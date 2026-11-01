using Lair.Backends.Cpu;
using Lair.Core.Contracts;
using Lair.Kv;

namespace Lair.Tests;

/// <summary>
/// Paged KV cache tests: free-list integrity under randomized alloc/release, exact addressing across
/// full and partial blocks with untouched slots staying zero, refcount-gated release, and the
/// all-layers-in-one-block size formula.
/// </summary>
[Trait("Category", "Kv")]
public class KvTests
{
    [Fact]
    public void Allocate_AllCapacity_ThrowsKvFullNamingCapacity_AndReleaseReopensSlot()
    {
        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 4);

        int[] blocks = new int[pool.CapacityBlocks];
        for (int i = 0; i < blocks.Length; i++)
        {
            blocks[i] = pool.Allocate();
        }

        Assert.Equal(0, pool.FreeBlocks);
        Assert.Equal(pool.CapacityBlocks, pool.UsedBlocks);
        Assert.Equal(blocks.Length, blocks.Distinct().Count());

        KvFullException full = Assert.Throws<KvFullException>(() => pool.Allocate());
        Assert.Contains("all 4 blocks", full.Message);
        Assert.Contains("kv_blocks", full.Message);

        pool.Release(blocks[2]);
        Assert.Equal(1, pool.FreeBlocks);
        Assert.Equal(blocks[2], pool.Allocate());
        Assert.Equal(0, pool.FreeBlocks);
    }

    [Fact]
    public void FreeList_Integrity_UnderRandomizedAllocRelease()
    {
        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 8);

        Random rng = new(20_261_101);
        HashSet<int> live = [];
        int allocOps = 0;
        int releaseOps = 0;

        for (int step = 0; step < 400; step++)
        {
            bool allocateFirst = live.Count == 0 || (live.Count < pool.CapacityBlocks && rng.Next(2) == 0);
            if (allocateFirst)
            {
                int id = pool.Allocate();
                Assert.True(live.Add(id), $"block {id} was handed out twice");
                allocOps++;
            }
            else
            {
                List<int> owned = [.. live];
                int victim = owned[rng.Next(owned.Count)];
                pool.Release(victim);
                Assert.True(live.Remove(victim));
                releaseOps++;
            }

            Assert.Equal(pool.CapacityBlocks - live.Count, pool.FreeBlocks);
            Assert.Equal(live.Count, pool.UsedBlocks);
        }

        Assert.True(allocOps > 10 && releaseOps > 10, $"randomized sequence barely exercised both paths ({allocOps}/{releaseOps})");

        foreach (int block in live)
        {
            pool.Release(block);
        }

        Assert.Equal(pool.CapacityBlocks, pool.FreeBlocks);

        HashSet<int> drained = [];
        while (pool.FreeBlocks > 0)
        {
            Assert.True(drained.Add(pool.Allocate()), "drain handed out a duplicate block id");
        }

        Assert.Equal(pool.CapacityBlocks, drained.Count);
    }

    [Fact]
    public void Addressing_SpanningThreeBlocksWithPartialLast_ReadsBackExact()
    {
        const int Layers = 3;
        const int KvHeads = 2;
        const int HeadDim = 4;
        const int Tokens = 40; // three blocks: full, full, partial (8 of 16 slots used)

        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 5, layers: Layers, kvHeads: KvHeads, headDim: HeadDim);
        SequenceKv seq = new(pool);

        int kvDim = KvHeads * HeadDim;
        float[] kPattern = new float[kvDim];
        float[] vPattern = new float[kvDim];

        for (int pos = 0; pos < Tokens; pos++)
        {
            seq.Append();
            for (int layer = 0; layer < Layers; layer++)
            {
                for (int i = 0; i < kvDim; i++)
                {
                    kPattern[i] = pos * 1_000 + layer * 100 + i;
                    vPattern[i] = -(pos * 1_000 + layer * 100 + i) - 0.5f;
                }

                seq.WriteK(pos, layer, kPattern);
                seq.WriteV(pos, layer, vPattern);
            }
        }

        Assert.Equal(Tokens, seq.Length);
        Assert.Equal(3, seq.Blocks.Count);
        Assert.Equal(48, seq.TokenCapacity); // 3 blocks x 16 slots; the last block is only partly used

        for (int pos = 0; pos < Tokens; pos++)
        {
            int expectedBlockIndex = pos / pool.BlockTokens;
            Assert.True(seq.Blocks[expectedBlockIndex] == seq.BlockFor(pos), $"position {pos} resolved to the wrong physical block");
        }

        for (int pos = 0; pos < Tokens; pos++)
        {
            for (int layer = 0; layer < Layers; layer++)
            {
                ReadOnlySpan<float> k = seq.ReadK(pos, layer);
                ReadOnlySpan<float> v = seq.ReadV(pos, layer);
                Assert.Equal(kvDim, k.Length);

                for (int i = 0; i < kvDim; i++)
                {
                    float expectedK = pos * 1_000 + layer * 100 + i;
                    Assert.True(k[i] == expectedK, $"K mismatch at position {pos}, layer {layer}, element {i}: got {k[i]}");
                    Assert.True(v[i] == -expectedK - 0.5f, $"V mismatch at position {pos}, layer {layer}, element {i}: got {v[i]}");
                }
            }
        }

        // The tail of the partial block was never written and stays zero in both K and V halves.
        int lastBlock = seq.Blocks[^1];
        ReadOnlySpan<float> payload = pool.BlockView(lastBlock);
        int unusedTailFloats = (Tokens % pool.BlockTokens) * kvDim;
        for (int layer = 0; layer < Layers; layer++)
        {
            int kHalfStart = layer * 2 * pool.BlockTokens * kvDim;
            int vHalfStart = kHalfStart + pool.BlockTokens * kvDim;

            for (int i = unusedTailFloats; i < pool.BlockTokens * kvDim; i++)
            {
                Assert.True(payload[kHalfStart + i] == 0f, $"unused K slot at float {kHalfStart + i} of block {lastBlock} was written");
                Assert.True(payload[vHalfStart + i] == 0f, $"unused V slot at float {vHalfStart + i} of block {lastBlock} was written");
            }
        }

        seq.Dispose(); // ownership ends: every owned block returns to the pool
        Assert.Equal(pool.CapacityBlocks, pool.FreeBlocks);
    }

    [Fact]
    public void Addressing_UntouchedSlotsStayZero_NoCrossLayerOrBlockBleed()
    {
        // Hand-computed geometry for this fixture: 16 token slots per block, 3 layers, kvDim 8 (2 heads x 4 dims).
        // One layer region = 2 * 16 * 8 = 256 floats; the K half occupies its first 128 floats.
        const int BlockTokens = 16;
        const int Layers = 3;
        const int KvDim = 8;
        const int LayerRegionFloats = 2 * BlockTokens * KvDim; // 256

        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 4, blockTokens: BlockTokens, layers: Layers, headDim: KvDim / 2);
        Assert.Equal(LayerRegionFloats * sizeof(float), (int)pool.BytesPerBlock / Layers);

        // Write only K of position 0 in layer 1: expect ones at floats [256..263] and nothing else.
        SequenceKv first = new(pool);
        int blockA = first.Append().BlockId;
        Assert.Equal(0, blockA);
        first.WriteK(logicalPos: 0, layer: 1, Ones(KvDim));

        ReadOnlySpan<float> payloadA = pool.BlockView(blockA);
        for (int i = 0; i < payloadA.Length; i++)
        {
            bool inTargetRegion = i >= LayerRegionFloats && i < LayerRegionFloats + KvDim; // K half of layer 1, slot 0
            Assert.True(payloadA[i] == (inTargetRegion ? 1f : 0f), $"unexpected value at float index {i} after a single isolated write");
        }

        foreach (float f in first.ReadK(0, 1))
        {
            Assert.Equal(1f, f);
        }

        // Fill the rest of block A's positions, then write only V of position 15: expect ones at floats [248..255].
        for (int pos = 1; pos < BlockTokens; pos++)
        {
            first.Append();
        }

        int vHalfBaseFloats = BlockTokens * KvDim; // the V half of a layer region starts after its K half
        first.WriteV(logicalPos: BlockTokens - 1, layer: 0, Ones(KvDim));

        ReadOnlySpan<float> payloadB = pool.BlockView(blockA);
        for (int i = 0; i < payloadB.Length; i++)
        {
            bool inFirstWrite = i >= LayerRegionFloats && i < LayerRegionFloats + KvDim; // K, layer 1, position 0
            bool inSecondWrite = i >= vHalfBaseFloats + (BlockTokens - 1) * KvDim && i < vHalfBaseFloats + BlockTokens * KvDim; // V, layer 0, last slot
            Assert.True(payloadB[i] == ((inFirstWrite || inSecondWrite) ? 1f : 0f), $"unexpected value at float index {i} after the second isolated write");
        }

        // A separate sequence gets its own block and cannot disturb this one.
        using SequenceKv second = new(pool);
        int blockB = second.Append().BlockId;
        Assert.NotEqual(blockA, blockB);
        second.WriteV(logicalPos: 0, layer: 2, Ones(KvDim));

        ReadOnlySpan<float> payloadAfterOtherWrite = pool.BlockView(blockA);
        for (int i = 0; i < payloadAfterOtherWrite.Length; i++)
        {
            bool inFirstWrite = i >= LayerRegionFloats && i < LayerRegionFloats + KvDim; // K, layer 1, position 0
            bool inSecondWrite = i >= vHalfBaseFloats + (BlockTokens - 1) * KvDim && i < vHalfBaseFloats + BlockTokens * KvDim; // V, layer 0, last slot
            Assert.True(payloadAfterOtherWrite[i] == ((inFirstWrite || inSecondWrite) ? 1f : 0f), $"block {blockA} was disturbed by writes to block {blockB}");
        }

        // Block B holds only its own write: V of position 0 in layer 2, floats [512+128 .. 512+135].
        ReadOnlySpan<float> payloadBOnly = pool.BlockView(blockB);
        int bWriteStart = (Layers - 1) * LayerRegionFloats + vHalfBaseFloats; // V half of the last layer region, slot 0
        for (int i = 0; i < payloadBOnly.Length; i++)
        {
            bool inBWrite = i >= bWriteStart && i < bWriteStart + KvDim;
            Assert.True(payloadBOnly[i] == (inBWrite ? 1f : 0f), $"block {blockB} contains unexpected values at float index {i}");
        }

        first.Dispose();
    }

    [Fact]
    public void Refcount_SharedBlockStaysInUseUntilLastRelease()
    {
        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 2);

        int blockA = pool.Allocate();
        Assert.Equal(1, pool.RefCount(blockA));
        Assert.Equal(1, pool.FreeBlocks); // capacity 2: one block live, one still free

        pool.AddRef(blockA); // share groundwork: a second owner joins the same physical block
        Assert.Equal(2, pool.RefCount(blockA));
        Assert.Equal(1, pool.FreeBlocks);

        pool.Release(blockA);
        Assert.Equal(1, pool.RefCount(blockA));
        Assert.Equal(1, pool.FreeBlocks); // still in use by the remaining owner

        int other = pool.Allocate();
        Assert.NotEqual(blockA, other);
        Assert.Equal(0, pool.FreeBlocks); // both physical blocks now handed out

        pool.Release(blockA);
        Assert.Equal(0, pool.RefCount(blockA));
        Assert.Equal(1, pool.FreeBlocks);
        Assert.Equal(blockA, pool.Allocate()); // back in circulation with refcount 1

        pool.Release(blockA);
        pool.Release(other);
        Assert.Equal(pool.CapacityBlocks, pool.FreeBlocks);
    }

    [Fact]
    public void Refcount_DoubleReleaseAndAddRefOnFreeBlockRejected()
    {
        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 1);

        int block = pool.Allocate();
        pool.Release(block);
        Assert.Throws<InvalidOperationException>(() => pool.AddRef(block)); // free blocks cannot be shared into
        Assert.Throws<InvalidOperationException>(() => pool.Release(block)); // double release rejected

        int reallocated = pool.Allocate();
        Assert.Equal(1, pool.RefCount(reallocated));
    }

    [Fact]
    public void BytesPerBlock_AllLayersInOneBlock_MatchesHandComputedMath()
    {
        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 1, layers: 4, kvHeads: 2, headDim: 64);

        // 4 layers * 2 (K/V) * 16 tokens * 2 heads * 64 dims * 4 bytes = 65_536
        Assert.Equal(65_536L, pool.BytesPerBlock);
        Assert.Equal(4, pool.LayerCount);
        Assert.Equal(128, pool.KvDim); // 2 * 64 floats per token side
        Assert.Equal(16, pool.BlockTokens);

        using CpuBackend wide = new();
        using BlockPool widePool = NewPool(wide, capacityBlocks: 3, layers: 4, kvHeads: 8, headDim: 128);
        // 4 * 2 * 16 * 8 * 128 * 4 = 524_288
        Assert.Equal(524_288L, widePool.BytesPerBlock);

        using CpuBackend slots = new();
        using BlockPool slotPool = NewPool(slots, capacityBlocks: 3);
        Assert.Equal(slotPool.SlotCapacity, (long)slotPool.CapacityBlocks * slotPool.BlockTokens);
        Assert.Equal((long)pool.CapacityBlocks * pool.BlockTokens, pool.FreeSlots); // fully free at construction
    }

    [Fact]
    public void WriteK_WrongVectorLength_Throws_AndReadBeyondLengthThrows()
    {
        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 2, kvHeads: 2, headDim: 8); // kvDim 16
        SequenceKv seq = new(pool);
        seq.Append();

        Assert.Throws<ArgumentException>(() => seq.WriteK(0, 0, new float[pool.KvDim - 1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => seq.ReadV(1, 0)); // position not appended yet
        Assert.Throws<ArgumentOutOfRangeException>(() => seq.WriteK(-1, 0, new float[pool.KvDim]));

        int held = pool.Allocate(); // consume the remaining block so a crossing append must fail
        using SequenceKv overflow = new(pool);
        KvFullException full = Assert.Throws<KvFullException>(() => overflow.Append());
        Assert.Contains($"all {pool.CapacityBlocks} blocks", full.Message);

        seq.Dispose(); // releases its owned block back to the pool
        Assert.Equal(1, pool.FreeBlocks); // `held` is still live
        pool.Release(held);
    }

    private static BlockPool NewPool(
        IBackend backend,
        int capacityBlocks = 4,
        int blockTokens = 16,
        int layers = 2,
        int kvHeads = 2,
        int headDim = 8) => new(blockTokens: blockTokens, layerCount: layers, kvHeads: kvHeads, headDim: headDim, capacityBlocks: capacityBlocks, backend: backend);

    private static float[] Ones(int count)
    {
        float[] values = new float[count];
        Array.Fill(values, 1f);
        return values;
    }
}
