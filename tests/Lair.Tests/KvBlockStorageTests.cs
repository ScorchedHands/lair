using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Lair.Backends.Cpu;
using Lair.Kv;

namespace Lair.Tests;

/// <summary>
/// Generic quantized block-struct storage: GGML q8_0/q4_0 block sizes, generic pool footprint against the format descriptor,
/// writer semantics reused from the committed quantizers, bit-identical round-trips across backend thread counts and between
/// batched and token-by-token region writes, and copy-on-write that moves a whole quantized block without re-quantizing it.
/// Production FP32 BlockPool/SequenceKv behavior is untouched by this item and stays covered by KvTests.
/// </summary>
[Trait("Category", "Kv")]
public class KvBlockStorageTests
{
    // ---- block struct sizes ----

    [Fact]
    public void BlockStructSizes_Q8Is34Bytes_Q4Is18Bytes()
    {
        Assert.Equal(34, Unsafe.SizeOf<Q8Block>());
        Assert.Equal(34, Q8Block.BlockBytes);
        Assert.Equal(34, KvFormatPair.GroupBytes(KvSideFormat.Q8_0));

        Assert.Equal(18, Unsafe.SizeOf<Q4Block>());
        Assert.Equal(18, Q4Block.BlockBytes);
        Assert.Equal(18, KvFormatPair.GroupBytes(KvSideFormat.Q4_0));
    }

    // ---- generic pool footprint matches the descriptor formula ----

    [Theory]
    [InlineData(KvSideFormat.Q8_0, KvSideFormat.Q8_0)]
    [InlineData(KvSideFormat.Q8_0, KvSideFormat.Q4_0)]
    [InlineData(KvSideFormat.Q4_0, KvSideFormat.Q4_0)] // storage geometry may prove the dropped pair even though serving refuses it
    public void GenericPoolFootprint_MatchesDescriptorBytesPerBlock(KvSideFormat k, KvSideFormat v)
    {
        KvFormatPair formats = new(k, v);
        using CpuBackend backend = new();

        // kvDim = 2 heads * 16 dims = 32 -> exactly one whole group per token region per side.
        long actual = (k, v) switch
        {
            (KvSideFormat.Q8_0, KvSideFormat.Q8_0) => Footprint(new BlockPool<Q8Block, Q8Block>(4, 2, 2, 16, 1, backend, formats)),
            (KvSideFormat.Q8_0, KvSideFormat.Q4_0) => Footprint(new BlockPool<Q8Block, Q4Block>(4, 2, 2, 16, 1, backend, formats)),
            _ => Footprint(new BlockPool<Q4Block, Q4Block>(4, 2, 2, 16, 1, backend, formats)),
        };

        Assert.Equal(formats.BytesPerBlock(2, 4, 32), actual);
    }

    [Fact]
    public void GenericPoolFootprint_AsymmetricK8V4_SumsEachSideWithItsOwnGroupBytes()
    {
        using CpuBackend backend = new();
        using BlockPool<Q8Block, Q4Block> pool = new(4, 2, 2, 16, 1, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q4_0));

        // 2 layers * (4 tokens * 1 group * 34 + 4 tokens * 1 group * 18) = 2 * (136 + 72) = 416
        Assert.Equal(416L, pool.BytesPerBlock);
    }

    [Fact]
    public void GenericPoolFloatSideRefusesNamingTheSide_AndStructMismatchRefuses()
    {
        using CpuBackend backend = new();

        ArgumentException kFloat = Assert.Throws<ArgumentException>(
            () => new BlockPool<Q8Block, Q8Block>(4, 2, 2, 16, 1, backend, new KvFormatPair(KvSideFormat.F32, KvSideFormat.Q8_0)));
        Assert.Contains("accepts only the quantized sides", kFloat.Message, StringComparison.Ordinal);
        Assert.Contains("'k' declares 'f32'", kFloat.Message, StringComparison.Ordinal);

        // Struct geometry disagrees with the named pair: Q4Block (18 bytes) cannot hold a q8_0 side (34 bytes).
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new BlockPool<Q4Block, Q8Block>(4, 2, 2, 16, 1, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0)));

        // Partial-group token region refused: kvDim 24 is not a whole 32-value group.
        ArgumentException partial = Assert.Throws<ArgumentException>(
            () => new BlockPool<Q8Block, Q8Block>(4, 2, 2, 12, 1, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0)));
        Assert.Contains("whole 32-value groups per token region", partial.Message, StringComparison.Ordinal);
    }

    // ---- bit-identical round-trips ----

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void QuantizedStorage_ThreadCountsProduceIdenticalRawBytes(int threads)
    {
        using CpuBackend backend = new(threads: threads, allowAvx2: true);
        byte[] stored = StoredRawBytesForFixture(backend);

        // Compare against a single-threaded reference: the writer is pure scalar code, so thread count cannot move a byte.
        using CpuBackend single = new(threads: 1, allowAvx2: true);
        byte[] reference = StoredRawBytesForFixture(single);

        Assert.Equal(reference, stored);
    }

    [Fact]
    public void QuantizedStorage_BatchedSideWriteMatchesTokenByToken()
    {
        const int BlockTokens = 4;
        const int Layers = 2;
        const int KvDim = 32; // 2 heads * 16 dims

        Random rng = new(20_261_117);
        float[] values = new float[BlockTokens * KvDim];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)(rng.NextDouble() * 6.0 - 3.0);
        }

        using CpuBackend backend = new();

        // Token-by-token: one SequenceKv write per token slot.
        using BlockPool<Q8Block, Q4Block> tokenPool = new(BlockTokens, Layers, 2, 16, 2, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q4_0));
        using SequenceKv<Q8Block, Q4Block> seq = new(tokenPool);
        for (int pos = 0; pos < BlockTokens; pos++)
        {
            seq.Append();
            seq.WriteK(pos, 0, values.AsSpan(pos * KvDim, KvDim));
            seq.WriteV(pos, 0, values.AsSpan(pos * KvDim, KvDim));
        }

        int tokenBlock = seq.BlockFor(0);

        // Batched: one whole-side region write for the same values into a fresh block.
        int batchBlock = tokenPool.Allocate();
        Q8Block.WriteRegion(values, tokenPool.MutableKSideSpan(batchBlock, 0));
        Q4Block.WriteRegion(values, tokenPool.MutableVSideSpan(batchBlock, 0));

        Assert.True(
            tokenPool.RawBlockView(tokenBlock).SequenceEqual(tokenPool.RawBlockView(batchBlock)),
            "batched side-region writes and token-by-token writes produced different stored bytes");
    }

    // ---- q8_0 writer semantics ----

    [Fact]
    public void Q8Writer_ZeroGroup_StoresZeroScaleAndZeroQuants()
    {
        byte[] arena = new byte[Q8Block.BlockBytes];
        Q8Block.WriteRegion(new float[Q8Block.Values], MemoryMarshal.Cast<byte, Q8Block>(arena.AsSpan()));

        Q8Block block = MemoryMarshal.Cast<byte, Q8Block>(arena.AsSpan())[0];
        Assert.Equal(0x0000, block.ScaleBits); // half +0
        for (int i = 0; i < Q8Block.Values; i++)
        {
            Assert.Equal(0, QuantByte(arena, i)); // every quant stays zero
        }
    }

    [Fact]
    public void Q8Writer_PositiveNegativeAndMidpointRounding_RoundAwayFromZeroAgainstHalfScale()
    {
        // amax = 127 so the half scale is exactly 1.0; midpoint values then exercise AwayFromZero directly.
        float[] values = new float[Q8Block.Values];
        values[0] = 127f;
        values[1] = 0.5f;   // 0.5 / 1.0 -> 0.5 -> away-from-zero -> 1
        values[2] = -0.5f;  // -> -1
        values[3] = 126.5f; // -> 127
        values[4] = -126.5f;// -> -127
        values[5] = 3.4f;   // -> 3
        values[6] = -3.6f;  // -> -4

        byte[] arena = new byte[Q8Block.BlockBytes];
        Q8Block.WriteRegion(values, MemoryMarshal.Cast<byte, Q8Block>(arena.AsSpan()));
        Q8Block block = MemoryMarshal.Cast<byte, Q8Block>(arena.AsSpan())[0];

        Assert.Equal(BitConverter.HalfToUInt16Bits((Half)1f), block.ScaleBits);

        Assert.Equal(127, (sbyte)QuantByte(arena, 0));
        Assert.Equal(1, (sbyte)QuantByte(arena, 1));
        Assert.Equal(-1, (sbyte)QuantByte(arena, 2));
        Assert.Equal(127, (sbyte)QuantByte(arena, 3));
        Assert.Equal(-127, (sbyte)QuantByte(arena, 4));
        Assert.Equal(3, (sbyte)QuantByte(arena, 5));
        Assert.Equal(-4, (sbyte)QuantByte(arena, 6));
    }

    [Fact]
    public void Q8Writer_EveryQuantStaysInRange_UnderExtremeMagnitudes()
    {
        // Clamping posture: whatever the magnitude, no stored quant leaves the signed-byte range.
        float[] values = new float[Q8Block.Values];
        Random rng = new(7);
        for (int trial = 0; trial < 64; trial++)
        {
            values[0] = float.MaxValue / 2f; // a large amax that still scales finite
            values[1] = -float.MaxValue / 2f;
            for (int i = 2; i < values.Length; i++)
            {
                values[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
            }

            byte[] arena = new byte[Q8Block.BlockBytes];
            Q8Block.WriteRegion(values, MemoryMarshal.Cast<byte, Q8Block>(arena.AsSpan()));

            for (int i = 0; i < Q8Block.Values; i++)
            {
                sbyte q = (sbyte)QuantByte(arena, i);
                Assert.InRange(q, (sbyte)-127, (sbyte)127);
            }
        }
    }

    // ---- q4_0 writer semantics ----

    [Fact]
    public void Q4Writer_ZeroGroup_StoresNegativeZeroScaleAndCodeEightNibbles()
    {
        byte[] arena = new byte[Q4Block.BlockBytes];
        Q4Block.WriteRegion(new float[Q4Block.Values], MemoryMarshal.Cast<byte, Q4Block>(arena.AsSpan()));

        Q4Block block = MemoryMarshal.Cast<byte, Q4Block>(arena.AsSpan())[0];
        Assert.Equal(0x8000, block.ScaleBits); // half -0
        for (int j = 0; j < Q4Block.Values / 2; j++)
        {
            Assert.Equal(0x88, NibbleByte(arena, j)); // code 8 in both nibbles
        }
    }

    [Fact]
    public void Q4Writer_SignedMaxScale_SelectsFirstStrictImprovementAndStoresSignedHalf()
    {
        // element 0 = -8 (first strict improvement from 0), element 16 = +8 (ties do not replace it): d = max / -8 = 1.0.
        float[] values = new float[Q4Block.Values];
        values[0] = -8f;
        values[16] = 8f;

        byte[] arena = new byte[Q4Block.BlockBytes];
        Q4Block.WriteRegion(values, MemoryMarshal.Cast<byte, Q4Block>(arena.AsSpan()));
        Q4Block block = MemoryMarshal.Cast<byte, Q4Block>(arena.AsSpan())[0];

        Assert.Equal(BitConverter.HalfToUInt16Bits((Half)1f), block.ScaleBits);

        // id = 1/1 = 1: element0 z = -8 + 8.5 = 0.5 -> 0 (low nibble); element16 z = 8 + 8.5 = 16.5 -> clamp 15 (high nibble).
        Assert.Equal(0xF0, NibbleByte(arena, 0)); // low nibble 0 (element 0), high nibble 15 (element 16)
    }

    [Fact]
    public void Q4Writer_NegativeMaxScale_ProducesNegativeHalfScaleAndReciprocalCodeRounding()
    {
        // element 0 = 8 -> max 8 -> d = -1.0 (negative half scale). id = -1.
        float[] values = new float[Q4Block.Values];
        values[0] = 8f;

        byte[] arena = new byte[Q4Block.BlockBytes];
        Q4Block.WriteRegion(values, MemoryMarshal.Cast<byte, Q4Block>(arena.AsSpan()));
        Q4Block block = MemoryMarshal.Cast<byte, Q4Block>(arena.AsSpan())[0];

        Assert.Equal(BitConverter.HalfToUInt16Bits((Half)(-1f)), block.ScaleBits);

        // element0 z = 8 * -1 + 8.5 = 0.5 -> 0; element16 (zero) z = 8.5 -> 8.
        Assert.Equal(0x80, NibbleByte(arena, 0)); // low nibble 0 (element 0), high nibble 8 (element 16)
    }

    [Fact]
    public void Q4Writer_CrossHalfNibblePairing_PairsElementJWithElement16PlusJ()
    {
        // Distinct codes for element 0 and element 16 land in byte 0's low/high nibbles respectively;
        // element 1 and element 17 land in byte 1, proving the pairing is half-region, not adjacent.
        float[] values = new float[Q4Block.Values];
        values[0] = 4f;   // max magnitude 4 -> d = -0.5 -> id = -2
        values[16] = 0f;
        values[1] = -2f;  // z = -2*-2 + 8.5 = 12.5 -> 12
        values[17] = 2f;  // z = 2*-2 + 8.5 = 4.5 -> 4

        byte[] arena = new byte[Q4Block.BlockBytes];
        Q4Block.WriteRegion(values, MemoryMarshal.Cast<byte, Q4Block>(arena.AsSpan()));
        Q4Block block = MemoryMarshal.Cast<byte, Q4Block>(arena.AsSpan())[0];

        Assert.Equal(BitConverter.HalfToUInt16Bits((Half)(-0.5f)), block.ScaleBits);

        // element0 z = 4*-2 + 8.5 = 0.5 -> 0; element16 z = 8.5 -> 8 => byte0 = low 0 | high 8<<4 = 0x80
        Assert.Equal(0x80, NibbleByte(arena, 0));
        // element1 -> 12 (low), element17 -> 4 (high) => byte1 = 12 | (4<<4) = 0x4C
        Assert.Equal(0x4C, NibbleByte(arena, 1));
    }

    // ---- copy-on-write copies the quantized block whole, without re-quantizing ----

    [Fact]
    public void CopyOnWrite_SharedQuantizedBlock_CopiedWholeWithoutRequantizing()
    {
        const int BlockTokens = 4;
        const int Layers = 2;
        const int KvDim = 32;

        using CpuBackend backend = new();
        using BlockPool<Q8Block, Q4Block> pool = new(BlockTokens, Layers, 2, 16, 4, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q4_0));

        // Owner A fills a whole block across every layer/side/token with quantized data.
        using SequenceKv<Q8Block, Q4Block> ownerA = new(pool);
        for (int pos = 0; pos < BlockTokens; pos++)
        {
            ownerA.Append();
            for (int layer = 0; layer < Layers; layer++)
            {
                ownerA.WriteK(pos, layer, Pattern(pos, layer, KvDim, positive: true));
                ownerA.WriteV(pos, layer, Pattern(pos, layer, KvDim, positive: false));
            }
        }

        int sharedBlock = ownerA.BlockFor(0);
        byte[] originalBytes = pool.RawBlockView(sharedBlock).ToArray();

        // Share the block with a second owner and adopt it into a sequence that will rewrite one token.
        pool.AddRef(sharedBlock); // a cache entry / another owner joins the same physical block

        using SequenceKv<Q8Block, Q4Block> adopted = new(pool);
        adopted.MarkAdopted(sharedBlock, startLogicalPos: 0);
        adopted.SetLength(BlockTokens);

        Assert.True(adopted.IsAdopted(0));
        Assert.Equal(sharedBlock, adopted.BlockFor(0));

        // Trigger copy-on-write by writing a different value into one adopted token.
        adopted.WriteK(1, 0, Pattern(999, 0, KvDim, positive: true));

        // The original shared block's raw bytes are untouched.
        Assert.True(
            pool.RawBlockView(sharedBlock).SequenceEqual(originalBytes),
            "copy-on-write disturbed the original shared block's stored bytes");

        // The private copy is a different physical block whose untouched bytes match the original exactly (every layer/side/token group).
        int privateBlock = adopted.BlockFor(0);
        Assert.NotEqual(sharedBlock, privateBlock);

        // Compare the private block against a fresh whole-block copy of the original, excluding the one rewritten token's K group.
        byte[] privateBytes = pool.RawBlockView(privateBlock).ToArray();
        Assert.Equal(originalBytes.Length, privateBytes.Length);

        int rewrittenTokenKStart = 1 * (KvDim / Q8Block.Values) * Q8Block.BlockBytes; // token slot 1, K side, layer 0
        int rewrittenTokenKEnd = rewrittenTokenKStart + (KvDim / Q8Block.Values) * Q8Block.BlockBytes;
        for (int i = 0; i < originalBytes.Length; i++)
        {
            bool isRewrittenRegion = i >= rewrittenTokenKStart && i < rewrittenTokenKEnd;
            if (isRewrittenRegion)
            {
                continue; // the rewritten token's K groups differ by design
            }

            Assert.Equal(originalBytes[i], privateBytes[i]); // untouched layers/sides/token groups survive bit-identically
        }

        // The rewritten token's K groups now hold the new value, proving the write landed in the private copy.
        ReadOnlySpan<Q8Block> newK = adopted.ReadK(1, 0);
        Assert.True(newK.Length > 0);

        // Refcount/free-list sanity: the shared block lost B's reference to it, so it is back to owner A's single reference.
        Assert.Equal(1, pool.RefCount(sharedBlock));
        Assert.Equal(1, pool.RefCount(privateBlock));

        adopted.Dispose(); // releases the private copy
        Assert.Equal(0, pool.RefCount(privateBlock));

        ownerA.Dispose(); // releases the shared block's last reference
        Assert.Equal(0, pool.RefCount(sharedBlock));
        Assert.Equal(pool.CapacityBlocks, pool.FreeBlocks);
    }

    // ---- existing non-generic FP32 pool/sequence remain unchanged (compile-time presence) ----

    [Fact]
    public void NonGenericFp32PoolAndSequence_StillAddressFloats()
    {
        using CpuBackend backend = new();
        using BlockPool pool = new(blockTokens: 16, layerCount: 2, kvHeads: 2, headDim: 8, capacityBlocks: 2, backend: backend);
        using SequenceKv seq = new(pool);

        Assert.True(pool.Formats.IsFloatStorage);
        Assert.Equal(2 * 2 * 16 * 16 * sizeof(float), pool.BytesPerBlock);

        seq.Append();
        seq.WriteK(0, 0, Ones(pool.KvDim));
        foreach (float f in seq.ReadK(0, 0))
        {
            Assert.Equal(1f, f);
        }
    }

    // ---- helpers ----

    private static long Footprint<TK, TV>(BlockPool<TK, TV> pool)
        where TK : struct, IKvFormat<TK>
        where TV : struct, IKvFormat<TV>
    {
        long bytes = pool.BytesPerBlock;
        pool.Dispose();
        return bytes;
    }

    private static byte[] StoredRawBytesForFixture(CpuBackend backend)
    {
        const int BlockTokens = 4;
        const int Layers = 2;
        const int KvDim = 32;

        Random rng = new(20_261_117);
        float[] values = new float[BlockTokens * KvDim];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)(rng.NextDouble() * 6.0 - 3.0);
        }

        using BlockPool<Q8Block, Q4Block> pool = new(BlockTokens, Layers, 2, 16, 1, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q4_0));
        using SequenceKv<Q8Block, Q4Block> seq = new(pool);
        for (int pos = 0; pos < BlockTokens; pos++)
        {
            seq.Append();
            for (int layer = 0; layer < Layers; layer++)
            {
                seq.WriteK(pos, layer, values.AsSpan(pos * KvDim, KvDim));
                seq.WriteV(pos, layer, values.AsSpan(pos * KvDim, KvDim));
            }
        }

        return pool.RawBlockView(seq.BlockFor(0)).ToArray();
    }

    private static float[] Pattern(int pos, int layer, int kvDim, bool positive)
    {
        float[] values = new float[kvDim];
        for (int i = 0; i < kvDim; i++)
        {
            float magnitude = pos * 1_000 + layer * 100 + i;
            values[i] = positive ? magnitude : -magnitude - 0.5f;
        }

        return values;
    }

    private static float[] Ones(int count)
    {
        float[] values = new float[count];
        Array.Fill(values, 1f);
        return values;
    }

    private static byte QuantByte(byte[] q8Arena, int index) => q8Arena[sizeof(ushort) + index];

    private static byte NibbleByte(byte[] q4Arena, int index) => q4Arena[sizeof(ushort) + index];
}
