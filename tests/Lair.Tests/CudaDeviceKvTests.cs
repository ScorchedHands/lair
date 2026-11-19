using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;
using Lair.Kv;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// GPU gate for device-resident KV read/write on the transformer step: a pool opened with the same <c>vram</c> and
/// <c>split</c> vocabulary as MVP4/MVP5.1, but in the device-step execution posture, hands its persistent device copy to
/// the step as a store — the step's KV write lands on-device through the pool's own block layout (block-major, then
/// layer, then each layer's K and V half, then token slot and element), and its causal attention walk reads those same
/// resident blocks without any host arena read. The claims: a store-bound step writes and reads its own resident blocks
/// byte-for-byte where the pool's views would address them; the composed step crosses the seam once for its staged rows
/// and once for its attention result per layer, with no residency staging between kernels; the same geometry answers
/// bit-identically through a private arena or the store, walked once or twice. The host-read posture stays the default:
/// <c>vram</c> and <c>split</c> residencies opened without an execution posture keep attention reading the pool's own
/// arena byte-for-byte exactly as MVP4 shipped, their mirrors refuse to hand out a store, and a step bound to one
/// refuses naming both sides. Unsupported combinations — a device-step posture with no device-backed range, a quantized
/// pair, or geometry the store does not describe — refuse at open rather than falling back.
/// </summary>
[Trait("Category", "Kv")]
[Trait("Category", "Prefill")]
[Trait("Category", "Cuda")]
public sealed class CudaDeviceKvTests
{
    // The same recorded bands the private-arena device step carries against the host reference: a relative band plus an
    // absolute ceiling per element, unchanged — this posture runs the identical attention math through paged addressing,
    // so its drift allowance is the existing seam's, and no CPU band or CUDA GEMV/GEMM band moves with it.
    private const float AttentionBand = 3e-4f;

    private const float AttentionAbsCeiling = 1e-6f;

    // The rotation readback carries the device step's recorded rotation band against the host rotation, unchanged from
    // the private-arena seam: the device rope kernel and the host helper agree inside that band, not bit-for-bit.
    private const float RopeBand = 3e-5f;

    private const float RopeAbsCeiling = 1e-6f;

    private const int BlockTokens = 8; // a power-of-two paging width the store-bound kernels resolve by shift and mask

    private const int Layers = 2;

    private const int KvHeads = 2;

    private const int HeadDim = 8;

    private const int Heads = 4; // grouped-query 2:1 over the kv heads

    private const float AttnScale = 0.125f;

    private static float Value(int seed) => (float)Math.Sin(0.37 * seed + 0.13) * 1.7f;

    private static float[] Filled(int length, int seedOffset)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = Value(seedOffset + i);
        }

        return values;
    }

    private static float[] QueryRows(int rows, int seedOffset) => Filled(rows * Heads * HeadDim, seedOffset);

    private static float[] KeyValueRows(int rows, int seedOffset) => Filled(rows * KvHeads * HeadDim, seedOffset);

    // One window's slice of a longer row run: the K/V rows for positions [rowStart, rowStart + rowCount).
    private static float[] SliceRows(float[] kvRowValues, int rowStart, int rowCount) =>
        kvRowValues.AsSpan(rowStart * KvHeads * HeadDim, rowCount * KvHeads * HeadDim).ToArray();

    // A pool whose whole reservation is device-backed and runs the device-step posture: every block id the free list
    // hands out falls inside the store a bound step reads and writes.
    private static BlockPool OpenStoreBackedPool(IBackend backend, int capacityBlocks) =>
        new(BlockTokens, Layers, KvHeads, HeadDim, capacityBlocks, backend, BlockResidencyMode.Vram, KvMirrorExecution.DeviceStep);

    // ---------------------------------------------------------------- device-side write lands where the pool addresses it

    [Fact]
    public void DeviceKv_StoreBackedStep_WritesItsOwnResidentBlocks_AtPoolAddressedSlots()
    {
        using CudaBackend cuda = OpenCuda();
        using BlockPool pool = OpenStoreBackedPool(cuda, capacityBlocks: 4); // positions 0..31 across four blocks

        Assert.True(pool.DeviceExecuted);
        Assert.Equal(BlockResidencyMode.Vram, pool.Residency);
        Assert.Equal(KvMirrorExecution.DeviceStep, pool.Execution);
        Assert.Equal(4L * pool.BytesPerBlock, pool.DeviceMirrorBytes); // persistent store sized to the whole declared range
        Assert.Equal(0, cuda.PersistentDeviceWeightBytes); // a KV store opens no weight residency

        float[] cos = new float[32 * HeadDim / 2];
        float[] sin = new float[32 * HeadDim / 2];
        LayerOps.BuildRopeTables(thetaBase: 10_000.0, HeadDim, 32, cos, sin);

        using CudaDeviceStep step = cuda.OpenDeviceStep(StoreRequest(), pool.DeviceStepMirror!);
        Assert.True(step.KvStoreBacked);
        step.LoadRopeTables(cos, sin);

        // A window that crosses a block boundary: positions 5..12 live in blocks 0 and 1 at slots 5..7 and 0..4.
        const int PositionBase = 5;
        const int Rows = 8;
        float[] qRows = QueryRows(Rows, seedOffset: 13);
        float[] kRows = KeyValueRows(Rows, seedOffset: 113);
        float[] vRows = KeyValueRows(Rows, seedOffset: 213);

        // Rotate the keys the way the composed step does, so the readback checks the bytes the step actually wrote.
        float[] rotatedK = (float[])kRows.Clone();
        for (int r = 0; r < Rows; r++)
        {
            for (int h = 0; h < KvHeads; h++)
            {
                LayerOps.ApplyRopeToHead(
                    rotatedK.AsSpan(r * KvHeads * HeadDim + h * HeadDim, HeadDim),
                    HeadDim / 2,
                    cos.AsSpan((PositionBase + r) * (HeadDim / 2), HeadDim / 2),
                    sin.AsSpan((PositionBase + r) * (HeadDim / 2), HeadDim / 2));
            }
        }

        step.StageWindow(qRows, kRows, vRows);
        step.ApplyRope(PositionBase);
        step.WriteKv(layer: 0, PositionBase);

        Assert.Equal(2, pool.DeviceWrittenBlocks); // positions 5..12 touched exactly blocks 0 and 1
        Assert.Equal(0, pool.DeviceMirrorPendingWrites); // a device-side write owes no host-to-device crossing

        // Read each touched block back from the store and address it through the pool's documented layout: every
        // written position's K and V must sit exactly where SequenceKv's views address the same (block, slot, layer).
        int floatsPerBlock = Layers * 2 * BlockTokens * KvHeads * HeadDim;
        foreach (int blockId in new[] { 0, 1 })
        {
            float[] payload = new float[floatsPerBlock];
            pool.ReadDeviceBlockToHost(blockId, payload);

            for (int pos = PositionBase; pos < PositionBase + Rows; pos++)
            {
                if (pos / BlockTokens != blockId)
                {
                    continue;
                }

                int slot = pos % BlockTokens;
                for (int h = 0; h < KvHeads; h++)
                {
                    int element = h * HeadDim;
                    Span<float> kSlot = SlotOf(payload, slot, layer: 0, isV: false);
                    Span<float> vSlot = SlotOf(payload, slot, layer: 0, isV: true);

                    AssertWithinBand(
                        rotatedK.AsSpan((pos - PositionBase) * KvHeads * HeadDim + element, HeadDim),
                        kSlot.Slice(element, HeadDim),
                        RopeBand,
                        RopeAbsCeiling,
                        $"device-written K at position {pos} must land at the pool slot the host views address");
                    BatchKernelTests.AssertBitsEqual(
                        vRows.AsSpan((pos - PositionBase) * KvHeads * HeadDim + element, HeadDim),
                        vSlot.Slice(element, HeadDim).ToArray(),
                        $"device-written V at position {pos} must land at the pool slot the host views address");
                }
            }
        }

        // A block the step never wrote still holds the store's zeroed start — the same unwritten posture the arena has.
        float[] untouched = new float[floatsPerBlock];
        pool.ReadDeviceBlockToHost(3, untouched);
        Assert.True(untouched.All(static v => v == 0f), "a block a store-bound step never wrote must read back as the store's zeros");
    }

    // ---------------------------------------------------------------- attention reads the resident blocks, no staging per layer

    [Fact]
    public void DeviceKv_ComposedStepOverStore_RunsEveryLayerWithoutHostStaging()
    {
        using CudaBackend cuda = OpenCuda();
        using BlockPool pool = OpenStoreBackedPool(cuda, capacityBlocks: 4); // positions 0..31

        float[] cos = new float[32 * HeadDim / 2];
        float[] sin = new float[32 * HeadDim / 2];
        LayerOps.BuildRopeTables(thetaBase: 10_000.0, HeadDim, 32, cos, sin);

        using CudaDeviceStep step = cuda.OpenDeviceStep(StoreRequest(), pool.DeviceStepMirror!);
        step.LoadRopeTables(cos, sin);

        const int PositionBase = 5;
        const int Rows = 8;

        // The positions the walk reaches back over are written first, the way a forward pass runs them.
        float[] earlierK = KeyValueRows(PositionBase, seedOffset: 313);
        float[] earlierV = KeyValueRows(PositionBase, seedOffset: 413);
        step.StageWindow(QueryRows(PositionBase, seedOffset: 513), earlierK, earlierV);
        step.WriteKv(layer: 0, positionBase: 0);
        step.WriteKv(layer: 1, positionBase: 0);

        long residencyBytesBefore = pool.DeviceMirrorCrossedBytes;
        long transfersBefore = cuda.RuntimeTransferCallsTotal;

        float[] outLayer0 = new float[Rows * Heads * HeadDim];
        float[] outLayer1 = new float[Rows * Heads * HeadDim];
        step.RunStep(0, PositionBase, QueryRows(Rows, seedOffset: 13), KeyValueRows(Rows, seedOffset: 113), KeyValueRows(Rows, seedOffset: 213), AttnScale, softcap: 50f, window: 3, outLayer0);
        step.RunStep(1, PositionBase, QueryRows(Rows, seedOffset: 613), KeyValueRows(Rows, seedOffset: 713), KeyValueRows(Rows, seedOffset: 813), AttnScale, softcap: 50f, window: 3, outLayer1);

        // Four crossings per layer — three staged row groups in, one attention result out — and nothing else: no
        // whole-block residency move between the kernels, per layer or in total.
        Assert.Equal(8, cuda.RuntimeTransferCallsTotal - transfersBefore);
        Assert.Equal(residencyBytesBefore, pool.DeviceMirrorCrossedBytes);
        Assert.Equal(0, pool.DeviceMirrorPendingWrites);
        Assert.Equal(2, pool.DeviceWrittenBlocks); // the same two blocks, written on-device at both layers
        Assert.True(cuda.KernelBusyMicrosecondsTotal > 0, "the device step recorded no kernel busy time");

        // Both layers' answers match the host attention over the same written positions — each layer reads its own
        // layer-region of the resident blocks.
        for (int layer = 0; layer < Layers; layer++)
        {
            int seed = layer == 0 ? 13 : 613;
            float[] qRows = QueryRows(Rows, seedOffset: seed);
            float[] kRows = KeyValueRows(Rows, seedOffset: seed + 100);
            float[] vRows = KeyValueRows(Rows, seedOffset: seed + 200);

            float[] rotatedQ = (float[])qRows.Clone();
            float[] rotatedK = (float[])kRows.Clone();
            for (int r = 0; r < Rows; r++)
            {
                for (int h = 0; h < Heads; h++)
                {
                    LayerOps.ApplyRopeToHead(
                        rotatedQ.AsSpan(r * Heads * HeadDim + h * HeadDim, HeadDim),
                        HeadDim / 2,
                        cos.AsSpan((PositionBase + r) * (HeadDim / 2), HeadDim / 2),
                        sin.AsSpan((PositionBase + r) * (HeadDim / 2), HeadDim / 2));
                }

                for (int h = 0; h < KvHeads; h++)
                {
                    LayerOps.ApplyRopeToHead(
                        rotatedK.AsSpan(r * KvHeads * HeadDim + h * HeadDim, HeadDim),
                        HeadDim / 2,
                        cos.AsSpan((PositionBase + r) * (HeadDim / 2), HeadDim / 2),
                        sin.AsSpan((PositionBase + r) * (HeadDim / 2), HeadDim / 2));
                }
            }

            float[] expected = HostAttention(
                rotatedQ, Rows, Heads, KvHeads, HeadDim, layer,
                earlierK, earlierV, PositionBase,
                rotatedK, vRows, Rows,
                AttnScale, softcap: 50f, window: 3, positionBase: PositionBase);

            float[] actual = layer == 0 ? outLayer0 : outLayer1;
            Assert.False(actual.Any(float.IsNaN), $"store-bound attention produced a NaN at layer {layer}");
            AssertWithinBand(expected, actual, AttentionBand, AttentionAbsCeiling, $"store-bound composed step at layer {layer}");
        }
    }

    [Fact]
    public void DeviceKv_ResidentBlocksAnswerLaterWindowsWithoutRestagingOldKv()
    {
        using CudaBackend cuda = OpenCuda();
        using BlockPool pool = OpenStoreBackedPool(cuda, capacityBlocks: 4);

        using CudaDeviceStep step = cuda.OpenDeviceStep(StoreRequest(), pool.DeviceStepMirror!);

        const int FirstRows = 8; // positions 0..7 — block 0 entirely
        const int SecondRows = 2; // positions 8..9 — the head of block 1

        float[] allK = KeyValueRows(FirstRows + SecondRows, seedOffset: 117);
        float[] allV = KeyValueRows(FirstRows + SecondRows, seedOffset: 217);

        // First window: positions 0..7 written on-device and attended.
        step.StageWindow(QueryRows(FirstRows, seedOffset: 17), SliceRows(allK, 0, FirstRows), SliceRows(allV, 0, FirstRows));
        step.WriteKv(0, positionBase: 0);

        float[] firstOut = new float[FirstRows * Heads * HeadDim];
        step.Attend(0, positionBase: 0, AttnScale, softcap: 0f, window: 0, firstOut);

        long transfersBefore = cuda.RuntimeTransferCallsTotal;
        long residencyBytesBefore = pool.DeviceMirrorCrossedBytes;

        // Second window at positions 8..9: only its own rows cross the host seam, and its walk reads block 0 — the
        // resident block the first window wrote — with no whole-block residency move of any kind.
        step.StageWindow(QueryRows(SecondRows, seedOffset: 317), SliceRows(allK, FirstRows, SecondRows), SliceRows(allV, FirstRows, SecondRows));
        step.WriteKv(0, positionBase: FirstRows);
        Assert.Equal(3, cuda.RuntimeTransferCallsTotal - transfersBefore); // three staged row groups in — no block crosses

        float[] secondOut = new float[SecondRows * Heads * HeadDim];
        step.Attend(0, positionBase: FirstRows, AttnScale, softcap: 0f, window: 6, secondOut);
        Assert.Equal(4, cuda.RuntimeTransferCallsTotal - transfersBefore); // plus the attention result out
        Assert.Equal(residencyBytesBefore, pool.DeviceMirrorCrossedBytes); // block 0 crossed no second time
        Assert.Equal(2, pool.DeviceWrittenBlocks); // blocks 0 and 1, each written once on-device

        // A decode-style re-attend of the already-written window: only the queries in and the result out cross.
        long beforeReattend = cuda.RuntimeTransferCallsTotal;
        step.StageQueries(QueryRows(SecondRows, seedOffset: 317));
        float[] reattendOut = new float[SecondRows * Heads * HeadDim];
        step.Attend(0, positionBase: FirstRows, AttnScale, softcap: 0f, window: 6, reattendOut);
        Assert.Equal(2, cuda.RuntimeTransferCallsTotal - beforeReattend); // queries in and result out — the resident blocks cross no time
        Assert.Equal(residencyBytesBefore, pool.DeviceMirrorCrossedBytes);

        float[] expectedFirst = HostAttention(
            QueryRows(FirstRows, seedOffset: 17), FirstRows, Heads, KvHeads, HeadDim, layer: 0,
            allK, allV, written: FirstRows, AttnScale, 0f, window: 0);
        AssertWithinBand(expectedFirst, firstOut, AttentionBand, AttentionAbsCeiling, "first window over its own device writes");

        float[] expectedSecond = HostAttention(
            QueryRows(SecondRows, seedOffset: 317), SecondRows, Heads, KvHeads, HeadDim, layer: 0,
            allK, allV, written: FirstRows + SecondRows, AttnScale, 0f, window: 6, positionBase: FirstRows);
        AssertWithinBand(expectedSecond, secondOut, AttentionBand, AttentionAbsCeiling, "later window over the resident blocks");
        BatchKernelTests.AssertBitsEqual(secondOut, reattendOut, "one window answered twice from resident state");
    }

    // ---------------------------------------------------------------- determinism posture across the two stores

    [Fact]
    public void DeviceKv_StoreAndPrivateArenaAnswerIdenticalBytes()
    {
        using CudaBackend cuda = OpenCuda();
        using BlockPool pool = OpenStoreBackedPool(cuda, capacityBlocks: 4);

        float[] qRows = QueryRows(6, seedOffset: 23);
        float[] kRows = KeyValueRows(6, seedOffset: 123);
        float[] vRows = KeyValueRows(6, seedOffset: 223);

        using CudaDeviceStep privateStep = cuda.OpenDeviceStep(StoreRequest());
        using CudaDeviceStep storeStep = cuda.OpenDeviceStep(StoreRequest(), pool.DeviceStepMirror!);
        Assert.False(privateStep.KvStoreBacked);
        Assert.True(storeStep.KvStoreBacked);

        float[] cos = new float[32 * HeadDim / 2];
        float[] sin = new float[32 * HeadDim / 2];
        LayerOps.BuildRopeTables(thetaBase: 10_000.0, HeadDim, 32, cos, sin);
        privateStep.LoadRopeTables(cos, sin);
        storeStep.LoadRopeTables(cos, sin);

        // Positions 0..4 stand written in both stores before the composed window at 5..10 attends them.
        foreach (CudaDeviceStep step in new[] { privateStep, storeStep })
        {
            step.StageWindow(QueryRows(5, seedOffset: 323), KeyValueRows(5, seedOffset: 423), KeyValueRows(5, seedOffset: 523));
            step.WriteKv(0, positionBase: 0);
        }

        float[] privateOut = new float[6 * Heads * HeadDim];
        privateStep.RunStep(0, positionBase: 5, qRows, kRows, vRows, AttnScale, softcap: 5f, window: 0, privateOut);

        float[] storeOut = new float[6 * Heads * HeadDim];
        storeStep.RunStep(0, positionBase: 5, qRows, kRows, vRows, AttnScale, softcap: 5f, window: 0, storeOut);
        BatchKernelTests.AssertBitsEqual(privateOut, storeOut, "the same window through a private arena and the pool store");

        // The store-bound step walked again answers its own bytes: the walk reads only resident state, and the same
        // roped queries through the same walk give the same answer.
        storeStep.StageQueries(QueryRows(6, seedOffset: 23));
        storeStep.ApplyRope(5);
        float[] again = new float[6 * Heads * HeadDim];
        storeStep.Attend(0, positionBase: 5, AttnScale, softcap: 5f, window: 0, again);
        BatchKernelTests.AssertBitsEqual(storeOut, again, "one store-bound step walked twice");
    }

    // ---------------------------------------------------------------- host-read posture stays the unchanged non-default

    [Fact]
    public void DeviceKv_HostReadArenaMode_StillWorksUnchangedAsNonDefault()
    {
        using CudaBackend cuda = OpenCuda();

        // The posture every device-backed residency has always opened with: same vocabulary, no execution posture stated.
        using BlockPool hostRead = new(BlockTokens, Layers, KvHeads, HeadDim, capacityBlocks: 4, cuda, BlockResidencyMode.Vram);
        Assert.Equal(BlockResidencyMode.Vram, hostRead.Residency);
        Assert.Equal(KvMirrorExecution.HostArenaRead, hostRead.Execution);
        Assert.False(hostRead.DeviceExecuted);
        Assert.False(hostRead.TryGetDeviceStore(out KvDeviceStore _)); // a host-read mirror hands out no store

        // Attention keeps reading the pool's own arena byte-for-byte, exactly as MVP4 shipped.
        using CpuBackend cpu = new();
        SequenceKv seq = new(hostRead);
        for (int pos = 0; pos < 32; pos++)
        {
            seq.Append();
            for (int layer = 0; layer < Layers; layer++)
            {
                seq.WriteK(pos, layer, Filled(KvHeads * HeadDim, seedOffset: pos * 100 + layer));
                seq.WriteV(pos, layer, Filled(KvHeads * HeadDim, seedOffset: -(pos * 100 + layer) - 1));
            }
        }

        Assert.Equal(4, hostRead.DeviceMirrorPendingWrites); // host writes owe whole-block crossings, as before
        hostRead.FlushDeviceMirror();
        Assert.Equal(0, hostRead.DeviceMirrorPendingWrites);
        Assert.Equal(4 * hostRead.BytesPerBlock, hostRead.DeviceMirrorCrossedBytes);

        float[] qHeads = Filled(Heads * HeadDim, seedOffset: 7);
        float[] outHost = new float[Heads * HeadDim];
        LayerOps.CausalAttention(
            cpu, qHeads, Heads, KvHeads, HeadDim, seq, layer: 0, 31, AttnScale, softcap: 0f, window: 0, outHost);

        // The same writes and the same attention on a host-only pool answer identically: the residency axis did not move
        // a byte the host seam reads.
        using BlockPool dram = new(BlockTokens, Layers, KvHeads, HeadDim, capacityBlocks: 4, cpu);
        SequenceKv seqDram = new(dram);
        for (int pos = 0; pos < 32; pos++)
        {
            seqDram.Append();
            for (int layer = 0; layer < Layers; layer++)
            {
                seqDram.WriteK(pos, layer, Filled(KvHeads * HeadDim, seedOffset: pos * 100 + layer));
                seqDram.WriteV(pos, layer, Filled(KvHeads * HeadDim, seedOffset: -(pos * 100 + layer) - 1));
            }
        }

        float[] outDram = new float[Heads * HeadDim];
        LayerOps.CausalAttention(
            cpu, qHeads, Heads, KvHeads, HeadDim, seqDram, layer: 0, 31, AttnScale, softcap: 0f, window: 0, outDram);
        BatchKernelTests.AssertBitsEqual(outHost, outDram, "host-read residency answers attention byte-for-byte the host-only posture answers");

        seq.Dispose();
        seqDram.Dispose();

        // A step cannot bind to a host-read mirror: its copy is residency accounting, not a store.
        Assert.True(hostRead.TryGetDeviceMirror(out IKvDeviceMirror? hostMirror));
        ArgumentException refusal = Assert.Throws<ArgumentException>(() => cuda.OpenDeviceStep(StoreRequest(), hostMirror!));
        Assert.Contains("host-read mirror", refusal.Message, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- vocabulary: split divides the store

    [Fact]
    public void DeviceKv_SplitVocabulary_StoreCoversLeadingIds_HostOnlyTailStaysHostRead()
    {
        using CudaBackend cuda = OpenCuda();
        using BlockPool pool = new(BlockTokens, Layers, KvHeads, HeadDim, capacityBlocks: 8, cuda, BlockResidencyMode.Split, KvMirrorExecution.DeviceStep);

        Assert.Equal(BlockResidencyMode.Split, pool.Residency);
        Assert.Equal(4, pool.DeviceBackedBlockCount); // half the reservation, the leading ids, as every split residency declares
        Assert.True(pool.DeviceExecuted);

        CudaDeviceStepRequest request = new()
        {
            MaxPositions = 4 * BlockTokens, // the step addresses exactly the store's range
            LayerCount = Layers,
            HeadCount = Heads,
            KvHeadCount = KvHeads,
            HeadDim = HeadDim,
        };

        using CudaDeviceStep step = cuda.OpenDeviceStep(request, pool.DeviceStepMirror!);

        step.StageWindow(QueryRows(32, seedOffset: 17), KeyValueRows(32, seedOffset: 117), KeyValueRows(32, seedOffset: 217));
        step.WriteKv(0, positionBase: 0);

        Assert.Equal(4, pool.DeviceWrittenBlocks); // positions 0..31 cover exactly the store's four blocks
        Assert.True(pool.IsDeviceBacked(3));
        Assert.False(pool.IsDeviceBacked(4)); // the tail half stays host-only

        int floatsPerBlock = Layers * 2 * BlockTokens * KvHeads * HeadDim;
        float[] payload = new float[floatsPerBlock];
        pool.ReadDeviceBlockToHost(3, payload); // the last device-backed block
        for (int h = 0; h < KvHeads; h++)
        {
            BatchKernelTests.AssertBitsEqual(
                KeyValueRows(32, seedOffset: 117).AsSpan(31 * KvHeads * HeadDim + h * HeadDim, HeadDim),
                SlotOf(payload, slot: 7, layer: 0, isV: false).Slice(h * HeadDim, HeadDim).ToArray(),
                "the split store's last device-backed block holds the step's write at its own slot");
        }

        // The host-only tail is not in the store: the mirror refuses a read-back outside its range.
        ArgumentOutOfRangeException tailRefusal = Assert.Throws<ArgumentOutOfRangeException>(
            () => pool.ReadDeviceBlockToHost(4, payload));
        Assert.Contains("outside the device-backed range", tailRefusal.Message, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- refusals: unsupported combinations name themselves

    [Fact]
    public void DeviceKv_ExecutionRefusesCombinationsItCannotServe_NamingEach()
    {
        using CpuBackend cpu = new();

        // A device-step posture beside a host-only residency has no range to execute against.
        ArgumentException dramRefusal = Assert.Throws<ArgumentException>(() =>
            new BlockPool(BlockTokens, Layers, KvHeads, HeadDim, capacityBlocks: 4, cpu, BlockResidencyMode.Dram, KvMirrorExecution.DeviceStep));
        Assert.Contains("device-step KV execution", dramRefusal.Message, StringComparison.Ordinal);
        Assert.Contains("'dram'", dramRefusal.Message, StringComparison.Ordinal);

        // A quantized pair has no float addressing a step's kernels walk.
        ArgumentException quantRefusal = Assert.Throws<ArgumentException>(() =>
            new BlockPool(
                BlockTokens, Layers, KvHeads, HeadDim, capacityBlocks: 4, cpu,
                BlockResidencyMode.Vram, deviceBackedBlocks: 4, mirrorOwner: null,
                new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q4_0), KvMirrorExecution.DeviceStep));
        Assert.Contains("FP32 block storage", quantRefusal.Message, StringComparison.Ordinal);
        Assert.Contains("q8_0", quantRefusal.Message, StringComparison.Ordinal);

        // A CPU backend answers the posture the same way it answers any device-backed residency: refusing itself.
        NotSupportedException cpuRefusal = Assert.Throws<NotSupportedException>(() =>
            new BlockPool(BlockTokens, Layers, KvHeads, HeadDim, capacityBlocks: 4, cpu, BlockResidencyMode.Vram, KvMirrorExecution.DeviceStep));
        Assert.Contains("CpuBackend", cpuRefusal.Message, StringComparison.Ordinal);
        Assert.Contains("kv block arena", cpuRefusal.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DeviceKv_StepBindingRefusesGeometryTheStoreDoesNotDescribe_NamingBothSides()
    {
        using CudaBackend cuda = OpenCuda();
        using BlockPool pool = OpenStoreBackedPool(cuda, capacityBlocks: 4); // 32 positions, 2 layers, kv region 16

        IKvDeviceMirror mirror = pool.DeviceStepMirror!;

        ArgumentException positionRefusal = Assert.Throws<ArgumentException>(() => cuda.OpenDeviceStep(
            new CudaDeviceStepRequest
            {
                MaxPositions = 24, // the store covers 4 blocks x 8 tokens = 32
                LayerCount = Layers,
                HeadCount = Heads,
                KvHeadCount = KvHeads,
                HeadDim = HeadDim,
            },
            mirror));
        Assert.Contains("covers 4 blocks of 8 tokens = 32 positions", positionRefusal.Message, StringComparison.Ordinal);
        Assert.Contains("MaxPositions 24", positionRefusal.Message, StringComparison.Ordinal);

        ArgumentException layerRefusal = Assert.Throws<ArgumentException>(() => cuda.OpenDeviceStep(
            new CudaDeviceStepRequest
            {
                MaxPositions = 32,
                LayerCount = 3,
                HeadCount = Heads,
                KvHeadCount = KvHeads,
                HeadDim = HeadDim,
            },
            mirror));
        Assert.Contains("store holds 2 layers", layerRefusal.Message, StringComparison.Ordinal);

        ArgumentException kvRefusal = Assert.Throws<ArgumentException>(() => cuda.OpenDeviceStep(
            new CudaDeviceStepRequest
            {
                MaxPositions = 32,
                LayerCount = Layers,
                HeadCount = 4,
                KvHeadCount = 4, // a kv region of 32 against the store's 16
                HeadDim = HeadDim,
            },
            mirror));
        Assert.Contains("kv region is 16 values", kvRefusal.Message, StringComparison.Ordinal);

        // A block size the kernels cannot page by shift refuses rather than walking the wrong block mid-step.
        using BlockPool raggedPool = new(3, Layers, KvHeads, HeadDim, capacityBlocks: 3, cuda, BlockResidencyMode.Vram, KvMirrorExecution.DeviceStep);
        ArgumentException raggedRefusal = Assert.Throws<ArgumentException>(() => cuda.OpenDeviceStep(
            new CudaDeviceStepRequest
            {
                MaxPositions = 9,
                LayerCount = Layers,
                HeadCount = Heads,
                KvHeadCount = KvHeads,
                HeadDim = HeadDim,
            },
            raggedPool.DeviceStepMirror!));
        Assert.Contains("not a power of two", raggedRefusal.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DeviceKv_OpeningAStoreBackedPoolAndStepLeavesWeightResidencyAndMirrorsAlone()
    {
        using CudaBackend cuda = OpenCuda();
        using BlockPool pool = OpenStoreBackedPool(cuda, capacityBlocks: 4);
        using CudaDeviceStep step = cuda.OpenDeviceStep(StoreRequest(), pool.DeviceStepMirror!);

        Assert.Equal(1, cuda.OpenKvMirrorCount); // the store is the pool's mirror — the step opens no second copy
        Assert.Equal(1, cuda.OpenDeviceStepCount);
        Assert.Equal(0, cuda.DeviceWeightCopies);
        Assert.Equal(0, cuda.PersistentDeviceWeightBytes);
        Assert.True(step.DeviceBytes > 0);

        // Disposing the pool releases its store; a step outliving its pool refuses rather than walking a released store.
        step.Dispose();
        Assert.Equal(0, cuda.OpenDeviceStepCount);
    }

    // ---------------------------------------------------------------- fixtures and helpers

    private static CudaDeviceStepRequest StoreRequest() => new()
    {
        MaxPositions = 32,
        LayerCount = Layers,
        HeadCount = Heads,
        KvHeadCount = KvHeads,
        HeadDim = HeadDim,
    };

    private static CudaBackend OpenCuda(CudaWeightResidency residency = CudaWeightResidency.DeviceCopy)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        try
        {
            return new CudaBackend(residency);
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }
    }

    // Extracts one slot out of a captured whole-block payload read back from the store, using the pool's documented
    // layout and the slot's offset within that block rather than its absolute position.
    private static Span<float> SlotOf(float[] blockPayload, int slot, int layer, bool isV)
    {
        int floatsPerLayer = 2 * BlockTokens * KvHeads * HeadDim;
        int start = layer * floatsPerLayer + (isV ? BlockTokens * KvHeads * HeadDim : 0) + slot * KvHeads * HeadDim;
        return blockPayload.AsSpan(start, KvHeads * HeadDim);
    }

    // The shipped CPU attention over the same geometry: a host KV sequence holds every written position's K and V and
    // each row attends its own causal range through LayerOps.CausalAttention — the per-row walk the tiled host kernel
    // is documented to answer bit-for-bit.
    private static float[] HostAttention(
        float[] qRows,
        int rows,
        int heads,
        int kvHeads,
        int headDim,
        int layer,
        float[] earlierK,
        float[] earlierV,
        int earlierRows,
        float[] windowK,
        float[] windowV,
        int windowRows,
        float attnScale,
        float softcap,
        int window,
        int positionBase) =>
        HostAttention(
            qRows, rows, heads, kvHeads, headDim, layer,
            Concat(earlierK, windowK), Concat(earlierV, windowV), earlierRows + windowRows,
            attnScale, softcap, window, positionBase);

    private static float[] Concat(float[] first, float[] second)
    {
        float[] all = new float[first.Length + second.Length];
        first.CopyTo(all, 0);
        second.CopyTo(all, first.Length);
        return all;
    }

    private static float[] HostAttention(
        float[] qRows,
        int rows,
        int heads,
        int kvHeads,
        int headDim,
        int layer,
        float[] allK,
        float[] allV,
        int written,
        float attnScale,
        float softcap,
        int window,
        int positionBase = 0)
    {
        int kvDim = kvHeads * headDim;

        using CpuBackend cpu = new();
        using BlockPool pool = new(
            blockTokens: BlockTokens,
            layerCount: layer + 1,
            kvHeads: kvHeads,
            headDim: headDim,
            capacityBlocks: written / BlockTokens + 2,
            backend: cpu);

        SequenceKv kv = new(pool);
        try
        {
            for (int pos = 0; pos < written; pos++)
            {
                kv.Append();
                kv.WriteK(pos, layer, new ReadOnlySpan<float>(allK, pos * kvDim, kvDim));
                kv.WriteV(pos, layer, new ReadOnlySpan<float>(allV, pos * kvDim, kvDim));
            }

            float[] outRows = new float[rows * heads * headDim];
            for (int r = 0; r < rows; r++)
            {
                LayerOps.CausalAttention(
                    cpu,
                    new ReadOnlySpan<float>(qRows, r * heads * headDim, heads * headDim),
                    heads,
                    kvHeads,
                    headDim,
                    kv,
                    layer,
                    positionBase + r,
                    attnScale,
                    softcap,
                    window,
                    outRows.AsSpan(r * heads * headDim, heads * headDim));
            }

            return outRows;
        }
        finally
        {
            kv.Dispose();
        }
    }

    private static void AssertWithinBand(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float band, float absoluteCeiling, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        float worstRelative = 0f;
        int relativeIndex = 0;
        double worstAbsolute = 0d;
        int absoluteIndex = 0;
        int offender = -1;

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]), $"{context}: NaN at element {i}");

            float delta = Math.Abs(expected[i] - actual[i]);
            float scale = Math.Max(Math.Abs(expected[i]), Math.Abs(actual[i]));
            float relative = scale == 0f ? delta : delta / scale;

            if (relative > worstRelative)
            {
                worstRelative = relative;
                relativeIndex = i;
            }

            if (delta > worstAbsolute)
            {
                worstAbsolute = delta;
                absoluteIndex = i;
            }

            double allowance = Math.Max((double)band * scale, absoluteCeiling);
            if (delta > allowance && offender < 0)
            {
                offender = i;
            }
        }

        if (offender >= 0)
        {
            Assert.Fail(
                $"{context}: widest relative drift {worstRelative:E6} at element {relativeIndex} and widest absolute " +
                $"drift {worstAbsolute:E6} at element {absoluteIndex} against the recorded band {band:E2}, ceiling {absoluteCeiling:E2}" +
                $" — offending element {offender}: expected {expected[offender]:E9}, actual {actual[offender]:E9}.");
        }
    }
}
