using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;
using Lair.Kv;

namespace Lair.Tests;

/// <summary>
/// KV device-residency seam tests: a host-only pool keeps the exact free-list order and addressing it has always had,
/// a CPU backend refuses a device-backed arena naming itself, the tier, and the resource instead of answering with a
/// stand-in, CUDA-backed residencies open persistent device storage sized to exactly their declared block range while
/// attention keeps reading the pool's own host arena byte-for-byte, split divides which ids carry a device copy, and
/// every flush point — admission-window end, release-to-zero, an explicit call — is what carries dirty bytes across,
/// counted as crossed blocks and crossed bytes. Nothing here implies a kernel on the device reads the mirrored bytes:
/// crossing storage is residency, not a compute path.
/// </summary>
public sealed class KvResidencyTests
{
    // ---------------------------------------------------------------- host-only posture

    [Fact]
    [Trait("Category", "Kv")]
    public void Residency_DramMode_OpensNoMirror_KeepsFreeListOrder_AndFlushPointsDoNothing()
    {
        using CpuBackend backend = new();
        using BlockPool legacy = NewHostOnly(backend, capacityBlocks: 8); // the constructor every pool in this build has used
        using BlockPool explicitDram = NewWithResidency(backend, capacityBlocks: 8, BlockResidencyMode.Dram);

        Assert.Equal(BlockResidencyMode.Dram, explicitDram.Residency);
        Assert.Equal(0, explicitDram.DeviceBackedBlockCount);
        Assert.False(explicitDram.IsDeviceBacked(0));
        Assert.Equal(0L, explicitDram.DeviceMirrorBytes);
        Assert.Equal(0, explicitDram.DeviceMirrorCrossedBlocks);
        Assert.Equal(0L, explicitDram.DeviceMirrorCrossedBytes);
        Assert.Equal(0, explicitDram.DeviceMirrorPendingWrites);

        int[] legacyIds = AllocationOrderSignature(legacy);
        int[] dramIds = AllocationOrderSignature(explicitDram);
        Assert.Equal(legacyIds, dramIds); // id-for-id the same hand-out and re-hand-out sequence

        // Writes through the attention seam on a host-only pool owe no crossing at any flush point.
        SequenceKv seq = new(explicitDram);
        FillSequence(seq, explicitDram, 0, 48);
        explicitDram.BeginAdmissionWindow(1);
        explicitDram.EndAdmissionWindow();
        explicitDram.FlushDeviceMirror();
        Assert.Equal(0, explicitDram.DeviceMirrorPendingWrites);
        Assert.Equal(0, explicitDram.DeviceMirrorCrossedBlocks);
        Assert.Equal(0L, explicitDram.DeviceMirrorCrossedBytes);

        for (int pos = 0; pos < 48; pos++)
        {
            for (int layer = 0; layer < explicitDram.LayerCount; layer++)
            {
                SpansEqual(KPattern(explicitDram, pos, layer), seq.ReadK(pos, layer), $"host-only K drifted at position {pos}, layer {layer}");
                SpansEqual(VPattern(explicitDram, pos, layer), seq.ReadV(pos, layer), $"host-only V drifted at position {pos}, layer {layer}");
            }
        }

        seq.Dispose();
        Assert.Equal(explicitDram.CapacityBlocks, explicitDram.FreeBlocks);
    }

    // ---------------------------------------------------------------- CPU refusal

    [Fact]
    [Trait("Category", "Kv")]
    public void KvMirror_CpuBackendRefusesDeviceBackedArena_NamingBackendTierAndResource()
    {
        using CpuBackend cpu = new();
        KvMirrorRequest request = new()
        {
            HostArenaBase = 0x1_000, // never dereferenced: the refusal answers before any staging starts
            HostArenaBytes = 4 * 512,
            BytesPerBlock = 512,
            FirstBlockId = 0,
            BlockCount = 4,
            Tier = WeightTier.Vram,
            ResourceName = "kv block arena",
        };

        NotSupportedException refusal = Assert.Throws<NotSupportedException>(() => cpu.OpenKvMirror(request));
        Assert.Contains("CpuBackend", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("'cpu'", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("Vram", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("kv block arena", refusal.Message, StringComparison.Ordinal);

        NotSupportedException poolRefusal = Assert.Throws<NotSupportedException>(() =>
        {
            using BlockPool refused = NewWithResidency(cpu, capacityBlocks: 4, BlockResidencyMode.Vram);
        });
        Assert.Contains("CpuBackend", poolRefusal.Message, StringComparison.Ordinal);
        Assert.Contains("kv block arena", poolRefusal.Message, StringComparison.Ordinal);

        NotSupportedException splitRefusal = Assert.Throws<NotSupportedException>(() =>
        {
            using BlockPool refused = NewWithResidency(cpu, capacityBlocks: 8, BlockResidencyMode.Split);
        });
        Assert.Contains("CpuBackend", splitRefusal.Message, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- device-backed residencies (CUDA seam)

    [Fact]
    [Trait("Category", "Kv")]
    [Trait("Category", "Cache")]
    public void Residency_VramMode_BacksWithPersistentDeviceStorage_AndAttentionStillReadsHostArena()
    {
        using CudaBackend cuda = OpenCudaBackend();
        using BlockPool pool = NewWithResidency(cuda, capacityBlocks: 4, BlockResidencyMode.Vram);

        Assert.Equal(BlockResidencyMode.Vram, pool.Residency);
        Assert.Equal(pool.CapacityBlocks, pool.DeviceBackedBlockCount);
        Assert.Equal((long)pool.CapacityBlocks * pool.BytesPerBlock, pool.DeviceMirrorBytes); // persistent device copy sized to the whole declared range
        for (int id = 0; id < pool.CapacityBlocks; id++)
        {
            Assert.True(pool.IsDeviceBacked(id), $"block {id} must fall inside a vram residency");
        }

        Assert.False(pool.IsDeviceBacked(pool.CapacityBlocks)); // ids outside the reservation are not device-backed

        SequenceKv seq = new(pool);
        FillSequence(seq, pool, 0, 40); // three blocks: full, full, partial

        for (int pos = 0; pos < 40; pos++)
        {
            for (int layer = 0; layer < pool.LayerCount; layer++)
            {
                SpansEqual(KPattern(pool, pos, layer), seq.ReadK(pos, layer), $"device-backed K read drifted at position {pos}, layer {layer}");
                SpansEqual(VPattern(pool, pos, layer), seq.ReadV(pos, layer), $"device-backed V read drifted at position {pos}, layer {layer}");
            }
        }

        // Attention addressed the host arena: a whole-block view of a device-backed block holds exactly what was written,
        // at that block's own slot offsets.
        int lastBlock = seq.Blocks[^1];
        float[] payload = pool.BlockView(lastBlock).ToArray();
        for (int pos = 32; pos < 40; pos++)
        {
            for (int layer = 0; layer < pool.LayerCount; layer++)
            {
                SpansEqual(KPattern(pool, pos, layer), SlotOf(payload, pool, pos % pool.BlockTokens, layer, isV: false), $"host payload of device-backed block missing K at position {pos}, layer {layer}");
            }
        }

        Assert.Equal(1, cuda.OpenKvMirrorCount); // the residency opened exactly one mirror for its range — and no weight crossed to device on account of it
        Assert.Equal(0, cuda.DeviceWeightCopies);

        Assert.Equal(3, pool.DeviceMirrorPendingWrites); // three written blocks are owed a crossing each
        pool.FlushDeviceMirror();
        Assert.Equal(0, pool.DeviceMirrorPendingWrites);
        Assert.Equal(3, pool.DeviceMirrorCrossedBlocks);
        Assert.Equal(3 * pool.BytesPerBlock, pool.DeviceMirrorCrossedBytes);

        seq.Dispose(); // release-to-zero crossings: each freed block's current host bytes replace whatever its device copy held
        long crossedAfterDispose = pool.DeviceMirrorCrossedBytes;
        int blocksAfterDispose = pool.DeviceMirrorCrossedBlocks;
        Assert.Equal(0, pool.DeviceMirrorPendingWrites);

        pool.Dispose();
        Assert.Equal(6 * pool.BytesPerBlock, crossedAfterDispose); // three crossings on flush, one more per block at release-to-zero
        Assert.Equal(3, blocksAfterDispose); // distinct ids stay three: repeats are counted in bytes, not as new blocks
        Assert.Equal(0, cuda.OpenKvMirrorCount); // disposing the pool releases its mirror's pin and device storage
    }

    [Fact]
    [Trait("Category", "Kv")]
    public void Residency_SplitMode_DividesIdsBetweenDeviceBackedAndHostOnlyRanges()
    {
        using CudaBackend cuda = OpenCudaBackend();
        using BlockPool pool = NewWithResidency(cuda, capacityBlocks: 8, BlockResidencyMode.Split);

        Assert.Equal(BlockResidencyMode.Split, pool.Residency);
        Assert.Equal(4, pool.DeviceBackedBlockCount); // half the reservation is device-backed
        Assert.Equal(4L * pool.BytesPerBlock, pool.DeviceMirrorBytes);

        SequenceKv first = new(pool);
        FillSequence(first, pool, 0, 64); // four blocks: ids 0..3 — exactly where ascending allocation lands
        SequenceKv second = new(pool);
        FillSequence(second, pool, 0, 64); // the next four ids fall in the host-only tail

        foreach (int id in first.Blocks)
        {
            Assert.True(id < 4 && pool.IsDeviceBacked(id), $"block {id} should sit in the device-backed range");
        }

        foreach (int id in second.Blocks)
        {
            Assert.True(id >= 4 && !pool.IsDeviceBacked(id), $"block {id} should sit in the host-only range");
        }

        // Both ranges address through the same host seam, byte-for-byte.
        SequenceKv[] both = [first, second];
        foreach (SequenceKv seq in both)
        {
            for (int pos = 0; pos < 64; pos++)
            {
                for (int layer = 0; layer < pool.LayerCount; layer++)
                {
                    SpansEqual(KPattern(pool, pos, layer), seq.ReadK(pos, layer), $"split K read drifted at position {pos}, layer {layer}");
                    SpansEqual(VPattern(pool, pos, layer), seq.ReadV(pos, layer), $"split V read drifted at position {pos}, layer {layer}");
                }
            }
        }

        Assert.Equal(4, pool.DeviceMirrorPendingWrites); // only the device-backed ids owe a crossing; the host-only tail owes none
        pool.FlushDeviceMirror();
        int crossed = pool.DeviceMirrorCrossedBlocks;
        long bytes = pool.DeviceMirrorCrossedBytes;
        Assert.Equal(0, pool.DeviceMirrorPendingWrites);

        first.Dispose(); // releasing the device-backed range crosses each freed block once more
        long bytesAfterRelease = pool.DeviceMirrorCrossedBytes;
        int crossedAfterRelease = pool.DeviceMirrorCrossedBlocks;

        second.Dispose(); // a host-only release causes no crossing at all
        Assert.Equal(crossedAfterRelease, pool.DeviceMirrorCrossedBlocks);
        Assert.Equal(bytesAfterRelease, pool.DeviceMirrorCrossedBytes);

        Assert.Equal(4, crossed); // four distinct device-backed ids crossed
        Assert.Equal(4 * pool.BytesPerBlock, bytes);
        Assert.Equal(4, crossedAfterRelease); // still the same four ids: a release re-crosses them rather than adding new ones
        Assert.Equal(8 * pool.BytesPerBlock, bytesAfterRelease); // each id crossed once on flush and again at release-to-zero
    }

    [Fact]
    [Trait("Category", "Kv")]
    public void Residency_FlushPoints_CarryDirtyDeviceBlocksAndCountEveryCrossing()
    {
        using CudaBackend cuda = OpenCudaBackend();
        using BlockPool pool = NewWithResidency(cuda, capacityBlocks: 4, BlockResidencyMode.Vram);

        SequenceKv seq = new(pool);
        FillSequence(seq, pool, 0, 16); // one device-backed block written

        Assert.Equal(1, pool.DeviceMirrorPendingWrites); // a written device-backed block is owed exactly one crossing
        Assert.Equal(0, pool.DeviceMirrorCrossedBlocks);
        Assert.Equal(0L, pool.DeviceMirrorCrossedBytes);

        pool.BeginAdmissionWindow(1);
        seq.ReserveAppends(16); // the reserved crossing block joins the pending set at allocation
        Assert.Equal(2, pool.DeviceMirrorPendingWrites);

        pool.EndAdmissionWindow(); // admission-window end is a flush point
        Assert.Equal(0, pool.DeviceMirrorPendingWrites);
        Assert.Equal(2, pool.DeviceMirrorCrossedBlocks);
        Assert.Equal(2 * pool.BytesPerBlock, pool.DeviceMirrorCrossedBytes);

        FillSequence(seq, pool, 16, 32); // the reserved block fills: its bytes are dirty again after crossing once
        Assert.Equal(1, pool.DeviceMirrorPendingWrites);

        pool.FlushDeviceMirror(); // an explicit flush is the third way pending bytes cross
        int distinctBlocks = pool.DeviceMirrorCrossedBlocks;
        long crossedSoFar = pool.DeviceMirrorCrossedBytes;
        Assert.Equal(0, pool.DeviceMirrorPendingWrites);

        seq.Dispose(); // release-to-zero: both blocks' freed host bytes cross once more each
        int blocksAtEnd = pool.DeviceMirrorCrossedBlocks;
        long bytesAtEnd = pool.DeviceMirrorCrossedBytes;
        Assert.Equal(0, pool.DeviceMirrorPendingWrites);

        pool.Dispose();

        Assert.Equal(2, distinctBlocks); // the same two ids, crossed repeatedly rather than counted twice as blocks
        Assert.Equal(3 * pool.BytesPerBlock, crossedSoFar);
        Assert.Equal(2, blocksAtEnd);
        Assert.Equal(5 * pool.BytesPerBlock, bytesAtEnd); // five crossings over two distinct device-backed blocks
    }

    [Fact]
    [Trait("Category", "Kv")]
    public void Residency_PoolDisposeFlushesItsMirror_AndBackendDisposeReleasesOneLeftOpen()
    {
        using CudaBackend cuda = OpenCudaBackend();
        using BlockPool pool = NewWithResidency(cuda, capacityBlocks: 2, BlockResidencyMode.Vram);

        SequenceKv seq = new(pool);
        FillSequence(seq, pool, 0, 32); // every block of the reservation written
        Assert.Equal(1, cuda.OpenKvMirrorCount);
        Assert.Equal(2, pool.DeviceMirrorPendingWrites);

        seq.Dispose();
        long crossedAtPoolDispose = pool.DeviceMirrorCrossedBytes;
        int blocksCrossed = pool.DeviceMirrorCrossedBlocks;

        pool.Dispose(); // the mirror's last crossing runs while its arena is still alive to stage from
        Assert.Equal(0, cuda.OpenKvMirrorCount);
        Assert.Equal(3 * pool.BytesPerBlock, crossedAtPoolDispose); // both blocks crossed on the first release, one more on the second
        Assert.Equal(2, blocksCrossed);

        // A host that never disposes its pool still gets an open mirror's pin and device storage back at backend dispose.
        CudaBackend abandonedHost = OpenCudaBackend();
        BlockPool abandonedPool = NewWithResidency(abandonedHost, capacityBlocks: 1, BlockResidencyMode.Split);
        Assert.Equal(1L * abandonedPool.BytesPerBlock, abandonedPool.DeviceMirrorBytes); // a split reservation of one block device-backs its leading id
        Assert.Equal(1, abandonedHost.OpenKvMirrorCount);

        abandonedHost.Dispose(); // releases the outstanding mirror instead of leaving a pin and an allocation behind
        Assert.Equal(0, abandonedHost.OpenKvMirrorCount);
    }

    // ---------------------------------------------------------------- helpers

    private static BlockPool NewHostOnly(IBackend backend, int capacityBlocks) =>
        new(blockTokens: 16, layerCount: 2, kvHeads: 2, headDim: 8, capacityBlocks: capacityBlocks, backend);

    private static BlockPool NewWithResidency(IBackend backend, int capacityBlocks, BlockResidencyMode residency) =>
        new(blockTokens: 16, layerCount: 2, kvHeads: 2, headDim: 8, capacityBlocks: capacityBlocks, backend, residency);

    private static CudaBackend OpenCudaBackend()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        try
        {
            return new CudaBackend();
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

    // The order a pool hands out and re-hands-out ids, in a fixed interleaving: full drain, releases in a rotated
    // order, then a second drain. A residency that changed free-list behavior would show up here as a different sequence.
    private static int[] AllocationOrderSignature(BlockPool pool)
    {
        int capacity = pool.CapacityBlocks;
        int[] firstPass = new int[capacity];
        for (int i = 0; i < capacity; i++)
        {
            firstPass[i] = pool.Allocate();
        }

        List<int> released = [];
        for (int i = 0; i < capacity; i++)
        {
            int id = firstPass[(i + 3) % capacity];
            pool.Release(id);
            released.Add(id);
        }

        int[] secondPass = new int[capacity];
        for (int i = 0; i < capacity; i++)
        {
            secondPass[i] = pool.Allocate();
        }

        foreach (int id in secondPass)
        {
            pool.Release(id);
        }

        return [.. firstPass, .. released, .. secondPass];
    }

    private static void FillSequence(SequenceKv seq, BlockPool pool, int fromPos, int toPos)
    {
        for (int pos = fromPos; pos < toPos; pos++)
        {
            if (pos >= seq.Length)
            {
                seq.Append();
            }

            for (int layer = 0; layer < pool.LayerCount; layer++)
            {
                seq.WriteK(pos, layer, KPattern(pool, pos, layer));
                seq.WriteV(pos, layer, VPattern(pool, pos, layer));
            }
        }
    }

    private static float[] KPattern(BlockPool pool, int pos, int layer) => Pattern(pos * 1_000 + layer * 100, pool.KvDim);

    private static float[] VPattern(BlockPool pool, int pos, int layer) => Pattern(-(pos * 1_000 + layer * 100) - 1, pool.KvDim);

    private static float[] Pattern(int seedValue, int length)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = seedValue + i;
        }

        return values;
    }

    // Extracts one slot out of a captured whole-block payload, using the pool's documented layout and the slot's offset
    // within that block rather than its absolute position.
    private static float[] SlotOf(float[] blockPayload, BlockPool pool, int offsetInBlock, int layer, bool isV)
    {
        int floatsPerLayer = 2 * pool.BlockTokens * pool.KvDim;
        int start = layer * floatsPerLayer + (isV ? pool.BlockTokens * pool.KvDim : 0) + offsetInBlock * pool.KvDim;
        return blockPayload[start..(start + pool.KvDim)];
    }

    private static void SpansEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string message)
    {
        Assert.True(expected.SequenceEqual(actual), $"{message} (expected {Fmt(expected)}, got {Fmt(actual)})");
    }

    private static string Fmt(ReadOnlySpan<float> values)
    {
        int shown = Math.Min(values.Length, 4);
        List<string> parts = [];
        for (int i = 0; i < shown; i++)
        {
            parts.Add(values[i].ToString("R"));
        }

        return string.Join(", ", parts) + (values.Length > shown ? ", ..." : string.Empty);
    }
}
