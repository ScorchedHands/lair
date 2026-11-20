using Lair.Contracts;

namespace Lair.Backends.Cuda;

/// <summary>
/// The CUDA-backed device mirror over a host KV arena another owner holds. The arena is pinned with the driver from its
/// first page through its last, and a persistent device allocation is opened for exactly one whole block per
/// device-backed block id; a flush copies each dirty block's whole host payload into that allocation with a
/// host-to-device copy, so the device-resident copy of the declared range tracks the host writes the pool makes. The
/// arena stays owned and readable by the caller. In the host-read posture nothing on the device is implied to read the
/// device copy: crossing a block's bytes is residency, not a compute path. In the device-step posture the same
/// allocation is a store: it is zeroed from the pool's own arena at open (the arena is zero-initialized, so the store
/// starts byte-identical to it), handed to a step through <see cref="TryGetDeviceStore"/> for on-device read/write, and
/// its device-side writes are recorded as residency accounting with no byte crossing. A host read of a block the device
/// holds goes through <see cref="ReadBlockToHost"/> rather than through arena bytes the device step never wrote.
/// </summary>
internal sealed unsafe class CudaKvMirror : IKvDeviceMirror
{
    private readonly CudaBackend _backend;
    private readonly KvMirrorRequest _request;

    // The pinned span covers exactly the host arena passed in: the backend allocation seam already returns a pointer that
    // satisfies the driver's registration alignment, and this mirror pins only bytes owned by the caller.
    private readonly nint _pinnedBase;
    private readonly long _pinnedBytes;
    private readonly ulong _devicePointer;

    private readonly HashSet<int> _dirty = [];
    private readonly HashSet<int> _crossed = [];
    private readonly HashSet<int> _deviceWritten = [];
    private long _crossedBytes;
    private bool _disposed;

    internal CudaKvMirror(CudaBackend backend, KvMirrorRequest request)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(request.BytesPerBlock);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(request.BlockCount);
        ArgumentOutOfRangeException.ThrowIfNegative(request.FirstBlockId);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(request.HostArenaBytes);
        if (request.HostArenaBase == 0)
        {
            throw new ArgumentException($"A KV mirror for '{request.ResourceName}' needs a live host arena to stage from.", nameof(request));
        }

        long deviceBytes;
        long arenaRangeEnd;
        try
        {
            deviceBytes = checked((long)request.BlockCount * request.BytesPerBlock);
            arenaRangeEnd = checked((request.FirstBlockId + (long)request.BlockCount) * request.BytesPerBlock);
        }
        catch (OverflowException)
        {
            throw new ArgumentException(
                $"'{request.ResourceName}' at tier {request.Tier}: block range {request.BlockCount} x {request.BytesPerBlock} " +
                $"bytes starting at block {request.FirstBlockId} overflows the addressable range.",
                nameof(request));
        }

        if (arenaRangeEnd > request.HostArenaBytes)
        {
            throw new ArgumentException(
                $"'{request.ResourceName}' at tier {request.Tier}: the device-backed range ends at byte {arenaRangeEnd} of a " +
                $"{request.HostArenaBytes}-byte host arena.",
                nameof(request));
        }

        // The driver seam sizes an allocation in native unsigned integers: on a 32-bit process that ceiling is uint.MaxValue,
        // and a device range beyond it is refused naming the resource rather than silently truncated.
        if (IntPtr.Size < 8 && deviceBytes > uint.MaxValue)
        {
            throw new ArgumentException(
                $"'{request.ResourceName}' at tier {request.Tier}: {deviceBytes} device bytes exceeds the allocation size the driver seam accepts.",
                nameof(request));
        }

        // A store a step addresses needs the pool geometry it pages through, and that geometry has to account for the
        // block bytes exactly. For the FP32 combo one block is layerCount * 2 * blockTokens * kvDim floats. For a
        // quantized combo each side region is whole groups of its block struct, so a block is layerCount * (kSideBytes +
        // vSideBytes) with each side rounded up to whole groups on its own. A request whose geometry disagrees with its
        // own block size states a layout no store can hold, so it refuses here naming both.
        if (request.Execution == KvMirrorExecution.DeviceStep)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(request.BlockTokens);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(request.LayerCount);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(request.KvDim);

            if (request.Combo == KvMirrorFormatCombo.Float)
            {
                long floatBlockBytes = checked(request.LayerCount * 2L * request.BlockTokens * request.KvDim * sizeof(float));
                if (floatBlockBytes != request.BytesPerBlock)
                {
                    throw new ArgumentException(
                        $"'{request.ResourceName}' at tier {request.Tier} as a device-step store: {request.LayerCount} layers x 2 halves x " +
                        $"{request.BlockTokens} tokens x {request.KvDim} values as FP32 is {floatBlockBytes} bytes, not the declared " +
                        $"{request.BytesPerBlock}-byte block — a float device step addresses FP32 block storage only.",
                        nameof(request));
                }
            }
            else
            {
                ArgumentOutOfRangeException.ThrowIfNegativeOrZero(request.KGroupBytes);
                ArgumentOutOfRangeException.ThrowIfNegativeOrZero(request.VGroupBytes);
                ArgumentOutOfRangeException.ThrowIfNegativeOrZero(request.GroupsPerTokenK);
                ArgumentOutOfRangeException.ThrowIfNegativeOrZero(request.GroupsPerTokenV);

                if (request.KvDim % request.GroupsPerTokenK != 0 || request.KvDim % request.GroupsPerTokenV != 0)
                {
                    throw new ArgumentException(
                        $"'{request.ResourceName}' at tier {request.Tier} as a {request.Combo} device-step store: a token region of " +
                        $"{request.KvDim} values is not a whole number of groups on each side " +
                        $"(k: {request.GroupsPerTokenK} groups, v: {request.GroupsPerTokenV} groups).",
                        nameof(request));
                }

                long kSide = checked((long)request.BlockTokens * request.GroupsPerTokenK * request.KGroupBytes);
                long vSide = checked((long)request.BlockTokens * request.GroupsPerTokenV * request.VGroupBytes);
                long layerRegion = checked(kSide + vSide);
                long quantBlockBytes = checked((long)request.LayerCount * layerRegion);
                if (request.KSideBytesPerLayer != kSide || request.LayerRegionBytes != layerRegion || quantBlockBytes != request.BytesPerBlock)
                {
                    throw new ArgumentException(
                        $"'{request.ResourceName}' at tier {request.Tier} as a {request.Combo} device-step store: {request.LayerCount} layers x " +
                        $"(k {request.BlockTokens} tokens x {request.GroupsPerTokenK} groups x {request.KGroupBytes} bytes + v " +
                        $"{request.BlockTokens} tokens x {request.GroupsPerTokenV} groups x {request.VGroupBytes} bytes) is " +
                        $"{quantBlockBytes} bytes, not the declared {request.BytesPerBlock}-byte block with its stated side geometry.",
                        nameof(request));
                }
            }
        }

        _backend = backend;
        _request = request;
        Tier = request.Tier;
        Execution = request.Execution;

        _pinnedBase = request.HostArenaBase;
        _pinnedBytes = request.HostArenaBytes;

        bool pinned = false;
        ulong allocated = 0;
        backend.EnterDeviceWork();
        try
        {
            CudaException.ThrowIf("cuMemHostRegister_v2", CudaNative.cuMemHostRegister((void*)_pinnedBase, (nuint)_pinnedBytes, 0));
            pinned = true;

            CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out allocated, (nuint)deviceBytes));
            _devicePointer = allocated;

            if (request.Execution == KvMirrorExecution.DeviceStep)
            {
                // The store starts byte-identical to the pool's own zeroed arena over the declared range: one residency
                // crossing at open, staged from the arena the mirror already pins, so a position a step has not written
                // yet reads back as the zeros the pool's arena holds — the same unwritten-slot posture every pool has.
                // The copy rides the owning backend's instance stream and joins it before answering: a device step that
                // consumes this store launches on that lane behind the upload, and the join keeps the pool free to write
                // its own arena again only after the bytes it staged have landed. A legacy-stream copy would order with
                // nothing here — an instance stream never rendezvouses with the default stream.
                long rangeOffset = checked((long)request.FirstBlockId * request.BytesPerBlock);
                CudaException.ThrowIf(
                    "cuMemcpyHtoDAsync_v2",
                    CudaNative.cuMemcpyHtoDAsync(allocated, (byte*)(_pinnedBase + rangeOffset), (nuint)deviceBytes, backend.ComputeStream));
                backend.SyncInstanceStream();
                for (int blockId = request.FirstBlockId; blockId < request.FirstBlockId + request.BlockCount; blockId++)
                {
                    _crossed.Add(blockId);
                }

                _crossedBytes = deviceBytes;
            }
        }
        catch
        {
            if (allocated != 0)
            {
                CudaNative.cuMemFree(allocated);
            }

            if (pinned)
            {
                CudaNative.cuMemHostUnregister((void*)_pinnedBase);
            }

            throw;
        }
        finally
        {
            backend.ExitDeviceWork();
        }

        backend.RegisterKvMirror(this);
    }

    public WeightTier Tier { get; }

    public KvMirrorExecution Execution { get; }

    public long DeviceBytes => _request.DeviceRangeBytes;

    public int CrossedBlocks => _crossed.Count;

    public long CrossedBytes => _crossedBytes;

    public int PendingWrites => _dirty.Count;

    public int DeviceWrittenBlocks => _deviceWritten.Count;

    public void MarkDirty(int blockId)
    {
        ThrowIfDisposed();
        ValidateRange(blockId, nameof(blockId));
        _dirty.Add(blockId);
    }

    /// <summary>
    /// Answers the store only in the device-step posture: a host-read mirror's copy is residency accounting no step is
    /// entitled to read, so it reports no store rather than handing out bytes no kernel produced.
    /// </summary>
    public bool TryGetDeviceStore(out KvDeviceStore store)
    {
        if (_request.Execution != KvMirrorExecution.DeviceStep || _disposed)
        {
            store = default;
            return false;
        }

        store = new KvDeviceStore
        {
            DeviceBase = _devicePointer,
            FirstBlockId = _request.FirstBlockId,
            BlockCount = _request.BlockCount,
            BlockTokens = _request.BlockTokens,
            LayerCount = _request.LayerCount,
            KvDim = _request.KvDim,
            BytesPerBlock = _request.BytesPerBlock,
            Combo = _request.Combo,
            KGroupBytes = _request.KGroupBytes,
            VGroupBytes = _request.VGroupBytes,
            GroupsPerTokenK = _request.GroupsPerTokenK,
            GroupsPerTokenV = _request.GroupsPerTokenV,
            KSideBytesPerLayer = _request.KSideBytesPerLayer,
            LayerRegionBytes = _request.LayerRegionBytes,
        };
        return true;
    }

    public void AccountDeviceWrite(int blockId)
    {
        ThrowIfDisposed();
        if (_request.Execution != KvMirrorExecution.DeviceStep)
        {
            throw new InvalidOperationException(
                $"block {blockId}: '{_request.ResourceName}' at tier {Tier} is a host-read mirror — its device copy is residency " +
                "accounting, not a store a device step writes, so a device-side write cannot be recorded against it.");
        }

        ValidateRange(blockId, nameof(blockId));
        _deviceWritten.Add(blockId);
    }

    public void ReadBlockToHost(int blockId, Span<float> output)
    {
        ThrowIfDisposed();
        if (_request.Execution != KvMirrorExecution.DeviceStep)
        {
            throw new InvalidOperationException(
                $"block {blockId}: '{_request.ResourceName}' at tier {Tier} is a host-read mirror — its host arena already is " +
                "the copy a host read uses, so its device copy is not a store to read back from.");
        }

        ValidateRange(blockId, nameof(blockId));
        if (output.Length * sizeof(float) < _request.BytesPerBlock)
        {
            throw new ArgumentException(
                $"reading block {blockId} of '{_request.ResourceName}' needs {checked(_request.BytesPerBlock / sizeof(float))} floats, got {output.Length}.",
                nameof(output));
        }

        long deviceOffset = (long)(blockId - _request.FirstBlockId) * _request.BytesPerBlock;
        _backend.EnterDeviceWork();
        try
        {
            fixed (float* p = output)
            {
                CudaException.ThrowIf(
                    "cuMemcpyDtoHAsync_v2",
                    CudaNative.cuMemcpyDtoHAsync((byte*)p, _devicePointer + (ulong)deviceOffset, (nuint)_request.BytesPerBlock, _backend.ComputeStream));
            }

            // The read-back rides the instance stream and joins it before answering: bytes a step kernel wrote on that
            // lane are ordered ahead of this copy, and they reach the caller's span only once the lane has drained.
            _backend.SyncInstanceStream();
        }
        finally
        {
            _backend.ExitDeviceWork();
        }
    }

    private void ValidateRange(int blockId, string paramName)
    {
        if ((uint)blockId - (uint)_request.FirstBlockId >= (uint)_request.BlockCount)
        {
            throw new ArgumentOutOfRangeException(
                paramName,
                $"block {blockId} lies outside the device-backed range [{_request.FirstBlockId}..{_request.FirstBlockId + _request.BlockCount - 1}] " +
                $"of '{_request.ResourceName}' at tier {Tier}.");
        }
    }

    /// <summary>
    /// Crosses every dirty block's whole payload to the device copy. A mirror already released — including one its
    /// backend let go before the pool that opened it reached its own teardown — has nothing left to cross and answers as
    /// a no-op instead of faulting a teardown path that only wanted the bytes current.
    /// </summary>
    public void Flush()
    {
        if (_disposed || _dirty.Count == 0)
        {
            return;
        }

        _backend.EnterDeviceWork();
        try
        {
            foreach (int blockId in _dirty)
            {
                long hostOffset = (long)blockId * _request.BytesPerBlock;
                long deviceOffset = (long)(blockId - _request.FirstBlockId) * _request.BytesPerBlock;
                CudaException.ThrowIf(
                    "cuMemcpyHtoDAsync_v2",
                    CudaNative.cuMemcpyHtoDAsync(
                        _devicePointer + (ulong)deviceOffset,
                        (byte*)(_request.HostArenaBase + hostOffset),
                        (nuint)_request.BytesPerBlock,
                        _backend.ComputeStream));

                _crossed.Add(blockId);
                _crossedBytes += _request.BytesPerBlock;
            }

            // Every crossing rides the owning backend's instance stream and this join closes it: a device step reading
            // these blocks launches on that lane behind them, and the pool regains write access to its own arena only
            // after what it staged here has landed. A legacy-stream copy would order with nothing under per-model streams.
            _backend.SyncInstanceStream();

            _dirty.Clear();
        }
        finally
        {
            _backend.ExitDeviceWork();
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        Flush(); // the device copy ends current: every pending crossing runs while the arena this mirror pins is still alive
        _disposed = true;

        _backend.EnterDeviceWork();
        try
        {
            // Join before releasing either side of the pairing: a host region unregistered or a device buffer freed while
            // an enqueued copy still reads them would strand that copy against memory this seam no longer owns. The lane
            // join is what makes both releases safe without reaching for the whole context.
            _backend.SyncInstanceStream();

            CudaNative.cuMemFree(_devicePointer);
            CudaNative.cuMemHostUnregister((void*)_pinnedBase);
        }
        finally
        {
            _backend.ExitDeviceWork();
        }

        _backend.UnregisterKvMirror(this);
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, typeof(CudaKvMirror));
}
