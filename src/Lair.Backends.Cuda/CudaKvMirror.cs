using Lair.Contracts;

namespace Lair.Backends.Cuda;

/// <summary>
/// The CUDA-backed device mirror over a host KV arena another owner holds. The arena is pinned with the driver from its
/// first page through its last, and a persistent device allocation is opened for exactly one whole block per
/// device-backed block id; a flush copies each dirty block's whole host payload into that allocation with a
/// host-to-device copy, so the device-resident copy of the declared range tracks the host writes the pool makes. The
/// arena stays owned and readable by the caller — attention keeps reading it unchanged — and nothing on the device is
/// implied to read the device copy: crossing a block's bytes is residency, not a compute path.
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

        _backend = backend;
        _request = request;
        Tier = request.Tier;

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

    public long DeviceBytes => _request.DeviceRangeBytes;

    public int CrossedBlocks => _crossed.Count;

    public long CrossedBytes => _crossedBytes;

    public int PendingWrites => _dirty.Count;

    public void MarkDirty(int blockId)
    {
        ThrowIfDisposed();
        if ((uint)blockId - (uint)_request.FirstBlockId >= (uint)_request.BlockCount)
        {
            throw new ArgumentOutOfRangeException(
                nameof(blockId),
                $"block {blockId} lies outside the device-backed range [{_request.FirstBlockId}..{_request.FirstBlockId + _request.BlockCount - 1}] " +
                $"of '{_request.ResourceName}' at tier {Tier}.");
        }

        _dirty.Add(blockId);
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
                    "cuMemcpyHtoD_v2",
                    CudaNative.cuMemcpyHtoD(
                        _devicePointer + (ulong)deviceOffset,
                        (byte*)(_request.HostArenaBase + hostOffset),
                        (nuint)_request.BytesPerBlock));

                _crossed.Add(blockId);
                _crossedBytes += _request.BytesPerBlock;
            }

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
