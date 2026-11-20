namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    private readonly Dictionary<(nint Pointer, long Length), ulong> _deviceWeights = [];
    private readonly List<ulong> _weightAllocations = [];

    /// <summary>Weight tensors this instance has copied to device memory so far.</summary>
    public int DeviceWeightCopies => _deviceWeights.Count;

    // The persistent residency accounting: every copy cached here is one allocation the instance holds for its whole lifetime and releases together at dispose. Each cache entry records the exact length its device buffer was allocated with, so the byte total reads off the same tracking that answers weight passes — no separate ledger to keep in step.
    /// <summary>Persistent device weight allocations this instance currently holds open: every copy made so far, freed as a set at dispose.</summary>
    public int PersistentDeviceWeightAllocations => _weightAllocations.Count;

    /// <summary>The device bytes these persistent weight copies hold: the sum over every cached tensor at its allocated length. A host-mapped posture records no allocation in this tracking and reports zero — its weights never enter persistent device memory.</summary>
    public long PersistentDeviceWeightBytes => _deviceWeights.Sum(entry => entry.Key.Length);

    // The copy residency: each (pointer, length) tensor is allocated once on the device and filled by one host-to-device
    // copy, cached for the instance's lifetime, and freed at dispose. This upload is persistent weight residency, not a
    // compute call's traffic — it is deliberately absent from the runtime transfer tallies, which answer only activation
    // inputs and kernel outputs crossing during a GEMV. Every byte held here reads off PersistentDeviceWeightBytes for as
    // long as its allocation stands, and drops out of that reading when the instance releases it.
    //
    // The fill rides the instance stream and joins it before answering, which is what keeps a resident tensor safe under
    // per-model streams: the copy that fills this region is enqueued ahead of every launch that will read it on this
    // lane, and the join guarantees the bytes are on the device before the caller's kernel can consume them. A
    // synchronous copy would land on the legacy default stream instead, and a non-blocking instance stream never
    // rendezvouses with that stream — a launch enqueued right after such a copy could read the region before its upload
    // had landed. The join also bounds the source's lifetime: the caller pins the weight bytes only for this call, so the
    // upload must be complete when the call answers. The allocation itself stays classic (one-time, per tensor): a
    // classic allocation is usable on every stream the moment it returns, and the buffer lives until dispose.
    private ulong EnsureDeviceWeights(nint hostPointer, nuint bytes)
    {
        if (_deviceWeights.TryGetValue((hostPointer, (long)bytes), out ulong existing))
        {
            return existing;
        }

        CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out ulong devicePointer, bytes));
        _weightAllocations.Add(devicePointer);
        _deviceWeights[(hostPointer, (long)bytes)] = devicePointer;

        unsafe
        {
            CudaException.ThrowIf("cuMemcpyHtoDAsync_v2", CudaNative.cuMemcpyHtoDAsync(devicePointer, (byte*)hostPointer, bytes, _stream));
        }

        SyncInstanceStream(); // the resident region is device-complete before any launch enqueued on this lane can read it

        return devicePointer;
    }

    private void FreeDeviceWeightAllocations()
    {
        foreach (ulong devicePointer in _weightAllocations)
        {
            CudaNative.cuMemFree(devicePointer);
        }

        _weightAllocations.Clear();
        _deviceWeights.Clear();
    }
}
