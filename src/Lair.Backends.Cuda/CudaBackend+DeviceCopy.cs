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
            CudaException.ThrowIf("cuMemcpyHtoD_v2", CudaNative.cuMemcpyHtoD(devicePointer, (byte*)hostPointer, bytes));
        }

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
