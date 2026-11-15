namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    private readonly Dictionary<(nint Pointer, long Length), ulong> _deviceWeights = [];
    private readonly List<ulong> _weightAllocations = [];

    /// <summary>Weight tensors this instance has copied to device memory so far.</summary>
    public int DeviceWeightCopies => _deviceWeights.Count;

    // The copy residency: each (pointer, length) tensor is allocated once on the device and filled by one host-to-device
    // copy, cached for the instance's lifetime, and freed at dispose.
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
