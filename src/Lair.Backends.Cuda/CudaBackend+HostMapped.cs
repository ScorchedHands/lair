namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    private readonly Dictionary<(nint Pointer, long Length), ulong> _hostMappedWeights = [];
    private readonly List<(nint Pointer, long Length)> _registeredHostRegions = [];

    /// <summary>Host weight regions this instance has registered with the driver so far — the mapped posture's residency accounting, which stays at zero copies crossed to device.</summary>
    public int HostMappedWeightRegions => _hostMappedWeights.Count;

    // The mapped residency: the host region the weight bytes already live in is registered once per (pointer, length)
    // and the kernel launches against the device pointer the driver answers for it, so no weight tensor is copied into
    // persistent device memory and the copy counter stays at zero. A region the driver already knows is answered from
    // the cache rather than registered twice.
    private ulong EnsureHostMappedWeights(nint hostPointer, nuint bytes)
    {
        if (_hostMappedWeights.TryGetValue((hostPointer, (long)bytes), out ulong existing))
        {
            return existing;
        }

        unsafe
        {
            CudaException.ThrowIf("cuMemHostRegister_v2", CudaNative.cuMemHostRegister((void*)hostPointer, bytes, 0));
        }

        _registeredHostRegions.Add((hostPointer, (long)bytes));

        try
        {
            ulong devicePointer;
            unsafe
            {
                CudaException.ThrowIf("cuMemHostGetDevicePointer_v2", CudaNative.cuMemHostGetDevicePointer(out devicePointer, (void*)hostPointer, 0));
            }

            _hostMappedWeights[(hostPointer, (long)bytes)] = devicePointer;
            return devicePointer;
        }
        catch
        {
            _registeredHostRegions.RemoveAt(_registeredHostRegions.Count - 1);
            unsafe
            {
                CudaNative.cuMemHostUnregister((void*)hostPointer);
            }

            throw;
        }
    }

    private void ReleaseHostMappedRegions()
    {
        // Each registered host region is released exactly once, and only its own registration is touched: the host memory itself stays owned by whatever source handed it out.
        foreach ((nint hostPointer, long _) in _registeredHostRegions)
        {
            unsafe
            {
                CudaNative.cuMemHostUnregister((void*)hostPointer);
            }
        }

        _registeredHostRegions.Clear();
        _hostMappedWeights.Clear();
    }
}
