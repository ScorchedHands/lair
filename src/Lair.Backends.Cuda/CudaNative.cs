using System.Runtime.InteropServices;

namespace Lair.Backends.Cuda;

internal static partial class CudaNative
{
    [LibraryImport("nvcuda", EntryPoint = "cuInit")]
    public static partial CudaResult cuInit(uint flags);

    [LibraryImport("nvcuda", EntryPoint = "cuDeviceGetCount")]
    public static partial CudaResult cuDeviceGetCount(out int count);

    [LibraryImport("nvcuda", EntryPoint = "cuDeviceGet")]
    public static partial CudaResult cuDeviceGet(out int device, int ordinal);

    [LibraryImport("nvcuda", EntryPoint = "cuDevicePrimaryCtxRetain")]
    public static partial CudaResult cuDevicePrimaryCtxRetain(out IntPtr context, int device);

    [LibraryImport("nvcuda", EntryPoint = "cuCtxSetCurrent")]
    public static partial CudaResult cuCtxSetCurrent(IntPtr context);

    [LibraryImport("nvcuda", EntryPoint = "cuDevicePrimaryCtxRelease")]
    public static partial CudaResult cuDevicePrimaryCtxRelease(int device);

    [LibraryImport("nvcuda", EntryPoint = "cuModuleLoadData")]
    public static unsafe partial CudaResult cuModuleLoadData(out IntPtr module, byte* image);

    [LibraryImport("nvcuda", EntryPoint = "cuModuleGetFunction")]
    public static partial CudaResult cuModuleGetFunction(out IntPtr function, IntPtr module, [MarshalAs(UnmanagedType.LPStr)] string name);

    [LibraryImport("nvcuda", EntryPoint = "cuMemAlloc_v2")]
    public static partial CudaResult cuMemAlloc(out ulong devicePointer, nuint bytes);

    [LibraryImport("nvcuda", EntryPoint = "cuMemcpyHtoD_v2")]
    public static unsafe partial CudaResult cuMemcpyHtoD(ulong destination, byte* source, nuint bytes);

    [LibraryImport("nvcuda", EntryPoint = "cuMemcpyDtoH_v2")]
    public static unsafe partial CudaResult cuMemcpyDtoH(byte* destination, ulong source, nuint bytes);

    [LibraryImport("nvcuda", EntryPoint = "cuLaunchKernel")]
    public static unsafe partial CudaResult cuLaunchKernel(IntPtr function, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, uint sharedBytes, IntPtr stream, void** kernelParams, void** extra);

    [LibraryImport("nvcuda", EntryPoint = "cuCtxSynchronize")]
    public static partial CudaResult cuCtxSynchronize();

    [LibraryImport("nvcuda", EntryPoint = "cuMemFree_v2")]
    public static partial CudaResult cuMemFree(ulong devicePointer);

    [LibraryImport("nvcuda", EntryPoint = "cuModuleUnload")]
    public static partial CudaResult cuModuleUnload(IntPtr module);
}
