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

    // Per-instance stream seam: each backend owns one non-blocking compute/copy stream on the shared primary context,
    // so its launches and async staging copies queue independently of every other instance's stream and never rendezvous
    // implicitly with the legacy NULL/default stream. Creation flags carry CU_STREAM_NON_BLOCKING (0x1).
    [LibraryImport("nvcuda", EntryPoint = "cuStreamCreate")]
    public static partial CudaResult cuStreamCreate(out IntPtr stream, uint flags);

    [LibraryImport("nvcuda", EntryPoint = "cuStreamDestroy_v2")]
    public static partial CudaResult cuStreamDestroy(IntPtr stream);

    [LibraryImport("nvcuda", EntryPoint = "cuStreamSynchronize")]
    public static partial CudaResult cuStreamSynchronize(IntPtr stream);

    // Answers whether everything enqueued on the stream has completed: Success when drained, CUDA_ERROR_NOT_READY while
    // work still stands. The concurrency gate reads it to prove two instances' device work is in flight together; no
    // production path consults it.
    [LibraryImport("nvcuda", EntryPoint = "cuStreamQuery")]
    public static partial CudaResult cuStreamQuery(IntPtr stream);

    [LibraryImport("nvcuda", EntryPoint = "cuMemAllocHost_v2")]
    public static unsafe partial CudaResult cuMemAllocHost(out void* hostPointer, nuint bytes);

    [LibraryImport("nvcuda", EntryPoint = "cuMemFreeHost")]
    public static unsafe partial CudaResult cuMemFreeHost(void* hostPointer);

    // Stream-ordered staging copies against pinned host memory: enqueued on the instance's own stream and awaited by a
    // per-stream join, so an activation input upload and a kernel output download queue without blocking either lane.
    [LibraryImport("nvcuda", EntryPoint = "cuMemcpyHtoDAsync_v2")]
    public static unsafe partial CudaResult cuMemcpyHtoDAsync(ulong destination, byte* source, nuint bytes, IntPtr stream);

    [LibraryImport("nvcuda", EntryPoint = "cuMemcpyDtoHAsync_v2")]
    public static unsafe partial CudaResult cuMemcpyDtoHAsync(byte* destination, ulong source, nuint bytes, IntPtr stream);

    [LibraryImport("nvcuda", EntryPoint = "cuMemFree_v2")]
    public static partial CudaResult cuMemFree(ulong devicePointer);

    // Stream-ordered allocation: the request and the release each land in the queue of the stream handed to them, so a
    // transient buffer allocated here is consumed only by work enqueued behind it on that same lane and released back
    // into that lane's pool after the last enqueued user, with no device-wide rendezvous anywhere in the hot path.
    [LibraryImport("nvcuda", EntryPoint = "cuMemAllocAsync")]
    public static partial CudaResult cuMemAllocAsync(out ulong devicePointer, nuint bytes, IntPtr stream);

    [LibraryImport("nvcuda", EntryPoint = "cuMemFreeAsync")]
    public static partial CudaResult cuMemFreeAsync(ulong devicePointer, IntPtr stream);

    // The device's default allocation pool, whose release threshold is raised once per backend so stream-ordered frees
    // keep their chunks for the next call instead of handing them back to the OS at every lane join.
    [LibraryImport("nvcuda", EntryPoint = "cuDeviceGetDefaultMemPool")]
    public static partial CudaResult cuDeviceGetDefaultMemPool(out IntPtr pool, int device);

    [LibraryImport("nvcuda", EntryPoint = "cuMemPoolSetAttribute")]
    public static unsafe partial CudaResult cuMemPoolSetAttribute(IntPtr pool, int attribute, void* value);

    [LibraryImport("nvcuda", EntryPoint = "cuMemHostRegister_v2")]
    public static unsafe partial CudaResult cuMemHostRegister(void* hostPointer, nuint bytes, uint flags);

    [LibraryImport("nvcuda", EntryPoint = "cuMemHostGetDevicePointer_v2")]
    public static unsafe partial CudaResult cuMemHostGetDevicePointer(out ulong devicePointer, void* hostPointer, uint flags);

    [LibraryImport("nvcuda", EntryPoint = "cuMemHostUnregister")]
    public static unsafe partial CudaResult cuMemHostUnregister(void* hostPointer);

    [LibraryImport("nvcuda", EntryPoint = "cuModuleUnload")]
    public static partial CudaResult cuModuleUnload(IntPtr module);

    // Device-side event timing, recorded on the same stream every launch here already uses. These answer how long a
    // kernel ran as the device measured it — the seam engine accounting reads instead of any OS-level utilization view.
    [LibraryImport("nvcuda", EntryPoint = "cuEventCreate")]
    public static partial CudaResult cuEventCreate(out IntPtr handle, uint flags);

    [LibraryImport("nvcuda", EntryPoint = "cuEventRecord")]
    public static partial CudaResult cuEventRecord(IntPtr handle, IntPtr stream);

    // Answers whether a recorded event has completed: CUDA_ERROR_NOT_READY while anything enqueued before it on that
    // stream still runs, Success once it has. The concurrency gate's pending probe rides on this.
    [LibraryImport("nvcuda", EntryPoint = "cuEventQuery")]
    public static partial CudaResult cuEventQuery(IntPtr handle);

    [LibraryImport("nvcuda", EntryPoint = "cuEventElapsedTime")]
    public static partial CudaResult cuEventElapsedTime(out float milliseconds, IntPtr start, IntPtr end);

    [LibraryImport("nvcuda", EntryPoint = "cuEventDestroy")]
    public static partial CudaResult cuEventDestroy(IntPtr handle);
}
