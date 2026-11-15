namespace Lair.Backends.Cuda;

/// <summary>The driver's own result vocabulary, named unchanged on a failure report.</summary>
public enum CudaResult
{
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    DeviceUnavailable = 100,
    NoDevice = 101,
    InvalidDevice = 102,
    InvalidImage = 200,
    InvalidContext = 201,
    ContextAlreadyCurrent = 202,
    MapFailed = 205,
    UnmapFailed = 206,
    NoBinaryForGpu = 209,
    InvalidPtx = 218,
    InvalidSource = 300,
    FileNotFound = 301,
    SharedObjectSymbolNotFound = 302,
    SharedObjectInitFailed = 303,
    OperatingSystem = 304,
    InvalidHandle = 400,
    NotFound = 500,
    NotReady = 600,
    NotSupported = 801,
    Unknown = 999,
}

/// <summary>The driver seam's own failure report: a CUDA call reached the driver and answered with an error. A host that opens this backend for a declared layer split names it when no usable device stands behind the seam.</summary>
public sealed class CudaException : Exception
{
    public CudaException(string function, CudaResult result)
        : base($"CUDA driver seam failure in {function}: {Describe(result)} ({(int)result}). nvcuda.dll was loaded but the call failed; ensure a CUDA-capable device and a driver exposing the CUDA Driver API are present.")
    {
        Function = function;
        Result = result;
    }

    public string Function { get; }

    public CudaResult Result { get; }

    public static void ThrowIf(string function, CudaResult result)
    {
        if (result != CudaResult.Success)
        {
            throw new CudaException(function, result);
        }
    }

    private static string Describe(CudaResult result) => result switch
    {
        CudaResult.InvalidValue => "CUDA_ERROR_INVALID_VALUE",
        CudaResult.OutOfMemory => "CUDA_ERROR_OUT_OF_MEMORY",
        CudaResult.NotInitialized => "CUDA_ERROR_NOT_INITIALIZED",
        CudaResult.Deinitialized => "CUDA_ERROR_DEINITIALIZED",
        CudaResult.DeviceUnavailable => "CUDA_ERROR_DEVICE_UNAVAILABLE",
        CudaResult.NoDevice => "CUDA_ERROR_NO_DEVICE",
        CudaResult.InvalidDevice => "CUDA_ERROR_INVALID_DEVICE",
        CudaResult.InvalidImage => "CUDA_ERROR_INVALID_IMAGE",
        CudaResult.InvalidContext => "CUDA_ERROR_INVALID_CONTEXT",
        CudaResult.ContextAlreadyCurrent => "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
        CudaResult.MapFailed => "CUDA_ERROR_MAP_FAILED",
        CudaResult.UnmapFailed => "CUDA_ERROR_UNMAP_FAILED",
        CudaResult.NoBinaryForGpu => "CUDA_ERROR_NO_BINARY_FOR_GPU",
        CudaResult.InvalidPtx => "CUDA_ERROR_INVALID_PTX",
        CudaResult.InvalidSource => "CUDA_ERROR_INVALID_SOURCE",
        CudaResult.FileNotFound => "CUDA_ERROR_FILE_NOT_FOUND",
        CudaResult.SharedObjectSymbolNotFound => "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
        CudaResult.SharedObjectInitFailed => "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
        CudaResult.OperatingSystem => "CUDA_ERROR_OPERATING_SYSTEM",
        CudaResult.InvalidHandle => "CUDA_ERROR_INVALID_HANDLE",
        CudaResult.NotFound => "CUDA_ERROR_NOT_FOUND",
        CudaResult.NotReady => "CUDA_ERROR_NOT_READY",
        CudaResult.NotSupported => "CUDA_ERROR_NOT_SUPPORTED",
        _ => "CUDA_ERROR_UNKNOWN",
    };
}
