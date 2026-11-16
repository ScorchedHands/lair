using System.Buffers;
using System.Runtime.InteropServices;

using Lair.Contracts;

namespace Lair.Backends.Cuda;

/// <summary>
/// CUDA compute backend: packed weight tensors reach device memory one of two ways per instance. The default posture
/// copies each tensor to device memory once per instance and every dense GEMV, single or batched, runs as an embedded
    /// PTX kernel against those copies for F32, F16, BF16, Q4_K, and Q6_K weights. The host-mapped posture instead registers each
/// tensor's live host region with the driver and launches the same kernels against the device pointer the registration
/// answers with, so no weight tensor is ever copied into persistent device memory and the weights stay resident exactly
/// where the weight source holds them. Formats with no kernel here refuse naming this backend rather than run through
/// some other route; the blocked matmul has no CUDA path at all yet. Activation vectors and outputs cross the PCIe
/// seam per call in either posture, so KV arenas and split-route staging stay host-resident aligned memory under this
/// device id until accelerator residency lands elsewhere.
/// </summary>
public sealed partial class CudaBackend : IBackend
{
    /// <summary>The accelerator device id every CUDA-backed route segment names.</summary>
    public const string DeviceIdValue = "cuda";

    private const int KernelThreads = 128;

    private readonly object _gate = new();
    private readonly CudaWeightResidency _weightResidency;
    private readonly int _device;
    private readonly IntPtr _context;
    private readonly IntPtr _module;
    private readonly IntPtr _gemvF32;
    private readonly IntPtr _gemvF16;
    private readonly IntPtr _gemvBf16;
    private readonly IntPtr _gemvQ4K;
    private readonly IntPtr _gemvQ6K;

    private bool _disposed;

    /// <summary>The weight residency posture this instance runs: the persistent device copy it opens with by default, or the host-mapped seam that launches against registered host regions.</summary>
    public CudaWeightResidency WeightResidency => _weightResidency;

    /// <summary>The tier this instance's kernels read weight bytes from: the device copies the default posture owns are the accelerator tier, and the mapped posture reads the host regions a weight source owns, so it reports the system-memory tier.</summary>
    public WeightTier WeightTier => _weightResidency == CudaWeightResidency.DeviceCopy
        ? Lair.Contracts.WeightTier.Vram
        : Lair.Contracts.WeightTier.Dram;

    public string DeviceId => DeviceIdValue;

    public CudaBackend()
        : this(CudaWeightResidency.DeviceCopy)
    {
    }

    /// <summary>
    /// Opens the backend with an explicit weight residency posture. The default posture copies each weight tensor to
    /// device memory once per instance; <see cref="CudaWeightResidency.HostMapped"/> never copies weight tensors into
    /// persistent device memory — it registers each tensor's live host region with the driver and launches against the
    /// device pointer the registration answers with, so the weights stay resident in the host memory the weight source
    /// owns. A mapped region must stay valid and page-aligned for the backend's whole lifetime, and no other
    /// registration may cover its pages while it is registered.
    /// </summary>
    public CudaBackend(CudaWeightResidency weightResidency)
    {
        _weightResidency = weightResidency;

        CudaException.ThrowIf("cuInit", CudaNative.cuInit(0));
        CudaException.ThrowIf("cuDeviceGetCount", CudaNative.cuDeviceGetCount(out int deviceCount));
        if (deviceCount < 1)
        {
            throw new CudaException("cuDeviceGetCount", CudaResult.NoDevice);
        }

        CudaException.ThrowIf("cuDeviceGet", CudaNative.cuDeviceGet(out _device, 0));
        CudaException.ThrowIf("cuDevicePrimaryCtxRetain", CudaNative.cuDevicePrimaryCtxRetain(out _context, _device));
        try
        {
            SetCurrentContext();

            unsafe
            {
                fixed (byte* ptxPtr = PtxBytes)
                {
                    CudaException.ThrowIf("cuModuleLoadData", CudaNative.cuModuleLoadData(out IntPtr module, ptxPtr));
                    _module = module;
                }
            }

            try
            {
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvF32, _module, "gemv_f32"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvF16, _module, "gemv_f16"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvBf16, _module, "gemv_bf16"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvQ4K, _module, "gemv_q4k"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvQ6K, _module, "gemv_q6k"));
            }
            catch
            {
                CudaNative.cuModuleUnload(_module);
                throw;
            }
        }
        catch
        {
            CudaException.ThrowIf("cuDevicePrimaryCtxRelease", CudaNative.cuDevicePrimaryCtxRelease(_device));
            throw;
        }
    }

    public unsafe MemoryHandle AllocateAligned(long bytes, int alignment)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0)
        {
            throw new ArgumentException("Alignment must be a positive power of two.", nameof(alignment));
        }

        void* pointer = NativeMemory.AlignedAlloc((nuint)bytes, (nuint)alignment);
        if (pointer is null)
        {
            throw new OutOfMemoryException($"Failed to allocate {bytes} aligned bytes.");
        }

        return new MemoryHandle(pointer);
    }

    public unsafe void FreeAligned(MemoryHandle handle) => NativeMemory.AlignedFree(handle.Pointer);

    public void Gemv(ReadOnlySpan<float> x, WeightRef w, Span<float> output)
    {
        ThrowIfDisposed();

        (int rows, int cols) = ValidateShape(w);

        if (x.Length != cols)
        {
            throw new ArgumentException($"Input vector length {x.Length} must equal the weight column count {cols}.", nameof(x));
        }

        if (output.Length < rows)
        {
            throw new ArgumentException($"Output span length {output.Length} is shorter than the weight row count {rows}.", nameof(output));
        }

        RunGemv(w, x, 1, output);
    }

    public void GemvBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs)
    {
        ThrowIfDisposed();

        (int rows, int cols) = ValidateShape(w);
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);

        if ((long)inputs.Length != (long)count * cols)
        {
            throw new ArgumentException(
                $"Batched input length {inputs.Length} must equal count {count} times the weight column count {cols}.",
                nameof(inputs));
        }

        if ((long)outputs.Length < (long)count * rows)
        {
            throw new ArgumentException(
                $"Batched output length {outputs.Length} is shorter than count {count} times the weight row count {rows}.",
                nameof(outputs));
        }

        RunGemv(w, inputs, count, outputs);
    }

    public void GemmBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs) =>
        throw new NotSupportedException(
            $"CudaBackend ('{DeviceIdValue}') has no blocked matmul kernel for format {w.Format}: the CUDA seam carries " +
            "GEMV kernels only, so a dense-matrix prefill selection is refused rather than run through some other route.");

    public bool SupportsGemm(QuantFormat format) => false;

    // Row dequantization stays host-side: it expands one row for embedding lookups and norm vectors, not a matrix
    // traversal, with the same scalar math every backend already agrees on. Formats without an expansion here refuse
    // naming this backend instead of borrowing another device's kernel.
    public unsafe void DequantRow(WeightRef w, int row, Span<float> output)
    {
        ThrowIfDisposed();

        (int rows, int cols) = ValidateShape(w);

        if ((uint)row >= (uint)rows)
        {
            throw new ArgumentOutOfRangeException(nameof(row), $"Row {row} is outside the weight of {rows} rows.");
        }

        if (output.Length < cols)
        {
            throw new ArgumentException($"Output span length {output.Length} is shorter than the weight column count {cols}.", nameof(output));
        }

        long stride = w.Shape.RowStrideBytes(w.Format);
        fixed (byte* pBytes = w.Bytes)
        {
            byte* rowPtr = pBytes + (long)row * stride;
            switch (w.Format)
            {
                case QuantFormat.F32:
                    DequantF32Row(rowPtr, cols, output);
                    break;
                case QuantFormat.F16:
                    DequantF16Row(rowPtr, cols, output);
                    break;
                case QuantFormat.BF16:
                    DequantBf16Row(rowPtr, cols, output);
                    break;
                case QuantFormat.Q4_K:
                    DequantQ4KRow(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
                    break;
                case QuantFormat.Q6_K:
                    DequantQ6KRow(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
                    break;
                default:
                    throw Unsupported(w.Format, "row dequantization");
            }
        }
    }

    public unsafe float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Dot operands must have the same length.", nameof(b));
        }

        fixed (float* pa = a)
        fixed (float* pb = b)
        {
            float acc = 0f;
            for (int i = 0; i < a.Length; i++)
            {
                acc += pa[i] * pb[i];
            }

            return acc;
        }
    }

    public unsafe void Axpy(float alpha, ReadOnlySpan<float> x, Span<float> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Axpy operands must have the same length.", nameof(y));
        }

        fixed (float* px = x)
        fixed (float* py = y)
        {
            for (int i = 0; i < x.Length; i++)
            {
                py[i] += alpha * px[i];
            }
        }
    }

    public void Fill(Span<float> y, float v) => y.Fill(v);

    public unsafe void Scale(Span<float> y, float s)
    {
        fixed (float* py = y)
        {
            for (int i = 0; i < y.Length; i++)
            {
                py[i] *= s;
            }
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            SetCurrentContext();

            FreeDeviceWeightAllocations();
            ReleaseHostMappedRegions();
        }

        CudaException.ThrowIf("cuModuleUnload", CudaNative.cuModuleUnload(_module));
        CudaException.ThrowIf("cuDevicePrimaryCtxRelease", CudaNative.cuDevicePrimaryCtxRelease(_device));
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CudaBackend));
        }
    }

    private void RunGemv(WeightRef w, ReadOnlySpan<float> activations, int count, Span<float> output)
    {
        (int rows, int cols) = ValidateShape(w);
        IntPtr function = KernelFor(w.Format); // refuses formats with no kernel before any device work starts

        nuint weightBytes = checked((nuint)(rows * w.Shape.RowStrideBytes(w.Format)));
        nuint xBytes = checked((nuint)count * (nuint)cols * sizeof(float));
        nuint outBytes = checked((nuint)count * (nuint)rows * sizeof(float));

        lock (_gate)
        {
            SetCurrentContext(); // the driver API keeps the current context per thread, so every call re-binds this instance's retained primary one

            ulong deviceWeights = 0;
            unsafe
            {
                fixed (byte* wBytes = w.Bytes)
                {
                    // The mapped posture launches against the device pointer of a registered host region; the default posture keeps its once-copied device buffer. Either answer is pinned for exactly this call and cached per tensor for the instance's lifetime.
                    deviceWeights = _weightResidency == CudaWeightResidency.HostMapped
                        ? EnsureHostMappedWeights((nint)wBytes, weightBytes)
                        : EnsureDeviceWeights((nint)wBytes, weightBytes); // pinned for exactly this call; the copy runs now and never again per tensor
                }
            }

            ulong deviceX = 0;
            ulong deviceOut = 0;
            try
            {
                CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out deviceX, xBytes));
                CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out deviceOut, outBytes));

                unsafe
                {
                    fixed (float* pIn = activations)
                    {
                        CudaException.ThrowIf("cuMemcpyHtoD_v2", CudaNative.cuMemcpyHtoD(deviceX, (byte*)pIn, xBytes));
                    }

                    uint rowsU = (uint)rows;
                    uint colsU = (uint)cols;
                    uint countU = (uint)count;
                    void** kernelParams = stackalloc void*[6];
                    kernelParams[0] = &deviceWeights;
                    kernelParams[1] = &deviceX;
                    kernelParams[2] = &deviceOut;
                    kernelParams[3] = &rowsU;
                    kernelParams[4] = &colsU;
                    kernelParams[5] = &countU;

                    uint blocks = (uint)((rows + KernelThreads - 1) / KernelThreads);
                    CudaException.ThrowIf("cuLaunchKernel", CudaNative.cuLaunchKernel(function, blocks, 1, 1, KernelThreads, 1, 1, 0, IntPtr.Zero, kernelParams, null));
                    CudaException.ThrowIf("cuCtxSynchronize", CudaNative.cuCtxSynchronize());

                    fixed (float* pOut = output)
                    {
                        CudaException.ThrowIf("cuMemcpyDtoH_v2", CudaNative.cuMemcpyDtoH((byte*)pOut, deviceOut, outBytes));
                    }
                }
            }
            finally
            {
                if (deviceOut != 0)
                {
                    CudaNative.cuMemFree(deviceOut);
                }

                if (deviceX != 0)
                {
                    CudaNative.cuMemFree(deviceX);
                }
            }
        }
    }

    private IntPtr KernelFor(QuantFormat format) => format switch
    {
        QuantFormat.F32 => _gemvF32,
        QuantFormat.F16 => _gemvF16,
        QuantFormat.BF16 => _gemvBf16,
        QuantFormat.Q4_K => _gemvQ4K,
        QuantFormat.Q6_K => _gemvQ6K,
        _ => throw Unsupported(format, "GEMV"),
    };

    private static NotSupportedException Unsupported(QuantFormat format, string kernel) => new(
        $"CudaBackend ('{DeviceIdValue}') has no {kernel} kernel for format {format}: the CUDA seam carries F32, F16, BF16, Q4_K, and Q6_K kernels only, " +
        "so this weight pass is refused naming that backend rather than run through a CPU fallback.");

    private static (int Rows, int Cols) ValidateShape(WeightRef w)
    {
        int[] dims = w.Shape.Dims;
        if (dims.Length < 2 || dims[0] <= 0 || dims[^1] <= 0)
        {
            throw new ArgumentException("Weight shape must be a matrix with positive row and column counts.", nameof(w));
        }

        int rows = dims[0];
        int cols = dims[^1];

        long strideBytes = w.Shape.RowStrideBytes(w.Format);
        if (strideBytes > int.MaxValue)
        {
            throw new ArgumentException("Row byte size exceeds the addressable span limit.", nameof(w));
        }

        long expectedBytes = rows * strideBytes;
        if ((long)w.Bytes.Length != expectedBytes)
        {
            throw new ArgumentException(
                $"WeightRef byte length is {w.Bytes.Length} but shape {rows}x{cols} in format {w.Format} " +
                $"requires {expectedBytes} bytes ({rows} rows x {strideBytes}-byte row stride).",
                nameof(w));
        }

        return (rows, cols);
    }

    private void SetCurrentContext() => CudaException.ThrowIf("cuCtxSetCurrent", CudaNative.cuCtxSetCurrent(_context));
}
