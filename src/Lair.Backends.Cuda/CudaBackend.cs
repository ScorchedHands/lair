using System.Buffers;
using System.Runtime.InteropServices;

using Lair.Contracts;

namespace Lair.Backends.Cuda;

/// <summary>
    /// CUDA compute backend: packed weight tensors reach device memory one of two ways per instance. The default posture
    /// copies each tensor to device memory once per instance and every dense GEMV, single or batched, runs as an embedded
    /// PTX kernel against those copies for F32, F16, BF16, Q8_0, Q5_0, Q2_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, IQ4_NL, and IQ4_XS
    /// weights - the two IQ4 layouts dot against FP activations with fused nibble-table lookups against pinned scratch - plus
    /// the five sub-4-bit i-quants (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, and IQ3_S), whose kernels instead consume a host-side Q8_K
    /// quantization of each activation vector crossed over PCIe beside their pinned grid tables. The host-mapped posture instead registers each

/// tensor's live host region with the driver and launches the same kernels against the device pointer the registration
/// answers with, so no weight tensor is ever copied into persistent device memory and the weights stay resident exactly
/// where the weight source holds them. Formats with no kernel here refuse naming this backend rather than run through
/// some other route; blocked matmul runs on the device for each format whose windowed kernel stands today — Q8_0/Q5_0/
/// Q2_0, every K-quant, and the IQ4 pair (each one its own GEMV kernel's unpacking walked once per activation-row window
/// instead of once per vector) plus F32/F16/BF16 (each a hand-tiled embedded-PTX dense matmul that walks that format's own
/// GEMV weight widening — plain f32 loads, integer-bit half widening, or the bf16 high-half shift — once per window with no
/// precision conversion), all in a `GemmBatch` entry point beside the GEMV. The Q8_K i-quants (IQ2_XXS/Q2_XS/IQ2_S/
/// IQ3_XXS/IQ3_S) carry no blocked matmul yet, so their dense-matrix selection is refused naming this backend. No native
/// math library is linked anywhere on this seam: every kernel here is embedded PTX in the `CudaBackend+Ptx` family.
/// Activation vectors and outputs cross the PCIe
/// seam per call in either posture, and a KV arena stays host-resident for attention reads in both: the device KV mirror
/// seam opens a persistent device copy of a declared block range that dirty blocks cross on flush, which is residency
/// accounting only — no device-side attention kernel stands behind a mirror, and opening one implies none. Beside the
/// weight-pass family this seam also carries the on-device transformer-step family: <see cref="OpenDeviceStep"/> opens
/// persistent device buffers for one step's rotary tables, staged window rows, and per-layer KV arena, and the step's
/// rotation, KV write, and causal <c>softmax(QK^T)V</c> walk run against those buffers as embedded PTX, so a dense step
/// crosses the seam once for its staged window and once for its attention result rather than per layer. That arena is
/// the step's own storage, not a mirror: a mirror still holds bytes for residency, and no kernel here reads one. A step
/// opened over a mirror's device store — the device-step execution posture of a <c>vram</c> or <c>split</c> KV residency
/// — takes that reconciliation further: its KV write and attention walk address the pool's own resident blocks through
/// the pool's block layout, so the resident copy is the storage the GPU step reads and writes rather than a copy a host
/// seam reads back.
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
    private readonly IntPtr _gemvQ8_0;
    private readonly IntPtr _gemvQ5_0;
    private readonly IntPtr _gemvQ2_0;
    private readonly IntPtr _gemvQ5K;
    private readonly IntPtr _gemvQ2K;
    private readonly IntPtr _gemvQ3K;
    private readonly IntPtr _gemvIQ4Nl;
    private readonly IntPtr _gemvIQ4Xs;
    private readonly IntPtr _gemvIQ2XXS;
    private readonly IntPtr _gemvIQ2XS;
    private readonly IntPtr _gemvIQ2S;
    private readonly IntPtr _gemvIQ3X;
    private readonly IntPtr _gemvIQ3S;
    private readonly IntPtr _gemmQ8_0;
    private readonly IntPtr _gemmQ5_0;
    private readonly IntPtr _gemmQ2_0;
    private readonly IntPtr _gemmQ4K;
    private readonly IntPtr _gemmQ6K;
    private readonly IntPtr _gemmQ5K;
    private readonly IntPtr _gemmQ2K;
    private readonly IntPtr _gemmQ3K;
    private readonly IntPtr _gemmIQ4Nl;
    private readonly IntPtr _gemmIQ4Xs;
    private readonly IntPtr _gemmF32;
    private readonly IntPtr _gemmF16;
    private readonly IntPtr _gemmBf16;
    private readonly IntPtr _ropePairs;
    private readonly IntPtr _ropeHalf;
    private readonly IntPtr _kvStore;
    private readonly IntPtr _kvStorePaged;
    private readonly IntPtr _attnCausal;
    private readonly IntPtr _attnCausalPaged;
    private readonly IntPtr _attnCausalQ8Q8;
    private readonly IntPtr _attnCausalK8V4;

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
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvQ8_0, _module, "gemv_q8_0"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvQ5_0, _module, "gemv_q5_0"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvQ2_0, _module, "gemv_q2_0"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvQ5K, _module, "gemv_q5k"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvQ2K, _module, "gemv_q2k"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvQ3K, _module, "gemv_q3k"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvIQ4Nl, _module, "gemv_iq4nl"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvIQ4Xs, _module, "gemv_iq4xs"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvIQ2XXS, _module, "gemv_iq2xxs"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvIQ2XS, _module, "gemv_iq2xs"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvIQ2S, _module, "gemv_iq2s"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvIQ3X, _module, "gemv_iq3xxs"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemvIQ3S, _module, "gemv_iq3s"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemmQ8_0, _module, "gemm_q8_0"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemmQ5_0, _module, "gemm_q5_0"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemmQ2_0, _module, "gemm_q2_0"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemmQ4K, _module, "gemm_q4k"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemmQ6K, _module, "gemm_q6k"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemmQ5K, _module, "gemm_q5k"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemmQ2K, _module, "gemm_q2k"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemmQ3K, _module, "gemm_q3k"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemmIQ4Nl, _module, "gemm_iq4nl"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemmIQ4Xs, _module, "gemm_iq4xs"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemmF32, _module, "gemm_f32"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemmF16, _module, "gemm_f16"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _gemmBf16, _module, "gemm_bf16"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _ropePairs, _module, "rope_pairs"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _ropeHalf, _module, "rope_half"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _kvStore, _module, "kv_store"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _kvStorePaged, _module, "kv_store_paged"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _attnCausal, _module, "attn_causal"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _attnCausalPaged, _module, "attn_causal_paged"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _attnCausalQ8Q8, _module, "attn_causal_paged_q8q8"));
                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out _attnCausalK8V4, _module, "attn_causal_paged_k8v4"));
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

    /// <summary>
    /// Whether this backend's GEMV kernels execute for weights in <paramref name="format"/>: the CUDA seam carries GEMV
    /// kernels for <see cref="QuantFormat.F32"/>, <see cref="QuantFormat.F16"/>, <see cref="QuantFormat.BF16"/>,
    /// <see cref="QuantFormat.Q8_0"/>, <see cref="QuantFormat.Q5_0"/>, <see cref="QuantFormat.Q2_0"/>,
    /// <see cref="QuantFormat.Q4_K"/>, <see cref="QuantFormat.Q5_K"/>, <see cref="QuantFormat.Q6_K"/>,
    /// <see cref="QuantFormat.Q2_K"/>, and <see cref="QuantFormat.Q3_K"/> only, so a route naming any other format is
    /// refused at load naming this backend rather than raised mid-decode by <see cref="SelectGemv"/> or run through some
    /// other device's kernels.
    /// </summary>
    public bool SupportsGemv(QuantFormat format) => HasGemvKernel(format);

    /// <summary>
    /// The capability fact every CUDA GEMV posture answers from: the formats the embedded PTX GEMV kernels this backend resolves
    /// function handles for — every QuantFormat the seam now carries. A load-time capability gate consults it through
    /// <see cref="SupportsGemv"/>; it stays static so the posture is inspectable without opening a device. The sub-4-bit i-quant
    /// family answers true here because its integer-dot kernels ride on host-side Q8_K quantization of the activation, and the IQ4
    /// pair rides on fused nibble-table lookups against pinned scratch — no format left in the enum lacks a kernel today.
    /// </summary>
    public static bool HasGemvKernel(QuantFormat format) => true; // every QuantFormat member now resolves a GEMV function handle here

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
                case QuantFormat.Q8_0:
                    DequantQ8_0Row(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
                    break;
                case QuantFormat.Q5_0:
                    DequantQ5_0Row(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
                    break;
                case QuantFormat.Q2_0:
                    DequantQ2_0Row(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
                    break;
                case QuantFormat.Q5_K:
                    DequantQ5KRow(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
                    break;
                case QuantFormat.Q2_K:
                    DequantQ2KRow(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
                    break;
                case QuantFormat.Q3_K:
                    DequantQ3KRow(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
                    break;
                case QuantFormat.IQ4_NL:
                    DequantIQ4NlRow(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
                    break;
                case QuantFormat.IQ4_XS:
                    DequantIQ4XRow(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
                    break;
                case QuantFormat.IQ2_XXS:
                    DequantIQ2XXSRow(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
                    break;
                case QuantFormat.IQ2_XS:
                    DequantIQ2XSRow(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
                    break;
                case QuantFormat.IQ2_S:
                    DequantIQ2SRow(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
                    break;
                case QuantFormat.IQ3_XXS:
                    DequantIQ3XRow(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
                    break;
                case QuantFormat.IQ3_S:
                    DequantIQ3SRow(new ReadOnlySpan<byte>(rowPtr, (int)stride), cols, output);
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

            ReleaseKvMirrors(); // mirrors flush and release their pins while this instance's context is still current
            ReleaseDeviceSteps(); // a step's buffers belong to this context too, so they go before it does
            FreeIqScratch(); // the pinned i-quant tables belong to this context too, so they go before it does
            FreeDeviceWeightAllocations();
            ReleaseHostMappedRegions();
            ReleaseTimingEvents(); // the launch-timing pair belongs to this context too, so it goes before the context does
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

        if (w.Format is QuantFormat.IQ2_XXS or QuantFormat.IQ2_XS or QuantFormat.IQ2_S or QuantFormat.IQ3_XXS or QuantFormat.IQ3_S)
        {
            RunGemvQ8K(w, activations, count, output, function); // the sub-4-bit family dots against a host-side Q8_K quantization of the activation instead.
            return;
        }

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

            bool iqTab = w.Format is QuantFormat.IQ4_NL or QuantFormat.IQ4_XS; // these two kernels look every nibble up in the shared scratch table.
            if (iqTab)
            {
                EnsureIqScratch(); // bound once per instance beside every other pinned table this seam reads.
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
                        AccountRuntimeCopy((long)xBytes); // this call's activation input crossing the seam
                    }

                    uint rowsU = (uint)rows;
                    uint colsU = (uint)cols;
                    uint countU = (uint)count;
                    ulong tabBase = _tabIq4Values; // only the IQ4 pair reads it: every other kernel in this family declares six parameters.
                    void** kernelParams = stackalloc void*[iqTab ? 7 : 6];
                    kernelParams[0] = &deviceWeights;
                    kernelParams[1] = &deviceX;
                    kernelParams[2] = &deviceOut;
                    kernelParams[3] = &rowsU;
                    kernelParams[4] = &colsU;
                    kernelParams[5] = &countU;

                    if (iqTab)
                    {
                        kernelParams[6] = &tabBase; // the scratch table cursor rides last, matching every other launch parameter order on this seam.
                    }

                    uint blocks = (uint)((rows + KernelThreads - 1) / KernelThreads);
                    bool timed = BeginKernelTiming(); // device-measured busy time for exactly this launch, when the seam is open here
                    CudaException.ThrowIf("cuLaunchKernel", CudaNative.cuLaunchKernel(function, blocks, 1, 1, KernelThreads, 1, 1, 0, IntPtr.Zero, kernelParams, null));
                    if (timed)
                    {
                        StampKernelEnd();
                    }

                    CudaException.ThrowIf("cuCtxSynchronize", CudaNative.cuCtxSynchronize());
                    if (timed)
                    {
                        FoldKernelElapsed(); // both stamps have completed by the time this instance's context synchronize returns
                    }

                    fixed (float* pOut = output)
                    {
                        CudaException.ThrowIf("cuMemcpyDtoH_v2", CudaNative.cuMemcpyDtoH((byte*)pOut, deviceOut, outBytes));
                        AccountRuntimeCopy((long)outBytes); // this call's kernel output crossing back to host memory
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
        QuantFormat.Q8_0 => _gemvQ8_0,
        QuantFormat.Q5_0 => _gemvQ5_0,
        QuantFormat.Q2_0 => _gemvQ2_0,
        QuantFormat.Q5_K => _gemvQ5K,
        QuantFormat.Q2_K => _gemvQ2K,
        QuantFormat.Q3_K => _gemvQ3K,
        QuantFormat.IQ4_NL => _gemvIQ4Nl,
        QuantFormat.IQ4_XS => _gemvIQ4Xs,
        QuantFormat.IQ2_XXS => _gemvIQ2XXS,
        QuantFormat.IQ2_XS => _gemvIQ2XS,
        QuantFormat.IQ2_S => _gemvIQ2S,
        QuantFormat.IQ3_XXS => _gemvIQ3X,
        QuantFormat.IQ3_S => _gemvIQ3S,
        _ => throw Unsupported(format, "GEMV"),
    };

    private static NotSupportedException Unsupported(QuantFormat format, string kernel) => new(
        $"CudaBackend ('{DeviceIdValue}') has no {kernel} kernel for format {format}: the CUDA seam carries GEMV kernels " +
        "for every QuantFormat it accepts today (F32, F16, BF16, Q8_0, Q5_0, Q2_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, IQ4_NL, " +
        "IQ4_XS, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, and IQ3_S) and windowed blocked matmul for the same set minus the Q8_K " +
        "i-quants (F32/F16/BF16 ride their own GEMV widening walked once per activation-row window; only IQ2_XXS, IQ2_XS, " +
        "IQ2_S, IQ3_XXS, and IQ3_S carry none), so this weight pass is refused naming that backend rather than run through a CPU fallback.");

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
