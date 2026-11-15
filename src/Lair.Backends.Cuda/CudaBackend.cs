using System.Buffers;
using System.Runtime.InteropServices;
using System.Text;

using Lair.Contracts;

namespace Lair.Backends.Cuda;

/// <summary>
/// CUDA compute backend: packed weight tensors are copied to device memory once per instance and every dense GEMV,
/// single or batched, runs as an embedded PTX kernel against those copies for F32, F16, and BF16 weights. Formats with
/// no kernel here refuse naming this backend rather than run through some other route; the blocked matmul has no CUDA
/// path at all yet. Activation vectors and outputs cross the PCIe seam per call, so KV arenas and split-route staging
/// stay host-resident aligned memory under this device id until accelerator residency lands elsewhere.
/// </summary>
public sealed class CudaBackend : IBackend
{
    /// <summary>The accelerator device id every CUDA-backed route segment names.</summary>
    public const string DeviceIdValue = "cuda";

    private const int KernelThreads = 128;

    // Address arithmetic stays in computed-register form throughout: this driver seam rejects [base+reg] load forms, so
    // each element address is mad'd from its index, widened, shifted by the element size, and added to the base. Half
    // and bf16 weights widen through integer bit math into a per-thread local slot reloaded as f32; cvt.f16 forms are
    // rejected outright (CUDA_ERROR_INVALID_PTX) on this seam. Accumulation is plain mul then add in ascending column
    // order - the same chain the scalar CPU reference uses, so one device result matches one host dot bit for bit.
    private const string Ptx =
        ".version 7.0\n" +
        ".target sm_50\n" +
        ".address_size 64\n" +
        "\n" +
        ".visible .entry gemv_f32(\n" +
        "    .param .u64 w_ptr,\n" +
        "    .param .u64 x_ptr,\n" +
        "    .param .u64 out_ptr,\n" +
        "    .param .u32 rows_arg,\n" +
        "    .param .u32 cols_arg,\n" +
        "    .param .u32 count_arg\n" +
        ") {\n" +
        "    .reg .u64 w, xin, o, addr;\n" +
        "    .reg .u32 row, col, bb, rows_, cols_, cnt_, tid, t0;\n" +
        "    .reg .pred p;\n" +
        "    .reg .f32 acc, wv, xv;\n" +
        "\n" +
        "    ld.param.u64 w, [w_ptr];\n" +
        "    ld.param.u64 xin, [x_ptr];\n" +
        "    ld.param.u64 o, [out_ptr];\n" +
        "    ld.param.u32 rows_, [rows_arg];\n" +
        "    ld.param.u32 cols_, [cols_arg];\n" +
        "    ld.param.u32 cnt_, [count_arg];\n" +
        "\n" +
        "    mov.u32 tid, %ctaid.x;\n" +
        "    mov.u32 t0, %ntid.x;\n" +
        "    mul.lo.u32 tid, tid, t0;\n" +
        "    mov.u32 t0, %tid.x;\n" +
        "    add.u32 row, tid, t0;\n" +
        "    setp.ge.u32 p, row, rows_;\n" +
        "    @p bra done;\n" +
        "\n" +
        "    mov.u32 bb, 0;\n" +
        "batch:\n" +
        "    setp.ge.u32 p, bb, cnt_;\n" +
        "    @p bra done;\n" +
        "    mov.f32 acc, 0f00000000;\n" +
        "    mov.u32 col, 0;\n" +
        "_loop_col:\n" +
        "    setp.ge.u32 p, col, cols_;\n" +
        "    @p bra store;\n" +
        "    mad.lo.u32 t0, row, cols_, col;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, w, addr;\n" +
        "    ld.global.f32 wv, [addr];\n" +
        "    mad.lo.u32 t0, bb, cols_, col;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, xin, addr;\n" +
        "    ld.global.f32 xv, [addr];\n" +
        "    mul.f32 wv, wv, xv;\n" +
        "    add.f32 acc, acc, wv;\n" +
        "    add.u32 col, col, 1;\n" +
        "    bra _loop_col;\n" +
        "store:\n" +
        "    mad.lo.u32 t0, bb, rows_, row;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, o, addr;\n" +
        "    st.global.f32 [addr], acc;\n" +
        "    add.u32 bb, bb, 1;\n" +
        "    bra batch;\n" +
        "done:\n" +
        "    ret;\n" +
        "}\n" +
        "\n" +
        ".visible .entry gemv_f16(\n" +
        "    .param .u64 w_ptr,\n" +
        "    .param .u64 x_ptr,\n" +
        "    .param .u64 out_ptr,\n" +
        "    .param .u32 rows_arg,\n" +
        "    .param .u32 cols_arg,\n" +
        "    .param .u32 count_arg\n" +
        ") {\n" +
        "    .reg .u64 w, xin, o, addr;\n" +
        "    .reg .u32 row, col, bb, rows_, cols_, cnt_, tid, t0;\n" +
        "    .reg .u32 bits, payload, exp, mant, sign, fbits;\n" +
        "    .reg .b16 hw;\n" +
        "    .reg .pred p;\n" +
        "    .reg .f32 acc, wv, xv, subv;\n" +
        "    .local .align 4 .b8 slot[4];\n" +
        "\n" +
        "    ld.param.u64 w, [w_ptr];\n" +
        "    ld.param.u64 xin, [x_ptr];\n" +
        "    ld.param.u64 o, [out_ptr];\n" +
        "    ld.param.u32 rows_, [rows_arg];\n" +
        "    ld.param.u32 cols_, [cols_arg];\n" +
        "    ld.param.u32 cnt_, [count_arg];\n" +
        "\n" +
        "    mov.u32 tid, %ctaid.x;\n" +
        "    mov.u32 t0, %ntid.x;\n" +
        "    mul.lo.u32 tid, tid, t0;\n" +
        "    mov.u32 t0, %tid.x;\n" +
        "    add.u32 row, tid, t0;\n" +
        "    setp.ge.u32 p, row, rows_;\n" +
        "    @p bra done;\n" +
        "\n" +
        "    mov.u32 bb, 0;\n" +
        "batch:\n" +
        "    setp.ge.u32 p, bb, cnt_;\n" +
        "    @p bra done;\n" +
        "    mov.f32 acc, 0f00000000;\n" +
        "    mov.u32 col, 0;\n" +
        "_loop_col:\n" +
        "    setp.ge.u32 p, col, cols_;\n" +
        "    @p bra store;\n" +
        "    mad.lo.u32 t0, row, cols_, col;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 1;\n" +
        "    add.u64 addr, w, addr;\n" +
        "    ld.global.b16 hw, [addr];\n" +
        "    mov.b32 bits, {hw, 0};\n" +
        "    and.b32 sign, bits, 32768;\n" +
        "    and.b32 payload, bits, 32767;\n" +
        "    shr.u32 exp, payload, 10;\n" +
        "    and.b32 mant, payload, 1023;\n" +
        "    shl.b32 sign, sign, 16;\n" +
        "    setp.le.u32 p, exp, 0;\n" +
        "    @p bra widen_sub;\n" +
        "    add.u32 fbits, exp, 112;\n" +
        "    shl.b32 fbits, fbits, 23;\n" +
        "    shl.b32 mant, mant, 13;\n" +
        "    or.b32 fbits, fbits, mant;\n" +
        "    xor.b32 fbits, fbits, sign;\n" +
        "    st.local.u32 [slot], fbits;\n" +
        "    ld.local.f32 wv, [slot];\n" +
        "    bra widen_done;\n" +
        "widen_sub:\n" +
        "    cvt.rn.f32.u32 subv, mant;\n" +
        "    mul.f32 wv, subv, 0f33800000;\n" +
        "    setp.eq.u32 p, sign, 32768;\n" +
        "    @!p bra widen_done;\n" +
        "    mul.f32 wv, wv, 0f80000000;\n" +
        "widen_done:\n" +
        "    mad.lo.u32 t0, bb, cols_, col;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, xin, addr;\n" +
        "    ld.global.f32 xv, [addr];\n" +
        "    mul.f32 wv, wv, xv;\n" +
        "    add.f32 acc, acc, wv;\n" +
        "    add.u32 col, col, 1;\n" +
        "    bra _loop_col;\n" +
        "store:\n" +
        "    mad.lo.u32 t0, bb, rows_, row;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, o, addr;\n" +
        "    st.global.f32 [addr], acc;\n" +
        "    add.u32 bb, bb, 1;\n" +
        "    bra batch;\n" +
        "done:\n" +
        "    ret;\n" +
        "}\n" +
        "\n" +
        ".visible .entry gemv_bf16(\n" +
        "    .param .u64 w_ptr,\n" +
        "    .param .u64 x_ptr,\n" +
        "    .param .u64 out_ptr,\n" +
        "    .param .u32 rows_arg,\n" +
        "    .param .u32 cols_arg,\n" +
        "    .param .u32 count_arg\n" +
        ") {\n" +
        "    .reg .u64 w, xin, o, addr;\n" +
        "    .reg .u32 row, col, bb, rows_, cols_, cnt_, tid, t0;\n" +
        "    .reg .u32 bits;\n" +
        "    .reg .b16 hw;\n" +
        "    .reg .pred p;\n" +
        "    .reg .f32 acc, wv, xv;\n" +
        "    .local .align 4 .b8 slot[4];\n" +
        "\n" +
        "    ld.param.u64 w, [w_ptr];\n" +
        "    ld.param.u64 xin, [x_ptr];\n" +
        "    ld.param.u64 o, [out_ptr];\n" +
        "    ld.param.u32 rows_, [rows_arg];\n" +
        "    ld.param.u32 cols_, [cols_arg];\n" +
        "    ld.param.u32 cnt_, [count_arg];\n" +
        "\n" +
        "    mov.u32 tid, %ctaid.x;\n" +
        "    mov.u32 t0, %ntid.x;\n" +
        "    mul.lo.u32 tid, tid, t0;\n" +
        "    mov.u32 t0, %tid.x;\n" +
        "    add.u32 row, tid, t0;\n" +
        "    setp.ge.u32 p, row, rows_;\n" +
        "    @p bra done;\n" +
        "\n" +
        "    mov.u32 bb, 0;\n" +
        "batch:\n" +
        "    setp.ge.u32 p, bb, cnt_;\n" +
        "    @p bra done;\n" +
        "    mov.f32 acc, 0f00000000;\n" +
        "    mov.u32 col, 0;\n" +
        "_loop_col:\n" +
        "    setp.ge.u32 p, col, cols_;\n" +
        "    @p bra store;\n" +
        "    mad.lo.u32 t0, row, cols_, col;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 1;\n" +
        "    add.u64 addr, w, addr;\n" +
        "    ld.global.b16 hw, [addr];\n" +
        "    mov.b32 bits, {hw, 0};\n" +
        "    shl.b32 t0, bits, 16;\n" +
        "    st.local.u32 [slot], t0;\n" +
        "    ld.local.f32 wv, [slot];\n" +
        "    mad.lo.u32 t0, bb, cols_, col;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, xin, addr;\n" +
        "    ld.global.f32 xv, [addr];\n" +
        "    mul.f32 wv, wv, xv;\n" +
        "    add.f32 acc, acc, wv;\n" +
        "    add.u32 col, col, 1;\n" +
        "    bra _loop_col;\n" +
        "store:\n" +
        "    mad.lo.u32 t0, bb, rows_, row;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, o, addr;\n" +
        "    st.global.f32 [addr], acc;\n" +
        "    add.u32 bb, bb, 1;\n" +
        "    bra batch;\n" +
        "done:\n" +
        "    ret;\n" +
        "}\n";

    private static readonly byte[] PtxBytes = Encoding.UTF8.GetBytes(Ptx + "\0");

    private readonly object _gate = new();
    private readonly Dictionary<(nint Pointer, long Length), ulong> _deviceWeights = [];
    private readonly List<ulong> _weightAllocations = [];
    private readonly int _device;
    private readonly IntPtr _context;
    private readonly IntPtr _module;
    private readonly IntPtr _gemvF32;
    private readonly IntPtr _gemvF16;
    private readonly IntPtr _gemvBf16;

    private bool _disposed;

    /// <summary>Weight tensors this instance has copied to device memory so far.</summary>
    public int DeviceWeightCopies => _deviceWeights.Count;

    public string DeviceId => DeviceIdValue;

    public CudaBackend()
    {
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
                    new ReadOnlySpan<float>(rowPtr, cols).CopyTo(output);
                    break;
                case QuantFormat.F16:
                    for (int i = 0; i < cols; i++)
                    {
                        output[i] = ReadHalfLE(rowPtr + 2 * i);
                    }

                    break;
                case QuantFormat.BF16:
                    for (int i = 0; i < cols; i++)
                    {
                        output[i] = ReadBf16LE(rowPtr + 2 * i);
                    }

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

            foreach (ulong devicePointer in _weightAllocations)
            {
                CudaNative.cuMemFree(devicePointer);
            }

            _weightAllocations.Clear();
            _deviceWeights.Clear();
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
                    deviceWeights = EnsureDeviceWeights((nint)wBytes, weightBytes); // pinned for exactly this call; the copy runs now and never again per tensor
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

    private IntPtr KernelFor(QuantFormat format) => format switch
    {
        QuantFormat.F32 => _gemvF32,
        QuantFormat.F16 => _gemvF16,
        QuantFormat.BF16 => _gemvBf16,
        _ => throw Unsupported(format, "GEMV"),
    };

    private static NotSupportedException Unsupported(QuantFormat format, string kernel) => new(
        $"CudaBackend ('{DeviceIdValue}') has no {kernel} kernel for format {format}: the CUDA seam carries F32, F16, and BF16 kernels only, " +
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

    // Little-endian half-float load widened to float.
    private static unsafe float ReadHalfLE(byte* p) => (float)BitConverter.Int16BitsToHalf((short)(*p | (p[1] << 8)));

    // Little-endian bf16 load widened to float: the 16 bits land in the f32 high half.
    private static unsafe float ReadBf16LE(byte* p) => BitConverter.UInt32BitsToSingle((uint)((p[0] | (p[1] << 8)) << 16));
}


