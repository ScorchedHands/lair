using Lair.Contracts;

namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // Activation-row window for the blocked-matmul family: one unpack of a weight row's quant bytes feeds this many
    // activation vectors before the row is walked again, which is the amortization the dense-matrix selection exists to
    // get. Every lane keeps its own accumulator chain advanced in ascending column order exactly as that vector's own
    // GEMV walk advances it, so a windowed dot and a solo dot of the same weight bytes produce the same sum for every
    // element: the per-format bands inherited from the GEMV kernels carry over unchanged. A lane past the end of the
    // batch reads its clamped row's values into an accumulator that is never stored, so no out-of-range activation ever
    // reaches memory and no output slot is written twice.
    private const int GemmWindow = 8;

    /// <inheritdoc/>
    public bool SupportsGemm(QuantFormat format) => HasGemmKernel(format);

    /// <summary>
    /// The capability fact every CUDA blocked-matmul posture answers from: exactly the formats this backend resolves a windowed function handle for.
    /// Every packed block quant (Q8_0, Q5_0, Q2_0, Q4_K, Q6_K, Q5_K, Q2_K, Q3_K) and the IQ4 pair carry one today — each is its own GEMV kernel's
    /// unpacking walked once per activation-row window instead of once per vector. The unquantized F32/F16/BF16 formats answer false because their
    /// dense matmul is the WMMA work this phase has not landed, and the Q8_K i-quant family (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) answers false
    /// until its windowed kernel lands: a route naming any of them keeps refusing at load rather than running through some other device's kernel.
    /// </summary>
    public static bool HasGemmKernel(QuantFormat format) => format switch
    {
        QuantFormat.Q8_0 or QuantFormat.Q5_0 or QuantFormat.Q2_0 or QuantFormat.Q4_K or QuantFormat.Q6_K
            or QuantFormat.Q5_K or QuantFormat.Q2_K or QuantFormat.Q3_K or QuantFormat.IQ4_NL or QuantFormat.IQ4_XS => true,
        _ => false, // F32/F16/BF16 have no blocked matmul in this build yet, and neither do the Q8_K i-quants.
    };

    private IntPtr GemmKernelFor(QuantFormat format) => format switch
    {
        QuantFormat.Q8_0 => _gemmQ8_0,
        QuantFormat.Q5_0 => _gemmQ5_0,
        QuantFormat.Q2_0 => _gemmQ2_0,
        QuantFormat.Q4_K => _gemmQ4K,
        QuantFormat.Q6_K => _gemmQ6K,
        QuantFormat.Q5_K => _gemmQ5K,
        QuantFormat.Q2_K => _gemmQ2K,
        QuantFormat.Q3_K => _gemmQ3K,
        QuantFormat.IQ4_NL => _gemmIQ4Nl,
        QuantFormat.IQ4_XS => _gemmIQ4Xs,
        _ => throw UnsupportedGemm(format),
    };

    private static NotSupportedException UnsupportedGemm(QuantFormat format) => new(
        $"CudaBackend ('{DeviceIdValue}') has no blocked matmul kernel for format {format}: the CUDA seam carries windowed " +
        "blocked-matmul kernels for Q8_0, Q5_0, Q2_0, Q4_K, Q6_K, Q5_K, Q2_K, Q3_K, IQ4_NL, and IQ4_XS today; F32, F16, and BF16 " +
        "carry none (their dense matmul is the unquantized WMMA work this build has not landed) and neither do the Q8_K i-quants " +
        "(IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, and IQ3_S), so this weight pass is refused naming that backend rather than run through a CPU fallback.");

    /// <inheritdoc/>
    public void GemmBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs)
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

        RunGemm(w, inputs, count, outputs); // refuses a format with no windowed kernel before any device work starts.
    }

    // The blocked-matmul launch: same residency posture (one pinned weight region per tensor for the instance's lifetime),
    // same activation/output staging and accounting as every GEMV here, and the same batch-major output layout — only the
    // kernel differs, walking each weight row once per activation-row window instead of once per activation vector. The two
    // IQ4 kernels read their shared nibble table beside their other operands exactly like their GEMV pair does.
    private void RunGemm(WeightRef w, ReadOnlySpan<float> activations, int count, Span<float> output)
    {
        (int rows, int cols) = ValidateShape(w);
        IntPtr function = GemmKernelFor(w.Format);

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
                    // The mapped posture launches against the device pointer of a registered host region; the default posture keeps its once-copied device buffer. Either answer is pinned for exactly this call and cached per tensor for the instance's lifetime — and it is the same cached region their GEMV calls already use, so switching kernels costs no extra residency.
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

                    uint blocks = (uint)((rows + KernelThreads - 1) / KernelThreads); // one weight row per thread: same grid geometry as every GEMV in the family.
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
}
