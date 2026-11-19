using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;

namespace Lair.Bench;

/// <summary>
/// Per-format CUDA-vs-CPU blocked-matmul bench for the MVP5.3 Bench Gate: every QuantFormat `CudaBackend.HasGemmKernel` answers true for is timed over
/// a deterministic pinned synthetic fixture - literally the same bytes `cuda-gemv` builds per format, so the two sweeps measure identical weight tensors and their numbers are comparable - at one shape per run with no RNG. For each format it measures wall time on three arms: a single-threaded scalar CPU chain used only for an untimed equivalence pass (the drift basis), the engine's auto-threaded CpuBackend timed over repeated calls, and `CudaBackend.GemmBatch` host wall time after an untimed warm-up of that fixture. Both CPU arms take the weight-pass route that format would actually use on this build today: `GemmBatch` where `CpuBackend.SupportsGemm` is true (Q8_0, Q4_K, IQ4_XS) and otherwise `GemvBatch`, the batched-row alternative a dense-matrix selection falls back to - each format's chosen route is printed as cpuRoute=gemm|gemv so every ratio states which CPU chain it compares. The CUDA wall arm includes every per-call activation upload, output download, launch, and context sync; the one-time weight copy a DeviceCopy posture caches on first use lands in the untimed warm-up instead. Device kernel-busy microseconds come from KernelBusyMicrosecondsTotal deltas around each timed launch exactly as the GEMV sweep reads them. Ratios are printed as comparable total passes (CPU wall vs CUDA wall) and as compute-only (CPU scalar-oracle wall against kernel-busy only, labeled as excluding PCIe/runtime transfer overhead), with GFLOP/s for every arm computed from the shape's own 2*rows*cols*batch operation count so a reader can reproduce it. One untimed equivalence pass per format reports max relative drift between the CPU chain of that format's route and the CUDA blocked matmul over identical inputs; no band is asserted here, so a run never widens or narrows any recorded gate - it only states what this box measured. Output goes to stdout only and carries the shape, reps, batch, format list, fixture description, selected routes, and host facts needed to reproduce every number.
/// </summary>
internal static class CudaGemmBench
{
    // Default shape: dense-matrix geometry that amortizes rather than decodes - a 512x1024 weight matrix consumed by a batch of 32 activation vectors, wide enough past both blocked kernels' tile granularity (8 CPU rows x one quant super-block; the CUDA window's own row-per-thread grid) for a repeated pass to measure steady-state kernel cost. All are overridable per run; --formats narrows the sweep when a subset is being re-measured.
    private const int DefaultRows = 512;
    private const int DefaultCols = 1024;
    private const int DefaultBatch = 32;
    private const int DefaultReps = 7;

    // Both CPU arms run the whole batch in one call, which is what a prefill window's weight pass does. This build's CPU backend caps activation vectors per batched call (its stack scratch sizes for it); naming that ceiling here refuses an over-wide --batch before any backend opens instead of throwing mid-sweep.
    private const int CpuBatchCeiling = 64;

    // The full supported sweep: exactly the formats `CudaBackend.HasGemmKernel` answers true for today - every packed block quant, the IQ4 pair, and the three unquantized dense formats. The Q8_K i-quants (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) carry no windowed kernel and are refused by --formats parsing rather than swept.
    private static readonly QuantFormat[] DefaultFormats =
    [
        QuantFormat.F32, QuantFormat.F16, QuantFormat.BF16,
        QuantFormat.Q8_0, QuantFormat.Q5_0, QuantFormat.Q2_0, QuantFormat.Q4_K, QuantFormat.Q6_K,
        QuantFormat.Q5_K, QuantFormat.Q2_K, QuantFormat.Q3_K,
        QuantFormat.IQ4_NL, QuantFormat.IQ4_XS,
    ];

    public static int Run(string[] args)
    {
        int rows = DefaultRows;
        int cols = DefaultCols;
        int batch = DefaultBatch;
        int reps = DefaultReps;
        List<QuantFormat>? formatsOverride = null;

        for (int i = 1; i < args.Length; i++)
        {
            if (i + 1 >= args.Length)
            {
                Console.WriteLine($"cuda-gemm: option '{args[i]}' needs a value");
                return 1;
            }

            switch (args[i])
            {
                case "--rows":
                    if (!TryParsePositive(args[++i], out rows))
                    {
                        Console.WriteLine($"cuda-gemm: --rows must be a positive integer (got '{args[i]}')");
                        return 1;
                    }

                    break;
                case "--cols":
                    if (!TryParsePositive(args[++i], out cols))
                    {
                        Console.WriteLine($"cuda-gemm: --cols must be a positive integer (got '{args[i]}')");
                        return 1;
                    }

                    break;
                case "--batch":
                    if (!TryParsePositive(args[++i], out batch))
                    {
                        Console.WriteLine($"cuda-gemm: --batch must be a positive integer (got '{args[i]}')");
                        return 1;
                    }

                    break;
                case "--reps":
                    if (!TryParsePositive(args[++i], out reps))
                    {
                        Console.WriteLine($"cuda-gemm: --reps must be a positive integer (got '{args[i]}')");
                        return 1;
                    }

                    break;
                case "--formats":
                    string list = args[++i];
                    formatsOverride = [];
                    foreach (string part in list.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                    {
                        if (!Enum.TryParse(part, ignoreCase: true, out QuantFormat parsed) || !CudaBackend.HasGemmKernel(parsed))
                        {
                            Console.WriteLine($"cuda-gemm: unknown or kernel-less format '{part}'; expected one of {string.Join(",", DefaultFormats)}");
                            return 1;
                        }

                        if (!formatsOverride.Contains(parsed))
                        {
                            formatsOverride.Add(parsed);
                        }
                    }

                    if (formatsOverride.Count == 0)
                    {
                        Console.WriteLine("cuda-gemm: --formats listed no formats");
                        return 1;
                    }

                    break;
                default:
                    Console.WriteLine($"cuda-gemm: unknown option '{args[i]}' (--rows, --cols, --batch, --reps, --formats)");
                    return 1;
            }
        }

        if ((long)batch * cols > int.MaxValue || (long)batch * rows > int.MaxValue)
        {
            Console.WriteLine("cuda-gemm: batch x columns or batch x rows exceeds the addressable span limit");
            return 1;
        }

        if (batch > CpuBatchCeiling)
        {
            // The CUDA arm would accept a wider batch, but every CPU weight-pass arm here is one call over the whole sweep's activation vectors - refusing up front keeps each printed ratio comparable across all three arms.
            Console.WriteLine($"cuda-gemm: --batch must stay within {CpuBatchCeiling} (the CPU backend runs one batched call per pass): got '{batch}'");
            return 1;
        }

        IReadOnlyList<QuantFormat> formats = formatsOverride ?? (IReadOnlyList<QuantFormat>)DefaultFormats;

        // Open every participant before timing anything so no backend-open cost lands inside a measured rep. The CUDA instance stays open for the whole sweep: its DeviceCopy posture caches each fixture tensor once, and its cumulative kernel-busy/transfer tallies are read as per-format deltas around that format's timed reps.
        using CpuBackend cpuScalar = new(threads: 1, allowAvx2: false); // the untimed scalar reference chain every drift figure is measured against
        using CpuBackend cpuAuto = new(); // engine posture: auto thread choice; its effective worker count is reported explicitly

        CudaBackend? cuda;
        try
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                Console.WriteLine("cuda-gemm: the CUDA driver seam runs on Windows against nvcuda.dll; this host reports otherwise");
                return 2;
            }

            cuda = new CudaBackend(CudaWeightResidency.DeviceCopy);
        }
        catch (EntryPointNotFoundException ex)
        {
            Console.WriteLine($"cuda-gemm: CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point ({ex.Message})");
            return 2;
        }
        catch (DllNotFoundException ex)
        {
            Console.WriteLine($"cuda-gemm: CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API ({ex.Message})");
            return 2;
        }
        catch (CudaException ex)
        {
            Console.WriteLine($"cuda-gemm: CUDA backend refused to open on this host: {CollapseWhitespace(ex.Message)}");
            return 2;
        }

        try
        {
            Console.WriteLine("cuda-gemm bench gate measurement");
            Console.WriteLine(
                $"  shape rows={rows} cols={cols} batch={batch} reps={reps} formats={formats.Count} " +
                $"cpuArms=scalar1T(threads=1,avx2=false untimed equivalence)+auto(writers below) cudaPosture=deviceCopy");
            Console.WriteLine(
                $"  host avx2={(Avx2.IsSupported ? "yes" : "no")} physicalCores={CpuBackend.DetectedPhysicalCoreCount} " +
                $"logicalProcessors={Environment.ProcessorCount} cpuAutoWorkers={cpuAuto.EffectiveWorkerCount}");
            Console.WriteLine(
                "  fixture=deterministic pinned block patterns (fixed arithmetic progressions, no RNG; the same packing classes the GPU parity fixtures walk) activations=Sin(0.11*c+7*v)*1.3 fixed formula");
            Console.WriteLine(
                "  basis: every arm runs CpuBackend/CudaBackend batched matmul over identical bytes (count activation vectors of cols, count*rows output layout); cuda warm-up (including the one-time per-tensor device copy) is untimed; cuda wall includes each call's activation upload, launch, sync, and download but not weight residency");
            Console.WriteLine(
                "  routes: both CPU arms take that format's own dense-matrix weight pass - GemmBatch where CpuBackend.SupportsGemm is true (Q8_0/Q4_K/IQ4_XS), otherwise GemvBatch as the batched-row alternative a blocked matmul falls back to on this build; each format prints cpuRoute=gemm|gemv so every ratio states which CPU chain it compares");
            Console.WriteLine(
                "  flops: GFLOP/s figures count one multiply and one add per dot-product element, i.e. 2*rows*cols*batch per pass - reproducible from the printed shape for every arm");

            List<FormatRow> table = [];

            foreach (QuantFormat format in formats)
            {
                FormatRow row = MeasureFormat(format, rows, cols, batch, reps, cpuScalar, cpuAuto, cuda);
                table.Add(row);
            }

            Console.WriteLine();
            Console.WriteLine("SUMMARY TABLE median-based; ratios >1 means CUDA faster");
            string[] headers = ["format", "cpu_route", "packedMiB", "cpu_scalar_ms", "cpu_auto_ms", "cuda_wall_ms", "busy_us/rep", "ratio_scalar_cuda", "ratio_auto_cuda", "ratio_compute_only", "equiv_max_rel"];
            Console.WriteLine(string.Join("  ", headers.Select(h => h.PadLeft(Math.Max(h.Length, 13)))));
            foreach (FormatRow row in table)
            {
                string[] cells =
                [
                    row.Format.ToString(),
                    row.CpuRoute,
                    $"{row.PackedBytes / (1024.0 * 1024.0):F2}",
                    $"{row.CpuScalarMedMs:F3}",
                    $"{row.CpuAutoMedMs:F3}",
                    $"{row.CudaWallMedMs:F3}",
                    row.BusyUsAvgPerRep is null ? "n/a" : $"{row.BusyUsAvgPerRep.Value:F1}",
                    $"{row.CpuScalarMedMs / row.CudaWallMedMs:F2}",
                    $"{row.CpuAutoMedMs / row.CudaWallMedMs:F2}",
                    row.ComputeOnlyRatio is null ? "n/a" : $"{row.ComputeOnlyRatio.Value:F2}",
                    row.EquivMaxRelDrift.ToString("G6", CultureInfo.InvariantCulture),
                ];

                Console.WriteLine(string.Join("  ", cells.Select(c => c.PadLeft(13))));
            }

            Console.WriteLine();
            Console.WriteLine(
                "  ratio_compute_only = cpu_scalar wall ms vs device-measured kernel-busy us: compute-only, it excludes PCIe/runtime transfer, launch enqueue, and sync overhead");
            Console.WriteLine(
                "  equiv_max_rel = max |cpu-|cuda| / max(|cpu|,|cuda|) over the batched outputs of one untimed paired pass on identical inputs; 'bit_exact=yes' per format marks bit-equal chains. A large relative value usually names a near-zero row sum where both chains agree in absolute terms - read it together with max_abs_drift before comparing against bands recorded on the parity fixtures' pinned shapes (this bench's fixture magnitudes and shape differ). Bands recorded in tests are not re-asserted or widened here.");
            Console.WriteLine(
                "  caveat: wall numbers include GPU contention from other workloads on this box; kernel-busy is device-measured and excludes queue waits only to the extent the events measure launch-to-completion on the default stream");
            Console.WriteLine(
                "  route asymmetry: a cpuRoute=gemv format streams its packed weight rows once per activation vector in both CPU arms while the CUDA blocked matmul walks each row once for its whole activation window, so that format's ratios compare today's fallback pass against the device kernel rather than two identical routes; cpuRoute=gemm formats run the same tile-amortized shape on both sides");

            return 0;
        }
        finally
        {
            cuda.Dispose();
        }
    }

    private static FormatRow MeasureFormat(QuantFormat format, int rows, int cols, int batch, int reps, CpuBackend cpuScalar, CpuBackend cpuAuto, CudaBackend cuda)
    {
        TensorShape shape = new([rows, cols]);
        long strideBytes = shape.RowStrideBytes(format);

        // The pinned fixture builder and activation family are the GEMV sweep's own: identical weight bytes per format across both benches, so a reader can compare this run against cuda-gemv on the same box without re-deriving fixtures.
        byte[] weights = CudaGemvBench.BuildFixture(format, rows, cols);
        WeightRef w = new(weights, shape, format);

        float[] inputs = new float[batch * cols];
        for (int v = 0; v < batch; v++)
        {
            for (int c = 0; c < cols; c++)
            {
                inputs[v * cols + c] = CudaGemvBench.Activation(c + v * 7); // the same activation family the parity batch cases use, per vector offset
            }
        }

        float[] cpuOut = new float[batch * rows];
        float[] cudaOut = new float[batch * rows];

        // The capability fact decides both CPU arms' route once per format: CpuBackend answers it identically for either instance. Formats without a blocked matmul run their batched-row GEMV, which is the pass today's dense-matrix selection falls back to on CPU.
        bool cpuBlockedMatmul = cpuScalar.SupportsGemm(format);
        string cpuRoute = cpuBlockedMatmul ? "gemm" : "gemv";

        Console.WriteLine();
        Console.WriteLine(
            $"FORMAT {format} rows={rows} cols={cols} batch={batch} strideBytes={strideBytes} " +
            $"bpw={(strideBytes * 8.0 / cols):F2} packedMiB={(weights.Length / (1024.0 * 1024.0)):F2}");
        Console.WriteLine(
            $"  cpu_route={cpuRoute}" +
            (cpuBlockedMatmul ? " (CpuBackend.SupportsGemm=true: blocked matmul tile path)" : " (no CPU blocked matmul for this format: batched-row GEMV is today's dense-matrix weight-pass alternative)"));

        // Untimed CUDA warm-up first: it caches this fixture's device weight copy and settles the timing-event pair, so every timed rep below measures a steady-state launch instead of residency setup.
        cuda.GemmBatch(inputs, batch, w, cudaOut);

        // Equivalence sanity pass (untimed): one CPU chain on that format's route and one CUDA blocked matmul over identical inputs and bytes.
        if (cpuBlockedMatmul)
        {
            cpuScalar.GemmBatch(inputs, batch, w, cpuOut);
        }
        else
        {
            cpuScalar.GemvBatch(inputs, batch, w, cpuOut);
        }

        float[] equivalenceCpu = [.. cpuOut];
        cuda.GemmBatch(inputs, batch, w, cudaOut);

        bool exactChain = true;
        bool nanSeen = false;
        double worstRel = 0.0;
        double worstAbs = 0.0;
        for (int i = 0; i < equivalenceCpu.Length; i++)
        {
            if (float.IsNaN(equivalenceCpu[i]) || float.IsNaN(cudaOut[i]))
            {
                nanSeen = true;
            }

            if (BitConverter.SingleToInt32Bits(equivalenceCpu[i]) != BitConverter.SingleToInt32Bits(cudaOut[i]))
            {
                exactChain = false;
            }

            double distance = Math.Abs(equivalenceCpu[i] - cudaOut[i]);
            double scale = Math.Max(Math.Abs(equivalenceCpu[i]), Math.Abs(cudaOut[i]));
            if (distance > worstAbs) worstAbs = distance;
            double relative = scale == 0.0 ? distance : distance / scale; // a zero against zero is exact, and any drift off zero counts fully
            if (relative > worstRel) worstRel = relative;
        }

        Console.WriteLine(
            $"  equivalence cpu_{cpuRoute}1T vs cuda_gemm_batched: max_rel_drift={worstRel:G6} max_abs_drift={worstAbs:G6} bit_exact={(exactChain ? "yes" : "no")}" +
            (nanSeen ? " NaN_SEEN=yes" : string.Empty));

        // CPU arms: warm-up call then timed reps, same batch shape and output layout as the CUDA arm. The scalar chain is only the drift basis; the auto arm states its worker quota so a reader can tell which posture each ratio compares. Equivalence values are already kept in their own array, so every timed pass writes through one scratch output.
        float[] scratch = new float[batch * rows];

        Action<ReadOnlySpan<float>, int, WeightRef, Span<float>> scalarCall = cpuBlockedMatmul ? cpuScalar.GemmBatch : cpuScalar.GemvBatch;
        Action<ReadOnlySpan<float>, int, WeightRef, Span<float>> autoCall = cpuBlockedMatmul ? cpuAuto.GemmBatch : cpuAuto.GemvBatch;

        double[] cpuScalarWall = TimeBackend(scalarCall, inputs, batch, w, scratch, reps);
        double[] cpuAutoWall = TimeBackend(autoCall, inputs, batch, w, scratch, reps);

        // CUDA arm: per-rep wall around the full GemmBatch call plus that launch's device-measured busy microseconds and transfer accounting read as deltas of this instance's cumulative tallies.
        double[] cudaWall = new double[reps];
        long[] busyUs = new long[reps];
        long transfersBefore = cuda.RuntimeTransferBytesTotal;
        for (int rep = 0; rep < reps; rep++)
        {
            long busyStart = cuda.KernelBusyMicrosecondsTotal;
            Stopwatch sw = Stopwatch.StartNew();
            cuda.GemmBatch(inputs, batch, w, cudaOut);
            sw.Stop();
            cudaWall[rep] = sw.Elapsed.TotalMilliseconds;
            busyUs[rep] = cuda.KernelBusyMicrosecondsTotal - busyStart; // the seam adds only whole microseconds and skips zeros: a zero delta means unmeasured-or-sub-microsecond for that launch
        }

        long transferDelta = cuda.RuntimeTransferBytesTotal - transfersBefore;
        double transferBytesPerRep = (double)transferDelta / reps;
        bool kernelBusyReported = busyUs.Sum() > 0; // zero across every rep means the timing seam reported nothing, not that the kernel took no time

        double cpuScalarMed = Median(cpuScalarWall);
        double cpuAutoMed = Median(cpuAutoWall);
        double cudaWallMed = Median(cudaWall);
        double busyAvgUsPerRep = (double)busyUs.Sum() / reps;

        // One operation count for the whole pass, identical across arms: 2 flops per dot-product element. GFLOP/s divides it by that arm's median wall time (or kernel-busy microseconds for the compute-only figure).
        double passFlops = 2.0 * rows * cols * batch;

        Console.WriteLine(
            $"  cpu_scalar1T[{cpuRoute}] wall ms med={cpuScalarMed:F3} min={cpuScalarWall.Min():F3} max={cpuScalarWall.Max():F3} " +
            $"GFLOP/s={(passFlops / (cpuScalarMed * 0.001) / 1e9):F2}");
        Console.WriteLine(
            $"  cpu_auto[{cpuRoute}]    wall ms med={cpuAutoMed:F3} min={cpuAutoWall.Min():F3} max={cpuAutoWall.Max():F3} " +
            $"GFLOP/s={(passFlops / (cpuAutoMed * 0.001) / 1e9):F2} workers={cpuAuto.EffectiveWorkerCount}");

        if (kernelBusyReported)
        {
            Console.WriteLine(
                $"  cuda_gemm      wall ms med={cudaWallMed:F3} min={cudaWall.Min():F3} max={cudaWall.Max():F3} " +
                $"GFLOP/s={(passFlops / (cudaWallMed * 0.001) / 1e9):F2} xferBytesPerRep={transferBytesPerRep:F0}");
            long[] busySorted = [.. busyUs];
            Array.Sort(busySorted);
            int nBusy = busySorted.Length;
            double busyMed = nBusy % 2 == 1 ? busySorted[nBusy / 2] : (busySorted[nBusy / 2 - 1] + busySorted[nBusy / 2]) / 2.0;
            Console.WriteLine(
                $"  cuda_busy      kernel-busy us/rep avg={busyAvgUsPerRep:F1} med={busyMed:F1} min={busyUs.Min():F1} max={busyUs.Max():F1} " +
                $"GFLOP/s={(passFlops / (busyAvgUsPerRep * 0.000_001) / 1e9):F2} (device-measured, excludes PCIe/runtime transfer)");
        }
        else
        {
            Console.WriteLine(
                $"  cuda_gemm      wall ms med={cudaWallMed:F3} min={cudaWall.Min():F3} max={cudaWall.Max():F3} " +
                $"GFLOP/s={(passFlops / (cudaWallMed * 0.001) / 1e9):F2} xferBytesPerRep={transferBytesPerRep:F0} kernel_busy=unreported (timing events unavailable or every launch rounded below 1us)");
        }

        Console.WriteLine(
            $"  ratios         scalar_vs_cuda_wall={cpuScalarMed / cudaWallMed:F2} auto_vs_cuda_wall={cpuAutoMed / cudaWallMed:F2}" +
            (kernelBusyReported ? $" scalar_wall_vs_kernel_busy_only(compute-only)={cpuScalarMed * 1000.0 / busyAvgUsPerRep:F2}" : " compute_only=n/a"));

        return new FormatRow(
            format, weights.Length, cpuRoute, cpuScalarMed, cpuAutoMed, cudaWallMed,
            kernelBusyReported ? busyAvgUsPerRep : null,
            kernelBusyReported ? cpuScalarMed * 1000.0 / busyAvgUsPerRep : null,
            worstRel);
    }

    private static double[] TimeBackend(Action<ReadOnlySpan<float>, int, WeightRef, Span<float>> call, float[] inputs, int batch, WeightRef w, float[] scratchOutput, int reps)
    {
        // Warm-up untimed (rep -1), then every timed rep recorded so med/min/max all come from the same pass set.
        call(inputs, batch, w, scratchOutput);

        double[] wall = new double[reps];
        for (int rep = 0; rep < reps; rep++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            call(inputs, batch, w, scratchOutput);
            sw.Stop();
            wall[rep] = sw.Elapsed.TotalMilliseconds;
        }

        return wall;
    }

    private static double Median(double[] values)
    {
        double[] sorted = [.. values];
        Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    private static bool TryParsePositive(string value, out int parsed) =>
        int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out parsed) && parsed >= 1;

    private static string CollapseWhitespace(string text) => string.Join(' ', text.Split(" \t\r\n", StringSplitOptions.RemoveEmptyEntries));

    internal sealed record FormatRow(QuantFormat Format, long PackedBytes, string CpuRoute, double CpuScalarMedMs, double CpuAutoMedMs, double CudaWallMedMs, double? BusyUsAvgPerRep, double? ComputeOnlyRatio, double EquivMaxRelDrift);
}
