using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;

namespace Lair.Bench;

/// <summary>
/// Per-format CUDA-vs-CPU GEMV bench for the MVP5.2 Bench Gate: every QuantFormat the seam carries today is timed over a
/// deterministic pinned synthetic fixture (the same hand-built block patterns the GPU parity tests validate, as fixed
/// arithmetic progressions with no RNG) at one shape per run, so repeated runs measure identical bytes. For each format it
/// measures CpuBackend.GemvBatch host wall time on two labeled arms - a single-threaded scalar oracle arm (the basis every
/// recorded CUDA parity band was measured against) and the engine's auto-threaded posture with its effective worker count
/// reported explicitly - against CudaBackend.GemvBatch host wall time over repeated launches after an untimed warm-up, plus
/// the device-measured kernel-busy microseconds read off KernelBusyMicrosecondsTotal deltas around each timed launch. The
/// CUDA wall arm includes every per-call activation upload, output download, launch, and context sync the backend does; the
/// one-time weight copy a DeviceCopy posture caches on first use lands in the untimed warm-up instead. Ratios are printed
/// both as comparable total passes (CPU wall vs CUDA wall) and as compute-only (CPU scalar wall vs kernel-busy only), with
/// the second labeled as excluding PCIe/runtime transfer overhead. One untimed equivalence pass per format reports the max
/// relative drift between the CPU scalar chain's batched output and the CUDA chain's over identical inputs; no band is
/// asserted here, so a run never widens or narrows any recorded gate - it only states what this box measured. Output goes
/// to stdout only and carries the shape, reps, format list, and fixture description needed to reproduce every number.
/// </summary>
internal static class CudaGemvBench
{
    // Default shape: modest but representative decode-side geometry (a 512x1024 weight matrix consumed by a batch of 32
    // activation vectors), with enough reps that the first-measured pass and its warm passes separate without inflating
    // any format's median. All are overridable per run; --formats narrows the sweep when a subset is being re-measured.
    private const int DefaultRows = 512;
    private const int DefaultCols = 1024;
    private const int DefaultBatch = 32;
    private const int DefaultReps = 7;

    // The full supported sweep: every QuantFormat the CUDA seam resolves a GEMV kernel for today. F32/F16/BF16/Q8_0/
    // Q4_K/Q5_0/Q6_K are the pre-existing baselines; Q5_K/Q2_K/Q3_K/Q2_0 and the seven i-quants are the newly covered set.
    private static readonly QuantFormat[] DefaultFormats =
    [
        QuantFormat.F32, QuantFormat.F16, QuantFormat.BF16,
        QuantFormat.Q8_0, QuantFormat.Q4_K, QuantFormat.Q5_0, QuantFormat.Q6_K,
        QuantFormat.Q5_K, QuantFormat.Q2_K, QuantFormat.Q3_K, QuantFormat.Q2_0,
        QuantFormat.IQ4_NL, QuantFormat.IQ4_XS,
        QuantFormat.IQ2_XXS, QuantFormat.IQ2_XS, QuantFormat.IQ2_S, QuantFormat.IQ3_XXS, QuantFormat.IQ3_S,
    ];

    // Half-float headers chosen per block exactly as the GPU parity fixtures choose them: a normal magnitude, a negative
    // subnormal whose sign must survive widening on both sides of the seam, and an upper-range one. Deterministic cycling.
    private static readonly Half[] HeaderHalves = [(Half)0.25f, (Half)(-7.5e-06f), (Half)3.5f];

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
                Console.WriteLine($"cuda-gemv: option '{args[i]}' needs a value");
                return 1;
            }

            switch (args[i])
            {
                case "--rows":
                    if (!TryParsePositive(args[++i], out rows))
                    {
                        Console.WriteLine($"cuda-gemv: --rows must be a positive integer (got '{args[i]}')");
                        return 1;
                    }

                    break;
                case "--cols":
                    if (!TryParsePositive(args[++i], out cols))
                    {
                        Console.WriteLine($"cuda-gemv: --cols must be a positive integer (got '{args[i]}')");
                        return 1;
                    }

                    break;
                case "--batch":
                    if (!TryParsePositive(args[++i], out batch))
                    {
                        Console.WriteLine($"cuda-gemv: --batch must be a positive integer (got '{args[i]}')");
                        return 1;
                    }

                    break;
                case "--reps":
                    if (!TryParsePositive(args[++i], out reps))
                    {
                        Console.WriteLine($"cuda-gemv: --reps must be a positive integer (got '{args[i]}')");
                        return 1;
                    }

                    break;
                case "--formats":
                    string list = args[++i];
                    formatsOverride = [];
                    foreach (string part in list.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                    {
                        if (!Enum.TryParse(part, ignoreCase: true, out QuantFormat parsed) || !CudaBackend.HasGemvKernel(parsed))
                        {
                            Console.WriteLine($"cuda-gemv: unknown format '{part}'; expected one of {string.Join(",", DefaultFormats)}");
                            return 1;
                        }

                        if (!formatsOverride.Contains(parsed))
                        {
                            formatsOverride.Add(parsed);
                        }
                    }

                    if (formatsOverride.Count == 0)
                    {
                        Console.WriteLine("cuda-gemv: --formats listed no formats");
                        return 1;
                    }

                    break;
                default:
                    Console.WriteLine($"cuda-gemv: unknown option '{args[i]}' (--rows, --cols, --batch, --reps, --formats)");
                    return 1;
            }
        }

        if ((long)batch * cols > int.MaxValue || (long)batch * rows > int.MaxValue)
        {
            Console.WriteLine("cuda-gemv: batch x columns or batch x rows exceeds the addressable span limit");
            return 1;
        }

        IReadOnlyList<QuantFormat> formats = formatsOverride ?? (IReadOnlyList<QuantFormat>)DefaultFormats;

        // Open every participant before timing anything so no backend-open cost lands inside a measured rep. The CUDA
        // instance stays open for the whole sweep: its DeviceCopy posture caches each fixture tensor once, and its
        // cumulative kernel-busy/transfer tallies are read as per-format deltas around that format's timed reps.
        using CpuBackend cpuScalar = new(threads: 1, allowAvx2: false); // the scalar oracle arm: every recorded CUDA parity band was measured against this chain
        using CpuBackend cpuAuto = new(); // engine posture: auto thread choice; its effective worker count is reported explicitly

        CudaBackend? cuda;
        try
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                Console.WriteLine("cuda-gemv: the CUDA driver seam runs on Windows against nvcuda.dll; this host reports otherwise");
                return 2;
            }

            cuda = new CudaBackend(CudaWeightResidency.DeviceCopy);
        }
        catch (EntryPointNotFoundException ex)
        {
            Console.WriteLine($"cuda-gemv: CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point ({ex.Message})");
            return 2;
        }
        catch (DllNotFoundException ex)
        {
            Console.WriteLine($"cuda-gemv: CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API ({ex.Message})");
            return 2;
        }
        catch (CudaException ex)
        {
            Console.WriteLine($"cuda-gemv: CUDA backend refused to open on this host: {CollapseWhitespace(ex.Message)}");
            return 2;
        }

        try
        {
            Console.WriteLine("cuda-gemv bench gate measurement");
            Console.WriteLine(
                $"  shape rows={rows} cols={cols} batch={batch} reps={reps} formats={formats.Count} " +
                $"cpuArms=scalar1T(threads=1,avx2=false)+auto(writers below) cudaPosture=deviceCopy");
            Console.WriteLine(
                $"  host avx2={(Avx2.IsSupported ? "yes" : "no")} physicalCores={CpuBackend.DetectedPhysicalCoreCount} " +
                $"logicalProcessors={Environment.ProcessorCount} cpuAutoWorkers={cpuAuto.EffectiveWorkerCount}");
            Console.WriteLine(
                "  fixture=deterministic pinned block patterns (fixed arithmetic progressions, no RNG; the same packing " +
                "classes the GPU parity fixtures walk) activations=Sin(0.11*c+7*v)*1.3 fixed formula");
            Console.WriteLine(
                "  basis: cpu arms and cuda wall all run CpuBackend/CudaBackend GemvBatch over identical bytes, batch " +
                "activation vectors, count*rows output layout; cuda warm-up (including the one-time per-tensor device copy) is untimed; " +
                "cuda wall includes each call's activation upload, launch, sync, and download but not weight residency");

            List<FormatRow> table = [];

            foreach (QuantFormat format in formats)
            {
                FormatRow row = MeasureFormat(format, rows, cols, batch, reps, cpuScalar, cpuAuto, cuda);
                table.Add(row);
            }

            Console.WriteLine();
            Console.WriteLine("SUMMARY TABLE median-based; ratios >1 means CUDA faster");
            string[] headers = ["format", "packedMiB", "cpu_scalar_ms", "cpu_auto_ms", "cuda_wall_ms", "busy_us/rep", "ratio_scalar_cuda", "ratio_auto_cuda", "ratio_compute_only", "equiv_max_rel"];
            Console.WriteLine(string.Join("  ", headers.Select(h => h.PadLeft(Math.Max(h.Length, 13)))));
            foreach (FormatRow row in table)
            {
                string[] cells =
                [
                    row.Format.ToString(),
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
                "  equiv_max_rel = max |cpu-|cuda| / max(|cpu|,|cuda|) over the batched outputs of one untimed paired pass on identical inputs; 'exact' per format marks bit-equal chains. A large relative value usually names a near-zero row sum where both chains agree in absolute terms - read it together with max_abs_drift before comparing against bands recorded on the parity fixtures' pinned shapes (this bench's fixture magnitudes and shape differ). Bands recorded in tests are not re-asserted or widened here.");
            Console.WriteLine(
                "  caveat: wall numbers include GPU contention from other workloads on this box; kernel-busy is device-measured and excludes queue waits only to the extent the events measure launch-to-completion on the default stream");

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
        byte[] weights = BuildFixture(format, rows, cols); // deterministic pinned pattern: identical bytes on every run of this build
        WeightRef w = new(weights, shape, format);

        float[] inputs = new float[batch * cols];
        for (int v = 0; v < batch; v++)
        {
            for (int c = 0; c < cols; c++)
            {
                inputs[v * cols + c] = Activation(c + v * 7); // the same activation family the parity batch cases use, per vector offset
            }
        }

        float[] cpuOut = new float[batch * rows];
        float[] cudaOut = new float[batch * rows];

        Console.WriteLine();
        Console.WriteLine(
            $"FORMAT {format} rows={rows} cols={cols} batch={batch} strideBytes={strideBytes} " +
            $"bpw={(strideBytes * 8.0 / cols):F2} packedMiB={(weights.Length / (1024.0 * 1024.0)):F2}");

        // Untimed CUDA warm-up first: it caches this fixture's device weight copy and settles the timing-event pair, so
        // every timed rep below measures a steady-state launch instead of residency setup.
        cuda.GemvBatch(inputs, batch, w, cudaOut);

        // Equivalence sanity pass (untimed): one CPU scalar chain and one CUDA chain over identical inputs and bytes.
        cpuScalar.GemvBatch(inputs, batch, w, cpuOut);
        float[] equivalenceCpu = [.. cpuOut];
        cuda.GemvBatch(inputs, batch, w, cudaOut);

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
            $"  equivalence cpu_scalar1T vs cuda_batched: max_rel_drift={worstRel:G6} max_abs_drift={worstAbs:G6} bit_exact={(exactChain ? "yes" : "no")}" +
            (nanSeen ? " NaN_SEEN=yes" : string.Empty));

        // CPU arms: warm-up call then timed reps, same batch shape and output layout as the CUDA arm. The scalar arm is the
        // parity-band basis; the auto arm states its worker quota so a reader can tell which posture each ratio compares.
        // Equivalence values are already kept in their own array, so every timed pass writes through one scratch output.
        float[] scratch = new float[batch * rows];

        double[] cpuScalarWall = TimeBackend(cpuScalar.GemvBatch, inputs, batch, w, scratch, reps);
        double[] cpuAutoWall = TimeBackend(cpuAuto.GemvBatch, inputs, batch, w, scratch, reps);

        // CUDA arm: per-rep wall around the full GemvBatch call plus that launch's device-measured busy microseconds and
        // transfer accounting read as deltas of this instance's cumulative tallies.
        double[] cudaWall = new double[reps];
        long[] busyUs = new long[reps];
        long transfersBefore = cuda.RuntimeTransferBytesTotal;
        for (int rep = 0; rep < reps; rep++)
        {
            long busyStart = cuda.KernelBusyMicrosecondsTotal;
            Stopwatch sw = Stopwatch.StartNew();
            cuda.GemvBatch(inputs, batch, w, cudaOut);
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

        Console.WriteLine(
            $"  cpu_scalar1T   wall ms med={cpuScalarMed:F3} min={cpuScalarWall.Min():F3} max={cpuScalarWall.Max():F3} " +
            $"GB/s={(weights.Length * batch * 1000.0 / cpuScalarMed / 1e9):F2}");
        Console.WriteLine(
            $"  cpu_auto       wall ms med={cpuAutoMed:F3} min={cpuAutoWall.Min():F3} max={cpuAutoWall.Max():F3} " +
            $"GB/s={(weights.Length * batch * 1000.0 / cpuAutoMed / 1e9):F2} workers={cpuAuto.EffectiveWorkerCount}");

        if (kernelBusyReported)
        {
            Console.WriteLine(
                $"  cuda_wall      wall ms med={cudaWallMed:F3} min={cudaWall.Min():F3} max={cudaWall.Max():F3} " +
                $"xferBytesPerRep={transferBytesPerRep:F0}");
            long[] busySorted = [.. busyUs];
            Array.Sort(busySorted);
            int nBusy = busySorted.Length;
            double busyMed = nBusy % 2 == 1 ? busySorted[nBusy / 2] : (busySorted[nBusy / 2 - 1] + busySorted[nBusy / 2]) / 2.0;
            Console.WriteLine(
                $"  cuda_busy      kernel-busy us/rep avg={busyAvgUsPerRep:F1} med={busyMed:F1} min={busyUs.Min():F1} max={busyUs.Max():F1} (device-measured, excludes PCIe/runtime transfer)");
        }
        else
        {
            Console.WriteLine(
                $"  cuda_wall      wall ms med={cudaWallMed:F3} min={cudaWall.Min():F3} max={cudaWall.Max():F3} " +
                $"xferBytesPerRep={transferBytesPerRep:F0} kernel_busy=unreported (timing events unavailable or every launch rounded below 1us)");
        }

        Console.WriteLine(
            $"  ratios         scalar_vs_cuda_wall={cpuScalarMed / cudaWallMed:F2} auto_vs_cuda_wall={cpuAutoMed / cudaWallMed:F2}" +
            (kernelBusyReported ? $" scalar_wall_vs_kernel_busy_only(compute-only)={cpuScalarMed * 1000.0 / busyAvgUsPerRep:F2}" : " compute_only=n/a"));

        return new FormatRow(
            format, weights.Length, cpuScalarMed, cpuAutoMed, cudaWallMed,
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

    internal static float Activation(int c) => (float)Math.Sin(0.11 * c + 1.7) * 1.3f; // the parity fixtures' deterministic activation family, shared with cuda-gemm's sweep so both benches feed identical vectors

    private static void WriteHeaderLE(Span<byte> dst, int offset, Half value)
    {
        ushort bits = (ushort)BitConverter.HalfToInt16Bits(value);
        dst[offset] = (byte)bits;
        dst[offset + 1] = (byte)(bits >> 8);
    }

    // A nibble window whose low and high codes walk the whole four-bit range across successive bytes, as in the IQ fixtures.
    private static byte NibblePair(int k, int r, int b) =>
        (byte)(((k * 5 + r * 3 + b * 7) & 0xF) | (((k * 11 + r * 13 + b * 2) & 0xF) << 4));

    // Every fixture below is the pinned pattern from the GPU parity fixtures for its format (same field offsets, scale
    // choices, and code progressions), so these bytes are already known to decode identically on both backends. Values
    // stay bounded by construction: half-float headers cycle through one normal, one negative subnormal, and one upper-range magnitude.

    // internal because the cuda-gemm sweep builds its fixtures through this same builder: identical pinned weight bytes per format across both benches is what makes their numbers comparable on one box.
    internal static byte[] BuildFixture(QuantFormat format, int rows, int cols) => format switch
    {
        QuantFormat.F32 or QuantFormat.F16 or QuantFormat.BF16 => BuildDenseFixture(format, rows, cols),
        QuantFormat.Q8_0 => BuildQ8_0Fixture(rows, cols),
        QuantFormat.Q5_0 => BuildQ5_0Fixture(rows, cols),
        QuantFormat.Q2_0 => BuildQ2_0Fixture(rows, cols),
        QuantFormat.Q4_K => BuildQ4KFixture(rows, cols),
        QuantFormat.Q6_K => BuildQ6KFixture(rows, cols),
        QuantFormat.Q5_K => BuildQ5KFixture(rows, cols),
        QuantFormat.Q2_K => BuildQ2KFixture(rows, cols),
        QuantFormat.Q3_K => BuildQ3KFixture(rows, cols),
        QuantFormat.IQ4_NL => BuildIQ4NlFixture(rows, cols),
        QuantFormat.IQ4_XS => BuildIQ4XsFixture(rows, cols),
        _ => BuildQ8KSuperBlockFixture(format, rows, cols), // IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S share one swept-body shape
    };

    private static byte[] BuildDenseFixture(QuantFormat format, int rows, int cols)
    {
        long stride = new TensorShape([rows, cols]).RowStrideBytes(format);
        byte[] bytes = new byte[checked((int)(rows * stride))];

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                // A fixed integer ramp mapped into [0.15, 1.15): deterministic across runs, bounded in magnitude, and
                // positive-biased so row sums stay away from systematic cancellation - a uniform +/- ramp would make the
                // equivalence check report ULP-level agreement as percent-scale drift against near-zero totals instead.
                float v = 0.15f + ((r * 7 + c * 13) % 256) / 256f;

                int offset = (int)(r * stride) + c * (format == QuantFormat.F32 ? 4 : 2);
                if (format == QuantFormat.F32)
                {
                    uint bits = unchecked((uint)BitConverter.SingleToInt32Bits(v));
                    bytes[offset] = (byte)bits;
                    bytes[offset + 1] = (byte)(bits >> 8);
                    bytes[offset + 2] = (byte)(bits >> 16);
                    bytes[offset + 3] = (byte)(bits >> 24);
                }
                else
                {
                    ushort bits = (ushort)BitConverter.HalfToInt16Bits((Half)v);
                    if (format == QuantFormat.BF16)
                    {
                        bits = (ushort)(bits << 8); // the half's precision widened to bf16 storage: same value class, this format's own bit layout
                    }

                    bytes[offset] = (byte)bits;
                    bytes[offset + 1] = (byte)(bits >> 8);
                }
            }
        }

        return bytes;
    }

    private static byte[] BuildQ8_0Fixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 31) / 32;
        byte[] bytes = new byte[rows * blocksPerRow * 34];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 34;
                WriteHeaderLE(bytes, block, HeaderHalves[(r + b) % HeaderHalves.Length]);

                int limit = Math.Min(32, cols - b * 32); // padded quants stay zero: the real column count bounds them.
                for (int l = 0; l < limit; l++)
                {
                    sbyte q = (sbyte)(((r * 17 + b * 5 + l * 41) % 256) - 128); // walks the whole int8 range including both extremes
                    bytes[block + 2 + l] = (byte)q;
                }
            }
        }

        return bytes;
    }

    private static byte[] BuildQ5_0Fixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 31) / 32;
        byte[] bytes = new byte[rows * blocksPerRow * 22];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 22;
                WriteHeaderLE(bytes, block, HeaderHalves[(r * 2 + b) % HeaderHalves.Length]);

                int limit = Math.Min(32, cols - b * 32);
                for (int l = 0; l < limit; l++)
                {
                    byte nibble = (byte)((r * 3 + b * 7 + l * 11) & 0xF);
                    if (l < 16)
                    {
                        bytes[block + 6 + l] |= nibble; // low nibbles elements 0..15, high nibbles 16..31 of the same pair byte
                    }
                    else
                    {
                        bytes[block + 6 + (l - 16)] |= (byte)(nibble << 4);
                    }

                    int fifthBit = (r * 5 + b * 3 + l * 7) & 1; // both states across every element position
                    bytes[block + 2 + (l >> 3)] |= (byte)(fifthBit << (l & 7)); // little-endian fifth-bit word: element l keeps bit l
                }
            }
        }

        return bytes;
    }

    private static byte[] BuildQ2_0Fixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 63) / 64;
        byte[] bytes = new byte[rows * blocksPerRow * 18];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 18;
                WriteHeaderLE(bytes, block, HeaderHalves[(r + b * 3) % HeaderHalves.Length]);

                int limit = Math.Min(64, cols - b * 64);
                for (int l = 0; l < limit; l++)
                {
                    byte two = (byte)((r * 3 + b * 5 + l) & 3); // every shift class of the shared window byte appears per block
                    bytes[block + 2 + (l >> 2)] |= (byte)(two << (2 * (l & 3)));
                }
            }
        }

        return bytes;
    }

    private static byte[] BuildQ4KFixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 255) / 256;
        byte[] bytes = new byte[rows * blocksPerRow * 144];

        Half dHalf = (Half)0.25f;
        Half mHalf = (Half)0.125f;
        ushort dBits = (ushort)BitConverter.HalfToInt16Bits(dHalf);
        ushort mBits = (ushort)BitConverter.HalfToInt16Bits(mHalf);

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 144;

                bytes[block] = (byte)dBits;
                bytes[block + 1] = (byte)(dBits >> 8);
                bytes[block + 2] = (byte)mBits;
                bytes[block + 3] = (byte)(mBits >> 8);

                byte[] sc = new byte[8];
                byte[] mn = new byte[8];
                for (int j = 0; j < 8; j++)
                {
                    sc[j] = (byte)(5 + 7 * j); // groups 4-7 land at 33..54: their high bits live in header words one and two
                    mn[j] = (byte)(3 + 5 * j); // groups 4-7 land at 23..38 likewise
                }

                for (int k = 0; k < 4; k++)
                {
                    bytes[block + 4 + k] = (byte)((sc[k] & 0x3F) | ((sc[k + 4] >> 4) << 6));
                    bytes[block + 8 + k] = (byte)((mn[k] & 0x3F) | ((mn[k + 4] >> 4) << 6));
                    bytes[block + 12 + k] = (byte)((sc[k + 4] & 0xF) | ((mn[k + 4] & 0xF) << 4));
                }

                for (int j = 0; j < 8; j++)
                {
                    int subStart = j * 32;
                    if (subStart >= cols - b * 256)
                    {
                        break; // padded tail: no real columns in this group
                    }

                    int limit = Math.Min(32, cols - b * 256 - subStart);
                    int window = block + 16 + 32 * (j / 2);
                    for (int l = 0; l < limit; l++)
                    {
                        byte code = (byte)((j * 3 + l) & 0xF);
                        if ((j & 1) == 0)
                        {
                            bytes[window + l] |= code;
                        }
                        else
                        {
                            bytes[window + l] |= (byte)(code << 4);
                        }
                    }
                }
            }
        }

        return bytes;
    }

    private static byte[] BuildQ6KFixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 255) / 256;
        byte[] bytes = new byte[rows * blocksPerRow * 210];

        Half dHalf = (Half)0.25f; // a fixed normal magnitude: the fixture exercises every nibble class, not header extremes per block
        ushort dBits = (ushort)BitConverter.HalfToInt16Bits(dHalf);

        sbyte[] scalesByGroup = [88, 47, 110, -128, 57, 106, -72, -59, 84, 86, -62, 95, 78, -120, -73, 71]; // signed group scales including both extremes

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 210;

                bytes[block + 208] = (byte)dBits;
                bytes[block + 209] = (byte)(dBits >> 8);

                for (int g = 0; g < 16; g++)
                {
                    int subStart = g * 16;
                    if (subStart >= cols - b * 256)
                    {
                        break; // padded tail: no real columns in this group
                    }

                    bytes[block + 192 + g] = (byte)scalesByGroup[g];

                    int h = g >> 3;      // 128-element half of the super-block
                    int k = g & 7;       // 16-element group inside the half
                    int quarter = k >> 1; // which 32-element quarter of the half this group starts in
                    int limit = Math.Min(16, cols - b * 256 - subStart);

                    for (int e = 0; e < limit; e++)
                    {
                        int six = ((g * 4 + e) & 0x3F) ^ (((r + b) << 3) & 0x3F); // distinct codes across every nibble class

                        ref byte qbyte = ref bytes[block + 64 * h + 16 * (k & 3) + e];
                        if ((quarter >> 1) == 0)
                        {
                            qbyte = (byte)((qbyte & 0xF0) | (six & 0xF));
                        }
                        else
                        {
                            qbyte = (byte)((qbyte & 0x0F) | (six << 4));
                        }

                        bytes[block + 128 + 32 * h + 16 * (k & 1) + e] |= (byte)((six >> 4) << (2 * quarter));
                    }
                }
            }
        }

        return bytes;
    }

    private static byte[] BuildQ5KFixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 255) / 256;
        byte[] bytes = new byte[rows * blocksPerRow * 176];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 176;
                WriteHeaderLE(bytes, block, HeaderHalves[(r + b) % HeaderHalves.Length]); // d at bytes 0..1 ...
                WriteHeaderLE(bytes, block + 2, (Half)(-0.5f)); // ... and min at bytes 2..3

                byte[] sc = new byte[8];
                byte[] mn = new byte[8];
                for (int j = 0; j < 8; j++)
                {
                    sc[j] = (byte)(6 + 7 * j); // groups 4-7 land at 34..55: their high bits split across two header words
                    mn[j] = (byte)((j == 2) ? 0 : (1 + 6 * j)); // one zero-scaled group sits beside non-zero ones
                }

                for (int k = 0; k < 4; k++)
                {
                    bytes[block + 4 + k] = (byte)((sc[k] & 0x3F) | ((sc[k + 4] >> 4) << 6));
                    bytes[block + 8 + k] = (byte)((mn[k] & 0x3F) | ((mn[k + 4] >> 4) << 6));
                    bytes[block + 12 + k] = (byte)((sc[k + 4] & 0xF) | ((mn[k + 4] & 0xF) << 4));
                }

                for (int j = 0; j < 8; j++)
                {
                    int subStart = j * 32;
                    if (subStart >= cols - b * 256)
                    {
                        break; // padded tail: no real columns in this group
                    }

                    int limit = Math.Min(32, cols - b * 256 - subStart);
                    for (int l = 0; l < limit; l++)
                    {
                        byte nibble = (byte)((r * 3 + j * 7 + l * 5) & 0xF); // distinct codes across both halves of a pair window
                        int windowByte = block + 48 + 32 * (j / 2) + l;
                        if ((j & 1) == 0)
                        {
                            bytes[windowByte] |= nibble;
                        }
                        else
                        {
                            bytes[windowByte] |= (byte)(nibble << 4);
                        }

                        int fifthBit = (r * 5 + j * 3 + l * 11) & 1; // one plane byte carries eight different groups' bits
                        bytes[block + 16 + l] |= (byte)(fifthBit << j);
                    }
                }
            }
        }

        return bytes;
    }

    private static byte[] BuildQ2KFixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 255) / 256;
        byte[] bytes = new byte[rows * blocksPerRow * 84];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 84;

                for (int g = 0; g < 16; g++)
                {
                    byte scaleNibble = (byte)((g == 3 || g == 12) ? 0 : ((r * 5 + b * 3 + g * 7) & 0xF)); // zero-scaled groups beside non-zero ones
                    byte minNibble = (byte)((b * 9 + g * 5 + r) & 0xF);
                    bytes[block + g] = (byte)(scaleNibble | (minNibble << 4)); // one byte carries both this group's four-bit scale and its four-bit min
                }

                WriteHeaderLE(bytes, block + 80, HeaderHalves[(r * 2 + b) % HeaderHalves.Length]); // d at bytes 80..81 ...
                WriteHeaderLE(bytes, block + 82, (Half)(-3.5f)); // ... and min at bytes 82..83

                for (int g = 0; g < 16; g++)
                {
                    int h = g >> 3;
                    int k = g & 7;

                    int subStart = g * 16;
                    if (subStart >= cols - b * 256)
                    {
                        break; // padded tail: no real columns in this group
                    }

                    int limit = Math.Min(16, cols - b * 256 - subStart);
                    for (int e = 0; e < limit; e++)
                    {
                        byte two = (byte)((r * 3 + g * 7 + e * 5) & 3); // every shift class of the shared window byte appears per quarter
                        bytes[block + 16 + 32 * h + 16 * (k & 1) + e] |= (byte)(two << (2 * (k >> 1)));
                    }
                }
            }
        }

        return bytes;
    }

    private static byte[] BuildQ3KFixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 255) / 256;
        byte[] bytes = new byte[rows * blocksPerRow * 110];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 110;

                byte[] scales = new byte[12]; // twelve bytes pack sixteen six-bit group scales at bytes 96..107
                for (int g = 0; g < 16; g++)
                {
                    int code = (r * 5 + b * 3 + g * 4) & 0x3F; // walks below, at, and above the centering point of 32
                    byte low = (byte)(code & 0xF);
                    byte high = (byte)((code >> 4) & 0x3);

                    if (g < 8)
                    {
                        scales[g] |= low; // groups 0..7 keep their scale's low nibble in scales[g] ...
                    }
                    else
                    {
                        scales[g - 8] |= (byte)(low << 4); // ... groups 8..15 take the high nibble of scales[g-8]
                    }

                    scales[8 + (g & 3)] |= (byte)(high << (2 * (g >> 2))); // every scale's four high bits share one byte at a constant shift
                }

                for (int i = 0; i < 12; i++)
                {
                    bytes[block + 96 + i] = scales[i];
                }

                WriteHeaderLE(bytes, block + 108, HeaderHalves[(r + b * 3) % HeaderHalves.Length]); // d at bytes 108..109

                for (int g = 0; g < 16; g++)
                {
                    int h = g >> 3;
                    int k = g & 7;
                    int m = 4 * h + (k >> 1); // this group's bit position inside every element of its quarter's hmask byte

                    int subStart = g * 16;
                    if (subStart >= cols - b * 256)
                    {
                        break; // padded tail: no real columns in this group
                    }

                    int limit = Math.Min(16, cols - b * 256 - subStart);
                    for (int e = 0; e < limit; e++)
                    {
                        byte two = (byte)((r * 3 + g * 7 + e * 5) & 3); // low two bits of the three-bit code, at this group's shift ...
                        bytes[block + 32 + 32 * h + 16 * (k & 1) + e] |= (byte)(two << (2 * (k >> 1)));

                        int highBit = (r * 5 + g * 3 + e * 7) & 1; // ... and its high bit, both states across every group of a half
                        bytes[block + 16 * (k & 1) + e] |= (byte)(highBit << m);
                    }
                }
            }
        }

        return bytes;
    }

    private static byte[] BuildIQ4NlFixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 31) / 32;
        byte[] bytes = new byte[rows * blocksPerRow * 18];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 18;

                WriteHeaderLE(bytes, block, HeaderHalves[(r * 2 + b) % HeaderHalves.Length]); // d at bytes 0..1.
                for (int k = 0; k < 16; k++)
                {
                    bytes[block + 2 + k] = NibblePair(k, r, b); // every table code value appears somewhere across the sweep.
                }
            }
        }

        return bytes;
    }

    private static byte[] BuildIQ4XsFixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 255) / 256;
        byte[] bytes = new byte[rows * blocksPerRow * 136];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 136;

                WriteHeaderLE(bytes, block, HeaderHalves[(r * 2 + b) % HeaderHalves.Length]); // d at bytes 0..1.

                ushort scalesH = unchecked((ushort)(0x5A3C + r * 97 + b * 61)); // every sub-block's two high bits vary, including the zero pair and the three pair
                bytes[block + 2] = (byte)scalesH;
                bytes[block + 3] = (byte)(scalesH >> 8);

                for (int k = 0; k < 4; k++)
                {
                    bytes[block + 4 + k] = unchecked((byte)(k * 17 + r * 53 + b * 29)); // low nibbles walk the whole range in both halves of each byte.
                }

                for (int ib = 0; ib < 8; ib++)
                {
                    for (int k = 0; k < 16; k++)
                    {
                        bytes[block + 8 + 16 * ib + k] = NibblePair(k, r + ib, b); // each window walks the whole code range independently.
                    }
                }
            }
        }

        return bytes;
    }

    private static byte[] BuildQ8KSuperBlockFixture(QuantFormat format, int rows, int cols)
    {
        int stride = format switch
        {
            QuantFormat.IQ2_XXS => 66,
            QuantFormat.IQ2_XS => 74,
            QuantFormat.IQ2_S => 82,
            QuantFormat.IQ3_XXS => 98,
            QuantFormat.IQ3_S => 110,
            _ => throw new ArgumentException($"{format} is not a sub-4-bit i-quant format", nameof(format)),
        };

        int blocksPerRow = (cols + 255) / 256; // all five run in Q8_K super-blocks
        byte[] bytes = new byte[rows * blocksPerRow * stride];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * stride;

                for (int i = 2; i < stride; i++) // the swept body: every position passes through its whole byte range across cases.
                {
                    bytes[block + i] = unchecked((byte)(i * 13 + r * 97 + b * 53));
                }

                WriteHeaderLE(bytes, block, HeaderHalves[(r * 2 + b) % HeaderHalves.Length]); // d at bytes 0..1.
            }
        }

        return bytes;
    }

    private static bool TryParsePositive(string value, out int parsed) =>
        int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out parsed) && parsed >= 1;

    private static string CollapseWhitespace(string text) => string.Join(' ', text.Split(" \t\r\n", StringSplitOptions.RemoveEmptyEntries));

    internal sealed record FormatRow(QuantFormat Format, long PackedBytes, double CpuScalarMedMs, double CpuAutoMedMs, double CudaWallMedMs, double? BusyUsAvgPerRep, double? ComputeOnlyRatio, double EquivMaxRelDrift);
}
