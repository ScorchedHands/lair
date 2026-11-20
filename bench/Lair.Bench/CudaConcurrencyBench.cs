using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

using Lair.Backends.Cuda;
using Lair.Contracts;

namespace Lair.Bench;

/// <summary>
/// MVP6.1 co-scheduling payoff measurement: can one process / one shared primary CUDA context run two independent weight passes
/// faster on two CUstreams than the seam's own serialized default-stream pipeline? The workloads are the same embedded-PTX kernels
/// and deterministic pinned fixtures the CUDA GEMV/GEMM benches already time - identical function handles (resolved from the module a
/// live CudaBackend loads), identical grid geometry, identical kernel parameter order, identical activation family and weight bytes.
/// Only stream usage differs between arms:
///   serialized arm  = workload A then workload B exactly as the seam does today per call (synchronous H2D upload on the legacy NULL
///                     stream -> cuLaunchKernel(stream=0) -> cuCtxSynchronize -> synchronous D2H download), so each pass fully
///                     completes before the next one starts;
///   overlapped arm  = A's async H2D + kernel launch + async D2H enqueued on CUstream A and B's on CUstream B, both pipelines submitted
///                     back to back without awaiting either, then both streams awaited - what two served models' lanes would look like
///                     if launches carried explicit streams.
/// Each workload also gets a single-workload baseline arm in the same interleaved rep schedule, so the serialized total can be checked
/// against its parts and an overlapped wall compared with max(A,B) as well as with the serial sum. Device-measured busy time comes from
/// driver events recorded on each arm's own stream (the same cuEventRecord/cuEventElapsedTime pair engine accounting uses), so host-side
/// payoff and device-busy concurrency are reported separately. One untimed equivalence pass per workload compares serialized-arm outputs
/// against overlapped-arm outputs bit for bit: it proves both arms measured the same work, and no band is asserted or widened anywhere
/// here - this bench only states what this box did.
///
/// Production posture is untouched by this file: CudaBackend's public Gemv/GemvBatch/GemmBatch still launch on the legacy default stream
/// with a per-call context sync, serve paths are unchanged, and no config surface exists for these streams. The two seams it depends on
/// are measurement-only by construction - it reads the seam's already-loaded module handle (a private field through reflection) to resolve
/// kernel function objects and hand-launches them itself with explicit CUstream arguments; nothing in src/Lair.Backends.Cuda is modified,
/// no default changes anywhere, and normal serving never sees a stream or event created here. The structural finding this bench exists to
/// quantify is exactly why it hand-rolls launches: today CudaBackend cannot express concurrent launches at all (one instance-wide gate
/// around every device call, per-call activation/output alloc+free, a hardcoded NULL-stream launch, and a cuCtxSynchronize inside each
/// call), so overlap has no seam to land on without an MVP6.1 stream/event change - which this spike measures the payoff of rather than
/// implements.
/// </summary>
internal static class CudaConcurrencyBench
{
    // Grid geometry mirrors the seam exactly: one weight row per thread in a 128-thread block, ceil(rows/128) blocks - the same
    // KernelThreads constant and formula every GEMV and blocked-matmul launch on this seam uses. It is hardcoded here because CudaBackend's
    // copy is a compile-time constant with no runtime-visible form; if that geometry ever moves, these launches move with it or stop naming
    // the real kernels (the resolved function handles are then the only thing still borrowed from the seam).
    private const int KernelThreads = 128;

    // CU_STREAM_NON_BLOCKING: the two lanes must not implicitly rendezvous with the legacy NULL stream, or "overlapped" would serialize
    // against anything else in the process exactly like today does.
    private const uint StreamNonBlocking = 0x1;

    // Default shape is a representative decode-side weight matrix (2048x2048 consumed by a batch of 32 activation vectors): big enough that
    // each kernel's device-measured busy time sits well above submission noise, small enough to keep every pair's resident weights modest
    // on any CUDA host. All overridable per run.
    private const int DefaultRows = 2048;
    private const int DefaultCols = 2048;
    private const int DefaultBatch = 32;
    private const int DefaultReps = 7;   // timed reps per arm, interleaved with the other arms so clock boost cannot favor one
    private const int DefaultSweeps = 3; // whole-measurement repeats: medians are taken within a sweep and across every rep of every sweep

    // The default pair set covers the three co-scheduling shapes MVP6.1 cares about - each pair is two independent weight passes on identical
    // geometry with different formats/kernels: one GEMV lane beside one blocked-matmul (prefill-like) lane, two GEMV lanes, and two
    // blocked-matmul lanes. Formats stay inside the six-parameter kernel family (no IQ4 scratch-table cursor, no Q8_K host-side activation
    // quantization), so every launch here is a faithful copy of that one parameter order on the seam rather than a reimplementation of an
    // i-quant helper's staging.
    private static readonly Pair[] DefaultPairs =
    [
        new("mixed_gemv_q4k__gemm_f16", KernelKind.GemV, QuantFormat.Q4_K, KernelKind.Gemm, QuantFormat.F16),
        new("two_gemv_q8_0__q6_k", KernelKind.GemV, QuantFormat.Q8_0, KernelKind.GemV, QuantFormat.Q6_K),
        new("two_gemm_q8_0__bf16", KernelKind.Gemm, QuantFormat.Q8_0, KernelKind.Gemm, QuantFormat.BF16),
    ];

    public static int Run(string[] args)
    {
        int rows = DefaultRows;
        int cols = DefaultCols;
        int batch = DefaultBatch;
        int reps = DefaultReps;
        int sweeps = DefaultSweeps;
        List<Pair>? pairsOverride = null;
        string? outPath = null;

        for (int i = 1; i < args.Length; i++)
        {
            if (i + 1 >= args.Length)
            {
                Console.WriteLine($"cuda-concurrency: option '{args[i]}' needs a value");
                return 1;
            }

            switch (args[i])
            {
                case "--rows":
                    if (!TryParsePositive(args[++i], out rows))
                    {
                        Console.WriteLine($"cuda-concurrency: --rows must be a positive integer (got '{args[i]}')");
                        return 1;
                    }

                    break;
                case "--cols":
                    if (!TryParsePositive(args[++i], out cols))
                    {
                        Console.WriteLine($"cuda-concurrency: --cols must be a positive integer (got '{args[i]}')");
                        return 1;
                    }

                    break;
                case "--batch":
                    if (!TryParsePositive(args[++i], out batch))
                    {
                        Console.WriteLine($"cuda-concurrency: --batch must be a positive integer (got '{args[i]}')");
                        return 1;
                    }

                    break;
                case "--reps":
                    if (!TryParsePositive(args[++i], out reps))
                    {
                        Console.WriteLine($"cuda-concurrency: --reps must be a positive integer (got '{args[i]}')");
                        return 1;
                    }

                    break;
                case "--sweeps":
                    if (!TryParsePositive(args[++i], out sweeps))
                    {
                        Console.WriteLine("cuda-concurrency: --sweeps must be a positive integer (got '{args[i]}')");
                        return 1;
                    }

                    break;
                case "--pairs":
                    string list = args[++i];
                    pairsOverride = [];
                    foreach (string part in list.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                    {
                        Pair? match = DefaultPairs.FirstOrDefault(p => p.Name.Equals(part, StringComparison.OrdinalIgnoreCase));
                        if (match is null)
                        {
                            Console.WriteLine($"cuda-concurrency: unknown pair '{part}'; expected one of {string.Join(",", DefaultPairs.Select(p => p.Name))}");
                            return 1;
                        }

                        if (!pairsOverride.Contains(match))
                        {
                            pairsOverride.Add(match);
                        }
                    }

                    if (pairsOverride.Count == 0)
                    {
                        Console.WriteLine("cuda-concurrency: --pairs listed no pairs");
                        return 1;
                    }

                    break;
                case "--out":
                    outPath = args[++i];
                    break;
                default:
                    Console.WriteLine($"cuda-concurrency: unknown option '{args[i]}' (--rows, --cols, --batch, --reps, --sweeps, --pairs, --out)");
                    return 1;
            }
        }

        if ((long)batch * cols > int.MaxValue || (long)batch * rows > int.MaxValue)
        {
            Console.WriteLine("cuda-concurrency: batch x columns or batch x rows exceeds the addressable span limit");
            return 1;
        }

        IReadOnlyList<Pair> pairs = pairsOverride ?? (IReadOnlyList<Pair>)DefaultPairs;

        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            Console.WriteLine("cuda-concurrency: the CUDA driver seam runs on Windows against nvcuda.dll; this host reports otherwise");
            return 2;
        }

        // Open the real backend first, exactly as a served model would. It validates device availability, retains the shared primary context,
        // and loads the embedded PTX module whose function handles every launch below reuses - so these measurements run against byte-identical
        // kernels to serving rather than a second JIT'd copy of the source text.
        CudaBackend backend;
        try
        {
            backend = new CudaBackend(CudaWeightResidency.DeviceCopy);
        }
        catch (EntryPointNotFoundException ex)
        {
            Console.WriteLine($"cuda-concurrency: CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point ({ex.Message})");
            return 2;
        }
        catch (DllNotFoundException ex)
        {
            Console.WriteLine($"cuda-concurrency: CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API ({ex.Message})");
            return 2;
        }
        catch (CudaException ex)
        {
            Console.WriteLine($"cuda-concurrency: CUDA backend refused to open on this host: {CollapseWhitespace(ex.Message)}");
            return 2;
        }

        try
        {
            // The seam's already-loaded module is the only thing read out of it. This is a measurement-only borrow: no src change, no stream
            // becomes default anywhere, and serve paths never see any handle created here.
            FieldInfo? moduleField = typeof(CudaBackend).GetField("_module", BindingFlags.NonPublic | BindingFlags.Instance);
            if (moduleField is null)
            {
                Console.WriteLine("cuda-concurrency: could not locate the seam's loaded PTX module field; this measurement cannot name the real kernels");
                return 2;
            }

            IntPtr module = (IntPtr)(moduleField.GetValue(backend) ?? throw new InvalidOperationException("the backend's module handle is unset"));

            // Re-retain the same primary context and bind it on this thread explicitly, so every launch below runs in the process-wide shared
            // primary context a served model uses. Primary contexts are singletons: this answers with the very one CudaBackend retained (device
            // ordinal 0 there too), only carrying an extra reference this bench releases at the end.
            ConcurrencyCudaNative.Check("cuInit", ConcurrencyCudaNative.cuInit(0));
            ConcurrencyCudaNative.Check("cuDeviceGetCount", ConcurrencyCudaNative.cuDeviceGetCount(out int deviceCount));
            if (deviceCount < 1)
            {
                Console.WriteLine("cuda-concurrency: the driver reports no CUDA device");
                return 2;
            }

            ConcurrencyCudaNative.Check("cuDeviceGet", ConcurrencyCudaNative.cuDeviceGet(out int device, 0));
            ConcurrencyCudaNative.Check("cuCtxSetCurrent", ConcurrencyCudaNative.cuDevicePrimaryCtxRetain(out IntPtr context, device));
            ConcurrencyCudaNative.Check("cuCtxSetCurrent", ConcurrencyCudaNative.cuCtxSetCurrent(context));

            try
            {
                return Measure(module, device, pairs, rows, cols, batch, reps, sweeps, outPath);
            }
            finally
            {
                ConcurrencyCudaNative.Check("cuDevicePrimaryCtxRelease", ConcurrencyCudaNative.cuDevicePrimaryCtxRelease(device));
            }
        }
        catch (ConcurrencyCudaException ex)
        {
            Console.WriteLine($"cuda-concurrency: driver call failed on this host: {CollapseWhitespace(ex.Message)}");
            return 2;
        }
        catch (EntryPointNotFoundException ex)
        {
            // A driver older than the stream/async-copy entry points this measurement needs: report it instead of pretending.
            Console.WriteLine($"cuda-concurrency: the installed driver lacks a required entry point ({ex.Message})");
            return 2;
        }
        finally
        {
            backend.Dispose(); // releases the seam's module and its primary-context reference after every measurement is done
        }
    }

    private static int Measure(IntPtr module, int deviceOrdinal, IReadOnlyList<Pair> pairs, int rows, int cols, int batch, int reps, int sweeps, string? outPath)
    {
        unsafe
        {
            ConcurrencyCudaNative.Check("cuDriverGetVersion", ConcurrencyCudaNative.cuDriverGetVersion(out int driverVersion));

            byte* nameBuffer = stackalloc byte[128]; // same ordinal the seam opens on: device 0 of whatever this host reports
            ConcurrencyCudaNative.Check("cuDeviceGetName", ConcurrencyCudaNative.cuDeviceGetName(nameBuffer, 128, deviceOrdinal));
            string deviceName = ReadAsciiNulTerminated(nameBuffer);

            Console.WriteLine("cuda-concurrency co-scheduling payoff measurement");
            Console.WriteLine(
                $"  shape rows={rows} cols={cols} batch={batch} reps={reps} sweeps={sweeps} pairs={pairs.Count} " +
                $"arms=single+serialized+twoStreamSerialized(control)+overlapped kernelThreads={KernelThreads} streamFlags=CU_STREAM_NON_BLOCKING");
            Console.WriteLine(
                $"  device ordinal={deviceOrdinal} name='{deviceName}' driverApiVersion={driverVersion} osDescription=\"{RuntimeInformation.OSDescription}\" " +
                $"logicalProcessors={Environment.ProcessorCount} primaryContext=shared(retained by CudaBackend, re-retained here)");
            Console.WriteLine(
                "  fixture=deterministic pinned block patterns from cuda-gemv's builder (fixed arithmetic progressions, no RNG; identical bytes " +
                "per format on every run of this build), activations=cuda-gemv/cuda-gemm's Sin(0.11*c+7*v)*1.3 family");
            Console.WriteLine(
                "  basis: both arms launch the same seam kernels (function handles resolved from the module CudaBackend loaded, same grid geometry " +
                "ceil(rows/128) x 128 threads and same six-parameter order). Weights are uploaded once per workload (untimed residency), activation staging " +
                "is page-locked host memory allocated by cuMemAllocHost_v2, device activation/output buffers persist across reps - so every timed number " +
                "measures steady-state launches rather than alloc churn. Serialized arm reproduces the seam's per-call pipeline (sync H2D -> launch on legacy " +
                "NULL stream -> ctx sync -> sync D2H) for A then B; overlapped arm submits each workload's async H2D + kernel + async D2H to its own non-blocking " +
                "CUstream before awaiting either. A third control arm uses the same lanes and async staging but awaits each lane before submitting the next workload, so the ratio between that control and the overlapped arm is concurrency alone while the ratio between today's pipeline and the control is what the lane machinery itself costs. Kernel busy is device-measured by events recorded on that arm's stream (kernel-only: the staging copies sit " +
                "outside the stamp span exactly as engine accounting puts them outside its tallies).");

            // Two persistent lanes for the whole run: created once, like two model lanes would be, never awaited between pairs.
            ConcurrencyCudaNative.Check("cuStreamCreate", ConcurrencyCudaNative.cuStreamCreate(out IntPtr laneA, StreamNonBlocking));
            try
            {
                ConcurrencyCudaNative.Check("cuStreamCreate", ConcurrencyCudaNative.cuStreamCreate(out IntPtr laneB, StreamNonBlocking));
                try
                {
                    List<PairResult> results = [];

                    foreach (Pair pair in pairs)
                    {
                        if (!CudaBackend.HasGemmKernel(pair.FormatA) || !CudaBackend.HasGemmKernel(pair.FormatB))
                        {
                            // The seam refuses a blocked matmul for the Q8_K i-quants; this bench keeps to formats it can name.
                            Console.WriteLine($"PAIR {pair.Name}: refused - the seam carries no windowed kernel for one of its formats");
                            continue;
                        }

                        results.Add(MeasurePair(module, pair, rows, cols, batch, reps, sweeps, laneA, laneB));
                    }

                    if (results.Count == 0)
                    {
                        Console.WriteLine("cuda-concurrency: no pair could be measured on this host");
                        return 2;
                    }

                    Console.WriteLine();
                    Console.WriteLine("AGGREGATE across pairs; ratios are medians of rep-by-rep paired ratios (both arms timed inside the same rep); ratio >1 means the overlapped arm finished two workloads faster than today's serialized pipeline; twoStreamSerialized is the same lane machinery awaited between workloads, so its ratio against overlapped isolates concurrency from stream/staging overhead");
                    string[] headers = ["pair", "serialized_ms", "twostream_ms", "overlapped_ms", "ratio_ser_over_ov", "ratio_2stream_over_ov", "ratio_ser_over_2stream", "ideal_serial_sum/ov", "busySum/ovWall", "tok/s_equiv_ser", "tok/s_equiv_ov"];
                    Console.WriteLine(string.Join("  ", headers.Select(h => h.PadLeft(Math.Max(h.Length, 14)))));
                    foreach (PairResult r in results)
                    {
                        string[] cells =
                        [
                            r.Name,
                            $"{r.SerializedMedMs:F3}",
                            $"{r.TwoStreamMedMs:F3}",
                            $"{r.OverlappedMedMs:F3}",
                            $"{r.RatioSerializedOverOverlapped:F2}",
                            $"{r.RatioTwoStreamOverOverlapped:F2}",
                            $"{r.RatioSerializedOverTwoStream:F2}",
                            $"{r.IdealSerialSumOverOverlapped:F2}",
                            $"{r.BusyConcurrencyRatio:F2}",
                            $"{r.TokSSerialized:F0}",
                            $"{r.TokSOverlapped:F0}",
                        ];

                        Console.WriteLine(string.Join("  ", cells.Select(c => c.PadLeft(14))));
                    }

                    foreach (PairResult r in results)
                    {
                        Console.WriteLine(
                            $"verdict pair={r.Name} overlap_payoff={r.Verdict} bitExactEquivalence={(r.BitExactEquivalence ? "yes" : "no")} " +
                            $"legacy_pipeline_over_lane_machinery={r.RatioSerializedOverTwoStream:F2} " +
                            $"paired_ratio_spread={r.RatioSerializedOverOverlappedSpread:F2} arm_median_ratio_ser_over_ov={r.ArmMedianRatioSerializedOverOverlapped:F2}");
                    }

                    if (outPath is not null)
                    {
                        File.WriteAllText(outPath, BuildJson(deviceOrdinal, deviceName, driverVersion, pairs, rows, cols, batch, reps, sweeps, results));
                        Console.WriteLine($"raw measurement JSON written to {outPath}");
                    }

                    return 0;
                }
                finally
                {
                    ConcurrencyCudaNative.Check("cuStreamDestroy", ConcurrencyCudaNative.cuStreamDestroy(laneB));
                }
            }
            finally
            {
                ConcurrencyCudaNative.Check("cuStreamDestroy", ConcurrencyCudaNative.cuStreamDestroy(laneA));
            }
        }
    }

    private static PairResult MeasurePair(IntPtr module, Pair pair, int rows, int cols, int batch, int reps, int sweeps, IntPtr laneA, IntPtr laneB)
    {
        Console.WriteLine();
        Console.WriteLine(
            $"PAIR {pair.Name} kindA={KindName(pair.KindA)} formatA={pair.FormatA} " +
            $"kindB={KindName(pair.KindB)} formatB={pair.FormatB} rows={rows} cols={cols} batch={batch}");

        Workload a = new(module, pair.KindA, pair.FormatA, rows, cols, batch);
        try
        {
            Workload b = new(module, pair.KindB, pair.FormatB, rows, cols, batch);
            try
            {
                Console.WriteLine(
                    $"  workload A entry={a.EntryName} packedMiB={(a.PackedBytes / (1024.0 * 1024.0)):F2} blocks={a.Blocks} " +
                    $"xferInBytesPerCall={a.InputBytes} xferOutBytesPerCall={a.OutputBytes}");
                Console.WriteLine(
                    $"  workload B entry={b.EntryName} packedMiB={(b.PackedBytes / (1024.0 * 1024.0)):F2} blocks={b.Blocks} " +
                    $"xferInBytesPerCall={b.InputBytes} xferOutBytesPerCall={b.OutputBytes}");

                // Untimed warm-up per workload: the first launch of a function pays its setup, and neither arm should see it.
                a.LaunchSerialized();
                b.LaunchSerialized();

                // One untimed equivalence pass per workload: serialized-arm output versus overlapped-arm output on identical bytes, bit for bit. It
                // proves both arms measure the same work; no band is asserted or widened here.
                bool exactA = a.EquivalenceCheck(laneA);
                bool exactB = b.EquivalenceCheck(laneB);
                Console.WriteLine($"  equivalence serialized_vs_overlapped outputs: A={(exactA ? "bit_exact" : "differs")} B={(exactB ? "bit_exact" : "differs")}");

                List<double> allSingleAWallMs = [];
                List<double> allSingleBWallMs = [];
                List<double> allSerializedWallMs = [];
                List<double> allTwoStreamSerializedWallMs = [];
                List<double> allOverlappedWallMs = [];
                List<double> allBusySumUs = [];
                List<double> allRatioSerOverOv = [];
                List<double> allRatio2StreamOverOv = [];
                List<double> allRatioSerOver2Stream = [];
                List<double> sweepRatios = [];

                for (int s = 0; s < sweeps; s++)
                {
                    double[] singleAWallMs = new double[reps];
                    double[] singleBWallMs = new double[reps];
                    double[] serializedWallMs = new double[reps];
                    double[] twoStreamSerializedWallMs = new double[reps];
                    double[] overlappedWallMs = new double[reps];
                    double[] busySumUs = new double[reps];
                    double[] ratioSerOverOv = new double[reps];
                    double[] ratio2StreamOverOv = new double[reps];
                    double[] ratioSerOver2Stream = new double[reps];

                    // Arms alternate rep by rep so GPU clock boost and other-workload noise cannot land on only one arm.
                    for (int rep = 0; rep < reps; rep++)
                    {
                        Stopwatch swSingleA = Stopwatch.StartNew();
                        double singleABusyUs = a.LaunchSerialized() * 1000.0;
                        singleAWallMs[rep] = swSingleA.Elapsed.TotalMilliseconds;

                        Stopwatch swSingleB = Stopwatch.StartNew();
                        double singleBBusyUs = b.LaunchSerialized() * 1000.0;
                        singleBWallMs[rep] = swSingleB.Elapsed.TotalMilliseconds;

                        // Serialized arm: A fully completes (upload, launch on the legacy stream, context sync, download) before B starts. Its busy
                        // tally is read inside each call exactly as engine accounting folds it after that call's synchronize returned.
                        Stopwatch swSerial = Stopwatch.StartNew();
                        double busyA = a.LaunchSerialized();
                        double busyB = b.LaunchSerialized();
                        serializedWallMs[rep] = swSerial.Elapsed.TotalMilliseconds;
                        busySumUs[rep] = (busyA + busyB) * 1000.0;

                        // Two-stream serialized control: the same lane machinery and the same async staging as the overlapped arm, but each lane is
                        // awaited before the next workload is submitted. It separates what stream creation, async copies, and per-stream joins cost
                        // from what actual concurrency buys, so a payoff claim never rests on both at once.
                        Stopwatch swTwoStream = Stopwatch.StartNew();
                        a.EnqueueOnStream(laneA);
                        ConcurrencyCudaNative.Check("cuStreamSynchronize", ConcurrencyCudaNative.cuStreamSynchronize(laneA));
                        double busyATwoStreamUs = a.ReadBusyMicroseconds();
                        b.EnqueueOnStream(laneB);
                        ConcurrencyCudaNative.Check("cuStreamSynchronize", ConcurrencyCudaNative.cuStreamSynchronize(laneB));
                        double busyBTwoStreamUs = b.ReadBusyMicroseconds();
                        twoStreamSerializedWallMs[rep] = swTwoStream.Elapsed.TotalMilliseconds;

                        // Overlapped arm: both pipelines are submitted before either lane is awaited, then both lanes are joined inside the timed span.
                        Stopwatch swOverlap = Stopwatch.StartNew();
                        a.EnqueueOnStream(laneA);
                        b.EnqueueOnStream(laneB);
                        ConcurrencyCudaNative.Check("cuStreamSynchronize", ConcurrencyCudaNative.cuStreamSynchronize(laneA));
                        double busyAOverlappedUs = a.ReadBusyMicroseconds();
                        ConcurrencyCudaNative.Check("cuStreamSynchronize", ConcurrencyCudaNative.cuStreamSynchronize(laneB));
                        double busyBOverlappedUs = b.ReadBusyMicroseconds();
                        overlappedWallMs[rep] = swOverlap.Elapsed.TotalMilliseconds;

                        // Pa ratios are computed rep by rep, from arms timed back to back inside the same rep, so a stretch of other workloads on the
                        // device slows both sides of a ratio together instead of splitting them across sweeps. This is the number the verdict rests on;
                        // the arm medians alone are too easy for contention to move.
                        ratioSerOverOv[rep] = serializedWallMs[rep] / overlappedWallMs[rep];
                        ratio2StreamOverOv[rep] = twoStreamSerializedWallMs[rep] / overlappedWallMs[rep];
                        ratioSerOver2Stream[rep] = serializedWallMs[rep] / twoStreamSerializedWallMs[rep];

                        Console.WriteLine(
                            $"  sweep {s} rep {rep}: singleA={singleAWallMs[rep]:F3} ms (busy {singleABusyUs:F1} us) singleB={singleBWallMs[rep]:F3} ms (busy {singleBBusyUs:F1} us) " +
                            $"serialized={serializedWallMs[rep]:F3} ms busySum={busySumUs[rep]:F1} us twoStreamSerialized={twoStreamSerializedWallMs[rep]:F3} ms " +
                            $"(busyA={busyATwoStreamUs:F1} us busyB={busyBTwoStreamUs:F1} us) overlapped={overlappedWallMs[rep]:F3} ms busyA={busyAOverlappedUs:F1} us busyB={busyBOverlappedUs:F1} us " +
                            $"ratios ser/ov={ratioSerOverOv[rep]:F2} 2stream/ov={ratio2StreamOverOv[rep]:F2} ser/2stream={ratioSerOver2Stream[rep]:F2}");

                        // Touch each staged output so no measured pass can be dead work and a bad kernel surfaces immediately.
                        a.TouchOutput();
                        b.TouchOutput();
                    }

                    Console.WriteLine(
                        $"  sweep {s} medians: singleA={Median(singleAWallMs):F3} ms singleB={Median(singleBWallMs):F3} ms " +
                        $"serialized(min/med/max)={Min(serializedWallMs):F3}/{Median(serializedWallMs):F3}/{Max(serializedWallMs):F3} ms busySumUs(med)={Median(busySumUs):F1} | " +
                        $"twoStreamSerialized(min/med/max)={Min(twoStreamSerializedWallMs):F3}/{Median(twoStreamSerializedWallMs):F3}/{Max(twoStreamSerializedWallMs):F3} ms | " +
                        $"overlapped(min/med/max)={Min(overlappedWallMs):F3}/{Median(overlappedWallMs):F3}/{Max(overlappedWallMs):F3} ms");

                    double medSerialized = Median(serializedWallMs);
                    double medTwoStream = Median(twoStreamSerializedWallMs);
                    double medOverlapped = Median(overlappedWallMs);
                    Console.WriteLine(
                        $"  sweep {s} paired ratios (median over the {reps} rep-by-rep ratios; both arms timed inside the same rep) " +
                        $"serialized/overlapped={Median(ratioSerOverOv):F2} (min/med/max={Min(ratioSerOverOv):F2}/{Median(ratioSerOverOv):F2}/{Max(ratioSerOverOv):F2}) " +
                        $"twoStreamSerialized/overlapped={Median(ratio2StreamOverOv):F2} (concurrency alone) " +
                        $"serialized/twoStreamSerialized={Median(ratioSerOver2Stream):F2} (control: same lanes and async staging, awaited between workloads instead of after both)");
                    Console.WriteLine(
                        $"  sweep {s} arm-median ratios for reference serialized/overlapped={medSerialized / medOverlapped:F2} " +
                        $"twoStreamSerialized/overlapped={medTwoStream / medOverlapped:F2} serialized/twoStreamSerialized={medSerialized / medTwoStream:F2} " +
                        $"ideal_serial_sum(singleA+singleB)/overlapped={(Median(singleAWallMs) + Median(singleBWallMs)) / medOverlapped:F2} " +
                        $"busySum/overlappedWall={Median(busySumUs) / (medOverlapped * 1000.0):F2} (>1 means the device did more work than the overlapped wall had time for, i.e. the lanes ran together)");

                    Console.WriteLine(
                        $"  sweep {s} tok/s-equivalent (weight-pass vectors: 2 x batch / arm wall; synthetic equivalence, not served tokens) " +
                        $"serialized={1000.0 * 2 * batch / medSerialized:F0} twoStreamSerialized={1000.0 * 2 * batch / medTwoStream:F0} overlapped={1000.0 * 2 * batch / medOverlapped:F0}");

                    allSingleAWallMs.AddRange(singleAWallMs);
                    allSingleBWallMs.AddRange(singleBWallMs);
                    allSerializedWallMs.AddRange(serializedWallMs);
                    allTwoStreamSerializedWallMs.AddRange(twoStreamSerializedWallMs);
                    allOverlappedWallMs.AddRange(overlappedWallMs);
                    allBusySumUs.AddRange(busySumUs);
                    allRatioSerOverOv.AddRange(ratioSerOverOv);
                    allRatio2StreamOverOv.AddRange(ratio2StreamOverOv);
                    allRatioSerOver2Stream.AddRange(ratioSerOver2Stream);
                    sweepRatios.Add(Median(ratioSerOverOv));
                }

                double serializedMed = Median(allSerializedWallMs);
                double twoStreamMed = Median(allTwoStreamSerializedWallMs);
                double overlappedMed = Median(allOverlappedWallMs);

                Console.WriteLine(
                    $"  aggregate over {sweeps * reps} timed reps per arm: serialized(min/med/max)={Min(allSerializedWallMs):F3}/{serializedMed:F3}/{Max(allSerializedWallMs):F3} ms " +
                    $"twoStreamSerialized(min/med/max)={Min(allTwoStreamSerializedWallMs):F3}/{twoStreamMed:F3}/{Max(allTwoStreamSerializedWallMs):F3} ms " +
                    $"overlapped(min/med/max)={Min(allOverlappedWallMs):F3}/{overlappedMed:F3}/{Max(allOverlappedWallMs):F3} ms");
                Console.WriteLine(
                    $"  aggregate paired ratios (med/min/max) serialized/overlapped={Median(allRatioSerOverOv):F2}/{Min(allRatioSerOverOv):F2}/{Max(allRatioSerOverOv):F2} " +
                    $"twoStreamSerialized/overlapped={Median(allRatio2StreamOverOv):F2}/{Min(allRatio2StreamOverOv):F2}/{Max(allRatio2StreamOverOv):F2} " +
                    $"serialized/twoStreamSerialized={Median(allRatioSerOver2Stream):F2}/{Min(allRatioSerOver2Stream):F2}/{Max(allRatioSerOver2Stream):F2}");

                return new PairResult(
                    pair.Name, serializedMed, Min(allSerializedWallMs), Max(allSerializedWallMs),
                    twoStreamMed, Min(allTwoStreamSerializedWallMs), Max(allTwoStreamSerializedWallMs),
                    overlappedMed, Min(allOverlappedWallMs), Max(allOverlappedWallMs),
                    Median(allSingleAWallMs), Median(allSingleBWallMs), Median(allBusySumUs),
                    Median(allRatioSerOverOv), Min(allRatioSerOverOv), Max(allRatioSerOverOv),
                    Median(allRatio2StreamOverOv), Min(allRatio2StreamOverOv), Max(allRatio2StreamOverOv),
                    Median(allRatioSerOver2Stream), Min(allRatioSerOver2Stream), Max(allRatioSerOver2Stream),
                    sweepRatios, 2 * batch, exactA && exactB);
            }
            finally
            {
                b.Dispose();
            }
        }
        finally
        {
            a.Dispose();
        }
    }

    private static string KindName(KernelKind kind) => kind == KernelKind.GemV ? "gemv" : "gemm";

    // Payoff classification thresholds on the serialized/overlapped wall ratio, taken per sweep as the median of that sweep's rep-by-rep paired ratios:
    // every sweep must clear it for a claim, so one lucky run cannot mark overlap as a win. A ratio at or under 0.98 everywhere means splitting into two
    // streams measured slower than today's serialization; anything between is reported inconclusive rather than rounded toward whichever side it leans on.
    private static string ClassifyPayoff(IReadOnlyList<double> sweepRatios) =>
        sweepRatios.All(r => r >= 1.05) ? "positive" :
        sweepRatios.All(r => r <= 0.98) ? "negative" : "inconclusive";

    private static double Median(IReadOnlyList<double> values)
    {
        double[] sorted = [.. values];
        Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    private static double Min(IReadOnlyList<double> values) => values.Min();

    private static double Max(IReadOnlyList<double> values) => values.Max();

    private static unsafe string ReadAsciiNulTerminated(byte* buffer)
    {
        int length = 0;
        while (length < 128 && buffer[length] != 0)
        {
            length++;
        }

        return Encoding.ASCII.GetString(new ReadOnlySpan<byte>(buffer, length));
    }

    private static string BuildJson(
        int deviceOrdinal, string deviceName, int driverVersion, IReadOnlyList<Pair> pairs, int rows, int cols, int batch, int reps, int sweeps, IReadOnlyList<PairResult> results)
    {
        StringBuilder sb = new();
        sb.Append('{');
        sb.Append("\"command\":\"cuda-concurrency\",");
        sb.Append($"\"args\":{{\"rows\":{rows},\"cols\":{cols},\"batch\":{batch},\"repsPerArm\":{reps},\"sweeps\":{sweeps}}},").Append('\n');
        sb.Append($"\"host\":{{\"osDescription\":{JsonString(RuntimeInformation.OSDescription)},\"logicalProcessors\":{Environment.ProcessorCount}}},").Append('\n');
        sb.Append($"\"device\":{{\"ordinal\":{deviceOrdinal},\"name\":{JsonString(deviceName)},\"driverApiVersion\":{driverVersion}}}").Append(',');
        sb.Append(
            "\"basis\":{\"serializedArm\":\"sync cuMemcpyHtoD -> cuLaunchKernel(stream=legacy NULL) -> cuCtxSynchronize -> sync cuMemcpyDtoH, workload A then B\", " +
            "\"twoStreamSerializedArm\":\"async H2D + launch + async D2H on CU_STREAM_NON_BLOCKING lanes with each lane awaited before the next workload is submitted (control for stream and staging overhead)\", " +
            "\"overlappedArm\":\"cuMemcpyHtoDAsync + kernel launch + async D2H per workload on CU_STREAM_NON_BLOCKING streams, both submitted before either is awaited\", " +
            $"\"kernelThreads\":{KernelThreads}, \"residentDeviceWeightsPerWorkload\":true, \"deviceActivationAndOutputBuffersPersistentAcrossReps\":true}},");

        sb.Append("\"pairs\":[");
        for (int i = 0; i < results.Count; i++)
        {
            if (i > 0)
            {
                sb.Append(',');
            }

            PairResult r = results[i];
            sb.Append('{');
            sb.Append($"\"name\":{JsonString(r.Name)},");
            sb.Append($"\"serializedMs\":{{\"min\":{Num(r.SerializedMinMs)},\"med\":{Num(r.SerializedMedMs)},\"max\":{Num(r.SerializedMaxMs)}}},");
            sb.Append($"\"twoStreamSerializedMs\":{{\"min\":{Num(r.TwoStreamMinMs)},\"med\":{Num(r.TwoStreamMedMs)},\"max\":{Num(r.TwoStreamMaxMs)}}},");
            sb.Append($"\"overlappedMs\":{{\"min\":{Num(r.OverlappedMinMs)},\"med\":{Num(r.OverlappedMedMs)},\"max\":{Num(r.OverlappedMaxMs)}}},");
            sb.Append($"\"singleArmMedianMs\":{{\"A\":{Num(r.SingleAMedMs)},\"B\":{Num(r.SingleBMedMs)}}},");
            sb.Append($"\"serializedKernelBusySumUsMedian\":{Num(r.BusySumMedianUs)},");
            sb.Append($"\"ratioSerializedOverOverlapped\":{Num(r.RatioSerializedOverOverlapped)},");
            sb.Append($"\"ratioSerializedOverOverlappedRepSpread\":{Num(r.RatioSerializedOverOverlappedSpread)},");
            sb.Append($"\"ratioTwoStreamOverOverlapped\":{Num(r.RatioTwoStreamOverOverlapped)},");
            sb.Append($"\"ratioTwoStreamOverOverlappedRepSpread\":{Num(r.RatioTwoStreamOverOverlappedSpread)},");
            sb.Append($"\"ratioSerializedOverTwoStream\":{Num(r.RatioSerializedOverTwoStream)},");
            sb.Append($"\"ratioSerializedOverTwoStreamRepSpread\":{Num(r.RatioSerializedOverTwoStreamSpread)},");
            sb.Append($"\"armMedianRatioSerializedOverOverlapped\":{Num(r.ArmMedianRatioSerializedOverOverlapped)},");
            sb.Append($"\"armMedianRatioTwoStreamOverOverlapped\":{Num(r.ArmMedianRatioTwoStreamOverOverlapped)},");
            sb.Append($"\"idealSerialSumOverOverlapped\":{Num(r.IdealSerialSumOverOverlapped)},");
            sb.Append($"\"busyConcurrencyRatio\":{Num(r.BusyConcurrencyRatio)},");
            sb.Append($"\"tokSEquivalent\":{{\"serialized\":{Num(r.TokSSerialized)},\"overlapped\":{Num(r.TokSOverlapped)}}},").Append('\n');
            sb.Append("\"sweepRatios\":[");
            for (int s = 0; s < r.SweepRatios.Count; s++)
            {
                if (s > 0)
                {
                    sb.Append(',');
                }

                sb.Append(Num(r.SweepRatios[s]));
            }

            sb.Append($"],\"bitExactEquivalence\":{(r.BitExactEquivalence ? "true" : "false")},");
            sb.Append($"\"verdict\":{JsonString(r.Verdict)}");
            sb.Append('}');
        }

        sb.Append("]}");
        return sb.ToString();

        static string Num(double value) => value.ToString("G6", CultureInfo.InvariantCulture);
    }

    private static string JsonString(string value)
    {
        StringBuilder sb = new(value.Length + 2);
        sb.Append('"');
        foreach (char c in value)
        {
            switch (c)
            {
                case '"':
                    sb.Append("\\\"");
                    break;
                case '\\':
                    sb.Append("\\\\");
                    break;
                default:
                    if (c < 0x20)
                    {
                        sb.Append($"\\u{(int)c:X4}");
                    }
                    else
                    {
                        sb.Append(c);
                    }

                    break;
            }
        }

        sb.Append('"');
        return sb.ToString();
    }

    private static bool TryParsePositive(string value, out int parsed) =>
        int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out parsed) && parsed >= 1;

    private static string CollapseWhitespace(string text) => string.Join(' ', text.Split(" \t\r\n", StringSplitOptions.RemoveEmptyEntries));

    internal sealed record Pair(string Name, KernelKind KindA, QuantFormat FormatA, KernelKind KindB, QuantFormat FormatB);

    internal enum KernelKind
    {
        GemV,
        Gemm,
    }

    /// <summary>One pair's aggregate: arm medians/min/max over every timed rep of every sweep, the single-arm baselines, the serialized busy sum, the rep-by-rep paired ratios (both arms timed inside the same rep), per-sweep payoff ratios, and the verdict those ratios support. The two-stream control reports the same lane machinery awaited between workloads, so the concurrency-only gain is separated from the cost of streams and async staging.</summary>
    private sealed class PairResult(
        string name, double serializedMedMs, double serializedMinMs, double serializedMaxMs, double twoStreamMedMs, double twoStreamMinMs, double twoStreamMaxMs,
        double overlappedMedMs, double overlappedMinMs, double overlappedMaxMs,
        double singleAMedMs, double singleBMedMs, double busySumMedianUs,
        double ratioSerOverOvMed, double ratioSerOverOvMin, double ratioSerOverOvMax,
        double ratio2StreamOverOvMed, double ratio2StreamOverOvMin, double ratio2StreamOverOvMax,
        double ratioSerOver2StreamMed, double ratioSerOver2StreamMin, double ratioSerOver2StreamMax,
        IReadOnlyList<double> sweepRatios, int vectorsPerPairPass, bool bitExactEquivalence)
    {
        public string Name { get; } = name;
        public double SerializedMedMs { get; } = serializedMedMs;
        public double SerializedMinMs { get; } = serializedMinMs;
        public double SerializedMaxMs { get; } = serializedMaxMs;
        public double TwoStreamMedMs { get; } = twoStreamMedMs;
        public double TwoStreamMinMs { get; } = twoStreamMinMs;
        public double TwoStreamMaxMs { get; } = twoStreamMaxMs;
        public double OverlappedMedMs { get; } = overlappedMedMs;
        public double OverlappedMinMs { get; } = overlappedMinMs;
        public double OverlappedMaxMs { get; } = overlappedMaxMs;
        public double SingleAMedMs { get; } = singleAMedMs;
        public double SingleBMedMs { get; } = singleBMedMs;
        public double BusySumMedianUs { get; } = busySumMedianUs;
        public IReadOnlyList<double> SweepRatios { get; } = sweepRatios;
        public bool BitExactEquivalence { get; } = bitExactEquivalence;

        // Primary ratios are the rep-by-rep paired medians: contention-proof because both arms of each ratio were timed back to back in the same rep.
        public double RatioSerializedOverOverlapped { get; } = ratioSerOverOvMed;
        public double RatioSerializedOverOverlappedSpread { get; } = ratioSerOverOvMax - ratioSerOverOvMin;
        public double RatioTwoStreamOverOverlapped { get; } = ratio2StreamOverOvMed; // the payoff concurrency alone buys, with stream and async-staging overhead already inside both arms
        public double RatioTwoStreamOverOverlappedSpread { get; } = ratio2StreamOverOvMax - ratio2StreamOverOvMin;
        public double RatioSerializedOverTwoStream { get; } = ratioSerOver2StreamMed; // the overhead the lane machinery itself costs against today's pipeline
        public double RatioSerializedOverTwoStreamSpread { get; } = ratioSerOver2StreamMax - ratioSerOver2StreamMin;

        // Arm-median ratios kept for transparency: they name the printed medians directly, and a gap between them and the paired ratios is contention, not signal.
        public double ArmMedianRatioSerializedOverOverlapped { get; } = serializedMedMs / overlappedMedMs;
        public double ArmMedianRatioTwoStreamOverOverlapped { get; } = twoStreamMedMs / overlappedMedMs;
        public double IdealSerialSumOverOverlapped { get; } = (singleAMedMs + singleBMedMs) / overlappedMedMs;
        public double BusyConcurrencyRatio { get; } = busySumMedianUs / (overlappedMedMs * 1000.0); // >1 means device-busy work exceeded the overlapped wall's time: the lanes ran together
        public double TokSSerialized { get; } = 1000.0 * vectorsPerPairPass / serializedMedMs;
        public double TokSOverlapped { get; } = 1000.0 * vectorsPerPairPass / overlappedMedMs;

        public string Verdict { get; } = ClassifyPayoff(sweepRatios);
    }

    /// <summary>One measured workload: its own resident device weight copy, page-locked host activation staging and output staging, persistent device activation/output buffers, one kernel function handle resolved from the seam's loaded module, and one reusable event pair for device-measured busy time. Every launch replicates the seam's parameter order exactly.</summary>
    private sealed class Workload : IDisposable
    {
        public string EntryName { get; }
        public long PackedBytes => _packedBytes;
        public int Blocks => (_rows + KernelThreads - 1) / KernelThreads;
        public long InputBytes => (long)_batch * _cols * sizeof(float);
        public long OutputBytes => (long)_batch * _rows * sizeof(float);

        /// <summary>Device-measured microseconds of this workload's most recent timed launch, or 0 when event timing is unavailable.</summary>
        public double LastBusyMicroseconds { get; private set; }

        private readonly IntPtr _function;
        private readonly int _rows;
        private readonly int _cols;
        private readonly int _batch;
        private readonly long _packedBytes;
        private ulong _deviceWeights;
        private ulong _deviceX;
        private ulong _deviceOut;
        private IntPtr _hostX; // page-locked via cuMemAllocHost_v2: async copies from it are genuinely enqueued, not staged synchronously
        private IntPtr _hostOut;
        private IntPtr _eventStart;
        private IntPtr _eventEnd;

        public Workload(IntPtr module, KernelKind kind, QuantFormat format, int rows, int cols, int batch)
        {
            _rows = rows;
            _cols = cols;
            _batch = batch;

            EntryName = PtxEntry(kind, format);
            ConcurrencyCudaNative.Check("cuModuleGetFunction", ConcurrencyCudaNative.cuModuleGetFunction(out _function, module, EntryName));

            // Deterministic pinned fixture and activation family shared with cuda-gemv/cuda-gemm: identical bytes every run of this build.
            byte[] weights = CudaGemvBench.BuildFixture(format, rows, cols);
            _packedBytes = weights.Length;

            ConcurrencyCudaNative.Check("cuMemAlloc", ConcurrencyCudaNative.cuMemAlloc(out _deviceWeights, (nuint)_packedBytes));
            unsafe
            {
                fixed (byte* pWeights = weights)
                {
                    ConcurrencyCudaNative.Check("cuMemcpyHtoD", ConcurrencyCudaNative.cuMemcpyHtoD(_deviceWeights, pWeights, (nuint)_packedBytes)); // untimed residency upload: the seam caches each tensor's device copy once too
                }

                ConcurrencyCudaNative.Check("cuMemAllocHost_v2", ConcurrencyCudaNative.cuMemAllocHost(out _hostX, (nuint)InputBytes));
                ConcurrencyCudaNative.Check("cuMemAllocHost_v2", ConcurrencyCudaNative.cuMemAllocHost(out _hostOut, (nuint)OutputBytes));

                // The same per-vector activation family both CUDA benches use: Sin(0.11*c + 7*v)*1.3.
                float* x = (float*)_hostX;
                for (int v = 0; v < batch; v++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        x[v * cols + c] = CudaGemvBench.Activation(c + v * 7);
                    }
                }
            }

            ConcurrencyCudaNative.Check("cuMemAlloc", ConcurrencyCudaNative.cuMemAlloc(out _deviceX, (nuint)InputBytes));
            ConcurrencyCudaNative.Check("cuMemAlloc", ConcurrencyCudaNative.cuMemAlloc(out _deviceOut, (nuint)OutputBytes));

            // One reusable timing pair for this workload's whole life, like the seam's accounting events: a driver that refuses to create them
            // leaves busy reporting at zero instead of refusing the measurement behind it.
            if (ConcurrencyCudaNative.cuEventCreate(out _eventStart, 0) != ConcurrencyCudaNative.Success ||
                ConcurrencyCudaNative.cuEventCreate(out _eventEnd, 0) != ConcurrencyCudaNative.Success)
            {
                _eventStart = IntPtr.Zero;
                _eventEnd = IntPtr.Zero;
            }
        }

        /// <summary>The seam-faithful serialized pipeline for exactly this workload: synchronous upload on the legacy stream, launch with a timed stamp pair, context synchronize, then synchronous download. Records device-measured kernel busy in LastBusyMicroseconds and returns it in milliseconds.</summary>
        public double LaunchSerialized()
        {
            unsafe
            {
                ConcurrencyCudaNative.Check("cuMemcpyHtoD", ConcurrencyCudaNative.cuMemcpyHtoD(_deviceX, (byte*)_hostX, (nuint)InputBytes));

                bool timed = Timed;
                if (timed)
                {
                    // Lands in stream order after the upload already crossed: measures kernel work alone, as engine accounting does.
                    ConcurrencyCudaNative.Check("cuEventRecord", ConcurrencyCudaNative.cuEventRecord(_eventStart, IntPtr.Zero));
                }

                LaunchOnStream(IntPtr.Zero);

                if (timed)
                {
                    ConcurrencyCudaNative.Check("cuEventRecord", ConcurrencyCudaNative.cuEventRecord(_eventEnd, IntPtr.Zero));
                }

                ConcurrencyCudaNative.Check("cuCtxSynchronize", ConcurrencyCudaNative.cuCtxSynchronize());

                double busyMs = timed ? ReadElapsedMilliseconds() : 0.0;
                LastBusyMicroseconds = busyMs * 1000.0;

                ConcurrencyCudaNative.Check("cuMemcpyDtoH", ConcurrencyCudaNative.cuMemcpyDtoH((byte*)_hostOut, _deviceOut, (nuint)OutputBytes));
                return busyMs;
            }
        }

        /// <summary>Submit this workload's whole pipeline to one lane without awaiting it: async upload on that stream, a timed stamp pair after the copy lands in queue order, launch on that stream, then async download. The caller joins the lanes.</summary>
        public void EnqueueOnStream(IntPtr stream)
        {
            unsafe
            {
                ConcurrencyCudaNative.Check("cuMemcpyHtoDAsync", ConcurrencyCudaNative.cuMemcpyHtoDAsync(_deviceX, (byte*)_hostX, (nuint)InputBytes, stream));

                if (Timed)
                {
                    ConcurrencyCudaNative.Check("cuEventRecord", ConcurrencyCudaNative.cuEventRecord(_eventStart, stream));
                }

                LaunchOnStream(stream);

                if (Timed)
                {
                    ConcurrencyCudaNative.Check("cuEventRecord", ConcurrencyCudaNative.cuEventRecord(_eventEnd, stream));
                }

                ConcurrencyCudaNative.Check("cuMemcpyDtoHAsync", ConcurrencyCudaNative.cuMemcpyDtoHAsync((byte*)_hostOut, _deviceOut, (nuint)OutputBytes, stream));
            }
        }

        /// <summary>Device-measured microseconds for the most recent timed launch of this workload. Call only after that lane's synchronize returned.</summary>
        public double ReadBusyMicroseconds()
        {
            if (!Timed)
            {
                return 0.0;
            }

            unsafe
            {
                LastBusyMicroseconds = ReadElapsedMilliseconds() * 1000.0;
            }

            return LastBusyMicroseconds;
        }

        private bool Timed => _eventStart != IntPtr.Zero && _eventEnd != IntPtr.Zero;

        /// <summary>Read the staged output's first element and report a NaN, so no measured pass can be dead work and a bad kernel surfaces immediately.</summary>
        public void TouchOutput()
        {
            unsafe
            {
                float* outp = (float*)_hostOut;
                if (outp != null && _batch * _rows > 0 && float.IsNaN(outp[0]))
                {
                    Console.WriteLine($"  NaN seen in output of workload entry={EntryName}");
                }
            }
        }

        /// <summary>One untimed serialized pass and one overlapped pass on the given lane, compared bit for bit: identical kernels must produce byte-identical outputs whichever arm enqueues them.</summary>
        public bool EquivalenceCheck(IntPtr stream)
        {
            unsafe
            {
                LaunchSerialized();

                int count = _batch * _rows;
                float* staged = (float*)_hostOut;
                float[] serializedCopy = new float[count];
                for (int i = 0; i < count; i++)
                {
                    serializedCopy[i] = staged[i];
                }

                EnqueueOnStream(stream);
                ConcurrencyCudaNative.Check("cuStreamSynchronize", ConcurrencyCudaNative.cuStreamSynchronize(stream));

                for (int i = 0; i < count; i++)
                {
                    if (BitConverter.SingleToInt32Bits(serializedCopy[i]) != BitConverter.SingleToInt32Bits(staged[i]))
                    {
                        return false;
                    }
                }

                return true;
            }
        }

        private void LaunchOnStream(IntPtr stream)
        {
            unsafe
            {
                // Exactly the seam's parameter order and grid geometry: weight cursor, activation cursor, output cursor, rows, cols, batch count.
                ulong deviceWeights = _deviceWeights;
                ulong deviceX = _deviceX;
                ulong deviceOut = _deviceOut;
                uint rowsU = (uint)_rows;
                uint colsU = (uint)_cols;
                uint countU = (uint)_batch;

                void** kernelParams = stackalloc void*[6];
                kernelParams[0] = &deviceWeights;
                kernelParams[1] = &deviceX;
                kernelParams[2] = &deviceOut;
                kernelParams[3] = &rowsU;
                kernelParams[4] = &colsU;
                kernelParams[5] = &countU;

                uint blocks = (uint)Blocks;
                ConcurrencyCudaNative.Check(
                    "cuLaunchKernel",
                    ConcurrencyCudaNative.cuLaunchKernel(_function, blocks, 1, 1, KernelThreads, 1, 1, 0, stream, kernelParams, null));
            }
        }

        private double ReadElapsedMilliseconds() => ReadElapsedInternal();

        private double ReadElapsedInternal()
        {
            ConcurrencyCudaNative.Check("cuEventElapsedTime", ConcurrencyCudaNative.cuEventElapsedTime(out float milliseconds, _eventStart, _eventEnd));
            return Math.Max(0f, milliseconds);
        }

        public void Dispose()
        {
            if (_deviceOut != 0)
            {
                ConcurrencyCudaNative.Check("cuMemFree", ConcurrencyCudaNative.cuMemFree(_deviceOut));
                _deviceOut = 0;
            }

            if (_deviceX != 0)
            {
                ConcurrencyCudaNative.Check("cuMemFree", ConcurrencyCudaNative.cuMemFree(_deviceX));
                _deviceX = 0;
            }

            if (_deviceWeights != 0)
            {
                ConcurrencyCudaNative.Check("cuMemFree", ConcurrencyCudaNative.cuMemFree(_deviceWeights));
                _deviceWeights = 0;
            }

            if (_hostOut != IntPtr.Zero)
            {
                unsafe
                {
                    ConcurrencyCudaNative.Check("cuMemFreeHost", ConcurrencyCudaNative.cuMemFreeHost((byte*)_hostOut));
                }

                _hostOut = IntPtr.Zero;
            }

            if (_hostX != IntPtr.Zero)
            {
                unsafe
                {
                    ConcurrencyCudaNative.Check("cuMemFreeHost", ConcurrencyCudaNative.cuMemFreeHost((byte*)_hostX));
                }

                _hostX = IntPtr.Zero;
            }

            if (_eventStart != IntPtr.Zero)
            {
                ConcurrencyCudaNative.cuEventDestroy(_eventStart);
                _eventStart = IntPtr.Zero;
            }

            if (_eventEnd != IntPtr.Zero)
            {
                ConcurrencyCudaNative.cuEventDestroy(_eventEnd);
                _eventEnd = IntPtr.Zero;
            }
        }

        private static string PtxEntry(KernelKind kind, QuantFormat format) => (kind, format) switch
        {
            // The same entry names CudaBackend's constructor resolves its own function handles from. Any name this build lacks fails at
            // cuModuleGetFunction with a clear driver error rather than measuring some other kernel.
            (KernelKind.GemV, QuantFormat.F32) => "gemv_f32",
            (KernelKind.GemV, QuantFormat.F16) => "gemv_f16",
            (KernelKind.GemV, QuantFormat.BF16) => "gemv_bf16",
            (KernelKind.GemV, QuantFormat.Q4_K) => "gemv_q4k",
            (KernelKind.GemV, QuantFormat.Q6_K) => "gemv_q6k",
            (KernelKind.GemV, QuantFormat.Q8_0) => "gemv_q8_0",
            (KernelKind.GemV, QuantFormat.Q5_0) => "gemv_q5_0",
            (KernelKind.GemV, QuantFormat.Q2_0) => "gemv_q2_0",
            (KernelKind.GemV, QuantFormat.Q5_K) => "gemv_q5k",
            (KernelKind.GemV, QuantFormat.Q2_K) => "gemv_q2k",
            (KernelKind.GemV, QuantFormat.Q3_K) => "gemv_q3k",
            (KernelKind.Gemm, QuantFormat.F32) => "gemm_f32",
            (KernelKind.Gemm, QuantFormat.F16) => "gemm_f16",
            (KernelKind.Gemm, QuantFormat.BF16) => "gemm_bf16",
            (KernelKind.Gemm, QuantFormat.Q4_K) => "gemm_q4k",
            (KernelKind.Gemm, QuantFormat.Q6_K) => "gemm_q6k",
            (KernelKind.Gemm, QuantFormat.Q8_0) => "gemm_q8_0",
            (KernelKind.Gemm, QuantFormat.Q5_0) => "gemm_q5_0",
            (KernelKind.Gemm, QuantFormat.Q2_0) => "gemm_q2_0",
            (KernelKind.Gemm, QuantFormat.Q5_K) => "gemm_q5k",
            (KernelKind.Gemm, QuantFormat.Q2_K) => "gemm_q2k",
            (KernelKind.Gemm, QuantFormat.Q3_K) => "gemm_q3k",
            _ => throw new NotSupportedException($"cuda-concurrency has no kernel mapping for {kind} on format {format}: the six-parameter GEMV/GEMM family only (IQ4 scratch-table and Q8_K host-quantized kernels need staging this bench does not replicate)"),
        };
    }
}

/// <summary>Driver entry points this measurement needs that the seam's own internal P/Invoke class does not declare - streams, async copies, page-locked host allocation, events, and device facts. Declared bench-side so no production file changes and the measurement seam stays bench-only.</summary>
internal static partial class ConcurrencyCudaNative
{
    public const int Success = 0;

    [LibraryImport("nvcuda", EntryPoint = "cuInit")]
    public static partial int cuInit(uint flags);

    [LibraryImport("nvcuda", EntryPoint = "cuDeviceGetCount")]
    public static partial int cuDeviceGetCount(out int count);

    [LibraryImport("nvcuda", EntryPoint = "cuDeviceGet")]
    public static partial int cuDeviceGet(out int device, int ordinal);

    [LibraryImport("nvcuda", EntryPoint = "cuDeviceGetName")]
    public static unsafe partial int cuDeviceGetName(byte* name, int length, int device);

    [LibraryImport("nvcuda", EntryPoint = "cuDriverGetVersion")]
    public static partial int cuDriverGetVersion(out int version);

    [LibraryImport("nvcuda", EntryPoint = "cuDevicePrimaryCtxRetain")]
    public static partial int cuDevicePrimaryCtxRetain(out IntPtr context, int device);

    [LibraryImport("nvcuda", EntryPoint = "cuDevicePrimaryCtxRelease")]
    public static partial int cuDevicePrimaryCtxRelease(int device);

    [LibraryImport("nvcuda", EntryPoint = "cuCtxSetCurrent")]
    public static partial int cuCtxSetCurrent(IntPtr context);

    [LibraryImport("nvcuda", EntryPoint = "cuModuleGetFunction")]
    public static partial int cuModuleGetFunction(out IntPtr function, IntPtr module, [MarshalAs(UnmanagedType.LPStr)] string name);

    [LibraryImport("nvcuda", EntryPoint = "cuMemAlloc_v2")]
    public static partial int cuMemAlloc(out ulong devicePointer, nuint bytes);

    [LibraryImport("nvcuda", EntryPoint = "cuMemFree_v2")]
    public static partial int cuMemFree(ulong devicePointer);

    [LibraryImport("nvcuda", EntryPoint = "cuMemcpyHtoD_v2")]
    public static unsafe partial int cuMemcpyHtoD(ulong destination, byte* source, nuint bytes);

    [LibraryImport("nvcuda", EntryPoint = "cuMemcpyDtoH_v2")]
    public static unsafe partial int cuMemcpyDtoH(byte* destination, ulong source, nuint bytes);

    [LibraryImport("nvcuda", EntryPoint = "cuMemcpyHtoDAsync_v2")]
    public static unsafe partial int cuMemcpyHtoDAsync(ulong destination, byte* source, nuint bytes, IntPtr stream);

    [LibraryImport("nvcuda", EntryPoint = "cuMemcpyDtoHAsync_v2")]
    public static unsafe partial int cuMemcpyDtoHAsync(byte* destination, ulong source, nuint bytes, IntPtr stream);

    [LibraryImport("nvcuda", EntryPoint = "cuMemAllocHost_v2")]
    public static unsafe partial int cuMemAllocHost(out IntPtr hostPointer, nuint bytes);

    [LibraryImport("nvcuda", EntryPoint = "cuMemFreeHost")]
    public static unsafe partial int cuMemFreeHost(byte* hostPointer);

    [LibraryImport("nvcuda", EntryPoint = "cuStreamCreate")]
    public static partial int cuStreamCreate(out IntPtr stream, uint flags);

    [LibraryImport("nvcuda", EntryPoint = "cuStreamSynchronize")]
    public static partial int cuStreamSynchronize(IntPtr stream);

    [LibraryImport("nvcuda", EntryPoint = "cuStreamDestroy")]
    public static partial int cuStreamDestroy(IntPtr stream);

    [LibraryImport("nvcuda", EntryPoint = "cuLaunchKernel")]
    public static unsafe partial int cuLaunchKernel(
        IntPtr function, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, uint sharedBytes, IntPtr stream, void** kernelParams, void** extra);

    [LibraryImport("nvcuda", EntryPoint = "cuCtxSynchronize")]
    public static partial int cuCtxSynchronize();

    [LibraryImport("nvcuda", EntryPoint = "cuEventCreate")]
    public static partial int cuEventCreate(out IntPtr handle, uint flags);

    [LibraryImport("nvcuda", EntryPoint = "cuEventRecord")]
    public static partial int cuEventRecord(IntPtr handle, IntPtr stream);

    [LibraryImport("nvcuda", EntryPoint = "cuEventElapsedTime")]
    public static partial int cuEventElapsedTime(out float milliseconds, IntPtr start, IntPtr end);

    [LibraryImport("nvcuda", EntryPoint = "cuEventDestroy")]
    public static partial int cuEventDestroy(IntPtr handle);

    [LibraryImport("nvcuda", EntryPoint = "cuGetErrorString")]
    public static partial int cuGetErrorString(int error, out IntPtr message);

    /// <summary>Raises a bench-local exception naming the driver call and its code (with the driver's own name for that code when it answers), so a failure reports what broke instead of surfacing as an unmarshalled non-zero.</summary>
    public static void Check(string operation, int result)
    {
        if (result == Success)
        {
            return;
        }

        string detail = string.Empty;
        try
        {
            if (cuGetErrorString(result, out IntPtr messagePtr) == Success && messagePtr != IntPtr.Zero)
            {
                detail = $" ({Marshal.PtrToStringAnsi(messagePtr)})";
            }
        }
        catch (EntryPointNotFoundException)
        {
            // An ancient driver without the error-name helper: report the code alone.
        }

        throw new ConcurrencyCudaException($"{operation} failed with CUDA result {result}{detail}");
    }
}

internal sealed class ConcurrencyCudaException : Exception
{
    public ConcurrencyCudaException(string message)
        : base(message)
    {
    }
}
