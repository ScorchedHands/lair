using System.Diagnostics;
using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Bench;

/// <summary>
/// CPU-vs-CUDA dense prefill measurement for the device composition seam: every prompt length in the sweep forwards the
/// same deterministic token window twice — once through this engine's own CPU route (its weight passes on that backend's
/// batched-row or blocked-matmul path, attention per row over host-resident KV blocks) and once with each layer of each
/// sub-batch composed through an opened <see cref="IDeviceAttentionStep"/> whose KV write and causal walk run against a
/// pool block store resident on the device. Wall time is measured around whole prefills; for the CUDA arm the same reps
/// also report that backend's cumulative deltas — device-measured kernel busy, runtime activation-in/attention-out copy
/// calls and bytes — beside the pool's residency facts (blocks the step wrote on-device, whole-block crossings the host
/// KV writes owed), because those two tallies measure different things: compute traffic against residency moves. One
/// untimed paired pass per length reports max absolute logit drift between the arms and whether their argmax continuations
/// agree — a basis statement only; no band is asserted here, so this bench never widens or narrows any recorded gate. The
/// default fixture is the pinned dense model (models/SmolLM-135M-Instruct-F16.gguf); when it is absent the run reports
/// that and exits without measuring anything. Output goes to stdout only — no report file, no config posture, no engine
/// surface change: this composition reaches the engine through its internal open seam alone.
/// </summary>
internal static class CudaPrefillBench
{
    private const string DefaultModelRelativePath = "models/SmolLM-135M-Instruct-F16.gguf";

    // The sweep defaults to the window lengths a prefill decision is actually made at, and repeats each arm enough times that median/min/max separate steady-state cost from first-touch page residency. Lengths are capped by the pool sizing this bench builds; --lengths overrides them.
    private static readonly int[] DefaultLengths = [64, 128, 256];

    // KV block granularity every suite and engine posture in this repo uses; capacity carries slack past the longest window so no measurement trips the pool mid-forward rather than measuring a refusal.
    private const int BlockTokens = 16;

    public static int Run(string[] args)
    {
        string modelPath = DefaultModelRelativePath;
        List<int> lengths = [.. DefaultLengths];
        int reps = 3;
        int threads = 0; // the CPU arm's backend sizing, same posture as the serve surface with engine.threads unset

        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--lengths":
                    if (i + 1 >= args.Length)
                    {
                        Console.WriteLine("cuda-prefill: option '--lengths' needs a value");
                        return 1;
                    }

                    List<int> parsed = [];
                    foreach (string part in args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                    {
                        if (!int.TryParse(part, out int length) || length < 1)
                        {
                            Console.WriteLine($"cuda-prefill: invalid length '{part}'");
                            return 1;
                        }

                        parsed.Add(length);
                    }

                    if (parsed.Count == 0)
                    {
                        Console.WriteLine("cuda-prefill: --lengths listed no positive lengths");
                        return 1;
                    }

                    lengths = [.. parsed.Distinct().OrderBy(l => l)];
                    break;
                case "--reps":
                    if (i + 1 >= args.Length || !int.TryParse(args[++i], out reps) || reps < 1)
                    {
                        Console.WriteLine("cuda-prefill: --reps must be a positive integer");
                        return 1;
                    }

                    break;
                case "--threads":
                    if (i + 1 >= args.Length || !int.TryParse(args[++i], out threads) || threads < 0)
                    {
                        Console.WriteLine("cuda-prefill: --threads must be a non-negative integer");
                        return 1;
                    }

                    break;
                default:
                    if (modelPath == DefaultModelRelativePath && !args[i].StartsWith("--", StringComparison.Ordinal))
                    {
                        modelPath = args[i];
                    }
                    else
                    {
                        Console.WriteLine($"cuda-prefill: unknown option '{args[i]}'");
                        return 1;
                    }

                    break;
            }
        }

        string? resolved = ResolveModelPath(modelPath);
        if (resolved is null)
        {
            Console.WriteLine($"cuda-prefill: fixture missing — no file at '{modelPath}' relative to this process or its enclosing repository");
            return 2;
        }

        using GgufFile gguf = GgufFile.Open(resolved);
        ModelPlan geometry = ModelPlans.Resolve(gguf);

        int maxLength = lengths[^1];
        int maxContextTokens = maxLength; // the engine's own context ceiling for a measured window; prefills never address past it, and nothing here decodes
        int capacityBlocks = (maxLength + BlockTokens - 1) / BlockTokens + 8;

        // A store-bound step addresses exactly the range its pool declares: positions come from block count x block tokens on both sides of that binding. The engine's context may sit below it — a composed window only ever reads and writes what this storage covers, never past its own ceiling.
        int deviceStorePositions = checked(capacityBlocks * BlockTokens);

        Console.WriteLine("cuda-prefill composition measurement");
        Console.WriteLine(
            $"  file={gguf.Path} bytes={gguf.FileSize} lengths={string.Join(",", lengths)} reps={reps} cpuThreads={(threads == 0 ? "auto" : threads.ToString())}");
        Console.WriteLine(
            $"  arch={geometry.Architecture} layers={geometry.LayerCount} hidden={geometry.HiddenSize} heads={geometry.HeadCount} " +
            $"kvHeads={geometry.KvHeadCount} headDim={geometry.HeadDim} vocab={geometry.VocabSize} resolvedPrefillRowWidth={geometry.PrefillRowWidth}");

        CudaBackend? cuda;
        try
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                Console.WriteLine("cuda-prefill: the CUDA driver seam runs on Windows against nvcuda.dll; this host reports otherwise");
                return 2;
            }

            cuda = new CudaBackend(CudaWeightResidency.DeviceCopy);
        }
        catch (EntryPointNotFoundException ex)
        {
            Console.WriteLine($"cuda-prefill: CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point ({ex.Message})");
            return 2;
        }
        catch (DllNotFoundException ex)
        {
            Console.WriteLine($"cuda-prefill: CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API ({ex.Message})");
            return 2;
        }
        catch (CudaException ex)
        {
            Console.WriteLine($"cuda-prefill: CUDA backend refused to open on this host: {CollapseWhitespace(ex.Message)}");
            return 2;
        }

        try
        {
            using CpuBackend cpu = new(threads: threads);
            string cpuRoute = ResolveCpuWeightPass(cpu, gguf); // the CPU arm takes the weight-pass route its own backend carries for this file's dense matrices — reported beside every ratio so each number states which chain it measured

            Console.WriteLine($"  arms: cpu(weightPass={cpuRoute}, attention=host per-row over pool blocks) + cuda(device step composed per layer, kvStore=pool device store resident on the device)");
            Console.WriteLine(
                "  basis: both arms forward identical token windows through TransformerModel.Prefill; wall time brackets each whole prefill (one untimed warm-up rep first); CUDA tallies are cumulative deltas across that arm's timed reps and count compute crossings only — pool residency moves print apart from them");

            foreach (int length in lengths)
            {
                int[] prompt = BuildPrompt(length, geometry.VocabSize);

                double cpuMedianMs = 0;
                double cpuMinMs = 0;
                double cpuMaxMs = 0;
                float[] referenceLogits = [];

                using (TransformerModel cpuModel = OpenCpuModel(gguf, cpu, maxContextTokens, cpuRoute))
                using (BlockPool cpuPool = NewHostPool(geometry, capacityBlocks, cpu))
                {
                    MeasureArm("cpu", $"length={length} tokens", () => Forward(cpuModel, cpuPool, prompt), reps, out cpuMedianMs, out cpuMinMs, out cpuMaxMs);

                    using SequenceKv probeKv = new(cpuPool);
                    referenceLogits = [.. cpuModel.Prefill(prompt, probeKv)];
                }

                long busyBefore = cuda.KernelBusyMicrosecondsTotal;
                long transferBytesBefore = cuda.RuntimeTransferBytesTotal;
                long transferCallsBefore = cuda.RuntimeTransferCallsTotal;

                double gpuMedianMs;
                double gpuMinMs;
                double gpuMaxMs;
                float[] deviceLogits;

                using (BlockPool pool = NewDeviceStorePool(geometry, capacityBlocks, cpu, cuda))
                using (IDeviceAttentionStep step = OpenComposedStep(cuda, geometry, deviceStorePositions, pool.DeviceStepMirror!))
                using (TransformerModel gpuModel = TransformerModel.Open(gguf, cuda, maxContextTokens, logLine: null, PrefillKernel.Gemm, WholeModelCudaRoute(geometry.Architecture), step))
                {
                    if (!gpuModel.DevicePrefillActive)
                    {
                        Console.WriteLine("cuda-prefill: the opened engine reported no device composition");
                        return 2;
                    }

                    MeasureArm("cuda", $"length={length} tokens", () => Forward(gpuModel, pool, prompt), reps, out gpuMedianMs, out gpuMinMs, out gpuMaxMs);

                    using SequenceKv probeKv = new(pool);
                    deviceLogits = [.. gpuModel.Prefill(prompt, probeKv)];

                    long busyUs = cuda.KernelBusyMicrosecondsTotal - busyBefore;
                    long transferBytes = cuda.RuntimeTransferBytesTotal - transferBytesBefore;
                    long transferCalls = cuda.RuntimeTransferCallsTotal - transferCallsBefore;

                    // The payoff gate reads this direction directly: a ratio above 1 means the composed GPU window answered in less wall time than CPU prefill at this length, and below 1 records it as no win rather than asserting fast prefill.
                    double wallRatioCpuOverGpu = gpuMedianMs > 0 ? cpuMedianMs / gpuMedianMs : double.NaN;

                    Console.WriteLine($"length={length} arm=cpu wallMedianMs={cpuMedianMs:F2} wallMinMs={cpuMinMs:F2} wallMaxMs={cpuMaxMs:F2} tokPerSecMedian={length / (cpuMedianMs / 1000.0):F1}");
                    Console.WriteLine(
                        $"length={length} arm=cuda wallMedianMs={gpuMedianMs:F2} wallMinMs={gpuMinMs:F2} wallMaxMs={gpuMaxMs:F2} tokPerSecMedian={length / (gpuMedianMs / 1000.0):F1} " +
                        $"wallRatioCpuOverGpu={wallRatioCpuOverGpu:F3} gpuWins={(gpuMedianMs < cpuMedianMs)}");
                    Console.WriteLine(
                        $"  cuda tallies reps={reps}: kernelBusyUs={busyUs} runtimeTransferCalls={transferCalls} runtimeTransferBytes={transferBytes} " +
                        $"bytesPerCall={(transferCalls > 0 ? (double)transferBytes / transferCalls : 0):F0}");
                    Console.WriteLine(
                        $"  kv residency: deviceWrittenBlocks={pool.DeviceWrittenBlocks} crossedBlocks={pool.DeviceMirrorCrossedBlocks} crossedBytes={pool.DeviceMirrorCrossedBytes} " +
                        $"residentDeviceKvBytes={pool.DeviceMirrorBytes} (residency moves, outside the runtime tallies above)");

                    int referenceArgmax = ArgMax(referenceLogits);
                    int deviceArgmax = ArgMax(deviceLogits);
                    double drift = MaxAbsDrift(referenceLogits, deviceLogits);
                    Console.WriteLine(
                        $"  parity basis length={length}: maxAbsLogitDrift={drift:E3} argmax cpu={referenceArgmax} cuda={deviceArgmax} " +
                        $"argmaxMatch={(referenceArgmax == deviceArgmax ? "yes" : "no")} (basis only — no band asserted here)");
                }

                Console.WriteLine();
            }

            Console.WriteLine($"  cuda facts: weightCopiesToDevice={cuda.DeviceWeightCopies} persistentDeviceWeightBytes={cuda.PersistentDeviceWeightBytes}");
            Console.WriteLine($"  peakWorkingSetMb={Process.GetCurrentProcess().PeakWorkingSet64 / (1024.0 * 1024.0):F1} across this process run");

            return 0;
        }
        finally
        {
            cuda.Dispose();
        }
    }

    // One arm's timed reps: an untimed warm-up pass first (device weight residency, worker pools, and the engine's first-touch allocations all land there), then median/min/max over whole-prefill wall times. The forward closure returns nothing measurable — logits are read once more outside timing for the parity basis line.
    private static void MeasureArm(string label, string what, Action forward, int reps, out double medianMs, out double minMs, out double maxMs)
    {
        forward();

        double[] samples = new double[reps];
        for (int rep = 0; rep < reps; rep++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            forward();
            sw.Stop();
            samples[rep] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(samples);
        medianMs = samples[reps / 2];
        minMs = samples[0];
        maxMs = samples[^1];
    }

    private static void Forward(TransformerModel model, BlockPool pool, int[] prompt)
    {
        using SequenceKv kv = new(pool);
        Span<float> logits = model.Prefill(prompt, kv);

        if (float.IsNaN(logits[0]))
        {
            Console.WriteLine("  NaN in prefill logits");
        }
    }

    // The pool the CPU arm runs its host attention over: plain DRAM-resident blocks on this engine's own backend — exactly what every default posture opens.
    private static BlockPool NewHostPool(ModelPlan geometry, int capacityBlocks, CpuBackend cpu) =>
        new(BlockTokens, geometry.LayerCount, geometry.KvHeadCount, geometry.HeadDim, capacityBlocks, cpu);

    // The pool the composed arm runs against: a host-owned arena whose whole declared range carries a persistent device copy through this CUDA backend's mirror in the device-step posture — the store-bound footing a GPU step reads and writes. Host KV bytes stay authoritative for anything this engine decodes afterward; the step records its own on-device writes as residency.
    private static BlockPool NewDeviceStorePool(ModelPlan geometry, int capacityBlocks, CpuBackend arenaOwner, CudaBackend mirrorOwner) =>
        new(
            BlockTokens, geometry.LayerCount, geometry.KvHeadCount, geometry.HeadDim, capacityBlocks, arenaOwner,
            BlockResidencyMode.Vram, deviceBackedBlocks: capacityBlocks, mirrorOwner, KvFormatPair.Default, KvMirrorExecution.DeviceStep);

    private static IDeviceAttentionStep OpenComposedStep(CudaBackend cuda, ModelPlan geometry, int maxPositions, IKvDeviceMirror kvStore) =>
        cuda.OpenAttentionStep(
            new CudaDeviceStepRequest
            {
                MaxPositions = maxPositions,
                LayerCount = geometry.LayerCount,
                HeadCount = geometry.HeadCount,
                KvHeadCount = geometry.KvHeadCount,
                HeadDim = geometry.HeadDim,
                Pairing = RopePairing.AdjacentPairs, // the llama-family pairing this seam rotates at; a composed window arrives already rotated, so the walk itself never applies it
                MaxWindowRows = Math.Max(CudaDeviceStepRequest.DefaultMaxWindowRows, geometry.PrefillRowWidth),
            },
            kvStore);

    private static TransformerModel OpenCpuModel(GgufFile gguf, CpuBackend cpu, int maxContextTokens, string cpuRoute) =>
        cpuRoute == PrefillKernelNames.Gemm
            ? TransformerModel.Open(gguf, cpu, maxContextTokens, logLine: null, PrefillKernel.Gemm)
            : TransformerModel.Open(gguf, cpu, maxContextTokens, logLine: null, PrefillKernel.GemV);

    // Which weight-pass route the CPU arm can actually run for this file: every dense attention/FFN matrix format its own backend multiplies in bulk selects 'gemm'; a format it has no blocked matmul for falls to batched rows — reported rather than decided silently, and the same gate the engine answers at open.
    private static string ResolveCpuWeightPass(CpuBackend cpu, GgufFile gguf)
    {
        foreach (LogicalTensor tensor in gguf.Tensors.Tensors)
        {
            if (!IsDenseMatrix(tensor.Name))
            {
                continue;
            }

            if (!cpu.SupportsGemm(tensor.Format))
            {
                return PrefillKernelNames.GemV;
            }
        }

        return PrefillKernelNames.Gemm;
    }

    private static bool IsDenseMatrix(string name) =>
        name.Contains(".attn_", StringComparison.Ordinal) || name.Contains(".ffn_", StringComparison.Ordinal);

    // A deterministic in-vocabulary window: token values only drive embedding lookups, so cycling ids exercises the same weight traversals and attention shapes every length measures. The same prompt drives both arms of a length.
    private static int[] BuildPrompt(int tokens, int vocabSize)
    {
        int usable = Math.Max(8, vocabSize - 2);
        int[] ids = new int[tokens];

        for (int i = 0; i < tokens; i++)
        {
            ids[i] = 1 + i % usable;
        }

        return ids;
    }

    private static ModelRoute WholeModelCudaRoute(string architecture) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = $"cuda-prefill-{architecture}",
            Stages = [new RouteStage(ModelRoutes.PrefillStage, CudaBackend.DeviceIdValue), new RouteStage(ModelRoutes.DecodeStage, CudaBackend.DeviceIdValue)],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaBackend.DeviceIdValue],
        });

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[best])
            {
                best = i;
            }
        }

        return best;
    }

    private static double MaxAbsDrift(ReadOnlySpan<float> reference, ReadOnlySpan<float> actual)
    {
        int shared = Math.Min(reference.Length, actual.Length);
        double worst = 0;

        for (int i = 0; i < shared; i++)
        {
            double drift = Math.Abs(reference[i] - actual[i]);
            if (drift > worst)
            {
                worst = drift;
            }
        }

        return worst;
    }

    // The fixture seam the test host uses, mirrored for the bench: walk up from this process's directory so a runner that changes working directories still finds the pinned file. An explicit rooted or plain path is taken as given when it exists.
    private static string? ResolveModelPath(string requested)
    {
        if (File.Exists(requested))
        {
            return Path.GetFullPath(requested);
        }

        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        for (int depth = 0; directory is not null && depth < 8; depth++)
        {
            string candidate = Path.Combine(directory.FullName, requested.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(candidate))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        return null;
    }

    private static string CollapseWhitespace(string text) =>
        string.Join(' ', text.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
}
