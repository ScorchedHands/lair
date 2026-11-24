using System.Diagnostics;
using System.Globalization;
using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Tokenizer;

namespace Lair.Bench;

// Manual Quant Recommendation measurement: the production CPU quantized-KV serving routes - the quantized TransformerModel
// Open overload closed over a KvFormatPair plus the backend's resolved IntegerDotPath, the typed BlockPool<Q8Block,...> the
// pair's group layout sizes, and SequenceKv<TK,TV> / IKvSequence forwards over the gemv batched-row window - put against the
// untouched FP32 f32/f32 baseline route on a pinned GGUF fixture, window by window, so a bench report can state what each
// landed posture costs and drifts without any config or serve-surface change.
//
// Nothing here opens a spike seam or a synthetic mirror: every forward runs the engine's own closed attention route, and the
// baseline mode opens exactly the float posture every host serves today - no quantized pair, no typed pool, no route argument.
internal static class KvQuantDecodeBench
{
    private const string DefaultModelRelativePath = "models/SmolLM-135M-Instruct-F16.gguf";

    internal static int Run(string[] args)
    {
        List<string> rest = [.. args[1..]];
        string modelPath = ResolveModelPath(rest.Count > 0 && !rest[0].StartsWith("--", StringComparison.Ordinal) ? rest[0] : DefaultModelRelativePath);

        if (rest.Count > 0 && !rest[0].StartsWith("--", StringComparison.Ordinal))
        {
            rest.RemoveAt(0);
        }

        int threads = ParseInt(rest, "--threads", 8);
        int context = ParseInt(rest, "--context", 1024);
        int decodeTokens = ParseInt(rest, "--decode", 32);
        int windows = ParseInt(rest, "--windows", 5);
        List<string> modes = [.. ParseList(rest, "--modes", ["baseline", "q8q8", "k8v4"])];
        bool stageProfile = rest.Contains("--stage-profile"); // off keeps every report line exactly as this bench has always printed them

        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"kv-quant: no fixture at {modelPath}");
            return 1;
        }

        if (threads < 1)
        {
            Console.WriteLine("kv-quant: --threads must be at least 1");
            return 1;
        }

        if (context < 1 || decodeTokens < 1 || windows < 1)
        {
            Console.WriteLine("kv-quant: --context, --decode, and --windows must all be at least 1");
            return 1;
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        ModelPlan geometry = ModelPlans.Resolve(gguf); // the plan an engine open resolves; the route its single CPU segment carries is what a quantized pair opens over
        ModelRoute route = geometry.Route;

        int blockTokens = 16; // the granularity every pool in this build uses
        int capacity = context + (windows + 1) * decodeTokens; // every position the run addresses: prefilled, warmed up, and decoded
        int capacityBlocks = (capacity + blockTokens - 1) / blockTokens + 4;
        int maxContext = capacity + blockTokens;

        using CpuBackend backend = new(threads: threads, allowAvx2: true);

        Console.WriteLine($"kv-quant file={gguf.Path} bytes={gguf.FileSize} threads={backend.EffectiveWorkerCount} isaResolved={ResolvedIntegerDotNames.Of(backend.ResolvedIntegerDotPath)}");
        Console.WriteLine(
            $"  arch={geometry.Architecture} layers={geometry.LayerCount} hidden={geometry.HiddenSize} heads={geometry.HeadCount} " +
            $"kvHeads={geometry.KvHeadCount} headDim={geometry.HeadDim} vocab={geometry.VocabSize} context={context} " +
            $"decode={decodeTokens} windows={windows} blockTokens={blockTokens} poolBlocks={capacityBlocks}");

        float[]? baselineProbe = null;
        bool anyRefused = false;

        foreach (string mode in modes)
        {
            if (mode is not ("baseline" or "q8q8" or "k8v4"))
            {
                Console.WriteLine($"kv-quant: unknown mode '{mode}' - expected baseline, q8q8, k8v4");
                return 1;
            }

            try
            {
                float[] probe = RunMode(mode, gguf, geometry, route, backend, maxContext, capacityBlocks, blockTokens, context, decodeTokens, windows, stageProfile);

                if (mode == "baseline")
                {
                    baselineProbe = probe;
                }
                else
                {
                    ReportEquivalence(mode, probe, baselineProbe, geometry.VocabSize);
                }
            }
            catch (ModelException refusal)
            {
                // A posture this build's quantized route cannot serve reports the engine's own refusal and stops this mode:
                // the bench never substitutes the float posture for a pair that refused.
                Console.WriteLine($"kv-quant mode={mode} status=refused reason={CollapseWhitespace(refusal.Message)}");
                anyRefused = true;
            }

            Console.WriteLine();
        }

        double peakWorkingSetMb = Process.GetCurrentProcess().PeakWorkingSet64 / (1024.0 * 1024.0);
        Console.WriteLine($"  peakWorkingSetMb={peakWorkingSetMb:F1} across this process run");

        return anyRefused ? 1 : 0;
    }

    // One mode end to end: the production open posture its pair names, the pool that pair's footprint sizes, and a decode
    // loop in the shape a served sequence runs - prompt prefilled once as setup outside timing, one untimed warm-up window,
    // then timed windows of a fixed token id and a fixed decode count over the same sequence.
    private static float[] RunMode(
        string mode, GgufFile gguf, ModelPlan geometry, ModelRoute route, CpuBackend backend,
        int maxContext, int capacityBlocks, int blockTokens, int context, int decodeTokens, int windows, bool stageProfile)
    {
        KvFormatPair pair = mode switch
        {
            "q8q8" => new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0),
            "k8v4" => new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q4_0),
            _ => KvFormatPair.Default,
        };

        // The baseline opens the untouched float posture - the same overload every host configuration reaches today, with no
        // quantized pair and no route argument. A quantized mode opens the production quantized overload: the pair closed
        // once at open beside the integer-dot encoding this backend resolved, over the plan's single CPU segment and the
        // gemv batched-row window the quantized kernel carries.
        using TransformerModel model = pair.IsFloatStorage
            ? TransformerModel.Open(gguf, backend, maxContext, logLine: null, PrefillKernel.GemV)
            : TransformerModel.Open(
                gguf,
                backend,
                maxContext,
                logLine: null,
                PrefillKernel.GemV, // the quantized route runs prompt windows through the batched-row window only
                route,
                pair,
                backend.ResolvedIntegerDotPath);

        if (stageProfile)
        {
            model.EnableDenseForwardProfiling(); // additive clock reads around the engine's existing dense spans only: every forward below still leaves the same logits and KV bytes an unprofiled run does
        }

        using IKvPool pool = pair switch
        {
            { K: KvSideFormat.Q8_0, V: KvSideFormat.Q8_0 } => new BlockPool<Q8Block, Q8Block>(blockTokens, geometry.LayerCount, geometry.KvHeadCount, geometry.HeadDim, capacityBlocks, backend, pair),
            { K: KvSideFormat.Q8_0, V: KvSideFormat.Q4_0 } => new BlockPool<Q8Block, Q4Block>(blockTokens, geometry.LayerCount, geometry.KvHeadCount, geometry.HeadDim, capacityBlocks, backend, pair),
            _ => new BlockPool(blockTokens, geometry.LayerCount, geometry.KvHeadCount, geometry.HeadDim, capacityBlocks, backend),
        };

        using IKvSequence kv = pool.NewSequence();

        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        int[] prompt = BuildPrompt(tokenizer, context);

        if (stageProfile)
        {
            model.ResetDenseForwardProfile(); // the setup window's profile is read alone, before any decode forward adds to it
        }

        Stopwatch prefillWatch = Stopwatch.StartNew();
        model.Prefill(prompt, kv); // fills every position the decode windows attend over: setup cost, reported apart from timing
        double prefillMs = prefillWatch.Elapsed.TotalMilliseconds;

        string? setupStageLine = stageProfile ? DenseDecodeStageLine(model, mode, "prefill_setup", prompt.Length, 1) : null;

        long reservedBytes = checked(pool.BytesPerBlock * pool.CapacityBlocks);
        Console.WriteLine(
            $"kv-quant mode={mode} storage={pool.Formats} attention={(pair.IsFloatStorage ? "production span<float> kernel" : "production quantized integer-dot kernel")} " +
            $"prefillSetupMs={prefillMs:F1} promptTokens={prompt.Length}");
        Console.WriteLine(
            $"kv-quant mode={mode} footprint bytesPerBlock={pool.BytesPerBlock} capacityBlocks={pool.CapacityBlocks} " +
            $"reservedKvBytes={reservedBytes} reservedKvMiB={reservedBytes / (1024.0 * 1024.0):F2} " +
            $"descriptorBytesPerBlock={pair.BytesPerBlock(geometry.LayerCount, blockTokens, pool.KvDim)}");

        if (setupStageLine is not null)
        {
            Console.WriteLine(setupStageLine);
        }

        return RunWindows(model, kv, mode, prompt[0], decodeTokens, windows, stageProfile);
    }

    private static float[] RunWindows(TransformerModel model, IKvSequence kv, string mode, int nextToken, int decodeTokens, int windows, bool stageProfile)
    {
        Stopwatch warmWatch = Stopwatch.StartNew();
        for (int i = 0; i < decodeTokens; i++)
        {
            model.DecodeNext(nextToken, kv); // JIT warm-up window: never timed
        }

        double warmMs = warmWatch.Elapsed.TotalMilliseconds;
        Console.WriteLine($"kv-quant mode={mode} warmWindowTok/s={(decodeTokens / (warmMs / 1000.0)):F3}");

        List<double> perWindowTokS = [];
        float[] probe = [];

        for (int w = 0; w < windows; w++)
        {
            int firstPosition = kv.Length;

            if (stageProfile)
            {
                model.ResetDenseForwardProfile(); // each window's stage totals describe that window alone: the untimed warm-up forwards and any earlier window are reset out
            }

            Stopwatch windowWatch = Stopwatch.StartNew();
            for (int i = 0; i < decodeTokens; i++)
            {
                Span<float> logits = model.DecodeNext(nextToken, kv);

                if (w == windows - 1 && i == decodeTokens - 1)
                {
                    probe = [.. logits]; // the final forced token's full distribution: copied out because the engine reuses its logit buffer every step
                }
            }

            double ms = windowWatch.Elapsed.TotalMilliseconds;
            perWindowTokS.Add(decodeTokens / (ms / 1000.0));

            Console.WriteLine(
                $"kv-quant mode={mode} window={w + 1} positions={firstPosition}..{kv.Length - 1} decodeMs={ms:F1} " +
                $"tok/s={(decodeTokens / (ms / 1000.0)):F3}");

            if (stageProfile)
            {
                Console.WriteLine(DenseDecodeStageLine(model, mode, "decode", kv.Length - firstPosition, decodeTokens));
            }
        }

        Console.WriteLine(FormatStats(mode, perWindowTokS));

        int top = ArgMax(probe);
        Console.WriteLine($"kv-quant mode={mode} probeTopId={top}/{probe.Length} probeTopLogit={probe[top]:F4}");

        return probe;
    }

    // The measured equivalence fact a quantized posture carries against this run's own FP32 baseline: the same prompt, the
    // same forced continuation, and the same final position, so any drift belongs to the opened storage pair and its kernel.
    // Reported as values, never asserted: a bench run states what the route costs and drifts, and parity gates live in tests.
    private static void ReportEquivalence(string mode, float[] probe, float[]? baselineProbe, int vocabSize)
    {
        if (probe.Length == 0)
        {
            return;
        }

        if (baselineProbe is null || baselineProbe.Length != probe.Length)
        {
            Console.WriteLine($"kv-quant mode={mode} probeEquivalence status=no-baseline-probe run baseline in the same invocation for the comparison");

            return;
        }

        int top = ArgMax(probe);
        int baselineTop = ArgMax(baselineProbe);

        double maxAbs = 0d;
        for (int i = 0; i < probe.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(probe[i] - baselineProbe[i]));
        }

        Console.WriteLine(
            $"kv-quant mode={mode} probeEquivalence argmaxMatch={(top == baselineTop)} baselineTopId={baselineTop}/{vocabSize} " +
            $"logitsMaxAbsDiffVsBaseline={maxAbs:F4}");
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }

    private static string FormatStats(string mode, List<double> values)
    {
        double median = Median(values);
        double min = values.Min();
        double max = values.Max();

        return $"kv-quant mode={mode} tok/s=[{string.Join(", ", values.Select(v => v.ToString("F3", CultureInfo.InvariantCulture)))}] " +
               $"median={median:F3} min={min:F3} max={max:F3} spread={(max - min) / median * 100.0:F2}%";
    }

    private static double Median(List<double> values)
    {
        List<double> sorted = [.. values];
        sorted.Sort();

        return sorted.Count % 2 == 1
            ? sorted[sorted.Count / 2]
            : (sorted[sorted.Count / 2 - 1] + sorted[sorted.Count / 2]) / 2.0;
    }

    // A prompt of exactly the requested token count from the repeated sentence every prefill harness uses: token values only
    // drive embedding lookups, so any in-vocabulary ids exercise the same weight and attention work.
    private static int[] BuildPrompt(GgufTokenizer tokenizer, int tokens)
    {
        const string line = "The quick brown fox jumps over the lazy dog while the patient cat watches the rain. ";

        List<int> ids = [];
        for (int i = 0; ids.Count < tokens && i < 4096; i++)
        {
            foreach (int id in tokenizer.Encode(line))
            {
                ids.Add(id);

                if (ids.Count == tokens)
                {
                    break;
                }
            }
        }

        return [.. ids.Take(tokens)];
    }

    // The fixture this bench defaults to, located the way the test host finds it: walk up from the assembly directory
    // until the repository-relative candidate exists, so a run from the repo root or from a worktree resolves the same file.
    private static string ResolveModelPath(string requested)
    {
        if (Path.IsPathRooted(requested) || File.Exists(requested))
        {
            return requested;
        }

        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        for (int depth = 0; directory is not null && depth < 8; depth++)
        {
            string candidate = Path.Combine(directory.FullName, requested);
            if (File.Exists(candidate))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        return requested; // unresolvable: the caller reports the fixture absence against the requested path
    }

    // One machine-readable dense stage line for a measured window group in this mode: the engine's own accumulated stage and per-layer totals since
    // its last reset, with the three heaviest whole-block layers ranked beside them. Parent stages cover their substages — attention weight/kernel/KV-write
    // times partition the attention stage, feed-forward traversal partitions its parent — so a reader names the dominant substage without double-counting it.
    // forwardCalls states how many forwards those totals describe: every stage figure is a sum over that group rather than one window's cost.
    private static string DenseDecodeStageLine(TransformerModel model, string mode, string windowKind, int tokensCovered, int forwardCalls)
    {
        DenseForwardProfile profile = model.DenseForwardProfileSummary();

        List<string> topLayers = [.. profile.Layers
            .OrderByDescending(timing => timing.TotalMs)
            .Take(3)
            .Select(timing => $"{timing.Layer}:{timing.TotalMs:F3}")];

        return $"kv-quant mode={mode} stage window={windowKind} tokens={tokensCovered} forwardCalls={forwardCalls}" +
               $" seed_ms={profile.SeedMs:F3} cache_append_ms={profile.CacheAppendMs:F3}" +
               $" attention_ms={profile.AttentionMs:F3} attention_weight_ms={profile.AttentionWeightMs:F3}" +
               $" attention_kernel_ms={profile.AttentionKernelMs:F3} kv_write_ms={profile.KvWriteMs:F3}" +
               $" feed_forward_ms={profile.FeedForwardMs:F3} feed_forward_weight_ms={profile.FeedForwardWeightMs:F3}" +
               $" logits_head_ms={profile.LogitsHeadMs:F3} total_ms={profile.TotalMs:F3}" +
               $" top_layers={string.Join(",", topLayers)}";
    }

    private static string CollapseWhitespace(string text) => string.Join(' ', text.Split(" \t\r\n", StringSplitOptions.RemoveEmptyEntries));

    private static int ParseInt(List<string> args, string key, int fallback)
    {
        int index = args.IndexOf(key);
        if (index < 0 || index + 1 >= args.Count || !int.TryParse(args[index + 1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int value))
        {
            return fallback;
        }

        return value;
    }

    private static IEnumerable<string> ParseList(List<string> args, string key, IEnumerable<string> fallback)
    {
        int index = args.IndexOf(key);
        if (index < 0 || index + 1 >= args.Count)
        {
            return fallback;
        }

        return [.. args[index + 1].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)];
    }
}
