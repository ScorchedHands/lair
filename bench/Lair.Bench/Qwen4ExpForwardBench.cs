using System.Diagnostics;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Bench;

/// <summary>
/// CPU-only throughput harness over the assembly-internal bounded qwen4exp forward seam: open one pinned (possibly split)
/// GGUF through the ordinary loader/plan path on a plain CPU backend, run short synthetic prompt and generation windows
/// through chunk forwards, and publish prefill and decode tokens/sec beside peak-working-set facts — or print the engine's
/// own refusal verbatim when this file cannot open that seam. Synthetic token ids come from the plan's resolved vocabulary;
/// no tokenizer is consulted, because a target declaring an unsupported pre-tokenizer profile still has to be measured on
/// its weights and attention geometry. Every window stays inside the opened context ceiling: the bounded seam sizes its
/// sequence caches at open and never grows them. Serving routes (Prefill/DecodeNext/DecodeBatch), KV pools, dedicated lanes,
/// offloaded or crossing backends, device prefill steps and quantized KV postures are deliberately absent — every one of
/// them is refused by the seam anyway, so this harness opens none. <c>--threads T</c> is the backend's global worker request
/// (T &gt; 0) so the reported <c>threads=</c> equals <c>backend.EffectiveWorkerCount</c>, and every line carries stage timings
/// (plan_ms, open_ms) beside the prefill/decode/argmax timings so model preparation cost separates from forward cost.
/// Bench-only: no posture, config or production change.
///
/// Exit codes: 0 ran both windows; 1 malformed options (nothing opened); 2 a loader/plan/engine refusal naming what this
/// build cannot run against that file — reported as one machine-readable line rather than an unhandled crash.
/// </summary>
internal static class Qwen4ExpForwardBench
{
    private const int DefaultPromptTokens = 64;   // the first Xeon smoke window: short enough to stay inside a small context ceiling
    private const int DefaultGenTokens = 32;      // generation steps measured after an untimed prompt warm start on the same sequence
    private const int DefaultChunk = 32;          // rows per chunk forward, beside the seam's own sub-batch shape

    /// <summary>
    /// The name a --skip-ple tensor map points each n-gram module role at. No GGUF carries it — real files spell this family's
    /// PLE tensors under blk.N.ple_* and the table as per_layer_token_embd.weight — so the seam's presence scan resolves
    /// nothing, refuses nothing, and its forward never reads those bytes (no PLE gather is wired into this seam at all). Every
    /// role stays declared with its real scope; only what it points at changes.
    /// </summary>
    private const string PleSkipSentinel = "blk.{layer}.lair_bench_ple_ngram_module_not_loaded.weight";

    public static int Run(string[] args)
    {
        if (args.Length < 2 || args[1].StartsWith("-", StringComparison.Ordinal))
        {
            Console.WriteLine("qwen4exp-forward-file: a GGUF path is required");
            return Usage();
        }

        string path = args[1];
        int promptTokens = DefaultPromptTokens;
        int genTokens = DefaultGenTokens;
        int chunk = DefaultChunk;
        int threads = 0; // 0 keeps the backend's auto-sizing, the same posture a serve opens with engine.threads unset
        int reps = 1;
        int context = 0; // 0 resolves to this run's windows plus slack so no forward trips the opened ceiling
        bool skipPle = false;
        bool profileLayers = false;

        for (int i = 2; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--prompt-tokens":
                    if (!TryPositive(args, ref i, out promptTokens))
                    {
                        return OptionError(args[i]);
                    }

                    break;
                case "--gen-tokens":
                    if (!TryPositive(args, ref i, out genTokens))
                    {
                        return OptionError(args[i]);
                    }

                    break;
                case "--chunk":
                    if (!TryPositive(args, ref i, out chunk))
                    {
                        return OptionError(args[i]);
                    }

                    break;
                case "--reps":
                    if (!TryPositive(args, ref i, out reps))
                    {
                        return OptionError(args[i]);
                    }

                    break;
                case "--context":
                    if (!TryPositive(args, ref i, out context))
                    {
                        return OptionError(args[i]);
                    }

                    break;
                case "--threads":
                    if (i + 1 >= args.Length || !int.TryParse(args[i + 1], out threads) || threads < 0)
                    {
                        return OptionError(args[i]);
                    }

                    i++;
                    break;
                case "--skip-ple":
                    skipPle = true;
                    break;
                case "--profile-layers":
                    profileLayers = true;
                    break;
                default:
                    Console.WriteLine($"qwen4exp-forward-file: unknown option '{args[i]}'");
                    return Usage();
            }
        }

        int requiredContext = promptTokens + genTokens;
        if (context != 0 && context < requiredContext)
        {
            // The seam sizes every sequence cache at the opened ceiling and refuses a forward that would pass it, so an
            // under-sized window is a usage error here rather than a mid-run engine exception.
            Console.WriteLine($"qwen4exp-forward-file: --context {context} cannot carry this run's windows (needs at least {requiredContext}: prompt {promptTokens} + generation {genTokens})");
            return 1;
        }

        int openedContext = context == 0 ? requiredContext + Math.Max(8, requiredContext / 4) : context;

        GgufFile? gguf = null;
        try
        {
            gguf = GgufFile.Open(path);
            return RunOpened(gguf, promptTokens, genTokens, chunk, reps, openedContext, threads, skipPle, profileLayers);
        }
        catch (Exception failure) when (failure is GgufException or ModelException or IOException or UnauthorizedAccessException)
        {
            // A missing shard, a header this loader cannot parse, an oversized span the source refuses to hand back: all of
            // them land on one refusal line naming what was asked for instead of an unhandled crash.
            Console.WriteLine($"qwen4exp-forward-file file={path} status=refused stage=open reason={CollapseWhitespace(failure.Message)}");
            PrintPeakWorkingSet();
            return 2;
        }
        finally
        {
            gguf?.Dispose();
        }
    }

    static int RunOpened(GgufFile gguf, int promptTokens, int genTokens, int chunk, int reps, int openedContext, int threads, bool skipPle, bool profileLayers)
    {
        string declaredArchitecture = gguf.Metadata.GetString("general.architecture", "general");

        if (skipPle && !string.Equals(declaredArchitecture, "qwen4exp", StringComparison.Ordinal))
        {
            Console.WriteLine($"qwen4exp-forward-file file={gguf.Path} arch={declaredArchitecture}{Options(promptTokens, genTokens, chunk, reps, openedContext, threads, skipPle)} status=refused stage=options reason='--skip-ple' applies only to a general.architecture = qwen4exp file");
            return 2;
        }

        // One profile instance serves both the pre-resolved reporting plan and the engine open behind it: an ordinary qwen4exp
        // file resolves through the registry selection exactly as every other path does, and only a --skip-ple run supplies one.
        ModelProfile? pleSkipped = skipPle ? PleSkippedProfile() : null;

        ModelPlan? plan = null;
        Stopwatch planWatch = Stopwatch.StartNew();
        try
        {
            plan = pleSkipped is null ? ModelPlans.Resolve(gguf) : ModelPlans.Resolve(gguf, pleSkipped);
        }
        catch (Exception failure) when (failure is GgufException or ModelException)
        {
            // A header that parsed but whose tensor table contradicts the family's declared interleave still reports its own
            // architecture name beside the plan-level refusal, so a reader sees which side disagreed. The plan stage is timed
            // even on refusal so a slow resolve is visible in the same machine-readable shape a successful run prints.
            planWatch.Stop();
            Console.WriteLine(HeaderLine(gguf, declaredArchitecture, promptTokens, genTokens, chunk, reps, openedContext, threads, skipPle) + $" plan_ms={planWatch.Elapsed.TotalMilliseconds:F3} status=refused stage=plan reason={CollapseWhitespace(failure.Message)}");
            PrintPeakWorkingSet();
            return 2;
        }
        planWatch.Stop();
        double planMs = planWatch.Elapsed.TotalMilliseconds;

        // `--threads T` is the backend's global worker request (T > 0); `--threads 0` keeps the backend's auto posture.
        // This harness has no separate quota knob, so the request stands uncapped and EffectiveWorkerCount reports
        // exactly the thread count the run asked for — the posture the stage timings are read against.
        using CpuBackend backend = new(threads: threads, allowAvx2: true);

        TransformerModel? model = null;
        Stopwatch openWatch = Stopwatch.StartNew();
        try
        {
            model = pleSkipped is null
                ? TransformerModel.Open(gguf, backend, openedContext)
                : TransformerModel.Open(gguf, backend, openedContext, pleSkipped, logLine: null);
        }
        catch (Exception failure) when (failure is ModelException or InvalidOperationException)
        {
            openWatch.Stop();
            Console.WriteLine(IdentityLine(plan, gguf, promptTokens, genTokens, chunk, reps, openedContext, backend.EffectiveWorkerCount, skipPle) + StageTimings(planMs, openWatch.Elapsed.TotalMilliseconds) + $" status=refused stage=open reason={CollapseWhitespace(failure.Message)}");
            PrintPeakWorkingSet();
            return 2;
        }
        openWatch.Stop();
        double openMs = openWatch.Elapsed.TotalMilliseconds;

        using (model)
        {
            if (!model.HasQwen4ExpForward)
            {
                // The seam is chosen by the plan's family geometry: a file whose architecture resolves no qwen4exp geometry
                // never opens it, and this harness reports that rather than reaching for some other route.
                Console.WriteLine(IdentityLine(plan, gguf, promptTokens, genTokens, chunk, reps, openedContext, backend.EffectiveWorkerCount, skipPle) + StageTimings(planMs, openMs) + " status=refused stage=open reason='resolved no bounded qwen4exp forward seam (only a plan whose architecture resolves the qwen4exp family geometry opens it)'");
                PrintPeakWorkingSet();
                return 2;
            }

            int[] prompt = BuildSyntheticPrompt(plan.VocabSize, promptTokens);
            bool pleTensorsPresent = CarriesPleTensors(gguf);
            string parityField = skipPle && pleTensorsPresent ? "not_full_target_parity" : "bounded_seam";

            if (profileLayers)
            {
                model.EnableQwen4ExpLayerProfiling();
            }

            try
            {
                (double bestPrefillSeconds, double meanPrefillSeconds) = MeasurePrefill(model, prompt, chunk, reps, profileLayers);
                Qwen4ExpLayerProfile? prefillProfile = profileLayers ? model.Qwen4ExpLayerProfileSummary() : null;

                (double decodeSeconds, double argmaxSeconds, bool finiteDecode) = MeasureDecode(model, prompt, genTokens, chunk, profileLayers);
                Qwen4ExpLayerProfile? decodeProfile = profileLayers ? model.Qwen4ExpLayerProfileSummary() : null;

                string ropeNote = model.Qwen4ExpRotaryCoverage is { } coverage ? $" ropeHeadRegions={coverage.HeadRegions} ropeIndexerRegions={coverage.IndexerRegions}" : string.Empty;

                Console.WriteLine(
                    IdentityLine(plan, gguf, promptTokens, genTokens, chunk, reps, openedContext, backend.EffectiveWorkerCount, skipPle) +
                    StageTimings(planMs, openMs) +
                    $" prefill_tps={promptTokens / bestPrefillSeconds:F2} decode_tps={genTokens / decodeSeconds:F2}" +
                    $" prefill_best_ms={bestPrefillSeconds * 1000.0:F3} prefill_mean_ms={meanPrefillSeconds * 1000.0:F3} decode_ms={decodeSeconds * 1000.0:F3} argmax_ms={argmaxSeconds * 1000.0:F3}" +
                    $" parity={parityField} ple_tensors_present={(pleTensorsPresent ? "true" : "false")}" +
                    $" finite_logits={(finiteDecode ? "true" : "false")}{ropeNote} status=ok");

                if (prefillProfile is { } prefill && decodeProfile is { } decode)
                {
                    Console.WriteLine(ProfileLine("prefill", prefill));
                    Console.WriteLine(ProfileLine("decode", decode));
                }

                PrintPeakWorkingSet();
            }
            catch (Exception failure) when (failure is InvalidOperationException or ArgumentException || failure is ModelException || failure is GgufException)
            {
                // A forward that trips the seam — a window past the opened ceiling, an operand its own composition cannot run,
                // or a payload span wider than this source can hand back — lands on the same refusal shape as open does, so no
                // measurement path can crash this harness unhandled.
                Console.WriteLine(IdentityLine(plan, gguf, promptTokens, genTokens, chunk, reps, openedContext, backend.EffectiveWorkerCount, skipPle) + StageTimings(planMs, openMs) + $" status=refused stage=forward reason={CollapseWhitespace(failure.Message)}");
                PrintPeakWorkingSet();
                return 2;
            }
        }

        return 0;
    }

    // Prefill windows: every rep runs the whole prompt through chunk forwards on a fresh sequence (the seam's carried state is
    // per-sequence, so reusing one would score later reps against positions an earlier rep already wrote), and only the timed
    // reps count. The first pass over mapped weights pays page residency rather than kernel cost, which is why best and mean
    // land beside the tokens/sec figure a comparison reads off. With --profile-layers the warmup rep's contribution is reset
    // away so the reported stage/layer times describe the measured reps only.
    static (double BestSeconds, double MeanSeconds) MeasurePrefill(TransformerModel model, int[] prompt, int chunk, int reps, bool profile)
    {
        double best = double.PositiveInfinity;
        double total = 0.0;

        for (int rep = 0; rep < reps + 1; rep++)
        {
            Qwen4ExpSequence sequence = model.NewQwen4ExpSequence();

            Stopwatch sw = Stopwatch.StartNew();
            foreach ((int offset, int length) in Windows(prompt.Length, chunk))
            {
                model.ForwardQwen4ExpChunk(sequence, prompt.AsSpan(offset, length));
            }

            sw.Stop();

            if (rep == 0 && profile)
            {
                model.ResetQwen4ExpLayerProfile(); // the untimed warmup pass must not land in the measured profile
            }

            if (rep > 0)
            {
                best = Math.Min(best, sw.Elapsed.TotalSeconds);
                total += sw.Elapsed.TotalSeconds;
            }
        }

        return (best, total / reps);
    }

    // Decode steps: the prompt warms the sequence untimed through the same chunk seam, then each generation step forwards one
    // token alone against that carried state — exactly the shape a decode loop takes — and its next token is the argmax of the
    // logits row that forward produced. Timing covers all requested steps at once so tokens/sec describes the steady-state
    // path rather than one step with an allocation tax attached. The vocab-sized argmax scan is timed apart from the loop so a
    // reader can split a decode step into its forward cost (decode_ms minus argmax_ms) and its selection cost. With
    // --profile-layers the profile is reset after the untimed prompt warm start so the reported stage/layer times describe
    // the measured generation steps only.
    static (double Seconds, double ArgMaxSeconds, bool Finite) MeasureDecode(TransformerModel model, int[] prompt, int genTokens, int chunk, bool profile)
    {
        Qwen4ExpSequence sequence = model.NewQwen4ExpSequence();
        foreach ((int offset, int length) in Windows(prompt.Length, chunk))
        {
            model.ForwardQwen4ExpChunk(sequence, prompt.AsSpan(offset, length));
        }

        if (profile)
        {
            model.ResetQwen4ExpLayerProfile(); // the warm-start prompt forwards are not part of the measured decode profile
        }

        bool finite = true;
        int token = ArgMax(model.LastQwen4ExpLogits(sequence), ref finite);

        Stopwatch sw = Stopwatch.StartNew();
        double argmaxSeconds = 0.0;
        for (int step = 0; step < genTokens; step++)
        {
            model.ForwardQwen4ExpChunk(sequence, [token]);

            Stopwatch argmaxWatch = Stopwatch.StartNew();
            token = ArgMax(model.LastQwen4ExpLogits(sequence), ref finite);
            argmaxWatch.Stop();
            argmaxSeconds += argmaxWatch.Elapsed.TotalSeconds;
        }

        sw.Stop();
        return (sw.Elapsed.TotalSeconds, argmaxSeconds, finite);
    }

    static IEnumerable<(int Offset, int Length)> Windows(int tokens, int chunk)
    {
        for (int offset = 0; offset < tokens; offset += chunk)
        {
            yield return (offset, Math.Min(chunk, tokens - offset));
        }
    }

    // Synthetic in-vocabulary ids from the plan's resolved vocabulary: token values drive embedding lookups only, so this
    // exercises every weight and attention path without a tokenizer, exactly as the prefill profile's prompts do. Id 0 stays
    // out because it is the BOS slot on every family that declares one.
    static int[] BuildSyntheticPrompt(int vocabSize, int tokens)
    {
        int safeVocab = Math.Max(2, vocabSize);
        int[] prompt = new int[tokens];
        for (int i = 0; i < tokens; i++)
        {
            prompt[i] = 1 + (int)((long)i * 1_000_003L % (safeVocab - 1));
        }

        return prompt;
    }

    static int ArgMax(ReadOnlySpan<float> logits, ref bool finite)
    {
        int bestIndex = 0;
        float bestValue = float.NegativeInfinity;

        for (int i = 0; i < logits.Length; i++)
        {
            if (!float.IsFinite(logits[i]))
            {
                finite = false;
                continue;
            }

            if (logits[i] > bestValue)
            {
                bestValue = logits[i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    // Whether this file actually carries the family's n-gram module tensors. The scan reads the assembled table by name only:
    // a --skip-ple over a file that shipped none is not a parity delta, and the multi-gigabyte per_layer_token_embd.weight
    // table stays unread — it lives on its own shard and this harness resolves no span over it (which also keeps the mmap
    // source's oversized-span guard out of the picture entirely).
    static bool CarriesPleTensors(GgufFile gguf)
    {
        foreach (LogicalTensor tensor in gguf.Tensors.Tensors)
        {
            if (tensor.Name.Contains(".ple_", StringComparison.Ordinal) ||
                string.Equals(tensor.Name, "per_layer_token_embd.weight", StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }

    // The plan and profile the PLE-skipped open resolves through: same family semantics and geometry validation as an ordinary
    // open, with only the six n-gram module roles pointed at a name no file carries — which is what keeps the seam's presence
    // scan from refusing a target whose PLE forward this build does not compose.
    static ModelProfile PleSkippedProfile()
    {
        ModelProfile qwen4Exp = ModelProfiles.Resolve("qwen4exp");

        List<TensorDescriptor> descriptors = [];
        foreach (TensorDescriptor descriptor in qwen4Exp.Tensors.Descriptors)
        {
            bool isPleRole = descriptor.Role switch
            {
                TensorRole.PleConvolutionKernel or TensorRole.PleKeyProjection or TensorRole.PleValueProjection
                    or TensorRole.PleConvolutionNorm or TensorRole.PleKeyNorm or TensorRole.PleQueryNorm => true,
                _ => false,
            };

            descriptors.Add(isPleRole ? descriptor with { Name = PleSkipSentinel, Aliases = [] } : descriptor);
        }

        return qwen4Exp with { Tensors = new TensorMap(descriptors) };
    }

    static string HeaderLine(GgufFile gguf, string architecture, int promptTokens, int genTokens, int chunk, int reps, int context, int threads, bool skipPle) =>
        $"qwen4exp-forward-file file={gguf.Path} arch={architecture}{Options(promptTokens, genTokens, chunk, reps, context, threads, skipPle)}";

    static string IdentityLine(ModelPlan plan, GgufFile gguf, int promptTokens, int genTokens, int chunk, int reps, int context, int effectiveThreads, bool skipPle) =>
        $"qwen4exp-forward-file file={gguf.Path} arch={plan.Architecture}" +
        $" layers={plan.LayerCount} hidden={plan.HiddenSize} heads={plan.HeadCount} kvHeads={plan.KvHeadCount} headDim={plan.HeadDim} vocab={plan.VocabSize}" +
        (plan.Qwen4Exp is { } geometry ? $" linearLayers={geometry.LinearLayerCount} sparseFullLayers={geometry.FullLayerCount}" : string.Empty) +
        Options(promptTokens, genTokens, chunk, reps, context, effectiveThreads, skipPle);

    static string Options(int promptTokens, int genTokens, int chunk, int reps, int context, int threads, bool skipPle) =>
        $" promptTokens={promptTokens} genTokens={genTokens} chunk={chunk} reps={reps} context={context} threads={threads}" +
        (skipPle ? " skipPle=true" : string.Empty);

    // One machine-readable profile line per measured window: the stage partition of the engine's bounded forward plus the top
    // layers by whole-block time, compact enough to parse without swallowing the output. A layer entry is
    // layer:kind:total_ms with kind in {linear, sparse}. The moe_* fields are the sparse feed-forward stage's substage
    // partition beside moe_ffn_ms: router scores/selection/dispatch, routed gate, routed up plus its SwiGLU fold, routed
    // down, the routed weighted fold into each token's output row, and the shared expert as one region.
    static string ProfileLine(string stage, Qwen4ExpLayerProfile profile)
    {
        List<Qwen4ExpLayerTiming> ranked = [.. profile.Layers];
        ranked.Sort((a, b) => b.TotalMs.CompareTo(a.TotalMs));

        string topLayers = string.Join(',', ranked.Take(8).Select(entry =>
            $"{entry.Layer}:{(entry.Kind == AttentionKind.LinearGatedDeltaNet ? "linear" : "sparse")}:{entry.TotalMs:F3}"));

        return $"qwen4exp-forward-file profile stage={stage}" +
            $" seed_ms={profile.SeedMs:F3} hyper_mix_ms={profile.HyperMixMs:F3}" +
            $" hyper_mix_attn_ms={profile.HyperMixAttentionMs:F3} hyper_mix_ffn_ms={profile.HyperMixFeedForwardMs:F3}" +
            $" attention_linear_ms={profile.LinearAttentionMs:F3} attention_sparse_ms={profile.SparseAttentionMs:F3}" +
            $" moe_ffn_ms={profile.MoeFeedForwardMs:F3} moe_router_ms={profile.MoeRouterMs:F3}" +
            $" moe_gate_ms={profile.MoeGateMs:F3} moe_up_ms={profile.MoeUpMs:F3}" +
            $" moe_down_ms={profile.MoeDownMs:F3} moe_shared_ms={profile.MoeSharedMs:F3}" +
            $" moe_combine_ms={profile.MoeCombineMs:F3}" +
            $" dense_ffn_ms={profile.DenseFeedForwardMs:F3}" +
            $" combine_ms={profile.CombineMs:F3} logits_ms={profile.LogitsMs:F3}" +
            $" total_ms={profile.TotalMs:F3} top_layers={topLayers}";
    }

    // Stage timings that separate model preparation from the measured forwards: plan_ms covers ModelPlans.Resolve, open_ms
    // covers TransformerModel.Open. open_ms is omitted where the open never completed (a plan-stage refusal), so every line
    // reports exactly the stages that actually ran.
    static string StageTimings(double planMs, double? openMs = null) =>
        $" plan_ms={planMs:F3}" + (openMs is double open ? $" open_ms={open:F3}" : string.Empty);

    static void PrintPeakWorkingSet() =>
        Console.WriteLine($"qwen4exp-forward-file peak_working_set_mb={Process.GetCurrentProcess().PeakWorkingSet64 / (102.0 * 1024.0):F1}");

    // Advances the option cursor past a consumed value, so every flag reads its own argument through one shape.
    static bool TryPositive(string[] args, ref int index, out int value)
    {
        if (index + 1 < args.Length && int.TryParse(args[index + 1], out value) && value > 0)
        {
            index++;
            return true;
        }

        value = 0;
        return false;
    }

    static int OptionError(string option)
    {
        Console.WriteLine($"qwen4exp-forward-file: option '{option}' needs a positive integer (use --threads 0 for the backend's auto posture)");
        return Usage();
    }

    static string CollapseWhitespace(string text) => string.Join(' ', text.Split(" \t\r\n", StringSplitOptions.RemoveEmptyEntries));

    private static int Usage()
    {
        Console.WriteLine("qwen4exp-forward-file <gguf path> [--prompt-tokens N] [--gen-tokens M] [--context C] [--chunk K] [--threads T] [--reps R] [--skip-ple] [--profile-layers]" +
            " (CPU-only bounded qwen4exp forward measurement: synthetic in-vocabulary prompt ids and argmax-driven generation steps run through the engine's" +
            " assembly-internal chunk seam over a plain CPU backend, publishing prefill/decode tokens-per-second plus peak working-set facts beside the" +
            " plan-resolved geometry; --threads T sets the backend's global worker request so the reported threads= matches the effective worker count," +
            " --threads 0 keeps the backend's auto posture, an omitted --context sizes the opened ceiling from the requested windows;" +
            " every line carries stage timings (plan_ms for plan resolution, open_ms for model open) beside prefill_best_ms, prefill_mean_ms, decode_ms and" +
            " argmax_ms so load/open cost separates from forward cost;" +
            " --skip-ple opens a qwen4exp file whose n-gram module tensors this seam has no composition over by pointing those six roles at absent names," +
            " which measures every non-PLE block of that target and marks parity=not_full_target_parity rather than claiming full-target forward parity;" +
            " --profile-layers turns on the engine's bounded stage/layer profiling for the measured windows only (warmup forwards are reset out) and" +
            " prints one machine-readable profile line per window: stage=prefill|decode with seed_ms, hyper_mix_ms, hyper_mix_attn_ms, hyper_mix_ffn_ms, attention_linear_ms, attention_sparse_ms," +
            " moe_ffn_ms beside its substage partition moe_router_ms, moe_gate_ms, moe_up_ms, moe_down_ms, moe_shared_ms and moe_combine_ms, dense_ffn_ms, combine_ms, logits_ms, total_ms and" +
            " top_layers=layer:kind:ms entries ranked by whole-block time;" +
            " any loader/plan/engine refusal prints one status=refused line naming what this build cannot run against that file, with exit code 2)");
        return 1;
    }
}
