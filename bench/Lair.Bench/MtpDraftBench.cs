using System.Diagnostics;
using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Sampling;
using Lair.Sampling.Drafting;

namespace Lair.Bench;

// CPU speculative-draft measurement seam on a pinned small fixture: the same TransformerModel decode forwards that serve today,
// driven through SpeculativeDecoder with scripted draft heads, put against an undrafted greedy baseline of identical sampling
// posture. Arms report median/min/max tok/s per window, accept-rate tallies, and drafted-vs-baseline token equality; nothing
// here loads a flagship artifact or claims a speedup the engine cannot yet express: verification through this seam is one verify
// forward per authoritative token (accept-prefix), so multipliers measure drafting overhead against an equivalent undrafted run
// until single-forward batched verification over unaccepted windows exists in the engine.
internal static class MtpDraftBench
{
    private const string DefaultModelRelativePath = "models/SmolLM-135M-Instruct-F16.gguf";
    private const int SentinelEos = -1; // no tokenizer stop token here: runs finish on their generation ceiling

    internal static int Run(string[] args)
    {
        List<string> rest = [.. args[1..]];
        string modelPath = ResolveModelPath(rest.Count > 0 && !rest[0].StartsWith("--", StringComparison.Ordinal) ? rest[0] : DefaultModelRelativePath);

        if (rest.Count > 0 && !rest[0].StartsWith("--", StringComparison.Ordinal))
        {
            rest.RemoveAt(0);
        }

        int threads = ParseInt(rest, "--threads", 8);
        int context = ParseInt(rest, "--context", 512);
        int decodeTokens = ParseInt(rest, "--decode", 32);
        int windows = ParseInt(rest, "--windows", 4);
        int draftWindow = ParseInt(rest, "--draft-window", 4);

        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"mtp: no fixture at {modelPath}");
            return 1;
        }

        if (threads < 1 || context < 1 || decodeTokens < 1 || windows < 1)
        {
            Console.WriteLine("mtp: --threads, --context, --decode, and --windows must all be at least 1");
            return 1;
        }

        if (draftWindow < 1)
        {
            Console.WriteLine("mtp: --draft-window must be at least 1");
            return 1;
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        ModelPlan geometry = ModelPlans.Resolve(gguf);

        int blockTokens = 8; // the granularity every pool in this bench uses for its sequences
        int capacityBlocks = (context + (windows + 2) * decodeTokens + draftWindow) / blockTokens + 4;

        using CpuBackend backend = new(threads: threads, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: context + (windows + 1) * decodeTokens + draftWindow + blockTokens);

        Console.WriteLine($"mtp file={gguf.Path} bytes={gguf.FileSize} threads={backend.EffectiveWorkerCount}");
        Console.WriteLine(
            $"  arch={geometry.Architecture} layers={geometry.LayerCount} hidden={geometry.HiddenSize} vocab={geometry.VocabSize} " +
            $"context={context} decode={decodeTokens} windows={windows} draftWindow={draftWindow} blockTokens={blockTokens} poolBlocks={capacityBlocks}");

        Console.WriteLine("  posture: verify forwards are one ordinary decode step per authoritative token (accept-prefix); multipliers below therefore document drafting overhead against an equivalent undrafted run, not a batched-verification speedup claim.");

        int[] prompt = [.. Enumerable.Repeat(7, 16)]; // fixed in-vocabulary ids; the bench measures decode cost, not tokenizer behavior
        SamplingParams greedy = new() { Temperature = 0f };

        using BlockPool pool = new(blockTokens, geometry.LayerCount, geometry.KvHeadCount, geometry.HeadDim, capacityBlocks, backend);

        // Reference continuation for the scripted heads comes from one untimed undrafted pass with identical sampling posture.
        List<int> reference = [];
        {
            using IKvSequence kv = pool.NewSequence();
            Sampler sampler = new(greedy);
            _ = model.Prefill(prompt, kv);
            int first = sampler.Sample(model.LastLogits());
            reference.Add(first);
            int pending = first;
            while (reference.Count < decodeTokens)
            {
                pending = sampler.Sample(model.DecodeNext(pending, kv));
                reference.Add(pending);
            }

            Console.WriteLine($"  reference: undrafted greedy continuation length={reference.Count} head5=[{string.Join(',', reference.Take(5))}]");
        }

        bool anyFailure = false;

        double baselineMedian = RunBaseline(model, pool, prompt, greedy, decodeTokens, windows);
        (double MedianTokPerSec, MtpTallySnapshot Stats, bool StreamMatch) oracleRun = RunDraftedArm("draftedOracle", model, pool, prompt, greedy, reference, draftWindow, oracle: true, decodeTokens, windows);
        (double MedianTokPerSec, MtpTallySnapshot Stats, bool StreamMatch) wrongRun = RunDraftedArm("draftedWrong", model, pool, prompt, greedy, reference, draftWindow, oracle: false, decodeTokens, windows);

        anyFailure |= !oracleRun.StreamMatch || !wrongRun.StreamMatch; // equivalence failures are surfaced by the run's exit code as well as in its per-arm lines
        Console.WriteLine();
        Console.WriteLine($"  baseline median={Fmt(baselineMedian)} tok/s");
        Console.WriteLine(
            $"  draftedOracle median={Fmt(oracleRun.MedianTokPerSec)} tok/s multiplierVsBaseline={Ratio(oracleRun.MedianTokPerSec, baselineMedian)} " +
            $"acceptRate={oracleRun.Stats.AcceptRate:F4} proposed={oracleRun.Stats.DraftsProposed} accepted={oracleRun.Stats.DraftTokensAccepted} rejected={oracleRun.Stats.DraftsRejected}");
        Console.WriteLine(
            $"  draftedWrong median={Fmt(wrongRun.MedianTokPerSec)} tok/s multiplierVsBaseline={Ratio(wrongRun.MedianTokPerSec, baselineMedian)} " +
            $"acceptRate={wrongRun.Stats.AcceptRate:F4} proposed={wrongRun.Stats.DraftsProposed} accepted={wrongRun.Stats.DraftTokensAccepted} rejected={wrongRun.Stats.DraftsRejected}");


        Console.WriteLine("  equivalence: drafted token streams are compared against the reference continuation per window; a mismatch prints and fails the run.");
        if (anyFailure)
        {
            return 1;
        }

        return 0;
    }

    // Undrafted greedy arm: prefill once as setup outside timing, an untimed warm advance on the live sequence, then timed windows advancing decodeTokens each.
    private static double RunBaseline(TransformerModel model, BlockPool pool, int[] prompt, SamplingParams sampling, int decodeTokens, int windows)
    {
        List<double> perWindow = [];

        using IKvSequence kv = pool.NewSequence();
        Sampler sampler = new(sampling); // one RNG stream for the whole arm, matching how a served pipeline owns its sampler
        _ = model.Prefill(prompt, kv);
        int pending = sampler.Sample(model.LastLogits());

        AdvanceBaseline(model, kv, ref pending, sampler, decodeTokens); // warm-up advance: the same forwards run untimed before any window is measured

        for (int w = 0; w < windows; w++)
        {
            Stopwatch clock = Stopwatch.StartNew();
            int emitted = AdvanceBaseline(model, kv, ref pending, sampler, decodeTokens); // timed window: same forwards the untimed warm-up ran
            clock.Stop();

            double tokPerSec = emitted * 1_000_000.0 / Math.Max(1, clock.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);
            perWindow.Add(tokPerSec);
        }

        Console.WriteLine($"  baseline windows=[{string.Join(',', perWindow.Select(v => Fmt(v)))}] median={Fmt(Median(perWindow))} min={Fmt(perWindow.Min())} max={Fmt(perWindow.Max())}");
        return Median(perWindow);
    }

    private static int AdvanceBaseline(TransformerModel model, IKvSequence kv, ref int pending, Sampler sampler, int tokens)
    {
        for (int i = 0; i < tokens; i++)
        {
            Span<float> logits = model.DecodeNext(pending, kv);
            pending = sampler.Sample(logits); // greedy: argmax of each row, no RNG consumption drift between arms
        }

        return tokens;
    }

    private static (double MedianTokPerSec, MtpTallySnapshot Stats, bool StreamMatch) RunDraftedArm(
        string armName,
        TransformerModel model,
        BlockPool pool,
        int[] prompt,
        SamplingParams sampling,
        IReadOnlyList<int> reference,
        int draftWindow,
        bool oracle,
        int decodeTokens,
        int windows)
    {
        IDraftHead head = oracle ? OracleHead(reference, prompt.Length, draftWindow) : WrongHead(reference, draftWindow);

        using IKvSequence kv = pool.NewSequence();
        SpeculativeDecoder decoder = new(sampling, head, model.Desc.VocabSize); // the arm's own sampling authority, same posture as the baseline sampler

        _ = model.Prefill(prompt, kv);
        int? firstFromLogits = decoder.StartFromLogits(model.LastLogits(), prompt, SentinelEos) ?? throw new InvalidOperationException("mtp: reference continuation must not end at a stop token");
        bool streamMatch = firstFromLogits == reference[0]; // the drafted run's generated stream starts with its sampled first token and is compared position-by-position against the undrafted reference
        if (!streamMatch)
        {
            Console.WriteLine($"  {armName}: FAIL first-token drift drafted={firstFromLogits} baseline={reference[0]}");
        }

        AdvanceDrafted(model, kv, decoder, prompt, decodeTokens); // warm-up advance outside timing

        List<double> perWindow = [];

        for (int w = 0; w < windows; w++)
        {
            int beforeGenerated = decoder.Generated.Count;
            Stopwatch clock = Stopwatch.StartNew();
            AdvanceDrafted(model, kv, decoder, prompt, decodeTokens); // timed window over the live drafted sequence
            clock.Stop();

            int emitted = decoder.Generated.Count - beforeGenerated;
            double tokPerSec = emitted * 1_000_000.0 / Math.Max(1, clock.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);
            perWindow.Add(tokPerSec);

            for (int i = beforeGenerated; i < decoder.Generated.Count && i < reference.Count; i++)
            {
                if (decoder.Generated[i] != reference[i])
                {
                    Console.WriteLine($"  {armName}: FAIL stream drift at position {i} drafted={decoder.Generated[i]} baseline={reference[i]}");
                    streamMatch = false;
                }
            }
        }

        MtpTallySnapshot stats = decoder.Tallies.Snapshot();
        Console.WriteLine(
            $"  {armName} windows=[{string.Join(',', perWindow.Select(v => Fmt(v)))}] median={Fmt(Median(perWindow))} min={Fmt(perWindow.Min())} max={Fmt(perWindow.Max())} " +
            $"stepsAdvanced={stats.StepsAdvanced} fullyAcceptedSteps={stats.FullyAcceptedSteps} mismatchedSteps={stats.MismatchedSteps} noProposalSteps={stats.NoProposalSteps} equivalence={(streamMatch ? "match" : "MISMATCH")}");

        return (Median(perWindow), stats, streamMatch);
    }

    // Advances the drafted run by exactly `tokens` generated tokens across whole boundaries, each boundary consulting its clamped window.
    private static void AdvanceDrafted(TransformerModel model, IKvSequence kv, SpeculativeDecoder decoder, int[] prompt, int tokens)
    {
        int targetGenerated = decoder.Generated.Count + tokens; // a fixed ceiling for the whole advance: boundaries stop at it exactly as a served run's budget would
        while (decoder.Generated.Count < targetGenerated)
        {
            DraftedAdvance step = decoder.Advance((token, seq) => model.DecodeNext(token, seq), kv, prompt, generationTokenCeiling: targetGenerated, eosId: SentinelEos);
            if (step.Finished is not null)
            {
                return; // a finished sequence stops the arm's advance just as it would stop a served run
            }
        }
    }

    private static ScriptedDraftHead OracleHead(IReadOnlyList<int> reference, int promptLen, int window) => new(window, ctx =>
    {
        List<int> drafts = [];
        int generatedSoFar = ctx.History.Length - promptLen; // the drafted sequence's accepted count: heads see prompt + accepted tokens only
        for (int i = 0; i < window && generatedSoFar + i < reference.Count; i++)
        {
            drafts.Add(reference[generatedSoFar + i]); // proposals replay the known-good continuation, so whole windows verify clean
        }

        return drafts;
    });

    private static ScriptedDraftHead WrongHead(IReadOnlyList<int> reference, int window)
    {
        HashSet<int> ids = [.. reference];
        int forbidden = ids.Contains(0) ? 1 : 0; // provably outside the continuation: under identical posture it can never be what the model samples at these positions

        return new ScriptedDraftHead(window, ctx =>
        {
            List<int> drafts = [];
            for (int i = 0; i < window; i++)
            {
                drafts.Add(forbidden);
            }

            return drafts;
        });
    }

    private static string Fmt(double v) => v.ToString("F3", System.Globalization.CultureInfo.InvariantCulture);

    private static double Median(List<double> values)
    {
        List<double> sorted = [.. values];
        sorted.Sort();
        int n = sorted.Count;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    private static string Ratio(double drafted, double baseline) =>
        (drafted / Math.Max(baseline, 1e-9)).ToString("F4", System.Globalization.CultureInfo.InvariantCulture);

    private static int ParseInt(List<string> args, string key, int fallback)
    {
        int index = args.IndexOf(key);
        if (index >= 0 && index + 1 < args.Count && int.TryParse(args[index + 1], out int value))
        {
            return value;
        }

        return fallback;
    }

    private static string ResolveModelPath(string requested)
    {
        if (Path.IsPathRooted(requested))
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

        return requested; // unresolved relative paths surface through the exists check below with a clear message
    }
}
