using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;

namespace Lair.Tests;

/// <summary>
/// Step-time prefill budget selection on a synthetic model: with the knob absent, chunk width stays exactly what it was configured to be;
/// with it present, each future chunk is sized from measured step cost instead — inside its floor of one token and its ceiling of the
/// configured maximum, never handing the engine more sub-batch rows than that engine's own resolved row count allows. Measured feedback is
/// what moves the width: a target this engine cannot reach lands on the floor rather than pretending a wider step fit, an edit mid-prefill
/// resizes from its next chunk issue while the step already chosen finishes whole, and every split the knob produced leaves the same token
/// stream as the run that split nothing. Chunk token count and sub-batch row count are reported apart from each other because they bound
/// different things — one is what a budget aims at, the other is the row window the engine splits it into and caps itself at.
/// </summary>
[Trait("Category", "Prefill")]
[Collection("model-loaded")] // shares the serialized real-model lane with the parity suites so these runs do not load alongside another engine
public sealed class PrefillStepBudgetSelectionTests : IDisposable
{
    private const int MaxBatchTokens = 64;

    // A prompt wide enough that a budgeted run needs several boundaries at this ceiling, and long enough for measured feedback to move the width.
    private const int LongPromptTokens = 240;

    // The step-time target these synthetic cases aim at: reachable on this engine, which is what the floor case below deliberately is not.
    private const int BudgetMs = 2;

    // Targets for the real fixture's mid-prefill edit case: one wide enough that its own ceiling pins every full boundary (so a pre-edit step lasts
    // long enough to be watched), and one narrow enough that measured cost has to pull future widths well below that ceiling.
    private const int WideBudgetMs = 400;
    private const int NarrowBudgetMs = 4;

    private readonly List<string> _tempFiles = [];

    [Fact]
    public async Task AbsentBudget_ForwardsChunksAtConfiguredWidth_ChunkCountAndRowsUnchanged()
    {
        string path = SyntheticLlamaGguf.Write("budget-absent");
        _tempFiles.Add(path);

        int[] prompt = BuildPrompt(LongPromptTokens);

        using BudgetHarness harness = new(path, maxBatchTokens: MaxBatchTokens, prefillStepBudgetMs: null);
        (SequenceOutcome outcome, List<ChunkChoice> choices) = await RunAndSampleAsync(harness.Scheduler, Request(prompt, 8), prompt.Length);

        Assert.Equal(LongPromptTokens, outcome.PromptTokens); // every prompt token rode a forward regardless of how the width was chosen
        Assert.True(outcome.CompletionTokens >= 1, "the run produced no tokens");

        // No knob in force: the configured ceiling is still exactly what each step forwards.
        Assert.Equal(0, harness.Scheduler.PrefillStepBudgetMicros);
        Assert.Equal((LongPromptTokens + MaxBatchTokens - 1) / MaxBatchTokens, harness.Scheduler.Stats.PrefillChunksIssued);
        Assert.Equal(LongPromptTokens, harness.Scheduler.Stats.PrefillTokensForwarded);

        // Selection with no budget takes min(remaining, ceiling), so this prompt's boundaries are three widths straight from the configuration and one
        // remainder — the last boundary recorded proves that shape directly. Wall-clock sampling on a loaded box cannot be trusted to catch any
        // specific earlier one, so it only carries bounds here rather than being the evidence itself.
        Assert.Equal(LongPromptTokens % MaxBatchTokens, harness.Scheduler.LastPrefillChosenTokens);
        Assert.All(choices, choice => Assert.InRange(choice.Tokens, 1, MaxBatchTokens));

        // Width and rows stay separate numbers even at the ceiling: a full-width step runs more than one row sub-batch under this engine's window.
        Assert.All(choices, choice => Assert.Equal(Math.Min(choice.Tokens, harness.Model.ResolvedPrefillRowWidth), choice.SubBatchRows));
        Assert.True(
            harness.Scheduler.Stats.PrefillSubBatchSum > harness.Scheduler.Stats.PrefillChunksIssued,
            $"{harness.Scheduler.Stats.PrefillSubBatchSum} sub-batch rows across {harness.Scheduler.Stats.PrefillChunksIssued} chunks — a ceiling-wide step never split into this engine's row windows");
    }

    [Fact]
    public async Task BudgetFromOpen_SizesFutureChunksFromMeasuredCost_InsideFloorAndCeiling_RowsStayCapped_OutputUnchanged()
    {
        string path = SyntheticLlamaGguf.Write("budget-open");
        _tempFiles.Add(path);

        int[] prompt = BuildPrompt(LongPromptTokens);

        // Baseline: the same engine and ceiling with no budget at all — every comparison anchors on its token stream.
        List<int> baselineTokens;
        long baselineChunks;
        {
            using BudgetHarness plain = new(path, maxBatchTokens: MaxBatchTokens, prefillStepBudgetMs: null);
            (SequenceOutcome baselineRun, _) = await RunAndSampleAsync(plain.Scheduler, Request(prompt, 8), prompt.Length);
            baselineTokens = [.. baselineRun.GeneratedTokens];
            baselineChunks = plain.Scheduler.Stats.PrefillChunksIssued;
        }

        Assert.Equal((LongPromptTokens + MaxBatchTokens - 1) / MaxBatchTokens, baselineChunks); // the un-budgeted shape this case compares against

        using BudgetHarness harness = new(path, maxBatchTokens: MaxBatchTokens, prefillStepBudgetMs: BudgetMs);
        Assert.Equal((long)BudgetMs * 1_000, harness.Scheduler.PrefillStepBudgetMicros); // the knob is in force from open

        (SequenceOutcome budgeted, List<ChunkChoice> choices) = await RunAndSampleAsync(harness.Scheduler, Request(prompt, 8), prompt.Length);

        Assert.Equal(baselineTokens, [.. budgeted.GeneratedTokens]); // different chosen widths moved nothing
        Assert.Equal(LongPromptTokens, harness.Scheduler.Stats.PrefillTokensForwarded); // narrowing split the same tokens, it did not drop any

        // The budget narrowed the run: more boundaries and narrower steps than the ceiling alone produced.
        Assert.True(
            harness.Scheduler.Stats.PrefillChunksIssued > baselineChunks,
            $"a budgeted prefill issued {harness.Scheduler.Stats.PrefillChunksIssued} chunks against an un-budgeted {baselineChunks} — measured cost never narrowed the step");

        double meanTokensPerChunk = harness.Scheduler.Stats.PrefillTokensForwarded / (double)harness.Scheduler.Stats.PrefillChunksIssued;
        Assert.True(
            meanTokensPerChunk < MaxBatchTokens,
            $"budgeted chunks averaged {meanTokensPerChunk:F1} tokens at a ceiling of {MaxBatchTokens} — the budget never reached selection");

        // Floor and ceiling held on every boundary this run recorded, and they hold for its aggregates whatever sampling managed to catch.
        Assert.All(choices, choice => Assert.InRange(choice.Tokens, 1, MaxBatchTokens));
        Assert.InRange(harness.Scheduler.LastPrefillChosenTokens, 1, MaxBatchTokens);

        // Sub-batch rows stayed inside what this engine sizes its batched streams for, whatever width was chosen above them.
        int rowCeiling = harness.Model.ResolvedPrefillRowWidth;
        Assert.All(choices, choice => Assert.Equal(Math.Min(choice.Tokens, rowCeiling), choice.SubBatchRows));
        Assert.InRange(harness.Scheduler.LastPrefillSubBatchRows, 1, rowCeiling);
        Assert.InRange(harness.Scheduler.Stats.PeakPrefillSubBatchRows, 1, rowCeiling);

        // Measured cost is what the widths came from: a per-token estimate fed back from an actual chunk width exists, a measured boundary width is on
        // record, and every step carries a duration for future boundaries to scale from. How far a target pulls widths below this engine's ceiling
        // depends on that engine's own cost curve — which the mid-prefill case measures against the real fixture rather than guessing here — so this
        // case asserts only what is invariant: narrowing happened relative to an un-budgeted run of the same prompt, inside floor and ceiling bounds.
        Assert.True(harness.Scheduler.EstimatedPrefillMicrosecondsPerToken > 0.0, "no measured prefill cost reached selection");
        Assert.True(harness.Scheduler.MeasuredPrefillWidthTokens >= 1, "measured cost was never fed back from a chunk width");
        Assert.InRange(harness.Scheduler.MeasuredPrefillWidthTokens, 1, MaxBatchTokens);

        Assert.True(
            harness.Scheduler.LastPrefillStepMicros >= 1,
            "the most recent prefill step recorded no measurement for future boundaries to aim at");
    }

    [Fact]
    public async Task BudgetNarrowerThanOneForwardStepOfThisEngine_SettlesAtFloorWidthWithOneRowPerSubBatch()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        int[] prompt = BuildRealPrompt(modelPath, 24);

        using GgufFile gguf = GgufFile.Open(modelPath);
        using CpuBackend backend = new(threads: 1);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 512, logLine: null, prefillKernel: PrefillKernel.Tiled); // the route this suite's latency gates run on

        try
        {
            using BlockPool pool = new(
                blockTokens: 16,
                layerCount: model.Desc.LayerCount,
                kvHeads: model.Desc.KvHeadCount,
                headDim: model.Desc.HeadDim,
                capacityBlocks: 32,
                backend);

            // The narrowest target the knob accepts. A single prompt position on this engine costs milliseconds, so measured cost finds no width that fits inside it and the choice has to stop at its floor rather than pretend a step did.
            using GenerationScheduler scheduler = new(model, pool, maxBatchTokens: MaxBatchTokens, cache: null, prefillStepBudgetMs: 1);

            SequenceOutcome outcome = await scheduler.Submit(Request(prompt, 2), CancellationToken.None).Completion;

            // The opening boundary takes the deliberate probe width (no measurement exists yet), and every later one lands on a single position:
            // measured cost cannot find any width inside this target, so selection stops at its floor instead of pretending some step fit.
            Assert.Equal(prompt.Length - GenerationScheduler.InitialPrefillProbeTokens + 1, scheduler.Stats.PrefillChunksIssued);
            Assert.Equal(prompt.Length, scheduler.Stats.PrefillTokensForwarded);
            Assert.True(scheduler.EstimatedPrefillMicrosecondsPerToken > 1_000.0, "measured prefill cost did not exceed this target");

            // At the floor every step is narrower than this engine's row window, so each one ran a single sub-batched forward: the run's sub-batch
            // count equals its chunk count exactly, and no step handed the engine more rows than that probe width or its own single position.
            Assert.Equal(1, scheduler.LastPrefillSubBatchRows);
            Assert.Equal(scheduler.Stats.PrefillChunksIssued, scheduler.Stats.PrefillSubBatchSum);
            Assert.InRange(scheduler.Stats.PeakPrefillSubBatchRows, 1, Math.Min(GenerationScheduler.InitialPrefillProbeTokens, model.ResolvedPrefillRowWidth));

            Assert.True(outcome.CompletionTokens >= 0, "the floor-bound run failed");
        }
        finally
        {
            backend.Dispose(); // disposed before the fixture file so a Windows mmap view is never left open on it
        }
    }

    [Fact]
    public async Task BudgetEditMidPrefill_ResizesFromNextChunkIssue_StepAlreadyChosenCompletes_OutputUnchanged()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        int[] prompt = BuildRealPrompt(modelPath, 170);

        using GgufFile gguf = GgufFile.Open(modelPath);
        using CpuBackend backend = new(threads: 1);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 1024, logLine: null, prefillKernel: PrefillKernel.Tiled);

        try
        {
            using BlockPool pool = new(
                blockTokens: 16,
                layerCount: model.Desc.LayerCount,
                kvHeads: model.Desc.KvHeadCount,
                headDim: model.Desc.HeadDim,
                capacityBlocks: 64,
                backend);

            List<int> baselineTokens;
            {
                using GenerationScheduler plain = new(model, pool, maxBatchTokens: MaxBatchTokens, cache: null);
                SequenceOutcome baselineRun = await RunAsync(plain, Request(prompt, 8));
                baselineTokens = [.. baselineRun.GeneratedTokens];
            }

            // Wide first: a target this engine cannot reach inside its own ceiling pins every full boundary at that ceiling. A step here costs tens of
            // milliseconds, which is what makes the moment mid-prefill observable — boundaries land slowly enough to be watched as they arrive.
            using GenerationScheduler budgeted = new(model, pool, maxBatchTokens: MaxBatchTokens, cache: null, prefillStepBudgetMs: WideBudgetMs);

            PipelineHandle handle = budgeted.Submit(Request(prompt, 8), CancellationToken.None);

            await WaitForAsync(
                () => budgeted.Stats.PrefillChunksIssued >= 3 && budgeted.LastPrefillChosenTokens * 2 >= MaxBatchTokens,
                "the chunked prefill never grew its chosen width toward the ceiling while active");

            long chunksAtEdit = budgeted.Stats.PrefillChunksIssued;
            long tokensAtEdit = budgeted.Stats.PrefillTokensForwarded;
            long microsWithWideTarget = budgeted.Stats.PrefillStepMicrosTotal;
            int widthAtEdit = budgeted.LastPrefillChosenTokens; // the family selection had reached when this target tightened

            // Tighten the target while a step is running: nothing about that step changes, and only later boundaries see the new value.
            budgeted.ApplyPrefillStepBudget(NarrowBudgetMs);
            Assert.Equal((long)NarrowBudgetMs * 1_000, budgeted.PrefillStepBudgetMicros);

            // Boundaries are sampled as they land from the edit on, so a resize is compared against what was already chosen.
            List<ChunkChoice> postEditChoices = [];
            long seenChunks = chunksAtEdit;

            while (!handle.Completion.IsCompleted)
            {
                long chunks = budgeted.Stats.PrefillChunksIssued;
                if (chunks > seenChunks)
                {
                    postEditChoices.Add(new ChunkChoice(budgeted.LastPrefillChosenTokens, budgeted.LastPrefillSubBatchRows));
                    seenChunks = chunks;
                }

                await Task.Delay(1);
            }

            SequenceOutcome outcome = await handle.Completion;

            Assert.True(
                budgeted.Stats.PrefillChunksIssued - chunksAtEdit >= 3,
                $"prefill finished before the edit applied ({chunksAtEdit} of {budgeted.Stats.PrefillChunksIssued} chunks) — widen this prompt");

            long tokensAfterEdit = budgeted.Stats.PrefillTokensForwarded - tokensAtEdit;
            long chunksAfterEdit = budgeted.Stats.PrefillChunksIssued - chunksAtEdit;
            double meanAfter = tokensAfterEdit / (double)chunksAfterEdit;

            Assert.True(
                meanAfter < widthAtEdit / 3.0,
                $"post-edit chunks averaged {meanAfter:F1} against a pre-edit width of {widthAtEdit} across {chunksAfterEdit} chunks — the tightened budget never narrowed future steps");

            // Boundaries that landed after the edit stayed inside floor and ceiling whatever family they came from, and every sub-batch they ran
            // stayed under this engine's own row window. A forward already issued when the target tightened runs whole at the width it was chosen
            // at — nothing here splits a step mid-forward, which is why the run's total forwarded tokens stay exactly the prompt's length below.
            Assert.NotEmpty(postEditChoices);
            Assert.All(postEditChoices, choice => Assert.InRange(choice.Tokens, 1, MaxBatchTokens));
            Assert.All(postEditChoices, choice => Assert.Equal(Math.Min(choice.Tokens, model.ResolvedPrefillRowWidth), choice.SubBatchRows));

            // Step time followed the same drop as width did, and nothing was forwarded twice or dropped.
            double meanStepMicrosBefore = microsWithWideTarget / (double)chunksAtEdit;
            double meanStepMicrosAfter = (budgeted.Stats.PrefillStepMicrosTotal - microsWithWideTarget) / (double)chunksAfterEdit;
            Assert.True(
                meanStepMicrosAfter < meanStepMicrosBefore,
                $"steps averaged {meanStepMicrosAfter:F0} us after the edit against {meanStepMicrosBefore:F0} us before it — a tightened target never shortened a step");

            Assert.Equal(prompt.Length, budgeted.Stats.PrefillTokensForwarded);
            Assert.True(outcome.CompletionTokens >= 1, "the resized run produced no tokens");

            // And identity across both widths: identical bytes to the run that split nothing.
            Assert.Equal(baselineTokens, [.. outcome.GeneratedTokens]);
        }
        finally
        {
            backend.Dispose(); // disposed before the fixture file so a Windows mmap view is never left open on it
        }
    }

[Fact]
    public async Task BudgetUnderConcurrency_NarrowsLongPrefillWhileShortRequestsAdvance_InsideFloorAndCeiling()
    {
        string path = SyntheticLlamaGguf.Write("budget-concurrency");
        _tempFiles.Add(path);

        int[] longPrompt = BuildPrompt(LongPromptTokens);
        int[] shortA = [1, 2, 3];       // hello world ! — ordinary entries of the synthetic vocabulary
        int[] shortB = [3, 1, 2];

        using BudgetHarness harness = new(path, maxBatchTokens: MaxBatchTokens, prefillStepBudgetMs: BudgetMs);

        // The short requests are submitted first and hold their slots across a slow stream of tokens: all three pipelines are provably advancing at
        // once no matter how the test thread is scheduled between submissions, and each hook fires on a boundary where this scheduler has forwarded
        // only part of that long prompt — the shorts advance while its chunked prefill is still running.
        List<long> partialPrefillTokensA = [];
        List<long> partialPrefillTokensB = [];
        PipelineHandle firstShort = harness.Scheduler.Submit(
            Request(shortA, 6, _ => { partialPrefillTokensA.Add(harness.Scheduler.Stats.PrefillTokensForwarded); Thread.Sleep(20); }),
            CancellationToken.None);
        PipelineHandle secondShort = harness.Scheduler.Submit(
            Request(shortB, 6, _ => { partialPrefillTokensB.Add(harness.Scheduler.Stats.PrefillTokensForwarded); Thread.Sleep(20); }),
            CancellationToken.None);
        PipelineHandle longPrefill = harness.Scheduler.Submit(Request(longPrompt, 8), CancellationToken.None);

        SequenceOutcome[] outcomes = await Task.WhenAll(firstShort.Completion, secondShort.Completion, longPrefill.Completion);

        Assert.Equal(6, outcomes[0].CompletionTokens); // a stop id outside the vocabulary keeps every run on its token budget
        Assert.Equal(6, outcomes[1].CompletionTokens);
        Assert.Equal(8, outcomes[2].CompletionTokens);

        // Every pipeline held its admission slot at once, and the shorts streamed while the long prompt was still prefilling: their hooks
        // fired on boundaries where this scheduler had forwarded only part of that prompt.
        Assert.Equal(3, harness.Scheduler.PeakAdvancingPipelines);
        Assert.True(partialPrefillTokensA.Count >= 4 && partialPrefillTokensB.Count >= 4, "a short request stopped advancing during the long prefill");
        Assert.True(
            partialPrefillTokensA.Any(tokensForwarded => tokensForwarded < LongPromptTokens) ||
            partialPrefillTokensB.Any(tokensForwarded => tokensForwarded < LongPromptTokens),
            $"no short token landed mid-prefill: A=[{string.Join(",", [.. partialPrefillTokensA, .. partialPrefillTokensB])}]");

        // Concurrency held the same bounds a solo run holds to, and the budget narrowed the long prompt's steps while they advanced.
        Assert.InRange(harness.Scheduler.Stats.PeakPrefillSubBatchRows, 1, harness.Model.ResolvedPrefillRowWidth);
        Assert.Equal(LongPromptTokens + shortA.Length + shortB.Length, harness.Scheduler.Stats.PrefillTokensForwarded);

        Assert.True(
            harness.Scheduler.Stats.PrefillChunksIssued > (LongPromptTokens + MaxBatchTokens - 1) / MaxBatchTokens,
            "the budget never narrowed the long prompt's chunks while other pipelines were advancing");
        Assert.InRange(harness.Scheduler.LastPrefillChosenTokens, 1, MaxBatchTokens);

        // The knob active under concurrency left every pipeline's stream identical to a run of that pipeline alone on an un-budgeted engine.
        foreach ((int[] prompt, int maxTokens, SequenceOutcome outcome) in new[] { (shortA, 6, outcomes[0]), (shortB, 6, outcomes[1]), (longPrompt, 8, outcomes[2]) })
        {
            using BudgetHarness solo = new(path, maxBatchTokens: MaxBatchTokens, prefillStepBudgetMs: null);
            SequenceOutcome alone = await RunAsync(solo.Scheduler, Request(prompt, maxTokens));
            Assert.Equal([.. alone.GeneratedTokens], [.. outcome.GeneratedTokens]);
        }

        Assert.True(harness.Scheduler.EstimatedPrefillMicrosecondsPerToken > 0.0, "no measured cost reached selection under concurrency");
    }

    // ---------------------------------------------------------------- harness and helpers

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
                // A model still holding its mmap view keeps the file on Windows; the temp directory absorbs it.
            }
        }

        _tempFiles.Clear();
    }

    /// <summary>One prefill boundary as the scheduler recorded it: tokens chosen for the step, and the rows that step's forward ran under this engine's resolved row count.</summary>
    private readonly record struct ChunkChoice(int Tokens, int SubBatchRows);

    // A stop id outside this synthetic vocabulary keeps every run on the length path instead of ending on the model's real EOS.
    private static ScheduledRequest Request(int[] promptTokens, int maxTokens, Action<int>? onTokenId = null) => new()
    {
        PromptTokens = promptTokens,
        MaxTokens = maxTokens,
        Sampling = new SamplingParams { Temperature = 0f }, // greedy: the only thing a width change could move is nothing at all
        EosId = 999,
        OnTokenId = onTokenId,
    };

    private static async Task<SequenceOutcome> RunAsync(GenerationScheduler scheduler, ScheduledRequest request) =>
        await scheduler.Submit(request, CancellationToken.None).Completion;

    /// <summary>Runs one request and records the boundaries it chose as they land. Sampling can skip a boundary on a fast engine, which is why these gates also assert each run's aggregates rather than only what was observed here.</summary>
    private static async Task<(SequenceOutcome Outcome, List<ChunkChoice> Choices)> RunAndSampleAsync(GenerationScheduler scheduler, ScheduledRequest request, int promptTokens)
    {
        PipelineHandle handle = scheduler.Submit(request, CancellationToken.None);
        List<ChunkChoice> choices = [];
        long seenChunks = 0;

        while (!handle.Completion.IsCompleted && scheduler.Stats.PrefillTokensForwarded < promptTokens)
        {
            long chunks = scheduler.Stats.PrefillChunksIssued;
            if (chunks > seenChunks)
            {
                choices.Add(new ChunkChoice(scheduler.LastPrefillChosenTokens, scheduler.LastPrefillSubBatchRows));
                seenChunks = chunks;
            }

            await Task.Delay(1);
        }

        long finalChunks = scheduler.Stats.PrefillChunksIssued;
        if (finalChunks > seenChunks && scheduler.Stats.PrefillTokensForwarded >= promptTokens)
        {
            choices.Add(new ChunkChoice(scheduler.LastPrefillChosenTokens, scheduler.LastPrefillSubBatchRows));
        }

        return (await handle.Completion, choices);
    }

    private static async Task WaitForAsync(Func<bool> condition, string what)
    {
        long deadline = Environment.TickCount64 + 25_000;
        while (!condition())
        {
            if (Environment.TickCount64 >= deadline)
            {
                Assert.Fail(what);
            }

            await Task.Delay(10);
        }
    }

    // A prompt of exactly the requested token count cycling the synthetic vocabulary's ordinary entries.
    // A prompt of the requested size built from the real fixture's tokenizer, so its token count is what this engine actually counts.
    private static int[] BuildRealPrompt(string modelPath, int minimumTokens)
    {
        using GgufFile gguf = GgufFile.Open(modelPath);

        const string line = "The quick brown fox jumps over the lazy dog while the patient cat watches the rain.";
        List<int> ids = [];
        for (int round = 0; ids.Count < minimumTokens && round < 16; round++)
        {
            ids.AddRange(Lair.Tokenizer.GgufTokenizer.FromGguf(gguf).Encode(line));
        }

        Assert.True(ids.Count >= minimumTokens, $"fixture prompt built only {ids.Count} tokens");
        return [.. ids];
    }

    private static int[] BuildPrompt(int tokens)
    {
        int[] cycle = [1, 2, 3];

        List<int> ids = [];
        while (ids.Count < tokens)
        {
            ids.AddRange(cycle);
        }

        return [.. ids.Take(tokens)];
    }

    private sealed class BudgetHarness : IDisposable
    {
        public GgufFile Gguf { get; }
        public CpuBackend Backend { get; }
        public TransformerModel Model { get; }
        public BlockPool Pool { get; }
        public GenerationScheduler Scheduler { get; }

        public BudgetHarness(string modelPath, int maxBatchTokens, int? prefillStepBudgetMs)
        {
            Gguf = GgufFile.Open(modelPath);
            Backend = new CpuBackend(threads: 1);
            Model = TransformerModel.Open(Gguf, Backend, maxContextTokens: 512);
            Pool = new BlockPool(
                blockTokens: 4,
                layerCount: Model.Desc.LayerCount,
                kvHeads: Model.Desc.KvHeadCount,
                headDim: Model.Desc.HeadDim,
                capacityBlocks: 96,
                Backend);
            Scheduler = new GenerationScheduler(Model, Pool, maxBatchTokens, cache: null, prefillStepBudgetMs: prefillStepBudgetMs);
        }

        public void Dispose()
        {
            Scheduler.Dispose();
            Pool.Dispose();
            Model.Dispose();
            Gguf.Dispose();
            Backend.Dispose();
        }
    }
}
