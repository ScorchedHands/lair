using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Pipeline control-plane behavior on a synthetic tiny model: an admitted request is advanced as its own state machine, over-cap
/// submissions wait for an advancement slot instead of being rejected or preempting work already running, and a pipeline's token stream
/// is identical whether it stepped alone, rode in shared steps with other pipelines, or advanced under a cap that forces one-at-a-time
/// stepping. Every request here runs greedy, so scheduling shape is the only thing these tests could move — which makes an unchanged
/// token stream evidence rather than coincidence.
/// </summary>
[Trait("Category", "Scheduling")]
[Collection("model-loaded")]
public sealed class PipelineSchedulingTests : IDisposable
{
    private const int GenerationCeiling = 12;

    // A prompt long enough to need many chunk boundaries at this harness's chunk width, so it holds its advancement slot while other
    // pipelines are still arriving and waiting behind the cap.
    private const int AnchorPromptTokens = 30;

    private readonly List<string> _tempFiles = [];

    [Fact]
    public async Task ConcurrencyCap_QueuesOverCapWithoutRejection_AndNeverAdvancesMoreThanTheCap()
    {
        string path = SyntheticLlamaGguf.Write("pipeline-cap");
        _tempFiles.Add(path);

        const int cap = 2;
        using PipelineHarness harness = new(path, capacityBlocks: 48, maxContextTokens: 64, maxBatchTokens: 2, maxConcurrentRequests: cap);

        ScheduledRequest anchor = Request(BuildLongPrompt(harness.Tokenizer), GenerationCeiling);
        ScheduledRequest shortA = Request(Encode(harness.Tokenizer, "hello world"), GenerationCeiling);
        ScheduledRequest shortB = Request(Encode(harness.Tokenizer, "world !"), GenerationCeiling);

        // The anchor's many chunk boundaries hold one slot; the two submissions behind it share whatever remains.
        PipelineHandle anchorHandle = harness.Scheduler.Submit(anchor, CancellationToken.None);
        PipelineHandle handleA = harness.Scheduler.Submit(shortA, CancellationToken.None);
        PipelineHandle handleB = harness.Scheduler.Submit(shortB, CancellationToken.None);

        SequenceOutcome[] outcomes = await Task.WhenAll(anchorHandle.Completion, handleA.Completion, handleB.Completion);

        // Submissions behind the cap were admitted and ran to their ceilings rather than refused.
        Assert.All(outcomes, outcome =>
        {
            Assert.Equal(FinishReason.Length, outcome.FinishReason);
            Assert.Equal(GenerationCeiling, outcome.CompletionTokens);
        });

        // The cap held in both accounts of concurrency: peak slots ever handed out, and per-step window occupancy.
        Assert.Equal(cap, harness.Scheduler.MaxConcurrentRequests);
        Assert.Equal(cap, harness.Scheduler.PeakAdvancingPipelines);
        Assert.True(
            harness.Scheduler.Stats.WindowSequenceSum <= harness.Scheduler.Stats.DecodeSteps * cap,
            $"windows advanced {harness.Scheduler.Stats.WindowSequenceSum} pipeline steps across {harness.Scheduler.Stats.DecodeSteps} decode steps — past a cap of {cap}");

        // The anchor's chunked prefill plus two short prompts guarantee at least one window held two pipelines: the cap bounds
        // concurrency, it does not serialize everything.
        Assert.True(
            harness.Scheduler.Stats.WindowSequenceSum > harness.Scheduler.Stats.DecodeSteps,
            "no step ever advanced more than one pipeline under a cap of 2 — advancement was serialized rather than capped");

        Assert.Equal(0, harness.Scheduler.AdvancingPipelineCount); // every slot came back as its pipeline finished
    }

    [Fact]
    public async Task QueuedPipeline_CancellingBeforeItEverAdvanced_FinishesCanceledWithNoTokens()
    {
        string path = SyntheticLlamaGguf.Write("pipeline-queued-cancel");
        _tempFiles.Add(path);

        using PipelineHarness harness = new(path, capacityBlocks: 48, maxContextTokens: 64, maxBatchTokens: 2, maxConcurrentRequests: 1);

        // The anchor holds the only slot across many chunk boundaries, so a pipeline submitted behind it waits at admission.
        PipelineHandle anchor = harness.Scheduler.Submit(Request(BuildLongPrompt(harness.Tokenizer), GenerationCeiling), CancellationToken.None);

        using var cancel = new CancellationTokenSource();
        PipelineHandle queued = harness.Scheduler.Submit(Request(Encode(harness.Tokenizer, "hello"), GenerationCeiling), cancel.Token);
        cancel.Cancel();

        SequenceOutcome queuedOutcome = await queued.Completion;
        Assert.Equal(FinishReason.Canceled, queuedOutcome.FinishReason);
        Assert.Empty(queuedOutcome.GeneratedTokens); // a waiting pipeline issued no forward before its cancellation was observed
        Assert.Equal(0, queuedOutcome.PrefillCachedTokens);

        SequenceOutcome anchorOutcome = await anchor.Completion;
        Assert.Equal(FinishReason.Length, anchorOutcome.FinishReason);
        Assert.True(
            anchorOutcome.CompletionTokens == GenerationCeiling,
            $"cancelling the waiting pipeline disturbed the advancing one: {anchorOutcome.FinishReason} after {anchorOutcome.CompletionTokens} tokens");

        Assert.Equal(1, harness.Scheduler.PeakAdvancingPipelines); // nothing ever advanced alongside the anchor
    }

    [Fact]
    public async Task PipelineTokenStreams_AreIdenticalRunAloneMergedAndUnderACapOfOne()
    {
        string path = SyntheticLlamaGguf.Write("pipeline-identity");
        _tempFiles.Add(path);

        using PipelineHarness prompts = new(path, capacityBlocks: 8, maxContextTokens: 64, maxBatchTokens: 2, maxConcurrentRequests: null);

        ScheduledRequest[] requests =
        [
            Request(Encode(prompts.Tokenizer, "hello"), GenerationCeiling),
            Request(Encode(prompts.Tokenizer, "world !"), GenerationCeiling),
            Request(Encode(prompts.Tokenizer, "! hello world"), GenerationCeiling),
            Request(BuildLongPrompt(prompts.Tokenizer), GenerationCeiling),
        ];

        Assert.True(requests[0].PromptTokens.Length < requests[3].PromptTokens.Length, "merged runs need pipelines with different KV histories");

        using PipelineHarness solo = new(path, capacityBlocks: 48, maxContextTokens: 64, maxBatchTokens: 2, maxConcurrentRequests: null);
        SequenceOutcome[] alone = await RunSolo(solo, requests);

        Assert.True(
            solo.Scheduler.Stats.DecodeSteps > 0 && solo.Scheduler.Stats.WindowSequenceSum == solo.Scheduler.Stats.DecodeSteps,
            "the baseline did not step one pipeline at a time — the comparison needs that shape");

        using PipelineHarness merged = new(path, capacityBlocks: 48, maxContextTokens: 64, maxBatchTokens: 2, maxConcurrentRequests: null);
        SequenceOutcome[] together = await RunTogether(merged, requests);

        Assert.True(
            merged.Scheduler.Stats.WindowSequenceSum > merged.Scheduler.Stats.DecodeSteps && merged.Scheduler.PeakAdvancingPipelines >= 2,
            $"the uncapped run never merged pipelines: {merged.Scheduler.Stats.WindowSequenceSum} pipeline steps across {merged.Scheduler.Stats.DecodeSteps} steps, peak {merged.Scheduler.PeakAdvancingPipelines}");

        AssertOutcomesEqual(alone, together); // same tokens whether a pipeline stepped in a window of one or rode shared windows

        using PipelineHarness capped = new(path, capacityBlocks: 48, maxContextTokens: 64, maxBatchTokens: 2, maxConcurrentRequests: 1);
        SequenceOutcome[] oneAtATime = await RunTogether(capped, requests);

        AssertOutcomesEqual(alone, oneAtATime); // advancement forced through a cap of one changes nothing either

        Assert.True(
            capped.Scheduler.PeakAdvancingPipelines == 1 && capped.Scheduler.Stats.DecodeSteps == capped.Scheduler.Stats.WindowSequenceSum,
            $"the capped scheduler advanced {capped.Scheduler.Stats.WindowSequenceSum} pipelines across {capped.Scheduler.Stats.DecodeSteps} steps with peak {capped.Scheduler.PeakAdvancingPipelines} — expected one pipeline per step");
    }

    [Fact]
    public async Task PerPipelineLogitTransforms_StayOnTheirOwnRow_MergedOrAlone()
    {
        string path = SyntheticLlamaGguf.Write("pipeline-transforms");
        _tempFiles.Add(path);

        using PipelineHarness prompts = new(path, capacityBlocks: 8, maxContextTokens: 64, maxBatchTokens: 2, maxConcurrentRequests: null);

        // One pipeline carries penalties and a logit bias, the other none. Each transform reads only its own prompt plus already-generated
        // history and adjusts only its own logits row; every logprob capture comes off that same transformed row, so comparing captures
        // compares what each pipeline actually saw — not just which tokens it picked.
        ScheduledRequest plain = Request(Encode(prompts.Tokenizer, "hello world"), GenerationCeiling) with { LogProbTopN = 3 };
        ScheduledRequest transformed = plain with
        {
            PromptTokens = Encode(prompts.Tokenizer, "world ! hello"),
            Sampling = new SamplingParams
            {
                Temperature = 0f,
                PresencePenalty = 0.4f,
                FrequencyPenalty = -0.3f,
                LogitBias = new Dictionary<int, float> { [1] = 2.5f, [3] = -2.5f },
            },
        };

        Assert.True(transformed.Sampling.HasLogitTransform, "the transform-carrying request resolved to no logit transform");

        ScheduledRequest[] requests = [plain, transformed];

        using PipelineHarness solo = new(path, capacityBlocks: 48, maxContextTokens: 64, maxBatchTokens: 2, maxConcurrentRequests: null);
        SequenceOutcome[] alone = await RunSolo(solo, requests);

        using PipelineHarness merged = new(path, capacityBlocks: 48, maxContextTokens: 64, maxBatchTokens: 2, maxConcurrentRequests: null);
        SequenceOutcome[] together = await RunTogether(merged, requests);

        AssertOutcomesEqual(alone, together);

        for (int i = 0; i < requests.Length; i++)
        {
            IReadOnlyList<TokenLogprob> expected = alone[i].LogProbs ?? [];
            IReadOnlyList<TokenLogprob> actual = together[i].LogProbs ?? [];

            Assert.Equal(GenerationCeiling, expected.Count);
            Assert.Equal(expected.Count, actual.Count);

            for (int token = 0; token < expected.Count; token++)
            {
                Assert.Equal(expected[token].TokenId, actual[token].TokenId);
                Assert.Equal(expected[token].LogProb, actual[token].LogProb); // exact double equality: the same transformed row either way

                IReadOnlyList<(int TokenId, double LogProb)> expectedTop = expected[token].TopLogProbs;
                IReadOnlyList<(int TokenId, double LogProb)> actualTop = actual[token].TopLogProbs;

                Assert.Equal(expectedTop.Count, actualTop.Count);
                for (int c = 0; c < expectedTop.Count; c++)
                {
                    Assert.Equal(expectedTop[c].TokenId, actualTop[c].TokenId);
                    Assert.Equal(expectedTop[c].LogProb, actualTop[c].LogProb);
                }
            }
        }
    }

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

    private static void AssertOutcomesEqual(SequenceOutcome[] expected, SequenceOutcome[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].FinishReason, actual[i].FinishReason);
            Assert.Equal(expected[i].CompletionTokens, actual[i].CompletionTokens);
            Assert.Equal(expected[i].GeneratedTokens, actual[i].GeneratedTokens); // token-for-token identical, alone or merged
        }
    }

    /// <summary>Runs each request on a scheduler of its own with nothing else live: the shape every pipeline's stream is compared against.</summary>
    private static async Task<SequenceOutcome[]> RunSolo(PipelineHarness harness, ScheduledRequest[] requests)
    {
        List<SequenceOutcome> outcomes = [];
        foreach (ScheduledRequest request in requests)
        {
            outcomes.Add(await harness.Scheduler.Submit(request, CancellationToken.None).Completion);
        }

        SequenceOutcome[] results = [.. outcomes];
        AssertCompleteRuns(results);
        return results;
    }

    /// <summary>Admits every request before awaiting any of them, so their steps overlap however the scheduler composes them.</summary>
    private static async Task<SequenceOutcome[]> RunTogether(PipelineHarness harness, ScheduledRequest[] requests)
    {
        PipelineHandle[] handles = [.. requests.Select(request => harness.Scheduler.Submit(request, CancellationToken.None))];
        SequenceOutcome[] outcomes = await Task.WhenAll(handles.Select(handle => handle.Completion));

        AssertCompleteRuns(outcomes);
        return outcomes;
    }

    private static void AssertCompleteRuns(SequenceOutcome[] outcomes)
    {
        foreach (SequenceOutcome outcome in outcomes)
        {
            Assert.Equal(FinishReason.Length, outcome.FinishReason);
            Assert.True(
                outcome.CompletionTokens == GenerationCeiling,
                $"a pipeline ended on {outcome.FinishReason} after {outcome.CompletionTokens} tokens");
        }
    }

    private static SamplingParams Greedy() => new() { Temperature = 0f };

    private static ScheduledRequest Request(int[] promptTokens, int maxTokens) => new()
    {
        PromptTokens = promptTokens,
        MaxTokens = maxTokens,
        Sampling = Greedy(),
        // A stop id outside this synthetic vocabulary keeps every run on the length path instead of ending on the model's real EOS.
        EosId = 999,
    };

    private static int[] Encode(GgufTokenizer tokenizer, string text) => [.. tokenizer.Encode(text)];

    private static int[] BuildLongPrompt(GgufTokenizer tokenizer)
    {
        List<int> ids = [];
        for (int round = 0; ids.Count < AnchorPromptTokens && round < 64; round++)
        {
            ids.AddRange(tokenizer.Encode("hello world !"));
        }

        Assert.True(ids.Count >= AnchorPromptTokens, $"long prompt built only {ids.Count} tokens; a many-chunk pipeline is required");
        return [.. ids];
    }

    private sealed class PipelineHarness : IDisposable
    {
        public GgufFile Gguf { get; }
        public GgufTokenizer Tokenizer { get; }
        public CpuBackend Backend { get; }
        public TransformerModel Model { get; }
        public BlockPool Pool { get; }
        public GenerationScheduler Scheduler { get; }

        public PipelineHarness(string modelPath, int capacityBlocks, int maxContextTokens, int maxBatchTokens, int? maxConcurrentRequests)
        {
            Gguf = GgufFile.Open(modelPath);
            Tokenizer = GgufTokenizer.FromGguf(Gguf);
            Backend = new CpuBackend(threads: 1);
            Model = TransformerModel.Open(Gguf, Backend, maxContextTokens);
            Pool = new BlockPool(
                blockTokens: 4,
                layerCount: Model.Desc.LayerCount,
                kvHeads: Model.Desc.KvHeadCount,
                headDim: Model.Desc.HeadDim,
                capacityBlocks: capacityBlocks,
                Backend);

            Scheduler = maxConcurrentRequests is int cap
                ? new GenerationScheduler(Model, Pool, maxBatchTokens, cache: null, maxConcurrentRequests: cap)
                : new GenerationScheduler(Model, Pool, maxBatchTokens);
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

/// <summary>
/// Advancement-slot accounting on its own, with no engine behind it: the gate hands out at most one slot per advancing pipeline and as
/// many of them as its cap allows, returns a slot when that pipeline finishes, lowers future acquisitions without taking anything back,
/// and bounds nothing under an uncapped cap. These are the progress decisions the scheduler makes about pipelines; capacity limits stay
/// where they were measured against real resources and never consult this accounting.
/// </summary>
[Trait("Category", "Scheduling")]
public sealed class PipelineConcurrencyGateTests
{
    [Fact]
    public void ProgressGate_HandsOutOnlyAsManySlotsAsTheCapAndReturnsThemOnFinish()
    {
        PipelineConcurrencyGate gate = new(maxConcurrentRequests: 2);

        Assert.True(gate.TryAcquireSlot());
        Assert.True(gate.TryAcquireSlot());
        Assert.False(gate.TryAcquireSlot()); // cap fully held: the next pipeline waits rather than advancing

        Assert.Equal(2, gate.HeldSlots);
        Assert.Equal(2, gate.PeakHeldSlots);

        gate.ReleaseSlot();
        Assert.Equal(1, gate.HeldSlots);
        Assert.True(gate.TryAcquireSlot()); // a returned slot is immediately available to the next pipeline at its boundary

        Assert.Equal(2, gate.PeakHeldSlots); // peak records concurrency ever reached, not just the current moment
    }

    [Fact]
    public void ProgressGate_LoweringTheCapStopsNewAdvancementWithoutPreemptingHeldWork()
    {
        PipelineConcurrencyGate gate = new(maxConcurrentRequests: 3);
        Assert.True(gate.TryAcquireSlot());
        Assert.True(gate.TryAcquireSlot());

        gate.ApplyMaxConcurrentRequests(1);

        Assert.Equal(1, gate.MaxConcurrentRequests);
        Assert.Equal(2, gate.HeldSlots); // work already advancing keeps its slot: the cap change stops new advancement only
    }

    [Fact]
    public void ProgressGate_UncappedHandsEveryPipelineASlot()
    {
        PipelineConcurrencyGate gate = new(PipelineConcurrencyGate.Uncapped);

        for (int i = 0; i < 64; i++)
        {
            Assert.True(gate.TryAcquireSlot());
        }

        Assert.Equal(64, gate.HeldSlots);
        Assert.Equal(64, gate.PeakHeldSlots);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    public void ProgressGate_RejectsCapsBelowOneAtOpenAndOnApply(int cap)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PipelineConcurrencyGate(cap));

        PipelineConcurrencyGate gate = new(maxConcurrentRequests: 2);
        Assert.Throws<ArgumentOutOfRangeException>(() => gate.ApplyMaxConcurrentRequests(cap));
    }

    [Fact]
    public void ProgressGate_ReleasingASlotThatWasNeverHeldIsAFault()
    {
        PipelineConcurrencyGate gate = new(maxConcurrentRequests: 1);
        Assert.True(gate.TryAcquireSlot());
        gate.ReleaseSlot();

        InvalidOperationException fault = Assert.Throws<InvalidOperationException>(() => gate.ReleaseSlot());
        Assert.Contains("no advancement slot is held", fault.Message, StringComparison.Ordinal);
    }
}
