using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// What the route-aware composition seam does to real stepping on a synthetic tiny model: merged steps stay token-for-token identical to
/// the same sequences stepped alone, sharing one plan and its whole CPU-only route merges them rather than splitting them, and a candidate
/// that cannot fit — past the segment's batch cap or out of quota headroom against it — is deferred with the blocking limit named in an
/// internal reason while every pipeline still advances to its token budget. A deferral never reaches admission: submissions behind one are
/// admitted, hold their slots and reservations, and ride a later step instead of being refused (a submission past the scheduler's own
/// concurrency cap is a separate refusal decided at admission — see the scheduling suite).
/// </summary>
[Trait("Category", "Scheduling")]
[Trait("Category", "Concurrency")]
[Collection("model-loaded")] // shares one serialized lane with the other suites that open engines on this box
public sealed class RouteStepCompositionTests : IDisposable
{
    private const int GenerationCeiling = 24;

    private readonly List<string> _tempFiles = [];

    [Fact]
    public async Task MergedSteps_RemainTokenIdenticalToSoloStepping_WhenRouteIdentityComposesThem()
    {
        string path = SyntheticLlamaGguf.Write("composition-identity");
        _tempFiles.Add(path);

        ScheduledRequest[] requests =
        [
            Request(Encode(path, "hello")),
            Request(Encode(path, "world !")),
            Request(Encode(path, "! hello world")),
            Request(Encode(path, "hello world ! hello world !")),
        ];

        using Harness solo = new(path);
        SequenceOutcome[] alone = await RunSolo(solo, requests);

        Assert.True(
            solo.Scheduler.Stats.DecodeSteps > 0 && solo.Scheduler.Stats.WindowSequenceSum == solo.Scheduler.Stats.DecodeSteps,
            "the baseline did not step one pipeline at a time — the comparison needs that shape");

        using Harness merged = new(path);
        SequenceOutcome[] together = await RunTogether(merged, requests);

        AssertOutcomesEqual(alone, together); // same tokens whether a pipeline rode a window of one or a shared merged window

        // The merge happened over route identity: this model's whole execution path is its single CPU span, and every pipeline advanced
        // under the very plan instance the scheduler composes on. Nothing was split per device because there was nothing to split on.
        Assert.Equal(0, merged.Scheduler.SteppingSegmentIndex);
        Assert.Equal(ModelRoute.CpuDeviceId, merged.Model.Plan.Route.Segments[merged.Scheduler.SteppingSegmentIndex].DeviceId);
        Assert.True(
            merged.Scheduler.PeakAdvancingPipelines >= 2 && merged.Scheduler.Stats.WindowSequenceSum > merged.Scheduler.Stats.DecodeSteps,
            $"the run never composed a merged step: {merged.Scheduler.Stats.WindowSequenceSum} pipeline steps across {merged.Scheduler.Stats.DecodeSteps} decode steps");

        // A quota-less CPU segment bounds merged sequences by its batch cap alone, and nothing here came near it.
        Assert.Equal(0, merged.Scheduler.DeferredCompositionCandidates);
        Assert.Empty(merged.Scheduler.LastCompositionRefusals);
    }

    [Fact]
    public async Task CandidatePastTheSegmentsBatchCap_IsDeferred_NamingThatCap_AndEveryPipelineStillCompletes()
    {
        string path = SyntheticLlamaGguf.Write("composition-batch-cap");
        _tempFiles.Add(path);

        const int batchCap = 2;
        ModelRoute capped = RouteWithBatchCap("composition-batch-cap", batchCap);

        ScheduledRequest[] requests =
        [
            Request(Encode(path, "hello")),
            Request(Encode(path, "world !")),
            Request(Encode(path, "! hello world")),
        ];

        using Harness baselineHarness = new(path);
        SequenceOutcome[] alone = await RunSolo(baselineHarness, requests);

        using Harness cappedHarness = new(path, route: capped);
        SequenceOutcome[] together = await RunTogether(cappedHarness, requests);

        AssertOutcomesEqual(alone, together); // a composition kept short changed no pipeline's tokens — each still sampled its own row

        Assert.True(
            cappedHarness.Scheduler.DeferredCompositionCandidates > 0,
            "no candidate was ever kept out of a merged step: the segment's batch cap did not bind this run");

        IReadOnlyList<DeferredStepCandidate> refusals = cappedHarness.Scheduler.LastCompositionRefusals;
        Assert.True(
            refusals.Count > 0,
            "the last boundary that kept a candidate out recorded nothing: the segment's batch cap did not bind this run");

        // Only one limit is declared on this route, so every refusal from that boundary names it and the segment it belongs to.
        Assert.All(refusals, refusal =>
        {
            Assert.Equal(CompositionBlock.BatchCap, refusal.Block);
            Assert.Contains($"already holds its batch cap of {batchCap}", refusal.Reason, StringComparison.Ordinal);
            Assert.Contains("segment 0 ('cpu' at 'Dram')", refusal.Reason, StringComparison.Ordinal);
        });

        // The cap bounded composition, not advancement: every window stayed inside it, and the progress gate still handed out a slot per
        // live pipeline — an over-cap candidate waits inside step composition rather than being refused admission.
        Assert.True(
            cappedHarness.Scheduler.Stats.WindowSequenceSum <= cappedHarness.Scheduler.Stats.DecodeSteps * batchCap,
            $"windows advanced {cappedHarness.Scheduler.Stats.WindowSequenceSum} pipelines across {cappedHarness.Scheduler.Stats.DecodeSteps} decode steps — past a segment batch cap of {batchCap}");

        Assert.True(
            cappedHarness.Scheduler.PeakAdvancingPipelines > batchCap,
            $"only {cappedHarness.Scheduler.PeakAdvancingPipelines} pipelines ever advanced at once: deferral read like an advancement refusal rather than a composition limit");

        Assert.Equal(batchCap, cappedHarness.Model.Plan.Route.Segments[cappedHarness.Scheduler.SteppingSegmentIndex].BatchCap);
    }

    [Fact]
    public async Task CandidateWithoutThreadQuotaHeadroom_IsDeferred_NamingTheQuota_AndNothingMergesOnThatSegment()
    {
        string path = SyntheticLlamaGguf.Write("composition-thread-quota");
        _tempFiles.Add(path);

        ModelRoute threadBound = RouteWithThreadQuota("composition-thread-quota", threads: 1);

        ScheduledRequest[] requests =
        [
            Request(Encode(path, "hello")),
            Request(Encode(path, "world !")),
            Request(Encode(path, "! hello world")),
        ];

        using Harness baselineHarness = new(path);
        SequenceOutcome[] alone = await RunSolo(baselineHarness, requests);

        using Harness boundHarness = new(path, route: threadBound);
        SequenceOutcome[] oneAtATime = await RunTogether(boundHarness, requests);

        AssertOutcomesEqual(alone, oneAtATime); // the same sequences stepped under a fully claimed quota still produce the same tokens

        Assert.True(
            boundHarness.Scheduler.DeferredCompositionCandidates > 0,
            "no candidate was kept out of a merged step: the segment's thread quota did not bind this run");

        Assert.Contains(
            boundHarness.Scheduler.LastCompositionRefusals,
            refusal => refusal.Block == CompositionBlock.ThreadQuota && refusal.Reason.Contains("'1'-thread quota is fully claimed", StringComparison.Ordinal));

        // One sequence per window here: the quota claim was exhausted by the first pipeline every time composition offered more.
        Assert.Equal(boundHarness.Scheduler.Stats.DecodeSteps, boundHarness.Scheduler.Stats.WindowSequenceSum);
    }

    [Fact]
    public async Task LiveThreadQuotaReResolution_RetargetsFutureCompositionHeadroom_WithoutPreemptingAdmittedWork()
    {
        string path = SyntheticLlamaGguf.Write("composition-live-quota");
        _tempFiles.Add(path);

        ModelRoute quotaBound = RouteWithThreadQuota("composition-live-quota", threads: 4);

        ScheduledRequest[] soloRequests =
        [
            Request(Encode(path, "hello")),
            Request(Encode(path, "world !")),
            Request(Encode(path, "! hello world")),
        ];

        // The live run hooks every pipeline so their slots are certainly still held when the quota is re-resolved mid-serve on a busy box.
        ScheduledRequest[] liveRequests =
        [
            Request(Encode(path, "hello"), HoldingPipelineHook()),
            Request(Encode(path, "world !"), HoldingPipelineHook()),
            Request(Encode(path, "! hello world"), HoldingPipelineHook()),
        ];

        using Harness baselineHarness = new(path);
        SequenceOutcome[] alone = await RunSolo(baselineHarness, soloRequests);

        using Harness liveHarness = new(path, route: quotaBound);
        PipelineHandle[] handles = [.. liveRequests.Select(request => liveHarness.Scheduler.Submit(request, CancellationToken.None))];

        // The load-resolved quota of 4 has headroom for all three pipelines: the early merged steps kept nothing out.
        await WaitForDecodeStepsAsync(liveHarness.Scheduler, minimum: 2);
        Assert.Equal(0, liveHarness.Scheduler.DeferredCompositionCandidates);

        // Lowering the stepping segment's worker-thread quota mid-serve re-resolves headroom for future compositions only — pipelines already
        // admitted keep their slots and ride on. From here a merged window fits one sequence, so later boundaries defer what will not fit.
        liveHarness.Scheduler.ApplySteppingThreadQuota(1);

        SequenceOutcome[] together = await Task.WhenAll(handles.Select(handle => handle.Completion));

        AssertCompleteRuns(together); // nothing was preempted: every admitted pipeline still ran to its token budget
        AssertOutcomesEqual(alone, together); // and a retargeted quota moved no sequence's tokens

        Assert.True(
            liveHarness.Scheduler.DeferredCompositionCandidates > 0,
            "no candidate was kept out after the lowered quota: composition never read the re-resolved headroom");

        Assert.Contains(
            liveHarness.Scheduler.LastCompositionRefusals,
            refusal => refusal.Block == CompositionBlock.ThreadQuota && refusal.Reason.Contains("'1'-thread quota is fully claimed", StringComparison.Ordinal));
    }

    private static async Task WaitForDecodeStepsAsync(GenerationScheduler scheduler, long minimum)
    {
        long deadline = Environment.TickCount64 + 25_000;
        while (scheduler.Stats.DecodeSteps < minimum)
        {
            if (Environment.TickCount64 >= deadline)
            {
                Assert.Fail($"the run never reached {minimum} decode steps — the composition window to re-resolve inside never opened");
            }

            await Task.Delay(10);
        }
    }

    // ---------------------------------------------------------------- helpers

    private static SamplingParams Greedy() => new() { Temperature = 0f };

    private static ScheduledRequest Request(int[] promptTokens, Action<int>? onTokenId = null) => new()
    {
        PromptTokens = promptTokens,
        MaxTokens = GenerationCeiling,
        Sampling = Greedy(),
        // A stop id outside this synthetic vocabulary keeps every run on the length path instead of ending on the model's real EOS.
        EosId = 999,
        OnTokenId = onTokenId,
    };

    /// <summary>Keeps a pipeline live for a measurable window: its streaming hook parks the scheduler thread briefly at every sampled token, so mid-serve changes land while its slots are certainly still held on a busy box.</summary>
    private static Action<int> HoldingPipelineHook() => _ => Thread.Sleep(15);

    private static int[] Encode(string modelPath, string text)
    {
        using GgufFile gguf = GgufFile.Open(modelPath);
        return [.. GgufTokenizer.FromGguf(gguf).Encode(text)];
    }

    private static ModelRoute RouteWithBatchCap(string modelName, int batchCap) => ModelRoutes.Resolve(new RouteRequest
    {
        ModelName = modelName,
        Stages = [new(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
        AvailableDevices = [ModelRoute.CpuDeviceId],
        BatchCap = batchCap,
    });

    private static ModelRoute RouteWithThreadQuota(string modelName, int threads) => ModelRoutes.Resolve(new RouteRequest
    {
        ModelName = modelName,
        Stages = [new(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
        AvailableDevices = [ModelRoute.CpuDeviceId],
        ThreadQuota = threads,
    });

    /// <summary>Runs each request on a scheduler of its own with nothing else live: the shape every pipeline's stream is compared against.</summary>
    private static async Task<SequenceOutcome[]> RunSolo(Harness harness, ScheduledRequest[] requests)
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

    /// <summary>Admits every request before awaiting any of them, so their steps overlap however composition composes them.</summary>
    private static async Task<SequenceOutcome[]> RunTogether(Harness harness, ScheduledRequest[] requests)
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

    private static void AssertOutcomesEqual(SequenceOutcome[] expected, SequenceOutcome[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].FinishReason, actual[i].FinishReason);
            Assert.Equal(expected[i].CompletionTokens, actual[i].CompletionTokens);
            Assert.Equal(expected[i].GeneratedTokens, actual[i].GeneratedTokens); // token-for-token identical, composed or alone
        }
    }

    private sealed class Harness : IDisposable
    {
        public GgufFile Gguf { get; }
        public CpuBackend Backend { get; }
        public TransformerModel Model { get; }
        public BlockPool Pool { get; }
        public GenerationScheduler Scheduler { get; }

        /// <param name="modelPath">The synthetic model every harness in one test shares, so both runs forward identical weights.</param>
        /// <param name="route">A resolved route landed on the plan at open — what composition reads as this model's execution path. Null keeps the CPU-only span the file resolves to by itself.</param>
        public Harness(string modelPath, ModelRoute? route = null)
        {
            Gguf = GgufFile.Open(modelPath);
            Backend = new CpuBackend(threads: 1);
            Model = route is null
                ? TransformerModel.Open(Gguf, Backend, maxContextTokens: 64)
                : TransformerModel.Open(Gguf, Backend, maxContextTokens: 64, logLine: null, PrefillKernel.GemV, route);

            Pool = new BlockPool(
                blockTokens: 4,
                layerCount: Model.Desc.LayerCount,
                kvHeads: Model.Desc.KvHeadCount,
                headDim: Model.Desc.HeadDim,
                capacityBlocks: 64,
                Backend);

            Scheduler = new GenerationScheduler(Model, Pool, maxBatchTokens: 2);
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
}
