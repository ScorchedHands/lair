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
/// The device window as a scheduler consumes it: a model registered with a compute share advances its own scheduler with every merged
/// window bounded to its share of the window's step slots, the candidates past the grant defer naming the share, the model, and the device
/// segment the window bounds, and every deferred pipeline still runs to its token budget — a deferral is a launch decision, never a
/// preemption. Two models resident on one window each hold their own share without bounding the other, token streams stay identical to an
/// unbounded run, and a live re-share widens later boundaries from the next grant on while sequences already launched keep running.
/// </summary>
[Trait("Category", "Scheduling")]
[Collection("model-loaded")]
public sealed class DeviceFairShareSchedulerTests : IDisposable
{
    private const int GenerationCeiling = 12;

    private readonly List<string> _tempFiles = [];

    [Fact]
    public async Task ComputeShare_BoundsEveryWindowToItsCap_AndDefersTheExcess_NamingShareModelAndSegment()
    {
        string path = SyntheticLlamaGguf.Write("fair-share-bounds");
        _tempFiles.Add(path);

        DeviceCoScheduler seam = new(ModelRoute.CpuDeviceId);
        seam.RegisterModel("fair-share-model", 25); // cap 2 of the window's 8 step slots

        using FairShareHarness harness = new(path, "fair-share-model", seam, capacityBlocks: 48);

        PipelineHandle[] handles = [.. Enumerable.Range(0, 6).Select(id =>
            harness.Scheduler.Submit(Request(Encode(harness.Tokenizer, id % 2 == 0 ? "hello world" : "world !")), CancellationToken.None))];

        SequenceOutcome[] outcomes = await Task.WhenAll(handles.Select(handle => handle.Completion));

        AssertCompleteRuns(outcomes); // every deferred pipeline still ran to its token budget — deferral launched nothing, it preempted nothing

        GenerationScheduler scheduler = harness.Scheduler;
        Assert.True(
            scheduler.Stats.WindowSequenceSum <= scheduler.Stats.DecodeSteps * 2,
            $"windows advanced {scheduler.Stats.WindowSequenceSum} sequences across {scheduler.Stats.DecodeSteps} steps — past a 25% share of the 8-slot window");
        Assert.True(scheduler.Stats.WindowSequenceSum > scheduler.Stats.DecodeSteps, "the share serialized every step instead of bounding it");

        Assert.True(scheduler.DeferredCompositionCandidates > 0, "a 25% share against six live pipelines deferred nothing at any boundary");
        Assert.Contains(
            scheduler.LastCompositionRefusals,
            refusal =>
                refusal.Block == CompositionBlock.DeviceShare
                && refusal.Reason.Contains("compute_share 25%", StringComparison.Ordinal)
                && refusal.Reason.Contains("fair-share-model", StringComparison.Ordinal)
                && refusal.Reason.Contains(ModelRoute.CpuDeviceId, StringComparison.Ordinal));

        Assert.Equal(0, seam.OutstandingStepSlots("fair-share-model")); // every launched slot came back as its forward returned
    }

    [Fact]
    public async Task ComputeShare_LeavesTokenStreamsIdenticalToTheUnboundedRun_EvenAtTheOneSlotFloor()
    {
        string path = SyntheticLlamaGguf.Write("fair-share-identity");
        _tempFiles.Add(path);

        using FairShareHarness unbounded = new(path, "fair-share-identity", coScheduler: null, capacityBlocks: 48);
        SequenceOutcome[] baseline = await RunTogether(unbounded, BuildRequests(unbounded.Tokenizer));

        Assert.True(
            unbounded.Scheduler.Stats.WindowSequenceSum > unbounded.Scheduler.Stats.DecodeSteps,
            "the unbounded baseline never merged pipelines — the comparison needs that shape");

        DeviceCoScheduler seam = new(ModelRoute.CpuDeviceId);
        seam.RegisterModel("fair-share-identity", 1); // the floor: one slot of the window, every excess candidate deferred at every boundary

        using FairShareHarness serialized = new(path, "fair-share-identity", seam, capacityBlocks: 48);
        SequenceOutcome[] bounded = await RunTogether(serialized, BuildRequests(serialized.Tokenizer));

        AssertOutcomesEqual(baseline, bounded); // the window changed when device work launched, not a single sampled token

        Assert.True(
            serialized.Scheduler.Stats.WindowSequenceSum == serialized.Scheduler.Stats.DecodeSteps,
            $"a 1% share advanced {serialized.Scheduler.Stats.WindowSequenceSum} sequences across {serialized.Scheduler.Stats.DecodeSteps} steps — the floor should serialize windows, not bound them wider");
        Assert.True(serialized.Scheduler.DeferredCompositionCandidates > 0);
    }

    [Fact]
    public async Task TwoModelsOnOneWindow_EachHoldsItsOwnShare_WithoutBoundingTheOther()
    {
        string path = SyntheticLlamaGguf.Write("fair-share-pair");
        _tempFiles.Add(path);

        DeviceCoScheduler seam = new(ModelRoute.CpuDeviceId);
        seam.RegisterModel("small-share", 25); // cap 2
        seam.RegisterModel("large-share", 75); // cap 6

        using FairShareHarness small = new(path, "small-share", seam, capacityBlocks: 48);
        using FairShareHarness large = new(path, "large-share", seam, capacityBlocks: 48);

        PipelineHandle[] smallHandles = [.. Enumerable.Range(0, 6).Select(id =>
            small.Scheduler.Submit(Request(Encode(small.Tokenizer, id % 2 == 0 ? "hello world" : "world !")), CancellationToken.None))];
        PipelineHandle[] largeHandles = [.. Enumerable.Range(0, 6).Select(id =>
            large.Scheduler.Submit(Request(Encode(large.Tokenizer, id % 2 == 0 ? "hello world" : "world !")), CancellationToken.None))];

        SequenceOutcome[] smallOutcomes = await Task.WhenAll(smallHandles.Select(handle => handle.Completion));
        SequenceOutcome[] largeOutcomes = await Task.WhenAll(largeHandles.Select(handle => handle.Completion));

        AssertCompleteRuns(smallOutcomes);
        AssertCompleteRuns(largeOutcomes); // both models ran to their budgets while sharing one window

        Assert.True(
            small.Scheduler.Stats.WindowSequenceSum <= small.Scheduler.Stats.DecodeSteps * 2,
            $"the 25% model advanced {small.Scheduler.Stats.WindowSequenceSum} sequences across {small.Scheduler.Stats.DecodeSteps} steps — past its cap of 2");
        Assert.True(small.Scheduler.DeferredCompositionCandidates > 0);

        Assert.Equal(0, large.Scheduler.DeferredCompositionCandidates); // six candidates fit the 75% model's cap of 6: the other model's occupancy bounded nothing here
        Assert.True(
            large.Scheduler.Stats.WindowSequenceSum > large.Scheduler.Stats.DecodeSteps,
            "the 75% model never merged a window — its own cap should not have serialized it");

        Assert.Equal(0, seam.OutstandingStepSlots("small-share"));
        Assert.Equal(0, seam.OutstandingStepSlots("large-share"));
    }

    [Fact]
    public async Task LiveComputeShareReResolution_WidensLaterBoundaries_WithoutDisturbingAdmittedSequences()
    {
        string path = SyntheticLlamaGguf.Write("fair-share-live");
        _tempFiles.Add(path);

        // Baseline: a 1% share held in force for the whole run serializes every window.
        DeviceCoScheduler held = new(ModelRoute.CpuDeviceId);
        held.RegisterModel("fair-share-live", 1);
        using FairShareHarness baseline = new(path, "fair-share-live", held, capacityBlocks: 48);
        SequenceOutcome[] baselineOutcomes = await RunTogether(baseline, BuildRequests(baseline.Tokenizer));
        Assert.True(
            baseline.Scheduler.Stats.WindowSequenceSum == baseline.Scheduler.Stats.DecodeSteps,
            "the baseline share did not serialize every window — the comparison needs that shape");

        // Live: the same 1% share re-resolved to a full share while pipelines are mid-flight.
        DeviceCoScheduler live = new(ModelRoute.CpuDeviceId);
        live.RegisterModel("fair-share-live", 1);
        using FairShareHarness harness = new(path, "fair-share-live", live, capacityBlocks: 48);

        ScheduledRequest[] requests = [.. BuildRequests(harness.Tokenizer).Select(request =>
            request with { OnTokenId = _ => Thread.Sleep(15) })]; // the hooks keep every pipeline live across many boundaries while the re-share lands

        PipelineHandle[] handles = [.. requests.Select(request => harness.Scheduler.Submit(request, CancellationToken.None))];

        Thread.Sleep(120); // some boundaries composed under the 1-slot cap before this lands
        harness.Scheduler.ApplyComputeShare(100);
        Assert.Equal(100, live.ShareFor("fair-share-live"));

        SequenceOutcome[] outcomes = await Task.WhenAll(handles.Select(handle => handle.Completion));

        AssertOutcomesEqual(baselineOutcomes, outcomes); // the re-share moved no token any sequence had already sampled
        Assert.True(
            harness.Scheduler.Stats.WindowSequenceSum > harness.Scheduler.Stats.DecodeSteps,
            "a live raise to a full share never widened a later window — the re-share took effect nowhere");
    }

    // ---------------------------------------------------------------- helpers

    private static ScheduledRequest Request(int[] promptTokens) => new()
    {
        PromptTokens = promptTokens,
        MaxTokens = GenerationCeiling,
        Sampling = new SamplingParams { Temperature = 0f },
        // A stop id outside this synthetic vocabulary keeps every run on the length path instead of ending on the model's real EOS.
        EosId = 999,
    };

    private static ScheduledRequest[] BuildRequests(GgufTokenizer tokenizer) =>
    [
        Request(Encode(tokenizer, "hello")),
        Request(Encode(tokenizer, "world !")),
        Request(Encode(tokenizer, "! hello world")),
        Request(Encode(tokenizer, "hello world ! hello world ! hello")),
    ];

    private static int[] Encode(GgufTokenizer tokenizer, string text) => [.. tokenizer.Encode(text)];

    private static async Task<SequenceOutcome[]> RunTogether(FairShareHarness harness, ScheduledRequest[] requests)
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
            Assert.Equal(expected[i].GeneratedTokens, actual[i].GeneratedTokens); // token-for-token identical, window-bounded or not
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

    private sealed class FairShareHarness : IDisposable
    {
        public GgufFile Gguf { get; }
        public GgufTokenizer Tokenizer { get; }
        public CpuBackend Backend { get; }
        public TransformerModel Model { get; }
        public BlockPool Pool { get; }
        public GenerationScheduler Scheduler { get; }

        public FairShareHarness(string modelPath, string modelName, DeviceCoScheduler? coScheduler, int capacityBlocks)
        {
            Gguf = GgufFile.Open(modelPath);
            Tokenizer = GgufTokenizer.FromGguf(Gguf);
            Backend = new CpuBackend(threads: 1);
            Model = TransformerModel.Open(Gguf, Backend, maxContextTokens: 64);
            Pool = new BlockPool(
                blockTokens: 16,
                layerCount: Model.Desc.LayerCount,
                kvHeads: Model.Desc.KvHeadCount,
                headDim: Model.Desc.HeadDim,
                capacityBlocks: capacityBlocks,
                Backend);
            Scheduler = new GenerationScheduler(
                Model,
                Pool,
                maxBatchTokens: 8,
                cache: null,
                modelName: modelName,
                coScheduler: coScheduler);
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
