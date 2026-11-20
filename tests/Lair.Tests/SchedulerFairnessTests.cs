using System.Diagnostics;

using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Http;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;

namespace Lair.Tests;

/// <summary>The fairness class runs alone, never concurrently with any other collection: its two-model loads hold every worker of a
/// partitioned posture busy at once, and the suite's wall-clock-guarded cases must not be starved by them.</summary>
[CollectionDefinition("fairness-serialized", DisableParallelization = true)]
public sealed class SchedulerFairnessCollection;

/// <summary>
/// Scheduling fairness across two served models: step composition bounds each model inside its own granted core set, so a model that
/// saturates the share it was seated for defers only its own candidates and never reaches into another model's seat. Admission stays
/// first-come — an over-cap submission is refused naming the model and its quota rather than queued behind what runs or allowed to displace
/// it — one model's pipelines advance in admission order at every boundary, and a claim that finds nothing left runs on what remains while
/// every seated share keeps exactly what it was granted. Nothing here preempts an admitted sequence: deferral waits, refusals turn new work
/// away, and running sequences finish whole. Every gate below reads scheduler/ledger counters rather than elapsed time.
/// </summary>
[Trait("Category", "Fairness")]
[Collection("fairness-serialized")]
public sealed class SchedulerFairnessTests : IDisposable
{
    // The bounded step budget this gate holds the untouched model's sequence to: one scheduler boundary per generated token. A candidate kept
    // out of a merged window would push it past that, so zero deferrals plus exact per-boundary advance is what "not starved past its budget"
    // reads as on these counters — no wall-clock threshold anywhere in this class.
    private const int QuietModelTokenCeiling = 64;

    // Four sequences against a two-core partition: from the third candidate of every boundary onward this model is over its own granted set,
    // which is what makes its saturation visible on its deferral counters instead of on timing.
    private const int SaturatingModelSequences = 4;
    private const int SaturatingModelTokenCeiling = 64;

    private readonly List<string> _tempFiles = [];

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

    [Fact]
    public async Task ModelSaturatingItsGrantedCoreSet_DefersOnlyItsOwnCandidates_KeepsTheOtherModelAdvancingEveryBoundary_AndNeitherSeatMoves()
    {
        ComputeCapacityLedger host = new(SyntheticPartitionedPosture()); // one four-processor posture, reported as two whole partitions.

        using ModelPair pair = new("fairness-partition", threads: 2);

        // What a host seating these two models records for them, in open order against the same posture their engines ran on.
        ComputeGrant seatedFirst = host.RegisterClaim(pair.ModelAName, requestedCores: 2);
        ComputeGrant seatedSecond = host.RegisterClaim(pair.ModelBName, requestedCores: 2);

        Assert.Equal(0, host.RemainingCores); // both claims hold a partition whole: nothing oversubscribed itself.
        Assert.False(seatedFirst.SharesHeldCores);
        Assert.False(seatedSecond.SharesHeldCores);

        PipelineHandle quiet = pair.SchedulerA.Submit(Request([1, 2, 3], QuietModelTokenCeiling), CancellationToken.None);

        List<PipelineHandle> saturatingHandles = [];
        for (int index = 0; index < SaturatingModelSequences; index++)
        {
            saturatingHandles.Add(pair.SchedulerB.Submit(Request([1, 2], SaturatingModelTokenCeiling), CancellationToken.None));
        }

        SequenceOutcome quietOutcome = await quiet.Completion;
        SequenceOutcome[] saturatingOutcomes = await Task.WhenAll([.. saturatingHandles.Select(handle => handle.Completion)]);

        // All four held admission slots at once, so this model really did run more sequences than its granted set can advance together.
        Assert.Equal(SaturatingModelSequences, pair.SchedulerB.PeakAdvancingPipelines);

        // Its over-cap candidates waited rather than reaching anywhere else: a boundary that could not fit one named its own pinned core set.
        Assert.True(pair.SchedulerB.DeferredCompositionCandidates > 0, "no candidate was kept out of a merged step — this model never saturated its granted set");
        Assert.Contains(
            pair.SchedulerB.LastCompositionRefusals,
            refusal => refusal.Block == CompositionBlock.CoreSetQuota && refusal.Reason.Contains("core set [2, 3]", StringComparison.Ordinal));

        // Deferral cost those sequences later boundaries only: every one of them still advanced to its ceiling.
        Assert.All(saturatingOutcomes, outcome =>
        {
            Assert.Equal(FinishReason.Length, outcome.FinishReason);
            Assert.Equal(SaturatingModelTokenCeiling, outcome.CompletionTokens);
        });

        // The other model was never kept out of a window and advanced exactly one sequence at every boundary it ran — its whole run inside the
        // one-step-per-token budget this gate holds, while the partition beside it stayed saturated.
        Assert.Equal(0, pair.SchedulerA.DeferredCompositionCandidates); // nothing of this model was ever kept out of a window...
        Assert.Equal(FinishReason.Length, quietOutcome.FinishReason);
        Assert.Equal(QuietModelTokenCeiling, quietOutcome.CompletionTokens);
        Assert.Equal(pair.SchedulerA.Stats.DecodeSteps, pair.SchedulerA.Stats.WindowSequenceSum); // ...and every boundary it ran advanced exactly this one sequence.

        int quietBoundaries = (int)pair.SchedulerA.Stats.WindowSequenceSum;
        Assert.True(
            quietBoundaries >= QuietModelTokenCeiling / 2,
            $"the untouched model crossed {quietBoundaries} boundaries for {quietOutcome.CompletionTokens} tokens — starved past its bounded step budget");

        // Neither seat moved across the whole two-model load: each claim keeps what it was granted and no share reports overlap.
        ComputeGrant firstAfter = host.GrantFor(pair.ModelAName)!;
        ComputeGrant secondAfter = host.GrantFor(pair.ModelBName)!;
        Assert.Equal(2, firstAfter.ExclusiveCores);
        Assert.False(firstAfter.SharesHeldCores);
        Assert.Equal(2, secondAfter.ExclusiveCores);
        Assert.False(secondAfter.SharesHeldCores);
        Assert.Equal(0, host.RemainingCores);
    }

    [Fact]
    public async Task ModelAddedWhenEveryPartitionIsSeated_RunsOnWhatRemains_WithoutMovingASeatedShare_AndServingKeepsRunning()
    {
        string alphaPath = SyntheticModel("fairness-seat-alpha");
        string betaPath = SyntheticModel("fairness-seat-beta");

        ComputeCapacityLedger compute = new(SyntheticPartitionedPosture()); // the posture every claim below seats against.

        LairConfig start = ConfigWith([PinnedModel("alpha", alphaPath, "0-1"), PinnedModel("beta", betaPath, "2-3")], threads: 2);
        using ModelRegistry registry = ModelRegistry.Open(start, logger: null, transport: null, ledger: null, compute);

        Assert.Equal(["alpha", "beta"], registry.AdvertisedIds());
        Assert.Equal(0, compute.RemainingCores); // both partitions are held before anything is added.
        ComputeGrant alphaSeat = compute.GrantFor("alpha")!;
        ComputeGrant betaSeat = compute.GrantFor("beta")!;

        using ModelRegistry.Lease runningLease = registry.BeginRequest("alpha");
        Task<GenerationResult> inFlight = Task.Run(() => Generate(runningLease.Engine, maxTokens: 48)); // admitted work a later claim must not disturb.

        IReadOnlyList<FieldResult> results = registry.ApplyConfiguration(ConfigWith(
            [PinnedModel("alpha", alphaPath, "0-1"), PinnedModel("beta", betaPath, "2-3"), UnpinnedModel("gamma", SyntheticModel("fairness-seat-gamma"))],
            threads: 2));

        FieldResult addition = Assert.Single(results, result => result.Change.Field == "models[2]");
        Assert.Equal(ConfigApplyOutcome.Applied, addition.Outcome); // the addition is accepted into the posture and seats behind what stands there.

        WaitUntil(() => registry.AdvertisedIds().Contains("gamma"), "'gamma' to advertise after its background open");

        ComputeGrant added = compute.GrantFor("gamma")!;
        Assert.Equal(2, added.RequestedCores);   // it asked for the engine's whole worker share against a posture whose partitions are held...
        Assert.Equal(1, added.GrantedCores);     // ...and runs on the one worker its predecessors left.
        Assert.Equal(0, added.ExclusiveCores);
        Assert.True(added.SharesHeldCores);      // what could not be seated as its own is reported overlap — never a displacement of a seated claim.

        ComputeGrant alphaAfter = compute.GrantFor("alpha")!;
        ComputeGrant betaAfter = compute.GrantFor("beta")!;
        Assert.Equal(alphaSeat.NodeAllocations, alphaAfter.NodeAllocations); // the pinned seats are untouched by the addition...
        Assert.Equal(betaSeat.NodeAllocations, betaAfter.NodeAllocations);   // ...including their per-node shares.
        Assert.False(alphaAfter.SharesHeldCores);
        Assert.False(betaAfter.SharesHeldCores);

        GenerationService gammaEngine = registry.AdvertisedEngines().Single(engine => engine.Model.Id == "gamma");
        Assert.Equal(1, gammaEngine.EffectiveWorkerCount); // participation capped inside what remained.

        GenerationResult completed = await inFlight; // the admitted sequence finished whole: a later claim never preempted it.
        Assert.True(completed.CompletionTokens > 0);

        using ModelRegistry.Lease seatedPair = registry.BeginRequest("beta");
        Assert.True(Generate(seatedPair.Engine, maxTokens: 8).CompletionTokens > 0); // every model already served kept serving untouched.

        using ModelRegistry.Lease addedModel = registry.BeginRequest("gamma");
        Assert.True(Generate(addedModel.Engine, maxTokens: 8).CompletionTokens > 0); // and the partial share still runs the model it was seated for.
    }

    [Fact]
    public async Task OverCapSubmissionIsRefusedInsteadOfDisplacingActiveWork_WhileTheOtherServedModelKeepsAdvancing_AndAdmissionResumesWhenASlotFrees()
    {
        string alphaPath = SyntheticModel("fairness-cap-alpha");
        string betaPath = SyntheticModel("fairness-cap-beta");

        ComputeCapacityLedger compute = new(SyntheticPartitionedPosture());

        LairConfig start = ConfigWith([LongSequenceCappedPinnedModel("alpha", alphaPath, "0-1"), PinnedModel("beta", betaPath, "2-3")], threads: 2);
        using ModelRegistry registry = ModelRegistry.Open(start, logger: null, transport: null, ledger: null, compute);

        GenerationService alphaEngine = registry.AdvertisedEngines().Single(engine => engine.Model.Id == "alpha");
        GenerationService betaEngine = registry.AdvertisedEngines().Single(engine => engine.Model.Id == "beta");

        using ModelRegistry.Lease heldSlot = registry.BeginRequest("alpha");

        // A long budget on a pool that can seat it: this sequence stays admitted and advancing for the whole window below, so every answer read here
        // is an answer about a cap that was provably full at that instant.
        Task<GenerationResult> admitted = Task.Run(() => Generate(heldSlot.Engine, maxTokens: 4096));

        WaitUntil(() => alphaEngine.ServedScheduler.AdvancingPipelineCount >= 1 && alphaEngine.ServedScheduler.Stats.DecodeTokensAdvanced > 0, "the first sequence to hold its slot and start advancing"); // from here the cap is provably full at every instant below.

        EngineBusyException refusal;
        using (ModelRegistry.Lease overCapLease = registry.BeginRequest("alpha"))
        {
            // The cap is already fully held by the admitted sequence, so this submission answers at admission rather than queueing behind it.
            refusal = Assert.Throws<EngineBusyException>(() => Generate(overCapLease.Engine, maxTokens: 16));
        }

        Assert.Equal("engine_overloaded", refusal.Code); // a bounded-overload refusal naming what held the cap — never displacement of running work.
        Assert.Contains("'alpha'", refusal.Message);
        Assert.Contains("max_concurrent_requests", refusal.Message);

        using ModelRegistry.Lease otherModel = registry.BeginRequest("beta"); // the other model admits and advances beside that cap untouched.
        GenerationResult otherOutcome = Generate(otherModel.Engine, maxTokens: 64);
        Assert.True(otherOutcome.CompletionTokens > 0);

        Assert.Equal(1, alphaEngine.ServedScheduler.MaxConcurrentRequests); // the cap this model serves under is its declared one.
        Assert.Equal(1, alphaEngine.ServedScheduler.PeakAdvancingPipelines); // and it bounded concurrency rather than queueing or displacing anything.

        GenerationResult firstFinished = await admitted; // and the sequence that held the cap ran whole on its own slot.
        Assert.True(firstFinished.CompletionTokens > 0);

        using (ModelRegistry.Lease refilled = registry.BeginRequest("alpha")) // the refusal turned room away, not state: it admits normally once a slot frees.
        {
            GenerationResult refilledOutcome = Generate(refilled.Engine, maxTokens: 8);
            Assert.True(refilledOutcome.CompletionTokens > 0);
        }

        ComputeGrant alphaAfter = compute.GrantFor("alpha")!;
        ComputeGrant betaAfter = compute.GrantFor("beta")!;
        Assert.Equal(2, alphaAfter.ExclusiveCores); // neither model's seated share moved across the whole admission posture.
        Assert.Equal(2, betaAfter.ExclusiveCores);
        Assert.False(alphaAfter.SharesHeldCores);
        Assert.False(betaAfter.SharesHeldCores);

        Assert.Equal(2, betaEngine.EffectiveWorkerCount); // the model beside that cap kept its whole seated participation across it.
    }

    [Fact]
    public async Task PipelinesOfOneModelAdvanceInAdmissionOrder_WhileItsOverCapCandidatesWait_AndTheOtherModelRunsOnUnchanged()
    {
        using ModelPair pair = new("fairness-fifo", threads: 2);

        // One worker of headroom per merged window on this model: every boundary advances one candidate, so the order candidates ride in is their
        // admission order. Hooks fire only on this scheduler's own thread, in step-window order, so what they record here is a total arrival order
        // — and first-arrival-per-sequence is exactly the sequence the boundaries advanced them in.
        pair.SchedulerA.ApplySteppingThreadQuota(1);

        List<int> tokenArrivals = [];

        PipelineHandle[] admitted =
        [
            pair.SchedulerA.Submit(
                Request([1, 2], QuietModelTokenCeiling) with { OnTokenId = _ => Add(tokenArrivals, 0) },
                CancellationToken.None),
            pair.SchedulerA.Submit(
                Request([1, 2, 3], QuietModelTokenCeiling) with { OnTokenId = _ => Add(tokenArrivals, 1) },
                CancellationToken.None),
            pair.SchedulerA.Submit(
                Request([1, 2, 3, 4], QuietModelTokenCeiling) with { OnTokenId = _ => Add(tokenArrivals, 2) },
                CancellationToken.None),
        ];

        PipelineHandle other = pair.SchedulerB.Submit(Request([1, 2], QuietModelTokenCeiling), CancellationToken.None);

        SequenceOutcome[] outcomes = await Task.WhenAll([.. admitted.Select(handle => handle.Completion)]);
        SequenceOutcome otherOutcome = await other.Completion;

        Assert.Equal(3, pair.SchedulerA.PeakAdvancingPipelines); // all three held admission slots while only one advanced per boundary.
        Assert.True(pair.SchedulerA.DeferredCompositionCandidates > 0, "no candidate was kept out — this model's headroom never bound a step");
        Assert.Contains(
            pair.SchedulerA.LastCompositionRefusals,
            refusal => refusal.Block == CompositionBlock.ThreadQuota && refusal.Reason.Contains("1'-thread quota", StringComparison.Ordinal));

        // The first token of each sequence arrived in submission order: the boundary that advanced one left every later candidate waiting at a
        // deferral rather than promoting it past what was admitted before it.
        Assert.Equal(new[] { 0, 1, 2 }, tokenArrivals.Distinct().ToArray()); // first arrival per sequence is its admission order.

        // Waiting cost those pipelines later boundaries only: none was displaced or failed.
        Assert.All(outcomes, outcome =>
        {
            Assert.Equal(FinishReason.Length, outcome.FinishReason);
            Assert.Equal(QuietModelTokenCeiling, outcome.CompletionTokens);
        });

        // The other model ran on its own partition untouched: no deferral of its kind anywhere in its scheduler.
        Assert.Equal(0, pair.SchedulerB.DeferredCompositionCandidates);
        Assert.Equal(FinishReason.Length, otherOutcome.FinishReason);
        Assert.Equal(QuietModelTokenCeiling, otherOutcome.CompletionTokens);
    }

    private static ScheduledRequest Request(int[] promptTokens, int maxTokens) => new()
    {
        PromptTokens = promptTokens,
        MaxTokens = maxTokens,
        Sampling = new SamplingParams { Temperature = 0f }, // greedy: scheduling shape is the only thing these gates could move.
        EosId = 999, // a stop id outside this synthetic vocabulary keeps every run on the length path instead of ending on the model's real EOS.
    };

    // Token arrival bookkeeping for the ordering gate: hooks fire on one scheduler thread in step-window order, and this guard only exists so the
    // list stays safe however a boundary happens to be scheduled against it.
    private static void Add(List<int> arrivals, int index)
    {
        lock (arrivals) arrivals.Add(index);
    }

    private static GenerationResult Generate(GenerationService engine, string prompt = "hello world", int maxTokens = 8) =>
        engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = prompt,
            MaxTokens = maxTokens,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult();

    private static ModelConfig PinnedModel(string name, string path, string coreSet) => new()
    {
        Name = name,
        Path = path,
        MaxContextTokens = 512,
        KvBlocks = 32,
        Placement = new PlacementConfig { CoreSet = ParseCoreRanges(coreSet) },
    };

    // The capped model of the admission-posture gate needs room for one long sequence plus its prompt: a slot held by work that really runs is what makes "refused rather than queued" observable at an instant.
    private static ModelConfig LongSequenceCappedPinnedModel(string name, string path, string coreSet) => PinnedModel(name, path, coreSet) with { MaxConcurrentRequests = 1, MaxContextTokens = 8192, KvBlocks = 640 };


    // A model asking the engine's whole worker share while pinning nothing: what the posture seats for it is this ledger's answer.
    private static ModelConfig UnpinnedModel(string name, string path) => new()
    {
        Name = name,
        Path = path,
        MaxContextTokens = 512,
        KvBlocks = 8,
    };

    private static LairConfig ConfigWith(IReadOnlyList<ModelConfig> models, int threads) => new()
    {
        Server = new ServerConfig(),
        Engine = new EngineConfig { Threads = threads },
        Models = [.. models],
        Host = null, // the posture these gates seat against arrives through the injected ledger.
    };

    private static IReadOnlySet<int> ParseCoreRanges(string spec) => new HashSet<int>(
        spec.Split(',').SelectMany(part =>
        {
            string[] edges = part.Split('-');
            return edges.Length == 2
                ? Enumerable.Range(int.Parse(edges[0]), int.Parse(edges[1]) - int.Parse(edges[0]) + 1)
                : new[] { int.Parse(part) };
        }));

    // A four-processor posture reported as two whole partitions: each pinned model's core set is one node, so both seats are their own.
    private static HostTopology SyntheticPartitionedPosture() => new()
    {
        Nodes =
        [
            new HostNodeCapacity { Index = 0, LogicalProcessors = 2, CoreIds = [0, 1] },
            new HostNodeCapacity { Index = 1, LogicalProcessors = 2, CoreIds = [2, 3] },
        ],
        DerivedFromReportedTopology = true, // synthetic posture stands in for what a real host would have reported.
    };

    private string SyntheticModel(string modelName)
    {
        string path = SyntheticLlamaGguf.Write(modelName);
        _tempFiles.Add(path);
        return path;
    }

    private void WaitUntil(Func<bool> condition, string what)
    {
        Stopwatch clock = Stopwatch.StartNew();
        while (!condition())
        {
            if (clock.Elapsed > TimeSpan.FromSeconds(30))
            {
                throw new TimeoutException($"timed out waiting for {what}");
            }

            Thread.Sleep(50);
        }
    }

    /// <summary>Two synthetic model engines whose resolved routes carry the two partitions of one posture, each with its own scheduler and pool. The
    /// core sets land on the route exactly as a host's placement resolves them; no worker thread is pinned here, so what these gates read is the
    /// scheduling partition itself rather than whatever the operating system happens to schedule.</summary>
    private sealed class ModelPair : IDisposable
    {
        public string ModelAName { get; } = "fairness-quiet";
        public string ModelBName { get; } = "fairness-saturating";

        public GenerationScheduler SchedulerA { get; }
        public GenerationScheduler SchedulerB { get; }

        private readonly GgufFile _fileA;
        private readonly GgufFile _fileB;
        private readonly CpuBackend _backendA;
        private readonly CpuBackend _backendB;
        private readonly TransformerModel _modelA;
        private readonly TransformerModel _modelB;
        private readonly BlockPool _poolA;
        private readonly BlockPool _poolB;

        public ModelPair(string modelName, int threads)
        {
            string pathA = SyntheticLlamaGguf.Write(modelName);
            string pathB = SyntheticLlamaGguf.Write(modelName + "-twin"); // a second load: two models advance under two distinct plans.

            _fileA = GgufFile.Open(pathA);
            _fileB = GgufFile.Open(pathB);

            _backendA = new CpuBackend(threads: threads);
            _backendB = new CpuBackend(threads: threads);

            _modelA = TransformerModel.Open(_fileA, _backendA, maxContextTokens: 256, logLine: null, PrefillKernel.GemV, Route(ModelAName, [0, 1]));
            _modelB = TransformerModel.Open(_fileB, _backendB, maxContextTokens: 256, logLine: null, PrefillKernel.GemV, Route(ModelBName, [2, 3]));

            _poolA = new BlockPool(
                blockTokens: 16,
                layerCount: _modelA.Desc.LayerCount,
                kvHeads: _modelA.Desc.KvHeadCount,
                headDim: _modelA.Desc.HeadDim,
                capacityBlocks: 24,
                backend: _backendA);

            _poolB = new BlockPool(
                blockTokens: 16,
                layerCount: _modelB.Desc.LayerCount,
                kvHeads: _modelB.Desc.KvHeadCount,
                headDim: _modelB.Desc.HeadDim,
                capacityBlocks: 24,
                backend: _backendB);

            SchedulerA = new GenerationScheduler(_modelA, _poolA, maxBatchTokens: 8, cache: null, modelName: ModelAName);
            SchedulerB = new GenerationScheduler(_modelB, _poolB, maxBatchTokens: 8, cache: null, modelName: ModelBName);
        }

        private static ModelRoute Route(string model, int[] cores) => ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = model,
            Stages = [new RouteStage(ModelRoutes.DecodeStage, "cpu")],
            AvailableDevices = ["cpu"],
            CoreSet = new HashSet<int>(cores), // the partition this model's step composition bounds itself against.
        });

        public void Dispose()
        {
            SchedulerA.Dispose();
            SchedulerB.Dispose();
            _poolA.Dispose();
            _poolB.Dispose();
            _modelA.Dispose();
            _modelB.Dispose();
            _backendA.Dispose();
            _backendB.Dispose();
            _fileA.Dispose();
            _fileB.Dispose();
        }
    }
}
