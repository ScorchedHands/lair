using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Models.Scheduling;

namespace Lair.Tests;

/// <summary>
/// The merge predicate itself, decided over candidate pipelines rather than over executed forwards: candidates join one logical step when
/// they advance the same resolved plan across that plan's whole ordered route — so a hybrid-shaped group merges as one composition instead
/// of splitting per device or segment position — and only while their KV residency is addressable by the stepped segment, its batch cap has
/// room for them, and quota headroom survives against it for what already joined. Every refusal names which cap or quota blocked it. A
/// synthetic accelerator device stands in exactly as route resolution uses one: nothing executes that shape here, so grouping itself is what
/// these gates own.
/// </summary>
[Trait("Category", "Scheduling")]
[Collection("model-loaded")] // shares the serialized lane with the suites that open engines from these fixtures on this box
public sealed class StepCompositionArbiterTests : IDisposable
{
    // The device identity a resolved hybrid route names for a segment no backend in this build reports: real material for grouping, never executable here.
    private const string AcceleratorDeviceId = "cuda";

    private readonly List<string> _tempFiles = [];

    [Fact]
    public void Arbiter_MergesCandidatesSharingAHybridRouteIntoOneStep_NotPerSegmentOrDevice()
    {
        ModelPlan plan = ResolvePlanWith("arbiter-hybrid-route", new RouteRequest
        {
            ModelName = "arbiter-hybrid-route",
            Stages = [new(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId), new("split_block", AcceleratorDeviceId), new(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId, AcceleratorDeviceId],
        });

        Assert.Equal(3, plan.Route.Segments.Count); // a crossing that survives resolution: CPU span, accelerator block, CPU span again

        StepCompositionArbiter arbiter = new();
        CompositionReport report = arbiter.Compose([Candidate(plan, 1), Candidate(plan, 2), Candidate(plan, 3)], steppingSegmentIndex: 2);

        // One composition over the whole route — not one per device and not one per segment position of that route.
        StepComposition composition = Assert.Single(report.Compositions);
        Assert.Equal(3, composition.MergedSequenceCount);
        Assert.Empty(composition.Deferred);
        Assert.Empty(report.Deferrals);

        // The crossing stayed inside the merged group's own route: composition neither dropped it nor re-split the group around it.
        Assert.Equal(AcceleratorDeviceId, plan.Route.Segments[1].DeviceId);
        Assert.Equal(ModelRoute.CpuDeviceId, composition.SteppingSegment.DeviceId);

        Assert.Equal(0, arbiter.DeferredCandidateCount);
    }

    [Fact]
    public void Arbiter_KeepsSeparatePlansApart_EvenWhenTheirResolvedValuesMatch()
    {
        ModelPlan planA = ResolvePlan("arbiter-plan-a");
        ModelPlan planB = ResolvePlan("arbiter-plan-b");

        // Same resolved shape from two loads. Each carries its own quota objects on its segments, so even their route values are distinct — one more reason composition keys on the plan instance rather than matching shapes across models.
        DeviceSegment segmentA = Assert.Single(planA.Route.Segments);
        DeviceSegment segmentB = Assert.Single(planB.Route.Segments);
        Assert.Equal(ModelRoute.CpuDeviceId, segmentA.DeviceId);
        Assert.Equal(segmentB.DeviceId, segmentA.DeviceId);
        Assert.Equal(segmentA.BatchCap, segmentB.BatchCap);

        Assert.False(new StepRouteKey(planA, planA.Route).Equals(new StepRouteKey(planB, planB.Route)));

        CompositionReport report = new StepCompositionArbiter().Compose([Candidate(planA, 1), Candidate(planB, 2)], steppingSegmentIndex: 0);

        Assert.Equal(2, report.Compositions.Count);
        Assert.Single(report.CompositionFor(new StepRouteKey(planA, planA.Route))!.Composed);
        Assert.Single(report.CompositionFor(new StepRouteKey(planB, planB.Route))!.Composed);
    }

    [Fact]
    public void Arbiter_KeysOnWholeOrderedRoute_NotTheDeviceUnderAnyOnePosition()
    {
        ModelPlan cpuFirst = ResolvePlanWith("arbiter-order-a", new RouteRequest
        {
            ModelName = "arbiter-order-a",
            Stages = [new(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId), new(ModelRoutes.DecodeStage, AcceleratorDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId, AcceleratorDeviceId],
        });

        ModelPlan acceleratorFirst = ResolvePlanWith("arbiter-order-b", new RouteRequest
        {
            ModelName = "arbiter-order-b",
            Stages = [new(ModelRoutes.PrefillStage, AcceleratorDeviceId), new(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId, AcceleratorDeviceId],
        });

        Assert.Equal(2, cpuFirst.Route.Segments.Count);
        Assert.Equal(2, acceleratorFirst.Route.Segments.Count);

        // The same devices in a different order is a different step: the ordered route is part of what identity keys on.
        Assert.False(new StepRouteKey(cpuFirst, cpuFirst.Route).Equals(new StepRouteKey(acceleratorFirst, acceleratorFirst.Route)));

        CompositionReport report = new StepCompositionArbiter().Compose(
            [Candidate(cpuFirst, 1), Candidate(acceleratorFirst, 2), Candidate(cpuFirst, 3), Candidate(acceleratorFirst, 4)],
            steppingSegmentIndex: 0);

        // Two groups, each merged whole — no candidate crossed the boundary between routes just because both name 'cpu' somewhere.
        Assert.Equal(2, report.Compositions.Count);
        StepComposition cpuSide = report.CompositionFor(new StepRouteKey(cpuFirst, cpuFirst.Route))!;
        StepComposition crossedSide = report.CompositionFor(new StepRouteKey(acceleratorFirst, acceleratorFirst.Route))!;

        Assert.Equal(2, cpuSide.MergedSequenceCount);
        Assert.Equal(ModelRoute.CpuDeviceId, cpuSide.SteppingSegment.DeviceId);
        Assert.Equal(2, crossedSide.MergedSequenceCount);
        Assert.Equal(AcceleratorDeviceId, crossedSide.SteppingSegment.DeviceId);
    }

    [Fact]
    public void Arbiter_KeepsCandidateWhoseKvResidencyTheSteppedSegmentCannotAddress_OutOfThatStep()
    {
        ModelPlan plan = ResolvePlan("arbiter-residency");

        CompositionReport report = new StepCompositionArbiter().Compose(
            [Candidate(plan, 1), Candidate(plan, 2) with { KvResidency = WeightTier.Vram }, Candidate(plan, 3)],
            steppingSegmentIndex: 0);

        StepComposition composition = Assert.Single(report.Compositions);
        Assert.Equal(2, composition.MergedSequenceCount); // the candidates whose residency fits still merged as one step
        DeferredStepCandidate refusal = Assert.Single(composition.Deferred);
        Assert.Equal(CompositionBlock.KvResidency, refusal.Block);
        Assert.Contains("holds its KV at tier 'Vram'", refusal.Reason, StringComparison.Ordinal);
        Assert.Contains("segment 0 ('cpu' at 'Dram') cannot address", refusal.Reason, StringComparison.Ordinal);

        // The same rule read the other way: a segment resident on an accelerator steps candidates resident at that residency,
        // which is a route resolution can only resolve once a device-backed KV mirror is opened for those blocks.
        ModelPlan acceleratorResident = ResolvePlanWith("arbiter-residency-accelerator", new RouteRequest
        {
            ModelName = "arbiter-residency-accelerator",
            Stages = [new(ModelRoutes.DecodeStage, AcceleratorDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId, AcceleratorDeviceId],
            KvResidency = BlockResidencyMode.Vram,
            DeviceBackedKvMirrorSupport = true,
        });

        CompositionReport stepped = new StepCompositionArbiter().Compose(
            [Candidate(acceleratorResident, 4) with { KvResidency = WeightTier.Vram }, Candidate(acceleratorResident, 5) with { KvResidency = WeightTier.Vram }],
            steppingSegmentIndex: 0);

        Assert.Equal(2, Assert.Single(stepped.Compositions).MergedSequenceCount);
        Assert.Empty(stepped.Deferrals);
    }

    [Fact]
    [Trait("Category", "Budget")]
    public void Arbiter_KeepsCandidatePastTheSteppingSegmentsBatchCap_Out_NamingThatCap()
    {
        ModelPlan plan = ResolvePlanWith("arbiter-batch-cap", new RouteRequest
        {
            ModelName = "arbiter-batch-cap",
            Stages = [new(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId],
            BatchCap = 2,
        });

        CompositionReport report = new StepCompositionArbiter().Compose([Candidate(plan, 1), Candidate(plan, 2), Candidate(plan, 3)], steppingSegmentIndex: 0);

        StepComposition composition = Assert.Single(report.Compositions);
        Assert.Equal(2, composition.MergedSequenceCount); // the cap held two sequences before this one was kept out
        DeferredStepCandidate refusal = Assert.Single(composition.Deferred);
        Assert.Equal(3, refusal.Candidate.PipelineId);
        Assert.Equal(CompositionBlock.BatchCap, refusal.Block);
        Assert.Contains("already holds its batch cap of 2", refusal.Reason, StringComparison.Ordinal);

        // A hybrid-shaped group obeys the same ceiling on the segment it steps: segments elsewhere in its route neither widen nor bypass it.
        ModelPlan crossed = ResolvePlanWith("arbiter-batch-cap-crossed", new RouteRequest
        {
            ModelName = "arbiter-batch-cap-crossed",
            Stages = [new(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId), new("split_block", AcceleratorDeviceId), new(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId, AcceleratorDeviceId],
            BatchCap = 1,
        });

        CompositionReport acrossCrossing = new StepCompositionArbiter().Compose([Candidate(crossed, 4), Candidate(crossed, 5)], steppingSegmentIndex: 2);

        StepComposition crossingStep = Assert.Single(acrossCrossing.Compositions);
        Assert.Equal(1, crossingStep.MergedSequenceCount);
        Assert.Contains("batch cap of 1", Assert.Single(crossingStep.Deferred).Reason, StringComparison.Ordinal);
    }

    [Fact]
    [Trait("Category", "Budget")]
    public void Arbiter_KeepsCandidatesOutWhenThreadQuotaOrPinnedCoresRunOut_NamingWhichDid()
    {
        ModelPlan basePlan = ResolvePlan("arbiter-quota");

        ModelPlan threadBound = ResolvePlanWith("arbiter-quota-threads", new RouteRequest
        {
            ModelName = "arbiter-quota-threads",
            Stages = [new(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId],
            ThreadQuota = 2, // batch cap stays at the engine ceiling here: only thread headroom can bind this step
        });

        CompositionReport threadsFull = new StepCompositionArbiter().Compose(
            [Candidate(threadBound, 1), Candidate(threadBound, 2), Candidate(threadBound, 3)],
            steppingSegmentIndex: 0);

        Assert.Equal(2, Assert.Single(threadsFull.Compositions).MergedSequenceCount);
        DeferredStepCandidate threadRefusal = Assert.Single(threadsFull.Deferrals);
        Assert.Equal(CompositionBlock.ThreadQuota, threadRefusal.Block);
        Assert.Contains("'2'-thread quota is fully claimed", threadRefusal.Reason, StringComparison.Ordinal);

        ModelPlan coreBound = ResolvePlanWith("arbiter-quota-cores", new RouteRequest
        {
            ModelName = "arbiter-quota-cores",
            Stages = [new(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId],
            CoreSet = new HashSet<int> { 4, 9 }, // pinned cores bind here; no thread cap is declared at all
        });

        CompositionReport coresFull = new StepCompositionArbiter().Compose(
            [Candidate(coreBound, 1), Candidate(coreBound, 2), Candidate(coreBound, 3)],
            steppingSegmentIndex: 0);

        Assert.Equal(2, Assert.Single(coresFull.Compositions).MergedSequenceCount);
        DeferredStepCandidate coreRefusal = Assert.Single(coresFull.Deferrals);
        Assert.Equal(CompositionBlock.CoreSetQuota, coreRefusal.Block);
        Assert.Contains("pinned core set [4, 9] is fully claimed", coreRefusal.Reason, StringComparison.Ordinal);

        // A segment declaring neither quota claims nothing against itself: its batch cap alone bounds merged sequences.
        CompositionReport unbounded = new StepCompositionArbiter().Compose(
            [.. Enumerable.Range(1, TransformerModel.MaxBatchSequences).Select(id => Candidate(basePlan, id))],
            steppingSegmentIndex: 0);

        Assert.Equal(TransformerModel.MaxBatchSequences, Assert.Single(unbounded.Compositions).MergedSequenceCount);
        Assert.Empty(unbounded.Deferrals);
    }

    [Fact]
    public void Arbiter_ReportsNoSegmentAtTheSteppedPosition_AsItsOwnRefusal()
    {
        ModelPlan plan = ResolvePlan("arbiter-segment-shape");

        Assert.Single(plan.Route.Segments); // the CPU-only span every model resolved without placement carries

        CompositionReport report = new StepCompositionArbiter().Compose([Candidate(plan, 1), Candidate(plan, 2)], steppingSegmentIndex: 3);

        Assert.Equal(2, report.Deferrals.Count);
        Assert.All(report.Compositions, composition => Assert.Empty(composition.Composed));
        Assert.Contains("names 1 segment(s)", report.Deferrals[0].Reason, StringComparison.Ordinal);
    }

    [Fact]
    [Trait("Category", "Budget")]
    public void Arbiter_LeavesAHybridRoutesResolvedValuesIntact_AcrossACompositionDeferral()
    {
        ModelPlan plan = ResolvePlanWith("arbiter-budget-retention", new RouteRequest
        {
            ModelName = "arbiter-budget-retention",
            Stages = [new(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId), new("split_block", AcceleratorDeviceId), new(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId, AcceleratorDeviceId],
            ThreadQuota = 2,
            CoreSet = new HashSet<int> { 3, 8 },
            BatchCap = 2,
            AcceleratorStreamPriority = 4,
            AcceleratorSmSharePercent = 60.0,
        });

        Assert.Equal(ModelRoute.CpuDeviceId, plan.Route.Segments[2].DeviceId); // the CPU tail is what this composition steps on

        CompositionReport report = new StepCompositionArbiter().Compose(
            [Candidate(plan, 1), Candidate(plan, 2), Candidate(plan, 3)], steppingSegmentIndex: 2);

        StepComposition composition = Assert.Single(report.Compositions);
        Assert.Equal(2, composition.MergedSequenceCount); // the cap admitted two sequences and kept the third out of this step alone

        DeferredStepCandidate refusal = Assert.Single(composition.Deferred);
        Assert.Equal(CompositionBlock.BatchCap, refusal.Block);
        Assert.Contains("already holds its batch cap of 2", refusal.Reason, StringComparison.Ordinal);

        // What bounded the step is exactly what load resolved: one ceiling on every segment of this route, both device classes' claims untouched.
        foreach (DeviceSegment segment in plan.Route.Segments)
        {
            Assert.Equal(2, segment.BatchCap);
            Assert.Equal(WeightTier.Dram, segment.WeightTier);
        }

        DeviceSegment acceleratorSide = plan.Route.Segments[1];
        Assert.Equal(AcceleratorDeviceId, acceleratorSide.DeviceId);
        Assert.Equal(4, acceleratorSide.Quota.StreamPriority);
        Assert.Equal(60.0, acceleratorSide.Quota.SmSharePercent);
        Assert.Null(acceleratorSide.Quota.Threads);

        DeviceSegment stepping = composition.SteppingSegment;
        Assert.Equal(2, stepping.Quota.Threads);
        Assert.True(stepping.Quota.Cores.IsSupersetOf([3, 8]));
    }

    // ---------------------------------------------------------------- helpers

    private static StepCandidate Candidate(ModelPlan plan, long pipelineId) => new() { PipelineId = pipelineId, Plan = plan };

    /// <summary>Resolves one synthetic model's own route — the CPU-only span its file resolves to on its own.</summary>
    private ModelPlan ResolvePlan(string modelName) => ResolvePlanWith(modelName, request: null);

    /// <summary>The same model with a placement-declared route landed on its plan, which is how a host resolved from configuration carries one.</summary>
    private ModelPlan ResolvePlanWith(string modelName, RouteRequest? request)
    {
        string path = SyntheticLlamaGguf.Write(modelName);
        _tempFiles.Add(path);

        using GgufFile gguf = GgufFile.Open(path);
        ModelPlan plan = ModelPlans.Resolve(gguf);
        return request is null ? plan : plan with { Route = ModelRoutes.Resolve(request) };
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
