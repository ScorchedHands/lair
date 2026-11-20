using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Scheduling;

namespace Lair.Tests;

/// <summary>
/// The device fair-share limit as one more composition headroom: candidates past the slots the host window granted this model defer with
/// the same posture every other cap or quota uses — slot kept, reservation kept, no forward ridden at this boundary, re-composed at the
/// next — and the deferral names the model, its compute share, and the device segment the window bounds. A composition carrying no device
/// limit composes exactly as it did before any fair-share existed, and the limit belongs to one model's occupancy: another group sharing
/// nothing composes untouched beside it.
/// </summary>
[Trait("Category", "Scheduling")]
[Collection("model-loaded")] // shares the serialized lane with the suites that open engines from these fixtures on this box
public sealed class DeviceFairShareCompositionTests : IDisposable
{
    private const string AcceleratorDeviceId = "cuda";

    private readonly List<string> _tempFiles = [];

    [Fact]
    public void Arbiter_DefersCandidatesPastTheGrantedDeviceSlots_NamingShareModelAndSegment()
    {
        ModelPlan plan = ResolveHybridPlan("fair-share-compose");
        Assert.Equal(AcceleratorDeviceId, plan.Route.Segments[1].DeviceId); // the device segment the window bounds

        DeviceShareLimit granted = new(
            GrantedStepSlots: 2,
            SharePercent: 30,
            ModelName: "fair-share-compose",
            SegmentDescription: plan.Route.Segments[1].Describe(1),
            Plan: plan);

        StepCompositionArbiter arbiter = new();
        CompositionReport report = arbiter.Compose(
            [Candidate(plan, 1), Candidate(plan, 2), Candidate(plan, 3), Candidate(plan, 4)],
            steppingSegmentIndex: 2, // the CPU tail is what this build steps on; the device limit bounds the cuda segment's work
            deviceShare: granted);

        StepComposition composition = Assert.Single(report.Compositions);
        Assert.Equal(2, composition.MergedSequenceCount); // the window's grant admitted exactly what it granted

        Assert.Equal(2, composition.Deferred.Count);
        Assert.All(composition.Deferred, refusal =>
        {
            Assert.Equal(CompositionBlock.DeviceShare, refusal.Block);
            Assert.Contains("device co-scheduling window", refusal.Reason, StringComparison.Ordinal);
            Assert.Contains("model 'fair-share-compose' compute_share 30%", refusal.Reason, StringComparison.Ordinal);
            Assert.Contains(AcceleratorDeviceId, refusal.Reason, StringComparison.Ordinal); // the segment whose work the window bounds is named
            Assert.Contains("hold all 2 slot(s) granted for this boundary", refusal.Reason, StringComparison.Ordinal);
        });

        Assert.Equal(2, arbiter.DeferredCandidateCount);
        Assert.Equal(2, arbiter.MostRecentRefusals.Count); // the refusing boundary stays reportable as it did for every other composition limit
    }

    [Fact]
    public void Arbiter_WithoutADeviceLimit_ComposesExactlyAsBeforeFairShareExisted_AndLimitsStayPerModel()
    {
        ModelPlan bounded = ResolveHybridPlan("fair-share-bounded");
        ModelPlan other = ResolveHybridPlan("fair-share-other");

        // No limit on the call: the whole group merges, exactly as every composition before this seam resolved it.
        CompositionReport unbounded = new StepCompositionArbiter().Compose(
            [Candidate(bounded, 1), Candidate(bounded, 2), Candidate(bounded, 3)],
            steppingSegmentIndex: 2);

        Assert.Equal(3, Assert.Single(unbounded.Compositions).MergedSequenceCount);
        Assert.Empty(unbounded.Deferrals);

        // The limit belongs to the plan it was granted for: a second group on its own route composes untouched beside a bounded one.
        DeviceShareLimit granted = new(1, 10, "fair-share-bounded", bounded.Route.Segments[1].Describe(1), bounded);
        CompositionReport sideBySide = new StepCompositionArbiter().Compose(
            [Candidate(bounded, 1), Candidate(bounded, 2), Candidate(other, 3), Candidate(other, 4)],
            steppingSegmentIndex: 2,
            deviceShare: granted);

        Assert.Equal(2, sideBySide.Compositions.Count);

        StepComposition boundedStep = sideBySide.CompositionFor(new StepRouteKey(bounded, bounded.Route))!;
        Assert.Single(boundedStep.Composed);
        DeferredStepCandidate refusal = Assert.Single(boundedStep.Deferred);
        Assert.Equal(CompositionBlock.DeviceShare, refusal.Block);
        Assert.Contains("compute_share 10%", refusal.Reason, StringComparison.Ordinal);

        StepComposition otherStep = sideBySide.CompositionFor(new StepRouteKey(other, other.Route))!;
        Assert.Equal(2, otherStep.MergedSequenceCount); // the other model's candidates never deferred against this model's occupancy
        Assert.Empty(otherStep.Deferred);
    }

    [Fact]
    public void Arbiter_AZeroSlotGrant_KeepsEveryCandidateOutOfThatStep_AsADeferralNotARefusal()
    {
        ModelPlan plan = ResolveHybridPlan("fair-share-starved");

        DeviceShareLimit granted = new(0, 1, "fair-share-starved", plan.Route.Segments[1].Describe(1), plan); // the model already holds every slot its 1% share resolves to
        CompositionReport report = new StepCompositionArbiter().Compose(
            [Candidate(plan, 1), Candidate(plan, 2)],
            steppingSegmentIndex: 2,
            deviceShare: granted);

        StepComposition composition = Assert.Single(report.Compositions);
        Assert.Empty(composition.Composed);
        Assert.Equal(2, composition.Deferred.Count);
        Assert.All(composition.Deferred, refusal => Assert.Equal(CompositionBlock.DeviceShare, refusal.Block));
    }

    // ---------------------------------------------------------------- helpers

    private static StepCandidate Candidate(ModelPlan plan, long pipelineId) => new() { PipelineId = pipelineId, Plan = plan };

    private ModelPlan ResolveHybridPlan(string modelName)
    {
        string path = SyntheticLlamaGguf.Write(modelName);
        _tempFiles.Add(path);

        using GgufFile gguf = GgufFile.Open(path);
        ModelPlan plan = ModelPlans.Resolve(gguf);

        return plan with
        {
            Route = ModelRoutes.Resolve(new RouteRequest
            {
                ModelName = modelName,
                Stages =
                [
                    new(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId),
                    new("split_block", AcceleratorDeviceId),
                    new(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId),
                ],
                AvailableDevices = [ModelRoute.CpuDeviceId, AcceleratorDeviceId],
            }),
        };
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
