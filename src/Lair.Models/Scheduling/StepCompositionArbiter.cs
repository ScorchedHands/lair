namespace Lair.Models.Scheduling;

using System.Runtime.CompilerServices;

/// <summary>
/// What one advancing pipeline offers to step composition: its identity, the resolved model plan it advances under — whose route is
/// the execution path every candidate of a merged step shares — and where its KV blocks are resident while it steps. A candidate
/// carries no engine handle or device claim: composition is decided from what load already resolved about the work, so a route this
/// build cannot execute across devices still composes under exactly the same rules as an executable one, and what keeps a candidate out
/// names itself either way.
/// </summary>
internal readonly record struct StepCandidate
{
    public StepCandidate()
    {
        KvResidency = WeightTier.Dram; // the residency every pool in this build holds blocks at, which is what a candidate carries when its caller names no other tier
    }

    public required long PipelineId { get; init; }

    /// <summary>The resolved plan this pipeline advances under. Two candidates share a step only when they advance the very same plan instance, whose route is their shared execution path: two loads that happen to match field for field are still two steps.</summary>
    public required ModelPlan Plan { get; init; }

    /// <summary>Where this pipeline's KV blocks live while it advances. Every pool in this build holds DRAM-resident blocks, so every candidate admitted today carries that tier; a candidate at another residency joins only a step over a segment that can address it.</summary>
    public WeightTier KvResidency { get; init; }

    /// <summary>The identity this candidate composes under: its shared plan instance plus the ordered route that plan resolves to. Merge groups are keyed on exactly this value, so one hybrid-shaped route is one group over its whole segment list rather than one group per device or per segment.</summary>
    public StepRouteKey RouteKey => new(Plan, Plan.Route);
}

/// <summary>The kind of headroom a composition ran out of for one candidate. Every deferral names which limit blocked it — the stepping segment's batch cap, its worker-thread quota, its pinned core set, this candidate's KV residency against that segment, or a route shape with no segment at the position being stepped.</summary>
internal enum CompositionBlock
{
    /// <summary>Pipelines already in this step hold every sequence slot of the stepping segment's batch cap.</summary>
    BatchCap,

    /// <summary>Pipelines already in this step claim every worker thread of the stepping segment's declared thread quota.</summary>
    ThreadQuota,

    /// <summary>Pipelines already in this step claim every core of the stepping segment's pinned core set.</summary>
    CoreSetQuota,

    /// <summary>This candidate holds its KV at a residency the segment being stepped cannot address.</summary>
    KvResidency,

    /// <summary>The shared route names no segment at the position this composition steps on, so nothing can merge onto it here.</summary>
    SegmentShape,
}

/// <summary>A candidate the arbiter kept out of this composition, with the limit that blocked it and a reason naming which quota or cap ran out. A deferral is not an admission refusal: the pipeline keeps its advancement slot and its whole-lifetime capacity reservation, rides no forward at this boundary, and composes again at the next one.</summary>
internal sealed record DeferredStepCandidate(StepCandidate Candidate, int SegmentIndex, CompositionBlock Block, string Reason);

/// <summary>The identity a merged step is keyed on: one resolved model plan instance plus the execution path it resolves to. The plan compares by reference — sharing a plan means advancing the same loaded model, not two loads whose fields happen to agree — and the route compares as its own value set, so segment order is part of the identity. Each load's segments carry that load's quota objects with them, which keeps two models' structurally similar routes distinct values rather than one shared group.</summary>
internal readonly record struct StepRouteKey(ModelPlan Plan, ModelRoute Route)
{
    public bool Equals(StepRouteKey other) => ReferenceEquals(Plan, other.Plan) && Route.Equals(other.Route);

    public override int GetHashCode() => HashCode.Combine(RuntimeHelpers.GetHashCode(Plan), Route.GetHashCode());
}

/// <summary>One logical step the arbiter composed over one shared route: the candidates that advance together on its stepping segment, and the candidates this composition could not fit — each named with what blocked it. A group sharing a hybrid-shaped route is a single value here however many devices or segments that route spans; splitting such a group per device would be a different step, and nothing in composition does that.</summary>
internal sealed record StepComposition(
    StepRouteKey Key,
    int SegmentIndex,
    IReadOnlyList<StepCandidate> Composed,
    IReadOnlyList<DeferredStepCandidate> Deferred)
{
    /// <summary>The segment this composition advances its merged step on: the route position named when the composition was decided.</summary>
    public DeviceSegment SteppingSegment => Key.Route.Segments[SegmentIndex];

    /// <summary>Candidates that share a forward pass with one another — more than one means their math rode one merged batch, and each still sampled its own logits row either way.</summary>
    public int MergedSequenceCount => Composed.Count;
}

/// <summary>The composition decisions of one boundary: every route group stepped at it and every candidate kept out. The arbiter publishes this as a single immutable value replaced wholesale rather than lists mutated in place, so an observer reads either the previous boundary's decision or this one — never a half-written list.</summary>
internal sealed record CompositionReport(
    IReadOnlyList<StepComposition> Compositions,
    IReadOnlyList<DeferredStepCandidate> Deferrals)
{
    /// <summary>The composition decided for <paramref name="key"/>, or null when no group advanced under that route at this boundary.</summary>
    public StepComposition? CompositionFor(StepRouteKey key)
    {
        foreach (StepComposition composition in Compositions)
        {
            if (composition.Key.Equals(key))
            {
                return composition;
            }
        }

        return null;
    }
}

/// <summary>
/// Route-aware step composition over candidate pipelines: who may advance their math in the same forward pass, decided on route
/// identity rather than device identity. Candidates merge into one logical step when they advance the same resolved model plan and
/// share that plan's whole ordered route — a hybrid-shaped group merges as one step across its segments instead of splitting per
/// device or per segment — and only while their KV residency is addressable by the segment being stepped, that segment's batch cap
/// still has room for them, and they leave quota headroom against it (worker threads, pinned cores) for every pipeline already in
/// the step. A candidate that cannot fit is deferred with a reason naming which cap or quota blocked it; deferral keeps its slot and
/// reservation intact and simply rides no forward at this boundary, so nothing about merging can move another sequence's tokens: each
/// pipeline samples from its own logits row whether the step advanced one sequence or many. Composition reads only what load resolved —
/// plan geometry, route segments with their caps and quotas, candidate residency — claims nothing new against any device ledger, opens
/// no backend, allocates no buffer, and never reorders a route's segments.
/// </summary>
internal sealed class StepCompositionArbiter
{
    private static readonly IReadOnlyList<StepComposition> NoCompositions = [];
    private static readonly IReadOnlyList<DeferredStepCandidate> NoDeferrals = [];

    // Cumulative count over every boundary of candidates kept out of a composition. Written by the composing thread and read as a
    // volatile snapshot by observers: diagnostic material for tests and internal reporting, never an input to any admission decision.
    private long _deferredCandidates;

    // The most recent boundary's decisions, replaced atomically as one immutable report (see CompositionReport).
    private CompositionReport? _report;

    // Refusals from the last boundary that refused anything, kept whole so a later all-fitting boundary does not erase which cap or quota
    // bound an earlier step. Immutable arrays replaced wholesale: observers never see partial state.
    private DeferredStepCandidate[] _refusals = [];

    /// <summary>Candidates this arbiter has kept out of a composition since it opened — across every route group and every boundary.</summary>
    public long DeferredCandidateCount => Volatile.Read(ref _deferredCandidates);

    /// <summary>The compositions and deferrals decided at the most recent boundary, or an empty report before any boundary composed anything.</summary>
    public CompositionReport LastReport => Volatile.Read(ref _report) ?? new(NoCompositions, NoDeferrals);

    /// <summary>The refusals recorded by the last boundary that kept anything out of a composition — each naming the cap or quota that blocked it. A boundary that fit every candidate leaves these in place rather than clearing them, so what bound a step stays reportable after later boundaries succeed.</summary>
    public IReadOnlyList<DeferredStepCandidate> MostRecentRefusals => Volatile.Read(ref _refusals);

    /// <summary>
    /// Composes one logical step per shared route identity across <paramref name="candidates"/>, stepping every group on the route
    /// position named by <paramref name="steppingSegmentIndex"/>. Groups form in candidate order and stay independent: candidates that
    /// do not share a plan instance plus its whole route never ride one forward pass, whatever devices their routes name or how similar
    /// their geometry is. Inside a group, candidates are considered in the order given — ascending pipeline id is what keeps composition
    /// identical across runs — and each admission re-checks residency compatibility against the stepping segment, that segment's batch
    /// cap, and its quota headroom for the pipelines already admitted. Every candidate left out lands in that group's deferral list with
    /// the limit that blocked it named; a deferred candidate is refused nowhere else and composes again at the next boundary.
    /// </summary>
    public CompositionReport Compose(IReadOnlyList<StepCandidate> candidates, int steppingSegmentIndex)
    {
        ArgumentNullException.ThrowIfNull(candidates);

        // Merge groups keyed on route identity in first-candidate order: the first candidate of a group opens it and every later
        // candidate sharing that plan-and-route joins it, so one hybrid-shaped route is one composition no matter how many segments or
        // devices it spans. A repeat over another plan instance — identical geometry notwithstanding — opens its own group.
        List<StepRouteKey> keys = [];
        List<List<StepCandidate>> groups = [];
        Dictionary<StepRouteKey, int> groupOf = new();

        foreach (StepCandidate candidate in candidates)
        {
            StepRouteKey key = candidate.RouteKey;
            if (!groupOf.TryGetValue(key, out int index))
            {
                index = keys.Count;
                groupOf.Add(key, index);
                keys.Add(key);
                groups.Add([]);
            }

            groups[index].Add(candidate);
        }

        List<StepComposition> compositions = [];
        List<DeferredStepCandidate> deferrals = [];

        for (int g = 0; g < groups.Count; g++)
        {
            StepRouteKey key = keys[g];
            ModelRoute route = key.Route;
            bool namesSegment = steppingSegmentIndex >= 0 && steppingSegmentIndex < route.Segments.Count;
            DeviceSegment segment = namesSegment ? route.Segments[steppingSegmentIndex] : null!;

            // Quota ledger for this step alone: what the pipelines already admitted hold against the stepping segment. A segment that
            // declares no thread cap and pins no cores claims nothing, so composition over it is bounded by its batch cap alone — which
            // is every model resolved without placement quotas today.
            List<StepCandidate> admitted = [];
            List<DeferredStepCandidate> refused = [];
            int[] pinnedCores = namesSegment ? OrderedCores(segment.Quota.Cores) : [];
            int threadClaims = 0;
            int coreClaims = 0;

            foreach (StepCandidate candidate in groups[g])
            {
                if (!namesSegment)
                {
                    refused.Add(new DeferredStepCandidate(
                        candidate,
                        steppingSegmentIndex,
                        CompositionBlock.SegmentShape,
                        $"pipeline {candidate.PipelineId} composes on no segment of this route: it names {route.Segments.Count} " +
                        $"segment(s), so position {steppingSegmentIndex} has nothing to step a merged batch onto."));
                    continue;
                }

                if (candidate.KvResidency != segment.WeightTier)
                {
                    refused.Add(new DeferredStepCandidate(
                        candidate,
                        steppingSegmentIndex,
                        CompositionBlock.KvResidency,
                        $"pipeline {candidate.PipelineId} holds its KV at tier '{candidate.KvResidency}', which " +
                        $"{segment.Describe(steppingSegmentIndex)} cannot address — a merged step advances the residency that segment " +
                        $"reads, so this pipeline steps on its own instead."));
                    continue;
                }

                if (admitted.Count >= segment.BatchCap)
                {
                    refused.Add(new DeferredStepCandidate(
                        candidate,
                        steppingSegmentIndex,
                        CompositionBlock.BatchCap,
                        $"pipeline {candidate.PipelineId} waits for headroom on {segment.Describe(steppingSegmentIndex)}: this step " +
                        $"already holds its batch cap of {segment.BatchCap} sequence(s)."));
                    continue;
                }

                if (segment.Quota.Threads is int threadCap && threadClaims >= threadCap)
                {
                    refused.Add(new DeferredStepCandidate(
                        candidate,
                        steppingSegmentIndex,
                        CompositionBlock.ThreadQuota,
                        $"pipeline {candidate.PipelineId} waits for headroom on {segment.Describe(steppingSegmentIndex)}: its " +
                        $"'{threadCap}'-thread quota is fully claimed by the sequences already in this step."));
                    continue;
                }

                if (pinnedCores.Length > 0 && coreClaims >= pinnedCores.Length)
                {
                    refused.Add(new DeferredStepCandidate(
                        candidate,
                        steppingSegmentIndex,
                        CompositionBlock.CoreSetQuota,
                        $"pipeline {candidate.PipelineId} waits for headroom on {segment.Describe(steppingSegmentIndex)}: its pinned " +
                        $"core set [{string.Join(", ", pinnedCores)}] is fully claimed by the sequences already in this step."));
                    continue;
                }

                admitted.Add(candidate);

                // Each pipeline joining a merged step holds one share of what the stepping segment declares against it: one worker thread
                // against a declared cap, and one core from its pinned set. A segment declaring neither claims nothing for its sequences.
                if (segment.Quota.Threads is not null)
                {
                    threadClaims++;
                }

                if (pinnedCores.Length > 0)
                {
                    coreClaims++;
                }
            }

            compositions.Add(new StepComposition(key, steppingSegmentIndex, admitted, refused));
            deferrals.AddRange(refused);
        }

        Volatile.Write(ref _deferredCandidates, Volatile.Read(ref _deferredCandidates) + deferrals.Count);

        if (deferrals.Count > 0)
        {
            Volatile.Write(ref _refusals, [.. deferrals]);
        }

        CompositionReport report = new([.. compositions], [.. deferrals]);
        Volatile.Write(ref _report, report);
        return report;
    }

    // The pinned core ids of a segment in ascending order: the claim order composition walks so headroom is decided identically for one
    // route whatever iteration order an unordered set reports its cores in. Empty means the segment pins nothing and claims no core share,
    // which leaves only its batch cap as the limit on merged sequences.
    private static int[] OrderedCores(IReadOnlySet<int> cores) => [.. cores.Order()];
}
