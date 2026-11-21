namespace Lair.Contracts;

/// <summary>
/// A residency tier: one layer of the storage hierarchy that can hold units of expert work resident so a step reads them from
/// the tier instead of re-reading them cold, and whose hot set is chosen by a capacity/cost model — value = router hit frequency ×
/// gather cost saved, budget = RAM quota minus the working-set reserve. Selection is neither a static top-N (which misses workload
/// drift) nor plain LRU (which ignores that experts differ in how expensive they are to gather cold): units rank by expected bytes
/// their residency saves per step, and swaps happen only when an incoming unit beats its victim by a margin over decayed counts.
/// A tier answers slab reads through <see cref="IExpertSlabSource"/>: resident units resolve from private copies the tier owns,
/// everything else falls through to the cold backing store — so whichever way one resolution lands, the bytes and the arithmetic
/// stay bit-identical with the dense-equivalent forward. Hot-set size, hits/misses, and a saved-bytes estimate are cumulative
/// since open and ride the metrics bus; <see cref="IsResident"/> is the residency view routing policy reads when it is allowed to
/// prefer what already lives here.
/// </summary>
public interface IResidencyTier
{
    /// <summary>The tier posture name reported on the metrics bus (<c>ram_hot_experts</c> for the RAM implementation).</summary>
    string Kind { get; }

    /// <summary>The byte ceiling this tier may hold resident: its RAM quota less the working-set reserve carved out beside it.</summary>
    long BudgetBytes { get; }

    /// <summary>Bytes currently held in residency across every admitted unit — never above <see cref="BudgetBytes"/>.</summary>
    long ResidentBytes { get; }

    /// <summary>How many expert units (per layer) are fully resident right now.</summary>
    int HotUnits { get; }

    /// <summary>Cumulative slab resolutions this tier answered from residency since it opened.</summary>
    long HitsTotal { get; }

    /// <summary>Cumulative slab resolutions that fell through to the cold backing store since the tier opened.</summary>
    long MissesTotal { get; }

    /// <summary>Estimated bytes those resident answers kept out of the cold read path — slab bytes answered hot, summed per resolution.</summary>
    long SavedBytesEstimateTotal { get; }

    /// <summary>
    /// Fold one sparse-FFN layer step's active-expert set into this tier's capacity/cost model and run any due residency swap round.
    /// <paramref name="expertIndexes"/> is that step's union: every expert the batch's router pointers select, each exactly once, in
    /// ascending order (the same set the gather walks); <paramref name="memberCounts"/> gives how many selection slots routed to each
    /// position — empty means one each. The three roles carry the stacks and per-expert byte costs a cold gather would pay for this
    /// unit across gate/up/down, which is what ranks candidates against victims. A call may promote copies into residency or demote
    /// beaten units; it changes no arithmetic on any slab read by this step.
    /// </summary>
    void NoteActiveExperts(ExpertSlabRole gate, ExpertSlabRole up, ExpertSlabRole down, ReadOnlySpan<int> expertIndexes, ReadOnlySpan<int> memberCounts);

    /// <summary>Whether the given stacked tensor's expert is fully resident in this tier right now — the residency view a routing policy reads.</summary>
    bool IsResident(LogicalTensor stack, int expertIndex);
}
