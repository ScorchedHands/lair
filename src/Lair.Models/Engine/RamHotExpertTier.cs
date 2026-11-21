using System.Collections.Concurrent;

using Lair.Contracts;
using Lair.Loader;

namespace Lair.Models.Engine;

/// <summary>
/// The RAM implementation of the residency tier behind the expert gather path: a hot set of experts held as private byte copies,
/// chosen by capacity/cost rather than insertion order or recency. Each sparse-FFN layer step folds its union active-expert set into
/// decayed per-expert selection counts; every swap round ages those counts (old popularity fades so the hot set tracks workload drift)
/// and rebalances residency so units rank by expected gather cost saved — hit frequency × that unit's cold slab bytes, since experts
/// differ in how expensive they are to read cold. Promotion fills free budget best-value-first; a full budget admits an incoming unit
/// only by evicting resident ones its value beats by the min-gain margin, so demotion is always cost-justified and the hot set never
/// grows past <see cref="BudgetBytes"/> (the RAM quota less the working-set reserve — the KV-side carve-out the opener states). A unit
/// admitted here answers every slab resolution for its expert from private copies; anything else resolves through contiguous cold views
/// exactly as a plain slab source would. Copies are byte-exact, so gather output stays bit-identical whichever side one step's slabs
/// land on — residency changes where bytes come from, never the arithmetic over them. The tier reads whole tensors straight through to its
/// backing store untouched; only expert-slab resolutions take the hot path. Whoever opened the cold source owns it: it must outlive this
/// tier and is disposed by that caller, not here. Demoting a unit drops only the tier's own reference — any gather span already handed out
/// from its copy stays readable for as long as its consumer borrows it.
/// </summary>
public sealed class RamHotExpertTier : IWeightSource, IExpertSlabSource, IResidencyTier
{
    /// <summary>The metrics-bus posture name for this tier.</summary>
    public const string TierKind = "ram_hot_experts";

    /// <summary>Swap rounds default to every 4096 layer-step notes — hot sets need re-ranking far less often than that, and one aging/rebalance pass per round amortizes to a rounding error beside the GEMVs it feeds.</summary>
    public const int DefaultSwapPeriodLayerSteps = 4096;

    /// <summary>How fast selection counts age: each swap round halves them, so a unit's standing reflects recent routing traffic and stale hot experts lose their claim as the workload drifts.</summary>
    private const float AgingFactorPerRound = 0.5f;

    /// <summary>The min-gain gate: an incoming unit must beat its victim's value by this factor before eviction happens, so near-ties never churn residency (the adaptive-swap posture — swap only on a real margin).</summary>
    private const double MinGainMargin = 1.25;

    /// <summary>Scores at or below this read as no expected traffic: they age out instead of ranking.</summary>
    private const float DeadScoreFloor = 1e-6f;

    private readonly IWeightSource _coldBacking;
    private readonly ExpertSlabWeightSource _coldSlabs;
    private readonly ConcurrentDictionary<string, LayerState> _layersByStackName = new(StringComparer.Ordinal);

    // Every recorded layer exactly once: the name map points gate/up/down stack names at one shared state object each, so policy passes iterate this list instead of the dictionary's values — which would visit the same layer three times and count its units repeatedly.
    private readonly List<LayerState> _layers = [];
    private readonly object _policyGate = new();
    private readonly int _swapPeriodLayerSteps;

    // Answer tallies: written by whichever thread resolves a slab (decode and lane forwards share one tier), sampled once per metrics tick.
    private long _hitsTotal;
    private long _missesTotal;
    private long _savedBytesEstimateTotal;
    private long _promotionsTotal;
    private long _demotionsTotal;

    // Residency accounting: published volatile for readers while only the policy lock writes it.
    private long _residentBytes;
    private int _hotUnits;
    private int _notesSinceSwap;
    private int _nextLayerId;

    public RamHotExpertTier(IWeightSource coldBacking, long ramQuotaBytes, long workingSetReserveBytes = 0, int swapPeriodLayerSteps = DefaultSwapPeriodLayerSteps)
    {
        ArgumentNullException.ThrowIfNull(coldBacking);
        ArgumentOutOfRangeException.ThrowIfNegative(ramQuotaBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(workingSetReserveBytes);
        if (swapPeriodLayerSteps < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(swapPeriodLayerSteps), "a residency tier needs a swap period of at least one layer-step note");
        }

        _coldBacking = coldBacking;
        _coldSlabs = new ExpertSlabWeightSource(coldBacking);
        BudgetBytes = Math.Max(0, ramQuotaBytes - workingSetReserveBytes); // the quota less its reserve is the whole admission ceiling — a reserve at or past the quota leaves nothing hot
        _swapPeriodLayerSteps = swapPeriodLayerSteps;
    }

    /// <inheritdoc />
    public string Kind => TierKind;

    /// <summary>The admission ceiling: RAM quota minus the working-set reserve stated at open.</summary>
    public long BudgetBytes { get; }

    /// <inheritdoc />
    public long ResidentBytes => Volatile.Read(ref _residentBytes);

    /// <inheritdoc />
    public int HotUnits => Volatile.Read(ref _hotUnits);

    /// <inheritdoc />
    public long HitsTotal => Volatile.Read(ref _hitsTotal);

    /// <inheritdoc />
    public long MissesTotal => Volatile.Read(ref _missesTotal);

    /// <inheritdoc />
    public long SavedBytesEstimateTotal => Volatile.Read(ref _savedBytesEstimateTotal);

    /// <summary>Cumulative promotions into residency since open — how often the capacity/cost model moved a unit in.</summary>
    public long PromotionsTotal => Volatile.Read(ref _promotionsTotal);

    /// <summary>Cumulative margin-gated demotions out of residency since open — every one beaten by its replacement's value.</summary>
    public long DemotionsTotal => Volatile.Read(ref _demotionsTotal);

    /// <inheritdoc />
    public ReadOnlySpan<byte> Resolve(LogicalTensor tensor) => _coldBacking.Resolve(tensor);

    /// <inheritdoc />
    public ReadOnlySpan<byte> ResolveSlab(LogicalTensor stack, int expertIndex, long expertBytes)
    {
        ArgumentNullException.ThrowIfNull(stack);
        ArgumentOutOfRangeException.ThrowIfNegative(expertIndex);
        ArgumentOutOfRangeException.ThrowIfLessThan(expertBytes, 1L);

        if (expertIndex > 0 && expertBytes > stack.SizeBytes / expertIndex)
        {
            throw new ArgumentException(
                $"slab of expert {expertIndex} at {expertBytes} bytes per expert reaches past the {stack.SizeBytes}-byte " +
                $"payload of stacked tensor '{stack.Name}'", nameof(expertIndex));
        }

        _layersByStackName.TryGetValue(stack.Name, out LayerState? state);
        if (state is not null && expertIndex < state.Hot.Length)
        {
            ResidentEntry? entry = Volatile.Read<ResidentEntry?>(ref state.Hot[expertIndex]);
            if (entry is not null)
            {
                long hotRoleBytes = RoleBytesFor(state, stack.Name);
                if (hotRoleBytes != expertBytes)
                {
                    throw new ArgumentException(
                        $"slab of resident expert {expertIndex} in '{stack.Name}' holds {hotRoleBytes} bytes per expert, " +
                        $"{expertBytes} asked for — the tier records one geometry per stacked role", nameof(expertBytes));
                }

                Interlocked.Increment(ref _hitsTotal);
                Interlocked.Add(ref _savedBytesEstimateTotal, expertBytes); // this resolution read resident bytes instead of cold ones: the estimate counts exactly what it avoided
                byte[] copy = stack.Name == state.Gate.Stack.Name ? entry.Gate : stack.Name == state.Up.Stack.Name ? entry.Up : entry.Down;
                return new ReadOnlySpan<byte>(copy);
            }

            Interlocked.Increment(ref _missesTotal); // a known expert role answered cold: the gather paid its slab read through the backing store this time
        }

        return _coldSlabs.ResolveSlab(stack, expertIndex, expertBytes); // same contiguous-view arithmetic an unhooked slab source answers with — byte-identical bytes either way
    }

    /// <inheritdoc />
    public void NoteActiveExperts(ExpertSlabRole gate, ExpertSlabRole up, ExpertSlabRole down, ReadOnlySpan<int> expertIndexes, ReadOnlySpan<int> memberCounts)
    {
        if (expertIndexes.Length == 0)
        {
            return; // a step that gathered nothing routes nothing into the cost model
        }

        if (!memberCounts.IsEmpty && memberCounts.Length < expertIndexes.Length)
        {
            throw new ArgumentException(
                $"a residency note carries {memberCounts.Length} member counts for {expertIndexes.Length} union positions", nameof(memberCounts));
        }

        lock (_policyGate)
        {
            LayerState state = GetOrAddLayerLocked(gate, up, down);

            for (int position = 0; position < expertIndexes.Length; position++)
            {
                int expertIndex = expertIndexes[position];
                if (expertIndex >= state.Scores.Length)
                {
                    throw new ArgumentException(
                        $"union expert index {expertIndex} exceeds the {state.Scores.Length}-expert geometry recorded for '{gate.Stack.Name}'", nameof(expertIndexes));
                }

                float selections = memberCounts.IsEmpty ? 1f : Math.Max(0, memberCounts[position]);
                state.Scores[expertIndex] += selections; // router hit frequency per expert: every selection slot routed here this step counts once (DN-M7-3's union-set accounting)
            }

            _notesSinceSwap++;
            if (_notesSinceSwap >= _swapPeriodLayerSteps)
            {
                _notesSinceSwap = 0;
                AgeAndRebalanceLocked();
            }
        }
    }

    /// <inheritdoc />
    public bool IsResident(LogicalTensor stack, int expertIndex)
    {
        ArgumentNullException.ThrowIfNull(stack);
        ArgumentOutOfRangeException.ThrowIfNegative(expertIndex);

        return _layersByStackName.TryGetValue(stack.Name, out LayerState? state)
            && expertIndex < state.Hot.Length
            && Volatile.Read(ref state.Hot[expertIndex]) is not null;
    }

    /// <summary>
    /// Zeros the answer tallies (hits/misses/saved estimate/promotions/demotions) so a bench or test harness can measure one dedicated replay pass.
    /// Harness-facing on purpose: serving posture never calls it — metrics snapshots read cumulative-since-open counters, and only harnesses reset them.
    /// </summary>
    public void ResetAnswerTallies()
    {
        Interlocked.Exchange(ref _hitsTotal, 0);
        Interlocked.Exchange(ref _missesTotal, 0);
        Interlocked.Exchange(ref _savedBytesEstimateTotal, 0);
        Interlocked.Exchange(ref _promotionsTotal, 0);
        Interlocked.Exchange(ref _demotionsTotal, 0);
    }

    // First note of a layer records its three stacked roles and their geometry: expert capacity is what each stack's payload holds at
    // that role's per-expert byte block (the same arithmetic the slab views slice by), so one figure pins scores and hot slots for every role.
    private LayerState GetOrAddLayerLocked(ExpertSlabRole gate, ExpertSlabRole up, ExpertSlabRole down)
    {
        if (_layersByStackName.TryGetValue(gate.Stack.Name, out LayerState? existing))
        {
            if (existing.Gate.ExpertBytes != gate.ExpertBytes || existing.Up.ExpertBytes != up.ExpertBytes || existing.Down.ExpertBytes != down.ExpertBytes)
            {
                throw new ArgumentException(
                    $"'{gate.Stack.Name}' was recorded at gate={existing.Gate.ExpertBytes} up={existing.Up.ExpertBytes} down={existing.Down.ExpertBytes} bytes per expert " +
                    $"and now arrives as gate={gate.ExpertBytes} up={up.ExpertBytes} down={down.ExpertBytes} — a stacked role keeps one geometry", nameof(gate));
            }

            return existing;
        }

        long capacity = Math.Min(
            gate.Stack.SizeBytes / gate.ExpertBytes,
            Math.Min(up.Stack.SizeBytes / up.ExpertBytes, down.Stack.SizeBytes / down.ExpertBytes));
        if (capacity < 1 || capacity > int.MaxValue)
        {
            throw new ArgumentException(
                $"expert stacks '{gate.Stack.Name}'/{up.Stack.Name}/{down.Stack.Name} hold no complete per-expert slab at " +
                $"{gate.ExpertBytes}/{up.ExpertBytes}/{down.ExpertBytes} bytes each", nameof(gate));
        }

        LayerState state = new()
        {
            Gate = gate,
            Up = up,
            Down = down,
            Scores = new float[(int)capacity],
            Hot = new ResidentEntry?[(int)capacity],
            UnitBytes = gate.ExpertBytes + up.ExpertBytes + down.ExpertBytes, // one expert's whole chain is the unit: a hot expert answers every role from residency or none — no half-resident experts
            LayerId = _nextLayerId++,
        };

        RegisterName(state.Gate.Stack.Name, state);
        if (state.Up.Stack.Name != state.Gate.Stack.Name)
        {
            RegisterName(state.Up.Stack.Name, state);
        }

        if (state.Down.Stack.Name != state.Gate.Stack.Name && state.Down.Stack.Name != state.Up.Stack.Name)
        {
            RegisterName(state.Down.Stack.Name, state);
        }

        _layers.Add(state); // one entry per layer: the name map above is lookup-only, policy iteration reads this list
        return state;
    }

    private void RegisterName(string name, LayerState state)
    {
        if (_layersByStackName.TryGetValue(name, out LayerState? other) && !ReferenceEquals(other, state))
        {
            throw new ArgumentException(
                $"'{name}' already belongs to layer {other.LayerId} ('{other.Gate.Stack.Name}') — a tier tracks one expert geometry per stacked tensor", nameof(name));
        }

        _layersByStackName[name] = state;
    }

    // One swap round: age every recorded score, then rebalance. Free budget fills best-value-first over units ranked by expected bytes their
    // residency saves (decayed selection count × that unit's cold slab bytes); a full budget admits an incoming unit only by demoting its
    // weakest residents — and only when the candidate beats each victim by the min-gain margin, so near-ties leave residency alone.
    private void AgeAndRebalanceLocked()
    {
        foreach (LayerState state in _layers) // distinct layers only — the name map would hand this pass each layer once per role stack it registered
        {
            float[] scores = state.Scores;
            for (int e = 0; e < scores.Length; e++)
            {
                if (scores[e] != 0f)
                {
                    scores[e] *= AgingFactorPerRound;
                    if (scores[e] <= DeadScoreFloor)
                    {
                        scores[e] = 0f; // traffic that aged out stops ranking at all
                    }
                }
            }
        }

        List<UnitRef> candidates = [];
        List<UnitRef> residents = [];
        foreach (LayerState state in _layers)
        {
            for (int e = 0; e < state.Scores.Length; e++)
            {
                double value = (double)state.Scores[e] * state.UnitBytes; // expected gather cost saved per round if this unit stays resident: hit frequency × bytes a cold gather pays for it — the cost asymmetry plain LRU ignores
                ResidentEntry? entry = Volatile.Read<ResidentEntry?>(ref state.Hot[e]);
                if (entry is not null)
                {
                    residents.Add(new UnitRef(state, e, value));
                }
                else if (state.Scores[e] > 0f)
                {
                    candidates.Add(new UnitRef(state, e, value));
                }
            }
        }

        Comparison<UnitRef> rank = static (a, b) => a.Value != b.Value ? b.Value.CompareTo(a.Value) // best expected saving first; ties resolve deterministically by layer then expert index
            : a.State.LayerId != b.State.LayerId ? a.State.LayerId.CompareTo(b.State.LayerId)
            : a.Expert.CompareTo(b.Expert);
        candidates.Sort(rank);

        for (int i = 0; i < candidates.Count && ResidentBytes < BudgetBytes; i++) // free budget admits best-value units until the ceiling binds — growth stops there and continues only through justified swaps below
        {
            UnitRef candidate = candidates[i];
            if (ResidentBytes + candidate.State.UnitBytes <= BudgetBytes)
            {
                PromoteLocked(candidate);
            }
        }

        for (int i = 0; i < candidates.Count; i++)
        {
            UnitRef candidate = candidates[i];
            ResidentEntry? alreadyHot = Volatile.Read<ResidentEntry?>(ref candidate.State.Hot[candidate.Expert]);
            if (alreadyHot is not null)
            {
                continue; // the fill pass already admitted it this round
            }

            int demotedThisSwap = 0;
            while (ResidentBytes + candidate.State.UnitBytes > BudgetBytes && residents.Count > 0)
            {
                UnitRef worst = residents[0];
                for (int r = 1; r < residents.Count; r++) // the least valuable resident is the eviction candidate...
                {
                    if (IsWorse(residents[r], worst))
                    {
                        worst = residents[r];
                    }
                }

                if (worst.Value * MinGainMargin >= candidate.Value)
                {
                    break; // ...unless nothing resident is beaten by the margin: a near-tie never churns residency, so this unit stays cold and the hot set stops growing here.
                }

                DemoteLocked(worst);
                residents.Remove(worst);
                demotedThisSwap++;
            }

            if (demotedThisSwap > 0 && ResidentBytes + candidate.State.UnitBytes <= BudgetBytes)
            {
                PromoteLocked(candidate); // the room came from units this candidate beats by margin: admit it where that room landed
            }
        }
    }

    private static bool IsWorse(UnitRef a, UnitRef b) =>
        a.Value != b.Value ? a.Value < b.Value
        : a.State.LayerId != b.State.LayerId ? a.State.LayerId > b.State.LayerId
        : a.Expert > b.Expert;

    private void PromoteLocked(UnitRef unit)
    {
        LayerState state = unit.State;

        // Byte-exact copies of what a cold gather would have read for this expert's whole chain — same bytes, resident from now on.
        byte[] gateCopy = _coldSlabs.ResolveSlab(state.Gate.Stack, unit.Expert, state.Gate.ExpertBytes).ToArray();
        byte[] upCopy = _coldSlabs.ResolveSlab(state.Up.Stack, unit.Expert, state.Up.ExpertBytes).ToArray();
        byte[] downCopy = _coldSlabs.ResolveSlab(state.Down.Stack, unit.Expert, state.Down.ExpertBytes).ToArray();

        Volatile.Write(ref state.Hot[unit.Expert], new ResidentEntry(gateCopy, upCopy, downCopy)); // publish last: readers either see no entry (cold path) or a complete chain
        _residentBytes += state.UnitBytes;
        _hotUnits++;
        Interlocked.Increment(ref _promotionsTotal);
    }

    private void DemoteLocked(UnitRef unit)
    {
        LayerState state = unit.State;
        ResidentEntry? entry = Volatile.Read<ResidentEntry?>(ref state.Hot[unit.Expert]);
        if (entry is null)
        {
            return; // already gone — a repeated demotion never double-counts the room it freed
        }

        Volatile.Write(ref state.Hot[unit.Expert], null); // in-flight spans borrowed from these arrays stay readable until their consumers let go: dropping the tier's reference is safe, GC keeps copies alive while referenced
        _residentBytes -= state.UnitBytes;
        _hotUnits--;
        Interlocked.Increment(ref _demotionsTotal);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        // The tier owns only its own hot copies, which GC reclaims; the cold backing store belongs to whoever opened it and stays theirs. Disposal drops every resident reference so no copy outlives a disposed tier — spans already handed out remain readable until their consumers let go, exactly as demotion guarantees them.
        foreach (LayerState state in _layers) // each recorded layer once: hot slots belong to a shared state its three role names all point at
        {
            ResidentEntry?[] hot = state.Hot;
            for (int e = 0; e < hot.Length; e++)
            {
                Volatile.Write(ref hot[e], null);
            }
        }

        Volatile.Write(ref _residentBytes, 0);
        Volatile.Write(ref _hotUnits, 0);
    }

    private static long RoleBytesFor(LayerState state, string stackName) =>
        stackName == state.Gate.Stack.Name ? state.Gate.ExpertBytes
        : stackName == state.Up.Stack.Name ? state.Up.ExpertBytes
        : stackName == state.Down.Stack.Name ? state.Down.ExpertBytes
        : throw new ArgumentException($"'{stackName}' is not one of the recorded roles '{state.Gate.Stack.Name}', '{state.Up.Stack.Name}', '{state.Down.Stack.Name}'", nameof(stackName));

    private readonly record struct UnitRef(LayerState State, int Expert, double Value);

    private sealed class ResidentEntry
    {
        public ResidentEntry(byte[] gate, byte[] up, byte[] down)
        {
            Gate = gate;
            Up = up;
            Down = down;
        }

        public byte[] Gate { get; }
        public byte[] Up { get; }
        public byte[] Down { get; }
    }

    private sealed class LayerState
    {
        public required ExpertSlabRole Gate { get; init; }
        public required ExpertSlabRole Up { get; init; }
        public required ExpertSlabRole Down { get; init; }
        public required float[] Scores { get; init; }
        public required ResidentEntry?[] Hot { get; init; }
        public required long UnitBytes { get; init; }
        public required int LayerId { get; init; }
    }
}
