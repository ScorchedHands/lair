namespace Lair.Models;

/// <summary>
/// What one tier's capacity seam reports for one resource: the ceiling it holds in that resource's native unit
/// (megabytes for weights and prefill buffers, KV blocks for KV), or the explicit unbounded posture — the answer a
/// host gives when it backs the tier but states no public total. A missing seam is never guessed at: a caller that
/// reports none says the tier is not opened for that resource, and a seam never invents a total nobody declared.
/// </summary>
public sealed record CapacitySeam
{
    public required BudgetTier Tier { get; init; }

    public required BudgetResource Resource { get; init; }

    /// <summary>The ceiling this seam holds in the resource's native unit; null states the unbounded posture rather than guessing a total.</summary>
    public long? Capacity { get; init; }

    /// <summary>The unit a claim against this seam counts in.</summary>
    public string UnitName() => DeviceLedger.UnitName(Resource);

    /// <summary>The identity a diagnostic names this seam by: its tier, its resource, and its capacity posture.</summary>
    public string Describe() =>
        $"'{BudgetTierNames.Of(Tier)}' for '{BudgetResourceNames.Of(Resource)}' " +
        (Capacity is null ? "(unbounded capacity posture)" : $"(capacity {Capacity} {UnitName()}s)");
}

/// <summary>
/// One working-copy claim the ledger seated: the model it belongs to, the resource it holds, the tier and chain
/// position it landed on, the amount in the resource's native unit, and its registration order — the sequence every
/// claim takes and the order first-come-first-served seats them in. A claim seated at a read-side origin store holds
/// no ledger capacity: the resource's complete set already lives there.
/// </summary>
public sealed record DeviceLedgerClaim
{
    public required int Sequence { get; init; }

    public required string Model { get; init; }

    public required BudgetResource Resource { get; init; }

    public required BudgetTier Tier { get; init; }

    /// <summary>The amount the claim holds, in the resource's native unit (megabytes for weights and prefill buffers, KV blocks for KV).</summary>
    public required long Amount { get; init; }

    /// <summary>The index of the chain position the claim landed on, counting every position its chain declared.</summary>
    public required int ChainPosition { get; init; }

    /// <summary>True when the claim landed below the chain's first copy-holding position — the chain's own demotion, never a displacement of an earlier claim.</summary>
    public required bool Demoted { get; init; }

    /// <summary>True when the claim landed at the read-side origin store: the complete set already lives there, so the seating consumes no ledger capacity.</summary>
    public required bool AtOriginStore { get; init; }

    /// <summary>The claim as the text a diagnostic or log reports it by: model, resource, amount with its unit, tier, and chain position.</summary>
    public string Describe() =>
        $"'{Model}' {BudgetResourceNames.Of(Resource)} {Amount} {DeviceLedger.UnitName(Resource)}s at " +
        $"'{BudgetTierNames.Of(Tier)}' (chain position {ChainPosition}{(Demoted ? ", demoted" : string.Empty)})";
}

/// <summary>One position a claim walked while registering: the tier, whether it holds copies at all, and why the claim seated there or passed it by.</summary>
public sealed record ClaimChainStep
{
    public required BudgetTier Tier { get; init; }

    public required bool IsOriginStore { get; init; }

    /// <summary>What the claim decided at this position: <c>seated</c>, <c>seated at origin store</c>, <c>held by earlier claims</c>, or <c>no opened capacity seam</c>.</summary>
    public required string Outcome { get; init; }

    /// <summary>What the tier already held for this resource when this claim reached it.</summary>
    public long Occupied { get; init; }

    /// <summary>The tier's capacity posture for this resource; null states unbounded.</summary>
    public long? Capacity { get; init; }

    /// <summary>The ceiling this model declares at the chain position; null when its declaration states none there.</summary>
    public long? DeclaredCap { get; init; }
}

/// <summary>What registering one claim decided: the seated claim with its chain walk, or the exhausted chain with the breakdown of what every consulted position holds.</summary>
public sealed record ClaimOutcome
{
    public required bool Seated { get; init; }

    public DeviceLedgerClaim? Claim { get; init; }

    /// <summary>The last copy-holding tier the exhausted claim consulted; null when the claim seated.</summary>
    public BudgetTier? ExhaustedAtTier { get; init; }

    /// <summary>Every chain position the claim walked, in chain order, with what each held and decided.</summary>
    public required IReadOnlyList<ClaimChainStep> ChainWalk { get; init; }

    /// <summary>The ledger breakdown an exhaustion reports: every consulted position with what it holds against its capacity posture, and the amount nothing could seat.</summary>
    public string Breakdown { get; init; } = string.Empty;
}

/// <summary>
/// Host-scoped capacity accounting over the pools a host opened: every working-copy claim a model registers lands on
/// the first position of its own resolved chain that an opened capacity seam backs, whose declared ceiling fits it, and
/// where earlier claims still leave room against that seam's total — all in registration order. Earlier claims are never
/// revoked, preempted, or displaced to seat a later one: a claim too large for the model's own declared ceiling at a
/// position, or unable to fit beside what an opened finite seam already holds, lands on the next position its chain names
/// (a read-side origin store included, where the complete set already lives), and only a chain with no position left
/// reports exhaustion. Claims are keyed by model, resource, and tier; occupancy is the sum of seated claims against each
/// tier's seam. Landed admissions report through <see cref="ObserveAdmission"/> — they ride inside what their own model
/// already seats instead of registering beside it, so one engine never competes with itself for a tier, while only other
/// models' standing claims compete when host-seam totals leave no room for the landing. Nothing here opens a pool,
/// enforces an engine's existing soft cap, relocates a copy, or gates a request: it accounts against declared chains and
/// reported seams, recording where each claim decided to sit.
/// </summary>
public sealed class DeviceLedger
{
    private readonly object _gate = new();
    private readonly Dictionary<(BudgetTier Tier, BudgetResource Resource), CapacitySeam> _seams;
    private readonly List<CapacitySeam> _seamTable;
    private readonly List<DeviceLedgerClaim> _claims = [];
    private int _nextSequence;

    /// <summary>Opens a ledger over the capacity seams a caller reports — today's host supplies unbounded system-memory seams, and a synthetic harness supplies finite capacity tables in each resource's native unit.</summary>
    public DeviceLedger(IReadOnlyList<CapacitySeam> seams)
    {
        ArgumentNullException.ThrowIfNull(seams);

        _seams = [];
        _seamTable = [.. seams];
        foreach (CapacitySeam seam in seams)
        {
            ArgumentNullException.ThrowIfNull(seam);
            if (!Enum.IsDefined(seam.Resource) || seam.Capacity is < 0)
            {
                throw new ModelException(
                    $"a capacity seam for tier '{seam.Tier}' resource '{seam.Resource}' is malformed: a seam names a known resource and a capacity of 0 or greater, or states the unbounded posture");
            }

            _seams[(seam.Tier, seam.Resource)] = seam;
        }
    }

    /// <summary>
    /// The ledger a host opens when it reports which tiers its pools back but states no public totals: every opened
    /// tier gets an explicit unbounded seam for every budgetable resource. A missing public total is represented as
    /// the unbounded posture — never guessed at from process or machine state.
    /// </summary>
    public static DeviceLedger OpenedTiersUnbounded(IReadOnlyList<BudgetTier> openedPoolTiers)
    {
        ArgumentNullException.ThrowIfNull(openedPoolTiers);

        BudgetResource[] resources = [BudgetResource.Weights, BudgetResource.Kv, BudgetResource.PrefillBuffers];
        return new DeviceLedger([.. openedPoolTiers.SelectMany(tier => resources.Select(resource => new CapacitySeam
        {
            Tier = tier,
            Resource = resource,
            Capacity = null,
        }))]);
    }

    /// <summary>The seams this ledger accounts against, in the order they were supplied.</summary>
    public IReadOnlyList<CapacitySeam> Seams => _seamTable;

    /// <summary>Every claim seated so far, in registration order — the order first-come-first-served seated them in.</summary>
    public IReadOnlyList<DeviceLedgerClaim> Claims
    {
        get
        {
            lock (_gate)
            {
                return [.. _claims];
            }
        }
    }

    /// <summary>The seam a tier reports for one resource; null when no opened seam backs it.</summary>
    public CapacitySeam? SeamFor(BudgetTier tier, BudgetResource resource) =>
        _seams.TryGetValue((tier, resource), out CapacitySeam? seam) ? seam : null;

    /// <summary>The sum of seated claims holding working copies at one tier for one resource. Claims seated at an origin store hold none.</summary>
    public long Occupied(BudgetTier tier, BudgetResource resource)
    {
        lock (_gate)
        {
            return OccupiedAt(tier, resource);
        }
    }

    /// <summary>The claims sharing one tier for one resource, in registration order — the contested holders an operator view reports when more than one model shares a tier.</summary>
    public IReadOnlyList<DeviceLedgerClaim> ClaimsAt(BudgetTier tier, BudgetResource resource)
    {
        lock (_gate)
        {
            return [.. _claims.Where(claim => claim.Tier == tier && claim.Resource == resource && !claim.AtOriginStore)];
        }
    }

    /// <summary>
    /// Registers one working-copy claim against one model's resolved chain, in the caller's registration order. The
    /// claim walks the chain's positions as declared: the first copy-holding position an opened seam backs, whose
    /// declared ceiling fits this model's cumulative claims at that tier, and where earlier claims still leave room
    /// takes it; a position held to its capacity by claims registered ahead is passed by, never taken from; a position
    /// too small for the claim this model declares there is also passed by rather than half-seated; a read-side origin
    /// position seats the claim without consuming capacity (the complete set already lives there); and a copy-holding
    /// position no seam backs is passed by as what this host does not open. A claim that seats at any position below the
    /// chain's first copy-holding position records the chain's own demotion. Only a chain with no position left reports
    /// exhaustion — with the tier it exhausted at and the breakdown of what every consulted position holds.
    /// </summary>
    public ClaimOutcome RegisterClaim(string model, ResolvedBudgetChain chain, long amount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        ArgumentNullException.ThrowIfNull(chain);
        ArgumentOutOfRangeException.ThrowIfNegative(amount);

        lock (_gate)
        {
            List<ClaimChainStep> walk = [];

            int firstCopyHolding = -1;
            for (int index = 0; index < chain.Tiers.Count; index++)
            {
                if (!chain.Tiers[index].IsOriginStore)
                {
                    firstCopyHolding = index;
                    break;
                }
            }

            for (int index = 0; index < chain.Tiers.Count; index++)
            {
                ResolvedBudgetTier position = chain.Tiers[index];

                if (position.IsOriginStore)
                {
                    // The complete set already lives at the read-side origin store: seating there is a posture, not a capacity claim.
                    DeviceLedgerClaim seated = Seat(model, chain.Resource, position.Tier, amount, index, firstCopyHolding >= 0 && index != firstCopyHolding, atOriginStore: true);
                    walk.Add(new ClaimChainStep { Tier = position.Tier, IsOriginStore = true, Outcome = "seated at origin store" });
                    return new ClaimOutcome { Seated = true, Claim = seated, ChainWalk = [.. walk] };
                }

                CapacitySeam? seam = SeamFor(position.Tier, chain.Resource);
                long occupied = OccupiedAt(position.Tier, chain.Resource);
                if (position.Cap is not null && OccupiedFor(model, position.Tier, chain.Resource) + amount > position.Cap.Value)
                {
                    walk.Add(new ClaimChainStep
                    {
                        Tier = position.Tier,
                        IsOriginStore = false,
                        Outcome = "declared ceiling below demand",
                        Occupied = occupied,
                        Capacity = seam?.Capacity,
                        DeclaredCap = position.Cap,
                    });
                    continue;
                }

                if (seam is null)
                {
                    walk.Add(new ClaimChainStep
                    {
                        Tier = position.Tier,
                        IsOriginStore = false,
                        Outcome = "no opened capacity seam",
                        Occupied = occupied,
                        DeclaredCap = position.Cap,
                    });
                    continue;
                }

                if (seam.Capacity is null || occupied + amount <= seam.Capacity.Value)
                {
                    DeviceLedgerClaim seated = Seat(model, chain.Resource, position.Tier, amount, index, index != firstCopyHolding, atOriginStore: false);
                    walk.Add(new ClaimChainStep
                    {
                        Tier = position.Tier,
                        IsOriginStore = false,
                        Outcome = "seated",
                        Occupied = occupied + amount,
                        Capacity = seam.Capacity,
                        DeclaredCap = position.Cap,
                    });
                    return new ClaimOutcome { Seated = true, Claim = seated, ChainWalk = [.. walk] };
                }

                walk.Add(new ClaimChainStep
                {
                    Tier = position.Tier,
                    IsOriginStore = false,
                    Outcome = "held by earlier claims",
                    Occupied = occupied,
                    Capacity = seam.Capacity,
                    DeclaredCap = position.Cap,
                });
            }

            ClaimOutcome exhausted = new()
            {
                Seated = false,
                ExhaustedAtTier = walk.Last(step => !step.IsOriginStore).Tier,
                ChainWalk = [.. walk],
                Breakdown = BuildBreakdown(model, chain.Resource, amount, walk),
            };
            return exhausted;
        }
    }

    /// <summary>The seated claims one model holds, as the breakdown text a diagnostic reports: each claim's resource, amount with its unit, tier, and chain position.</summary>
    public string BreakdownFor(string model, BudgetResource resource)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);

        lock (_gate)
        {
            List<DeviceLedgerClaim> claims = [.. _claims.Where(claim => claim.Model == model && claim.Resource == resource)];
            if (claims.Count == 0)
            {
                return $"model '{model}' holds no '{BudgetResourceNames.Of(resource)}' claims on this ledger";
            }

            return string.Join("; ", claims.Select(claim => claim.Describe()));
        }
    }

    /// <summary>The unit one resource's claims and seams count in: megabytes for weights and prefill buffers, KV blocks for KV.</summary>
    public static string UnitName(BudgetResource resource) => resource == BudgetResource.Kv ? "KV block" : "megabyte";

    /// <summary>
    /// Reports what one admission's landed reservation needs against the chain this model resolved, after the existing
    /// seams already admitted it. A standing claim is where its model's pool backing lives: a landed sequence rides
    /// inside that seat rather than registering beside it, so an observation never double-counts one model's own memory
    /// and never displaces another model's earlier seating — only what <em>other</em> models hold competes for the tier.
    /// The landing walks its chain analogously to registration against host seams: a copy-holding position with room left
    /// after other claims (or an unbounded seam) seats it; a read-side origin position seats without consuming capacity;
    /// and only a chain whose every consulted host-backed position is filled by earlier claims of others reports
    /// exhaustion. The model's own declared ceilings are shown in breakdowns for context, but this observation does not
    /// re-decide them or the engine pool's live soft cap — those decisions belong to the existing seams. Observations seat
    /// or refuse without registering a claim, so occupancy stays the standing claims' sum.
    /// </summary>
    public ClaimOutcome ObserveAdmission(string model, ResolvedBudgetChain chain, long needed)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        ArgumentNullException.ThrowIfNull(chain);
        ArgumentOutOfRangeException.ThrowIfNegative(needed);

        lock (_gate)
        {
            List<ClaimChainStep> walk = [];

            for (int index = 0; index < chain.Tiers.Count; index++)
            {
                ResolvedBudgetTier position = chain.Tiers[index];

                if (position.IsOriginStore)
                {
                    // The read-side origin store holds the complete set already: a landed sequence rides there without competing for capacity.
                    walk.Add(new ClaimChainStep { Tier = position.Tier, IsOriginStore = true, Outcome = "seated at origin store" });
                    return new ClaimOutcome { Seated = true, ChainWalk = [.. walk] };
                }

                CapacitySeam? seam = SeamFor(position.Tier, chain.Resource);
                long heldByOthers = OccupiedAt(position.Tier, chain.Resource, excludingModel: model);
                if (seam is null)
                {
                    walk.Add(new ClaimChainStep
                    {
                        Tier = position.Tier,
                        IsOriginStore = false,
                        Outcome = "no opened capacity seam",
                        Occupied = heldByOthers,
                        DeclaredCap = position.Cap,
                    });
                    continue;
                }

                if (seam.Capacity is null || heldByOthers + needed <= seam.Capacity.Value)
                {
                    walk.Add(new ClaimChainStep
                    {
                        Tier = position.Tier,
                        IsOriginStore = false,
                        Outcome = "seated",
                        Occupied = heldByOthers + needed,
                        Capacity = seam.Capacity,
                        DeclaredCap = position.Cap,
                    });
                    return new ClaimOutcome { Seated = true, ChainWalk = [.. walk] };
                }

                walk.Add(new ClaimChainStep
                {
                    Tier = position.Tier,
                    IsOriginStore = false,
                    Outcome = "held by earlier claims",
                    Occupied = heldByOthers,
                    Capacity = seam.Capacity,
                    DeclaredCap = position.Cap,
                });
            }

            return new ClaimOutcome
            {
                Seated = false,
                ExhaustedAtTier = walk.Last(step => !step.IsOriginStore).Tier,
                ChainWalk = [.. walk],
                Breakdown = BuildBreakdown(model, chain.Resource, needed, walk),
            };
        }
    }

    /// <summary>
    /// Re-reports one tier's capacity posture — the answer a host gives when the totals behind its opened seams change.
    /// Seated claims stay whole: an earlier claim is never revoked or displaced by a lowered total, only what later
    /// admissions and registrations may still seat tightens to it, so a tier whose standing claims now exceed its stated
    /// ceiling holds them while refusing nothing that already stands there. A seam no caller ever reported stays absent;
    /// the unbounded posture is restated by passing null capacity.
    /// </summary>
    public void UpdateCapacity(BudgetTier tier, BudgetResource resource, long? capacity)
    {
        if (capacity is < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity), "a reported capacity is a ceiling of 0 or greater, or null to state the unbounded posture.");
        }

        lock (_gate)
        {
            if (!_seams.TryGetValue((tier, resource), out CapacitySeam? seam))
            {
                throw new ModelException($"no opened capacity seam backs tier '{tier}' for resource '{resource}': a re-reported total updates only what this ledger accounts against");
            }

            CapacitySeam updated = new() { Tier = tier, Resource = resource, Capacity = capacity };
            _seams[(tier, resource)] = updated;
            for (int index = 0; index < _seamTable.Count; index++)
            {
                if (_seamTable[index].Tier == tier && _seamTable[index].Resource == resource)
                {
                    _seamTable[index] = updated;
                    break;
                }
            }
        }
    }

    /// <summary>
    /// Updates the amount of one model's latest seated working-copy claim for a resource, recording that its own live
    /// engine seam retargeted what that standing seat covers. The claim stays at its tier and registration order: no
    /// earlier or later seating moves, displaces anything, or re-runs first-come selection. Returns false when the model
    /// has no seated working-copy claim for that resource.
    /// </summary>
    public bool UpdateStandingClaimAmount(string model, BudgetResource resource, long newAmount) =>
        UpdateStandingClaimAmount(model, resource, newAmount, atTier: null);

    /// <summary>
    /// Updates the amount of one model's latest seated working-copy claim for a resource, recording that its own live
    /// engine seam retargeted what that standing seat covers. When a tier is named, only that model's claims holding
    /// working copies at that tier are candidates: an engine that seats part of one resource across tiers updates the
    /// seat its live cap governs and leaves every other seat whole. The claim stays at its tier and registration order:
    /// no earlier or later seating moves, displaces anything, or re-runs first-come selection. Returns false when the
    /// model has no seated working-copy claim matching what was named.
    /// </summary>
    public bool UpdateStandingClaimAmount(string model, BudgetResource resource, long newAmount, BudgetTier? atTier)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        ArgumentOutOfRangeException.ThrowIfNegative(newAmount);

        lock (_gate)
        {
            int index = -1;
            for (int candidate = 0; candidate < _claims.Count; candidate++)
            {
                DeviceLedgerClaim claim = _claims[candidate];
                if (claim.Model == model && claim.Resource == resource && !claim.AtOriginStore
                    && (atTier is null || claim.Tier == atTier))
                {
                    index = candidate;
                }
            }

            if (index < 0)
            {
                return false;
            }

            DeviceLedgerClaim current = _claims[index];
            _claims[index] = current with { Amount = newAmount };
            return true;
        }
    }

    /// <summary>
    /// Releases every claim one model holds on this ledger — the answer a host gives when that model's engine is being
    /// torn down. Released claims free their tier seats for later registrations and admissions immediately; nothing already
    /// seated moves, re-runs first-come selection, or changes order otherwise. Returns what was released so the caller can
    /// report it. A model holding no claims releases nothing and reports none.
    /// </summary>
    public IReadOnlyList<DeviceLedgerClaim> ReleaseClaims(string model)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);

        lock (_gate)
        {
            List<DeviceLedgerClaim> released = [];
            for (int index = _claims.Count - 1; index >= 0; index--)
            {
                if (_claims[index].Model == model)
                {
                    released.Add(_claims[index]);
                    _claims.RemoveAt(index);
                }
            }

            released.Reverse(); // reported in the registration order they were seated in
            return released;
        }
    }

    private DeviceLedgerClaim Seat(string model, BudgetResource resource, BudgetTier tier, long amount, int chainPosition, bool demoted, bool atOriginStore)
    {
        DeviceLedgerClaim claim = new()
        {
            Sequence = _nextSequence++,
            Model = model,
            Resource = resource,
            Tier = tier,
            Amount = amount,
            ChainPosition = chainPosition,
            Demoted = demoted,
            AtOriginStore = atOriginStore,
        };

        _claims.Add(claim);
        return claim;
    }

    private long OccupiedAt(BudgetTier tier, BudgetResource resource, string? excludingModel = null)
    {
        long occupied = 0;
        foreach (DeviceLedgerClaim claim in _claims)
        {
            if (claim.Tier == tier && claim.Resource == resource && !claim.AtOriginStore && claim.Model != excludingModel)
            {
                occupied += claim.Amount;
            }
        }

        return occupied;
    }

    private long OccupiedFor(string model, BudgetTier tier, BudgetResource resource)
    {
        long occupied = 0;
        foreach (DeviceLedgerClaim claim in _claims)
        {
            if (claim.Tier == tier && claim.Resource == resource && !claim.AtOriginStore && claim.Model == model)
            {
                occupied += claim.Amount;
            }
        }

        return occupied;
    }

    // The exhaustion breakdown: every position the claim walked, what it holds against its capacity posture, which
    // earlier claims hold it, and the amount no position could seat — the text an operator acts on.
    private string BuildBreakdown(string model, BudgetResource resource, long amount, IReadOnlyList<ClaimChainStep> walk)
    {
        string unit = UnitName(resource);
        List<string> positions = [];

        foreach (ClaimChainStep step in walk)
        {
            string tier = $"'{BudgetTierNames.Of(step.Tier)}'";

            if (step.IsOriginStore)
            {
                positions.Add($"{tier} is this chain's origin store");
                continue;
            }

            if (step.Outcome == "declared ceiling below demand")
            {
                string declared = step.DeclaredCap is null ? "a ceiling" : $"{step.DeclaredCap.Value} {unit}s";
                positions.Add($"{tier} declares {declared} here — less than the {amount} {unit}s this claim needs");
                continue;
            }

            if (step.Outcome == "no opened capacity seam")
            {
                string declaredSuffix = step.DeclaredCap is null ? string.Empty : $" — this model declares {step.DeclaredCap.Value} {unit}s there";
                positions.Add($"{tier} has no opened capacity seam{declaredSuffix}");
                continue;
            }

            IReadOnlyList<DeviceLedgerClaim> holders = [.. _claims.Where(claim => claim.Tier == step.Tier && claim.Resource == resource && !claim.AtOriginStore)];
            string holderText = holders.Count == 0
                ? "no claims"
                : string.Join(", ", holders.Select(claim => $"'{claim.Model}' {claim.Amount} {unit}s"));

            string capacity = step.Capacity is null ? "unbounded" : $"{step.Capacity} {unit}s";
            string declaredCeiling = step.DeclaredCap is null ? string.Empty : $" — this model declares {step.DeclaredCap.Value} {unit}s there";
            positions.Add($"{tier} holds {step.Occupied} of {capacity} ({holderText}){declaredCeiling}");
        }

        return $"model '{model}' budgets for resource '{BudgetResourceNames.Of(resource)}' found no position its chain seats {amount} {unit}s at: " +
               $"chain [{string.Join(", ", walk.Select(step => $"'{BudgetTierNames.Of(step.Tier)}'"))}] — {string.Join("; ", positions)}";
    }
}
