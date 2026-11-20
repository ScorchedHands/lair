namespace Lair.Models;

/// <summary>
/// One model's logical-processor share as this ledger seated it: what the model asked to run with, what it holds as its own, and
/// whether any part of its participation rides cores another grant already holds. A later oversubscribed claim is clamped to what
/// earlier claims left rather than refused or allowed to preempt them; an opt-in overlap keeps the share the model asked for and
/// records the overlap instead of hiding it. Nothing here decides which node a share draws from — that belongs to placement, not
/// accounting — so grants are seated against the host's reported logical-processor total in registration order.
/// </summary>
public sealed record ComputeGrant
{
    /// <summary>The position this claim took in registration order.</summary>
    public required int Sequence { get; init; }

    public required string Model { get; init; }

    /// <summary>The share the model asked to run with: its engine thread choice, capped by its own quota where it declared one.</summary>
    public required int RequestedCores { get; init; }

    /// <summary>The logical processors this claim participates with. Never zero: a claim that finds nothing left runs beside standing grants rather than being refused.</summary>
    public required int GrantedCores { get; init; }

    /// <summary>The part of that participation taken from capacity ahead of later claims. Clamped to what earlier claims left, so it never exceeds the reported total across all claims.</summary>
    public required int ExclusiveCores { get; init; }

    /// <summary>True when some or all of this claim's participation rides logical processors another grant already holds — an opt-in overlap, or a share seated after capacity ran out.</summary>
    public bool SharesHeldCores { get; init; }

    /// <summary>The logical processors this claim took as its own on each reported node, keyed by node index and ordered by node. Empty when the claim seated none as its own.</summary>
    public IReadOnlyDictionary<int, int> NodeAllocations { get; init; } = new Dictionary<int, int>();

    /// <summary>The model as the text a diagnostic names this grant by: requested against granted, with its own share and overlap posture stated.</summary>
    public string Describe() =>
        $"'{Model}' requested {RequestedCores} core{(RequestedCores == 1 ? "" : "s")} / granted {GrantedCores}" +
        (SharesHeldCores ? $" ({ExclusiveCores} exclusive, shared with earlier grants)" : string.Empty) +
        (NodeAllocations.Count > 0 ? $" [{NodeAllocationText()}]" : string.Empty);

    /// <summary>Grant-versus-request as one report field: what was asked for against what this claim runs with.</summary>
    public string GrantVersusRequest() => $"{RequestedCores}/{GrantedCores}";

    /// <summary>The per-node share as one report field: each node this claim seated on beside what it took there.</summary>
    public string NodeAllocationText() => string.Join(",", NodeAllocations.OrderBy(entry => entry.Key).Select(entry => $"{entry.Key}:{entry.Value}"));
}

/// <summary>
/// What a declared tier total reads against the reported topology: each declared number beside what the host stated, and whether
/// they agree. A posture that cannot be compared states why rather than pretending agreement — byte totals nothing reported are not
/// a mismatch with a declared ceiling, they are an answer this host never gave.
/// </summary>
public sealed record DeclaredTotalsReport(
    long? DeclaredCores,
    int ReportedCores,
    bool CoresAgree,
    long? DeclaredMemoryMb,
    long? ReportedMemoryMb)
{
    /// <summary>True when a declared core total disagrees with the reported logical-processor total — warn-and-accept territory.</summary>
    public bool ReportsCoreMismatch => DeclaredCores is not null && !CoresAgree;

    /// <summary>True when both sides stated bytes and those numbers disagree. A byte posture nothing reported reads as incomparable, never as a mismatch.</summary>
    public bool ReportsByteMismatch => DeclaredMemoryMb is not null && ReportedMemoryMb is not null && DeclaredMemoryMb.Value != ReportedMemoryMb.Value;

    /// <summary>The comparison as the text a warning line reports: each declared total beside what was reported, or why nothing compared.</summary>
    public string Describe()
    {
        List<string> parts = [];

        if (DeclaredCores is null)
        {
            parts.Add("declared no core total");
        }
        else
        {
            parts.Add(CoresAgree
                ? $"declared cores {DeclaredCores.Value} match reported {ReportedCores}"
                : $"declared cores {DeclaredCores.Value} against reported {ReportedCores}");
        }

        if (DeclaredMemoryMb is null)
        {
            parts.Add("declared no byte total");
        }
        else if (ReportedMemoryMb is null)
        {
            parts.Add($"declared megabytes {DeclaredMemoryMb.Value} against unreported node bytes");
        }
        else
        {
            parts.Add(ReportsByteMismatch
                ? $"declared megabytes {DeclaredMemoryMb.Value} against reported {ReportedMemoryMb.Value}"
                : $"declared megabytes {DeclaredMemoryMb.Value} match reported {ReportedMemoryMb.Value}");
        }

        return string.Join("; ", parts);
    }
}

/// <summary>
/// Host-scoped compute accounting over the logical processors a host reports: every model's worker claim registers in
/// registration order, first-come-first-served. Earlier claims are never revoked, preempted, or displaced — an oversubscribed
/// later claim is clamped to what its predecessors left (a partial grant), and only a model that opts into core overcommit keeps
/// the whole share it asked for while overlapping cores another grant holds. Claims seat on the nodes their placement chose:
/// a placement names the node or nodes a share draws from, capacity is tracked per reported node, and a claim clamps to what
/// its chosen nodes left. A claim registered without a placement spreads across every reported node in node order, which is
/// exactly the host-total accounting this ledger had before per-node seats existed — on a one-node posture the two readings
/// are the same numbers.
/// </summary>
public sealed class ComputeCapacityLedger
{
    private readonly object _gate = new();
    private readonly List<ComputeGrant> _grants = [];
    private readonly Dictionary<int, int> _nodeExclusive = [];
    private int _exclusiveCores;

    /// <summary>Opens accounting over one reported compute posture.</summary>
    public ComputeCapacityLedger(HostTopology topology)
    {
        ArgumentNullException.ThrowIfNull(topology);
        if (topology.Nodes.Count == 0 || topology.TotalLogicalProcessors < 1)
        {
            throw new ModelException(
                "a compute capacity ledger needs a reported posture naming at least one node carrying logical processors: an empty posture states no capacity to account against.");
        }

        foreach (HostNodeCapacity node in topology.Nodes)
        {
            if (node.LogicalProcessors < 1)
            {
                throw new ModelException($"reported node {node.Index} carries no logical processors: a reported posture names what each node holds.");
            }

            _nodeExclusive[node.Index] = 0;
        }

        Topology = topology;
    }

    /// <summary>The compute posture this ledger accounts against, exactly as its caller reported it.</summary>
    public HostTopology Topology { get; }

    /// <summary>The logical processors every reported node carries together — the total a claim counts against.</summary>
    public int TotalCores => Topology.TotalLogicalProcessors;

    /// <summary>The logical processors no seated grant has taken as its own. A shared overlap never consumes this.</summary>
    public int RemainingCores
    {
        get
        {
            lock (_gate)
            {
                return Math.Max(TotalCores - _exclusiveCores, 0);
            }
        }
    }

    /// <summary>The logical processors no seated grant has taken as its own on one reported node. A node this posture does not report holds nothing.</summary>
    public int RemainingCoresForNode(int nodeIndex)
    {
        lock (_gate)
        {
            HostNodeCapacity? node = Topology.Nodes.FirstOrDefault(candidate => candidate.Index == nodeIndex);
            return node is null ? 0 : Math.Max(node.LogicalProcessors - _nodeExclusive[nodeIndex], 0);
        }
    }

    /// <summary>Every reported node beside the logical processors seated grants have taken as their own on it, in node order — what placement reads as load when it chooses a node.</summary>
    public IReadOnlyDictionary<int, int> NodeAllocations
    {
        get
        {
            lock (_gate)
            {
                Dictionary<int, int> held = [];
                foreach (HostNodeCapacity node in Topology.Nodes)
                {
                    held[node.Index] = _nodeExclusive[node.Index];
                }

                return held;
            }
        }
    }

    /// <summary>Every seated grant so far, in registration order.</summary>
    public IReadOnlyList<ComputeGrant> Grants
    {
        get
        {
            lock (_gate)
            {
                return [.. _grants];
            }
        }
    }

    /// <summary>The grant one model holds, or null when it has registered none.</summary>
    public ComputeGrant? GrantFor(string model)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);

        lock (_gate)
        {
            return _grants.LastOrDefault(claim => claim.Model == model);
        }
    }

    /// <summary>The models whose participation rides logical processors another grant holds, in registration order — empty when every seated share is its own.</summary>
    public IReadOnlyList<string> ModelsSharingHeldCores
    {
        get
        {
            lock (_gate)
            {
                return [.. _grants.Where(claim => claim.SharesHeldCores).Select(claim => claim.Model)];
            }
        }
    }

    /// <summary>
    /// Registers one model's worker claim without a placement, in the caller's registration order: the claim spreads across
    /// every reported node in node order, which is the host-total accounting this ledger has always done — on a one-node
    /// posture, the same numbers a placement would produce. The claim takes what earlier claims left as its own share; asking
    /// for more than that is a partial grant naming requested against granted, never a refusal and never a displacement of an
    /// earlier claim. A claim that opts into overcommit keeps the whole share it asked for and records the overlap with standing grants.
    /// When nothing is left to seat, the model still runs — at one worker riding what stands there — because compute accounting has no
    /// authority to refuse a served model, only to report what its participation costs the host.
    /// </summary>
    public ComputeGrant RegisterClaim(string model, int requestedCores, bool allowOvercommit = false) =>
        RegisterClaim(model, requestedCores, placement: null, allowOvercommit);

    /// <summary>
    /// Registers one model's worker claim on the nodes its placement chose. Capacity is tracked per reported node: the claim
    /// fills its chosen nodes in node order, clamping to what each of those nodes still has unclaimed, so a later claim
    /// never displaces an earlier seat — it clamps to what its chosen nodes left, and a claim whose chosen nodes are full runs
    /// at one worker riding what stands there. A pinned posture's node is authoritative: this seam moves a claim to another
    /// node for neither a full seat nor a partial one. Every node the claim seated on is recorded in the grant. A null placement
    /// registers today's host-total spread across every reported node.
    /// </summary>
    public ComputeGrant RegisterClaim(string model, int requestedCores, NumaPlacement? placement, bool allowOvercommit = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        ArgumentOutOfRangeException.ThrowIfLessThan(requestedCores, 1);

        if (placement is not null && placement.Model != model)
        {
            throw new ModelException(
                $"model '{model}' registers a claim against the placement derived for '{placement.Model}': a claim seats on the placement resolved for it.");
        }

        lock (_gate)
        {
            if (_grants.Exists(claim => claim.Model == model))
            {
                throw new ModelException($"model '{model}' already holds a compute grant on this ledger: one model claims its worker share once.");
            }

            List<int> chosen = placement?.NodeIndices.Count > 0
                ? [.. placement.NodeIndices]
                : [.. Topology.Nodes.Select(node => node.Index)]; // no placement named: today's posture spreads across every reported node

            foreach (int nodeIndex in chosen)
            {
                if (!_nodeExclusive.ContainsKey(nodeIndex))
                {
                    throw new ModelException(
                        $"model '{model}' places on node {nodeIndex} this posture does not report ({Topology.Describe()}): a claim seats on nodes the host named.");
                }
            }

            int left = requestedCores;
            Dictionary<int, int> allocations = [];
            int exclusive = 0;

            // The claim fills its chosen nodes in node order, taking what each still has unclaimed and handing the rest to the
            // next chosen node: the same first-come-first-served clamp this ledger has always done, now stated node by node.
            foreach (int nodeIndex in chosen)
            {
                int room = Math.Max(Topology.Nodes.First(node => node.Index == nodeIndex).LogicalProcessors - _nodeExclusive[nodeIndex], 0);
                int taken = Math.Min(left, room);

                if (taken == 0)
                {
                    continue;
                }

                allocations[nodeIndex] = taken;
                exclusive += taken;
                left -= taken;
            }

            bool sharesHeldCores;
            int granted;

            if (allowOvercommit)
            {
                // An opted-in claim keeps what it asked for: whatever capacity its chosen nodes left could not seat comes from cores standing grants already hold.
                granted = requestedCores;
                sharesHeldCores = exclusive < requestedCores && _grants.Count > 0;
            }
            else if (exclusive == 0)
            {
                // Nothing is left to seat: the model runs beside what stands there rather than being refused, and that overlap is reported.
                granted = 1;
                sharesHeldCores = _grants.Count > 0;
            }
            else
            {
                granted = exclusive;
                sharesHeldCores = false;
            }

            ComputeGrant seated = new()
            {
                Sequence = _grants.Count,
                Model = model,
                RequestedCores = requestedCores,
                GrantedCores = granted,
                ExclusiveCores = exclusive,
                SharesHeldCores = sharesHeldCores,
                NodeAllocations = allocations,
            };

            _grants.Add(seated);
            _exclusiveCores += exclusive;
            foreach (KeyValuePair<int, int> allocation in allocations)
            {
                _nodeExclusive[allocation.Key] += allocation.Value;
            }

            return seated;
        }
    }

    /// <summary>
    /// Releases one model's seated worker share — the answer a host gives when that model's engine is being torn down. The
    /// grant leaves the registration table and its exclusive cores return to what later claims may seat against, per node;
    /// every other claim keeps its position, clamp, and overlap posture exactly as it registered them (a clamped share does
    /// not widen itself because capacity came free). Returns true when a grant was released. A model holding no grant
    /// releases nothing and reports false.
    /// </summary>
    public bool ReleaseGrant(string model)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);

        lock (_gate)
        {
            int index = _grants.FindIndex(claim => claim.Model == model);
            if (index < 0)
            {
                return false;
            }

            ComputeGrant released = _grants[index];
            _grants.RemoveAt(index);
            _exclusiveCores -= released.ExclusiveCores;
            foreach (KeyValuePair<int, int> allocation in released.NodeAllocations)
            {
                _nodeExclusive[allocation.Key] -= allocation.Value;
            }

            return true;
        }
    }

    /// <summary>
    /// Reads declared tier totals against this posture. Nothing here accepts or rejects a configuration: a mismatch is reported so its
    /// owner can log it warn-and-accept, and a byte total the host never reported reads as incomparable rather than as disagreement with
    /// what an operator declared.
    /// </summary>
    public DeclaredTotalsReport CheckDeclaredTotals(long? declaredDramCores, long? declaredDramMb) => new(
        declaredDramCores,
        TotalCores,
        CoresAgree: declaredDramCores is null || declaredDramCores.Value == TotalCores,
        declaredDramMb,
        Topology.ReportedMemoryMb);

    /// <summary>The breakdown one diagnostic reports this ledger's posture through: what it accounts against and every seated grant.</summary>
    public string Breakdown()
    {
        lock (_gate)
        {
            if (_grants.Count == 0)
            {
                return $"compute capacity {Topology.Describe()} holds no claims";
            }

            return $"compute capacity {Topology.Describe()} — " +
                   $"{string.Join("; ", _grants.Select(claim => claim.Describe()))} / remaining {RemainingCores} cores";
        }
    }
}
