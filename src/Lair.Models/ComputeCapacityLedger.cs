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

    /// <summary>The model as the text a diagnostic names this grant by: requested against granted, with its own share and overlap posture stated.</summary>
    public string Describe() =>
        $"'{Model}' requested {RequestedCores} core{(RequestedCores == 1 ? "" : "s")} / granted {GrantedCores}" +
        (SharesHeldCores ? $" ({ExclusiveCores} exclusive, shared with earlier grants)" : string.Empty);

    /// <summary>Grant-versus-request as one report field: what was asked for against what this claim runs with.</summary>
    public string GrantVersusRequest() => $"{RequestedCores}/{GrantedCores}";
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
/// Host-scoped compute accounting over the logical processors a host reports: every model's worker claim registers against that
/// total in registration order, first-come-first-served. Earlier claims are never revoked, preempted, or displaced — an oversubscribed
/// later claim is clamped to what its predecessors left (a partial grant), and only a model that opts into core overcommit keeps the
/// whole share it asked for while overlapping cores another grant holds. Per-node capacity stays reported through <see cref="Topology"/>;
/// which node each share draws from is placement's decision, so this ledger seats shares against the host total without choosing nodes.
/// </summary>
public sealed class ComputeCapacityLedger
{
    private readonly object _gate = new();
    private readonly List<ComputeGrant> _grants = [];
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
    /// Registers one model's worker claim, in the caller's registration order. The claim takes what earlier claims left as its own share;
    /// asking for more than that is a partial grant naming requested against granted, never a refusal and never a displacement of an
    /// earlier claim. A claim that opts into overcommit keeps the whole share it asked for and records the overlap with standing grants.
    /// When nothing is left to seat, the model still runs — at one worker riding what stands there — because compute accounting has no
    /// authority to refuse a served model, only to report what its participation costs the host.
    /// </summary>
    public ComputeGrant RegisterClaim(string model, int requestedCores, bool allowOvercommit = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        ArgumentOutOfRangeException.ThrowIfLessThan(requestedCores, 1);

        lock (_gate)
        {
            if (_grants.Exists(claim => claim.Model == model))
            {
                throw new ModelException($"model '{model}' already holds a compute grant on this ledger: one model claims its worker share once.");
            }

            int remaining = Math.Max(TotalCores - _exclusiveCores, 0);
            int exclusive = Math.Min(requestedCores, remaining);
            bool sharesHeldCores;
            int granted;

            if (allowOvercommit)
            {
                // An opted-in claim keeps what it asked for: whatever capacity left could not seat comes from cores standing grants already hold.
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
            };

            _grants.Add(seated);
            _exclusiveCores += exclusive;
            return seated;
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
