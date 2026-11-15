namespace Lair.Models;

/// <summary>
/// The NUMA spread posture one model's placement declares: <c>auto</c> lets placement choose, <c>local</c> asks for one
/// node, and <c>interleave</c> asks for a spread posture across the nodes the host reports. A posture is a preference
/// placement resolves against the reported topology — it is never a promise that the operating system moved anything.
/// </summary>
public enum NumaSpreadPreference
{
    /// <summary>Placement chooses: the least-loaded reported node, lowest node index as tiebreak. A pinned core set beside it resolves to the node carrying that set.</summary>
    Auto,

    /// <summary>One node carries the model: the node a pinned core set maps to, or the least-loaded node when the model pins nothing.</summary>
    Local,

    /// <summary>A spread posture across every node the host reports — a posture that names nodes, never cores.</summary>
    Interleave,
}

/// <summary>The config vocabulary one placement spread key accepts, in the order a refusal reports it.</summary>
public static class NumaSpreadNames
{
    public const string Auto = "auto";
    public const string Local = "local";
    public const string Interleave = "interleave";

    public static IReadOnlyList<string> DeclaredNames { get; } = [Auto, Local, Interleave];

    public static string Of(NumaSpreadPreference preference) => preference switch
    {
        NumaSpreadPreference.Local => Local,
        NumaSpreadPreference.Interleave => Interleave,
        _ => Auto,
    };

    /// <summary>Reads one config name as a spread posture. An unrecognized name is not a posture: callers report it naming the accepted vocabulary.</summary>
    public static bool TryParse(string name, out NumaSpreadPreference preference)
    {
        switch (name)
        {
            case Auto:
                preference = NumaSpreadPreference.Auto;
                return true;

            case Local:
                preference = NumaSpreadPreference.Local;
                return true;

            case Interleave:
                preference = NumaSpreadPreference.Interleave;
                return true;

            default:
                preference = default;
                return false;
        }
    }

    /// <summary>The accepted set as the text a refusal names it by.</summary>
    public static string AcceptedVocabulary => $"only '{Auto}', '{Local}', or '{Interleave}'";
}

/// <summary>
/// One model's claim as placement reads it: the model it names, the worker share it asks to run with, the spread posture
/// its placement declares, and the core ids it pins. A claim carries no decision about which node it lands on — that is
/// what <see cref="NumaPlacementResolver"/> produces from the ordered claims against the reported posture.
/// </summary>
public sealed record NumaClaim
{
    public required string Model { get; init; }

    /// <summary>The worker share this claim asks to run with — what placement counts against a node when it chooses one.</summary>
    public required int RequestedCores { get; init; }

    /// <summary>The spread posture this model's placement declares. An absent posture reads as auto.</summary>
    public NumaSpreadPreference Spread { get; init; } = NumaSpreadPreference.Auto;

    /// <summary>Pinned logical core ids; empty means the model pins nothing.</summary>
    public IReadOnlySet<int> CoreSet { get; init; } = new HashSet<int>();
}

/// <summary>
/// The placement one claim resolved to: the posture it asked for beside the posture placement actually applied, the node
/// or nodes chosen, the core ids the posture maps onto them, and — where a posture could not be honoured from what the
/// host reported — the honest advisory stating why instead of an emulation. Chosen nodes are reported in node order;
/// an applied posture of <c>local</c> beside a requested <c>auto</c> is a pin that localized the choice.
/// </summary>
public sealed record NumaPlacement
{
    public required string Model { get; init; }

    /// <summary>The spread posture the model's placement declared.</summary>
    public required NumaSpreadPreference Requested { get; init; }

    /// <summary>The posture placement resolved to. A requested auto that a pinned set mapped onto one node applies as local.</summary>
    public required NumaSpreadPreference Applied { get; init; }

    /// <summary>The node or nodes this placement chose, in reported node order. A single-node posture names exactly one.</summary>
    public required IReadOnlyList<int> NodeIndices { get; init; }

    /// <summary>The pinned core ids this posture carries onto the chosen node(s); empty when the model pins nothing.</summary>
    public IReadOnlyList<int> CoreIds { get; init; } = [];

    /// <summary>Why this posture could not be honoured from what the host reported, or null when it was honoured whole.</summary>
    public string? Advisory { get; init; }

    /// <summary>True when this placement spread across more than one reported node.</summary>
    public bool IsSpread => NodeIndices.Count > 1;

    /// <summary>The placement as the text a report names it by: requested against applied posture, chosen node(s), and any advisory.</summary>
    public string Describe() =>
        $"'{Model}' {NumaSpreadNames.Of(Requested)}→{NumaSpreadNames.Of(Applied)} node{(NodeIndices.Count == 1 ? "" : "s")} {string.Join(", ", NodeIndices)}" +
        (CoreIds.Count > 0 ? $" cores [{string.Join(", ", CoreIds)}]" : string.Empty) +
        (Advisory is null ? string.Empty : $" — advisory: {Advisory}");
}

/// <summary>
/// The placement seam that turns ordered model claims into a deterministic placement map over one reported posture.
/// Claim order is the order the claims arrive in — configuration declaration order at every caller — and the map is
/// pure: the same posture and the same claims produce the same placements on every call, with no host state consulted.
/// Auto chooses the least-loaded node, lowest node index as tiebreak, where load is the worker share earlier claims in
/// this run seated on each node beside whatever the caller seeds as already held. Local chooses one node and maps a
/// pinned core set to the node carrying it when the posture reports core ids. Interleave records a spread posture across
/// the reported nodes. A non-empty core set beside interleave is refused naming model, key, and conflict; a pinned set
/// spanning reported nodes is refused for local or auto when the posture reports core ids. Where the posture reports no
/// core ids at all, a pinned posture is carried as an advisory pin rather than emulated as a node mapping.
/// </summary>
public static class NumaPlacementResolver
{
    /// <summary>The placement key whose posture these rules resolve, named verbatim in every refusal.</summary>
    public const string ConfigKey = "placement.numa_spread";

    /// <summary>The placement key a pinned posture declares, named verbatim beside the spread key in every conflict refusal.</summary>
    public const string CoreSetKey = "placement.core_set";

    /// <summary>
    /// The named advisory every surface carries where byte placement cannot be stated from this build: first-touch follows
    /// the workers pinned to the chosen node before they serve, and nothing here prefaults a model file or moves pages on
    /// a host whose placement is advisory.
    /// </summary>
    public const string PagePlacementAdvisory =
        "byte placement is first-touch discipline only: workers pinned to the chosen node before serving decide it, and this build neither prefaults the model file nor moves pages where placement is advisory";

    /// <summary>
    /// Resolves one ordered claim set into its placement map. <paramref name="initialNodeLoads"/> seeds what a caller's
    /// standing accounting already holds per node — a shared ledger's seated shares — so a later claim's auto choice sees
    /// the load earlier claims left rather than starting from an empty machine. Null seeds an empty one.
    /// </summary>
    public static IReadOnlyList<NumaPlacement> Resolve(HostTopology topology, IReadOnlyList<NumaClaim> claims, IReadOnlyDictionary<int, int>? initialNodeLoads = null)
    {
        ArgumentNullException.ThrowIfNull(topology);
        ArgumentNullException.ThrowIfNull(claims);

        if (topology.Nodes.Count == 0)
        {
            throw new ModelException(
                "placement needs a reported posture naming at least one node: an empty posture states no node to place a model on.");
        }

        Dictionary<int, int> load = [];
        foreach (HostNodeCapacity node in topology.Nodes)
        {
            load[node.Index] = initialNodeLoads is not null && initialNodeLoads.TryGetValue(node.Index, out int held) ? Math.Max(held, 0) : 0;
        }

        List<NumaPlacement> placements = [];
        foreach (NumaClaim claim in claims)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(claim.Model);
            ArgumentOutOfRangeException.ThrowIfLessThan(claim.RequestedCores, 1);
            placements.Add(ResolveOne(topology, claim, load));
        }

        return placements;
    }

    private static NumaPlacement ResolveOne(HostTopology topology, NumaClaim claim, Dictionary<int, int> load)
    {
        int[] pinned = [.. claim.CoreSet.Order()];
        bool reportsCoreIds = topology.Nodes.Any(node => node.CoreIds.Count > 0);

        if (claim.Spread == NumaSpreadPreference.Interleave)
        {
            if (pinned.Length > 0)
            {
                throw new ModelException(
                    $"model '{claim.Model}' declares {ConfigKey} '{NumaSpreadNames.Interleave}' with a non-empty {CoreSetKey} [{string.Join(", ", pinned)}]: " +
                    $"a spread posture names nodes, not cores — drop the core set or declare {ConfigKey} '{NumaSpreadNames.Auto}' / '{NumaSpreadNames.Local}' for a single-node posture.");
            }

            IReadOnlyList<HostNodeCapacity> nodes = [.. topology.Nodes];
            foreach (HostNodeCapacity node in nodes)
            {
                load[node.Index] += ShareOf(claim.RequestedCores, nodes.Count, node.Index);
            }

            return new NumaPlacement
            {
                Model = claim.Model,
                Requested = NumaSpreadPreference.Interleave,
                Applied = NumaSpreadPreference.Interleave,
                NodeIndices = [.. nodes.Select(node => node.Index)],
            };
        }

        // local and auto share the single-node rules: a pinned set the posture can map chooses its node outright,
        // a pinned set that spans reported nodes is a conflict, and a pinned set nothing can map rides as an advisory.
        SortedSet<int> carryingNodes = [];
        HostNodeCapacity? carrying = null;
        foreach (int coreId in pinned)
        {
            HostNodeCapacity? node = topology.NodeForCoreId(coreId);
            if (node is null)
            {
                continue; // an id no reported node carries: the posture cannot map it, which reads as advisory rather than as a span
            }

            if (carryingNodes.Add(node.Index))
            {
                carrying = node;
            }
        }

        if (carryingNodes.Count > 1)
        {
            throw new ModelException(
                $"model '{claim.Model}' declares {CoreSetKey} [{string.Join(", ", pinned)}] beside {ConfigKey} '{NumaSpreadNames.Of(claim.Spread)}': " +
                $"the pinned core set spans reported nodes {string.Join(", ", carryingNodes)} — a single-node posture cannot span nodes. " +
                $"Name cores that sit on one node, or declare {ConfigKey} '{NumaSpreadNames.Interleave}' for a spread posture.");
        }

        NumaSpreadPreference applied = claim.Spread;
        IReadOnlyList<int> chosen;
        string? advisory = null;

        if (carryingNodes.Count == 1)
        {
            // The posture mapped the pin: that node carries the model, and a requested auto reads back as the local posture the pin became.
            chosen = [carrying!.Index];
            if (claim.Spread == NumaSpreadPreference.Auto)
            {
                applied = NumaSpreadPreference.Local;
            }
        }
        else
        {
            chosen = [LeastLoadedNode(topology, load).Index];
            if (pinned.Length > 0)
            {
                advisory = reportsCoreIds
                    ? $"the declared {CoreSetKey} [{string.Join(", ", pinned)}] names no node this posture reports: the pin is carried as a pin posture, not as a node mapping"
                    : $"the reported posture states no core ids: the declared {CoreSetKey} [{string.Join(", ", pinned)}] is carried as a pin posture, not as a node mapping";
            }
        }

        foreach (int nodeIndex in chosen)
        {
            load[nodeIndex] += ShareOf(claim.RequestedCores, chosen.Count, nodeIndex);
        }

        return new NumaPlacement
        {
            Model = claim.Model,
            Requested = claim.Spread,
            Applied = applied,
            NodeIndices = chosen,
            CoreIds = [.. pinned],
            Advisory = advisory,
        };
    }

    // The node no claim in this run has loaded as much as another, lowest node index as tiebreak — what an auto posture
    // chooses, and what a local posture with nothing to map falls back to.
    private static HostNodeCapacity LeastLoadedNode(HostTopology topology, Dictionary<int, int> load)
    {
        HostNodeCapacity chosen = topology.Nodes[0];
        int chosenLoad = load[chosen.Index];

        foreach (HostNodeCapacity node in topology.Nodes)
        {
            int nodeLoad = load[node.Index];
            if (nodeLoad < chosenLoad || (nodeLoad == chosenLoad && node.Index < chosen.Index))
            {
                chosen = node;
                chosenLoad = nodeLoad;
            }
        }

        return chosen;
    }

    // One claim's share of one node when a posture spreads across several: a floor split with the remainder going to the
    // lowest node indices, so a five-core interleave over two nodes is three on node 0 and two on node 1 — the same split
    // on every call.
    private static int ShareOf(int requestedCores, int nodeCount, int nodeIndex)
    {
        if (nodeCount <= 1)
        {
            return requestedCores;
        }

        int baseShare = requestedCores / nodeCount;
        int remainder = requestedCores % nodeCount;
        return baseShare + (nodeIndex < remainder ? 1 : 0);
    }
}
