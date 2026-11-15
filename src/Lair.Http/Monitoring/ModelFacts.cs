using Lair.Host.Config;

namespace Lair.Http.Monitoring;

/// <summary>
/// Configuration- and backend-backed facts about the model a bus observes, attached once by the owning service
/// (and re-pushed when hot reload changes them). They feed only snapshot payload fields — sampling cost is one
/// volatile record read per tick — so an unattached bus publishes exactly its v1 shape with nulls in these slots.
/// The trailing quota and prefill-lane fields default to the uncapped, shared posture so records built without them
/// stay well-formed; the owning service fills them from what its configuration and opened lane posture resolve.
/// </summary>
public sealed record ModelFacts(
    string Type,
    int WorkerThreads,
    string? CoreSet,
    int? ThreadQuota,
    string DecodeDevice,
    string PrefillDevice,
    string KvResidency,
    string PrefillKernel,

    /// <summary>The quant formats the served model's dense weight matrices carry: what a prefill selection needs kernels for, and what an unsupported file reports its refusal against.</summary>
    string MatmulWeightFormats,

    /// <summary>The admission concurrency cap configured for this model, or null when nothing bounds how many requests it may hold admitted at once. A live edit of the cap re-pushes it with the rest of the facts.</summary>
    int? MaxConcurrentRequests = null,

    /// <summary>The resolved execution route's segments in order, already in wire shape, or null when the attaching side has no resolved route to report.</summary>
    IReadOnlyList<ModelRouteSegment>? RouteSegments = null,

    /// <summary>Which prefill-lane posture the model opened with: <c>shared</c> or <c>dedicated</c>.</summary>
    string PrefillLane = PlacementPrefillLaneValues.Shared,

    /// <summary>The effective worker claim a dedicated split was derived from — zero when the posture shares one backend.</summary>
    int PrefillClaimThreads = 0,

    /// <summary>The decode lane's worker cap as resolved — zero when shared.</summary>
    int PrefillDecodeShareThreads = 0,

    /// <summary>The prefill lane's own worker cap — zero when shared.</summary>
    int PrefillLaneShareThreads = 0,

    /// <summary>The decode lane's pinned core ids as compressed ranges, null when the posture pins nothing.</summary>
    string? PrefillDecodeCoreSet = null,

    /// <summary>The prefill lane's pinned core ids as compressed ranges, null when the posture pins nothing.</summary>
    string? PrefillLaneCoreSet = null,

    /// <summary>Total bytes of mutable activation storage the dedicated lane family holds at open — zero for a shared posture.</summary>
    long PrefillActivationBytes = 0,

    /// <summary>The row width the opened engine resolved for its prompt windows, before any declared prefill-buffer ceiling.</summary>
    int PrefillRowWidth = 0,

    /// <summary>The row width the opened engine actually runs its prompt windows at: the resolved width when nothing declared bounds it, narrowed to what a prefill-buffer ceiling allows otherwise.</summary>
    int EffectivePrefillRowWidth = 0,

    /// <summary>Total mutable activation bytes the opened engine holds for its prompt windows' working set — the lane family where one opened, the effective row window's cost beside its fixed tail where the posture shares one backend.</summary>
    long PrefillWorkingSetBytes = 0,

    /// <summary>What narrowed the engine's row window, stated by the ceiling that binds it: tier and cap, one sub-batch row's activation bytes, and requested against allowed width — null when no declared ceiling bound the window.</summary>
    string? PrefillRowClamp = null,

    /// <summary>What the host device ledger has seated claims against, one entry per tier position, or null when the ledger has seated none.</summary>
    IReadOnlyList<LedgerOccupancyEntry>? LedgerOccupancy = null,

    /// <summary>The worker share this model asked to run with — its engine thread choice capped by its own quota — or null until the owning service attaches compute facts.</summary>
    int? ComputeRequestedThreads = null,

    /// <summary>The logical processors that claim was granted: below requested where an earlier seat clamped it, equal when nothing contended for the host's capacity.</summary>
    int? ComputeGrantedThreads = null,

    /// <summary>True when this model opted into sharing cores another claim holds; null until facts attach or while the disjoint posture stands.</summary>
    bool? CoreOvercommit = null,

    /// <summary>The models whose participation rides logical processors a grant already holds — null when every seated share is its own.</summary>
    IReadOnlyList<string>? SharedCoreModels = null,

    /// <summary>The single node this engine's worker claim seated on, or null for a posture that spread across nodes and until the owning service attaches placement facts.</summary>
    int? PlacementNode = null,

    /// <summary>Every node the opened placement chose, as a comma-separated list of node positions — the same reading a spread posture reports.</summary>
    string? PlacementNodes = null,

    /// <summary>The spread posture placement applied when the engine opened (<c>auto</c>, <c>local</c>, <c>interleave</c>), or null until placement facts attach.</summary>
    string? PlacementSpread = null,

    /// <summary>The spread posture the model's configuration declared, beside the posture placement resolved it to.</summary>
    string? PlacementSpreadRequested = null,

    /// <summary>The logical processors this model's grant seated as its own on each node, as <c>node:cores</c> pairs — null until compute facts attach or when the claim seated none.</summary>
    string? PlacementNodeAllocations = null,

    /// <summary>Why the opened posture could not be honoured whole from what the host reported, or null when it was honoured.</summary>
    string? PlacementAdvisory = null);
