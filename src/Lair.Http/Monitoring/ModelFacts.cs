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
    long PrefillActivationBytes = 0);
