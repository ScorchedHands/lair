namespace Lair.Models;

/// <summary>
/// The dedicated prefill-lane posture a model's placement resolves to at load: the worker claim it was split from,
/// each lane's fixed worker cap, and — when the model pins cores — which declared core ids belong to which lane. A
/// shared-posture model carries no layout; its forwards keep running on one execution backend exactly as before any
/// lane vocabulary existed. Worker pools, masks, activation buffers, and collision posture are load-time facts: this
/// value is resolved once when the engine opens, never re-split while a model runs.
/// </summary>
public sealed record PrefillLaneLayout
{
    /// <summary>The served model's name, named in every refusal and diagnostic this layout reports.</summary>
    public required string ModelName { get; init; }

    /// <summary>The effective worker claim the split was derived from at load: min(engine thread choice, placement quota).</summary>
    public int EffectiveClaimThreads { get; init; }

    /// <summary>Prefill-lane worker cap resolved at load. Fixed for the model's lifetime: a live edit never re-splits it.</summary>
    public required int PrefillShareThreads { get; init; }

    /// <summary>Decode-lane worker cap as first derived (the claim minus the prefill share). A valid live quota edit may retarget this side alone.</summary>
    public required int DecodeShareThreads { get; init; }

    /// <summary>The declared core ids claimed by the decode lane, ascending. Empty when the model pins nothing: an unpinned posture bounds participation only.</summary>
    public IReadOnlyList<int> DecodeCoreIds { get; init; } = [];

    /// <summary>The declared core ids claimed by the prefill lane, ascending — disjoint from <see cref="DecodeCoreIds"/>, with the two covering every pinned id exactly once. Empty when unpinned.</summary>
    public IReadOnlyList<int> PrefillCoreIds { get; init; } = [];

    /// <summary>True when this layout partitions a declared core set: both lane masks are disjoint and their union is the full claimed set.</summary>
    public bool IsPinned => DecodeCoreIds.Count > 0 || PrefillCoreIds.Count > 0;
}

/// <summary>
/// Resolves a dedicated lane posture from placement values alone, and owns every rule that resolution enforces. The
/// default split takes floor(one quarter) of the model's effective worker claim for the prefill side — never less than
/// one worker when the claim can give both sides one — and leaves the rest to decode: an eight-worker claim becomes
/// six decoding plus two prefilling. A pinned posture then partitions the declared core ids disjointly so every id is
/// claimed by exactly one lane, with the prefill side taking a deterministic lower-numbered subset sized from its share
/// of the claim and the remainder going to decode; an unpinned posture bounds worker participation only. A claim or a
/// pinned set that cannot give both lanes at least one worker fails here naming the model, the placement key behind the
/// posture, and what to change — nothing opens on a half-lane engine. Nothing in this resolution executes a lane: it is
/// the value set an open call builds its two execution backends from.
/// </summary>
public static class ModelPrefillLanes
{
    /// <summary>The placement key whose dedicated posture resolves to a layout, named verbatim in every refusal.</summary>
    public const string ConfigKey = "placement.prefill_lane";

    private const int PrefillShareQuotient = 4; // the default split takes floor(1/4) of the claim for prefill

    /// <summary>
    /// Resolves one dedicated posture. Callers supply exactly what their configuration states: the engine's own thread
    /// choice (0-auto already resolved to a concrete count), the model's placement quota, and its pinned core set —
    /// cross-model disjointness of those sets is validated where models are compared, not here. Every refusal names the
    /// model, this key, and what must change instead: widen the claim, pin at least two cores, or run shared.
    /// </summary>
    public static PrefillLaneLayout ResolveDedicated(string modelName, int engineThreadsChoice, int? threadQuota, IReadOnlySet<int> coreSet)
    {
        ArgumentNullException.ThrowIfNull(coreSet);

        if (engineThreadsChoice < 1 || threadQuota is < 1)
        {
            throw new ModelException(
                $"model '{modelName}' declares {ConfigKey} 'dedicated' with a worker claim that cannot be resolved: the engine's " +
                $"thread choice must resolve to at least one worker and a declared quota must be 1 or greater — fix 'engine.threads'/placement before opening this model.");
        }

        int claim = Math.Min(engineThreadsChoice, threadQuota ?? engineThreadsChoice);

        // The default split: floor(one quarter) of the claim goes to prefill, never below one worker; decode keeps the rest.
        int prefillShare = Math.Max(1, claim / PrefillShareQuotient);
        int decodeShare = claim - prefillShare;

        if (decodeShare < 1)
        {
            throw new ModelException(
                $"model '{modelName}' declares {ConfigKey} 'dedicated' but its effective worker claim of {claim} " +
                $"{DescribeQuota(threadQuota, engineThreadsChoice)} cannot support both lane shares: a dedicated posture needs at least one " +
                $"prefill and one decode worker. Raise the thread quota or 'engine.threads', or set {ConfigKey} to 'shared'.");
        }

        int[] pinnedIds = [.. coreSet.Order()];
        IReadOnlyList<int> decodeCores = [];
        IReadOnlyList<int> prefillCores = [];

        if (pinnedIds.Length > 0)
        {
            // A pinned posture partitions the declared ids disjointly: the union must stay the full claimed set, so each
            // side needs at least one id. The prefill mask takes a deterministic lower-numbered subset sized from its
            // share of the claim (never empty when both sides can have one); decode keeps every remaining higher id.
            int prefillCoreCount = Math.Max(1, (int)((long)pinnedIds.Length * prefillShare / claim));

            if (pinnedIds.Length - prefillCoreCount < 1)
            {
                throw new ModelException(
                    $"model '{modelName}' declares {ConfigKey} 'dedicated' but its pinned core set [{string.Join(", ", pinnedIds)}] " +
                    "cannot partition disjointly into both lane masks: a dedicated posture needs at least two declared cores so each " +
                    $"lane claims one. Pin more cores, drop the pinning, or set {ConfigKey} to 'shared'.");
            }

            prefillCores = pinnedIds[..prefillCoreCount];
            decodeCores = pinnedIds[prefillCoreCount..];
        }

        return new PrefillLaneLayout
        {
            ModelName = modelName,
            EffectiveClaimThreads = claim,
            PrefillShareThreads = prefillShare,
            DecodeShareThreads = decodeShare,
            DecodeCoreIds = decodeCores,
            PrefillCoreIds = prefillCores,
        };
    }

    /// <summary>
    /// Maps a live thread-quota edit onto an already-open dedicated posture: the desired claim re-resolves as min(engine
    /// thread choice, quota) and the decode side alone follows it — the load-resolved prefill share stays fixed because its
    /// worker pool, mask, activation buffers, and collision posture were established when the engine opened. The edit maps
    /// only while both lanes keep at least one worker; anything else reports why so the apply path can refuse it as a lane-
    /// layout change that takes restart instead of pretending to re-split live state.
    /// </summary>
    public static bool TryMapLiveThreadQuota(PrefillLaneLayout layout, int engineThreadsChoice, int? threadQuota, out int decodeShareThreads)
    {
        ArgumentNullException.ThrowIfNull(layout);

        if (engineThreadsChoice < 1 || threadQuota is < 1)
        {
            decodeShareThreads = 0; // a claim that cannot resolve maps to no lane posture at all: the caller reports it as unmappable
            return false;
        }

        int claim = Math.Min(engineThreadsChoice, threadQuota ?? engineThreadsChoice);
        int decodeShare = claim - layout.PrefillShareThreads;

        if (decodeShare < 1)
        {
            decodeShareThreads = 0; // the fixed prefill share plus this claim leave no worker for decode — unmappable while the model stays open
            return false;
        }

        decodeShareThreads = decodeShare;
        return true;
    }

    private static string DescribeQuota(int? threadQuota, int engineThreadsChoice) =>
        threadQuota is null
            ? $"(engine threads {engineThreadsChoice}, no placement quota)"
            : $"(quota {threadQuota} against an 'engine.threads' choice of {engineThreadsChoice})";
}
