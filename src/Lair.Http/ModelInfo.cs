using Lair.Models;

namespace Lair.Http;

/// <summary>
/// What a served model advertises: the id requests must name, the effective context window, the KV pool size,
/// the dominant tensor quantization, the architecture, the vocabulary size, the tool output parser and reasoning
/// visibility policy resolved for this model at load, the sparse-FFN strategy its plan resolved to for a model whose
/// plan carries an expert-routed block, and the admission concurrency cap configured for this model — null when nothing
/// bounds how many requests it may hold admitted at once. Both policy fields default to the configuration's own defaults so records
/// built without a resolved plan stay well-formed; every engine-opened record carries its resolved values instead.
/// </summary>
public sealed record ModelInfo(
    string Id,
    int ContextLength,
    int KvBlocks,
    string Quant,
    string Architecture,
    int VocabSize,
    string ToolParser = "auto",
    string Reasoning = "keep",
    string? SparseFfnStrategy = null,
    int? MaxConcurrentRequests = null);

/// <summary>
/// The wire name of each resolved sparse-FFN strategy, shared by the load log and the advertised model facts. A dense
/// model carries no value at all: these names exist only for a block that routes experts.
/// </summary>
public static class SparseFfnStrategyNames
{
    public const string DenseResident = "dense_resident";
    public const string LowRankSharedBasis = "low_rank_shared_basis";

    public static string Of(SparseFfnStrategy strategy) => strategy switch
    {
        SparseFfnStrategy.LowRankSharedBasis => LowRankSharedBasis,
        _ => DenseResident,
    };
}

/// <summary>
/// The wire name of each resolved weight-residency tier, shared by the advertised route segments and the monitor's
/// copy of them. A tier is reported as the lowercase name a placement declaration uses for it.
/// </summary>
public static class WeightTierNames
{
    public const string Dram = "dram";
    public const string Vram = "vram";

    public static string Of(WeightTier tier) => tier switch
    {
        WeightTier.Vram => Vram,
        _ => Dram,
    };
}

/// <summary>
/// One resolved route segment as the wire reports it: the compute device it runs on, where its weights stay resident
/// while it computes, the batch cap a step may advance on it, and its quota as resolved — a thread cap (null when the
/// segment runs at the engine's own thread choice) and its pinned core ids rendered as compressed ranges, absent when
/// it pins nothing. Ordered as the route orders its segments; the same record shape rides <c>/v1/models</c> and the
/// monitor snapshot's copy of the route, with one spelling on every surface because each field names itself.
/// </summary>
public sealed record ModelRouteSegment(
    [property: System.Text.Json.Serialization.JsonPropertyName("device_id")] string DeviceId,
    [property: System.Text.Json.Serialization.JsonPropertyName("weight_tier")] string WeightTier,
    [property: System.Text.Json.Serialization.JsonPropertyName("batch_cap")] int BatchCap,
    [property: System.Text.Json.Serialization.JsonPropertyName("thread_quota")] int? ThreadQuota,
    [property: System.Text.Json.Serialization.JsonPropertyName("core_set")] string? CoreSet);

/// <summary>
/// One recent admission refusal as the monitor reports it: the limit kind that blocked the submission, the model it
/// was submitting to, and the concise reason the refusal carries. A diagnostic view of refusals the request surface
/// already answered — it never changes what a refused request's own response says. Each field names itself so the
/// nested shape reads alike wherever the entry is serialized.
/// </summary>
public sealed record AdmissionRefusalEntry(
    [property: System.Text.Json.Serialization.JsonPropertyName("kind")] string Kind,
    [property: System.Text.Json.Serialization.JsonPropertyName("model")] string Model,
    [property: System.Text.Json.Serialization.JsonPropertyName("reason")] string Reason);
