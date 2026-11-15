using System.Text.Json;
using System.Text.Json.Serialization;

namespace Lair.Http.Monitoring;

/// <summary>
/// Wire vocabulary for the versioned monitor stream (schema v1). Every frame on <c>/monitor/stream</c> is one of
/// these event names, carries a server-scoped incrementing SSE id, and its payload repeats the schema version —
/// as <c>metrics_events_version</c> inside snapshot payloads (the snapshots' established <c>metrics_</c> field
/// vocabulary) and plain <c>events_version</c> on every other frame. Later versions add fields without reshaping
/// v1 ones, so clients can branch on the version marker alone.
/// </summary>
public static class MonitorEventNames
{
    /// <summary>A per-model engine snapshot; payload is a <see cref="MetricsSnapshot"/>.</summary>
    public const string Snapshot = "snapshot";

    /// <summary>A generation entered the scheduler: its id, model, endpoint kind, and prompt size are known from here on.</summary>
    public const string RequestStart = "request_start";

    /// <summary>An in-flight request advanced: generated-so-far count, elapsed time, current tok/s rate, and sampled tok/s series.</summary>
    public const string RequestUpdate = "request_update";

    /// <summary>A generation finished with a status (eos, length, stop literal, cancellation, or failure) and its full tok/s sample series.</summary>
    public const string RequestComplete = "request_complete";

    /// <summary>A connect-time state dump: the current in-flight request set plus the recent completed-request ring. Sent when a reconnect cannot fully replay from event ids (unknown, stale, or too-coalesced id) so clients always converge to live truth.</summary>
    public const string Resync = "resync";
}

/// <summary>The schema version carried by every monitor stream payload.</summary>
public static class MonitorEventSchema
{
    public const int Version = 1;
}

/// <summary>
/// The limit-kind vocabulary the monitor uses for recorded admission refusals, naming the same blocking limits the
/// request layer's refusals name: the KV pool cannot back the request's prompt plus token budget, the resolved
/// context window leaves no token free for generation, or every admission slot of the model's concurrency cap is
/// held. A recorded refusal is a diagnostic copy of a refusal its caller already answered — these names describe
/// which limit blocked, never a new error surface.
/// </summary>
public static class AdmissionRefusalKinds
{
    /// <summary>The submission's prompt plus token budget outruns the token slots the KV pool can back right now.</summary>
    public const string KvCapacity = "kv_capacity";

    /// <summary>The prompt does not fit the model's resolved context window with a token left free for generation.</summary>
    public const string ContextLength = "context_length";

    /// <summary>Every admission slot of the model's <c>max_concurrent_requests</c> cap is held by sequences already admitted.</summary>
    public const string ConcurrencyQuota = "concurrency_quota";

    /// <summary>The host ledger's shared copy-holding chain could not seat this submission's reservation anywhere — host accounting across opened pools, not the pool's own cap.</summary>
    public const string BudgetExhaustion = "budget_exhaustion";
}

/// <summary>
/// One point of a request's tok/s series: at the moment its Nth token was produced, N tokens had taken this many
/// milliseconds since generation began. Series are step-boundary tick samples over a small fixed window (oldest
/// points drop first), so sparklines read recent speed rather than a full history.
/// </summary>
public sealed record TokenSample(
    [property: JsonPropertyName("at_token")] int AtToken,
    [property: JsonPropertyName("elapsed_ms")] long ElapsedMs);

/// <summary>
/// One request-lifecycle frame payload (start/update/complete share one shape; fields not yet known are null and
/// omitted from the JSON). The id is the existing <c>x-lair-request-id</c>, so stream rows correlate with HTTP
/// responses and their log lines. Update frames carry a snapshot of the live tok/s series; completion frames carry
/// the final one plus cached tokens and the finish status. In-flight views inside a resync payload use this shape too.
/// </summary>
public sealed record RequestPayload(
    [property: JsonPropertyName("events_version")] int EventsVersion,
    [property: JsonPropertyName("request_id")] string RequestId,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("endpoint")] string Endpoint,
    [property: JsonPropertyName("prompt_tokens")] int PromptTokens,
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("generated_tokens")] int? GeneratedTokens = null,
    [property: JsonPropertyName("elapsed_ms")] long? ElapsedMs = null,
    [property: JsonPropertyName("tok_s")] double? TokensPerSec = null,
    [property: JsonPropertyName("cached_tokens")] int? CachedTokens = null,
    [property: JsonPropertyName("finish_reason")] string? FinishReason = null,
    [property: JsonPropertyName("samples")] IReadOnlyList<TokenSample>? Samples = null,
    [property: JsonPropertyName("budget_exhausted")] string? BudgetExhausted = null)
{
    /// <summary>The snapshot as one compact JSON document — the SSE payload.</summary>
    public string ToJson() => JsonSerializer.Serialize(this);

    internal static readonly JsonSerializerOptions Wire = new() { WriteIndented = false };
}

/// <summary>A connect-time (or replay-fallback) state dump: everything live right now, in request-payload shape.</summary>
public sealed record ResyncPayload(
    [property: JsonPropertyName("events_version")] int EventsVersion,
    [property: JsonPropertyName("in_flight")] IReadOnlyList<RequestPayload> InFlight,
    [property: JsonPropertyName("recent_requests")] IReadOnlyList<RequestPayload> RecentRequests)
{
    public string ToJson() => JsonSerializer.Serialize(this);
}
