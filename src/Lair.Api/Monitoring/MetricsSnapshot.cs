using System.Text.Json;
using System.Text.Json.Serialization;

namespace Lair.Api.Monitoring;

/// <summary>
/// One published view of a served engine's live counters, process facts included. Every field serializes with a
/// consistent <c>metrics_</c> prefix so snapshot log lines and stream events share one vocabulary with the
/// per-request structured logs. Rates are windowed: they compare this snapshot against the previous one for the
/// same model (or the engine-open baseline for the first), so an idle stretch reads as zero rather than a stale
/// average; cumulative fields always read total-since-open. Process facts describe the serving process, not the OS or hardware — wider system telemetry is out of scope here and later phases may extend it; CPU share is relative to one logical core, so a multithreaded engine can report more than 100 percent.
/// </summary>
public sealed record MetricsSnapshot(
    [property: JsonPropertyName("metrics_ts")] string TimestampUtcIso8601,
    [property: JsonPropertyName("metrics_window_ms")] long WindowMs,

    [property: JsonPropertyName("metrics_model")] string Model,
    [property: JsonPropertyName("metrics_tok_s")] double TokensPerSecDecodeWindowed,
    [property: JsonPropertyName("metrics_step_time_avg_ms")] double AvgStepTimeMsWindowed,
    [property: JsonPropertyName("metrics_decode_steps_total")] long DecodeStepsTotal,
    [property: JsonPropertyName("metrics_batch_occupancy_seqs_per_step")] double BatchOccupancyAvgSequencesPerStep,
    [property: JsonPropertyName("metrics_prefill_chunks_total")] long PrefillChunksTotal,
    [property: JsonPropertyName("metrics_prefill_tokens_forwarded_total")] long PrefillTokensForwardedTotal,

    [property: JsonPropertyName("metrics_cache_hit_sequences_total")] long CacheHitSequencesTotal,
    [property: JsonPropertyName("metrics_cache_skipped_tokens_total")] long CacheSkippedTokensTotal,

    [property: JsonPropertyName("metrics_kv_used_blocks")] int KvUsedBlocks,
    [property: JsonPropertyName("metrics_kv_capacity_blocks")] int KvCapacityBlocks,
    [property: JsonPropertyName("metrics_kv_cached_blocks")] int KvCachedBlocks,
    [property: JsonPropertyName("metrics_active_sequences")] int ActiveSequences,

    [property: JsonPropertyName("metrics_requests_total")] long RequestsTotal,
    [property: JsonPropertyName("metrics_requests_eos_total")] long RequestsEosTotal,
    [property: JsonPropertyName("metrics_requests_length_total")] long RequestsLengthTotal,
    [property: JsonPropertyName("metrics_requests_stop_total")] long RequestsStopTotal,
    [property: JsonPropertyName("metrics_requests_cancelled_total")] long RequestsCancelledTotal,
    [property: JsonPropertyName("metrics_requests_failed_total")] long RequestsFailedTotal,

    [property: JsonPropertyName("metrics_process_working_set_mb")] double ProcessWorkingSetMb,
    [property: JsonPropertyName("metrics_process_cpu_percent")] double ProcessCpuPercent)
{
    /// <summary>The snapshot as one compact JSON document — the SSE payload and the structured log line's body.</summary>
    public string ToJson() => JsonSerializer.Serialize(this);

    internal static readonly JsonSerializerOptions Wire = new() { WriteIndented = false };
}
