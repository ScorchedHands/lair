using System.Text.Json;
using System.Text.Json.Serialization;

namespace Lair.Http.Monitoring;

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
    [property: JsonPropertyName("metrics_process_cpu_percent")] double ProcessCpuPercent,

    /// <summary>The monitor stream schema version this snapshot's payload belongs to.</summary>
    [property: JsonPropertyName("metrics_events_version")] int EventsVersion = MonitorEventSchema.Version,

    /// <summary>The server-scoped event id the bus assigned this snapshot; equals its SSE frame id and anchors Last-Event-ID replay.</summary>
    [property: JsonPropertyName("metrics_event_seq")] long EventSeq = 0,

    /// <summary>Totals of stream frames dropped from subscriber queues since the bus opened — how often slow consumers lost events.</summary>
    [property: JsonPropertyName("metrics_dropped_events_total")] long DroppedEventsTotal = 0,

    // Additive model-fact fields for operator views (configured-model detail rows). Every one is a trailing optional
    // parameter with an inert default, so v1 payloads and constructions read exactly as before; pool-derived values
    // are always filled, configuration-backed ones serialize null until the owning service attaches facts.

    /// <summary>The configured model kind (<c>chat</c>, <c>embed</c>) this engine serves.</summary>
    [property: JsonPropertyName("metrics_model_type")] string? ModelType = null,

    /// <summary>Tokens per KV block — the fixed block geometry of this engine's pool.</summary>
    [property: JsonPropertyName("metrics_kv_block_tokens")] int KvBlockTokens = 0,

    /// <summary>The soft KV usage cap admission currently spends from, within the physical reservation.</summary>
    [property: JsonPropertyName("metrics_kv_budget_blocks")] int KvBudgetBlocks = 0,

    /// <summary>Total bytes physically reserved for the KV pool across all blocks, rounded to megabytes.</summary>
    [property: JsonPropertyName("metrics_kv_pool_mb")] double KvPoolMb = 0.0,

    /// <summary>Engine worker threads currently participating in kernels (the effective count after quota).</summary>
    [property: JsonPropertyName("metrics_worker_threads")] int? WorkerThreads = null,

    /// <summary>Pinned logical core ids rendered as compressed ranges (<c>0-23</c>), or null when unpinned.</summary>
    [property: JsonPropertyName("metrics_core_set")] string? CoreSet = null,

    /// <summary>The configured hard cap on this model's worker threads, or null when uncapped.</summary>
    [property: JsonPropertyName("metrics_thread_quota")] int? ThreadQuota = null,

    /// <summary>The compute device placement of the served model (decode side).</summary>
    [property: JsonPropertyName("metrics_decode_device")] string? DecodeDevice = null,

    /// <summary>The prefill-side device placement configured for this model.</summary>
    [property: JsonPropertyName("metrics_prefill_device")] string? PrefillDevice = null,

    /// <summary>The KV residency mode configured for this model (<c>dram</c>, ...).</summary>
    [property: JsonPropertyName("metrics_kv_residency")] string? KvResidency = null,

    /// <summary>The prefill kernel selection the engine's prompt-window route runs through (<c>gemv</c>, ...).</summary>
    [property: JsonPropertyName("metrics_prefill_kernel")] string? PrefillKernel = null,

    /// <summary>Layers whose resolved plan carries a finite attention window.</summary>
    [property: JsonPropertyName("metrics_kv_windowed_layers")] int WindowedLayers = 0,

    /// <summary>Logical KV blocks the declared windows select across all layers at the current soft budget.</summary>
    [property: JsonPropertyName("metrics_kv_read_footprint_blocks")] long KvReadFootprintBlocks = 0,

    /// <summary>Whole logical blocks excluded by declared windows at the current soft budget.</summary>
    [property: JsonPropertyName("metrics_kv_skipped_read_blocks")] long SkippedKvReadBlocks = 0)
{
    /// <summary>The snapshot as one compact JSON document — the SSE payload and the structured log line's body.</summary>
    public string ToJson() => JsonSerializer.Serialize(this);

    internal static readonly JsonSerializerOptions Wire = new() { WriteIndented = false };
}
