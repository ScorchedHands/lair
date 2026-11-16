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

    /// <summary>The quant formats the served model's dense weight matrices carry — the formats a prefill selection needs kernels for.</summary>
    [property: JsonPropertyName("metrics_weight_formats")] string? MatmulWeightFormats = null,

    /// <summary>Layers whose resolved plan carries a finite attention window.</summary>
    [property: JsonPropertyName("metrics_kv_windowed_layers")] int WindowedLayers = 0,

    /// <summary>Logical KV blocks the declared windows select across all layers at the current soft budget.</summary>
    [property: JsonPropertyName("metrics_kv_read_footprint_blocks")] long KvReadFootprintBlocks = 0,

    /// <summary>Whole logical blocks excluded by declared windows at the current soft budget.</summary>
    [property: JsonPropertyName("metrics_kv_skipped_read_blocks")] long SkippedKvReadBlocks = 0,

    // Additive quota, route, and admission-diagnostic fields. Every one is a trailing optional parameter with an
    // inert default, so earlier payloads and constructions read exactly as before; configuration-backed ones
    // serialize null until the owning service attaches facts.

    /// <summary>The configured admission concurrency cap for this model, or null when nothing bounds how many requests it may hold admitted at once.</summary>
    [property: JsonPropertyName("metrics_max_concurrent_requests")] int? MaxConcurrentRequests = null,

    /// <summary>The resolved execution route's segments in execution order — device identity, weight tier, batch cap, and resolved quota per segment — or null until the owning service attaches them.</summary>
    [property: JsonPropertyName("metrics_route_segments")] IReadOnlyList<ModelRouteSegment>? RouteSegments = null,

    /// <summary>Which prefill-lane posture this model opened with (<c>shared</c>, <c>dedicated</c>), or null until the owning service attaches facts.</summary>
    [property: JsonPropertyName("metrics_prefill_lane")] string? PrefillLane = null,

    /// <summary>The effective worker claim a dedicated split was derived from — zero when the posture shares one backend.</summary>
    [property: JsonPropertyName("metrics_prefill_claim_threads")] int PrefillClaimThreads = 0,

    /// <summary>The decode lane's worker cap as resolved — zero when shared.</summary>
    [property: JsonPropertyName("metrics_prefill_decode_share_threads")] int PrefillDecodeShareThreads = 0,

    /// <summary>The prefill lane's own worker cap — zero when shared.</summary>
    [property: JsonPropertyName("metrics_prefill_lane_share_threads")] int PrefillLaneShareThreads = 0,

    /// <summary>The decode lane's pinned core ids as compressed ranges, null when the posture pins nothing.</summary>
    [property: JsonPropertyName("metrics_prefill_decode_core_set")] string? PrefillDecodeCoreSet = null,

    /// <summary>The prefill lane's pinned core ids as compressed ranges, null when the posture pins nothing.</summary>
    [property: JsonPropertyName("metrics_prefill_lane_core_set")] string? PrefillLaneCoreSet = null,

    /// <summary>Total bytes of mutable activation storage the dedicated lane family holds at open — zero for a shared posture.</summary>
    [property: JsonPropertyName("metrics_prefill_activation_bytes")] long PrefillActivationBytes = 0,

    /// <summary>The row width the opened engine resolved for its prompt windows, before any declared prefill-buffer ceiling — reported beside the lane's activation bytes so a step-time fact and a buffer-bound fact read together.</summary>
    [property: JsonPropertyName("metrics_prefill_row_width")] int PrefillRowWidth = 0,

    /// <summary>The row width the opened engine actually runs its prompt windows at: the resolved width when nothing declared bounds it, narrowed to what a prefill-buffer ceiling allows otherwise.</summary>
    [property: JsonPropertyName("metrics_effective_prefill_row_width")] int EffectivePrefillRowWidth = 0,

    /// <summary>Total mutable activation bytes the opened engine holds for its prompt windows' working set — what a prefill-buffer claim states against its chain.</summary>
    [property: JsonPropertyName("metrics_prefill_working_set_bytes")] long PrefillWorkingSetBytes = 0,

    /// <summary>What narrowed the engine's row window, in the binding ceiling's own words — tier and cap, one sub-batch row's activation bytes, requested against allowed width; null when no declared ceiling bound the window.</summary>
    [property: JsonPropertyName("metrics_prefill_row_clamp")] string? PrefillRowClamp = null,

    /// <summary>Submissions the admission control plane refused since this bus opened, whatever their caller answered with.</summary>
    [property: JsonPropertyName("metrics_admission_refusals_total")] long AdmissionRefusalsTotal = 0,

    /// <summary>The most recent admission refusals, newest first: each names the limit kind that blocked, the model, and a concise reason. Bounded to the latest <see cref="EngineMetricsBus.AdmissionRefusalRingCapacity"/>.</summary>
    [property: JsonPropertyName("metrics_admission_refusals")] IReadOnlyList<AdmissionRefusalEntry>? AdmissionRefusals = null,

    /// <summary>Host device-ledger occupancy: one entry per tier position a seated claim holds, each with the tier's capacity posture and contested holders where several claimants share it; null until the owning service attaches facts and whenever no claim has seated.</summary>
    [property: JsonPropertyName("metrics_ledger_occupancy")] IReadOnlyList<LedgerOccupancyEntry>? LedgerOccupancy = null,

    /// <summary>The worker share this model asked to run with (engine thread choice capped by its own quota); null until the owning service attaches compute facts.</summary>
    [property: JsonPropertyName("metrics_compute_requested_threads")] int? ComputeRequestedThreads = null,

    /// <summary>The logical processors that claim was granted — below requested where an earlier seat clamped it; null until compute facts attach.</summary>
    [property: JsonPropertyName("metrics_compute_granted_threads")] int? ComputeGrantedThreads = null,

    /// <summary>True when this model opted into sharing cores another claim holds; null while the disjoint posture stands or before compute facts attach.</summary>
    [property: JsonPropertyName("metrics_core_overcommit")] bool? CoreOvercommit = null,

    /// <summary>The models whose participation rides logical processors a grant already holds — null whenever every seated share is its own.</summary>
    [property: JsonPropertyName("metrics_shared_cores_models")] IReadOnlyList<string>? SharedCoreModels = null,

    // Additive placement facts (earlier shapes untouched): the node an opened engine seated its worker claim on beside the
    // posture placement applied and the posture the configuration declared, the per-node share the grant seated, and any
    // advisory stating why a posture could not be honoured from what the host reported. A live edit of the spread key
    // records the preference for later placement decisions; these stay as the open resolved them.

    /// <summary>The single node this engine's worker claim seated on; null for a posture that spread across nodes or until placement facts attach.</summary>
    [property: JsonPropertyName("metrics_placement_node")] int? PlacementNode = null,

    /// <summary>Every node the opened placement chose, as a comma-separated list of node positions.</summary>
    [property: JsonPropertyName("metrics_placement_nodes")] string? PlacementNodes = null,

    /// <summary>The spread posture placement applied at open (<c>auto</c>, <c>local</c>, <c>interleave</c>); null until placement facts attach.</summary>
    [property: JsonPropertyName("metrics_placement_spread")] string? PlacementSpread = null,

    /// <summary>The spread posture the model's configuration declared, beside the posture placement resolved it to.</summary>
    [property: JsonPropertyName("metrics_placement_spread_requested")] string? PlacementSpreadRequested = null,

    /// <summary>The logical processors this model's grant seated as its own on each node, as <c>node:cores</c> pairs; null until compute facts attach or when the claim seated none.</summary>
    [property: JsonPropertyName("metrics_compute_node_allocations")] string? PlacementNodeAllocations = null,

    /// <summary>Why the opened posture could not be honoured whole from what the host reported; null when it was honoured.</summary>
    [property: JsonPropertyName("metrics_placement_advisory")] string? PlacementAdvisory = null,

    /// <summary>The report-only update candidate the served model's pinned pack has against the hub's current HEAD — the candidate revision and its delta summary — or null when the model declares no published source, when a check found no candidate, or when a check could not be consulted. A trailing optional field: snapshots of a model with no candidate read exactly as they did before it existed.</summary>
    [property: JsonPropertyName("metrics_update_candidate")] string? UpdateCandidate = null,

    // Additive accelerator facts (earlier shapes untouched): every one reads from what an opened compute backend answers
    // on the device side — driver-measured kernel time and its own copy tallies — never from OS or hardware telemetry. A
    // serve with no accelerator backend attached leaves all four null, so a CPU-only snapshot carries them as nothing at
    // all rather than as zeros an operator could mistake for idle hardware.

    /// <summary>Windowed share of this window the opened accelerator backends' kernels were busy on the device — delta device-measured kernel microseconds over elapsed wall milliseconds, summed across every backend instance this serve holds open; null until such a backend stands here.</summary>
    [property: JsonPropertyName("metrics_gpu_utilization_percent")] double? GpuUtilizationPercent = null,

    /// <summary>Cumulative bytes the opened accelerator backends crossed with runtime copies — activation inputs going to the device and kernel outputs coming back. One-time persistent weight uploads, host-region registrations, and KV mirror residency crossings stay outside this tally; null on a serve holding no accelerator backend.</summary>
    [property: JsonPropertyName("metrics_pcie_transfer_bytes_total")] long? PcieTransferBytesTotal = null,

    /// <summary>Cumulative count of those runtime copy operations — one per activation input and one per kernel output, whatever their sizes; null on a serve holding no accelerator backend.</summary>
    [property: JsonPropertyName("metrics_pcie_transfers_total")] long? PcieTransfersTotal = null,

    /// <summary>Device bytes the opened default-copy postures hold open for resident weights right now — host-mapped weight residency and transient activation buffers stay outside this reading, which drops as an instance releases its allocations; null on a serve holding no accelerator backend.</summary>
    [property: JsonPropertyName("metrics_persistent_vram_bytes")] long? PersistentVramBytes = null)
{
    /// <summary>The snapshot as one compact JSON document — the SSE payload and the structured log line's body.</summary>
    public string ToJson() => JsonSerializer.Serialize(this);

    internal static readonly JsonSerializerOptions Wire = new() { WriteIndented = false };
}
