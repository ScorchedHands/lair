using Lair.Kv;
using Lair.Models;

namespace Lair.Host.Config;

/// <summary>
/// Immutable, fully validated LAIR configuration: the server listener, the engine
/// knobs, and the single model the runtime serves.
/// </summary>
public sealed record LairConfig
{
    public required ServerConfig Server { get; init; }

    public required EngineConfig Engine { get; init; }

    public required IReadOnlyList<ModelConfig> Models { get; init; }

    public UiConfig Ui { get; init; } = new();

    /// <summary>
    /// The resident capacity an operator declares for this host — system-memory and accelerator byte totals, logical processors, a disk
    /// posture, with optional per-node overrides — or null when the configuration says nothing about any of it. These are declared values
    /// exactly as the loader parsed them: what a host actually carries is what its reported topology states, and a declaration that
    /// disagrees with it is accepted with a warning rather than rejected. Internal today: compute accounting reads these totals against
    /// the reported posture when an engine opens.
    /// </summary>
    internal HostCapacityConfig? Host { get; init; }
}

/// <summary>
/// Declared resident capacity for one host, in flat keys: system-memory bytes and logical processors, a disk path declared as presence
/// only, and optional per-node overrides keyed by node position. Every value is declared posture — an absent key declares no total rather
/// than zero, and a per-node override states only what its entry names.
/// </summary>
public sealed record HostCapacityConfig
{
    /// <summary>Megabytes the operator declares at system memory; null (absent) declares no byte total.</summary>
    public long? DramMb { get; init; }

    /// <summary>Megabytes the operator declares at accelerator memory — the ceiling a device-resident weight claim seats against when a compute backend opened with device-resident weight copies backs that residency. No driver-detected total is consulted: an absent key states no public total, which is the posture a host gives when it backs a residency without naming its size.</summary>
    public long? VramMb { get; init; }

    /// <summary>The logical processors this host carries — the unit a core set or thread quota counts in; null (absent) declares no core total.</summary>
    public int? CpuCores { get; init; }

    /// <summary>The disk path this host reads from, declared as presence and path only. No byte ceiling belongs to that position: the model file already is the complete set.</summary>
    public string? DiskPath { get; init; }

    /// <summary>Per-node overrides keyed by node position, each stating its own bytes and/or cores. An override never replaces the reported posture — it disagrees with it.</summary>
    public IReadOnlyList<NodeCapacityOverride> Nodes { get; init; } = [];
}

/// <summary>One declared per-node override: which node position it names, and only the totals that entry states.</summary>
public sealed record NodeCapacityOverride
{
    /// <summary>The node position this override names, counting the nodes a host posture reports from zero.</summary>
    public required int Index { get; init; }

    /// <summary>Megabytes declared for this node; null (absent) declares nothing about its bytes.</summary>
    public long? Mb { get; init; }

    /// <summary>Logical processors declared for this node; null (absent) declares nothing about its cores.</summary>
    public int? Cores { get; init; }
}

/// <summary>Monitor surface settings — what the live stats stream retains and replays.</summary>
public sealed record UiConfig
{
    /// <summary>Completed requests kept in the monitor's recent-request ring, newest first. In [1, 1000]; a hot edit takes effect immediately (shrinking drops the oldest entries).</summary>
    public int RecentRequests { get; init; } = 25;
}

/// <summary>HTTP listener settings.</summary>
public sealed record ServerConfig
{
    public string Bind { get; init; } = "127.0.0.1";

    public int Port { get; init; } = 8080;
}

/// <summary>Engine-wide settings shared by every model.</summary>
public sealed record EngineConfig
{
    /// <summary>Worker threads; <c>0</c> means "pick automatically".</summary>
    public int Threads { get; init; }

    /// <summary>Fixed weight block size.</summary>
    public int BlockSize { get; init; } = 16;

    /// <summary>Largest prompt chunk the scheduler prefills in one step.</summary>
    public int MaxBatchTokens { get; init; } = 512;

    /// <summary>
    /// Wall-clock budget, in milliseconds (1 or greater), that a prefill step aims at. Null (absent) keeps the chunk width as
    /// configured: every step forwards up to <see cref="MaxBatchTokens"/> prompt tokens. A present value lets the scheduler size
    /// each future chunk from measured step cost so its forward work targets this budget, still capped by
    /// <see cref="MaxBatchTokens"/> and never below one token; the engine's own row ceilings keep sub-batch rows bounded either
    /// way. A hot edit takes effect from the next prefill chunk issue — a step already in flight finishes at its old width.
    /// </summary>
    public int? PrefillStepBudgetMs { get; init; }

    /// <summary>
    /// Which attention/weight path the engine's prefill route selects: <c>gemv</c> (default — the batched row window),
    /// <c>tiled</c> (the same windows with row-tiled causal attention), or <c>gemm</c> (dense weight traversal by blocked
    /// matmul). The schema accepts all three; a selection this engine cannot run for the served file is refused at model
    /// open naming that file's own weight formats. A hot edit takes effect from the next prefill window boundary.
    /// </summary>
    public string PrefillKernel { get; init; } = "gemv";

    /// <summary>The engine-wide CPU posture block: which integer-dot instruction set the sub-4-bit i-quant kernels run.</summary>
    public EngineCpuConfig Cpu { get; init; } = new();
}

/// <summary>The keys one engine cpu block may declare.</summary>
public sealed record EngineCpuConfig
{
    /// <summary>
    /// Which integer-dot encoding the sub-4-bit i-quant kernels run: <c>auto</c> (default — the highest path this host's CPU
    /// reports, AVX-VNNI over AVX2), <c>avx2</c>, or <c>avx-vnni</c>. The axis resolves once at engine open and every row,
    /// batch vector, worker, and lane of the model runs that one encoding; forcing a path this CPU does not report fails the
    /// load naming the missing feature instead of falling back. All three spellings are accepted by the schema — what a host
    /// can run is decided when its backend opens.
    /// </summary>
    public string Isa { get; init; } = "auto";
}

/// <summary>One served model and how it is placed on the machine.</summary>
public sealed record ModelConfig
{
    public required string Name { get; init; }

    public required string Path { get; init; }

    /// <summary>
    /// What this model serves: <c>chat</c> (default — prompt forwards feed sampled decode, exactly as every model does today) or
    /// <c>embed</c> (encode-only serving posture: requests run bidirectional encoder forwards and answer sentence vectors, the chat
    /// and completion endpoints refuse it, and no decode pipeline ever advances on it). Which posture a loaded instance runs is
    /// established when the engine opens, so changing this key is a restart-only edit.
    /// </summary>
    public string Type { get; init; } = ModelTypeValues.Chat;

    public int MaxContextTokens { get; init; } = 4096;

    public int KvBlocks { get; init; } = 256;

    /// <summary>
    /// The per-resource budgets this model entry declares — weights, KV blocks, and prefill buffers each with their own ordered
    /// tier chain and ceilings — or null when the entry says nothing about any of them. These are declared values exactly as the
    /// loader parsed them; what a host can actually carry is resolved against its opened pools at engine open, not here. Internal:
    /// no monitor or API surface reports budget claims yet, while hosts resolve these chains onto the model's plan and reload
    /// classification routes each ceiling by its own mutation class.
    /// </summary>
    internal IReadOnlyList<BudgetChain>? Budgets { get; init; }

    /// <summary>True when this entry wrote <c>kv_blocks</c> explicitly — presence is what separates a spelled-out number from today's default where the legacy ceiling and budgets' system-memory ceiling must state the same thing.</summary>
    internal bool LegacyKvBlocksDeclared { get; init; }

    /// <summary>
    /// The KV cache format pair this model entry declares, resolved to canonical per-side formats exactly as the loader parsed them: an absent
    /// <c>kv</c> block, an absent <c>kv.quant</c> nesting, or an absent side all resolve to <c>f32</c>, which is today's storage. Whether a
    /// resolved pair can actually be served is what model open answers against the pool and attention paths it has landed — this value states
    /// posture only. Internal: no monitor or API surface reports it yet, while reload classification routes each side by its own mutation class.
    /// </summary>
    internal KvFormatPair KvQuant { get; init; } = KvFormatPair.Default;

    /// <summary>
    /// Tool-precedence override. When true, chat requests with declared tools always receive LAIR's fenced tool
    /// block even when the model's chat template reports tool support, and the declared tools are kept out of the
    /// template's own <c>tools</c> context so exactly one tool surface reaches the model. Default false: the loaded
    /// template's detected capabilities decide, and LAIR injection is the fallback for templates without tool support.
    /// </summary>
    public bool ForceFencedTools { get; init; }

    /// <summary>
    /// Which tool output parser the model resolves to at load: <c>auto</c> (default — the loaded template's detected
    /// capabilities decide, with LAIR's fenced block as fallback), <c>template</c> (the model's own chat-template
    /// rendering is authoritative and no fenced injection happens), <c>fenced</c> (LAIR's fenced tool block for every
    /// declared-tool request regardless of the template), or <c>none</c> (inert: no tool surface reaches the model,
    /// content stays byte-for-byte what was generated, and no <c>tool_calls</c> are reported).
    /// </summary>
    public string ToolParser { get; init; } = "auto";

    /// <summary>
    /// How a recognized reasoning region reaches the wire: <c>keep</c> (default — the region streams through as ordinary
    /// visible content, byte-for-byte as generated) or <c>strip</c> (the region is withheld from visible content while its
    /// boundaries stay recorded for the request). Recognition itself keys on the markup a generation contains, not on this
    /// key: with neither marker present nothing is recognized and either value leaves output untouched.
    /// </summary>
    public string Reasoning { get; init; } = "keep";

    /// <summary>Per-model reasoning-token budget carried with the model entry; null (absent) means unbudgeted — today's behavior, no ceiling applied.</summary>
    public int? MaxReasoningTokens { get; init; }

    /// <summary>Per-model response-token budget carried with the model entry; null (absent) means unbudgeted and leaves request token limits at their existing semantics.</summary>
    public int? MaxResponseTokens { get; init; }

    /// <summary>
    /// Hard cap on how many requests this model may hold admitted at once. Null (absent) bounds nothing: every submission is
    /// admitted as far as KV capacity reaches, exactly as before this key existed. A present value must be 1 or greater and is
    /// enforced at admission — a submission that would exceed it is refused naming the model and its quota slot rather than
    /// queued behind what already runs. Sequences already admitted keep their slots through a hot edit of this key; only later
    /// submissions see the new cap.
    /// </summary>
    public int? MaxConcurrentRequests { get; init; }

    /// <summary>
    /// How many submissions this model may hold waiting for a concurrency slot before new ones are refused. Null (absent) or an
    /// explicit 0 keeps today's posture: a submission arriving while every admission slot is held is refused immediately rather
    /// than queued. A positive value lets such submissions block on the slot instead, each revalidated against context, KV capacity,
    /// and budget chains when a slot frees; nothing a queued submission waits for costs the pool or a token budget before its own
    /// admission. Sequences already admitted are untouched by a hot edit of this key, and so are submissions that were already waiting.
    /// </summary>
    public int? MaxQueuedRequests { get; init; }

    /// <summary>
    /// How long a submission may block waiting for an admission slot before it answers with the overloaded refusal. A present value
    /// must be 1 or greater, in milliseconds. Null (absent) resolves to the documented default when queueing is enabled and carries
    /// no meaning while queueing is disabled — with <c>max_queued_requests</c> at zero nothing waits, so this key changes no refusal posture there.
    /// </summary>
    public int? QueueTimeoutMs { get; init; }

    /// <summary>
    /// This model's device fair-share: the percent (1 through 100) of one host device co-scheduling window its launched device work may
    /// hold against the accelerator its resolved route advances on. Null (absent) declares no share: step composition bounds this model by
    /// its segment batch caps and quota headroom alone, exactly as before this key existed. A present value bounds how much device work a
    /// later boundary may launch for this model — an over-subscribed candidate defers naming the share, the model, and the device segment
    /// the window bounds — and never preempts a sequence already running: the cap retargets future launches, and a model whose share would
    /// round below one window slot still gets one. Like a stepping segment's thread quota, a live edit follows from the next composition on.
    /// </summary>
    public int? ComputeShare { get; init; }

    /// <summary>
    /// How an embedding request's encoder forward collapses its per-position hidden states into the vector it answers with:
    /// <c>mean</c> (default — the average over every position) or <c>cls</c> (the first position's row verbatim). The choice reads
    /// what a request already tokenized, so a live edit retitles later requests only; nothing an encoder forward already computed changes.
    /// </summary>
    public string Pooling { get; init; } = PoolingModeValues.Mean;

    /// <summary>
    /// Whether pooled vectors are L2-normalized before they answer. Default false returns the raw pooled vector exactly as pooling
    /// produced it; true scales it to unit length (a zero vector stays untouched). Like <c>pooling</c>, this is a response-shape choice,
    /// so a live edit retitles later requests only.
    /// </summary>
    public bool Normalize { get; init; }

    /// <summary>
    /// The published source this model's files are pinned to — the repo, the revision as declared, and the file set with each
    /// file's pinned hash and size — or null when the entry says nothing about provenance, which serves the declared local
    /// path exactly as every model does today. The loader validates this block's shape at load like every other block;
    /// whether the bytes on disk match the pin is the pack manifest's answer, and a live edit of this posture is
    /// restart-only because loadability depends on what the manifest and fetch behavior established at open.
    /// </summary>
    public ModelSourceConfig? Source { get; init; }

    public PlacementConfig Placement { get; init; } = new();
}

/// <summary>Where a served model's files come from: the published repo, the revision pinned as declared, and the model files the pack carries.</summary>
public sealed record ModelSourceConfig
{
    /// <summary>The Hugging Face repo slug the files are fetched from.</summary>
    public required string Repo { get; init; }

    /// <summary>The commit sha or branch name the entry pins, stored exactly as declared: a branch becomes a commit sha when fetch resolves it into the manifest.</summary>
    public required string Revision { get; init; }

    /// <summary>The model files the source declares, in declared order, each with its pinned hash and size when the entry states them.</summary>
    public required IReadOnlyList<SourceFileConfig> Files { get; init; }
}

/// <summary>One file a source block declares: a GGUF path/name beside the repo, with the pinned sha256 and declared size when the entry states them.</summary>
public sealed record SourceFileConfig
{
    /// <summary>The file's path/name within the repo, kept as a readable name in the pack directory.</summary>
    public required string Path { get; init; }

    /// <summary>The pinned sha256 of this file's bytes, 64 lowercase hex characters; null (absent) pins no hash yet.</summary>
    public string? Sha256 { get; init; }

    /// <summary>The declared byte size of this file; null (absent) declares no size.</summary>
    public long? Size { get; init; }
}

/// <summary>The accepted values of a model entry's <c>type</c>: prompt-and-sample serving, or encode-only sentence-vector serving.</summary>
public static class ModelTypeValues
{
    /// <summary>Prompt forwards feed sampled decode through the generation scheduler — today's behavior, and what an absent key resolves to.</summary>
    public const string Chat = "chat";

    /// <summary>The model serves encoder forwards only: its requests run bidirectional encode passes and answer sentence vectors, and no decode pipeline ever advances on it.</summary>
    public const string Embed = "embed";
}

/// <summary>The accepted values of a model entry's <c>pooling</c>: how an encoder forward's per-position hidden states collapse into one vector.</summary>
public static class PoolingModeValues
{
    /// <summary>Average every position's final hidden row — the default sentence-vector recipe, and what an absent key resolves to.</summary>
    public const string Mean = "mean";

    /// <summary>Take the first position's final hidden row verbatim (CLS-style extraction).</summary>
    public const string Cls = "cls";
}

/// <summary>The accepted values of <c>placement.prefill_lane</c>: one execution backend for every forward, or a dedicated worker lane for prompt forwards.</summary>
public static class PlacementPrefillLaneValues
{
    /// <summary>Prefill and decode share the model's single execution posture — today's behavior, and what an absent key resolves to.</summary>
    public const string Shared = "shared";

    /// <summary>Prompt forwards run on a second CPU execution backend capped to its own lane share while decode composition keeps the rest of the claim.</summary>
    public const string Dedicated = "dedicated";
}

/// <summary>Device, residency, core-pinning, and prefill-lane choices for a model.</summary>
public sealed record PlacementConfig
{
    public string Device { get; init; } = "cpu";

    public string KvResidency { get; init; } = "dram";

    public string Prefill { get; init; } = "cpu";

    /// <summary>Pinned logical core ids; empty means unpinned.</summary>
    public IReadOnlySet<int> CoreSet { get; init; } = new HashSet<int>();

    /// <summary>Hard cap on this model's worker threads (1 or greater); null leaves the engine thread choice uncapped.</summary>
    public int? ThreadQuota { get; init; }

    /// <summary>Which prefill lane posture the model opens with: <c>shared</c> (default — prompt forwards ride the model's single execution backend, exactly as before this key existed) or <c>dedicated</c> (prompt forwards run on a second CPU execution backend capped to its own share of the worker claim). The split is resolved when the engine opens; changing it is a restart-only edit.</summary>
    public string PrefillLane { get; init; } = PlacementPrefillLaneValues.Shared;

    /// <summary>
    /// Whether this model's core-set claim may overlap cores another model already claims. Default false keeps today's posture: pinned
    /// core sets partition the machine disjointly and an overlap is refused at load naming both models. True relaxes it for this model's
    /// own claim, which then shares held cores under the compute grant rules instead of being refused. Overlap changes what a running
    /// engine's workers contend for, so this key is established when the engine opens — a restart-only edit like <c>core_set</c>.
    /// </summary>
    public bool CoreOvercommit { get; init; }

    /// <summary>
    /// The NUMA spread posture this model's placement declares: <c>auto</c> (default — placement chooses the least-loaded node the
    /// host reports, lowest node index as tiebreak, and a pinned core set resolves to the node carrying it), <c>local</c> (one node
    /// carries the model), or <c>interleave</c> (a spread posture across the reported nodes). The posture is decided when the engine
    /// opens: which node its workers run on and what its weight pages first-touch to are established then, so a live edit records the
    /// preference for later placement decisions without re-binding an opened worker pool.
    /// </summary>
    public string NumaSpread { get; init; } = "auto";

    /// <summary>
    /// How many of this model's transformer layers execution opens on device <c>cuda</c>, counted from layer 0, with every later
    /// layer and the output head staying on CPU. Zero (absent) keeps today's posture: every layer runs on this service's own CPU
    /// backend and no accelerator seam is opened at all. A present value must be 0 or greater; a split opens its CUDA compute backend
    /// when the engine opens, so changing it — like any key that decides which backends and route segments exist — is a restart-only edit.
    /// </summary>
    public int GpuLayers { get; init; }
}
