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
    /// Which attention/weight path the engine's prefill route selects: <c>gemv</c> (default — the batched row window),
    /// <c>tiled</c> (the same windows with row-tiled causal attention), or <c>gemm</c> (dense weight traversal by blocked
    /// matmul). The schema accepts all three; a selection this engine cannot run for the served file is refused at model
    /// open naming that file's own weight formats. A hot edit takes effect from the next prefill window boundary.
    /// </summary>
    public string PrefillKernel { get; init; } = "gemv";
}

/// <summary>One served model and how it is placed on the machine.</summary>
public sealed record ModelConfig
{
    public required string Name { get; init; }

    public required string Path { get; init; }

    public string Type { get; init; } = "chat";

    public int MaxContextTokens { get; init; } = 4096;

    public int KvBlocks { get; init; } = 256;

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

    public PlacementConfig Placement { get; init; } = new();
}

/// <summary>Device, residency, and core-pinning choices for a model.</summary>
public sealed record PlacementConfig
{
    public string Device { get; init; } = "cpu";

    public string KvResidency { get; init; } = "dram";

    public string Prefill { get; init; } = "cpu";

    /// <summary>Pinned logical core ids; empty means unpinned.</summary>
    public IReadOnlySet<int> CoreSet { get; init; } = new HashSet<int>();

    /// <summary>Hard cap on this model's worker threads (1 or greater); null leaves the engine thread choice uncapped.</summary>
    public int? ThreadQuota { get; init; }
}
