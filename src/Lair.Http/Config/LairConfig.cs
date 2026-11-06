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
