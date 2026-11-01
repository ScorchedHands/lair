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
}

/// <summary>One served model and how it is placed on the machine.</summary>
public sealed record ModelConfig
{
    public required string Name { get; init; }

    public required string Path { get; init; }

    public string Type { get; init; } = "chat";

    public int MaxContextTokens { get; init; } = 4096;

    public int KvBlocks { get; init; } = 256;

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
}
