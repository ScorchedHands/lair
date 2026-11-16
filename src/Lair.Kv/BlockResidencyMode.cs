namespace Lair.Kv;

/// <summary>
/// How a <see cref="BlockPool"/> places its physical block ids across host and device residency. <c>Dram</c> is the
/// posture every pool in this build has always run: all blocks live in the host arena and nothing else is opened.
/// <c>Vram</c> declares the whole reservation device-backed: every block id the pool can hand out falls inside the
/// range a device mirror keeps a persistent copy of. <c>Split</c> divides the reservation between a host-only range and
/// a device-backed range. In every posture attention keeps reading the pool's own host arena unchanged — the residency
/// axis is about where a copy of each block's bytes is kept, not about which memory the reads address.
/// </summary>
public enum BlockResidencyMode
{
    /// <summary>Every block lives only in the host arena. No device mirror is opened and none is required.</summary>
    Dram,

    /// <summary>The whole reservation is device-backed: every block id lies inside the device mirror's range, and its host bytes cross to the device copy on flush.</summary>
    Vram,

    /// <summary>The reservation is divided: the leading block ids are device-backed and the rest stay host-only, each range flushed or read according to where its ids fall.</summary>
    Split,
}

/// <summary>The vocabulary one model's KV residency declares, and the mode names a resolved residency carries, each in the order a refusal reports it.</summary>
public static class BlockResidencyModeNames
{
    /// <summary>The declaration that states no posture of its own: this build resolves it to host-resident blocks, exactly as an absent key does.</summary>
    public const string Auto = "auto";

    public const string Dram = "dram";
    public const string Vram = "vram";
    public const string Split = "split";

    /// <summary>The mode names a resolved residency carries, in the order a route refusal reports them.</summary>
    public static IReadOnlyList<string> DeclaredNames { get; } = [Dram, Vram, Split];

    /// <summary>Every name one placement declaration may state: each resolved mode plus the posture-neutral <c>auto</c>.</summary>
    public static IReadOnlyList<string> ConfiguredVocabulary { get; } = [Auto, Dram, Vram, Split];

    public static string Of(BlockResidencyMode mode) => mode switch
    {
        BlockResidencyMode.Vram => Vram,
        BlockResidencyMode.Split => Split,
        _ => Dram,
    };

    /// <summary>Reads one declared placement value as the residency this build runs it as: an absent or blank declaration and a posture-neutral one both land on host-resident blocks, while every named mode lands on itself. An unrecognized name is not a residency: callers report it naming the accepted vocabulary.</summary>
    public static bool TryResolvePlacementValue(string? declared, out BlockResidencyMode mode)
    {
        switch (declared)
        {
            case Auto or Dram or null:
                mode = BlockResidencyMode.Dram;
                return true;

            case Vram:
                mode = BlockResidencyMode.Vram;
                return true;

            case Split:
                mode = BlockResidencyMode.Split;
                return true;

            default:
                mode = default;
                return false;
        }
    }

    /// <summary>The accepted set as the text a refusal names it by.</summary>
    public static string AcceptedVocabulary => $"only {string.Join(", ", ConfiguredVocabulary.Select(name => $"'{name}'"))}";
}
