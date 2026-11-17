using System.Collections.Generic;
using System.Linq;

namespace Lair.Kv;

/// <summary>
/// One side of a served model's KV cache format pair. The cache is a pair of per-side formats, not a pair of scales:
/// the float sides store plain values and carry no stored scale, while the block-quant sides own their per-block group
/// scales privately inside the region those values live in. This is a cold-path vocabulary — it is what a configuration
/// resolves to, what a pool sizes itself against, and what a model-open refusal names — never something an attention
/// loop consults per token.
/// </summary>
public enum KvSideFormat
{
    /// <summary>Plain FP32 values with no stored scale: the storage every pool in this build has always run.</summary>
    F32,

    /// <summary>Half-precision values with no stored scale — the same addressing shape as <see cref="F32"/>, at half the bytes.</summary>
    F16,

    /// <summary>GGML q8_0 group geometry: 32 one-byte payloads plus one half-precision scale per whole 32-value group.</summary>
    Q8_0,

    /// <summary>GGML q4_0 group geometry: 16 nibble-packed bytes (32 values) plus one half-precision scale per whole 32-value group. A pair that names this format on both sides is the dropped combo and never resolves.</summary>
    Q4_0,
}

/// <summary>The canonical name one resolved KV side carries, every exact lowercase spelling a declaration may state, and the vocabulary a refusal reports them by — each in the order a refusal reports them.</summary>
public static class KvSideFormatNames
{
    public const string F32 = "f32";
    public const string F16 = "f16";
    public const string Q8_0 = "q8_0";
    public const string Q4_0 = "q4_0";

    /// <summary>The canonical names a resolved pair carries.</summary>
    public static IReadOnlyList<string> DeclaredNames { get; } = [F32, F16, Q8_0, Q4_0];

    /// <summary>Every exact lowercase spelling one side declaration may state: each canonical name beside its documented aliases — <c>fp32</c> and <c>fp16</c> for the float sides, <c>q8</c> and <c>int8</c> for <c>q8_0</c>.</summary>
    public static IReadOnlyList<string> ConfiguredVocabulary { get; } = [F32, "fp32", F16, "fp16", Q8_0, "q8", "int8", Q4_0];

    /// <summary>The accepted set as the text a refusal names it by.</summary>
    public static string AcceptedVocabulary => $"only {string.Join(", ", ConfiguredVocabulary.Select(name => $"'{name}'"))}";

    /// <summary>The canonical name a resolved side format carries.</summary>
    public static string Of(KvSideFormat format) => format switch
    {
        KvSideFormat.F16 => F16,
        KvSideFormat.Q8_0 => Q8_0,
        KvSideFormat.Q4_0 => Q4_0,
        _ => F32,
    };

    /// <summary>Reads one declared side value as the format this build resolves it to. Matching is exact and case-sensitive: an alias lands on its canonical format, and anything else — another casing, a misspelling, some other quant name — is not a KV side format and reports against <see cref="AcceptedVocabulary"/>.</summary>
    public static bool TryResolve(string? declared, out KvSideFormat format)
    {
        switch (declared)
        {
            case F32 or "fp32":
                format = KvSideFormat.F32;
                return true;

            case F16 or "fp16":
                format = KvSideFormat.F16;
                return true;

            case Q8_0 or "q8" or "int8":
                format = KvSideFormat.Q8_0;
                return true;

            case Q4_0:
                format = KvSideFormat.Q4_0;
                return true;

            default:
                format = default;
                return false;
        }
    }
}
