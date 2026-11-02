namespace Lair.Sampling;

/// <summary>
/// Immutable decode-time sampling configuration. <see cref="Greedy"/> is implied by a
/// non-positive temperature; <see cref="TopK"/> = 0 disables the top-k cut and
/// <see cref="TopP"/> = 1 disables the top-p cut. Values are validated when a
/// <see cref="Sampler"/> is constructed.
/// </summary>
public sealed record SamplingParams
{
    /// <summary>Softmax temperature. 0 (or lower) selects greedy decoding; negative values are invalid.</summary>
    public float Temperature { get; init; } = 1.0f;

    /// <summary>Number of highest-probability candidates kept before the top-p cut. 0 disables the cut; negative values are invalid.</summary>
    public int TopK { get; init; } = 40;

    /// <summary>Nucleus mass kept after the top-k cut, in (0, 1]. 1 keeps every surviving candidate.</summary>
    public float TopP { get; init; } = 0.9f;

    /// <summary>RNG seed for reproducible draws; null uses a fresh, unseeded generator.</summary>
    public long? Seed { get; init; } = null;

    /// <summary>
    /// Additive penalty on the logit of every token that has appeared at least once in the sequence's
    /// prompt plus already-generated history, applied before each draw. In [-2, 2]; 0 disables it.
    /// </summary>
    public float PresencePenalty { get; init; } = 0f;

    /// <summary>
    /// Additive penalty scaled by how often each token has appeared in the sequence's prompt plus
    /// already-generated history, applied before each draw. In [-2, 2]; 0 disables it.
    /// </summary>
    public float FrequencyPenalty { get; init; } = 0f;

    /// <summary>
    /// Constant per-token logit offsets added before every draw, keyed by token id. In [-100, 100] per
    /// entry; null disables the bias.
    /// </summary>
    public IReadOnlyDictionary<int, float>? LogitBias { get; init; } = null;

    /// <summary>True when any of the two penalties or the logit bias needs its logits transform applied.</summary>
    public bool HasLogitTransform => PresencePenalty != 0f || FrequencyPenalty != 0f || LogitBias is { Count: > 0 };

    /// <summary>True when the sampler decodes greedily (argmax) instead of sampling.</summary>
    public bool Greedy => Temperature <= 0f;
}
