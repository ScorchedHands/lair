using Lair.Loader;

namespace Lair.Models;

/// <summary>
/// Immutable resolved model plan, produced once per load from a <see cref="ModelProfile"/> plus the GGUF
/// container's KV metadata and one fact from its tensor table (whether an untied output matrix is present).
/// It carries the resolved profile name and metadata prefix, the concrete geometry, the profile's tensor map,
/// the strategy values the forward engine switches on, the declared feature flags, and the sparse-FFN slot.
/// Metadata keys follow the canonical GGUF naming (<c>&lt;prefix&gt;embedding_length</c>,
/// <c>&lt;prefix&gt;block_count</c>, <c>&lt;prefix&gt;attention.head_count</c>,
/// <c>&lt;prefix&gt;attention.head_count_kv</c>, <c>&lt;prefix&gt;feed_forward_length</c>,
/// <c>&lt;prefix&gt;rope.freq_base</c>, <c>&lt;prefix&gt;attention.layer_norm_rms_epsilon</c>,
/// <c>&lt;prefix&gt;vocab_size</c>), with <c>&lt;prefix&gt;attention.head_length</c> as the preferred source
/// for an explicit head dimension and <c>&lt;prefix&gt;attention.key_length</c> /
/// <c>&lt;prefix&gt;attention.value_length</c> accepted as equivalent sources; the head dimension no longer
/// has to split the hidden size exactly. A declared <c>&lt;prefix&gt;vocab_size</c> wins; without it the size
/// is the length of <c>tokenizer.ggml.tokens</c>. The declared feature flags read
/// <c>&lt;prefix&gt;attention.use_bias</c>, <c>&lt;prefix&gt;attention.norm_bias</c>, and
/// <c>&lt;prefix&gt;attention.use_qk_norm</c>; the first two default to <c>false</c> when the key is absent,
/// a declared false forces per-head QK-norm off, and an absent or true declaration keeps the resolved
/// profile's QK-norm mode. The norm placement comes from the resolved profile alone: no metadata key selects
/// it today. The declared softcaps read
/// <c>&lt;prefix&gt;attn_logit_softcapping</c> and <c>&lt;prefix&gt;final_logit_softcapping</c>, each
/// resolving to <c>0f</c> (disabled) when the key is absent and to a positive finite number otherwise. The
/// declared attention window reads <c>&lt;prefix&gt;attention.sliding_window</c>, resolving to <c>0</c>
/// (unbounded) when the key is absent and to a positive integer otherwise.
/// </summary>
public sealed record ModelPlan
{
    public required string Architecture { get; init; }

    /// <summary>
    /// Identity of the resolved profile: the architecture name itself for a registered profile, or the
    /// generic llama-shaped fallback identity when an unregistered architecture was accepted through the
    /// exact tensor-table match.
    /// </summary>
    public required string Profile { get; init; }

    /// <summary>Prefix the resolved profile's metadata keys use, e.g. <c>"llama."</c>.</summary>
    public required string Prefix { get; init; }

    public required int LayerCount { get; init; }

    public required int HiddenSize { get; init; }

    public required int HeadCount { get; init; }

    public required int KvHeadCount { get; init; }

    public required int HeadDim { get; init; }

    public required int IntermediateSize { get; init; }

    public required float RopeTheta { get; init; }

    public required float EpsRmsNorm { get; init; }

    public required int VocabSize { get; init; }

    /// <summary>Whether the tensor table carries an untied output matrix, i.e. the output head is not tied to the token embedding.</summary>
    public required bool HasOutputWeights { get; init; }

    /// <summary>Declared <c>&lt;prefix&gt;attention.use_bias</c>: whether the q/k/v/o projections carry additive biases.</summary>
    public required bool UseAttentionBias { get; init; }

    /// <summary>Declared <c>&lt;prefix&gt;attention.norm_bias</c>: whether the norm layers carry additive bias vectors.</summary>
    public required bool UseNormBias { get; init; }

    /// <summary>Resolved per-head QK-norm mode: a declared <c>&lt;prefix&gt;attention.use_qk_norm=false</c> forces <see cref="QkNormMode.Off"/>; absent the key resolves to the resolved profile's mode, and a declaration of <c>true</c> selects the weighted form on an architecture whose profile defaults it off.</summary>
    public required QkNormMode QkNorm { get; init; }

    /// <summary>The resolved profile's norm placement: whether each sublayer output is normalized again with its own vector before joining the residual stream, or only the streams feeding each sublayer are normalized. No metadata key decides it.</summary>
    public required NormPlacement NormPlacement { get; init; }

    /// <summary>Declared <c>&lt;prefix&gt;attn_logit_softcapping</c>: the attention-score softcap applied after scaling and before the softmax, or <c>0f</c> when the key is absent (disabled).</summary>
    public float AttentionSoftcap { get; init; }

    /// <summary>Declared <c>&lt;prefix&gt;final_logit_softcapping</c>: the final-logit softcap applied after the output-head product and before any softmax or sampling, or <c>0f</c> when the key is absent (disabled).</summary>
    public float FinalLogitSoftcap { get; init; }

    /// <summary>
    /// Declared <c>&lt;prefix&gt;attention.sliding_window</c>: the attention score window in positions. At query
    /// position <c>i</c> the causal range narrows to <c>[max(0, i - window + 1), i]</c>, and the bidirectional
    /// range to <c>[max(0, i - window + 1), i + window]</c> clamped to the sequence. <c>0</c> when the key is
    /// absent means unbounded. The value masks scores only: every position is still written to and readable
    /// from the KV cache.
    /// </summary>
    public int AttentionWindow { get; init; }

    /// <summary>The resolved profile's tensor map: the GGUF names behind every logical tensor role.</summary>
    public required TensorMap Tensors { get; init; }

    public required NormStrategy Norm { get; init; }

    public required ActivationStrategy Activation { get; init; }

    public required BiasPolicy Bias { get; init; }

    public required RopeStrategy Rope { get; init; }

    /// <summary>Declared sparse-FFN shape, or <c>null</c> for the dense gate/up/down block every supported profile declares today.</summary>
    public SparseFfnDescriptor? SparseFfn { get; init; }

    /// <summary>The flat scalar view of this plan's resolved geometry.</summary>
    public ModelDesc ToDesc() =>
        new(
            LayerCount,
            HiddenSize,
            HeadCount,
            KvHeadCount,
            HeadDim,
            IntermediateSize,
            RopeTheta,
            EpsRmsNorm,
            VocabSize,
            HasOutputWeights,
            Architecture,
            AttentionSoftcap,
            FinalLogitSoftcap);
}
