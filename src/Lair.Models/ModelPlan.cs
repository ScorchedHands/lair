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
/// <c>&lt;prefix&gt;rope.freq_base</c>, <c>&lt;prefix&gt;rope.freq_base_swa</c>,
/// <c>&lt;prefix&gt;attention.layer_norm_rms_epsilon</c>,
/// <c>&lt;prefix&gt;vocab_size</c>), with <c>&lt;prefix&gt;attention.head_length</c> as the preferred source
/// for an explicit head dimension and <c>&lt;prefix&gt;attention.key_length</c> /
/// <c>&lt;prefix&gt;attention.value_length</c> accepted as equivalent sources; the head dimension no longer
/// has to split the hidden size exactly. A declared <c>&lt;prefix&gt;vocab_size</c> wins; without it the size
/// is the length of <c>tokenizer.ggml.tokens</c>. The declared feature flags read
/// <c>&lt;prefix&gt;attention.use_bias</c>, <c>&lt;prefix&gt;attention.norm_bias</c>, and
/// <c>&lt;prefix&gt;attention.use_qk_norm</c>; a declaration overrides the resolved profile's feature default for
/// the first and third in either direction, <c>attention.norm_bias</c> defaults to <c>false</c> when the key is
/// absent, a declared false forces per-head QK-norm off, and an absent or true declaration keeps the resolved
/// profile's QK-norm mode. The norm placement comes from the resolved profile alone: no metadata key selects
/// it today. The block-norm epsilon reads <c>&lt;prefix&gt;attention.layer_norm_epsilon</c>, resolving to
/// <c>0f</c> when the key is absent, and feeds only the LayerNorm block norms; the RMS epsilon from
/// <c>&lt;prefix&gt;attention.layer_norm_rms_epsilon</c> keeps feeding the RMS norms and the QK-norm. The declared softcaps read
/// <c>&lt;prefix&gt;attn_logit_softcapping</c> and <c>&lt;prefix&gt;final_logit_softcapping</c>, each
/// resolving to <c>0f</c> (disabled) when the key is absent and to a positive finite number otherwise. The
/// declared attention window reads <c>&lt;prefix&gt;attention.sliding_window</c>, resolving to <c>0</c>
/// (unbounded) when the key is absent and to a positive integer otherwise. Which layers that window masks comes
/// from the resolved profile alone, like the norm placement and the input-embedding gain: no metadata key decides
/// it today. The sparse-FFN slot reads <c>&lt;prefix&gt;expert_count</c>, <c>&lt;prefix&gt;expert_used_count</c>,
/// <c>&lt;prefix&gt;expert_shared_count</c>, <c>&lt;prefix&gt;expert_weights_scale</c>, and
/// <c>&lt;prefix&gt;expert_weights_norm</c>: a declared expert count resolves the descriptor against the resolved
/// profile's declared shape, and a file declaring no expert count keeps the dense gate/up/down block. The same slot reads
/// <c>&lt;prefix&gt;expert_layout</c>, <c>&lt;prefix&gt;expert_rank</c>, and <c>&lt;prefix&gt;sparse_ffn_strategy</c>: the first two say how
/// the block's experts are stored, and the third resolves against the resolved profile's default into the plan's
/// expert-computation variant — a declaration can override that default in either direction but cannot select storage the
/// file does not declare. The rope
/// scaling reads <c>&lt;prefix&gt;rope.scaling.type</c>, <c>&lt;prefix&gt;rope.scaling.factor</c>,
/// <c>&lt;prefix&gt;rope.scaling.original_context_length</c>, <c>&lt;prefix&gt;rope.scaling.attn_factor</c>, and the
/// YaRN operands <c>&lt;prefix&gt;rope.scaling.yarn_log_multiplier</c>, <c>&lt;prefix&gt;rope.scaling.yarn_ext_factor</c>,
/// <c>&lt;prefix&gt;rope.scaling.yarn_beta_fast</c>, and <c>&lt;prefix&gt;rope.scaling.yarn_beta_slow</c>: an absent
/// variant declaration keeps the standard unscaled rotation and the declared <c>yarn</c> variant resolves the
/// scaling plan. The rotary slice reads <c>&lt;prefix&gt;rope.dimension_count</c>, resolving to the full head
/// dimension when the key is absent and to a declared positive even count not exceeding it otherwise.
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

    /// <summary>
    /// Declared <c>&lt;prefix&gt;rope.freq_base_swa</c>: the rotary base a layer that carries this plan's attention
    /// window rotates at. Absent the key it resolves to the same base a file declaring no rotary frequency is read at,
    /// and a declared value must land in the float range as a positive finite number — anything else fails naming the
    /// key and the declared value. It reaches only layers <see cref="AttentionWindowForLayer"/> narrows; an unbounded
    /// layer rotates at <see cref="RopeTheta"/>, and see <see cref="RopeThetaForLayer"/> for what one layer actually
    /// rotates at. A plan whose every layer is unbounded never consults it.
    /// </summary>
    public float RopeWindowedTheta { get; init; } = DefaultRopeTheta;

    /// <summary>The rotary base a file declaring no <c>&lt;prefix&gt;rope.freq_base</c> is read at.</summary>
    public const float DefaultRopeTheta = 10_000f;

    public required float EpsRmsNorm { get; init; }

    /// <summary>
    /// The epsilon under the root of the LayerNorm block norms: <c>0f</c> unless the file declares
    /// <c>&lt;prefix&gt;attention.layer_norm_epsilon</c>. The RMS norms and the QK-norm keep
    /// <see cref="EpsRmsNorm"/> whatever this resolves to.
    /// </summary>
    public float EpsLayerNorm { get; init; }

    public required int VocabSize { get; init; }

    /// <summary>Whether the tensor table carries an untied output matrix, i.e. the output head is not tied to the token embedding.</summary>
    public required bool HasOutputWeights { get; init; }

    /// <summary>Resolved attention bias: the declared <c>&lt;prefix&gt;attention.use_bias</c>, or the resolved profile's default when that key is absent — whether each layer's attention projections carry additive biases: the q/k/v bias vectors are required at load, and an output-projection bias is used when the file carries one.</summary>
    public required bool UseAttentionBias { get; init; }

    /// <summary>Declared <c>&lt;prefix&gt;attention.norm_bias</c>: whether the norm layers carry additive bias vectors.</summary>
    public required bool UseNormBias { get; init; }

    /// <summary>Resolved QK-norm mode: a declared <c>&lt;prefix&gt;attention.use_qk_norm=false</c> forces <see cref="QkNormMode.Off"/>; absent the key resolves to the resolved profile's mode, and a declaration of <c>true</c> keeps that profile's own weighted form when it has one, or selects the per-head weighted form on an architecture whose profile defaults it off.</summary>
    public required QkNormMode QkNorm { get; init; }

    /// <summary>The resolved profile's norm placement: whether each sublayer output is normalized again with its own vector before joining the residual stream, or only the streams feeding each sublayer are normalized. No metadata key decides it.</summary>
    public required NormPlacement NormPlacement { get; init; }

    /// <summary>Declared <c>&lt;prefix&gt;attn_logit_softcapping</c>: the attention-score softcap applied after scaling and before the softmax, or <c>0f</c> when the key is absent (disabled).</summary>
    public float AttentionSoftcap { get; init; }

    /// <summary>Declared <c>&lt;prefix&gt;final_logit_softcapping</c>: the final-logit softcap applied after the output-head product and before any softmax or sampling, or <c>0f</c> when the key is absent (disabled).</summary>
    public float FinalLogitSoftcap { get; init; }

    /// <summary>
    /// Declared <c>&lt;prefix&gt;attention.sliding_window</c>: the attention window in positions. At query position
    /// <c>i</c> a layer that carries this window reads <c>[max(0, i - window + 1), i]</c> on every path — the causal
    /// decode range and the encoder's bidirectional pass alike: a declared window reaches behind the query and never
    /// grants visibility past the query itself. A layer the plan resolves to zero attends the whole sequence in both
    /// directions instead; see <see cref="AttentionWindowForLayer"/> for what one layer actually resolves. <c>0</c> when
    /// the key is absent means unbounded, and <see cref="WindowInterleave"/> decides which layers the declared value
    /// narrows. Every position's K/V is still written to and stays readable in the KV cache — the window bounds what a
    /// layer reads, not what the sequence stores, because another layer of the same model may attend the whole range.
    /// </summary>
    public int AttentionWindow { get; init; }

    /// <summary>
    /// The resolved profile's window interleave: whether the declared attention window narrows the score range of
    /// every layer, only the even-indexed layers leaving the odd-indexed ones unbounded, or all but the last layer of
    /// each group of six leaving that one unbounded. No metadata key decides it. A resolved <see cref="AttentionWindow"/>
    /// of <c>0</c> leaves every layer unbounded whatever the pattern.
    /// </summary>
    public WindowInterleave WindowInterleave { get; init; } = WindowInterleave.AllLayers;

    /// <summary>
    /// The window one layer reads within, resolved here from <see cref="AttentionWindow"/> against
    /// <see cref="WindowInterleave"/> so the plan carries the per-layer value lair.kv's block selection honors: every
    /// layer under <see cref="WindowInterleave.AllLayers"/>, even-indexed layers only under
    /// <see cref="WindowInterleave.AlternatingLocalFirst"/>, where the odd-indexed layers resolve unbounded, and all but
    /// the last layer of each group of six under <see cref="WindowInterleave.SixLayerGroupsLocalFirst"/>, where that
    /// last layer resolves unbounded. A resolved window of <c>0</c> is unbounded whatever the pattern, and a pattern this
    /// engine cannot resolve fails naming it instead of silently masking some other set of layers. The value bounds what
    /// that layer reads; storage stays whole.
    /// </summary>
    public int AttentionWindowForLayer(int layer)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(layer);

        switch (WindowInterleave)
        {
            case WindowInterleave.AllLayers:
                return AttentionWindow;

            case WindowInterleave.AlternatingLocalFirst:
                return layer % 2 == 0 ? AttentionWindow : 0;

            case WindowInterleave.SixLayerGroupsLocalFirst:
                return layer % 6 == 5 ? 0 : AttentionWindow;

            default:
                throw new ModelException($"window interleave '{WindowInterleave}' is not implemented by the forward engine");
        }
    }

    /// <summary>
    /// The rotary base one layer rotates at, resolved here from this plan's pair of bases against its own window so a
    /// model whose reference graph rotates its windowed layers at a different frequency than its unbounded ones reads
    /// each layer the way that graph does: <see cref="RopeWindowedTheta"/> for a layer carrying a window,
    /// <see cref="RopeTheta"/> for one attending the whole range. A plan whose every layer is unbounded resolves to
    /// <see cref="RopeTheta"/> at every layer whatever the declared second base says.
    /// </summary>
    public float RopeThetaForLayer(int layer) => AttentionWindowForLayer(layer) > 0 ? RopeWindowedTheta : RopeTheta;

    /// <summary>
    /// The resolved profile's input-embedding gain: <c>sqrt(hidden size)</c> on a profile whose reference graph
    /// scales the token-embedding row before the first layer, <c>0f</c> everywhere else. No metadata key decides
    /// it, and the conversion leaves the gain out of the weights.
    /// </summary>
    public float InputEmbeddingScale { get; init; }

    /// <summary>The resolved profile's tensor map: the GGUF names behind every logical tensor role.</summary>
    public required TensorMap Tensors { get; init; }

    public required NormStrategy Norm { get; init; }

    public required ActivationStrategy Activation { get; init; }

    public required BiasPolicy Bias { get; init; }

    public required RopeStrategy Rope { get; init; }

    /// <summary>
    /// The rope-scaling shape resolved from the file's declared <c>&lt;prefix&gt;rope.scaling.*</c> metadata, or
    /// <c>null</c> when the file declares no scaling variant: the engine then rotates the full head with the
    /// unscaled angles. A resolved plan names the variant, the multiplier, the original context length, the
    /// attention factor the score path applies, and the correction scale and beta pair the angle tables blend.
    /// </summary>
    public RopeScalingPlan? RopeScaling { get; init; }

    /// <summary>
    /// Declared <c>&lt;prefix&gt;rope.dimension_count</c>: the leading head dimensions the rotation covers, with
    /// the remaining dimensions passed through unchanged. <c>0</c> when the key is absent means the full head
    /// dimension, and a declared value must be an even count not exceeding the head dimension.
    /// </summary>
    public int RopeRotaryDimensions { get; init; }

    /// <summary>
    /// The sparse-FFN shape resolved from the file's declared expert metadata (<c>&lt;prefix&gt;expert_count</c>,
    /// <c>&lt;prefix&gt;expert_used_count</c>, <c>&lt;prefix&gt;expert_shared_count</c>,
    /// <c>&lt;prefix&gt;expert_weights_scale</c>, and <c>&lt;prefix&gt;expert_weights_norm</c>) against the
    /// resolved profile's sparse-FFN shape, or <c>null</c> when the file declares no expert count and the dense
    /// gate/up/down block runs. A resolved descriptor names the router and per-expert tensor roles every expert
    /// of the block is resident under: all experts stay in RAM, so routing selects among them rather than
    /// fetching them. It also carries how those experts are stored — the declared <c>&lt;prefix&gt;expert_layout</c>
    /// (stacked unless the file declares a shared basis), that layout's rank when it has one, and the tensor roles a
    /// factorized declaration reaches for. Which of those shapes anything computes with is the strategy value below.
    /// </summary>
    public SparseFfnDescriptor? SparseFfn { get; init; }

    /// <summary>
    /// The expert-computation variant this model resolves to: the declared <c>&lt;prefix&gt;sparse_ffn_strategy</c>
    /// overriding the resolved profile's default, exactly as the QK-norm mode resolves. <c>null</c> when the
    /// file declares no experts and no block routes anything. A value names what the load decided, not what proved
    /// runnable: a variant the forward engine cannot yet run stays recorded here and reports at engine open instead of
    /// falling back to another strategy invisibly.
    /// </summary>
    public SparseFfnStrategy? SparseFfnStrategy { get; init; }

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
            EpsLayerNorm,
            VocabSize,
            HasOutputWeights,
            Architecture,
            AttentionSoftcap,
            FinalLogitSoftcap,
            InputEmbeddingScale);
}
