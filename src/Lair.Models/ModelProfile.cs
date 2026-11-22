namespace Lair.Models;

/// <summary>
/// Load-time model profile for one <c>general.architecture</c> value: the metadata prefix its keys use, the
/// strategy values the forward engine switches on, the complete tensor map naming every logical tensor role,
/// and the sparse-FFN slot (unset while no profile declares experts). Resolved once at load into a
/// <see cref="ModelPlan"/> so the hot forward path stays concrete and never dispatches per token.
/// </summary>
public sealed record ModelProfile
{
    public required string Architecture { get; init; }

    /// <summary>Prefix every architecture-scoped metadata key of this profile carries, e.g. <c>"llama."</c>.</summary>
    public required string Prefix { get; init; }

    public required NormStrategy Norm { get; init; }

    public required ActivationStrategy Activation { get; init; }

    public required BiasPolicy Bias { get; init; }

    public required RopeStrategy Rope { get; init; }

    /// <summary>
    /// How this architecture's block projects q/k/v: three matrices the tensor map names one role each, or one fused
    /// buffer whose head regions each projection reads as its own slice of that payload. A profile defaults to
    /// <see cref="Lair.Models.QkvLayout.SeparateTensors"/>; no metadata key selects it, and a profile resolving the
    /// fused form names that buffer in its tensor map instead of leaving the role undeclared.
    /// </summary>
    public QkvLayout AttentionWeights { get; init; } = QkvLayout.SeparateTensors;

    /// <summary>
    /// Which tensor supplies this block's gate half: a matrix of its own named by the feed-forward gate role, or the
    /// leading rows of the up tensor for a conversion that ships the gated pair as one buffer and names no gate matrix.
    /// A profile defaults to <see cref="Lair.Models.FeedForwardGateSource.SeparateGateMatrix"/>; no metadata key selects it.
    /// </summary>
    public FeedForwardGateSource GateSource { get; init; } = FeedForwardGateSource.SeparateGateMatrix;

    public required TensorMap Tensors { get; init; }

    /// <summary>
    /// The sparse-FFN shape the profile's reference block names — the router and expert roles its tensor map
    /// declares and whether its reference graph routes through a shared expert — or <c>null</c> for a dense
    /// gate/up/down block. The counts the plan carries come from the file's declared expert metadata, never
    /// from the profile.
    /// </summary>
    public SparseFfnDescriptor? SparseFfn { get; init; }

    /// <summary>
    /// Feature default for which expert-computation variant a sparse-FFN block of this family resolves to, applied when
    /// the file declares no <c>&lt;prefix&gt;sparse_ffn_strategy</c>: a declaration overrides it in either direction and
    /// only then does the plan carry a strategy value at all. A profile defaults this to
    /// <see cref="Lair.Models.SparseFfnStrategy.DenseResident"/>, which is every registered family today, so selecting
    /// another variant takes a declaration rather than a family's own preference — and no default can name a variant the
    /// forward engine cannot run.
    /// </summary>
    public SparseFfnStrategy DefaultSparseFfnStrategy { get; init; } = SparseFfnStrategy.DenseResident;

    /// <summary>
    /// Feature default for QK-norm, applied when the file declares no
    /// <c>&lt;prefix&gt;attention.use_qk_norm</c>: a declared false overrides it to <see cref="QkNormMode.Off"/>
    /// whatever the profile defaults to, and a profile defaulting to a weighted mode makes that mode's norm vectors
    /// required at load for a file that declares nothing — each side-sized as its mode names them. A profile defaults
    /// this to <see cref="Lair.Models.QkNormMode.Off"/>.
    /// </summary>
    public QkNormMode DefaultQkNorm { get; init; } = QkNormMode.Off;

    /// <summary>
    /// Feature default for the attention biases, applied when the file declares no
    /// <c>&lt;prefix&gt;attention.use_bias</c>: a declaration overrides it in either direction, and a profile
    /// defaulting to <c>true</c> makes every layer's q/k/v bias tensors required at load for a file that declares
    /// nothing. A profile defaults this to <c>false</c>.
    /// </summary>
    public bool DefaultAttentionBias { get; init; }

    /// <summary>
    /// Where this architecture's forward places its norm vectors relative to the residual add: a profile defaults to
    /// <see cref="Lair.Models.NormPlacement.PreOnly"/>, and
    /// <see cref="Lair.Models.NormPlacement.PostSublayerOutputs"/> makes every layer's post-sublayer norm tensors
    /// required at load. No metadata key selects it: the profile decides, so a file carrying those tensors under a
    /// pre-only placement is a layout failure rather than ignored material.
    /// </summary>
    public NormPlacement NormPlacement { get; init; } = NormPlacement.PreOnly;

    /// <summary>
    /// Feature default for the input-embedding gain: when true the plan resolves a gain of
    /// <c>sqrt(hidden size)</c> that the engine multiplies every token-embedding row by before the first layer,
    /// the way the gemma-family reference graphs scale their embedding input. No metadata key selects it; the
    /// conversion leaves the gain out of the weights. A profile defaults this to <c>false</c>.
    /// </summary>
    public bool ScaleInputEmbeddingsBySqrtHidden { get; init; }

    /// <summary>
    /// Which layers this architecture's forward masks with its declared attention window: a profile defaults to
    /// <see cref="Lair.Models.WindowInterleave.AllLayers"/>,
    /// <see cref="Lair.Models.WindowInterleave.AlternatingLocalFirst"/> narrows the mask to the even-indexed layers
    /// and leaves the odd-indexed ones unbounded, which is how the gemma2 reference graph applies its window, and
    /// <see cref="Lair.Models.WindowInterleave.SixLayerGroupsLocalFirst"/> leaves the last layer of each group of six
    /// unbounded and masks the rest, which is how the reference applies this family's window. No metadata key selects
    /// it: the profile decides, and a declared window of <c>0</c> is unbounded whatever the pattern.
    /// </summary>
    public WindowInterleave WindowInterleave { get; init; } = WindowInterleave.AllLayers;

    /// <summary>
    /// Whether this family's blocks carry gated-delta-net linear attention, a sparse indexer on its full-attention
    /// layers, and widened hyper-connected residual streams: the profile whose map names those roles sets it true, and
    /// only such a profile resolves the per-layer geometry slot from the file's declared groups. A profile defaults to
    /// <c>false</c>; no metadata key selects it — which tensors a family ships is the profile's fact, not the file's.
    /// </summary>
    public bool DeclaresQwen4ExpSemantics { get; init; }
}
