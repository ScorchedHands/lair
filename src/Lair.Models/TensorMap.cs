using System.Globalization;

using Lair.Contracts;

namespace Lair.Models;

/// <summary>
/// Identifies one logical tensor role the forward engine consumes, independent of the GGUF names a profile
/// maps it to.
/// </summary>
public enum TensorRole
{
    TokenEmbedding,

    /// <summary>
    /// One layer's fused q/k/v projection buffer: a single matrix whose rows are every head of q, then every head of k,
    /// then every head of v. Only a family whose conversion ships that layout names it, and the engine reads each side
    /// as its own slice rather than splitting the tensor into three matrices — see <see cref="QkvLayout"/>. A profile
    /// whose block projects through three separate tensors leaves this role out of its map entirely.
    /// </summary>
    AttentionQkv,

    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    FeedForwardGate,
    FeedForwardUp,
    FeedForwardDown,
    AttentionNorm,
    FeedForwardNorm,
    AttentionQueryBias,
    AttentionKeyBias,
    AttentionValueBias,
    AttentionOutputBias,
    OutputNorm,
    OutputHead,
    OutputBias,
    AttentionQueryNorm,
    AttentionKeyNorm,
    AttentionNormBias,
    FeedForwardNormBias,
    OutputNormBias,
    PostAttentionNorm,
    PostFeedForwardNorm,
    MoeRouter,
    MoeExpertGate,
    MoeExpertUp,
    MoeExpertDown,
    MoeSharedExpertGate,
    MoeSharedExpertUp,
    MoeSharedExpertDown,

    /// <summary>
    /// The shared expert's per-token scalar gate: one row of the hidden size whose score selects the shared
    /// expert for this token. It is a distinct tensor from <see cref="MoeSharedExpertGate"/> — that role names the
    /// shared block's SwiGLU gate matrix, this one names the selector vector the reference multiplies the shared
    /// output by — and only some families ship it at all.
    /// </summary>
    MoeSharedExpertInputGate,

    /// <summary>
    /// The factorized layout's shared input basis: one matrix per layer read once over the hidden stream to produce
    /// a rank-sized vector every expert of that layer consumes. Only a file declaring
    /// <see cref="SparseFfnLayout.FactorizedSharedBasis"/> ships it, and no dense-stacked family names it at all.
    /// </summary>
    MoeExpertInputBasis,

    /// <summary>
    /// The factorized layout's per-expert coefficients: one stacked tensor whose outermost dimension indexes experts
    /// exactly as the dense expert roles do, holding each expert's contribution in the shared basis' compressed space.
    /// </summary>
    MoeExpertCoefficients,

    /// <summary>
    /// The factorized layout's shared output projection: one matrix per layer that expands the accumulated
    /// coefficient-space sum back to the hidden stream, which is what lets a weighted expert sum collapse into a single
    /// application instead of one down-projection per expert.
    /// </summary>
    MoeExpertOutputBasis,

    /// <summary>
    /// One linear-attention block's z gate: its own projection rows (one value-head slice wide) whose sigmoid gates the
    /// normalized head outputs at this token. Only a family with gated-delta-net blocks names it; the fused qkv buffer
    /// of such a block stays named by <see cref="AttentionQkv"/> because that tensor carries exactly its conv channels.
    /// </summary>
    LinearAttentionGate,

    // The recurrent-state operands one gated-delta-net block reads: each value head's stored decay weight (the name
    // this family spells without a .weight suffix), the merged beta and alpha projections' rows split into two named
    // tensors here because the file ships them as two, the depthwise convolution kernel over every fused channel row,
    // each head's dt bias added before softplus, the shared per-head output norm vector, and the block output matrix.

    /// <summary>Per value head stored decay weight of a gated-delta-net block.</summary>
    SsmDecayWeight,

    /// <summary>The alpha projection rows feeding one linear block's decay gate (softplus over these plus each head's dt bias).</summary>
    SsmAlphaProjection,

    /// <summary>The beta projection rows selecting how much of the value a token writes into its recurrent state.</summary>
    SsmBetaProjection,

    /// <summary>The causal depthwise convolution kernel: one row per fused qkv channel carrying that channel's taps contiguously.</summary>
    SsmConvolutionKernel,

    /// <summary>Per value head dt bias added to the raw alpha before softplus.</summary>
    SsmTimeStepBias,

    /// <summary>The shared norm vector each linear block normalizes its per-head outputs with (head-dimension wide).</summary>
    SsmHeadNorm,

    /// <summary>A gated-delta-net block's output projection over the concatenated value-head widths.</summary>
    SsmOutputProjection,

    // The sparse indexer every full-attention block of this family scores its selection through: one query region per
    // scoring head, the shared single key projection, and their component-width norm vectors. Attention itself stays
    // named by the ordinary q/k/v/output roles — selection restricts which cache cells those read, not what they are.

    /// <summary>The indexer's stacked query projections: one component region per scoring head.</summary>
    IndexerQueryProjection,

    /// <summary>The indexer's single key projection every cached cell passes through before pooling.</summary>
    IndexerKeyProjection,

    /// <summary>The norm vector the indexer queries are RMS-normalized with (component-width).</summary>
    IndexerQueryNorm,

    /// <summary>The norm vector pooled keys are RMS-normalized with (component-width).</summary>
    IndexerKeyNorm,

    // Hyper-connections: each block mixes a widened residual stream around both its sublayers. The inject scores pick
    // which streams feed the sublayer, the down/up pair compresses the flattened stream through the declared low rank,
    // and the norm vector normalizes every element of that flattened width before mixing. Both sides carry the same
    // four tensors under their own names; a final output-side trio mixes the streams into the logits' hidden side.

    /// <summary>Per-stream inject scores selecting which residual streams feed this block's attention.</summary>
    AttentionStreamInject,

    /// <summary>The low-rank expansion adding this block's attention output back across the flattened stream width.</summary>
    AttentionStreamUp,

    /// <summary>The projection compressing the flattened stream to the declared rank before the attention sublayer.</summary>
    AttentionStreamDown,

    /// <summary>The norm vector covering the whole flattened residual feeding this block's attention (flattened-width).</summary>
    AttentionStreamNorm,

    /// <summary>Per-stream inject scores selecting which residual streams feed this block's feed-forward.</summary>
    FeedForwardStreamInject,

    /// <summary>The low-rank expansion adding this block's feed-forward output back across the flattened stream width.</summary>
    FeedForwardStreamUp,

    /// <summary>The projection compressing the flattened stream to the declared rank before the feed-forward sublayer.</summary>
    FeedForwardStreamDown,

    /// <summary>The norm vector covering the whole flattened residual feeding this block's feed-forward (flattened-width).</summary>
    FeedForwardStreamNorm,

    /// <summary>The global output-side projection compressing the final flattened stream to the declared rank before mixing.</summary>
    OutputStreamDown,

    /// <summary>The global norm vector covering the final flattened residual stream (flattened-width).</summary>
    OutputStreamNorm,

    /// <summary>The global low-rank expansion that mixes the final streams into each token's hidden row.</summary>
    OutputStreamUp,

    // The per-layer ngram input-embedding tensors this family ships on its declared PLE layers: a causal convolution
    // kernel over the gathered table rows, the key/value projections of the small attention mixing them with the
    // residual, and their three norm vectors. All are optional in every map — only files declaring PLE carry them, and
    // nothing resident reads them through this map yet (the gather seam resolves that tensor by name).

    /// <summary>A PLE block's causal convolution kernel over its gathered table rows.</summary>
    PleConvolutionKernel,

    /// <summary>The key projection of a PLE block's small mixing attention.</summary>
    PleKeyProjection,

    /// <summary>The value projection returning the mixed per-layer input embedding to the hidden stream.</summary>
    PleValueProjection,

    /// <summary>The norm vector covering this PLE block's convolved table rows before mixing.</summary>
    PleConvolutionNorm,

    /// <summary>The norm vector covering a PLE block's projected keys (hidden-width).</summary>
    PleKeyNorm,

    /// <summary>The norm vector covering the hidden row a PLE block queries against (hidden-width).</summary>
    PleQueryNorm,
}

/// <summary>Whether a tensor name is a single global name or a per-block name carrying a <c>{layer}</c> placeholder.</summary>
public enum TensorScope
{
    Global,
    PerLayer,
}

/// <summary>
/// One slot of a profile's <see cref="TensorMap"/>: the role it fills, the GGUF name it maps to (a
/// <c>{layer}</c> template for <see cref="TensorScope.PerLayer"/> roles), whether the tensor is required,
/// and alias names tried after the primary name. Aliases are declared against the same scope rules as the
/// primary name; the llama and qwen2 profiles declare none, so their map resolves to exactly the canonical
/// GGUF names.
/// </summary>
public sealed record TensorDescriptor(
    TensorRole Role,
    TensorScope Scope,
    string Name,
    bool Required,
    IReadOnlyList<string> Aliases)
{
    /// <summary>
    /// The concrete GGUF tensor name for <paramref name="layer"/>: the primary name with the placeholder
    /// substituted for per-layer roles, and the name itself for global roles (the layer is ignored there).
    /// </summary>
    public string NameFor(int layer) =>
        Scope == TensorScope.PerLayer ? Name.Replace("{layer}", layer.ToString(CultureInfo.InvariantCulture)) : Name;

    /// <summary>
    /// Finds this descriptor's tensor in <paramref name="tensors"/>: the primary name first, then each alias
    /// in order. Returns <c>null</c> when no candidate is present.
    /// </summary>
    public LogicalTensor? Resolve(LogicalTensorTable tensors, int layer = -1)
    {
        LogicalTensor? hit = tensors.Find(NameFor(layer));
        if (hit is not null)
        {
            return hit;
        }

        foreach (string alias in Aliases)
        {
            hit = tensors.Find(Scope == TensorScope.PerLayer ? alias.Replace("{layer}", layer.ToString(CultureInfo.InvariantCulture)) : alias);
            if (hit is not null)
            {
                return hit;
            }
        }

        return null;
    }
}

/// <summary>
/// A profile's complete tensor map: exactly one <see cref="TensorDescriptor"/> per <see cref="TensorRole"/> except the
/// roles a family's block may not ship at all — the sparse-FFN roles, which a dense profile leaves out of its map
/// entirely, and the fused qkv buffer, which only a family projecting through one fused tensor names. Resolved once
/// at load and carried immutably by the <see cref="ModelPlan"/>. The forward engine looks tensors up through this map
/// instead of through hardcoded name suffixes.
/// </summary>
public sealed class TensorMap
{
    private readonly TensorDescriptor[] _byRole;

    public TensorMap(IReadOnlyList<TensorDescriptor> descriptors)
    {
        ArgumentNullException.ThrowIfNull(descriptors);

        _byRole = new TensorDescriptor[Enum.GetValues<TensorRole>().Length];
        foreach (TensorDescriptor descriptor in descriptors)
        {
            if (_byRole[(int)descriptor.Role] is not null)
            {
                throw new ModelException($"tensor map declares role '{descriptor.Role}' more than once");
            }

            _byRole[(int)descriptor.Role] = descriptor;
        }

        foreach (TensorRole role in Enum.GetValues<TensorRole>())
        {
            if (_byRole[(int)role] is null && !IsProfileOptionalRole(role))
            {
                throw new ModelException($"tensor map is missing a descriptor for role '{role}'");
            }
        }

        // Sparse and fused-layout roles stay absent from a profile whose block ships none of them: the map names only
        // the tensors its forward can consume, so such a map carries no router, expert, or fused-qkv descriptor and a
        // tensor under one of those names is material its map cannot match.
        List<TensorDescriptor> declared = [];
        foreach (TensorDescriptor descriptor in _byRole)
        {
            if (descriptor is not null)
            {
                declared.Add(descriptor);
            }
        }

        Descriptors = declared;
    }

    /// <summary>Every descriptor, in role order.</summary>
    public IReadOnlyList<TensorDescriptor> Descriptors { get; }

    /// <summary>The descriptor for one role.</summary>
    public TensorDescriptor this[TensorRole role] => _byRole[(int)role];

    /// <summary>
    /// Whether a role belongs to the sparse-FFN block: the router, the expert matrices dense or shared, the
    /// shared expert's scalar gate vector, and the factorized layout's basis/coefficient/output-basis tensors.
    /// A dense profile names these roles for the load-time report only — its forward consumes none of them,
    /// and the generic llama-shaped fallback treats a tensor under a sparse name as material its map cannot
    /// match rather than as a matchable optional role.
    /// </summary>
    public static bool IsSparseRole(TensorRole role) =>
        role is TensorRole.MoeRouter
            or TensorRole.MoeExpertGate
            or TensorRole.MoeExpertUp
            or TensorRole.MoeExpertDown
            or TensorRole.MoeSharedExpertGate
            or TensorRole.MoeSharedExpertUp
            or TensorRole.MoeSharedExpertDown
            or TensorRole.MoeSharedExpertInputGate
            or TensorRole.MoeExpertInputBasis
            or TensorRole.MoeExpertCoefficients
            or TensorRole.MoeExpertOutputBasis;

    /// <summary>
    /// Whether a role belongs to an architecture family other than the dense and MoE shapes — this build's gated-delta-net
    /// linear-attention, sparse-indexer, hyper-connection, and per-layer-input-embedding roles. No existing profile names
    /// them; only a family that runs those blocks declares them in its map. Like the sparse roles, they stay out of the
    /// llama-shaped fallback's nameable set: a tensor under one of these names is material a dense-shaped file cannot run.
    /// </summary>
    public static bool IsFamilySpecificRole(TensorRole role) =>
        IsSparseRole(role)
            || role is TensorRole.LinearAttentionGate
            or TensorRole.SsmDecayWeight
            or TensorRole.SsmAlphaProjection
            or TensorRole.SsmBetaProjection
            or TensorRole.SsmConvolutionKernel
            or TensorRole.SsmTimeStepBias
            or TensorRole.SsmHeadNorm
            or TensorRole.SsmOutputProjection
            or TensorRole.IndexerQueryProjection
            or TensorRole.IndexerKeyProjection
            or TensorRole.IndexerQueryNorm
            or TensorRole.IndexerKeyNorm
            or TensorRole.AttentionStreamInject
            or TensorRole.AttentionStreamUp
            or TensorRole.AttentionStreamDown
            or TensorRole.AttentionStreamNorm
            or TensorRole.FeedForwardStreamInject
            or TensorRole.FeedForwardStreamUp
            or TensorRole.FeedForwardStreamDown
            or TensorRole.FeedForwardStreamNorm
            or TensorRole.OutputStreamDown
            or TensorRole.OutputStreamNorm
            or TensorRole.OutputStreamUp
            or TensorRole.PleConvolutionKernel
            or TensorRole.PleKeyProjection
            or TensorRole.PleValueProjection
            or TensorRole.PleConvolutionNorm
            or TensorRole.PleKeyNorm
            or TensorRole.PleQueryNorm;

    /// <summary>
    /// Whether a profile's map may leave this role out entirely: the sparse-FFN roles above, which no dense block ships,
    /// every other family-specific role named by <see cref="IsFamilySpecificRole"/>, and
    /// <see cref="TensorRole.AttentionQkv"/>, which only a family whose conversion fuses q/k/v into one tensor names.
    /// A map that omits such a role has no name to resolve for it, so nothing may ask the indexer for its descriptor;
    /// every profile today either consumes the fused layout or leaves the role undeclared.
    /// </summary>
    public static bool IsProfileOptionalRole(TensorRole role) =>
        IsFamilySpecificRole(role) || role == TensorRole.AttentionQkv;
}
