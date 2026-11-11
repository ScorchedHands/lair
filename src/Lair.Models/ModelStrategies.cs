namespace Lair.Models;

/// <summary>
/// Normalization applied to each layer input stream and to the final stream before the output head.
/// Resolved once per model from its <see cref="ModelProfile"/>; the forward path switches on the value and
/// never dispatches through a per-token interface.
/// </summary>
public enum NormStrategy
{
    /// <summary>RMSNorm scaled by a per-layer weight vector: <c>src / sqrt(mean(src^2) + eps) * weight</c>.</summary>
    RmsNorm,

    /// <summary>
    /// LayerNorm scaled by a per-layer weight vector and shifted by its bias when one is present:
    /// <c>(src - mean) / sqrt(var + eps) * weight + bias</c> with population variance over the vector.
    /// </summary>
    LayerNorm,
}

/// <summary>
/// Gated feed-forward activation applied between the up and down projections. Resolved once per model from
/// its <see cref="ModelProfile"/>; the forward path switches on the value and never dispatches through a
/// per-token interface.
/// </summary>
public enum ActivationStrategy
{
    /// <summary><c>silu(gate) * up</c>.</summary>
    SwiGLU,

    /// <summary><c>gelu_tanh(gate) * up</c>: the tanh-approximated GELU gate over the same gate/up/down tensor block.</summary>
    GeGlu,

    /// <summary><c>gelu(gate) * up</c>: the exact GELU gate over the same gate/up/down tensor block.</summary>
    GeGluExact,
}

/// <summary>
/// How the engine decides whether a projection carries an additive bias. Resolved once per model from its
/// <see cref="ModelProfile"/>.
/// </summary>
public enum BiasPolicy
{
    /// <summary>A bias is applied exactly when its tensor is present in the file, per layer and per projection.</summary>
    ByTensorPresence,

    /// <summary>
    /// The resolved plan decides: <c>&lt;prefix&gt;attention.use_bias</c> overrides the profile's feature default for
    /// the attention biases (absent the key, a profile that defaults the feature on resolves it on), and
    /// <c>&lt;prefix&gt;attention.norm_bias</c> (absent means <c>false</c>) lands in the plan. A resolved-on attention
    /// bias makes every layer's q/k/v bias vectors required at load; a resolved-off one ignores present tensors with
    /// one structured log line naming them.
    /// </summary>
    Declared,
}

/// <summary>
/// Where each layer applies its norm vectors relative to the residual add. Resolved once per model from its
/// <see cref="ModelProfile"/>; no metadata key decides it today, and the forward path switches on the value and
/// never dispatches through a per-token interface.
/// </summary>
public enum NormPlacement
{
    /// <summary>
    /// Each sublayer consumes only the norm that precedes it: the attention input stream before the QKV GEMVs and
    /// the feed-forward input stream before the gate/up GEMVs, each added to unchanged afterwards. No tensor after a
    /// sublayer is normalized again before its residual add.
    /// </summary>
    PreOnly,

    /// <summary>
    /// Each sublayer output is normalized once more with its own norm vector before it joins the residual stream:
    /// the attention output after the output projection and before the first residual add, and the feed-forward
    /// output after the down projection and before the second. The pre-sublayer norms still apply exactly as under
    /// <see cref="PreOnly"/>; this adds a second norm on each output rather than replacing the first.
    /// </summary>
    PostSublayerOutputs,
}

/// <summary>
/// Normalization applied to the projected q and k streams after their projection (and any bias) and before RoPE —
/// either over each head's own slice or over the whole stream at once. Resolved once per model from its
/// <see cref="ModelProfile"/> default and the file's declared <c>&lt;prefix&gt;attention.use_qk_norm</c>; the forward
/// path switches on the value and never dispatches through a per-token interface.
/// </summary>
public enum QkNormMode
{
    /// <summary>No normalization: q and k go from their projection straight to RoPE.</summary>
    Off,

    /// <summary>
    /// RMS norm over each head's own <c>headDim</c> slice scaled by the layer's per-head weight vector, which the
    /// file must carry for every layer: <c>src / sqrt(mean(src_h^2) + eps) * w[i]</c>.
    /// </summary>
    PerHeadWeighted,

    /// <summary>
    /// RMS norm over each head's own <c>headDim</c> slice with the plan epsilon alone and no weight vector:
    /// <c>src / sqrt(mean(src_h^2) + eps)</c>, numerically identical to a weight of ones. It consumes no tensor, so
    /// it demands none and ignores any per-head norm vectors the file carries.
    /// </summary>
    PerHeadUnweighted,

    /// <summary>
    /// RMS norm over each side's whole projected stream — every head of q normalized jointly under one mean over that
    /// side's full width (<c>head_count × head_dim</c>, and likewise <c>kv_head_count × head_dim</c> for k), scaled
    /// element by element by the layer's stream-sized weight vector: <c>src / sqrt(mean(src^2) + eps) * w[i]</c>. This
    /// is the form the reference runs when it normalizes q and k before splitting them into heads, as the olmoe block
    /// does with one [hidden]-sized vector per layer; the file must carry a vector covering each side's whole width.
    /// </summary>
    StreamWeighted,
}

/// <summary>
/// Rotary position embedding pairing applied to the q and k heads. Resolved once per model from its
/// <see cref="ModelProfile"/>; the pair layout is the profile's, while the file's declared metadata can narrow
/// the rotation to a leading slice of the head and select a scaling variant for the angle tables.
/// </summary>
public enum RopeStrategy
{
    /// <summary>
    /// Adjacent-pair rotation over the full head dimension with no scaling variant: pair <c>i</c> occupies head
    /// dimensions <c>2i</c> and <c>2i + 1</c>. This is the pairing the llama-family checkpoints lay their rotary
    /// rows in, and the pairing the reference applies to them.
    /// </summary>
    StandardFullHead,

    /// <summary>
    /// Rotate-half (non-adjacent) pairing over the full head dimension with no scaling variant: pair <c>i</c>
    /// occupies head dimensions <c>i</c> and <c>i + headDim/2</c>. The qwen3 checkpoints lay their rotary rows in
    /// this order, so the engine must pair the head halves rather than adjacent dimensions to reproduce them.
    /// </summary>
    RotateHalfFullHead,
}

/// <summary>
/// How a sparse-FFN block's expert matrices are laid out in the file it loads from: one dense-shaped matrix per
/// role, or a factorized shape whose experts share an input basis and carry only their own coefficients. Resolved
/// once at load from the declared <c>&lt;prefix&gt;expert_layout</c> (absent means the stacked layout every family
/// ships today) and carried by the <see cref="SparseFfnDescriptor"/>, so a factorized file is recognizable before
/// any kernel decision runs.
/// </summary>
public enum SparseFfnLayout
{
    /// <summary>
    /// Dense-stacked expert storage: one tensor per role whose outermost dimension indexes experts, each expert block
    /// carrying the full matrix for its side of the gate/up/down chain. This is what every published MoE conversion
    /// ships and what the resident expert execution reads today.
    /// </summary>
    StackedExperts,

    /// <summary>
    /// Factorized expert storage: a shared input basis read once per layer plus one coefficient matrix per expert in
    /// that compressed space, expanded through one shared output projection. The layout names the tensors it needs;
    /// whether anything computes with them is decided by the resolved strategy, not here.
    /// </summary>
    FactorizedSharedBasis,
}

/// <summary>
/// Which expert-computation variant a sparse-FFN block resolves to: the resident dense-stacked chain, or the shared-
/// basis low-rank collapse over factorized coefficients. Resolved once per model from the resolved profile's default and
/// the file's declared <c>&lt;prefix&gt;sparse_ffn_strategy</c>, exactly as the QK-norm mode resolves against its profile
/// default, and never re-decided per token or per layer — a variant is chosen by the plan once, not dispatched inside the
/// forward. Selecting a variant is independent of whether any engine can run it: an unimplemented selection stays visible
/// in the plan and reports at engine open rather than falling back to what happens to work.
/// </summary>
public enum SparseFfnStrategy
{
    /// <summary>
    /// Every declared expert stays resident and routing selects among them, each selected expert running its own
    /// gate/up/down chain over the normalized stream. The dense-equivalent reference forward describes this variant,
    /// so it is what every sparse file executes today whatever its layout declaration says about factorization.
    /// </summary>
    DenseResident,

    /// <summary>
    /// Factor each routed expert as shared basis times per-expert coefficients: apply the input basis once per layer,
    /// accumulate small coefficient-space vectors across the selected experts under their routing weights, and expand
    /// that sum through one output projection instead of running a full down-projection per expert. It requires a file
    /// declaring <see cref="SparseFfnLayout.FactorizedSharedBasis"/> with its rank and tensor roles carried in the
    /// descriptor.
    /// </summary>
    LowRankSharedBasis,
}

/// <summary>
/// Which layers a model's declared attention window masks. Resolved once per model from its
/// <see cref="ModelProfile"/>; no metadata key decides it today, and the forward path switches on the value and
/// never dispatches through a per-token interface.
/// </summary>
public enum WindowInterleave
{
    /// <summary>Every layer's attention narrows its score range to the declared window.</summary>
    AllLayers,

    /// <summary>
    /// Even-indexed layers narrow their score ranges to the declared window and odd-indexed layers attend the whole
    /// range, which is how the gemma2 reference graph applies its window. A declared window of <c>0</c> leaves every
    /// layer unbounded whatever the pattern.
    /// </summary>
    AlternatingLocalFirst,
}
