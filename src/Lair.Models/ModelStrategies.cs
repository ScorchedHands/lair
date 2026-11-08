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
/// Per-head normalization applied to the q and k heads after their projection (and any bias) and before RoPE.
/// Resolved once per model from its <see cref="ModelProfile"/> default and the file's declared
/// <c>&lt;prefix&gt;attention.use_qk_norm</c>; the forward path switches on the value and never dispatches through a
/// per-token interface.
/// </summary>
public enum QkNormMode
{
    /// <summary>No per-head normalization: q and k go from their projection straight to RoPE.</summary>
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
}

/// <summary>
/// Rotary position embedding applied to the q and k heads. Resolved once per model from its
/// <see cref="ModelProfile"/>; scaling variants and partial rotary slices are outside the implemented set.
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
