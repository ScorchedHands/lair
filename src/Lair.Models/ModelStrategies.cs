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
    /// The file's declaration decides: <c>&lt;prefix&gt;attention.use_bias</c> and
    /// <c>&lt;prefix&gt;attention.norm_bias</c> (each absent means <c>false</c>) land in the plan. Declared
    /// <c>true</c> makes the corresponding tensors required at load; declared <c>false</c> ignores present
    /// tensors with one structured log line naming them.
    /// </summary>
    Declared,
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
