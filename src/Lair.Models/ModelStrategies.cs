namespace Lair.Models;

/// <summary>
/// Normalization applied to each layer input stream and to the final stream before the output head.
/// Resolved once per model from its <see cref="ModelProfile"/>; the forward path switches on the value and
/// never dispatches through a per-token interface.
/// </summary>
public enum NormStrategy
{
    /// <summary>RMSNorm scaled by a per-layer weight vector: the only norm implemented today.</summary>
    RmsNorm,
}

/// <summary>
/// Gated feed-forward activation applied between the up and down projections. Resolved once per model from
/// its <see cref="ModelProfile"/>; the forward path switches on the value and never dispatches through a
/// per-token interface.
/// </summary>
public enum ActivationStrategy
{
    /// <summary><c>silu(gate) * up</c>: the only activation implemented today.</summary>
    SwiGLU,
}

/// <summary>
/// How the engine decides whether a projection carries an additive bias. Resolved once per model from its
/// <see cref="ModelProfile"/>.
/// </summary>
public enum BiasPolicy
{
    /// <summary>A bias is applied exactly when its tensor is present in the file, per layer and per projection.</summary>
    ByTensorPresence,
}

/// <summary>
/// Rotary position embedding applied to the q and k heads. Resolved once per model from its
/// <see cref="ModelProfile"/>; scaling variants and partial rotary slices are outside the implemented set.
/// </summary>
public enum RopeStrategy
{
    /// <summary>Adjacent-pair rotation over the full head dimension with no scaling variant: the only RoPE implemented today.</summary>
    StandardFullHead,
}
