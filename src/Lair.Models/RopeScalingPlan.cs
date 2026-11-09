namespace Lair.Models;

/// <summary>
/// Declared rope-scaling shape resolved from the file's <c>&lt;prefix&gt;rope.scaling.*</c> metadata: the variant
/// name the file declares, the context multiplier, the context length the variant is calibrated against, the
/// attention-score factor the reference folds into the rope magnitude, the YaRN correction scale, and the beta
/// pair that bounds the correction band. The slot exists in every <see cref="ModelPlan"/>; a file that declares
/// no scaling variant keeps <c>null</c> and the engine keeps the standard full-head rotation.
/// </summary>
public sealed record RopeScalingPlan(
    /// <summary>Declared <c>&lt;prefix&gt;rope.scaling.type</c>: the scaling variant the engine applies.</summary>
    string Variant,
    /// <summary>Declared <c>&lt;prefix&gt;rope.scaling.factor</c>: the context multiplier, <c>1f</c> when the key is absent.</summary>
    float Multiplier,
    /// <summary>Declared <c>&lt;prefix&gt;rope.scaling.original_context_length</c>: the context length the variant interpolates against.</summary>
    int OriginalContextLength,
    /// <summary>
    /// The resolved attention factor: the score-side magnitude the reference pairs with the variant, resolved
    /// from the multiplier and the declared <c>&lt;prefix&gt;rope.scaling.attn_factor</c> (and, when the variant
    /// declares one, the declared <c>&lt;prefix&gt;rope.scaling.yarn_log_multiplier</c>).
    /// </summary>
    float AttentionFactor,
    /// <summary>Declared <c>&lt;prefix&gt;rope.scaling.yarn_ext_factor</c>: the correction scale blended into the angles, <c>1f</c> when the key is absent.</summary>
    float CorrectionScale,
    /// <summary>Declared <c>&lt;prefix&gt;rope.scaling.yarn_beta_fast</c>: the low end of the correction band, <c>32f</c> when the key is absent.</summary>
    float BetaFast,
    /// <summary>Declared <c>&lt;prefix&gt;rope.scaling.yarn_beta_slow</c>: the high end of the correction band, <c>1f</c> when the key is absent.</summary>
    float BetaSlow);
