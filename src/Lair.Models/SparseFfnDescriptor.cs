namespace Lair.Models;

/// <summary>
/// Declared shape of a sparse (expert-routed) feed-forward block: the expert count and routing count the
/// file declares under <c>&lt;prefix&gt;expert_count</c> and <c>&lt;prefix&gt;expert_used_count</c>, the
/// shared expert the block declares through <c>&lt;prefix&gt;expert_shared_count</c>, the tensor roles the
/// resolved profile's map names for the router and the expert matrices, and the routing-weight operands
/// the reference reads from <c>&lt;prefix&gt;expert_weights_scale</c> and <c>&lt;prefix&gt;expert_weights_norm</c>.
/// The slot exists in every <see cref="ModelPlan"/> so a profile can declare one; a file that declares no
/// expert count keeps <c>null</c> and the engine keeps the dense gate/up/down path.
/// </summary>
public sealed record SparseFfnDescriptor(
    /// <summary>Number of expert gate/up/down triplets the block declares.</summary>
    int ExpertCount,
    /// <summary>Experts each token routes to, out of <see cref="ExpertCount"/>.</summary>
    int RoutedExpertsPerToken,
    /// <summary>Whether one expert is shared by every token in addition to the routed experts.</summary>
    bool HasSharedExpert,
    /// <summary>Tensor role of the per-token routing matrix over the experts.</summary>
    TensorRole RouterRole,
    /// <summary>Tensor role of the stacked per-expert gate matrices.</summary>
    TensorRole ExpertGateRole,
    /// <summary>Tensor role of the stacked per-expert up matrices.</summary>
    TensorRole ExpertUpRole,
    /// <summary>Tensor role of the stacked per-expert down matrices.</summary>
    TensorRole ExpertDownRole,
    /// <summary>Tensor role of the shared expert's gate matrix, or <c>null</c> when the block declares no shared expert.</summary>
    TensorRole? SharedGateRole,
    /// <summary>Tensor role of the shared expert's up matrix, or <c>null</c> when the block declares no shared expert.</summary>
    TensorRole? SharedUpRole,
    /// <summary>Tensor role of the shared expert's down matrix, or <c>null</c> when the block declares no shared expert.</summary>
    TensorRole? SharedDownRole,
    /// <summary>
    /// Tensor role of the shared expert's per-token scalar gate vector — the row whose score selects the shared
    /// expert for a token. Declared only when the resolved profile names it and the block declares a shared expert;
    /// whether the file actually ships that tensor decides at load, where an absent vector leaves the shared
    /// contribution ungated.
    /// </summary>
    TensorRole? SharedInputGateRole,
    /// <summary>Declared <c>&lt;prefix&gt;expert_weights_scale</c>: the gain the reference multiplies the routing weights by, <c>0f</c> (disabled) when the key is absent.</summary>
    float ExpertWeightsScale,
    /// <summary>Declared <c>&lt;prefix&gt;expert_weights_norm</c>: whether the reference normalizes the routing weights before the routed expert outputs are summed, <c>false</c> when the key is absent.</summary>
    bool NormalizeExpertWeights);
