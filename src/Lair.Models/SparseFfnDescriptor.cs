namespace Lair.Models;

/// <summary>
/// Declared shape of a sparse (expert-routed) feed-forward block: the expert count and routing count the
/// file declares under <c>&lt;prefix&gt;expert_count</c> and <c>&lt;prefix&gt;expert_used_count</c>, the
/// shared expert the block declares through <c>&lt;prefix&gt;expert_shared_count</c>, the tensor roles the
/// resolved profile's map names for the router and the expert matrices, the routing-weight operands
/// the reference reads from <c>&lt;prefix&gt;expert_weights_scale</c> and <c>&lt;prefix&gt;expert_weights_norm</c>,
/// and how the block's experts are stored: the declared layout (<c>&lt;prefix&gt;expert_layout</c>), its factorization
/// rank under <c>&lt;prefix&gt;expert_rank</c>, and — for a factorized declaration only — the tensor roles naming the
/// shared input basis, the per-expert coefficients, and the shared output projection. The slot exists in every
/// <see cref="ModelPlan"/> so a profile can declare one; a file that declares no expert count keeps <c>null</c> and
/// the engine keeps the dense gate/up/down path. A descriptor whose layout is stacked carries no rank and names none
/// of the factorized roles, which is what keeps an un-factorizable file out of any factorized selection.
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
    /// <summary>
    /// How this block's experts are stored: dense-stacked per-role matrices, or a factorized shape sharing one input
    /// basis across the block's experts. Absent the declared <c>&lt;prefix&gt;expert_layout</c> it is
    /// <see cref="SparseFfnLayout.StackedExperts"/>, which is what every published conversion ships.
    /// </summary>
    SparseFfnLayout Layout,
    /// <summary>
    /// Declared <c>&lt;prefix&gt;expert_rank</c>: the rank the factorized storage compresses each expert to, present
    /// exactly when <see cref="Layout"/> is <see cref="SparseFfnLayout.FactorizedSharedBasis"/> and <c>null</c> for a
    /// stacked block. A later kernel reads it as the coefficient-space width; nothing in this build computes with it.
    /// </summary>
    int? Rank,
    /// <summary>
    /// Tensor role of the factorized layout's shared input basis — the matrix applied once per layer to every expert's
    /// input — or <c>null</c> on a stacked block. The resolved profile names it and the file decides whether it ships.
    /// </summary>
    TensorRole? ExpertInputBasisRole,
    /// <summary>
    /// Tensor role of the factorized layout's per-expert coefficients, stacked with the expert count as its outermost
    /// dimension exactly like <see cref="ExpertGateRole"/> — or <c>null</c> on a stacked block.
    /// </summary>
    TensorRole? ExpertCoefficientRole,
    /// <summary>
    /// Tensor role of the factorized layout's shared output projection that expands the accumulated coefficient-space
    /// sum back to the hidden stream, or <c>null</c> on a stacked block.
    /// </summary>
    TensorRole? ExpertOutputBasisRole,
    /// <summary>Declared <c>&lt;prefix&gt;expert_weights_scale</c>: the gain the reference multiplies the routing weights by, <c>0f</c> (disabled) when the key is absent.</summary>
    float ExpertWeightsScale,
    /// <summary>Declared <c>&lt;prefix&gt;expert_weights_norm</c>: whether the reference normalizes the routing weights before the routed expert outputs are summed, <c>false</c> when the key is absent.</summary>
    bool NormalizeExpertWeights);
