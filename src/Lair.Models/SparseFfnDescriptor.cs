namespace Lair.Models;

/// <summary>
/// Declared shape of a sparse (expert-routed) feed-forward block. The slot exists in every
/// <see cref="ModelPlan"/> so a profile can declare one; no profile declares it yet, so every plan carries
/// <c>null</c> and the engine keeps the dense gate/up/down path.
/// </summary>
public sealed record SparseFfnDescriptor(
    /// <summary>Number of expert gate/up/down triplets the block declares.</summary>
    int ExpertCount,
    /// <summary>Experts each token routes to, out of <see cref="ExpertCount"/>.</summary>
    int RoutedExpertsPerToken,
    /// <summary>Whether one expert is shared by every token in addition to the routed experts.</summary>
    bool HasSharedExpert);
