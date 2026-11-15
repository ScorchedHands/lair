namespace Lair.Models;

/// <summary>
/// A model's own budget chain ran out of positions its opened host seats its working copies at: every copy-holding
/// position the chain names is held to its capacity by claims registered ahead of this one — first-come-first-served
/// never revokes, preempts, or displaces an earlier claim to seat a later one — and the chain names no further
/// position its opened host backs. The refusal names the model, the resource, the tier it exhausted at, and the
/// ledger breakdown of what every consulted position holds, so the actionable answer (raise the ceiling the chain
/// declares, name a position an opened pool can back, or free the tier) reads off the message.
/// </summary>
public sealed class BudgetExhaustedException : Exception
{
    public BudgetExhaustedException(string model, BudgetResource resource, BudgetTier tier, string breakdown)
        : base(BuildMessage(model, resource, tier, breakdown))
    {
        Model = model;
        Resource = resource;
        Tier = tier;
        Breakdown = breakdown;
    }

    /// <summary>The model whose chain exhausted.</summary>
    public string Model { get; }

    /// <summary>The resource whose chain exhausted.</summary>
    public BudgetResource Resource { get; }

    /// <summary>The last copy-holding tier the exhausted claim consulted.</summary>
    public BudgetTier Tier { get; }

    /// <summary>The ledger breakdown: every consulted chain position with what it holds against its capacity posture.</summary>
    public string Breakdown { get; }

    private static string BuildMessage(string model, BudgetResource resource, BudgetTier tier, string breakdown) =>
        $"model '{model}' budgets for resource '{BudgetResourceNames.Of(resource)}' exhausted its copy-holding chain at tier " +
        $"'{BudgetTierNames.Of(tier)}': {breakdown}. Raise the ceiling this chain declares, restart with a chain naming a position an " +
        "opened pool can back, or free the tier before opening this model.";
}
