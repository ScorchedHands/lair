namespace Lair.Http.Monitoring;

/// <summary>
/// Live residency accounting an opened engine answers from its attached expert residency tier: what the hot set holds right now and how it has
/// answered slab reads since open. Only a serve that opened a model over a residency-backed weight source answers this at all — every other host
/// reports null, so snapshots keep carrying these slots as nothing rather than zeros an operator could mistake for an idle tier. The saved-bytes
/// figure is the estimate of cold-read bytes resident answers kept out of the gather path, summed per resolution.
/// </summary>
public sealed record ResidencyTierAccounting(
    int HotUnits,
    long BudgetBytes,
    long ResidentBytes,
    long HitsTotal,
    long MissesTotal,
    long SavedBytesEstimateTotal);
