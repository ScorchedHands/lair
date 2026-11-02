namespace Lair.Models.Scheduling;

/// <summary>
/// One generated token's log-probability entry captured at its sampling step: the token id, its log
/// probability under the distribution actually sampled from, and the <c>topLogProb</c> highest-probability
/// tokens of that distribution (the chosen token included) sorted descending by log probability.
/// </summary>
/// <param name="TopLogProbs">The distribution's highest-probability entries as (token id, log probability) pairs, sorted descending; may exceed <c>topLogProb</c> by one when the chosen token was not among them.</param>
public sealed record TokenLogprob(
    int TokenId,
    double LogProb,
    IReadOnlyList<(int TokenId, double LogProb)> TopLogProbs);
