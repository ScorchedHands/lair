namespace Lair.Sampling.Drafting;

/// <summary>
/// A deterministic <see cref="IDraftHead"/> whose proposals come from a caller-supplied selector over the proposal context —
/// no model weights, no tensors. It is the seam's stand-in for families that ship no draft-head artifacts: tests drive it
/// from recorded continuations to exercise accept and reject paths with fixed outcomes, and benches replay a pinned greedy
/// stream as proposals so accept-rate tallies measure against a known-correct plan rather than an unmeasured guess. The
/// selector decides what is proposed; this head validates shape only (count within the cap, ids inside the vocabulary) and
/// never mutates sequence state.
/// </summary>
public sealed class ScriptedDraftHead : IDraftHead
{
    private readonly Func<DraftProposalContext, IReadOnlyList<int>> _selector;

    public ScriptedDraftHead(int maxDraftTokens, Func<DraftProposalContext, IReadOnlyList<int>> selector)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxDraftTokens);
        ArgumentNullException.ThrowIfNull(selector);
        MaxDraftTokens = maxDraftTokens;
        _selector = selector;
    }

    /// <inheritdoc/>
    public int MaxDraftTokens { get; }

    /// <summary>Runs the selector and validates its shape: at most <see cref="MaxDraftTokens"/> ids, each inside the target vocabulary.</summary>
    /// <exception cref="InvalidOperationException">The selector returned null or an out-of-vocabulary token id.</exception>
    public IReadOnlyList<int> Propose(DraftProposalContext context)
    {
        IReadOnlyList<int> drafts = _selector(context) ?? throw new InvalidOperationException("draft head selector returned no proposal list.");
        if (drafts.Count > MaxDraftTokens)
        {
            throw new InvalidOperationException($"draft head proposed {drafts.Count} tokens but its cap is {MaxDraftTokens}.");
        }

        for (int i = 0; i < drafts.Count; i++)
        {
            int id = drafts[i];
            if (id < 0 || id >= context.VocabSize)
            {
                throw new InvalidOperationException($"draft head proposed token {id} at position {i}, outside the target vocabulary of {context.VocabSize}.");
            }
        }

        return drafts;
    }
}
