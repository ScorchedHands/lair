namespace Lair.Sampling.Drafting;

/// <summary>
/// A draft head for speculative decoding: it proposes the tokens a drafter believes will follow the sequence's current
/// history, and those proposals are only ever believed after the target model has verified them. The contract carries no
/// weights of its own — a head may wrap model-native multi-token-prediction tensors when the artifact ships them, or any
/// CPU-only stand-in (a scripted plan, an n-gram table, a smaller side model) supplied by the host. A target whose file
/// names no draft-head tensors is not eligible for native drafting: the host must supply a head here or leave drafting off.
/// Proposals are advisory and may be wrong; verification decides every acceptance.
/// </summary>
public interface IDraftHead
{
    /// <summary>Largest number of tokens this head will propose for one decode boundary.</summary>
    int MaxDraftTokens { get; }

    /// <summary>
    /// Proposes the next up to <see cref="MaxDraftTokens"/> tokens after the history in <paramref name="context"/>, most
    /// immediate first. Fewer (possibly zero) proposals are valid — a head that is unsure proposes nothing and the step
    /// proceeds as an ordinary decode step would. Implementations must be deterministic for identical contexts so accept-rate
    /// measurements repeat across runs.
    /// </summary>
    IReadOnlyList<int> Propose(DraftProposalContext context);
}
