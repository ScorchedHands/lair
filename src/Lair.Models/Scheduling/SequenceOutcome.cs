namespace Lair.Models.Scheduling;

/// <summary>
/// The result a scheduled sequence hands back when it finishes: every token it generated (eos excluded), the prompt
/// size, and why decoding stopped. A sequence cancelled or failed mid-flight reports the tokens it had already
/// generated; its KV blocks are already released by the time the outcome is published.
/// </summary>
/// <param name="Error">The exception that ended a <see cref="FinishReason.Failed"/> sequence; null otherwise.</param>
/// <param name="PrefillCachedTokens">Prompt tokens whose KV blocks were adopted from the prefix cache instead of recomputed; 0 when no prefix was shared.</param>
/// <param name="LogProbs">One entry per generated token capturing its sampled-step log probability and top candidates, in generation order; null or empty when the request did not ask for them.</param>
public sealed record SequenceOutcome(
    IReadOnlyList<int> GeneratedTokens,
    int PromptTokens,
    int CompletionTokens,
    FinishReason FinishReason,
    Exception? Error = null,
    int PrefillCachedTokens = 0,
    IReadOnlyList<TokenLogprob>? LogProbs = null);
