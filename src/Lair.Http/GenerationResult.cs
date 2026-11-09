using Lair.Models.Scheduling;

namespace Lair.Http;

/// <summary>
/// The outcome of one generation: the decoded text, token accounting, and why decoding stopped —
/// <c>"eos"</c> (end-of-token reached), <c>"length"</c> (token budget exhausted), or <c>"stop"</c>
/// (a stop string appeared in the output and the text was truncated before it).
/// </summary>
/// <param name="CachedPromptTokens">Prompt tokens whose KV blocks were adopted from the engine's prefix cache instead of recomputed; 0 when nothing was shared.</param>
/// <param name="LogProbs">One entry per generated token with its sampled-step log probability and top candidates, in generation order; null when the request did not ask for logprobs.</param>
/// <param name="ToolCalls">Fenced-JSON calls recognized in the decoded text, in document order, each naming a declared tool with compact arguments JSON; non-null only when at least one call was accepted and <see cref="FinishReason"/> is then "tool_calls" and <see cref="Text"/> has the fences removed.</param>
/// <param name="ReasoningRegions">Half-open spans of the reasoning regions recognized from the markup this generation contains, in document order; non-null only when at least one region was recognized. Under a strip policy these spans are the parts <see cref="Text"/> does not carry.</param>
/// <param name="Reasoning">The visibility policy this sequence ran on — the one resolved when it was admitted, which a later reload does not change. Streamed content follows this value rather than whatever policy is current when generation ends.</param>
public sealed record GenerationResult(
    string Text,
    int PromptTokens,
    int CompletionTokens,
    string FinishReason,
    int CachedPromptTokens = 0,
    IReadOnlyList<TokenLogprob>? LogProbs = null,
    IReadOnlyList<ToolCall>? ToolCalls = null,
    IReadOnlyList<ReasoningRegion>? ReasoningRegions = null,
    ReasoningPolicy Reasoning = ReasoningPolicy.Keep);
