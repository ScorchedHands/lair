namespace Lair.Api;

/// <summary>
/// The outcome of one generation: the decoded text, token accounting, and why decoding stopped —
/// <c>"eos"</c> (end-of-token reached), <c>"length"</c> (token budget exhausted), or <c>"stop"</c>
/// (a stop string appeared in the output and the text was truncated before it).
/// </summary>
/// <param name="CachedPromptTokens">Prompt tokens whose KV blocks were adopted from the engine's prefix cache instead of recomputed; 0 when nothing was shared.</param>
public sealed record GenerationResult(
    string Text,
    int PromptTokens,
    int CompletionTokens,
    string FinishReason,
    int CachedPromptTokens = 0);
