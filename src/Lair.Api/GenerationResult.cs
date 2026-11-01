namespace Lair.Api;

/// <summary>
/// The outcome of one generation: the decoded text, token accounting, and why decoding stopped —
/// <c>"eos"</c> (end-of-token reached), <c>"length"</c> (token budget exhausted), or <c>"stop"</c>
/// (a stop string appeared in the output and the text was truncated before it).
/// </summary>
public sealed record GenerationResult(
    string Text,
    int PromptTokens,
    int CompletionTokens,
    string FinishReason);
