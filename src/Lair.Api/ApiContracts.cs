using Lair.Tokenizer.Templating;

namespace Lair.Api;

// Wire-format records for the OpenAI-compatible surface. JSON names are the C# names run through
// the snake_case naming policy configured in the app builder and in the streaming serializer.

/// <summary>Request body of <c>/v1/chat/completions</c>; sampling knobs fall back to the engine defaults.</summary>
internal sealed record ChatCompletionBody(
    string? Model,
    IReadOnlyList<MessageBody>? Messages,
    bool? Stream,
    int? MaxTokens,
    float? Temperature,
    float? TopP,
    int? TopK,
    long? Seed,
    IReadOnlyList<string>? Stop);

/// <summary>Request body of <c>/v1/completions</c>; the raw prompt replaces the messages list.</summary>
internal sealed record CompletionBody(
    string? Model,
    string? Prompt,
    bool? Stream,
    int? MaxTokens,
    float? Temperature,
    float? TopP,
    int? TopK,
    long? Seed,
    IReadOnlyList<string>? Stop);

/// <summary>A conversation turn as it arrives on the wire.</summary>
internal sealed record MessageBody(string? Role, string? Content);

/// <summary>Token accounting reported with every non-streaming completion.</summary>
internal sealed record Usage(int PromptTokens, int CompletionTokens, int TotalTokens);

/// <summary>Non-streaming <c>/v1/chat/completions</c> result.</summary>
internal sealed record ChatCompletionResponse(
    string Id,
    string Object,
    long Created,
    string Model,
    IReadOnlyList<ChatChoice> Choices,
    Usage Usage);

/// <summary>One chat choice: the assistant message plus why generation ended.</summary>
internal sealed record ChatChoice(int Index, ChatMessage Message, string FinishReason);

/// <summary>Non-streaming <c>/v1/completions</c> result.</summary>
internal sealed record TextCompletionResponse(
    string Id,
    string Object,
    long Created,
    string Model,
    IReadOnlyList<TextChoice> Choices,
    Usage Usage);

/// <summary>One completion choice: the continuation text plus why generation ended.</summary>
internal sealed record TextChoice(int Index, string Text, string FinishReason);

/// <summary>One SSE frame of a streamed completion; the choice carries the incremental delta.</summary>
internal sealed record ChunkPayload(
    string Id,
    string Object,
    long Created,
    string Model,
    IReadOnlyList<ChunkChoice> Choices);

/// <summary>A streamed choice: the delta plus <c>finish_reason</c> on the final content chunk.</summary>
internal sealed record ChunkChoice(int Index, ChunkDelta Delta, string? FinishReason);

/// <summary>The incremental piece of a streamed message: a role, a content fragment, or both.</summary>
internal sealed record ChunkDelta(string? Role = null, string? Content = null);

/// <summary><c>/v1/models</c> envelope.</summary>
internal sealed record ModelsResponse(string Object, IReadOnlyList<ModelEntry> Data);

/// <summary>One advertised model: OpenAI baseline fields plus the context/quant/architecture metadata.</summary>
internal sealed record ModelEntry(
    string Id,
    string Object,
    long Created,
    string OwnedBy,
    int ContextLength,
    string Quant,
    string Architecture);

/// <summary><c>/healthz</c> body.</summary>
internal sealed record HealthResponse(string Status, bool ModelReady);

/// <summary>OpenAI error envelope: <c>{error:{message,type,code}}</c>.</summary>
internal sealed record ErrorResponse(ErrorDetail Error);

/// <summary>The error facts inside the envelope.</summary>
internal sealed record ErrorDetail(string Message, string Type, string Code);
