using System.Text.Json.Serialization;

namespace Lair.Http;

// Wire-format records for the OpenAI-compatible surface. JSON names are the C# names run through
// the snake_case naming policy configured in the app builder and in the streaming serializer.

/// <summary>Request body of <c>/v1/chat/completions</c>; sampling knobs fall back to the engine defaults. The tool fields are read straight from the request document so their raw shape can be validated; they are a chat-only surface.</summary>
internal sealed record ChatCompletionBody(
    string? Model,
    IReadOnlyList<MessageBody>? Messages,
    bool? Stream,
    int? MaxTokens,
    float? Temperature,
    float? TopP,
    int? TopK,
    long? Seed,
    IReadOnlyList<string>? Stop,
    float? PresencePenalty,
    float? FrequencyPenalty,
    IReadOnlyDictionary<string, float>? LogitBias,
    [property: JsonPropertyName("logprobs")] bool? LogProbs,
    int? TopLogprobs);

/// <summary>Request body of <c>/v1/completions</c>; the raw prompt replaces the messages list. Tool declarations are a chat-only surface and this endpoint ignores them.</summary>
internal sealed record CompletionBody(
    string? Model,
    string? Prompt,
    bool? Stream,
    int? MaxTokens,
    float? Temperature,
    float? TopP,
    int? TopK,
    long? Seed,
    IReadOnlyList<string>? Stop,
    float? PresencePenalty,
    float? FrequencyPenalty,
    IReadOnlyDictionary<string, float>? LogitBias,
    [property: JsonPropertyName("logprobs")] bool? LogProbs,
    int? TopLogprobs);

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
internal sealed record ChatChoice(
    int Index,
    AssistantMessage Message,
    string FinishReason,
    [property: JsonPropertyName("logprobs")] LogProbsReport? LogProbs = null);

/// <summary>The non-streaming chat choice's message: the assistant turn with its visible content and any recognized tool calls; the call list is absent when there were none.</summary>
internal sealed record AssistantMessage(
    string Role,
    string Content,
    [property: JsonPropertyName("tool_calls")] IReadOnlyList<ToolCallWire>? ToolCalls);

/// <summary>One wire-format tool call: a generated id, the function kind, and the called function's name with its arguments as a compact JSON string.</summary>
internal sealed record ToolCallWire(
    string Id,
    string Type,
    FunctionCallWire Function);

/// <summary>The function half of a wire tool call.</summary>
internal sealed record FunctionCallWire(string Name, string Arguments);

/// <summary>Non-streaming <c>/v1/completions</c> result.</summary>
internal sealed record TextCompletionResponse(
    string Id,
    string Object,
    long Created,
    string Model,
    IReadOnlyList<TextChoice> Choices,
    Usage Usage);

/// <summary>One completion choice: the continuation text plus why generation ended.</summary>
internal sealed record TextChoice(
    int Index,
    string Text,
    string FinishReason,
    [property: JsonPropertyName("logprobs")] LogProbsReport? LogProbs = null);

/// <summary>Per-token log-probability report attached to a non-streaming choice or streamed delta; absent when not requested.</summary>
internal sealed record LogProbsReport(IReadOnlyList<LogProbEntry> Content);

/// <summary>One token's logprob entry: the token text, its vocabulary id when known, and the distribution's top candidates of the same shape.</summary>
internal sealed record LogProbEntry(int? TokenId, string Token, double Logprob, IReadOnlyList<LogProbEntry>? TopLogprobs);

/// <summary>One SSE frame of a streamed completion; the choice carries the incremental delta.</summary>
internal sealed record ChunkPayload(
    string Id,
    string Object,
    long Created,
    string Model,
    IReadOnlyList<ChunkChoice> Choices);

/// <summary>A streamed choice: the delta plus <c>finish_reason</c> on the final content chunk.</summary>
internal sealed record ChunkChoice(int Index, ChunkDelta Delta, string? FinishReason);

/// <summary>The incremental piece of a streamed message: a role, a content fragment with its logprob entry, recognized tool calls, or combinations.</summary>
internal sealed record ChunkDelta(
    string? Role = null,
    string? Content = null,
    LogProbEntry? Logprobs = null,
    [property: JsonPropertyName("tool_calls")] IReadOnlyList<ToolCallWire>? ToolCalls = null);

/// <summary><c>/v1/models</c> envelope.</summary>
internal sealed record ModelsResponse(string Object, IReadOnlyList<ModelEntry> Data);

/// <summary>One advertised model: OpenAI baseline fields plus the context/quant/architecture metadata and the configured model kind. The type field is additive payload so chat surfaces can exclude embed models on a fact; it defaults to "chat" because the config loader accepts no other value in this build.</summary>
internal sealed record ModelEntry(
    string Id,
    string Object,
    long Created,
    string OwnedBy,
    int ContextLength,
    string Quant,
    string Architecture,
    string Type = "chat");

/// <summary><c>/healthz</c> body.</summary>
internal sealed record HealthResponse(string Status, bool ModelReady);

/// <summary>OpenAI error envelope: <c>{error:{message,type,code}}</c>.</summary>
internal sealed record ErrorResponse(ErrorDetail Error);

/// <summary>The error facts inside the envelope.</summary>
internal sealed record ErrorDetail(string Message, string Type, string Code);
