namespace Lair.Templating.Minja;

/// <summary>
/// One chat message in a <see cref="ChatTemplateInput"/>. <see cref="Content"/> is a <see cref="MinjaValue"/> so a
/// caller can supply either a plain string or a typed-content list (a list of <c>{"type": "text", "text": ...}</c>
/// maps) exactly as an HF-style template expects; the wrapper normalizes between the two shapes according to the
/// template's detected capabilities. <see cref="ToolCalls"/> carries assistant tool-call requests, and
/// <see cref="Name"/>/<see cref="ToolCallId"/> carry the tool name and call id used by tool-response messages.
/// </summary>
public sealed record ChatMessage
{
    public required string Role { get; init; }

    /// <summary>Message content: a string or a typed-content list; none renders as empty text.</summary>
    public MinjaValue Content { get; init; } = MinjaValue.None;

    /// <summary>Assistant tool-call requests attached to this message, or null when the message makes none.</summary>
    public IReadOnlyList<ChatToolCall>? ToolCalls { get; init; }

    /// <summary>Optional message name, used by some templates for tool-response/tool routing.</summary>
    public string? Name { get; init; }

    /// <summary>The tool-call id a <c>role: tool</c> response answers, when the caller supplies one.</summary>
    public string? ToolCallId { get; init; }

    /// <summary>Convenience factory for a plain-text message.</summary>
    public static ChatMessage Text(string role, string content) => new() { Role = role, Content = MinjaValue.Of(content) };
}

/// <summary>
/// One assistant tool-call request. <see cref="Arguments"/> is a <see cref="MinjaValue"/> so a caller can supply
/// either a map of arguments or a JSON string, matching the shape a given template expects; the wrapper converts
/// between the two according to the template's object-argument capability. <see cref="Id"/> is the call id some
/// templates echo back for tool-response correlation.
/// </summary>
public sealed record ChatToolCall
{
    public required string Name { get; init; }

    /// <summary>Tool arguments as a map or a JSON string; none means no arguments.</summary>
    public MinjaValue Arguments { get; init; } = MinjaValue.EmptyMap;

    public string? Id { get; init; }
}

/// <summary>
/// A single render request for <see cref="MinjaChatTemplate.Render"/>. It carries the HF-style chat inputs a chat
/// template consumes: the message list, optional tool definitions, the <c>add_generation_prompt</c> flag, extra
/// context keys injected verbatim, an optional fixed <c>Now</c> for <c>strftime_now</c> determinism, and an optional
/// <c>date_string</c>. Tool definitions are supplied as raw <see cref="MinjaValue"/> maps because published chat
/// templates disagree on the tool-definition shape (OpenAI-style nested <c>function</c> objects versus flat
/// <c>name</c>/<c>parameters</c> maps); the wrapper passes them through unchanged and only probes them for
/// capability detection.
/// </summary>
public sealed record ChatTemplateInput
{
    public required IReadOnlyList<ChatMessage> Messages { get; init; }

    /// <summary>Tool definitions to expose to the template, or null to omit the <c>tools</c> variable entirely.</summary>
    public IReadOnlyList<MinjaValue>? Tools { get; init; }

    /// <summary>The <c>add_generation_prompt</c> context flag; defaults to true like the HF renderer.</summary>
    public bool AddGenerationPrompt { get; init; } = true;

    /// <summary>Extra context keys (e.g. <c>custom_tools</c>, <c>tools_in_user_message</c>) bound directly into the template scope.</summary>
    public IReadOnlyDictionary<string, MinjaValue>? ExtraContext { get; init; }

    /// <summary>Fixed clock value for <c>strftime_now</c>. When null, a single value is captured at render time so all calls in one render agree.</summary>
    public DateTimeOffset? Now { get; init; }

    /// <summary>Optional <c>date_string</c> context value. When null the wrapper leaves <c>date_string</c> undefined so a template's own default branch still runs.</summary>
    public string? DateString { get; init; }
}
