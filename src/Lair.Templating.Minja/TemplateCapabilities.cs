namespace Lair.Templating.Minja;

/// <summary>
/// Capability facts detected for a chat template by <see cref="MinjaChatTemplate"/>. These are the facts the later
/// LAIR adapter uses to decide whether to inject its own fenced-tool fallback and how to normalize message shapes:
/// a template that reports <see cref="SupportsTools"/> renders tools itself, while one that does not needs LAIR's
/// injection or the wrapper's tool-response/system-role fallbacks. Each fact is derived from dummy-render probes,
/// so it reflects what the template actually renders rather than a guess from its source text.
/// </summary>
public sealed record TemplateCapabilities
{
    /// <summary>The template renders supplied tool definitions (a dummy tool name appears in its output).</summary>
    public bool SupportsTools { get; init; }

    /// <summary>The template renders assistant tool-call requests (a dummy call name/argument appears).</summary>
    public bool SupportsToolCalls { get; init; }

    /// <summary>The template renders <c>role: tool</c> responses natively (dummy response content appears).</summary>
    public bool SupportsToolResponses { get; init; }

    /// <summary>The template renders a <c>role: system</c> message's content (dummy system content appears).</summary>
    public bool SupportsSystemRole { get; init; }

    /// <summary>The template renders more than one tool call per assistant message (both dummy names appear).</summary>
    public bool SupportsParallelToolCalls { get; init; }

    /// <summary>The template echoes a tool-call id (a dummy id appears in its output).</summary>
    public bool SupportsToolCallId { get; init; }

    /// <summary>
    /// The template requires tool-call arguments to be maps/objects rather than JSON strings: a dummy object
    /// argument renders but a stringified one does not.
    /// </summary>
    public bool RequiresObjectArguments { get; init; }

    /// <summary>
    /// The template requires typed-content lists rather than plain strings: a dummy typed-content list renders but a
    /// plain string of the same text does not.
    /// </summary>
    public bool RequiresTypedContent { get; init; }

    /// <summary>
    /// The template fails (or renders nothing) when a message's content is null/none, so callers must always supply
    /// non-null content.
    /// </summary>
    public bool RequiresNonNullContent { get; init; }
}

/// <summary>Per-render behavior toggles for <see cref="MinjaChatTemplate.Render"/>.</summary>
public sealed record ChatTemplateOptions
{
    /// <summary>
    /// When true (default), the wrapper rewrites message/tool shapes to fit the template's detected capabilities
    /// (typed-content wrapping/flattening, argument object/string conversion, tool-response and system-role
    /// fallbacks). When false the input is rendered exactly as supplied, leaving normalization to the caller.
    /// </summary>
    public bool NormalizeMessages { get; init; } = true;

    /// <summary>
    /// When true (default), capabilities are derived by dummy-render probes. When false the wrapper skips probing
    /// and reports no capabilities, which is cheaper but leaves every fallback disabled.
    /// </summary>
    public bool DetectCapabilities { get; init; } = true;
}

/// <summary>
/// The result of a chat-template render: the rendered prompt text, the template's capability facts, and whether the
/// rendered prompt already begins with the model BOS token. <see cref="StartsWithBos"/> lets the later LAIR adapter
/// suppress the tokenizer's own BOS so the prompt is not double-prefixed; the wrapper itself never edits the
/// tokenizer.
/// </summary>
public sealed record ChatTemplateRenderResult
{
    public required string Prompt { get; init; }

    public required TemplateCapabilities Capabilities { get; init; }

    /// <summary>True when the rendered prompt already starts with the model BOS token supplied to the template.</summary>
    public bool StartsWithBos { get; init; }
}
