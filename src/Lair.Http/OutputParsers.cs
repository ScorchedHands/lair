using Lair.Templating.Minja;

namespace Lair.Http;

/// <summary>
/// The tool output parser resolved for one loaded model. The choice is decided once at model load from the loaded
/// chat template's capabilities plus the model's <c>tool_parser</c> configuration, and the request path applies that
/// resolved value to every later chat request — it is never re-decided per request or inside the decode loop.
/// </summary>
public enum ToolParser
{
    /// <summary>The model's own chat-template rendering is the tool surface: declared tools enter the template context, LAIR injects no fenced block, and fenced-JSON detection still reads calls out of the decoded text.</summary>
    Template,

    /// <summary>LAIR's fenced tool block is injected for declared tools regardless of what the loaded template renders itself; fenced-JSON detection reads calls out of the decoded text.</summary>
    Fenced,

    /// <summary>Inert: no tool surface reaches the model and no decoded text is read as a call — content stays byte-for-byte what was generated and carries no <c>tool_calls</c>.</summary>
    None,
}

/// <summary>The wire name of each resolved parser choice, shared by the load log and the advertised model facts.</summary>
public static class ToolParserNames
{
    public const string Template = "template";
    public const string Fenced = "fenced";
    public const string None = "none";

    public static string Of(ToolParser parser) => parser switch
    {
        ToolParser.Template => Template,
        ToolParser.Fenced => Fenced,
        _ => None,
    };
}

/// <summary>
/// A recognized reasoning region of generated output: the half-open span offsets. The recognizer that fills this
/// slot is separate work; every plan resolved today carries no region, so no reader consumes it yet.
/// </summary>
public sealed record ReasoningRegion(int Start, int End);

/// <summary>
/// The resolved output-parser plan for one loaded model: the tool parser choice, the parser instance that choice
/// runs (null when the plan is inert), and the reasoning-region slot a later recognizer fills. Built once at model
/// load from the probed chat-template capabilities plus configuration and stored with the loaded engine, so request
/// behavior consumes a resolved value instead of deciding again; only a new-requests-only reload re-resolves it.
/// </summary>
public sealed record OutputParserPlan(ToolParser ToolParser, IToolOutputParser? Parser, ReasoningRegion? Reasoning)
{
    public OutputParserPlan(ToolParser toolParser, IToolOutputParser? parser)
        : this(toolParser, parser, null)
    {
    }
}

/// <summary>
/// One model's recognized-call reading over fully decoded generated text — the shape the request pipeline needs.
/// It receives the exact generated text and the names one request may call, and reports every accepted call plus
/// the visible text with each accepted fence removed.
/// </summary>
public interface IToolOutputParser
{
    /// <summary>Reads recognized tool calls out of one fully decoded generation.</summary>
    ToolDetection ReadCalls(string text, IReadOnlyCollection<string> allowedNames);
}

/// <summary>LAIR's fenced-JSON parser: the existing call detector behind the parser interface. Accepted fences are stripped from visible text with their arguments compacted; anything else stays ordinary text byte-for-byte.</summary>
public sealed class FencedJsonToolOutputParser : IToolOutputParser
{
    public ToolDetection ReadCalls(string text, IReadOnlyCollection<string> allowedNames) =>
        ToolCallDetector.Detect(text, allowedNames);
}

/// <summary>
/// The parser registry: resolves one model's configured <c>tool_parser</c> key against the chat-template capabilities
/// probed at load into a concrete plan. Selection keys are template capabilities and configuration only — never an
/// architecture name. <c>auto</c> keeps today's precedence exactly: a loaded template that renders tools owns the tool
/// surface, LAIR's fenced block is the fallback for a template without tool support (and for a model with no chat
/// template at all), and the model's <c>force_fenced_tools</c> override selects it even for a tool-capable template.
/// </summary>
public static class OutputParserRegistry
{
    /// <summary>The plan one model runs: resolved from its configuration plus its load-probed capabilities.</summary>
    public static OutputParserPlan Resolve(string toolParser, bool forceFencedTools, TemplateCapabilities? capabilities) => toolParser switch
    {
        ToolParserNames.Template => new OutputParserPlan(ToolParser.Template, new FencedJsonToolOutputParser()),
        ToolParserNames.Fenced => new OutputParserPlan(ToolParser.Fenced, new FencedJsonToolOutputParser()),
        ToolParserNames.None => new OutputParserPlan(ToolParser.None, parser: null),

        // "auto" (the only remaining loader-accepted value): today's capability precedence, expressed as a resolved choice.
        _ => forceFencedTools || capabilities is null || !capabilities.SupportsTools
            ? new OutputParserPlan(ToolParser.Fenced, new FencedJsonToolOutputParser())
            : new OutputParserPlan(ToolParser.Template, new FencedJsonToolOutputParser()),
    };
}
