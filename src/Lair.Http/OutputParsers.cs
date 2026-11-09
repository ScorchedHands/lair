using System.Text;
using System.Text.Json;
using Lair.Templating.Minja;

namespace Lair.Http;

/// <summary>
/// The tool output parser resolved for one loaded model. The choice is decided once at model load from the loaded
/// chat template's capabilities plus the model's <c>tool_parser</c> configuration, and the request path applies that
/// resolved value to every later chat request — it is never re-decided per request or inside the decode loop.
/// </summary>
public enum ToolParser
{
    /// <summary>The model's own chat-template rendering is the tool surface: declared tools enter the template context, LAIR injects no fenced block, and the template's own call shape is read out of the decoded text.</summary>
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
/// Reads the call shape a tool-capable chat template itself instructs the model to emit: one JSON object holding exactly
/// <c>name</c> (a declared tool) and <c>parameters</c> — the same object such a template renders back into an assistant
/// tool-call turn, so a recognized call round-trips through the template's own rendering on the next request. A candidate
/// is any brace-balanced object that parses to exactly that two-property shape with a string name the choice allows; the
/// fenced contract's <c>arguments</c> property is not this shape, so a fenced block stays ordinary text under this parser.
/// Accepted objects are removed from the visible text with their arguments compacted; anything else — unknown names,
/// malformed or unterminated objects, extra properties, wrong property names — stays ordinary text byte-for-byte.
/// </summary>
public sealed class TemplateNativeToolOutputParser : IToolOutputParser
{
    public ToolDetection ReadCalls(string text, IReadOnlyCollection<string> allowedNames)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(allowedNames);

        List<ToolCall> calls = [];
        StringBuilder visible = new(text.Length);
        int cursor = 0;

        while (cursor < text.Length)
        {
            int openIndex = text.IndexOf('{', cursor);
            if (openIndex < 0)
            {
                break;
            }

            int closeIndex = FindObjectEnd(text, openIndex);
            if (closeIndex < 0)
            {
                break; // an unterminated object can only be ordinary output
            }

            ToolCall? call = ParseCall(text[openIndex..(closeIndex + 1)], allowedNames);
            if (call is null)
            {
                AppendVisibleSegment(visible, text[cursor..(openIndex + 1)]);
                cursor = openIndex + 1; // not a call candidate: copy the opening brace verbatim and keep scanning after it
                continue;
            }

            AppendVisibleSegment(visible, text[cursor..openIndex]);
            calls.Add(call);
            cursor = closeIndex + 1; // whatever follows the object stays in the following segment
        }

        AppendVisibleSegment(visible, text[cursor..]);
        return new ToolDetection(calls, visible.ToString());
    }

    // The matching close of the object opened at openIndex, found by brace counting that skips JSON string bodies so braces
    // inside quoted values never end the candidate; the exact pairing is left to the JSON parse, which rejects any slice
    // this bound over-matched.
    private static int FindObjectEnd(string text, int openIndex)
    {
        int depth = 0;
        bool inString = false;
        bool escaped = false;

        for (int i = openIndex; i < text.Length; i++)
        {
            char c = text[i];
            if (inString)
            {
                if (escaped)
                {
                    escaped = false;
                }
                else if (c == '\\')
                {
                    escaped = true;
                }
                else if (c == '"')
                {
                    inString = false;
                }

                continue;
            }

            switch (c)
            {
                case '"':
                    inString = true;
                    break;
                case '{' or '[':
                    depth++;
                    break;
                case '}' or ']':
                    depth--;
                    if (depth == 0)
                    {
                        return i;
                    }

                    break;
            }
        }

        return -1;
    }

    // One candidate slice is a call only when it parses to exactly one object with exactly the properties name (string, allowed)
    // and parameters; everything else returns null so the text stays ordinary.
    private static ToolCall? ParseCall(string body, IReadOnlyCollection<string> allowedNames)
    {
        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(body);
        }
        catch (JsonException)
        {
            return null;
        }

        using (document)
        {
            if (document.RootElement.ValueKind != JsonValueKind.Object)
            {
                return null;
            }

            int properties = 0;
            string? name = null;
            bool hasParameters = false;
            foreach (JsonProperty property in document.RootElement.EnumerateObject())
            {
                if (++properties > 2)
                {
                    return null;
                }

                switch (property.Name)
                {
                    case "name":
                        if (property.Value.ValueKind != JsonValueKind.String || name is not null)
                        {
                            return null;
                        }

                        name = property.Value.GetString();
                        break;
                    case "parameters":
                        // Any JSON value is accepted and re-encoded compactly, matching the wire's string form.
                        if (hasParameters)
                        {
                            return null;
                        }

                        hasParameters = true;
                        break;
                    default:
                        return null;
                }
            }

            if (name is null || !hasParameters || properties != 2)
            {
                return null;
            }

            if (!allowedNames.Contains(name))
            {
                return null;
            }

            // Serialize the parameters through a writer so they land compactly on the wire regardless of how the model spaced them; raw-text passthrough would echo source whitespace.
            string? canonical = Compact(document.RootElement.GetProperty("parameters").GetRawText());
            return canonical is null ? null : new ToolCall(name, canonical);
        }
    }

    private static string? Compact(string rawParameters)
    {
        try
        {
            using JsonDocument document = JsonDocument.Parse(rawParameters);

            using MemoryStream buffer = new();
            using (Utf8JsonWriter writer = new(buffer))
            {
                document.RootElement.WriteTo(writer);
            }

            return Encoding.UTF8.GetString(buffer.ToArray());
        }
        catch (JsonException)
        {
            // Unreachable: the enclosing object only parsed because every member's raw text is valid JSON.
            return null;
        }
    }

    // Copies a stripped-out segment, collapsing at most one boundary space so removing a call does not fuse surrounding words.
    private static void AppendVisibleSegment(StringBuilder visible, string segment)
    {
        if (segment.Length > 0 && visible.Length > 0)
        {
            char edge = visible[^1];
            bool leftSpace = edge is ' ' or '\n';

            if (leftSpace && segment[0] == ' ')
            {
                int trailingSpaces = 0;
                while (trailingSpaces < segment.Length && segment[trailingSpaces] == ' ')
                {
                    trailingSpaces++;
                }

                visible.Append(segment, trailingSpaces, segment.Length - trailingSpaces);
                return;
            }
        }

        visible.Append(segment);
    }
}

/// <summary>
/// The parser registry: resolves one model's configured <c>tool_parser</c> key against the chat-template capabilities
/// probed at load into a concrete plan. Selection keys are template capabilities and configuration only — never an
/// architecture name. <c>auto</c> keeps today's precedence exactly: a loaded template that renders tools owns the tool
/// surface (and its own call shape is what the decoded text is read for), LAIR's fenced block plus fenced-JSON reading is
/// the fallback for a template without tool support (and for a model with no chat template at all), and the model's
/// <c>force_fenced_tools</c> override selects it even for a tool-capable template.
/// </summary>
public static class OutputParserRegistry
{
    /// <summary>The plan one model runs: resolved from its configuration plus its load-probed capabilities.</summary>
    public static OutputParserPlan Resolve(string toolParser, bool forceFencedTools, TemplateCapabilities? capabilities) => toolParser switch
    {
        ToolParserNames.Template => new OutputParserPlan(ToolParser.Template, new TemplateNativeToolOutputParser()),
        ToolParserNames.Fenced => new OutputParserPlan(ToolParser.Fenced, new FencedJsonToolOutputParser()),
        ToolParserNames.None => new OutputParserPlan(ToolParser.None, parser: null),

        // "auto" (the only remaining loader-accepted value): today's capability precedence, expressed as a resolved choice.
        _ => forceFencedTools || capabilities is null || !capabilities.SupportsTools
            ? new OutputParserPlan(ToolParser.Fenced, new FencedJsonToolOutputParser())
            : new OutputParserPlan(ToolParser.Template, new TemplateNativeToolOutputParser()),
    };
}
