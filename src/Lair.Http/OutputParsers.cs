using System.Text;
using System.Text.Json;
using Lair.Templating.Minja;
using Lair.Tokenizer;

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

/// <summary>How one model's recognized reasoning region reaches the wire.</summary>
public enum ReasoningPolicy
{
    /// <summary>The region streams through as ordinary visible content, byte-for-byte as generated — the behavior every model has today.</summary>
    Keep,

    /// <summary>The region is withheld from visible content while its boundaries stay recorded for the request.</summary>
    Strip,
}

/// <summary>The wire name of each reasoning visibility choice, shared by the load log and the advertised model facts.</summary>
public static class ReasoningPolicyNames
{
    public const string Keep = "keep";
    public const string Strip = "strip";

    public static string Of(ReasoningPolicy policy) => policy switch
    {
        ReasoningPolicy.Strip => Strip,
        _ => Keep,
    };
}

/// <summary>Half-open span offsets of one recognized reasoning region over the generated text.</summary>
public sealed record ReasoningRegion(int Start, int End);

/// <summary>The open/close markup pair a model's plan resolves its reasoning region to — the same markers the recognizer reads over generated text, plus the single vocabulary token each marker arrives as in this model's tokenizer. A ceiling bounds only a region this resolved, so request-time budget rules consult it instead of guessing a shape per request.</summary>
public sealed record ReasoningRegionMarkers(string OpenMarkup, string CloseMarkup)
{
    /// <summary>The id one sampled token must carry for the open marker to arrive whole.</summary>
    public required int OpenTokenId { get; init; }

    /// <summary>The id one sampled token must carry for the close marker to arrive whole.</summary>
    public required int CloseTokenId { get; init; }
}

/// <summary>The reasoning regions recognized in one generation plus the visible text with them removed.</summary>
public sealed record ReasoningDetection(IReadOnlyList<ReasoningRegion> Regions, string VisibleText);

/// <summary>
/// Recognizes reasoning regions from the markup a model emits for itself: an region opens at <c>&lt;think&gt;</c> and
/// closes at the first <c>&lt;/think&gt;</c> after it. Recognition keys on that markup alone, never on an architecture
/// or model name, so a generation without it recognizes nothing. A close with no open is ordinary text; an open that
/// never closes runs to the end of the generated text and stays recognized as a region — reasoning cut off by a token
/// budget is still reasoning. Regions do not nest: markup inside a region is its content, so only one boundary pair at
/// a time is live.
/// </summary>
public static class ReasoningRecognition
{
    public const string OpenMarkup = "<think>";
    public const string CloseMarkup = "</think>";

    /// <summary>The half-open spans of every recognized region over one fully decoded generation, in document order.</summary>
    public static IReadOnlyList<ReasoningRegion> FindRegions(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        List<ReasoningRegion>? regions = null;
        int cursor = 0;

        while (cursor < text.Length)
        {
            int open = text.IndexOf(OpenMarkup, cursor, StringComparison.Ordinal);
            if (open < 0)
            {
                break;
            }

            int close = text.IndexOf(CloseMarkup, open + OpenMarkup.Length, StringComparison.Ordinal);
            regions ??= [];

            if (close < 0)
            {
                regions.Add(new ReasoningRegion(open, text.Length));
                break;
            }

            regions.Add(new ReasoningRegion(open, close + CloseMarkup.Length));
            cursor = close + CloseMarkup.Length;
        }

        return regions ?? [];
    }

    /// <summary>Recognized regions over one generation and the visible text with each region's span removed verbatim.</summary>
    public static ReasoningDetection Strip(string text)
    {
        IReadOnlyList<ReasoningRegion> regions = FindRegions(text);
        if (regions.Count == 0)
        {
            return new ReasoningDetection(regions, text);
        }

        StringBuilder visible = new(text.Length);
        int cursor = 0;
        foreach (ReasoningRegion region in regions)
        {
            visible.Append(text, cursor, region.Start - cursor);
            cursor = region.End;
        }

        visible.Append(text, cursor, text.Length - cursor);
        return new ReasoningDetection(regions, visible.ToString());
    }
}

/// <summary>
/// The same recognition applied to a generation as it streams, piece by piece. A boundary marker split across pieces is
/// held back rather than guessed at: the tail of an unconsumed buffer that could still begin or complete a marker stays
/// pending until the following piece settles it, so a region opened mid-piece and closed two pieces later is recognized
/// exactly once and its content never reaches the wire. Visible spans are handed out in stream order and each boundary
/// pair records one <see cref="ReasoningRegion"/> with offsets over the whole generation.
/// </summary>
public sealed class ReasoningRegionFilter
{
    private const string Open = ReasoningRecognition.OpenMarkup;
    private const string Close = ReasoningRecognition.CloseMarkup;

    private readonly List<ReasoningRegion> _regions = [];
    private string _buffer = "";
    private int _bufferStart;
    private int _regionOpen;
    private bool _inRegion;

    /// <summary>The regions recognized so far, in document order.</summary>
    public IReadOnlyList<ReasoningRegion> Regions => _regions;

    /// <summary>Fed one streamed piece of generated text: hands back the visible text it settles, which may be empty.</summary>
    public IReadOnlyList<string> Feed(string piece)
    {
        ArgumentNullException.ThrowIfNull(piece);

        if (_buffer.Length == 0)
        {
            _buffer = piece;
        }
        else
        {
            _buffer += piece;
        }

        return Drain(final: false);
    }

    /// <summary>Closes the stream: hands back whatever visible text is still pending and records an open region that never closed.</summary>
    public IReadOnlyList<string> Finish() => Drain(final: true);

    private List<string> Drain(bool final)
    {
        List<string>? visible = null;

        while (true)
        {
            if (_inRegion)
            {
                int close = _buffer.IndexOf(Close, StringComparison.Ordinal);
                if (close >= 0)
                {
                    _regions.Add(new ReasoningRegion(_regionOpen, _bufferStart + close + Close.Length));
                    _bufferStart += close + Close.Length;
                    _buffer = _buffer[(close + Close.Length)..];
                    _inRegion = false;
                    continue;
                }

                if (final)
                {
                    // An open region that the stream never closed runs to the end of generation.
                    _regions.Add(new ReasoningRegion(_regionOpen, _bufferStart + _buffer.Length));
                    _buffer = "";
                    break;
                }

                int heldClose = Math.Max(0, _buffer.Length - (Close.Length - 1));
                if (heldClose > 0)
                {
                    _bufferStart += heldClose;
                    _buffer = _buffer[heldClose..]; // region content is withheld as it settles
                }

                break;
            }

            int open = _buffer.IndexOf(Open, StringComparison.Ordinal);
            if (open >= 0)
            {
                visible ??= [];
                if (open > 0)
                {
                    visible.Add(_buffer[..open]);
                }

                _regionOpen = _bufferStart + open;
                _bufferStart += open + Open.Length;
                _buffer = _buffer[(open + Open.Length)..];
                _inRegion = true;
                continue;
            }

            if (final)
            {
                visible ??= [];
                if (_buffer.Length > 0)
                {
                    visible.Add(_buffer); // a pending tail that settled as ordinary text still belongs on the wire
                    _bufferStart += _buffer.Length;
                    _buffer = "";
                }

                break;
            }

            int heldOpen = Math.Max(0, _buffer.Length - (Open.Length - 1));
            if (heldOpen > 0)
            {
                visible ??= [];
                visible.Add(_buffer[..heldOpen]);
                _bufferStart += heldOpen;
                _buffer = _buffer[heldOpen..]; // the tail could still open a region, so it stays pending
            }

            break;
        }

        return visible ?? [];
    }
}

/// <summary>
/// The resolved output-parser plan for one loaded model: the tool parser choice, the parser instance that choice runs
/// (null when the plan is inert), and the reasoning visibility policy. Built once at model load from the probed
/// chat-template capabilities plus configuration and stored with the loaded engine, so request behavior consumes a
/// resolved value instead of deciding again; only a new-requests-only reload re-resolves it. Reasoning boundaries are
/// recognized per generation against the markup that generation actually contains.
/// </summary>
public sealed record OutputParserPlan(ToolParser ToolParser, IToolOutputParser? Parser, ReasoningPolicy Reasoning)
{
    public OutputParserPlan(ToolParser toolParser, IToolOutputParser? parser)
        : this(toolParser, parser, ReasoningPolicy.Keep)
    {
    }

    /// <summary>The reasoning region this model resolves at load: the marker pair its budgets bound. Null when the model entry declares no reasoning budget — nothing is resolved here to bound, and a per-request reasoning override names that refusal rather than guessing a region.</summary>
    public ReasoningRegionMarkers? ReasoningRegion { get; init; }

    /// <summary>True when this plan resolves a reasoning region, which is what makes a reasoning ceiling mean something for this model.</summary>
    public bool HasResolvedReasoningRegion => ReasoningRegion is not null;
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
/// The parser registry: resolves one model's configured <c>tool_parser</c> and <c>reasoning</c> keys against the
/// chat-template capabilities probed at load into a concrete plan. Selection keys are template capabilities and
/// configuration only — never an architecture name; reasoning boundaries come from markup a generation contains, so the
/// resolved policy says what to do with them once they are found. <c>auto</c> keeps today's precedence exactly: a loaded
/// template that renders tools owns the tool surface (and its own call shape is what the decoded text is read for),
/// LAIR's fenced block plus fenced-JSON reading is the fallback for a template without tool support (and for a model with
/// no chat template at all), and the model's <c>force_fenced_tools</c> override selects it even for a tool-capable
/// template. An absent or unrecognized reasoning key lands on <c>keep</c>, which is today's wire behavior, and a model entry that declares no
/// reasoning ceiling resolves no reasoning region at all — the absence request-time budget rules refuse against instead of guessing one.
/// </summary>
public static class OutputParserRegistry
{
    /// <summary>The plan one model runs: resolved from its configuration plus its load-probed capabilities. The reasoning region is the vocabulary-proven marker pair carried in, never a guess.</summary>
    public static OutputParserPlan Resolve(string toolParser, bool forceFencedTools, TemplateCapabilities? capabilities, string reasoning = ReasoningPolicyNames.Keep, ReasoningRegionMarkers? reasoningRegion = null) => toolParser switch
    {
        ToolParserNames.Template => Plan(ToolParser.Template, new TemplateNativeToolOutputParser(), reasoning, reasoningRegion),
        ToolParserNames.Fenced => Plan(ToolParser.Fenced, new FencedJsonToolOutputParser(), reasoning, reasoningRegion),
        ToolParserNames.None => Plan(ToolParser.None, parser: null, reasoning, reasoningRegion),

        // "auto" (the only remaining loader-accepted value): today's capability precedence, expressed as a resolved choice.
        _ => forceFencedTools || capabilities is null || !capabilities.SupportsTools
            ? Plan(ToolParser.Fenced, new FencedJsonToolOutputParser(), reasoning, reasoningRegion)
            : Plan(ToolParser.Template, new TemplateNativeToolOutputParser(), reasoning, reasoningRegion),
    };

    private static OutputParserPlan Plan(ToolParser toolParser, IToolOutputParser? parser, string reasoning, ReasoningRegionMarkers? region) =>
        new(toolParser, parser, reasoning == ReasoningPolicyNames.Strip ? ReasoningPolicy.Strip : ReasoningPolicy.Keep) { ReasoningRegion = region };

    /// <summary>
    /// Proves what a reasoning ceiling could bound in this model's vocabulary: each marker must arrive whole as one token — an entry whose single-token decode is exactly the markup the recognizer reads over generated text. The proof does not depend on whether a ceiling is declared, because a hot-reloaded ceiling can arrive after load; a caller that declares a ceiling without a proven pair refuses it rather than serving a bound that would bind nothing.
    /// </summary>
    public static bool TryResolveMarkerPair(GgufTokenizer tokenizer, out ReasoningRegionMarkers? pair, out string refusalDetail)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);

        if (!TryResolveMarker(ReasoningRecognition.OpenMarkup, tokenizer, out int openId, out refusalDetail))
        {
            pair = null;
            return false;
        }

        if (!TryResolveMarker(ReasoningRecognition.CloseMarkup, tokenizer, out int closeId, out refusalDetail))
        {
            pair = null;
            return false;
        }

        pair = new ReasoningRegionMarkers(ReasoningRecognition.OpenMarkup, ReasoningRecognition.CloseMarkup)
        {
            OpenTokenId = openId,
            CloseTokenId = closeId,
        };
        refusalDetail = string.Empty;
        return true;
    }

    private static bool TryResolveMarker(string markup, GgufTokenizer tokenizer, out int tokenId, out string refusalDetail)
    {
        if (tokenizer.IdOf(markup) is not int id)
        {
            tokenId = -1;
            refusalDetail = $"'{markup}' is not an entry in this model's vocabulary";
            return false;
        }

        string decoded = tokenizer.Decode(new[] { id });
        if (!string.Equals(decoded, markup, StringComparison.Ordinal))
        {
            tokenId = -1;
            refusalDetail = $"the '{markup}' entry decodes to '{decoded}' rather than the marker itself (this build drops that token type from generated text)";
            return false;
        }

        tokenId = id;
        refusalDetail = string.Empty;
        return true;
    }
}
