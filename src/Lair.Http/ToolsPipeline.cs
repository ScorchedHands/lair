using System.Text;
using System.Text.Json;
using Lair.Templating.Minja;

namespace Lair.Http;

/// <summary>How the request constrains tool use.</summary>
public enum ToolChoiceMode
{
    /// <summary>The model may call any declared tool or none.</summary>
    Auto,

    /// <summary>No schemas are injected and no output is read as a call.</summary>
    None,

    /// <summary>A named function is the only callable one: every declared schema is still injected, but detection accepts only this name.</summary>
    Function,
}

/// <summary>The resolved tool-choice selection carried on a generation request.</summary>
public sealed record ToolChoiceSelection(ToolChoiceMode Mode, string? FunctionName)
{
    public static readonly ToolChoiceSelection Auto = new(ToolChoiceMode.Auto, null);
    public static readonly ToolChoiceSelection None = new(ToolChoiceMode.None, null);

    public static ToolChoiceSelection Function(string name) => new(ToolChoiceMode.Function, name);
}

/// <summary>One declared tool: the function's name, its optional description, and its parameter schema as raw JSON text.</summary>
public sealed record ToolDefinition(string Name, string? Description, string ParametersSchema);

/// <summary>
/// Appends the deterministic tool block to a rendered chat prompt before it is encoded. The append is family-agnostic:
/// every template receives the same instruction line, one fenced <c>```json</c> schema per declared tool, and the reply
/// contract naming the exact call shape the detector accepts. A model that never honors the contract still produces
/// ordinary text; only the injection, detection, and wire shape are guaranteed by the runtime.
/// </summary>
public static class ToolPromptComposer
{
    private const string InstructionLine = "You can call these tools:";

    /// <summary>The exact reply-contract line appended after the tool schemas.</summary>
    public static readonly string ContractLine =
        "To call a function, reply with ONLY a fenced ```json block containing an object of the form {\"name\":\"<tool_name>\",\"arguments\":{...}} naming one declared tool.";

    /// <summary>The rendered prompt plus the appended instruction line, per-tool schema fences, and the contract line.</summary>
    public static string Compose(string renderedPrompt, IReadOnlyList<ToolDefinition> tools)
    {
        ArgumentNullException.ThrowIfNull(renderedPrompt);
        ArgumentNullException.ThrowIfNull(tools);

        StringBuilder builder = new(renderedPrompt.Length + 256);
        builder.Append(renderedPrompt)
            .Append('\n')
            .Append(InstructionLine)
            .Append('\n');

        foreach (ToolDefinition tool in tools)
        {
            builder.Append("```json\n")
                .Append(SchemaJson(tool))
                .Append("\n```\n");
        }

        return builder.Append(ContractLine).ToString();
    }

    // The schema object written into each fence: name, optional description, and the parameter schema verbatim.
    private static string SchemaJson(ToolDefinition tool)
    {
        using MemoryStream buffer = new();
        using (Utf8JsonWriter writer = new(buffer))
        {
            writer.WriteStartObject();

            writer.WriteString("name", tool.Name);
            if (tool.Description is not null)
            {
                writer.WriteString("description", tool.Description);
            }

            writer.WritePropertyName("parameters");
            writer.WriteRawValue(tool.ParametersSchema);

            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.ToArray());
    }
}

/// <summary>
/// Converts one LAIR tool declaration into the HF-style tool-definition map a minja chat template consumes:
/// <c>{"type":"function","function":{"name":...,"description":...,"parameters":{...}}}</c> — the same shape the
/// wrapper's capability probe validated when it reported <c>SupportsTools</c>, so declared tools land where the
/// template already knows how to read them. The parameter schema is parsed from its raw JSON text; a schema that
/// does not parse is rejected naming the tool, matching the fenced path's requirement that schemas are JSON.
/// </summary>
internal static class TemplateToolValues
{
    public static MinjaValue Of(ToolDefinition tool)
    {
        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(tool.ParametersSchema);
        }
        catch (JsonException)
        {
            throw new ArgumentException(
                $"'tools' contains the declaration '{tool.Name}' whose 'parameters' is not valid JSON.", nameof(tool));
        }

        using (document)
        {
            List<KeyValuePair<string, MinjaValue>> function = [new("name", MinjaValue.Of(tool.Name))];

            if (tool.Description is not null)
            {
                function.Add(new("description", MinjaValue.Of(tool.Description)));
            }

            function.Add(new("parameters", FromElement(document.RootElement)));

            return MinjaValue.Map([
                new KeyValuePair<string, MinjaValue>("type", MinjaValue.Of("function")),
                new KeyValuePair<string, MinjaValue>("function", MinjaValue.Map(function)),
            ]);
        }
    }

    private static MinjaValue FromElement(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.Object => MinjaValue.Map(element.EnumerateObject().Select(property =>
            new KeyValuePair<string, MinjaValue>(property.Name, FromElement(property.Value)))),

        JsonValueKind.Array => MinjaValue.List(element.EnumerateArray().Select(FromElement)),

        JsonValueKind.String => MinjaValue.Of(element.GetString() ?? string.Empty),

        JsonValueKind.True => MinjaValue.True,
        JsonValueKind.False => MinjaValue.False,
        JsonValueKind.Null => MinjaValue.None,

        JsonValueKind.Number => element.TryGetInt64(out long integer)
            ? MinjaValue.Of(integer)
            : MinjaValue.Of(element.GetDouble()),

        _ => MinjaValue.None,
    };
}

/// <summary>One fenced-JSON object recognized as a call: the declared tool's name and its arguments re-encoded compactly.</summary>
public sealed record ToolCall(string Name, string Arguments);

/// <summary>The detector's outcome: every accepted call in document order plus the visible text with each accepted fence removed.</summary>
public sealed record ToolDetection(IReadOnlyList<ToolCall> Calls, string VisibleText);

/// <summary>
/// Reads fenced <c>```json</c> blocks out of generated text as tool calls. A block is a call only when it holds one JSON
/// object with exactly the properties <c>name</c> (a declared tool) and <c>arguments</c>; anything else — malformed or
/// trailing JSON, wrong property names, an unknown name under an open choice, another name under a forced choice, extra
/// properties — stays ordinary text, byte-for-byte. Accepted fences are removed from the visible text: each fence plus its
/// language tag line is cut out and a whitespace run touching either cut boundary collapses to one character, so removing
/// a fence never fuses surrounding words together or leaves doubled spaces behind.
/// </summary>
public static class ToolCallDetector
{
    /// <summary>Detects calls in fully decoded text against the set of names this request may call.</summary>
    public static ToolDetection Detect(string text, IReadOnlyCollection<string> allowedNames)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(allowedNames);

        List<ToolCall> calls = [];
        StringBuilder visible = new(text.Length);
        int cursor = 0;

        while (cursor < text.Length)
        {
            int fenceStart = text.IndexOf("```", cursor, StringComparison.Ordinal);
            if (fenceStart < 0)
            {
                break;
            }

            // The language tag runs to the end of its line; only a bare 'json' tag opens a candidate call block.
            int lineEnd = text.IndexOf('\n', fenceStart + 3);
            if (lineEnd < 0 || !IsJsonTag(text.AsSpan(fenceStart + 3, lineEnd - fenceStart - 3)))
            {
                AppendVisibleSegment(visible, text[cursor..fenceStart]);
                cursor = fenceStart; // not a call candidate: copy the marker verbatim and keep scanning after it
                visible.Append("```");
                cursor += 3;
                continue;
            }

            int closeIndex = text.IndexOf("```", lineEnd + 1, StringComparison.Ordinal);
            if (closeIndex < 0)
            {
                break; // an unterminated fence can only be ordinary output
            }

            string body = text[(lineEnd + 1)..closeIndex];
            ToolCall? call = ParseCall(body, allowedNames);
            if (call is null)
            {
                AppendVisibleSegment(visible, text[cursor..fenceStart]);
                cursor = fenceStart; // ordinary fenced output: copy it verbatim and keep scanning after it
                visible.Append("```");
                cursor += 3;
                continue;
            }

            AppendVisibleSegment(visible, text[cursor..fenceStart]);
            calls.Add(call);
            cursor = closeIndex + 3; // the fence's trailing newline stays in the following segment
        }

        AppendVisibleSegment(visible, text[cursor..]);
        return new ToolDetection(calls, visible.ToString());
    }

    private static bool IsJsonTag(ReadOnlySpan<char> tagLine) => tagLine.Trim().SequenceEqual("json");

    // One fence body is a call only when it parses to exactly one object with exactly the properties name (string, allowed)
    // and arguments; everything else returns null so the block stays ordinary text.
    private static ToolCall? ParseCall(string body, IReadOnlyCollection<string> allowedNames)
    {
        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(body.Trim());
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
            bool hasArguments = false;
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
                    case "arguments":
                        // Any JSON value is accepted and re-encoded compactly, matching the wire's string form.
                        if (hasArguments)
                        {
                            return null;
                        }

                        hasArguments = true;
                        break;
                    default:
                        return null;
                }
            }

            if (name is null || !hasArguments || properties != 2)
            {
                return null;
            }

            if (!allowedNames.Contains(name))
            {
                return null;
            }

            // Serialize the arguments through a writer so they land compactly on the wire regardless of how the model spaced them; raw-text passthrough would echo source whitespace.
            string? canonical = Compact(document.RootElement.GetProperty("arguments").GetRawText());
            return canonical is null ? null : new ToolCall(name, canonical);
        }
    }

    private static string? Compact(string rawArguments)
    {
        try
        {
            using JsonDocument document = JsonDocument.Parse(rawArguments);

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

    // Copies a stripped-out segment, collapsing at most one boundary space so removing a fence does not fuse surrounding words.
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

// Wire-side parsing of the request body's 'tools' and 'tool_choice' members, run while the parsed document is still alive so
// only typed results outlive it. Malformed shapes throw ArgumentException naming the offending field, which the HTTP error
// middleware maps to an invalid_request response.
internal static class ToolWire
{
    // Parses the body's 'tools' array into typed definitions: every entry must be a function object with a non-empty unique name
    // and an object-valued parameter schema. Absent or null declares nothing; request order is preserved.
    public static IReadOnlyList<ToolDefinition> ParseTools(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object
            || !root.TryGetProperty("tools", out JsonElement tools)
            || tools.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
        {
            return [];
        }

        if (tools.ValueKind != JsonValueKind.Array)
        {
            throw new ArgumentException("'tools' must be an array of function tool objects.");
        }

        List<ToolDefinition> parsed = new(tools.GetArrayLength());
        HashSet<string> names = new(StringComparer.Ordinal);
        int index = 0;

        foreach (JsonElement entry in tools.EnumerateArray())
        {
            if (entry.ValueKind != JsonValueKind.Object)
            {
                throw new ArgumentException(
                    $"'tools[{index}]' must be a function tool object of the form {{\"type\":\"function\",\"function\":{{...}}}}.");
            }

            string type = entry.TryGetProperty("type", out JsonElement typeElement) && typeElement.ValueKind == JsonValueKind.String
                ? typeElement.GetString() ?? string.Empty
                : string.Empty;
            if (type != "function")
            {
                throw new ArgumentException(
                    $"'tools[{index}].type' must be \"function\", got {(type.Length == 0 ? "no 'type'" : $"\"{type}\"")}.");
            }

            if (!entry.TryGetProperty("function", out JsonElement function) || function.ValueKind != JsonValueKind.Object)
            {
                throw new ArgumentException($"'tools[{index}].function' must be an object naming the tool.");
            }

            if (!function.TryGetProperty("name", out JsonElement nameElement) || nameElement.ValueKind != JsonValueKind.String)
            {
                throw new ArgumentException($"'tools[{index}].function.name' is required and must be a string.");
            }

            string name = nameElement.GetString()!;
            if (name.Length == 0)
            {
                throw new ArgumentException($"'tools[{index}].function.name' must not be empty.");
            }

            if (!names.Add(name))
            {
                throw new ArgumentException($"'tools' declares the tool '{name}' more than once.");
            }

            string? description = function.TryGetProperty("description", out JsonElement descriptionElement)
                && descriptionElement.ValueKind == JsonValueKind.String
                ? descriptionElement.GetString()
                : null;

            if (!function.TryGetProperty("parameters", out JsonElement parameters) || parameters.ValueKind != JsonValueKind.Object)
            {
                throw new ArgumentException($"'tools[{index}].function.parameters' is required and must be an object.");
            }

            parsed.Add(new ToolDefinition(name, description, parameters.GetRawText()));
            index++;
        }

        return parsed;
    }

    // Resolves the body's 'tool_choice': absent or "auto" allows every declared tool, "none" closes the pipeline entirely, and a
    // function object forces one named member of the declared set. Anything else names 'tool_choice' as invalid.
    public static ToolChoiceSelection ParseChoice(JsonElement root, IReadOnlyList<ToolDefinition> tools)
    {
        if (root.ValueKind != JsonValueKind.Object
            || !root.TryGetProperty("tool_choice", out JsonElement choice)
            || choice.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
        {
            return ToolChoiceSelection.Auto;
        }

        switch (choice.ValueKind)
        {
            case JsonValueKind.String:
                string mode = choice.GetString()!;
                return mode switch
                {
                    "auto" => ToolChoiceSelection.Auto,
                    "none" => ToolChoiceSelection.None,
                    _ => throw new ArgumentException(
                        $"'tool_choice' must be \"auto\", \"none\", or a function object naming a declared tool, got \"{mode}\"."),
                };

            case JsonValueKind.Object:
                if (!choice.TryGetProperty("type", out JsonElement typeElement) || typeElement.GetString() != "function")
                {
                    throw new ArgumentException(
                        "'tool_choice' must be \"auto\", \"none\", or an object of the form {\"type\":\"function\",\"function\":{\"name\":...}}.");
                }

                if (!choice.TryGetProperty("function", out JsonElement function) || function.ValueKind != JsonValueKind.Object)
                {
                    throw new ArgumentException("'tool_choice.function' must be an object naming the forced tool.");
                }

                if (!function.TryGetProperty("name", out JsonElement nameElement) || nameElement.ValueKind != JsonValueKind.String)
                {
                    throw new ArgumentException($"'tool_choice.function.name' is required and must be a string.");
                }

                string forced = nameElement.GetString()!;
                bool declared = false;
                foreach (ToolDefinition tool in tools)
                {
                    if (string.Equals(tool.Name, forced, StringComparison.Ordinal))
                    {
                        declared = true;
                        break;
                    }
                }

                if (!declared)
                {
                    throw new ArgumentException(
                        $"'tool_choice.function.name' names '{forced}', which is not one of the declared tools.");
                }

                return ToolChoiceSelection.Function(forced);

            default:
                throw new ArgumentException(
                    "'tool_choice' must be \"auto\", \"none\", or a function object naming a declared tool.");
        }
    }
}
