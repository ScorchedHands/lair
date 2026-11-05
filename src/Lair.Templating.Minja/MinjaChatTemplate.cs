using System.Globalization;
using System.Text;
using System.Text.Json;

namespace Lair.Templating.Minja;

/// <summary>
/// Library-level minja chat-template wrapper: a parsed chat template plus its BOS/EOS tokens, rendered against
/// HF-style chat inputs. It supplies the minja chat context (<c>messages</c>, <c>add_generation_prompt</c>,
/// <c>bos_token</c>, <c>eos_token</c>, <c>tools</c> when provided, extra context and a deterministic
/// <c>strftime_now</c>), detects the template's tool/system/tool-call/tool-response/typed-content capabilities
/// through dummy-render probes, and normalizes message and tool shapes to fit those capabilities. It is the
/// reusable chat layer the later LAIR adapter builds on; it does not touch the tokenizer, HTTP or request pipeline.
/// </summary>
public sealed class MinjaChatTemplate
{
    private readonly TemplateNode _template;
    private readonly string _bosToken;
    private readonly string _eosToken;

    private ProbeFacts? _facts;

    /// <summary>
    /// Parses a chat template once. <paramref name="bosToken"/> and <paramref name="eosToken"/> are the model's
    /// literal BOS/EOS strings bound to the template's <c>bos_token</c>/<c>eos_token</c> context variables; empty
    /// strings leave those names defined-but-empty, matching templates that guard their use.
    /// </summary>
    /// <exception cref="MinjaSyntaxException">The template source does not parse.</exception>
    public MinjaChatTemplate(string templateSource, string bosToken = "", string eosToken = "")
    {
        ArgumentNullException.ThrowIfNull(templateSource);

        _template = MinjaParser.Parse(templateSource);
        _bosToken = bosToken;
        _eosToken = eosToken;
    }

    /// <summary>
    /// Returns the template's capability facts, probing once and caching the result. Probing renders the template
    /// against dummy messages/tools and records what appears in the output; a probe that fails to render counts as
    /// that capability being absent.
    /// </summary>
    public TemplateCapabilities DetectCapabilities() => ToCapabilities(EnsureProbed());

    /// <summary>
    /// Renders the chat template against a <see cref="ChatTemplateInput"/>. The returned
    /// <see cref="ChatTemplateRenderResult"/> carries the prompt, the capability facts (so the caller can decide
    /// whether to inject its own tool fallback), and whether the prompt already starts with the model BOS.
    /// </summary>
    /// <exception cref="MinjaRenderException">The template failed to render the (possibly normalized) input.</exception>
    public ChatTemplateRenderResult Render(ChatTemplateInput input, ChatTemplateOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(input);

        options ??= new ChatTemplateOptions();

        // Normalization needs real capability facts, so it only runs when capabilities are detected. With detection
        // off the caller renders raw and reads no capability guidance from this call.
        ProbeFacts? facts = options.DetectCapabilities ? EnsureProbed() : null;
        TemplateCapabilities capabilities = facts is not null ? ToCapabilities(facts) : new TemplateCapabilities();

        IReadOnlyList<ChatMessage> messages = facts is not null && options.NormalizeMessages
            ? Normalize(input.Messages, facts)
            : input.Messages;

        DateTimeOffset now = input.Now ?? DateTimeOffset.UtcNow;

        Dictionary<string, MinjaValue> context = BuildContext(messages, input);

        TemplateScope scope = TemplateScope.Root(context);
        scope.DefineCallable("strftime_now", new StrftimeNowValue(now));

        string prompt = MinjaEvaluator.Render(_template, scope);

        return new ChatTemplateRenderResult
        {
            Prompt = prompt,
            Capabilities = capabilities,
            StartsWithBos = _bosToken.Length > 0 && prompt.StartsWith(_bosToken, StringComparison.Ordinal),
        };
    }

    private ProbeFacts EnsureProbed() => _facts ??= Probe();

    // ---------------------------------------------------------------- context construction

    private Dictionary<string, MinjaValue> BuildContext(IReadOnlyList<ChatMessage> messages, ChatTemplateInput input)
    {
        Dictionary<string, MinjaValue> context = new(StringComparer.Ordinal)
        {
            ["messages"] = MinjaValue.List(messages.Select(MessageValue)),
            ["add_generation_prompt"] = MinjaValue.Of(input.AddGenerationPrompt),
            ["bos_token"] = MinjaValue.Of(_bosToken),
            ["eos_token"] = MinjaValue.Of(_eosToken),
        };

        // minja only defines `tools` when tools are supplied; templates branch on `tools is defined`.
        if (input.Tools is { Count: > 0 } tools)
            context["tools"] = MinjaValue.List(tools);

        if (input.ExtraContext is not null)
        {
            foreach (KeyValuePair<string, MinjaValue> entry in input.ExtraContext)
                context[entry.Key] = entry.Value;
        }

        // `date_string` is only bound when the caller supplies it, so a template's own default branch (which calls
        // strftime_now) still runs when it is absent.
        if (input.DateString is not null)
            context["date_string"] = MinjaValue.Of(input.DateString);

        return context;
    }

    /// <summary>Converts one typed chat record into the map a template iterates over.</summary>
    private static MinjaValue MessageValue(ChatMessage message)
    {
        List<KeyValuePair<string, MinjaValue>> entries = [new("role", MinjaValue.Of(message.Role)), new("content", message.Content)];

        if (message.ToolCalls is { Count: > 0 } toolCalls)
            entries.Add(new("tool_calls", MinjaValue.List(toolCalls.Select(ToolCallValue))));

        if (message.Name is not null)
            entries.Add(new("name", MinjaValue.Of(message.Name)));

        if (message.ToolCallId is not null)
            entries.Add(new("tool_call_id", MinjaValue.Of(message.ToolCallId)));

        return MinjaValue.Map(entries);
    }

    private static MinjaValue ToolCallValue(ChatToolCall call)
    {
        MinjaValue function = MinjaValue.Map([
            new KeyValuePair<string, MinjaValue>("name", MinjaValue.Of(call.Name)),
            new KeyValuePair<string, MinjaValue>("arguments", call.Arguments),
        ]);

        List<KeyValuePair<string, MinjaValue>> entries = [
            new("type", MinjaValue.Of("function")),
            new("function", function),
        ];

        if (call.Id is not null)
            entries.Add(new("id", MinjaValue.Of(call.Id)));

        return MinjaValue.Map(entries);
    }

    // ---------------------------------------------------------------- message normalization

    /// <summary>
    /// Rewrites message shapes to fit the template's capabilities: typed-content wrapping/flattening, tool-call
    /// argument object/string conversion, tool-response fallback for templates that cannot render <c>role: tool</c>,
    /// and system-role fallback for templates that cannot render <c>role: system</c>. The rewrite is a copy; the
    /// caller's input records are never mutated.
    /// </summary>
    private static IReadOnlyList<ChatMessage> Normalize(IReadOnlyList<ChatMessage> messages, ProbeFacts facts)
    {
        List<ChatMessage> normalized = [];

        foreach (ChatMessage message in messages)
        {
            MinjaValue content = NormalizeContent(message.Content, facts);
            IReadOnlyList<ChatToolCall>? toolCalls = message.ToolCalls is null
                ? null
                : [.. message.ToolCalls.Select(call => NormalizeToolCall(call, facts))];

            normalized.Add(message with { Content = content, ToolCalls = toolCalls });
        }

        if (!facts.SupportsToolResponses)
            normalized = [.. normalized.Select(ConvertToolResponse)];

        if (!facts.SupportsSystemRole)
            normalized = MergeSystemIntoUser(normalized);

        return normalized;
    }

    private static MinjaValue NormalizeContent(MinjaValue content, ProbeFacts facts)
    {
        if (facts.RequiresTypedContent)
            return content.Kind == MinjaValueKind.String ? WrapTypedContent(content.StringValue) : content;

        // Typed content is not required: flatten a typed list to plain text when every part is a text part.
        if (content.Kind == MinjaValueKind.List)
            return TryFlattenTypedContent(content, out string text) ? MinjaValue.Of(text) : content;

        return content;
    }

    /// <summary>Wraps plain text as a single text part, the shape typed-content templates expect.</summary>
    private static MinjaValue WrapTypedContent(string text) => MinjaValue.List([MinjaValue.Map([
        new KeyValuePair<string, MinjaValue>("type", MinjaValue.Of("text")),
        new KeyValuePair<string, MinjaValue>("text", MinjaValue.Of(text)),
    ])]);

    private static bool TryFlattenTypedContent(MinjaValue list, out string text)
    {
        StringBuilder builder = new();

        foreach (MinjaValue part in list.Items)
        {
            if (part.Kind != MinjaValueKind.Map)
            {
                text = string.Empty;
                return false;
            }

            MinjaValue type = part.MapValue.Get("type");

            if (!type.IsNone && type.Kind == MinjaValueKind.String && type.StringValue != "text")
            {
                // A non-text part (image/audio/etc.) cannot be flattened to plain text safely.
                text = string.Empty;
                return false;
            }

            builder.Append(part.MapValue.Get("text").ToString());
        }

        text = builder.ToString();
        return true;
    }

    private static ChatToolCall NormalizeToolCall(ChatToolCall call, ProbeFacts facts) => facts.ArgMode switch
    {
        // Template needs object arguments: convert a JSON-string argument back into a map/list.
        ArgumentMode.RequireObject when call.Arguments.Kind == MinjaValueKind.String =>
            call with { Arguments = ParseJsonArguments(call.Arguments.StringValue) },

        // Template wants pre-stringified text and cannot read maps: dump the supplied map as minja-compatible JSON.
        ArgumentMode.StringifyMap when call.Arguments.Kind == MinjaValueKind.Map =>
            call with { Arguments = MinjaValue.Of(call.Arguments.ToJson()) },

        _ => call,
    };

    /// <summary>
    /// Parses a JSON-argument string back into a map/list for templates that require object arguments. JSON is
    /// parsed with System.Text.Json and mapped onto minja values (objects become ordered maps, <c>true</c>/
    /// <c>false</c>/<c>null</c> become minja booleans/none); a value that is not valid JSON is left as the original
    /// string rather than failing the whole render.
    /// </summary>
    private static MinjaValue ParseJsonArguments(string json)
    {
        try
        {
            using JsonDocument document = JsonDocument.Parse(json);

            return JsonElementToMinja(document.RootElement);
        }
        catch (JsonException)
        {
            return MinjaValue.Of(json);
        }
    }

    private static MinjaValue JsonElementToMinja(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.Object => MinjaValue.Map(element.EnumerateObject().Select(property =>
            new KeyValuePair<string, MinjaValue>(property.Name, JsonElementToMinja(property.Value)))),

        JsonValueKind.Array => MinjaValue.List(element.EnumerateArray().Select(JsonElementToMinja)),

        JsonValueKind.String => MinjaValue.Of(element.GetString() ?? string.Empty),

        JsonValueKind.True => MinjaValue.True,
        JsonValueKind.False => MinjaValue.False,
        JsonValueKind.Null => MinjaValue.None,

        JsonValueKind.Number => element.TryGetInt64(out long integer)
            ? MinjaValue.Of(integer)
            : MinjaValue.Of(element.GetDouble()),

        _ => MinjaValue.None,
    };

    /// <summary>
    /// Converts a <c>role: tool</c> response into a user/tool-response fallback shape for templates that cannot
    /// render tool responses natively: the message becomes a <c>user</c> message that keeps its content and still
    /// carries the tool name (<c>name</c>) and call id (<c>tool_call_id</c>), so minja-style templates that branch
    /// on message name or render user content surface the response instead of dropping it.
    /// </summary>
    private static ChatMessage ConvertToolResponse(ChatMessage message)
    {
        if (message.Role != "tool")
            return message;

        return message with { Role = "user" };
    }

    /// <summary>
    /// Fallback for templates without system-role support: system content is merged into the first user message
    /// (separated by a blank line). When no user message exists, a user message carrying the system content is
    /// inserted before the remaining messages so the system instruction is never dropped.
    /// </summary>
    private static List<ChatMessage> MergeSystemIntoUser(List<ChatMessage> messages)
    {
        List<string> systemParts = [.. messages.Where(message => message.Role == "system").Select(message => message.Content.ToString())];

        if (systemParts.Count == 0)
            return messages;

        List<ChatMessage> rest = [.. messages.Where(message => message.Role != "system")];
        string mergedSystem = string.Join("\n\n", systemParts.Where(part => part.Length > 0));

        int firstUser = rest.FindIndex(message => message.Role == "user");

        if (firstUser >= 0)
        {
            ChatMessage user = rest[firstUser];
            string existing = user.Content.ToString();
            string combined = mergedSystem.Length == 0 ? existing : existing.Length == 0 ? mergedSystem : mergedSystem + "\n\n" + existing;

            rest[firstUser] = user with { Content = MinjaValue.Of(combined) };

            return rest;
        }

        ChatMessage fallback = new() { Role = "user", Content = MinjaValue.Of(mergedSystem) };

        return [fallback, .. rest];
    }

    // ---------------------------------------------------------------- capability probing

    private ProbeFacts Probe()
    {
        DateTimeOffset now = new(2024, 1, 2, 3, 4, 5, TimeSpan.Zero);

        bool requiresTyped = ProbeTypedContent(now);
        bool requiresNonNull = ProbeNonNullContent(now);

        // Later probes use the content shape the template prefers so they do not fail for the wrong reason.
        Func<string, MinjaValue> content = requiresTyped ? (Func<string, MinjaValue>)TypedContent : (s => MinjaValue.Of(s));

        return new ProbeFacts
        {
            SupportsTools = Contains(ProbeTools(now), ProbeToolName),
            SupportsSystemRole = Contains(ProbeSystem(now, content), ProbeSystemContent),
            SupportsToolCalls = Contains(ProbeToolCall(now, content), ProbeCallName),
            SupportsParallelToolCalls = ProbeParallelCalls(now, content),
            SupportsToolCallId = Contains(ProbeToolCallId(now, content), ProbeCallId),
            SupportsToolResponses = Contains(ProbeToolResponse(now, content), ProbeToolResponseContent),
            RequiresTypedContent = requiresTyped,
            RequiresNonNullContent = requiresNonNull,
            ArgMode = ProbeArgumentMode(now, content),
        };
    }

    /// <summary>
    /// Decides how tool-call arguments must be shaped by rendering the same dummy call with map arguments and with
    /// a JSON-string argument. If only the map renders the template needs objects (convert strings to maps); if only
    /// the string renders it wants pre-stringified text (dump maps to JSON); if both or neither render, leave the
    /// caller's shape untouched so templates that dump arguments themselves are not double-encoded.
    /// </summary>
    private ArgumentMode ProbeArgumentMode(DateTimeOffset now, Func<string, MinjaValue> content)
    {
        MinjaValue argumentMap = MapOf(("probe_arg", ProbeArgValue));

        bool rendersObjects = Contains(ProbeRender([AssistantWithCalls(content("a"), [Call(ProbeCallName, argumentMap, ProbeCallId)])], null, now), ProbeArgValue);
        bool rendersStrings = Contains(ProbeRender([AssistantWithCalls(content("a"), [Call(ProbeCallName, MinjaValue.Of(argumentMap.ToJson()), ProbeCallId)])], null, now), ProbeArgValue);

        if (rendersObjects && !rendersStrings)
            return ArgumentMode.RequireObject;

        if (!rendersObjects && rendersStrings)
            return ArgumentMode.StringifyMap;

        return ArgumentMode.AsSupplied;
    }

    private static TemplateCapabilities ToCapabilities(ProbeFacts facts) => new()
    {
        SupportsTools = facts.SupportsTools,
        SupportsToolCalls = facts.SupportsToolCalls,
        SupportsToolResponses = facts.SupportsToolResponses,
        SupportsSystemRole = facts.SupportsSystemRole,
        SupportsParallelToolCalls = facts.SupportsParallelToolCalls,
        SupportsToolCallId = facts.SupportsToolCallId,
        RequiresObjectArguments = facts.ArgMode == ArgumentMode.RequireObject,
        RequiresTypedContent = facts.RequiresTypedContent,
        RequiresNonNullContent = facts.RequiresNonNullContent,
    };

    private string? ProbeRender(IReadOnlyList<ChatMessage> messages, IReadOnlyList<MinjaValue>? tools, DateTimeOffset now)
    {
        Dictionary<string, MinjaValue> context = new(StringComparer.Ordinal)
        {
            ["messages"] = MinjaValue.List(messages.Select(MessageValue)),
            ["add_generation_prompt"] = MinjaValue.True,
            ["bos_token"] = MinjaValue.Of(_bosToken),
            ["eos_token"] = MinjaValue.Of(_eosToken),
        };

        if (tools is { Count: > 0 })
            context["tools"] = MinjaValue.List(tools);

        try
        {
            TemplateScope scope = TemplateScope.Root(context);
            scope.DefineCallable("strftime_now", new StrftimeNowValue(now));

            return MinjaEvaluator.Render(_template, scope);
        }
        catch (Exception failure) when (failure is MinjaRenderException or MinjaSyntaxException)
        {
            // A probe that cannot render is treated as the capability being absent.
            return null;
        }
    }

    private static bool Contains(string? output, string marker) => output is not null && output.Contains(marker, StringComparison.Ordinal);

    private bool ProbeTypedContent(DateTimeOffset now)
    {
        // A typed list renders but the same text as a plain string does not (or fails) → typed content required.
        string typed = ProbeRender([Text("user", TypedContent(ProbeTypedText))], null, now) ?? string.Empty;
        string plain = ProbeRender([Text("user", MinjaValue.Of(ProbeTypedText))], null, now) ?? string.Empty;

        bool typedShows = typed.Contains(ProbeTypedText, StringComparison.Ordinal);
        bool plainShows = plain.Contains(ProbeTypedText, StringComparison.Ordinal);

        // Typed content is required when the typed form is understood but the plain form is not rendered.
        return typedShows && !plainShows;
    }

    private bool ProbeNonNullContent(DateTimeOffset now)
    {
        string withText = ProbeRender([Text("user", "x")], null, now) ?? string.Empty;
        string? withNull = ProbeRender([new ChatMessage { Role = "user", Content = MinjaValue.None }], null, now);

        // Non-null content is required when a null-content message cannot render while a text one can.
        return withNull is null && withText.Length > 0;
    }

    private string? ProbeTools(DateTimeOffset now) =>
        ProbeRender([Text("user", "q")], [ProbeToolDefinition()], now);

    private string? ProbeSystem(DateTimeOffset now, Func<string, MinjaValue> content) =>
        ProbeRender([new ChatMessage { Role = "system", Content = content(ProbeSystemContent) }], null, now);

    private string? ProbeToolCall(DateTimeOffset now, Func<string, MinjaValue> content) =>
        ProbeRender([AssistantWithCalls(content("a"), [Call(ProbeCallName, MapOf(("probe_arg", ProbeArgValue)), ProbeCallId)])], null, now);

    private bool ProbeParallelCalls(DateTimeOffset now, Func<string, MinjaValue> content)
    {
        string? output = ProbeRender([
            AssistantWithCalls(content("a"), [
                Call(ProbeCallName, MapOf(("probe_arg", ProbeArgValue)), ProbeCallId),
                Call(ProbeCallName2, MapOf(("probe_arg", ProbeArgValue2)), ProbeCallId2),
            ]),
        ], null, now);

        return output is not null
            && output.Contains(ProbeCallName, StringComparison.Ordinal)
            && output.Contains(ProbeCallName2, StringComparison.Ordinal);
    }

    private string? ProbeToolCallId(DateTimeOffset now, Func<string, MinjaValue> content) =>
        ProbeRender([AssistantWithCalls(content("a"), [Call(ProbeCallName, MapOf(("probe_arg", ProbeArgValue)), ProbeCallId)])], null, now);

    private string? ProbeToolResponse(DateTimeOffset now, Func<string, MinjaValue> content)
    {
        ChatMessage response = new()
        {
            Role = "tool",
            Content = content(ProbeToolResponseContent),
            Name = ProbeToolName,
            ToolCallId = ProbeCallId,
        };

        return ProbeRender([response], null, now);
    }

    // ---------------------------------------------------------------- probe fixtures

    private const string ProbeToolName = "__lair_probe_tool_name__";
    private const string ProbeToolName2 = "__lair_probe_tool_name2__";
    private const string ProbeCallName = "__lair_probe_call_name__";
    private const string ProbeCallName2 = "__lair_probe_call_name2__";
    private const string ProbeCallId = "__lair_probe_call_id__";
    private const string ProbeCallId2 = "__lair_probe_call_id2__";
    private const string ProbeArgValue = "__lair_probe_arg_value__";
    private const string ProbeArgValue2 = "__lair_probe_arg_value2__";
    private const string ProbeSystemContent = "__lair_probe_system__";
    private const string ProbeToolResponseContent = "__lair_probe_tool_response__";
    private const string ProbeTypedText = "__lair_probe_typed_text__";

    private static MinjaValue ProbeToolDefinition() => MinjaValue.Map([
        new KeyValuePair<string, MinjaValue>("type", MinjaValue.Of("function")),
        new KeyValuePair<string, MinjaValue>("function", MinjaValue.Map([
            new KeyValuePair<string, MinjaValue>("name", MinjaValue.Of(ProbeToolName)),
            new KeyValuePair<string, MinjaValue>("description", MinjaValue.Of("probe description")),
            new KeyValuePair<string, MinjaValue>("parameters", MinjaValue.Map([
                new KeyValuePair<string, MinjaValue>("type", MinjaValue.Of("object")),
                new KeyValuePair<string, MinjaValue>("properties", MinjaValue.Map([
                    new KeyValuePair<string, MinjaValue>("probe_arg", MinjaValue.Map([
                        new KeyValuePair<string, MinjaValue>("type", MinjaValue.Of("string")),
                    ])),
                ])),
            ])),
        ])),
    ]);

    private static ChatMessage Text(string role, MinjaValue content) => new() { Role = role, Content = content };

    private static ChatMessage Text(string role, string content) => new() { Role = role, Content = MinjaValue.Of(content) };

    private static ChatMessage AssistantWithCalls(MinjaValue content, IReadOnlyList<ChatToolCall> calls) =>
        new() { Role = "assistant", Content = content, ToolCalls = calls };

    private static ChatToolCall Call(string name, MinjaValue arguments, string id) =>
        new() { Name = name, Arguments = arguments, Id = id };

    private static MinjaValue TypedContent(string text) => MinjaValue.List([MinjaValue.Map([
        new KeyValuePair<string, MinjaValue>("type", MinjaValue.Of("text")),
        new KeyValuePair<string, MinjaValue>("text", MinjaValue.Of(text)),
    ])]);

    private static MinjaValue MapOf(params (string Key, string Value)[] entries) =>
        MinjaValue.Map(entries.Select(entry => new KeyValuePair<string, MinjaValue>(entry.Key, MinjaValue.Of(entry.Value))));

    // ---------------------------------------------------------------- strftime_now

    /// <summary>
    /// The <c>strftime_now(fmt)</c> context callable: it renders a fixed clock value through a bounded set of
    /// strftime tokens used by chat templates. The value is captured once per render so every call in one render
    /// agrees, keeping output deterministic when the caller supplies a fixed <c>Now</c>.
    /// </summary>
    private sealed class StrftimeNowValue(DateTimeOffset now) : MinjaCallable
    {
        public MinjaValue Invoke(IReadOnlyList<CallArgument> arguments, TemplateScope callSite, int offset, CallerBlockValue? caller, MinjaValue? input)
        {
            if (arguments.Count != 1)
                throw new MinjaRenderException("strftime_now", offset, $"strftime_now takes exactly one format string but found {arguments.Count}");

            if (arguments[0].Name is not null)
                throw new MinjaRenderException("strftime_now", offset, $"strftime_now does not take a named argument '{arguments[0].Name}'");

            string format = MinjaEvaluator.EvaluateInScope(arguments[0].Value, callSite).ToString();

            return MinjaValue.Of(Format(now, format));
        }

        /// <summary>Maps the strftime tokens chat templates use to invariant-culture .NET formatting.</summary>
        private static string Format(DateTimeOffset value, string format)
        {
            StringBuilder builder = new(format.Length + 8);

            for (int i = 0; i < format.Length; i++)
            {
                if (format[i] != '%' || i + 1 >= format.Length)
                {
                    builder.Append(format[i]);
                    continue;
                }

                char token = format[i + 1];
                i++;

                switch (token)
                {
                    case 'Y':
                        builder.Append(value.ToString("yyyy", CultureInfo.InvariantCulture));
                        break;

                    case 'm':
                        builder.Append(value.ToString("MM", CultureInfo.InvariantCulture));
                        break;

                    case 'd':
                        builder.Append(value.ToString("dd", CultureInfo.InvariantCulture));
                        break;

                    case 'b':
                        builder.Append(value.ToString("MMM", CultureInfo.InvariantCulture));
                        break;

                    case 'H':
                        builder.Append(value.ToString("HH", CultureInfo.InvariantCulture));
                        break;

                    case 'M':
                        builder.Append(value.ToString("mm", CultureInfo.InvariantCulture));
                        break;

                    case 'S':
                        builder.Append(value.ToString("ss", CultureInfo.InvariantCulture));
                        break;

                    case '%':
                        builder.Append('%');
                        break;

                    default:
                        // Unknown tokens pass through verbatim (the literal %token), matching a permissive strftime.
                        builder.Append('%').Append(token);
                        break;
                }
            }

            return builder.ToString();
        }
    }

    /// <summary>How tool-call arguments must be shaped for this template, derived from the dummy-argument probes.</summary>
    private enum ArgumentMode
    {
        /// <summary>The caller's argument shape is used unchanged (the template accepts both shapes or neither).</summary>
        AsSupplied,

        /// <summary>The template renders object arguments but not JSON strings: convert string arguments to maps.</summary>
        RequireObject,

        /// <summary>The template renders a pre-stringified argument but not a map: dump supplied maps to JSON.</summary>
        StringifyMap,
    }

    /// <summary>
    /// Raw capability probe results. These drive normalization directly; the public <see cref="TemplateCapabilities"/>
    /// is derived from them for callers that only need the reported facts.
    /// </summary>
    private sealed record ProbeFacts
    {
        public bool SupportsTools { get; init; }
        public bool SupportsToolCalls { get; init; }
        public bool SupportsToolResponses { get; init; }
        public bool SupportsSystemRole { get; init; }
        public bool SupportsParallelToolCalls { get; init; }
        public bool SupportsToolCallId { get; init; }
        public bool RequiresTypedContent { get; init; }
        public bool RequiresNonNullContent { get; init; }
        public ArgumentMode ArgMode { get; init; } = ArgumentMode.AsSupplied;
    }
}
