using Lair.Templating.Minja;

namespace Lair.Tests;

/// <summary>
/// Goldens for the library-level minja chat-template wrapper: ChatML/Phi rendering through MinjaChatTemplate,
/// BOS/EOS context handling, capability detection from dummy-render probes, and the message/tool polyfills (typed
/// content, object vs string tool-call arguments, tool-response and system-role fallbacks). These tests exercise the
/// wrapper API only and do not touch the tokenizer or the LAIR request pipeline.
/// </summary>
[Trait("Category", "MinjaChat")]
public sealed class MinjaChatTests
{
    private const string ChatMlTemplate =
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    private const string PhiTemplate =
        "{% for message in messages %}{{ message['role'] + '\n' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant\n' }}{% endif %}";

    // ChatML-family template that renders tool definitions, assistant tool calls (name + JSON arguments + id) and
    // ordinary turns, and supports multiple calls per assistant message.
    private const string ToolCallTemplate =
        "{%- if tools %}{{- 'TOOLS:\n' }}{%- for tool in tools %}{{- tool['function']['name'] + '\n' }}{%- endfor %}{%- endif %}"
        + "{%- for message in messages %}"
        + "{%- if message['tool_calls'] %}"
        + "{{- '<|call_start|>' }}"
        + "{%- for call in message['tool_calls'] %}{{- 'name=' + call['function']['name'] + '|args=' + (call['function']['arguments'] | tojson) + '|id=' + call['id'] + ';' }}{%- endfor %}"
        + "{%- else %}"
        + "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        + "{%- endif %}"
        + "{%- endfor %}"
        + "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}";

    // Tool-call template that reads arguments as a map (via the items filter) and therefore fails on a JSON string.
    private const string ObjectArgumentTemplate =
        "{%- for message in messages %}{%- if message['tool_calls'] %}{%- for call in message['tool_calls'] %}"
        + "CALL:{{ call['function']['name'] }}{%- for (key, value) in (call['function']['arguments'] | items) %}[{{ key }}={{ value }}]{%- endfor %}"
        + "{%- endfor %}{%- endif %}{%- endfor %}";

    // ChatML-family template that renders only system/user/assistant turns and ignores other roles.
    private const string GuardedChatMlTemplate =
        "{% for message in messages %}{% if message['role'] in ['system', 'user', 'assistant'] %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    // ChatML-family template that renders only user/assistant turns and drops a system role.
    private const string NoSystemTemplate =
        "{% for message in messages %}{% if message['role'] in ['user', 'assistant'] %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    // Template whose message content must be a typed list of {"type": "text", "text": ...} parts.
    private const string TypedContentTemplate =
        "{%- for message in messages %}{{- '<|start|>' + message['role'] + '\n' }}{%- for part in message['content'] %}{{ part['text'] }}{%- endfor %}{{- '\n' }}{%- endfor %}{% if add_generation_prompt %}{{ '<|start|>assistant\n' }}{% endif %}";

    // ---------------------------------------------------------------- ChatML / Phi goldens through the wrapper

    [Fact]
    public void ChatMl_WithGenerationPrompt_RendersExactChatML()
    {
        MinjaChatTemplate template = new(ChatMlTemplate);
        ChatTemplateRenderResult result = template.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("system", "You are helpful."), ChatMessage.Text("user", "Hello\nworld")],
            AddGenerationPrompt = true,
        });

        Assert.Equal(
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            + "<|im_start|>user\nHello\nworld<|im_end|>\n"
            + "<|im_start|>assistant\n",
            result.Prompt);
        Assert.False(result.StartsWithBos);
    }

    [Fact]
    public void ChatMl_WithoutGenerationPrompt_OmitsAssistantTurn()
    {
        MinjaChatTemplate template = new(ChatMlTemplate);
        ChatTemplateRenderResult result = template.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("system", "You are helpful."), ChatMessage.Text("user", "Hello\nworld")],
            AddGenerationPrompt = false,
        });

        Assert.Equal(
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            + "<|im_start|>user\nHello\nworld<|im_end|>\n",
            result.Prompt);
    }

    [Fact]
    public void PhiStyle_RendersThroughWrapperUnchanged()
    {
        MinjaChatTemplate template = new(PhiTemplate);
        ChatTemplateRenderResult result = template.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("system", "You are helpful."), ChatMessage.Text("user", "Hello\nworld")],
            AddGenerationPrompt = true,
        });

        Assert.Equal("system\nYou are helpful.\nuser\nHello\nworld\nassistant\n", result.Prompt);
    }

    // ---------------------------------------------------------------- BOS / EOS context

    [Fact]
    public void BosToken_InContextAndStartsWithBosWhenTemplateEmitsIt()
    {
        const string template = "{{ bos_token }}{% for message in messages %}{{ message['role'] + ':' + message['content'] + eos_token + '\n' }}{% endfor %}";
        MinjaChatTemplate chat = new(template, bosToken: "<s>", eosToken: "</s>");

        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "hi")],
            AddGenerationPrompt = false,
        });

        Assert.True(result.StartsWithBos);
        Assert.Equal("<s>user:hi</s>\n", result.Prompt);
    }

    [Fact]
    public void BosToken_NotReportedWhenTemplateDoesNotEmitIt()
    {
        MinjaChatTemplate chat = new(ChatMlTemplate, bosToken: "<s>", eosToken: "</s>");

        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "hi")],
            AddGenerationPrompt = false,
        });

        Assert.False(result.StartsWithBos);
        Assert.DoesNotContain("<s>", result.Prompt);
    }

    // ---------------------------------------------------------------- tool-capable templates

    [Fact]
    public void ToolCapableTemplate_RendersSuppliedToolsAndReportsToolSupport()
    {
        MinjaChatTemplate chat = new(ToolCallTemplate);
        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "weather?")],
            Tools = [Tool("get_weather")],
            AddGenerationPrompt = true,
        });

        Assert.True(result.Capabilities.SupportsTools);
        Assert.Contains("get_weather", result.Prompt);
        Assert.Contains("TOOLS:", result.Prompt);
    }

    [Fact]
    public void ToolCapableTemplate_OmitsToolSectionWhenNoToolsSupplied()
    {
        MinjaChatTemplate chat = new(ToolCallTemplate);
        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "weather?")],
            AddGenerationPrompt = true,
        });

        Assert.DoesNotContain("TOOLS:", result.Prompt);
    }

    [Fact]
    public void NonToolTemplate_ReportsNoToolSupportAndDoesNotInjectTools()
    {
        MinjaChatTemplate chat = new(ChatMlTemplate);
        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "hi")],
            Tools = [Tool("get_weather")],
            AddGenerationPrompt = false,
        });

        Assert.False(result.Capabilities.SupportsTools);
        Assert.DoesNotContain("get_weather", result.Prompt);
    }

    // ---------------------------------------------------------------- tool-call rendering

    [Fact]
    public void ToolCallTemplate_RendersCallNameJsonArgumentsAndId()
    {
        MinjaChatTemplate chat = new(ToolCallTemplate);
        ChatMessage assistant = new()
        {
            Role = "assistant",
            Content = MinjaValue.EmptyString,
            ToolCalls = [new ChatToolCall { Name = "lookup", Arguments = Map(("city", S("Paris"))), Id = "call_1" }],
        };

        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [assistant],
            AddGenerationPrompt = false,
        });

        Assert.True(result.Capabilities.SupportsToolCalls);
        Assert.True(result.Capabilities.SupportsToolCallId);
        Assert.Contains("name=lookup", result.Prompt);
        Assert.Contains("\"city\": \"Paris\"", result.Prompt);
        Assert.Contains("id=call_1", result.Prompt);
        // The template dumps arguments itself, so maps are passed through (not pre-stringified) and not double-encoded.
        Assert.False(result.Capabilities.RequiresObjectArguments);
    }

    [Fact]
    public void ToolCallTemplate_ReportsParallelCallsWhenBothNamesRender()
    {
        MinjaChatTemplate chat = new(ToolCallTemplate);
        ChatMessage assistant = new()
        {
            Role = "assistant",
            Content = MinjaValue.EmptyString,
            ToolCalls =
            [
                new ChatToolCall { Name = "first_call", Arguments = Map(("a", S("1"))), Id = "id1" },
                new ChatToolCall { Name = "second_call", Arguments = Map(("b", S("2"))), Id = "id2" },
            ],
        };

        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [assistant],
            AddGenerationPrompt = false,
        });

        Assert.True(result.Capabilities.SupportsParallelToolCalls);
        Assert.Contains("first_call", result.Prompt);
        Assert.Contains("second_call", result.Prompt);
    }

    [Fact]
    public void ObjectArgumentTemplate_RequiresObjectArgumentsAndConvertsStringArguments()
    {
        MinjaChatTemplate chat = new(ObjectArgumentTemplate);

        // The caller supplies a JSON-string argument; the wrapper parses it into a map so the items-based template renders it.
        ChatMessage assistant = new()
        {
            Role = "assistant",
            Content = MinjaValue.EmptyString,
            ToolCalls = [new ChatToolCall { Name = "lookup", Arguments = MinjaValue.Of("{\"city\":\"Paris\"}"), Id = "call_1" }],
        };

        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [assistant],
            AddGenerationPrompt = false,
        });

        Assert.True(result.Capabilities.RequiresObjectArguments);
        Assert.Contains("CALL:lookup", result.Prompt);
        Assert.Contains("[city=Paris]", result.Prompt);
    }

    [Fact]
    public void ObjectArgumentTemplate_RendersMapArgumentsDirectly()
    {
        MinjaChatTemplate chat = new(ObjectArgumentTemplate);
        ChatMessage assistant = new()
        {
            Role = "assistant",
            Content = MinjaValue.EmptyString,
            ToolCalls = [new ChatToolCall { Name = "lookup", Arguments = Map(("city", S("Paris"))), Id = "call_1" }],
        };

        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [assistant],
            AddGenerationPrompt = false,
        });

        Assert.Contains("[city=Paris]", result.Prompt);
    }

    // ---------------------------------------------------------------- tool-response fallback

    [Fact]
    public void ToolResponseFallback_ConvertsToolRoleToUserWhenTemplateCannotRenderIt()
    {
        MinjaChatTemplate chat = new(GuardedChatMlTemplate);

        ChatMessage assistant = new()
        {
            Role = "assistant",
            Content = MinjaValue.EmptyString,
            ToolCalls = [new ChatToolCall { Name = "get_weather", Arguments = Map(("city", S("Paris"))), Id = "call_1" }],
        };
        ChatMessage toolResponse = new() { Role = "tool", Content = S("42C"), Name = "get_weather", ToolCallId = "call_1" };

        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "weather?"), assistant, toolResponse],
            AddGenerationPrompt = true,
        });

        Assert.False(result.Capabilities.SupportsToolResponses);
        Assert.Contains("42C", result.Prompt);
        // The response surfaces under a user turn rather than a raw tool role.
        Assert.DoesNotContain("<|im_start|>tool", result.Prompt);
    }

    [Fact]
    public void ChatMl_HandlesToolRoleNativelySoNoFallbackTriggers()
    {
        MinjaChatTemplate chat = new(ChatMlTemplate);
        ChatMessage toolResponse = new() { Role = "tool", Content = S("42C"), Name = "get_weather", ToolCallId = "call_1" };

        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [toolResponse],
            AddGenerationPrompt = false,
        });

        // The generic loop renders any role's content, so tool responses are supported and left as-is.
        Assert.True(result.Capabilities.SupportsToolResponses);
        Assert.Contains("<|im_start|>tool\n42C", result.Prompt);
    }

    // ---------------------------------------------------------------- typed content

    [Fact]
    public void TypedContentTemplate_RequiresTypedContentAndWrapsPlainStrings()
    {
        MinjaChatTemplate chat = new(TypedContentTemplate);
        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "hi")],
            AddGenerationPrompt = false,
        });

        Assert.True(result.Capabilities.RequiresTypedContent);
        Assert.Contains("hi", result.Prompt);
    }

    [Fact]
    public void TypedContentTemplate_PassesSuppliedTypedListsThrough()
    {
        MinjaChatTemplate chat = new(TypedContentTemplate);
        MinjaValue typed = MinjaValue.List([Map(("type", S("text")), ("text", S("typed part")))]);

        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [new ChatMessage { Role = "user", Content = typed }],
            AddGenerationPrompt = false,
        });

        Assert.Contains("typed part", result.Prompt);
    }

    [Fact]
    public void PlainTemplate_FlattensSuppliedTypedListsToText()
    {
        MinjaChatTemplate chat = new(ChatMlTemplate);
        MinjaValue typed = MinjaValue.List([
            Map(("type", S("text")), ("text", S("hello"))),
            Map(("type", S("text")), ("text", S(" world"))),
        ]);

        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [new ChatMessage { Role = "user", Content = typed }],
            AddGenerationPrompt = false,
        });

        Assert.False(result.Capabilities.RequiresTypedContent);
        Assert.Contains("hello world", result.Prompt);
        Assert.DoesNotContain("{'type'", result.Prompt);
    }

    [Fact]
    public void PlainTemplate_LeavesNonTextTypedListUntouched()
    {
        // A dump-style template so an un-flattened typed list is observable instead of failing a string concatenation.
        MinjaChatTemplate chat = new("{% for message in messages %}{{ message['content'] | tojson }}{% endfor %}");
        MinjaValue typed = MinjaValue.List([Map(("type", S("image")), ("url", S("http://x")))]);

        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [new ChatMessage { Role = "user", Content = typed }],
            AddGenerationPrompt = false,
        });

        // A list with a non-text part is not flattened to plain text; it passes through unchanged and dumps as JSON.
        Assert.False(result.Capabilities.RequiresTypedContent);
        Assert.Contains("image", result.Prompt);
        Assert.Contains("http://x", result.Prompt);
    }

    // ---------------------------------------------------------------- system-role fallback

    [Fact]
    public void SystemRoleFallback_MergesSystemIntoFirstUserMessage()
    {
        MinjaChatTemplate chat = new(NoSystemTemplate);
        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("system", "Be terse."), ChatMessage.Text("user", "Hello")],
            AddGenerationPrompt = false,
        });

        Assert.False(result.Capabilities.SupportsSystemRole);
        Assert.Contains("Be terse.", result.Prompt);
        Assert.Contains("Hello", result.Prompt);
        Assert.DoesNotContain("<|im_start|>system", result.Prompt);
    }

    [Fact]
    public void SystemRoleFallback_InsertsUserMessageWhenNoUserExists()
    {
        MinjaChatTemplate chat = new(NoSystemTemplate);
        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("system", "Only system.")],
            AddGenerationPrompt = false,
        });

        Assert.Contains("Only system.", result.Prompt);
        Assert.Contains("<|im_start|>user", result.Prompt);
    }

    [Fact]
    public void ChatMl_SupportsSystemRoleSoNoMergeHappens()
    {
        MinjaChatTemplate chat = new(ChatMlTemplate);
        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("system", "Be terse."), ChatMessage.Text("user", "Hello")],
            AddGenerationPrompt = false,
        });

        Assert.True(result.Capabilities.SupportsSystemRole);
        Assert.Contains("<|im_start|>system\nBe terse.<|im_end|>", result.Prompt);
    }

    // ---------------------------------------------------------------- context polyfills

    [Fact]
    public void ExtraContextKeys_AreBoundDirectlyIntoTemplateScope()
    {
        const string template = "{{ custom_tools }}|{{ tools_in_user_message }}";
        MinjaChatTemplate chat = new(template);

        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "hi")],
            AddGenerationPrompt = false,
            ExtraContext = new Dictionary<string, MinjaValue>
            {
                ["custom_tools"] = MinjaValue.Of("custom"),
                ["tools_in_user_message"] = MinjaValue.True,
            },
        });

        Assert.Equal("custom|True", result.Prompt);
    }

    [Fact]
    public void StrftimeNow_RendersFixedNowWithCommonTokens()
    {
        const string template = "{{ strftime_now('%Y-%m-%d %H:%M:%S %b') }}";
        MinjaChatTemplate chat = new(template);

        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "hi")],
            AddGenerationPrompt = false,
            Now = new DateTimeOffset(2024, 1, 2, 3, 4, 5, TimeSpan.Zero),
        });

        Assert.Equal("2024-01-02 03:04:05 Jan", result.Prompt);
    }

    [Fact]
    public void DateString_IsBoundWhenSuppliedAndOmittedOtherwise()
    {
        const string template = "{% if date_string is defined %}{{ date_string }}{% else %}none{% endif %}";
        MinjaChatTemplate chat = new(template);

        ChatTemplateRenderResult supplied = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "hi")],
            AddGenerationPrompt = false,
            DateString = "Jan 02 2024",
        });
        Assert.Equal("Jan 02 2024", supplied.Prompt);

        ChatTemplateRenderResult omitted = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "hi")],
            AddGenerationPrompt = false,
        });
        Assert.Equal("none", omitted.Prompt);
    }

    // ---------------------------------------------------------------- options

    [Fact]
    public void NormalizeDisabled_RendersInputExactlyAsSupplied()
    {
        MinjaChatTemplate chat = new(NoSystemTemplate);
        ChatTemplateRenderResult result = chat.Render(
            new ChatTemplateInput
            {
                Messages = [ChatMessage.Text("system", "Be terse."), ChatMessage.Text("user", "Hello")],
                AddGenerationPrompt = false,
            },
            new ChatTemplateOptions { NormalizeMessages = false });

        // With normalization off the system message is dropped by the template instead of merged.
        Assert.DoesNotContain("Be terse.", result.Prompt);
    }

    [Fact]
    public void DetectCapabilitiesDisabled_ReportsNoCapabilitiesAndSkipsNormalization()
    {
        MinjaChatTemplate chat = new(NoSystemTemplate);
        ChatTemplateRenderResult result = chat.Render(
            new ChatTemplateInput
            {
                Messages = [ChatMessage.Text("system", "Be terse."), ChatMessage.Text("user", "Hello")],
                AddGenerationPrompt = false,
            },
            new ChatTemplateOptions { DetectCapabilities = false });

        Assert.False(result.Capabilities.SupportsSystemRole);
        Assert.DoesNotContain("Be terse.", result.Prompt);
    }

    // ---------------------------------------------------------------- Llama-family-shaped smoke

    [Fact]
    public void LlamaFamilyShapedTemplate_SmokeRendersWithToolsDateNamespaceAndIndent()
    {
        // A Llama-family-shaped template exercising the context variables and constructs the later parity item will
        // need: bos_token, custom_tools extra context, tools_in_user_message, strftime_now, date_string,
        // tojson(indent=4), inline list/map literals and namespace objects. Parity with the real Llama-3.2 template
        // is a later item; this only proves the wrapper renders such a template end to end.
        const string template =
            "{{ bos_token }}"
            + "{%- set date_string = date_string if date_string is defined else strftime_now('%b %d %Y') -%}"
            + "{%- set ns = namespace(has_images=false) -%}"
            + "{%- set custom_tools = custom_tools if custom_tools is defined else tools -%}"
            + "{%- if tools_in_user_message is not defined -%}{%- set tools_in_user_message = true -%}{%- endif -%}"
            + "{%- if custom_tools -%}"
            + "{{- 'Tools: ' + (custom_tools | map(attribute='name') | join(', ')) + '\n' }}"
            + "{%- endif -%}"
            + "{{- 'Date: ' + date_string + '\n' }}"
            + "{{- 'Now: ' + strftime_now('%H:%M:%S') + '\n' }}"
            + "{{- 'Flags: ' + ([1, 2] | tojson(indent=4)) + ' ' + ({'a': 1} | tojson) + '\n' }}"
            + "{%- if ns.has_images -%}{{- 'images\n' }}{%- endif -%}"
            + "{%- for message in messages -%}"
            + "{{- message['role'] + '\n' + message['content'] + '\n' }}"
            + "{%- endfor -%}"
            + "{{- eos_token }}"
            + "{%- if add_generation_prompt -%}{{- 'assistant\n' }}{%- endif -%}";

        MinjaChatTemplate chat = new(template, bosToken: "<|im_start|>", eosToken: "<|im_end|>");

        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "Hello")],
            ExtraContext = new Dictionary<string, MinjaValue>
            {
                ["custom_tools"] = MinjaValue.List([Map(("name", S("get_weather")))]),
                ["tools_in_user_message"] = MinjaValue.True,
            },
            Now = new DateTimeOffset(2024, 1, 2, 3, 4, 5, TimeSpan.Zero),
            AddGenerationPrompt = true,
        });

        Assert.True(result.StartsWithBos);
        Assert.Contains("Tools: get_weather", result.Prompt);
        Assert.Contains("Date: Jan 02 2024", result.Prompt);
        Assert.Contains("Now: 03:04:05", result.Prompt);
        // Inline list literal dumped with indent=4 and an inline map literal dumped compactly.
        Assert.Contains("[\n    1,\n    2\n]", result.Prompt);
        Assert.Contains("{\"a\": 1}", result.Prompt);
        Assert.Contains("<|im_start|>", result.Prompt);
        Assert.Contains("assistant\n", result.Prompt);
    }

    // ---------------------------------------------------------------- helpers

    private static MinjaValue S(string value) => MinjaValue.Of(value);

    private static MinjaValue Map(params (string Key, MinjaValue Value)[] entries) =>
        MinjaValue.Map(entries.Select(entry => new KeyValuePair<string, MinjaValue>(entry.Key, entry.Value)));

    private static MinjaValue Tool(string name)
    {
        MinjaValue parameters = Map(("type", S("object")));
        MinjaValue function = Map(
            ("name", S(name)),
            ("description", S("description")),
            ("parameters", parameters));

        return Map(
            ("type", S("function")),
            ("function", function));
    }
}
