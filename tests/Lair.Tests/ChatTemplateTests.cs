using Lair.Loader;
using Lair.Templating.Minja;

namespace Lair.Tests;

/// <summary>
/// Golden renders for the chat template path: hand-verified exact output for ChatML, conditional, and Phi-style
/// templates through <see cref="MinjaChatTemplate"/> (the renderer the LAIR adapter now runs), the template
/// semantics the adapter inherits from minja (strict addition, Python json.dumps escaping, whitespace control,
/// index access), and a real-model integration render of tokenizer.chat_template from
/// models/SmolLM-135M-Instruct-F16.gguf when the fixture is present.
/// </summary>
[Trait("Category", "Template")]
public sealed class ChatTemplateTests
{
    private static readonly ChatMessage[] Conversation =
    [
        ChatMessage.Text("system", "You are helpful."),
        ChatMessage.Text("user", "Hello\nworld"),
    ];

    private const string ChatMlTemplate =
        """
        {% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
        """;

    private const string ConditionalTemplate =
        """
        {% for message in messages %}{% if message['role'] == 'system' %}[SYS]{{ message['content'] | tojson }}{% else %}[USER]{{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}GEN{% else %}NOGEN{% endif %}
        """;

    private const string PhiTemplate =
        """
        {% for message in messages %}{{ message['role'] + '\n' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant\n' }}{% endif %}
        """;

    // ---------------------------------------------------------------- goldens

    [Fact]
    public void ChatMl_WithGenerationPrompt_RendersExactChatML()
    {
        string rendered = Render(ChatMlTemplate, Conversation, addGenerationPrompt: true);

        Assert.Equal(
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            + "<|im_start|>user\nHello\nworld<|im_end|>\n"
            + "<|im_start|>assistant\n",
            rendered);
    }

    [Fact]
    public void ChatMl_WithoutGenerationPrompt_OmitsAssistantTurn()
    {
        string rendered = Render(ChatMlTemplate, Conversation, addGenerationPrompt: false);

        Assert.Equal(
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            + "<|im_start|>user\nHello\nworld<|im_end|>\n",
            rendered);
    }

    [Fact]
    public void Conditional_WithGenerationPrompt_TakesSystemBranchAndJsonEscapes()
    {
        string rendered = Render(ConditionalTemplate, Conversation, addGenerationPrompt: true);

        Assert.Equal("[SYS]\"You are helpful.\"[USER]Hello\nworldGEN", rendered);
    }

    [Fact]
    public void Conditional_WithoutGenerationPrompt_TakesElseBranches()
    {
        string rendered = Render(ConditionalTemplate, Conversation, addGenerationPrompt: false);

        Assert.Equal("[SYS]\"You are helpful.\"[USER]Hello\nworldNOGEN", rendered);
    }

    [Fact]
    public void PhiStyle_WithGenerationPrompt_RendersRoleContentLoop()
    {
        string rendered = Render(PhiTemplate, Conversation, addGenerationPrompt: true);

        Assert.Equal("system\nYou are helpful.\nuser\nHello\nworld\nassistant\n", rendered);
    }

    [Fact]
    public void PhiStyle_WithoutGenerationPrompt_OmitsTrailingRole()
    {
        string rendered = Render(PhiTemplate, Conversation, addGenerationPrompt: false);

        Assert.Equal("system\nYou are helpful.\nuser\nHello\nworld\n", rendered);
    }

    // ---------------------------------------------------------------- adapter-path semantics

    [Fact]
    public void Addition_StringWithInteger_FailsActionablyInsteadOfCoercing()
    {
        // The retired ChatTemplateRenderer coerced integers into concatenations; minja does not. A template that
        // mixes a string and an integer fails as an actionable render error naming the construct.
        MinjaRenderException error = Assert.Throws<MinjaRenderException>(
            () => Render("{{ 'n=' + 3 }}", Conversation, false));

        Assert.Equal("addition", error.Construct);
        Assert.Contains("cannot add", error.Message);
    }

    [Fact]
    public void Output_RendersBooleansAsPythonNames()
    {
        // The adapter's renderer takes Python-style capitalized literals; the lowercase spellings are the same
        // literals (minja and the pinned reference accept both), and values render with Python names.
        Assert.Equal(
            "True|False",
            Render("{{ True }}|{{ False }}", Conversation, false));
    }

    [Fact]
    public void If_CoversNotAndEquality()
    {
        Assert.Equal(
            "hidden",
            Render("{% if not add_generation_prompt %}hidden{% endif %}", Conversation, false));
        Assert.Equal(
            "ne",
            Render("{% if 'a' != 'b' %}ne{% else %}eq{% endif %}", Conversation, false));
    }

    [Fact]
    public void If_ElifChain_TakesRoleBranchesInOrder()
    {
        // Pinned with normalization off: a role-conditional template cannot surface the wrapper's dummy probe
        // content, so the capability probe reports system-role support as absent and the wrapper's fallback would
        // merge the system message into the user turn. This test pins the template branch semantics themselves;
        // the fallback behavior is pinned in the wrapper's own category.
        Assert.Equal(
            "SU",
            RenderExact(
                """
                {% for message in messages %}{% if message['role'] == 'system' %}S{% elif message['role'] == 'user' %}U{% else %}?{% endif %}{% endfor %}
                """, Conversation, false));
    }

    [Fact]
    public void Tojson_MatchesPythonJsonDumpsEscaping()
    {
        ChatMessage[] tricky = [ChatMessage.Text("system", "a\tb\"c\\d\n\u0001€\U0001F600<|x|>")];

        // minja's tojson mirrors Python json.dumps with its default separators and ensure_ascii escaping: control
        // characters and non-ASCII text (including the emoji's surrogate pair) land as \uXXXX escapes.
        Assert.Equal(
            """{"role": "system", "content": "a\tb\"c\\d\n\u0001\u20ac\ud83d\ude00<|x|>"}""",
            Render("{{ messages[0] | tojson }}", tricky, false));    }

    [Fact]
    public void Tojson_EncodesConstructedListsAndMaps()
    {
        Assert.Equal(
            """[1, "a", true, false]""",
            Render("{{ [1, 'a', True, False] | tojson }}", Conversation, false));
        Assert.Equal(
            """{"k": "v"}""",
            Render("{{ {'k': 'v'} | tojson }}", Conversation, false));
    }

    [Fact]
    public void TrimFilter_TrimsWhitespace()
    {
        Assert.Equal(
            "pad",
            Render("{{ '  pad  ' | trim }}", Conversation, false));
    }

    [Fact]
    public void WhitespaceControl_ElmDashTrimsAdjacentWhitespace()
    {
        Assert.Equal(
            "abc",
            Render("a\n{%- if True -%}\n  b\n{%- endif -%}\nc", Conversation, false));
        Assert.Equal(
            "xyesy",
            Render("x {%- if False -%} nope {%- else -%} yes {%- endif -%} y", Conversation, false));
    }

    [Fact]
    public void IndexAccess_NestsListThenMap()
    {
        // Pinned with normalization off for the same reason as the elif-chain test: the single-message probe cannot
        // demonstrate system-role rendering for a fixed-index template, so the wrapper fallback would reshape the
        // conversation before this index semantics test runs.
        Assert.Equal(
            "Hello\nworld",
            RenderExact("{{ messages[1]['content'] }}", Conversation, false));
    }

    // ---------------------------------------------------------------- real-model integration

    [Fact]
    public void RealModel_ChatTemplate_ParsesAndRendersTwoMessages()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        string? template = gguf.Metadata.ChatTemplate;
        Assert.NotNull(template);

        MinjaChatTemplate chat = new(template!);
        ChatMessage[] messages = [ChatMessage.Text("system", "You are a helpful assistant."), ChatMessage.Text("user", "Hello")];

        string withPrompt = chat.Render(new ChatTemplateInput { Messages = messages, AddGenerationPrompt = true }).Prompt;
        Assert.Contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", withPrompt);
        Assert.Contains("<|im_start|>user\nHello<|im_end|>\n", withPrompt);
        Assert.EndsWith("<|im_start|>assistant\n", withPrompt);

        string withoutPrompt = chat.Render(new ChatTemplateInput { Messages = messages, AddGenerationPrompt = false }).Prompt;
        Assert.Contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", withoutPrompt);
        Assert.Contains("<|im_start|>user\nHello<|im_end|>\n", withoutPrompt);
        Assert.DoesNotContain("<|im_start|>assistant\n", withoutPrompt);
    }

    // ---------------------------------------------------------------- helpers

    // Renders through the adapter path: the wrapper with its default options (capability detection and message
    // normalization on), exactly as GenerationService renders a chat request.
    private static string Render(string template, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt) =>
        new MinjaChatTemplate(template).Render(new ChatTemplateInput { Messages = messages, AddGenerationPrompt = addGenerationPrompt }).Prompt;

    // Renders the supplied messages exactly as given: template semantics pinned without the wrapper's capability fallbacks.
    private static string RenderExact(string template, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt) =>
        new MinjaChatTemplate(template).Render(
            new ChatTemplateInput { Messages = messages, AddGenerationPrompt = addGenerationPrompt },
            new ChatTemplateOptions { NormalizeMessages = false }).Prompt;
}
