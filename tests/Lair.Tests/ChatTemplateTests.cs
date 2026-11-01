using Lair.Loader;
using Lair.Tokenizer.Templating;

namespace Lair.Tests;

/// <summary>
/// Golden renders for the supported Mini Jinja subset: hand-verified exact output for ChatML, conditional, and
/// Phi-style templates, rejection of constructs outside the subset, and a real-model integration render of
/// tokenizer.chat_template from models/SmolLM-135M-Instruct-F16.gguf when the fixture is present.
/// </summary>
[Trait("Category", "Template")]
public sealed class ChatTemplateTests
{
    private static readonly ChatMessage[] Conversation =
    [
        new("system", "You are helpful."),
        new("user", "Hello\nworld"),
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
        string rendered = ChatTemplateRenderer.Parse(ChatMlTemplate).Render(Conversation, addGenerationPrompt: true);

        Assert.Equal(
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            + "<|im_start|>user\nHello\nworld<|im_end|>\n"
            + "<|im_start|>assistant\n",
            rendered);
    }

    [Fact]
    public void ChatMl_WithoutGenerationPrompt_OmitsAssistantTurn()
    {
        string rendered = ChatTemplateRenderer.Parse(ChatMlTemplate).Render(Conversation, addGenerationPrompt: false);

        Assert.Equal(
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            + "<|im_start|>user\nHello\nworld<|im_end|>\n",
            rendered);
    }

    [Fact]
    public void Conditional_WithGenerationPrompt_TakesSystemBranchAndJsonEscapes()
    {
        string rendered = ChatTemplateRenderer.Parse(ConditionalTemplate).Render(Conversation, addGenerationPrompt: true);

        Assert.Equal("[SYS]\"You are helpful.\"[USER]Hello\nworldGEN", rendered);
    }

    [Fact]
    public void Conditional_WithoutGenerationPrompt_TakesElseBranches()
    {
        string rendered = ChatTemplateRenderer.Parse(ConditionalTemplate).Render(Conversation, addGenerationPrompt: false);

        Assert.Equal("[SYS]\"You are helpful.\"[USER]Hello\nworldNOGEN", rendered);
    }

    [Fact]
    public void PhiStyle_WithGenerationPrompt_RendersRoleContentLoop()
    {
        string rendered = ChatTemplateRenderer.Parse(PhiTemplate).Render(Conversation, addGenerationPrompt: true);

        Assert.Equal("system\nYou are helpful.\nuser\nHello\nworld\nassistant\n", rendered);
    }

    [Fact]
    public void PhiStyle_WithoutGenerationPrompt_OmitsTrailingRole()
    {
        string rendered = ChatTemplateRenderer.Parse(PhiTemplate).Render(Conversation, addGenerationPrompt: false);

        Assert.Equal("system\nYou are helpful.\nuser\nHello\nworld\n", rendered);
    }

    // ---------------------------------------------------------------- unsupported constructs

    [Fact]
    public void SetTag_ThrowsNamingConstructAndPosition()
    {
        TemplateSyntaxException ex = Assert.Throws<TemplateSyntaxException>(
            () => ChatTemplateRenderer.Parse("{% set loop_messages = messages %}"));

        Assert.Contains("'{% set %}'", ex.Message);
        Assert.Equal(3, ex.Position);
    }

    [Fact]
    public void UnknownFilter_ThrowsNamingFilterAndPosition()
    {
        TemplateSyntaxException ex = Assert.Throws<TemplateSyntaxException>(
            () => ChatTemplateRenderer.Parse("{{ 'shout' | upper }}"));

        Assert.Contains("'upper'", ex.Message);
        Assert.Equal(13, ex.Position);
    }

    [Fact]
    public void MacroTag_ThrowsNamingConstructAndPosition()
    {
        TemplateSyntaxException ex = Assert.Throws<TemplateSyntaxException>(
            () => ChatTemplateRenderer.Parse("{% macro render(msg) %}{{ msg }}{% endmacro %}"));

        Assert.Contains("'{% macro %}'", ex.Message);
        Assert.Equal(3, ex.Position);
    }

    [Fact]
    public void SliceSyntax_ThrowsNamingConstructAndPosition()
    {
        TemplateSyntaxException ex = Assert.Throws<TemplateSyntaxException>(
            () => ChatTemplateRenderer.Parse("{{ messages[0:2] }}"));

        Assert.Contains("'slice'", ex.Message);
        Assert.Equal(13, ex.Position);
    }

    [Fact]
    public void UnclosedFor_ThrowsAtKeywordPosition()
    {
        TemplateSyntaxException ex = Assert.Throws<TemplateSyntaxException>(
            () => ChatTemplateRenderer.Parse("{% for message in messages %}{{ message['role'] }}"));

        Assert.Contains("unclosed '{% for %}'", ex.Message);
        Assert.Equal(3, ex.Position);
    }

    // ---------------------------------------------------------------- subset semantics

    [Fact]
    public void Concat_CoercesIntegersToText()
    {
        Assert.Equal("n=3", ChatTemplateRenderer.Parse("{{ 'n=' + 3 }}").Render(Conversation, false));
    }

    [Fact]
    public void Output_RendersBooleansAsPythonNames()
    {
        Assert.Equal(
            "True|False",
            ChatTemplateRenderer.Parse("{{ true }}|{{ False }}").Render(Conversation, false));
    }

    [Fact]
    public void If_CoversNotEqualityAndElifChain()
    {
        Assert.Equal(
            "hidden",
            ChatTemplateRenderer.Parse("{% if not add_generation_prompt %}hidden{% endif %}").Render(Conversation, false));
        Assert.Equal(
            "ne",
            ChatTemplateRenderer.Parse("{% if 'a' != 'b' %}ne{% else %}eq{% endif %}").Render(Conversation, false));
        Assert.Equal(
            "SU",
            ChatTemplateRenderer.Parse(
                """
                {% for message in messages %}{% if message['role'] == 'system' %}S{% elif message['role'] == 'user' %}U{% else %}?{% endif %}{% endfor %}
                """).Render(Conversation, false));
    }

    [Fact]
    public void Tojson_MatchesPythonJsonDumpsEscaping()
    {
        ChatMessage[] tricky = [new("system", "a\tb\"c\\d\n€\U0001F600<|x|>")];

        Assert.Equal(
            """{"role":"system","content":"a\tb\"c\\d\n\u0001€😀<|x|>"}""",
            ChatTemplateRenderer.Parse("{{ messages[0] | tojson }}").Render(tricky, false));
    }

    [Fact]
    public void Tojson_EncodesConstructedListsAndMaps()
    {
        Assert.Equal(
            """[1,"a",true,false]""",
            ChatTemplateRenderer.Parse("{{ [1, 'a', true, false] | tojson }}").Render(Conversation, false));
        Assert.Equal(
            """{"k":"v"}""",
            ChatTemplateRenderer.Parse("{{ {'k': 'v'} | tojson }}").Render(Conversation, false));
    }

    [Fact]
    public void TrimFilter_TrimsWhitespace()
    {
        Assert.Equal(
            "pad",
            ChatTemplateRenderer.Parse("{{ '  pad  ' | trim }}").Render(Conversation, false));
    }

    [Fact]
    public void WhitespaceControl_ElmDashTrimsAdjacentWhitespace()
    {
        Assert.Equal(
            "abc",
            ChatTemplateRenderer.Parse("a\n{%- if true -%}\n  b\n{%- endif -%}\nc").Render(Conversation, false));
        Assert.Equal(
            "xyesy",
            ChatTemplateRenderer.Parse("x {%- if false -%} nope {%- else -%} yes {%- endif -%} y").Render(Conversation, false));
    }

    [Fact]
    public void IndexAccess_NestsListThenMap()
    {
        Assert.Equal(
            "Hello\nworld",
            ChatTemplateRenderer.Parse("{{ messages[1]['content'] }}").Render(Conversation, false));
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

        ChatTemplateRenderer renderer = ChatTemplateRenderer.Parse(template!);
        ChatMessage[] messages = [new("system", "You are a helpful assistant."), new("user", "Hello")];

        string withPrompt = renderer.Render(messages, addGenerationPrompt: true);
        Assert.Contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", withPrompt);
        Assert.Contains("<|im_start|>user\nHello<|im_end|>\n", withPrompt);
        Assert.EndsWith("<|im_start|>assistant\n", withPrompt);

        string withoutPrompt = renderer.Render(messages, addGenerationPrompt: false);
        Assert.Contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", withoutPrompt);
        Assert.Contains("<|im_start|>user\nHello<|im_end|>\n", withoutPrompt);
        Assert.DoesNotContain("<|im_start|>assistant\n", withoutPrompt);
    }
}
