using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Lair.Loader;
using Lair.Templating.Minja;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Byte-exact template parity against the pinned llama.cpp b12032 build: golden prompts captured from
/// <c>llama-server /apply-template</c> for the real Llama-3.2-1B-Instruct-Q4_K_M <c>tokenizer.chat_template</c>
/// (see scripts/generate-template-parity.ps1) are compared against LAIR's adapter-path render — the same
/// <see cref="MinjaChatTemplate"/> the loaded-model adapter runs, with the model's literal BOS/EOS bound and the
/// reference's captured <c>date_string</c> pinned so the render is clock-free. Guards: the model file's sha256
/// and the chat_template's sha256 must match the fixture. The whole category skips cleanly when the fixture or
/// the pinned model file is absent; the fixture itself is committed, so the comparisons never need llama.cpp.
/// Two documented divergences are pinned too: llama.cpp strips the template-emitted leading BOS from the
/// rendered text (it re-adds it at tokenization), so the goldens carry no BOS while LAIR's adapter render
/// carries it and suppresses the tokenizer's own BOS; and llama.cpp refuses a conversation that ends in an
/// assistant tool-call message, which LAIR renders.
/// </summary>
[Trait("Category", "TemplateParity")]
public sealed class TemplateParityTests
{
    private const string FixtureRelativePath = "tests/fixtures/parity-templates-llama32.json";

    private static readonly Lazy<ParityContext?> _context = new(BuildContext);

    // ---------------------------------------------------------------- guards

    [Fact]
    public void FixtureAndModel_GuardsConfirmPinnedReference()
    {
        ParityContext? context = Context();
        if (context is null)
        {
            return; // fixture or pinned model absent: nothing to guard against
        }

        Assert.Equal(context.Fixture.ModelSha256, Sha256File(context.ModelPath!));
        Assert.Equal(context.Fixture.ChatTemplateSha256, Sha256Text(context.Fixture.ChatTemplate));
        Assert.Equal(context.Fixture.BosToken, context.BosToken);
        Assert.Equal(context.Fixture.EosToken, context.EosToken);
    }

    // ---------------------------------------------------------------- byte-exact goldens

    [Fact]
    public void SysUser_WithGenerationPrompt_MatchesReferenceRender() => AssertCase("sys_user_gen");

    [Fact]
    public void SysUser_WithoutGenerationPrompt_MatchesReferenceRender() => AssertCase("sys_user_no_gen");

    [Fact]
    public void MultiTurn_WithGenerationPrompt_MatchesReferenceRender() => AssertCase("multi_turn_gen");

    [Fact]
    public void MultiTurn_WithoutGenerationPrompt_MatchesReferenceRender() => AssertCase("multi_turn_no_gen");

    [Fact]
    public void ToolCallWithResponse_WithGenerationPrompt_MatchesReferenceRender() => AssertCase("tool_call_with_response_gen");

    [Fact]
    public void ToolCallWithResponse_WithoutGenerationPrompt_MatchesReferenceRender() => AssertCase("tool_call_with_response_no_gen");

    [Fact]
    public void ToolResponse_WithGenerationPrompt_MatchesReferenceRender() => AssertCase("tool_response_gen");

    [Fact]
    public void ToolResponse_WithoutGenerationPrompt_MatchesReferenceRender() => AssertCase("tool_response_no_gen");

    [Fact]
    public void ToolsDeclared_WithGenerationPrompt_MatchesReferenceRender() => AssertCase("tools_declared_gen");

    [Fact]
    public void ToolsDeclared_WithoutGenerationPrompt_MatchesReferenceRender() => AssertCase("tools_declared_no_gen");

    [Fact]
    public void ToolsWithToolResponse_WithGenerationPrompt_MatchesReferenceRender() => AssertCase("tools_tool_response_gen");

    [Fact]
    public void ToolsWithToolResponse_WithoutGenerationPrompt_MatchesReferenceRender() => AssertCase("tools_tool_response_no_gen");

    // ---------------------------------------------------------------- documented reference refusal

    [Fact]
    public void TrailingToolCall_ReferenceRefusalIsDocumented_AndLairStillRenders()
    {
        ParityContext? context = Context();
        if (context is null)
        {
            return; // fixture or pinned model absent
        }

        foreach (string name in new[] { "tool_call_trailing_gen", "tool_call_trailing_no_gen" })
        {
            FixtureCase refusalCase = FindCase(context.Fixture, name);
            Assert.NotNull(refusalCase.ReferenceRefusal);
            Assert.Contains("Cannot continue an assistant message that contains tool calls", refusalCase.ReferenceRefusal);
        }

        // The divergence itself: llama.cpp's /apply-template refuses this message set, while LAIR's adapter path
        // renders it through the template's own tool-call branch. Pin that LAIR behavior against the exact
        // tool-call block the reference renders for the same call when a tool response follows it.
        FixtureCase trailing = FindCase(context.Fixture, "tool_call_trailing_gen");
        string prompt = RenderCase(context, trailing);
        Assert.Contains("{\"name\": \"calculator\", \"parameters\": {\"expression\": \"2+2\"}}<|eot_id|>", prompt);
        Assert.EndsWith("<|start_header_id|>assistant<|end_header_id|>\n\n", prompt);
    }

    // ---------------------------------------------------------------- helpers

    private static void AssertCase(string name)
    {
        ParityContext? context = Context();
        if (context is null)
        {
            return; // fixture or pinned model absent: no recorded outputs to compare against
        }

        FixtureCase entry = FindCase(context.Fixture, name);
        Assert.NotNull(entry.ExpectedPrompt);

        string prompt = RenderCase(context, entry);

        // llama.cpp strips a template-emitted leading BOS from the rendered text when the model's add_bos
        // policy is true and re-adds it at tokenization; LAIR keeps the BOS in the rendered string and
        // suppresses the tokenizer's own BOS instead. The token sequence is identical, so the comparison
        // adds the model BOS back onto the reference string.
        Assert.Equal(context.Fixture.BosToken + entry.ExpectedPrompt, prompt);
    }

    private static string RenderCase(ParityContext context, FixtureCase entry)
    {
        ChatTemplateInput input = new()
        {
            Messages = MapMessages(entry.Messages),
            AddGenerationPrompt = entry.AddGenerationPrompt,
            // The reference renders with its capture clock's date; pinning the captured date_string here makes
            // the adapter render deterministic without llama.cpp and without a live clock.
            DateString = context.Fixture.DateString,
            Tools = entry.Tools is JsonElement tools ? [.. tools.EnumerateArray().Select(JsonToMinja)] : null,
        };

        ChatTemplateRenderResult result = context.Chat.Render(input);

        Assert.True(result.StartsWithBos); // the adapter-bound BOS is in the rendered prompt

        return result.Prompt;
    }

    /// <summary>Maps the fixture's OpenAI-shaped message set onto the adapter's chat input records.</summary>
    private static IReadOnlyList<ChatMessage> MapMessages(JsonElement messages)
    {
        List<ChatMessage> mapped = [];

        foreach (JsonElement message in messages.EnumerateArray())
        {
            ChatMessage mappedMessage = new()
            {
                Role = message.GetProperty("role").GetString()!,
                Content = MinjaValue.Of(message.GetProperty("content").GetString() ?? string.Empty),
            };

            if (message.TryGetProperty("name", out JsonElement name))
            {
                mappedMessage = mappedMessage with { Name = name.GetString() };
            }

            if (message.TryGetProperty("tool_call_id", out JsonElement toolCallId))
            {
                mappedMessage = mappedMessage with { ToolCallId = toolCallId.GetString() };
            }

            if (message.TryGetProperty("tool_calls", out JsonElement toolCalls))
            {
                mappedMessage = mappedMessage with { ToolCalls = [.. toolCalls.EnumerateArray().Select(MapToolCall)] };
            }

            mapped.Add(mappedMessage);
        }

        return mapped;
    }

    /// <summary>
    /// Maps one OpenAI tool-call record onto the adapter's chat tool-call input. The arguments arrive as the
    /// OpenAI JSON string; llama.cpp parses them into a template value before rendering and the adapter's
    /// RequireObject normalization (<c>ParseJsonArguments</c>) does the same, so the parity input carries the
    /// parsed map.
    /// </summary>
    private static ChatToolCall MapToolCall(JsonElement call)
    {
        JsonElement function = call.GetProperty("function");

        return new ChatToolCall
        {
            Name = function.GetProperty("name").GetString()!,
            Arguments = JsonToMinja(function.GetProperty("arguments")),
            Id = call.TryGetProperty("id", out JsonElement id) ? id.GetString() : null,
        };
    }

    /// <summary>
    /// Maps a JSON element onto minja values preserving document order (objects become ordered maps). A JSON
    /// string that is itself a JSON document (the OpenAI tool-call arguments shape) is parsed one level deeper,
    /// mirroring how the reference and the adapter normalize tool-call arguments.
    /// </summary>
    private static MinjaValue JsonToMinja(JsonElement element)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                return MinjaValue.Map(element.EnumerateObject().Select(property =>
                    new KeyValuePair<string, MinjaValue>(property.Name, JsonToMinja(property.Value))));

            case JsonValueKind.Array:
                return MinjaValue.List(element.EnumerateArray().Select(JsonToMinja));

            case JsonValueKind.String:
                string text = element.GetString() ?? string.Empty;

                // Tool-call arguments ride as a JSON string carrying an object; the reference parses it into a
                // value before rendering, so the parity input does too.
                if (text.Length > 1 && (text[0] == '{' || text[0] == '['))
                {
                    try
                    {
                        using JsonDocument nested = JsonDocument.Parse(text);
                        return JsonToMinja(nested.RootElement);
                    }
                    catch (JsonException)
                    {
                        return MinjaValue.Of(text);
                    }
                }

                return MinjaValue.Of(text);

            case JsonValueKind.True:
                return MinjaValue.True;

            case JsonValueKind.False:
                return MinjaValue.False;

            case JsonValueKind.Null:
                return MinjaValue.None;

            case JsonValueKind.Number:
                return element.TryGetInt64(out long integer)
                    ? MinjaValue.Of(integer)
                    : MinjaValue.Of(element.GetDouble());

            default:
                return MinjaValue.None;
        }
    }

    private static ParityContext? Context() => _context.Value;

    private static ParityContext? BuildContext()
    {
        string? fixtureFile = ModelFixture.ResolveRepoRelative(FixtureRelativePath);
        if (fixtureFile is null)
        {
            return null; // the parity fixture is absent: nothing to compare against
        }

        ParityFixture fixture = LoadFixture(fixtureFile);

        string? modelPath = ModelFixture.ResolveRepoRelative(fixture.ModelFile);
        if (modelPath is null)
        {
            return null; // real-model cases skip when the pinned GGUF is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        string modelSha = Sha256File(modelPath);
        if (modelSha != fixture.ModelSha256)
        {
            throw new InvalidOperationException(
                $"template parity guard: {fixture.ModelFile} hashes to {modelSha} but the fixture was captured against {fixture.ModelSha256}; " +
                "regenerate the fixture with scripts/generate-template-parity.ps1 against the pinned model.");
        }

        using GgufFile gguf = GgufFile.Open(modelPath);

        string? template = gguf.Metadata.ChatTemplate
            ?? throw new InvalidOperationException($"model {modelPath} has no tokenizer.chat_template but the fixture expects one");

        if (Sha256Text(template) != fixture.ChatTemplateSha256)
        {
            throw new InvalidOperationException(
                $"template parity guard: the model's tokenizer.chat_template hashes to {Sha256Text(template)} but the fixture expects {fixture.ChatTemplateSha256}; " +
                "regenerate the fixture with scripts/generate-template-parity.ps1.");
        }

        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        // The same literal BOS/EOS pieces GenerationService binds into the template context.
        string bosToken = tokenizer.BosId >= 0 ? tokenizer.Decode([tokenizer.BosId], includeSpecial: true) : string.Empty;
        string eosToken = tokenizer.EosId >= 0 ? tokenizer.Decode([tokenizer.EosId], includeSpecial: true) : string.Empty;

        if (bosToken != fixture.BosToken || eosToken != fixture.EosToken)
        {
            throw new InvalidOperationException(
                $"template parity guard: the model's literal BOS/EOS pieces are '{bosToken}'/'{eosToken}' but the fixture recorded '{fixture.BosToken}'/'{fixture.EosToken}'.");
        }

        return new ParityContext(fixture, modelPath, new MinjaChatTemplate(template, bosToken, eosToken), bosToken, eosToken);
    }

    private static FixtureCase FindCase(ParityFixture fixture, string name)
    {
        foreach (FixtureCase entry in fixture.Cases)
        {
            if (entry.Name == name)
            {
                return entry;
            }
        }

        throw new InvalidOperationException($"the parity fixture has no case named '{name}'");
    }

    private static ParityFixture LoadFixture(string path)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;

        List<FixtureCase> cases = [];
        foreach (JsonElement entry in root.GetProperty("cases").EnumerateArray())
        {
            cases.Add(new FixtureCase(
                entry.GetProperty("name").GetString()!,
                entry.GetProperty("messages").Clone(),
                entry.GetProperty("add_generation_prompt").GetBoolean(),
                entry.TryGetProperty("expected_prompt", out JsonElement prompt) ? prompt.GetString() : null,
                entry.TryGetProperty("reference_refusal", out JsonElement refusal) ? refusal.GetString() : null,
                entry.TryGetProperty("tools", out JsonElement tools) ? tools.Clone() : null));
        }

        return new ParityFixture(
            root.GetProperty("model_file").GetString()!,
            root.GetProperty("model_sha256").GetString()!,
            root.GetProperty("chat_template").GetString()!,
            root.GetProperty("chat_template_sha256").GetString()!,
            root.GetProperty("bos_token").GetString()!,
            root.GetProperty("eos_token").GetString()!,
            root.GetProperty("date_string").GetString()!,
            cases);
    }

    private static string Sha256File(string path)
    {
        using SHA256 hasher = SHA256.Create();
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(hasher.ComputeHash(stream)).ToLowerInvariant();
    }

    private static string Sha256Text(string text)
    {
        using SHA256 hasher = SHA256.Create();
        return Convert.ToHexString(hasher.ComputeHash(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();
    }

    private sealed record ParityFixture(
        string ModelFile,
        string ModelSha256,
        string ChatTemplate,
        string ChatTemplateSha256,
        string BosToken,
        string EosToken,
        string DateString,
        IReadOnlyList<FixtureCase> Cases);

    private sealed record FixtureCase(
        string Name,
        JsonElement Messages,
        bool AddGenerationPrompt,
        string? ExpectedPrompt,
        string? ReferenceRefusal,
        JsonElement? Tools);

    private sealed record ParityContext(
        ParityFixture Fixture,
        string? ModelPath,
        MinjaChatTemplate Chat,
        string BosToken,
        string EosToken);
}
