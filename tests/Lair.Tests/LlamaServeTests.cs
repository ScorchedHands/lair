using System.Reflection;
using System.Text.Json;
using Lair.Http;
using Lair.Host.Config;
using Lair.Templating.Minja;
using Xunit.Abstractions;

namespace Lair.Tests;

/// <summary>
/// End-to-end serve fixture for the real <c>models/Llama-3.2-1B-Instruct-Q4_K_M.gguf</c>: the GGUF is opened
/// through the production load path (<see cref="GenerationService.Open"/>), a multi-turn conversation is
/// rendered through the model's own <c>tokenizer.chat_template</c> by the adapter's load-time parse, the
/// rendered prompt is byte-compared against the pinned llama.cpp parity golden, and a greedy continuation
/// produces a finite, deterministic first-token continuation with finite per-step logit distributions.
/// The whole category skips cleanly when the pinned model file is absent. Memory budget: exactly one model
/// instance for the whole collection (one <see cref="LlamaServeFixture"/> engine, collection-serialized
/// tests), a modest context/KV pool, and at most 32 generated tokens per request, so the resident set is the
/// pinned weights plus a small pool and nothing else.
/// </summary>
[CollectionDefinition("Llama Serve")]
public sealed class LlamaServeCollection : ICollectionFixture<LlamaServeFixture>;

/// <summary>
/// The single loaded Llama-3.2-1B engine shared by every test in the collection. The open is lazy so the
/// category skips cleanly when the pinned model file is absent, and the engine is disposed once when the
/// collection finishes — no second copy of the 770 MB model is ever resident.
/// </summary>
public sealed class LlamaServeFixture : IDisposable
{
    private const string ModelRelativePath = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
    private const string ModelName = "llama-3.2-serve";

    private GenerationService? _engine;

    public string ModelPath => ModelFixture.ResolveRepoRelative(ModelRelativePath) ?? throw new InvalidOperationException(
        $"{ModelRelativePath} was not found walking up from the test assembly directory.");

    public string ModelId => ModelName;

    /// <summary>Opens the engine once on first use; null when the pinned model file is absent.</summary>
    public GenerationService? OpenEngine(ITestOutputHelper? output = null)
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(ModelRelativePath);
        if (modelPath is null)
        {
            output?.WriteLine($"SKIP Serve: {ModelRelativePath} not found walking up from the test assembly directory; the llama serve fixture needs the pinned model file.");
            return null;
        }

        lock (Gate)
        {
            if (_engine is not null)
            {
                return _engine;
            }

            // Modest pool: 2 048-token context and 64 KV blocks (about 32 MB of KV for this model) alongside
            // the 770 MB Q4_K_M weights.
            _engine = GenerationService.Open(new LairConfig
            {
                Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
                Engine = new EngineConfig(),
                Models =
                [
                    new ModelConfig
                    {
                        Name = ModelName,
                        Path = modelPath,
                        MaxContextTokens = 2048,
                        KvBlocks = 64,
                    },
                ],
            });

            return _engine;
        }
    }

    private static readonly object Gate = new();

    public void Dispose()
    {
        lock (Gate)
        {
            _engine?.Dispose();
            _engine = null;
        }
    }
}

[Trait("Category", "Serve")]
[Collection("Llama Serve")]
public sealed class LlamaServeTests
{
    private const string ParityFixtureRelativePath = "tests/fixtures/parity-templates-llama32.json";
    private const int MaxGeneratedTokens = 32;

    private readonly LlamaServeFixture _fixture;

    public LlamaServeTests(LlamaServeFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        Output = output;
    }

    private ITestOutputHelper Output { get; }

    // ---------------------------------------------------------------- load + render + parity golden

    [Fact]
    public void Serve_LoadAndRender_MatchesPinnedParityPrompt()
    {
        ParityContext? parity = ResolveParity();
        if (parity is null)
        {
            return; // model or parity fixture absent: nothing to compare against
        }

        GenerationService engine = _fixture.OpenEngine(Output) ?? throw new InvalidOperationException("engine open returned null");

        // The load path produced the adapter facts for the model's own template: the {% set %}/{% for %}
        // plumbing parsed and the capability probe ran at load.
        Assert.NotNull(engine.ChatTemplateCapabilities);
        Assert.Equal("llama", engine.Model.Architecture);

        // The adapter renders the multi-turn conversation with the model's own chat template. Pinning the
        // captured date_string (the parity fixture's clock) makes the render clock-free and byte-comparable.
        MinjaChatTemplate chat = EngineChatTemplate(engine);
        ChatTemplateRenderResult render = chat.Render(
            new ChatTemplateInput
            {
                Messages = parity.MultiTurnMessages,
                AddGenerationPrompt = true,
                DateString = parity.DateString,
            });

        Assert.True(render.StartsWithBos, "the model's own template did not emit its BOS token in the rendered prompt");
        Assert.Equal(parity.BosToken + parity.ExpectedPrompt, render.Prompt);

        // BOS accounting guard: the rendered prompt carries exactly one leading BOS — the template's own —
        // so the adapter suppresses the tokenizer's duplicate instead of stacking a second one.
        Assert.StartsWith(parity.BosToken, render.Prompt);
        Assert.False(render.Prompt.StartsWith(parity.BosToken + parity.BosToken), "the rendered prompt carries a doubled BOS");
    }

    // ---------------------------------------------------------------- greedy continuation

    [Fact]
    public async Task Serve_GreedyContinuation_IsFiniteDeterministicAndCoherent()
    {
        ParityContext? parity = ResolveParity();
        if (parity is null)
        {
            return; // model absent: nothing to continue
        }

        GenerationService engine = _fixture.OpenEngine(Output) ?? throw new InvalidOperationException("engine open returned null");

        GenerationRequest Request() => new()
        {
            Model = _fixture.ModelId,
            IsChat = true,
            RequestId = "req-serve-continuation",
            Messages = [.. parity.MultiTurnMessages.Select(m => new Lair.Tokenizer.Templating.ChatMessage(m.Role, m.Content.ToString()))],
            MaxTokens = MaxGeneratedTokens,
            Temperature = 0f,
            Seed = 7,
            LogProbs = true,
            TopLogprobs = 3,
        };

        GenerationResult first = await engine.GenerateAsync(Request(), CancellationToken.None);
        GenerationResult second = await engine.GenerateAsync(Request(), CancellationToken.None);

        // Finite: the run stopped within its token budget and the per-step distributions are finite numbers.
        Assert.True(first.CompletionTokens > 0, "the continuation generated no tokens");
        Assert.True(first.CompletionTokens <= MaxGeneratedTokens, $"the continuation generated {first.CompletionTokens} tokens over the {MaxGeneratedTokens}-token budget");
        Assert.Contains(first.FinishReason, new[] { "eos", "length", "stop" });
        Assert.NotNull(first.LogProbs);
        Assert.Equal(first.CompletionTokens, first.LogProbs.Count);
        foreach (Lair.Models.Scheduling.TokenLogprob entry in first.LogProbs)
        {
            Assert.True(double.IsFinite(entry.LogProb), $"logprob for token {entry.TokenId} is not finite: {entry.LogProb}");
            foreach ((int _, double topLogProb) in entry.TopLogProbs)
            {
                Assert.True(double.IsFinite(topLogProb), $"top logprob for token {entry.TokenId} is not finite: {topLogProb}");
            }
        }

        // Deterministic: greedy decoding with a fixed seed repeats byte-for-byte, including the first token.
        Assert.Equal(first.Text, second.Text);
        Assert.Equal(first.CompletionTokens, second.CompletionTokens);
        Assert.Equal(first.PromptTokens, second.PromptTokens);
        Assert.Equal(first.FinishReason, second.FinishReason);

        // Coherent: real vocabulary text, no control-character spam, no newline runs.
        Assert.False(string.IsNullOrEmpty(first.Text), "the continuation text is empty");
        Assert.True(first.Text.Trim().Length > 0, "the continuation is only whitespace");
        Assert.False(first.Text.Contains("<|", StringComparison.Ordinal), "special-token markup leaked into the decoded text");
        Assert.DoesNotContain("\n\n\n", first.Text);
        Assert.False(first.Text.Any(c => char.IsControl(c) && c != '\n'), "the continuation contains control characters");

        Output.WriteLine($"continuation: finish={first.FinishReason} promptTokens={first.PromptTokens} completionTokens={first.CompletionTokens}");
        Output.WriteLine($"first tokens: {Describe(first.Text)}");
    }

    // ---------------------------------------------------------------- helpers

    /// <summary>
    /// The parse-once chat template the loaded engine actually renders with — the same reflection seam the
    /// diagnostics suite uses for the load-time parse — so this test compares the adapter's own template, not a
    /// second copy built in the test.
    /// </summary>
    private static MinjaChatTemplate EngineChatTemplate(GenerationService engine)
    {
        FieldInfo templateField = typeof(GenerationService).GetField("_chatTemplate", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("the loaded model no longer caches its parsed chat template");

        return (MinjaChatTemplate)(templateField.GetValue(engine)
            ?? throw new InvalidOperationException("the served model loaded without a chat template"));
    }

    private ParityContext? ResolveParity()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative("models/Llama-3.2-1B-Instruct-Q4_K_M.gguf");
        if (modelPath is null)
        {
            Output.WriteLine("SKIP Serve: models/Llama-3.2-1B-Instruct-Q4_K_M.gguf not found walking up from the test assembly directory; the llama serve fixture needs the pinned model file.");
            return null;
        }

        string? fixtureFile = ModelFixture.ResolveRepoRelative(ParityFixtureRelativePath);
        if (fixtureFile is null)
        {
            Output.WriteLine($"SKIP Serve: {ParityFixtureRelativePath} not found; the parity comparison has no pinned reference.");
            return null;
        }

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(fixtureFile));
        JsonElement root = document.RootElement;

        string dateString = root.GetProperty("date_string").GetString()!;
        string bosToken = root.GetProperty("bos_token").GetString()!;
        string chatTemplate = root.GetProperty("chat_template").GetString()!;

        JsonElement caseElement = default;
        foreach (JsonElement entry in root.GetProperty("cases").EnumerateArray())
        {
            if (entry.GetProperty("name").GetString() == "multi_turn_gen")
            {
                caseElement = entry;
                break;
            }
        }

        if (caseElement.ValueKind == JsonValueKind.Undefined)
        {
            throw new InvalidOperationException($"the parity fixture has no 'multi_turn_gen' case in {ParityFixtureRelativePath}");
        }

        List<ChatMessage> messages = [];
        foreach (JsonElement message in caseElement.GetProperty("messages").EnumerateArray())
        {
            messages.Add(new ChatMessage
            {
                Role = message.GetProperty("role").GetString()!,
                Content = MinjaValue.Of(message.GetProperty("content").GetString() ?? string.Empty),
            });
        }

        string expectedPrompt = caseElement.GetProperty("expected_prompt").GetString()!;

        // Guard the pinned reference against a swapped model file before comparing anything.
        string modelSha = Sha256File(modelPath);
        if (modelSha != root.GetProperty("model_sha256").GetString())
        {
            throw new InvalidOperationException(
                $"serve parity guard: models/Llama-3.2-1B-Instruct-Q4_K_M.gguf hashes to {modelSha} but the fixture was captured against {root.GetProperty("model_sha256").GetString()}; " +
                "regenerate the fixture with scripts/generate-template-parity.ps1 against the pinned model.");
        }

        using (GgufFileGuard gguf = GgufFileGuard.Open(modelPath))
        {
            if (gguf.ChatTemplate != chatTemplate)
            {
                throw new InvalidOperationException(
                    "serve parity guard: the model's tokenizer.chat_template is not the template the fixture's goldens were captured against");
            }
        }

        return new ParityContext(messages, expectedPrompt, dateString, bosToken);
    }

    private static string Describe(string text)
    {
        string preview = text.Length <= 160 ? text : text[..160] + "...";
        return preview.Replace("\n", "\\n");
    }

    private static string Sha256File(string path)
    {
        using System.Security.Cryptography.SHA256 hasher = System.Security.Cryptography.SHA256.Create();
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(hasher.ComputeHash(stream)).ToLowerInvariant();
    }

    // Opens the GGUF only to read the chat_template metadata guard; disposed immediately.
    private sealed class GgufFileGuard : IDisposable
    {
        private readonly Lair.Loader.GgufFile _file;

        private GgufFileGuard(Lair.Loader.GgufFile file) => _file = file;

        public static GgufFileGuard Open(string path) => new(Lair.Loader.GgufFile.Open(path));

        public string? ChatTemplate => _file.Metadata.ChatTemplate;

        public void Dispose() => _file.Dispose();
    }

    private sealed record ParityContext(
        IReadOnlyList<ChatMessage> MultiTurnMessages,
        string ExpectedPrompt,
        string DateString,
        string BosToken);
}
