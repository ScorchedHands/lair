using Lair.Http;
using Lair.Host.Config;
using Lair.Loader;
using Lair.Tokenizer;
using Xunit.Abstractions;

namespace Lair.Tests;

/// <summary>
/// Serve fixture for the real <c>models/gemma-2-2b-it.Q4_K_M.gguf</c>: the file opens through the production
/// load path, its SentencePiece vocabulary is resolved from <c>tokenizer.ggml.model</c> (not a default), and a
/// two-turn chat exchange runs end to end — a first user turn, its assistant reply appended to the conversation,
/// and a second user turn that refers back to that reply. The assertions are behavioral: each turn yields
/// non-empty visible text with an <c>eos</c> or <c>length</c> finish, and the second turn's rendered prompt is the
/// exact string the engine tokenizes and provably carries the first reply. The whole class skips cleanly when the
/// pinned model file is absent. One engine serves the class (a static lazy open under a lock) on the serialized
/// <c>model-loaded</c> lane, so no second engine loads alongside it.
/// </summary>
[Trait("Category", "Serve")]
[Collection("model-loaded")]
public sealed class Gemma2ServeTests
{
    private const string ModelRelativePath = "models/gemma-2-2b-it.Q4_K_M.gguf";
    private const string ModelName = "gemma2-serve";
    private const int MaxGeneratedTokens = 64;

    private static readonly object Gate = new();
    private static GenerationService? _engine;

    private readonly ITestOutputHelper _output;

    public Gemma2ServeTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task Serve_TwoTurnChat_CarriesTheFirstReplyIntoTheSecondPrompt()
    {
        GenerationService? engine = OpenEngine();
        if (engine is null)
        {
            return; // pinned model absent: nothing to serve
        }

        Assert.Equal("gemma2", engine.Model.Architecture);

        // The vocabulary implementation is the one the file's metadata selects: it declares the SentencePiece
        // marker and a pre value the BPE pre-split registry does not define, so the load resolving to SPM is the
        // selection seam — the engine would have failed at open on a BPE default.
        AssertVocabularyResolvedFromMetadata();

        GenerationRequest FirstTurnRequest() => new()
        {
            Model = ModelName,
            IsChat = true,
            RequestId = "req-gemma2-serve-first",
            Messages = [new Lair.Tokenizer.Templating.ChatMessage("user", "What is the capital of France? Reply with just the city name.")],
            MaxTokens = MaxGeneratedTokens,
            Temperature = 0f,
            Seed = 7,
        };

        GenerationResult first = await engine.GenerateAsync(FirstTurnRequest(), CancellationToken.None);

        _output.WriteLine($"serve turn 1: finish={first.FinishReason} prompt={first.PromptTokens} completion={first.CompletionTokens} text={Describe(first.Text)}");
        Assert.True(first.CompletionTokens > 0, "the first turn generated no tokens");
        Assert.Contains(first.FinishReason, new[] { "eos", "length" });
        AssertVisibleText(first.Text, "first turn");

        string firstReply = first.Text;

        // Turn two continues the conversation: the assistant reply from turn one is a real message in the history,
        // and the new user turn refers back to it instead of restating the answer.
        GenerationRequest SecondTurnRequest() => new()
        {
            Model = ModelName,
            IsChat = true,
            RequestId = "req-gemma2-serve-second",
            Messages =
            [
                new Lair.Tokenizer.Templating.ChatMessage("user", "What is the capital of France? Reply with just the city name."),
                new Lair.Tokenizer.Templating.ChatMessage("assistant", firstReply),
                new Lair.Tokenizer.Templating.ChatMessage("user", "What is the name of the city you just named? Reply with just the city name."),
            ],
            MaxTokens = MaxGeneratedTokens,
            Temperature = 0f,
            Seed = 11,
        };

        GenerationResult second = await engine.GenerateAsync(SecondTurnRequest(), CancellationToken.None);

        _output.WriteLine($"serve turn 2: finish={second.FinishReason} prompt={second.PromptTokens} completion={second.CompletionTokens} text={Describe(second.Text)}");
        Assert.True(second.CompletionTokens > 0, "the second turn generated no tokens");
        Assert.Contains(second.FinishReason, new[] { "eos", "length" });
        AssertVisibleText(second.Text, "second turn");

        // The rendered prompt the engine tokenizes for turn two is the observable hand-off: it carries turn one's
        // reply as the assistant message, and turn one's own prompt — which had no assistant reply — does not.
        string promptOne = engine.BuildChatPrompt(FirstTurnRequest(), CancellationToken.None);
        string promptTwo = engine.BuildChatPrompt(SecondTurnRequest(), CancellationToken.None);

        string carried = firstReply.Trim();
        Assert.True(carried.Length > 0, "turn one's reply is only whitespace, so there is nothing to carry");
        Assert.Contains(carried, promptTwo, StringComparison.Ordinal);
        Assert.DoesNotContain(carried, promptOne, StringComparison.Ordinal);
        Assert.True(promptTwo.Length > promptOne.Length, "the second prompt did not grow with the appended assistant reply");

        // Greedy decoding with a fixed seed repeats, so the exchange is reproducible run to run.
        GenerationResult repeat = await engine.GenerateAsync(SecondTurnRequest(), CancellationToken.None);
        Assert.Equal(second.Text, repeat.Text);
        Assert.Equal(second.FinishReason, repeat.FinishReason);
    }

    // ---------------------------------------------------------------- vocabulary selection seam

    private void AssertVocabularyResolvedFromMetadata()
    {
        string modelPath = ModelFixture.ResolveRepoRelative(ModelRelativePath)!;

        using GgufFile gguf = GgufFile.Open(modelPath);

        // The pinned file names the SentencePiece marker and a pre value the BPE registry rejects; the load that
        // produced this engine resolved SPM from those keys rather than falling back to the BPE default.
        Assert.Equal("llama", gguf.Metadata.GetString("tokenizer.ggml.model", ""));
        Assert.Equal("default", gguf.Metadata.GetString("tokenizer.ggml.pre", ""));

        // The special ids come from the file's own declarations, not from a profile default: the SentencePiece
        // path declares no fallback ids, so an unset bos would read back as -1. The served tokenizer carries the
        // declared ids, and its add_bos policy is the declared value.
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        Assert.Equal(gguf.Metadata.GetTokens().Length, tokenizer.VocabSize);
        Assert.True(tokenizer.AddBosToken);
        Assert.Equal(2, tokenizer.BosId);
        Assert.Equal(1, tokenizer.EosId);
        Assert.Equal(0, tokenizer.PadId);
        Assert.Equal(3, tokenizer.UnkId);
    }

    // ---------------------------------------------------------------- helpers

    private static void AssertVisibleText(string text, string turn)
    {
        Assert.False(string.IsNullOrEmpty(text), $"{turn} produced empty text");
        Assert.True(text.Trim().Length > 0, $"{turn} produced only whitespace");
        Assert.False(text.Any(c => char.IsControl(c) && c != '\n'), $"{turn} contains control characters");
    }

    private static string Describe(string text)
    {
        string preview = text.Length <= 160 ? text : text[..160] + "...";
        return preview.Replace("\n", "\\n");
    }

    // One engine for the whole class: opened lazily on first use, never disposed mid-run so a later case in this
    // serialized lane can reuse it. The process exit releases the weights.
    private GenerationService? OpenEngine()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(ModelRelativePath);
        if (modelPath is null)
        {
            _output.WriteLine($"SKIP Serve: {ModelRelativePath} not found walking up from the test assembly directory; the gemma2 serve fixture needs the pinned model file.");
            return null;
        }

        lock (Gate)
        {
            return _engine ??= GenerationService.Open(new LairConfig
            {
                Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
                Engine = new EngineConfig(),
                Models =
                [
                    new ModelConfig
                    {
                        Name = ModelName,
                        Path = modelPath,
                        MaxContextTokens = 1024,
                        KvBlocks = 64,
                    },
                ],
            });
        }
    }
}
