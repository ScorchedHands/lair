using Lair.Http;
using Lair.Host.Config;
using Xunit.Abstractions;

namespace Lair.Tests;

/// <summary>
/// Serve fixture for reasoning visibility on the real <c>models/Qwen3-0.6B-Q8_0.gguf</c>: the file opens through the
/// production load path, one greedy chat request is generated under the default <c>keep</c> policy so its emitted
/// <c>&lt;think&gt;</c> … <c>&lt;/think&gt;</c> region reaches visible content as generated, and a live
/// <see cref="GenerationService.ApplyConfiguration"/> flip to <c>strip</c> changes what the next request carries while
/// leaving decoding itself untouched — same prompt, same seed, so token counts and the finish reason repeat exactly and
/// only the reasoning region vanishes from visible content. The whole class skips cleanly when the pinned model file is
/// absent. One engine serves every case in this class (a static lazy open under a lock): two copies of the weights would
/// buy nothing, and the class shares the serialized <c>model-loaded</c> lane so no second engine loads alongside it.
/// </summary>
[Trait("Category", "Serve")]
[Collection("model-loaded")]
public sealed class Qwen3ReasoningServeTests
{
    private const string ModelRelativePath = "models/Qwen3-0.6B-Q8_0.gguf";
    private const string ModelName = "qwen3-reasoning-serve";
    private const int MaxGeneratedTokens = 320; // enough budget for this model to close its thinking region and answer

    private static readonly object Gate = new();
    private static GenerationService? _engine;

    private readonly ITestOutputHelper _output;

    public Qwen3ReasoningServeTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task Serve_KeepPolicy_EmitsTheRecognizedRegionInVisibleContent_AndRecordsItsBoundaries()
    {
        GenerationService? engine = OpenEngine();
        if (engine is null)
        {
            return; // pinned model absent: nothing to serve
        }

        Assert.Equal(ReasoningPolicy.Keep, engine.ResolvedOutputParser.Reasoning);
        Assert.Equal("keep", engine.Model.Reasoning);

        GenerationResult kept = await engine.GenerateAsync(Request(), CancellationToken.None);

        _output.WriteLine($"serve keep: finish={kept.FinishReason} completion={kept.CompletionTokens} regions={kept.ReasoningRegions?.Count ?? 0}");
        Assert.True(kept.CompletionTokens > 0, "the continuation generated no tokens");
        Assert.Contains(kept.FinishReason, new[] { "eos", "length", "stop" });

        // The model's own thinking markup is what recognition keys on: this fixture emits it, so the region is recognized.
        ReasoningRegion region = Assert.Single(kept.ReasoningRegions!);
        string span = kept.Text[region.Start..region.End];
        Assert.StartsWith("<think>", span);

        // Under keep nothing leaves: visible content still carries both boundaries exactly as generated.
        Assert.Contains("<think>", kept.Text, StringComparison.Ordinal);
        Assert.Contains("</think>", kept.Text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Serve_StripFlipWithholdsTheRegionOnNextRequest_WithoutChangingDecoding()
    {
        GenerationService? engine = OpenEngine();
        if (engine is null)
        {
            return; // pinned model absent: nothing to serve
        }

        GenerationResult kept = await engine.GenerateAsync(Request(), CancellationToken.None);
        Assert.NotNull(kept.ReasoningRegions);

        IReadOnlyList<FieldResult> applied = engine.ApplyConfiguration(ConfigFor("strip"));
        FieldResult result = Assert.Single(applied);
        Assert.Equal("models[0].reasoning", result.Change.Field);
        Assert.Equal(ConfigApplyOutcome.Applied, result.Outcome);
        _output.WriteLine($"serve apply: {result.Detail}");
        Assert.Equal("strip", engine.Model.Reasoning);

        GenerationResult stripped = await engine.GenerateAsync(Request(), CancellationToken.None);

        // Decoding is untouched by the policy: identical prompt and seed repeat token-for-token, so only visibility differs.
        Assert.Equal(kept.PromptTokens, stripped.PromptTokens);
        Assert.Equal(kept.CompletionTokens, stripped.CompletionTokens);
        Assert.Equal(kept.FinishReason, stripped.FinishReason);

        // The region is gone from visible content while its boundaries stay recorded.
        Assert.DoesNotContain("<think>", stripped.Text, StringComparison.Ordinal);
        Assert.DoesNotContain("</think>", stripped.Text, StringComparison.Ordinal);
        ReasoningRegion region = Assert.Single(stripped.ReasoningRegions!);
        Assert.True(region.End > region.Start, "the recorded region is empty");

        string visibleSpan = kept.Text[region.Start..region.End];
        Assert.Contains("<think>", visibleSpan, StringComparison.Ordinal); // the withheld span is exactly what keep put on the wire
        Assert.NotEqual(kept.Text, stripped.Text);
    }

    private static GenerationRequest Request() => new()
    {
        Model = ModelName,
        IsChat = true,
        RequestId = "req-reasoning-serve",
        Messages = [new Lair.Tokenizer.Templating.ChatMessage("user", "What is the capital of France? Explain your reasoning, then answer.")],
        MaxTokens = MaxGeneratedTokens,
        Temperature = 0f,
        Seed = 11,
    };

    private static LairConfig ConfigFor(string reasoning) => new()
    {
        Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
        Engine = new EngineConfig(),
        Models =
        [
            new ModelConfig
            {
                Name = ModelName,
                Path = ModelFixture.ResolveRepoRelative(ModelRelativePath)!,
                MaxContextTokens = 1024,
                KvBlocks = 64,
                Reasoning = reasoning,
            },
        ],
    };

    // One engine for the whole class: opened lazily on first use, never disposed mid-run so a later case in this
    // serialized lane can reuse it. The process exit releases the weights.
    private GenerationService? OpenEngine()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(ModelRelativePath);
        if (modelPath is null)
        {
            _output.WriteLine($"SKIP Serve: {ModelRelativePath} not found walking up from the test assembly directory; the reasoning serve fixture needs the pinned model file.");
            return null;
        }

        lock (Gate)
        {
            return _engine ??= GenerationService.Open(ConfigFor("keep"));
        }
    }
}
