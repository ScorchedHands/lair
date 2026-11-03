using Lair.Http;
using Lair.Host.Config;
using Lair.Tokenizer.Templating;

namespace Lair.Tests;

/// <summary>
/// Generation facade tests: exception messages and payloads, request defaults and sampling-knob overrides,
/// model-info shape, and the Open guards that fail before any model file is touched. No GGUF fixture is loaded.
/// </summary>
[Trait("Category", "Generation")]
public class GenerationServiceTests
{
    [Fact]
    public void ModelNotFoundException_ReportsRequestedId_AndActionableMessage()
    {
        ModelNotFoundException ex = new("mistral-7b");

        Assert.Equal("mistral-7b", ex.Requested);
        Assert.Contains("'mistral-7b' is not served", ex.Message);
        Assert.Contains("models list", ex.Message);
    }

    [Fact]
    public void ContextLengthExceededException_ReportsPromptAndLimit_AndActionableMessage()
    {
        ContextLengthExceededException ex = new(9_000, 4_096);

        Assert.Equal(9_000, ex.PromptTokens);
        Assert.Equal(4_096, ex.Limit);
        Assert.Contains("9000", ex.Message);
        Assert.Contains("4096", ex.Message);
        Assert.Contains("Shorten the prompt", ex.Message);
    }

    [Fact]
    public void EngineBusyException_ExplainsKvCapacity_AndRetry()
    {
        EngineBusyException ex = new();

        Assert.Contains("cannot admit this request", ex.Message);
        Assert.Contains("KV pool", ex.Message);
        Assert.Contains("kv_blocks", ex.Message);
        Assert.Contains("Retry", ex.Message);
    }

    [Fact]
    public void GenerationRequest_DefaultsMatchEngineConventions()
    {
        GenerationRequest request = new() { Model = "test-model" };

        Assert.Equal("test-model", request.Model);
        Assert.False(request.IsChat);
        Assert.Null(request.Prompt);
        Assert.Null(request.Messages);
        Assert.Equal(128, request.MaxTokens);
        Assert.Equal(1.0f, request.Temperature);
        Assert.Equal(0.9f, request.TopP);
        Assert.Equal(40, request.TopK);
        Assert.Null(request.Seed);
        Assert.Null(request.Stop);
        Assert.False(request.Stream);
        Assert.Null(request.OnToken);
    }

    [Fact]
    public void GenerationRequest_SamplingKnobsAndStreamingAreOverridable()
    {
        GenerationRequest request = new()
        {
            Model = "test-model",
            IsChat = true,
            Messages = [new ChatMessage("user", "hi")],
            MaxTokens = 16,
            Temperature = 0.2f,
            TopP = 0.5f,
            TopK = 5,
            Seed = 42,
            Stop = ["\n", "END"],
            Stream = true,
            OnToken = _ => { },
        };

        Assert.True(request.IsChat);
        Assert.Equal(16, request.MaxTokens);
        Assert.Equal(0.2f, request.Temperature);
        Assert.Equal(0.5f, request.TopP);
        Assert.Equal(5, request.TopK);
        Assert.Equal(42, request.Seed);
        Assert.Equal(["\n", "END"], request.Stop);
        Assert.True(request.Stream);
        Assert.NotNull(request.OnToken);
    }

    [Fact]
    public void ModelInfo_ExposesAdvertisedFacts()
    {
        ModelInfo info = new("smollm-test", 512, 32, "F16", "llama", 49152);

        Assert.Equal("smollm-test", info.Id);
        Assert.Equal(512, info.ContextLength);
        Assert.Equal(32, info.KvBlocks);
        Assert.Equal("F16", info.Quant);
        Assert.Equal("llama", info.Architecture);
        Assert.Equal(49152, info.VocabSize);
    }

    [Fact]
    public void Open_EmptyModelsList_FailsWithActionableMessage()
    {
        LairConfig config = new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig(),
            Models = [],
        };

        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(() => GenerationService.Open(config));
        Assert.Contains("no models", ex.Message);
    }

    [Fact]
    public void Open_NonexistentModelPath_FailsBeforeEngineUse()
    {
        LairConfig config = new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig(),
            Models = [new ModelConfig { Name = "missing", Path = Path.Combine(Path.GetTempPath(), "lair-missing-model.gguf") }],
        };

        Exception ex = Record.Exception(() => GenerationService.Open(config));
        Assert.NotNull(ex);
        Assert.Contains("lair-missing-model.gguf", ex.Message);
    }
}