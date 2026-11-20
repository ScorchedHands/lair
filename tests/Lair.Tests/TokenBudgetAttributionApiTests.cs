using System.Net;
using System.Text;
using System.Text.Json;
using Lair.Host.Config;
using Lair.Http;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;

namespace Lair.Tests;

/// <summary>
    /// Wire-side budget attribution on the pinned Qwen3 fixture: a reasoning-capped exchange reports the reasoning ceiling, a
    /// response-capped exchange reports the response ceiling, and both keep the ordinary <c>finish_reason: length</c> vocabulary.
    /// The class skips cleanly when the pinned model file is absent, and its live-server bursts run in the serialized API lane so unrelated suites cannot starve this measurement under shared-host load.
    /// </summary>
[Trait("Category", "Api")]
[Collection("lair-api-serialized")]
public sealed class TokenBudgetAttributionApiTests
{
    private const string ModelRelativePath = "models/Qwen3-0.6B-Q8_0.gguf";
    private const string ModelName = "qwen3-token-budget-attribution";

    [Fact]
    public async Task ReasoningCappedExchange_ReportsReasoningExhausted_AndKeepsLengthFinishReason()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(ModelRelativePath);
        if (modelPath is null)
        {
            return; // pinned model absent: this reasoning-attribution proof needs the real thinking fixture
        }

        await using ServedModel served = await OpenAsync(modelPath, maxReasoningTokens: 8, maxResponseTokens: null);

        JsonElement body = await PostJsonAsync(served, """{"model":"qwen3-token-budget-attribution","temperature":0,"messages":[{"role":"user","content":"What is the capital of France? Explain your reasoning, then answer."}]}""");

        Assert.Equal("length", body.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        Assert.Equal("reasoning", body.GetProperty("budget_exhausted").GetString());
    }

    [Fact]
    public async Task ResponseCappedExchange_ReportsResponseExhausted_AndKeepsLengthFinishReason()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(ModelRelativePath);
        if (modelPath is null)
        {
            return; // pinned model absent: this response-attribution proof needs the real thinking fixture
        }

        await using ServedModel served = await OpenAsync(modelPath, maxReasoningTokens: 256, maxResponseTokens: 4);

        JsonElement body = await PostJsonAsync(served, """{"model":"qwen3-token-budget-attribution","temperature":0,"messages":[{"role":"user","content":"What is the capital of France? Explain your reasoning, then answer."}]}""");

        Assert.Equal("length", body.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        Assert.Equal("response", body.GetProperty("budget_exhausted").GetString());
    }

    private static async Task<ServedModel> OpenAsync(string modelPath, int? maxReasoningTokens, int? maxResponseTokens)
    {
        LairConfig config = new()
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
                    MaxReasoningTokens = maxReasoningTokens,
                    MaxResponseTokens = maxResponseTokens,
                },
            ],
        };

        GenerationService engine = GenerationService.Open(config);
        try
        {
            WebApplication app = ApiApp.Build(config, engine);
            await ((IHost)app).StartAsync(CancellationToken.None);
            return new ServedModel(app, engine);
        }
        catch
        {
            engine.Dispose();
            throw;
        }
    }

    private static async Task<JsonElement> PostJsonAsync(ServedModel served, string jsonBody)
    {
        using StringContent content = new(jsonBody, Encoding.UTF8, "application/json");
        HttpResponseMessage response = await served.Http.PostAsync(served.Url + "/v1/chat/completions", content);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        return JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement;
    }

    private sealed class ServedModel : IAsyncDisposable
    {
        private readonly WebApplication _app;
        private readonly GenerationService _engine;

        public ServedModel(WebApplication app, GenerationService engine)
        {
            _app = app;
            _engine = engine;
            Http = new HttpClient();
            Url = app.Urls.Single();
        }

        public HttpClient Http { get; }

        public string Url { get; }

        public async ValueTask DisposeAsync()
        {
            await ((IHost)_app).StopAsync(TimeSpan.FromSeconds(5));
            await _app.DisposeAsync();
            _engine.Dispose();
            Http.Dispose();
        }
    }
}
