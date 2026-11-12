using System.Net;
using System.Text;
using System.Text.Json;
using Lair.Host.Config;
using Lair.Http;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;

namespace Lair.Tests;

/// <summary>
/// The vocabulary gate a reasoning budget is resolved against, at model load and on reload: a model declaring
/// <c>max_reasoning_tokens</c> on a tokenizer that cannot carry <c>&lt;think&gt;</c> and <c>&lt;/think&gt;</c> as whole tokens is
/// refused naming the model, the key, and the marker; a response-only ceiling on that same marker-less model opens and
/// bounds visible tokens exactly as before; and on a vocabulary that does carry the markers, the ceiling closes the region
/// at the ceiling through the ids the gate resolved. Every case runs on synthetic tiny models.
/// </summary>
[Trait("Category", "Api")]
[Collection("lair-api-serialized")]
public sealed class TokenBudgetMarkerGateTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    [Fact]
    public void ReasoningCeilingOnMarkerLessVocabulary_FailsModelOpenNamingModelKeyAndMarker()
    {
        string path = SyntheticLlamaGguf.Write("gate-unbound-reasoning", withReasoningMarkers: false);
        _tempFiles.Add(path);

        LairConfig config = ConfigFor(path, "gate-unbound-reasoning", maxReasoningTokens: 16, maxResponseTokens: null);

        InvalidOperationException failure = Assert.Throws<InvalidOperationException>(() => GenerationService.Open(config));

        Assert.Contains("gate-unbound-reasoning", failure.Message, StringComparison.Ordinal);
        Assert.Contains("max_reasoning_tokens", failure.Message, StringComparison.Ordinal);
        Assert.Contains("<think>", failure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ResponseCeilingOnMarkerLessVocabulary_Opens_AndBoundsVisibleTokens()
    {
        await using ServedModel served = await ServeAsync("gate-response-only", maxReasoningTokens: null, maxResponseTokens: 3, withReasoningMarkers: false);

        JsonElement body = await PostJsonAsync(served, "/v1/completions", CompletionJson(
            "gate-response-only",
            ("prompt", "\"hello world\""),
            ("max_tokens", "8"),
            ("temperature", "0")));

        Assert.Equal("length", body.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        Assert.Equal(3, body.GetProperty("usage").GetProperty("completion_tokens").GetInt32());
        Assert.Equal("response", body.GetProperty("budget_exhausted").GetString());
    }

    [Fact]
    public async Task ReasoningCeiling_OnMarkerBearingVocabulary_ClosesRegionAtTheCeiling()
    {
        await using ServedModel served = await ServeAsync("gate-region-close", maxReasoningTokens: 1, maxResponseTokens: null, withReasoningMarkers: true);

        JsonElement body = await PostJsonAsync(served, "/v1/chat/completions", ChatJson(
            "gate-region-close",
            ("max_tokens", "8"),
            ("temperature", "0"),
            ("logit_bias", """{"6":10}""")));

        string content = body.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()!;
        Assert.StartsWith("<think></think>", content, StringComparison.Ordinal);
        Assert.Equal("length", body.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        Assert.Equal("reasoning", body.GetProperty("budget_exhausted").GetString());
    }

    [Fact]
    public void ReloadAddingReasoningCeilingOnMarkerLessModel_IsRefusedNamingKeyAndMarker()
    {
        string path = SyntheticLlamaGguf.Write("gate-reload-unbound", withReasoningMarkers: false);
        _tempFiles.Add(path);

        LairConfig config = ConfigFor(path, "gate-reload-unbound", maxReasoningTokens: null, maxResponseTokens: null);
        using GenerationService engine = GenerationService.Open(config);

        IReadOnlyList<FieldResult> results = engine.ApplyConfiguration(config with
        {
            Models = [config.Models[0] with { MaxReasoningTokens = 8 }],
        });

        FieldResult result = Assert.Single(results);
        Assert.Equal(ConfigApplyOutcome.Refused, result.Outcome);
        Assert.Contains("max_reasoning_tokens", result.Change.Field, StringComparison.Ordinal);
        Assert.Contains("<think>", result.Detail, StringComparison.Ordinal);
    }

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
                // A model still holding its mmap view keeps the file on Windows; the temp directory absorbs it.
            }
        }

        _tempFiles.Clear();
    }

    private static LairConfig ConfigFor(string modelPath, string modelName, int? maxReasoningTokens, int? maxResponseTokens) => new()
    {
        Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
        Engine = new EngineConfig(),
        Models =
        [
            new ModelConfig
            {
                Name = modelName,
                Path = modelPath,
                MaxContextTokens = 512,
                KvBlocks = 32,
                MaxReasoningTokens = maxReasoningTokens,
                MaxResponseTokens = maxResponseTokens,
            },
        ],
    };

    private async Task<ServedModel> ServeAsync(string modelName, int? maxReasoningTokens, int? maxResponseTokens, bool withReasoningMarkers)
    {
        string path = SyntheticLlamaGguf.Write(modelName, withReasoningMarkers);
        _tempFiles.Add(path);

        LairConfig config = ConfigFor(path, modelName, maxReasoningTokens, maxResponseTokens);
        GenerationService engine = GenerationService.Open(config);
        WebApplication app = ApiApp.Build(config, engine);
        await ((IHost)app).StartAsync(CancellationToken.None);
        return new ServedModel(app, engine);
    }

    // ---------------------------------------------------------------- request plumbing

    private static string ChatJson(string model, params (string Key, string RawJson)[] fields)
    {
        List<(string Key, string RawJson)> all = [("model", Str(model)), ("messages", """[{"role":"user","content":"hello"}]""")];
        all.AddRange(fields);
        return JsonBody(all);
    }

    private static string CompletionJson(string model, params (string Key, string RawJson)[] fields)
    {
        List<(string Key, string RawJson)> all = [("model", Str(model))];
        all.AddRange(fields);
        return JsonBody(all);
    }

    private static string JsonBody(List<(string Key, string RawJson)> fields)
    {
        using MemoryStream buffer = new();
        using (Utf8JsonWriter writer = new(buffer))
        {
            writer.WriteStartObject();
            foreach ((string key, string raw) in fields)
            {
                writer.WritePropertyName(key);
                writer.WriteRawValue(raw);
            }

            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.ToArray());
    }

    private static string Str(string value) => "\"" + value + "\"";

    private static async Task<JsonElement> PostJsonAsync(ServedModel served, string path, string jsonBody)
    {
        using StringContent content = new(jsonBody, Encoding.UTF8, "application/json");
        HttpResponseMessage response = await served.Http.PostAsync(served.Url + path, content);
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
