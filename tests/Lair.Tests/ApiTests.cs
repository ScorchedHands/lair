using System.Diagnostics;
using System.Net;
using System.Text;
using System.Text.Json;
using Lair.Api;
using Lair.Host.Config;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;

namespace Lair.Tests;

/// <summary>
/// Serves the real SmolLM fixture through <see cref="ApiApp"/> on an ephemeral loopback port:
/// one engine instance and one started web app shared by every API test in the class.
/// </summary>
public sealed class ApiFixture : IAsyncLifetime
{
    public const string ModelName = "smollm-test";

    private GenerationService? _engine;
    private WebApplication? _app;

    public string BaseUrl { get; private set; } = string.Empty;

    public async Task InitializeAsync()
    {
        string modelPath = ModelFixture.Resolve() ?? throw new InvalidOperationException(
            $"GGUF fixture '{ModelFixture.SmolLmRelativePath}' was not found walking up from the test assembly directory.");

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
                    MaxContextTokens = 512,
                    KvBlocks = 32,
                },
            ],
        };

        _engine = GenerationService.Open(config);
        _app = ApiApp.Build(config, _engine);
        await ((IHost)_app).StartAsync(CancellationToken.None);
        BaseUrl = _app.Urls.Single();
    }

    public async Task DisposeAsync()
    {
        if (_app is not null)
        {
            await ((IHost)_app).StopAsync(TimeSpan.FromSeconds(5));
            await _app.DisposeAsync();
        }

        _engine?.Dispose();
    }
}

/// <summary>
/// HTTP-surface tests: OpenAI-shaped success bodies, SSE streaming, the error taxonomy, and the
/// request-id header, all against the shared live server in <see cref="ApiFixture"/>.
/// </summary>
[Trait("Category", "Api")]
public class ApiTests : IClassFixture<ApiFixture>
{
    private static readonly HttpClient Http = new();

    private static readonly string[] FinishReasons = ["stop", "length", "eos"];

    private readonly ApiFixture _fixture;

    public ApiTests(ApiFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public async Task ChatNonStream_ReturnsCompletionWithUsage()
    {
        (HttpResponseMessage response, JsonElement body) = await PostJsonAsync(
            "/v1/chat/completions",
            """{"model":"smollm-test","messages":[{"role":"user","content":"Reply with one short word."}],"max_tokens":8,"temperature":0}""");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
        Assert.True(response.Headers.TryGetValues(ApiApp.RequestIdHeader, out _));

        Assert.Equal("chat.completion", body.GetProperty("object").GetString());
        Assert.StartsWith("chatcmpl-", body.GetProperty("id").GetString());
        Assert.Equal(ApiFixture.ModelName, body.GetProperty("model").GetString());
        Assert.True(body.GetProperty("created").GetInt64() > 0);

        JsonElement choice = body.GetProperty("choices")[0];
        Assert.Equal(0, choice.GetProperty("index").GetInt32());
        Assert.Equal("assistant", choice.GetProperty("message").GetProperty("role").GetString());
        Assert.False(string.IsNullOrEmpty(choice.GetProperty("message").GetProperty("content").GetString()));
        Assert.Contains(choice.GetProperty("finish_reason").GetString(), FinishReasons);

        JsonElement usage = body.GetProperty("usage");
        int promptTokens = usage.GetProperty("prompt_tokens").GetInt32();
        int completionTokens = usage.GetProperty("completion_tokens").GetInt32();
        Assert.True(promptTokens > 0);
        Assert.Equal(promptTokens + completionTokens, usage.GetProperty("total_tokens").GetInt32());
    }

    [Fact]
    public async Task ChatStream_EmitsChunkEventsAndDoneSentinel()
    {
        using HttpRequestMessage request = new(HttpMethod.Post, Url("/v1/chat/completions"))
        {
            Content = new StringContent(
                """{"model":"smollm-test","messages":[{"role":"user","content":"Count from one to four."}],"stream":true,"max_tokens":8,"temperature":0}""",
                Encoding.UTF8,
                "application/json"),
        };

        using HttpResponseMessage response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);
        Assert.True(response.Headers.TryGetValues(ApiApp.RequestIdHeader, out _));

        List<JsonElement> chunks = [];
        string lastDataLine = string.Empty;

        using Stream stream = await response.Content.ReadAsStreamAsync();
        using StreamReader reader = new(stream);

        while (true)
        {
            string? line = await reader.ReadLineAsync();
            if (line is null)
            {
                break;
            }

            if (line.Length == 0)
            {
                continue;
            }

            Assert.StartsWith("data: ", line);
            lastDataLine = line;

            if (line == "data: [DONE]")
            {
                break;
            }

            chunks.Add(JsonDocument.Parse(line["data: ".Length..]).RootElement);
        }

        Assert.True(chunks.Count >= 3, $"expected at least three chunk events, saw {chunks.Count}");
        Assert.Equal("data: [DONE]", lastDataLine);

        foreach (JsonElement chunk in chunks)
        {
            Assert.Equal("chat.completion.chunk", chunk.GetProperty("object").GetString());
            Assert.StartsWith("chatcmpl-", chunk.GetProperty("id").GetString());
        }

        Assert.Equal("assistant", chunks[0].GetProperty("choices")[0].GetProperty("delta").GetProperty("role").GetString());

        string streamedContent = string.Concat(
            chunks
                .Select(chunk => chunk.GetProperty("choices")[0].GetProperty("delta"))
                .Where(delta => delta.TryGetProperty("content", out _))
                .Select(delta => delta.GetProperty("content").GetString()));
        Assert.False(string.IsNullOrEmpty(streamedContent));

        Assert.Contains(chunks[^1].GetProperty("choices")[0].GetProperty("finish_reason").GetString(), FinishReasons);
    }

    [Fact]
    public async Task CompletionsNonStream_ReturnsChoiceText()
    {
        (HttpResponseMessage response, JsonElement body) = await PostJsonAsync(
            "/v1/completions",
            """{"model":"smollm-test","prompt":"The capital of France is","max_tokens":8}""");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text_completion", body.GetProperty("object").GetString());
        Assert.StartsWith("cmpl-", body.GetProperty("id").GetString());

        JsonElement choice = body.GetProperty("choices")[0];
        Assert.Equal(0, choice.GetProperty("index").GetInt32());
        Assert.False(string.IsNullOrEmpty(choice.GetProperty("text").GetString()));
        Assert.Contains(choice.GetProperty("finish_reason").GetString(), FinishReasons);

        Assert.True(body.GetProperty("usage").GetProperty("prompt_tokens").GetInt32() > 0);
        Assert.True(body.GetProperty("usage").GetProperty("total_tokens").GetInt32() > 0);
    }

    [Fact]
    public async Task Models_ListsServedModelWithMetadata()
    {
        HttpResponseMessage response = await Http.GetAsync(Url("/v1/models"));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.True(response.Headers.TryGetValues(ApiApp.RequestIdHeader, out _));

        JsonElement body = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement;
        Assert.Equal("list", body.GetProperty("object").GetString());

        JsonElement entry = body.GetProperty("data")[0];
        Assert.Equal("smollm-test", entry.GetProperty("id").GetString());
        Assert.Equal("model", entry.GetProperty("object").GetString());
        Assert.Equal("lair", entry.GetProperty("owned_by").GetString());
        Assert.Equal(512, entry.GetProperty("context_length").GetInt32());
        Assert.Equal("F16", entry.GetProperty("quant").GetString());
        Assert.False(string.IsNullOrEmpty(entry.GetProperty("architecture").GetString()));
        Assert.Equal("chat", entry.GetProperty("type").GetString()); // the additive kind flag chat surfaces filter on
    }

    [Fact]
    public async Task ChatUnknownModel_Returns404ModelNotFound()
    {
        (HttpResponseMessage response, JsonElement body) = await PostJsonAsync(
            "/v1/chat/completions",
            """{"model":"nope","messages":[{"role":"user","content":"hi"}],"max_tokens":4}""");

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        Assert.True(response.Headers.TryGetValues(ApiApp.RequestIdHeader, out _));

        JsonElement error = body.GetProperty("error");
        Assert.Equal("model_not_found", error.GetProperty("code").GetString());
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        Assert.False(string.IsNullOrEmpty(error.GetProperty("message").GetString()));
    }

    [Fact]
    public async Task ConcurrentChat_RequestsRunTogetherThroughTheBatchScheduler()
    {
        // The batch scheduler admits both requests instead of holding a single engine slot: the second one
        // arrives while the first is still generating and completes normally alongside it.
        Task<(HttpResponseMessage, JsonElement)> first = PostJsonAsync(
            "/v1/chat/completions",
            """{"model":"smollm-test","messages":[{"role":"user","content":"Write a short paragraph about the sea."}],"max_tokens":32,"temperature":0}""");

        await Task.Delay(250);

        (HttpResponseMessage second, JsonElement secondBody) = await PostJsonAsync(
            "/v1/chat/completions",
            """{"model":"smollm-test","messages":[{"role":"user","content":"hi"}],"max_tokens":4}""");

        Assert.Equal(HttpStatusCode.OK, second.StatusCode);
        Assert.True(second.Headers.TryGetValues(ApiApp.RequestIdHeader, out _));
        Assert.Equal("chat.completion", secondBody.GetProperty("object").GetString());

        (HttpResponseMessage firstResponse, JsonElement firstBody) = await first;
        Assert.Equal(HttpStatusCode.OK, firstResponse.StatusCode);
        Assert.Equal("chat.completion", firstBody.GetProperty("object").GetString());
    }

    [Fact]
    [Trait("Category", "Cancel")]
    public async Task ClientAbortMidStream_LeavesTheServerHealthyForLaterRequests()
    {
        // The SSE body is drained only after the generation completes, so the line after the opening role chunk lands at the
        // very end of the run: its absence within the window proves the sequence is still decoding when the client hangs up.
        using HttpRequestMessage request = new(HttpMethod.Post, Url("/v1/completions"))
        {
            Content = new StringContent(
                """{"model":"smollm-test","prompt":"Once upon a time in a land far away,","max_tokens":48,"temperature":0,"stream":true}""",
                Encoding.UTF8,
                "application/json"),
        };

        using HttpResponseMessage response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);

        Stream stream = await response.Content.ReadAsStreamAsync();
        using StreamReader reader = new(stream);

        string? opening = await reader.ReadLineAsync(CancellationToken.None);
        Assert.StartsWith("data: ", opening); // the stream is live: the role chunk is already on the wire

        Task<string?> nextChunk = ReadDataLineAsync(reader);
        Assert.NotSame(nextChunk, await Task.WhenAny(nextChunk, Task.Delay(500))); // the run is still generating

        response.Dispose(); // the client hangs up mid-stream: RequestAborted reaches the sequence still decoding
        _ = nextChunk.ContinueWith(static faulted => _ = faulted.Exception, TaskContinuationOptions.OnlyOnFaulted);

        // The aborted request left the engine and the batch loop intact: the server answers the next requests normally.
        HttpResponseMessage health = await Http.GetAsync(Url("/healthz"));
        Assert.Equal(HttpStatusCode.OK, health.StatusCode);

        (HttpResponseMessage followUp, JsonElement body) = await PostJsonAsync(
            "/v1/completions",
            """{"model":"smollm-test","prompt":"The capital of France is","max_tokens":8,"temperature":0}""");

        Assert.Equal(HttpStatusCode.OK, followUp.StatusCode);
        Assert.Equal("text_completion", body.GetProperty("object").GetString());
        Assert.False(string.IsNullOrEmpty(body.GetProperty("choices")[0].GetProperty("text").GetString()));

        // A fresh streaming request runs the whole pipeline again: prefill, shared decode steps, SSE chunks, and the done sentinel.
        using HttpRequestMessage replay = new(HttpMethod.Post, Url("/v1/completions"))
        {
            Content = new StringContent(
                """{"model":"smollm-test","prompt":"My favorite food is","max_tokens":8,"temperature":0,"stream":true}""",
                Encoding.UTF8,
                "application/json"),
        };

        using HttpResponseMessage replayResponse = await Http.SendAsync(replay, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(HttpStatusCode.OK, replayResponse.StatusCode);
        Assert.Contains("data: [DONE]", await replayResponse.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task PromptBeyondContextWindow_Returns400ContextLengthExceeded()
    {
        string prompt = string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. ", 60));

        (HttpResponseMessage response, JsonElement body) = await PostJsonAsync(
            "/v1/completions",
            BinaryJson(("model", "smollm-test"), ("prompt", prompt), ("max_tokens", 4)));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.True(response.Headers.TryGetValues(ApiApp.RequestIdHeader, out _));

        JsonElement error = body.GetProperty("error");
        Assert.Equal("context_length_exceeded", error.GetProperty("code").GetString());
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
    }

    [Fact]
    public async Task MalformedJsonBody_Returns400InvalidRequest()
    {
        (HttpResponseMessage response, JsonElement body) = await PostJsonAsync("/v1/chat/completions", "{ not json");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.True(response.Headers.TryGetValues(ApiApp.RequestIdHeader, out _));

        JsonElement error = body.GetProperty("error");
        Assert.Equal("invalid_request", error.GetProperty("code").GetString());
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
    }

    [Fact]
    public async Task Healthz_ReportsOkAndModelReady()
    {
        HttpResponseMessage response = await Http.GetAsync(Url("/healthz"));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.True(response.Headers.TryGetValues(ApiApp.RequestIdHeader, out _));

        JsonElement body = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement;
        Assert.Equal("ok", body.GetProperty("status").GetString());
        Assert.True(body.GetProperty("model_ready").GetBoolean());
    }

    [Fact]
    public async Task Embeddings_Responds404UntilImplemented()
    {
        HttpResponseMessage response = await PostJsonRawAsync(
            "/v1/embeddings",
            $"{{\"model\":\"{ApiFixture.ModelName}\",\"input\":\"hello\"}}");

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        Assert.True(response.Headers.TryGetValues(ApiApp.RequestIdHeader, out _));
    }

    private string Url(string path) => _fixture.BaseUrl + path;

    // Reads the next SSE data line, skipping the blank separator lines between events.
    private static async Task<string?> ReadDataLineAsync(StreamReader reader)
    {
        while (true)
        {
            string? line = await reader.ReadLineAsync();
            if (line is null || line.Length > 0)
            {
                return line;
            }
        }
    }

    private async Task<(HttpResponseMessage, JsonElement)> PostJsonAsync(string path, string jsonBody)
    {
        HttpResponseMessage response = await PostJsonRawAsync(path, jsonBody);
        return (response, JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement);
    }

    private async Task<HttpResponseMessage> PostJsonRawAsync(string path, string jsonBody)
    {
        using HttpRequestMessage request = new(HttpMethod.Post, Url(path))
        {
            Content = new StringContent(jsonBody, Encoding.UTF8, "application/json"),
        };

        return await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
    }

    private static string BinaryJson(params (string Key, object Value)[] fields)
    {
        using MemoryStream buffer = new();
        using (Utf8JsonWriter writer = new(buffer))
        {
            writer.WriteStartObject();
            foreach ((string key, object value) in fields)
            {
                switch (value)
                {
                    case string text:
                        writer.WriteString(key, text);
                        break;
                    case int number:
                        writer.WriteNumber(key, number);
                        break;
                    default:
                        throw new ArgumentException($"unsupported fixture value type {value.GetType()}", nameof(fields));
                }
            }

            writer.WriteEndObject();
        }

        return System.Text.Encoding.UTF8.GetString(buffer.ToArray());
    }
}
