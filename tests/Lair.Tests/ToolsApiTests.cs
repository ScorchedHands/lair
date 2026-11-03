using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Lair.Http;
using Lair.Host.Config;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;

namespace Lair.Tests;

/// <summary>
/// Serves the real SmolLM fixture for the tool-shape tests on an ephemeral loopback port. When the GGUF is absent the
/// app never starts and every test in this file skips, matching how the other real-model suites skip. Runs inside the
/// serialized API collection so its generation bursts cannot starve the suite's wall-clock-guarded classes.
/// </summary>
[Collection("lair-api-serialized")]
public sealed class ToolsApiFixture : IAsyncLifetime
{
    public const string ModelName = "smollm-test";

    private GenerationService? _engine;
    private WebApplication? _app;

    /// <summary>The server's base URL, or empty when the fixture model is absent and nothing was started.</summary>
    public string BaseUrl { get; private set; } = string.Empty;

    public async Task InitializeAsync()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

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
/// Live tool-call wire shapes against the real SmolLM fixture: a chat request that declares tools returns an OpenAI-shaped
/// result (tool_calls absent or valid, finish_reason in {stop, length, eos, tool_calls}), SSE carries complete-call deltas
/// before the final chunk, tool_choice none closes the pipeline without touching prompt size, and malformed tool
/// declarations are refused as invalid_request naming the field. Emission itself is model-dependent per the detection
/// contract; only injection, shape, and validation are asserted here.
/// </summary>
[Trait("Category", "Api")]
public class ToolsApiTests : IClassFixture<ToolsApiFixture>
{
    private static readonly HttpClient Http = new();

    // A recognized call is optional from a small fixture model: the contract covers every ordinary ending too.
    private static readonly string[] FinishReasons = ["stop", "length", "eos", "tool_calls"];

    private const string ToolsJson = """[{"type":"function","function":{"name":"get_weather","description":"Fetch current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]""";
    private static readonly Regex CallIdPattern = new("^call_[0-9a-f]{12}$", RegexOptions.Compiled);

    private readonly ToolsApiFixture _fixture;

    public ToolsApiTests(ToolsApiFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public async Task ChatWithTools_NonStream_ReturnsWellFormedToolShape()
    {
        if (_fixture.BaseUrl.Length == 0)
        {
            return; // skip when the SmolLM fixture is absent
        }

        JsonElement body = await PostJsonAsync("/v1/chat/completions", JsonBody(
            ("model", Str(ToolsApiFixture.ModelName)),
            ("messages", """[{"role":"user","content":"What is the weather in Oslo? Reply briefly."}]"""),
            ("max_tokens", "16"),
            ("temperature", "0"),
            ("tools", ToolsJson)));

        Assert.Equal("chat.completion", body.GetProperty("object").GetString());

        JsonElement choice = body.GetProperty("choices")[0];
        Assert.Contains(choice.GetProperty("finish_reason").GetString(), FinishReasons);

        JsonElement message = choice.GetProperty("message");
        Assert.Equal("assistant", message.GetProperty("role").GetString());
        Assert.True(message.TryGetProperty("content", out _)); // content is always reported, possibly empty

        if (message.TryGetProperty("tool_calls", out JsonElement calls))
        {
            AssertToolCallArray(calls);
            Assert.Equal("tool_calls", choice.GetProperty("finish_reason").GetString());
        }

        int completionTokens = body.GetProperty("usage").GetProperty("completion_tokens").GetInt32();
        Assert.True(completionTokens > 0);
    }

    [Fact]
    public async Task ChatWithTools_Stream_CarriesToolCallDeltasBeforeTheFinalChunk()
    {
        if (_fixture.BaseUrl.Length == 0)
        {
            return; // skip when the SmolLM fixture is absent
        }

        List<JsonElement> chunks = await StreamAsync("/v1/chat/completions", JsonBody(
            ("model", Str(ToolsApiFixture.ModelName)),
            ("messages", """[{"role":"user","content":"What is the weather in Oslo? Reply briefly."}]"""),
            ("max_tokens", "12"),
            ("temperature", "0"),
            ("tools", ToolsJson),
            ("stream", "true")));

        Assert.Equal("assistant", chunks[0].GetProperty("choices")[0].GetProperty("delta").GetProperty("role").GetString());

        string finishReason = chunks[^1].GetProperty("choices")[0].GetProperty("finish_reason").GetString()!;
        Assert.Contains(finishReason, FinishReasons);

        List<JsonElement> callDeltas = chunks
            .Select(chunk => chunk.GetProperty("choices")[0].GetProperty("delta"))
            .Where(delta => delta.TryGetProperty("tool_calls", out _))
            .ToList();

        foreach (JsonElement delta in callDeltas)
        {
            AssertToolCallArray(delta.GetProperty("tool_calls"));
        }

        // A recognized call must surface on the wire and flip the finish reason; content deltas stay well-formed either way.
        if (finishReason == "tool_calls")
        {
            Assert.NotEmpty(callDeltas);
        }
    }

    [Fact]
    public async Task ToolChoiceNone_DoesNotInjectSchemas_AutoDoes()
    {
        if (_fixture.BaseUrl.Length == 0)
        {
            return; // skip when the SmolLM fixture is absent
        }

        const string messages = """[{"role":"user","content":"What is the weather in Oslo? Reply briefly."}]""";

        JsonElement plain = await PostJsonAsync("/v1/chat/completions", ChatBody(messages, null));
        int plainTokens = plain.GetProperty("usage").GetProperty("prompt_tokens").GetInt32();

        // tool_choice "none" closes the pipeline: same prompt size as no tools at all, and nothing call-shaped on the wire.
        JsonElement noneChoice = await PostJsonAsync("/v1/chat/completions", ChatBody(
            messages, """[{"type":"function","function":{"name":"get_weather","parameters":{}}}]""", Str("none")));
        Assert.Equal(plainTokens, noneChoice.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());
        Assert.False(noneChoice.GetProperty("choices")[0].GetProperty("message").TryGetProperty("tool_calls", out _));

        // The default choice injects: the appended instruction line, schema fences, and contract grow the prompt.
        JsonElement auto = await PostJsonAsync("/v1/chat/completions", ChatBody(
            messages, """[{"type":"function","function":{"name":"get_weather","parameters":{}}}]"""));
        Assert.True(
            auto.GetProperty("usage").GetProperty("prompt_tokens").GetInt32() > plainTokens,
            "declaring tools must enlarge the rendered prompt through injection");
    }

    [Theory]
    [InlineData("function.name", """[{"type":"function","function":{"parameters":{}}}]""")]          // name missing
    [InlineData("function.parameters", """[{"type":"function","function":{"name":"x"}}]""")]         // parameters missing
    [InlineData("tools[0].type", """[{"function":{"name":"x","parameters":{}}}]""")]                 // type member missing
    [InlineData("tools", """{"not":"array"}""")]                                                     // not an array at all
    public async Task MalformedTools_Return400InvalidRequestNamingTheField(string fieldHint, string toolsJson)
    {
        if (_fixture.BaseUrl.Length == 0)
        {
            return; // skip when the SmolLM fixture is absent
        }

        (HttpResponseMessage response, JsonElement body) = await PostJsonExpectingAsync(
            "/v1/chat/completions", ChatBody("""[{"role":"user","content":"hi"}]""", toolsJson));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.True(response.Headers.TryGetValues(ApiApp.RequestIdHeader, out _));
        Assert.Equal("invalid_request_error", body.GetProperty("error").GetProperty("type").GetString());
        Assert.Equal("invalid_request", body.GetProperty("error").GetProperty("code").GetString());
        Assert.Contains(fieldHint, body.GetProperty("error").GetProperty("message").GetString());
    }

    [Fact]
    public async Task ToolChoiceForcingUndeclaredTool_Returns400NamingToolChoice()
    {
        if (_fixture.BaseUrl.Length == 0)
        {
            return; // skip when the SmolLM fixture is absent
        }

        (HttpResponseMessage response, JsonElement body) = await PostJsonExpectingAsync(
            "/v1/chat/completions", ChatBody(
                """[{"role":"user","content":"hi"}]""",
                ToolsJson,
                """{"type":"function","function":{"name":"not_declared"}}"""));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains("tool_choice", body.GetProperty("error").GetProperty("message").GetString());
    }

    [Theory]
    [InlineData("""[{"type":"function","function":{"name":"get_weather","parameters":{}}}]""")] // valid shape: still ignored here
    [InlineData("""{"not":"array"}""")]                                                         // malformed too: also ignored
    public async Task CompletionsEndpoint_IgnoresTheToolsField(string toolsJson)
    {
        if (_fixture.BaseUrl.Length == 0)
        {
            return; // skip when the SmolLM fixture is absent
        }

        JsonElement body = await PostJsonAsync("/v1/completions", JsonBody(
            ("model", Str(ToolsApiFixture.ModelName)),
            ("prompt", "\"The capital of France is\""),
            ("max_tokens", "8"),
            ("temperature", "0"),
            ("tools", toolsJson)));

        Assert.Equal("text_completion", body.GetProperty("object").GetString());
        string[] completionFinishReasons = ["stop", "length", "eos"];
        Assert.Contains(body.GetProperty("choices")[0].GetProperty("finish_reason").GetString(), completionFinishReasons);
        Assert.DoesNotContain("tool_calls", body.ToString()); // chat-only surface: nothing tool-shaped comes back
    }

    // A chat body with optional tool declarations and an optional tool_choice, built so the raw shapes stay visible.
    private static string ChatBody(string messagesJson, string? toolsJson, string? toolChoiceJson = null)
    {
        List<(string Key, string RawJson)> fields =
        [
            ("model", Str(ToolsApiFixture.ModelName)),
            ("messages", messagesJson),
            ("max_tokens", "8"),
            ("temperature", "0"),
        ];

        if (toolsJson is not null)
        {
            fields.Add(("tools", toolsJson));
        }

        if (toolChoiceJson is not null)
        {
            fields.Add(("tool_choice", toolChoiceJson));
        }

        return JsonBody([.. fields]);
    }

    private static string JsonBody(params (string Key, string RawJson)[] fields)
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

    // The OpenAI tool-call entry shape: call_<12 hex> id, function type, declared name, and a parseable arguments JSON string.
    private static void AssertToolCallArray(JsonElement calls)
    {
        Assert.True(calls.GetArrayLength() >= 1);

        foreach (JsonElement call in calls.EnumerateArray())
        {
            Assert.Matches(CallIdPattern, call.GetProperty("id").GetString());
            Assert.Equal("function", call.GetProperty("type").GetString());

            JsonElement function = call.GetProperty("function");
            Assert.Equal("get_weather", function.GetProperty("name").GetString()); // the only declared tool in these tests

            string arguments = function.GetProperty("arguments").GetString()!;
            Assert.NotEmpty(arguments);

            // The arguments must be valid, parseable JSON that already sits compactly in the wire string.
            using JsonDocument parsed = JsonDocument.Parse(arguments);
            Assert.Equal(arguments, parsed.RootElement.GetRawText());
        }
    }

    private async Task<List<JsonElement>> StreamAsync(string path, string jsonBody)
    {
        using HttpRequestMessage request = new(HttpMethod.Post, Url(path))
        {
            Content = new StringContent(jsonBody, Encoding.UTF8, "application/json"),
        };

        using HttpResponseMessage response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);

        List<JsonElement> chunks = [];
        string lastDataLine = string.Empty;
        using (Stream stream = await response.Content.ReadAsStreamAsync())
        using (StreamReader reader = new(stream))
        {
            while (true)
            {
                string? line = await reader.ReadLineAsync();
                if (line is null || line.Length == 0)
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
        }

        Assert.Equal("data: [DONE]", lastDataLine); // the done sentinel is unchanged by tool deltas
        return chunks;
    }

    private string Url(string path) => _fixture.BaseUrl + path;

    private async Task<JsonElement> PostJsonAsync(string path, string jsonBody)
    {
        (HttpResponseMessage response, JsonElement body) = await PostJsonExpectingAsync(path, jsonBody);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        return body;
    }

    private async Task<(HttpResponseMessage, JsonElement)> PostJsonExpectingAsync(string path, string jsonBody)
    {
        using HttpRequestMessage request = new(HttpMethod.Post, Url(path))
        {
            Content = new StringContent(jsonBody, Encoding.UTF8, "application/json"),
        };

        HttpResponseMessage response = await Http.SendAsync(request);
        return (response, JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement);
    }
}

