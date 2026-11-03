using System.Net;
using System.Text;
using System.Text.Json;
using Lair.Http;

namespace Lair.Tests;

/// <summary>The sampling-params API class runs alone, never concurrently with any other collection.</summary>
[CollectionDefinition("lair-api-serialized", DisableParallelization = true)]
public sealed class SamplingParamsApiCollection;

/// <summary>
/// OpenAI sampling-parameter surface over the live server: <c>presence_penalty</c>, <c>frequency_penalty</c>,
/// <c>logit_bias</c>, and <c>logprobs</c>/<c>top_logprobs</c> on both the non-stream and SSE shapes, plus the
/// <c>invalid_request</c> rejections for out-of-range values and unresolvable bias keys. All runs go through
/// the shared <see cref="ApiFixture"/> engine. The class runs in a collection that is never parallel with any
/// other: its generation bursts saturate every core, and the suite's wall-clock-guarded cancellation and
/// latency tests must not be starved by them under debug-JIT kernel speed.
/// </summary>
[Collection("lair-api-serialized")]
[Trait("Category", "Api")]
public class SamplingParamsApiTests : IClassFixture<ApiFixture>
{
    private static readonly HttpClient Http = new();

    private static readonly string[] FinishReasons = ["stop", "length", "eos"];

    // Calibrated against the fixture: greedy continuation of this prompt starts with token 253 (' a'),
    // so biasing token 9678 (' chocolate') by +10 flips the first generated token.
    private const string BiasPrompt = "My favorite food is";
    private const int BaselineFirstToken = 253;
    private const int ForcedToken = 9678;
    private const string ForcedTokenText = " chocolate";

    // A fixture prompt that keeps generating well past eight tokens under greedy decoding.
    private const string LongPrompt = "Once upon a time in a land far away,";

    private readonly ApiFixture _fixture;

    public SamplingParamsApiTests(ApiFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public async Task LogitBias_NumericTokenId_ForcesItAsFirstGeneratedToken()
    {
        JsonElement baseline = await PostJsonAsync("/v1/completions", LogProbsBody(BiasPrompt, temperature: "0"));

        // Default behavior stays identical when the knob is absent: no logprobs key on a plain call.
        JsonElement plain = await PostJsonAsync("/v1/completions", JsonBody(
            ("model", Str(ApiFixture.ModelName)),
            ("prompt", Str(BiasPrompt)),
            ("max_tokens", "8"),
            ("temperature", "0")));
        Assert.False(plain.GetProperty("choices")[0].TryGetProperty("logprobs", out _));

        JsonElement first = ReportOf(baseline).GetProperty("content")[0];
        Assert.Equal(BaselineFirstToken, first.GetProperty("token_id").GetInt32());

        JsonElement biased = await PostJsonAsync("/v1/completions", JsonBody(
            ("model", Str(ApiFixture.ModelName)),
            ("prompt", Str(BiasPrompt)),
            ("max_tokens", "8"),
            ("temperature", "0"),
            ("logprobs", "true"),
            ("top_logprobs", "5"),
            ("logit_bias", "{\"" + ForcedToken + "\":10}")));

        JsonElement forced = ReportOf(biased).GetProperty("content")[0];
        Assert.Equal(ForcedToken, forced.GetProperty("token_id").GetInt32());
        Assert.Equal(ForcedTokenText, forced.GetProperty("token").GetString());
    }

    [Fact]
    public async Task LogitBias_ChatStyleTokenString_ResolvesThroughTheTokenizer()
    {
        JsonElement biased = await PostJsonAsync("/v1/completions", JsonBody(
            ("model", Str(ApiFixture.ModelName)),
            ("prompt", Str(BiasPrompt)),
            ("max_tokens", "8"),
            ("temperature", "0"),
            ("logprobs", "true"),
            ("top_logprobs", "5"),
            ("logit_bias", "{\"" + ForcedTokenText + "\":10}")));

        JsonElement first = ReportOf(biased).GetProperty("content")[0];
        Assert.Equal(ForcedToken, first.GetProperty("token_id").GetInt32());
        Assert.Equal(ForcedTokenText, first.GetProperty("token").GetString());
    }

    [Fact]
    public async Task Penalties_SameSeedProducesIdenticalOutputAcrossRuns()
    {
        string body = JsonBody(
            ("model", Str(ApiFixture.ModelName)),
            ("prompt", Str(LongPrompt)),
            ("max_tokens", "16"),
            ("temperature", "1.0"),
            ("seed", "42"),
            ("presence_penalty", "1.2"),
            ("frequency_penalty", "0.6"));

        JsonElement first = await PostJsonAsync("/v1/completions", body);
        JsonElement second = await PostJsonAsync("/v1/completions", body);

        string firstText = first.GetProperty("choices")[0].GetProperty("text").GetString()!;
        string secondText = second.GetProperty("choices")[0].GetProperty("text").GetString()!;
        Assert.False(string.IsNullOrEmpty(firstText));
        Assert.Equal(firstText, secondText);
    }

    [Fact]
    public async Task Penalties_ChangeTheSeededContinuation()
    {
        string without = JsonBody(
            ("model", Str(ApiFixture.ModelName)),
            ("prompt", Str(LongPrompt)),
            ("max_tokens", "16"),
            ("temperature", "1.0"),
            ("seed", "42"));
        string with = JsonBody(
            ("model", Str(ApiFixture.ModelName)),
            ("prompt", Str(LongPrompt)),
            ("max_tokens", "16"),
            ("temperature", "1.0"),
            ("seed", "42"),
            ("presence_penalty", "1.2"),
            ("frequency_penalty", "0.6"));

        JsonElement off = await PostJsonAsync("/v1/completions", without);
        JsonElement on = await PostJsonAsync("/v1/completions", with);

        string offText = off.GetProperty("choices")[0].GetProperty("text").GetString()!;
        string onText = on.GetProperty("choices")[0].GetProperty("text").GetString()!;
        Assert.False(string.IsNullOrEmpty(offText));
        Assert.False(string.IsNullOrEmpty(onText));
        Assert.NotEqual(offText, onText);
    }

    [Fact]
    public async Task LogProbs_NonStreamChat_ReportMatchesCompletionTokensAndTopCandidates()
    {
        JsonElement body = await PostJsonAsync("/v1/chat/completions", JsonBody(
            ("model", Str(ApiFixture.ModelName)),
            ("messages", "[{\"role\":\"user\",\"content\":\"Reply with one short word.\"}]"),
            ("max_tokens", "8"),
            ("temperature", "0"),
            ("logprobs", "true"),
            ("top_logprobs", "5")));

        int completionTokens = body.GetProperty("usage").GetProperty("completion_tokens").GetInt32();
        Assert.True(completionTokens > 0);

        JsonElement content = ReportOf(body).GetProperty("content");
        Assert.Equal(completionTokens, content.GetArrayLength());

        foreach (JsonElement entry in content.EnumerateArray())
        {
            Assert.True(entry.TryGetProperty("token_id", out _));
            Assert.True(entry.TryGetProperty("token", out _));
            AssertReport(entry);
        }
    }

    [Fact]
    public async Task LogProbs_NonStreamCompletions_ChosenTokenIsAmongTopCandidates()
    {
        JsonElement body = await PostJsonAsync("/v1/completions", JsonBody(
            ("model", Str(ApiFixture.ModelName)),
            ("prompt", "\"The capital of France is\""),
            ("max_tokens", "8"),
            ("temperature", "1.0"),
            ("seed", "7"),
            ("logprobs", "true"),
            ("top_logprobs", "3")));

        int completionTokens = body.GetProperty("usage").GetProperty("completion_tokens").GetInt32();
        Assert.True(completionTokens > 0);

        JsonElement content = ReportOf(body).GetProperty("content");
        Assert.Equal(completionTokens, content.GetArrayLength());

        foreach (JsonElement entry in content.EnumerateArray())
        {
            int chosen = entry.GetProperty("token_id").GetInt32();
            JsonElement top = entry.GetProperty("top_logprobs");
            Assert.True(top.GetArrayLength() >= 3, $"expected at least three top candidates, saw {top.GetArrayLength()}");
            Assert.Contains(
                chosen,
                top.EnumerateArray().Select(candidate => candidate.GetProperty("token_id").GetInt32()));
        }
    }

    [Fact]
    public async Task LogProbs_Stream_EmitsPerTokenEntriesInEveryContentChunk()
    {
        string requestBody = JsonBody(
            ("model", Str(ApiFixture.ModelName)),
            ("prompt", Str(LongPrompt)),
            ("max_tokens", "8"),
            ("temperature", "0"),
            ("logprobs", "true"),
            ("top_logprobs", "5"),
            ("stream", "true"));

        using HttpRequestMessage request = new(HttpMethod.Post, Url("/v1/completions"))
        {
            Content = new StringContent(requestBody, Encoding.UTF8, "application/json"),
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

        Assert.Equal("data: [DONE]", lastDataLine); // the done sentinel closes the stream as before
        Assert.Equal("assistant", chunks[0].GetProperty("choices")[0].GetProperty("delta").GetProperty("role").GetString());
        Assert.Contains(chunks[^1].GetProperty("choices")[0].GetProperty("finish_reason").GetString(), FinishReasons);

        List<JsonElement> logprobChunks = chunks
            .Select(chunk => chunk.GetProperty("choices")[0].GetProperty("delta"))
            .Where(delta => delta.TryGetProperty("logprobs", out _))
            .ToList();
        Assert.True(logprobChunks.Count >= 3, $"expected at least three logprob chunks, saw {logprobChunks.Count}");

        // Every content chunk carries its logprob entry, and every logprob chunk carries its content piece.
        int contentChunks = chunks
            .Select(chunk => chunk.GetProperty("choices")[0].GetProperty("delta"))
            .Count(delta => delta.TryGetProperty("content", out _));
        Assert.Equal(contentChunks, logprobChunks.Count);

        foreach (JsonElement delta in logprobChunks)
        {
            Assert.True(delta.TryGetProperty("content", out _));
            AssertReport(delta.GetProperty("logprobs"));
        }

        // The streamed entries mirror the non-stream report for the same request.
        JsonElement nonStream = await PostJsonAsync("/v1/completions", LogProbsBody(LongPrompt, temperature: "0"));
        Assert.Equal(
            nonStream.GetProperty("usage").GetProperty("completion_tokens").GetInt32(),
            logprobChunks.Count);
    }

    [Theory]
    [InlineData("presence_penalty", "2.5")]
    [InlineData("presence_penalty", "-2.5")]
    [InlineData("frequency_penalty", "3.0")]
    public async Task PenaltyOutOfRange_Returns400InvalidRequest(string knob, string value)
    {
        (HttpResponseMessage response, JsonElement body) = await PostJsonExpectingAsync(
            "/v1/completions",
            JsonBody(
                ("model", Str(ApiFixture.ModelName)),
                ("prompt", Str(BiasPrompt)),
                ("max_tokens", "4"),
                ("temperature", "0"),
                (knob, value)));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.True(response.Headers.TryGetValues(ApiApp.RequestIdHeader, out _));
        Assert.Equal("invalid_request", body.GetProperty("error").GetProperty("code").GetString());
        Assert.Equal("invalid_request_error", body.GetProperty("error").GetProperty("type").GetString());
        Assert.Contains(knob, body.GetProperty("error").GetProperty("message").GetString());
    }

    [Fact]
    public async Task TopLogprobsOutOfRange_Returns400InvalidRequest()
    {
        (HttpResponseMessage response, JsonElement body) = await PostJsonExpectingAsync(
            "/v1/completions",
            JsonBody(
                ("model", Str(ApiFixture.ModelName)),
                ("prompt", Str(BiasPrompt)),
                ("max_tokens", "4"),
                ("temperature", "0"),
                ("logprobs", "true"),
                ("top_logprobs", "21")));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal("invalid_request", body.GetProperty("error").GetProperty("code").GetString());
        Assert.Contains("top_logprobs", body.GetProperty("error").GetProperty("message").GetString());
    }

    [Fact]
    public async Task LogitBiasValueOutOfRange_Returns400InvalidRequest()
    {
        (HttpResponseMessage response, JsonElement body) = await PostJsonExpectingAsync(
            "/v1/completions",
            JsonBody(
                ("model", Str(ApiFixture.ModelName)),
                ("prompt", Str(BiasPrompt)),
                ("max_tokens", "4"),
                ("temperature", "0"),
                ("logit_bias", "{\"" + ForcedToken + "\":150}")));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal("invalid_request", body.GetProperty("error").GetProperty("code").GetString());
        Assert.Contains("logit_bias", body.GetProperty("error").GetProperty("message").GetString());
    }

    [Fact]
    public async Task LogitBiasUnresolvableKey_Returns400NamingTheKey()
    {
        const string badKey = "not a single token xyz";
        (HttpResponseMessage response, JsonElement body) = await PostJsonExpectingAsync(
            "/v1/completions",
            JsonBody(
                ("model", Str(ApiFixture.ModelName)),
                ("prompt", Str(BiasPrompt)),
                ("max_tokens", "4"),
                ("temperature", "0"),
                ("logit_bias", "{\"" + badKey + "\":1.0}")));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal("invalid_request", body.GetProperty("error").GetProperty("code").GetString());
        Assert.Contains(badKey, body.GetProperty("error").GetProperty("message").GetString());
    }

    [Fact]
    public async Task LogitBiasTokenIdOutsideVocabulary_Returns400InvalidRequest()
    {
        (HttpResponseMessage response, JsonElement body) = await PostJsonExpectingAsync(
            "/v1/completions",
            JsonBody(
                ("model", Str(ApiFixture.ModelName)),
                ("prompt", Str(BiasPrompt)),
                ("max_tokens", "4"),
                ("temperature", "0"),
                ("logit_bias", "{\"999999\":1.0}")));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal("invalid_request", body.GetProperty("error").GetProperty("code").GetString());
        Assert.Contains("logit_bias", body.GetProperty("error").GetProperty("message").GetString());
    }

    // Builds a request body from (property name, raw JSON value) pairs so the fixtures stay readable.
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

    // A completions body that asks for logprobs with five reported candidates.
    private static string LogProbsBody(string prompt, string temperature) => JsonBody(
        ("model", Str(ApiFixture.ModelName)),
        ("prompt", Str(prompt)),
        ("max_tokens", "8"),
        ("temperature", temperature),
        ("logprobs", "true"),
        ("top_logprobs", "5"));

    // One logprob entry's value contract: probabilities ≤ 1 (logprobs ≤ 0), top candidates sorted descending,
    // and the chosen token present among them.
    private static void AssertReport(JsonElement entry)
    {
        double logprob = entry.GetProperty("logprob").GetDouble();
        Assert.True(logprob <= 0.0, $"logprob {logprob} exceeds log(1)");
        Assert.True(double.IsFinite(logprob));

        JsonElement top = entry.GetProperty("top_logprobs");
        Assert.True(top.GetArrayLength() >= 1);

        int chosen = entry.TryGetProperty("token_id", out JsonElement id) ? id.GetInt32() : -1;
        double previous = double.PositiveInfinity;
        bool chosenPresent = false;
        foreach (JsonElement candidate in top.EnumerateArray())
        {
            double candidateLogprob = candidate.GetProperty("logprob").GetDouble();
            Assert.True(candidateLogprob <= previous, "top_logprobs are not sorted descending");
            Assert.True(candidateLogprob <= 0.0);
            previous = candidateLogprob;

            if (candidate.GetProperty("token_id").GetInt32() == chosen)
            {
                chosenPresent = true;
            }
        }

        Assert.True(chosenPresent, "chosen token missing from top_logprobs");
    }

    private static string Str(string value) => "\"" + value + "\"";

    private static JsonElement ReportOf(JsonElement body) => body.GetProperty("choices")[0].GetProperty("logprobs");

    private string Url(string path) => _fixture.BaseUrl + path;

    // POSTs that must succeed: asserts 200 and hands back the parsed body.
    private async Task<JsonElement> PostJsonAsync(string path, string jsonBody)
    {
        (HttpResponseMessage response, JsonElement body) = await PostJsonExpectingAsync(path, jsonBody);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        return body;
    }

    // POSTs whose status is the assertion: hands back both the response and its parsed body.
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
