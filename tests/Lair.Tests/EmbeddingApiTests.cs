using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

using Lair.Contracts;
using Lair.Http;
using Lair.Http.Monitoring;
using Lair.Host.Config;
using Lair.Kv;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;

namespace Lair.Tests;

/// <summary>
/// The embeddings endpoint over a synthetic encoder-posture model: OpenAI's list envelope with one datum per input at its own
/// index and prompt-side usage, both input spellings, order and repeat-input identity across a merged batch, batched answers
/// matching the same inputs sent as separate requests within float slack, mean-vs-CLS divergence and L2 normalization as
/// configured posture, the endpoint refusals in both directions (a sampling model against this endpoint, an encoder model
/// against the sampling endpoints), the additive pooling facts on <c>/v1/models</c>, the request-shape refusals, the encoder
/// lane's bounded-queue posture under contention, and an up-front KV-capacity refusal naming the encoder positions it needs.
/// </summary>
[Trait("Category", "Embeddings")]
[Trait("Category", "Api")]
[Collection("lair-api-serialized")] // the contention bursts below are CPU-wide: they share the serialized API lane so live-server measurements in other API suites are not starved by them
public sealed class EmbeddingApiTests : IDisposable
{
    // The synthetic fixture's geometry: hidden 4 is the vector length every embedding answer carries.
    private const int Hidden = 4;
    private const string ModelName = "synthetic-embed";

    private readonly List<string> _tempFiles = [];

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

    // ---------------------------------------------------------------- envelope and input spellings

    [Fact]
    public async Task Embeddings_SingleStringInput_AnswersOpenAiListEnvelope()
    {
        await using EmbeddingServer server = await EmbeddingServer.StartAsync(NewModel(), pooling: "mean", normalize: false);

        (HttpResponseMessage response, JsonElement body) = await PostEmbeddingsAsync(server, "\"hello\"");

        Assert.True(response.IsSuccessStatusCode, $"the embeddings request was refused: {body.GetRawText()}");
        Assert.True(response.Headers.TryGetValues(ApiApp.RequestIdHeader, out _));

        Assert.Equal("list", body.GetProperty("object").GetString());
        Assert.Equal(ModelName, body.GetProperty("model").GetString());

        JsonElement data = body.GetProperty("data");
        Assert.Equal(1, data.GetArrayLength());

        JsonElement datum = data[0];
        Assert.Equal("embedding", datum.GetProperty("object").GetString());
        Assert.Equal(0, datum.GetProperty("index").GetInt32());
        Assert.Equal(Hidden, datum.GetProperty("embedding").GetArrayLength());

        JsonElement usage = body.GetProperty("usage");
        Assert.Equal(1, usage.GetProperty("prompt_tokens").GetInt32()); // "hello" is one declared piece
        Assert.Equal(usage.GetProperty("prompt_tokens").GetInt32(), usage.GetProperty("total_tokens").GetInt32()); // nothing was generated to count
    }

    [Fact]
    public async Task Embeddings_ArrayInputs_PreserveOrderAndIndex_AndARepeatInputRepeatsItsVector()
    {
        await using EmbeddingServer server = await EmbeddingServer.StartAsync(NewModel(), pooling: "mean", normalize: false);

        (HttpResponseMessage response, JsonElement body) = await PostEmbeddingsAsync(
            server, """["hello","world!","hello"]""");

        response.EnsureSuccessStatusCode();

        JsonElement data = body.GetProperty("data");
        Assert.Equal(3, data.GetArrayLength());

        float[][] vectors = [.. data.EnumerateArray().Select(datum =>
        {
            Assert.Equal("embedding", datum.GetProperty("object").GetString());
            return ReadVector(datum);
        })];

        // Index is arrival order, and the same text in the same batch answers with the same bits.
        for (int i = 0; i < data.GetArrayLength(); i++)
        {
            Assert.Equal(i, data[i].GetProperty("index").GetInt32());
            Assert.Equal(Hidden, vectors[i].Length);
        }

        Assert.Equal(vectors[0], vectors[2]);
        Assert.True(!vectors[0].SequenceEqual(vectors[1]), "different inputs must not answer with the same vector");

        JsonElement usage = body.GetProperty("usage");
        Assert.Equal(4, usage.GetProperty("prompt_tokens").GetInt32()); // "hello" (1) + "world!" (2) + "hello" (1)
        Assert.Equal(4, usage.GetProperty("total_tokens").GetInt32());
    }

    [Fact]
    public async Task Embeddings_MeanAndClsPosturesDiffer_AndNormalizeRescalesPooledVectors()
    {
        string path = NewModel();
        const string input = "hello world !"; // three positions: a mean over them is not the first row

        await using EmbeddingServer raw = await EmbeddingServer.StartAsync(path, pooling: "mean", normalize: false);
        await using EmbeddingServer cls = await EmbeddingServer.StartAsync(path, pooling: "cls", normalize: true);

        float[] meanVector = await SingleVectorAsync(raw, input);
        float[] clsVector = await SingleVectorAsync(cls, input);

        Assert.Equal(Hidden, meanVector.Length);
        Assert.Equal(Hidden, clsVector.Length);
        Assert.True(MaxAbs(meanVector) > 1e-6f && MaxAbs(clsVector) > 1e-6f, "both postures must answer with a non-degenerate vector");

        double squaredDistance = 0;
        for (int i = 0; i < Hidden; i++)
        {
            squaredDistance += (double)meanVector[i] - clsVector[i];
            squaredDistance *= squaredDistance;
        }

        Assert.True(squaredDistance >= 1e-6, $"mean and CLS pooling answered the same vector (distance {Math.Sqrt(squaredDistance):G6})");

        // The normalized posture answers at unit length; the raw posture answers with what pooling produced, unrescaled.
        Assert.True(Math.Abs(L2Norm(clsVector) - 1.0) <= 1e-5, $"normalized vector has L2 norm {L2Norm(clsVector):G9}, expected ~1");
        Assert.True(MaxAbs(meanVector) > MaxAbs(clsVector) * 0.5f || Math.Abs(L2Norm(meanVector) - 1.0) > 1e-3, "the raw posture must not hand back a rescaled vector");
    }

    [Fact]
    public async Task Embeddings_BatchedRequestMatchesTheSameInputsSentSeparately()
    {
        await using EmbeddingServer server = await EmbeddingServer.StartAsync(NewModel(), pooling: "mean", normalize: false);

        string[] inputs = ["hello", "world! world", "hello world !"];

        (HttpResponseMessage batchedResponse, JsonElement batched) = await PostEmbeddingsAsync(
            server,
            $$"""[{{string.Join(",", inputs.Select(input => JsonSerializer.Serialize(input)))}}]""");

        batchedResponse.EnsureSuccessStatusCode();
        float[][] batchedVectors = [.. batched.GetProperty("data").EnumerateArray().Select(ReadVector)];
        Assert.Equal(inputs.Length, batchedVectors.Length);

        // The same inputs as separate single-input requests: the merged forward and the per-request forwards must agree within float slack.
        for (int i = 0; i < inputs.Length; i++)
        {
            float[] solo = await SingleVectorAsync(server, inputs[i]);
            AssertClose(batchedVectors[i], solo, $"batched input {i} vs its own request");
        }
    }

    // ---------------------------------------------------------------- endpoint posture refusals

    [Fact]
    public async Task Embeddings_SamplingEndpointsRefuseEncoderModel_NamingEndpointAndType()
    {
        await using EmbeddingServer server = await EmbeddingServer.StartAsync(NewModel(), pooling: "mean", normalize: false);

        (HttpResponseMessage chatResponse, JsonElement chatError) = await PostJsonAsync(
            server, "/v1/chat/completions", $$"""{"model":"{{ModelName}}","messages":[{"role":"user","content":"hello"}]}""");

        Assert.Equal(HttpStatusCode.BadRequest, chatResponse.StatusCode);
        Assert.Equal("invalid_request_error", chatError.GetProperty("error").GetProperty("type").GetString());
        Assert.Equal("invalid_request", chatError.GetProperty("error").GetProperty("code").GetString());
        Assert.Contains("/v1/chat/completions", chatError.GetProperty("error").GetProperty("message").GetString());
        Assert.Contains("/v1/embeddings", chatError.GetProperty("error").GetProperty("message").GetString());

        (HttpResponseMessage completionResponse, JsonElement completionError) = await PostJsonAsync(
            server, "/v1/completions", $$"""{"model":"{{ModelName}}","prompt":"hello","max_tokens":4}""");

        Assert.Equal(HttpStatusCode.BadRequest, completionResponse.StatusCode);
        Assert.Contains("/v1/completions", completionError.GetProperty("error").GetProperty("message").GetString());

        // The refusal answered without touching the engine's serving state: this model still embeds.
        await SingleVectorAsync(server, "hello");
    }

    [Fact]
    public async Task Embeddings_QuantizedKvModel_RefusesWithEncoderUnsupported_NamingWhatToOmit()
    {
        // The encoder pass reads float KV slots; a model opened on a block-quant pair answers with the posture refusal that names the config it would have to drop, not a generic engine fault. (A 32-wide head region is what the quantized attention kernel needs to open at all.)
        string quantPath = Track(SyntheticLlamaGguf.Write(ModelName, withReasoningMarkers: false, QuantFormat.F32, new SyntheticLlamaGguf.Geometry(2, 64, 2, 1, 64)));

        await using EmbeddingServer server = await EmbeddingServer.StartAsync(
            quantPath, pooling: "mean", normalize: false, kvBlocks: 8, quantizedKv: true);

        (HttpResponseMessage response, JsonElement body) = await PostEmbeddingsAsync(server, "\"hello\"");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        JsonElement error = body.GetProperty("error");
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        Assert.Equal("encoder_unsupported", error.GetProperty("code").GetString());
        Assert.Contains(ModelName, error.GetProperty("message").GetString());
        Assert.Contains("kv.quant", error.GetProperty("message").GetString());
    }

    // ---------------------------------------------------------------- advertised posture and request-shape refusals

    [Fact]
    public async Task Models_CarriesPoolingPostureOnlyForEncoderModels()
    {
        string path = NewModel();

        await using EmbeddingServer encoder = await EmbeddingServer.StartAsync(path, pooling: "cls", normalize: true);
        JsonElement encoderEntry = await FirstModelEntryAsync(encoder);

        Assert.Equal("embed", encoderEntry.GetProperty("type").GetString());
        Assert.Equal("cls", encoderEntry.GetProperty("pooling").GetString());
        Assert.True(encoderEntry.GetProperty("normalize").GetBoolean());

        await using EmbeddingServer sampler = await EmbeddingServer.StartAsync(path, type: "chat", pooling: "mean", normalize: false);
        JsonElement samplerEntry = await FirstModelEntryAsync(sampler);

        Assert.Equal("chat", samplerEntry.GetProperty("type").GetString());

        // A sampling model answers nothing about sentence-vector posture, so its entry stays exactly as it was: no new keys.
        Assert.False(samplerEntry.TryGetProperty("pooling", out _));
        Assert.False(samplerEntry.TryGetProperty("normalize", out _));
    }

    [Fact]
    public async Task Embeddings_MissingInputEmptyInputAndUnknownModel_AnswerOpenAiRefusals()
    {
        await using EmbeddingServer server = await EmbeddingServer.StartAsync(NewModel(), pooling: "mean", normalize: false);

        (HttpResponseMessage absent, JsonElement absentError) = await PostJsonAsync(server, "/v1/embeddings", """{"input":null}""");
        Assert.Equal(HttpStatusCode.BadRequest, absent.StatusCode);
        Assert.Equal("invalid_request", absentError.GetProperty("error").GetProperty("code").GetString());
        Assert.Contains("'input'", absentError.GetProperty("error").GetProperty("message").GetString());

        (HttpResponseMessage empty, JsonElement emptyError) = await PostJsonAsync(server, "/v1/embeddings", """{"input":[]}""");
        Assert.Equal(HttpStatusCode.BadRequest, empty.StatusCode);
        Assert.Contains("non-empty 'input'", emptyError.GetProperty("error").GetProperty("message").GetString());

        (HttpResponseMessage wrongType, JsonElement wrongTypeError) = await PostJsonAsync(server, "/v1/embeddings", """{"input":42}""");
        Assert.Equal(HttpStatusCode.BadRequest, wrongType.StatusCode);
        Assert.Contains("'input' must be a string or a list of strings", wrongTypeError.GetProperty("error").GetProperty("message").GetString());

        (HttpResponseMessage unknown, JsonElement unknownError) = await PostJsonAsync(
            server, "/v1/embeddings", """{"model":"not-served","input":"hello"}""");
        Assert.Equal(HttpStatusCode.NotFound, unknown.StatusCode);
        Assert.Equal("model_not_found", unknownError.GetProperty("error").GetProperty("code").GetString());
    }

    // ---------------------------------------------------------------- encoder-lane admission posture

    [Fact]
    public async Task Embeddings_HeldEncoderLaneRefusesSubmission_AndBoundedQueueAbsorbsTheSameContention()
    {
        string path = NewModel();
        string wideInput = string.Concat(Enumerable.Repeat("world! ", 60)); // 120 positions: a forward wide enough for arrivals to overlap it

        // Nothing queues behind a held lane: contention is the bounded-overload refusal, naming the lane and the posture that would let a submission wait.
        for (int attempt = 1; attempt <= 8; attempt++)
        {
            await using EmbeddingServer immediate = await EmbeddingServer.StartAsync(
                path, pooling: "mean", normalize: false, kvBlocks: 16);

            HttpResponseMessage[] responses = await PostConcurrentlyAsync(immediate, wideInput, requests: 16);

            HttpResponseMessage[] refused = [.. responses.Where(response => response.StatusCode == HttpStatusCode.ServiceUnavailable)];
            if (refused.Length == 0)
            {
                continue; // this attempt's arrivals never overlapped a held forward: fresh state, try again
            }

            foreach (HttpResponseMessage ok in responses.Where(response => response.StatusCode == HttpStatusCode.OK))
            {
                ok.EnsureSuccessStatusCode();
            }

            using JsonDocument document = JsonDocument.Parse(await refused[0].Content.ReadAsStringAsync());
            JsonElement error = document.RootElement.GetProperty("error");
            Assert.Equal("engine_overloaded_error", error.GetProperty("type").GetString());
            Assert.Equal("engine_overloaded", error.GetProperty("code").GetString());
            Assert.Contains("encoder lane", error.GetProperty("message").GetString());
            Assert.Contains(ModelName, error.GetProperty("message").GetString());

            Assert.True(immediate.Engine.Metrics!.AdmissionRefusalsTotal >= 1);
            Assert.Contains(
                immediate.Engine.Metrics.RecentAdmissionRefusals(),
                entry => entry.Kind == AdmissionRefusalKinds.ConcurrencyQuota && entry.Model == ModelName);

            // The same arrivals against a declared depth: every submission either takes the lane or waits for it, so nothing is refused.
            await using EmbeddingServer bounded = await EmbeddingServer.StartAsync(
                path, pooling: "mean", normalize: false, kvBlocks: 16, maxQueuedRequests: 16, queueTimeoutMs: 60_000);

            HttpResponseMessage[] absorbed = await PostConcurrentlyAsync(bounded, wideInput, requests: 16);

            Assert.Equal(0, absorbed.Count(response => response.StatusCode == HttpStatusCode.ServiceUnavailable));
            foreach (HttpResponseMessage response in absorbed)
            {
                response.EnsureSuccessStatusCode();
            }

            Assert.Equal(0, bounded.Engine.Metrics!.AdmissionRefusalsTotal); // a queue seat is not a refusal
            return;
        }

        Assert.Fail("no embedding submission was refused against a held encoder lane across 8 attempts — arrivals never overlapped a running forward");
    }

    [Fact]
    public async Task Embeddings_KvCapacityShortfall_RefusesUpFront_NamingTheEncoderPositionsItNeeds()
    {
        await using EmbeddingServer server = await EmbeddingServer.StartAsync(
            NewModel(), pooling: "mean", normalize: false, kvBlocks: 8); // 128 slots total: eight 120-position inputs need eight blocks apiece

        // Each input clears the per-input context ceiling on its own; it is the call's whole batch that outruns what the pool can back.
        string wideInput = string.Concat(Enumerable.Repeat("world! ", 60));
        string inputs = $$"""["{{string.Join("\",\"", Enumerable.Repeat(wideInput, 8))}}"]""";

        (HttpResponseMessage response, JsonElement body) = await PostEmbeddingsAsync(server, inputs);

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        JsonElement error = body.GetProperty("error");
        Assert.Equal("engine_overloaded_error", error.GetProperty("type").GetString());
        Assert.Equal("engine_overloaded", error.GetProperty("code").GetString());
        Assert.Contains("kv_capacity", error.GetProperty("message").GetString());

        // The refusal names both sides of the shortfall: whole blocks for this call's positions against what the pool can back right now.
        string message = error.GetProperty("message").GetString()!;
        Match neededMatch = Regex.Match(message, @"(?<blocks>\d+) KV blocks back");
        Assert.True(neededMatch.Success, $"refusal does not name the blocks it needs: {message}");
        Assert.True(int.Parse(neededMatch.Groups["blocks"].Value) > 8, $"refusal names {neededMatch.Groups["blocks"].Value} blocks needed against an 8-block pool");
        Assert.Contains("only 8 are free", message);
        Assert.Contains(ModelName, error.GetProperty("message").GetString());

        Assert.Contains(
            server.Engine.Metrics!.RecentAdmissionRefusals(),
            entry => entry.Kind == AdmissionRefusalKinds.KvCapacity && entry.Model == ModelName);

        // The refusal spent nothing: the pool still backs a normal-sized embedding call.
        await SingleVectorAsync(server, "hello");
    }

    // ---------------------------------------------------------------- shared helpers

    private string NewModel() => Track(SyntheticLlamaGguf.Write(ModelName));

    // SyntheticLlamaGguf writes the file itself; this only remembers it for cleanup.
    private string Track(string path)
    {
        _tempFiles.Add(path);
        return path;
    }

    // The caller hands over the raw JSON spelling of the `input` value — one string or a list of strings — and this wraps it in the model field every request carries.
    private static async Task<(HttpResponseMessage, JsonElement)> PostEmbeddingsAsync(EmbeddingServer server, string inputValueJson) =>
        await PostJsonAsync(server, "/v1/embeddings", $$"""{"model":"{{server.ModelName}}","input":{{inputValueJson}}}""");

    private static async Task<(HttpResponseMessage, JsonElement)> PostJsonAsync(EmbeddingServer server, string path, string jsonBody)
    {
        HttpResponseMessage response = await server.Http.PostAsync(
            server.BaseUrl + path, new StringContent(jsonBody, Encoding.UTF8, "application/json"));

        using JsonDocument document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        return (response, document.RootElement.Clone()); // detached: assertions outlive the parsed document
    }

    private static async Task<HttpResponseMessage[]> PostConcurrentlyAsync(EmbeddingServer server, string input, int requests)
    {
        string body = $$"""{"model":"{{server.ModelName}}","input":{{JsonSerializer.Serialize(input)}}}""";

        return await Task.WhenAll(Enumerable.Range(0, requests).Select(_ => Task.Run(async () =>
        {
            using HttpRequestMessage request = new(HttpMethod.Post, server.BaseUrl + "/v1/embeddings")
            {
                Content = new StringContent(body, Encoding.UTF8, "application/json"),
            };

            return await server.Http.SendAsync(request);
        })));
    }

    private static async Task<float[]> SingleVectorAsync(EmbeddingServer server, string input)
    {
        (HttpResponseMessage response, JsonElement body) = await PostEmbeddingsAsync(
            server, JsonSerializer.Serialize(input));

        Assert.True(response.IsSuccessStatusCode, $"the embeddings request was refused: {body.GetRawText()}");
        return ReadVector(body.GetProperty("data")[0]);
    }

    private static async Task<JsonElement> FirstModelEntryAsync(EmbeddingServer server)
    {
        HttpResponseMessage response = await server.Http.GetAsync(server.BaseUrl + "/v1/models");
        response.EnsureSuccessStatusCode();

        using JsonDocument document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        return document.RootElement.GetProperty("data")[0].Clone();
    }

    private static float[] ReadVector(JsonElement datum)
    {
        JsonElement embedding = datum.GetProperty("embedding");
        float[] vector = new float[embedding.GetArrayLength()];

        int i = 0;
        foreach (JsonElement value in embedding.EnumerateArray())
        {
            vector[i++] = value.GetSingle();
        }

        return vector;
    }

    private static double L2Norm(ReadOnlySpan<float> values)
    {
        double sum = 0;
        foreach (float v in values)
        {
            sum += (double)v * v;
        }

        return Math.Sqrt(sum);
    }

    private static float MaxAbs(ReadOnlySpan<float> values)
    {
        float max = 0f;
        foreach (float v in values)
        {
            max = Math.Max(max, Math.Abs(v));
        }

        return max;
    }

    private static void AssertClose(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{context}: slot {i} is not finite");
            Assert.True(
                Math.Abs(actual[i] - expected[i]) <= 1e-5,
                $"{context}: slot {i} expected {expected[i]:G9}, got {actual[i]:G9}");
        }

        Assert.True(MaxAbs(expected) > 1e-6f, $"{context}: the reference vector looks degenerate");
    }

    /// <summary>One synthetic engine on an ephemeral port, opened in the serving posture the embeddings endpoint answers with.</summary>
    private sealed class EmbeddingServer : IAsyncDisposable
    {
        public GenerationService Engine { get; }
        public WebApplication App { get; }
        public HttpClient Http { get; } = new();
        public string BaseUrl { get; }
        public string ModelName { get; }

        private EmbeddingServer(GenerationService engine, WebApplication app, string modelName)
        {
            Engine = engine;
            App = app;
            ModelName = modelName;
            BaseUrl = app.Urls.Single();
        }

        public static async Task<EmbeddingServer> StartAsync(
            string modelPath,
            string pooling,
            bool normalize,
            string type = "embed",
            int kvBlocks = 8,
            int? maxQueuedRequests = null,
            int? queueTimeoutMs = null,
            bool quantizedKv = false)
        {
            LairConfig config = new()
            {
                Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
                Engine = new EngineConfig { Threads = 1 },
                Models =
                [
                    new ModelConfig
                    {
                        Name = EmbeddingApiTests.ModelName,
                        Path = modelPath,
                        Type = type,
                        Pooling = pooling,
                        Normalize = normalize,
                        MaxContextTokens = 512,
                        KvBlocks = kvBlocks,
                        MaxQueuedRequests = maxQueuedRequests,
                        QueueTimeoutMs = queueTimeoutMs,
                        KvQuant = quantizedKv
                            ? new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0)
                            : KvFormatPair.Default,
                    },
                ],
            };

            GenerationService engine = GenerationService.Open(config);
            try
            {
                WebApplication app = ApiApp.Build(config, engine);
                await ((IHost)app).StartAsync(CancellationToken.None);
                return new EmbeddingServer(engine, app, EmbeddingApiTests.ModelName);
            }
            catch
            {
                engine.Dispose();
                throw;
            }
        }

        public async ValueTask DisposeAsync()
        {
            Http.Dispose();
            await ((IHost)App).StopAsync(TimeSpan.FromSeconds(5));
            await App.DisposeAsync();
            Engine.Dispose();
        }
    }
}

/// <summary>
/// The sentence-vector posture as a live-editable knob: both keys apply as new-requests-only changes, the advertised entry and
/// the engine's own posture facts follow the edit without a restart, later requests answer with the new posture, and what a
/// model's type is stays a restart-only fact beside them.
/// </summary>
[Trait("Category", "Embeddings")]
[Trait("Category", "HotReload")]
public sealed class EmbeddingPostureReloadTests : IDisposable
{
    private const int Hidden = 4;
    private const string ModelName = "synthetic-embed-reload";

    private readonly List<string> _tempFiles = [];

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
            }
        }

        _tempFiles.Clear();
    }

    [Fact]
    public async Task PoolingAndNormalizeHotEdit_AppliesForLaterRequests_AndReadvertisesWithoutRestart()
    {
        string path = WriteModel();

        await using EmbeddingHarness server = await EmbeddingHarness.StartAsync(path, pooling: "mean", normalize: false);

        float[] before = await SingleVectorAsync(server, "hello world !");
        JsonElement beforeEntry = await FirstModelEntryAsync(server);
        Assert.Equal("mean", beforeEntry.GetProperty("pooling").GetString());
        Assert.False(beforeEntry.GetProperty("normalize").GetBoolean());

        IReadOnlyList<FieldResult> applied = server.Engine.ApplyConfiguration(
            server.Engine.EffectiveConfiguration with
            {
                Models = [server.Engine.EffectiveConfiguration.Models[0] with { Pooling = "cls", Normalize = true }],
            });

        FieldResult pooling = Assert.Single(applied, result => result.Change.Field == "models[0].pooling");
        Assert.Equal(ConfigApplyOutcome.Applied, pooling.Outcome);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, pooling.Change.MutationClass); // what a later request answers with, not what any forward computed

        FieldResult normalize = Assert.Single(applied, result => result.Change.Field == "models[0].normalize");
        Assert.Equal(ConfigApplyOutcome.Applied, normalize.Outcome);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, normalize.Change.MutationClass);

        Assert.Equal("cls", server.Engine.PoolingMode);
        Assert.True(server.Engine.NormalizesPooledVectors);

        JsonElement afterEntry = await FirstModelEntryAsync(server);
        Assert.Equal("cls", afterEntry.GetProperty("pooling").GetString()); // the advertised posture follows the live edit
        Assert.True(afterEntry.GetProperty("normalize").GetBoolean());

        float[] after = await SingleVectorAsync(server, "hello world !"); // a request admitted after the edit answers with the new posture
        Assert.Equal(Hidden, after.Length);
        Assert.True(Math.Abs(L2Norm(after) - 1.0) <= 1e-5, $"post-edit vector has L2 norm {L2Norm(after):G9}, expected ~1");
        Assert.True(!after.SequenceEqual(before), "the post-edit posture answered the same vector the old posture produced");
    }

    [Fact]
    public async Task ModelTypeHotEdit_StaysRestartOnly_BesideTheNewRequestsOnlyPostureKeys()
    {
        string path = WriteModel();

        await using EmbeddingHarness server = await EmbeddingHarness.StartAsync(path, pooling: "mean", normalize: false);

        IReadOnlyList<FieldResult> applied = server.Engine.ApplyConfiguration(
            server.Engine.EffectiveConfiguration with
            {
                Models = [server.Engine.EffectiveConfiguration.Models[0] with { Type = "chat", Pooling = "cls" }],
            });

        FieldResult type = Assert.Single(applied, result => result.Change.Field == "models[0].type");
        Assert.Equal(ConfigMutationClass.RestartOnly, type.Change.MutationClass); // what a model is opened as is decided when it opens
        Assert.Equal(ConfigApplyOutcome.Refused, type.Outcome);

        FieldResult pooling = Assert.Single(applied, result => result.Change.Field == "models[0].pooling");
        Assert.Equal(ConfigApplyOutcome.Applied, pooling.Outcome); // the response-shape keys apply while the type edit is refused

        Assert.Equal("embed", server.Engine.ModelType); // the engine stays on the posture it opened with
        Assert.Equal("cls", server.Engine.PoolingMode);
    }

    // ---------------------------------------------------------------- shared helpers

    private string WriteModel()
    {
        string path = SyntheticLlamaGguf.Write(ModelName);
        _tempFiles.Add(path);
        return path;
    }

    private static async Task<(HttpResponseMessage, JsonElement)> PostEmbeddingsAsync(EmbeddingHarness server, string inputValueJson)
    {
        HttpResponseMessage response = await server.Http.PostAsync(
            server.BaseUrl + "/v1/embeddings",
            new StringContent($$"""{"model":"{{server.ModelName}}","input":{{inputValueJson}}}""", Encoding.UTF8, "application/json"));

        using JsonDocument document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        return (response, document.RootElement.Clone());
    }

    private static async Task<float[]> SingleVectorAsync(EmbeddingHarness server, string input)
    {
        (HttpResponseMessage response, JsonElement body) = await PostEmbeddingsAsync(
            server, JsonSerializer.Serialize(input));

        response.EnsureSuccessStatusCode();

        JsonElement embedding = body.GetProperty("data")[0].GetProperty("embedding");        float[] vector = new float[embedding.GetArrayLength()];

        int i = 0;
        foreach (JsonElement value in embedding.EnumerateArray())
        {
            vector[i++] = value.GetSingle();
        }

        return vector;
    }

    private static async Task<JsonElement> FirstModelEntryAsync(EmbeddingHarness server)
    {
        HttpResponseMessage response = await server.Http.GetAsync(server.BaseUrl + "/v1/models");
        response.EnsureSuccessStatusCode();

        using JsonDocument document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        return document.RootElement.Clone().GetProperty("data")[0].Clone();
    }

    private static double L2Norm(ReadOnlySpan<float> values)
    {
        double sum = 0;
        foreach (float v in values)
        {
            sum += (double)v * v;
        }

        return Math.Sqrt(sum);
    }

    /// <summary>One synthetic encoder-posture engine opened for a live-edit run.</summary>
    private sealed class EmbeddingHarness : IAsyncDisposable
    {
        public GenerationService Engine { get; }
        public WebApplication App { get; }
        public HttpClient Http { get; } = new();
        public string BaseUrl { get; }
        public string ModelName { get; }

        private EmbeddingHarness(GenerationService engine, WebApplication app, string modelName)
        {
            Engine = engine;
            App = app;
            ModelName = modelName;
            BaseUrl = app.Urls.Single();
        }

        public static async Task<EmbeddingHarness> StartAsync(string modelPath, string pooling, bool normalize)
        {
            LairConfig config = new()
            {
                Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
                Engine = new EngineConfig { Threads = 1 },
                Models =
                [
                    new ModelConfig
                    {
                        Name = EmbeddingPostureReloadTests.ModelName,
                        Path = modelPath,
                        Type = "embed",
                        Pooling = pooling,
                        Normalize = normalize,
                        MaxContextTokens = 512,
                        KvBlocks = 8,
                    },
                ],
            };

            GenerationService engine = GenerationService.Open(config);
            try
            {
                WebApplication app = ApiApp.Build(config, engine);
                await ((IHost)app).StartAsync(CancellationToken.None);
                return new EmbeddingHarness(engine, app, EmbeddingPostureReloadTests.ModelName);
            }
            catch
            {
                engine.Dispose();
                throw;
            }
        }

        public async ValueTask DisposeAsync()
        {
            Http.Dispose();
            await ((IHost)App).StopAsync(TimeSpan.FromSeconds(5));
            await App.DisposeAsync();
            Engine.Dispose();
        }
    }
}
