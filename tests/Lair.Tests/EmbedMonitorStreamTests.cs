using System.Diagnostics;
using System.Net.Http;
using System.Text;
using System.Text.Json;

using Lair.Http;
using Lair.Http.Monitoring;
using Lair.Host.Config;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;

namespace Lair.Tests;

/// <summary>
/// The multi-instance monitor stream over the flagship side-by-side posture: a sampling model and an encoder-only model
/// served by one registry, with the stream attached while both take load. A model opened as <c>embed</c> advances no
/// scheduler sequence at all, so its snapshots exist only because the facade writes encode-side activity around each
/// encoder call — these facts prove a live embedding request is visible on <c>/monitor/stream</c> at all, carrying the
/// embed model's own id, its resolved route/device facts, and non-zero encode-side counters, while the sampling engine
/// beside it keeps reporting zero for work it never ran. Ids stay host-unique across both engines because every bus a
/// registry attaches publishes through one host event line. Synthetic tiny models keep both serves cheap; no real
/// weights are opened, and each embedding goes to its own engine's single encoder lane so nothing here contends on it.
/// </summary>
[Trait("Category", "Embeddings")]
[Trait("Category", "MonitorStream")]
public sealed class EmbedMonitorStreamTests : IDisposable
{
    private const string ChatModel = "gpu-coding";
    private const string EmbedModel = "cpu-embed";

    private readonly List<string> _tempPaths = [];

    public void Dispose()
    {
        foreach (string path in _tempPaths)
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

        _tempPaths.Clear();
    }

    [Fact]
    public async Task MonitorStream_CarriesEncoderEngineSnapshots_SideBySideWithSamplingEngine()
    {
        using CancellationTokenSource deadline = new(TimeSpan.FromSeconds(150));

        await using Serve serve = await Serve.StartAsync(this);
        await using Stream body = await serve.OpenStreamAsync(deadline.Token);
        StreamReader reader = new(body);

        // The stream is pumped from the moment it attaches, exactly as an operator's client reads it: frames land in a
        // shared collection while load runs against both engines concurrently on their own encoder/decode lanes.
        FramePump pump = FramePump.Start(reader);

        for (int round = 0; round < 3 && !deadline.IsCancellationRequested; round++)
        {
            await Task.WhenAll(
                serve.PostEmbeddingsAsync(EmbedModel, ["hello", "world! world", "hello world !"], deadline.Token),
                serve.PostChatAsync(ChatModel, deadline.Token));

            await Task.Delay(TimeSpan.FromMilliseconds(400), CancellationToken.None); // let each engine's snapshot tick land inside its own window
        }

        IReadOnlyDictionary<string, List<SseFrame>> snapshots = await pump.WaitForSnapshotsByModelAsync(
            [EmbedModel, ChatModel], requiredPerModel: 1, deadline.Token);

        SseFrame embedSnapshot = snapshots[EmbedModel][0];
        SseFrame chatSnapshot = snapshots[ChatModel][0];

        Assert.True(embedSnapshot.Id > 0 && chatSnapshot.Id > 0, "a snapshot frame carried no server-scoped id");
        Assert.NotEqual(embedSnapshot.Id, chatSnapshot.Id); // host-unique ids: the two engines never repeat one another's

        AssertEmbedFacts(embedSnapshot);
        AssertSamplingEngineCarriesNoEncoderWork(chatSnapshot);
    }

    [Fact]
    public async Task SnapshotIds_StayHostUnique_AcrossEngines_ServingInterleavedLoads()
    {
        using CancellationTokenSource deadline = new(TimeSpan.FromSeconds(150));

        await using Serve serve = await Serve.StartAsync(this);
        await using Stream body = await serve.OpenStreamAsync(deadline.Token);
        StreamReader reader = new(body);

        // Alternating load on the two engines while this stream is pumped, each round separated past the host's snapshot cadence so every
        // burst gets its own closing tick: a burst that ends between ticks still publishes once for that window, and quiet windows publish nothing.
        FramePump pump = FramePump.Start(reader);

        for (int round = 0; round < 3 && !deadline.IsCancellationRequested; round++)
        {
            await serve.PostEmbeddingsAsync(EmbedModel, ["hello world"], deadline.Token);
            await serve.PostChatAsync(ChatModel, deadline.Token);
            await Task.Delay(TimeSpan.FromSeconds(3), CancellationToken.None); // comfortably past the 2s snapshot cadence
        }

        List<SseFrame> collected = await pump.WaitForSnapshotCountAsync(4, deadline.Token);

        Dictionary<long, string> idsSeen = []; // id -> model id: a repeated id across frames is the collision this asserts against
        HashSet<string> modelsSeen = [];

        foreach (SseFrame frame in collected)
        {
            long id = frame.Id ?? throw new InvalidOperationException($"a snapshot frame carried no server-scoped id: {frame.Data}");
            string model = ModelOf(frame)!;

            Assert.True(idsSeen.TryAdd(id, model), $"event id {id} repeated across engines ({idsSeen[id]} and {model})");
            modelsSeen.Add(model);
        }

        Assert.True(idsSeen.Count >= 4, $"expected at least four distinct host ids across the two engines, got {idsSeen.Count}"); // every frame drew from one host event line and none repeated an id
        Assert.Contains(EmbedModel, modelsSeen); // the encoder model produced its own frame on this stream
    }

    // ---------------------------------------------------------------- snapshot fact assertions

    // The embed model's own frame must speak the embed engine: its id, its posture, its resolved route/device facts, and a
    // non-zero encode-side tally — while every decode-side field stays at zero because nothing there decoded.
    private static void AssertEmbedFacts(SseFrame snapshot)
    {
        using JsonDocument document = JsonDocument.Parse(snapshot.Data);
        JsonElement root = document.RootElement;

        Assert.Equal(EmbedModel, root.GetProperty("metrics_model").GetString());
        Assert.Equal(snapshot.Id!.Value, root.GetProperty("metrics_event_seq").GetInt64());
        Assert.Equal("embed", root.GetProperty("metrics_model_type").GetString());

        Assert.True(
            root.TryGetProperty("metrics_route_segments", out JsonElement segments) && segments.ValueKind == JsonValueKind.Array && segments.GetArrayLength() > 0,
            $"snapshot for '{EmbedModel}' must carry its resolved route: {snapshot.Data}");

        JsonElement segment = segments[0];
        Assert.False(string.IsNullOrEmpty(segment.GetProperty("device_id").GetString()));
        Assert.False(string.IsNullOrEmpty(segment.GetProperty("weight_tier").GetString()));

        // The encode-side facts are what this engine's snapshots exist for: a completed embedding forward, the positions it
        // carried, and no in-flight call left over once its request returned.
        Assert.True(
            root.GetProperty("metrics_encoder_requests_total").GetInt64() >= 1,
            $"the encoder model must report at least one completed embedding request: {snapshot.Data}");
        Assert.True(
            root.GetProperty("metrics_encoder_tokens_forwarded_total").GetInt64() >= 1,
            $"the encoder model must credit the positions its forwards carried: {snapshot.Data}");

        // Decode-side fields keep reading what actually happened on this engine — nothing at all.
        Assert.Equal(0L, root.GetProperty("metrics_decode_steps_total").GetInt64());
        Assert.Equal(0, root.GetProperty("metrics_active_sequences").GetInt32());
    }

    // The additive encode-side fields must not leak across engines: the sampling model beside the encoder model ran no
    // encoder pass, so it reports those same fields at their inert defaults while its own decode facts stay non-zero.
    private static void AssertSamplingEngineCarriesNoEncoderWork(SseFrame snapshot)
    {
        using JsonDocument document = JsonDocument.Parse(snapshot.Data);
        JsonElement root = document.RootElement;

        Assert.Equal(ChatModel, root.GetProperty("metrics_model").GetString());

        // The additive encode-side fields are present on every snapshot shape now, but a model opened for sampling runs no encoder pass: they read their inert defaults here.
        Assert.True(
            root.TryGetProperty("metrics_active_encoder_requests", out JsonElement activeEncoders) && activeEncoders.GetInt32() == 0
                && root.GetProperty("metrics_encoder_requests_total").GetInt64() == 0,
            $"a sampling model must not report encoder-lane work: {snapshot.Data}");

        Assert.True(
            root.GetProperty("metrics_model_type").GetString() != "embed",
            $"the sampling engine's snapshot must name its own posture: {snapshot.Data}");
    }

    // ---------------------------------------------------------------- SSE reading helpers (same frame contract MonitorMultiInstanceTests reads with)

    private static string? ModelOf(SseFrame frame)
    {
        try
        {
            using JsonDocument document = JsonDocument.Parse(frame.Data);
            return document.RootElement.TryGetProperty("metrics_model", out JsonElement model) ? model.GetString() : null;
        }
        catch (JsonException)
        {
            return null; // keep-alive comments and non-snapshot frames carry no model key
        }
    }

    private sealed record SseFrame(long? Id, string Event, string Data);

    /// <summary>
    /// Reads the live stream on its own task from attach time onward — what an operator's client does while load runs — and
    /// keeps every snapshot frame it sees, keyed by the model id its payload names. A test then waits on this collection
    /// instead of reading frames inline, so a busy engine cannot starve the assertion path behind request-lifecycle traffic.
    /// </summary>
    private sealed class FramePump
    {
        private readonly List<SseFrame> _snapshots = []; // guarded by _gate; published under it and copied out on read
        private readonly object _gate = new();

        private FramePump()
        {
        }

        public static FramePump Start(StreamReader reader)
        {
            FramePump pump = new();

            _ = Task.Run(async () =>
            {
                try
                {
                    while (true)
                    {
                        SseFrame? frame = await ReadNextFrameAsync(reader);
                        if (frame is null)
                        {
                            return; // stream closed: nothing more to collect
                        }

                        if (frame.Event == MonitorEventNames.Snapshot && ModelOf(frame) is not null)
                        {
                            lock (pump._gate)
                            {
                                pump._snapshots.Add(frame);
                            }
                        }
                    }
                }
                catch (Exception)
                {
                    // A closed or aborted stream ends the pump; the waiting test reports what it did collect.
                }
            });

            return pump;
        }

        /// <summary>Waits until the pump has collected at least <paramref name="requiredPerModel"/> snapshots for every listed model.</summary>
        public async Task<IReadOnlyDictionary<string, List<SseFrame>>> WaitForSnapshotsByModelAsync(
            IReadOnlyList<string> models, int requiredPerModel, CancellationToken ct)
        {
            Stopwatch clock = Stopwatch.StartNew();

            while (true)
            {
                Dictionary<string, List<SseFrame>> keyed = KeyedByModel();

                if (models.All(model => keyed.TryGetValue(model, out List<SseFrame>? frames) && frames.Count >= requiredPerModel))
                {
                    return keyed;
                }

                ct.ThrowIfCancellationRequested();

                if (clock.Elapsed > TimeSpan.FromSeconds(120))
                {
                    Assert.Fail($"no snapshot frame per model within {clock.Elapsed}: collected {Describe(keyed)}");
                }

                await Task.Delay(50, CancellationToken.None);
            }
        }

        /// <summary>Waits until the pump has collected at least <paramref name="requiredTotal"/> snapshot frames from any engines.</summary>
        public async Task<List<SseFrame>> WaitForSnapshotCountAsync(int requiredTotal, CancellationToken ct)
        {
            Stopwatch clock = Stopwatch.StartNew();

            while (true)
            {
                List<SseFrame> all = SnapshotFrames();

                if (all.Count >= requiredTotal)
                {
                    return all;
                }

                ct.ThrowIfCancellationRequested();

                if (clock.Elapsed > TimeSpan.FromSeconds(120))
                {
                    Assert.Fail($"only {all.Count} snapshot frames collected within {clock.Elapsed}: {Describe(all)}");
                }

                await Task.Delay(50, CancellationToken.None);
            }
        }

        private static string Describe(IReadOnlyDictionary<string, List<SseFrame>> keyed) =>
            string.Join(", ", keyed.Select(pair => $"{pair.Key} x{pair.Value.Count}"));

        private static string Describe(IReadOnlyList<SseFrame> frames) =>
            string.Join(" | ", frames.Select(frame => $"{ModelOf(frame)}@{frame.Id}"));

        private Dictionary<string, List<SseFrame>> KeyedByModel()
        {
            Dictionary<string, List<SseFrame>> keyed = new(StringComparer.Ordinal);

            foreach (SseFrame frame in SnapshotFrames())
            {
                string model = ModelOf(frame)!;
                if (!keyed.TryGetValue(model, out List<SseFrame>? frames))
                {
                    frames = [];
                    keyed[model] = frames;
                }

                frames.Add(frame);
            }

            return keyed;
        }

        private List<SseFrame> SnapshotFrames()
        {
            lock (_gate)
            {
                return [.. _snapshots];
            }
        }
    }

    // One SSE frame per blank-line dispatch: id:/event:/data: fields accumulate, comment lines are ignored.
    private static async Task<SseFrame?> ReadNextFrameAsync(StreamReader reader)
    {
        long? id = null;
        string eventName = "message";
        StringBuilder data = new();

        while (true)
        {
            string? line = await reader.ReadLineAsync(CancellationToken.None);
            if (line is null)
            {
                return null; // stream ended mid-frame or clean: no more frames either way
            }

            if (line.Length == 0)
            {
                if (data.Length > 0)
                {
                    return new SseFrame(id, eventName, data.ToString());
                }

                continue; // bare separator after a comment frame carries nothing
            }

            if (line[0] == ':')
            {
                continue; // ": connected" / ": ping" keep-alive comments
            }

            int colon = line.IndexOf(':');
            if (colon <= 0)
            {
                continue; // malformed field line: skip it, the stream contract survives
            }

            string field = line[..colon];
            string value = line[(colon + 1)..].TrimStart();

            switch (field)
            {
                case "id":
                    id = long.TryParse(value, out long parsedId) ? parsedId : null;
                    break;
                case "event":
                    eventName = value;
                    break;
                case "data":
                    if (data.Length > 0)
                    {
                        data.Append('\n');
                    }

                    data.Append(value);
                    break;
            }
        }
    }

    /// <summary>One registry-backed two-model serve: a sampling model and an encoder-only model on an ephemeral loopback port.</summary>
    private sealed class Serve : IAsyncDisposable
    {
        private static readonly HttpClient Http = new();

        public ModelRegistry Registry { get; }
        public string BaseUrl { get; }

        private readonly WebApplication _app;

        private Serve(ModelRegistry registry, WebApplication app)
        {
            Registry = registry;
            _app = app;
            BaseUrl = app.Urls.Single();
        }

        public static async Task<Serve> StartAsync(EmbedMonitorStreamTests owner)
        {
            string chatPath = SyntheticLlamaGguf.Write($"embed-monitor-{ChatModel}");
            string embedPath = SyntheticLlamaGguf.Write($"embed-monitor-{EmbedModel}");
            owner._tempPaths.Add(chatPath);
            owner._tempPaths.Add(embedPath);

            // Small single-threaded batch posture, as the multi-instance monitor suite declares it: step boundaries stay frequent so each completed load
            // publishes its closing snapshot at debug-JIT forward speed. Both models share one host event line through this registry's monitor attach.
            LairConfig config = new()
            {
                Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
                Engine = new EngineConfig { Threads = 1, MaxBatchTokens = 8 },
                Models =
                [
                    new ModelConfig { Name = ChatModel, Path = chatPath, MaxContextTokens = 512, KvBlocks = 32 },
                    new ModelConfig { Name = EmbedModel, Path = embedPath, Type = "embed", Pooling = "mean", Normalize = false, MaxContextTokens = 512, KvBlocks = 16 },
                ],
            };

            ModelRegistry registry = ModelRegistry.Open(config);
            try
            {
                WebApplication app = ApiApp.Build(config, registry);
                await ((IHost)app).StartAsync(CancellationToken.None);
                return new Serve(registry, app);
            }
            catch
            {
                registry.Dispose();
                throw;
            }
        }

        public async Task<Stream> OpenStreamAsync(CancellationToken ct)
        {
            using HttpRequestMessage request = new(HttpMethod.Get, BaseUrl + "/monitor/stream");
            HttpResponseMessage response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
            Assert.True(response.IsSuccessStatusCode, $"stream request answered {(int)response.StatusCode}");

            return await response.Content.ReadAsStreamAsync(ct);
        }

        public async Task PostChatAsync(string model, CancellationToken ct)
        {
            string body = JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["model"] = model,
                ["messages"] = new[] { new { role = "user", content = "hello world" } },
                ["max_tokens"] = 8,
                ["temperature"] = 0f,
                ["stream"] = false,
            });

            using HttpResponseMessage response = await Http.PostAsync(BaseUrl + "/v1/chat/completions", new StringContent(body, Encoding.UTF8, "application/json"), ct);
            string payload = await response.Content.ReadAsStringAsync(ct);
            Assert.True(response.IsSuccessStatusCode, $"the chat request was refused ({(int)response.StatusCode}): {payload}");
        }

        public async Task PostEmbeddingsAsync(string model, IReadOnlyList<string> inputs, CancellationToken ct)
        {
            string body = $$"""{"model":"{{model}}","input":[{{string.Join(",", inputs.Select(input => JsonSerializer.Serialize(input)))}}]}""";

            using HttpResponseMessage response = await Http.PostAsync(BaseUrl + "/v1/embeddings", new StringContent(body, Encoding.UTF8, "application/json"), ct);
            string payload = await response.Content.ReadAsStringAsync(ct);
            Assert.True(response.IsSuccessStatusCode, $"the embeddings request was refused ({(int)response.StatusCode}): {payload}");
        }

        public async ValueTask DisposeAsync()
        {
            await ((IHost)_app).StopAsync(TimeSpan.FromSeconds(5));
            await _app.DisposeAsync();
            Registry.Dispose();
        }
    }
}
