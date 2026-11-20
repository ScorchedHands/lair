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
/// The monitor stream over a multi-instance registry: every advertised engine's snapshot bus joins the one stream
/// through a host event line, so a two-model serve carries snapshot frames for both model ids — each speaking its
/// own model and route facts — and a reconnect from an id older than both buses' newest snapshots replays newer
/// frames from every bus, oldest-first with host-unique ids, then closes each bus's transient state with its own
/// resync. A model a reload opens later joins the monitor posture the host registered, so streams attached after
/// its load sample it too. Synthetic tiny models keep every serve cheap; the existing single-engine monitor suites
/// cover the one-bus posture unchanged.
/// </summary>
[Trait("Category", "MonitorStream")]
[Trait("Category", "Registry")]
public sealed class MonitorMultiInstanceTests : IDisposable
{
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
                // already gone; nothing depends on it.
            }
        }
    }

    [Fact]
    public async Task SnapshotFrames_CarryEveryAdvertisedModel_WithItsOwnModelAndRouteFacts()
    {
        using CancellationTokenSource deadline = new(TimeSpan.FromSeconds(120));

        await using Serve serve = await Serve.StartAsync(this, "alpha", "beta");
        await using Stream body = await serve.OpenStreamAsync(null, deadline.Token);
        StreamReader reader = new(body);

        // Both models take load while the stream is attached: each engine's closing tick publishes its snapshot onto
        // the shared subscription, so the stream must show both model ids.
        await serve.PostChatAsync("alpha", deadline.Token);
        await serve.PostChatAsync("beta", deadline.Token);

        IReadOnlyDictionary<string, SseFrame> snapshots = await ReadSnapshotsForAsync(reader, ["alpha", "beta"], deadline.Token);

        SseFrame alphaSnapshot = snapshots["alpha"];
        SseFrame betaSnapshot = snapshots["beta"];

        Assert.True(alphaSnapshot.Id > 0 && betaSnapshot.Id > 0, "a snapshot frame carried no server-scoped id");
        Assert.NotEqual(alphaSnapshot.Id, betaSnapshot.Id); // host-unique ids: two engines never repeat one

        AssertSnapshotFacts(alphaSnapshot, "alpha");
        AssertSnapshotFacts(betaSnapshot, "beta");
    }

    [Fact]
    public async Task LastEventId_ReplaysNewerSnapshots_FromEveryBus_ThenResyncsEach()
    {
        using CancellationTokenSource deadline = new(TimeSpan.FromSeconds(180));

        await using Serve serve = await Serve.StartAsync(this, "alpha", "beta");

        // First round: one completion per model with the stream attached, so each bus publishes a snapshot and the
        // test learns an id both buses have already passed.
        await using (Stream body = await serve.OpenStreamAsync(null, deadline.Token))
        {
            StreamReader reader = new(body);

            await serve.PostChatAsync("alpha", deadline.Token);
            await serve.PostChatAsync("beta", deadline.Token);

            IReadOnlyDictionary<string, SseFrame> firstRound = await ReadSnapshotsForAsync(reader, ["alpha", "beta"], deadline.Token);

            long headerId = Math.Min(firstRound["alpha"].Id!.Value, firstRound["beta"].Id!.Value); // older than both buses' newest retained snapshot

            // Let each bus's window roll so its closing snapshot for this round lands in its ring before the next
            // round starts — the reconnect must find strictly newer retained snapshots on every bus.
            await Task.Delay(TimeSpan.FromSeconds(4), deadline.Token); // comfortably past the snapshot cadence

            // Second round: every bus publishes again, strictly newer than the header this reconnect replays from.
            await serve.PostChatAsync("alpha", deadline.Token);
            await serve.PostChatAsync("beta", deadline.Token);
            await Task.Delay(TimeSpan.FromSeconds(4), deadline.Token);

            await using Stream replayBody = await serve.OpenStreamAsync(headerId.ToString(), deadline.Token);
            StreamReader replayReader = new(replayBody);

            long previousId = headerId;
            int replayed = 0;
            HashSet<string> replayedModels = [];
            List<SseFrame> resyncs = [];

            // Replay frames arrive oldest-first across buses, then one resync per bus; live-tail frames may interleave.
            while (resyncs.Count < 2 || replayedModels.Count < 2)
            {
                SseFrame frame = await ReadFramesUntilAsync(
                    replayReader,
                    f => f.Event is MonitorEventNames.Snapshot or MonitorEventNames.Resync,
                    deadline.Token);

                if (frame.Event == MonitorEventNames.Resync)
                {
                    Assert.True(frame.Id is null or 0, "resync frames are transient state and must not claim a replayable id");
                    resyncs.Add(frame);
                    continue;
                }

                Assert.True(frame.Id > previousId, $"replayed snapshot ids must climb: {previousId} -> {frame.Id}");
                previousId = frame.Id!.Value;
                replayed++;
                replayedModels.Add(ModelOf(frame) ?? throw new InvalidOperationException($"a replayed snapshot frame carried no model id: {frame.Data}"));
            }

            Assert.True(replayed >= 2, $"the reconnect replayed {replayed} snapshots — expected newer frames from both buses");
            Assert.Equal(2, resyncs.Count); // each bus's transient state closes with its own dump

            using JsonDocument document = JsonDocument.Parse(resyncs[0].Data);
            Assert.Equal(1, document.RootElement.GetProperty("events_version").GetInt32());
        }
    }

    [Fact]
    public async Task HotAddedModel_JoinsTheMonitorPosture_StreamsAttachedAfterItsLoad()
    {
        using CancellationTokenSource deadline = new(TimeSpan.FromSeconds(180));

        await using Serve serve = await Serve.StartAsync(this, "alpha");

        await serve.PostChatAsync("alpha", deadline.Token); // the served model answers before the reload touches the posture

        serve.Registry.ApplyConfiguration(serve.Config with
        {
            Models =
            [
                serve.Config.Models[0],
                new ModelConfig { Name = "beta", Path = serve.SecondPath, MaxContextTokens = 512, KvBlocks = 32 },
            ],
        });

        WaitUntil(() => serve.Registry.AdvertisedIds().Contains("beta"), "'beta' to advertise after its background open");

        // The hot addition opened with a bus: the registry replays the host's monitor attach onto every engine it
        // opens, so a stream attached now samples both engines without rebuilding the app. Streams attached before
        // the addition loaded simply never subscribed to it — that stays out of scope.
        Assert.All(serve.Registry.AdvertisedEngines(), engine => Assert.NotNull(engine.Metrics));

        await using Stream body = await serve.OpenStreamAsync(null, deadline.Token);
        StreamReader reader = new(body);

        // Both models take load while this stream is attached: the hot addition publishes its snapshot, and the
        // engine that was already serving publishes again, so the one stream must show both ids.
        await serve.PostChatAsync("beta", deadline.Token);
        await serve.PostChatAsync("alpha", deadline.Token);

        IReadOnlyDictionary<string, SseFrame> snapshots = await ReadSnapshotsForAsync(reader, ["alpha", "beta"], deadline.Token);

        Assert.NotEqual(snapshots["alpha"].Id, snapshots["beta"].Id); // the hot addition publishes through the same host event line

        AssertSnapshotFacts(snapshots["beta"], "beta");
    }

    // A snapshot frame's own payload must agree with the frame it rode: the model id, the event seq, and the resolved
    // route segments the owning engine attached.
    private static void AssertSnapshotFacts(SseFrame snapshot, string modelId)
    {
        using JsonDocument document = JsonDocument.Parse(snapshot.Data);
        JsonElement root = document.RootElement;

        Assert.Equal(modelId, root.GetProperty("metrics_model").GetString());
        Assert.Equal(snapshot.Id!.Value, root.GetProperty("metrics_event_seq").GetInt64());

        Assert.True(
            root.TryGetProperty("metrics_route_segments", out JsonElement segments) && segments.ValueKind == JsonValueKind.Array && segments.GetArrayLength() > 0,
            $"snapshot for '{modelId}' must carry its resolved route: {snapshot.Data}");

        JsonElement segment = segments[0];
        Assert.False(string.IsNullOrEmpty(segment.GetProperty("device_id").GetString()));
        Assert.False(string.IsNullOrEmpty(segment.GetProperty("weight_tier").GetString()));
    }

    // Reads snapshot frames until every requested model has produced one, keeping the first frame per model: buses
    // interleave on one stream, so a scan that keeps each model's frame is how a multi-engine stream is read.
    private static async Task<IReadOnlyDictionary<string, SseFrame>> ReadSnapshotsForAsync(
        StreamReader reader, IReadOnlyList<string> models, CancellationToken ct)
    {
        Dictionary<string, SseFrame> found = [];
        while (found.Count < models.Count)
        {
            SseFrame frame = await ReadFramesUntilAsync(reader, f => f.Event == MonitorEventNames.Snapshot, ct);
            string? model = ModelOf(frame);
            if (model is not null && models.Contains(model) && !found.ContainsKey(model))
            {
                found[model] = frame;
            }
        }

        return found;
    }

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

    // Reads SSE frames until one matches — comments and unmatched frames are skipped; a closed stream fails the wait.
    private static async Task<SseFrame> ReadFramesUntilAsync(StreamReader reader, Func<SseFrame, bool> match, CancellationToken ct)
    {
        while (true)
        {
            SseFrame? frame = await ReadNextFrameAsync(reader, ct);
            if (frame is null)
            {
                throw new InvalidOperationException("the monitor stream closed before a matching frame arrived");
            }

            if (match(frame))
            {
                return frame;
            }
        }
    }

    // One SSE frame per blank-line dispatch: id:/event:/data: fields accumulate, comment lines are ignored.
    private static async Task<SseFrame?> ReadNextFrameAsync(StreamReader reader, CancellationToken ct)
    {
        long? id = null;
        string eventName = "message";
        StringBuilder data = new();

        while (true)
        {
            string? line = await reader.ReadLineAsync(ct);
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

    private static void WaitUntil(Func<bool> condition, string what)
    {
        Stopwatch clock = Stopwatch.StartNew();
        while (!condition())
        {
            if (clock.Elapsed > TimeSpan.FromSeconds(30))
            {
                throw new TimeoutException($"timed out waiting for {what}");
            }

            Thread.Sleep(50);
        }
    }

    private sealed record SseFrame(long? Id, string Event, string Data);

    /// <summary>One registry-backed serve (one or two declared models) on an ephemeral loopback port.</summary>
    private sealed class Serve : IAsyncDisposable
    {
        private static readonly HttpClient Http = new();

        public ModelRegistry Registry { get; }
        public LairConfig Config { get; }
        public string SecondPath { get; }
        public string BaseUrl { get; }

        private readonly WebApplication _app;

        private Serve(ModelRegistry registry, LairConfig config, WebApplication app, string secondPath)
        {
            Registry = registry;
            Config = config;
            _app = app;
            SecondPath = secondPath;
            BaseUrl = app.Urls.Single();
        }

        public static async Task<Serve> StartAsync(MonitorMultiInstanceTests owner, params string[] modelNames)
        {
            string firstPath = SyntheticLlamaGguf.Write($"monitor-{modelNames[0]}");
            string secondPath = SyntheticLlamaGguf.Write("monitor-beta"); // declared for a two-model posture, or held for a hot addition otherwise
            owner._tempPaths.Add(firstPath);
            owner._tempPaths.Add(secondPath);

            // Small single-threaded batch posture: step boundaries stay frequent so each completed load publishes its
            // closing snapshot at debug-JIT forward speed, like the single-engine monitor suites do.
            LairConfig config = new()
            {
                Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
                Engine = new EngineConfig { Threads = 1, MaxBatchTokens = 8 },
                Models = [.. modelNames.Select(name => new ModelConfig
                {
                    Name = name,
                    Path = firstPath,
                    MaxContextTokens = 512,
                    KvBlocks = 32,
                })],
            };

            ModelRegistry registry = ModelRegistry.Open(config);
            try
            {
                WebApplication app = ApiApp.Build(config, registry);
                await ((IHost)app).StartAsync(CancellationToken.None);
                return new Serve(registry, config, app, secondPath);
            }
            catch
            {
                registry.Dispose();
                throw;
            }
        }

        public Task<Stream> OpenStreamAsync(string? lastEventId, CancellationToken ct) =>
            OpenStreamCoreAsync(BaseUrl, lastEventId, ct);

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

            using HttpResponseMessage response = await Http.PostAsync(
                BaseUrl + "/v1/chat/completions", new StringContent(body, Encoding.UTF8, "application/json"), ct);
            response.EnsureSuccessStatusCode();
        }

        private static async Task<Stream> OpenStreamCoreAsync(string baseUrl, string? lastEventId, CancellationToken ct)
        {
            HttpRequestMessage request = new(HttpMethod.Get, baseUrl + "/monitor/stream");
            if (lastEventId is not null)
            {
                request.Headers.Add("Last-Event-ID", lastEventId);
            }

            HttpResponseMessage response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
            Assert.True(response.IsSuccessStatusCode, $"stream request answered {(int)response.StatusCode}");
            return await response.Content.ReadAsStreamAsync(ct);
        }

        public async ValueTask DisposeAsync()
        {
            await ((IHost)_app).StopAsync(TimeSpan.FromSeconds(5));
            await _app.DisposeAsync();
            Registry.Dispose();
        }
    }
}
