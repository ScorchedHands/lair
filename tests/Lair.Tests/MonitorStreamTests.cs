using System.Net.Http;
using System.Text;
using System.Text.Json;
using Lair.Http;
using Lair.Http.Monitoring;
using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;
using Lair.Tokenizer;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// The versioned monitor stream (schema v1): snapshot frames with their event ids and version markers, request
/// lifecycle frames correlated with x-lair-request-id including tok/s sample series, the recent-request ring bounded
/// by ui.recent_requests (live hot-reload included), Last-Event-ID replay with resync fallback, and latest-only
/// subscriber coalescing with counted drops. Engine-driven tests use the real SmolLM fixture and skip when it is
/// absent; bus-level semantics run against direct harnesses so they assert deterministically.
/// </summary>
[Trait("Category", "MonitorUi")]
public sealed class MonitorStreamTests
{
    private static readonly HttpClient Http = new();

    // Proven multi-token greedy fixtures: the long one keeps snapshots flowing across a reconnect window, the short
    // ones keep ring-bound tests quick even at debug-JIT forward speed.
    private const string LongPrompt = "It was a dark and stormy night, and";
    private const string ShortPrompt = "Three tips for staying healthy are:";

    [Fact]
    public async Task SnapshotFrames_AreVersioned_Identified_AndAdditiveDuringLoad()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        await using LiveServer server = await LiveServer.StartAsync("smollm-uiver", uiRecentRequests: 25);

        Task<string> load = PostCompletionAsync(server, LongPrompt, maxTokens: 64);

        using CancellationTokenSource deadline = new(TimeSpan.FromSeconds(150));
        await using Stream body = await OpenStreamAsync(server, lastEventId: null, deadline.Token);
        StreamReader reader = new(body);

        SseFrame snapshot = await ReadFramesUntilAsync(reader, f => f.Event == MonitorEventNames.Snapshot, deadline.Token);

        Assert.True(snapshot.Id > 0, "the snapshot frame carried no server-scoped id");

        using JsonDocument document = JsonDocument.Parse(snapshot.Data);
        JsonElement root = document.RootElement;

        Assert.Equal(1, root.GetProperty("metrics_events_version").GetInt32());
        Assert.Equal(snapshot.Id!.Value, root.GetProperty("metrics_event_seq").GetInt64());

        // The established snapshot vocabulary survives intact — additive-only.
        foreach (string legacy in new[] { "metrics_ts", "metrics_model", "metrics_tok_s", "metrics_kv_used_blocks" })
        {
            Assert.True(root.TryGetProperty(legacy, out _), $"snapshot JSON lost legacy field '{legacy}': {snapshot.Data}");
        }

        // Every top-level property still speaks the metrics_ vocabulary.
        foreach (JsonProperty property in root.EnumerateObject())
        {
            Assert.StartsWith("metrics_", property.Name);
        }

        try
        {
            _ = await load;
        }
        catch (Exception)
        {
            // swallowed on purpose — this test only inspects the frame contract
        }
    }

    [Fact]
    public async Task RequestLifecycleFrames_CorrelateWithRequestId_AndCarryTokSeries()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        await using LiveServer server = await LiveServer.StartAsync("smollm-lifecycle", uiRecentRequests: 25);

        using CancellationTokenSource deadline = new(TimeSpan.FromSeconds(180));

        // Subscribe before generating: lifecycle frames are transient, so a late subscriber would miss them by design.
        await using Stream body = await OpenStreamAsync(server, lastEventId: null, deadline.Token);
        StreamReader reader = new(body);

        CompletionOutcome outcome = await PostCompletionDetailedAsync(server, LongPrompt, maxTokens: 40);

        // Collect every frame that carries this request's id — the three lifecycle frames arrive interleaved with
        // snapshots, so scan forward until all kinds are in hand.
        List<SseFrame> seen = [];
        SseFrame? completeFrame = null;
        while (completeFrame is null || !seen.Any(f => f.Event == MonitorEventNames.RequestStart) || !seen.Any(f => f.Event == MonitorEventNames.RequestUpdate))
        {
            SseFrame? frame = await ReadNextFrameAsync(reader, deadline.Token);
            if (frame is null)
            {
                throw new InvalidOperationException("the monitor stream closed before the request lifecycle completed on it");
            }

            if (RequestIdOf(frame) == outcome.RequestId && frame.Event is MonitorEventNames.RequestStart or MonitorEventNames.RequestUpdate or MonitorEventNames.RequestComplete)
            {
                seen.Add(frame);
                if (frame.Event == MonitorEventNames.RequestComplete)
                {
                    completeFrame = frame;
                }
            }
        }

        SseFrame complete = completeFrame!;
        Assert.True(complete.Id > 0, "the completion frame carried no server-scoped id");

        Assert.Contains(seen, f => f.Event == MonitorEventNames.RequestStart);
        Assert.Contains(seen, f => f.Event == MonitorEventNames.RequestUpdate);

        foreach (SseFrame frame in seen)
        {
            using JsonDocument document = JsonDocument.Parse(frame.Data);
            JsonElement root = document.RootElement;

            Assert.Equal(1, root.GetProperty("events_version").GetInt32());
            Assert.Equal(outcome.RequestId, root.GetProperty("request_id").GetString());
            Assert.Equal("smollm-lifecycle", root.GetProperty("model").GetString());
            Assert.Equal("completion", root.GetProperty("endpoint").GetString()); // the POST went to /v1/completions
            Assert.True(root.GetProperty("prompt_tokens").GetInt32() > 0);

            if (frame.Event != MonitorEventNames.RequestStart)
            {
                Assert.True(root.TryGetProperty("samples", out JsonElement samples) && samples.ValueKind == JsonValueKind.Array,
                    $"update/complete frames must carry the tok/s series: {frame.Data}");
            }
        }

        using (JsonDocument completeDoc = JsonDocument.Parse(complete.Data))
        {
            JsonElement root = completeDoc.RootElement;

            int generated = root.GetProperty("generated_tokens").GetInt32();
            Assert.True(generated > 1, $"expected a multi-token completion for the series assertion, got {generated}");

            JsonElement samples = root.GetProperty("samples");
            Assert.True(samples.GetArrayLength() > 0, "completion carried an empty tok/s sample series");

            JsonElement last = samples[samples.GetArrayLength() - 1];
            Assert.Equal(generated, last.GetProperty("at_token").GetInt32()); // the final point names the full token count
            Assert.True(last.GetProperty("elapsed_ms").GetInt64() >= 0);

            Assert.True(root.GetProperty("tok_s").GetDouble() > 0.0, "completion tok/s should reflect the sampled series");
            Assert.True(root.TryGetProperty("finish_reason", out JsonElement finish) && !string.IsNullOrEmpty(finish.GetString()));
            Assert.Equal(outcome.FinishReason, finish.GetString()); // stream status matches the HTTP outcome
            Assert.True(root.TryGetProperty("cached_tokens", out _), "completion frames report cached prompt tokens");
        }
    }

    [Fact]
    public async Task RequestCompleteFrame_CarriesBudgetExhaustedAttribution_TheLiveFrameSpeaksWhatTheBodyReported()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        await using LiveServer server = await LiveServer.StartAsync("smollm-budget", uiRecentRequests: 25);

        using CancellationTokenSource deadline = new(TimeSpan.FromSeconds(180));

        await using Stream body = await OpenStreamAsync(server, lastEventId: null, deadline.Token);
        StreamReader reader = new(body);

        string requestBody = JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["model"] = server.ModelName,
            ["prompt"] = ShortPrompt,
            ["max_tokens"] = 16,
            ["max_response_tokens"] = 3,
            ["temperature"] = 0f,
            ["stream"] = false,
        });

        using HttpResponseMessage response = await Http.PostAsync(
            $"{server.BaseUrl}/v1/completions", new StringContent(requestBody, Encoding.UTF8, "application/json"));
        response.EnsureSuccessStatusCode();

        string requestId = response.Headers.TryGetValues(ApiApp.RequestIdHeader, out IEnumerable<string>? values)
            ? values.FirstOrDefault() ?? throw new InvalidOperationException("the response carried no x-lair-request-id")
            : throw new InvalidOperationException("the response carried no x-lair-request-id");

        using JsonDocument responseBody = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("response", responseBody.RootElement.GetProperty("budget_exhausted").GetString());

        SseFrame complete = await ReadFramesUntilAsync(
            reader,
            f => f.Event == MonitorEventNames.RequestComplete && RequestIdOf(f) == requestId,
            deadline.Token);

        using JsonDocument document = JsonDocument.Parse(complete.Data);
        JsonElement root = document.RootElement;

        // The frame's token series can stop one short of the completion count when the combined ceiling ends a run, so this
        // asserts the attribution the completion carries, not the tally.
        Assert.Equal("response", root.GetProperty("budget_exhausted").GetString());
        Assert.Equal("length", root.GetProperty("finish_reason").GetString());
        Assert.True(root.GetProperty("generated_tokens").GetInt32() > 0);
    }

    [Fact]
    public async Task RecentRequestRing_IsBoundedByUiConfig_ResyncCarriesOnlyNewest()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        await using LiveServer server = await LiveServer.StartAsync("smollm-ring", uiRecentRequests: 2);

        List<string> requestIds = [];
        for (int i = 0; i < 5; i++)
        {
            CompletionOutcome outcome = await PostCompletionDetailedAsync(server, ShortPrompt, maxTokens: 6);
            requestIds.Add(outcome.RequestId);
        }

        Assert.Equal(2, server.Bus.RecentRequestCount);
        IReadOnlyList<RequestPayload> retained = server.Bus.RecentRequests();
        Assert.Equal(requestIds[4], retained[0].RequestId); // newest first
        Assert.Equal(requestIds[3], retained[1].RequestId);

        using CancellationTokenSource deadline = new(TimeSpan.FromSeconds(150));

        // A header id older than every retained event (and inside an unwrapped ring) replays the snapshots and then
        // dumps state: the resync's recent list must show exactly the newest two completions.
        await using Stream body = await OpenStreamAsync(server, lastEventId: "1", deadline.Token);
        StreamReader reader = new(body);

        SseFrame resync = await ReadFramesUntilAsync(reader, f => f.Event == MonitorEventNames.Resync, deadline.Token);
        Assert.True(resync.Id is null or 0, "resync frames are transient state and must not claim a replayable id");

        using JsonDocument document = JsonDocument.Parse(resync.Data);
        JsonElement root = document.RootElement;

        Assert.Equal(1, root.GetProperty("events_version").GetInt32());
        Assert.True(root.TryGetProperty("in_flight", out _), "resync must name the in-flight set (empty when idle)");

        JsonElement recent = root.GetProperty("recent_requests");
        Assert.Equal(2, recent.GetArrayLength());
        Assert.Equal(requestIds[4], recent[0].GetProperty("request_id").GetString());
        Assert.Equal(requestIds[3], recent[1].GetProperty("request_id").GetString());
    }

    [Fact]
    public async Task RecentRequests_HotReload_ResizesRingWithoutRestart()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        await using LiveServer server = await LiveServer.StartAsync("smollm-ringreload", uiRecentRequests: 25);

        List<string> requestIds = [];
        for (int i = 0; i < 3; i++)
        {
            CompletionOutcome outcome = await PostCompletionDetailedAsync(server, ShortPrompt, maxTokens: 6);
            requestIds.Add(outcome.RequestId);
        }

        Assert.Equal(3, server.Bus.RecentRequestCount); // default capacity admits all three so far

        LairConfig shrinkCandidate = server.Engine.EffectiveConfiguration with { Ui = new UiConfig { RecentRequests = 2 } };
        FieldResult shrink = Assert.Single(server.Engine.ApplyConfiguration(shrinkCandidate));

        Assert.Equal("ui.recent_requests", shrink.Change.Field);
        Assert.Equal(ConfigMutationClass.Live, shrink.Change.MutationClass);
        Assert.Equal(ConfigApplyOutcome.Applied, shrink.Outcome);

        Assert.Equal(2, server.Bus.RecentRequestCount); // shrinking dropped the oldest without a restart
        IReadOnlyList<RequestPayload> retained = server.Bus.RecentRequests();
        Assert.Equal(requestIds[2], retained[0].RequestId);
        Assert.Equal(requestIds[1], retained[1].RequestId);

        LairConfig growCandidate = server.Engine.EffectiveConfiguration with { Ui = new UiConfig { RecentRequests = 4 } };
        FieldResult grow = Assert.Single(server.Engine.ApplyConfiguration(growCandidate));
        Assert.Equal(ConfigApplyOutcome.Applied, grow.Outcome);

        CompletionOutcome sixth = await PostCompletionDetailedAsync(server, ShortPrompt, maxTokens: 6);
        Assert.Equal(3, server.Bus.RecentRequestCount); // the grown capacity admitted a new completion again
        Assert.Equal(sixth.RequestId, server.Bus.RecentRequests()[0].RequestId);

        Assert.Equal(4, server.Engine.EffectiveConfiguration.Ui.RecentRequests); // effective configuration carries the live value
    }

    [Fact]
    public async Task LastEventId_ReplaysMissedSnapshots_ThenResyncsState()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        await using LiveServer server = await LiveServer.StartAsync("smollm-replay", uiRecentRequests: 25);

        Task<string> firstLoad = PostCompletionAsync(server, LongPrompt, maxTokens: 64);

        using CancellationTokenSource deadline = new(TimeSpan.FromSeconds(180));
        long lastSeenId;

        {
            await using Stream body = await OpenStreamAsync(server, lastEventId: null, deadline.Token);
            StreamReader reader = new(body);

            SseFrame firstSnapshot = await ReadFramesUntilAsync(reader, f => f.Event == MonitorEventNames.Snapshot, deadline.Token);
            lastSeenId = firstSnapshot.Id!.Value;

            // Stay attached until a newer snapshot lands: the ring must hold something strictly newer than the id we
            // will replay from before this stream hangs up.
            await ReadFramesUntilAsync(reader, f => f.Event == MonitorEventNames.Snapshot && f.Id > lastSeenId, deadline.Token);
        } // hang up mid-load: everything published after lastSeenId is missed by this stream

        Task<string> secondLoad = PostCompletionAsync(server, LongPrompt + " the lighthouse keeper never slept", maxTokens: 64);

        await using Stream replayBody = await OpenStreamAsync(server, lastEventId: lastSeenId.ToString(), deadline.Token);
        StreamReader replayReader = new(replayBody);

        long previousId = lastSeenId;
        int replayedSnapshots = 0;
        SseFrame resync = default!;
        while (true)
        {
            SseFrame frame = await ReadFramesUntilAsync(
                replayReader,
                f => f.Event is MonitorEventNames.Snapshot or MonitorEventNames.Resync,
                deadline.Token);

            if (frame.Event == MonitorEventNames.Resync)
            {
                resync = frame;
                break;
            }

            Assert.True(frame.Id > previousId, $"replayed snapshot ids must climb: {previousId} -> {frame.Id}");
            previousId = frame.Id!.Value;
            replayedSnapshots++;
        }

        Assert.True(replayedSnapshots >= 1, "the reconnect replayed no missed snapshot");

        using JsonDocument document = JsonDocument.Parse(resync.Data);
        Assert.Equal(1, document.RootElement.GetProperty("events_version").GetInt32());

        try
        {
            _ = await firstLoad;
            _ = await secondLoad;
        }
        catch (Exception)
        {
            // loads may have ended when the stream closed; replay ordering is what this test asserts
        }
    }

    [Fact]
    public async Task LastEventId_DecisionInputs_DistinguishTransientAndStaleIds()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // harness needs the fixture model + tokenizer
        }

        using MonitorHarness harness = new();
        await using EngineMetricsBus bus = new(
            "decision-model", harness.Scheduler, harness.Pool, cache: null, NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(25));

        // No engine load ran: zero snapshots published. Synthetic lifecycle frames move the id line forward so a
        // header can name something retained-but-not-replayable (a transient request frame) versus nothing at all.
        for (int i = 1; i <= 3; i++)
        {
            bus.PublishRequest(MonitorEventNames.RequestUpdate, SyntheticPayload($"decision-{i}"));
        }

        Assert.Equal(0, bus.PublishedTotal); // quiet-while-idle: no snapshots were invented
        Assert.Equal(3, bus.LatestEventId);
        Assert.Equal(0, bus.OldestRetainedSnapshotId);
        Assert.Empty(bus.SnapshotReplayAfter(2)); // an id naming a transient frame has nothing snapshot-newer — the handler's resync-fallback condition

        ResyncPayload resync = bus.BuildResync();
        Assert.NotNull(resync.InFlight);
        Assert.NotNull(resync.RecentRequests);
    }

    [Fact]
    public async Task SlowSubscriber_SeesOnlyNewestSnapshot_AndDropsAreCounted()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // harness needs the fixture model + tokenizer
        }

        using MonitorHarness harness = new();
        await using EngineMetricsBus bus = new(
            "coalesce-model", harness.Scheduler, harness.Pool, cache: null, NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(30));

        MonitorSubscription subscriber = bus.Subscribe(); // attached but never read — the slow-consumer shape

        System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();
        int[] promptTokens = [.. harness.Tokenizer.Encode("hi")];

        while (bus.PublishedTotal <= 12 && clock.Elapsed < TimeSpan.FromSeconds(90))
        {
            SequenceHandle handle = harness.Scheduler.Submit(new ScheduledRequest
            {
                PromptTokens = promptTokens,
                MaxTokens = 3,
                Sampling = new SamplingParams { Temperature = 0f },
                EosId = harness.Tokenizer.EosId,
            }, CancellationToken.None);

            _ = await handle.Completion;
        }

        Assert.True(bus.PublishedTotal > 12, $"only {bus.PublishedTotal} snapshots published — the load never outlived the coalescing window");

        // A capacity-one latest-pending queue holds exactly one frame no matter how many were published: draining it
        // yields a single snapshot, and it is the newest in the ring.
        int drained = 0;
        long? drainedId = null;
        while (subscriber.Snapshots.Reader.TryRead(out MonitorFrame? frame))
        {
            Assert.NotNull(frame);
            drained++;
            drainedId = frame!.Id;
        }

        Assert.Equal(1, drained); // only the newest snapshot survived for this consumer
        IReadOnlyList<MetricsSnapshot> ring = bus.SnapshotRing();
        Assert.Equal(ring[0].EventSeq, drainedId ?? -1); // and it is that newest one

        Assert.True(bus.DroppedEventsTotal >= 5, $"coalescing {bus.PublishedTotal} snapshots into one slot should count drops, got {bus.DroppedEventsTotal}");

        long beforeRequestDrops = bus.DroppedEventsTotal;
        for (int i = 0; i < EngineMetricsBus.RequestQueueCapacity + 10; i++)
        {
            bus.PublishRequest(MonitorEventNames.RequestUpdate, SyntheticPayload($"flood-{i}"));
        }

        Assert.True(bus.DroppedEventsTotal - beforeRequestDrops >= 10, $"flooding past the bounded queue should count drops, got {bus.DroppedEventsTotal - beforeRequestDrops}");

        bus.Unsubscribe(subscriber); // detach: queued frames drop with the completed channels
    }

    private static RequestPayload SyntheticPayload(string requestId) => new(
        MonitorEventSchema.Version,
        requestId,
        "synthetic-model",
        "completion",
        PromptTokens: 4,
        Status: "running",
        GeneratedTokens: 1);

    private static string? RequestIdOf(SseFrame frame)
    {
        try
        {
            using JsonDocument document = JsonDocument.Parse(frame.Data);
            return document.RootElement.TryGetProperty("request_id", out JsonElement id) ? id.GetString() : null;
        }
        catch (JsonException)
        {
            return null; // keep-alive comments and other frames carry no request key
        }
    }

    private static async Task<Stream> OpenStreamAsync(LiveServer server, string? lastEventId, CancellationToken ct)
    {
        HttpRequestMessage request = new(HttpMethod.Get, $"{server.BaseUrl}/monitor/stream");
        if (lastEventId is not null)
        {
            request.Headers.Add("Last-Event-ID", lastEventId);
        }

        HttpResponseMessage response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        Assert.True(response.IsSuccessStatusCode, $"stream request answered {(int)response.StatusCode}");
        return await response.Content.ReadAsStreamAsync(ct);
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

    private sealed record SseFrame(long? Id, string Event, string Data);

    private sealed record CompletionOutcome(string FinishReason, string RequestId);

    private static async Task<string> PostCompletionAsync(LiveServer server, string prompt, int maxTokens) =>
        (await PostCompletionDetailedAsync(server, prompt, maxTokens)).FinishReason;

    // One greedy completion through the live API; returns its finish reason and the x-lair-request-id it answered with.
    private static async Task<CompletionOutcome> PostCompletionDetailedAsync(LiveServer server, string prompt, int maxTokens)
    {
        string body = JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["model"] = server.ModelName,
            ["prompt"] = prompt,
            ["max_tokens"] = maxTokens,
            ["temperature"] = 0f,
            ["stream"] = false,
        });

        using HttpResponseMessage response = await Http.PostAsync(
            $"{server.BaseUrl}/v1/completions", new StringContent(body, Encoding.UTF8, "application/json"));
        response.EnsureSuccessStatusCode();

        string requestId = response.Headers.TryGetValues(ApiApp.RequestIdHeader, out IEnumerable<string>? values)
            ? values.FirstOrDefault() ?? throw new InvalidOperationException("the response carried no x-lair-request-id")
            : throw new InvalidOperationException("the response carried no x-lair-request-id");

        using JsonDocument document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        string finish = document.RootElement.GetProperty("choices")[0].GetProperty("finish_reason").GetString() ?? string.Empty;
        return new CompletionOutcome(finish, requestId);
    }

    /// <summary>One engine + API host on an ephemeral port with a chosen ui.recent_requests capacity.</summary>
    private sealed class LiveServer : IAsyncDisposable
    {
        public GenerationService Engine { get; }
        public WebApplication App { get; }
        public string BaseUrl { get; }
        public string ModelName { get; }

        // The API host attaches the bus while building; this reaches the same instance through the service.
        public EngineMetricsBus Bus => Engine.Metrics ?? throw new InvalidOperationException("the API host attached no metrics bus");

        private LiveServer(GenerationService engine, WebApplication app, string modelName)
        {
            Engine = engine;
            App = app;
            ModelName = modelName;
            BaseUrl = app.Urls.Single();
        }

        public static async Task<LiveServer> StartAsync(string modelName, int uiRecentRequests)
        {
            string modelPath = ModelFixture.Resolve() ?? throw new InvalidOperationException(
                $"GGUF fixture '{ModelFixture.SmolLmRelativePath}' was not found walking up from the test assembly directory.");

            LairConfig config = new()
            {
                Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },

                // Single-threaded on purpose (the metrics suite's reasoning): these fixtures run beside other real-model
                // suites, and small prefill chunks keep step boundaries frequent at debug-JIT forward speed.
                Engine = new EngineConfig { Threads = 1, MaxBatchTokens = 8 },
                Models =
                [
                    new ModelConfig
                    {
                        Name = modelName,
                        Path = modelPath,
                        MaxContextTokens = 512,
                        KvBlocks = 48,
                    },
                ],
                Ui = new UiConfig { RecentRequests = uiRecentRequests },
            };

            GenerationService engine = GenerationService.Open(config);
            try
            {
                WebApplication app = ApiApp.Build(config, engine);
                await ((IHost)app).StartAsync(CancellationToken.None);
                return new LiveServer(engine, app, modelName);
            }
            catch
            {
                engine.Dispose();
                throw;
            }
        }

        public async ValueTask DisposeAsync()
        {
            await ((IHost)App).StopAsync(TimeSpan.FromSeconds(5));
            await App.DisposeAsync();
            Engine.Dispose();
        }
    }

    /// <summary>Model + pool + scheduler opened directly against the fixture for bus-level semantics.</summary>
    private sealed class MonitorHarness : IDisposable
    {
        public GgufFile Gguf { get; }
        public GgufTokenizer Tokenizer { get; }
        public CpuBackend Backend { get; }
        public TransformerModel Model { get; }
        public BlockPool Pool { get; }
        public GenerationScheduler Scheduler { get; }

        public MonitorHarness()
        {
            string modelPath = ModelFixture.Resolve() ?? throw new InvalidOperationException(
                $"GGUF fixture '{ModelFixture.SmolLmRelativePath}' was not found walking up from the test assembly directory.");

            Gguf = GgufFile.Open(modelPath);
            Tokenizer = GgufTokenizer.FromGguf(Gguf);
            Backend = new CpuBackend(threads: 1);
            Model = TransformerModel.Open(Gguf, Backend, 512);
            Pool = new BlockPool(
                blockTokens: 16,
                layerCount: Model.Desc.LayerCount,
                kvHeads: Model.Desc.KvHeadCount,
                headDim: Model.Desc.HeadDim,
                capacityBlocks: 24,
                Backend);
            Scheduler = new GenerationScheduler(Model, Pool, maxBatchTokens: 512);
        }

        public void Dispose()
        {
            Scheduler.Dispose();
            Pool.Dispose();
            Model.Dispose();
            Gguf.Dispose();
            Backend.Dispose();
        }
    }
}
