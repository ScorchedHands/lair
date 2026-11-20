using System.Text.Json;

using Lair.Http;
using Lair.Http.Monitoring;
using Lair.Host.Config;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// Monitor coverage for the encode-only serving posture. A model opened as <c>embed</c> advances no scheduler sequence,
/// so decode-side counters alone would leave it permanently silent on <c>/monitor/stream</c>: these facts prove its
/// snapshots publish while its encoder forwards run and stay quiet while nothing runs, that the additive encode-side
/// fields carry what those forwards actually did while every decode-side field keeps reading its own zero fact, and that
/// an admitted encoder call mints request-lifecycle frames landing in the recent-request ring with no fabricated tok/s
/// series. Engines here are synthetic so each assertion describes exactly one bus's counters.
/// </summary>
[Trait("Category", "Embeddings")]
[Trait("Category", "MonitorStream")]
public sealed class EmbedMonitorSnapshotTests : IDisposable
{
    private const string ModelName = "embed-monitor";

    private static readonly TimeSpan Interval = TimeSpan.FromMilliseconds(40);

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

    [Fact]
    public async Task EncoderEngineStaysQuietWhileIdle_AndPublishesOnceAnEncoderCallRuns()
    {
        using GenerationService engine = OpenEmbedEngine(out EngineMetricsBus bus);

        // Several idle ticks pass with nothing submitted: the quiet-while-idle posture is unchanged, and an encode-only
        // engine has no decode counter that could accidentally make it look busy.
        await Task.Delay(Interval * 4);
        Assert.Equal(0L, bus.RingCount);

        EmbeddingOutcome outcome = await engine.EmbedAsync(Request("embed-monitor-idle"), CancellationToken.None);
        MetricsSnapshot snapshot = WaitForSnapshot(bus, s => s.EncoderRequestsTotal >= 1 && s.ActiveEncoderRequests == 0);

        Assert.Single(outcome.Vectors);
        Assert.Equal(1L, snapshot.EncoderRequestsTotal); // the closing tick of a burst publishes even after the load ended mid-window
        Assert.Equal(0, snapshot.ActiveEncoderRequests); // nothing runs once its call returned
    }

    [Fact]
    public async Task SnapshotCarriesEncodeSideFacts_AndKeepsEveryDecodeFieldAtItsZeroFact()
    {
        using GenerationService engine = OpenEmbedEngine(out EngineMetricsBus bus);

        string[] inputs = ["hello", "world! world", "hello world !"]; // distinct token counts per input, each its own call
        long forwardedTotal = 0;

        for (int i = 0; i < inputs.Length; i++)
        {
            EmbeddingOutcome outcome = await engine.EmbedAsync(Request($"embed-monitor-facts-{i}", inputs[i]), CancellationToken.None);
            forwardedTotal += outcome.TotalInputTokens;
        }

        MetricsSnapshot snapshot = WaitForSnapshot(bus, s => s.EncoderRequestsTotal >= inputs.Length && s.EncoderTokensForwardedTotal == forwardedTotal && s.ActiveEncoderRequests == 0);

        Assert.Equal(inputs.Length, snapshot.EncoderRequestsTotal); // every call that pooled all of its inputs counts once
        Assert.Equal(forwardedTotal, snapshot.EncoderTokensForwardedTotal); // positions carried through forwards that completed their pooling step
        Assert.Equal(0, snapshot.ActiveEncoderRequests);

        // Every window that carried encoder work publishes (its counters differ), so at least one retained snapshot reports the throughput of its own interval.
        Assert.True(
            bus.SnapshotRing().Any(s => s.EncoderTokensPerSecWindowed > 0.0),
            $"no published snapshot carried an encode rate: {string.Join(" | ", bus.SnapshotRing().Select(s => s.ToJson()))}");

        // Decode-side fields keep reading what actually happened on this engine — nothing at all — instead of standing in for encoder activity.
        Assert.Equal(0L, snapshot.DecodeStepsTotal);
        Assert.Equal(0.0, snapshot.TokensPerSecDecodeWindowed);
        Assert.Equal(0L, snapshot.PrefillChunksTotal);
        Assert.Equal(0L, snapshot.PrefillTokensForwardedTotal);
        Assert.Equal(0, snapshot.ActiveSequences);
        Assert.Equal(0L, snapshot.RequestsTotal);

        // Wire shape: the new fields joined the metrics_ vocabulary without renaming anything.
        using JsonDocument document = JsonDocument.Parse(snapshot.ToJson());
        JsonElement root = document.RootElement;

        foreach (JsonProperty property in root.EnumerateObject())
        {
            Assert.StartsWith("metrics_", property.Name);
        }

        foreach (string field in new[]
        {
            "metrics_active_encoder_requests", "metrics_encoder_requests_total",
            "metrics_encoder_tokens_forwarded_total", "metrics_encoder_tok_s",
        })
        {
            Assert.True(root.TryGetProperty(field, out _), $"snapshot JSON is missing '{field}': {snapshot.ToJson()}");
        }

        Assert.Equal("embed", root.GetProperty("metrics_model_type").GetString()); // the posture a snapshot names, not an inference from which fields are non-zero
    }

    [Fact]
    public async Task EncoderCallMintsLifecycleFrames_AndLandsInTheRecentRequestRing()
    {
        using GenerationService engine = OpenEmbedEngine(out EngineMetricsBus bus);

        MonitorSubscription subscription = bus.Subscribe();
        try
        {
            EmbeddingOutcome outcome = await engine.EmbedAsync(Request("embed-monitor-lifecycle"), CancellationToken.None);

            List<MonitorFrame> frames = DrainRequestFrames(subscription, TimeSpan.FromMilliseconds(500));
            Assert.NotEmpty(frames);

            MonitorFrame start = Assert.Single(frames, frame => frame.EventName == MonitorEventNames.RequestStart);
            MonitorFrame complete = Assert.Single(frames, frame => frame.EventName == MonitorEventNames.RequestComplete);
            Assert.DoesNotContain(frames, frame => frame.EventName == MonitorEventNames.RequestUpdate); // nothing advanced token by token between start and completion

            using JsonDocument startDoc = JsonDocument.Parse(start.Json);
            JsonElement startRoot = startDoc.RootElement;
            Assert.Equal("embed-monitor-lifecycle", startRoot.GetProperty("request_id").GetString());
            Assert.Equal(ModelName, startRoot.GetProperty("model").GetString());
            Assert.Equal("embeddings", startRoot.GetProperty("endpoint").GetString());
            Assert.Equal(outcome.TotalInputTokens, startRoot.GetProperty("prompt_tokens").GetInt32()); // the call's whole input side is known from admission on
            Assert.Equal("running", startRoot.GetProperty("status").GetString());

            using JsonDocument completeDoc = JsonDocument.Parse(complete.Json);
            JsonElement completeRoot = completeDoc.RootElement;
            Assert.Equal("encoded", completeRoot.GetProperty("status").GetString()); // an encoder call has no sampling finish to report
            Assert.Equal(JsonValueKind.Null, completeRoot.GetProperty("finish_reason").ValueKind); // the field stands there as nothing, exactly as a generation frame's unknown-at-start fields do
            Assert.Equal(JsonValueKind.Null, completeRoot.GetProperty("samples").ValueKind); // no tok/s series exists for a posture that samples nothing
            Assert.Equal(0, completeRoot.GetProperty("generated_tokens").GetInt32());
            Assert.True(completeRoot.GetProperty("elapsed_ms").GetInt64() >= 0);

            Assert.True(start.Id < complete.Id); // one host event line orders both frames of this call
        }
        finally
        {
            bus.Unsubscribe(subscription);
        }

        IReadOnlyList<RequestPayload> recent = bus.RecentRequests();
        Assert.NotEmpty(recent); // a completed encoder call is replayable state, not just a live-tail frame
        RequestPayload row = recent[0];
        Assert.Equal("embed-monitor-lifecycle", row.RequestId);
        Assert.Equal("embeddings", row.Endpoint);
        Assert.Equal("encoded", row.Status);
    }

    // ---------------------------------------------------------------- shared helpers

    private GenerationService OpenEmbedEngine(out EngineMetricsBus bus)
    {
        string path = SyntheticLlamaGguf.Write(ModelName);
        _tempFiles.Add(path);

        LairConfig config = new()
        {
            Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
            Engine = new EngineConfig { Threads = 1 },
            Models =
            [
                new ModelConfig
                {
                    Name = ModelName,
                    Path = path,
                    Type = "embed",
                    Pooling = "mean",
                    Normalize = false,
                    MaxContextTokens = 512,
                    KvBlocks = 8,
                },
            ],
        };

        GenerationService engine = GenerationService.Open(config);
        try
        {
            bus = engine.AttachMetrics(NullLoggerFactory.Instance, Interval); // the same facts path a serving host wires through its monitor build
            return engine;
        }
        catch
        {
            engine.Dispose();
            throw;
        }
    }

    private static EmbeddingRequest Request(string requestId, string input = "hello") => new()
    {
        Model = ModelName,
        Inputs = [input],
        RequestId = requestId,
    };

    // Waits for the bus's own cadence to publish: a snapshot whose encode-side tally satisfies the predicate, newest first.
    private static MetricsSnapshot WaitForSnapshot(EngineMetricsBus bus, Func<MetricsSnapshot, bool>? ready = null)
    {
        for (int attempt = 0; attempt < 200; attempt++)
        {
            IReadOnlyList<MetricsSnapshot> ring = bus.SnapshotRing();
            if (ring.Count > 0 && (ready is null || ready(ring[0])))
            {
                return ring[0];
            }

            Task.Delay(25).GetAwaiter().GetResult();
        }

        throw new TimeoutException($"no snapshot satisfied the wait: bus published {bus.PublishedTotal} frames");
    }

    private static List<MonitorFrame> DrainRequestFrames(MonitorSubscription subscription, TimeSpan window)
    {
        List<MonitorFrame> frames = [];
        long deadline = Environment.TickCount64 + (long)window.TotalMilliseconds;

        while (Environment.TickCount64 < deadline)
        {
            if (subscription.Requests.Reader.TryRead(out MonitorFrame? frame) && frame is not null)
            {
                frames.Add(frame);
                continue;
            }

            Task.Delay(25).GetAwaiter().GetResult();
        }

        return frames;
    }
}
