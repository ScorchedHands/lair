using System.Diagnostics;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using Lair.Api;
using Lair.Api.Monitoring;
using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Host;
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
/// Metrics bus, snapshot JSON, the live <c>/monitor/stream</c> endpoint, and the monitor CLI table — all driven
/// through the real SmolLM fixture: counter correctness after a scripted shared-prefix load, ring boundedness under
/// sustained scheduler traffic, wire shape of stream events, and clean cancellation against a test server. Each API
/// test opens its own engine so cumulative counters describe exactly that test's load.
/// </summary>
[Trait("Category", "Metrics")]
public sealed class MetricsTests
{
    private static readonly HttpClient Http = new();

    // Multi-token greedy content proven by the concurrency suite, extended with four repetitions of the chunked-prefill
    // pangram (well past two 16-token blocks) so requests 1-3 share whole cached blocks and later ones adopt prompt
    // prefixes; request 4 runs a distinct cold prompt. Kept modest so even debug-JIT prefill finishes inside the suite's patience.
    private static readonly string SharedPrefix = string.Join(" ", Enumerable.Repeat(
        "The quick brown fox jumps over the lazy dog while the patient cat watches the rain.", 4));

    private static readonly string[] LoadPrompts =
    [
        SharedPrefix + " In a distant library, maps of imaginary cities lined every wall.",
        SharedPrefix,
        SharedPrefix + " The librarian kept a red ledger of borrowed dreams.",
        "Three tips for staying healthy are:",
    ];

    // The API host's snapshot cadence is two seconds; settled reads wait out three full windows plus margin.
    private static readonly TimeSpan Settle = TimeSpan.FromSeconds(7);

    [Fact]
    public async Task CountersAfterSharedPrefixLoad_MatchHttpOutcomes_AndRecordCacheSkips()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        await using LiveServer server = await LiveServer.StartAsync("smollm-metrics");

        Dictionary<string, int> httpFinishCounts = [];
        foreach (string prompt in LoadPrompts)
        {
            string finish = await PostCompletionAsync(server, prompt, maxTokens: 24);
            httpFinishCounts[finish] = httpFinishCounts.GetValueOrDefault(finish) + 1;
        }

        // Let the cadence settle and publish its closing snapshot before reading counters.
        await Task.Delay(Settle);

        IReadOnlyList<MetricsSnapshot> ring = server.Bus.SnapshotRing();
        Assert.True(ring.Count > 0, "the bus published no snapshots after a load");

        MetricsSnapshot final = ring[0]; // newest first
        Assert.Equal("smollm-metrics", final.Model);

        long httpTotal = httpFinishCounts.Values.Sum();
        Assert.Equal(httpTotal, final.RequestsTotal);
        Assert.Equal(CountOf(httpFinishCounts, "eos"), final.RequestsEosTotal);
        Assert.Equal(CountOf(httpFinishCounts, "length"), final.RequestsLengthTotal);
        Assert.Equal(CountOf(httpFinishCounts, "stop"), final.RequestsStopTotal);

        Assert.True(final.DecodeStepsTotal > 0, "no decode steps were recorded after a full load");
        Assert.True(
            ring.Any(s => s.TokensPerSecDecodeWindowed > 0),
            $"no snapshot recorded decode throughput after {httpTotal} completions");
        Assert.True(
            ring.Any(s => s.DecodeStepsTotal > 0 && s.AvgStepTimeMsWindowed > 0),
            "snapshots never paired executed decode steps with a measured step time");

        // Shared prefixes: the two later shared-prefix requests each adopted at least one full cached block (16 tokens).
        Assert.True(
            final.CacheHitSequencesTotal >= 2,
            $"expected at least two sequences to adopt cached prefixes, got {final.CacheHitSequencesTotal}");
        Assert.True(
            final.CacheSkippedTokensTotal >= 2 * 16,
            $"expected cache adoptions to skip prefill for shared prefixes, got {final.CacheSkippedTokensTotal} skipped tokens");

        // KV utilization stays inside the pool's capacity while sequences ran through it.
        Assert.Equal(48, final.KvCapacityBlocks);
        Assert.InRange(final.KvUsedBlocks, 0, final.KvCapacityBlocks);
        Assert.True(ring.Any(s => s.KvUsedBlocks > 0), "no snapshot ever reported KV blocks in use");
    }

    [Fact]
    public async Task StreamEvents_CarryWellFormedSnapshotJsonWithMetricsFields()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        await using LiveServer server = await LiveServer.StartAsync("smollm-stream");

        // A short concurrency-proven prompt decodes almost immediately: snapshots start flowing within one step even at debug-JIT speed.
        Task<string> load = PostCompletionAsync(server, "It was a dark and stormy night, and", maxTokens: 64); // keep the engine active so snapshots flow

        using CancellationTokenSource timeout = new(TimeSpan.FromSeconds(90));

        HttpResponseMessage response = await Http.SendAsync(
            new HttpRequestMessage(HttpMethod.Get, $"{server.BaseUrl}/monitor/stream"),
            HttpCompletionOption.ResponseHeadersRead,
            timeout.Token);

        Assert.True(response.IsSuccessStatusCode, $"stream request answered {(int)response.StatusCode}");
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);

        using (response)
        {
            StreamReader reader = new(await response.Content.ReadAsStreamAsync(timeout.Token));

            string? dataLine = null;
            while (dataLine is null && !timeout.IsCancellationRequested)
            {
                string? line = await reader.ReadLineAsync(timeout.Token);
                if (line is null)
                {
                    break; // stream closed before the first snapshot — caught by Assert.NotNull below
                }

                if (line.StartsWith("data: ", StringComparison.Ordinal))
                {
                    dataLine = line["data: ".Length..];
                }
            }

            Assert.NotNull(dataLine);

            using JsonDocument document = JsonDocument.Parse(dataLine!);
            JsonElement root = document.RootElement;

            string[] expectedFields =
            [
                "metrics_ts", "metrics_window_ms", "metrics_model", "metrics_tok_s", "metrics_step_time_avg_ms",
                "metrics_decode_steps_total", "metrics_batch_occupancy_seqs_per_step", "metrics_prefill_chunks_total",
                "metrics_prefill_tokens_forwarded_total", "metrics_cache_hit_sequences_total", "metrics_cache_skipped_tokens_total",
                "metrics_kv_used_blocks", "metrics_kv_capacity_blocks", "metrics_kv_cached_blocks", "metrics_active_sequences",
                "metrics_requests_total", "metrics_requests_eos_total", "metrics_requests_length_total", "metrics_requests_stop_total",
                "metrics_requests_cancelled_total", "metrics_requests_failed_total", "metrics_process_working_set_mb",
                "metrics_process_cpu_percent",
            ];

            foreach (string field in expectedFields)
            {
                Assert.True(root.TryGetProperty(field, out _), $"snapshot JSON is missing '{field}': {dataLine}");
            }

            // Every top-level property speaks the metrics_ vocabulary.
            foreach (JsonProperty property in root.EnumerateObject())
            {
                Assert.StartsWith("metrics_", property.Name);
            }

            Assert.Equal("smollm-stream", root.GetProperty("metrics_model").GetString());
            Assert.Matches(@"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", root.GetProperty("metrics_ts").GetString()!);
            Assert.Equal(JsonValueKind.Number, root.GetProperty("metrics_kv_capacity_blocks").ValueKind);
            Assert.True(root.GetProperty("metrics_window_ms").GetInt64() > 0);
        }

        // The load may have ended early when the stream read finished; its outcome is not asserted here.
        try
        {
            _ = await load;
        }
        catch (Exception)
        {
            // swallowed on purpose — this test only watches snapshots
        }
    }

    [Fact]
    public async Task RingStaysBounded_UnderSustainedSchedulerTraffic()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using MetricsHarness harness = new(capacityBlocks: 24);
        await using EngineMetricsBus bus = new(
            "ring-model", harness.Scheduler, harness.Pool, cache: null, NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(30));

        // Sustained tiny requests keep the engine active long enough to publish well past the ring capacity.
        Stopwatch clock = Stopwatch.StartNew();
        int[] promptTokens = [.. harness.Tokenizer.Encode("hi")];

        while (bus.PublishedTotal <= EngineMetricsBus.RingCapacity && clock.Elapsed < TimeSpan.FromSeconds(60))
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

        Assert.True(
            bus.PublishedTotal > EngineMetricsBus.RingCapacity,
            $"only {bus.PublishedTotal} snapshots published — the load never outlived the ring");

        IReadOnlyList<MetricsSnapshot> ring = bus.SnapshotRing();
        Assert.Equal(EngineMetricsBus.RingCapacity, bus.RingCount);
        Assert.Equal(EngineMetricsBus.RingCapacity, ring.Count);

        // Newest first: timestamps are monotone going down the list.
        for (int i = 0; i + 1 < ring.Count; i++)
        {
            Assert.True(
                string.CompareOrdinal(ring[i].TimestampUtcIso8601, ring[i + 1].TimestampUtcIso8601) >= 0,
                $"ring entry {i} is older than the one before it");
        }
    }

    [Fact]
    public async Task MonitorCommand_RendersLiveRowsFromTestServer_AndCancelsCleanly()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        await using LiveServer server = await LiveServer.StartAsync("smollm-monitor");

        // Load in flight while the monitor attaches: a short prompt decodes fast, so rows land within one snapshot window.
        Task<string> load = PostCompletionAsync(server, "It was a dark and stormy night, and", maxTokens: 64);

        StringWriter output = new();
        using CancellationTokenSource cancel = new();
        Task<int> run = MonitorCommand.RunAsync(new Uri(server.BaseUrl), output, cancel.Token);

        Stopwatch clock = Stopwatch.StartNew();
        bool sawRow = false;
        while (clock.Elapsed < TimeSpan.FromSeconds(90) && !sawRow)
        {
            sawRow = DataRows(output.ToString()).Count > 0;
            if (!sawRow)
            {
                await Task.Delay(250);
            }
        }

        cancel.Cancel(); // Ctrl+C-equivalent: the run ends on its own with 0
        int exitCode = await run;

        try
        {
            _ = await load;
        }
        catch (Exception)
        {
            // the generation may have ended when the engine disposed; row rendering is what this test asserts
        }

        Assert.True(sawRow, $"the monitor never rendered a table row:{Environment.NewLine}{output}");
        Assert.Equal(0, exitCode);

        List<string> lines = [.. output.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries)];
        Assert.StartsWith("model", lines[0]); // header line
        Assert.Contains(lines, l => !l.StartsWith("model") && l.Contains("smollm-monitor"));
    }

    [Fact]
    public async Task MonitorCommand_UnreachableServer_PrintsOneActionableError()
    {
        StringWriter output = new();

        int exitCode = await MonitorCommand.RunAsync(new Uri("http://127.0.0.1:1"), output, CancellationToken.None);

        Assert.NotEqual(0, exitCode);
        List<string> lines = [.. output.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries)];
        Assert.Single(lines);
        Assert.StartsWith("error:", lines[0]);
        Assert.Contains("/monitor/stream", lines[0]); // names what it tried to reach
    }

    private static int CountOf(Dictionary<string, int> counts, string reason) => counts.GetValueOrDefault(reason);

    private static List<string> DataRows(string text) =>
        [.. text.Split('\n', StringSplitOptions.RemoveEmptyEntries).Where(l => !l.StartsWith("model"))];

    // One greedy completion through the live API; returns its finish reason.
    private static async Task<string> PostCompletionAsync(LiveServer server, string prompt, int maxTokens)
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

        using JsonDocument document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        return document.RootElement.GetProperty("choices")[0].GetProperty("finish_reason").GetString() ?? string.Empty;
    }

    /// <summary>One engine + API host on an ephemeral port, metrics bus attached by the build itself.</summary>
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

        public static async Task<LiveServer> StartAsync(string modelName)
        {
            string modelPath = ModelFixture.Resolve() ?? throw new InvalidOperationException(
                $"GGUF fixture '{ModelFixture.SmolLmRelativePath}' was not found walking up from the test assembly directory.");

            LairConfig config = new()
            {
                Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },

                // Single-threaded on purpose: these fixtures run beside other real-model suites in parallel, and
                // auto-threaded GEMV pools would starve the timing-sensitive waits those suites depend on. Small
                // prefill chunks keep step boundaries frequent so snapshots flow even at debug-JIT forward speed.
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

    /// <summary>Model + pool + scheduler opened directly against the fixture, like the cancellation suite's harness.</summary>
    private sealed class MetricsHarness : IDisposable
    {
        public GgufFile Gguf { get; }
        public GgufTokenizer Tokenizer { get; }
        public CpuBackend Backend { get; }
        public TransformerModel Model { get; }
        public BlockPool Pool { get; }
        public GenerationScheduler Scheduler { get; }

        public MetricsHarness(int capacityBlocks)
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
                capacityBlocks: capacityBlocks,
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
