using System.Net;
using System.Text;
using System.Text.Json;
using Lair.Http;
using Lair.Host.Config;
using Lair.Http.Monitoring;
using Lair.Backends.Cpu;
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
/// Quota, route, and admission diagnostics on the two operator surfaces: <c>/v1/models</c> reports the configured
/// concurrency cap (explicitly capped or uncapped) beside the resolved route segments, and <c>/monitor/stream</c>
/// snapshots carry the same route facts plus prefill-lane posture and a bounded ring of recent admission refusals —
/// each naming the limit kind that blocked. Engines here are synthetic so every assertion runs without the real-model
/// fixture; capture wiring is driven through the real facade path, and bus-level semantics run against direct harnesses.
/// </summary>
[Trait("Category", "Api")]
[Trait("Category", "Metrics")]
public sealed class AdmissionDiagnosticsTests : IDisposable
{
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

    private string NewSyntheticModel()
    {
        string path = SyntheticLlamaGguf.Write("admission-diagnostics");
        _tempFiles.Add(path);
        return path;
    }

    // --- /v1/models quota and route reporting ---------------------------------------------------------

    [Fact]
    public async Task Models_ListsServedModel_ReportsConcurrencyCap_AndResolvedRouteSegments()
    {
        await using SyntheticServer server = await SyntheticServer.StartAsync(
            NewSyntheticModel(), maxConcurrentRequests: 3, threadQuota: 2, coreSet: new HashSet<int> { 2, 4, 5 });

        JsonElement entry = await GetFirstModelEntryAsync(server);

        Assert.Equal("synthetic-quota", entry.GetProperty("id").GetString());
        Assert.Equal(3, entry.GetProperty("max_concurrent_requests").GetInt32());
        Assert.Equal("capped", entry.GetProperty("concurrency_cap_posture").GetString());

        JsonElement segments = entry.GetProperty("route_segments");
        Assert.Equal(JsonValueKind.Array, segments.ValueKind);
        Assert.True(segments.GetArrayLength() >= 1);

        // A CPU-only placement collapses its stages into one executable segment carrying this model's resolved quota.
        JsonSegmentAssert.CpuSingleSegment(segments[0], threadQuota: 2, coreSetRanges: "2, 4-5");
    }

    [Fact]
    public async Task Models_UncappedModel_ReportsNullCap_AndUncappedPosture()
    {
        await using SyntheticServer server = await SyntheticServer.StartAsync(NewSyntheticModel());

        JsonElement entry = await GetFirstModelEntryAsync(server);

        // The cap keys are always reported: a null value plus the uncapped posture says nothing bounds admissions,
        // so a reader never infers "uncapped" from a missing key.
        Assert.True(entry.TryGetProperty("max_concurrent_requests", out JsonElement cap));
        Assert.Equal(JsonValueKind.Null, cap.ValueKind);
        Assert.Equal("uncapped", entry.GetProperty("concurrency_cap_posture").GetString());

        // Route identity rides every served model's entry even with no placement declared: the engine's own CPU segment.
        JsonSegmentAssert.CpuSingleSegment(entry.GetProperty("route_segments")[0], threadQuota: null, coreSetRanges: null);
    }

    [Fact]
    public async Task MaxConcurrentRequestsHotEdit_ReadvertisesOnModelsEndpoint_WithoutRestart()
    {
        await using SyntheticServer server = await SyntheticServer.StartAsync(
            NewSyntheticModel(), maxConcurrentRequests: 3, threadQuota: 2);

        FieldResult applied = Assert.Single(server.Engine.ApplyConfiguration(
            server.Engine.EffectiveConfiguration with
            {
                Models = [server.Engine.EffectiveConfiguration.Models[0] with { MaxConcurrentRequests = 5 }],
            }));
        Assert.Equal("models[0].max_concurrent_requests", applied.Change.Field);
        Assert.Equal(ConfigApplyOutcome.Applied, applied.Outcome);

        JsonElement entry = await GetFirstModelEntryAsync(server);
        Assert.Equal(5, entry.GetProperty("max_concurrent_requests").GetInt32()); // the advertised cap follows the live edit
    }

    [Fact]
    public async Task ModelsResponse_KeepsBaselineFields_AndAddsQuotaFactsAdditively()
    {
        await using SyntheticServer server = await SyntheticServer.StartAsync(
            NewSyntheticModel(), maxConcurrentRequests: 2, threadQuota: 8);

        JsonElement entry = await GetFirstModelEntryAsync(server);

        // The OpenAI baseline and the earlier LAIR additions read exactly as before; quota facts land beside them.
        Assert.Equal("model", entry.GetProperty("object").GetString());
        Assert.Equal("lair", entry.GetProperty("owned_by").GetString());
        Assert.True(entry.GetProperty("context_length").GetInt32() > 0);
        Assert.False(string.IsNullOrEmpty(entry.GetProperty("quant").GetString()));
        Assert.False(string.IsNullOrEmpty(entry.GetProperty("architecture").GetString()));
        Assert.Equal("chat", entry.GetProperty("type").GetString());

        // The route segment's quota is what load resolved from placement: the declared cap and its pinned cores.
        JsonSegmentAssert.CpuSingleSegment(entry.GetProperty("route_segments")[0], threadQuota: 8, coreSetRanges: null);
    }

    // --- admission refusal capture through the real facade path ---------------------------------------

    [Fact]
    public async Task ContextLengthRefusal_AnswersAsContextLengthError_AndIsRecordedInMonitorRing()
    {
        string path = NewSyntheticModel();
        (int window, string longPrompt, string shortProbe) = MeasureContextSizing(path);

        await using SyntheticServer server = await SyntheticServer.StartAsync(
            path, maxContextTokens: window); // the configured window is exactly what the fixture tokenizer measures for this prompt — admission refuses it on equality

        (HttpResponseMessage response, JsonElement body) = await PostCompletionAsync(server, longPrompt, maxTokens: 4);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal("context_length_exceeded", body.GetProperty("error").GetProperty("code").GetString());

        EngineMetricsBus bus = server.Bus;
        Assert.Equal(1, bus.AdmissionRefusalsTotal);

        AdmissionRefusalEntry refusal = Assert.Single(bus.RecentAdmissionRefusals());
        Assert.Equal(AdmissionRefusalKinds.ContextLength, refusal.Kind);
        Assert.Equal(server.ModelName, refusal.Model);
        Assert.Contains("context window", refusal.Reason); // the concise reason is the refusal the caller already saw
    }

    [Fact]
    public async Task KvCapacityContention_RefusesSubmission_AndRingNamesKvCapacity()
    {
        await using SyntheticServer server = await SyntheticServer.StartAsync(
            NewSyntheticModel(), maxContextTokens: 512, kvBlocks: 1); // a one-block pool: any admitted sequence holds the only block for its whole lifetime

        EngineBusyException busy = await RunUntilRefusedAsync(server, probeMaxTokens: 4);
        Assert.Contains("kv_capacity", busy.Message);

        AdmissionRefusalEntry refusal = Assert.Single(server.Bus.RecentAdmissionRefusals()); // one refused submission was recorded across the attempts that produced it
        Assert.Equal(AdmissionRefusalKinds.KvCapacity, refusal.Kind); // the blocking limit named on the diagnostic matches what blocked admission
        Assert.Equal(server.ModelName, refusal.Model);
    }

    [Fact]
    public async Task ConcurrencyQuotaContention_RefusesSubmission_AndRingNamesConcurrencyQuota()
    {
        await using SyntheticServer server = await SyntheticServer.StartAsync(
            NewSyntheticModel(), maxContextTokens: 512, kvBlocks: 8, maxConcurrentRequests: 1);

        EngineBusyException busy = await RunUntilRefusedAsync(server, probeMaxTokens: 4);
        Assert.Contains("max_concurrent_requests", busy.Message); // the refusal names its cap exactly as before this diagnostic existed

        AdmissionRefusalEntry refusal = Assert.Single(server.Bus.RecentAdmissionRefusals());
        Assert.Equal(AdmissionRefusalKinds.ConcurrencyQuota, refusal.Kind);
        Assert.Equal(server.ModelName, refusal.Model);
    }

    [Fact]
    public async Task RecentAdmissionRefusals_SurviveIdleWindows_AndRideTheNextPublishedSnapshot()
    {
        string path = NewSyntheticModel();
        (int window, string longPrompt, string shortProbe) = MeasureContextSizing(path);

        await using SyntheticServer server = await SyntheticServer.StartAsync(
            path, maxContextTokens: window, kvBlocks: 2);

        (HttpResponseMessage refusedResponse, _) = await PostCompletionAsync(server, longPrompt, maxTokens: 4);
        Assert.Equal(HttpStatusCode.BadRequest, refusedResponse.StatusCode);

        // A refusal alone never wakes the quiet-while-idle sampler; a small completion publishes the closing snapshot, and that snapshot must carry the recorded refusal.
        (HttpResponseMessage okResponse, JsonElement body) = await PostCompletionAsync(server, shortProbe, maxTokens: 2);
        okResponse.EnsureSuccessStatusCode();

        using CancellationTokenSource deadline = new(TimeSpan.FromSeconds(90));
        IReadOnlyList<MetricsSnapshot> ring = await WaitForPublishedSnapshotAsync(server.Bus, deadline.Token);

        MetricsSnapshot newest = ring[0]; // newest first
        Assert.Equal(1, newest.AdmissionRefusalsTotal);
        AdmissionRefusalEntry refusal = Assert.Single(newest.AdmissionRefusals!);
        Assert.Equal(AdmissionRefusalKinds.ContextLength, refusal.Kind);
    }

    // --- bus-level ring semantics and snapshot wire shape ---------------------------------------------

    [Fact]
    public async Task AdmissionRefusalsRing_KeepsLatestTenNewestFirst_AndTotalCountsEveryRefusal()
    {
        using BusHarness harness = new();
        await using EngineMetricsBus bus = new(
            "ring-refusal-model", harness.Scheduler, harness.Pool, cache: null, NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(30));

        for (int i = 1; i <= 12; i++)
        {
            bus.RecordAdmissionRefusal(AdmissionRefusalKinds.ConcurrencyQuota, "ring-refusal-model", $"refusal-{i}");
        }

        Assert.Equal(12, bus.AdmissionRefusalsTotal); // every refusal counts since open, ring retention aside

        IReadOnlyList<AdmissionRefusalEntry> retained = bus.RecentAdmissionRefusals();
        Assert.Equal(EngineMetricsBus.AdmissionRefusalRingCapacity, retained.Count); // bounded to the latest ten
        Assert.Equal("refusal-12", retained[0].Reason); // newest first
        Assert.Equal($"refusal-{12 - EngineMetricsBus.AdmissionRefusalRingCapacity + 1}", retained[^1].Reason);
    }

    [Fact]
    public async Task Snapshots_CarryRouteQuotaLaneAndRefusalFacts_InTheMetricsVocabulary()
    {
        using BusHarness harness = new();
        await using EngineMetricsBus bus = new(
            "facts-model", harness.Scheduler, harness.Pool, cache: null, NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(30));

        bus.AttachModelFacts(new ModelFacts(
            Type: "chat",
            WorkerThreads: 4,
            CoreSet: "0-7",
            ThreadQuota: 6,
            DecodeDevice: "cpu",
            PrefillDevice: "cpu",
            KvResidency: "dram",
            PrefillKernel: "gemv",
            MatmulWeightFormats: "F32",
            MaxConcurrentRequests: 4,
            RouteSegments: [new ModelRouteSegment("cpu", WeightTierNames.Dram, BatchCap: 16, ThreadQuota: 6, CoreSet: "0-7")],
            PrefillLane: PlacementPrefillLaneValues.Dedicated,
            PrefillClaimThreads: 8,
            PrefillDecodeShareThreads: 6,
            PrefillLaneShareThreads: 2,
            PrefillDecodeCoreSet: "4-7",
            PrefillLaneCoreSet: "0-3",
            PrefillActivationBytes: 1_966_080));

        bus.RecordAdmissionRefusal(AdmissionRefusalKinds.KvCapacity, "facts-model", "kv refusal");
        bus.RecordAdmissionRefusal(AdmissionRefusalKinds.ConcurrencyQuota, "facts-model", "quota refusal");

        // One admitted sequence moves the counters so a closing snapshot publishes with everything attached.
        int[] prompt = [.. harness.Tokenizer.Encode("world!")]; // declared pieces: two tokens under any tokenizer posture here
        PipelineHandle handle = harness.Scheduler.Submit(new ScheduledRequest
        {
            PromptTokens = prompt,
            MaxTokens = 4,
            Sampling = new SamplingParams { Temperature = 0f },
            EosId = harness.Tokenizer.EosId,
        }, CancellationToken.None);
        _ = await handle.Completion;

        using CancellationTokenSource deadline = new(TimeSpan.FromSeconds(60));
        IReadOnlyList<MetricsSnapshot> ring = await WaitForPublishedSnapshotAsync(bus, deadline.Token);
        MetricsSnapshot snapshot = ring[0];

        Assert.Equal(4, snapshot.MaxConcurrentRequests);
        ModelRouteSegment segment = Assert.Single(snapshot.RouteSegments!);
        Assert.Equal("cpu", segment.DeviceId);
        Assert.Equal(WeightTierNames.Dram, segment.WeightTier);
        Assert.Equal(16, segment.BatchCap);
        Assert.Equal(6, segment.ThreadQuota);
        Assert.Equal("0-7", segment.CoreSet);

        Assert.Equal(PlacementPrefillLaneValues.Dedicated, snapshot.PrefillLane);
        Assert.Equal(8, snapshot.PrefillClaimThreads);
        Assert.Equal(6, snapshot.PrefillDecodeShareThreads);
        Assert.Equal(2, snapshot.PrefillLaneShareThreads);
        Assert.Equal("4-7", snapshot.PrefillDecodeCoreSet);
        Assert.Equal("0-3", snapshot.PrefillLaneCoreSet);
        Assert.Equal(1_966_080, snapshot.PrefillActivationBytes);

        Assert.Equal(2, snapshot.AdmissionRefusalsTotal);
        IReadOnlyList<AdmissionRefusalEntry> refusals = snapshot.AdmissionRefusals!; // the published slice is newest-first too
        Assert.Equal("quota refusal", refusals[0].Reason);
        Assert.Equal("kv refusal", refusals[1].Reason);

        // Wire shape: every top-level property speaks the metrics_ vocabulary, and the new arrays nest plain field names.
        using JsonDocument document = JsonDocument.Parse(snapshot.ToJson());
        JsonElement root = document.RootElement;
        foreach (JsonProperty property in root.EnumerateObject())
        {
            Assert.StartsWith("metrics_", property.Name);
        }

        Assert.True(root.TryGetProperty("metrics_max_concurrent_requests", out _));
        // The nested segment carries its own quota, not the model-level cap: this route segment was declared 6 threads wide.
        Assert.Equal(6, root.GetProperty("metrics_route_segments")[0].GetProperty("thread_quota").GetInt32());
        Assert.Equal("cpu", root.GetProperty("metrics_route_segments")[0].GetProperty("device_id").GetString());
        Assert.Equal("dram", root.GetProperty("metrics_route_segments")[0].GetProperty("weight_tier").GetString());
        Assert.Equal(16, root.GetProperty("metrics_route_segments")[0].GetProperty("batch_cap").GetInt32());
        Assert.Equal("dedicated", root.GetProperty("metrics_prefill_lane").GetString());
        Assert.True(root.TryGetProperty("metrics_admission_refusals_total", out _));

        JsonElement refusalJson = root.GetProperty("metrics_admission_refusals")[0];
        Assert.Equal(AdmissionRefusalKinds.ConcurrencyQuota, refusalJson.GetProperty("kind").GetString());
        Assert.Equal("facts-model", refusalJson.GetProperty("model").GetString());
        Assert.Equal("quota refusal", refusalJson.GetProperty("reason").GetString());
    }

    [Fact]
    public async Task MonitorStream_SnapshotsCarryAdmissionRefusalDiagnostics_OnTheOperatorSurface()
    {
        string path = NewSyntheticModel();
        (int window, string longPrompt, string shortProbe) = MeasureContextSizing(path);

        await using SyntheticServer server = await SyntheticServer.StartAsync(
            path, maxContextTokens: window, kvBlocks: 2);

        (HttpResponseMessage refusedResponse, _) = await PostCompletionAsync(server, longPrompt, maxTokens: 4);
        Assert.Equal(HttpStatusCode.BadRequest, refusedResponse.StatusCode);

        // A small completion publishes a closing snapshot; the refusal ring rides it.
        (HttpResponseMessage okResponse, _) = await PostCompletionAsync(server, shortProbe, maxTokens: 2);
        okResponse.EnsureSuccessStatusCode();

        using CancellationTokenSource deadline = new(TimeSpan.FromSeconds(150));
        HttpRequestMessage request = new(HttpMethod.Get, $"{server.BaseUrl}/monitor/stream");
        request.Headers.Add("Last-Event-ID", "0"); // replay every retained snapshot: the refusal-carrying one is among them

        using HttpResponseMessage response = await server.Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, deadline.Token);
        Assert.True(response.IsSuccessStatusCode, $"stream request answered {(int)response.StatusCode}");

        StreamReader reader = new(await response.Content.ReadAsStreamAsync(deadline.Token));

        bool sawRefusalOnSnapshot = false;
        while (!deadline.IsCancellationRequested && !sawRefusalOnSnapshot)
        {
            string? line;
            try
            {
                line = await reader.ReadLineAsync(deadline.Token);
            }
            catch (OperationCanceledException)
            {
                break; // the deadline expired mid-read: fall through to the assertion below with its message
            }

            if (line is null)
            {
                break; // stream closed before the diagnostic arrived
            }

            if (!line.StartsWith("data: ", StringComparison.Ordinal))
            {
                continue;
            }

            using JsonDocument document = JsonDocument.Parse(line["data: ".Length..]);
            JsonElement root = document.RootElement;
            if (!root.TryGetProperty("metrics_admission_refusals", out JsonElement refusals) || refusals.ValueKind != JsonValueKind.Array)
            {
                continue; // request lifecycle and resync payloads carry no snapshot refusal array
            }

            foreach (JsonElement entry in refusals.EnumerateArray())
            {
                if (entry.GetProperty("kind").GetString() == AdmissionRefusalKinds.ContextLength
                    && entry.GetProperty("model").GetString() == server.ModelName)
                {
                    sawRefusalOnSnapshot = true; // the diagnostic names what blocked, on the live feed its consumer reads
                    break;
                }
            }
        }

        Assert.True(sawRefusalOnSnapshot, "no snapshot frame carried the recorded admission refusal");
    }

    // --- helpers ---------------------------------------------------------------------------------------

    private static int EncodedCount(string modelPath, string text)
    {
        using GgufFile file = GgufFile.Open(modelPath);
        return GgufTokenizer.FromGguf(file).Encode(text).Count; // the tokenizer borrows the file's views: disposing the file releases both
    }

    /// <summary>
    /// A prompt pair sized from what this fixture's own tokenizer measures, not from a guessed tokenization: the long
    /// prompt is measured past a chosen short one by at least four tokens, and its exact measured length becomes the
    /// configured window — so admission refuses it on equality while the short probe still fits with room to generate.
    /// </summary>
    private static (int Window, string LongPrompt, string ShortPrompt) MeasureContextSizing(string modelPath)
    {
        string[] candidates = ["world!", "hello", "!"];
        string? shortPrompt = candidates.FirstOrDefault(p => EncodedCount(modelPath, p) > 0);
        Assert.NotNull(shortPrompt); // a synthetic vocabulary whose every piece drops would make this fixture useless

        int shortTokens = EncodedCount(modelPath, shortPrompt);
        foreach (string join in new[] { " ", "! ", "!!" })
        {
            for (int reps = 1; reps <= 64; reps++)
            {
                string longPrompt = string.Concat(Enumerable.Repeat(shortPrompt + join, reps));
                int tokens = EncodedCount(modelPath, longPrompt);
                if (tokens >= shortTokens + 4)
                {
                    return (tokens, longPrompt, shortPrompt); // the window equals the measured prompt: equality trips the gate exactly
                }
            }
        }

        throw new InvalidOperationException("no repeated probe grew past its own token count — synthetic fixture cannot express a context refusal");
    }

    private static async Task<IReadOnlyList<MetricsSnapshot>> WaitForPublishedSnapshotAsync(EngineMetricsBus bus, CancellationToken ct)
    {
        while (true)
        {
            IReadOnlyList<MetricsSnapshot> ring = bus.SnapshotRing();
            if (ring.Count > 0)
            {
                return ring;
            }

            ct.ThrowIfCancellationRequested();
            await Task.Delay(50, CancellationToken.None);
        }
    }

    // Drives the facade until a submission is refused while another sequence holds what admission needs: the live
    // request runs stream-first so its first token proves it was admitted and still occupies the engine when the probe
    // submits. Each attempt is fresh state, so a run that ended before the probe landed simply retries.
    private static async Task<EngineBusyException> RunUntilRefusedAsync(SyntheticServer server, int probeMaxTokens)
    {
        const int maxAttempts = 60;

        for (int attempt = 1; attempt <= maxAttempts; attempt++)
        {
            using CancellationTokenSource holdLive = new();
            TaskCompletionSource<bool> firstToken = new(TaskCreationOptions.RunContinuationsAsynchronously);

            bool parkFirstTokenOnce = false;

            GenerationRequest live = new()
            {
                Model = server.ModelName,
                IsChat = false,
                Prompt = "world!world!", // four tokens from declared pieces: the sequence is non-empty under any tokenizer posture here
                MaxTokens = 24,
                Temperature = 0f,
                Stream = true,
                OnToken = _ =>
                {
                    firstToken.TrySetResult(true);

                    if (!parkFirstTokenOnce)
                    {
                        parkFirstTokenOnce = true;

                        // Hold the admitted sequence briefly after its first streamed token so the probe lands while that sequence still owns its KV block.
                        Thread.Sleep(250);
                    }
                },
            };

            Task<GenerationResult> liveTask = server.Engine.GenerateAsync(live, holdLive.Token);

            if (await TryProbeWhileLiveAsync(server, liveTask, firstToken.Task, probeMaxTokens) is EngineBusyException busy)
            {
                // Let the held sequence finish before returning so the engine and its pool settle.
                try
                {
                    _ = await liveTask;
                }
                catch (OperationCanceledException)
                {
                    // The refusal path may have ended it first; nothing this gate owns depends on its outcome.
                }

                return busy;
            }

            _ = await liveTask; // the live sequence finished before the probe could overlap: retry with fresh state
        }

        throw new InvalidOperationException($"no submission was refused across {maxAttempts} attempts — contention never overlapped an admitted sequence");
    }

    private static async Task<EngineBusyException?> TryProbeWhileLiveAsync(
        SyntheticServer server,
        Task<GenerationResult> liveTask,
        Task firstTokenSignal,
        int probeMaxTokens)
    {
        // Wait for proof the live sequence was admitted and advanced (its first token), but only while it still runs.
        Task winner = await Task.WhenAny(firstTokenSignal, liveTask);
        if (winner != firstTokenSignal || liveTask.IsCompleted)
        {
            return null;
        }

        GenerationRequest probe = new()
        {
            Model = server.ModelName,
            IsChat = false,
            Prompt = "world!", // two declared-piece tokens: admission is what this gate probes, never tokenization
            MaxTokens = probeMaxTokens,
            Temperature = 0f,
        };

        try
        {
            _ = await server.Engine.GenerateAsync(probe, CancellationToken.None);
            return null; // admitted while the live sequence ran: not this attempt's shape — retry with fresh state
        }
        catch (EngineBusyException busy)
        {
            return busy;
        }
    }

    private static async Task<JsonElement> GetFirstModelEntryAsync(SyntheticServer server)
    {
        HttpResponseMessage response = await server.Http.GetAsync($"{server.BaseUrl}/v1/models");
        response.EnsureSuccessStatusCode();

        using JsonDocument document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        return document.RootElement.GetProperty("data")[0].Clone(); // a detached copy: assertions outlive the parsed document
    }

    private static async Task<(HttpResponseMessage Response, JsonElement Body)> PostCompletionAsync(
        SyntheticServer server, string prompt, int maxTokens)
    {
        string body = JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["model"] = server.ModelName,
            ["prompt"] = prompt,
            ["max_tokens"] = maxTokens,
            ["temperature"] = 0f,
            ["stream"] = false,
        });

        HttpResponseMessage response = await server.Http.PostAsync(
            $"{server.BaseUrl}/v1/completions", new StringContent(body, Encoding.UTF8, "application/json"));

        using JsonDocument document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        return (response, document.RootElement.Clone()); // detached: callers assert after the parsed document is gone
    }

    /// <summary>Assertions shared by every route-segment report on either surface.</summary>
    private static class JsonSegmentAssert
    {
        public static void CpuSingleSegment(JsonElement segment, int? threadQuota, string? coreSetRanges)
        {
            Assert.Equal("cpu", segment.GetProperty("device_id").GetString());
            Assert.Equal(WeightTierNames.Dram, segment.GetProperty("weight_tier").GetString());
            Assert.Equal(TransformerModel.MaxBatchSequences, segment.GetProperty("batch_cap").GetInt32());

            if (threadQuota is int quota)
            {
                Assert.Equal(quota, segment.GetProperty("thread_quota").GetInt32());
            }
            else
            {
                // Resolved at the engine's own thread choice: the model-listing serializer omits null nested values and
                // the snapshot serializer writes them as JSON null — either spelling reads as "no cap on this segment".
                Assert.True(!segment.TryGetProperty("thread_quota", out JsonElement cap) || cap.ValueKind == JsonValueKind.Null);
            }

            if (coreSetRanges is string ranges)
            {
                Assert.Equal(ranges, segment.GetProperty("core_set").GetString());
            }
            else
            {
                // Unpinned reads the same way on both surfaces: absent or null.
                Assert.True(!segment.TryGetProperty("core_set", out _) || segment.GetProperty("core_set").ValueKind == JsonValueKind.Null);
            }
        }
    }

    /// <summary>One synthetic engine on an ephemeral port with chosen quota, placement, and pool knobs.</summary>
    private sealed class SyntheticServer : IAsyncDisposable
    {
        public GenerationService Engine { get; }
        public WebApplication App { get; }
        public HttpClient Http { get; } = new();
        public string BaseUrl { get; }
        public string ModelName { get; }

        // The API host attaches the bus while building; this reaches the same instance through the service.
        public EngineMetricsBus Bus => Engine.Metrics ?? throw new InvalidOperationException("the API host attached no metrics bus");

        private SyntheticServer(GenerationService engine, WebApplication app, string modelName)
        {
            Engine = engine;
            App = app;
            ModelName = modelName;
            BaseUrl = app.Urls.Single();
        }

        public static async Task<SyntheticServer> StartAsync(
            string modelPath,
            int? maxConcurrentRequests = null,
            int? threadQuota = null,
            IReadOnlySet<int>? coreSet = null,
            int maxContextTokens = 512,
            int kvBlocks = 8)
        {
            LairConfig config = new()
            {
                Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },

                // Single-threaded on purpose: these synthetic engines run beside other suites in parallel, and a tiny
                // pool of workers keeps step boundaries cheap for the contention attempts below.
                Engine = new EngineConfig { Threads = 1, MaxBatchTokens = 8 },
                Models =
                [
                    new ModelConfig
                    {
                        Name = "synthetic-quota",
                        Path = modelPath,
                        MaxContextTokens = maxContextTokens,
                        KvBlocks = kvBlocks,
                        MaxConcurrentRequests = maxConcurrentRequests,
                        Placement = new PlacementConfig
                        {
                            ThreadQuota = threadQuota,
                            CoreSet = coreSet ?? new HashSet<int>(),
                        },
                    },
                ],
            };

            GenerationService engine = GenerationService.Open(config);
            try
            {
                WebApplication app = ApiApp.Build(config, engine);
                await ((IHost)app).StartAsync(CancellationToken.None);
                return new SyntheticServer(engine, app, "synthetic-quota");
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

    /// <summary>Synthetic model + pool + scheduler opened directly against the bus, like the metrics suite's harness.</summary>
    private sealed class BusHarness : IDisposable
    {
        public GgufFile Gguf { get; }
        public GgufTokenizer Tokenizer { get; }
        public CpuBackend Backend { get; }
        public TransformerModel Model { get; }
        public BlockPool Pool { get; }
        public GenerationScheduler Scheduler { get; }

        public BusHarness()
        {
            string modelPath = SyntheticLlamaGguf.Write("bus-harness");

            Gguf = GgufFile.Open(modelPath);
            Tokenizer = GgufTokenizer.FromGguf(Gguf);
            Backend = new CpuBackend(threads: 1);
            Model = TransformerModel.Open(Gguf, Backend, 512);
            Pool = new BlockPool(
                blockTokens: 16,
                layerCount: Model.Desc.LayerCount,
                kvHeads: Model.Desc.KvHeadCount,
                headDim: Model.Desc.HeadDim,
                capacityBlocks: 8,
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
