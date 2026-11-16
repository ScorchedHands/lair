using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

using Lair.Contracts;
using Lair.Host;
using Lair.Host.Config;
using Lair.Http;
using Lair.Http.Monitoring;
using Lair.Loader;
using Lair.Models;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// Monitor counter gate for the hybrid posture, at serve level and on synthetic fixtures only: a whole-model split opened
/// through the monitor's own bus publishes engine-side accelerator facts — windowed kernel busy share, cumulative runtime
/// transfer bytes and calls, and persistent device weight residency — while its decode runs, with those readings coming
/// from what the opened CUDA backend answers about its own device work. The same gate opens a CPU-only serve on the tiny
/// fixture and proves its snapshots carry nothing in those slots rather than zeros no engine fact supports, so no surface
/// downstream can render an idle-hardware reading for a model that holds no accelerator open. Persistent residency is held
/// under 256 MiB by construction here; DRAM evidence measures this gate's own working-set growth instead of whole-process
/// residue shared with unrelated cases.
/// </summary>
[Trait("Category", "Cuda")]
[Trait("Category", "MonitorCounters")]
[Collection("model-loaded")] // shares the serialized engine lane with the suites that measure latency budgets here: opening these synthetic engines never starves a composition measurement elsewhere in the assembly
public sealed class HybridMonitorCounterTests : IDisposable
{
    // The residency envelope this gate holds its synthetic split inside, byte-for-byte the ceiling an automated hybrid case declares.
    private const long PersistentDeviceWeightEnvelopeBytes = 256L * (1L << 20);

    // Growth ceiling above the floor sampled before any engine opens: what these synthetic postures may add to the working set —
    // model views, activation buffers, and the driver context a split route retains. Absolute process totals stay unasserted here.
    private const long ProcessWorkingSetGrowthCeilingBytes = 768L * (1L << 20);

    // The shape the residency gate declares: hidden 1024 over eight heads of dimension 128, four kv heads, feed-forward matching
    // the hidden size across six layers — dense matrices big enough to make a device-residency reading meaningful while staying
    // far inside the envelope.
    private static readonly SyntheticLlamaGguf.Geometry ResidencyShape = new(Layers: 6, HiddenSize: 1024, HeadCount: 8, KvHeadCount: 4, FeedForwardLength: 1024);

    private const int DenseMatricesPerLayer = 7;
    private const long MatrixElementBytes = sizeof(short); // F16 payloads on both sides of the seam

    private static readonly string[] AcceleratorFields =
    [
        "metrics_gpu_utilization_percent",
        "metrics_pcie_transfer_bytes_total",
        "metrics_pcie_transfers_total",
        "metrics_persistent_vram_bytes",
    ];

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
    public void HybridDecodeSnapshots_PopulateAcceleratorFacts_AndKeepPersistentResidencyWithinEnvelope()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return; // the CUDA driver seam is exercised on Windows against nvcuda.dll.
        }

        long processFloor = Environment.WorkingSet;

        string path = SyntheticModel("monitor-hybrid-counters", ResidencyShape);
        int layerCount = LayerCount(path);

        try
        {
            using GenerationService hybrid = GenerationService.Open(ConfigFor(path, "monitor-hybrid-counters", gpuLayers: layerCount), NullLogger.Instance);
            EngineMetricsBus bus = hybrid.AttachMetrics(NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(50));

            Assert.Equal(2, hybrid.ResolvedRoute.Segments.Count); // every transformer layer on CUDA, the output head as its host-side tail span.
            AcceleratorAccounting openedFacts = hybrid.OpenedAcceleratorAccounting
                ?? throw new InvalidOperationException("the opened hybrid engine answered no accelerator accounting.");

            // Nothing has crossed yet: residency materializes when a weight pass first reaches the span, and these tallies
            // fold only what a compute call crosses — backend open alone answers traffic nothing at all.
            Assert.Equal(0L, openedFacts.PersistentDeviceWeightBytes);
            Assert.Equal(0L, openedFacts.RuntimeTransferCallsTotal);
            Assert.Equal(0L, openedFacts.RuntimeTransferBytesTotal);
            Assert.Empty(bus.SnapshotRing()); // nothing has advanced yet: the bus invented no snapshot for an idle engine.

            string text = Generate(hybrid, maxTokens: 4);
            Assert.False(string.IsNullOrEmpty(text));

            // Let the sampling cadence close its window before reading what the bus published: a fast synthetic load can end
            // between ticks, and the interval that saw those completions publishes the snapshot carrying their tallies.
            Thread.Sleep(TimeSpan.FromMilliseconds(300));

            MetricsSnapshot snapshot = WaitForAcceleratorSnapshot(bus);

            AcceleratorAccounting decodeFacts = hybrid.OpenedAcceleratorAccounting
                ?? throw new InvalidOperationException("the hybrid engine stopped answering accelerator accounting mid-serve.");
            Assert.True(
                decodeFacts.RuntimeTransferCallsTotal > 0L,
                "no runtime copy operation was accounted while CUDA compute ran: the transfer tallies never saw a GEMV crossing.");
            Assert.True(decodeFacts.RuntimeTransferBytesTotal > 0L, "compute ran but no activation or output byte crossed into the transfer tally.");

            // The counters answered actual decode rather than backend opening, and stay bounded well clear of any unit mix-up:
            // every copy here is one vector crossing, so a handful of decode steps cannot reach megabytes of traffic let alone
            // the weight residency itself.
            Assert.True(
                decodeFacts.RuntimeTransferBytesTotal < PersistentDeviceWeightEnvelopeBytes / 4,
                $"runtime transfers total {decodeFacts.RuntimeTransferBytesTotal} bytes for a few synthetic decode steps — above a quarter of the residency envelope.");

            double utilization = snapshot.GpuUtilizationPercent!.Value;
            Assert.True(utilization > 0.0, "kernel busy measured zero across a window in which CUDA kernels ran.");

            // A serialized stream's kernel busy share cannot approach twice the wall clock it was measured against: this bound
            // catches a unit mix-up (device microseconds read as milliseconds or seconds) without pinning device speed.
            Assert.True(utilization < 200.0, $"kernel busy reads {utilization}% of its window — beyond any honest reading of one serve's kernels.");

            long persistentVramBytes = snapshot.PersistentVramBytes!.Value;

            // Weight residency reads as what the copy posture holds while its kernels run. The deterministic byte total is
            // asserted against the engine's own settled reading — every dense matrix of the split span, cached once at its
            // full allocated length — and against that ceiling the published fact must stand inside too: a snapshot taken mid-
            // serve can legitimately carry less than the finished span, so it bounds rather than pins. Either way the gate's
            // authoritative device-residency claim stays under 256 MiB while hybrid decode is active.
            Assert.Equal(ExpectedSplitSpanBytes(), decodeFacts.PersistentDeviceWeightBytes);
            Assert.True(
                decodeFacts.PersistentDeviceWeightBytes < PersistentDeviceWeightEnvelopeBytes,
                $"whole-model split persistent device weights hold {decodeFacts.PersistentDeviceWeightBytes} bytes — at or above the 256 MiB envelope.");

            Assert.InRange(persistentVramBytes, 1L, PersistentDeviceWeightEnvelopeBytes - 1L);
            Assert.True(
                persistentVramBytes <= decodeFacts.PersistentDeviceWeightBytes,
                $"the published persistent VRAM fact ({persistentVramBytes}) exceeds what the opened backend holds ({decodeFacts.PersistentDeviceWeightBytes}).");

            // Transfer tallies read cumulative-since-open on every surface: what the closing window published stands at or below
            // what the engine answers now, and a later snapshot can only carry more. Exact equality here would pin token counts
            // to this host's forward speed rather than the accounting itself.
            Assert.InRange(snapshot.PcieTransferBytesTotal!.Value, 1L, decodeFacts.RuntimeTransferBytesTotal);
            Assert.InRange(snapshot.PcieTransfersTotal!.Value, 1L, decodeFacts.RuntimeTransferCallsTotal);

            // The wire contract: every published property speaks metrics_ vocabulary and carries a value here.
            using (JsonDocument document = JsonDocument.Parse(snapshot.ToJson()))
            {
                JsonElement root = document.RootElement;
                foreach (string field in AcceleratorFields)
                {
                    Assert.True(root.TryGetProperty(field, out JsonElement value), $"snapshot JSON carried no '{field}': {snapshot.ToJson()}");
                    Assert.NotEqual(JsonValueKind.Null, value.ValueKind);
                }

                foreach (JsonProperty property in root.EnumerateObject())
                {
                    Assert.StartsWith("metrics_", property.Name);
                }

                Assert.True(root.GetProperty("metrics_gpu_utilization_percent").GetDouble() > 0.0);
            }

            long workingSet = Environment.WorkingSet; // sampled while this engine is open and its weight passes ran
            AssertWorkingSetWithinEnvelope(floor: processFloor, measured: workingSet);
        }
        catch (Exception ex) when (IsUnavailableCudaSeam(ex))
        {
            return; // no usable driver or device behind this host's accelerator seam: the synthetic gate cannot exercise a hybrid posture here.
        }

        AssertWorkingSetWithinEnvelopeFloorOnly(processFloor);
    }

    [Fact]
    public void CpuOnlyServeSnapshots_LeaveAcceleratorFactsNull_AndPublishNoDeviceReading()
    {
        long processFloor = Environment.WorkingSet;

        string path = SyntheticModel("monitor-cpu-only"); // the tiny fixture: no accelerator posture is declared at all.

        using GenerationService cpuOnly = GenerationService.Open(ConfigFor(path, "monitor-cpu-only", gpuLayers: 0), NullLogger.Instance);
        EngineMetricsBus bus = cpuOnly.AttachMetrics(NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(50));

        Assert.Null(cpuOnly.OpenedAcceleratorAccounting); // a CPU-only open attached no accelerator accounting: it has no backend to answer from.

        string text = Generate(cpuOnly, maxTokens: 4);
        Assert.False(string.IsNullOrEmpty(text));

        IReadOnlyList<MetricsSnapshot> ring = WaitForSnapshots(bus, TimeSpan.FromSeconds(30));
        Assert.True(ring.Count > 0, "the CPU-only serve published no snapshot while its decode ran — the gate has nothing to inspect.");

        foreach (MetricsSnapshot snapshot in ring)
        {
            Assert.Null(snapshot.GpuUtilizationPercent);
            Assert.Null(snapshot.PcieTransferBytesTotal);
            Assert.Null(snapshot.PcieTransfersTotal);
            Assert.Null(snapshot.PersistentVramBytes);

            using JsonDocument document = JsonDocument.Parse(snapshot.ToJson());
            foreach (string field in AcceleratorFields)
            {
                // Absent or explicitly null both read as nothing: a downstream surface must never mistake either for an idle device.
                if (document.RootElement.TryGetProperty(field, out JsonElement value))
                {
                    Assert.Equal(JsonValueKind.Null, value.ValueKind);
                }
            }

            foreach (JsonProperty property in document.RootElement.EnumerateObject())
            {
                Assert.StartsWith("metrics_", property.Name); // the additive fields joined the vocabulary without renaming anything.
            }
        }

        AssertWorkingSetWithinEnvelopeFloorOnly(processFloor);
    }

    [Fact]
    public async Task CpuOnlyMonitorRows_KeepClassicColumns_AndPrintNoAcceleratorTail()
    {
        long processFloor = Environment.WorkingSet;

        string path = SyntheticModel("monitor-cpu-cli");
        LairConfig config = ConfigFor(path, "monitor-cpu-cli", gpuLayers: 0) with
        {
            Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
        };

        using GenerationService engine = GenerationService.Open(config, NullLogger.Instance);
        WebApplication app = ApiApp.Build(config, engine);
        await ((IHost)app).StartAsync(CancellationToken.None);
        try
        {
            // Load in flight while the monitor attaches: snapshots reach a subscriber only from its own subscription onward.
            Task<string> load = Task.Run(() => Generate(engine, maxTokens: 8));

            StringWriter output = new();
            using CancellationTokenSource cancel = new(TimeSpan.FromSeconds(30));
            Task<int> run = MonitorCommand.RunAsync(new Uri(app.Urls.Single()), output, cancel.Token);

            Stopwatch clock = Stopwatch.StartNew();
            bool sawRow = false;
            while (clock.Elapsed < TimeSpan.FromSeconds(25) && !sawRow)
            {
                sawRow = DataRows(output.ToString()).Count > 0;
                if (!sawRow)
                {
                    await Task.Delay(200);
                }
            }

            cancel.Cancel(); // Ctrl+C-equivalent: the monitor ends on its own with 0
            int exitCode = await run;
            Assert.False(string.IsNullOrEmpty(await load)); // the host's own engine completed its greedy run

            Assert.True(sawRow, $"the monitor never rendered a table row:{Environment.NewLine}{output}");
            Assert.Equal(0, exitCode);

            List<string> rows = DataRows(output.ToString());
            Assert.Contains(rows, row => row.Contains("monitor-cpu-cli")); // the classic columns still render for a CPU-only serve.
            Assert.DoesNotContain(rows, row => row.Contains("persistent-vram", StringComparison.Ordinal));
            Assert.DoesNotContain(rows, row => row.Contains(" gpu ", StringComparison.Ordinal)); // no device column without an engine fact behind it.
        }
        finally
        {
            await ((IHost)app).StopAsync(TimeSpan.FromSeconds(5));
            await app.DisposeAsync();
        }

        AssertWorkingSetWithinEnvelopeFloorOnly(processFloor);
    }

    // ---------------------------------------------------------------- helpers

    // Rendered monitor rows only: the table header names its columns instead of a model.
    private static List<string> DataRows(string text) =>
        [.. text.Split('\n', StringSplitOptions.RemoveEmptyEntries).Where(l => !l.StartsWith("model", StringComparison.Ordinal))];

    // The ring once this engine has published at least one snapshot — a fast synthetic load can finish inside a single
    // sampling window, and the closing tick that carries it lands just after the load returned.
    private static IReadOnlyList<MetricsSnapshot> WaitForSnapshots(EngineMetricsBus bus, TimeSpan timeout)
    {
        using CancellationTokenSource deadline = new(timeout);
        while (!deadline.IsCancellationRequested)
        {
            if (bus.PublishedTotal > 0)
            {
                return bus.SnapshotRing();
            }

            Thread.Sleep(100);
        }

        return bus.SnapshotRing(); // the caller's own assertion reports an empty ring with its own message.
    }

    // The newest published window that actually measured device work: kernel busy above zero, weight residency held open, and
    // runtime copies tallied. Windows covering only host-side admission can legitimately report nothing yet, so the gate reads
    // whichever published window saw compute rather than assuming the last one did.
    private static MetricsSnapshot WaitForAcceleratorSnapshot(EngineMetricsBus bus)
    {
        using CancellationTokenSource deadline = new(TimeSpan.FromSeconds(120));
        while (!deadline.IsCancellationRequested)
        {
            foreach (MetricsSnapshot snapshot in bus.SnapshotRing()) // newest first already
            {
                if (snapshot.GpuUtilizationPercent is > 0.0
                    && snapshot.PersistentVramBytes is > 0L
                    && snapshot.PcieTransferBytesTotal is > 0L)
                {
                    return snapshot;
                }
            }

            Thread.Sleep(100);
        }

        Assert.Fail("no published window measured kernel busy, weight residency, or runtime copies while the hybrid engine was serving: the bus read no accounting from an opened CUDA backend.");
        return null!;
    }

    private static bool IsUnavailableCudaSeam(Exception error) =>
        error is DllNotFoundException
            or EntryPointNotFoundException
            or TypeInitializationException
            || (error is InvalidOperationException invalid && invalid.Message.Contains(ModelRoute.CudaBackendSeam, StringComparison.Ordinal))
            || (error is Lair.Backends.Cuda.CudaException cuda && cuda.Result is Lair.Backends.Cuda.CudaResult.NoDevice or Lair.Backends.Cuda.CudaResult.DeviceUnavailable);

    private string SyntheticModel(string modelName, SyntheticLlamaGguf.Geometry? geometry = null)
    {
        string path = geometry is null
            ? SyntheticLlamaGguf.Write(modelName)
            : SyntheticLlamaGguf.Write(modelName, withReasoningMarkers: true, QuantFormat.F16, geometry);
        _tempFiles.Add(path);
        return path;
    }

    private static int LayerCount(string modelPath)
    {
        using GgufFile probe = GgufFile.Open(modelPath);
        return ModelPlans.Resolve(probe, logLine: null).LayerCount;
    }

    // The deterministic byte total the copy posture must hold for this fixture's split span: every dense matrix of every layer at
    // its full allocated length in the shipped element size. Norm vectors, embeddings, and the output head stay host-side.
    private static long ExpectedSplitSpanBytes()
    {
        int headDim = ResidencyShape.HiddenSize / ResidencyShape.HeadCount;
        long qRows = (long)ResidencyShape.HeadCount * headDim;
        long kvRows = (long)ResidencyShape.KvHeadCount * headDim;

        long perLayerElements = 2L * qRows * ResidencyShape.HiddenSize // attn_q and attn_output
            + 2L * kvRows * ResidencyShape.HiddenSize                  // attn_k and attn_v
            + 3L * ResidencyShape.FeedForwardLength * ResidencyShape.HiddenSize; // ffn_gate, ffn_up, and ffn_down

        return perLayerElements * MatrixElementBytes * ResidencyShape.Layers;
    }

    private static void AssertWorkingSetWithinEnvelope(long floor, long measured)
    {
        Assert.True(
            measured - floor <= ProcessWorkingSetGrowthCeilingBytes,
            $"these hybrid postures grew the process working set by {measured - floor} bytes, above the growth ceiling for a synthetic split run.");
    }

    // Nothing opened past this point may have leaked into the working set beyond what these synthetic engines hold while open.
    private static void AssertWorkingSetWithinEnvelopeFloorOnly(long floor)
    {
        long measured = Environment.WorkingSet;
        Assert.True(
            measured - floor <= ProcessWorkingSetGrowthCeilingBytes,
            $"after teardown the process working set sits {measured - floor} bytes above its pre-open floor — this gate left more resident than it opened.");
    }

    private static LairConfig ConfigFor(string modelPath, string modelName, int gpuLayers) => new()
    {
        Server = new ServerConfig(),
        Engine = new EngineConfig { Threads = 1 },
        Models =
        [
            new ModelConfig
            {
                Name = modelName,
                Path = modelPath,
                MaxContextTokens = 512,
                KvBlocks = 8,
                Placement = new PlacementConfig { GpuLayers = gpuLayers },
            },
        ],
    };

    private static string Generate(GenerationService engine, int maxTokens) =>
        engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = "hello",
            MaxTokens = maxTokens,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult().Text;
}
