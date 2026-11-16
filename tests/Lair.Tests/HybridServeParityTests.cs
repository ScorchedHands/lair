using System.Runtime.InteropServices;

using Lair.Contracts;
using Lair.Host.Config;
using Lair.Http;
using Lair.Loader;
using Lair.Models;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// Serve-level hybrid parity at the whole-model transformer boundary: a synthetic model opened CPU-only and reopened with every
/// transformer layer on device 'cuda' serves greedy tokens identically, while its CUDA span reports accelerator-resident weights.
    /// The same posture is measured against the owner resource envelope: persistent device weight residency stays under 256 MiB per gate
    /// run and these synthetic engines' own working-set growth remains bounded well inside 4 GiB, so every automated hybrid case here
/// runs on synthetic fixtures only — a larger pinned real-model fixture once exercised this boundary by hand, but its evidence is
/// historical documentation, not an active gate at these sizes. A finite accelerator weight ledger proves that a declared VRAM-backed
/// chain refuses at load naming model, resource, tier, and amounts before an engine half-runs it.
/// </summary>
[CollectionDefinition("resource-envelope", DisableParallelization = true)]
public sealed class ResourceEnvelopeCollection;

[Trait("Category", "Parity-Hybrid")]
[Collection("resource-envelope")]
public sealed class HybridServeParityTests : IDisposable
{
    // The synthetic footprint these gates hold inside: persistent device weight residency under 256 MiB, plus a gate-scoped working-set increment bounded well below the declared 4 GiB DRAM ceiling. Whole-process absolute residue from unrelated cases is deliberately not asserted here; live serve measures absolute RSS separately.
    private const long PersistentDeviceWeightEnvelopeBytes = 256L * (1L << 20);

    // The growth ceiling above the floor sampled before any engine opens: what these synthetic postures may add to the working set — model views, activation buffers, and the driver context a split route retains. Measured growth for both gates on this host sits near 400 MiB; the ceiling leaves generous headroom for driver-generation variance while staying well below the declared DRAM envelope.
    private const long ProcessWorkingSetGrowthCeilingBytes = 768L * (1L << 20);

    // The wider synthetic shape only the residency gate declares: hidden 1024 over eight heads of dimension 128, four kv heads, a feed-forward width matching the hidden size. Its dense transformer matrices are exactly what the accelerator span must hold resident on device.
    private static readonly SyntheticLlamaGguf.Geometry ResidencyShape = new(Layers: 6, HiddenSize: 1024, HeadCount: 8, KvHeadCount: 4, FeedForwardLength: 1024);

    // The dense matrices per transformer layer the copy posture caches on device — q/k/v/output and gate/up/down. Norm vectors cross through host-side row dequantization; embeddings and the output head stay off this span in a whole-model split.
    private const int DenseMatricesPerLayer = 7;

    // The quantized element size the residency fixture ships its matrices in: half-precision, which both backends read without widening loss on these payloads.
    private const long MatrixElementBytes = sizeof(short);

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
    public void WholeModelTransformerSpan_ServesTokenIdenticalToCpuOnly_AndResolvesCudaSegmentAtVram()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return; // the CUDA driver seam is exercised on Windows against nvcuda.dll.
        }

        long processFloor = ProcessBytes();

        string path = SyntheticModel("hybrid-whole-split");
        int layerCount = LayerCount(path);

        using GenerationService baseline = GenerationService.Open(ConfigFor(path, "hybrid-cpu-baseline", gpuLayers: 0), NullLogger.Instance);
        Assert.Single(baseline.ResolvedRoute.Segments);
        string cpuText = Generate(baseline, maxTokens: 4);

        try
        {
            using GenerationService hybrid = GenerationService.Open(ConfigFor(path, "hybrid-cuda-whole", gpuLayers: layerCount), NullLogger.Instance);

            Assert.Equal(2, hybrid.ResolvedRoute.Segments.Count); // every transformer layer runs on CUDA; the output head remains a host-side tail span.
            Assert.Equal("cuda", hybrid.ResolvedRoute.Segments[0].DeviceId);
            Assert.Equal(ModelRoute.CpuDeviceId, hybrid.ResolvedRoute.Segments[1].DeviceId);

            Assert.Equal(WeightTier.Vram, hybrid.ResolvedRoute.Segments[0].WeightTier); // the opened accelerator backend reports device-resident weights for its span.
            Assert.Equal(WeightTier.Dram, hybrid.ResolvedRoute.Segments[1].WeightTier);  // the host-side tail answers from CPU storage.

            string hybridText = Generate(hybrid, maxTokens: 4);
            Assert.Equal(cpuText, hybridText); // prompt and continued steps are token-identical across the whole-model transformer span.

            long deviceWeightBytes = hybrid.SplitExecutionPersistentDeviceWeightBytes!.Value;
            Assert.True(deviceWeightBytes > 0L, "the opened accelerator span cached no weight copy at all: the residency accounting read back nothing while its kernels ran.");
            Assert.True(
                deviceWeightBytes < PersistentDeviceWeightEnvelopeBytes,
                $"whole-model split persistent device weights hold {deviceWeightBytes} bytes, at or above the 256 MiB envelope.");

            long workingSet = ProcessBytes();
            AssertWorkingSetWithinEnvelope(floor: processFloor, measured: workingSet);
        }
        catch (Exception ex) when (IsUnavailableCudaSeam(ex))
        {
            return; // no usable driver or device behind this host's accelerator seam: the synthetic gate cannot exercise a hybrid posture here.
        }
    }

    [Fact]
    public void WholeModelTransformerSpan_KeepsPersistentDeviceWeightResidencyWithinEnvelope()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return; // the CUDA driver seam is exercised on Windows against nvcuda.dll.
        }

        long processFloor = ProcessBytes();

        string path = SyntheticModel("hybrid-residency-envelope", ResidencyShape);
        int layerCount = LayerCount(path);

        try
        {
            using GenerationService hybrid = GenerationService.Open(ConfigFor(path, "hybrid-cuda-whole", gpuLayers: layerCount), NullLogger.Instance);

            Assert.Equal(2, hybrid.ResolvedRoute.Segments.Count); // the wider fixture keeps a meaningful whole-model split: every transformer layer on CUDA, the output head as its host-side tail span.
            Assert.Equal("cuda", hybrid.ResolvedRoute.Segments[0].DeviceId);
            Assert.Equal(ModelRoute.CpuDeviceId, hybrid.ResolvedRoute.Segments[1].DeviceId);

            Assert.Equal(WeightTier.Vram, hybrid.ResolvedRoute.Segments[0].WeightTier); // the opened accelerator backend reports device-resident weights for its span.
            Assert.Equal(WeightTier.Dram, hybrid.ResolvedRoute.Segments[1].WeightTier);  // the host-side tail answers from CPU storage.

            string text = Generate(hybrid, maxTokens: 3); // weight passes land on the accelerator span and cache every dense matrix exactly once.
            Assert.False(string.IsNullOrEmpty(text));

            int allocations = hybrid.SplitExecutionPersistentDeviceWeightAllocations!.Value;
            long deviceBytes = hybrid.SplitExecutionPersistentDeviceWeightBytes!.Value;

            Assert.Equal(ResidencyShape.Layers * DenseMatricesPerLayer, allocations); // one persistent allocation per dense transformer matrix across the split span.
            Assert.Equal(ExpectedSplitSpanBytes(), deviceBytes); // deterministic accounting over the declared geometry: every cached tensor at its full allocated length.
            Assert.True(
                deviceBytes < PersistentDeviceWeightEnvelopeBytes,
                $"whole-model split persistent device weights hold {deviceBytes} bytes, at or above the 256 MiB envelope.");

            long workingSet = ProcessBytes(); // sampled while this engine is open and its weight passes ran: model views resident on host, device copies held by the backend.
            AssertWorkingSetWithinEnvelope(floor: processFloor, measured: workingSet);
        }
        catch (Exception ex) when (IsUnavailableCudaSeam(ex))
        {
            return; // no usable driver or device behind this host's accelerator seam: the residency posture cannot be opened here.
        }
    }

    [Fact]
    public void HybridWeightClaimAgainstFiniteVramLedger_RefusesOpenNamingWeightsTierAndAmounts()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return; // the CUDA driver seam is exercised on Windows against nvcuda.dll.
        }

        string path = SyntheticModel("hybrid-budget-exhaust"); // the tiny synthetic fixture: this gate proves refusal at load, so it never opens an engine or copies a weight byte anywhere.
        int layerCount = LayerCount(path);

        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Vram, Resource = BudgetResource.Weights, Capacity = 0 }]);

        try
        {
            GenerationService engine = GenerationService.Open(
                ConfigFor(path, "hybrid-budget-exhaust", gpuLayers: layerCount) with
                {
                    Models =
                    [
                        new ModelConfig
                        {
                            Name = "hybrid-budget-exhaust",
                            Path = path,
                            MaxContextTokens = 512,
                            KvBlocks = 8,
                            Placement = new PlacementConfig { GpuLayers = layerCount },
                            Budgets = [new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Vram] }],
                        },
                    ],
                },
                NullLogger.Instance,
                ledger);

            Assert.Fail($"expected the finite accelerator weight seam to refuse '{engine.Model.Id}' before this model opened");
        }
        catch (BudgetExhaustedException error)
        {
            Assert.Contains("model 'hybrid-budget-exhaust'", error.Message);
            Assert.Contains("'weights'", error.Message);
            Assert.Contains("tier 'vram'", error.Message);
            Assert.Contains("'vram' holds 0 of 0 megabytes", error.Message);
            Assert.Contains("megabytes", error.Message); // the refusal states this resource's unit beside every consulted position.
        }
        catch (Exception ex) when (IsUnavailableCudaSeam(ex))
        {
            return; // no usable driver or device behind this host's accelerator seam: the finite VRAM posture cannot be opened here.
        }
    }

    private static bool IsUnavailableCudaSeam(Exception error) =>
        error is DllNotFoundException
            or EntryPointNotFoundException
            or TypeInitializationException
            || (error is InvalidOperationException invalid && invalid.Message.Contains(ModelRoute.CudaBackendSeam, StringComparison.Ordinal));

    [Fact]
    public void HybridSplitWithoutUsableDevice_LeavesNoHalfOpenedEngine()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return; // the CUDA driver seam is exercised on Windows against nvcuda.dll.
        }

        string path = SyntheticModel("hybrid-nodevice");
        int layerCount = LayerCount(path);

        try
        {
            using GenerationService engine = GenerationService.Open(ConfigFor(path, "hybrid-nodevice", gpuLayers: layerCount), NullLogger.Instance);

            // A host with a usable seam opens the split normally; this gate proves no half-opened posture reaches serving when that seam is absent.
            Assert.Equal(2, engine.ResolvedRoute.Segments.Count);
        }
        catch (InvalidOperationException error) when (error.Message.Contains("'placement.gpu_layers'") && error.Message.Contains(ModelRoute.CudaBackendSeam))
        {
            // The refusal names the declared split and where accelerator-backed spans run before any serving engine exists.
        }
    }

    private string SyntheticModel(string modelName, SyntheticLlamaGguf.Geometry? geometry = null)
    {
        // Only the residency gate declares a wider shape; every other case keeps the original tiny fixture byte-for-byte. F16 ships this file's matrices in half-precision payloads that survive stored formats unchanged on either side of a split route.
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

    // The deterministic byte total the copy posture must hold for this fixture's split span: every dense matrix of every layer at its full allocated length in the shipped element size. Norm vectors, embeddings, and the output head stay host-side and contribute nothing here.
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
        // This gate measures only what these synthetic engines add. Shared test assemblies can carry unrelated resident pages before and after this case, so an absolute process total here would measure the host rather than the posture under test; live serve covers absolute RSS separately.
        Assert.True(
            measured - floor <= ProcessWorkingSetGrowthCeilingBytes,
            $"these hybrid postures grew the process working set by {measured - floor} bytes, above the growth ceiling for a synthetic split run.");
    }

    // The current process's resident pages as the runtime reports them: model file views touched since their opens, activation buffers, and whatever the driver seam maps while an accelerator posture is alive. Sampled only inside these scoped gates so unrelated cases sharing this process cannot move one of their assertions on their own — a concurrent large-model holder inflates both floor and measured samples alike, leaving every growth comparison it skews bounded by its own ceiling check instead.
    private static long ProcessBytes() => Environment.WorkingSet;

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
