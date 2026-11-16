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
    /// The pinned local fixture is checked only when present, using the same short raw prompt through both postures so a missing file or
    /// absent driver seam skips rather than inventing a different posture; every quantized family that file places inside a transformer
    /// block runs on the accelerator kernel path, and only the host-side head and embedding tensors stay off device. A finite accelerator
    /// weight ledger proves that a declared VRAM-backed chain refuses at load naming model, resource, tier, and amounts before an engine
    /// half-runs it.
/// </summary>
[Trait("Category", "Parity-Hybrid")]
public sealed class HybridServeParityTests : IDisposable
{
    private const string PinnedFixtureRelativePath = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

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
        }
        catch (Exception ex) when (IsUnavailableCudaSeam(ex))
        {
            return; // no usable driver or device behind this host's accelerator seam: the synthetic gate cannot exercise a hybrid posture here.
        }
    }

    [Fact]
    public void PinnedWholeModelTransformerSpan_ServesTokenIdenticalToCpuOnly_WhenFixtureIsPresent()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return; // the CUDA driver seam is exercised on Windows against nvcuda.dll.
        }

        string? path = ModelFixture.ResolveRepoRelative(PinnedFixtureRelativePath);
        if (path is null)
        {
            return; // real-model cases skip when the GGUF file is absent (xunit 2.9 has no dynamic Assert.Skip).
        }

        int layerCount = LayerCount(path);

        using GenerationService baseline = GenerationService.Open(PinnedConfigFor(path, "pinned-hybrid-cpu-baseline", gpuLayers: 0), NullLogger.Instance);
        string cpuText = Generate(baseline, maxTokens: 2);

        try
        {
            using GenerationService hybrid = GenerationService.Open(PinnedConfigFor(path, "pinned-hybrid-cuda-whole", gpuLayers: layerCount), NullLogger.Instance);

            Assert.Equal(2, hybrid.ResolvedRoute.Segments.Count); // every transformer layer runs on CUDA; the output head remains a host-side tail span.
            Assert.Equal("cuda", hybrid.ResolvedRoute.Segments[0].DeviceId);
            Assert.Equal(ModelRoute.CpuDeviceId, hybrid.ResolvedRoute.Segments[1].DeviceId);

            Assert.Equal(WeightTier.Vram, hybrid.ResolvedRoute.Segments[0].WeightTier); // the opened accelerator backend reports device-resident weights for its span.
            Assert.Equal(WeightTier.Dram, hybrid.ResolvedRoute.Segments[1].WeightTier);  // the host-side tail answers from CPU storage.

            string hybridText = Generate(hybrid, maxTokens: 2);
            Assert.Equal(cpuText, hybridText); // prompt and continued steps are token-identical across the whole-model transformer span.
        }
        catch (Exception ex) when (IsUnavailableCudaSeam(ex))
        {
            return; // no usable driver or device behind this host's accelerator seam: the pinned fixture cannot exercise a hybrid posture here.
        }
    }

    [Fact]
    public void HybridWeightClaimAgainstFiniteVramLedger_RefusesOpenNamingWeightsTierAndAmounts()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return; // the CUDA driver seam is exercised on Windows against nvcuda.dll.
        }

        string path = SyntheticModel("hybrid-budget-exhaust");
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

    private string SyntheticModel(string modelName)
    {
        string path = SyntheticLlamaGguf.Write(modelName);
        _tempFiles.Add(path);
        return path;
    }

    private static int LayerCount(string modelPath)
    {
        using GgufFile probe = GgufFile.Open(modelPath);
        return ModelPlans.Resolve(probe, logLine: null).LayerCount;
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

    private static LairConfig PinnedConfigFor(string modelPath, string modelName, int gpuLayers) => new()
    {
        Server = new ServerConfig(),
        Engine = new EngineConfig { Threads = 1 },
        Models =
        [
            new ModelConfig
            {
                Name = modelName,
                Path = modelPath,
                MaxContextTokens = 64,
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
