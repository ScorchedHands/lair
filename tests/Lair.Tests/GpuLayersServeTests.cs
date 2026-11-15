using System.Runtime.InteropServices;

using Lair.Host.Config;
using Lair.Http;
using Lair.Models;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// The layer split at the serve seam: a model opened with a declared gpu_layers count resolves its placement to a hybrid route — device 'cuda' from layer 0, cpu carrying every later layer and the output head beside it — serves greedy tokens identical to the same file opened CPU-only, and keeps reporting decode-side facts through this service's own CPU backend. An absent or zero key leaves the opened posture untouched: one CPU segment over DRAM-resident weights with no accelerator seam consulted at all. A declared split past what the model carries refuses the open naming the model, its route segments, and where accelerator-backed spans run — before any serving engine exists to half-run it.
/// </summary>
[Trait("Category", "Parity")]
[Trait("Category", "Cuda")]
public sealed class GpuLayersServeTests : IDisposable
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

    [Fact]
    public void DeclaredSplit_ServesTokenIdenticalToCpuOnly_AndResolvesHybridRouteWithDecodeSideFactsOnCpu()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        string path = SyntheticModel("split-serve");

        LairConfig cpuOnly = Config(path, "split-serve-cpu", gpuLayers: 0);
        using GenerationService baseline = GenerationService.Open(cpuOnly, NullLogger.Instance);

        Assert.Single(baseline.ResolvedRoute.Segments); // today's posture stands untouched when no split is declared.
        string cpuText = Generate(baseline, "hello world");

        LairConfig hybrid = Config(path, "split-serve-hybrid", gpuLayers: 1);

        try
        {
            using GenerationService split = GenerationService.Open(hybrid, NullLogger.Instance);

            Assert.Equal(2, split.ResolvedRoute.Segments.Count); // the declared split resolved to two segments in execution order.
            Assert.Equal("cuda", split.ResolvedRoute.Segments[0].DeviceId); // device 'cuda' from layer 0...
            Assert.Equal(ModelRoute.CpuDeviceId, split.ResolvedRoute.Segments[1].DeviceId); // ...cpu carrying every later layer and the output head.

            string hybridText = Generate(split, "hello world");

            Assert.Equal(cpuText, hybridText); // the served text is token-identical across the crossing — a serving engine never samples differently behind a device it runs part of its model on.
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }
    }

    [Fact]
    public void DeclaredSplitPastTheModelLayers_RefusesOpenNamingSegmentsAndSeam_WithoutHalfOpeningAnEngine()
    {
        string path = SyntheticModel("split-serve-past"); // the refusal lands before any weight pass, so no CUDA device is consulted.

        LairConfig candidate = new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig { Threads = 1 },
            Models =
            [
                new ModelConfig
                {
                    Name = "split-serve-past",
                    Path = path,
                    MaxContextTokens = 512,
                    KvBlocks = 8,
                    Placement = new PlacementConfig { GpuLayers = 5 }, // the synthetic fixture ships two layers: a span starting past them runs nothing.
                },
            ],
        };

        ModelException error = Assert.Throws<ModelException>(() => GenerationService.Open(candidate, NullLogger.Instance));

        Assert.Contains("starts at boundary '5', past this file's last layer", error.Message); // the engine names what lands short exactly as its own split-route refusals always have.
    }

    [Fact]
    public void DeclaredSplitWithDedicatedPrefillLane_RefusesOpenNamingBothPlacementKeys_AndCarriesNoStageSideCrossing()
    {
        string path = SyntheticModel("split-serve-lane");

        LairConfig candidate = new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig { Threads = 2 },
            Models =
            [
                new ModelConfig
                {
                    Name = "split-serve-lane",
                    Path = path,
                    MaxContextTokens = 512,
                    KvBlocks = 8,
                    Placement = new PlacementConfig
                    {
                        GpuLayers = 1,
                        PrefillLane = PlacementPrefillLaneValues.Dedicated,
                    },
                },
            ],
        };

        ModelException error = Assert.Throws<ModelException>(() => GenerationService.Open(candidate, NullLogger.Instance));

        Assert.Contains("model 'split-serve-lane'", error.Message); // the refusal names what it refuses to open.
        Assert.Contains("'placement.gpu_layers' 1", error.Message);
        Assert.Contains($"'placement.prefill_lane' '{PlacementPrefillLaneValues.Dedicated}'", error.Message); // both posture keys stand beside each other in the actionable message.
        Assert.Contains("no crossing across a split route", error.Message); // this build carries one stage-side backend, so it refuses rather than dropping either side silently.
    }

    [Fact]
    public void DeclaredSplitWithoutUsableDevice_FailsFastNamingModelKeyAndSeam_NeverSilentCpuFallback()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        string path = SyntheticModel("split-serve-nodevice"); // the refusal lands before any weight pass; it needs a declared split and whatever this host's driver seam answers.

        LairConfig candidate = new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig { Threads = 1 },
            Models =
            [
                new ModelConfig
                {
                    Name = "split-serve-nodevice",
                    Path = path,
                    MaxContextTokens = 512,
                    KvBlocks = 8,
                    Placement = new PlacementConfig { GpuLayers = 1 },
                },
            ],
        };

        try
        {
            using GenerationService engine = GenerationService.Open(candidate, NullLogger.Instance);

            // A host with a usable seam opens the split normally: what this gate proves is the refusal shape below on hosts without one.
            Assert.Equal(2, engine.ResolvedRoute.Segments.Count);
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("'placement.gpu_layers' 1") && ex.Message.Contains(ModelRoute.CudaBackendSeam))
        {
            // The declared split names the model, the key as declared, and where accelerator-backed segments run — with no fallback to some other posture.
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }
    }

    // ---------------------------------------------------------------- helpers

    private static LairConfig Config(string modelPath, string modelName, int gpuLayers) => new()
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

    private static string Generate(GenerationService engine, string prompt) =>
        engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = prompt,
            MaxTokens = 8,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult().Text;

    private string SyntheticModel(string modelName)
    {
        string path = SyntheticLlamaGguf.Write(modelName);
        _tempFiles.Add(path);
        return path;
    }
}
