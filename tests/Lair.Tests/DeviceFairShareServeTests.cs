using System.Runtime.InteropServices;

using Lair.Host.Config;
using Lair.Http;
using Lair.Models;
using Lair.Models.Scheduling;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// The device window at the serve seam on a real accelerator: two engines whose routes resolve onto device 'cuda' advance one shared
/// co-scheduling window — the second open adopts the first engine's seam — each model registered with its declared compute share, and a
/// share-bound model defers its own excess candidates naming that share while the model holding the wider share keeps advancing beside it.
/// Every served token stays identical to the same request run alone: fair-share changes when device work launches, never what a forward
/// answers, and a deferral never preempts a sequence already launched.
/// </summary>
[Trait("Category", "Cuda")]
[Collection("model-loaded")] // shares the serialized engine lane with the suites that measure working-set envelopes here: an extra accelerator context must never overlap their measurements
public sealed class DeviceFairShareServeTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    [Fact]
    public void TwoDeviceResidentEngines_ShareOneWindow_AndEachModelRegistersItsDeclaredShare()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        string heavyPath = SyntheticModel("fair-share-heavy");
        string smallPath = SyntheticModel("fair-share-small");

        try
        {
            using GenerationService heavy = GenerationService.Open(Config(heavyPath, "fair-share-heavy", 1), NullLogger.Instance);
            using GenerationService small = GenerationService.Open(Config(smallPath, "fair-share-small", 100), NullLogger.Instance, ledger: null, compute: null, coScheduler: heavy.HostCoScheduler);

            Assert.Equal("cuda", heavy.HostCoScheduler!.DeviceId); // the resolved split route's accelerator segment is what the window bounds
            Assert.Equal("cuda", small.HostCoScheduler!.DeviceId);
            Assert.Same(heavy.HostCoScheduler, small.HostCoScheduler); // the second engine advanced under the first engine's window, not one of its own

            Assert.Equal(1, heavy.HostCoScheduler.ShareFor("fair-share-heavy"));
            Assert.Equal(100, heavy.HostCoScheduler.ShareFor("fair-share-small"));

            Assert.Equal(2, heavy.ResolvedRoute.Segments.Count);
            Assert.Equal("cuda", heavy.ResolvedRoute.Segments[0].DeviceId);
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
    public void ShareBoundModel_DefersItsOwnExcess_OnTheSharedWindow_WhileTheWiderShareModel_KeepsAdvancingTokenIdentical()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        string heavyPath = SyntheticModel("fair-share-contend-heavy");
        string smallPath = SyntheticModel("fair-share-contend-small");

        // The same three prompts answer on both models: solo first, then under contention on one shared window.
        string[] prompts = ["hello world", "world !", "! hello"];

        try
        {
            // Solo baselines: the exact text each request answers with when nothing else contends for the device.
            string[] heavySolo;
            string[] smallSolo;

            using (GenerationService heavyBaseline = GenerationService.Open(Config(heavyPath, "fair-share-contend-heavy", 1), NullLogger.Instance))
            {
                heavySolo = [.. prompts.Select(prompt => Generate(heavyBaseline, prompt))];
            }

            using (GenerationService smallBaseline = GenerationService.Open(Config(smallPath, "fair-share-contend-small", 100), NullLogger.Instance))
            {
                smallSolo = [.. prompts.Select(prompt => Generate(smallBaseline, prompt))];
            }

            // Contention: a 1%-share model holding three concurrent requests beside a full-share model holding three of its own, on one window.
            using GenerationService heavy = GenerationService.Open(Config(heavyPath, "fair-share-contend-heavy", 1), NullLogger.Instance);
            using GenerationService small = GenerationService.Open(Config(smallPath, "fair-share-contend-small", 100), NullLogger.Instance, ledger: null, compute: null, coScheduler: heavy.HostCoScheduler);

            Task<string>[] heavyRuns = [.. prompts.Select(prompt => Task.Run(() => Generate(heavy, prompt)))];
            Task<string>[] smallRuns = [.. prompts.Select(prompt => Task.Run(() => Generate(small, prompt)))];

            string[] heavyTexts = [.. heavyRuns.Select(task => task.GetAwaiter().GetResult())];
            string[] smallTexts = [.. smallRuns.Select(task => task.GetAwaiter().GetResult())];

            Assert.Equal(heavySolo, heavyTexts); // the share-bound model answered every deferred request exactly as it ran alone — deferral launched nothing, it preempted nothing
            Assert.Equal(smallSolo, smallTexts); // and the wider-share model's tokens moved nowhere either

            GenerationScheduler heavyScheduler = heavy.ServedScheduler;
            Assert.True(heavyScheduler.DeferredCompositionCandidates > 0, "a 1% share against three concurrent requests deferred nothing on the shared window");
            Assert.Contains(
                heavyScheduler.LastCompositionRefusals,
                refusal =>
                    refusal.Block == CompositionBlock.DeviceShare
                    && refusal.Reason.Contains("compute_share 1%", StringComparison.Ordinal)
                    && refusal.Reason.Contains("fair-share-contend-heavy", StringComparison.Ordinal)
                    && refusal.Reason.Contains("cuda", StringComparison.Ordinal));

            GenerationScheduler smallScheduler = small.ServedScheduler;
            Assert.Equal(0, smallScheduler.DeferredCompositionCandidates); // three candidates fit the full-share model's cap of 8: the other model's occupancy bounded nothing here
            Assert.True(smallScheduler.Stats.DecodeSteps > 0);

            // The wider-share model kept a bounded cadence while the device ran the share-bound model beside it — the ITL side of the window.
            long averageStepMicros = smallScheduler.Stats.DecodeWindowMicrosTotal / Math.Max(1, smallScheduler.Stats.DecodeSteps);
            Assert.True(
                averageStepMicros < 500_000,
                $"the full-share model averaged {averageStepMicros} microseconds per decode step while a 1% model contended on the same device");

            Assert.Equal(0, heavy.HostCoScheduler!.OutstandingStepSlots("fair-share-contend-heavy"));
            Assert.Equal(0, heavy.HostCoScheduler.OutstandingStepSlots("fair-share-contend-small")); // every launched slot returned as its forward completed
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

    private static LairConfig Config(string modelPath, string modelName, int computeShare) => new()
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
                Placement = new PlacementConfig { GpuLayers = 1 },
                ComputeShare = computeShare,
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
}
