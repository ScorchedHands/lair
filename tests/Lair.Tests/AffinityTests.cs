using System.Runtime.InteropServices;
using Lair.Http;
using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Host.Config;

namespace Lair.Tests;

/// <summary>
/// Threading-contract tests: pinned worker threads land inside the configured core set, a model's thread quota caps
/// its backend (with generation staying byte-identical), and auto-sizing follows the physical-core rule. Sampling is
/// the backend's own per-block instrumentation (<c>CpuBackend.EnableExecutionSampling</c> /
/// <c>CpuBackend.ExecutionSamples</c>): each worker block records the logical processor it executed on;
/// dispatcher-executed blocks are deliberately not sampled because that thread belongs to the caller, not to the
/// pool. Pinning itself is Windows-only — these landing and quota assertions gate on
/// <see cref="RuntimeInformation.IsOSPlatform"/> so a non-Windows host skips them rather than faking affinity.
/// </summary>
[Trait("Category", "Affinity")]
public sealed class AffinityTests
{
    // Two low ids inside every realistic single-group machine; the mask pins exactly these logical processors.
    private static readonly HashSet<int> PinSet = [0, 1];

    private static bool HasAffinityApi =>
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && RuntimeInformation.ProcessArchitecture == Architecture.X64;

    [Fact]
    public void PinnedWorkers_LandInsideConfiguredCoreSet()
    {
        if (!HasAffinityApi)
        {
            return; // pinning is Windows/x64-only in this build; nothing to assert elsewhere
        }

        using CpuBackend backend = new(threads: 3, coreSet: PinSet);

        (byte[] weights, float[] x, float[] output) = Workload();
        backend.EnableExecutionSampling();

        // Repeated 16384-row calls (256 blocks each): a job this size outlives the workers' wake-up latency, so both
        // pinned workers stream claims and eight calls clear the 100-sample budget with room to spare. Small jobs
        // would let the dispatcher drain every block before the pool wakes — legal for compute, useless for sampling.
        for (int call = 0; call < 8; call++)
        {
            backend.Gemv(x, new WeightRef(weights, MatrixShape(16384, 576), QuantFormat.F32), output);
        }

        IReadOnlyList<int> samples = backend.ExecutionSamples;
        Assert.True(samples.Count >= 100, $"expected at least 100 sampled worker executions but got {samples.Count}.");

        foreach (int processor in samples)
        {
            Assert.Contains(processor, PinSet); // every sampled execution ran on a pinned logical core
        }
    }

    [Fact]
    public void UnpinnedModel_RunsPoolWithoutMaskAndStaysUnsampledFree()
    {
        if (!HasAffinityApi)
        {
            return;
        }

        using CpuBackend backend = new(threads: 3); // no core set: workers must run unpinned (mask never applied)

        (byte[] weights, float[] x, float[] output) = Workload();
        backend.EnableExecutionSampling();

        for (int call = 0; call < 8; call++)
        {
            backend.Gemv(x, new WeightRef(weights, MatrixShape(16384, 576), QuantFormat.F32), output);
        }

        // The pool ran and sampled freely: samples may fall on any logical processor because no mask was applied.
        Assert.True(backend.ExecutionSamples.Count >= 100, $"expected at least 100 sampled worker executions but got {backend.ExecutionSamples.Count}.");
    }

    [Fact]
    public async Task ThreadQuota_CapsBackendWorkers_AndGenerationStaysByteIdentical()
    {
        string modelPath = ModelFixture.Resolve() ?? throw new InvalidOperationException(
            $"GGUF fixture '{ModelFixture.SmolLmRelativePath}' was not found walking up from the test assembly directory.");

        LairConfig ConfigWith(int? quota) => new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig { Threads = 8 },
            Models =
            [
                new ModelConfig
                {
                    Name = "smollm-test",
                    Path = modelPath,
                    MaxContextTokens = 512,
                    KvBlocks = 32,
                    Placement = new PlacementConfig { ThreadQuota = quota },
                },
            ],
        };

        using GenerationService capped = GenerationService.Open(ConfigWith(2));
        Assert.Equal(2, capped.EffectiveWorkerCount); // min(global threads=8, thread_quota=2)

        using GenerationService uncapped = GenerationService.Open(ConfigWith(null));
        Assert.Equal(8, uncapped.EffectiveWorkerCount); // no quota: the engine request stands as configured

        // Greedy decoding (temperature 0) has no random draw and kernels are bit-identical across thread counts, so a
        // quota-capped completion must match the uncapped one byte for byte.
        GenerationRequest Request() => new()
        {
            Model = "smollm-test",
            Prompt = "The capital of France is",
            MaxTokens = 8,
            Temperature = 0f,
        };

        GenerationResult cappedRun = await capped.GenerateAsync(Request(), CancellationToken.None);
        GenerationResult uncappedRun = await uncapped.GenerateAsync(Request(), CancellationToken.None);

        Assert.Equal(uncappedRun.Text, cappedRun.Text);
        Assert.Equal(uncappedRun.CompletionTokens, cappedRun.CompletionTokens);
    }

    [Fact]
    public void AutoSizing_UsesPhysicalCoreRuleWithinLogicalCount()
    {
        int detected = CpuBackend.DetectedPhysicalCoreCount;
        bool windowsTopology = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && IntPtr.Size == 8;
        if (windowsTopology)
        {
            Assert.True(detected >= 1, "physical-core detection must return at least one core on Windows x64.");
        }

        Assert.True(
            detected <= Environment.ProcessorCount,
            $"detected physical cores ({detected}) cannot exceed the logical processor count ({Environment.ProcessorCount}).");

        using CpuBackend auto = new(); // threads: 0 — the documented rule must be what the backend applied
        int expected = detected > 0 ? Math.Min(detected, Environment.ProcessorCount) : Environment.ProcessorCount;
        Assert.Equal(expected, auto.EffectiveWorkerCount);

        if (!windowsTopology && detected == 0)
        {
            // Documented fallback: detection unavailable means the logical processor count sizes the pool, never a single worker.
            Assert.Equal(Environment.ProcessorCount, auto.EffectiveWorkerCount);
        }
    }

    [Fact]
    public void ExplicitThreadRequest_IsGlobalWorkerCount_AndQuotaCapsItWithoutExceedingTheRequest()
    {
        // An explicit nonzero thread request is the backend's global worker count, uncapped when no quota is supplied:
        // EffectiveWorkerCount reports exactly the requested posture, which is what the bench's threads= field reads off.
        using CpuBackend explicitRequest = new(threads: 4);
        Assert.Equal(4, explicitRequest.EffectiveWorkerCount);
        Assert.Equal(4, explicitRequest.UncappedThreadChoice);

        // A quota caps participation below the request; the uncapped ceiling still records the engine's own choice.
        using CpuBackend capped = new(threads: 4, threadQuota: 2);
        Assert.Equal(2, capped.EffectiveWorkerCount);
        Assert.Equal(4, capped.UncappedThreadChoice);

        // A quota above the engine request cannot raise participation past it — placement caps, never expands, the worker count.
        using CpuBackend overQuota = new(threads: 4, threadQuota: 8);
        Assert.Equal(4, overQuota.EffectiveWorkerCount); // min(global 4, quota 8)
        Assert.Equal(4, overQuota.UncappedThreadChoice);

        // A live quota edit re-derives participation against the uncapped ceiling: lowering idles workers, raising back to or
        // above the engine choice restores it but never exceeds that choice.
        using CpuBackend live = new(threads: 4, threadQuota: 2);
        Assert.Equal(1, live.ApplyThreadQuota(1));
        Assert.Equal(4, live.ApplyThreadQuota(8)); // capped at the engine's own 4, not the quota's 8
        Assert.Equal(4, live.EffectiveWorkerCount);
    }

    private static (byte[] Weights, float[] X, float[] Output) Workload()
    {
        const int Rows = 16384;
        const int Cols = 576; // SmolLM hidden size: the real decode shape this contract is about

        byte[] weights = new byte[Rows * Cols * sizeof(float)];
        Random random = new(20261102);
        foreach (ref float value in MemoryMarshal.Cast<byte, float>(weights.AsSpan()))
        {
            value = (float)(random.NextDouble() - 0.5) * 0.01f; // small weights keep every output finite
        }

        float[] x = new float[Cols];
        foreach (ref float value in x.AsSpan())
        {
            value = (float)(random.NextDouble() - 0.5);
        }

        return (weights, x, new float[Rows]);
    }

    private static TensorShape MatrixShape(int rows, int cols) => new([rows, cols]);
}





