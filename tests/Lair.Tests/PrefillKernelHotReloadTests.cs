using Lair.Host;
using Lair.Host.Config;
using Lair.Http;

namespace Lair.Tests;

/// <summary>
/// The hot-reload path for <c>engine.prefill_kernel</c>: an edit of the key classifies new-requests-only, a selection
/// with no kernel in this build is refused naming it and what the forward engine lacks while the running engine keeps
/// serving on the route it opened with (no silent fallback), sequences already mid-generation finish byte-for-byte as
/// they started, and a restart carrying the same selection serves the identical token stream against the baseline run.
/// Runs inside the serialized API collection so its real-model bursts do not starve timing-guarded suites elsewhere in
/// the assembly.
/// </summary>
[Collection("lair-api-serialized")]
[Trait("Category", "HotReload")]
public sealed class PrefillKernelHotReloadTests
{
    private const string ModelName = "smollm-prefill-kernel-reload";

    [Fact]
    public async Task PrefillKernelEdit_IsClassifiedNewRequestsOnly_AndRestartWithSameSelection_ServesIdenticalStream()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        string prompt = "The quick brown fox jumps over the lazy dog while the patient cat watches the rain. " +
                        "In a distant library, maps of imaginary cities lined every wall.";

        // Baseline: an engine opened from the default configuration — no key at all, so gemv by default — runs the
        // request greedy. Every later comparison anchors on its bytes.
        string baselineText;
        int baselineCompletion;
        {
            using GenerationService plain = OpenPlain(kernelSelection: null);
            GenerationResult baseline = await plain.GenerateAsync(Completion(prompt, 24), CancellationToken.None);
            Assert.False(string.IsNullOrEmpty(baseline.Text));
            baselineText = baseline.Text;
            baselineCompletion = baseline.CompletionTokens;
        }

        using LiveConfigHarness harness = new(BuildYaml("gemv"));

        // A generation is already mid-flight when the edit lands: a refused selection must not disturb it, and an
        // applied one only re-reads at the next window boundary.
        Task<GenerationResult> running = harness.Engine.GenerateAsync(Completion(prompt, 24), CancellationToken.None);
        await WaitForAsync(() => harness.Engine.Stats.ActiveSequenceCount >= 1, "the in-flight generation never became active");

        harness.EditYaml(BuildYaml("tiled"));

        ReloadSummary summary = await harness.WaitForSummaryAsync(
            s => s.ValidationOk && s.Results.Any(r => r.Field == "engine.prefill_kernel" && r.Outcome == "refused"),
            "the prefill_kernel edit never reached the engine as a refused change");

        ReloadFieldResult result = Assert.Single(summary.Results, r => r.Field == "engine.prefill_kernel");
        Assert.Equal(MutationClassNames.NewRequests, result.MutationClass); // classified new-requests-only like every admission-side choice key
        Assert.Equal("gemv", result.OldValue);
        Assert.Equal("tiled", result.NewValue);
        Assert.Contains("'tiled' is not implemented by the forward engine", result.Detail); // refused naming value and missing kernel
        Assert.Contains("no tiled attention/weight path exists in this build", result.Detail);

        GenerationResult inflight = await running;
        Assert.Equal(baselineText, inflight.Text); // the sequence already mid-generation finished untouched
        Assert.Equal(baselineCompletion, inflight.CompletionTokens);

        // The refused edit retained what the engine opened with instead of falling back or landing: later requests on
        // the same live engine keep serving the gemv route byte-for-byte.
        Assert.Equal("gemv", harness.Engine.EffectiveConfiguration.Engine.PrefillKernel);

        GenerationResult afterRefusal = await harness.Engine.GenerateAsync(Completion(prompt, 24), CancellationToken.None);
        Assert.Equal(baselineText, afterRefusal.Text);
        Assert.Equal(baselineCompletion, afterRefusal.CompletionTokens);

        // A restart carrying the same selection — an engine opened with the key stated explicitly rather than defaulted —
        // produces the identical stream against that baseline.
        using GenerationService restarted = OpenPlain("gemv");
        GenerationResult restartRun = await restarted.GenerateAsync(Completion(prompt, 24), CancellationToken.None);
        Assert.Equal(baselineText, restartRun.Text);
        Assert.Equal(baselineCompletion, restartRun.CompletionTokens);

        // The configuration in force after the refused edit is exactly what a new engine would open with: the same
        // selection, so "restart carrying the same value" and "the live engine that took the hot edit" describe one state.
        Assert.Equal("gemv", restarted.EffectiveConfiguration.Engine.PrefillKernel);
    }

    // ---------------------------------------------------------------- fixture plumbing

    private static GenerationRequest Completion(string prompt, int maxTokens) => new()
    {
        Model = ModelName,
        IsChat = false,
        Prompt = prompt,
        MaxTokens = maxTokens,
        Temperature = 0f, // greedy: byte-comparable across engines and reload states
    };

    private static string BuildYaml(string? prefillKernel)
    {
        string modelPath = ModelFixture.Resolve()!;

        // The baseline engine carries no key at all — the default selection — while every compared run states it.
        string kernelLine = prefillKernel is null ? string.Empty : $"\n  prefill_kernel: {prefillKernel}";

        return $"""
            server:
              bind: 127.0.0.1
              port: 8199

            engine:
              threads: 4
              block_size: 16
              max_batch_tokens: 8{kernelLine}

            models:
              - name: {ModelName}
                path: '{modelPath}'
                type: chat
                max_context_tokens: 512
                kv_blocks: 24
                placement:
                  device: cpu
                  kv_residency: dram
                  prefill: cpu
            """;
    }

    private static GenerationService OpenPlain(string? kernelSelection) => GenerationService.Open(
        YamlConfigLoader.Load(WriteTempConfig(BuildYaml(kernelSelection))));

    private static string WriteTempConfig(string yaml)
    {
        string dir = Path.Combine(Path.GetTempPath(), $"lair-prefill-kernel-plain-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "lair.yaml");
        File.WriteAllText(path, yaml);

        // The plain harness has no watcher; its temp directory is cleaned when the process exits.
        AppDomain.CurrentDomain.ProcessExit += (_, _) => TryDeleteTree(dir);
        return path;
    }

    /// <summary>One live engine with a watching hot-reload host over a temporary config file, capturing every structured log line.</summary>
    private sealed class LiveConfigHarness : IDisposable
    {
        public string Dir { get; }
        public string ConfigPath { get; }
        public GenerationService Engine { get; }

        private readonly object _logGate = new();
        private readonly List<string> _lines = [];
        private readonly ConfigWatcher _watcher;

        public LiveConfigHarness(string yaml)
        {
            Dir = Path.Combine(Path.GetTempPath(), $"lair-prefill-kernel-hotreload-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Dir);
            ConfigPath = Path.Combine(Dir, "lair.yaml");
            File.WriteAllText(ConfigPath, yaml);

            Engine = GenerationService.Open(YamlConfigLoader.Load(ConfigPath));
            _watcher = new ConfigWatcher(ConfigPath, Engine, line =>
            {
                lock (_logGate)
                {
                    _lines.Add(line);
                }
            });
            _watcher.Start(); // started after the initial write: opening never looks like an edit
        }

        public void EditYaml(string yaml) => File.WriteAllText(ConfigPath, yaml);

        public IReadOnlyList<string> LogLinesSnapshot()
        {
            lock (_logGate)
            {
                return [.. _lines];
            }
        }

        /// <summary>Waits for a completed reload attempt whose summary satisfies the predicate; failures dump captured log lines into the failure message.</summary>
        public async Task<ReloadSummary> WaitForSummaryAsync(Func<ReloadSummary, bool> done, string what)
        {
            long deadline = Environment.TickCount64 + 25_000; // debounce + settle + load each pass is sub-second; this covers debug-JIT contention generously
            while (true)
            {
                ReloadSummary? summary = Engine.LastReload;
                if (summary is not null && done(summary))
                {
                    return summary;
                }

                if (Environment.TickCount64 >= deadline)
                {
                    Assert.Fail($"{what}. Captured log lines:\n{string.Join("\n", LogLinesSnapshot())}");
                }

                await Task.Delay(50);
            }
        }

        public void Dispose()
        {
            _watcher.Dispose(); // stops observing before the engine it reports into goes away
            Engine.Dispose();
            TryDeleteTree(Dir);
        }
    }

    private static async Task WaitForAsync(Func<bool> condition, string what)
    {
        long deadline = Environment.TickCount64 + 25_000;
        while (!condition())
        {
            if (Environment.TickCount64 >= deadline)
            {
                Assert.Fail(what);
            }

            await Task.Delay(25);
        }
    }

    private static void TryDeleteTree(string dir)
    {
        try
        {
            Directory.Delete(dir, recursive: true);
        }
        catch (IOException)
        {
            // Temp scratch under the test user's path; a locked leftover is noise, not a failure signal.
        }
        catch (UnauthorizedAccessException)
        {
        }
    }
}
