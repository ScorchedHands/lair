using System.Text.RegularExpressions;
using Lair.Http;
using Lair.Host;
using Lair.Host.Config;

namespace Lair.Tests;

/// <summary>
/// The hot-reload application path end to end: a config file watcher with debounce drives validated edits through
/// the change-application bus while an engine is mid-generation on the real SmolLM fixture. Live knobs (prefill chunk
/// width, worker participation quota) visibly alter engine behavior without restart and leave output bytes identical;
/// the KV soft cap shrinks by draining cached prefix blocks around a live sequence and grows back within the physical
/// reservation to admit previously unaffordable work; max context changes gate later admissions only. Restart-only
/// edits are refused naming field and mutation class while the engine keeps serving old values, and broken files are
/// rejected loudly with the runtime untouched. Runs inside the serialized API collection so its real-model bursts do
/// not starve timing-guarded suites elsewhere in the assembly.
/// </summary>
[Collection("lair-api-serialized")]
[Trait("Category", "HotReload")]
public sealed class HotReloadTests
{
    private const string ModelName = "smollm-hotreload";

    // The concurrency/metrics suites' proven multi-token fixture sentences: shared prefixes here adopt cached blocks,
    // distinct tails publish new ones.
    private static readonly string Pangram =
        "The quick brown fox jumps over the lazy dog while the patient cat watches the rain.";

    [Fact]
    public async Task MaxBatchTokensEditMidPrefill_ResizesSubsequentChunks_OutputUnchanged()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        string prompt = SharedPrefix(5) + " In a distant library, maps of imaginary cities lined every wall.";

        // Baseline: an engine opened from the same configuration with no watcher runs the request at a constant chunk
        // width. The live run must reproduce its bytes exactly despite splitting prefill differently mid-flight.
        string baselineText;
        int baselineCompletion;
        {
            using GenerationService plain = OpenPlain();
            GenerationResult baseline = await plain.GenerateAsync(Completion(prompt, 16), CancellationToken.None);
            Assert.False(string.IsNullOrEmpty(baseline.Text));
            baselineText = baseline.Text;
            baselineCompletion = baseline.CompletionTokens;
        }

        using LiveConfigHarness harness = new(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512));

        Task<GenerationResult> running = harness.Engine.GenerateAsync(Completion(prompt, 16), CancellationToken.None);

        // Catch the prefill while chunks are still ahead of it.
        await WaitForAsync(
            () => harness.Engine.Stats.PrefillChunksIssued >= 2 && harness.Engine.Stats.ActiveSequenceCount >= 1,
            "the long prompt's chunked prefill never showed two issued chunks while active");

        harness.EditYaml(BuildYaml(kvBlocks: 24, maxBatchTokens: 4, maxContextTokens: 512));

        ReloadSummary summary = await harness.WaitForSummaryAsync(
            s => s.ValidationOk && s.Results.Any(r => r.Field == "engine.max_batch_tokens" && r.Outcome == "applied"),
            "the max_batch_tokens edit never reached the engine as an applied live change");

        long chunksAtApply = harness.Engine.Stats.PrefillChunksIssued;
        long tokensAtApply = harness.Engine.Stats.PrefillTokensForwarded;

        GenerationResult result = await running;

        Assert.Contains(
            summary.Results,
            r => r.Field == "engine.max_batch_tokens" && r.MutationClass == MutationClassNames.Live);

        // The reload landed mid-prefill: chunks kept coming after it was applied.
        long chunksAfterApply = harness.Engine.Stats.PrefillChunksIssued - chunksAtApply;
        Assert.True(
            chunksAfterApply >= 2,
            $"prefill finished before the edit applied (chunks at apply {chunksAtApply}, total {harness.Engine.Stats.PrefillChunksIssued}) — widen the fixture prompt");

        double avgBefore = tokensAtApply / (double)chunksAtApply;
        long tokensAfterApply = harness.Engine.Stats.PrefillTokensForwarded - tokensAtApply;
        double avgAfter = tokensAfterApply / (double)chunksAfterApply;

        Assert.True(avgBefore > 6.0 && avgBefore <= 8.0, $"pre-edit chunks averaged {avgBefore}, expected the configured width of 8");
        // One chunk straddling the apply boundary legitimately finishes at its old width (documented behavior: a reload
        // resizes from the next issue). With the fixture's prompt that single outlier can lift the post-edit average by
        // at most ~0.3 above 4; anything near the pre-edit widths means the new value never reached prefill.
        Assert.True(avgAfter <= 5.0, $"post-edit chunks averaged {avgAfter} tokens wide — the new max_batch_tokens=4 never reached prefill");

        // Correctness invariant: chunk boundaries move forwards, not results.
        Assert.Equal(baselineText, result.Text);
        Assert.Equal(baselineCompletion, result.CompletionTokens);
        Assert.True(result.FinishReason is "eos" or "length");
    }

    [Fact]
    public async Task ThreadQuotaEditMidServe_RetargetsWorkerParticipation_OutputUnchanged()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        string prompt = SharedPrefix(3);

        using LiveConfigHarness harness = new(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512, threadQuota: 3));

        // The engine opened with the quota already applied (min(engine threads 4, quota 3) = 3); a baseline run under
        // it anchors the output bytes.
        GenerationResult beforeRun = await harness.Engine.GenerateAsync(Completion(SharedPrefix(3), 10), CancellationToken.None);
        Assert.Equal(3, harness.Engine.EffectiveWorkerCount);


        Task<GenerationResult> running = harness.Engine.GenerateAsync(Completion(prompt + " A lighthouse keeper catalogued every passing ship.", 24), CancellationToken.None);
        await WaitForAsync(() => harness.Engine.Stats.ActiveSequenceCount >= 1, "the in-flight generation never became active");

        // Lower the quota mid-generation: participation drops from the next kernel call; workers idle.
        harness.EditYaml(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512, threadQuota: 1));
        await harness.WaitForSummaryAsync(
            s => s.Results.Any(r => r.Field == "models[0].placement.thread_quota" && r.Outcome == "applied" && r.NewValue == "1"),
            "the thread_quota edit never reached the engine as an applied live change");

        Assert.Equal(1, harness.Engine.EffectiveWorkerCount);

        GenerationResult during = await running; // the in-flight sequence finishes on lowered participation, output untouched

        // Raise it again past the engine's own thread choice: participation returns, clamped at that ceiling.
        harness.EditYaml(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512, threadQuota: 5));
        await harness.WaitForSummaryAsync(
            s => s.Results.Any(r => r.Field == "models[0].placement.thread_quota" && r.Outcome == "applied" && r.NewValue == "5"),
            "the raised thread_quota edit never reached the engine as an applied live change");

        Assert.Equal(4, harness.Engine.EffectiveWorkerCount); // min(engine threads 4, quota 5): placement cannot exceed the engine's choice

        GenerationResult after = await harness.Engine.GenerateAsync(Completion(prompt + " A lighthouse keeper catalogued every passing ship.", 24), CancellationToken.None);
        Assert.Equal(during.Text, after.Text);
        Assert.False(string.IsNullOrEmpty(beforeRun.Text)); // rows stay independent at any worker count: same bytes as the quota-6 baseline run family

        Assert.Contains(
            harness.Engine.LastReload!.Results,
            r => r.Field == "models[0].placement.thread_quota" && r.MutationClass == MutationClassNames.Live);
    }

    [Fact]
    public async Task KvSoftCapShrink_DrainsCachedBlocksAroundLiveSequence_GrowBackAdmitsRejectedWork()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using LiveConfigHarness harness = new(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512));

        Assert.Equal(24, harness.Engine.ReservedKvBlocks); // physical reservation fixed at open — a hot edit never re-reserves it
        Assert.Equal(24, harness.Engine.KvBudgetBlocks);   // budget starts equal to the reservation

        // Warm the prefix cache well past the cap the test will shrink to: each distinct tail publishes new full blocks.
        string warmPrefix = SharedPrefix(5);
        int cachedBefore = 0;
        for (int round = 0; round < 8 && harness.Engine.CachedPrefixBlocks < 10; round++)
        {
            GenerationResult warm = await harness.Engine.GenerateAsync(
                Completion($"{warmPrefix} A distinct archival note number {round} records the season quietly.", 4),
                CancellationToken.None);
            Assert.False(string.IsNullOrEmpty(warm.Text));
            cachedBefore = harness.Engine.CachedPrefixBlocks;
        }

        Assert.True(cachedBefore >= 10, $"the warm-up only established {cachedBefore} cached prefix blocks — extend it before shrinking below the cap");

        // A live sequence adopting those cached blocks runs while the shrink lands.
        Task<GenerationResult> active = harness.Engine.GenerateAsync(
            Completion(warmPrefix + " The librarian kept a red ledger of borrowed dreams.", 24),
            CancellationToken.None);
        await WaitForAsync(() => harness.Engine.Stats.ActiveSequenceCount >= 1, "the adopting sequence never became active");

        harness.EditYaml(BuildYaml(kvBlocks: 8, maxBatchTokens: 8, maxContextTokens: 512));

        ReloadSummary shrink = await harness.WaitForSummaryAsync(
            s => s.ValidationOk && s.Results.Any(r => r.Field == "models[0].kv_blocks" && r.Outcome == "applied" && r.NewValue == "8"),
            "the kv_blocks shrink never reached the engine as an applied live change");

        ReloadFieldResult shrunk = shrink.Results.Single(r => r.Field == "models[0].kv_blocks");
        Assert.Equal(MutationClassNames.Live, shrunk.MutationClass);
        Assert.Contains("drained", shrunk.Detail); // eviction path reported right in the per-field outcome

        GenerationResult activeRun = await active; // completes normally: its blocks were never revoked by the drain
        Assert.True(activeRun.FinishReason is "eos" or "length", $"the live sequence ended as '{activeRun.FinishReason}' during the shrink");

        Assert.Equal(8, harness.Engine.KvBudgetBlocks);
        Assert.Equal(24, harness.Engine.ReservedKvBlocks); // reservation untouched — only usage budget moved
        Assert.True(harness.Engine.CachedPrefixBlocks < cachedBefore, "the shrink drained no cached prefix blocks");

        // At the shrunk cap a fresh request cannot be admitted. The soft cap backs only 8 * 16 = 128 token slots, so the
        // advertised window admission validates against tightens to exactly that — a prompt past it is gated out while
        // every already-running sequence keeps its blocks untouched.
        string probe = SharedPrefix(9) + " A cartographer unrolled charts of coastlines that never existed.";
        await Assert.ThrowsAsync<ContextLengthExceededException>(() => harness.Engine.GenerateAsync(Completion(probe, 8), CancellationToken.None));

        // Grow the cap back within the reservation: the window and budget widen together, and the gated prompt fits again.
        harness.EditYaml(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512));
        await harness.WaitForSummaryAsync(
            s => s.Results.Any(r => r.Field == "models[0].kv_blocks" && r.Outcome == "applied" && r.NewValue == "24"),
            "the kv_blocks grow-back never reached the engine as an applied live change");

        Assert.Equal(24, harness.Engine.KvBudgetBlocks); // raise reaches exactly up to the reserved physical capacity

        GenerationResult admitted = await harness.Engine.GenerateAsync(Completion(probe, 8), CancellationToken.None);
        Assert.True(admitted.FinishReason is "eos" or "length", $"the grown-back admission ended as '{admitted.FinishReason}'");
        Assert.False(string.IsNullOrEmpty(admitted.Text));
    }

    [Fact]
    public async Task MaxContextRaiseMidServe_AdmitsPreviouslyTooLongRequest_InFlightUnaffected()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        string warmPrefix = SharedPrefix(5);

        // Baseline for the in-flight sequence: the same request run start-to-finish on an untouched engine. The live
        // run spans a mid-serve context raise and must reproduce its bytes regardless of what admission rules moved.
        string baselineInflight;
        {
            using GenerationService plain = OpenPlain();
            // Short prefix: it fits the 128-token window with a full 20-token budget on both engines, and its prompt is
            // itself a cached-chain prefix of the longer probe below.
            GenerationResult solo = await plain.GenerateAsync(Completion(SharedPrefix(3), 20), CancellationToken.None);
            Assert.False(string.IsNullOrEmpty(solo.Text));
            baselineInflight = solo.Text;
        }

        using LiveConfigHarness harness = new(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 128));
        Assert.Equal(128, harness.Engine.Model.ContextLength); // admission window in force at open (pool backs more than this)

        GenerationResult warm = await harness.Engine.GenerateAsync(Completion(warmPrefix, 4), CancellationToken.None);
        Assert.False(string.IsNullOrEmpty(warm.Text));

        // Nine prefix sentences plus a tail sit well past the 128-token admission window while staying inside what the
        // pool can back — exactly the band a new-requests-only edit moves.
        string tail = " A cartographer unrolled charts of coastlines that never existed anywhere on any map.";
        string tooLong = SharedPrefix(9) + tail;

        ContextLengthExceededException refused = await Assert.ThrowsAsync<ContextLengthExceededException>(
            () => harness.Engine.GenerateAsync(Completion(tooLong, 8), CancellationToken.None));

        // The rejection names its token count — the raise is sized just past it.
        Match match = Regex.Match(refused.Message, @"prompt is (\d+) tokens");
        Assert.True(match.Success, $"refusal message did not carry a prompt token count: {refused.Message}");
        int probeTokens = int.Parse(match.Groups[1].ToString());

        // An in-flight generation spans the reload window; its bytes must survive the admission change untouched.
        Task<GenerationResult> inflight = harness.Engine.GenerateAsync(Completion(SharedPrefix(3), 20), CancellationToken.None);
        await WaitForAsync(() => harness.Engine.Stats.ActiveSequenceCount >= 1, "the in-flight sequence never became active");

        harness.EditYaml(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: probeTokens + 32)); // within the reservation's backing
        await harness.WaitForSummaryAsync(
            s => s.Results.Any(r => r.Field == "models[0].max_context_tokens" && r.Outcome == "applied" && r.NewValue == (probeTokens + 32).ToString()),
            "the max_context_tokens raise never reached the engine as an applied new-requests change");

        GenerationResult inflightRun = await inflight; // unaffected: it runs on its already-admitted limits

        ReloadSummary summary = harness.Engine.LastReload!;
        ReloadFieldResult raised = summary.Results.Single(r => r.Field == "models[0].max_context_tokens");
        Assert.Equal(MutationClassNames.NewRequests, raised.MutationClass);

        GenerationResult admitted = await harness.Engine.GenerateAsync(Completion(tooLong, 8), CancellationToken.None);
        Assert.True(admitted.PromptTokens >= probeTokens, $"the previously-too-long prompt measured {admitted.PromptTokens} tokens");
        Assert.False(string.IsNullOrEmpty(admitted.Text));
        Assert.Equal(baselineInflight, inflightRun.Text); // in-flight bytes identical to the untouched-engine run across the mid-serve raise

        // Shrink the window again: requests that fit only under the raise are gated out once more, while bytes already
        // generated stay exactly as produced.
        harness.EditYaml(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: probeTokens - 8)); // still above the loader's floor of 128 and below what the prompt needs
        await harness.WaitForSummaryAsync(
            s => s.Results.Any(r => r.Field == "models[0].max_context_tokens" && r.Outcome == "applied" && r.NewValue == (probeTokens - 8).ToString()),
            "the max_context_tokens shrink never reached the engine as an applied new-requests change");

        Assert.True(harness.Engine.Model.ContextLength <= probeTokens - 8);
        await Assert.ThrowsAsync<ContextLengthExceededException>(
            () => harness.Engine.GenerateAsync(Completion(tooLong, 8), CancellationToken.None));
    }

    [Fact]
    public async Task RestartOnlyEdits_AreRefusedNamingFieldAndClass_EngineKeepsServingOldValues()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using LiveConfigHarness harness = new(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512));

        GenerationResult beforeRun = await harness.Engine.GenerateAsync(Completion(SharedPrefix(3), 6), CancellationToken.None);

        // Atomic-save pattern (temp file renamed onto the config name) carrying two restart-only edits at once. The
        // candidate fails validation outright (block size fixed; path absent), so it is dropped whole — yet each
        // changed key still gets a per-field refusal naming its mutation class in both summary and log lines.
        string edited = BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512, blockSize: 32, modelPathOverride: Path.Combine(harness.Dir, "relocated-model.gguf"));
        harness.WriteAtomic(edited);

        ReloadSummary summary = await harness.WaitForSummaryAsync(
            s => !s.ValidationOk && s.Results.Any(r => r.Field == "models[0].path" && r.Outcome == "refused"),
            "the restart-only edit never produced a refused reload summary");

        Assert.False(summary.ValidationOk); // dropped whole: nothing was applied from an invalid candidate

        ReloadFieldResult path = summary.Results.Single(r => r.Field == "models[0].path");
        Assert.Equal(MutationClassNames.RestartOnly, path.MutationClass);
        Assert.Contains("dropped whole", path.Detail);

        ReloadFieldResult blockSize = summary.Results.Single(r => r.Field == "engine.block_size");
        Assert.Equal(MutationClassNames.RestartOnly, blockSize.MutationClass);

        // The structured log lines name field and class exactly as the summary does.
        IReadOnlyList<string> logs = harness.LogLinesSnapshot();
        Assert.Contains(logs, l => l.Contains("config_reload") && l.Contains("field=models[0].path") && l.Contains("class=restart-only") && l.Contains("outcome=refused"));
        Assert.Contains(logs, l => l.Contains("field=engine.block_size") && l.Contains("class=restart-only"));

        // Old values retained and the engine keeps serving them.
        GenerationResult afterRun = await harness.Engine.GenerateAsync(Completion(SharedPrefix(3), 6), CancellationToken.None);
        Assert.Equal(beforeRun.Text, afterRun.Text);
        Assert.Equal(16, harness.Engine.EffectiveConfiguration.Engine.BlockSize);

        // A fully valid candidate that swaps the model path (the file exists — validation cannot catch this one) and
        // raises a live knob at the same time: the live half applies, the path is refused with its actionable message.
        string otherExistingFile = ModelFixture.ResolveRepoRelative("lair.example.yaml")!;

        harness.EditYaml(BuildYaml(kvBlocks: 24, maxBatchTokens: 512, maxContextTokens: 512, modelPathOverride: otherExistingFile));
        ReloadSummary mixed = await harness.WaitForSummaryAsync(
            s => s.ValidationOk && s.Results.Any(r => r.Field == "engine.max_batch_tokens" && r.Outcome == "applied"),
            "the live edit beside a restart-only path swap never applied");

        ReloadFieldResult smuggledPath = mixed.Results.Single(r => r.Field == "models[0].path");
        Assert.Equal(MutationClassNames.RestartOnly, smuggledPath.MutationClass);
        Assert.Contains("requires restart (restart-only)", smuggledPath.Detail); // the actionable refusal naming field and class verbatim

        Assert.Equal(512, harness.Engine.EffectiveConfiguration.Engine.MaxBatchTokens);  // live change in force from this candidate
        Assert.NotEqual(otherExistingFile, harness.Engine.EffectiveConfiguration.Models[0].Path); // refused field kept its old value

        GenerationResult served = await harness.Engine.GenerateAsync(Completion(SharedPrefix(3), 6), CancellationToken.None);
        Assert.Equal(beforeRun.Text, served.Text); // still serving the original model file's bytes after refusal and application alike
    }

    [Fact]
    public async Task BrokenYamlMidServe_IsRejectedLoudly_RuntimeStaysUntouched()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        string original = BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512);
        using LiveConfigHarness harness = new(original);

        GenerationResult beforeRun = await harness.Engine.GenerateAsync(Completion("Describe the sea in one sentence.", 6), CancellationToken.None);

        // A live generation is running when the file breaks — rejection must not disturb it.
        Task<GenerationResult> inflight = harness.Engine.GenerateAsync(Completion(SharedPrefix(3) + " The harbor master logged each arriving boat by color.", 14), CancellationToken.None);
        await WaitForAsync(() => harness.Engine.Stats.ActiveSequenceCount >= 1, "the in-flight sequence never became active");

        // A mid-serve edit the loader must refuse: a non-integer knob value. The file still parses, so every changed
        // key is named with its mutation class even though nothing can be applied from it.
        harness.EditYaml(original.Replace("max_batch_tokens: 8", "max_batch_tokens: zero"));

        ReloadSummary summary = await harness.WaitForSummaryAsync(
            s => !s.ValidationOk && !string.IsNullOrEmpty(s.ValidationError),
            "the broken file never produced a rejected reload summary");

        Assert.False(summary.ValidationOk); // loud, structured rejection naming the read/validation failure
        Assert.NotEmpty(summary.Results);  // every changed key the raw parse could still name is refused with its class — nothing applied

        ReloadFieldResult liveAttempt = summary.Results.Single(r => r.Field == "engine.max_batch_tokens");
        Assert.Equal(MutationClassNames.Live, liveAttempt.MutationClass);
        Assert.Equal("refused", liveAttempt.Outcome); // validation failure drops the candidate whole: even a live-class edit stays unapplied

        GenerationResult inflightRun = await inflight; // runs on, untouched by the rejected candidate
        Assert.True(inflightRun.FinishReason is "eos" or "length");

        // Repairing the file restores normal reloads and serving behavior identical to before.
        harness.EditYaml(original);
        await WaitForAsync(() => harness.Engine.LastReload?.ValidationOk == true, "the repaired configuration never validated");

        GenerationResult afterRun = await harness.Engine.GenerateAsync(Completion("Describe the sea in one sentence.", 6), CancellationToken.None);
        Assert.Equal(beforeRun.Text, afterRun.Text);
    }

    // --- fixture plumbing -----------------------------------------------------------------

    private static string SharedPrefix(int sentences) => string.Join(" ", Enumerable.Repeat(Pangram, sentences));

    private static GenerationRequest Completion(string prompt, int maxTokens) => new()
    {
        Model = ModelName,
        IsChat = false,
        Prompt = prompt,
        MaxTokens = maxTokens,
        Temperature = 0f, // greedy: byte-comparable across engines and reload states
    };

    private static string BuildYaml(
        int kvBlocks,
        int maxBatchTokens,
        int maxContextTokens,
        int? threadQuota = null,
        int blockSize = 16,
        string? modelPathOverride = null)
    {
        string modelPath = modelPathOverride ?? ModelFixture.Resolve()!;

        string yaml = $"""
            server:
              bind: 127.0.0.1
              port: 8199

            engine:
              threads: 4
              block_size: {blockSize}
              max_batch_tokens: {maxBatchTokens}

            models:
              - name: {ModelName}
                path: '{modelPath}'
                type: chat
                max_context_tokens: {maxContextTokens}
                kv_blocks: {kvBlocks}
                placement:
                  device: cpu
                  kv_residency: dram
                  prefill: cpu
            """;

        return threadQuota is int quota ? yaml + $"\n      thread_quota: {quota}\n" : yaml;
    }

    private static GenerationService OpenPlain() => GenerationService.Open(YamlConfigLoader.Load(WriteTempConfig(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512))));

    private static string WriteTempConfig(string yaml)
    {
        string dir = Path.Combine(Path.GetTempPath(), $"lair-hotreload-plain-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "lair.yaml");
        File.WriteAllText(path, yaml);

        // The plain harness has no watcher; its temp directory is cleaned when the config file itself closes.
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
            Dir = Path.Combine(Path.GetTempPath(), $"lair-hotreload-{Guid.NewGuid():N}");
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

        /// <summary>The editor pattern that atomic saves use — temp file renamed onto the config name.</summary>
        public void WriteAtomic(string yaml)
        {
            string staging = Path.Combine(Dir, "lair.yaml.tmp");
            File.WriteAllText(staging, yaml);
            File.Move(staging, ConfigPath, overwrite: true);
        }

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
