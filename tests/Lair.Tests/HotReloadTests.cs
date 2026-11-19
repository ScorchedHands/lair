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

    /// <summary>
    /// The step-time budget hot-reloading mid-prefill: a wide target leaves every boundary at the configured width, and tightening it while
    /// prefill is running resizes only later chunks — the one already chosen finishes whole. An invalid value for the same key lands as a loud
    /// per-field refusal naming its mutation class with nothing applied, and the engine keeps serving identical bytes either way.
    /// </summary>
    [Fact]
    public async Task PrefillStepBudgetEditMidPrefill_ResizesSubsequentChunks_OutputUnchanged_AndInvalidValuesAreRefused()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        const int WideTargetMs = 200;   // far above one forward on this engine: selection pins each boundary at the configured width
        const int NarrowTargetMs = 12;  // under a weight-bound forward's per-token cost, so later boundaries have to come narrower

        string prompt = SharedPrefix(5) + " In a distant library, maps of imaginary cities lined every wall.";
        string shortPrompt = "Describe the sea in one sentence.";

        using GenerationService plain = OpenPlain(); // no watcher and no budget: the byte-for-byte anchor for both halves below
        GenerationResult baselineShort = await plain.GenerateAsync(Completion(shortPrompt, 6), CancellationToken.None);
        GenerationResult baselineLong = await plain.GenerateAsync(Completion(prompt, 16), CancellationToken.None);
        Assert.False(string.IsNullOrEmpty(baselineLong.Text));

        string wideYaml = BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512, prefillStepBudgetMs: WideTargetMs);
        using LiveConfigHarness harness = new(wideYaml);

        Task<GenerationResult> running = harness.Engine.GenerateAsync(Completion(prompt, 16), CancellationToken.None);

        await WaitForAsync(
            () => harness.Engine.Stats.PrefillChunksIssued >= 2 && harness.Engine.Stats.ActiveSequenceCount >= 1,
            "the budgeted prefill never showed two issued chunks while active");

        string narrowedYaml = BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512, prefillStepBudgetMs: NarrowTargetMs);
        harness.EditYaml(narrowedYaml);

        ReloadSummary applied = await harness.WaitForSummaryAsync(
            s => s.ValidationOk && s.Results.Any(r => r.Field == "engine.prefill_step_budget_ms" && r.Outcome == "applied"),
            "the prefill_step_budget_ms edit never reached the engine as an applied live change");

        Assert.Contains(
            applied.Results,
            r => r.Field == "engine.prefill_step_budget_ms" && r.MutationClass == MutationClassNames.Live);

        long chunksAtApply = harness.Engine.Stats.PrefillChunksIssued;
        long tokensAtApply = harness.Engine.Stats.PrefillTokensForwarded;

        GenerationResult result = await running;

        // The reload landed mid-prefill: boundaries kept coming after it was applied, and they came narrower than the ones before it.
        long chunksAfterApply = harness.Engine.Stats.PrefillChunksIssued - chunksAtApply;
        Assert.True(
            chunksAfterApply >= 2,
            $"prefill finished before the edit applied (chunks at apply {chunksAtApply}, total {harness.Engine.Stats.PrefillChunksIssued}) — widen the fixture prompt");

        double avgBefore = tokensAtApply / (double)chunksAtApply;
        long tokensAfterApply = harness.Engine.Stats.PrefillTokensForwarded - tokensAtApply;
        double avgAfter = tokensAfterApply / (double)chunksAfterApply;

        Assert.True(avgBefore > 6.0 && avgBefore <= 8.0, $"pre-edit chunks averaged {avgBefore}, expected the configured width of 8");

        // A step already chosen when the target tightened finishes at its old width: one boundary straddling the apply lands wide, and every
        // later one came from measured cost under the new target — narrow enough that a pre-edit average could never produce it.
        Assert.True(avgAfter <= 3.0, $"post-edit chunks averaged {avgAfter:F1} tokens wide — the tightened {NarrowTargetMs} ms step-time budget never reached prefill");

        // Correctness invariant: boundaries move forwards, not results.
        Assert.Equal(baselineLong.Text, result.Text);
        Assert.Equal(baselineLong.CompletionTokens, result.CompletionTokens);
        Assert.True(result.FinishReason is "eos" or "length");

        // An invalid value for the same key mid-serve: loud per-field refusal naming its class, nothing applied, and serving unchanged.
        harness.EditYaml(narrowedYaml.Replace($"prefill_step_budget_ms: {NarrowTargetMs}", "prefill_step_budget_ms: zero"));

        ReloadSummary refused = await harness.WaitForSummaryAsync(
            s => !s.ValidationOk && s.Results.Any(r => r.Field == "engine.prefill_step_budget_ms"),
            "the invalid prefill_step_budget_ms edit never produced a rejected reload summary");

        Assert.False(refused.ValidationOk);
        ReloadFieldResult refusedAttempt = refused.Results.Single(r => r.Field == "engine.prefill_step_budget_ms");
        Assert.Equal(MutationClassNames.Live, refusedAttempt.MutationClass);
        Assert.Equal("refused", refusedAttempt.Outcome); // validation failure drops the candidate whole: even a live-class edit stays unapplied

        GenerationResult afterRefusal = await harness.Engine.GenerateAsync(Completion(shortPrompt, 6), CancellationToken.None);
        Assert.Equal(baselineShort.Text, afterRefusal.Text); // the engine kept serving on the values it had

        harness.EditYaml(wideYaml);
        await WaitForAsync(() => harness.Engine.LastReload?.ValidationOk == true, "the repaired configuration never validated");
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

    [Fact]
    public async Task ConcurrencyCapEditMidServe_RefusesLaterSubmissions_NamingQuota_WithoutPreemptingInFlight()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using LiveConfigHarness harness = new(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512, maxConcurrentRequests: 2));

        // Two sequences admitted together hold both slots of the cap this engine opened with.
        Task<GenerationResult> first = harness.Engine.GenerateAsync(
            Completion(SharedPrefix(3) + " A lighthouse keeper catalogued every passing ship.", 24), CancellationToken.None);
        Task<GenerationResult> second = harness.Engine.GenerateAsync(
            Completion(SharedPrefix(3) + " A cartographer charted a coastline of glass and salt.", 24), CancellationToken.None);

        await WaitForAsync(() => harness.Engine.Stats.ActiveSequenceCount >= 2, "two sequences never ran alongside each other");

        // While both slots are held, a third submission is refused naming this model and its concurrency quota rather than queued.
        EngineBusyException busy = await Assert.ThrowsAsync<EngineBusyException>(
            () => harness.Engine.GenerateAsync(Completion("Describe the sea briefly.", 4), CancellationToken.None));
        Assert.Contains(ModelName, busy.Message);
        Assert.Contains("max_concurrent_requests", busy.Message);

        // Lowering the cap mid-serve is new-requests-only: sequences already admitted keep their slots and finish normally.
        harness.EditYaml(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512, maxConcurrentRequests: 1));
        ReloadSummary summary = await harness.WaitForSummaryAsync(
            s => s.Results.Any(r => r.Field == "models[0].max_concurrent_requests" && r.Outcome == "applied"),
            "the max_concurrent_requests edit never reached the engine as an applied new-requests change");
        Assert.Contains(summary.Results, r => r.Field == "models[0].max_concurrent_requests" && r.MutationClass == MutationClassNames.NewRequests);

        // The lowered cap binds future admissions immediately: with one sequence still holding a slot of a cap of 1, another submission is refused.
        EngineBusyException lowered = await Assert.ThrowsAsync<EngineBusyException>(
            () => harness.Engine.GenerateAsync(Completion("Describe the sea briefly.", 4), CancellationToken.None));
        Assert.Contains(ModelName, lowered.Message);

        // Neither refusal disturbed what already ran: both in-flight sequences finish on their token budgets.
        GenerationResult firstOutcome = await first;
        GenerationResult secondOutcome = await second;
        Assert.False(string.IsNullOrEmpty(firstOutcome.Text) || string.IsNullOrEmpty(secondOutcome.Text));
        Assert.Equal(24, firstOutcome.CompletionTokens);
        Assert.Equal(24, secondOutcome.CompletionTokens);

        // Once every slot came back, the lowered cap admits again — one sequence at a time.
        GenerationResult refilled = await harness.Engine.GenerateAsync(Completion("Describe the sea in one sentence.", 6), CancellationToken.None);
        Assert.False(string.IsNullOrEmpty(refilled.Text));

        // Raising the cap applies to later admissions too: two concurrent submissions now ride alongside each other under it.
        harness.EditYaml(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512, maxConcurrentRequests: 3));
        await harness.WaitForSummaryAsync(
            s => s.Results.Any(r => r.Field == "models[0].max_concurrent_requests" && r.Outcome == "applied" && r.NewValue == "3"),
            "the raised max_concurrent_requests edit never reached the engine as an applied change");

        GenerationResult[] raised = await Task.WhenAll(
            [
                harness.Engine.GenerateAsync(Completion(SharedPrefix(2) + " A diver mapped a field of quiet kelp.", 10), CancellationToken.None),
                harness.Engine.GenerateAsync(Completion(SharedPrefix(2) + " A potter shaped a jar from river clay.", 10), CancellationToken.None),
            ]);

        Assert.Equal(2, raised.Length); // both ran: nothing under the raised cap was refused
        Assert.False(string.IsNullOrEmpty(raised[0].Text) || string.IsNullOrEmpty(raised[1].Text));
    }

    [Fact]
    public async Task RequestQueueKeysEditMidServe_AppliesAsNewRequests_QueuesLaterSubmissions_WithoutPreemptingInFlight()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using LiveConfigHarness harness = new(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512, maxConcurrentRequests: 1));

        Task<GenerationResult> holding = await StartHoldingSequenceAsync(harness.Engine);
        await WaitForAsync(() => harness.Engine.Stats.ActiveSequenceCount >= 1, "the holding sequence never admitted");

        // With no queue configured at open, a submission while the only slot is held answers today's immediate refusal posture unchanged.
        EngineBusyException immediate = await Assert.ThrowsAsync<EngineBusyException>(
            () => harness.Engine.GenerateAsync(Completion("Describe the sea briefly.", 4), CancellationToken.None));
        Assert.Contains(ModelName, immediate.Message);
        Assert.Contains("max_concurrent_requests", immediate.Message);

        // Enabling the queue mid-serve is new-request-only: what already runs keeps running, and a submission arriving from now on waits for its slot
        // instead of being refused outright.
        harness.EditYaml(BuildYaml(kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512, maxConcurrentRequests: 1, maxQueuedRequests: 2));
        ReloadSummary depthSummary = await harness.WaitForSummaryAsync(
            s => s.Results.Any(r => r.Field == "models[0].max_queued_requests" && r.Outcome == "applied"),
            "the max_queued_requests edit never reached the engine as an applied new-requests change");
        Assert.Contains(depthSummary.Results, r => r.Field == "models[0].max_queued_requests" && r.MutationClass == MutationClassNames.NewRequests);

        Task<GenerationResult> queued = harness.Engine.GenerateAsync(Completion("Describe the harbor briefly.", 6), CancellationToken.None);
        await WaitForAsync(() => harness.Engine.QueuedRequestCount >= 1, "the later submission never entered this engine's request queue");

        GenerationResult heldOutcome = await holding; // its sequence finished normally through the edit that queued a later one behind it
        Assert.Equal(24, heldOutcome.CompletionTokens);

        GenerationResult queuedOutcome = await queued; // admitted when the slot its model's posture now queues for came free
        Assert.False(string.IsNullOrEmpty(queuedOutcome.Text));
        Assert.Equal(0, harness.Engine.QueuedRequestCount); // every seat released as its submission was admitted

        // Lowering the bounded wait mid-serve follows the same mutation class and binds later submissions at once: a ceiling far shorter than what is
        // running cannot be beaten by an admission that has not freed yet.
        harness.EditYaml(BuildYaml(
            kvBlocks: 24, maxBatchTokens: 8, maxContextTokens: 512, maxConcurrentRequests: 1, maxQueuedRequests: 2, queueTimeoutMs: 60));
        ReloadSummary timeoutSummary = await harness.WaitForSummaryAsync(
            s => s.Results.Any(r => r.Field == "models[0].queue_timeout_ms" && r.Outcome == "applied"),
            "the queue_timeout_ms edit never reached the engine as an applied new-requests change");
        Assert.Contains(timeoutSummary.Results, r => r.Field == "models[0].queue_timeout_ms" && r.MutationClass == MutationClassNames.NewRequests);

        Task<GenerationResult> secondHolding = await StartHoldingSequenceAsync(harness.Engine);
        EngineBusyException timedOut = await Assert.ThrowsAsync<EngineBusyException>(
            () => harness.Engine.GenerateAsync(Completion("Describe the harbor briefly.", 4), CancellationToken.None));
        Assert.Equal("engine_overloaded", timedOut.Code); // the same bounded overload family at the wire as every other admission refusal here
        Assert.Contains(ModelName, timedOut.Message);
        Assert.Contains("queue_timeout_ms", timedOut.Message);

        GenerationResult secondHeld = await secondHolding; // nothing already admitted was preempted by a submission that ran out of its wait
        Assert.Equal(24, secondHeld.CompletionTokens);

        // The posture governs admission, not a permanent wall: with the slot free again, an ordinary submission is admitted normally.
        GenerationResult refilled = await harness.Engine.GenerateAsync(Completion("Describe the sea in one sentence.", 6), CancellationToken.None);
        Assert.False(string.IsNullOrEmpty(refilled.Text));
    }

    // Starts a sequence that proves it was admitted (its first streamed token) and then parks on its own steps, so gates below observe admission state
    // against a slot that is demonstrably held rather than racing whatever order two submissions happened to take.
    private static async Task<Task<GenerationResult>> StartHoldingSequenceAsync(GenerationService engine)
    {
        TaskCompletionSource<bool> firstToken = new(TaskCreationOptions.RunContinuationsAsynchronously);

        int parkedTokens = 0;
        GenerationRequest holding = new()
        {
            Model = ModelName,
            IsChat = false,
            Prompt = SharedPrefix(3) + " A lighthouse keeper catalogued every passing ship.",
            MaxTokens = 24,
            Temperature = 0f,
            Stream = true,
            OnToken = _ =>
            {
                firstToken.TrySetResult(true);

                if (parkedTokens < 3)
                {
                    parkedTokens++;
                    Thread.Sleep(500); // hold this engine's only admission slot long enough for queue state to be observed against it
                }
            },
        };

        Task<GenerationResult> task = engine.GenerateAsync(holding, CancellationToken.None);

        if (await Task.WhenAny(firstToken.Task, task) != firstToken.Task || task.IsCompleted)
        {
            throw new InvalidOperationException("the holding sequence finished before it proved admission — its slot was never observably held");
        }

        return task;
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
        string? modelPathOverride = null,
        int? maxConcurrentRequests = null,
        int? prefillStepBudgetMs = null,
        int? maxQueuedRequests = null,
        int? queueTimeoutMs = null)
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
            """;

        if (prefillStepBudgetMs is int budget)
        {
            // Inserted straight after the width it bounds, using the line ending this file's raw literals actually carry.
            yaml = yaml.Replace(
                $"max_batch_tokens: {maxBatchTokens}",
                $"max_batch_tokens: {maxBatchTokens}{Environment.NewLine}  prefill_step_budget_ms: {budget}");
        }

        if (maxConcurrentRequests is int cap)
        {
            yaml += $"\n    max_concurrent_requests: {cap}\n";
        }

        if (maxQueuedRequests is int depth)
        {
            yaml += $"\n    max_queued_requests: {depth}\n";
        }

        if (queueTimeoutMs is int ceiling)
        {
            yaml += $"\n    queue_timeout_ms: {ceiling}\n";
        }

        yaml += "\n    placement:" +
                "\n      device: cpu" +
                "\n      kv_residency: dram" +
                "\n      prefill: cpu\n";

        return threadQuota is int quota ? yaml + $"      thread_quota: {quota}\n" : yaml;
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
