using Lair.Backends.Cpu;
using Lair.Host;
using Lair.Host.Config;
using Lair.Http;
using Lair.Kv;
using Lair.Models;

namespace Lair.Tests;

/// <summary>
/// Dedicated prefill-lane tests: the placement key's loader posture, the lane split a dedicated posture resolves to
/// (default shares, quota-capped claims, disjoint pinned partitions, and the refusals that keep a half-lane engine from
/// opening), the live thread-quota mapping that moves only the decode side, the KV reservation a lane forward consumes
/// instead of allocating mid-flight, and — against the real-model fixture — a dedicated engine serving byte-identical
/// bytes to its shared-posture baseline while its prompt window runs beside decode work.
/// </summary>
[Trait("Category", "Prefill")]
public sealed class PrefillLaneTests : IDisposable
{
    private const string ModelName = "smollm-prefilllane";

    private static readonly string Pangram =
        "The quick brown fox jumps over the lazy dog while the patient cat watches the rain.";

    private readonly List<string> _tempPaths = [];

    void IDisposable.Dispose()
    {
        foreach (string path in _tempPaths)
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
                // Best-effort cleanup of temp fixtures.
            }
        }
    }

    // --- loader posture -------------------------------------------------------------------------------

    [Fact]
    public void Load_PrefillLaneOmitted_DefaultsShared()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig(PlacementYaml("prefill_lane: dedicated")));
        Assert.Equal("dedicated", config.Models[0].Placement.PrefillLane);

        LairConfig shared = YamlConfigLoader.Load(WriteConfig(
            "models:\n" +
            "  - name: qwen-test\n" +
            $"    path: '{TempModelFile()}'\n"));
        Assert.Equal("shared", shared.Models[0].Placement.PrefillLane);
    }

    [Theory]
    [InlineData("shared")]
    [InlineData("dedicated")]
    public void Load_PrefillLaneAcceptsTheTwoPostures(string posture)
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig(PlacementYaml($"prefill_lane: {posture}")));
        Assert.Equal(posture, config.Models[0].Placement.PrefillLane);
    }

    [Fact]
    public void Load_PrefillLaneInvalidScalar_FailsNamingModelKeyAndAcceptedSet()
    {
        ConfigException error = Assert.Throws<ConfigException>(
            () => YamlConfigLoader.Load(WriteConfig(PlacementYaml("prefill_lane: sticky"))));

        Assert.Contains("models[0].placement.prefill_lane 'sticky'", error.Message);
        Assert.Contains("qwen-test", error.Message);
        Assert.Contains("'shared'", error.Message);
        Assert.Contains("'dedicated'", error.Message);
    }

    [Fact]
    public void Load_PrefillLaneNonScalar_FailsLikeEveryOtherPlacementKey()
    {
        ConfigException error = Assert.Throws<ConfigException>(
            () => YamlConfigLoader.Load(WriteConfig(PlacementYaml("prefill_lane: [shared, dedicated]"))));

        Assert.Contains("models[0].placement.prefill_lane must be a text value.", error.Message);
    }

    [Fact]
    public void Diff_PrefillLaneFlip_ClassifiesRestartOnly()
    {
        string modelPath = TempModelFile();
        string Shared() => "models:\n" +
            "  - name: qwen-test\n" +
            $"    path: '{modelPath}'\n" +
            "    placement:\n" +
            "      prefill_lane: shared\n";

        LairConfig shared = YamlConfigLoader.Load(WriteConfig(Shared()));
        LairConfig dedicated = YamlConfigLoader.Load(WriteConfig(Shared().Replace("prefill_lane: shared", "prefill_lane: dedicated")));

        FieldChange change = Assert.Single(ConfigMutationTable.Diff(shared, dedicated));

        Assert.Equal("models[0].placement.prefill_lane", change.Field);
        Assert.Equal("shared", change.OldValue);
        Assert.Equal("dedicated", change.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, change.MutationClass);
    }

    // --- lane split resolution ------------------------------------------------------------------------

    [Fact]
    public void ResolveDedicated_DefaultEightWorkerClaim_SplitsTwoAndSix()
    {
        PrefillLaneLayout layout = ModelPrefillLanes.ResolveDedicated("m", 8, null, new HashSet<int>());

        Assert.Equal(8, layout.EffectiveClaimThreads);
        Assert.Equal(2, layout.PrefillShareThreads);
        Assert.Equal(6, layout.DecodeShareThreads);
        Assert.False(layout.IsPinned);
        Assert.Empty(layout.PrefillCoreIds);
        Assert.Empty(layout.DecodeCoreIds);
    }

    [Fact]
    public void ResolveDedicated_QuotaCapsClaimBeforeSplit()
    {
        PrefillLaneLayout layout = ModelPrefillLanes.ResolveDedicated("m", 8, 4, new HashSet<int>());

        Assert.Equal(4, layout.EffectiveClaimThreads);
        Assert.Equal(1, layout.PrefillShareThreads);
        Assert.Equal(3, layout.DecodeShareThreads);
    }

    [Fact]
    public void ResolveDedicated_TwoWorkerClaim_SplitsOneAndOne()
    {
        PrefillLaneLayout layout = ModelPrefillLanes.ResolveDedicated("m", 2, null, new HashSet<int>());

        Assert.Equal(1, layout.PrefillShareThreads);
        Assert.Equal(1, layout.DecodeShareThreads);
    }

    [Fact]
    public void ResolveDedicated_ClaimTooNarrowToSplit_RefusesNamingModelKeyAndOptions()
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelPrefillLanes.ResolveDedicated("tiny", 1, null, new HashSet<int>()));

        Assert.Contains("model 'tiny'", error.Message);
        Assert.Contains("placement.prefill_lane", error.Message);
        Assert.Contains("'shared'", error.Message);
    }

    [Fact]
    public void ResolveDedicated_UnresolvableClaim_RefusesBeforeSplitting()
    {
        Assert.Throws<ModelException>(() => ModelPrefillLanes.ResolveDedicated("m", 0, null, new HashSet<int>()));
        Assert.Throws<ModelException>(() => ModelPrefillLanes.ResolveDedicated("m", 8, 0, new HashSet<int>()));
    }

    [Fact]
    public void ResolveDedicated_PinnedSet_PartitionsLowerNumberedIdsForPrefill()
    {
        HashSet<int> cores = [0, 1, 2, 3, 4, 5, 6, 7];
        PrefillLaneLayout layout = ModelPrefillLanes.ResolveDedicated("m", 8, null, cores);

        Assert.True(layout.IsPinned);
        Assert.True(layout.PrefillCoreIds.SequenceEqual([0, 1]));
        Assert.True(layout.DecodeCoreIds.SequenceEqual([2, 3, 4, 5, 6, 7]));

        // Disjoint, and the two masks cover every declared id exactly once.
        Assert.Empty(layout.PrefillCoreIds.Intersect(layout.DecodeCoreIds));
        Assert.Equal(cores.Count, layout.PrefillCoreIds.Count + layout.DecodeCoreIds.Count);
    }

    [Fact]
    public void ResolveDedicated_PinnedFloorSplitWhenProportionalWouldEmptyDecode()
    {
        HashSet<int> cores = [0, 1, 2, 3];
        PrefillLaneLayout layout = ModelPrefillLanes.ResolveDedicated("m", 8, null, cores);

        // floor(4 * 2/8) = 1: the lower-numbered id takes the prefill side and every remaining id goes to decode.
        Assert.True(layout.PrefillCoreIds.SequenceEqual([0]));
        Assert.True(layout.DecodeCoreIds.SequenceEqual([1, 2, 3]));
    }

    [Fact]
    public void ResolveDedicated_PinnedSingleCore_RefusesNamingWhatToChange()
    {
        ModelException error = Assert.Throws<ModelException>(
            () => ModelPrefillLanes.ResolveDedicated("tight", 8, null, new HashSet<int> { 3 }));

        Assert.Contains("model 'tight'", error.Message);
        Assert.Contains("at least two declared cores", error.Message);
    }

    [Fact]
    public void TryMapLiveThreadQuota_MovesOnlyTheDecodeSide_AndRefusesWhenNoWorkerIsLeft()
    {
        PrefillLaneLayout layout = ModelPrefillLanes.ResolveDedicated("m", 8, null, new HashSet<int>());

        Assert.True(ModelPrefillLanes.TryMapLiveThreadQuota(layout, 8, 4, out int lowered));
        Assert.Equal(2, lowered);

        Assert.True(ModelPrefillLanes.TryMapLiveThreadQuota(layout, 8, 8, out int raised));
        Assert.Equal(6, raised);

        Assert.True(ModelPrefillLanes.TryMapLiveThreadQuota(layout, 8, null, out int unset));
        Assert.Equal(6, unset);

        // A quota whose claim minus the fixed prefill share leaves nothing for decode maps to no posture at all.
        Assert.False(ModelPrefillLanes.TryMapLiveThreadQuota(layout, 8, 2, out _));
        Assert.False(ModelPrefillLanes.TryMapLiveThreadQuota(layout, 0, 8, out _));

        // The layout itself never moved: the prefill side is fixed for the model's lifetime.
        Assert.Equal(2, layout.PrefillShareThreads);
        Assert.Equal(6, layout.DecodeShareThreads);
    }

    // --- reservation seam -----------------------------------------------------------------------------

    [Fact]
    public void ReserveAppends_ReservesCrossingBlocksUpFront_AndAppendsConsumeThemWithoutAllocating()
    {
        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 4);
        SequenceKv seq = new(pool);

        seq.ReserveAppends(33); // positions 0, 16, and 32 each cross into a fresh block

        Assert.Equal(1, pool.FreeBlocks); // three of four blocks are now held by this sequence alone
        Assert.Equal(3, pool.UsedBlocks);

        for (int i = 0; i < 33; i++)
        {
            seq.Append();
        }

        Assert.Equal(33, seq.Length);
        Assert.Equal(48, seq.TokenCapacity);
        Assert.Equal(1, pool.FreeBlocks); // every crossing append rode a reserved block: the free list never moved
        Assert.Equal(3, pool.UsedBlocks);

        seq.Dispose();
        Assert.Equal(4, pool.FreeBlocks);
    }

    [Fact]
    public void ReserveAppends_DisposeReturnsReservationsTheRunNeverConsumed()
    {
        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 4);
        SequenceKv seq = new(pool);

        seq.ReserveAppends(17);
        seq.Append(); // consumes one of the two reserved blocks

        seq.Dispose();
        Assert.Equal(4, pool.FreeBlocks); // the unconsumed reservation comes back with everything else
    }

    [Fact]
    public void ReserveAppends_PoolCannotSatisfy_ThrowsKvFull_AndHeldBlocksReturnOnDispose()
    {
        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 2);
        SequenceKv seq = new(pool);

        Assert.Throws<KvFullException>(() => seq.ReserveAppends(33)); // three crossings against a two-block pool

        seq.Dispose();
        Assert.Equal(2, pool.FreeBlocks); // whatever the refused reservation managed to take returns
    }

    // --- engine posture against the real-model fixture ------------------------------------------------

    [Fact]
    public async Task DedicatedOpen_ServesIdenticalBytesToSharedBaseline_WithLaneForwardAndDecodeWork()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        string longPrompt = SharedPrefix(5) + " In a distant library, maps of imaginary cities lined every wall.";
        string shortPrompt = "Describe the sea in one sentence.";

        string baselineText;
        int baselineCompletion;
        {
            using GenerationService plain = OpenPlain(shared: true);
            GenerationResult baseline = await plain.GenerateAsync(Completion(longPrompt, 16), CancellationToken.None);
            Assert.False(string.IsNullOrEmpty(baseline.Text));
            baselineText = baseline.Text;
            baselineCompletion = baseline.CompletionTokens;
        }

        using GenerationService lane = OpenPlain(shared: false);

        Assert.NotNull(lane.DedicatedPrefillLanes);
        Assert.Equal(1, lane.DedicatedPrefillLanes!.PrefillShareThreads); // threads: 4 → claim 4 → floor-quarter prefill
        Assert.Equal(3, lane.DedicatedPrefillLanes.DecodeShareThreads);
        Assert.Equal(1, lane.DedicatedPrefillLaneWorkerCap); // the lane backend holds its share alone

        GenerationResult solo = await lane.GenerateAsync(Completion(longPrompt, 16), CancellationToken.None);
        Assert.Equal(baselineText, solo.Text);
        Assert.Equal(baselineCompletion, solo.CompletionTokens);
        Assert.True(solo.FinishReason is "eos" or "length");
        Assert.True(lane.Stats.PrefillChunksIssued >= 1);
        Assert.True(lane.Stats.PrefillTokensForwarded > 0);

        // Overlapped composition: a prefilling prompt and decoding siblings run at the same boundaries, and each
        // sequence still samples its own logits row — identical bytes to the shared-posture baseline.
        GenerationResult[] concurrent = await Task.WhenAll(
            lane.GenerateAsync(Completion(longPrompt, 16), CancellationToken.None),
            lane.GenerateAsync(Completion(shortPrompt, 8), CancellationToken.None),
            lane.GenerateAsync(Completion(SharedPrefix(2) + " A potter shaped a jar from river clay.", 8), CancellationToken.None));

        Assert.Equal(baselineText, concurrent[0].Text);
        Assert.All(concurrent, r => Assert.False(string.IsNullOrEmpty(r.Text)));
    }

    [Fact]
    public async Task PrefillLaneFlipMidServe_RefusedRestartOnly_ValueRetained_ServingUnchanged()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        string prompt = SharedPrefix(3) + " A lighthouse keeper logged every passing ship.";

        using GenerationService plain = OpenPlain(shared: true);
        GenerationResult baseline = await plain.GenerateAsync(Completion(prompt, 10), CancellationToken.None);

        using LaneHarness harness = new(BuildYaml(prefillLane: "dedicated"));

        harness.EditYaml(BuildYaml(prefillLane: "shared"));

        ReloadSummary summary = await harness.WaitForSummaryAsync(
            s => s.ValidationOk && s.Results.Any(r => r.Field == "models[0].placement.prefill_lane"),
            "the prefill_lane flip never reached the engine as a classified change");

        ReloadFieldResult refused = summary.Results.Single(r => r.Field == "models[0].placement.prefill_lane");
        Assert.Equal(MutationClassNames.RestartOnly, refused.MutationClass);
        Assert.Equal("refused", refused.Outcome);
        Assert.Contains("requires restart (restart-only)", refused.Detail);

        // The refused field keeps its opened value: the engine still runs the lane posture it opened with.
        Assert.Equal("dedicated", harness.Engine.EffectiveConfiguration.Models[0].Placement.PrefillLane);
        Assert.NotNull(harness.Engine.DedicatedPrefillLanes);

        GenerationResult served = await harness.Engine.GenerateAsync(Completion(prompt, 10), CancellationToken.None);
        Assert.Equal(baseline.Text, served.Text);
    }

    [Fact]
    public async Task ThreadQuotaEditUnderDedicated_RefusedWhenDecodeSideWouldEmpty_AppliedOtherwise()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        // threads: 4 with no quota: claim 4 → the load split is 1 prefill / 3 decode, fixed for the engine's lifetime.
        using LaneHarness harness = new(BuildYaml(prefillLane: "dedicated", threads: 4));
        Assert.Equal(1, harness.Engine.DedicatedPrefillLanes!.PrefillShareThreads);

        string prompt = SharedPrefix(2) + " A cartographer sketched an unpublished strait.";

        // A quota whose claim minus the fixed prefill share leaves no decode worker: refused naming the posture, nothing applied.
        harness.EditYaml(BuildYaml(prefillLane: "dedicated", threads: 4, threadQuota: 1));
        ReloadSummary refused = await harness.WaitForSummaryAsync(
            s => s.ValidationOk && s.Results.Any(r => r.Field == "models[0].placement.thread_quota" && r.Outcome == "refused"),
            "the unmappable quota edit never landed as a refusal");

        ReloadFieldResult refusal = refused.Results.Single(r => r.Field == "models[0].placement.thread_quota");
        Assert.Contains("dedicated prefill lane", refusal.Detail);
        Assert.Null(harness.Engine.EffectiveConfiguration.Models[0].Placement.ThreadQuota); // refused field keeps its old value

        // A quota that leaves both sides workers applies to the decode side alone.
        harness.EditYaml(BuildYaml(prefillLane: "dedicated", threads: 4, threadQuota: 3));
        ReloadSummary applied = await harness.WaitForSummaryAsync(
            s => s.ValidationOk && s.Results.Any(r => r.Field == "models[0].placement.thread_quota" && r.Outcome == "applied"),
            "the mappable quota edit never applied");

        Assert.Contains(
            applied.Results,
            r => r.Field == "models[0].placement.thread_quota" && r.MutationClass == MutationClassNames.Live);
        Assert.Equal(3, harness.Engine.EffectiveConfiguration.Models[0].Placement.ThreadQuota);
        Assert.Equal(1, harness.Engine.DedicatedPrefillLanes!.PrefillShareThreads); // the prefill side never moved

        GenerationResult served = await harness.Engine.GenerateAsync(Completion(prompt, 8), CancellationToken.None);
        Assert.False(string.IsNullOrEmpty(served.Text));
    }

    // --- fixture plumbing -----------------------------------------------------------------------------

    private static string SharedPrefix(int sentences) => string.Join(" ", Enumerable.Repeat(Pangram, sentences));

    private static GenerationRequest Completion(string prompt, int maxTokens) => new()
    {
        Model = ModelName,
        IsChat = false,
        Prompt = prompt,
        MaxTokens = maxTokens,
        Temperature = 0f, // greedy: byte-comparable across engines and postures
    };

    private static string BuildYaml(string prefillLane, int threads = 4, int? threadQuota = null)
    {
        string yaml = $"""
            server:
              bind: 127.0.0.1
              port: 8200

            engine:
              threads: {threads}
              block_size: 16
              max_batch_tokens: 8

            models:
              - name: {ModelName}
                path: '{ModelFixture.Resolve()!}'
                type: chat
                max_context_tokens: 512
                kv_blocks: 24
                placement:
                  device: cpu
                  kv_residency: dram
                  prefill: cpu
                  prefill_lane: {prefillLane}
            """;

        return threadQuota is int quota ? yaml + $"\n      thread_quota: {quota}\n" : yaml;
    }

    private GenerationService OpenPlain(bool shared)
    {
        string dir = Path.Combine(Path.GetTempPath(), $"lair-prefilllane-plain-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "lair.yaml");
        File.WriteAllText(path, BuildYaml(shared ? "shared" : "dedicated"));
        _tempPaths.Add(path);

        return GenerationService.Open(YamlConfigLoader.Load(path));
    }

    private string PlacementYaml(string placementLine) =>
        "models:\n" +
        "  - name: qwen-test\n" +
        $"    path: '{TempModelFile()}'\n" +
        "    placement:\n" +
        $"      {placementLine}\n";

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-prefilllane-config-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, yaml);
        _tempPaths.Add(path);
        return path;
    }

    private string TempModelFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-prefilllane-model-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, [0x47, 0x47, 0x55, 0x46]);
        _tempPaths.Add(path);
        return path;
    }

    private static BlockPool NewPool(
        Lair.Contracts.IBackend backend,
        int capacityBlocks = 4,
        int blockTokens = 16,
        int layers = 2,
        int kvHeads = 2,
        int headDim = 8) => new(blockTokens: blockTokens, layerCount: layers, kvHeads: kvHeads, headDim: headDim, capacityBlocks: capacityBlocks, backend: backend);

    /// <summary>One live engine with a watching hot-reload host over a temporary config file, mirroring the hot-reload suite's harness.</summary>
    private sealed class LaneHarness : IDisposable
    {
        public string Dir { get; }
        public string ConfigPath { get; }
        public GenerationService Engine { get; }

        private readonly object _logGate = new();
        private readonly List<string> _lines = [];
        private readonly ConfigWatcher _watcher;

        public LaneHarness(string yaml)
        {
            Dir = Path.Combine(Path.GetTempPath(), $"lair-prefilllane-{Guid.NewGuid():N}");
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
            _watcher.Start();
        }

        public void EditYaml(string yaml) => File.WriteAllText(ConfigPath, yaml);

        public IReadOnlyList<string> LogLinesSnapshot()
        {
            lock (_logGate)
            {
                return [.. _lines];
            }
        }

        public async Task<ReloadSummary> WaitForSummaryAsync(Func<ReloadSummary, bool> done, string what)
        {
            long deadline = Environment.TickCount64 + 25_000;
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

        void IDisposable.Dispose()
        {
            _watcher.Dispose();
            Engine.Dispose();

            try
            {
                Directory.Delete(Dir, recursive: true);
            }
            catch (IOException)
            {
            }
            catch (UnauthorizedAccessException)
            {
            }
        }
    }
}
