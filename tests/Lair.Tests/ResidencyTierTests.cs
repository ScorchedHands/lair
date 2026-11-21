using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Http.Monitoring;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// The RAM residency tier behind the expert gather path: a hot set chosen by capacity/cost — decayed router hit frequency
/// times each unit's cold slab bytes — rather than insertion order, recency, or raw cumulative counts. Promotion fills free
/// budget best-value-first and stops at the ceiling (RAM quota less working-set reserve); admission past the ceiling happens
/// only by demoting residents the candidate beats by the min-gain margin, so near-ties never churn residency; aged scores let
/// a drifting workload swap out experts that stopped earning their bytes. Hot slabs are byte-exact copies of what a cold slab
/// view answers with — gate, up, and down alike — so gather output stays bit-identical whichever side one step's slabs land on,
/// proved end-to-end against the same file opened through plain mmap weights. Answer tallies (hits/misses/saved-bytes estimate)
/// count every slab resolution; the metrics bus carries them additively with a null posture when no tier is attached.
/// </summary>
[Trait("Category", "Residency")]
public sealed class ResidencyTierTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

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
            }
        }

        _tempFiles.Clear();
    }

    // ---------------------------------------------------------------- policy mechanics, straight against the tier

    [Fact]
    public void Tier_PromotedExpertAnswersEveryRoleFromResidency_ByteIdenticalToColdSlabViews()
    {
        // A union note admits an expert's whole gate/up/down chain; from then on every role resolves out of the tier's own copies,
        // byte-for-byte what the plain slab provider would have sliced out of the stacked payload — and each such resolution is tallied.
        const int experts = 2;
        const int hidden = 4;
        const int inter = 2;

        FixtureSource cold = new();
        (ExpertSlabRole gate, ExpertSlabRole up, ExpertSlabRole down) stacks = Stacks(experts, hidden, inter);

        using RamHotExpertTier tier = new(cold, ramQuotaBytes: long.MaxValue, swapPeriodLayerSteps: 1);
        ExpertSlabWeightSource reference = new(cold); // a plain view provider over the same source — the byte-for-byte comparison every hot slab is checked against; nothing to dispose here, it borrows the array-backed fixture which lives for the whole test

        Assert.False(tier.IsResident(stacks.gate.Stack, 0));

        tier.NoteActiveExperts(stacks.gate, stacks.up, stacks.down, [0], [3]); // one note with swap period 1: age + rebalance admit the unit immediately
        Assert.True(tier.IsResident(stacks.gate.Stack, 0));
        Assert.Equal(1, tier.HotUnits);

        long roleBytes = (long)inter * hidden * 4; // gate and up blocks are inter×hidden floats each...
        long downRoleBytes = (long)hidden * inter * 4; // ...and down is the flipped geometry — a resident expert must answer all three roles from its copies

        foreach ((ExpertSlabRole role, long perExpert) in new[] { (stacks.gate, roleBytes), (stacks.up, roleBytes), (stacks.down, downRoleBytes) })
        {
            byte[] hot = tier.ResolveSlab(role.Stack, 0, perExpert).ToArray();
            Assert.True(
                hot.SequenceEqual(reference.ResolveSlab(role.Stack, 0, perExpert).ToArray()),
                $"resident role '{role.Stack.Name}' must answer exactly the bytes a cold slab view carries");
        }

        Assert.Equal(3, tier.HitsTotal); // gate, up, down each answered from residency once
        Assert.Equal(0, tier.MissesTotal);
        Assert.Equal(roleBytes * 2 + downRoleBytes, tier.SavedBytesEstimateTotal); // the estimate counts exactly what those resolutions avoided reading cold

        // An expert that was never noted stays fully cold: same bytes through the backing store's contiguous views.
        byte[] coldSlab = tier.ResolveSlab(stacks.gate.Stack, 1, roleBytes).ToArray();
        Assert.True(coldSlab.SequenceEqual(reference.ResolveSlab(stacks.gate.Stack, 1, roleBytes).ToArray()));
    }

    [Fact]
    public void Tier_HotSetRanksByExpectedGatherCostSaved_NotInsertionOrderOrRawHitCount()
    {
        // Two layers with different per-expert chain sizes: the cheap layer's expert is noted first and picked more times, but its unit
        // saves fewer bytes cold than the expensive layer's less-picked unit — so capacity/cost residency holds the big-chain expert and
        // leaves the higher-count one out. Plain insertion order would have taken the small one; raw hit counts (4 > 3) too: only value =
        // decayed frequency × that unit's cold slab bytes ranks them this way.
        const int hidden = 4;

        FixtureSource cold = new();
        (ExpertSlabRole gate, ExpertSlabRole up, ExpertSlabRole down) smallStacks = Stacks(2, hidden, inter: 2); // unit = 3 blocks of 8 floats = 96 B per expert
        (ExpertSlabRole gate, ExpertSlabRole up, ExpertSlabRole down) bigStacks = Stacks(2, hidden, inter: 10, prefix: "big"); // ...against a chain five times as expensive to read cold — distinct stack names so the tier records it as its own layer geometry

        long smallUnit = 3 * 8 * 4L;
        long bigUnit = 3 * (long)10 * hidden * 4;

        using RamHotExpertTier tier = new(cold, ramQuotaBytes: bigUnit, swapPeriodLayerSteps: 2); // budget admits exactly one unit of the expensive chain — and nothing else can slip in beside it
        Assert.Equal(bigUnit, tier.BudgetBytes);

        tier.NoteActiveExperts(smallStacks.gate, smallStacks.up, smallStacks.down, [0], [4]); // noted first, four selections: wins on insertion order and hit count alone...
        tier.NoteActiveExperts(bigStacks.gate, bigStacks.up, bigStacks.down, [1], [3]); // ...but this note closes the swap round, and its unit ranks higher by expected bytes saved

        Assert.True(smallUnit < bigUnit, "the fixture must make one chain strictly cheaper to read cold than the other");

        Assert.True(tier.IsResident(bigStacks.gate.Stack, 1), "the expensive-chain expert must be resident: its value (selections × cold slab bytes) beats the cheaper one's");
        Assert.False(tier.IsResident(smallStacks.gate.Stack, 0), "the higher-count cheap unit must stay out when cost-weighted ranking puts it behind and budget is full");
        Assert.Equal(bigUnit, tier.ResidentBytes);
        Assert.True(tier.ResidentBytes <= tier.BudgetBytes, "residency may never exceed the stated ceiling");
        Assert.Equal(0, tier.DemotionsTotal); // nothing was evicted to make room: the winner simply filled free budget first — no justified swap happened because none was needed
    }

    [Fact]
    public void Tier_SwapOnlyOnMargin_NearTiesNeverChurn_AndAgedDriftSwapsWhenTheGainClears()
    {
        // Budget fits exactly one unit. An incoming unit that beats the resident by less than the min-gain margin leaves residency alone;
        // as scores age (each swap round halves them) and a workload drifts toward another expert, its standing clears the margin only after
        // several rounds — so the hot set follows earning power over time, not cumulative totals: raw counts would keep the first expert forever.
        const int hidden = 4;
        const int inter = 2;

        FixtureSource cold = new();
        (ExpertSlabRole gate, ExpertSlabRole up, ExpertSlabRole down) stacks = Stacks(2, hidden, inter); // unit: three blocks of 8 floats each — every score scales the same unit bytes here

        long unit = 3 * 8 * 4L;
        using RamHotExpertTier tier = new(cold, ramQuotaBytes: unit, swapPeriodLayerSteps: 1); // one note per swap round keeps the aging arithmetic exact (×0.5 is binary-exact on these counts)

        tier.NoteActiveExperts(stacks.gate, stacks.up, stacks.down, [0], [8]); // aged to 4 selections standing; promoted into free budget
        Assert.True(tier.IsResident(stacks.gate.Stack, 0));

        tier.NoteActiveExperts(stacks.gate, stacks.up, stacks.down, [1], [5]); // expert 1 fresh at 5 → aged 2.5 against the resident's aged 2: value ratio exactly the min-gain gate — `beats by margin` is strict, so nothing evicts yet
        Assert.False(tier.IsResident(stacks.gate.Stack, 1), "a candidate beating its victim by exactly the margin (and no more) must not evict anything");
        Assert.Equal(0, tier.DemotionsTotal);

        tier.NoteActiveExperts(stacks.gate, stacks.up, stacks.down, [1], [5]); // aged again: expert 0 decays to 1 while expert 1's standing reaches 3.75 — now past the margin against every resident...
        Assert.True(tier.IsResident(stacks.gate.Stack, 1), "the drifted workload's unit earns its residency once it clears the min-gain gate");
        Assert.False(tier.IsResident(stacks.gate.Stack, 0), "...and exactly then the stale hot expert is demoted to make room for it");
        Assert.Equal(1, tier.DemotionsTotal); // one justified swap, no churn before it
    }

    [Fact]
    public void Tier_ResidentBytesNeverExceedBudget_AndReserveCarveoutCanLeaveNothingHot()
    {
        // The ceiling binds under a churning workload: with room for exactly two units of this geometry, residency saturates there and stays
        // bounded however many experts keep cycling through the notes. A reserve at or past the quota leaves no admission room at all — the hot set is empty by construction.
        const int hidden = 4;

        FixtureSource cold = new();
        (ExpertSlabRole gate, ExpertSlabRole up, ExpertSlabRole down) stacks = Stacks(3, hidden, inter: 2); // three experts competing for room sized for two chains

        long unit = 3 * 8 * 4L;
        using RamHotExpertTier tier = new(cold, ramQuotaBytes: 2 * unit + 100, workingSetReserveBytes: 100, swapPeriodLayerSteps: 1); // quota less the stated reserve is exactly two units of ceiling
        Assert.Equal(2 * unit, tier.BudgetBytes);

        int[] cycle = [0, 1, 2];
        for (int round = 0; round < 4; round++) // a workload that keeps rotating the union through every swap round...
        {
            foreach (int expert in cycle)
            {
                tier.NoteActiveExperts(stacks.gate, stacks.up, stacks.down, [expert], [round % 3 + 1]);

                Assert.True(tier.ResidentBytes <= tier.BudgetBytes, $"residency crossed its ceiling mid-cycle: {tier.ResidentBytes} > {tier.BudgetBytes}");
                Assert.True(tier.HotUnits <= 2, "no more than two of these chains may ever be resident under this budget");
            }
        }

        using RamHotExpertTier starved = new(cold, ramQuotaBytes: unit, workingSetReserveBytes: unit); // reserve at the quota: nothing is left to admit...
        Assert.Equal(0, starved.BudgetBytes);
        foreach (int expert in cycle)
        {
            starved.NoteActiveExperts(stacks.gate, stacks.up, stacks.down, [expert], [1]);
        }

        Assert.Equal(0, starved.HotUnits); // ...so every slab keeps answering cold and the hot set stays empty by construction
        Assert.False(starved.IsResident(stacks.gate.Stack, 0));
    }

    // ---------------------------------------------------------------- engine-level parity: a tier-backed model answers identically to plain mmap weights

    [Fact]
    public void TransformerModelOpen_TierBackedWeights_LogitsBitIdenticalToPlainMmap_WithPopulatedHotTier()
    {
        // The same single-layer qwen2moe file opened two ways — straight mmap weight source, and wrapped in the residency tier as its weight
        // source (the engine picks it up as slab provider and feeds every sparse step's union into its cost model). Every token forwarded
        // through each must produce bit-identical logits: residency changes where slab bytes come from, never the arithmetic over them.
        string path = WriteTemp(BuildMoeImage());

        using CpuBackend backend = new(threads: 1, allowAvx2: false);
        using GgufFile gguf = GgufFile.Open(path);

        const int unitBytes = (6 * 8 + 6 * 8 + 8 * 6) * 4; // this fixture's per-expert chain: gate+up inter×hidden and down flipped, F32 — the figure the budget arithmetic below works against
        using GgufMmapWeightSource mmap = new(gguf);
        using RamHotExpertTier tier = new(mmap, ramQuotaBytes: 4 * unitBytes + 512, workingSetReserveBytes: 512, swapPeriodLayerSteps: 2); // budget 2048 B: room for three of the four experts' chains (3 × 576 = 1728), never all four

        using TransformerModel plain = TransformerModel.Open(gguf, backend, maxContextTokens: 8);
        using TransformerModel hot = TransformerModel.Open(gguf, tier, backend, maxContextTokens: 8);
        Assert.NotNull(hot.AttachedResidencyTier); // the injected source is recognized as the engine's residency seam
        Assert.Equal(RamHotExpertTier.TierKind, hot.AttachedResidencyTier!.Kind);

        using BlockPool plainPool = NewPool(plain, backend);
        using BlockPool hotPool = NewPool(hot, backend);

        int[] tokens = [0, 1, 2, 3];
        for (int step = 0; step < 8; step++) // cycling the fixture's discriminating router: repeated unions feed the cost model while both models forward identical streams
        {
            int token = tokens[step % tokens.Length];

            float[] plainLogits = plain.DecodeNext(token, new SequenceKv(plainPool)).ToArray();
            float[] hotLogits = hot.DecodeNext(token, new SequenceKv(hotPool)).ToArray();

            AssertBitsEqual(plainLogits, hotLogits, $"step {step} tier-backed logits vs the same model over plain mmap weights"); // every step — resident or not — carries identical bytes
        }

        IResidencyTier attached = hot.AttachedResidencyTier!;
        Assert.True(attached.HotUnits >= 1, "the replayed workload must have populated the hot tier through its notes");
        Assert.True(tier.ResidentBytes <= tier.BudgetBytes, "engine-fed residency stays under the stated ceiling");
        Assert.True(attached.HitsTotal > 0, $"a re-routed expert resident from an earlier step must answer slab resolutions from residency (hits {attached.HitsTotal}, misses {attached.MissesTotal})");
        Assert.Equal(tier.BudgetBytes, attached.BudgetBytes);

        // The hot tier's own slabs are byte-exact against the fixture bytes it wrapped — same bytes the parity above proved arithmetic-identical.
        LogicalTensor gateStack = new("blk.0.ffn_gate_exps.weight", new TensorShape([4, 6, 8]), QuantFormat.F32, 0, 4 * 6 * 8 * 4); // name-keyed lookup: the tier answers its recorded copy for this stacked role
        int residentExpert = Enumerable.Range(0, 4).First(e => attached.IsResident(gateStack, e)); // at least one expert is hot by the assertion above...

        float[] gatePayload = StackedExperts([0.35f, -0.25f, 0.45f, -0.15f], 4 * 6 * 8); // exactly what BuildMoeImage wrote into this tensor
        byte[] expectedSlab = new byte[6 * 8 * 4]; // one expert's gate block: the inter×hidden floats at that expert's offset in the stacked payload
        for (int i = 0; i < expectedSlab.Length / 4; i++)
        {
            BitConverter.TryWriteBytes(expectedSlab.AsSpan(i * 4, 4), gatePayload[residentExpert * 6 * 8 + i]);
        }

        Assert.True(
            tier.ResolveSlab(gateStack, residentExpert, 6 * 8 * 4L).ToArray().SequenceEqual(expectedSlab),
            "the engine-facing hot slab must carry exactly the bytes its expert's block carries in the file");
    }

    // ---------------------------------------------------------------- metrics seam: tier numbers ride snapshots additively; nothing attached stays null

    [Fact]
    public async Task EngineSnapshot_CarriesAttachedResidencyTierNumbers_AndStaysNull_WithoutOne()
    {
        string path = WriteTemp(BuildMoeImage());

        using CpuBackend backend = new(threads: 1, allowAvx2: false);
        using GgufFile gguf = GgufFile.Open(path);
        using GgufMmapWeightSource mmap = new(gguf);
        using RamHotExpertTier tier = new(mmap, ramQuotaBytes: 4096, workingSetReserveBytes: 256, swapPeriodLayerSteps: 1);

        static ModelFacts Facts(string? residencyTier) => new(
            "qwen2moe", // the fixture's architecture...
            WorkerThreads: 1,
            CoreSet: null,
            ThreadQuota: null,
            DecodeDevice: "cpu",
            PrefillDevice: "cpu",
            KvResidency: "dram",
            PrefillKernel: "cpu",
            MatmulWeightFormats: "f32", // ...a minimal facts posture otherwise — what matters here is the trailing residency slot, filled or left at its default null exactly as GenerationService pushes it
            ResidencyTier: residencyTier);

        using TransformerModel model = TransformerModel.Open(gguf, tier, backend, maxContextTokens: 8);
        using BlockPool pool = NewPool(model, backend);
        using GenerationScheduler scheduler = new(model, pool, maxBatchTokens: 4);

        await using EngineMetricsBus attachedBus = new("residency-tier", scheduler, pool, cache: null, NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(30));
        attachedBus.AttachModelFacts(Facts(tier.Kind)); // the posture name rides facts exactly as GenerationService pushes it...
        attachedBus.AttachResidencyProvider(() => new ResidencyTierAccounting(tier.HotUnits, tier.BudgetBytes, tier.ResidentBytes, tier.HitsTotal, tier.MissesTotal, tier.SavedBytesEstimateTotal));

        await using EngineMetricsBus plainBus = new("no-tier", scheduler, pool, cache: null, NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(30));
        plainBus.AttachModelFacts(Facts(null)); // ...and a bus with no provider attached — the default serving shape...

        // Snapshots publish only while an engine is active, so each bus gets one encoder-shaped burst to observe (the same counters a real encode-only serve lights up) — enough activity for the sampler's next tick to land a snapshot on both rings.
        attachedBus.BeginEncoderRequest();
        attachedBus.RecordEncoderPositions(2);
        attachedBus.EndEncoderRequest(allInputsPooled: true);

        plainBus.BeginEncoderRequest();
        plainBus.RecordEncoderPositions(1);
        plainBus.EndEncoderRequest(allInputsPooled: true);

        Stopwatch clock = Stopwatch.StartNew();
        while ((attachedBus.RingCount == 0 || plainBus.RingCount == 0) && clock.Elapsed < TimeSpan.FromSeconds(15))
        {
            await Task.Delay(25);
        }

        Assert.True(attachedBus.RingCount > 0, "the metrics bus published no snapshot for an attached residency tier");
        Assert.True(plainBus.RingCount > 0, "the metrics bus published no snapshot without a residency seam");

        MetricsSnapshot wired = attachedBus.SnapshotRing()[0]; // nothing decoded after attach: the counters in the snapshot are exactly what the provider reports now
        Assert.Equal(RamHotExpertTier.TierKind, wired.ResidencyTier);
        Assert.Equal(tier.BudgetBytes, wired.ResidencyBudgetBytes);
        Assert.Equal(tier.HotUnits, wired.ResidencyHotUnits ?? -1);
        Assert.True(wired.ResidencyResidentBytes >= 0 && wired.ResidencyHitsTotal >= 0 && wired.ResidencyMissesTotal >= 0 && wired.ResidencySavedBytesEstimateTotal >= 0, "attached tallies publish as live readings, not nulls");

        MetricsSnapshot defaultShape = plainBus.SnapshotRing()[0]; // ...and a host that attaches no provider keeps every residency slot reading as nothing rather than zero-filled
        Assert.Null(defaultShape.ResidencyTier);
        Assert.Null(defaultShape.ResidencyHotUnits);
        Assert.Null(defaultShape.ResidencyBudgetBytes);
        Assert.Null(defaultShape.ResidencyResidentBytes);
        Assert.Null(defaultShape.ResidencyHitsTotal);
        Assert.Null(defaultShape.ResidencyMissesTotal);
        Assert.Null(defaultShape.ResidencySavedBytesEstimateTotal); // ...so every residency slot on that bus reads as nothing rather than zero-filled, exactly the pre-seam snapshot shape
    }

    // ---------------------------------------------------------------- fixtures: a compact single-layer qwen2moe file and array-backed stacks

    private static BlockPool NewPool(TransformerModel model, CpuBackend backend) =>
        new(blockTokens: 16, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks: 8, backend);

    /// <summary>An array-backed whole-tensor weight source for direct tier tests: payloads sized straight from each stack's declared SizeBytes, distinct per expert so slab views are distinguishable.</summary>
    private sealed class FixtureSource : IWeightSource
    {
        public ReadOnlySpan<byte> Resolve(LogicalTensor tensor)
        {
            int elements = checked((int)(tensor.SizeBytes / 4)); // F32 payloads sized exactly to each stack's declared size, whatever geometry the test asked for

            byte[] bytes = new byte[elements * 4];
            for (int i = 0; i < elements; i++)
            {
                BitConverter.TryWriteBytes(bytes.AsSpan(i * 4, 4), 0.05f + 0.13f * (i % 97) + 0.02f * (i / 64)); // any distinct pattern: these tests only ever compare tier bytes against the same source's cold views
            }

            return bytes;
        }

        public void Dispose()
        {
        }
    }

    private static (ExpertSlabRole Gate, ExpertSlabRole Up, ExpertSlabRole Down) Stacks(int experts, int hidden, int inter, string prefix = "small")
    {
        LogicalTensor gate = new($"{prefix}.gate", new TensorShape([experts, inter, hidden]), QuantFormat.F32, 0, (long)experts * inter * hidden * 4);
        LogicalTensor up = new($"{prefix}.up", new TensorShape([experts, inter, hidden]), QuantFormat.F32, 0, (long)experts * inter * hidden * 4);
        LogicalTensor down = new($"{prefix}.down", new TensorShape([experts, hidden, inter]), QuantFormat.F32, 0, (long)experts * hidden * inter * 4);

        return (
            new ExpertSlabRole(gate, new TensorShape([inter, hidden]), (long)inter * hidden * 4),
            new ExpertSlabRole(up, new TensorShape([inter, hidden]), (long)inter * hidden * 4),
            new ExpertSlabRole(down, new TensorShape([hidden, inter]), (long)hidden * inter * 4));
    }

    // A single-layer qwen2moe file: hidden 8, heads 2/kv-heads 2/head dim 4, ffn 6, vocab 5, four routed experts with a discriminating router — the same shape MoeTests drives through gather. No shared expert is declared; routing alone feeds the tier's notes.
    private static byte[] BuildMoeImage()
    {
        const int hidden = 8;
        const int inter = 6;
        const int experts = 4;

        List<(string Key, object Value)> metadata =
        [
            ("general.architecture", "qwen2moe"),
            ("qwen2moe.embedding_length", hidden),
            ("qwen2moe.block_count", 1),
            ("qwen2moe.attention.head_count", 2),
            ("qwen2moe.attention.head_count_kv", 2),
            ("qwen2moe.feed_forward_length", inter),
            ("qwen2moe.attention.layer_norm_rms_epsilon", 1e-5f),
            ("qwen2moe.vocab_size", 5),
            ("qwen2moe.expert_count", experts),
            ("qwen2moe.expert_used_count", 2),
            ("qwen2moe.expert_weights_scale", 1f),
            ("qwen2moe.expert_weights_norm", true),
        ];

        float[] embeddings =
        [
            0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f, 0.70f, 0.80f,
            0.90f, 0.15f, 0.25f, 0.35f, 0.45f, 0.55f, 0.65f, 0.75f,
            0.12f, 0.88f, 0.16f, 0.84f, 0.20f, 0.80f, 0.24f, 0.76f,
            0.33f, 0.31f, 0.99f, 0.11f, 0.77f, 0.22f, 0.55f, 0.44f,
            0.61f, 0.17f, 0.28f, 0.39f, 0.50f, 0.91f, 0.13f, 0.72f,
        ];

        float[] router =
        [
            1.2f, -0.4f, 0.3f, -0.9f, 0.7f, -0.5f, 0.8f, -0.6f,
            -0.5f, 1.1f, -0.3f, 0.9f, -0.7f, 0.4f, -0.8f, 0.2f,
            0.4f, 0.7f, -1.0f, 0.5f, -0.6f, 0.3f, 0.9f, -0.8f,
            -0.9f, 0.2f, 0.8f, -0.4f, 0.6f, -1.1f, 0.1f, 0.7f,
        ];

        float[] attnQ = new float[hidden * hidden]; // nonzero q/k rows so attention is real arithmetic in both models alike — parity compares the engine's own outputs either way
        for (int i = 0; i < attnQ.Length; i++)
        {
            attnQ[i] = 0.4f * (0.1f + 0.37f * (i % 9));
        }

        float[] attnK = new float[hidden * hidden];
        for (int i = 0; i < attnK.Length; i++)
        {
            attnK[i] = 0.2f * (0.1f + 0.37f * (i % 9));
        }

        List<TensorDef> tensors =
        [
            new("token_embd.weight", [5, hidden], embeddings),
            new("blk.0.attn_q.weight", [hidden, hidden], attnQ),
            new("blk.0.attn_k.weight", [hidden, hidden], attnK),
            new("blk.0.attn_v.weight", [hidden, hidden], new float[hidden * hidden]), // zero v/o rows keep the forward's attention contribution a known constant — routing still discriminates on the embedding-row stream
            new("blk.0.attn_output.weight", [hidden, hidden], new float[hidden * hidden]),
            new("blk.0.attn_norm.weight", [hidden], Norm(1f)),
            new("blk.0.ffn_norm.weight", [hidden], Norm(1.1f)),
            new("output_norm.weight", [hidden], Norm(0.95f)),
            new("blk.0.ffn_gate_inp.weight", [experts, hidden], router),
            new("blk.0.ffn_gate_exps.weight", [experts, inter, hidden], StackedExperts([0.35f, -0.25f, 0.45f, -0.15f], experts * inter * hidden)),
            new("blk.0.ffn_up_exps.weight", [experts, inter, hidden], StackedExperts([0.30f, 0.28f, 0.26f, 0.24f], experts * inter * hidden)),
            new("blk.0.ffn_down_exps.weight", [experts, hidden, inter], StackedExperts([1.7f, -1.3f, 2.1f, -0.9f], experts * hidden * inter)),
        ];

        return BuildGguf(metadata, tensors);
    }

    private static float[] Norm(float factor)
    {
        float[] values = new float[8];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = factor * (1f + 0.05f * (i % 4));
        }

        return values;
    }

    private static float[] StackedExperts(float[] perExpertFactor, int total)
    {
        const int blockLength = 6 * 8; // inter×hidden and hidden×inter hold the same element count: one helper covers both geometries
        float[] stacked = new float[total];

        for (int e = 0; e < perExpertFactor.Length; e++)
        {
            for (int i = 0; i < blockLength; i++)
            {
                stacked[e * blockLength + i] = perExpertFactor[e] * (0.15f + 0.23f * (i % 6)) + 0.04f * e;
            }
        }

        return stacked;
    }

    private sealed record TensorDef(string Name, int[] Dims, float[] Payload); // dims in logical (row-major) order

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-residency-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    /// <summary>Builds a GGUF v3 image from KV pairs and F32 tensors — the same layout MoeTests writes.</summary>
    private static byte[] BuildGguf(IReadOnlyList<(string Key, object Value)> kvPairs, IReadOnlyList<TensorDef> tensors)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(GgufMagic);
        writer.Write(3u); // version
        writer.Write((ulong)tensors.Count);
        writer.Write((ulong)kvPairs.Count);

        foreach ((string key, object value) in kvPairs)
        {
            WriteGgufString(writer, key);
            switch (value)
            {
                case string s:
                    writer.Write(8u); // STRING
                    WriteGgufString(writer, s);
                    break;
                case int i:
                    writer.Write(5u); // INT32
                    writer.Write(i);
                    break;
                case float f:
                    writer.Write(6u); // FLOAT32
                    writer.Write(f);
                    break;
                case bool b:
                    writer.Write(7u); // BOOL
                    writer.Write((byte)(b ? 1 : 0));
                    break;
                default:
                    throw new ArgumentException($"Unsupported fixture value type {value.GetType().Name}.", nameof(kvPairs));
            }
        }

        long offset = 0;
        foreach (TensorDef tensor in tensors)
        {
            int[] stored = new int[tensor.Dims.Length]; // GGUF stores dims fastest-varying first: the logical row-major order is written reversed
            for (int d = 0; d < tensor.Dims.Length; d++)
            {
                stored[d] = tensor.Dims[tensor.Dims.Length - 1 - d];
            }

            WriteGgufString(writer, tensor.Name);
            writer.Write((uint)stored.Length);
            foreach (int dim in stored)
            {
                writer.Write((ulong)dim);
            }

            writer.Write(0u); // ggml type F32
            writer.Write((ulong)offset);

            long elements = 1;
            foreach (int dim in tensor.Dims)
            {
                elements *= dim;
            }

            offset += elements * 4;
        }

        long headerLength = stream.Position;
        long dataStart = (headerLength + 31) / 32 * 32; // GGUF default alignment is 32
        for (long i = headerLength; i < dataStart; i++)
        {
            writer.Write((byte)0);
        }

        foreach (TensorDef tensor in tensors)
        {
            long elements = 1;
            foreach (int dim in tensor.Dims)
            {
                elements *= dim;
            }

            if (tensor.Payload.Length != elements)
            {
                throw new ArgumentException($"tensor '{tensor.Name}' payload has {tensor.Payload.Length} entries, expected {elements}.", nameof(tensors));
            }

            foreach (float value in tensor.Payload)
            {
                writer.Write(value);
            }
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private static void AssertBitsEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        ReadOnlySpan<byte> expectedBytes = MemoryMarshal.AsBytes(expected);
        ReadOnlySpan<byte> actualBytes = MemoryMarshal.AsBytes(actual);

        for (int i = 0; i < expectedBytes.Length; i++)
        {
            Assert.True(
                expectedBytes[i] == actualBytes[i],
                $"{context}: first differing byte at {i} (float {i / 4}): expected 0x{expectedBytes[i]:X2}, got 0x{actualBytes[i]:X2}");
        }
    }
}
