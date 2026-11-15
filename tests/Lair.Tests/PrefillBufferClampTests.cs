using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Http;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;
using Microsoft.Extensions.Logging;

namespace Lair.Tests;

/// <summary>
/// Prefill working-buffer clamp: a model's resolved prefill-buffer ceilings narrow the row window its engine runs, and the narrowing reads
/// apart from the width a step-time budget chose — requested width against allowed width beside the binding ceiling and one sub-batch row's
/// activation cost. A chain that caps nothing, or caps positions its engine already fits, clamps nothing at all: chunk widths, sub-batch rows,
/// and token streams stay exactly what an engine carrying no budgets produces. Where a ceiling binds, the narrowed window splits the same tokens
/// into narrower sub-batches and moves nothing, and a chain that cannot seat even one row's bytes reports its exhaustion naming the model, the
/// resource, the tier, and the breakdown rather than running over the ceiling quietly.
/// </summary>
[Trait("Category", "Prefill")]
[Trait("Category", "Budget")]
public sealed class PrefillBufferClampTests : IDisposable
{
    // A row cost chosen so the caps below bind: 1 MiB of ceiling allows 32 rows of this cost, so a 64-row request fits a 2 MiB cap and narrows under a 1 MiB one.
    private const long RowBytes = 32 * 1024;

    private const int RequestedRows = 64;

    // The pinned thinking fixture whose sub-batch rows cost tens of kilobytes: a 1 MiB prefill-buffer ceiling genuinely narrows its default window.
    private const string QwenFixtureRelativePath = "models/Qwen3-0.6B-Q8_0.gguf";

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

    // ---------------------------------------------------------------- clamp resolution

    [Fact]
    public void Clamp_UncappedChainAndFittingCap_ClampNothing_AndASmallerCapNarrowsToRowsItsCeilingAllows()
    {
        ResolvedBudgetSet uncapped = Resolve([new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram] }]);
        Assert.Null(PrefillBufferClamp.Compute(uncapped.For(BudgetResource.PrefillBuffers)!, RowBytes, RequestedRows)); // a declared position stating no ceiling bounds no window

        ResolvedBudgetSet fitting = Resolve([new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 2 } }]);
        Assert.Null(PrefillBufferClamp.Compute(fitting.For(BudgetResource.PrefillBuffers)!, RowBytes, RequestedRows)); // 2 MiB allows 64 rows of this cost — the request already fits, so nothing narrows

        ResolvedBudgetSet binding = Resolve([new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1 } }]);
        PrefillRowWidthClamp clamp = PrefillBufferClamp.Compute(binding.For(BudgetResource.PrefillBuffers)!, RowBytes, RequestedRows)!;
        Assert.Equal(BudgetTier.Dram, clamp.Tier);
        Assert.Equal(1L, clamp.CapMegabytes);
        Assert.Equal(RowBytes, clamp.RowActivationBytes);
        Assert.Equal(RequestedRows, clamp.RequestedRowWidth); // requested stays stated beside allowed — the two are distinct numbers
        Assert.Equal(32, clamp.AllowedRowWidth);             // 1 MiB against a 32 KiB row cost allows 32 rows
        Assert.True(clamp.AllowsPositiveWidth);
    }

    [Fact]
    public void Clamp_MultipleCappedPositions_BindsAtTheOneAllowingTheFewestRows()
    {
        ResolvedBudgetSet chain = Resolve(
        [
            new BudgetChain
            {
                Resource = BudgetResource.PrefillBuffers,
                Order = [BudgetTier.Dram, BudgetTier.Vram],
                Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 8, [BudgetTier.Vram] = 1 },
            },
        ],
        openedPoolTiers: [BudgetTier.Dram, BudgetTier.Vram]);

        PrefillRowWidthClamp clamp = PrefillBufferClamp.Compute(chain.For(BudgetResource.PrefillBuffers)!, RowBytes, RequestedRows)!;
        Assert.Equal(BudgetTier.Vram, clamp.Tier); // 8 MiB allows 256 rows and 1 MiB allows 32 — the narrower position binds
        Assert.Equal(32, clamp.AllowedRowWidth);
        Assert.True(clamp.AllowedRowWidth < RequestedRows); // the binding cap, not the roomier sibling, decided the window
        Assert.Contains("'vram' caps prefill working buffers at 1 megabytes", clamp.Describe());
    }

    [Fact]
    public void Clamp_CapSmallerThanOneRow_ReportsAnUnfitWindowInsteadOfSilentlyRunningOverItsCeiling()
    {
        ResolvedBudgetSet chain = Resolve([new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1 } }]);

        PrefillRowWidthClamp clamp = PrefillBufferClamp.Compute(chain.For(BudgetResource.PrefillBuffers)!, 2L * 1024 * 1024, RequestedRows)!; // a row costing 2 MiB cannot fit the 1 MiB ceiling at all
        Assert.Equal(0, clamp.AllowedRowWidth);
        Assert.False(clamp.AllowsPositiveWidth);
        Assert.Contains("'dram' caps prefill working buffers at 1 megabytes", clamp.Describe());
    }

    // ---------------------------------------------------------------- engine-side clamp and identity

    [Fact]
    public void Engine_ResolvesEffectiveRowWidthEqualToRequested_WhenNoChainOrAFittingCapBounds_It()
    {
        string path = SyntheticModel("clamp-engine-unclamped");

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend backend = new(threads: 1);

        using TransformerModel plain = TransformerModel.Open(gguf, backend, maxContextTokens: 512);
        Assert.Equal(plain.Plan.PrefillRowWidth, plain.EffectivePrefillRowWidth); // an engine carrying no budgets runs at the resolved width, clamp absent
        Assert.Null(plain.PrefillRowClamp);
        Assert.True(plain.PrefillRowActivationBytes > 0); // every engine that runs prompt windows can state one row's cost

        ModelRoute route = plain.Plan.Route;
        ResolvedBudgetSet fitting = Resolve([new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1 } }]);
        using TransformerModel cappedButFitting = TransformerModel.Open(gguf, backend, maxContextTokens: 512, logLine: null, prefillKernel: PrefillKernel.GemV, route: route, budgets: fitting);
        Assert.Equal(cappedButFitting.Plan.PrefillRowWidth, cappedButFitting.EffectivePrefillRowWidth); // 1 MiB fits this synthetic engine's whole window cost — nothing narrows
        Assert.Null(cappedButFitting.PrefillRowClamp);
    }

    [Fact]
    public void Engine_ClampsRowWindowToDeclaredCeiling_NarrowsSubBatches_AndKeepsTokensIdentical()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(QwenFixtureRelativePath);
        if (modelPath is null)
        {
            return; // real-model clamp proof skips when the pinned fixture is absent
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        using CpuBackend backend = new(threads: 1);

        using TransformerModel unclamped = TransformerModel.Open(gguf, backend, maxContextTokens: 1024);
        Assert.Null(unclamped.PrefillRowClamp);

        // One megabyte against this model's per-row cost: the ceiling is real for a fixture whose rows cost tens of kilobytes, so the window must narrow below the resolved count.
        ResolvedBudgetSet budgets = Resolve([new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1 } }]);
        using TransformerModel clamped = TransformerModel.Open(gguf, backend, maxContextTokens: 1024, logLine: null, prefillKernel: PrefillKernel.GemV, route: unclamped.Plan.Route, budgets: budgets);

        Assert.NotNull(clamped.PrefillRowClamp);
        Assert.True(clamped.PrefillRowClamp!.AllowsPositiveWidth, $"a 1 MiB ceiling leaves no room for one row of {clamped.PrefillRowActivationBytes} bytes on this fixture");
        Assert.True(clamped.EffectivePrefillRowWidth < clamped.ResolvedPrefillRowWidth,
            $"effective width {clamped.EffectivePrefillRowWidth} did not narrow the resolved {clamped.ResolvedPrefillRowWidth} under a 1 MiB ceiling");
        Assert.Equal(clamped.PrefillRowClamp.AllowedRowWidth, clamped.EffectivePrefillRowWidth);
        Assert.Contains("caps prefill working buffers at 1 megabytes", clamped.PrefillRowClamp.Describe());
        Assert.Equal(clamped.Plan.PrefillRowWidth, clamped.PrefillRowClamp.RequestedRowWidth); // the plan still states what was requested, distinct from what runs

        // The narrowed window forwards the same positions the unclamped engine would: identical logits bytes across a window that spans several sub-batches either way.
        int[] prompt = BuildPrompt(96);
        using BlockPool pool = new(blockTokens: 16, layerCount: clamped.Plan.LayerCount, kvHeads: clamped.Plan.KvHeadCount, headDim: clamped.Plan.HeadDim, capacityBlocks: 32, backend);

        using SequenceKv wideKv = new(pool);
        float[] wideLogits = [.. unclamped.Prefill(prompt, wideKv)];
        using SequenceKv narrowKv = new(pool);
        float[] narrowLogits = [.. clamped.Prefill(prompt, narrowKv)];

        Assert.Equal(wideLogits.Length, narrowLogits.Length);
        Assert.Equal(ArgMax(wideLogits), ArgMax(narrowLogits));

        for (int layer = 0; layer < clamped.Plan.LayerCount; layer++)
        {
            for (int position = 0; position < prompt.Length; position++)
            {
                Assert.Equal(ReadBytes(wideKv.ReadK(position, layer)), ReadBytes(narrowKv.ReadK(position, layer)));
                Assert.Equal(ReadBytes(wideKv.ReadV(position, layer)), ReadBytes(narrowKv.ReadV(position, layer)));
            }
        }

        backend.Dispose(); // disposed before the fixture file so a Windows mmap view is never left open on it
    }

    // ---------------------------------------------------------------- scheduler diagnostics: requested versus effective

    [Fact]
    public async Task Scheduler_NarrowRowWindow_ReadsApartFromChunkWidth_AndUnboundedRunsStayAtTodayShape()
    {
        string path = SyntheticModel("clamp-scheduler-shape");

        int[] prompt = BuildPrompt(240);

        // Requested versus effective, distinguishable: a window narrowed below the configured chunk ceiling still forwards whole chunks and reports rows capped at its own width.
        using NarrowHarness narrow = new(path, maxBatchTokens: 64, prefillRowWidth: 8);
        SequenceOutcome narrowed = await narrow.Scheduler.Submit(Request(prompt, 8), CancellationToken.None).Completion;

        Assert.Equal(8, narrow.Scheduler.EffectivePrefillRowWidth);
        Assert.Null(narrow.Scheduler.ResolvedPrefillRowClamp); // a narrow resolved width with no declared ceiling is not a clamp — diagnostics keep the two apart
        Assert.Equal(0, narrow.Scheduler.Stats.ClampedPrefillChunksIssued);
        Assert.Equal(prompt.Length / 64 + 1, narrow.Scheduler.Stats.PrefillChunksIssued); // whole 64-token chunks and a remainder, exactly as an unbounded engine issued them
        Assert.Equal(8, narrow.Scheduler.LastPrefillSubBatchRows);
        Assert.True(narrow.Scheduler.LastPrefillChosenTokens > narrow.Scheduler.LastPrefillSubBatchRows, "a whole-chunk boundary never ran more sub-batch rows than its chosen width allowed");
        Assert.Equal(8, narrow.Scheduler.Stats.PeakPrefillSubBatchRows);
        Assert.True(narrow.Scheduler.Stats.PrefillSubBatchSum > narrow.Scheduler.Stats.PrefillChunksIssued);

        // The same prompt on the default-width engine: identical chunk shape and identical tokens — a narrower window moved nothing.
        using BaselineHarness baseline = new(path, maxBatchTokens: 64);
        SequenceOutcome plain = await baseline.Scheduler.Submit(Request(prompt, 8), CancellationToken.None).Completion;

        Assert.Equal(baseline.Model.ResolvedPrefillRowWidth, baseline.Scheduler.EffectivePrefillRowWidth);
        Assert.Null(baseline.Scheduler.ResolvedPrefillRowClamp);
        Assert.Equal(0, baseline.Scheduler.Stats.ClampedPrefillChunksIssued);
        Assert.Equal(narrow.Scheduler.Stats.PrefillChunksIssued, baseline.Scheduler.Stats.PrefillChunksIssued);
        Assert.Equal(prompt.Length, baseline.Scheduler.Stats.PrefillTokensForwarded);
        Assert.Equal([.. plain.GeneratedTokens], [.. narrowed.GeneratedTokens]);

        // A declared ceiling this engine's whole window cost fits clamps nothing: budgets present, shape unchanged, clamped counter still silent.
        using BudgetedHarness budgeted = new(path, maxBatchTokens: 64);
        SequenceOutcome capped = await budgeted.Scheduler.Submit(Request(prompt, 8), CancellationToken.None).Completion;

        Assert.Null(budgeted.Model.PrefillRowClamp); // 1 MiB bounds nothing this synthetic engine runs
        Assert.Null(budgeted.Scheduler.ResolvedPrefillRowClamp);
        Assert.Equal(0, budgeted.Scheduler.Stats.ClampedPrefillChunksIssued);
        Assert.Equal(baseline.Scheduler.Stats.PrefillChunksIssued, budgeted.Scheduler.Stats.PrefillChunksIssued);
        Assert.Equal([.. plain.GeneratedTokens], [.. capped.GeneratedTokens]);
    }

    [Fact]
    public async Task Scheduler_ClampedEngine_CountsClampedChunksApart_AndKeepsTokenStreamsIdentical()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(QwenFixtureRelativePath);
        if (modelPath is null)
        {
            return; // the clamped-chunk accounting proof needs the pinned fixture whose rows a 1 MiB ceiling binds
        }

        int[] prompt = BuildPrompt(96);

        using GgufFile gguf = GgufFile.Open(modelPath);
        using CpuBackend backend = new(threads: 1);
        using TransformerModel unclamped = TransformerModel.Open(gguf, backend, maxContextTokens: 1024);
        using TransformerModel clamped = TransformerModel.Open(gguf, backend, maxContextTokens: 1024, logLine: null, prefillKernel: PrefillKernel.GemV, route: unclamped.Plan.Route, budgets: Resolve([new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1 } }]));

        try
        {
            using BlockPool pool = new(blockTokens: 16, layerCount: clamped.Plan.LayerCount, kvHeads: clamped.Plan.KvHeadCount, headDim: clamped.Plan.HeadDim, capacityBlocks: 32, backend);

            SequenceOutcome plainRun;
            long plainChunks;
            long plainSubBatches;
            {
                using GenerationScheduler plain = new(unclamped, pool, maxBatchTokens: 64, cache: null);
                plainRun = await plain.Submit(Request(prompt, 8), CancellationToken.None).Completion;
                plainChunks = plain.Stats.PrefillChunksIssued;
                plainSubBatches = plain.Stats.PrefillSubBatchSum;
                Assert.Equal(0, plain.Stats.ClampedPrefillChunksIssued);
            }

            using GenerationScheduler scheduler = new(clamped, pool, maxBatchTokens: 64, cache: null);
            SequenceOutcome clampedRun = await scheduler.Submit(Request(prompt, 8), CancellationToken.None).Completion;

            Assert.NotNull(scheduler.ResolvedPrefillRowClamp);
            Assert.True(scheduler.EffectivePrefillRowWidth < unclamped.ResolvedPrefillRowWidth);

            // Chunk width selection never saw the clamp: the same boundaries were chosen, and every one of them ran under the ceiling-bound window.
            Assert.Equal(plainChunks, scheduler.Stats.PrefillChunksIssued);
            Assert.Equal(scheduler.Stats.PrefillChunksIssued, scheduler.Stats.ClampedPrefillChunksIssued);
            Assert.Equal(prompt.Length, scheduler.Stats.PrefillTokensForwarded);
            Assert.InRange(scheduler.Stats.PeakPrefillSubBatchRows, 1, scheduler.EffectivePrefillRowWidth);
            Assert.True(scheduler.Stats.PrefillSubBatchSum > plainSubBatches, "a clamped window never split the same chunks into narrower sub-batches");

            Assert.Equal([.. plainRun.GeneratedTokens], [.. clampedRun.GeneratedTokens]); // clamped versus unclamped: the same tokens, whatever the window narrowed
        }
        finally
        {
            backend.Dispose(); // disposed before the fixture file so a Windows mmap view is never left open on it
        }
    }

    // ---------------------------------------------------------------- exhaustion through the ledger

    [Fact]
    public void Open_PrefillChainWhoseEveryPositionHolds_ReportsExhaustionNamingModelResourceTierAndBreakdown()
    {
        string path = SyntheticModel("clamp-exhaust-open");
        CapturingLogger logger = new();

        // The neighbour seats the whole tier first; this engine's own working-set claim then has nowhere its chain seats it.
        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.PrefillBuffers, Capacity = 1 }]);
        ClaimOutcome neighbour = ledger.RegisterClaim("clamp-neighbour", Resolve([new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram] }]).For(BudgetResource.PrefillBuffers)!, 1);
        Assert.True(neighbour.Seated);

        BudgetExhaustedException loadFailure = Assert.Throws<BudgetExhaustedException>(
            () => GenerationService.Open(ConfigWith(path, "clamp-exhaust", budgets: [new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1 } }]), logger, ledger));

        Assert.Equal("clamp-exhaust", loadFailure.Model);
        Assert.Equal(BudgetResource.PrefillBuffers, loadFailure.Resource);
        Assert.Equal(BudgetTier.Dram, loadFailure.Tier);
        Assert.Contains("prefill_buffers", loadFailure.Message);
        Assert.Contains("'clamp-neighbour' 1 megabytes", loadFailure.Breakdown); // the breakdown names what stands at the tier against the claim nothing could seat
    }

    [Fact]
    public void Admission_PrefillWorkingSetNoLongerSeatable_RefusesWithBudgetCode_WithoutDisplacingStandingClaims()
    {
        string path = SyntheticModel("clamp-exhaust-admission");

        // Both engines seat their working sets against a two-megabyte tier; the host then re-reports a total no landing fits beside what stands there.
        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.PrefillBuffers, Capacity = 2 }]);

        using GenerationService first = GenerationService.Open(ConfigWith(path, "clamp-admit-a", budgets: [new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 2 } }]), null, ledger);
        using GenerationService second = GenerationService.Open(ConfigWith(path, "clamp-admit-b", budgets: [new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 2 } }]), null, ledger);

        Assert.Equal(2L, ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.PrefillBuffers).Count);

        GenerationResult admitted = Generate(first, "hello world", maxTokens: 4);
        Assert.True(admitted.CompletionTokens > 0); // admitted while the tier still held room beside the standing seats

        ledger.UpdateCapacity(BudgetTier.Dram, BudgetResource.PrefillBuffers, capacity: 1); // nothing seated is revoked; only later landings tighten

        EngineBusyException refusal = Assert.Throws<EngineBusyException>(() => Generate(second, "hello world", maxTokens: 4));
        Assert.Equal("budget_exhausted", refusal.Code);
        Assert.Contains("clamp-admit-b", refusal.Message);
        Assert.Contains("prefill_buffers", refusal.Message);

        EngineBusyException sameEngineToo = Assert.Throws<EngineBusyException>(() => Generate(first, "hello world", maxTokens: 4)); // the tier now reports no room beside what stands there for either model's next landing
        Assert.Equal("budget_exhausted", sameEngineToo.Code);

        IReadOnlyList<DeviceLedgerClaim> standing = ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.PrefillBuffers);
        Assert.Equal(2, standing.Count); // the refusals displaced nothing: both working sets stand where they seated
    }

    [Fact]
    public void Open_PrefillClaim_StatesTheWorkingSetAndRowWidths_BesideLaneBytes()
    {
        string path = SyntheticModel("clamp-claim-report");
        CapturingLogger logger = new();

        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.PrefillBuffers, Capacity = null }]);

        using GenerationService engine = GenerationService.Open(
            ConfigWith(path, "clamp-claim", budgets: [new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 4 } }]),
            logger,
            ledger);

        DeviceLedgerClaim claim = Assert.Single(ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.PrefillBuffers));
        Assert.Equal(1L, claim.Amount); // the synthetic engine's window cost rounds up to one megabyte of claim — the posture the ledger holds against the chain

        string[] claimLines = [.. logger.Messages(LogLevel.Information).Where(line => line.Contains("ledger_claim", StringComparison.Ordinal) && line.Contains("prefill_buffers", StringComparison.Ordinal))];
        string claimLine = Assert.Single(claimLines);
        Assert.Contains("row_width=", claimLine);
        Assert.Contains("effective_row_width=", claimLine);
        Assert.Contains("activation_bytes=", claimLine); // lane/working bytes reported beside the row widths the claim covers
    }

    // ---------------------------------------------------------------- harnesses and helpers

    private static ResolvedBudgetSet Resolve(IReadOnlyList<BudgetChain> declared, IReadOnlyList<BudgetTier>? openedPoolTiers = null) =>
        ModelBudgets.Resolve(new BudgetRequest
        {
            ModelName = "clamp-test",
            Declared = declared,
            OpenedPoolTiers = openedPoolTiers ?? [BudgetTier.Dram],
        });

    private string SyntheticModel(string modelName)
    {
        string path = SyntheticLlamaGguf.Write(modelName);
        _tempFiles.Add(path);
        return path;
    }

    private static LairConfig ConfigWith(string modelPath, string modelName, IReadOnlyList<BudgetChain> budgets) => new()
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
                KvBlocks = 32,
                Budgets = budgets,
            },
        ],
    };

    private static GenerationResult Generate(GenerationService engine, string prompt, int maxTokens) =>
        engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = prompt,
            MaxTokens = maxTokens,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult();

    private static ScheduledRequest Request(int[] promptTokens, int maxTokens) => new()
    {
        PromptTokens = promptTokens,
        MaxTokens = maxTokens,
        Sampling = new SamplingParams { Temperature = 0f }, // greedy: a window shape change could move nothing at all
        EosId = 999,
    };

    private static int[] BuildPrompt(int tokens)
    {
        int[] cycle = [1, 2, 3]; // hello / world / ! — ordinary entries of the synthetic vocabulary

        List<int> ids = [];
        while (ids.Count < tokens)
        {
            ids.AddRange(cycle);
        }

        return [.. ids.Take(tokens)];
    }

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        float best = float.NegativeInfinity;
        int bestIdx = -1;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > best)
            {
                best = logits[i];
                bestIdx = i;
            }
        }

        Assert.True(bestIdx >= 0, "logit vector had no selectable entry");
        return bestIdx;
    }

    private static IReadOnlyList<byte> ReadBytes(ReadOnlySpan<float> values) => [.. MemoryMarshal.AsBytes(values)];

    private sealed class BaselineHarness : IDisposable
    {
        public GgufFile Gguf { get; }
        public CpuBackend Backend { get; }
        public TransformerModel Model { get; }
        public BlockPool Pool { get; }
        public GenerationScheduler Scheduler { get; }

        public BaselineHarness(string modelPath, int maxBatchTokens)
        {
            Gguf = GgufFile.Open(modelPath);
            Backend = new CpuBackend(threads: 1);
            Model = TransformerModel.Open(Gguf, Backend, maxContextTokens: 512);
            Pool = new BlockPool(blockTokens: 4, layerCount: Model.Desc.LayerCount, kvHeads: Model.Desc.KvHeadCount, headDim: Model.Desc.HeadDim, capacityBlocks: 96, Backend);
            Scheduler = new GenerationScheduler(Model, Pool, maxBatchTokens, cache: null);
        }

        public void Dispose()
        {
            Scheduler.Dispose();
            Pool.Dispose();
            Model.Dispose();
            Gguf.Dispose();
            Backend.Dispose();
        }
    }

    /// <summary>The same engine opened through the assembly-internal row-width seam at a narrower resolved count — the shape a clamp leaves behind, without a declared ceiling.</summary>
    private sealed class NarrowHarness : IDisposable
    {
        public GgufFile Gguf { get; }
        public CpuBackend Backend { get; }
        public TransformerModel Model { get; }
        public BlockPool Pool { get; }
        public GenerationScheduler Scheduler { get; }

        public NarrowHarness(string modelPath, int maxBatchTokens, int prefillRowWidth)
        {
            Gguf = GgufFile.Open(modelPath);
            Backend = new CpuBackend(threads: 1);
            Model = TransformerModel.Open(Gguf, Backend, maxContextTokens: 512, logLine: null, prefillKernel: PrefillKernel.GemV, prefillRowWidth);
            Pool = new BlockPool(blockTokens: 4, layerCount: Model.Desc.LayerCount, kvHeads: Model.Desc.KvHeadCount, headDim: Model.Desc.HeadDim, capacityBlocks: 96, Backend);
            Scheduler = new GenerationScheduler(Model, Pool, maxBatchTokens, cache: null);
        }

        public void Dispose()
        {
            Scheduler.Dispose();
            Pool.Dispose();
            Model.Dispose();
            Gguf.Dispose();
            Backend.Dispose();
        }
    }

    /// <summary>An engine whose model declares a prefill-buffer ceiling this synthetic window cost fits: budgets present, clamp absent, scheduling shape unchanged.</summary>
    private sealed class BudgetedHarness : IDisposable
    {
        public GgufFile Gguf { get; }
        public CpuBackend Backend { get; }
        public TransformerModel Model { get; }
        public BlockPool Pool { get; }
        public GenerationScheduler Scheduler { get; }

        public BudgetedHarness(string modelPath, int maxBatchTokens)
        {
            Gguf = GgufFile.Open(modelPath);
            Backend = new CpuBackend(threads: 1);
            ResolvedBudgetSet budgets = ModelBudgets.Resolve(new BudgetRequest
            {
                ModelName = "clamp-budgeted",
                Declared = [new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1 } }],
                OpenedPoolTiers = [BudgetTier.Dram],
            });
            Model = TransformerModel.Open(Gguf, Backend, maxContextTokens: 512, logLine: null, prefillKernel: PrefillKernel.GemV, route: ModelPlans.Resolve(Gguf).Route, budgets: budgets);
            Pool = new BlockPool(blockTokens: 4, layerCount: Model.Desc.LayerCount, kvHeads: Model.Desc.KvHeadCount, headDim: Model.Desc.HeadDim, capacityBlocks: 96, Backend);
            Scheduler = new GenerationScheduler(Model, Pool, maxBatchTokens, cache: null);
        }

        public void Dispose()
        {
            Scheduler.Dispose();
            Pool.Dispose();
            Model.Dispose();
            Gguf.Dispose();
            Backend.Dispose();
        }
    }

    private sealed class CapturingLogger : ILogger
    {
        private readonly List<(LogLevel Level, string Message)> _entries = [];

        public IReadOnlyList<string> Messages(LogLevel level) =>
            [.. _entries.Where(entry => entry.Level == level).Select(entry => entry.Message)];

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            ArgumentNullException.ThrowIfNull(formatter);
            _entries.Add((logLevel, formatter(state, exception)));
        }
    }
}
