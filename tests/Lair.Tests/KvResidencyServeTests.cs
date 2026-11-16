using Lair.Contracts;
using Lair.Host.Config;
using Lair.Http;
using Lair.Kv;
using Lair.Models;
using Lair.Models.Scheduling;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// The KV residency axis at the serve seam: a posture-neutral or host-resident declaration opens pools that hold every block in the arena and step over system memory only, an accelerator-resident one is admitted against the finite ceiling its budgets declare and refuses before any device byte exists when that ceiling is missing or too small, and a resolved device-backed reservation keeps its whole weight set host-resident while its blocks carry persistent device copies — serving token-identical text to the same file opened host-only. What an opened engine reports is what its pool actually holds, never what placement declared it as.
/// </summary>
[Trait("Category", "Kv")]
public sealed class KvResidencyServeTests : IDisposable
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
    public void AbsentAndAutoPlacement_RunHostResidentPools_AndStepOverSystemMemoryOnly()
    {
        string path = SyntheticModel("kv-residency-host");

        using GenerationService absent = GenerationService.Open(Config(path, "kv-absent", kvBlocks: 8), NullLogger.Instance);
        using GenerationService auto = GenerationService.Open(
            Config(path, "kv-auto", kvBlocks: 8, placement: new PlacementConfig { KvResidency = BlockResidencyModeNames.Auto }),
            NullLogger.Instance);

        Assert.Equal(BlockResidencyMode.Dram, absent.ServedBlockPool.Residency);
        Assert.Equal(0, absent.ServedBlockPool.DeviceBackedBlockCount);
        Assert.Equal(0L, absent.ServedBlockPool.DeviceMirrorBytes); // a host-only reservation opens no mirror at all

        Assert.Equal(BlockResidencyMode.Dram, auto.ServedBlockPool.Residency);
        Assert.Equal(0, auto.ServedBlockPool.DeviceBackedBlockCount);

        Assert.Single(absent.ResolvedRoute.Segments);
        Assert.Equal([WeightTier.Dram], absent.ResolvedRoute.Segments[0].BlockResidencyTiers); // the posture-neutral declaration resolves to host-resident blocks, and its segments step over exactly that
        Assert.Equal([WeightTier.Dram], auto.ResolvedRoute.Segments[0].BlockResidencyTiers);

        Assert.Equal(Generate(absent, "hello world"), Generate(auto, "hello world")); // the same engine posture under both declarations
    }

    [Theory]
    [InlineData("vram")]
    [InlineData("split")]
    public void AcceleratorResidencyWithoutDeclaredCeiling_RefusesNamingModelResourceTierAndKey(string residency)
    {
        string path = SyntheticModel("kv-residency-no-ceiling"); // the refusal lands before any backend seam opens, so no device is consulted here.

        ModelException error = Assert.Throws<ModelException>(() => GenerationService.Open(
            Config(path, "kv-no-ceiling", kvBlocks: 8, placement: new PlacementConfig { KvResidency = residency }),
            NullLogger.Instance));

        Assert.Contains("model 'kv-no-ceiling'", error.Message, StringComparison.Ordinal);
        Assert.Contains($"'placement.kv_residency' '{residency}'", error.Message, StringComparison.Ordinal);
        Assert.Contains("resource 'kv'", error.Message, StringComparison.Ordinal);
        Assert.Contains("tier 'vram'", error.Message, StringComparison.Ordinal);
        Assert.Contains("models[].budgets.kv.caps.vram_blocks", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    [Trait("Category", "Cuda")]
    public void VramReservationOverItsDeclaredCeiling_RefusesAtTheAcceleratorTierWithLedgerBreakdown()
    {
        string path = SyntheticModel("kv-residency-over-cap");

        // The reservation is eight blocks and the chain bounds device-held blocks at four: an accelerator-resident declaration covers every id its pool hands out, so nothing shrinks to fit and the open refuses with the breakdown every exhausted chain reports — before a single block id exists on the device.
        ModelConfig model = ConfigModel("kv-over-cap", path, kvBlocks: 8, residency: "vram");

        try
        {
            using GenerationService engine = GenerationService.Open(ConfigFrom(model), NullLogger.Instance);

            Assert.Fail($"expected an exhausted KV chain at the accelerator tier, but '{engine.Model.Id}' opened with {engine.ServedBlockPool.DeviceBackedBlockCount} device-backed blocks.");
        }
        catch (BudgetExhaustedException ex)
        {
            Assert.Equal("kv-over-cap", ex.Model);
            Assert.Equal(BudgetResource.Kv, ex.Resource);
            Assert.Equal(BudgetTier.Vram, ex.Tier); // the tier its declared ceiling bounds device-held blocks at is where the chain ran out
            Assert.Contains("'vram'", ex.Message, StringComparison.Ordinal);
            Assert.NotEmpty(ex.Breakdown.Trim());
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains(ModelRoute.CudaBackendSeam))
        {
            throw new InvalidOperationException(
                "CUDA driver seam unavailable: a device-backed KV residency needs the backend that holds persistent block copies, and this host opened none.",
                ex);
        }
    }

    [Fact]
    [Trait("Category", "Cuda")]
    public void VramReservationWithinItsCeiling_BackEveryBlockKeepWeightsHostResident_AndServeIdenticalTextToHostOnly()
    {
        string path = SyntheticModel("kv-residency-vram");

        using GenerationService baseline = GenerationService.Open(Config(path, "kv-vram-baseline", kvBlocks: 8), NullLogger.Instance);
        string hostText = Generate(baseline, "hello world");

        CapturingLogger logger = new();

        try
        {
            using GenerationService deviceBacked = GenerationService.Open(
                Config(
                    path,
                    "kv-vram",
                    kvBlocks: 8,
                    placement: new PlacementConfig { KvResidency = "vram" },
                    budgets: [new BudgetChain
                    {
                        Resource = BudgetResource.Kv,
                        Order = [BudgetTier.Dram, BudgetTier.Vram],
                        Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Vram] = 8L },
                    }]),
                logger);

            BlockPool pool = deviceBacked.ServedBlockPool;

            Assert.Equal(BlockResidencyMode.Vram, pool.Residency); // what the opened pool holds is what this engine reports
            Assert.Equal(pool.CapacityBlocks, pool.DeviceBackedBlockCount); // an accelerator-resident reservation covers every id it hands out
            Assert.Equal((long)pool.CapacityBlocks * pool.BytesPerBlock, pool.DeviceMirrorBytes);

            Assert.Single(deviceBacked.ResolvedRoute.Segments);
            Assert.Equal(ModelRoute.CpuDeviceId, deviceBacked.ResolvedRoute.Segments[0].DeviceId); // compute — and every attention read — stays on the host backend that owns this arena
            Assert.Equal(WeightTier.Dram, deviceBacked.ResolvedRoute.Segments[0].WeightTier); // weights are placed independently of blocks: nothing here moved a weight to the device
            Assert.Equal([WeightTier.Vram], deviceBacked.ResolvedRoute.Segments[0].BlockResidencyTiers);

            IReadOnlyList<string> claims = logger.Messages(LogLevel.Information)
                .Where(message => message.Contains("ledger_claim", StringComparison.Ordinal))
                .ToList();

            Assert.Contains(claims, message => message.Contains("resource='kv' tier='dram'", StringComparison.Ordinal) && message.Contains("amount_blocks=8", StringComparison.Ordinal)); // the whole reservation seats where it was always seated
            Assert.Contains(claims, message => message.Contains("resource='kv' tier='vram'", StringComparison.Ordinal) && message.Contains("amount_blocks=8", StringComparison.Ordinal)); // and its device-held copy is a standing claim of its own against the declared ceiling

            string servedText = Generate(deviceBacked, "hello world");
            Assert.Equal(hostText, servedText); // residency changes where a block's bytes are kept, never what this engine serves from them

            // The candidate that advanced carried the residency its own blocks hold — inside this pool's device-backed range — so it stepped on the segment its route names instead of being held out for a residency mismatch.
            Assert.DoesNotContain(deviceBacked.ServedScheduler.LastStepComposition.Deferrals, deferral => deferral.Block == CompositionBlock.KvResidency);

            GenerateConcurrently(deviceBacked, 3); // several sequences whose blocks sit in the device-backed range compose together on that segment rather than each stepping alone.
            Assert.DoesNotContain(deviceBacked.ServedScheduler.LastStepComposition.Deferrals, deferral => deferral.Block == CompositionBlock.KvResidency);

            pool.FlushDeviceMirror();
            Assert.True(pool.DeviceMirrorCrossedBytes > 0L, $"serving {pool.CapacityBlocks} blocks across an accelerator-resident reservation crossed no device byte");
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains(ModelRoute.CudaBackendSeam))
        {
            throw new InvalidOperationException(
                "CUDA driver seam unavailable: a device-backed KV residency needs the backend that holds persistent block copies, and this host opened none.",
                ex);
        }
    }

    [Fact]
    [Trait("Category", "Cuda")]
    public void SplitCeilingNarrowsItsDefaultRange_AndHostOnlyTailKeepsReadingFromTheArena()
    {
        string path = SyntheticModel("kv-residency-split");

        using GenerationService baseline = GenerationService.Open(Config(path, "kv-split-baseline", kvBlocks: 8), NullLogger.Instance);
        string hostText = Generate(baseline, "hello world");

        try
        {
            // A ceiling wide enough to cover the divided default leaves that default standing: half of the reservation's ids carry a persistent device copy, in allocation order.
            using GenerationService halved = GenerationService.Open(
                Config(
                    path,
                    "kv-split-default",
                    kvBlocks: 8,
                    placement: new PlacementConfig { KvResidency = "split" },
                    budgets: [new BudgetChain
                    {
                        Resource = BudgetResource.Kv,
                        Order = [BudgetTier.Dram, BudgetTier.Vram],
                        Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Vram] = 8L },
                    }]),
                NullLogger.Instance);

            Assert.Equal(BlockResidencyMode.Split, halved.ServedBlockPool.Residency);
            Assert.Equal(4, halved.ServedBlockPool.DeviceBackedBlockCount); // the leading half of allocation order
            Assert.True(halved.ServedBlockPool.IsDeviceBacked(0));
            Assert.False(halved.ServedBlockPool.IsDeviceBacked(7)); // and its tail stays host-only

            Assert.Equal([WeightTier.Dram, WeightTier.Vram], halved.ResolvedRoute.Segments[0].BlockResidencyTiers); // a divided reservation is steppable at either tier
            Assert.Equal(hostText, Generate(halved, "hello world"));

            // A narrower ceiling narrows that range and never widens it: the declared count bounds what crosses, not what divides.
            using GenerationService narrowed = GenerationService.Open(
                Config(
                    path,
                    "kv-split-narrowed",
                    kvBlocks: 8,
                    placement: new PlacementConfig { KvResidency = "split" },
                    budgets: [new BudgetChain
                    {
                        Resource = BudgetResource.Kv,
                        Order = [BudgetTier.Dram, BudgetTier.Vram],
                        Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Vram] = 2L },
                    }]),
                NullLogger.Instance);

            Assert.Equal(2, narrowed.ServedBlockPool.DeviceBackedBlockCount);
            Assert.True(narrowed.ServedBlockPool.IsDeviceBacked(1));
            Assert.False(narrowed.ServedBlockPool.IsDeviceBacked(2));
            Assert.Equal(hostText, Generate(narrowed, "hello world")); // blocks outside the device-backed range read from this pool's arena exactly as host-only ones do
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains(ModelRoute.CudaBackendSeam))
        {
            throw new InvalidOperationException(
                "CUDA driver seam unavailable: a divided KV residency needs the backend that holds persistent block copies, and this host opened none.",
                ex);
        }
    }

    [Fact]
    [Trait("Category", "Cuda")]
    public void SystemMemoryWeightsBesideDeviceBackedBlocks_OpenAndSeatEveryClaimAtItsChainsTier()
    {
        string path = SyntheticModel("kv-residency-mixed");

        CapturingLogger logger = new();

        // The mixed shape this vocabulary already allows: weights resident where the file's copies are read, blocks admitted against their declared device ceiling — one model carrying both at once needs no key beyond today's placement axis and its budget chains.
        try
        {
            using GenerationService engine = GenerationService.Open(
                Config(
                    path,
                    "kv-mixed",
                    kvBlocks: 8,
                    placement: new PlacementConfig { KvResidency = "vram" },
                    budgets:
                    [
                        new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1024 } },
                        new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram, BudgetTier.Vram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Vram] = 8L } },
                    ]),
                logger);

            IReadOnlyList<string> claims = logger.Messages(LogLevel.Information)
                .Where(message => message.Contains("ledger_claim", StringComparison.Ordinal))
                .ToList();

            Assert.Contains(claims, message => message.Contains("resource='weights' tier='dram'", StringComparison.Ordinal)); // the weight claim seats at its chain's system-memory position beside an opened device-backed sibling
            Assert.Contains(claims, message => message.Contains("resource='kv' tier='vram' amount_blocks=8", StringComparison.Ordinal)); // and this reservation's device-held copy carries its own standing seat against the declared ceiling

            Assert.Equal(BlockResidencyMode.Vram, engine.ServedBlockPool.Residency);
            Assert.Equal(8, engine.ServedBlockPool.DeviceBackedBlockCount); // both resources landed on the tiers their chains named — nothing half-opened
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains(ModelRoute.CudaBackendSeam))
        {
            throw new InvalidOperationException(
                "CUDA driver seam unavailable: a device-backed KV residency needs the backend that holds persistent block copies, and this host opened none.",
                ex);
        }
    }

    // ---------------------------------------------------------------- helpers

    private static ModelConfig ConfigModel(string modelName, string modelPath, int kvBlocks, string residency) => new()
    {
        Name = modelName,
        Path = modelPath,
        MaxContextTokens = 512,
        KvBlocks = kvBlocks,
        Placement = new PlacementConfig { KvResidency = residency },
        Budgets = [new BudgetChain
        {
            Resource = BudgetResource.Kv,
            Order = [BudgetTier.Dram, BudgetTier.Vram],
            Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Vram] = 4L }, // narrower than the reservation this model reserves
        }],
    };

    private static LairConfig ConfigFrom(ModelConfig model) => new()
    {
        Server = new ServerConfig(),
        Engine = new EngineConfig { Threads = 1 },
        Models = [model],
    };

    private static LairConfig Config(
        string modelPath,
        string modelName,
        int kvBlocks,
        PlacementConfig? placement = null,
        IReadOnlyList<BudgetChain>? budgets = null) => new()
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
                KvBlocks = kvBlocks,
                Placement = placement ?? new PlacementConfig(),
                Budgets = budgets,
            },
        ],
    };

    private static void GenerateConcurrently(GenerationService engine, int count) =>
        Task.WhenAll([.. Enumerable.Range(0, count).Select(_ => engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = "hello world",
            MaxTokens = 8,
            Temperature = 0f,
        }, CancellationToken.None))]).GetAwaiter().GetResult();

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
