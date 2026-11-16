using Lair.Host.Config;
using Lair.Http;
using Lair.Models;
using Microsoft.Extensions.Logging;

namespace Lair.Tests;

/// <summary>
/// The shared host ledger across two programmatically opened engines: claims register in open order against one set of
/// capacity seams, the first engine keeps its tier while the second lands on the next position its own chain names (or
/// reports an exhausted chain as a load failure), and none of this disturbs what either engine's pool already enforces —
/// no live referenced block is ever displaced by another engine's claim. The host serves one model today, so cross-claim
/// contention is proven through the shared ledger seam itself rather than faked multi-model serving.
/// </summary>
[Trait("Category", "Scheduling")]
[Trait("Category", "Budget")]
public sealed class DeviceLedgerSharedHostTests : IDisposable
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
    public void SharedLedger_SecondsEngineWeightClaimDemotesToNextChainPosition_FirstKeepsItsTier_AndLogsRegistrationOrder()
    {
        string path = SyntheticModel("ledger-order");
        CapturingLogger logger = new();

        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Weights, Capacity = 1 }]);

        using GenerationService first = GenerationService.Open(ConfigWith(path, "ledger-a", weightsToOriginStore: true), logger, ledger);
        using GenerationService second = GenerationService.Open(ConfigWith(path, "ledger-b", weightsToOriginStore: true), logger, ledger);

        IReadOnlyList<DeviceLedgerClaim> claims = ledger.Claims; // one host accounting across both opened engines
        Assert.Equal(2, claims.Count);
        Assert.Equal(["ledger-a", "ledger-b"], [.. claims.Select(claim => claim.Model)]); // registration order is open order

        Assert.Equal(BudgetTier.Dram, claims[0].Tier);   // the first engine keeps its working-copy tier whole
        Assert.False(claims[0].Demoted);
        Assert.Equal(BudgetTier.Disk, claims[1].Tier);   // the second lands on its chain's next position — the origin store it already reads from
        Assert.True(claims[1].Demoted);

        string[] claimLines = [.. logger.Messages(LogLevel.Information).Where(line => line.Contains("ledger_claim", StringComparison.Ordinal))];
        Assert.Equal(2, claimLines.Length); // one ordered log line per registered claim
        Assert.Contains("'ledger-a'", claimLines[0]);
        Assert.Contains("tier='dram'", claimLines[0]);
        Assert.Contains("demoted=False", claimLines[0]);
        Assert.Contains("'ledger-b'", claimLines[1]);
        Assert.Contains("tier='disk'", claimLines[1]); // the demoted seat names its chain position in the accepted vocabulary.
        Assert.Contains("demoted=True", claimLines[1]);

        // A generation on the first engine runs normally after the second opened beside it: the demotion moved nothing.
        GenerationResult result = Generate(first, "hello world");
        Assert.True(result.CompletionTokens > 0);
    }

    [Fact]
    public void SharedLedger_KvReservationExhaustedAtOpen_IsALoadFailure_NamingModelResourceTierAndBreakdown()
    {
        string path = SyntheticModel("ledger-load-failure");
        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Kv, Capacity = 8 }]);

        using GenerationService first = GenerationService.Open(ConfigWith(path, "ledger-kv-a", kvBlocks: 8), null, ledger); // its reservation fills the tier
        Assert.Single(ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.Kv));

        BudgetExhaustedException loadFailure = Assert.Throws<BudgetExhaustedException>(
            () => GenerationService.Open(ConfigWith(path, "ledger-kv-b", kvBlocks: 4), null, ledger)); // no position of its chain seats the reservation

        Assert.Equal("ledger-kv-b", loadFailure.Model);
        Assert.Equal(BudgetResource.Kv, loadFailure.Resource);
        Assert.Equal(BudgetTier.Dram, loadFailure.Tier);
        Assert.Contains("'dram' holds 8 of 8 KV blocks", loadFailure.Breakdown);

        // The refused open registered nothing: only the first engine's claim stands on the shared ledger.
        Assert.Single(ledger.Claims);
    }

    [Fact]
    public void SharedLedger_LandedReservationsRideStandingClaims_BothEnginesKeepServing_AfterTheOtherOpened()
    {
        string path = SyntheticModel("ledger-admission");
        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Kv, Capacity = 16 }]);

        // Both engines' reservations already seat: eight blocks each against a sixteen-block tier. Nothing either engine admits afterwards competes with its own pool's standing seat.
        using GenerationService first = GenerationService.Open(ConfigWith(path, "ledger-admit-a", kvBlocks: 8), null, ledger);
        using GenerationService second = GenerationService.Open(ConfigWith(path, "ledger-admit-b", kvBlocks: 8), null, ledger);

        Assert.True(Generate(first, "hello world", maxTokens: 4).CompletionTokens > 0); // a full shared tier does not stop either engine serving what its own pool backs
        Assert.True(Generate(second, "hello world", maxTokens: 4).CompletionTokens > 0);

        Assert.Equal(2L, ledger.Claims.Count(c => c.Resource == BudgetResource.Kv)); // landed reservations rode inside the standing seats — observations register no claim of their own
    }

    [Fact]
    public void SharedLedger_ReReportedSmallerTierCapacity_RefusesLaterAdmissionsWithBudgetCode_WithoutDisplacingStandingClaims()
    {
        string path = SyntheticModel("ledger-shrink");
        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Kv, Capacity = 16 }]);

        using GenerationService first = GenerationService.Open(ConfigWith(path, "ledger-shrink-a", kvBlocks: 8), null, ledger);
        using GenerationService second = GenerationService.Open(ConfigWith(path, "ledger-shrink-b", kvBlocks: 8), null, ledger);

        Assert.True(Generate(first, "hello world", maxTokens: 4).CompletionTokens > 0); // admitted while the tier still held its reported total

        ledger.UpdateCapacity(BudgetTier.Dram, BudgetResource.Kv, capacity: 8); // the host re-reports a smaller total than what already stands there — nothing seated is revoked

        EngineBusyException refusal = Assert.Throws<EngineBusyException>(
            () => Generate(second, "hello world", maxTokens: 4)); // only other models' standing seats now compete for this tier, and they fill it

        Assert.Equal("budget_exhausted", refusal.Code);
        Assert.Contains("ledger-shrink-b", refusal.Message);
        Assert.Contains("'dram' holds 8 of 8 KV blocks ('ledger-shrink-a' 8 KV blocks, 'ledger-shrink-b' 8 KV blocks)", refusal.Message); // the breakdown names what stands at the tier against its re-reported total

        IReadOnlyList<DeviceLedgerClaim> standing = ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.Kv);
        Assert.Equal(["ledger-shrink-a", "ledger-shrink-b"], [.. standing.Select(claim => claim.Model)]); // both reservations stay seated exactly as registered — a lowered total tightens later seats only

        EngineBusyException sameEngineToo = Assert.Throws<EngineBusyException>(
            () => Generate(first, "hello world", maxTokens: 4)); // the tier now reports no room beside what stands there for either model's next landing

        Assert.Equal("budget_exhausted", sameEngineToo.Code);
    }

    [Fact]
    public void SharedLedger_LiveKvCeilingEditUpdatesTheStandingClaim_WithoutMovingItsSeat()
    {
        string path = SyntheticModel("ledger-live-kv");
        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Kv, Capacity = 16 }]);

        using GenerationService engine = GenerationService.Open(ConfigWith(path, "ledger-live", kvBlocks: 8), null, ledger);
        Assert.Equal(8L, Assert.Single(ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.Kv)).Amount); // the reservation opened on its standing seat.

        IReadOnlyList<FieldResult> results = engine.ApplyConfiguration(ConfigWith(path, "ledger-live", kvBlocks: 4));
        FieldResult applied = Assert.Single(results);
        Assert.Equal("models[0].budgets.kv.caps.dram_blocks", applied.Change.Field);
        Assert.Equal(ConfigApplyOutcome.Applied, applied.Outcome);

        DeviceLedgerClaim claim = Assert.Single(ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.Kv)); // the same registration stands at the same tier.
        Assert.Equal("ledger-live", claim.Model);
        Assert.False(claim.Demoted);
        Assert.Equal(4L, claim.Amount); // what it covers follows the live engine seam's applied value without re-running first-come selection.

        GenerationResult result = Generate(engine, "hello world", maxTokens: 4);
        Assert.True(result.CompletionTokens > 0);
    }

    private string SyntheticModel(string modelName)
    {
        string path = SyntheticLlamaGguf.Write(modelName);
        _tempFiles.Add(path);
        return path;
    }

    private static GenerationResult Generate(GenerationService engine, string prompt, int maxTokens = 8) =>
        engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = prompt,
            MaxTokens = maxTokens,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult();

    private static LairConfig ConfigWith(string modelPath, string modelName, int kvBlocks = 8, bool weightsToOriginStore = false) => new()
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
                Budgets = weightsToOriginStore
                    ? [new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Dram, BudgetTier.Disk], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1024 } }]
                    : [new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = kvBlocks } }],
            },
        ],
    };

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
