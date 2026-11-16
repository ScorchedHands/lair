using Lair.Models;

namespace Lair.Tests;

/// <summary>
/// Pure ledger accounting over synthetic capacity tables: claims seat in registration order against the first chain
/// position an opened seam backs and still seats, a claim that cannot seat where earlier ones hold a tier lands on
/// the next position its own chain names rather than evicting anything, a read-side origin store carries a demoted
/// weight claim without consuming capacity, an exhausted chain reports model, resource, tier, and the breakdown of what
/// every consulted position holds, and the unbounded posture is stated explicitly instead of a total being invented.
/// Nothing here opens a pool or runs a request: these are the ledger's own rules against tables it was handed.
/// </summary>
[Trait("Category", "Budget")]
public sealed class DeviceLedgerTests
{
    [Fact]
    public void RegisterClaim_FirstComeSeatsFirstTier_AndSecondDemotesInsteadOfEvictingFirst()
    {
        ResolvedBudgetChain chain = WeightsChain([BudgetTier.Dram, BudgetTier.Vram]);
        using TestLedger ledger = new(
            Capacity(BudgetTier.Dram, BudgetResource.Weights, 1),
            Capacity(BudgetTier.Vram, BudgetResource.Weights, 1));

        ClaimOutcome first = ledger.Ledger.RegisterClaim("model-a", chain, amount: 1);
        Assert.True(first.Seated);
        Assert.Equal(BudgetTier.Dram, first.Claim!.Tier);
        Assert.Equal(0, first.Claim.ChainPosition);
        Assert.False(first.Claim.Demoted);

        // The second claim walks past the tier the first one holds and lands on its chain's next position — an earlier claim is never displaced.
        ClaimOutcome second = ledger.Ledger.RegisterClaim("model-b", chain, amount: 1);
        Assert.True(second.Seated);
        Assert.Equal(BudgetTier.Vram, second.Claim!.Tier);
        Assert.Equal(1, second.Claim.ChainPosition);
        Assert.True(second.Claim.Demoted);

        Assert.Equal([0, 1], [.. ledger.Ledger.Claims.Select(claim => claim.Sequence)]); // registration order is the seating order
        Assert.Equal(1L, ledger.Ledger.Occupied(BudgetTier.Dram, BudgetResource.Weights));
        Assert.Single(ledger.Ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.Weights), claim => claim.Model == "model-a");
    }

    [Fact]
    public void RegisterClaim_ExhaustedChain_ReportsModelResourceTierAndLedgerBreakdown()
    {
        ResolvedBudgetChain chain = WeightsChain([BudgetTier.Dram]);
        using TestLedger ledger = new(Capacity(BudgetTier.Dram, BudgetResource.Weights, 1));

        Assert.True(ledger.Ledger.RegisterClaim("model-a", chain, amount: 1).Seated);

        ClaimOutcome exhausted = ledger.Ledger.RegisterClaim("model-b", chain, amount: 1);
        Assert.False(exhausted.Seated);
        Assert.Null(exhausted.Claim);
        Assert.Equal(BudgetTier.Dram, exhausted.ExhaustedAtTier);

        Assert.Contains("model 'model-b'", exhausted.Breakdown);
        Assert.Contains("'weights'", exhausted.Breakdown);
        Assert.Contains("'dram' holds 1 of 1 megabytes", exhausted.Breakdown);
        Assert.Contains("'model-a' 1 megabytes", exhausted.Breakdown); // the breakdown names who already holds the tier

        ClaimChainStep held = Assert.Single(exhausted.ChainWalk);
        Assert.Equal(BudgetTier.Dram, held.Tier);
        Assert.Equal("held by earlier claims", held.Outcome); // the position was walked and passed by, never taken from the claim registered ahead of it
    }

    [Fact]
    public void RegisterClaim_UnboundedSeamSeatsEveryClaim_AndStatesThePostureWithoutInventingATotal()
    {
        DeviceLedger ledger = DeviceLedger.OpenedTiersUnbounded([BudgetTier.Dram]); // what today's host reports: DRAM backed, no public total

        CapacitySeam seam = ledger.SeamFor(BudgetTier.Dram, BudgetResource.Weights)!;
        Assert.Null(seam.Capacity);
        Assert.Contains("unbounded capacity posture", seam.Describe());

        ResolvedBudgetChain chain = WeightsChain([BudgetTier.Dram]); // a declared ceiling binds the declaration, not this host's unbounded seam
        for (int claim = 1; claim <= 5; claim++)
        {
            ClaimOutcome outcome = ledger.RegisterClaim($"model-{claim}", chain, amount: 2);
            Assert.True(outcome.Seated); // no position of an unbounded chain exhausts
            Assert.Equal(BudgetTier.Dram, outcome.Claim!.Tier);
        }

        Assert.Equal(10L, ledger.Occupied(BudgetTier.Dram, BudgetResource.Weights));
    }

    [Fact]
    public void RegisterClaim_FiniteAcceleratorWeightSeam_ExhaustsAgainstDeclaredCapacity_NamingModelResourceTierAndAmounts()
    {
        ResolvedBudgetChain chain = WeightsChain([BudgetTier.Vram]);
        using TestLedger ledger = new(Capacity(BudgetTier.Vram, BudgetResource.Weights, 4));

        Assert.True(ledger.Ledger.RegisterClaim("model-a", chain, amount: 3).Seated);

        ClaimOutcome exhausted = ledger.Ledger.RegisterClaim("model-b", chain, amount: 2);
        Assert.False(exhausted.Seated);
        Assert.Equal(BudgetTier.Vram, exhausted.ExhaustedAtTier);

        Assert.Contains("model 'model-b'", exhausted.Breakdown);
        Assert.Contains("'weights'", exhausted.Breakdown);
        Assert.Contains("'vram' holds 3 of 4 megabytes", exhausted.Breakdown);
        Assert.Contains("'model-a' 3 megabytes", exhausted.Breakdown);
    }

    [Fact]
    public void RegisterClaim_ChainWithNoOpenedSeam_ReportsWhatThisHostDoesNotOpen()
    {
        using TestLedger ledger = new(); // a ledger with no capacity seams at all: nothing is guessed into existence
        ClaimOutcome outcome = ledger.Ledger.RegisterClaim("model-a", WeightsChain([BudgetTier.Dram]), amount: 1);

        Assert.False(outcome.Seated);
        Assert.Equal(BudgetTier.Dram, outcome.ExhaustedAtTier);
        Assert.Contains("'dram' has no opened capacity seam", outcome.Breakdown);
    }

    [Fact]
    public void RegisterClaim_WeightDemotionToOriginStore_SeatsWithoutConsumingCapacity()
    {
        ResolvedBudgetChain chain = new()
        {
            Resource = BudgetResource.Weights,
            Tiers =
            [
                new ResolvedBudgetTier { Tier = BudgetTier.Dram, Cap = 1 },
                new ResolvedBudgetTier { Tier = BudgetTier.Nvme, IsOriginStore = true }, // the file already holds the complete weight set there
            ],
        };

        using TestLedger ledger = new(Capacity(BudgetTier.Dram, BudgetResource.Weights, 1));

        Assert.True(ledger.Ledger.RegisterClaim("model-a", chain, amount: 1).Seated); // first claim keeps the working-copy tier
        ClaimOutcome demoted = ledger.Ledger.RegisterClaim("model-b", chain, amount: 1);

        Assert.True(demoted.Seated); // landing on the origin store is a posture, not an eviction of 'model-a'
        Assert.Equal(BudgetTier.Nvme, demoted.Claim!.Tier);
        Assert.True(demoted.Claim.AtOriginStore);
        Assert.True(demoted.Claim.Demoted);

        Assert.Equal(1L, ledger.Ledger.Occupied(BudgetTier.Dram, BudgetResource.Weights)); // the first claim still holds its tier whole
        Assert.Equal(0L, ledger.Ledger.Occupied(BudgetTier.Nvme, BudgetResource.Weights));  // an origin-store seating consumes no ledger capacity
    }

    [Fact]
    public void RegisterClaim_ModelDeclaredCeilingTooSmall_DemotesToNextOpenedPosition_WithoutConsumingTheRejectedTier()
    {
        ResolvedBudgetChain chain = new()
        {
            Resource = BudgetResource.Weights,
            Tiers =
            [
                new ResolvedBudgetTier { Tier = BudgetTier.Dram, Cap = 2 }, // the model declares only two megabytes here — a five-megabyte claim does not half-fit it.
                new ResolvedBudgetTier { Tier = BudgetTier.Vram },          // its chain's next position is opened by an unbounded seam and takes the whole working copy instead.
            ],
        };

        DeviceLedger ledger = new([Capacity(BudgetTier.Dram, BudgetResource.Weights, 10), new CapacitySeam { Tier = BudgetTier.Vram, Resource = BudgetResource.Weights }]);

        ClaimOutcome outcome = ledger.RegisterClaim("model-a", chain, amount: 5);
        Assert.True(outcome.Seated);
        Assert.Equal(BudgetTier.Vram, outcome.Claim!.Tier);
        Assert.Equal(1, outcome.Claim.ChainPosition);
        Assert.True(outcome.Claim.Demoted);

        ClaimChainStep rejected = Assert.Single(outcome.ChainWalk, step => step.Outcome == "declared ceiling below demand"); // the chain's own too-small position was passed by, not taken from anyone.
        Assert.Equal(BudgetTier.Dram, rejected.Tier);
        Assert.Equal(2L, rejected.DeclaredCap!.Value);

        Assert.Equal(0L, ledger.Occupied(BudgetTier.Dram, BudgetResource.Weights)); // the too-small tier consumed nothing on its way past it.
        Assert.Equal(5L, ledger.Occupied(BudgetTier.Vram, BudgetResource.Weights));  // the whole working copy landed at the next position instead of being split across tiers.
    }

    [Fact]
    public void RegisterClaim_KvChainCountsBlocks_AndNamesItsUnitInEveryBreakdown()
    {
        ResolvedBudgetChain chain = new() { Resource = BudgetResource.Kv, Tiers = [new ResolvedBudgetTier { Tier = BudgetTier.Dram }] };
        using TestLedger ledger = new(Capacity(BudgetTier.Dram, BudgetResource.Kv, 8));

        Assert.True(ledger.Ledger.RegisterClaim("model-a", chain, amount: 5).Seated);
        Assert.True(ledger.Ledger.RegisterClaim("model-b", chain, amount: 3).Seated); // 5 + 3 exactly fills the tier's declared blocks

        ClaimOutcome third = ledger.Ledger.RegisterClaim("model-c", chain, amount: 1);
        Assert.False(third.Seated);
        Assert.Contains("'kv'", third.Breakdown);
        Assert.Contains("holds 8 of 8 KV blocks", third.Breakdown); // the unit a KV claim counts in is blocks, never megabytes
    }

    [Fact]
    public void BreakdownFor_ListsSeatedClaimsInRegistrationOrder()
    {
        ResolvedBudgetChain chain = WeightsChain([BudgetTier.Dram]);
        using TestLedger ledger = new(Capacity(BudgetTier.Dram, BudgetResource.Weights, 10));

        Assert.True(ledger.Ledger.RegisterClaim("model-a", chain, amount: 4).Seated);
        Assert.True(ledger.Ledger.RegisterClaim("model-b", chain, amount: 3).Seated);

        string breakdown = ledger.Ledger.BreakdownFor("model-a", BudgetResource.Weights);
        Assert.Contains("'model-a' weights 4 megabytes at 'dram'", breakdown);

        IReadOnlyList<DeviceLedgerClaim> contested = ledger.Ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.Weights); // both claimants share the tier
        Assert.Equal(["model-a", "model-b"], [.. contested.Select(claim => claim.Model)]);
    }

    [Fact]
    public void ObserveAdmission_RidesInsideTheModelsOwnStandingClaim_WithoutCompetingWithItself()
    {
        ResolvedBudgetChain chain = new() { Resource = BudgetResource.Kv, Tiers = [new ResolvedBudgetTier { Tier = BudgetTier.Dram }] };
        using TestLedger ledger = new(Capacity(BudgetTier.Dram, BudgetResource.Kv, 4));

        Assert.True(ledger.Ledger.RegisterClaim("model-a", chain, amount: 4).Seated); // the pool's standing seat fills its tier

        ClaimOutcome observation = ledger.Ledger.ObserveAdmission("model-a", chain, needed: 3);
        Assert.True(observation.Seated); // a landed sequence rides inside what its own model already seats — an engine never competes with itself
        Assert.Equal(1L, ledger.Ledger.Claims.Count(c => c.Model == "model-a")); // observations seat or refuse; they register no claim

        ClaimChainStep seated = Assert.Single(observation.ChainWalk);
        Assert.Equal("seated", seated.Outcome);
    }

    [Fact]
    public void UpdateCapacity_TightensLaterSeats_AndAnObservationBesideOtherModelsReportsExhaustion()
    {
        ResolvedBudgetChain chain = new() { Resource = BudgetResource.Kv, Tiers = [new ResolvedBudgetTier { Tier = BudgetTier.Dram }] };
        using TestLedger ledger = new(Capacity(BudgetTier.Dram, BudgetResource.Kv, 8));

        Assert.True(ledger.Ledger.RegisterClaim("model-a", chain, amount: 4).Seated);
        Assert.True(ledger.Ledger.RegisterClaim("model-b", chain, amount: 2).Seated); // both standing seats fit the total reported today

        ledger.Ledger.UpdateCapacity(BudgetTier.Dram, BudgetResource.Kv, capacity: 3); // the host re-reports a smaller total; what already stands is never revoked
        Assert.True(ledger.Ledger.ObserveAdmission("model-a", chain, needed: 1).Seated); // what model b holds still leaves room for this landing inside model a's own seat

        ClaimOutcome refused = ledger.Ledger.ObserveAdmission("model-b", chain, needed: 1);
        Assert.False(refused.Seated); // what model a holds leaves no room beside it for anything else landing here
        Assert.Equal(BudgetTier.Dram, refused.ExhaustedAtTier);
        Assert.Contains("'dram' holds 4 of 3 KV blocks", refused.Breakdown); // the tier reports standing claims above its lowered total

        IReadOnlyList<DeviceLedgerClaim> stillStanding = ledger.Ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.Kv);
        Assert.Equal(["model-a", "model-b"], [.. stillStanding.Select(claim => claim.Model)]); // a lowered total tightens later seats only — earlier claims stay seated

        Assert.False(ledger.Ledger.RegisterClaim("model-c", chain, amount: 1).Seated); // and nothing new registers against a tier already held past its total
    }

    [Fact]
    public void UpdateStandingClaimAmount_ChangesTheAmount_WithoutMovingOrDuplicatingTheClaim()
    {
        ResolvedBudgetChain chain = new() { Resource = BudgetResource.Kv, Tiers = [new ResolvedBudgetTier { Tier = BudgetTier.Dram, Cap = 8 }] };
        using TestLedger ledger = new(Capacity(BudgetTier.Dram, BudgetResource.Kv, 16));

        Assert.True(ledger.Ledger.RegisterClaim("model-a", chain, amount: 8).Seated);
        Assert.True(ledger.Ledger.UpdateStandingClaimAmount("model-a", BudgetResource.Kv, newAmount: 4));

        DeviceLedgerClaim claim = Assert.Single(ledger.Ledger.Claims); // the same seated claim carries the updated amount, not a replacement entry.
        Assert.Equal("model-a", claim.Model);
        Assert.Equal(BudgetTier.Dram, claim.Tier);
        Assert.Equal(4L, claim.Amount);
        Assert.Equal(4L, ledger.Ledger.Occupied(BudgetTier.Dram, BudgetResource.Kv));
    }

    [Fact]
    public void UpdateStandingClaimAmount_AtOneTier_LeavesTheSameModelsClaimAtAnotherTier_Untouched()
    {
        ResolvedBudgetChain hostReservation = new() { Resource = BudgetResource.Kv, Tiers = [new ResolvedBudgetTier { Tier = BudgetTier.Dram }] };
        ResolvedBudgetChain deviceRange = new() { Resource = BudgetResource.Kv, Tiers = [new ResolvedBudgetTier { Tier = BudgetTier.Vram, Cap = 8 }] };

        using TestLedger ledger = new(
            Capacity(BudgetTier.Dram, BudgetResource.Kv, 16),
            Capacity(BudgetTier.Vram, BudgetResource.Kv, 8));

        Assert.True(ledger.Ledger.RegisterClaim("model-a", hostReservation, amount: 8).Seated);
        Assert.True(ledger.Ledger.RegisterClaim("model-a", deviceRange, amount: 4).Seated); // one model's blocks can stand at two positions of their own chain at once

        Assert.True(ledger.Ledger.UpdateStandingClaimAmount("model-a", BudgetResource.Kv, newAmount: 5, atTier: BudgetTier.Dram));

        Assert.Equal(5L, Assert.Single(ledger.Ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.Kv)).Amount); // the seat a live edit governs carries the new amount
        Assert.Equal(4L, Assert.Single(ledger.Ledger.ClaimsAt(BudgetTier.Vram, BudgetResource.Kv)).Amount); // and the claim seated beside it keeps its own standing amount rather than being retargeted twice

        Assert.Equal(5L, ledger.Ledger.Occupied(BudgetTier.Dram, BudgetResource.Kv));
        Assert.Equal(4L, ledger.Ledger.Occupied(BudgetTier.Vram, BudgetResource.Kv)); // what each position holds counts once, at the tier it stands at

        Assert.False(ledger.Ledger.UpdateStandingClaimAmount("model-a", BudgetResource.Kv, newAmount: 9, atTier: BudgetTier.Nvme)); // a seat this model never took is not moved to some other one
    }

    [Fact]
    public void ObserveAdmission_LandingRidesInsideEngineSeams_WithoutReDecidingItsOwnModelCeiling()
    {
        ResolvedBudgetChain chain = new() { Resource = BudgetResource.Kv, Tiers = [new ResolvedBudgetTier { Tier = BudgetTier.Dram, Cap = 4 }] };
        using TestLedger ledger = new(Capacity(BudgetTier.Dram, BudgetResource.Kv, 10));

        Assert.True(ledger.Ledger.RegisterClaim("model-a", chain, amount: 2).Seated); // the standing reservation fits what this model declares at its only copy-holding position.
        ClaimOutcome seated = ledger.Ledger.ObserveAdmission("model-a", chain, needed: 5);

        Assert.True(seated.Seated); // a landing admitted by the engine's own pool seam rides inside that seat; observation does not become a second KV gate beside it.
        ClaimChainStep landed = Assert.Single(seated.ChainWalk);
        Assert.Equal(BudgetTier.Dram, landed.Tier);
        Assert.Equal("seated", landed.Outcome);
        Assert.Equal(4L, landed.DeclaredCap!.Value); // the model's declared ceiling stays visible in diagnostics without being re-enforced here.

        Assert.Equal(2L, ledger.Ledger.Occupied(BudgetTier.Dram, BudgetResource.Kv)); // observations register nothing and displace nothing.
    }

    private static ResolvedBudgetChain WeightsChain(IReadOnlyList<BudgetTier> tiers) => new()
    {
        Resource = BudgetResource.Weights,
        Tiers = [.. tiers.Select(tier => new ResolvedBudgetTier { Tier = tier })],
    };

    private static CapacitySeam Capacity(BudgetTier tier, BudgetResource resource, long capacity) => new()
    {
        Tier = tier,
        Resource = resource,
        Capacity = capacity,
    };

    // A disposable wrapper keeps every test's ledger closed the same way a host closes its accounting; the ledger itself owns no OS handles.
    private sealed class TestLedger : IDisposable
    {
        public DeviceLedger Ledger { get; } = new([]);

        public TestLedger(params CapacitySeam[] seams) => Ledger = new DeviceLedger(seams);

        public void Dispose()
        {
        }
    }
}
