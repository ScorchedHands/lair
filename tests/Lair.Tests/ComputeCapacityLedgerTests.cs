using Lair.Models;

namespace Lair.Tests;

/// <summary>
/// Compute accounting over a reported posture: claims seat against the summed logical processors of every node in registration order,
/// first-come-first-served — an oversubscribed later claim is clamped to what earlier claims left (requested against granted) rather than
/// refused or allowed to displace them, and only a model that opts into core overcommit keeps its whole requested share while overlapping.
/// Exclusive capacity never exceeds the reported posture across every claim; declared tier totals read against the summed nodes as agreement
/// or a named mismatch, and byte totals nothing reported read as incomparable rather than disagreement.
/// </summary>
[Trait("Category", "Scheduling")]
public sealed class ComputeCapacityLedgerTests
{
    [Fact]
    public void ReportedPosture_SumsPerNodeLogicalProcessors_AndStatesNothingWhereBytesWereNotReported()
    {
        HostTopology posture = SyntheticTwoNodePosture(processors: 3, memoryMb: [4096, 8192]);

        Assert.Equal(2, posture.Nodes.Count);
        Assert.Equal(6, posture.TotalLogicalProcessors); // the total a claim counts against is what every node carries together.
        Assert.Equal(12288L, posture.ReportedMemoryMb);   // summed only because each node stated its own bytes.

        HostTopology halfReported = new()
        {
            Nodes =
            [
                new HostNodeCapacity { Index = 0, LogicalProcessors = 3, MemoryMb = 4096 },
                new HostNodeCapacity { Index = 1, LogicalProcessors = 3, MemoryMb = null },
            ],
        };

        Assert.Null(halfReported.ReportedMemoryMb); // one node left bytes unreported: the posture states nothing rather than guessing a machine total.
    }

    [Fact]
    public void HostProbe_ReportsOneWholeMachinePosture_CoveringEveryLogicalProcessorOnce()
    {
        HostTopology posture = HostTopologyProbe.Detect();

        Assert.NotEmpty(posture.Nodes);
        Assert.Equal(Environment.ProcessorCount, posture.TotalLogicalProcessors); // the seam accounts for exactly this machine's logical processors.

        HashSet<int> ids = [];
        foreach (HostNodeCapacity node in posture.Nodes)
        {
            Assert.True(node.LogicalProcessors >= 1);
            foreach (int coreId in node.CoreIds)
            {
                Assert.True(ids.Add(coreId)); // no logical processor is attributed to two nodes.
            }
        }

        if (posture.DerivedFromReportedTopology && posture.Nodes.Count > 0 && posture.Nodes[0].CoreIds.Count > 0)
        {
            Assert.Equal(Environment.ProcessorCount, ids.Count); // a reported posture partitions this machine's logical processors completely.
        }

        HostTopology cached = HostTopologyProbe.Detect();
        Assert.Same(posture, cached); // the posture is read once per process and reused from then on.
    }

    [Fact]
    public void CheckDeclaredTotals_AgreesAgainstSummedNodes_AndNamesAMismatchWithBothNumbers()
    {
        ComputeCapacityLedger ledger = new(SyntheticTwoNodePosture(processors: 3, memoryMb: null));

        DeclaredTotalsReport agreeing = ledger.CheckDeclaredTotals(declaredDramCores: 6, declaredDramMb: null);
        Assert.True(agreeing.CoresAgree); // a declared total that matches the summed nodes needs no warning.
        Assert.False(agreeing.ReportsCoreMismatch);

        DeclaredTotalsReport disagreeing = ledger.CheckDeclaredTotals(declaredDramCores: 32, declaredDramMb: null);
        Assert.True(disagreeing.ReportsCoreMismatch); // warn-and-accept territory: the mismatch is reported, not enforced here.
        Assert.Contains("declared cores 32 against reported 6", disagreeing.Describe());

        DeclaredTotalsReport unreportedBytes = ledger.CheckDeclaredTotals(declaredDramCores: null, declaredDramMb: 65536);
        Assert.False(unreportedBytes.ReportsByteMismatch); // bytes the host never stated are not a disagreement with what an operator declared.
        Assert.Contains("against unreported node bytes", unreportedBytes.Describe());

        DeclaredTotalsReport matchingBytes = new ComputeCapacityLedger(SyntheticTwoNodePosture(3, [4096, 8192]))
            .CheckDeclaredTotals(declaredDramCores: null, declaredDramMb: 12288);
        Assert.Contains("match reported", matchingBytes.Describe());

        DeclaredTotalsReport disagreeingBytes = new ComputeCapacityLedger(SyntheticTwoNodePosture(3, [4096, 8192]))
            .CheckDeclaredTotals(declaredDramCores: null, declaredDramMb: 1024);
        Assert.True(disagreeingBytes.ReportsByteMismatch);
    }

    [Fact]
    public void ClaimsSeatFirstComeFirstServed_LaterOversubscribedClaimIsClampedToWhatEarlierClaimsLeft()
    {
        ComputeCapacityLedger ledger = new(SyntheticTwoNodePosture(processors: 4, memoryMb: null)); // eight logical processors reported.

        ComputeGrant first = ledger.RegisterClaim("first", requestedCores: 5);
        Assert.Equal(5, first.GrantedCores);
        Assert.False(first.SharesHeldCores);
        Assert.Equal(3, ledger.RemainingCores);

        ComputeGrant second = ledger.RegisterClaim("second", requestedCores: 6); // asks past what the first claim left.
        Assert.Equal("5/5", first.GrantVersusRequest());
        Assert.Equal("6/3", second.GrantVersusRequest()); // partial grant naming requested against granted, never a refusal.
        Assert.Equal(3, second.ExclusiveCores);           // it holds as its own only what stood unclaimed ahead of it.
        Assert.False(second.SharesHeldCores);             // the clamped share is still its own cores, not an overlap.

        ComputeGrant third = ledger.RegisterClaim("third", requestedCores: 2);   // capacity has run out entirely by now.
        Assert.Equal(1, third.GrantedCores);              // a claim that finds nothing left still runs beside standing grants.
        Assert.True(third.SharesHeldCores);               // and that overlap is reported rather than hidden.

        Assert.Equal(["first", "second", "third"], [.. ledger.Grants.Select(grant => grant.Model)]); // registration order, earlier claims untouched.
        Assert.Equal(["third"], [.. ledger.ModelsSharingHeldCores]);

        _ = ledger.RegisterClaim("fourth", 1);            // a further model is still admitted after exhaustion rather than refused.
        Assert.Equal(0, ledger.RemainingCores);           // accounting floors at zero; it never goes negative through clamping.
    }

    [Fact]
    public void OptedInOvercommitKeepsRequestedShareAndRecordsTheOverlap_WhileExclusiveCapacityStaysWithinPosture()
    {
        ComputeCapacityLedger ledger = new(SyntheticTwoNodePosture(processors: 4, memoryMb: null));

        _ = ledger.RegisterClaim("first", requestedCores: 6); // six of the eight reported processors.

        ComputeGrant optedIn = ledger.RegisterClaim("second", requestedCores: 5, allowOvercommit: true);
        Assert.Equal(5, optedIn.GrantedCores);   // an opt-in claim keeps what it asked for instead of being clamped.
        Assert.True(optedIn.SharesHeldCores);    // whatever the remaining capacity could not seat rides cores 'first' holds.

        Assert.Equal(["second"], [.. ledger.ModelsSharingHeldCores]);
        Assert.Contains("shared with earlier grants", optedIn.Describe());

        int exclusiveTotal = ledger.Grants.Sum(grant => grant.ExclusiveCores);
        Assert.True(exclusiveTotal <= ledger.TotalCores); // overlap is accounted, never double-booked against the posture.
    }

    [Fact]
    public void EveryClaimSurvivesAnExhaustedPosture_AndExclusiveCapacityNeverExceedsWhatWasReported()
    {
        ComputeCapacityLedger ledger = new(SyntheticTwoNodePosture(processors: 4, memoryMb: null)); // eight logical processors.

        List<ComputeGrant> seated = [];
        for (int model = 1; model <= 20; model++)
        {
            seated.Add(ledger.RegisterClaim($"model-{model}", requestedCores: 4)); // twenty claims against capacity for two full ones.
        }

        Assert.Equal(20, seated.Count);                                        // nothing here refuses a served model outright.
        Assert.All(seated, grant => Assert.True(grant.GrantedCores >= 1));    // every claim participates with at least one worker.
        Assert.True(seated.Sum(grant => grant.ExclusiveCores) <= ledger.TotalCores);

        int exclusive = 0;
        foreach (ComputeGrant grant in seated)
        {
            exclusive += grant.ExclusiveCores;
            if (exclusive > ledger.TotalCores)
            {
                Assert.Fail("a claim took more as its own than the posture reported ahead of it");
            }

            Assert.Equal(grant.ExclusiveCores == 0, grant.SharesHeldCores); // a claim seated after capacity ran out reports exactly that overlap.
        }
    }

    [Fact]
    public void OneModelClaimsItsWorkerShareOnce_AndASecondClaimForItIsRefusedByName()
    {
        ComputeCapacityLedger ledger = new(SyntheticTwoNodePosture(processors: 2, memoryMb: null));

        _ = ledger.RegisterClaim("qwen", requestedCores: 1);

        ModelException error = Assert.Throws<ModelException>(() => ledger.RegisterClaim("qwen", requestedCores: 1));
        Assert.Contains("'qwen' already holds a compute grant", error.Message);
    }

    [Fact]
    public void LedgerRefusesAnEmptyPosture_AndANodeWithoutLogicalProcessors()
    {
        ModelException empty = Assert.Throws<ModelException>(() => new ComputeCapacityLedger(new HostTopology { Nodes = [] }));
        Assert.Contains("naming at least one node", empty.Message);

        ModelException deadNode = Assert.Throws<ModelException>(() => new ComputeCapacityLedger(new HostTopology
        {
            Nodes = [new HostNodeCapacity { Index = 0, LogicalProcessors = 4 }, new HostNodeCapacity { Index = 1, LogicalProcessors = 0 }],
        }));

        Assert.Contains("node 1 carries no logical processors", deadNode.Message);
    }

    private static HostTopology SyntheticTwoNodePosture(int processors, long?[]? memoryMb)
    {
        memoryMb ??= [null, null]; // a posture that states nothing about bytes leaves both nodes unreported.

        return new()
        {
            Nodes =
            [
                new HostNodeCapacity
                {
                    Index = 0,
                    LogicalProcessors = processors,
                    MemoryMb = memoryMb[0],
                    CoreIds = [.. Enumerable.Range(0, processors)],
                },
                new HostNodeCapacity
                {
                    Index = 1,
                    LogicalProcessors = processors,
                    MemoryMb = memoryMb[1],
                    CoreIds = [.. Enumerable.Range(processors, processors)],
                },
            ],
            DerivedFromReportedTopology = true, // synthetic posture stands in for what a real host would have reported.
        };
    }
}
