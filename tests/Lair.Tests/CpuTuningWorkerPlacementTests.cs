using Lair.Backends.Cpu;

namespace Lair.Tests;

/// <summary>
/// Worker-placement gates for the CPU backend's physical-core seam (WorkerPlacement + ThreadAffinity): sibling-group parsing from
/// sysfs-shaped fixtures, node-local preference with socket spanning only when quota requires it, SMT-sibling borrowing only beyond
/// distinct-core capacity, restricted-mask confinement that can narrow but never widen placement, and silent fallback to today's
/// unpinned posture where topology cannot be read. Parsing/selection are pure-managed so these gates run identically on Windows CI;
/// the sched_setaffinity application itself is exercised only by host-gated suites (AffinityTests), not here — nothing in this file
/// changes a thread's affinity or reads real /sys files on a Linux runner.
/// </summary>
[Trait("Category", "CpuTuning")]
public sealed class CpuTuningWorkerPlacementTests
{
    // A compact stand-in for the lab Xeon posture (report 2026-11-27T14-14-58Z-optimize-q4k-widebatch-xeon-pinning): node0/sock0 owns physical cores with primaries 0-7 and HT siblings 8-15; node1/sock1 owns primaries 16-23 with siblings 24-31.
    private static Dictionary<int, string?> MiniXeonSiblings()
    {
        var siblings = new Dictionary<int, string?>();

        for (int primary = 0; primary < 8; primary++)
        {
            string pair = $"{primary},{primary + 8}"; // each core lists its full sibling group, exactly as thread_siblings_list states it on both members
            siblings[primary] = pair;
            siblings[primary + 8] = pair;
        }

        for (int primary = 16; primary < 24; primary++)
        {
            string pair = $"{primary},{primary + 8}"; // primaries 16-23 with siblings 24-31: the second node's HT pairs
            siblings[primary] = pair;
            siblings[primary + 8] = pair;
        }

        return siblings;
    }

    private static Dictionary<int, int> MiniXeonNodes()
    {
        var nodes = new Dictionary<int, int>();

        for (int id = 0; id <= 15; id++)
        {
            nodes[id] = 0; // primaries and siblings alike belong to node0's cpulist "0-15"
        }

        for (int id = 16; id <= 31; id++)
        {
            nodes[id] = 1;
        }

        return nodes;
    }

    private static Dictionary<int, int> MiniXeonPackages()
    {
        var packages = new Dictionary<int, int>();

        for (int id = 0; id <= 15; id++)
        {
            packages[id] = 0; // physical_package_id states the socket per CPU; siblings share their core's package
        }

        for (int id = 16; id <= 31; id++)
        {
            packages[id] = 1;
        }

        return packages;
    }

    private static IReadOnlyList<PhysicalCore> MiniXeonCores() =>
        WorkerPlacement.BuildCores(MiniXeonSiblings(), MiniXeonNodes(), MiniXeonPackages());

    // ---------------------------------------------------------------- cpu-list parsing

    [Theory]
    [InlineData("0-3,8", new[] { 0, 1, 2, 3, 8 })] // the sysfs shape: ranges and singletons joined by commas
    [InlineData("5", new[] { 5 })]
    [InlineData("7", new[] { 7 })]
    [InlineData(" 4 , 6 ", new[] { 4, 6 })] // kernel files carry no padding; the parser tolerates it anyway
    public void ParseCpuList_AcceptsIdsAndRanges(string text, int[] expected)
    {
        SortedSet<int>? parsed = WorkerPlacement.ParseCpuList(text);

        Assert.NotNull(parsed);
        Assert.Equal(expected, parsed.ToArray());
    }

    [Theory]
    [InlineData("")] // an empty line states no ids: unreadable posture rather than a zero-id claim
    [InlineData("-1")] // a range with no head is not a bounded list
    [InlineData("7-3")] // reversed ranges are malformed, never silently inverted
    [InlineData("abc")]
    [InlineData("0-x")]
    public void ParseCpuList_MalformedText_ReadsAsUnparseable(string text)
    {
        Assert.Null(WorkerPlacement.ParseCpuList(text));
    }

    // ---------------------------------------------------------------- sibling-group derivation

    [Fact]
    public void BuildCores_CollapsesHtPairsIntoOneCorePerGroup()
    {
        IReadOnlyList<PhysicalCore> cores = MiniXeonCores();

        Assert.Equal(16, cores.Count); // 32 logical processors, 16 physical cores: each HT pair collapsed to one group

        PhysicalCore first = cores[0];
        Assert.Equal(0, first.PrimaryId);
        Assert.Equal([0, 8], first.SiblingIds); // primary is the lowest-numbered processor on the core; siblings ascend
        Assert.Equal(0, first.NodeIndex);
        Assert.Equal(0, first.PackageId);

        PhysicalCore node1First = cores[8];
        Assert.Equal(16, node1First.PrimaryId);
        Assert.Equal([16, 24], node1First.SiblingIds);
        Assert.Equal(1, node1First.NodeIndex);
    }

    [Fact]
    public void BuildCores_UnreadableEntries_AreDroppedRatherThanGuessed()
    {
        var siblings = new Dictionary<int, string?>
        {
            [0] = "0-1", // one HT pair stated by both of its members: a readable core
            [1] = "0-1",
            [2] = null, // sysfs did not state this CPU's sibling list (or reading it failed)
            [3] = "not a cpu list", // malformed text never becomes partial truth
        };

        IReadOnlyList<PhysicalCore> cores = WorkerPlacement.BuildCores(siblings);

        Assert.Single(cores);
        Assert.Equal([0, 1], cores[0].SiblingIds);
        Assert.Equal(-1, cores[0].NodeIndex); // no node attribution stated: the core still participates in placement, unattributed
    }

    [Fact]
    public void BuildCores_SingleThreadCore_ListsOnlyItself()
    {
        var siblings = new Dictionary<int, string?> { [4] = "4" }; // SMT-less hosts state a one-member sibling list

        IReadOnlyList<PhysicalCore> cores = WorkerPlacement.BuildCores(siblings);

        Assert.Single(cores);
        Assert.Equal([4], cores[0].SiblingIds);
    }

    // ---------------------------------------------------------------- node-local preference and socket spanning

    [Fact]
    public void Select_QuotaInsideOneNode_NeverSpansTheOtherSocket()
    {
        PlacementSelection selection = WorkerPlacement.Select(MiniXeonCores(), allowedProcessorIds: null, workerQuota: 4);

        Assert.True(selection.IsResolved);
        Assert.False(selection.UsesSmtSiblings); // quota fits distinct physical cores: no sibling borrowing anywhere below
        Assert.Equal([0, 1, 2, 3], selection.ProcessorIds); // one distinct core per worker, node-local first, primaries only

        foreach (int id in selection.ProcessorIds)
        {
            Assert.False(id >= 8 && id <= 15 || id >= 24 && id <= 31); // every picked processor is a primary: no HT sibling made it into the set
        }
    }

    [Fact]
    public void Select_QuotaBeyondOneNode_CoversDistinctCoresBeforeTouchingSiblings()
    {
        PlacementSelection selection = WorkerPlacement.Select(MiniXeonCores(), allowedProcessorIds: null, workerQuota: 16);

        Assert.False(selection.UsesSmtSiblings); // spanning sockets is required by quota vs capacity — but distinct cores still suffice whole
        Assert.Equal([0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23], selection.ProcessorIds); // both nodes' primaries, siblings untouched
    }

    [Fact]
    public void Select_QuotaBeyondPhysicalCapacity_BorrowsSiblingsRoundRobin()
    {
        PlacementSelection selection = WorkerPlacement.Select(MiniXeonCores(), allowedProcessorIds: null, workerQuota: 20);

        Assert.True(selection.IsResolved);
        Assert.Equal(20, selection.ProcessorIds.Count); // quota fits inside the logical capacity (32 processors) so placement resolves
        Assert.True(selection.UsesSmtSiblings); // distinct-core capacity ran out at 16; only then may HT siblings be borrowed

        IReadOnlyList<int> chosen = selection.ProcessorIds;
        int[] primaries = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23];
        foreach (int id in primaries)
        {
            Assert.Contains(id, chosen); // every distinct physical core is claimed before any sibling overflow pass runs
        }

        int[] borrowed = [.. chosen.Except(primaries)];
        Assert.Equal([8, 9, 10, 11], borrowed); // the four remaining workers land one per already-chosen core in round-robin order — never stacked on a single core's siblings and not revisiting any core before every node-local option was taken
    }

    [Fact]
    public void Select_QuotaBeyondLogicalCapacity_PlacesWhatItCan_AndReportsTheRest()
    {
        PlacementSelection selection = WorkerPlacement.Select(MiniXeonCores(), allowedProcessorIds: null, workerQuota: 40); // only 32 processors exist here

        Assert.Equal(32, selection.ProcessorIds.Count); // the posture places what it can express; callers treat a short placement as partial rather than inventing ids
        Assert.True(selection.UsesSmtSiblings);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-5)] // a non-positive quota asks for no placement: the unpinned posture stands untouched, whatever the topology says
    public void Select_NonPositiveQuota_RequestsNoPlacement(int quota)
    {
        PlacementSelection selection = WorkerPlacement.Select(MiniXeonCores(), allowedProcessorIds: null, workerQuota: quota);

        Assert.False(selection.IsResolved);
    }

    [Fact]
    public void Select_EmptyTopology_IsUnresolved_FallbackStaysSilent()
    {
        PlacementSelection selection = WorkerPlacement.Select([], allowedProcessorIds: null, workerQuota: 8);

        Assert.False(selection.IsResolved); // an unreadable posture never becomes a partial pin someone later misreads as placement
    }

    // ---------------------------------------------------------------- restricted-mask confinement (containers)

    [Fact]
    public void Select_RestrictedMask_ConfinesSelectionAndNeverWidensIt()
    {
        int[] allowed = [16, 17, 18, 19, 20, 21, 22, 23]; // a container cpuset holding node1's primaries only

        PlacementSelection selection = WorkerPlacement.Select(MiniXeonCores(), allowed, workerQuota: 4);

        Assert.Equal([16, 17, 18, 19], selection.ProcessorIds);
        foreach (int id in selection.ProcessorIds)
        {
            Assert.Contains(id, allowed); // placement narrowed to the mask; node0 cores were dropped outright
        }
    }

    [Fact]
    public void Select_RestrictedMask_UsesAllowedSiblingsWhenPrimaryIsOutside()
    {
        int[] allowed = [8, 9, 10, 11]; // only HT siblings of node0 cores remain inside the mask — those cores still participate through their allowed member

        PlacementSelection selection = WorkerPlacement.Select(MiniXeonCores(), allowed, workerQuota: 4);

        Assert.Equal([8, 9, 10, 11], selection.ProcessorIds);
    }

    [Fact]
    public void Select_RestrictedMask_OverflowCannotEscapeTheCpuset()
    {
        int[] allowed = [2, 3, 4, 5]; // quota wants eight workers; the mask holds four primaries and none of their siblings (8-15 are outside)

        PlacementSelection selection = WorkerPlacement.Select(MiniXeonCores(), allowed, workerQuota: 8);

        Assert.Equal([2, 3, 4, 5], selection.ProcessorIds); // a short placement inside the mask — never widened to sibling ids the container forbids
    }

    [Fact]
    public void Select_MaskWithNoUsableCore_IsUnresolved()
    {
        int[] allowed = [60]; // an id this synthetic topology does not name at all

        PlacementSelection selection = WorkerPlacement.Select(MiniXeonCores(), allowed, workerQuota: 2);

        Assert.False(selection.IsResolved); // nothing inside the mask can host a worker here: today's unpinned posture stands
    }

    [Fact]
    public void Select_EmptyAllowedList_ReadsAsUnrestrictedNotConfinement()
    {
        PlacementSelection selection = WorkerPlacement.Select(MiniXeonCores(), [], workerQuota: 2);

        Assert.Equal([0, 1], selection.ProcessorIds); // an unreported mask is "no confinement stated", not a cpuset of zero processors
    }

    // ---------------------------------------------------------------- multi-word masks and the unreadable-host fallback

    [Fact]
    public void BuildProcessorMask_CoversIdsPastOneGroupInWords()
    {
        IReadOnlyList<ulong> single = ThreadAffinity.BuildProcessorMask([0, 63]);
        Assert.Single(single);
        Assert.Equal(1UL | (1UL << 63), single[0]);

        IReadOnlyList<ulong> wide = ThreadAffinity.BuildProcessorMask([5, 70]); // id 70 lives in word one: bit 6 of the second processor set
        Assert.Equal(2, wide.Count);
        Assert.Equal(1UL << 5, wide[0]);
        Assert.Equal(1UL << 6, wide[1]);

        Assert.Empty(ThreadAffinity.BuildProcessorMask([])); // no ids asked for: nothing to apply
    }

    [Fact]
    public void BuildProcessorMask_NegativeId_FailsNamingTheContract()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ThreadAffinity.BuildProcessorMask([-1])); // a negative id is not a processor this seam can name
    }

    [Fact]
    public void PlacementSelection_BuildsTheSameMultiWordShape()
    {
        var cores = new List<PhysicalCore>
        {
            new(PrimaryId: 70, SiblingIds: [70], NodeIndex: 1, PackageId: 1),
            new(PrimaryId: 80, SiblingIds: [80], NodeIndex: 2, PackageId: 2),
        };

        PlacementSelection selection = WorkerPlacement.Select(cores, allowedProcessorIds: null, workerQuota: 2);

        IReadOnlyList<ulong> words = selection.BuildMaskWords();
        Assert.Equal(2, words.Count); // ids 70 and 80 both sit in word one of the processor mask
        Assert.Equal((1UL << (70 % 64)) | (1UL << (80 % 64)), words[1]);
    }

    [Fact]
    public void DetectLinuxPosture_OnHostWithoutSysfs_ReportsNoPosture()
    {
        if (OperatingSystem.IsLinux())
        {
            return; // on a Linux runner this seam reads the real host's sysfs; its shape is asserted by host-gated suites, not here
        }

        Assert.Null(WorkerPlacement.DetectLinuxPosture()); // no /sys tree: callers fall back to unpinned placement silently
    }

    [Fact]
    public void DetectAllowedProcessorIds_OnNonLinuxHost_ReportsUnrestricted()
    {
        if (OperatingSystem.IsLinux())
        {
            return; // the confined-container assertion belongs to Linux runners; here we assert only that no probe claims a mask this platform cannot report
        }

        Assert.Null(ThreadAffinity.DetectAllowedProcessorIds()); // null reads as "no confinement reported", never an empty cpuset that strands every worker
    }

    [Fact]
    public void ApplyCoreSet_ZeroMask_IsTheTrivialAccept()
    {
        Assert.True(ThreadAffinity.ApplyCoreSet(0UL)); // no processor set requested: nothing to apply, on any platform (matches the unpinned pool posture)
    }

    // ---------------------------------------------------------------- backend seam regressions (Windows-visible half)

    [Fact]
    public void Backend_UnpinnedOpen_OnNonLinuxHost_StaysUnpinned_AndAutoPlacementDerivesNothing()
    {
        if (OperatingSystem.IsLinux())
        {
            return; // auto placement is the Linux seam; on this CI host it must derive nothing and change nothing
        }

        using CpuBackend backend = new(threads: 2);

        Assert.False(backend.IsPinningConfigured); // today's unpinned posture stands exactly as before
        Assert.Null(backend.ResolvedAutoPlacement); // no mask was derived where the seam is Linux-only

        using CpuBackend pinned = new(threads: 2, coreSet: new HashSet<int>([0, 1]));
        Assert.True(pinned.IsPinningConfigured); // an explicit configured core set keeps its documented single-group pinning contract
    }
}
