using System.Globalization;

namespace Lair.Backends.Cpu;

/// <summary>
/// One physical core as the topology source reported it: its logical processors grouped from per-CPU SMT sibling lists, with the
/// NUMA node and socket/package ids sysfs stated for them. The primary id is the lowest-numbered processor on the core — what a
/// distinct-core placement pins a worker to when the core's SMT siblings are not needed. Node and package ids read -1 where no
/// source stated one: an unattributed core still participates in placement, it just cannot be preferred or avoided by node/socket rules.
/// </summary>
internal sealed record PhysicalCore(
    int PrimaryId,
    IReadOnlyList<int> SiblingIds,
    int NodeIndex,
    int PackageId)
{
    /// <summary>The order sysfs groups cores in: unreported nodes first (a -1 id never blocks placement), then package id, then the primary processor id.</summary>
    public static readonly IComparer<PhysicalCore> CanonicalOrder = Comparer<PhysicalCore>.Create((a, b) =>
    {
        int byNode = a.NodeIndex.CompareTo(b.NodeIndex);
        if (byNode != 0)
        {
            return byNode;
        }

        int byPackage = a.PackageId.CompareTo(b.PackageId);
        if (byPackage != 0)
        {
            return byPackage;
        }

        return a.PrimaryId.CompareTo(b.PrimaryId);
    });
}

/// <summary>What the placement seam could read about this host: every physical core sysfs stated, and — when sched_getaffinity answered — the processor set this process is confined to (null when unrestricted or unreadable).</summary>
internal sealed record CpuPlacementTopology(
    IReadOnlyList<PhysicalCore> Cores,
    IReadOnlyList<int>? AllowedProcessorIds)
{
    public int PhysicalCoreCount => Cores.Count;

    /// <summary>A posture that names no cores cannot place anything: callers fall back to unpinned placement silently.</summary>
    public bool IsResolvable => Cores.Count > 0;
}

/// <summary>The processor ids a quota was placed onto, plus whether the selection had to borrow SMT siblings because quota exceeded distinct physical cores. An empty set means this topology cannot express the quota — the documented fallback posture is unpinned.</summary>
internal sealed record PlacementSelection(
    IReadOnlyList<int> ProcessorIds,
    bool UsesSmtSiblings)
{
    public bool IsResolved => ProcessorIds.Count > 0;

    /// <summary>The selection as one-word affinity masks for <see cref="ThreadAffinity.ApplyProcessorMask"/>: word w covers processor ids [w*64, w*64+63).</summary>
    internal IReadOnlyList<ulong> BuildMaskWords()
    {
        if (ProcessorIds.Count == 0)
        {
            return [];
        }

        int highest = ProcessorIds.Max();
        ulong[] words = new ulong[highest / 64 + 1];
        foreach (int id in ProcessorIds)
        {
            words[id / 64] |= 1UL << (id % 64);
        }

        return words;
    }
}

/// <summary>
/// The CPU backend's worker-placement seam: pure-managed parsing of sysfs topology facts into physical cores, and the selection rule that places a thread quota onto them. Parsing and selection are OS-independent (unit tests exercise both against synthetic fixtures); only <see cref="DetectLinuxPosture"/> touches the filesystem or asks the kernel for this process's affinity mask, and it is reached on Linux hosts only.
///
/// The measured reason for these rules lives in committed reports: unpinned prefill lands wherever the OS puts threads while HT-sibling pinning measures worse than unpinned — tasksetting to node0 <em>physical</em> cores lifted implied weight bandwidth 0.415→0.54 GB/s on the lab Xeon lane (report directory 2026-11-27T14-14-58Z-optimize-q4k-widebatch-xeon-pinning). So placement prefers distinct physical cores, node-local first, and borrows SMT siblings only when quota exceeds available capacity. Nothing here changes kernel arithmetic; it chooses processor ids for worker threads only.
/// </summary>
internal static class WorkerPlacement
{
    /// <summary>
    /// Parses a sysfs cpu-list line — comma-separated ids and inclusive ranges, e.g. "0-15,32" or "0-7". Returns null when the text is missing or not a bounded list (a malformed entry reads as unreadable topology rather than as partial truth).
    /// </summary>
    public static SortedSet<int>? ParseCpuList(string? cpuListText)
    {
        if (cpuListText is null)
        {
            return null; // no text stated: that's the absent posture, not a parse failure to invent an id for
        }

        SortedSet<int> ids = [];
        foreach (string rawPart in cpuListText.Split(',', StringSplitOptions.RemoveEmptyEntries))
        {
            string part = rawPart.Trim();
            int dash = part.IndexOf('-');
            if (dash < 0)
            {
                if (!int.TryParse(part, NumberStyles.Integer, CultureInfo.InvariantCulture, out int single) || single < 0)
                {
                    return null;
                }

                ids.Add(single);
                continue;
            }

            if (!int.TryParse(part[..dash], NumberStyles.Integer, CultureInfo.InvariantCulture, out int from) ||
                !int.TryParse(part[(dash + 1)..], NumberStyles.Integer, CultureInfo.InvariantCulture, out int to) ||
                from < 0 ||
                to < from)
            {
                return null;
            }

            for (int id = from; id <= to; id++)
            {
                ids.Add(id);
            }
        }

        return ids.Count == 0 ? null : ids;
    }

    /// <summary>
    /// Derives physical cores from per-CPU sibling facts. Inputs are the raw file contents a sysfs probe read: each processor's
    /// <c>thread_siblings_list</c> text, its NUMA node index (from that node's cpulist — processors no node lists carry -1), and its
    /// <c>physical_package_id</c>. Processors whose sibling list is unreadable or self-inconsistent are dropped rather than guessed at;
    /// a core group merges every processor reporting the same sorted id set, so HT pairs collapse to one entry. Groups come back in
    /// canonical order (node, package, primary). An empty result states "this source stated no cores", which callers treat as unresolvable topology.
    /// </summary>
    public static IReadOnlyList<PhysicalCore> BuildCores(
        IReadOnlyDictionary<int, string?> threadSiblingsByProcessor,
        IReadOnlyDictionary<int, int>? nodeIndexByProcessor = null,
        IReadOnlyDictionary<int, int>? packageIdByProcessor = null)
    {
        ArgumentNullException.ThrowIfNull(threadSiblingsByProcessor);

        Dictionary<string, SortedSet<int>> groupsByKey = [];
        foreach ((int processor, string? siblingsText) in threadSiblingsByProcessor)
        {
            if (siblingsText is null || processor < 0)
            {
                continue; // a CPU whose sibling list this host did not state contributes no group rather than inventing one
            }

            SortedSet<int>? listed = ParseCpuList(siblingsText);
            if (listed is null)
            {
                continue; // malformed text: unreadable topology, dropped silently like every other fallback here
            }

            SortedSet<int> ids = [.. listed];
            ids.Add(processor); // a sysfs sibling list always includes its own processor; normalize so self-inconsistent entries still group deterministically

            string key = string.Join(',', ids);
            groupsByKey.TryAdd(key, ids);
        }

        List<PhysicalCore> cores = [];
        foreach (SortedSet<int> ids in groupsByKey.Values)
        {
            int primaryId = ids.Min; // the lowest-numbered processor on this core is its placement representative
            cores.Add(new PhysicalCore(
                primaryId,
                [.. ids],
                StatedAttribute(ids, nodeIndexByProcessor),
                StatedAttribute(ids, packageIdByProcessor)));
        }

        cores.Sort(PhysicalCore.CanonicalOrder); // -1 (unreported) sorts ahead of any stated id: an unattributed core never blocks placement
        return cores;
    }

    /// <summary>
    /// Places one thread quota onto a reported core set. Rules, in order: confine candidates to the allowed processor mask when this process is restricted (a container's cpuset can only ever narrow placement, never widen it); prefer node-local physical cores by packing lowest-index nodes and sockets first so a quota that fits inside one node never spans a socket; take exactly one distinct core per worker while quota has capacity; borrow SMT siblings round-robin across already-chosen cores only when quota exceeds the number of distinct usable physical cores. An empty selection means this topology cannot express the quota — callers fall back to unpinned placement silently.
    /// </summary>
    public static PlacementSelection Select(IReadOnlyList<PhysicalCore> cores, IReadOnlyCollection<int>? allowedProcessorIds, int workerQuota)
    {
        ArgumentNullException.ThrowIfNull(cores);

        if (workerQuota < 1 || cores.Count == 0)
        {
            return new PlacementSelection([], false);
        }

        HashSet<int>? allowed = allowedProcessorIds is null or { Count: 0 } ? null : [.. allowedProcessorIds];

        List<(PhysicalCore Core, int Representative)> candidates = [];
        foreach (PhysicalCore core in cores)
        {
            if (RepresentativeOf(core, allowed) is not int representative)
            {
                continue; // a core with no processor inside the mask this process may run on cannot host a worker here
            }

            candidates.Add((core, representative));
        }

        if (candidates.Count == 0)
        {
            return new PlacementSelection([], false);
        }

        List<(PhysicalCore Core, int Representative)> placed = [];
        HashSet<int> chosen = [];
        foreach ((PhysicalCore core, int representative) in candidates.OrderBy(candidate => candidate.Core, PhysicalCore.CanonicalOrder))
        {
            if (chosen.Count == workerQuota)
            {
                break; // quota fits distinct physical cores: no socket spanning and no sibling borrowing happens below this line
            }

            placed.Add((core, representative));
            chosen.Add(representative);
        }

        bool usedSiblings = false;
        for (int round = 0; chosen.Count < workerQuota && placed.Count > 0; round++)
        {
            // Quota beyond distinct-core capacity: spread the overflow across already-chosen cores' remaining siblings — one per core per round — rather than crowding a single core's HT pair. Every usable distinct physical core was taken above, so anything added here is genuinely an SMT borrow.
            bool anyCoreOfferedAtRound = false;

            foreach ((PhysicalCore core, int representative) in placed)
            {
                if (chosen.Count == workerQuota)
                {
                    break;
                }

                IReadOnlyList<int> siblings = core.SiblingIds;
                if (round >= siblings.Count)
                {
                    continue; // this core has no further sibling to offer at this round
                }

                anyCoreOfferedAtRound = true; // even a duplicate representative or masked-out member counts as an offer: the round made its pass, progress is decided by whether anything was added
                int siblingId = siblings[round];
                if (allowed is not null && !allowed.Contains(siblingId))
                {
                    continue; // the mask confines even the overflow: a restricted container never widens beyond its cpuset
                }

                if (chosen.Add(siblingId) && siblingId != representative)
                {
                    usedSiblings = true;
                }
            }

            if (!anyCoreOfferedAtRound)
            {
                break; // every chosen core is exhausted: the quota cannot be expressed inside this posture, and what was placed stands as a partial placement rather than an unpinned guess — callers decide.
            }
        }

        List<int> ids = [.. chosen];
        ids.Sort();
        return new PlacementSelection(ids, usedSiblings);
    }

    /// <summary>
    /// Reads this host's placement posture on Linux: per-CPU sibling/package facts from <c>/sys/devices/system/cpu/*&#47;topology</c>, node membership from each <c>/sys/devices/system/node/node*/cpulist</c>, and the process affinity mask through sched_getaffinity (via <see cref="ThreadAffinity"/>). Returns null when sysfs cannot be read at all — callers fall back silently, exactly as every other topology seam here does.
    /// </summary>
    public static CpuPlacementTopology? DetectLinuxPosture()
    {
        const string cpuRoot = "/sys/devices/system/cpu";
        if (!Directory.Exists(cpuRoot))
        {
            return null; // no sysfs tree to read: an unresolvable posture, reported as such by callers
        }

        Dictionary<int, string?> siblingsByCpu = [];
        Dictionary<int, int> packageIdByCpu = [];
        foreach (string directory in Directory.GetDirectories(cpuRoot, "cpu*").OrderBy(path => path, StringComparer.Ordinal))
        {
            if (!int.TryParse(Path.GetFileName(directory)["cpu".Length..], NumberStyles.Integer, CultureInfo.InvariantCulture, out int cpuIndex) || cpuIndex < 0)
            {
                continue; // "cpufreq", "cpuidle" and friends describe policy directories, not a processor's topology directory
            }

            string? siblings = ReadSysfsText(Path.Combine(directory, "topology", "thread_siblings_list"));
            if (siblings is null)
            {
                continue; // a CPU whose sibling list this host did not state contributes no group
            }

            siblingsByCpu[cpuIndex] = siblings;

            string? packageText = ReadSysfsText(Path.Combine(directory, "topology", "physical_package_id"));
            if (packageText is not null && int.TryParse(packageText.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int packageId) && packageId >= 0)
            {
                packageIdByCpu[cpuIndex] = packageId;
            }
        }

        if (siblingsByCpu.Count == 0)
        {
            return null; // sysfs named no processor topology at all: unresolvable, not a zero-core machine claim
        }

        Dictionary<int, int> nodeIndexByCpu = ReadNodeMembership();
        IReadOnlyList<PhysicalCore> cores = BuildCores(siblingsByCpu, nodeIndexByCpu.Count == 0 ? null : nodeIndexByCpu, packageIdByCpu);
        return new CpuPlacementTopology(cores, ThreadAffinity.DetectAllowedProcessorIds());
    }

    // Node membership from the kernel's own per-node cpulists. A processor no node lists stays unattributed (-1) rather than guessed onto one; a whole unreadable tree reads as "no nodes stated", which placement tolerates by treating every core as equally placed.
    private static Dictionary<int, int> ReadNodeMembership()
    {
        Dictionary<int, int> membership = [];
        const string nodeRoot = "/sys/devices/system/node";
        if (!Directory.Exists(nodeRoot))
        {
            return membership;
        }

        foreach (string directory in Directory.GetDirectories(nodeRoot, "node*").OrderBy(path => path, StringComparer.Ordinal))
        {
            if (!int.TryParse(Path.GetFileName(directory)["node".Length..], NumberStyles.Integer, CultureInfo.InvariantCulture, out int nodeIndex) || nodeIndex < 0)
            {
                continue; // "possible" and friends describe topology files, not a node directory with a cpulist
            }

            SortedSet<int>? processors = ParseCpuList(ReadSysfsText(Path.Combine(directory, "cpulist")));
            if (processors is null)
            {
                continue; // this node's list unreadable: its members stay unattributed rather than claimed wrongly
            }

            foreach (int processor in processors)
            {
                membership[processor] = nodeIndex;
            }
        }

        return membership;
    }

    private static string? ReadSysfsText(string path)
    {
        try
        {
            if (!File.Exists(path))
            {
                return null;
            }

            foreach (string line in File.ReadLines(path))
            {
                return line.Trim(); // sysfs carries these facts on one short line
            }

            return null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return null; // a container that cannot read topology files reads as unreadable, never as an error the backend can crash on
        }
    }

    // The processor this core may host a worker on: its primary id when unrestricted, else the lowest-numbered member inside the mask (a restricted posture often keeps only siblings of some cores — those cores still participate through their allowed sibling).
    private static int? RepresentativeOf(PhysicalCore core, HashSet<int>? allowed)
    {
        if (allowed is null)
        {
            return core.PrimaryId;
        }

        foreach (int id in core.SiblingIds) // SiblingIds are ascending, so the first hit is the lowest allowed member
        {
            if (allowed.Contains(id))
            {
                return id;
            }
        }

        return null;
    }

    private static int StatedAttribute(IReadOnlyCollection<int> ids, IReadOnlyDictionary<int, int>? attributeByProcessor)
    {
        if (attributeByProcessor is null)
        {
            return -1;
        }

        foreach (int id in ids) // ids ascend here: the lowest-numbered processor that states the attribute decides for the whole core
        {
            if (attributeByProcessor.TryGetValue(id, out int value))
            {
                return value;
            }
        }

        return -1;
    }
}
