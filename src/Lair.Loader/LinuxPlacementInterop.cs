using System.Globalization;
using System.Runtime.InteropServices;

namespace Lair.Loader;

/// <summary>
/// The placement primitives a measurement lane needs on Linux, and nothing more: anonymous huge-page arenas through libc <c>mmap</c>/<c>madvise</c>, worker or gather-thread
/// binding through <c>sched_setaffinity</c>, an optional first-touch memory policy through the raw <c>set_mempolicy</c> syscall, and placement accounting read back off the host's own maps files.
/// It reserves no hugetlb pages, lowers no sysctl, installs nothing, and never emulates a platform: every ask reports whether it applied or which errno refused it, so a measurement says what the kernel
/// actually did for this process rather than what the harness hoped for. Non-Linux hosts see <see cref="Supported"/> false and get plain committed arenas plus no placement claims.
/// </summary>
public static class LinuxPlacementInterop
{
    private const int ProtRead = 0x1;
    private const int ProtWrite = 0x2;

    private const int MapPrivate = 0x2;
    private const int MapAnonymous = 0x20; // MAP_ANONYMOUS on Linux/x86-64: bytes come from the process, not a file.

    private const int MadvHugePage = 14;
    public const int MadvWillNeed = 3;

    private const long SysSetMempolicy = 239; // x86-64 syscall table value: glibc carries no wrapper for the NUMA policy calls, so this lane reaches them through libc's own syscall().
    private const int MemPolicyDefault = 0;
    private const int MemPolicyBind = 2;

    private const int CpuSetBytes = 128; // cpu_set_t as one kernel word set of 1024 bits: this lane only ever sets ids it read out of the host's own cpulists.

    /// <summary>Whether these primitives exist on this host at all.</summary>
    public static bool Supported => OperatingSystem.IsLinux();

    [DllImport("libc", EntryPoint = "mmap", SetLastError = true)]
    private static extern nint Mmap(nint addr, nuint length, int prot, int flags, int fd, long offset);

    [DllImport("libc", EntryPoint = "munmap", SetLastError = true)]
    private static extern int MunmapCall(nint addr, nuint length);

    [DllImport("libc", EntryPoint = "madvise", SetLastError = true)]
    private static extern int Madvise(nint addr, nuint length, int advice);

    /// <summary>Map an anonymous private range for a hot-row arena: address space is reserved only — pages land on first touch.</summary>
    public static long MmapAnonymous(long bytes) => (long)Mmap(nint.Zero, checked((nuint)bytes), ProtRead | ProtWrite, MapPrivate | MapAnonymous, -1, 0);

    /// <summary>Release an arena this lane mapped. A failed unmap is reported as false rather than throwing out of a dispose path.</summary>
    public static bool Munmap(nint baseAddress, nuint length) => MunmapCall(baseAddress, length) == 0;

    /// <summary><c>madvise(MADV_HUGEPAGE)</c> over an anonymous range: true only when the kernel accepted the ask for this process.</summary>
    public static bool MadviseHugePage(nint baseAddress, nuint length) => Madvise(baseAddress, length, MadvHugePage) == 0;

    /// <summary><c>madvise(MADV_WILLNEED)</c> over a mapped range: the OS prefetch hint DN-M7-11 describes. True when accepted for this range on this host.</summary>
    public static bool MadviseWillNeed(nint baseAddress, nuint length) => Madvise(baseAddress, length, MadvWillNeed) == 0;

    /// <summary>Bind the calling thread to a NUMA node's own cpulist through libc sched_setaffinity. Reports the errno when the kernel refuses rather than pretending locality.</summary>
    public static PlacementAttempt BindCurrentThreadToNode(int node)
    {
        string cpulistPath = $"/sys/devices/system/node/node{node.ToString(CultureInfo.InvariantCulture)}/cpulist";

        if (!File.Exists(cpulistPath))
        {
            return new PlacementAttempt(false, $"node {node} declares no readable cpulist at '{cpulistPath}'");
        }

        string cpuList;

        try
        {
            cpuList = File.ReadAllText(cpulistPath).Trim();
        }
        catch (IOException ex)
        {
            return new PlacementAttempt(false, $"reading '{cpulistPath}' failed: {ex.Message}");
        }

        if (!TryParseCpuList(cpuList, out int[] cpus, out string parseError))
        {
            return new PlacementAttempt(false, $"'{cpuList}' from '{cpulistPath}' is not a cpu list this lane can bind to: {parseError}");
        }

        byte[] mask = BuildCpuSetMask(cpus);

        // pid 0 means the calling thread exactly as sched_setaffinity(2) documents it — no tid bookkeeping needed for worker-thread binding.
        if (SchedSetAffinity(0, CpuSetBytes, mask) != 0)
        {
            return new PlacementAttempt(false, $"sched_setaffinity to node {node} cpulist '{cpuList}' failed: errno={Marshal.GetLastPInvokeError()}");
        }

        return new PlacementAttempt(true, $"calling thread bound to node {node} cpulist '{cpuList}'");
    }

    [DllImport("libc", EntryPoint = "sched_setaffinity", SetLastError = true)]
    private static extern int SchedSetAffinity(int pid, nuint size, byte[] cpuSet);

    /// <summary>Read the calling thread's affinity mask back — the verification half of a placement claim.</summary>
    public static string? ReadCurrentThreadCpuList()
    {
        if (!Supported)
        {
            return null;
        }

        byte[] mask = new byte[CpuSetBytes];

        if (SchedGetAffinity(0, CpuSetBytes, mask) != 0)
        {
            return null;
        }

        List<int> cpus = [];

        for (int bit = 0; bit < CpuSetBytes * 8; bit++)
        {
            if ((mask[bit / 8] & (1 << (bit % 8))) != 0)
            {
                cpus.Add(bit);
            }
        }

        return string.Join(",", cpus);
    }

    [DllImport("libc", EntryPoint = "sched_getaffinity", SetLastError = true)]
    private static extern int SchedGetAffinity(int pid, nuint size, byte[] cpuSet);

    /// <summary>Ask the kernel to place this thread's future anonymous pages on one node (MPOL_BIND over the raw set_mempolicy syscall). An attempt a host refuses is reported with its errno — first-touch discipline beside an affinity binding stays the fallback either way.</summary>
    public static PlacementAttempt TryBindMemoryPolicyToNode(int node)
    {
        if (!Supported)
        {
            return new PlacementAttempt(false, "memory policy placement exists on Linux hosts only");
        }

        nuint[] nodemask = new nuint[8]; // one 64-bit word per eight nodes: this lane binds a single node id it read from sysfs.
        nodemask[node / 64] |= (nuint)1 << (node % 64);

        long result = SyscallBindPolicy(SysSetMempolicy, MemPolicyBind, nodemask, node + 2); // maxnode is one past the highest bit set: exactly what set_mempolicy(2) documents.

        if (result != 0)
        {
            return new PlacementAttempt(false, $"set_mempolicy(MPOL_BIND) to node {node} failed: errno={Marshal.GetLastPInvokeError()}");
        }

        return new PlacementAttempt(true, $"memory policy bound to node {node}: this thread's future anonymous pages land there first");
    }

    /// <summary>Reset the calling thread's memory policy so a later placement arm starts from the kernel default rather than inheriting an earlier binding.</summary>
    public static PlacementAttempt ClearMemoryPolicy()
    {
        if (!Supported)
        {
            return new PlacementAttempt(false, "memory policy placement exists on Linux hosts only");
        }

        long result = SyscallDefaultPolicy(SysSetMempolicy, MemPolicyDefault, nint.Zero, 0); // MPOL_DEFAULT with a null nodemask: the documented way back to default posture.

        return result == 0
            ? new PlacementAttempt(true, "memory policy reset to the kernel default")
            : new PlacementAttempt(false, $"set_mempolicy(MPOL_DEFAULT) failed: errno={Marshal.GetLastPInvokeError()}");
    }

    // syscall(2) has no fixed signature; these two call shapes are what this lane ever asks of it.
    [DllImport("libc", EntryPoint = "syscall", SetLastError = true)]
    private static extern long SyscallBindPolicy(long number, int mode, nuint[] nodemask, int maxnode);

    [DllImport("libc", EntryPoint = "syscall", SetLastError = true)]
    private static extern long SyscallDefaultPolicy(long number, int mode, nint nodemask, int maxnode);

    /// <summary>Touch one byte per page across an arena range from the calling thread: first-touch placement decides which node those pages land on.</summary>
    public static long FirstTouch(nint baseAddress, long length)
    {
        if (length <= 0 || !Supported)
        {
            return 0L;
        }

        long touched = 0;

        for (long offset = 0; offset < length; offset += 4096)
        {
            nint address = (nint)((long)baseAddress + offset); // the read-then-write is the fault: an untouched anonymous page has no physical frame and no home node yet.
            Marshal.WriteByte(address, (byte)(Marshal.ReadByte(address) | 0));
            touched++;
        }

        return touched;
    }

    /// <summary>The host's NUMA nodes with their cpulists, read from sysfs — the same facts numactl would print, without requiring it.</summary>
    public static List<(int Node, string CpuList)> DetectNumaNodes()
    {
        List<(int Node, string CpuList)> nodes = [];

        if (!Supported)
        {
            return nodes;
        }

        try
        {
            foreach (string entry in Directory.EnumerateFileSystemEntries("/sys/devices/system/node"))
            {
                string name = Path.GetFileName(entry);

                if (!name.StartsWith("node", StringComparison.Ordinal))
                {
                    continue; // "possible", "online" and friends describe topology, not a node directory with cpulists to read.
                }

                string indexText = name.Substring(4);

                if (!int.TryParse(indexText, NumberStyles.Integer, CultureInfo.InvariantCulture, out int nodeIndex))
                {
                    continue;
                }

                nodes.Add((nodeIndex, File.ReadAllText(Path.Combine(entry, "cpulist")).Trim()));
            }
        }
        catch (IOException)
        {
            // A host whose sysfs tree cannot be enumerated reports no nodes rather than inventing a single-node posture here.
        }

        return nodes;
    }

    /// <summary>Parse a kernel cpulist like "0-15,32-47" into individual processor ids.</summary>
    public static bool TryParseCpuList(string cpuList, out int[] cpus, out string error)
    {
        List<int> parsed = [];

        foreach (string part in cpuList.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int dash = part.IndexOf('-');

            if (dash < 0)
            {
                if (!int.TryParse(part, NumberStyles.Integer, CultureInfo.InvariantCulture, out int single) || single < 0 || single >= CpuSetBytes * 8)
                {
                    cpus = [];
                    error = $"'{part}' is not a processor id inside the mask this lane builds";
                    return false;
                }

                parsed.Add(single);
                continue;
            }

            string headText = part[..dash].Trim();
            string tailText = part[(dash + 1)..].Trim();

            if (!int.TryParse(headText, NumberStyles.Integer, CultureInfo.InvariantCulture, out int head) ||
                !int.TryParse(tailText, NumberStyles.Integer, CultureInfo.InvariantCulture, out int tail) ||
                head < 0 || tail < head || tail >= CpuSetBytes * 8)
            {
                cpus = [];
                error = $"'{part}' is not a bounded processor range inside the mask this lane builds";
                return false;
            }

            for (int id = head; id <= tail; id++)
            {
                parsed.Add(id);
            }
        }

        if (parsed.Count == 0)
        {
            cpus = [];
            error = "the list carried no processors";
            return false;
        }

        cpus = [.. parsed];
        error = string.Empty;
        return true;
    }

    private static byte[] BuildCpuSetMask(int[] cpus)
    {
        byte[] mask = new byte[CpuSetBytes]; // cpu_set_t: one bit per processor id, exactly the layout sched_setaffinity(2) expects.

        foreach (int cpu in cpus)
        {
            mask[cpu / 8] |= (byte)(1 << (cpu % 8));
        }

        return mask;
    }

    /// <summary>Transparent-huge-page posture and hugetlb pool facts as the host reports them — read-only sysfs/proc probes, never a reservation.</summary>
    public static Dictionary<string, object?> ReadHugePageFacts()
    {
        Dictionary<string, object?> facts = new()
        {
            ["supported_on_host_family"] = Supported,
        };

        if (!Supported)
        {
            return facts;
        }

        facts["thp_enabled_posture"] = ReadFirstLine("/sys/kernel/mm/transparent_hugepage/enabled");
        facts["hugetlb_pool_total_pages"] = ReadProcMeminfoValue("HugePages_Total:");
        facts["hugetlb_pool_free_pages"] = ReadProcMeminfoValue("HugePages_Free:");

        Dictionary<string, object?> perNode = new();

        foreach ((int node, string _) in DetectNumaNodes())
        {
            string hugeDir = $"/sys/devices/system/node/node{node.ToString(CultureInfo.InvariantCulture)}/hugepages/hugepages-2048kB";

            if (Directory.Exists(hugeDir))
            {
                perNode[$"node{node}"] = new Dictionary<string, object?>
                {
                    ["nr_hugepages"] = ReadSysfsValue(Path.Combine(hugeDir, "nr_hugepages")),
                    ["free_hugepages"] = ReadSysfsValue(Path.Combine(hugeDir, "free_hugepages")),
                };
            }
        }

        facts["hugetlb_per_node_dirs"] = perNode.Count == 0 ? null : perNode;
        return facts;
    }

    /// <summary>What this host exposes for hardware-counter work: whether perf exists and how paranoid its event access is. Reported, never changed.</summary>
    public static Dictionary<string, object?> ReadHardwareCounterFacts() => new()
    {
        ["perf_binary_present"] = File.Exists("/usr/bin/perf"),
        ["perf_event_paranoid"] = ReadFirstLine("/proc/sys/kernel/perf_event_paranoid") ?? "unreadable",
        ["numa_maps_readable"] = Supported && File.Exists("/proc/self/numa_maps"),
    };

    /// <summary>Minor and major fault counters for this process from /proc/self/stat (fields after the command paren: index 7 minor faults, index 9 major faults).</summary>
    public static (long MinorFaults, long MajorFaults) ReadMinorMajorFaultsOfCurrentProcess()
    {
        try
        {
            string stat = File.ReadAllText("/proc/self/stat");
            int closeParen = stat.LastIndexOf(')');

            if (closeParen < 0 || closeParen + 1 >= stat.Length)
            {
                return (0L, 0L);
            }

            // Fields are single-space separated after the parenthesized command, so a comm containing spaces cannot shift the indices.
            string[] fields = stat.Substring(closeParen + 1).Split(' ', StringSplitOptions.RemoveEmptyEntries);

            if (fields.Length < 10 || !long.TryParse(fields[7], NumberStyles.Integer, CultureInfo.InvariantCulture, out long minor) || !long.TryParse(fields[9], NumberStyles.Integer, CultureInfo.InvariantCulture, out long major))
            {
                return (0L, 0L);
            }

            return (minor, major);
        }
        catch (IOException)
        {
            return (0L, 0L); // a host without a readable /proc falls back to the streamed-byte tallies alone.
        }
    }

    /// <summary>What backs and where this process's own address range lives: THP coverage bytes from smaps plus per-node page counts from numa_maps.</summary>
    public static RangeMapsFacts ReadRangeMaps(nint baseAddress, long length)
    {
        if (!Supported || length <= 0)
        {
            return new RangeMapsFacts();
        }

        ulong start = (ulong)(nuint)baseAddress;
        ulong end = start + checked((ulong)length);

        long anonHugePagesBytes = 0;
        long rssBytes = 0;
        int smapsEntries = 0;
        Dictionary<string, object?> pagesByNode = new();
        int numaMapsEntries = 0;

        try
        {
            bool insideRange = false; // the property lines below belong to whichever range header preceded them.

            foreach (string line in File.ReadLines("/proc/self/smaps"))
            {
                if (TryParseSmapsRange(line, out ulong entryStart, out ulong entryEnd))
                {
                    insideRange = entryStart < end && start < entryEnd; // a kernel split of the arena still lands inside [start,end): count every piece.

                    if (insideRange)
                    {
                        smapsEntries++;
                    }

                    continue;
                }

                if (!insideRange)
                {
                    continue;
                }

                if (line.StartsWith("AnonHugePages:", StringComparison.Ordinal))
                {
                    anonHugePagesBytes += ParseKilobytesField(line); // THP bytes the kernel actually mapped over this range — coverage, not an ask.
                }
                else if (line.StartsWith("Rss:", StringComparison.Ordinal))
                {
                    rssBytes += ParseKilobytesField(line); // resident pages of every piece that overlaps this arena.
                }
            }
        }
        catch (IOException)
        {
            // smaps unreadable: the caller reports zero coverage and lets the backing facts' madvise line speak for itself rather than claiming THP it never confirmed.
        }

        try
        {
            foreach (string line in File.ReadLines("/proc/self/numa_maps"))
            {
                int space = line.IndexOf(' ');

                if (space <= 0 || !ulong.TryParse(line.AsSpan(0, space), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out ulong entryStart))
                {
                    continue; // numa_maps lines carry only the VMA start address: attribution matches any entry that begins inside this lane's range.
                }

                if (entryStart >= end)
                {
                    break; // entries are ascending by address: everything after is past this arena.
                }

                if (entryStart < start)
                {
                    continue;
                }

                numaMapsEntries++;

                foreach (string field in line.Substring(space + 1).Split(' ', StringSplitOptions.RemoveEmptyEntries))
                {
                    int equals = field.IndexOf('=');

                    if (equals <= 0 || !field.StartsWith("N", StringComparison.Ordinal) || !long.TryParse(field.AsSpan(equals + 1), NumberStyles.Integer, CultureInfo.InvariantCulture, out long pages))
                    {
                        continue; // "mapped=", "anon=", "dirty=" describe the VMA's state, not its node placement.
                    }

                    string nodeName = field[..equals]; // N0=…, N1=…: kernel page counts per node for this range piece.

                    pagesByNode.TryGetValue(nodeName, out object? seen);
                    pagesByNode[nodeName] = seen is long previous ? previous + pages : pages;
                }
            }
        }
        catch (IOException)
        {
            // numa_maps unreadable: placement falls back to the smaps coverage plus affinity facts.
        }

        return new RangeMapsFacts
        {
            AnonHugePagesBytesOverRange = anonHugePagesBytes,
            RssBytesOverRange = rssBytes,
            SmapsEntriesOverRange = smapsEntries,
            NumaMapsEntryCount = numaMapsEntries,
            PagesByNode = pagesByNode,
        };
    }

    private static bool TryParseSmapsRange(string line, out ulong start, out ulong end)
    {
        start = 0;
        end = 0;

        int space = line.IndexOf(' ');

        if (space <= 0)
        {
            return false; // a bare property line or an empty pathname tail is never a range header.
        }

        string head = line[..space];
        int dash = head.IndexOf('-');

        if (dash <= 0 || !Uri.IsHexDigit(head[0]))
        {
            return false; // "Size:", "AnonHugePages:" and friends have no leading hex range.
        }

        return ulong.TryParse(head.AsSpan(0, dash), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out start)
            && ulong.TryParse(head.AsSpan(dash + 1), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out end);
    }

    private static long ParseKilobytesField(string line)
    {
        foreach (string piece in line.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            if (!piece.Contains(':') && long.TryParse(piece, NumberStyles.Integer, CultureInfo.InvariantCulture, out long value))
            {
                return checked(value * 1024); // smaps reports these sizes in kilobytes.
            }
        }

        return 0L;
    }

    private static string? ReadFirstLine(string path)
    {
        try
        {
            foreach (string line in File.ReadLines(path))
            {
                return line.Trim();
            }
        }
        catch (IOException)
        {
        }

        return null;
    }

    private static long? ReadSysfsValue(string path)
    {
        string? text = ReadFirstLine(path);

        return text is not null && long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out long value) ? value : null;
    }

    private static long? ReadProcMeminfoValue(string keyPrefix)
    {
        try
        {
            foreach (string line in File.ReadLines("/proc/meminfo"))
            {
                if (!line.StartsWith(keyPrefix, StringComparison.Ordinal))
                {
                    continue;
                }

                string rest = line[keyPrefix.Length..].Trim();

                return long.TryParse(rest, NumberStyles.Integer, CultureInfo.InvariantCulture, out long value) ? value : null;
            }
        }
        catch (IOException)
        {
        }

        return null;
    }
}

/// <summary>One placement ask and what the kernel actually did about it: applied plus its wording, or refused naming the errno — a harness never claims locality this lane could not bind.</summary>
public sealed record PlacementAttempt(bool Applied, string Description);

/// <summary>What the host's own maps files say about one address range: THP coverage and RSS from smaps, per-node kernel-page counts from numa_maps. A zero count means "this host reported none", which differs from a missing file only through the entry counters.</summary>
public sealed record RangeMapsFacts
{
    public long AnonHugePagesBytesOverRange { get; init; }

    public long RssBytesOverRange { get; init; }

    public int SmapsEntriesOverRange { get; init; }

    public int NumaMapsEntryCount { get; init; }

    public Dictionary<string, object?> PagesByNode { get; init; } = new();
}
