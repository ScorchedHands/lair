using System.Globalization;
using System.Runtime.InteropServices;

namespace Lair.Models;

/// <summary>
/// One node as this host's posture reports it: its position in the reported posture, how many logical processors sit on
/// it, and what that source stated about bytes — null when nothing stated per-node byte totals. A count no source stated is
/// never guessed at: an unreported byte posture reads as absent rather than zero, because a total invented from process or
/// machine state would read back as though the host had reported it.
/// </summary>
public sealed record HostNodeCapacity
{
    /// <summary>The position of this node in the reported posture, ordered by the logical processor ids each node carries.</summary>
    public required int Index { get; init; }

    /// <summary>Logical processors this node reports, counted in the units a core set or a thread quota counts in.</summary>
    public required int LogicalProcessors { get; init; }

    /// <summary>Megabytes of memory this node stated for itself, or null when no source stated one.</summary>
    public long? MemoryMb { get; init; }

    /// <summary>The logical processor ids the reported posture attributes to this node. An empty set states that the source did not report them.</summary>
    public IReadOnlyList<int> CoreIds { get; init; } = [];

    /// <summary>Whether one reported core id belongs to this node.</summary>
    public bool CarriesCoreId(int coreId) => CoreIds.Contains(coreId);

    /// <summary>The node as the text a report names it by: its position, logical processors, and byte posture.</summary>
    public string Describe() =>
        $"node {Index} ({LogicalProcessors} logical processor{(LogicalProcessors == 1 ? "" : "s")}" +
        (MemoryMb is null ? ", bytes unreported)" : $", {MemoryMb.Value} MB)");
}

/// <summary>
/// The compute posture one host derives from what its operating system reports: every node it stated, each with the logical
/// processors on it and — where the source stated them — that node's own byte totals. Byte capacity is reported here but never
/// placed per node by this posture: which node a claim draws from belongs to placement, not accounting. A host whose full topology
/// cannot be read reports one node sized by <see cref="Environment.ProcessorCount"/> and states no per-node bytes at all.
/// </summary>
public sealed record HostTopology
{
    public required IReadOnlyList<HostNodeCapacity> Nodes { get; init; }

    /// <summary>True when this posture came from a source that stated the nodes, rather than the documented fallback.</summary>
    public bool DerivedFromReportedTopology { get; init; }

    /// <summary>The logical processors across every reported node — what a compute claim counts against.</summary>
    public int TotalLogicalProcessors => Nodes.Sum(node => node.LogicalProcessors);

    /// <summary>Megabytes summed over nodes that stated byte totals, or null when any node left them unreported. A posture with no per-node byte totals states nothing rather than guessing a machine total.</summary>
    public long? ReportedMemoryMb => ReportsNodeByteTotals ? Nodes.Sum(node => node.MemoryMb!.Value) : null;

    /// <summary>True only when every reported node stated its own memory, so the summed bytes read as what was actually reported.</summary>
    public bool ReportsNodeByteTotals => Nodes.Count > 0 && Nodes.All(node => node.MemoryMb is not null);

    /// <summary>The posture as one report line: node count, logical processors, and the byte posture it may state.</summary>
    public string Describe() =>
        $"{Nodes.Count} node{(Nodes.Count == 1 ? "" : "s")} / {TotalLogicalProcessors} logical processor{(TotalLogicalProcessors == 1 ? "" : "s")}" +
        (ReportedMemoryMb is null ? " / bytes unreported" : $" / {ReportedMemoryMb.Value} MB reported");

    /// <summary>The node one core id belongs to, or null when no reported node carries it.</summary>
    public HostNodeCapacity? NodeForCoreId(int coreId) => Nodes.FirstOrDefault(node => node.CarriesCoreId(coreId));
}

/// <summary>
/// The host topology seam the compute ledger accounts against: read once per process from what the operating system reports and
/// cached for the process lifetime, exactly as physical-core detection is. Windows counts each NUMA node's logical processors out of
/// its group masks through <c>GetLogicalProcessorInformationEx(RelationNumaNode)</c>; Linux reads sysfs under
/// <c>/sys/devices/system/node</c>, where each node states both its processor list and its own memory. A posture that cannot be read
/// whole falls back to one node sized by <see cref="Environment.ProcessorCount"/> with no per-node byte totals, so a host never
/// reports detail it did not get. Windows states no per-node memory through this seam, so its byte totals stay unreported there.
/// </summary>
public static unsafe partial class HostTopologyProbe
{
    // SYSTEM_RELATIONSHIP_TYPE as these hosts answer it: relationship 1 yields one record per NUMA node carrying that node's masks.
    private const int RelationNumaNode = 1;

    // Offsets inside a node record, read from what single-group records carry here: DWORD Relationship @ +0, DWORD Size @ +4,
    // WORD GroupCount @ +30, and the group mask array at +32 with each entry holding its KAFFINITY mask at offset 0.
    private const int RecordSizeOffset = 4;
    private const int NodeGroupCountOffset = 30;
    private const int NodeMaskOffset = 32;

    // One group-mask array entry: a 64-bit processor set and the group number it belongs to.
    private const int GroupEntryStride = 16;

    private static readonly object _probeGate = new();

    // Reads stay unsynchronized on the fast path: every caller reports one posture, because the first probe through the gate publishes it and later callers report what was published rather than probing again. Several engines opening beside each other therefore account against the same reported machine.
    private static HostTopology? _cachedPosture;

    /// <summary>
    /// This process's compute posture, probed once and cached. A host whose reported topology cannot be read whole — an unsupported
    /// call shape, a mask set that does not partition this machine's logical processors, or missing sysfs files — reports the
    /// single-node fallback instead of partial detail.
    /// </summary>
    public static HostTopology Detect()
    {
        HostTopology? cached = Volatile.Read(ref _cachedPosture);
        if (cached is not null)
        {
            return cached;
        }

        lock (_probeGate)
        {
            HostTopology? published = Volatile.Read(ref _cachedPosture); // a caller that reached the gate behind another reports what was already probed.
            if (published is null)
            {
                published = Probe();
                Volatile.Write(ref _cachedPosture, published);
            }

            return published;
        }
    }

    private static HostTopology Probe()
    {
        if (OperatingSystem.IsWindows())
        {
            HostTopology? windows = ProbeWindowsNodes();
            if (windows is not null)
            {
                return windows;
            }
        }
        else if (OperatingSystem.IsLinux())
        {
            HostTopology? linux = ProbeSysfsNodes();
            if (linux is not null)
            {
                return linux;
            }
        }

        return FallbackPosture();
    }

    // The documented fallback: one node carrying this process's logical processor count and no byte totals. Nothing here claims a
    // per-node number the host did not report, so an operator reading this posture sees exactly what could not be read.
    private static HostTopology FallbackPosture() => new()
    {
        Nodes = [new HostNodeCapacity
        {
            Index = 0,
            LogicalProcessors = Math.Max(Environment.ProcessorCount, 1),
            MemoryMb = null,
            CoreIds = [],
        }],
        DerivedFromReportedTopology = false,
    };

    private static HostTopology? ProbeWindowsNodes()
    {
        if (IntPtr.Size != 8)
        {
            return null; // the record offsets below are x64-specific: a group mask's processor set is pointer-sized.
        }

        uint length = 0;

        // First call with no buffer: it fails by design and reports the required size, like every other caller of this API.
        _ = GetLogicalProcessorInformationEx(RelationNumaNode, null, ref length);
        if (length == 0)
        {
            return null;
        }

        byte* buffer = (byte*)NativeMemory.Alloc(length);
        if (buffer is null)
        {
            return null; // a tiny allocation failure means no posture at all, never a crash in the host's accounting seam.
        }

        try
        {
            if (!GetLogicalProcessorInformationEx(RelationNumaNode, buffer, ref length))
            {
                return null;
            }

            List<IReadOnlyCollection<int>> reported = [];
            uint consumed = 0;
            while (consumed + NodeMaskOffset <= length)
            {
                byte* record = buffer + consumed;
                if (*(int*)record != RelationNumaNode)
                {
                    return null; // only node records were requested: anything else means the stream is not what this posture reads.
                }

                uint recordSize = *(uint*)(record + RecordSizeOffset);
                if (recordSize < NodeMaskOffset || consumed > length - recordSize)
                {
                    return null; // a malformed or truncated stream reports no posture rather than a partial one.
                }

                ushort groupCount = *(ushort*)(record + NodeGroupCountOffset);
                if (groupCount == 0)
                {
                    return null;
                }

                SortedSet<int> processors = [];
                for (int entry = 0; entry < groupCount && NodeMaskOffset + (entry + 1) * GroupEntryStride <= recordSize; entry++)
                {
                    byte* group = record + NodeMaskOffset + entry * GroupEntryStride;
                    ulong mask = *(ulong*)group;

                    for (int bit = 0; bit < 64; bit++)
                    {
                        if (((mask >> bit) & 1UL) != 0)
                        {
                            processors.Add(*(ushort*)(group + 8) * 64 + bit); // the group this mask belongs to, joined with its processor number.
                        }
                    }
                }

                if (processors.Count == 0)
                {
                    return null;
                }

                reported.Add([.. processors]);
                consumed += recordSize;
            }

            // Windows states no per-node memory through these records, so byte totals stay unreported on this platform.
            return BuildReportedPosture([.. reported.Select(processors => (processors, (long?)null))]);
        }
        finally
        {
            NativeMemory.Free(buffer);
        }
    }

    private static HostTopology? ProbeSysfsNodes()
    {
        const string rootNodeDirectory = "/sys/devices/system/node";
        if (!Directory.Exists(rootNodeDirectory))
        {
            return null;
        }

        List<(IReadOnlyCollection<int> Processors, long? MemoryMb)> reported = [];

        foreach (string directory in Directory.GetDirectories(rootNodeDirectory, "node*").OrderBy(path => path, StringComparer.Ordinal))
        {
            IReadOnlyCollection<int>? processors = ReadSysfsProcessorIds(Path.Combine(directory, "cpulist"));
            if (processors is null || processors.Count == 0)
            {
                return null; // a node whose processor list this cannot read leaves the whole posture unreadable.
            }

            reported.Add((processors, ReadSysfsMemoryMb(Path.Combine(directory, "meminfo"))));
        }

        if (reported.Count == 0)
        {
            return null;
        }

        // sysfs states each node's own memory, so byte totals ride on the posture only where every node stated them.
        bool allNodesStatedBytes = reported.All(entry => entry.MemoryMb is not null);
        return BuildReportedPosture([.. reported.Select(entry => (entry.Processors, allNodesStatedBytes ? entry.MemoryMb : null))]);
    }

    private static IReadOnlyCollection<int>? ReadSysfsProcessorIds(string cpulistPath)
    {
        try
        {
            if (!File.Exists(cpulistPath))
            {
                return null;
            }

            // A cpulist reads as a comma-separated list of ids and inclusive ranges, for example "0-15,32-47".
            SortedSet<int> processors = [];
            foreach (string rawPart in File.ReadAllText(cpulistPath).Split(',', StringSplitOptions.RemoveEmptyEntries))
            {
                string part = rawPart.Trim();
                int dash = part.IndexOf('-');
                if (dash < 0)
                {
                    if (!int.TryParse(part, NumberStyles.Integer, CultureInfo.InvariantCulture, out int single) || single < 0)
                    {
                        return null;
                    }

                    processors.Add(single);
                    continue;
                }

                if (!int.TryParse(part[..dash], NumberStyles.Integer, CultureInfo.InvariantCulture, out int from) ||
                    !int.TryParse(part[(dash + 1)..], NumberStyles.Integer, CultureInfo.InvariantCulture, out int to) ||
                    from < 0 ||
                    to < from)
                {
                    return null;
                }

                for (int processor = from; processor <= to; processor++)
                {
                    processors.Add(processor);
                }
            }

            return processors;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FormatException)
        {
            return null;
        }
    }

    // The node's own total memory, read from the line sysfs states it on. Anything this cannot name reads as unreported rather than zero.
    private static long? ReadSysfsMemoryMb(string meminfoPath)
    {
        try
        {
            if (!File.Exists(meminfoPath))
            {
                return null;
            }

            foreach (string rawLine in File.ReadLines(meminfoPath))
            {
                const string marker = "MemTotal:";
                int at = rawLine.IndexOf(marker, StringComparison.Ordinal);
                if (at < 0)
                {
                    continue;
                }

                string[] pieces = rawLine[(at + marker.Length)..].Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (pieces.Length == 0 || !long.TryParse(pieces[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out long kilobytes) || kilobytes < 0)
                {
                    return null;
                }

                return kilobytes / 1024;
            }

            return null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FormatException)
        {
            return null;
        }
    }

    // A reported posture is accepted only when the nodes partition this machine's logical processors exactly: every id from 0 to
    // Environment.ProcessorCount appears on one node, and no node names an id outside that range. Anything else reports nothing, so a
    // partial or misread mask set never becomes per-node capacity someone later accounts against. Node positions follow the ids each
    // node carries, lowest first, which is what a declared per-node override keys on.
    private static HostTopology? BuildReportedPosture(IReadOnlyList<(IReadOnlyCollection<int> Processors, long? MemoryMb)> reported)
    {
        int processorCount = Math.Max(Environment.ProcessorCount, 1);
        if (reported.Count == 0)
        {
            return null;
        }

        HashSet<int> claimed = [];
        foreach ((IReadOnlyCollection<int> processors, long? _) in reported)
        {
            foreach (int processor in processors)
            {
                if (processor < 0 || processor >= processorCount)
                {
                    return null; // an id outside this machine's logical processors means the read is not a posture anyone can account against.
                }

                if (!claimed.Add(processor))
                {
                    return null; // one logical processor claimed by two nodes means the topology was misread, not multi-homed.
                }
            }
        }

        if (claimed.Count != processorCount)
        {
            return null; // a logical processor no node claims means the topology was read only in part.
        }

        List<HostNodeCapacity> postureNodes = [];
        foreach ((IReadOnlyCollection<int> nodeProcessors, long? nodeMemoryMb) in reported.OrderBy(node => node.Item1.Min()))
        {
            postureNodes.Add(new HostNodeCapacity
            {
                Index = postureNodes.Count,
                LogicalProcessors = nodeProcessors.Count,
                MemoryMb = nodeMemoryMb,
                CoreIds = [.. nodeProcessors.OrderBy(processor => processor)],
            });
        }

        return new HostTopology { Nodes = [.. postureNodes], DerivedFromReportedTopology = true };
    }

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool GetLogicalProcessorInformationEx(int relationshipType, byte* buffer, ref uint returnedLength);
}
