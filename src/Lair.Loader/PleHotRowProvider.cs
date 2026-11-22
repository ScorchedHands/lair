using System.Globalization;
using System.Runtime.InteropServices;

using Lair.Contracts;

namespace Lair.Loader;

/// <summary>
/// Which backing a hot-row arena asks the host for. <see cref="TransparentHugePage"/> is an ask, not a promise: the provider madvise's its anonymous range
/// and reports what actually landed in <see cref="PleHotRowProvider.BackingFacts"/>, so a host whose THP posture cannot back it still gets residency with the
/// shortfall named. Explicit hugetlb pages are never reserved or required — where pre-reserved pages happen to be free they may land there, and every other host stays on transparent backing.
/// </summary>
public enum PleHotRowBackMode
{
    TransparentHugePage,
    PlainCommitted,
}

/// <summary>
/// What a hot-row admission did against its ceiling: how many rows the selection asked for, how many whole packed rows the cap allows at this table's row cost,
/// and which of those actually landed in residency. A selection the clamp cut is reported as refused rather than silently narrowed — an access-scheduling policy
/// that quietly dropped rows would make a measured hit rate describe something other than what ran. The allowed-row arithmetic is the same ceiling division
/// <see cref="Lair.Models.PleResidencyGate"/> applies to declared budgets, so admission and budget gating can never disagree about how many rows fit.
/// </summary>
public sealed record PleHotRowAdmission
{
    /// <summary>The byte ceiling this provider may hold resident.</summary>
    public required long CapBytes { get; init; }

    /// <summary>Packed bytes one admitted row costs on this table's quant format.</summary>
    public required int RowStrideBytes { get; init; }

    /// <summary>How many rows the selection asked to hold resident.</summary>
    public required long RequestedRows { get; init; }

    /// <summary>How many whole packed rows this ceiling allows in total. May state zero when even one row's bytes exceed it.</summary>
    public required long AllowedRows { get; init; }

    /// <summary>Rows actually admitted into residency.</summary>
    public required long AdmittedRows { get; init; }

    /// <summary>Rows the selection asked for that the ceiling refused — zero when the cap already fit everything.</summary>
    public long RefusedByClampRows => RequestedRows - AdmittedRows;

    /// <summary>Packed bytes now held resident across every admitted row.</summary>
    public required long ResidentBytes { get; init; }

    /// <summary>The clamp as the text a diagnostic reports: ceiling, per-row cost, and requested against allowed in rows.</summary>
    public string Describe() =>
        $"cap {CapBytes:N0} bytes at {RowStrideBytes} packed bytes/row allows {AllowedRows:N0} rows — selection asked for {RequestedRows:N0}, residency holds {AdmittedRows:N0}";
}

/// <summary>
/// A frequency sketch over gathered PLE row indices: the only access-scheduling input DN-M7-10's cheapest variant needs, because n-gram slot addresses are
/// deterministic from the token sequence and this class answers which of them deserve residency. Counts come from observing a real gather stream; ranks order by
/// observed frequency with ascending row index as tiebreak, so two identical streams rank identically across runs — policy comparisons stay reproducible without any learned predictor in play.
/// </summary>
public sealed class PleFrequencySketch
{
    private readonly Dictionary<long, long> _counts = new();

    /// <summary>Fold one gathered row index into the sketch.</summary>
    public void Observe(long rowIndex)
    {
        if (rowIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rowIndex), "a PLE gather resolves rows by non-negative table indices");
        }

        _counts.TryGetValue(rowIndex, out long seen);
        _counts[rowIndex] = seen + 1;
        TotalObservations++;
    }

    /// <summary>Distinct row indices this sketch has observed.</summary>
    public int DistinctRows => _counts.Count;

    /// <summary>Total observations folded into the sketch across every gathered position.</summary>
    public long TotalObservations { get; private set; }

    /// <summary>The rows ranked for hot-slot selection: highest observed count first, ties broken by ascending row index so ranking is a function of the stream alone.</summary>
    public List<long> RankedByFrequency()
    {
        List<KeyValuePair<long, long>> entries = [.. _counts];
        entries.Sort((left, right) =>
        {
            int byCount = right.Value.CompareTo(left.Value); // higher frequency first: a row gathered more often saves the next cold gather.
            return byCount != 0 ? byCount : left.Key.CompareTo(right.Key); // equal counts break deterministically on the table index itself.
        });

        List<long> ranked = new(entries.Count);
        foreach (KeyValuePair<long, long> entry in entries)
        {
            ranked.Add(entry.Key);
        }

        return ranked;
    }

    /// <summary>The observed count for one row — zero when this sketch never saw it.</summary>
    public long CountOf(long rowIndex) => _counts.TryGetValue(rowIndex, out long seen) ? seen : 0L;
}

/// <summary>
/// A hot-row PLE provider: the exact selected rows of a n-gram table copied once into an aligned anonymous arena under a declared byte ceiling, with every other
/// row resolved through the cold backing store. This is DN-M7-6's residency half as a first implementation — private copies keep gathered bytes bit-for-bit identical
/// to the shard they came from (admission parity-checks each copy against its source span before it counts as resident), and hits versus misses are tallied so an
/// access-scheduling measurement can attribute throughput to what actually stayed resident. On Linux the arena is a <c>mmap(MAP_PRIVATE|MAP_ANONYMOUS)</c> range madvise'd with
/// <c>MADV_HUGEPAGE</c>; on every other host it is plain committed aligned memory and the requested backing reports as not applied rather than emulated. The provider never reads
/// the whole table: admission copies exactly its selection, and a row outside the table's declared row space reaches only the cold source's own refusal.
/// </summary>
public sealed unsafe class PleHotRowProvider : IDisposable
{
    private const int PageSizeHint = 4096;

    // Admission ordering is internal: rows land in residency sorted ascending so membership resolves by binary search without a per-row hash of its own.
    private long[] _hotRows = [];

    private byte* _arenaBase;
    private readonly nuint _capacityBytes;
    private bool _disposed;
    private long _hitsTotal;
    private long _missesTotal;
    private long _copiedBytesTotal;
    private long _parityCheckedRows;

    public PleHotRowProvider(PleRowSource coldBacking, long maxResidentBytes, PleHotRowBackMode backMode = PleHotRowBackMode.TransparentHugePage)
    {
        ArgumentNullException.ThrowIfNull(coldBacking);

        ArgumentOutOfRangeException.ThrowIfLessThan(maxResidentBytes, 0L);

        ColdBacking = coldBacking;
        RowStrideBytes = coldBacking.RowStrideBytes;
        MaxResidentBytes = maxResidentBytes;
        RequestedBackMode = backMode;

        // The arena reserves address space only: pages land on first touch during admission copies, which is what makes a NUMA-bound admission thread decide where they live.
        nuint capacity = checked((nuint)Math.Max(PageSizeHint, maxResidentBytes));
        _capacityBytes = capacity;

        if (OperatingSystem.IsLinux())
        {
            long mapped = LinuxPlacementInterop.MmapAnonymous((long)capacity);

            if (mapped == -1)
            {
                throw new IOException($"hot-row arena: anonymous mmap of {capacity} bytes failed on this host");
            }

            _arenaBase = (byte*)mapped;
        }
        else
        {
            _arenaBase = (byte*)NativeMemory.AlignedAlloc(capacity, PageSizeHint);
        }

        bool hugePageAskSucceeded = false;

        if (backMode == PleHotRowBackMode.TransparentHugePage && OperatingSystem.IsLinux())
        {
            hugePageAskSucceeded = LinuxPlacementInterop.MadviseHugePage((nint)_arenaBase, capacity);
        }

        BackingFacts = new Dictionary<string, object?>
        {
            ["requested_back_mode"] = backMode.ToString(),
            ["host_family"] = OperatingSystem.IsLinux() ? "linux" : "other",
            ["anonymous_mmap_applied"] = OperatingSystem.IsLinux(),
            ["madvise_hugepage_requested"] = backMode == PleHotRowBackMode.TransparentHugePage && OperatingSystem.IsLinux(),
            ["madvise_hugepage_succeeded"] = hugePageAskSucceeded,
            ["explicit_hugetlb_used"] = false, // pre-reserved hugetlb pages are never reserved or required by this provider; transparent backing is the mode it asks for.
            ["arena_capacity_bytes"] = (long)capacity,
        };
    }

    /// <summary>The cold store every non-resident row resolves through.</summary>
    public PleRowSource ColdBacking { get; }

    /// <summary>Packed bytes one resident slot holds — the same fixed stride the gather addresses rows at.</summary>
    public int RowStrideBytes { get; }

    /// <summary>The byte ceiling this provider may hold resident across every admitted row.</summary>
    public long MaxResidentBytes { get; }

    /// <summary>The backing mode this provider asked the host for.</summary>
    public PleHotRowBackMode RequestedBackMode { get; }

    /// <summary>What backing actually landed, as machine-readable facts: which asks were made and which applied on this host.</summary>
    public Dictionary<string, object?> BackingFacts { get; }

    /// <summary>The last admission's clamp answer — null until a selection has been admitted.</summary>
    public PleHotRowAdmission? Admission { get; private set; }

    /// <summary>Cumulative row resolutions answered from the resident arena since this provider opened.</summary>
    public long HitsTotal => _hitsTotal;

    /// <summary>Cumulative row resolutions that fell through to the cold backing store since this provider opened.</summary>
    public long MissesTotal => _missesTotal;

    /// <summary>Packed bytes copied into residency across every admission — first-touch traffic this provider paid once per admitted row.</summary>
    public long CopiedBytesTotal => _copiedBytesTotal;

    /// <summary>How many admitted copies were byte-for-byte checked against their source span during admission.</summary>
    public long ParityCheckedRows => _parityCheckedRows;

    /// <summary>The arena's base address, for placement accounting through the host's own maps files.</summary>
    public nint ArenaBaseAddress => (nint)_arenaBase;

    /// <summary>Packed bytes currently held resident across every admitted row — never above <see cref="MaxResidentBytes"/>.</summary>
    public long ResidentBytes { get; private set; }

    /// <summary>The arena's reserved capacity in bytes.</summary>
    public long ArenaCapacityBytes => (long)_capacityBytes;

    /// <summary>
    /// Copy a priority-ordered selection into residency under the ceiling: rows are admitted while their packed cost fits, refused beyond it, and every admitted copy is
    /// checked byte-for-byte against the cold span it was taken from before it counts as resident. Admission order decides who gets cut — the caller ranks candidates (a frequency sketch does), this method only clamps.
    /// Re-admitting replaces the hot set: the arena keeps its capacity and old slots are overwritten by the new selection's copies. A row outside the table's declared
    /// row space is refused naming the tensor and shard, exactly as the cold gather refuses it — residency never invents rows no head range declared.
    /// </summary>
    public PleHotRowAdmission AdmitHotSet(IReadOnlyList<long> rowsInPriorityOrder)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        ArgumentNullException.ThrowIfNull(rowsInPriorityOrder);

        long allowed = MaxResidentBytes <= 0 || RowStrideBytes <= 0
            ? 0L
            : Math.Min(MaxResidentBytes / RowStrideBytes, long.MaxValue); // whole packed rows against this table's own stride — the same ceiling division PleResidencyGate.FitsRows performs on a declared budget.

        List<long> admitted = [];

        foreach (long rowIndex in rowsInPriorityOrder)
        {
            if (!ColdBacking.ContainsRow(rowIndex))
            {
                throw new ArgumentOutOfRangeException(
                    nameof(rowsInPriorityOrder),
                    $"row {rowIndex} is outside the {ColdBacking.Rows}-row PLE table '{ColdBacking.Table.Tensor.Name}' — residency cannot hold a row no head range declared");
            }

            if (admitted.Count >= allowed || admitted.Count >= int.MaxValue)
            {
                break; // everything past the clamp stays out of residency and shows up as refused through RequestedRows - AdmittedRows on the admission record.
            }

            admitted.Add(rowIndex);
        }

        admitted.Sort(); // ascending residency slots: membership resolves by binary search over this array alone.

        long residentBytes = checked((long)admitted.Count * RowStrideBytes);

        byte* arena = _arenaBase;

        {
            for (int slot = 0; slot < admitted.Count; slot++)
            {
                ReadOnlySpan<byte> source = ColdBacking.ResolveRow(admitted[slot]); // the copy reads exactly this row's packed bytes out of the shard mapping.

                byte* destination = arena + checked((long)slot * RowStrideBytes);
                source.CopyTo(new Span<byte>(destination, RowStrideBytes));

                if (!new ReadOnlySpan<byte>(destination, RowStrideBytes).SequenceEqual(source)) // residency that lies about its own contents would make every measured hit a fiction: admission refuses instead.
                {
                    throw new IOException($"hot-row arena: admitted row {admitted[slot]} did not copy byte-for-byte from shard '{ColdBacking.ShardPath}'");
                }

                _parityCheckedRows++;
                _copiedBytesTotal += RowStrideBytes; // first-touch traffic this admission paid for the page(s) holding this slot.
            }
        }

        _hotRows = [.. admitted];
        ResidentBytes = residentBytes;

        Admission = new PleHotRowAdmission
        {
            CapBytes = MaxResidentBytes,
            RowStrideBytes = RowStrideBytes,
            RequestedRows = rowsInPriorityOrder.Count,
            AllowedRows = allowed,
            AdmittedRows = admitted.Count,
            ResidentBytes = residentBytes,
        };

        return Admission;
    }

    /// <summary>Whether one row currently lives in the resident arena.</summary>
    public bool IsResident(long rowIndex) => SlotOf(rowIndex) >= 0;

    private int SlotOf(long rowIndex)
    {
        long[] hotRows = _hotRows; // a single read of the array reference: admission swaps whole arrays, so a gather never sees a half-updated hot set.
        return Array.BinarySearch(hotRows, rowIndex);
    }

    /// <summary>
    /// Resolve one packed table row: resident rows come from this provider's own arena and count as hits, everything else falls through to the cold backing store and counts
    /// as a miss. The bytes are identical either way — admission checked every resident copy against its source span when it took it.
    /// </summary>
    public ReadOnlySpan<byte> ResolveRow(long rowIndex)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int slot = SlotOf(rowIndex);

        if (slot >= 0)
        {
            _hitsTotal++; // the row lives in private committed pages: no shard page fault stands behind it.
            return new ReadOnlySpan<byte>(_arenaBase + checked((long)slot * RowStrideBytes), RowStrideBytes);
        }

        _missesTotal++;
        return ColdBacking.ResolveRow(rowIndex); // a miss is exactly the gather this seam already measures, refusal included.
    }

    /// <summary>
    /// What actually backs the arena right now, read off the host's own accounting: on Linux the provider reports transparent-huge-page coverage of its range from
    /// <c>/proc/self/smaps</c> plus per-node page counts from <c>/proc/self/numa_maps</c>. A non-Linux host gets a null facts map — this is a measurement seam, not an emulation.
    /// </summary>
    public Dictionary<string, object?>? ReadBackingMeasurement()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!OperatingSystem.IsLinux())
        {
            return null;
        }

        RangeMapsFacts maps = LinuxPlacementInterop.ReadRangeMaps((nint)_arenaBase, (long)_capacityBytes);

        return new Dictionary<string, object?>
        {
            ["arena_base_hex"] = ((nuint)(nint)_arenaBase).ToString("x16", CultureInfo.InvariantCulture),
            ["arena_capacity_bytes"] = ArenaCapacityBytes,
            ["smaps_anon_hugepages_bytes_over_arena"] = maps.AnonHugePagesBytesOverRange,
            ["smaps_rss_bytes_over_arena"] = maps.RssBytesOverRange,
            ["numa_maps_entries_over_arena"] = maps.NumaMapsEntryCount,
            ["numa_pages_by_node"] = maps.PagesByNode.Count == 0 ? null : new Dictionary<string, object?>(maps.PagesByNode),
        };
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_arenaBase is not null)
        {
            if (OperatingSystem.IsLinux())
            {
                LinuxPlacementInterop.Munmap((nint)_arenaBase, _capacityBytes);
            }
            else
            {
                NativeMemory.AlignedFree(_arenaBase);
            }

            _arenaBase = null;
        }
    }
}
