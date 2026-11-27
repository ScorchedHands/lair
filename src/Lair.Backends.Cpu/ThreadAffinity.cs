using System.Numerics;
using System.Runtime.InteropServices;

namespace Lair.Backends.Cpu;

/// <summary>
/// Platform-guarded threading helpers for the CPU backend: applying a logical-core affinity mask to the calling thread and detecting how many physical cores back the machine's logical processors. Windows pins through kernel32 <c>SetThreadAffinityMask</c>; Linux applies processor sets through libc <c>sched_setaffinity</c> (pid 0, which is the calling thread exactly as sched_setaffinity(2) documents it — no tid bookkeeping needed for worker-thread binding). On any other operating system <see cref="ApplyCoreSet"/> remains a documented no-op that reports itself once at startup and leaves threads unpinned rather than approximating affinity. Physical-core detection reads the kernel's own topology: on Windows one record per physical core from <c>GetLogicalProcessorInformationEx(RelationProcessorCore)</c>, which folds SMT siblings into each record; on Linux distinct SMT sibling groups read out of sysfs by <see cref="WorkerPlacement"/>, restricted to the processor set sched_getaffinity says this process may run on (a container's cpuset narrows detection, and a host whose files cannot be read reports 0). Any detection failure reports 0 and callers fall back to <see cref="Environment.ProcessorCount"/>.
/// </summary>
internal static unsafe partial class ThreadAffinity
{
    // Affinity masks are per-group 64-bit processor sets on Windows, so an explicit core set above id 63 cannot be expressed through the single-word API this seam has always carried. Multi-word processor masks (BuildProcessorMask/ApplyProcessorMask) carry the Linux auto-placement seam past one group; that path never crosses MaxCoreId on Windows.
    public const int MaxCoreId = 63;

    // cpu_set_t as one kernel word set of 1024 bits — the same size the loader's measurement lane builds, and enough for any host whose ids sysfs names below 1024. A mask wider than this reports unappliable rather than truncating into a wrong pin.
    private const int MaxAffinityBytes = 16 * sizeof(ulong);

    private const int RelationProcessorCore = 0;

    // SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX (x64): DWORD Relationship @ +0, DWORD Size @ +4, union member @ +8.
    // The core record's PROCESSOR_RELATIONSHIP: BYTE Flags @ +0 (LTP_PC_SMT when the core has SMT siblings),
    // WORD GroupCount @ +22, GROUP_AFFINITY[GroupCount] @ +24 with each 16-byte entry carrying its 64-bit Mask at
    // offset 0 and the group number at offset 8. A single-group core record is therefore always 48 bytes.
    private const int RecordSizeOffset = 4;
    private const int CoreGroupCountOffset = 30;
    private const int CoreMaskOffset = 32;

    // PROCESSOR_RELATIONSHIP.Flags carries LTP_PC_SMT when a core has more than one logical processor; the probe
    // below counts records rather than testing that flag, because SMT siblings are already folded into each record.

    private static readonly object _probeGate = new();

    // Probe cache for the process lifetime: NotProbed until first call, then a physical-core count or -1 when
    // detection is unavailable. Reads are unsynchronized on purpose; one extra probe under the gate is harmless.
    private const int NotProbed = -2;

    private static int _physicalCoreProbe = NotProbed;

    // Allowed-mask cache for Linux: null states "this platform has no affinity restriction to report" (Windows, macOS, or a failed probe) — callers treat it as unrestricted rather than as an empty mask.
    private static IReadOnlyList<int>? _allowedProcessorIds;
    private static int _allowedProbeDone;

    private static int _unsupportedPlatformWarningShown;

    /// <summary>True when this host's placement seam can derive a processor-group core set for unpinned models (Windows today). Mask application itself exists on Windows and Linux: the CPU backend places its own workers from detected physical cores on Linux instead of deriving a node mask — see WorkerPlacement and CpuBackend.</summary>
    public static bool SupportsPinning => OperatingSystem.IsWindows();

    /// <summary>Whether this platform can apply a processor set to worker threads at all (Windows and Linux today; anything else is the documented no-op).</summary>
    public static bool SupportsMaskApplication => OperatingSystem.IsWindows() || OperatingSystem.IsLinux();

    /// <summary>
    /// Builds the 64-bit processor mask for a logical core set. Core ids must lie within one processor group;
    /// anything outside [0, 63] is rejected so an out-of-group id never silently pins to the wrong processor.
    /// </summary>
    public static ulong BuildCoreMask(IReadOnlySet<int> coreSet)
    {
        ArgumentNullException.ThrowIfNull(coreSet);

        ulong mask = 0;
        foreach (int core in coreSet)
        {
            if (core < 0 || core > MaxCoreId)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(coreSet),
                    core,
                    $"Core id {core} is outside the affinity range [0, {MaxCoreId}] supported by processor masks.");
            }

            mask |= 1UL << core;
        }

        return mask;
    }

    /// <summary>
    /// Builds a multi-word processor mask covering ids beyond one 64-bit group: word w covers ids [w*64, w*64+63). This is the shape Linux auto placement hands to sched_setaffinity on hosts with more than 64 logical processors; Windows keeps its single-group contract and refuses anything wider rather than pinning a high id through the wrong group.
    /// </summary>
    public static IReadOnlyList<ulong> BuildProcessorMask(IReadOnlyCollection<int> processorIds)
    {
        ArgumentNullException.ThrowIfNull(processorIds);

        int highest = -1;
        foreach (int id in processorIds)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(id);
            if (id > highest)
            {
                highest = id;
            }
        }

        if (highest < 0)
        {
            return []; // an empty set requests no placement: nothing to express
        }

        ulong[] words = new ulong[highest / 64 + 1];
        foreach (int id in processorIds)
        {
            words[id / 64] |= 1UL << (id % 64);
        }

        return words;
    }

    /// <summary>
    /// Pins the calling thread to the processor set described by <paramref name="coreMask"/> for its whole lifetime.
    /// Returns true when the mask was applied. On Windows this is SetThreadAffinityMask; on Linux it is a one-word
    /// sched_setaffinity cpu_set_t (ids up to 63, which every explicit core set stays inside). On any other platform
    /// this remains a documented no-op: it reports once via a startup warning line and returns false, leaving threads
    /// unpinned rather than approximating affinity.
    /// </summary>
    public static bool ApplyCoreSet(ulong coreMask)
    {
        if (coreMask == 0)
        {
            return true; // no processor set requested: nothing to apply
        }

        if (OperatingSystem.IsWindows())
        {
            return SetThreadAffinityMask(GetCurrentThread(), coreMask) != 0;
        }

        if (OperatingSystem.IsLinux())
        {
            byte[] mask = new byte[MaxAffinityBytes]; // cpu_set_t: one bit per processor id, exactly the layout sched_setaffinity(2) expects. A single-group mask never reaches past word zero of this buffer.
            for (int bit = 0; bit < sizeof(ulong) * 8; bit++)
            {
                if (((coreMask >> bit) & 1UL) != 0)
                {
                    mask[bit / 8] |= (byte)(1 << (bit % 8));
                }
            }

            // pid 0 means the calling thread exactly as sched_setaffinity(2) documents it — no tid bookkeeping needed for worker-thread binding. A refused ask reads as false: placement falls back to unpinned, never to a partial pin someone else guessed.
            return SchedSetAffinity(0, (nuint)MaxAffinityBytes, mask) == 0;
        }

        ReportUnsupportedPlatformOnce();
        return false;
    }

    /// <summary>
    /// Applies a multi-word processor set to the calling thread — the seam Linux auto placement uses when selected ids cross one group. Windows applies only masks that fit its single-group contract and reports anything wider as unappliable rather than pinning through the wrong group; other platforms keep the documented no-op. An empty word list means no placement was requested and trivially succeeds, matching <see cref="ApplyCoreSet"/> with a zero mask.
    /// </summary>
    public static bool ApplyProcessorMask(IReadOnlyList<ulong> maskWords)
    {
        ArgumentNullException.ThrowIfNull(maskWords);

        if (maskWords.Count == 0)
        {
            return true; // no processor set requested: nothing to apply
        }

        if (OperatingSystem.IsWindows())
        {
            if (!AllZeroAboveFirstWord(maskWords))
            {
                return false; // ids above group zero cannot be expressed through SetThreadAffinityMask: reported as unappliable, never pinned to a wrong processor.
            }

            ulong single = maskWords[0];
            if (single == 0)
            {
                return true; // an all-zero first word asks for nothing — the same trivial accept ApplyCoreSet gives a zero mask.
            }

            return SetThreadAffinityMask(GetCurrentThread(), single) != 0;
        }

        if (!OperatingSystem.IsLinux())
        {
            ReportUnsupportedPlatformOnce();
            return false;
        }

        int bytes = maskWords.Count * sizeof(ulong);
        if (bytes > MaxAffinityBytes)
        {
            return false; // a processor set wider than the kernel word set this seam builds: reported as unappliable rather than truncated into a wrong pin.
        }

        byte[] cpuSet = new byte[MaxAffinityBytes];
        for (int word = 0; word < maskWords.Count; word++)
        {
            ulong value = maskWords[word]; // little-endian placement: bit i of word w names processor id w*64+i, the exact cpu_set_t layout.
            int offset = word * sizeof(ulong);

            for (int byteIndex = 0; byteIndex < sizeof(ulong); byteIndex++)
            {
                cpuSet[offset + byteIndex] |= (byte)(value >> (byteIndex * 8));
            }
        }

        return SchedSetAffinity(0, (nuint)MaxAffinityBytes, cpuSet) == 0;
    }

    /// <summary>
    /// Physical cores behind this machine's logical processors, probed once and cached: the number of
    /// RelationProcessorCore records from <c>GetLogicalProcessorInformationEx</c>, which never counts SMT siblings as
    /// extra cores. Returns 0 when no detection is available (unsupported call shape or API failure); callers document that fallback explicitly. On Linux, distinct sysfs sibling groups restricted to this process's affinity mask — so a container confined to part of the machine detects only its own physical cores.
    /// </summary>
    public static int DetectPhysicalCoreCount()
    {
        // Read-mostly cache: the probe runs once under a lock and every later read just returns the cached value.
        if (_physicalCoreProbe != NotProbed)
        {
            return Math.Max(_physicalCoreProbe, 0);
        }

        lock (_probeGate)
        {
            if (_physicalCoreProbe == NotProbed)
            {
                int detected = ProbePhysicalCoreCount();

                // A zero or negative probe result is cached as -1 so a failed detection never re-probes per backend.
                _physicalCoreProbe = detected > 0 ? detected : -1;
            }

            return Math.Max(_physicalCoreProbe, 0);
        }
    }

    /// <summary>
    /// The processor ids this process may run on, read once from sched_getaffinity on Linux. Null means unrestricted or unreadable (Windows, macOS, a failed probe): callers treat it as "no confinement reported", never as an empty mask that would strand every worker. A container's cpuset comes back narrowed — placement built against it can only ever shrink further.
    /// </summary>
    public static IReadOnlyList<int>? DetectAllowedProcessorIds()
    {
        if (Volatile.Read(ref _allowedProbeDone) != 0)
        {
            return Volatile.Read(ref _allowedProcessorIds);
        }

        lock (_probeGate)
        {
            if (Volatile.Read(ref _allowedProbeDone) == 0)
            {
                IReadOnlyList<int>? allowed = OperatingSystem.IsLinux() ? ProbeAllowedProcessorIds() : null;
                Volatile.Write(ref _allowedProcessorIds, allowed);
                Volatile.Write(ref _allowedProbeDone, 1);
            }

            return Volatile.Read(ref _allowedProcessorIds);
        }
    }

    private static unsafe int ProbePhysicalCoreCount()
    {
        if (OperatingSystem.IsLinux())
        {
            CpuPlacementTopology? posture = WorkerPlacement.DetectLinuxPosture();
            if (posture is null)
            {
                return 0; // sysfs unreadable: the documented fallback, never a crash in backend sizing.
            }

            IReadOnlyList<int>? allowed = posture.AllowedProcessorIds;
            int cores = 0;
            foreach (PhysicalCore core in posture.Cores)
            {
                if (allowed is null || core.SiblingIds.Any(allowed.Contains))
                {
                    cores++; // a core with no processor inside the mask this process may run on cannot back an auto worker here.
                }
            }

            return cores;
        }

        if (!OperatingSystem.IsWindows())
        {
            return 0;
        }

        // The record layout constants below are x64-specific (KAFFINITY is pointer-sized in GROUP_AFFINITY).
        if (IntPtr.Size != 8)
        {
            return 0;
        }

        uint length = 0;

        // First call with no buffer: it fails by design and reports the required size, like GetLogicalProcessorInformation.
        _ = GetLogicalProcessorInformationEx(RelationProcessorCore, null, ref length);
        if (length == 0)
        {
            return 0;
        }

        byte* buffer = (byte*)NativeMemory.Alloc(length);
        if (buffer is null)
        {
            return 0; // tiny allocation failed: treat as detection unavailable, never crash the backend
        }

        try
        {
            if (!GetLogicalProcessorInformationEx(RelationProcessorCore, buffer, ref length))
            {
                return 0;
            }

            int physicalCores = 0;
            uint consumed = 0;
            while (consumed + CoreMaskOffset <= length)
            {
                byte* record = buffer + consumed;
                if (*(int*)record != RelationProcessorCore)
                {
                    break; // only core records were requested; anything else means the stream is not what we expect
                }

                uint recordSize = *(uint*)(record + RecordSizeOffset);
                if (recordSize < CoreMaskOffset || consumed > length - recordSize)
                {
                    return physicalCores > 0 ? physicalCores : 0; // malformed or truncated stream: keep what was counted
                }

                ushort groupCount = *(ushort*)(record + CoreGroupCountOffset);
                if (groupCount == 0)
                {
                    return physicalCores > 0 ? physicalCores : 0; // malformed stream: keep what was already counted
                }

                ulong unionMask = 0;
                for (int group = 0; group < groupCount && CoreMaskOffset + group * 16 <= recordSize; group++)
                {
                    unionMask |= *(ulong*)(record + CoreMaskOffset + group * 16);
                }

                if (unionMask != 0)
                {
                    physicalCores++; // one core record per physical core; SMT siblings live inside the mask
                }

                consumed += recordSize;
            }

            return physicalCores;
        }
        finally
        {
            NativeMemory.Free(buffer);
        }
    }

    private static IReadOnlyList<int>? ProbeAllowedProcessorIds()
    {
        byte[] cpuSet = new byte[MaxAffinityBytes];

        // pid 0 reads the calling thread's affinity mask; a kernel that refuses (or cannot answer) leaves placement unrestricted rather than guessing one.
        if (SchedGetAffinity(0, (nuint)MaxAffinityBytes, cpuSet) != 0)
        {
            return null;
        }

        List<int> ids = [];
        for (int bit = 0; bit < MaxAffinityBytes * 8; bit++)
        {
            if ((cpuSet[bit / 8] & (1 << (bit % 8))) != 0)
            {
                ids.Add(bit);
            }
        }

        return ids.Count == 0 ? null : ids; // an all-zero read cannot be trusted as a confinement claim: treat it like no answer.
    }

    private static bool AllZeroAboveFirstWord(IReadOnlyList<ulong> maskWords)
    {
        for (int word = 1; word < maskWords.Count; word++)
        {
            if (maskWords[word] != 0)
            {
                return false;
            }
        }

        return true;
    }

    private static void ReportUnsupportedPlatformOnce()
    {
        if (Interlocked.Exchange(ref _unsupportedPlatformWarningShown, 1) == 0)
        {
            Console.Error.WriteLine(
                "lair: core-set pinning is implemented on Windows and Linux; configured core sets are ignored on this platform.");
        }
    }

    [LibraryImport("kernel32.dll")]
    private static partial nint GetCurrentThread();

    // Returns the thread's previous affinity mask; zero means the request failed.
    [LibraryImport("kernel32.dll", SetLastError = true)]
    private static partial ulong SetThreadAffinityMask(nint hThread, ulong dwProcessAffinityMask);

    // Logical processor number (within the current processor group) that is executing the calling thread right now.
    [LibraryImport("kernel32.dll")]
    private static partial uint GetCurrentProcessorNumber();

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static unsafe partial bool GetLogicalProcessorInformationEx(int relationshipType, byte* buffer, ref uint returnedLength);

    // Linux affinity seam — the same libc calls the loader's measurement lane already binds through. pid 0 names the calling thread; a nonzero return is the kernel refusing the mask (errno stays with Marshal).
    [LibraryImport("libc", EntryPoint = "sched_setaffinity", SetLastError = true)]
    private static partial int SchedSetAffinity(int pid, nuint cpusetsize, byte[] cpuSet);

    [LibraryImport("libc", EntryPoint = "sched_getaffinity", SetLastError = true)]
    private static partial int SchedGetAffinity(int pid, nuint cpusetsize, byte[] cpuSet);

    // Sampling hook for the backend's own execution instrumentation (test-visible): returns the processor number a worker block ran on, or -1 where this platform cannot answer through kernel32 — an unreported sample reads as absent rather than crashing a Linux pool that armed sampling. Kept here so every affinity-adjacent P/Invoke lives in one documented place.
    internal static int CurrentProcessorNumber() => OperatingSystem.IsWindows() ? (int)GetCurrentProcessorNumber() : -1;
}
