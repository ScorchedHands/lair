using System.Numerics;
using System.Runtime.InteropServices;

namespace Lair.Backends.Cpu;

/// <summary>
/// Platform-guarded threading helpers for the CPU backend: applying a logical-core affinity mask to the calling
/// thread and detecting how many physical cores back the machine's logical processors. Pinning is implemented on
/// Windows only via kernel32 <c>SetThreadAffinityMask</c>; on other operating systems <see cref="ApplyCoreSet"/> is
/// a documented no-op that reports itself once at startup (Linux affinity through sched_setaffinity is later
/// hardening and is deliberately not faked here). Physical-core detection reads the kernel's own topology via
/// <c>GetLogicalProcessorInformationEx(RelationProcessorCore)</c>, which returns one record per physical core with
/// its SMT siblings folded into that record's group mask, so counting records counts cores and never logical
/// processors. Any detection failure reports 0 and callers fall back to <see cref="Environment.ProcessorCount"/>.
/// </summary>
internal static unsafe partial class ThreadAffinity
{
    // Affinity masks are per-group 64-bit processor sets, so core ids above 63 cannot be expressed through this API.
    public const int MaxCoreId = 63;

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

    private static int _nonWindowsWarningShown;

    /// <summary>True when this platform has a real affinity implementation (pinning is Windows-only in this build).</summary>
    public static bool SupportsPinning => OperatingSystem.IsWindows();

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
    /// Pins the calling thread to the processor set described by <paramref name="coreMask"/> for its whole lifetime.
    /// Returns true when the mask was applied. On non-Windows hosts this is a documented no-op: it reports once via
    /// a startup warning line and returns false, leaving threads unpinned rather than approximating affinity.
    /// </summary>
    public static bool ApplyCoreSet(ulong coreMask)
    {
        if (coreMask == 0)
        {
            return true; // no processor set requested: nothing to apply
        }

        if (!OperatingSystem.IsWindows())
        {
            ReportNonWindowsNoOpOnce();
            return false;
        }

        return SetThreadAffinityMask(GetCurrentThread(), coreMask) != 0;
    }

    /// <summary>
    /// Physical cores behind this machine's logical processors, probed once and cached: the number of
    /// RelationProcessorCore records from <c>GetLogicalProcessorInformationEx</c>, which never counts SMT siblings as
    /// extra cores. Returns 0 when no detection is available (non-Windows, unsupported call shape, or API failure);
    /// callers document that fallback explicitly.
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

    private static unsafe int ProbePhysicalCoreCount()
    {
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

    private static void ReportNonWindowsNoOpOnce()
    {
        if (Interlocked.Exchange(ref _nonWindowsWarningShown, 1) == 0)
        {
            Console.Error.WriteLine(
                "lair: core-set pinning is implemented on Windows only; configured core sets are ignored on this platform.");
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

    // Sampling hook for the backend's own execution instrumentation (test-visible): returns the processor number a
    // worker block ran on. Kept here so every affinity-adjacent P/Invoke lives in one documented place.
    internal static int CurrentProcessorNumber() => (int)GetCurrentProcessorNumber();
}
