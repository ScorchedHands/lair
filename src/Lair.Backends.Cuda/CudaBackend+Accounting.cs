namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // Cumulative engine-side accounting for this instance's device work, read as volatile totals by any observer that
    // samples while kernels run. The kernel-busy total is measured by the device itself: a pair of driver events sits
    // around each launch on the same stream every launch here already uses, so the answer is how long the kernel was
    // busy on the device rather than what wall time elapsed around the call site. Runtime transfer totals count only
    // copies made by compute calls — activation inputs crossing to the device and kernel outputs coming back. The
    // one-time weight upload a default posture performs when it caches a tensor, host-region registrations, and a KV
    // mirror's residency crossings each stay outside these tallies: they are residency moves, not a compute call's traffic.
    private long _kernelBusyMicrosecondsTotal;
    private long _runtimeTransferBytesTotal;
    private long _runtimeTransferCallsTotal;

    // The event pair every timed launch reuses for the instance's lifetime. It opens lazily on the first kernel-bearing
    // call, and a driver that refuses to open it degrades this accounting instead of refusing the compute path behind it.
    private IntPtr _timingStart;
    private IntPtr _timingEnd;
    private bool _deviceTimingUnavailable;

    /// <summary>Device-measured microseconds this instance's kernels have been busy since it opened — the sum over every timed launch, not wall time around the call sites.</summary>
    public long KernelBusyMicrosecondsTotal => Volatile.Read(ref _kernelBusyMicrosecondsTotal);

    /// <summary>Cumulative bytes crossed by runtime activation input copies and kernel output copies this instance has made. Persistent weight uploads, host-region registrations, and mirror residency crossings are not in this tally.</summary>
    public long RuntimeTransferBytesTotal => Volatile.Read(ref _runtimeTransferBytesTotal);

    /// <summary>Cumulative count of those runtime copy operations — one per activation input copy and one per kernel output copy, whatever their byte sizes.</summary>
    public long RuntimeTransferCallsTotal => Volatile.Read(ref _runtimeTransferCallsTotal);

    // Opens the timing pair once, under the caller's device-work gate with this instance's context current. A failed
    // creation releases whatever half opened and marks the seam unavailable: kernel accounting then reports zero busy
    // time while every launch still runs exactly as it would have without measurement.
    private void OpenTimingEvents()
    {
        if (_timingStart != 0 || _deviceTimingUnavailable)
        {
            return;
        }

        CudaResult startError = CudaNative.cuEventCreate(out IntPtr start, 0);
        if (startError != CudaResult.Success)
        {
            _deviceTimingUnavailable = true;
            return;
        }

        CudaResult endError = CudaNative.cuEventCreate(out IntPtr end, 0);
        if (endError != CudaResult.Success)
        {
            CudaNative.cuEventDestroy(start);
            _deviceTimingUnavailable = true;
            return;
        }

        _timingStart = start;
        _timingEnd = end;
    }

    // Answers whether this launch is timed, and records the opening stamp on the default stream when it is. The stamp
    // lands in stream order after every earlier copy has already crossed, so what follows measures kernel work alone.
    private bool BeginKernelTiming()
    {
        if (_timingStart == 0)
        {
            OpenTimingEvents();
        }

        if (_deviceTimingUnavailable)
        {
            return false;
        }

        CudaException.ThrowIf("cuEventRecord", CudaNative.cuEventRecord(_timingStart, IntPtr.Zero));
        return true;
    }

    // Records the closing stamp immediately after the launch is enqueued, so elapsed time covers exactly this kernel.
    private void StampKernelEnd() => CudaException.ThrowIf("cuEventRecord", CudaNative.cuEventRecord(_timingEnd, IntPtr.Zero));

    // Reads the pair once both stamps have completed — every call site folds elapsed only after its launch's context
    // synchronize returned — and adds whole microseconds to this instance's busy total. A device clock that resolves a
    // sub-microsecond kernel to zero contributes nothing rather than an invented floor.
    private void FoldKernelElapsed()
    {
        CudaException.ThrowIf("cuEventElapsedTime", CudaNative.cuEventElapsedTime(out float milliseconds, _timingStart, _timingEnd));

        long microseconds = (long)Math.Round(Math.Max(0f, milliseconds) * 1000.0);
        if (microseconds > 0)
        {
            Interlocked.Add(ref _kernelBusyMicrosecondsTotal, microseconds);
        }
    }

    // Folds one runtime copy made by a compute call into the transfer tallies: bytes and calls together, so an observer
    // sees both the volume crossed and how many operations carried it.
    private void AccountRuntimeCopy(long bytes)
    {
        Interlocked.Add(ref _runtimeTransferBytesTotal, bytes);
        Interlocked.Increment(ref _runtimeTransferCallsTotal);
    }

    private void ReleaseTimingEvents()
    {
        if (_timingStart != 0)
        {
            CudaNative.cuEventDestroy(_timingStart);
            _timingStart = IntPtr.Zero;
        }

        if (_timingEnd != 0)
        {
            CudaNative.cuEventDestroy(_timingEnd);
            _timingEnd = IntPtr.Zero;
        }
    }
}
