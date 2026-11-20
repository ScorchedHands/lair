using System.Runtime.InteropServices;

using Lair.Contracts;

namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // CU_STREAM_NON_BLOCKING: an instance stream never implicitly rendezvous with the legacy NULL/default stream, so two
    // backends' kernels and staging copies queue and run independently on the shared primary context. The private _gate
    // still serializes single-flight ordering within one model; it no longer gates anything across instances.
    private const uint StreamNonBlocking = 0x1;

    // This instance's dedicated non-blocking compute/copy stream: every steady-state launch, staging copy, and timing
    // event on this seam carries the handle, never IntPtr.Zero. Residency moves made under EnterDeviceWork (a weight
    // tensor's one-time upload, an IQ scratch table fill, a mirror flush) keep their synchronous legacy-stream copies —
    // each completes before its call returns, so stream-ordering contracts against later launches hold unchanged.
    private IntPtr _stream;

    // Count of instance streams currently created across the process: opened backends plus any not yet disposed. The
    // concurrency gate reads it as a structural fact that every open CudaBackend carries exactly one live stream; no
    // production path consults it and it names no user-facing knob.
    private static long _liveInstanceStreams;

    // Cached pinned host staging arenas for steady-state activation traffic: the input arena holds this call's activation
    // bytes before their async upload, the output arena receives the kernel result through an async download enqueued on
    // the instance stream, and only after the per-stream join does the answer land in the caller's span. Pinned staging is
    // what lets both directions queue without a blocking pageable-copy rendezvous inside the driver; the arenas grow with
    // the largest shape seen and stay private to this instance. Production calls serialize through _gate, so one arena set
    // never serves two outstanding pipelines on the same instance — only the enqueue seam below can leave work pending,
    // and every pipeline it opens stages in its own pinned arrays instead of these arenas.
    private nint _pinnedIn;
    private nuint _pinnedInBytes;
    private nint _pinnedOut;
    private nuint _pinnedOutBytes;

    // The probe event the pending-check records behind the lane's queue. Opened lazily, released with the timing pair.
    private IntPtr _pendingProbe;

    /// <summary>This instance's dedicated non-blocking stream handle — zero until the backend opens it, and never the legacy NULL/default stream. Test/measurement visibility only.</summary>
    internal IntPtr ComputeStream => _stream;

    /// <summary>Streams currently live across all open CudaBackend instances in this process: one per opened backend that has not been disposed. Test/measurement visibility only.</summary>
    internal static long LiveInstanceStreams => Volatile.Read(ref _liveInstanceStreams);

    // Opens the instance stream beside the module load, under the constructor's context-current window so a creation
    // failure takes the same module-unload and context-release unwind every other startup failure walks.
    private void CreateInstanceStream()
    {
        CudaException.ThrowIf("cuStreamCreate", CudaNative.cuStreamCreate(out _stream, StreamNonBlocking));

        try
        {
            // Steady-state transient buffers ride this stream through the device's default allocation pool. The pool's
            // release threshold is raised once so a lane join returns freed chunks to the pool rather than to the OS,
            // which is what keeps a per-call alloc/copy/launch/free cycle from re-doing physical allocation each time.
            CudaException.ThrowIf("cuDeviceGetDefaultMemPool", CudaNative.cuDeviceGetDefaultMemPool(out IntPtr pool, _device));

            nuint releaseThreshold = PoolReleaseThresholdBytes;
            unsafe
            {
                CudaException.ThrowIf("cuMemPoolSetAttribute", CudaNative.cuMemPoolSetAttribute(pool, MemPoolAttrReleaseThreshold, &releaseThreshold));
            }

            Interlocked.Increment(ref _liveInstanceStreams);
        }
        catch
        {
            // nothing allocatable can fail here; the guard exists so a half-open stream never counts as live.
            CudaNative.cuStreamDestroy(_stream);
            _stream = IntPtr.Zero;
            throw;
        }
    }

    // CU_MEMPOOL_ATTR_RELEASE_THRESHOLD (4 in the driver API's CUmemPool_attribute): bytes of unused pool memory the
    // driver keeps after a lane join instead of releasing to the OS. Sized beside the largest steady-state transient
    // this seam stages (a batched activation window plus its result), so the common shapes recycle in-pool and a larger
    // shape still falls back to a physical allocation.
    private const int MemPoolAttrReleaseThreshold = 4;

    private const nuint PoolReleaseThresholdBytes = 256 * 1024 * 1024;

    private void DestroyInstanceStream()
    {
        if (_stream == IntPtr.Zero)
        {
            return;
        }

        // Drain before destroying: a stream destroy implicitly joins its own pending work, but syncing first keeps the
        // failure report naming cuStreamSynchronize rather than surfacing mid-teardown.
        CudaException.ThrowIf("cuStreamSynchronize", CudaNative.cuStreamSynchronize(_stream));
        CudaException.ThrowIf("cuStreamDestroy_v2", CudaNative.cuStreamDestroy(_stream));

        _stream = IntPtr.Zero;
        Interlocked.Decrement(ref _liveInstanceStreams);
    }

    // Joins this instance's own stream instead of the whole context: every steady-state call site awaits exactly the lane
    // it enqueued, so a second backend's kernels keep running while this one drains. Callers hold their device-work gate
    // with this instance's context current (RunGemv/RunGemm/RunGemvQ8K under _gate; step paths inside EnterDeviceWork).
    internal void SyncInstanceStream() => CudaException.ThrowIf("cuStreamSynchronize", CudaNative.cuStreamSynchronize(_stream));

    // Whether everything this instance's stream has enqueued is still in flight. Test/measurement visibility only: the
    // concurrency gate reads it to prove two instances' device work stands pending together, and no production path does.
    // The answer comes from a probe event recorded behind everything already queued on the lane: cuEventQuery reports
    // NOT_READY until every earlier launch and copy has completed. (A bare cuStreamQuery proved unreliable here — it
    // answered Success while a just-submitted millisecond launch was still running.)
    internal bool InstanceStreamPending()
    {
        ThrowIfDisposed();

        lock (_gate)
        {
            SetCurrentContext();

            if (_pendingProbe == 0)
            {
                CudaException.ThrowIf("cuEventCreate", CudaNative.cuEventCreate(out _pendingProbe, 0));
            }

            CudaException.ThrowIf("cuEventRecord", CudaNative.cuEventRecord(_pendingProbe, _stream));

            return CudaNative.cuEventQuery(_pendingProbe) == CudaResult.NotReady;
        }
    }

    // Grows one pinned arena to at least `bytes`, keeping the previous capacity when it already fits so a steady-state
    // shape reallocs never more than its first call. cuMemAllocHost answers page-aligned pinned memory, which both async
    // copy directions stage against without a pageable rendezvous.
    private void EnsurePinnedIn(nuint bytes) => EnsurePinnedArena(ref _pinnedIn, ref _pinnedInBytes, bytes);

    private void EnsurePinnedOut(nuint bytes) => EnsurePinnedArena(ref _pinnedOut, ref _pinnedOutBytes, bytes);

    private static unsafe void EnsurePinnedArena(ref nint arena, ref nuint capacity, nuint bytes)
    {
        if (capacity >= bytes && arena != 0)
        {
            return;
        }

        if (arena != 0)
        {
            CudaException.ThrowIf("cuMemFreeHost", CudaNative.cuMemFreeHost((void*)arena));
            arena = 0;
            capacity = 0;
        }

        nuint grown = Math.Max(bytes, 65_536); // a floor so the common activation shapes never churn the allocator.
        CudaException.ThrowIf("cuMemAllocHost_v2", CudaNative.cuMemAllocHost(out void* allocated, grown));
        arena = (nint)allocated;
        capacity = grown;
    }

    private unsafe void FreePinnedStaging()
    {
        if (_pinnedIn != 0)
        {
            CudaException.ThrowIf("cuMemFreeHost", CudaNative.cuMemFreeHost((void*)_pinnedIn));
            _pinnedIn = 0;
            _pinnedInBytes = 0;
        }

        if (_pinnedOut != 0)
        {
            CudaException.ThrowIf("cuMemFreeHost", CudaNative.cuMemFreeHost((void*)_pinnedOut));
            _pinnedOut = 0;
            _pinnedOutBytes = 0;
        }
    }

    /// <summary>
    /// A GEMV enqueued on this instance's stream without an implicit join: the whole pipeline — pinned activation upload,
    /// timed kernel launch, async result download into a pinned staging array — is submitted and left pending until the
    /// caller awaits it. This seam exists so the concurrency gate can submit two instances' device work before awaiting
    /// either (the posture MVP6.1 measures) and observe both streams busy at once; production compute calls keep their
    /// own internal join through <see cref="Gemv"/>/<see cref="GemvBatch"/> and never route through here. The launch
    /// carries the same function handle, ceil(rows / KernelThreads) grid geometry, and parameter order every GEMV on this
    /// seam uses, so a pipeline answer is bit-identical to the production call over the same bytes. Each pipeline stages
    /// its activation input and result in pinned arrays it owns for its whole pending lifetime, never through the shared
    /// arenas, so several pipelines may stand enqueued on one instance at once. The sub-4-bit Q8_K family keeps its
    /// host-side activation quantization in the production path only; this seam refuses those formats rather than re-stage
    /// their tables.
    /// </summary>
    internal GemvPipeline BeginGemv(WeightRef w, ReadOnlySpan<float> activations)
    {
        ThrowIfDisposed();

        (int rows, int cols) = ValidateShape(w);
        if (activations.Length != cols)
        {
            throw new ArgumentException($"Input vector length {activations.Length} must equal the weight column count {cols}.", nameof(activations));
        }

        IntPtr function = KernelFor(w.Format); // refuses formats with no kernel before any device work starts.
        if (w.Format is QuantFormat.IQ2_XXS or QuantFormat.IQ2_XS or QuantFormat.IQ2_S or QuantFormat.IQ3_XXS or QuantFormat.IQ3_S)
        {
            throw new NotSupportedException($"The enqueue pipeline seam covers the FP-activation GEMV family only; format {w.Format} dots against a host-side Q8_K quantization in the production path and is refused here.");
        }

        lock (_gate)
        {
            SetCurrentContext(); // same re-bind every launch does: the driver API keeps the current context per thread.

            nuint weightBytes = checked((nuint)(rows * w.Shape.RowStrideBytes(w.Format)));
            ulong deviceWeights;
            unsafe
            {
                fixed (byte* wBytes = w.Bytes)
                {
                    deviceWeights = _weightResidency == CudaWeightResidency.HostMapped
                        ? EnsureHostMappedWeights((nint)wBytes, weightBytes)
                        : EnsureDeviceWeights((nint)wBytes, weightBytes);
                }
            }

            bool iqTab = w.Format is QuantFormat.IQ4_NL or QuantFormat.IQ4_XS; // these two kernels look every nibble up in the shared scratch table.
            if (iqTab)
            {
                EnsureIqScratch();
            }

            nuint xBytes = checked((nuint)cols * sizeof(float));
            nuint outBytes = checked((nuint)rows * sizeof(float));

            // Each pipeline owns its pinned staging arrays for its whole pending lifetime: the async upload reads the input
            // array and the async download writes the output array, and both stay pinned until Await joins this stream.
            float[] stagingIn = activations.ToArray();
            GCHandle inPin = GCHandle.Alloc(stagingIn, GCHandleType.Pinned);

            ulong deviceX = 0;
            ulong deviceOut = 0;
            try
            {
                CudaException.ThrowIf("cuMemAllocAsync", CudaNative.cuMemAllocAsync(out deviceX, xBytes, _stream));
                CudaException.ThrowIf("cuMemAllocAsync", CudaNative.cuMemAllocAsync(out deviceOut, outBytes, _stream));

                float[] stagingOut = new float[rows];
                GCHandle outPin = GCHandle.Alloc(stagingOut, GCHandleType.Pinned);

                unsafe
                {
                    // AddrOfPinnedObject answers the array's element address; the handle value itself is not a data pointer.
                    byte* inHost = (byte*)inPin.AddrOfPinnedObject();
                    CudaException.ThrowIf("cuMemcpyHtoDAsync_v2", CudaNative.cuMemcpyHtoDAsync(deviceX, inHost, xBytes, _stream));

                    AccountRuntimeCopy((long)xBytes); // this pipeline's activation input crossing the seam.

                    uint rowsU = (uint)rows;
                    uint colsU = (uint)cols;
                    uint countU = 1;
                    ulong tabBase = _tabIq4Values;
                    void** kernelParams = stackalloc void*[iqTab ? 7 : 6];
                    kernelParams[0] = &deviceWeights;
                    kernelParams[1] = &deviceX;
                    kernelParams[2] = &deviceOut;
                    kernelParams[3] = &rowsU;
                    kernelParams[4] = &colsU;
                    kernelParams[5] = &countU;

                    if (iqTab)
                    {
                        kernelParams[6] = &tabBase; // the scratch table cursor rides last, matching every other launch parameter order on this seam.
                    }

                    uint blocks = (uint)((rows + KernelThreads - 1) / KernelThreads);
                    bool timed = BeginKernelTiming(); // device-measured busy time for exactly this launch, when the seam is open here
                    CudaException.ThrowIf("cuLaunchKernel", CudaNative.cuLaunchKernel(function, blocks, 1, 1, KernelThreads, 1, 1, 0, _stream, kernelParams, null));
                    if (timed)
                    {
                        StampKernelEnd(); // the closing stamp rides immediately behind this launch on the same lane, exactly as every production GEMV brackets it
                    }

                    byte* outHost = (byte*)outPin.AddrOfPinnedObject();
                    CudaException.ThrowIf("cuMemcpyDtoHAsync_v2", CudaNative.cuMemcpyDtoHAsync(outHost, deviceOut, outBytes, _stream));

                    // Both buffers are released back into this lane's pool behind the download that last touched them:
                    // stream order keeps the copy legal, and a later pipeline on the same lane may recycle the chunks
                    // while this one still stands pending — its answer already lives in the pinned staging array.
                    CudaException.ThrowIf("cuMemFreeAsync", CudaNative.cuMemFreeAsync(deviceOut, _stream));
                    CudaException.ThrowIf("cuMemFreeAsync", CudaNative.cuMemFreeAsync(deviceX, _stream));
                    deviceOut = 0;
                    deviceX = 0;

                    return new GemvPipeline(this, stagingOut, inPin, outPin, timed);
                }
            }
            catch
            {
                // A half-opened pipeline drains its own lane before releasing staging and buffers: pinned arrays must not
                // free while an enqueued copy still reads or writes them, so the stream is joined first whenever any part
                // of the enqueue landed. A pool chunk is released with the classic free only after that explicit join.
                // The un-timed unwind contributes nothing to the kernel-busy tally.
                if (deviceX != 0 || deviceOut != 0)
                {
                    CudaNative.cuStreamSynchronize(_stream);

                    if (deviceOut != 0)
                    {
                        CudaNative.cuMemFree(deviceOut);
                    }

                    if (deviceX != 0)
                    {
                        CudaNative.cuMemFree(deviceX);
                    }
                }

                inPin.Free();

                throw;
            }
        }
    }

    /// <summary>One enqueued GEMV pipeline on the instance stream: pending device work plus its pinned staging, released together at <see cref="Await"/>. The pipeline's device buffers already returned to the lane's pool behind the enqueued download, so awaiting joins the lane and releases the staging only.</summary>
    internal sealed class GemvPipeline
    {
        private readonly CudaBackend _backend;
        private readonly float[] _stagingOut;
        private readonly GCHandle _inPin;
        private readonly GCHandle _outPin;

        internal readonly bool _timed;

        // internal rather than private so the owning backend's AwaitPipeline can mark the join done exactly once.
        internal bool _awaited;

        internal GemvPipeline(CudaBackend backend, float[] stagingOut, GCHandle inPin, GCHandle outPin, bool timed)
        {
            _backend = backend;
            _stagingOut = stagingOut;
            _inPin = inPin;
            _outPin = outPin;
            _timed = timed;
        }

        /// <summary>The result rows, valid once the pipeline has been awaited.</summary>
        public float[] Output => _stagingOut;

        /// <summary>Awaits this pipeline's lane: joins the instance stream, folds the device-measured launch time into the kernel-busy tally exactly as a production GEMV does, downloads nothing further (the async copy already wrote the staging array), and releases the staging. Idempotent.</summary>
        public void Await()
        {
            if (_awaited)
            {
                return; // already awaited: one logical join per pipeline.
            }

            _backend.AwaitPipeline(this);
        }

        internal void ReleaseStaging()
        {
            if (_inPin.IsAllocated)
            {
                _inPin.Free();
            }

            if (_outPin.IsAllocated)
            {
                _outPin.Free();
            }
        }
    }

    private void AwaitPipeline(GemvPipeline pipeline)
    {
        ThrowIfDisposed();

        lock (_gate)
        {
            SetCurrentContext();

            SyncInstanceStream(); // joins only this instance's stream; the other lane keeps running untouched.

            if (pipeline._timed)
            {
                FoldKernelElapsed(); // both stamps have completed by the time this instance's stream synchronize returns.
            }

            AccountRuntimeCopy(pipeline.Output.Length * sizeof(float)); // this pipeline's kernel output crossing back to host memory.

            pipeline._awaited = true;

            pipeline.ReleaseStaging();
        }
    }
}
