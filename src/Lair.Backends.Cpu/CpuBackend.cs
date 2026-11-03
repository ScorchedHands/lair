using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Lair.Contracts;

namespace Lair.Backends.Cpu;

/// <summary>
/// CPU compute backend: aligned arena management plus dense kernels. GEMV fuses dequantization for F16/Q8_0/Q4_K/
/// Q5_0/Q5_K/Q6_K in a single pass over the packed bytes, taking an AVX2 fast path when the runtime supports it and
/// scalar reference paths otherwise. Rows are partitioned into blocks executed by a persistent worker pool; each
/// row is computed independently so results are bit-identical regardless of thread count.
/// </summary>
public sealed unsafe class CpuBackend : IBackend
{
    // Q8_0: 32 int8 quants with one half-float block scale per 32 elements (34 bytes per block).
    private const int Qk8 = 32;

    // Q4_K super-blocks hold 256 elements in 144 bytes. The eight 6-bit group scales/mins are packed as:
    //   byte j      (0..3): low 6 bits of scale_j,        high 2 bits of scale_{j+4}
    //   byte 4+j    (0..3): low 6 bits of min_j,          high 2 bits of min_{j+4}
    //   byte 8+k    (0..3): low nibble of scale_{4+k},    high nibble of min_{4+k}
    private const int QkK = 256;

    private const int Q4KBlockBytes = 144;

    // Q5_0 blocks hold 32 five-bit quants in 22 bytes: half-float d at byte 0, four little-endian bytes of 5th
    // quant bits at bytes 2..5 (element i takes bit i of the 32-bit qh word), and 16 nibble bytes at bytes 6..21:
    // element i < 16 is the low nibble of qs[i/2], element i >= 16 the high nibble of qs[(i-16)/2].
    // w = (q - 16) * d with q in [0, 31].
    private const int Q5_0BlockBytes = 22;

    // Q5_K super-blocks hold 256 elements in 176 bytes: the Q4_K scale layout at bytes 0..15 (half d, half dmin,
    // 12 packed 6-bit group scales/mins, see QkK), 32 bytes of 5th quant bits at bytes 16..47 (group j reads bit
    // j of the window byte), and 128 nibble bytes at bytes 48..175 in the same group-pair windows as Q4_K.
    // w = d * sc_j * q - dmin * m_j with q in [0, 31].
    private const int Q5KBlockBytes = 176;

    // Q6_K super-blocks hold 256 elements in 210 bytes: 128 low-nibble bytes at 0..127, 64 bytes carrying the
    // high 2 bits of six-bit quants at 128..191, 16 signed int8 group scales at 192..207, and half-float d at
    // bytes 208..209. Sixteen 16-element groups: group g covers elements [16g, 16g+16) with scale scales[g];
    // w = d * sc_g * (q - 32) with q in [0, 63].
    private const int Q6KBlockBytes = 210;

    // Output rows per scheduling unit for threaded GEMV.
    private const int RowBlock = 64;

    // Upper bound on activation vectors per batched GEMV call.
    private const int MaxBatch = 16;

    // Below this row count the threading overhead is not worth paying.
    private const int ParallelRowThreshold = 256;

    // The engine's own thread choice (0 = auto-sized) before any model quota capped it: the ceiling a hot quota edit
    // may restore participation up to — raising a quota beyond this would be an engine-wide change, not a placement one.
    private readonly int _uncappedThreads;

    // Workers currently let participate in threaded GEMV: min(uncapped choice, current quota). Read fresh at every
    // kernel call so an apply from another thread lands on the next step boundary; lowering it leaves pool workers
    // parked and idle rather than killing them.
    private volatile int _effectiveThreads;

    // Guards lazy pool creation/growth against racing with ApplyThreadQuota calls. Forward consumers of one backend
    // are serialized by the scheduler's gate, so a swap never races an in-flight kernel job.
    private readonly object _poolGate = new();

    private readonly ulong? _coreMask;
    private bool _samplingRequested; // test-only instrumentation flag, read when the pool is created and pushed on enable
    private readonly bool _useAvx2;
    private WorkerPool? _pool;

    /// <summary>
    /// Default auto-sizing rule: one worker per physical core, SMT siblings excluded. Dense GEMV decode is memory-
    /// bandwidth-bound and two threads on one core share that core's front-end, L1/L2 caches, and execution ports, so
    /// a sibling thread adds contention rather than bandwidth; logical processor counts therefore overstate usable
    /// parallelism for this workload. Physical cores come from <see cref="ThreadAffinity.DetectPhysicalCoreCount"/>
    /// (Windows topology API, probed once per process); when detection is unavailable the count falls back to
    /// <see cref="Environment.ProcessorCount"/>, which can only overestimate and is then capped by it. An explicit
    /// nonzero thread request wins over auto-sizing, and a per-model quota caps either choice.
    /// </summary>
    public static int AutoWorkerCount => Math.Min(Math.Max(ThreadAffinity.DetectPhysicalCoreCount(), 1), Environment.ProcessorCount);

    /// <summary>
    /// Opens the CPU backend for one model placement. <paramref name="threads"/> is the global worker request (0 = auto-size by
    /// <see cref="AutoWorkerCount"/>); <paramref name="threadQuota"/> optionally caps that count to 1 or more; and a non-empty
    /// <paramref name="coreSet"/> pins every persistent pool worker to those logical cores for their whole lifetime, so all of the
    /// model's compute stays inside its configured processor set. Pinning is applied on Windows only: on other platforms it is a
    /// documented no-op that warns once at startup (Linux affinity via sched_setaffinity is later hardening). Core ids must lie
    /// within one 64-bit processor group ([0, 63]) because that is what an affinity mask can express.
    /// </summary>
    public CpuBackend(int threads = 0, IReadOnlySet<int>? coreSet = null, int? threadQuota = null, bool allowAvx2 = true)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(threads);
        if (threadQuota is < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(threadQuota), threadQuota, "A thread quota must be at least 1; omit it to leave the worker count uncapped.");
        }

        int global = threads == 0 ? AutoWorkerCount : threads;
        _uncappedThreads = global;
        _effectiveThreads = Math.Min(global, threadQuota ?? global);

        if (coreSet is { Count: > 0 })
        {
            // Validate ids before any pool exists so an out-of-group core set fails the open call instead of a worker loop.
            _coreMask = ThreadAffinity.BuildCoreMask(coreSet);
        }

        _useAvx2 = allowAvx2 && Avx2.IsSupported && Fma.IsSupported && Ssse3.IsSupported;
    }

    public string DeviceId => "cpu";

    /// <summary>Worker count this backend actually runs threaded GEMV with (dispatcher plus pool workers), after quota and auto-sizing.</summary>
    public int EffectiveWorkerCount => _effectiveThreads;

    /// <summary>
    /// Applies a new per-model thread quota at runtime: the effective worker count becomes min(engine thread choice,
    /// quota) — lowering it idles pool workers starting with the next kernel call (the scheduler's next step), and
    /// raising participation back up to the engine's own thread choice re-grows the persistent pool once if needed.
    /// Participation above <c>engine.threads</c> is not a placement decision this model owns, so a quota beyond that
    /// ceiling simply reads as the uncapped count. Rows stay independent, so output bytes are unchanged at any worker
    /// count; returns the effective worker count now in force.
    /// </summary>
    public int ApplyThreadQuota(int? threadQuota)
    {
        if (threadQuota is < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(threadQuota), threadQuota, "A thread quota must be at least 1; omit it to leave the worker count uncapped.");
        }

        lock (_poolGate)
        {
            _effectiveThreads = Math.Min(_uncappedThreads, threadQuota ?? _uncappedThreads);
            return _effectiveThreads;
        }
    }

    /// <summary>Physical cores detected on the machine by the topology probe (0 when detection is unavailable); see <see cref="AutoWorkerCount"/>.</summary>
    public static int DetectedPhysicalCoreCount => ThreadAffinity.DetectPhysicalCoreCount();

    /// <summary>
    /// Turns on lightweight per-block execution sampling: each worker block records which logical processor it ran on, readable
    /// through <see cref="ExecutionSamples"/>. Off by default so production runs pay no instrumentation cost beyond one field read.
    /// </summary>
    internal void EnableExecutionSampling() => _samplingRequested = true;

    /// <summary>The processor number of each sampled worker-block execution since sampling was enabled (empty before the first threaded call).</summary>
    internal IReadOnlyList<int> ExecutionSamples => _pool?.SampleSnapshot() ?? [];

    public MemoryHandle AllocateAligned(long bytes, int alignment)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0)
        {
            throw new ArgumentException("Alignment must be a positive power of two.", nameof(alignment));
        }

        void* pointer = NativeMemory.AlignedAlloc((nuint)bytes, (nuint)alignment);
        if (pointer is null)
        {
            throw new OutOfMemoryException($"Failed to allocate {bytes} aligned bytes.");
        }

        return new MemoryHandle(pointer);
    }

    public void FreeAligned(MemoryHandle handle)
    {
        NativeMemory.AlignedFree(handle.Pointer);
    }

    public void Gemv(ReadOnlySpan<float> x, WeightRef w, Span<float> output)
    {
        int[] dims = w.Shape.Dims;
        if (dims.Length < 2 || dims[0] <= 0 || dims[^1] <= 0)
        {
            throw new ArgumentException("Weight shape must be a matrix with positive row and column counts.", nameof(w));
        }

        int rows = dims[0];
        int cols = dims[^1];

        long strideBytes = w.Shape.RowStrideBytes(w.Format);
        if (strideBytes > int.MaxValue)
        {
            throw new ArgumentException("Row byte size exceeds the addressable span limit.", nameof(w));
        }

        long expectedBytes = rows * strideBytes;
        if ((long)w.Bytes.Length != expectedBytes)
        {
            throw new ArgumentException(
                $"WeightRef byte length is {w.Bytes.Length} but shape {rows}x{cols} in format {w.Format} " +
                $"requires {expectedBytes} bytes ({rows} rows x {strideBytes}-byte row stride).",
                nameof(w));
        }

        if (x.Length != cols)
        {
            throw new ArgumentException($"Input vector length {x.Length} must equal the weight column count {cols}.", nameof(x));
        }

        if (output.Length < rows)
        {
            throw new ArgumentException($"Output span length {output.Length} is shorter than the weight row count {rows}.", nameof(output));
        }

        int stride = (int)strideBytes;
        QuantFormat format = w.Format;

        // The threaded job carries raw pointers and ints only: fixed locals, spans, and ref structs such as
        // WeightRef cannot be captured by delegates. Pinned spans are stable for the duration of the call.
        fixed (byte* pBytes = w.Bytes)
        fixed (float* pX = x)
        fixed (float* pOut = output)
        {
            // Participation is sampled per call so a quota applied on another thread takes effect from this kernel on.
            int workers = _effectiveThreads;
            bool parallel = workers > 1 && rows >= ParallelRowThreshold;

            if (!parallel)
            {
                for (int row = 0; row < rows; row++)
                {
                    pOut[row] = RowDot(pBytes + (long)row * stride, pX, cols, format, _useAvx2);
                }

                return;
            }

            WorkerPool pool = GetOrCreatePool();
            pool.Run((rows + RowBlock - 1) / RowBlock, new RowBlockJob(pBytes, pX, pOut, stride, rows, cols, format, _useAvx2));
        }
    }

    public void GemvBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs)
    {
        int[] dims = w.Shape.Dims;
        if (dims.Length < 2 || dims[0] <= 0 || dims[^1] <= 0)
        {
            throw new ArgumentException("Weight shape must be a matrix with positive row and column counts.", nameof(w));
        }

        int rows = dims[0];
        int cols = dims[^1];

        long strideBytes = w.Shape.RowStrideBytes(w.Format);
        if (strideBytes > int.MaxValue)
        {
            throw new ArgumentException("Row byte size exceeds the addressable span limit.", nameof(w));
        }

        long expectedBytes = rows * strideBytes;
        if ((long)w.Bytes.Length != expectedBytes)
        {
            throw new ArgumentException(
                $"WeightRef byte length is {w.Bytes.Length} but shape {rows}x{cols} in format {w.Format} " +
                $"requires {expectedBytes} bytes ({rows} rows x {strideBytes}-byte row stride).",
                nameof(w));
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, MaxBatch);

        if ((long)inputs.Length != (long)count * cols)
        {
            throw new ArgumentException(
                $"Batched input length {inputs.Length} must equal count {count} times the weight column count {cols}.",
                nameof(inputs));
        }

        if ((long)outputs.Length < (long)count * rows)
        {
            throw new ArgumentException(
                $"Batched output length {outputs.Length} is shorter than count {count} times the weight row count {rows}.",
                nameof(outputs));
        }

        int stride = (int)strideBytes;
        QuantFormat format = w.Format;

        // Same raw-pointer job contract as Gemv: pinned spans stay stable for the duration of the call and every
        // claimed block finishes before WorkerPool.Run returns. Rows are independent, so row partitioning stays
        // bit-identical; each batch vector is computed by its own accumulator chain exactly like a single GEMV.
        fixed (byte* pBytes = w.Bytes)
        fixed (float* pIn = inputs)
        fixed (float* pOut = outputs)
        {
            // Participation is sampled per call so a quota applied on another thread takes effect from this kernel on.
            int workers = _effectiveThreads;
            bool parallel = workers > 1 && rows >= ParallelRowThreshold;

            if (!parallel)
            {
                for (int row = 0; row < rows; row++)
                {
                    RowBatchDot(pBytes + (long)row * stride, pIn, count, rows, cols, format, _useAvx2, pOut + row);
                }

                return;
            }

            WorkerPool pool = GetOrCreatePool();
            pool.Run((rows + RowBlock - 1) / RowBlock, new BatchRowJob(pBytes, pIn, pOut, stride, rows, cols, count, format, _useAvx2));
        }
    }

    public void DequantRow(WeightRef w, int row, Span<float> output)
    {
        int[] dims = w.Shape.Dims;
        if (dims.Length < 2 || dims[0] <= 0 || dims[^1] <= 0)
        {
            throw new ArgumentException("Weight shape must be a matrix with positive row and column counts.", nameof(w));
        }

        int rows = dims[0];
        int cols = dims[^1];

        long strideBytes = w.Shape.RowStrideBytes(w.Format);
        if (strideBytes > int.MaxValue)
        {
            throw new ArgumentException("Row byte size exceeds the addressable span limit.", nameof(w));
        }

        long expectedBytes = rows * strideBytes;
        if ((long)w.Bytes.Length != expectedBytes)
        {
            throw new ArgumentException(
                $"WeightRef byte length is {w.Bytes.Length} but shape {rows}x{cols} in format {w.Format} " +
                $"requires {expectedBytes} bytes ({rows} rows x {strideBytes}-byte row stride).",
                nameof(w));
        }

        if ((uint)row >= (uint)rows)
        {
            throw new ArgumentOutOfRangeException(nameof(row), $"Row {row} is outside the weight of {rows} rows.");
        }

        if (output.Length < cols)
        {
            throw new ArgumentException($"Output span length {output.Length} is shorter than the weight column count {cols}.", nameof(output));
        }

        fixed (byte* pBytes = w.Bytes)
        fixed (float* pOut = output)
        {
            byte* rowPtr = pBytes + (long)row * strideBytes;
            switch (w.Format)
            {
                case QuantFormat.F32:
                    new ReadOnlySpan<float>(rowPtr, cols).CopyTo(new Span<float>(pOut, cols));
                    break;
                case QuantFormat.F16:
                    if (_useAvx2)
                    {
                        DequantF16Avx2(rowPtr, pOut, cols);
                    }
                    else
                    {
                        for (int i = 0; i < cols; i++)
                        {
                            pOut[i] = ReadHalfLE(rowPtr + 2 * i);
                        }
                    }

                    break;
                case QuantFormat.Q8_0:
                    if (_useAvx2)
                    {
                        DequantQ8Avx2(rowPtr, pOut, cols);
                    }
                    else
                    {
                        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk8)
                        {
                            byte* block = rowPtr + (long)(baseIdx / Qk8) * (Qk8 + 2);
                            float d = ReadHalfLE(block);

                            int limit = Math.Min(Qk8, cols - baseIdx); // padded quants are zero: stop at the real column count.
                            for (int l = 0; l < limit; l++)
                            {
                                pOut[baseIdx + l] = (float)(sbyte)block[2 + l] * d;
                            }
                        }
                    }

                    break;
                case QuantFormat.Q4_K:
                    if (_useAvx2)
                    {
                        DequantQ4KAvx2(rowPtr, pOut, cols);
                    }
                    else
                    {
                        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
                        {
                            byte* block = rowPtr + (long)(baseIdx / QkK) * Q4KBlockBytes;
                            float d = ReadHalfLE(block);
                            float min = ReadHalfLE(block + 2);

                            for (int j = 0; j < 8; j++)
                            {
                                int subStart = baseIdx + j * 32;
                                if (subStart >= cols)
                                {
                                    break; // zero-padded tail block: no elements left to consume.
                                }

                                GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                                float scale = d * sc;
                                float shift = min * m;

                                byte* q = block + 16 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                                int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.
                                for (int l = 0; l < limit; l++)
                                {
                                    pOut[subStart + l] = ((q[l] >> (4 * (j & 1))) & 0xF) * scale - shift;
                                }
                            }
                        }
                    }

                    break;
                case QuantFormat.Q5_0:
                    if (_useAvx2)
                    {
                        DequantQ5_0Avx2(rowPtr, pOut, cols);
                    }
                    else
                    {
                        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk8)
                        {
                            byte* block = rowPtr + (long)(baseIdx / Qk8) * Q5_0BlockBytes;
                            float d = ReadHalfLE(block);
                            uint qh = ReadQhLE(block);

                            int limit = Math.Min(Qk8, cols - baseIdx); // padded quants are zero: stop at the real column count.
                            for (int l = 0; l < limit; l++)
                            {
                                int five = ((block[6 + (l & 15)] >> (4 * (l >> 4))) & 0xF) | ((int)(qh >> l) & 1) << 4;
                                pOut[baseIdx + l] = (five - 16) * d;
                            }
                        }
                    }

                    break;
                case QuantFormat.Q5_K:
                    if (_useAvx2)
                    {
                        DequantQ5KAvx2(rowPtr, pOut, cols);
                    }
                    else
                    {
                        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
                        {
                            byte* block = rowPtr + (long)(baseIdx / QkK) * Q5KBlockBytes;
                            float d = ReadHalfLE(block);
                            float min = ReadHalfLE(block + 2);

                            for (int j = 0; j < 8; j++)
                            {
                                int subStart = baseIdx + j * 32;
                                if (subStart >= cols)
                                {
                                    break; // zero-padded tail block: no elements left to consume.
                                }

                                GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                                float scale = d * sc;
                                float shift = min * m;

                                byte* q = block + 48 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                                byte* qh = block + 16 + 32 * (j / 2); // 5th bits for the same window bytes, bit j per byte
                                int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.
                                for (int l = 0; l < limit; l++)
                                {
                                    int five = ((q[l] >> (4 * (j & 1))) & 0xF) | (((qh[l] >> j) & 1) << 4);
                                    pOut[subStart + l] = five * scale - shift;
                                }
                            }
                        }
                    }

                    break;
                case QuantFormat.Q6_K:
                    if (_useAvx2)
                    {
                        DequantQ6KAvx2(rowPtr, pOut, cols);
                    }
                    else
                    {
                        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
                        {
                            byte* block = rowPtr + (long)(baseIdx / QkK) * Q6KBlockBytes;
                            float d = ReadHalfLE(block + 208);

                            for (int g = 0; g < 16; g++)
                            {
                                int subStart = baseIdx + g * 16;
                                if (subStart >= cols)
                                {
                                    break; // zero-padded tail block: no elements left to consume.
                                }

                                int h = g >> 3; // 128-element half of the super-block
                                int k = g & 7; // 16-element group inside the half
                                int quarter = k >> 2; // 0..3: which 32-element quarter of the half
                                int l = 16 * (k & 1); // window offset inside the half: 0 or 16
                                byte* q = block + 64 * h + 32 * (quarter & 1) + l; // window: bytes [q, q+16) always inside the block
                                byte* qh = block + 128 + 32 * h + l; // high 2 bits for the same window bytes, shift 2*quarter
                                float scale = d * (sbyte)block[192 + g];
                                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.
                                for (int e = 0; e < limit; e++)
                                {
                                    int six = ((q[e] >> (4 * (quarter >> 1))) & 0xF) | (((qh[e] >> (2 * quarter)) & 3) << 4);
                                    pOut[subStart + e] = (six - 32) * scale;
                                }
                            }
                        }
                    }

                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(w), w.Format, "Unknown quant format.");
            }
        }
    }

    public float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Dot operands must have the same length.", nameof(b));
        }

        fixed (float* pa = a)
        fixed (float* pb = b)
        {
            return _useAvx2 ? DotCore(pa, pb, a.Length) : DotScalar(pa, pb, a.Length);
        }
    }

    public void Axpy(float alpha, ReadOnlySpan<float> x, Span<float> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Axpy operands must have the same length.", nameof(y));
        }

        fixed (float* px = x)
        fixed (float* py = y)
        {
            if (_useAvx2)
            {
                AxpyCore(px, py, alpha, x.Length);
            }
            else
            {
                for (int i = 0; i < x.Length; i++)
                {
                    py[i] += alpha * px[i];
                }
            }
        }
    }

    public void Fill(Span<float> y, float v)
    {
        y.Fill(v);
    }

    public void Scale(Span<float> y, float s)
    {
        fixed (float* py = y)
        {
            if (_useAvx2)
            {
                ScaleCore(py, s, y.Length);
            }
            else
            {
                for (int i = 0; i < y.Length; i++)
                {
                    py[i] *= s;
                }
            }
        }
    }

    public void Dispose()
    {
        _pool?.Dispose();
        _pool = null;
    }

    private WorkerPool GetOrCreatePool()
    {
        lock (_poolGate)
        {
            int workersNeeded = Math.Max(_effectiveThreads - 1, 0);

            WorkerPool? pool = _pool;
            if (pool is not null && pool.WorkerCapacity >= workersNeeded)
            {
                return pool;
            }

            // A quota raised after the first threaded call may want more participants than the existing pool hosts:
            // grow it once. Safe because forward consumers of one backend are serialized by the scheduler's gate, so
            // no job can be mid-Run on the old pool — every Run returns only after all its blocks finished.
            pool?.Dispose();

            WorkerPool grown = new WorkerPool(workersNeeded, _coreMask);

            if (_samplingRequested)
            {
                // sampling can be armed before the first threaded call ever builds the pool
                grown.EnableSampling();
            }

            return _pool = grown;
        }
    }

    /// <summary>Whether this backend pins its pool workers (a core set was configured) or leaves them unpinned.</summary>
    internal bool IsPinningConfigured => _coreMask is not null;

    private static float RowDot(byte* row, float* x, int cols, QuantFormat format, bool avx2)
    {
        return format switch
        {
            QuantFormat.F32 => avx2 ? DotCore((float*)row, x, cols) : DotScalar((float*)row, x, cols),
            QuantFormat.F16 => avx2 ? RowDotF16Avx2(row, x, cols) : RowDotF16Scalar(row, x, cols),
            QuantFormat.Q8_0 => avx2 ? RowDotQ8Avx2(row, x, cols) : RowDotQ8Scalar(row, x, cols),
            QuantFormat.Q4_K => avx2 ? RowDotQ4KAvx2(row, x, cols) : RowDotQ4KScalar(row, x, cols),
            QuantFormat.Q5_0 => avx2 ? RowDotQ5_0Avx2(row, x, cols) : RowDotQ5_0Scalar(row, x, cols),
            QuantFormat.Q5_K => avx2 ? RowDotQ5KAvx2(row, x, cols) : RowDotQ5KScalar(row, x, cols),
            QuantFormat.Q6_K => avx2 ? RowDotQ6KAvx2(row, x, cols) : RowDotQ6KScalar(row, x, cols),
            _ => throw new ArgumentOutOfRangeException(nameof(format), format, "Unknown quant format."),
        };
    }

    /// <summary>
    /// Row-block executor for one GEMV call. Raw pointers are valid only while the pinned spans in Gemv's fixed
    /// scope stay alive; WorkerPool.Run guarantees every claimed block finishes before it returns. Each row is a
    /// pure function of its own weight bytes and x, so any partition of blocks yields bit-identical output.
    /// </summary>
    private sealed unsafe class RowBlockJob : IGemvJob
    {
        private readonly byte* _bytes;
        private readonly float* _x;
        private readonly float* _out;
        private readonly int _stride;
        private readonly int _rows;
        private readonly int _cols;
        private readonly QuantFormat _format;
        private readonly bool _avx2;

        public RowBlockJob(byte* bytes, float* x, float* outPtr, int stride, int rows, int cols, QuantFormat format, bool avx2)
        {
            _bytes = bytes;
            _x = x;
            _out = outPtr;
            _stride = stride;
            _rows = rows;
            _cols = cols;
            _format = format;
            _avx2 = avx2;
        }

        public void ProcessBlock(int block)
        {
            int firstRow = block * RowBlock;
            int lastRow = Math.Min(firstRow + RowBlock, _rows);
            for (int row = firstRow; row < lastRow; row++)
            {
                _out[row] = RowDot(_bytes + (long)row * _stride, _x, _cols, _format, _avx2);
            }
        }
    }

    internal interface IGemvJob
    {
        void ProcessBlock(int block);
    }

    /// <summary>
    /// Row-block executor for one batched GEMV call. Each row computes its batch vector dots independently, so
    /// any partition of blocks yields bit-identical output; batch vectors are never split across workers.
    /// </summary>
    private sealed unsafe class BatchRowJob : IGemvJob
    {
        private readonly byte* _bytes;
        private readonly float* _inputs;
        private readonly float* _out;
        private readonly int _stride;
        private readonly int _rows;
        private readonly int _cols;
        private readonly int _count;
        private readonly QuantFormat _format;
        private readonly bool _avx2;

        public BatchRowJob(byte* bytes, float* inputs, float* outPtr, int stride, int rows, int cols, int count, QuantFormat format, bool avx2)
        {
            _bytes = bytes;
            _inputs = inputs;
            _out = outPtr;
            _stride = stride;
            _rows = rows;
            _cols = cols;
            _count = count;
            _format = format;
            _avx2 = avx2;
        }

        public void ProcessBlock(int block)
        {
            int firstRow = block * RowBlock;
            int lastRow = Math.Min(firstRow + RowBlock, _rows);
            for (int row = firstRow; row < lastRow; row++)
            {
                RowBatchDot(_bytes + (long)row * _stride, _inputs, _count, _rows, _cols, _format, _avx2, _out + row);
            }
        }
    }

    // Batched row evaluator for one row: outBase points at column `row` of the flat batch output buffer and
    // receives outBase[b * rows] = dot(row, inputs[b]), i.e. absolute position b*rows + row per
    // IBackend.GemvBatch's layout. The row's packed bytes are traversed exactly once: every dequantized chunk
    // is materialized one time into a stack scratch buffer and fed to all count accumulator chains. Each chain
    // advances exactly like the single-vector kernel's - eight-lane FMAs over successive chunks in the same
    // element order, then scalar tail accumulation, then a lane reduction identical to HSum - so every output
    // is bit-identical to a single-vector GEMV row.
    private static void RowBatchDot(byte* rowBytes, float* inputs, int count, int rows, int cols, QuantFormat format, bool avx2, float* outBase)
    {
        switch (format)
        {
            case QuantFormat.F32:
                if (avx2)
                {
                    BatchF32Avx2((float*)rowBytes, inputs, count, rows, cols, outBase);
                }
                else
                {
                    for (int b = 0; b < count; b++)
                    {
                        outBase[b * rows] = DotScalar((float*)rowBytes, inputs + (long)b * cols, cols);
                    }
                }

                break;
            case QuantFormat.F16:
                if (avx2)
                {
                    BatchF16Avx2(rowBytes, inputs, count, rows, cols, outBase);
                }
                else
                {
                    for (int b = 0; b < count; b++)
                    {
                        outBase[b * rows] = RowDotF16Scalar(rowBytes, inputs + (long)b * cols, cols);
                    }
                }

                break;
            case QuantFormat.Q8_0:
                if (avx2)
                {
                    BatchQ8Avx2(rowBytes, inputs, count, rows, cols, outBase);
                }
                else
                {
                    for (int b = 0; b < count; b++)
                    {
                        outBase[b * rows] = RowDotQ8Scalar(rowBytes, inputs + (long)b * cols, cols);
                    }
                }

                break;
            case QuantFormat.Q4_K:
                if (avx2)
                {
                    BatchQ4KAvx2(rowBytes, inputs, count, rows, cols, outBase);
                }
                else
                {
                    for (int b = 0; b < count; b++)
                    {
                        outBase[b * rows] = RowDotQ4KScalar(rowBytes, inputs + (long)b * cols, cols);
                    }
                }

                break;
            case QuantFormat.Q5_0:
                if (avx2)
                {
                    BatchQ5_0Avx2(rowBytes, inputs, count, rows, cols, outBase);
                }
                else
                {
                    for (int b = 0; b < count; b++)
                    {
                        outBase[b * rows] = RowDotQ5_0Scalar(rowBytes, inputs + (long)b * cols, cols);
                    }
                }

                break;
            case QuantFormat.Q5_K:
                if (avx2)
                {
                    BatchQ5KAvx2(rowBytes, inputs, count, rows, cols, outBase);
                }
                else
                {
                    for (int b = 0; b < count; b++)
                    {
                        outBase[b * rows] = RowDotQ5KScalar(rowBytes, inputs + (long)b * cols, cols);
                    }
                }

                break;
            case QuantFormat.Q6_K:
                if (avx2)
                {
                    BatchQ6KAvx2(rowBytes, inputs, count, rows, cols, outBase);
                }
                else
                {
                    for (int b = 0; b < count; b++)
                    {
                        outBase[b * rows] = RowDotQ6KScalar(rowBytes, inputs + (long)b * cols, cols);
                    }
                }

                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(format), format, "Unknown quant format.");
        }
    }

    // Batched F32 row: one weight window load per chunk feeds all accumulator chains, so the row is streamed
    // once for the whole batch. Accumulator lanes live in stack scratch (count x eight slots, reused per row)
    // and each chain's FMA operand order matches DotCore exactly.
    private static void BatchF32Avx2(float* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            int i = 0;
            for (; i + 8 <= cols; i += 8)
            {
                Vector256<float> wv = Avx.LoadVector256(row + i);
                for (int b = 0; b < count; b++)
                {
                    float* chain = pAcc + b * 8;
                    Avx.Store(chain, Fma.MultiplyAdd(wv, Avx.LoadVector256(inputs + (long)b * cols + i), Avx.LoadVector256(chain)));
                }
            }

            for (; i < cols; i++)
            {
                float w = row[i];
                for (int b = 0; b < count; b++)
                {
                    tail[b] += w * inputs[(long)b * cols + i];
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // Batched F16 row: one half-bit expansion per chunk feeds all accumulator chains with the same FMA operand
    // order as RowDotF16Avx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchF16Avx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            int i = 0;
            for (; i + 8 <= cols; i += 8)
            {
                Vector256<int> bits = Vector256.Create(
                    Sse41.ConvertToVector128Int32((ushort*)(row + 2L * i)),
                    Sse41.ConvertToVector128Int32((ushort*)(row + 2L * i) + 4));
                Vector256<float> w = HalfBitsToFloat(bits);
                for (int b = 0; b < count; b++)
                {
                    float* chain = pAcc + b * 8;
                    Avx.Store(chain, Fma.MultiplyAdd(w, Avx.LoadVector256(inputs + (long)b * cols + i), Avx.LoadVector256(chain)));
                }
            }

            for (; i < cols; i++)
            {
                float w = ReadHalfLE(row + 2 * i);
                for (int b = 0; b < count; b++)
                {
                    tail[b] += w * inputs[(long)b * cols + i];
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // Batched Q8_0 row: one quant unpack per eight-element chunk feeds all accumulator chains with the same
    // (x * q) * d association as RowDotQ8Avx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchQ8Avx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk8)
            {
                byte* block = row + (long)(baseIdx / Qk8) * (Qk8 + 2);
                float dHalf = ReadHalfLE(block);

                int limit = Math.Min(Qk8, cols - baseIdx); // padded quants are zero: stop at the real column count.
                for (int l = 0; l + 8 <= limit; l += 8)
                {
                    Vector256<int> qi = Avx2.ConvertToVector256Int32((sbyte*)(block + 2 + l)); // vpmovsxbd: eight signed quants, in-block only
                    Vector256<float> qf = Avx.ConvertToVector256Single(qi); // dequantization unpack, materialized once per chunk
                    float dLocal = dHalf;
                    Vector256<float> dv = Avx.BroadcastScalarToVector256(&dLocal);
                    for (int b = 0; b < count; b++)
                    {
                        float* chain = pAcc + b * 8;
                        Avx.Store(chain, Fma.MultiplyAdd(Avx.Multiply(Avx.LoadVector256(inputs + (long)b * cols + baseIdx + l), qf), dv, Avx.LoadVector256(chain)));
                    }
                }

                for (int k = (limit / 8) * 8; k < limit; k++)
                {
                    float w = (float)(sbyte)block[2 + k] * dHalf; // partial final block: at most seven elements per row
                    for (int b = 0; b < count; b++)
                    {
                        tail[b] += w * inputs[(long)b * cols + baseIdx + k];
                    }
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // Batched Q4_K row: one nibble unpack and fused scale/shift per chunk feeds all accumulator chains with the
    // same FMA shape as RowDotQ4KAvx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchQ4KAvx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
            {
                byte* block = row + (long)(baseIdx / QkK) * Q4KBlockBytes;
                byte* blockEnd = block + Q4KBlockBytes;
                float dHalf = ReadHalfLE(block);
                float minHalf = ReadHalfLE(block + 2);

                for (int j = 0; j < 8; j++)
                {
                    int subStart = baseIdx + j * 32;
                    if (subStart >= cols)
                    {
                        break; // zero-padded tail block: no elements left to consume.
                    }

                    GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                    float scale = dHalf * sc;
                    Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                    float negShift = -(minHalf * m);
                    Vector256<float> shiftV = Avx.BroadcastScalarToVector256(&negShift);

                    byte* q = block + 16 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                    int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.

                    for (int c = 0; c + 8 <= limit; c += 8)
                    {
                        Vector128<byte> window = LoadGroupBytes(q, blockEnd, c);

                        if ((j & 1) != 0)
                        {
                            // High nibbles per byte lane: srli_epi16 then a 0x0F byte mask (the classic SSE2 pattern).
                            window = Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte();
                        }

                        Vector128<byte> masked = Sse2.And(window, GetNibbleMask()); // needed nibbles now in each byte lane's low bits

                        Vector128<int> lo = Sse41.ConvertToVector128Int32(masked); // bytes 0..3 -> dword lanes (pmovzxbd)
                        Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes

                        Vector256<float> wchunk = Fma.MultiplyAdd(scaleV, Avx.ConvertToVector256Single(Vector256.Create(lo, hi)), shiftV); // dequantized chunk, materialized once
                        for (int b = 0; b < count; b++)
                        {
                            float* chain = pAcc + b * 8;
                            Avx.Store(chain, Fma.MultiplyAdd(Avx.LoadVector256(inputs + (long)b * cols + subStart + c), wchunk, Avx.LoadVector256(chain)));
                        }
                    }

                    for (int l = (limit / 8) * 8; l < limit; l++)
                    {
                        // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                        float nibble = ((q[l] >> (4 * (j & 1))) & 0xF) * scale - minHalf * m;
                        for (int b = 0; b < count; b++)
                        {
                            tail[b] += nibble * inputs[(long)b * cols + subStart + l];
                        }
                    }
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // Batched Q5_0 row: one 5-bit unpack per eight-element chunk feeds all accumulator chains with the same
    // fused (x * q) * d association as RowDotQ5_0Avx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchQ5_0Avx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk8)
            {
                byte* block = row + (long)(baseIdx / Qk8) * Q5_0BlockBytes;
                float dHalf = ReadHalfLE(block);
                uint qh = ReadQhLE(block);

                int limit = Math.Min(Qk8, cols - baseIdx); // padded quants are zero: stop at the real column count.
                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<float> qf = Avx.ConvertToVector256Single(LoadQ5_0Chunk(block, qh, c)); // dequantization unpack, materialized once per chunk
                    float dLocal = dHalf;
                    Vector256<float> dv = Avx.BroadcastScalarToVector256(&dLocal);
                    for (int b = 0; b < count; b++)
                    {
                        float* chain = pAcc + b * 8;
                        Avx.Store(chain, Fma.MultiplyAdd(Avx.Multiply(Avx.LoadVector256(inputs + (long)b * cols + baseIdx + c), qf), dv, Avx.LoadVector256(chain)));
                    }
                }

                for (int l = (limit / 8) * 8; l < limit; l++)
                {
                    int five = ((block[6 + (l & 15)] >> (4 * (l >> 4))) & 0xF) | ((int)(qh >> l) & 1) << 4;
                    float w = (five - 16) * dHalf; // partial final block: at most seven elements per row
                    for (int b = 0; b < count; b++)
                    {
                        tail[b] += w * inputs[(long)b * cols + baseIdx + l];
                    }
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // Batched Q5_K row: one nibble-plus-5th-bit unpack and fused scale/shift per chunk feeds all accumulator
    // chains with the same FMA shape as RowDotQ5KAvx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchQ5KAvx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
            {
                byte* block = row + (long)(baseIdx / QkK) * Q5KBlockBytes;
                byte* blockEnd = block + Q5KBlockBytes;
                float dHalf = ReadHalfLE(block);
                float minHalf = ReadHalfLE(block + 2);

                for (int j = 0; j < 8; j++)
                {
                    int subStart = baseIdx + j * 32;
                    if (subStart >= cols)
                    {
                        break; // zero-padded tail block: no elements left to consume.
                    }

                    GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                    float scale = dHalf * sc;
                    Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                    float negShift = -(minHalf * m);
                    Vector256<float> shiftV = Avx.BroadcastScalarToVector256(&negShift);

                    byte* q = block + 48 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                    byte* qh = block + 16 + 32 * (j / 2); // 5th bits for the same window bytes, bit j per byte
                    int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.

                    for (int c = 0; c + 8 <= limit; c += 8)
                    {
                        Vector256<int> five = LoadQ5KChunk(q, qh, blockEnd, j, c); // dequantization unpack, materialized once per chunk
                        Vector256<float> wchunk = Fma.MultiplyAdd(scaleV, Avx.ConvertToVector256Single(five), shiftV);
                        for (int b = 0; b < count; b++)
                        {
                            float* chain = pAcc + b * 8;
                            Avx.Store(chain, Fma.MultiplyAdd(Avx.LoadVector256(inputs + (long)b * cols + subStart + c), wchunk, Avx.LoadVector256(chain)));
                        }
                    }

                    for (int l = (limit / 8) * 8; l < limit; l++)
                    {
                        // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                        int five = ((q[l] >> (4 * (j & 1))) & 0xF) | (((qh[l] >> j) & 1) << 4);
                        float value = five * scale - minHalf * m;
                        for (int b = 0; b < count; b++)
                        {
                            tail[b] += value * inputs[(long)b * cols + subStart + l];
                        }
                    }
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // Batched Q6_K row: one 6-bit unpack and fused scale per chunk feeds all accumulator chains with the same
    // FMA shape as RowDotQ6KAvx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchQ6KAvx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
            {
                byte* block = row + (long)(baseIdx / QkK) * Q6KBlockBytes;
                float dHalf = ReadHalfLE(block + 208);

                for (int g = 0; g < 16; g++)
                {
                    int subStart = baseIdx + g * 16;
                    if (subStart >= cols)
                    {
                        break; // zero-padded tail block: no elements left to consume.
                    }

                    int h = g >> 3; // 128-element half of the super-block
                    int k = g & 7; // 16-element group inside the half
                    int quarter = k >> 2; // 0..3: which 32-element quarter of the half
                    int l = 16 * (k & 1); // window offset inside the half: 0 or 16
                    byte* q = block + 64 * h + 32 * (quarter & 1) + l; // window: bytes [q, q+16) always inside the block
                    byte* qh = block + 128 + 32 * h + l; // high 2 bits for the same window bytes, shift 2*quarter
                    float scale = dHalf * (sbyte)block[192 + g];
                    Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                    int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.

                    for (int c = 0; c + 8 <= limit; c += 8)
                    {
                        Vector256<int> six = LoadQ6KChunk(q, qh, quarter, c); // dequantization unpack, materialized once per chunk
                        Vector256<float> wchunk = Avx.Multiply(scaleV, Avx.ConvertToVector256Single(six)); // fused scale stays in the dot FMA chain
                        for (int b = 0; b < count; b++)
                        {
                            float* chain = pAcc + b * 8;
                            Avx.Store(chain, Fma.MultiplyAdd(Avx.LoadVector256(inputs + (long)b * cols + subStart + c), wchunk, Avx.LoadVector256(chain)));
                        }
                    }

                    for (int e = (limit / 8) * 8; e < limit; e++)
                    {
                        // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                        int six = ((q[e] >> (4 * (quarter >> 1))) & 0xF) | (((qh[e] >> (2 * quarter)) & 3) << 4);
                        float w = (six - 32) * scale;
                        for (int b = 0; b < count; b++)
                        {
                            tail[b] += w * inputs[(long)b * cols + subStart + e];
                        }
                    }
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // Batch scratch plumbing: zero the per-chain lanes and tails for one row. stackalloc scratch is reused for
    // every row and call; nothing here allocates.
    private static void BatchInit(Span<float> acc, Span<float> tail, int count)
    {
        acc.Clear();
        tail.Clear();
    }

    // Batch scratch plumbing: fold each chain's eight lanes with the same reduction the single kernels use and
    // write outBase[b * rows]. Scalar tails are added afterwards (F32/F16) or are already folded in (Q8_0/Q4_K,
    // whose chains end with an empty tail).
    private static void BatchFinish(float* pAcc, Span<float> tail, int count, int rows, float* outBase)
    {
        for (int b = 0; b < count; b++)
        {
            outBase[b * rows] = HSum(Avx.LoadVector256(pAcc + b * 8)) + tail[b];
        }
    }

    // Little-endian half-float load from raw memory.
    private static float ReadHalfLE(byte* p) => (float)BitConverter.Int16BitsToHalf((short)(*p | (p[1] << 8)));

    // Little-endian 32-bit load of the Q5_0 5th-bit field at block bytes 2..5.
    private static uint ReadQhLE(byte* block) => (uint)(block[2] | (block[3] << 8) | (block[4] << 16) | (block[5] << 24));

    // Q5_0 chunk loader for eight elements starting at c in {0, 8, 16, 24}: element j < 16 is the low nibble of
    // qs[j] and element 16+j the high nibble of the same byte, each OR-ed with bit i of the qh dword. The window
    // always loads the full 16-byte qs region (the only 16-byte read that stays inside the 22-byte block) and
    // slides the high half down for c in {8, 24}.
    private static Vector256<int> LoadQ5_0Chunk(byte* block, uint qh, int c)
    {
        byte* qs = block + 6;
        Vector128<byte> window = Sse2.LoadVector128(qs); // full nibble region, always inside the block
        if ((c & 8) != 0)
        {
            window = Ssse3.Shuffle(window, GetSlideHighBytesShuffle()); // qs bytes 8..15 land in lanes 0..7 for c in {8, 24}
        }

        if (c >= 16)
        {
            // High nibbles per byte lane: srli_epi16 then a 0x0F byte mask (the classic SSE2 pattern).
            window = Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte();
        }

        Vector128<byte> masked = Sse2.And(window, GetNibbleMask());

        Vector128<int> lo = Sse41.ConvertToVector128Int32(masked); // bytes 0..3 -> dword lanes (pmovzxbd)
        Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes

        byte qhByte = (byte)(c >> 3); // all eight 5th bits of this chunk live in one qh field byte
        Vector128<byte> replicated = Ssse3.Shuffle(Sse2.LoadVector128(block + 2), Vector128.Create(qhByte));
        Vector128<int> qhDwords = Sse41.ConvertToVector128Int32(replicated); // four dword lanes, each the qh field byte value
        Vector256<int> highBit = Avx2.And(
            Avx2.CompareGreaterThan(Avx2.And(Vector256.Create(qhDwords, qhDwords), GetQ5_0BitMasks()), Vector256<int>.Zero),
            Vector256.Create(16));

        return Avx2.Subtract(Avx2.Or(Vector256.Create(lo, hi), highBit), Vector256.Create(16)); // signed 5-bit values in element order
    }

    // Q5_K chunk loader for eight window bytes at offset c: low nibbles for even groups, high nibbles for odd,
    // each OR-ed with bit j of the matching 5th-bit byte. Window loads are clamped so they never pass the end
    // of the block, exactly like LoadGroupBytes.
    private static Vector256<int> LoadQ5KChunk(byte* q, byte* qh, byte* blockEnd, int j, int c)
    {
        Vector128<byte> window = LoadGroupBytes(q, blockEnd, c);

        if ((j & 1) != 0)
        {
            // High nibbles per byte lane: srli_epi16 then a 0x0F byte mask (the classic SSE2 pattern).
            window = Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte();
        }

        Vector128<byte> masked = Sse2.And(window, GetNibbleMask());

        Vector128<int> lo = Sse41.ConvertToVector128Int32(masked); // bytes 0..3 -> dword lanes (pmovzxbd)
        Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes

        // Widen the eight 5th-bit bytes to dwords and test bit j of each with a constant broadcast mask.
        Vector128<byte> qhWindow = Sse2.LoadVector128(qh + c); // reads [qh+c, qh+c+16), clamped reads stay inside the block
        Vector128<int> qhLo = Sse41.ConvertToVector128Int32(qhWindow); // bytes 0..3 -> dword lanes (pmovzxbd)
        Vector128<int> qhHi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(qhWindow, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes
        Vector256<int> highBit = Avx2.And(
            Avx2.CompareGreaterThan(Avx2.And(Vector256.Create(qhLo, qhHi), Vector256.Create(1 << j)), Vector256<int>.Zero),
            Vector256.Create(16));

        return Avx2.Or(Vector256.Create(lo, hi), highBit); // unsigned 5-bit values in element order
    }

    // Q6_K chunk loader for eight window bytes at offset c in {0, 8}: low nibbles for quarters 0..1, high
    // nibbles for quarters 2..3, each OR-ed with the 2-bit field at bit shift 2*quarter of the matching qh
    // byte. The conversion widens with sign, so the values land pre-centered at [-32, 31].
    private static Vector256<int> LoadQ6KChunk(byte* q, byte* qh, int quarter, int c)
    {
        Vector128<byte> window = Sse2.LoadVector128(q + c); // reads [q+c, q+c+16), always inside the 128-byte ql region

        if ((quarter >> 1) != 0)
        {
            // High nibbles per byte lane: srli_epi16 then a 0x0F byte mask (the classic SSE2 pattern).
            window = Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte();
        }

        Vector128<byte> masked = Sse2.And(window, GetNibbleMask());

        Vector128<int> lo = Sse41.ConvertToVector128Int32(masked); // bytes 0..3 -> dword lanes (pmovzxbd)
        Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes

        // Widen the eight high-bit bytes to dwords and extract the 2-bit field at bit shift 2*quarter with
        // constant immediates (the quarter loop variable selects the arm at compile time per unrolled group).
        Vector128<byte> qhWindow = Sse2.LoadVector128(qh + c); // reads [qh+c, qh+c+16), clamped reads stay inside the block
        Vector128<int> qhLo = Sse41.ConvertToVector128Int32(qhWindow); // bytes 0..3 -> dword lanes (pmovzxbd)
        Vector128<int> qhHi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(qhWindow, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes
        Vector256<int> qhDwords = Vector256.Create(qhLo, qhHi);
        Vector256<int> highBit = Avx2.ShiftLeftLogical(quarter switch
        {
            0 => Avx2.And(qhDwords, Vector256.Create(3)),
            1 => Avx2.And(Avx2.ShiftRightLogical(qhDwords, 2), Vector256.Create(3)),
            2 => Avx2.And(Avx2.ShiftRightLogical(qhDwords, 4), Vector256.Create(3)),
            _ => Avx2.And(Avx2.ShiftRightLogical(qhDwords, 6), Vector256.Create(3)),
        }, 4);

        return Avx2.Subtract(Avx2.Or(Vector256.Create(lo, hi), highBit), Vector256.Create(32));
    }

    // Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits across bytes).
    private static void GetScaleMinK4(int j, byte* scales, out byte sc, out byte m)
    {
        if (j < 4)
        {
            sc = (byte)(scales[j] & 0x3F);
            m = (byte)(scales[j + 4] & 0x3F);
        }
        else
        {
            sc = (byte)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
            m = (byte)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
        }
    }

    // F16 rows are raw little-endian half-floats.
    private static float RowDotF16Scalar(byte* row, float* x, int cols)
    {
        float acc = 0f;
        for (int i = 0; i < cols; i++)
        {
            acc += ReadHalfLE(row + 2 * i) * x[i];
        }

        return acc;
    }

    // AVX2: eight halves per step (one 16-byte window). .NET exposes no F16C vcvtph2ps intrinsic, so the
    // half-to-float expansion is integer bit math (HalfToFloat). Tail columns use scalar accumulation because an
    // odd column count has no padding to absorb a wide load. Live YMM: accumulator plus one conversion chain.
    private static float RowDotF16Avx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;

        int i = 0;
        for (; i + 8 <= cols; i += 8)
        {
            // pmovzxwd converts four half-bit patterns per call: two loads cover the eight-element chunk.
            Vector256<int> bits = Vector256.Create(
                Sse41.ConvertToVector128Int32((ushort*)(row + 2L * i)),
                Sse41.ConvertToVector128Int32((ushort*)(row + 2L * i) + 4));
            Vector256<float> w = HalfBitsToFloat(bits);
            Vector256<float> xv = Avx.LoadVector256(x + i); // safe: the row covers a full chunk of halves
            acc = Fma.MultiplyAdd(w, xv, acc);
        }

        float tail = 0f;
        for (; i < cols; i++)
        {
            tail += ReadHalfLE(row + 2 * i) * x[i];
        }

        return HSum(acc) + tail;
    }

    // Q8_0: each 34-byte block is half-float d followed by 32 int8 quants; w = d * q.
    private static float RowDotQ8Scalar(byte* row, float* x, int cols)
    {
        float acc = 0f;
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk8)
        {
            byte* block = row + (long)(baseIdx / Qk8) * (Qk8 + 2);
            float d = ReadHalfLE(block);

            int limit = Math.Min(Qk8, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int l = 0; l < limit; l++)
            {
                acc += x[baseIdx + l] * ((float)(sbyte)block[2 + l] * d);
            }
        }

        return acc;
    }

    // AVX2: broadcast the block scale once, widen eight signed quants per step with vpmovsxbd from a sbyte
    // pointer (reads only real block bytes), and FMA. Live YMM stays at five or fewer so no spills occur. The
    // padded tail of a partial final block falls to scalar accumulation.
    private static float RowDotQ8Avx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float tail = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk8)
        {
            byte* block = row + (long)(baseIdx / Qk8) * (Qk8 + 2);
            float dHalf = ReadHalfLE(block);

            int limit = Math.Min(Qk8, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int l = 0; l < limit; l += 8)
            {
                if (l + 8 > limit)
                {
                    break;
                }

                Vector256<int> qi = Avx2.ConvertToVector256Int32((sbyte*)(block + 2 + l)); // vpmovsxbd: eight signed quants, in-block only
                Vector256<float> qf = Avx.ConvertToVector256Single(qi);
                Vector256<float> xv = Avx.LoadVector256(x + baseIdx + l); // safe: these columns are real and block-covered

                float dLocal = dHalf;
                acc = Fma.MultiplyAdd(Avx.Multiply(xv, qf), Avx.BroadcastScalarToVector256(&dLocal), acc);
            }

            for (int k = (limit / 8) * 8; k < limit; k++)
            {
                tail += x[baseIdx + k] * ((float)(sbyte)block[2 + k] * dHalf); // partial final block: at most seven elements per row
            }
        }

        return HSum(acc) + tail;
    }

    // Q4_K: each 144-byte block covers 256 elements in eight 32-element groups j (0..7). Group j uses the low
    // nibbles of qs[32*(j/2) .. +31] when j is even and the high nibbles of those same bytes when odd;
    // w = d * sc_j * q - dmin * m_j. Scale/min unpacking follows the byte layout documented at QkK (scales at
    // block+4, quants at block+16).
    private static float RowDotQ4KScalar(byte* row, float* x, int cols)
    {
        float acc = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q4KBlockBytes;
            float d = ReadHalfLE(block);
            float min = ReadHalfLE(block + 2);

            for (int j = 0; j < 8; j++)
            {
                int subStart = baseIdx + j * 32;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                float scale = d * sc;
                float shift = min * m;

                byte* q = block + 16 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.
                for (int l = 0; l < limit; l++)
                {
                    float nibble = ((q[l] >> (4 * (j & 1))) & 0xF) * scale - shift;
                    acc += x[subStart + l] * nibble;
                }
            }
        }

        return acc;
    }

    // AVX2: per group broadcast the fused scale and negated min-shift, unpack eight nibbles at a time from one
    // 16-byte window load (masked low nibbles or srli-per-word high nibbles), then two FMAs for the four+four
    // dword lanes. Window loads are clamped so they never pass the end of the block. Live YMM: accumulator,
    // scaleV, shiftV plus short-lived unpack temporaries - at most seven simultaneously, so no spills occur.
    // Partial chunks fall to scalar accumulation.
    private static float RowDotQ4KAvx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float tail = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q4KBlockBytes;
            byte* blockEnd = block + Q4KBlockBytes;
            float dHalf = ReadHalfLE(block);
            float minHalf = ReadHalfLE(block + 2);

            for (int j = 0; j < 8; j++)
            {
                int subStart = baseIdx + j * 32;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                float scale = dHalf * sc;
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                float negShift = -(minHalf * m);
                Vector256<float> shiftV = Avx.BroadcastScalarToVector256(&negShift);

                byte* q = block + 16 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector128<byte> window = LoadGroupBytes(q, blockEnd, c);

                    if ((j & 1) != 0)
                    {
                        // High nibbles per byte lane: srli_epi16 then a 0x0F byte mask (the classic SSE2 pattern).
                        window = Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte();
                    }

                    Vector128<byte> masked = Sse2.And(window, GetNibbleMask()); // needed nibbles now in each byte lane's low bits

                    Vector128<int> lo = Sse41.ConvertToVector128Int32(masked); // bytes 0..3 -> dword lanes (pmovzxbd)
                    Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes

                    Vector256<float> qf = Avx.ConvertToVector256Single(Vector256.Create(lo, hi)); // eight nibble values in element order
                    Vector256<float> xv = Avx.LoadVector256(x + subStart + c); // safe: these eight columns are real and block-covered

                    acc = Fma.MultiplyAdd(xv, Fma.MultiplyAdd(scaleV, qf, shiftV), acc);
                }

                for (int l = (limit / 8) * 8; l < limit; l++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    float nibble = ((q[l] >> (4 * (j & 1))) & 0xF) * scale - minHalf * m;
                    tail += x[subStart + l] * nibble;
                }
            }
        }

        return HSum(acc) + tail;
    }

    // Q5_0: each 22-byte block covers 32 five-bit quants: half-float d at byte 0, the little-endian 32-bit qh
    // word at bytes 2..5, and 16 nibble bytes at bytes 6..21. Element j < 16 is the low nibble of qs[j] and
    // element 16+j the high nibble of the same byte; bit i of qh supplies the 5th bit. w = (q - 16) * d.
    private static float RowDotQ5_0Scalar(byte* row, float* x, int cols)
    {
        float acc = 0f;
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk8)
        {
            byte* block = row + (long)(baseIdx / Qk8) * Q5_0BlockBytes;
            float d = ReadHalfLE(block);
            uint qh = ReadQhLE(block);

            int limit = Math.Min(Qk8, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int l = 0; l < limit; l++)
            {
                int five = ((block[6 + (l & 15)] >> (4 * (l >> 4))) & 0xF) | ((int)(qh >> l) & 1) << 4;
                acc += x[baseIdx + l] * ((five - 16) * d);
            }
        }

        return acc;
    }

    // AVX2: broadcast the block scale once and unpack eight five-bit quants per step from the 16-byte nibble
    // window (low nibbles for elements 0..15, high nibbles for 16..31, each OR-ed with its 5th bit from the
    // qh dword), then the same fused (x * q) * d FMA chain as RowDotQ8Avx2. Window loads stay inside the block.
    // The padded tail of a partial final block falls to scalar accumulation.
    private static float RowDotQ5_0Avx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float tail = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk8)
        {
            byte* block = row + (long)(baseIdx / Qk8) * Q5_0BlockBytes;
            float dHalf = ReadHalfLE(block);
            uint qh = ReadQhLE(block);

            int limit = Math.Min(Qk8, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int l = 0; l < limit; l += 8)
            {
                if (l + 8 > limit)
                {
                    break;
                }

                Vector256<int> five = LoadQ5_0Chunk(block, qh, l);
                Vector256<float> qf = Avx.ConvertToVector256Single(five); // signed 5-bit values in element order
                Vector256<float> xv = Avx.LoadVector256(x + baseIdx + l); // safe: these columns are real and block-covered

                float dLocal = dHalf;
                acc = Fma.MultiplyAdd(Avx.Multiply(xv, qf), Avx.BroadcastScalarToVector256(&dLocal), acc);
            }

            for (int k = (limit / 8) * 8; k < limit; k++)
            {
                int five = ((block[6 + (k & 15)] >> (4 * (k >> 4))) & 0xF) | ((int)(qh >> k) & 1) << 4;
                tail += x[baseIdx + k] * ((five - 16) * dHalf); // partial final block: at most seven elements per row
            }
        }

        return HSum(acc) + tail;
    }

    // Q5_K: each 176-byte block covers 256 elements in eight 32-element groups j (0..7) with the Q4_K scale
    // layout at bytes 0..15. Group j uses the low nibbles of qs[32*(j/2) .. +31] (qs at byte 48) when j is even
    // and the high nibbles when odd; bit j of qh[32*(j/2) + l] (qh at byte 16) supplies the 5th bit.
    // w = d * sc_j * q - dmin * m_j with q in [0, 31].
    private static float RowDotQ5KScalar(byte* row, float* x, int cols)
    {
        float acc = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q5KBlockBytes;
            float d = ReadHalfLE(block);
            float min = ReadHalfLE(block + 2);

            for (int j = 0; j < 8; j++)
            {
                int subStart = baseIdx + j * 32;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                float scale = d * sc;
                float shift = min * m;

                byte* q = block + 48 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                byte* qh = block + 16 + 32 * (j / 2); // 5th bits for the same window bytes, bit j per byte
                int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.
                for (int l = 0; l < limit; l++)
                {
                    int five = ((q[l] >> (4 * (j & 1))) & 0xF) | (((qh[l] >> j) & 1) << 4);
                    acc += x[subStart + l] * (five * scale - shift);
                }
            }
        }

        return acc;
    }

    // AVX2: per group broadcast the fused scale and negated min-shift, unpack eight 5-bit quants at a time from
    // the nibble window (same clamped window loads as Q4_K) plus the 5th-bit byte, then two FMAs for the
    // four+four dword lanes. Partial chunks fall to scalar accumulation.
    private static float RowDotQ5KAvx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float tail = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q5KBlockBytes;
            byte* blockEnd = block + Q5KBlockBytes;
            float dHalf = ReadHalfLE(block);
            float minHalf = ReadHalfLE(block + 2);

            for (int j = 0; j < 8; j++)
            {
                int subStart = baseIdx + j * 32;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                float scale = dHalf * sc;
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                float negShift = -(minHalf * m);
                Vector256<float> shiftV = Avx.BroadcastScalarToVector256(&negShift);

                byte* q = block + 48 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                byte* qh = block + 16 + 32 * (j / 2); // 5th bits for the same window bytes, bit j per byte
                int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<int> five = LoadQ5KChunk(q, qh, blockEnd, j, c);

                    Vector256<float> qf = Avx.ConvertToVector256Single(five); // eight 5-bit values in element order
                    Vector256<float> xv = Avx.LoadVector256(x + subStart + c); // safe: these eight columns are real and block-covered

                    acc = Fma.MultiplyAdd(xv, Fma.MultiplyAdd(scaleV, qf, shiftV), acc);
                }

                for (int l = (limit / 8) * 8; l < limit; l++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    int five = ((q[l] >> (4 * (j & 1))) & 0xF) | (((qh[l] >> j) & 1) << 4);
                    tail += x[subStart + l] * (five * scale - minHalf * m);
                }
            }
        }

        return HSum(acc) + tail;
    }

    // Q6_K: each 210-byte block covers 256 elements in sixteen 16-element groups g (0..15). Group g lives in
    // half h = g/3 of the block, quarter k/4 of that half, at ql window 64h + 32*(quarter&1) + 16*(k&1): low
    // nibbles for even quarters, high nibbles for odd; qh byte 128h + 16*(k&1) + e carries the high 2 bits at
    // bit shift 2*quarter. w = d * sc_g * (q - 32) with q in [0, 63] and sc_g the signed int8 scale at byte
    // 192 + g.
    private static float RowDotQ6KScalar(byte* row, float* x, int cols)
    {
        float acc = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q6KBlockBytes;
            float d = ReadHalfLE(block + 208);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                int quarter = k >> 2; // 0..3: which 32-element quarter of the half
                int l = 16 * (k & 1); // window offset inside the half: 0 or 16
                byte* q = block + 64 * h + 32 * (quarter & 1) + l; // window: bytes [q, q+16) always inside the block
                byte* qh = block + 128 + 32 * h + l; // high 2 bits for the same window bytes, shift 2*quarter
                float scale = d * (sbyte)block[192 + g];
                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.
                for (int e = 0; e < limit; e++)
                {
                    int six = ((q[e] >> (4 * (quarter >> 1))) & 0xF) | (((qh[e] >> (2 * quarter)) & 3) << 4);
                    acc += x[subStart + e] * ((six - 32) * scale);
                }
            }
        }

        return acc;
    }

    // AVX2: per group broadcast the signed group scale, unpack eight 6-bit quants at a time from the nibble
    // window plus the 2-bit high-byte field (signed conversion absorbs the -32 centering), then the fused
    // scale and the dot FMA in the same chain as RowDotQ4KAvx2. Partial chunks fall to scalar accumulation.
    private static float RowDotQ6KAvx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float tail = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q6KBlockBytes;
            float dHalf = ReadHalfLE(block + 208);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                int quarter = k >> 2; // 0..3: which 32-element quarter of the half
                int l = 16 * (k & 1); // window offset inside the half: 0 or 16
                byte* q = block + 64 * h + 32 * (quarter & 1) + l; // window: bytes [q, q+16) always inside the block
                byte* qh = block + 128 + 32 * h + l; // high 2 bits for the same window bytes, shift 2*quarter
                float scale = dHalf * (sbyte)block[192 + g];
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<int> six = LoadQ6KChunk(q, qh, quarter, c);

                    Vector256<float> qf = Avx.ConvertToVector256Single(six); // signed 6-bit values in element order
                    Vector256<float> xv = Avx.LoadVector256(x + subStart + c); // safe: these eight columns are real and block-covered

                    acc = Fma.MultiplyAdd(xv, Avx.Multiply(scaleV, qf), acc);
                }

                for (int e = (limit / 8) * 8; e < limit; e++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    int six = ((q[e] >> (4 * (quarter >> 1))) & 0xF) | (((qh[e] >> (2 * quarter)) & 3) << 4);
                    tail += x[subStart + e] * ((six - 32) * scale);
                }
            }
        }

        return HSum(acc) + tail;
    }

    // Load the sixteen bytes starting at window offset c inside [q, q+32). For offsets whose wide load would pass
    // the end of the block (only possible for the last group pair of a row), load from offset 16 instead and slide
    // the needed eight bytes down with pshufb; both variants stay strictly inside the 144-byte block.
    private static Vector128<byte> LoadGroupBytes(byte* q, byte* blockEnd, int c)
    {
        if (q + c + 16 <= blockEnd)
        {
            return Sse2.LoadVector128(q + c); // full window read stays inside the block for c in {0, 8, 16}
        }

        Vector128<byte> tailWindow = Sse2.LoadVector128(q + 16); // reads [q+16, q+31], always inside the group-pair window
        return Ssse3.Shuffle(tailWindow, GetSlideHighBytesShuffle()); // window bytes c..c+7 (only reachable for c == 24) land in lanes 0..7
    }

    // Expand eight half-bit patterns (already widened to dwords) into floats with integer bit math; .NET exposes
    // no F16C vcvtph2ps intrinsic, so the expansion is done by hand. Infinities/NaNs in weights are not expected;
    // they decode to finite values here.
    private static Vector256<float> HalfBitsToFloat(Vector256<int> i)
    {
        Vector256<int> signBit = Avx2.And(i, Vector256.Create(0x8000));
        Vector256<int> payload = Avx2.And(i, Vector256.Create(0x7FFF));
        Vector256<int> exp = Avx2.ShiftRightLogical(payload, 10);
        Vector256<int> mant = Avx2.And(payload, Vector256.Create(0x3FF));

        // Normal values: rebias the exponent (half bias 15 to float bias 127 is +112) and left-shift the mantissa,
        // then reinterpret the assembled bits - no conversion instruction here.
        Vector256<float> normalF = Avx2.Or(
            Avx2.ShiftLeftLogical(Avx2.Add(exp, Vector256.Create(112)), 23),
            Avx2.ShiftLeftLogical(mant, 13)).AsSingle();

        // Subnormals and zero: mant * 2^-24. Infinities/NaNs in weights are not expected; they decode to finite values here.
        Vector256<float> subF = Avx.Multiply(Avx.ConvertToVector256Single(mant), Vector256.Create(1f / (1 << 24))); // genuine integer-to-float conversion

        Vector256<int> normalMask = Avx2.CompareGreaterThan(exp, Vector256<int>.Zero);
        Vector256<float> selected = Avx2.Or(
            Avx2.And(normalF.AsInt32(), normalMask),
            Avx2.AndNot(normalMask, subF.AsInt32())).AsSingle();

        return Avx2.Xor(selected.AsInt32(), Avx2.ShiftLeftLogical(signBit, 16)).AsSingle(); // apply the sign bit (0x8000 << 16)
    }

    // F16 row expansion: eight halves per chunk through HalfBitsToFloat; odd tails scalar. Same half-to-float math as RowDotF16Avx2.
    private static void DequantF16Avx2(byte* row, float* outPtr, int cols)
    {
        int i = 0;
        for (; i + 8 <= cols; i += 8)
        {
            Vector256<int> bits = Vector256.Create(
                Sse41.ConvertToVector128Int32((ushort*)(row + 2L * i)),
                Sse41.ConvertToVector128Int32((ushort*)(row + 2L * i) + 4));
            Avx.Store(outPtr + i, HalfBitsToFloat(bits));
        }

        for (; i < cols; i++)
        {
            outPtr[i] = ReadHalfLE(row + 2 * i);
        }
    }

    // Q8_0 row expansion: w = d * q per block. Same unpacking as RowDotQ8Avx2, storing instead of dotting.
    private static void DequantQ8Avx2(byte* row, float* outPtr, int cols)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk8)
        {
            byte* block = row + (long)(baseIdx / Qk8) * (Qk8 + 2);
            float dHalf = ReadHalfLE(block);

            int limit = Math.Min(Qk8, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int l = 0; l < limit; l += 8)
            {
                if (l + 8 > limit)
                {
                    break;
                }

                Vector256<int> qi = Avx2.ConvertToVector256Int32((sbyte*)(block + 2 + l)); // vpmovsxbd: eight signed quants, in-block only
                float dLocal = dHalf;
                Vector256<float> dv = Avx.BroadcastScalarToVector256(&dLocal);
                Avx.Store(outPtr + baseIdx + l, Avx.Multiply(Avx.ConvertToVector256Single(qi), dv));
            }

            for (int k = (limit / 8) * 8; k < limit; k++)
            {
                outPtr[baseIdx + k] = (float)(sbyte)block[2 + k] * dHalf; // partial final block: at most seven elements per row
            }
        }
    }

    // Q4_K row expansion: w = d*sc_j*q - dmin*m_j per group. Same nibble unpack and fused scale/shift as RowDotQ4KAvx2.
    private static void DequantQ4KAvx2(byte* row, float* outPtr, int cols)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q4KBlockBytes;
            byte* blockEnd = block + Q4KBlockBytes;
            float dHalf = ReadHalfLE(block);
            float minHalf = ReadHalfLE(block + 2);

            for (int j = 0; j < 8; j++)
            {
                int subStart = baseIdx + j * 32;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                float scale = dHalf * sc;
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                float negShift = -(minHalf * m);
                Vector256<float> shiftV = Avx.BroadcastScalarToVector256(&negShift);

                byte* q = block + 16 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector128<byte> window = LoadGroupBytes(q, blockEnd, c);

                    if ((j & 1) != 0)
                    {
                        // High nibbles per byte lane: srli_epi16 then a 0x0F byte mask (the classic SSE2 pattern).
                        window = Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte();
                    }

                    Vector128<byte> masked = Sse2.And(window, GetNibbleMask()); // needed nibbles now in each byte lane's low bits

                    Vector128<int> lo = Sse41.ConvertToVector128Int32(masked); // bytes 0..3 -> dword lanes (pmovzxbd)
                    Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes

                    Vector256<float> qf = Avx.ConvertToVector256Single(Vector256.Create(lo, hi)); // eight nibble values in element order
                    Avx.Store(outPtr + subStart + c, Fma.MultiplyAdd(scaleV, qf, shiftV));
                }

                for (int l = (limit / 8) * 8; l < limit; l++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    outPtr[subStart + l] = ((q[l] >> (4 * (j & 1))) & 0xF) * scale - minHalf * m;
                }
            }
        }
    }

    // Q5_0 row expansion: w = (q - 16) * d per block. Same 5-bit unpacking as RowDotQ5_0Avx2, storing instead of dotting.
    private static void DequantQ5_0Avx2(byte* row, float* outPtr, int cols)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk8)
        {
            byte* block = row + (long)(baseIdx / Qk8) * Q5_0BlockBytes;
            float dHalf = ReadHalfLE(block);
            uint qh = ReadQhLE(block);

            int limit = Math.Min(Qk8, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int l = 0; l < limit; l += 8)
            {
                if (l + 8 > limit)
                {
                    break;
                }

                Vector256<int> five = LoadQ5_0Chunk(block, qh, l);
                float dLocal = dHalf;
                Avx.Store(outPtr + baseIdx + l, Avx.Multiply(Avx.ConvertToVector256Single(five), Avx.BroadcastScalarToVector256(&dLocal)));
            }

            for (int k = (limit / 8) * 8; k < limit; k++)
            {
                int five = ((block[6 + (k & 15)] >> (4 * (k >> 4))) & 0xF) | ((int)(qh >> k) & 1) << 4;
                outPtr[baseIdx + k] = (five - 16) * dHalf; // partial final block: at most seven elements per row
            }
        }
    }

    // Q5_K row expansion: w = d*sc_j*q - dmin*m_j per group. Same nibble-plus-5th-bit unpack and fused scale/shift
    // as RowDotQ5KAvx2.
    private static void DequantQ5KAvx2(byte* row, float* outPtr, int cols)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q5KBlockBytes;
            byte* blockEnd = block + Q5KBlockBytes;
            float dHalf = ReadHalfLE(block);
            float minHalf = ReadHalfLE(block + 2);

            for (int j = 0; j < 8; j++)
            {
                int subStart = baseIdx + j * 32;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                float scale = dHalf * sc;
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                float negShift = -(minHalf * m);
                Vector256<float> shiftV = Avx.BroadcastScalarToVector256(&negShift);

                byte* q = block + 48 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                byte* qh = block + 16 + 32 * (j / 2); // 5th bits for the same window bytes, bit j per byte
                int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<int> five = LoadQ5KChunk(q, qh, blockEnd, j, c);
                    Vector256<float> qf = Avx.ConvertToVector256Single(five); // eight 5-bit values in element order
                    Avx.Store(outPtr + subStart + c, Fma.MultiplyAdd(scaleV, qf, shiftV));
                }

                for (int l = (limit / 8) * 8; l < limit; l++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    int five = ((q[l] >> (4 * (j & 1))) & 0xF) | (((qh[l] >> j) & 1) << 4);
                    outPtr[subStart + l] = five * scale - minHalf * m;
                }
            }
        }
    }

    // Q6_K row expansion: w = d * sc_g * (q - 32) per group. Same 6-bit unpack and signed conversion as
    // RowDotQ6KAvx2.
    private static void DequantQ6KAvx2(byte* row, float* outPtr, int cols)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q6KBlockBytes;
            float dHalf = ReadHalfLE(block + 208);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                int quarter = k >> 2; // 0..3: which 32-element quarter of the half
                int l = 16 * (k & 1); // window offset inside the half: 0 or 16
                byte* q = block + 64 * h + 32 * (quarter & 1) + l; // window: bytes [q, q+16) always inside the block
                byte* qh = block + 128 + 32 * h + l; // high 2 bits for the same window bytes, shift 2*quarter
                float scale = dHalf * (sbyte)block[192 + g];
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<int> six = LoadQ6KChunk(q, qh, quarter, c);
                    Avx.Store(outPtr + subStart + c, Avx.Multiply(scaleV, Avx.ConvertToVector256Single(six))); // signed 6-bit values, pre-centered
                }

                for (int e = (limit / 8) * 8; e < limit; e++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    int six = ((q[e] >> (4 * (quarter >> 1))) & 0xF) | (((qh[e] >> (2 * quarter)) & 3) << 4);
                    outPtr[subStart + e] = (six - 32) * scale;
                }
            }
        }
    }

    private static Vector128<byte> GetNibbleMask() => Vector128.Create((byte)0xF);

    // pshufb selector: vector byte lane i = source byte 4+i (bytes 4..7 down to lanes 0..3).
    private static Vector128<byte> GetUpperBytesShuffle() => Vector128.Create((byte)4, (byte)5, (byte)6, (byte)7, (byte)4, (byte)5, (byte)6, (byte)7, (byte)4, (byte)5, (byte)6, (byte)7, (byte)4, (byte)5, (byte)6, (byte)7);

    // pshufb selector: vector byte lane i = source byte 8+i (high half of the window down to lanes 0..7).
    private static Vector128<byte> GetSlideHighBytesShuffle() => Vector128.Create((byte)8, (byte)9, (byte)10, (byte)11, (byte)12, (byte)13, (byte)14, (byte)15, (byte)8, (byte)9, (byte)10, (byte)11, (byte)12, (byte)13, (byte)14, (byte)15);

    // Constant per-lane bit masks for Q5_0 5th-bit extraction: lane i tests bit i of the broadcast qh field byte.
    private static Vector256<int> GetQ5_0BitMasks() => Vector256.Create(1, 2, 4, 8, 16, 32, 64, 128);

    private static float DotCore(float* a, float* b, int count)
    {
        Vector256<float> acc = Vector256<float>.Zero;

        int i = 0;
        for (; i + 8 <= count; i += 8)
        {
            acc = Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), acc);
        }

        float tail = 0f;
        for (; i < count; i++)
        {
            tail += a[i] * b[i];
        }

        return HSum(acc) + tail;
    }

    private static float DotScalar(float* a, float* b, int count)
    {
        float acc = 0f;
        for (int i = 0; i < count; i++)
        {
            acc += a[i] * b[i];
        }

        return acc;
    }

    private static void AxpyCore(float* x, float* y, float alpha, int count)
    {
        Vector256<float> av = Avx.BroadcastScalarToVector256(&alpha);

        int i = 0;
        for (; i + 8 <= count; i += 8)
        {
            Avx.Store(y + i, Fma.MultiplyAdd(av, Avx.LoadVector256(x + i), Avx.LoadVector256(y + i)));
        }

        for (; i < count; i++)
        {
            y[i] += alpha * x[i];
        }
    }

    private static void ScaleCore(float* y, float s, int count)
    {
        Vector256<float> sv = Avx.BroadcastScalarToVector256(&s);

        int i = 0;
        for (; i + 8 <= count; i += 8)
        {
            Avx.Store(y + i, Avx.Multiply(Avx.LoadVector256(y + i), sv));
        }

        for (; i < count; i++)
        {
            y[i] *= s;
        }
    }

    private static float HSum(Vector256<float> v)
    {
        Vector128<float> sum = Sse.Add(v.GetLower(), v.GetUpper());

        sum = Sse.Add(sum, Sse.Shuffle(sum, sum, 0x4E)); // swap the two 64-bit halves
        sum = Sse.Add(sum, Sse.Shuffle(sum, sum, 0xB1)); // swap adjacent dword lanes
        return Sse.UnpackLow(sum, sum).GetElement(0);
    }

    /// <summary>
    /// Persistent worker pool: workers park on a monitor and are woken per job; the dispatcher thread joins the
    /// claim loop so small jobs pay no wake-up cost. All block claims happen under one lock, which is negligible
    /// next to a row-block of kernel work. Workers outlive individual GEMV calls until Dispose. Each worker pins
    /// itself to <c>coreMask</c> (when configured) at loop start and re-pins defensively after any wake-up gap —
    /// pinning lives in the thread body so pool bookkeeping can never undo it — and records the logical processor of
    /// every block it executes while execution sampling is enabled on the owning backend. The dispatcher's own blocks
    /// are deliberately neither pinned nor sampled: that thread belongs to the caller, not to the pool.
    /// </summary>
    private sealed class WorkerPool : IDisposable
    {
        // Sampling stays bounded so a long run cannot grow memory without limit; 4096 covers any test workload.
        private const int MaxSamples = 4096;

        private readonly object _gate = new();
        private readonly Thread[]? _workers;
        private readonly ulong? _coreMask; // processor set for every pool worker, or null when the model is unpinned
        private ConcurrentQueue<int>? _samples; // armed lazily via Interlocked so volatile-field type rules hold
        private int _sampleCount;

        private IGemvJob? _job;
        private int _totalBlocks;
        private int _nextBlock;
        private int _pendingBlocks;
        private bool _disposed;

        public WorkerPool(int workerCount, ulong? coreMask)
        {
            _coreMask = coreMask;

            if (workerCount == 0)
            {
                return; // pool exists only to satisfy GetOrCreatePool; Run is never called for it
            }

            _workers = new Thread[workerCount];
            for (int i = 0; i < workerCount; i++)
            {
                int index = i; // captured loop variable per worker name only
                Thread thread = new(WorkerLoop)
                {
                    IsBackground = true,
                    Name = $"lair-gemv-worker-{index}",
                };

                _workers[i] = thread;
                thread.Start();
            }
        }

        public int WorkerCapacity => _workers?.Length ?? 0;

        public void Run(int blockCount, IGemvJob job)
        {
            lock (_gate)
            {
                _job = job;
                _totalBlocks = blockCount;
                _nextBlock = 0;
                _pendingBlocks = blockCount;
                Monitor.PulseAll(_gate);
            }

            while (true)
            {
                int block;
                lock (_gate)
                {
                    if (_disposed || _nextBlock >= _totalBlocks)
                    {
                        break;
                    }

                    block = _nextBlock++;
                }

                job.ProcessBlock(block);

                lock (_gate)
                {
                    if (--_pendingBlocks == 0)
                    {
                        Monitor.PulseAll(_gate);
                    }
                }
            }

            lock (_gate)
            {
                while (_pendingBlocks != 0 && !_disposed)
                {
                    Monitor.Wait(_gate);
                }
            }
        }

        public void Dispose()
        {
            lock (_gate)
            {
                _disposed = true;
                Monitor.PulseAll(_gate);
            }

            if (_workers is null)
            {
                return;
            }

            foreach (Thread worker in _workers)
            {
                worker.Join();
            }
        }

        /// <summary>Arms per-block processor sampling; the interlocked store publishes the queue to every worker thread.</summary>
        internal void EnableSampling()
        {
            if (_samples is null)
            {
                Interlocked.CompareExchange(ref _samples, [], null); // a racing second arming keeps the first queue and drops its own
            }
        }

        /// <summary>Snapshots the processor number recorded by each sampled worker-block execution so far.</summary>
        internal IReadOnlyList<int> SampleSnapshot() => Volatile.Read(ref _samples) is ConcurrentQueue<int> queue ? [.. queue] : [];

        private void WorkerLoop()
        {
            // Pin at thread start: affinity is process-lifetime state of this thread, so one call covers the whole
            // loop. Unpinned models pass a zero mask and get the trivial accept back; Linux hosts get the documented
            // no-op here. The pin re-asserts itself below on every wake-up for defense in depth.
            ThreadAffinity.ApplyCoreSet(_coreMask ?? 0UL);

            while (true)
            {
                if (_coreMask is not null)
                {
                    // Defensive re-pin at each wake-up: cheap next to kernel work, and it keeps the pin even if an
                    // external tool or a thread-pool reuse ever reset this worker's affinity. The start-of-loop call
                    // above already covers unpinned models with a trivial no-op return.
                    ThreadAffinity.ApplyCoreSet(_coreMask.Value);
                }

                IGemvJob job;
                int block;
                bool claimed;
                lock (_gate)
                {
                    while (!_disposed && _nextBlock >= _totalBlocks)
                    {
                        Monitor.Wait(_gate);
                    }

                    if (_disposed)
                    {
                        return;
                    }

                    // A woken worker may find all blocks already claimed: claim fails, so loop back to sleep.
                    block = _nextBlock++;
                    claimed = block < _totalBlocks;
                    job = _job!;
                }

                if (_coreMask is not null)
                {
                    // Defensive re-pin on every wake-up: cheap next to kernel work, and it keeps the pin even if an
                    // external tool or a thread-pool reuse ever reset this worker's affinity.
                    ThreadAffinity.ApplyCoreSet(_coreMask.Value);
                }

                ConcurrentQueue<int>? samples = Volatile.Read(ref _samples);
                if (claimed && samples is not null && _sampleCount < MaxSamples)
                {
                    // Approximate cap under concurrent workers: a few blocks may race past it, which is fine for an
                    // unbounded-memory guard on a test-only hook. The number records where this block executes; the
                    // pin above guarantees a pinned worker answers from inside its core set.
                    _sampleCount++;
                    samples.Enqueue(ThreadAffinity.CurrentProcessorNumber());
                }

                job.ProcessBlock(block);

                lock (_gate)
                {
                    if (--_pendingBlocks == 0)
                    {
                        Monitor.PulseAll(_gate);
                    }
                }
            }
        }
    }
}
