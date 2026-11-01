using System.Buffers;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Lair.Core.Contracts;

namespace Lair.Backends.Cpu;

/// <summary>
/// CPU compute backend: aligned arena management plus dense kernels. GEMV fuses dequantization for F16/Q8_0/Q4_K
/// in a single pass over the packed bytes, taking an AVX2 fast path when the runtime supports it and scalar
/// reference paths otherwise. Rows are partitioned into blocks executed by a persistent worker pool; each row is
/// computed independently so results are bit-identical regardless of thread count.
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

    // Output rows per scheduling unit for threaded GEMV.
    private const int RowBlock = 64;

    // Upper bound on activation vectors per batched GEMV call.
    private const int MaxBatch = 16;

    // Below this row count the threading overhead is not worth paying.
    private const int ParallelRowThreshold = 256;

    private readonly int _threads;
    private readonly bool _useAvx2;
    private WorkerPool? _pool;

    public CpuBackend(int threads = 0, bool allowAvx2 = true)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(threads);
        _threads = threads == 0 ? Math.Min(Environment.ProcessorCount, 16) : threads;
        _useAvx2 = allowAvx2 && Avx2.IsSupported && Fma.IsSupported && Ssse3.IsSupported;
    }

    public string DeviceId => "cpu";

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
            bool parallel = _threads > 1 && rows >= ParallelRowThreshold;

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
            bool parallel = _threads > 1 && rows >= ParallelRowThreshold;

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
        WorkerPool? pool = _pool;
        if (pool is not null)
        {
            return pool;
        }

        // Never used when no extra workers exist: parallel GEMV requires threads > 1.
        return _pool = new WorkerPool(Math.Max(_threads - 1, 0));
    }

    private static float RowDot(byte* row, float* x, int cols, QuantFormat format, bool avx2)
    {
        return format switch
        {
            QuantFormat.F32 => avx2 ? DotCore((float*)row, x, cols) : DotScalar((float*)row, x, cols),
            QuantFormat.F16 => avx2 ? RowDotF16Avx2(row, x, cols) : RowDotF16Scalar(row, x, cols),
            QuantFormat.Q8_0 => avx2 ? RowDotQ8Avx2(row, x, cols) : RowDotQ8Scalar(row, x, cols),
            QuantFormat.Q4_K => avx2 ? RowDotQ4KAvx2(row, x, cols) : RowDotQ4KScalar(row, x, cols),
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

    private static Vector128<byte> GetNibbleMask() => Vector128.Create((byte)0xF);

    // pshufb selector: vector byte lane i = source byte 4+i (bytes 4..7 down to lanes 0..3).
    private static Vector128<byte> GetUpperBytesShuffle() => Vector128.Create((byte)4, (byte)5, (byte)6, (byte)7, (byte)4, (byte)5, (byte)6, (byte)7, (byte)4, (byte)5, (byte)6, (byte)7, (byte)4, (byte)5, (byte)6, (byte)7);

    // pshufb selector: vector byte lane i = source byte 8+i (high half of the window down to lanes 0..7).
    private static Vector128<byte> GetSlideHighBytesShuffle() => Vector128.Create((byte)8, (byte)9, (byte)10, (byte)11, (byte)12, (byte)13, (byte)14, (byte)15, (byte)8, (byte)9, (byte)10, (byte)11, (byte)12, (byte)13, (byte)14, (byte)15);

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
    /// next to a row-block of kernel work. Workers outlive individual GEMV calls until Dispose.
    /// </summary>
    private sealed class WorkerPool : IDisposable
    {
        private readonly object _gate = new();
        private readonly Thread[]? _workers;

        private IGemvJob? _job;
        private int _totalBlocks;
        private int _nextBlock;
        private int _pendingBlocks;
        private bool _disposed;

        public WorkerPool(int workerCount)
        {
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

        private void WorkerLoop()
        {
            while (true)
            {
                IGemvJob job;
                int block;
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
                    job = _job!;
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
