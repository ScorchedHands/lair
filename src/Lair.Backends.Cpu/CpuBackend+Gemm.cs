using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Lair.Contracts;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
    // Blocked-matmul geometry. A weight tile is one run of weight rows by one column block wide; it is dequantized
    // once and then consumed by every activation vector of the batch, so the unpacking work happens once per tile
    // instead of once per activation vector. The column block is one quant super-block wide, which keeps each row's
    // 32- and 256-element quant blocks entirely inside a single tile and makes the tile's eight-element chunks the
    // same chunks the single-vector and batched GEMV kernels walk: every output element therefore accumulates over
    // the same successive chunks, in the same order, with the same operand association as the GEMV reference for it.
    private const int GemmTileRows = 8;
    private const int GemmTileCols = QkK;

    // Quant blocks per column block, and the scratch each tile group needs: one eight-lane accumulator chain and one
    // scalar tail per (tile row, activation vector), plus the tile itself. All of it is stack scratch reused by
    // every tile group of every call; a blocked matmul allocates nothing on its hot path.
    private const int GemmBlocksPerTile = GemmTileCols / Qk8;
    private const int GemmAccLanes = GemmTileRows * MaxBatch * 8;
    private const int GemmAccTails = GemmTileRows * MaxBatch;

    /// <inheritdoc/>
    public bool SupportsGemm(QuantFormat format) =>
        format is QuantFormat.Q8_0 or QuantFormat.Q4_K or QuantFormat.IQ4_XS or QuantFormat.Q6_K;

    /// <inheritdoc/>
    public void GemmBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs)
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

        if (!SupportsGemm(w.Format))
        {
            throw new ArgumentException(
                $"no blocked matmul exists in this build for weights in format {w.Format}: only {QuantFormat.Q8_0}, " +
                $"{QuantFormat.Q4_K}, {QuantFormat.IQ4_XS}, and {QuantFormat.Q6_K} execute through the dense-matrix weight path.",
                nameof(w));
        }

        int stride = (int)strideBytes;
        QuantFormat format = w.Format;

        // This is a single-matrix dense batched pass, so it consults the same unpromoted static-partition crossover GemvBatch does: only a stated
        // width at or below this call's count asks for contiguous per-participant ranges, and every resolved posture (which states none) keeps dynamic claiming.
        CpuKernelTuning dispatchTuning = _tuning;
        bool staticRowPartition = dispatchTuning.StaticRowPartitionBatchWidth > 0 && count >= dispatchTuning.StaticRowPartitionBatchWidth;

        // Same raw-pointer job contract as GemvBatch: pinned spans stay stable for the duration of the call and every
        // claimed block finishes before WorkerPool.Run returns. A scheduling block owns whole weight rows and sweeps
        // every column for them, so each output element is produced by one uninterrupted ascending walk of its own
        // accumulator chain: partitioning rows across workers changes nothing about the bytes it writes.
        fixed (byte* pBytes = w.Bytes)
        fixed (float* pIn = inputs)
        fixed (float* pOut = outputs)
        {
            RunMatmulDispatch(pBytes, stride, pIn, pOut, rows, cols, count, format, staticRowPartition);
        }
    }

    // The blocked-matmul dispatch the dense-matrix selection and the batched-row hybrid route share: one tuning snapshot,
    // the same row-block partition, the same raw-pointer job contract. Callers hold the pinned spans. <paramref name="staticRowPartition"/>
    // is each call site's own decision against its tuning snapshot — a single-matrix dense pass may state it, a fused pair's two per-matrix
    // tile dispatches never does (that traversal keeps today's dynamic claim loop at any stated width), and an unthreaded run ignores it
    // entirely because there is no participant to partition. Whichever schedule runs, a block owns whole weight rows and sweeps every column
    // for them, so each output element still comes from one uninterrupted ascending walk of its own accumulator chain.
    private void RunMatmulDispatch(byte* pBytes, int stride, float* pIn, float* pOut, int rows, int cols, int count, QuantFormat format, bool staticRowPartition)
    {
        CpuKernelTuning tuning = _tuning;
        int workers = _effectiveThreads;
        bool parallel = workers > 1 && rows >= tuning.ParallelRowThreshold;

        if (!parallel)
        {
            RunGemmRows(pBytes, pIn, pOut, stride, rows, cols, count, format, _useAvx2, 0, rows);
            return;
        }

        WorkerPool pool = GetOrCreatePool();
        RunThreadedDispatch(pool, (rows + tuning.RowBlock - 1) / tuning.RowBlock, workers, staticRowPartition, new GemmRowJob(pBytes, pIn, pOut, stride, rows, cols, count, format, _useAvx2, tuning));
    }

    // One run of weight rows: they advance through the column walk together, in tiles of GemmTileRows rows so a tile
    // is built once and read by every activation vector of the batch.
    private static void RunGemmRows(byte* bytes, float* inputs, float* outBase, int stride, int rows, int cols, int count, QuantFormat format, bool avx2, int firstRow, int lastRow)
    {
        if (avx2)
        {
            switch (format)
            {
                case QuantFormat.Q8_0:
                    GemmQ8Avx2(bytes, stride, firstRow, lastRow, inputs, count, rows, cols, outBase);
                    return;
                case QuantFormat.Q4_K:
                    GemmQ4KAvx2(bytes, stride, firstRow, lastRow, inputs, count, rows, cols, outBase);
                    return;
                case QuantFormat.Q6_K:
                    GemmQ6KAvx2(bytes, stride, firstRow, lastRow, inputs, count, rows, cols, outBase);
                    return;
                default:
                    GemmIQ4XAvx2(bytes, stride, firstRow, lastRow, inputs, count, rows, cols, outBase);
                    return;
            }
        }

        switch (format)
        {
            case QuantFormat.Q8_0:
                GemmQ8Scalar(bytes, stride, firstRow, lastRow, inputs, count, rows, cols, outBase);
                break;
            case QuantFormat.Q4_K:
                GemmQ4KScalar(bytes, stride, firstRow, lastRow, inputs, count, rows, cols, outBase);
                break;
            case QuantFormat.Q6_K:
                GemmQ6KScalar(bytes, stride, firstRow, lastRow, inputs, count, rows, cols, outBase);
                break;
            default:
                GemmIQ4XScalar(bytes, stride, firstRow, lastRow, inputs, count, rows, cols, outBase);
                break;
        }
    }

    /// <summary>
    /// Row-block executor for one blocked-matmul call: each claimed block owns whole weight rows and walks every
    /// column for them, so any partition of blocks yields byte-identical output. Activation vectors are never split.
    /// </summary>
    private sealed unsafe class GemmRowJob : IGemvJob
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
        private readonly int _rowBlock; // captured from the dispatch snapshot so every claimed block uses the same width as Run's block count

        public GemmRowJob(byte* bytes, float* inputs, float* outPtr, int stride, int rows, int cols, int count, QuantFormat format, bool avx2, CpuKernelTuning tuning)
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
            _rowBlock = tuning.RowBlock;
        }

        public void ProcessBlock(int block)
        {
            int firstRow = block * _rowBlock;
            RunGemmRows(_bytes, _inputs, _out, _stride, _rows, _cols, _count, _format, _avx2, firstRow, Math.Min(firstRow + _rowBlock, _rows));
        }
    }

    // ---------------------------------------------------------------- Q8_0

    // Q8_0 tile: each block's 32 int8 quants widened to FP32, with the block's half-float scale carried alongside.
    // The reference multiplies an activation value by the quant and only then by the block scale, so the scale stays
    // outside the tile and the multiply-accumulate reproduces (x * q) * d exactly as RowDotQ8Avx2 does; the scalar
    // cleanup of a partial final block reproduces x * (q * d) exactly as that kernel's tail does.
    private static void GemmQ8Avx2(byte* bytes, int stride, int rowFrom, int rowTo, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> tile = stackalloc float[GemmTileRows * GemmTileCols];
        Span<float> scale = stackalloc float[GemmTileRows * GemmBlocksPerTile];
        Span<float> acc = stackalloc float[GemmAccLanes];
        Span<float> tail = stackalloc float[GemmAccTails];

        fixed (float* pTile = tile)
        fixed (float* pScale = scale)
        fixed (float* pAcc = acc)
        fixed (float* pTail = tail)
        {
            for (int rb = rowFrom; rb < rowTo; rb += GemmTileRows)
            {
                int rn = Math.Min(GemmTileRows, rowTo - rb);
                acc.Slice(0, rn * MaxBatch * 8).Clear();
                tail.Slice(0, rn * MaxBatch).Clear();

                for (int cb = 0; cb < cols; cb += GemmTileCols)
                {
                    int cn = Math.Min(GemmTileCols, cols - cb);
                    int blocks = (cn + Qk8 - 1) / Qk8;

                    for (int r = 0; r < rn; r++)
                    {
                        byte* row = bytes + (long)(rb + r) * stride;
                        float* tileRow = pTile + (long)r * GemmTileCols;

                        for (int blk = 0; blk < blocks; blk++)
                        {
                            byte* packed = row + (long)(cb / Qk8 + blk) * (Qk8 + 2);
                            float d = ReadHalfLE(packed);
                            pScale[r * GemmBlocksPerTile + blk] = d;

                            int limit = Math.Min(Qk8, cols - (cb + blk * Qk8)); // padded quants are zero: stop at the real column count.
                            int lane = blk * Qk8;
                            for (int l = 0; l + 8 <= limit; l += 8)
                            {
                                Avx.Store(tileRow + lane + l, Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32((sbyte*)(packed + 2 + l))));
                            }

                            for (int l = (limit / 8) * 8; l < limit; l++)
                            {
                                tileRow[lane + l] = (float)(sbyte)packed[2 + l];
                            }
                        }
                    }

                    for (int r = 0; r < rn; r++)
                    {
                        float* tileRow = pTile + (long)r * GemmTileCols;

                        for (int b = 0; b < count; b++)
                        {
                            float* chain = pAcc + (r * MaxBatch + b) * 8;
                            float* tailSlot = pTail + r * MaxBatch + b;
                            float* xRow = inputs + (long)b * cols + cb;

                            Vector256<float> accV = Avx.LoadVector256(chain);
                            float tailAcc = *tailSlot;

                            for (int blk = 0; blk < blocks; blk++)
                            {
                                int limit = Math.Min(Qk8, cols - (cb + blk * Qk8));
                                float dLocal = pScale[r * GemmBlocksPerTile + blk];
                                Vector256<float> dv = Avx.BroadcastScalarToVector256(&dLocal);
                                int lane = blk * Qk8;

                                for (int c = 0; c + 8 <= limit; c += 8)
                                {
                                    accV = Fma.MultiplyAdd(
                                        Avx.Multiply(Avx.LoadVector256(xRow + lane + c), Avx.LoadVector256(tileRow + lane + c)),
                                        dv,
                                        accV);
                                }

                                for (int k = (limit / 8) * 8; k < limit; k++)
                                {
                                    tailAcc += xRow[lane + k] * (tileRow[lane + k] * dLocal);
                                }
                            }

                            Avx.Store(chain, accV);
                            *tailSlot = tailAcc;
                        }
                    }
                }

                FinishGemmTile(pAcc, pTail, rn, count, rows, rb, outBase);
            }
        }
    }

    // Q8_0 scalar tile: w = d * q per block, one accumulator advanced in ascending column order exactly as
    // RowDotQ8Scalar advances its own.
    private static void GemmQ8Scalar(byte* bytes, int stride, int rowFrom, int rowTo, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> tile = stackalloc float[GemmTileRows * GemmTileCols];

        fixed (float* pTile = tile)
        {
            for (int rb = rowFrom; rb < rowTo; rb += GemmTileRows)
            {
                int rn = Math.Min(GemmTileRows, rowTo - rb);

                // The scalar route carries each output element's running sum in its own output slot: opened here once
                // per tile group, then advanced by the ascending column walk below and written nowhere else.
                for (int r = 0; r < rn; r++)
                {
                    for (int b = 0; b < count; b++)
                    {
                        outBase[(long)b * rows + rb + r] = 0f;
                    }
                }

                for (int cb = 0; cb < cols; cb += GemmTileCols)
                {
                    int cn = Math.Min(GemmTileCols, cols - cb);
                    int blocks = (cn + Qk8 - 1) / Qk8;

                    for (int r = 0; r < rn; r++)
                    {
                        byte* row = bytes + (long)(rb + r) * stride;
                        float* tileRow = pTile + (long)r * GemmTileCols;

                        for (int blk = 0; blk < blocks; blk++)
                        {
                            byte* packed = row + (long)(cb / Qk8 + blk) * (Qk8 + 2);
                            float d = ReadHalfLE(packed);
                            int limit = Math.Min(Qk8, cols - (cb + blk * Qk8));
                            int lane = blk * Qk8;

                            for (int l = 0; l < limit; l++)
                            {
                                tileRow[lane + l] = (float)(sbyte)packed[2 + l] * d;
                            }
                        }
                    }

                    for (int r = 0; r < rn; r++)
                    {
                        float* tileRow = pTile + (long)r * GemmTileCols;

                        for (int b = 0; b < count; b++)
                        {
                            float* xRow = inputs + (long)b * cols + cb;
                            float* outSlot = outBase + (long)b * rows + rb + r;
                            float acc = *outSlot;

                            for (int l = 0; l < cn; l++)
                            {
                                acc += xRow[l] * tileRow[l];
                            }

                            *outSlot = acc;
                        }
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------- Q4_K

    // Q4_K tile: w = d*sc_j*q - dmin*m_j per group, materialized with the same fused scale-and-shift the reference
    // chunk builds, and with the same scalar arithmetic for the partial final chunk of a group that kernel leaves to
    // its scalar tail. The column block is one 256-element super-block wide, so the tile walk visits the same eight
    // 32-element groups and the same eight-element chunks the vector kernels walk for a row.
    //
    // The dot is batch-major: one activation-vector load per chunk feeds all eight tile rows' accumulator chains, where
    // each chain stays live in a YMM register across the whole column block instead of round-tripping through stack
    // scratch once per FMA — the shape the Q6_K tile kernel already runs. Each chain still advances through the same
    // ascending group and chunk positions with the same FMA association and the same scalar-tail cleanup, so every
    // output lands bit-identical to a row-major walk of that row's chunks: grouping here only chooses which tile rows
    // share one activation pass, never a chain's element order or its reduction.
    private static void GemmQ4KAvx2(byte* bytes, int stride, int rowFrom, int rowTo, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> tile = stackalloc float[GemmTileRows * GemmTileCols];
        Span<float> acc = stackalloc float[GemmAccLanes];
        Span<float> tail = stackalloc float[GemmAccTails];

        fixed (float* pTile = tile)
        fixed (float* pAcc = acc)
        fixed (float* pTail = tail)
        {
            for (int rb = rowFrom; rb < rowTo; rb += GemmTileRows)
            {
                int rn = Math.Min(GemmTileRows, rowTo - rb);
                acc.Slice(0, rn * MaxBatch * 8).Clear();
                tail.Slice(0, rn * MaxBatch).Clear();

                for (int cb = 0; cb < cols; cb += GemmTileCols)
                {
                    for (int r = 0; r < rn; r++)
                    {
                        byte* block = bytes + (long)(rb + r) * stride + (long)(cb / QkK) * Q4KBlockBytes;
                        byte* blockEnd = block + Q4KBlockBytes;
                        float* tileRow = pTile + (long)r * GemmTileCols;

                        float dHalf = ReadHalfLE(block);
                        float minHalf = ReadHalfLE(block + 2);

                        for (int j = 0; j < 8; j++)
                        {
                            int subStart = cb + j * 32;
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
                                    window = Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte();
                                }

                                Vector128<byte> masked = Sse2.And(window, GetNibbleMask());

                                Vector128<int> lo = Sse41.ConvertToVector128Int32(masked);
                                Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle()));

                                Avx.Store(tileRow + j * 32 + c, Fma.MultiplyAdd(scaleV, Avx.ConvertToVector256Single(Vector256.Create(lo, hi)), shiftV));
                            }

                            for (int k = (limit / 8) * 8; k < limit; k++)
                            {
                                tileRow[j * 32 + k] = ((q[k] >> (4 * (j & 1))) & 0xF) * scale - minHalf * m;
                            }
                        }
                    }

                    if (rn == GemmTileRows)
                    {
                        // Full tile: the batch-major register route. Eight chains (one per tile row) stay live across the
                        // whole column block beside the activation vector; the tile rows are the FMA memory operands.
                        for (int b = 0; b < count; b++)
                        {
                            float* xRow = inputs + (long)b * cols + cb;

                            Vector256<float> c0 = Avx.LoadVector256(pAcc + (0 * MaxBatch + b) * 8);
                            Vector256<float> c1 = Avx.LoadVector256(pAcc + (1 * MaxBatch + b) * 8);
                            Vector256<float> c2 = Avx.LoadVector256(pAcc + (2 * MaxBatch + b) * 8);
                            Vector256<float> c3 = Avx.LoadVector256(pAcc + (3 * MaxBatch + b) * 8);
                            Vector256<float> c4 = Avx.LoadVector256(pAcc + (4 * MaxBatch + b) * 8);
                            Vector256<float> c5 = Avx.LoadVector256(pAcc + (5 * MaxBatch + b) * 8);
                            Vector256<float> c6 = Avx.LoadVector256(pAcc + (6 * MaxBatch + b) * 8);
                            Vector256<float> c7 = Avx.LoadVector256(pAcc + (7 * MaxBatch + b) * 8);

                            float t0 = pTail[0 * MaxBatch + b];
                            float t1 = pTail[1 * MaxBatch + b];
                            float t2 = pTail[2 * MaxBatch + b];
                            float t3 = pTail[3 * MaxBatch + b];
                            float t4 = pTail[4 * MaxBatch + b];
                            float t5 = pTail[5 * MaxBatch + b];
                            float t6 = pTail[6 * MaxBatch + b];
                            float t7 = pTail[7 * MaxBatch + b];

                            for (int j = 0; j < 8; j++)
                            {
                                int subStart = cb + j * 32;
                                if (subStart >= cols)
                                {
                                    break;
                                }

                                int limit = Math.Min(32, cols - subStart);
                                int lane = j * 32;

                                for (int c = 0; c + 8 <= limit; c += 8)
                                {
                                    Vector256<float> xv = Avx.LoadVector256(xRow + lane + c); // one activation load feeds all eight tile rows' chains.
                                    c0 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + lane + c), c0);
                                    c1 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + GemmTileCols + lane + c), c1);
                                    c2 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + 2L * GemmTileCols + lane + c), c2);
                                    c3 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + 3L * GemmTileCols + lane + c), c3);
                                    c4 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + 4L * GemmTileCols + lane + c), c4);
                                    c5 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + 5L * GemmTileCols + lane + c), c5);
                                    c6 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + 6L * GemmTileCols + lane + c), c6);
                                    c7 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + 7L * GemmTileCols + lane + c), c7);
                                }

                                for (int k = (limit / 8) * 8; k < limit; k++)
                                {
                                    int p = lane + k;
                                    t0 += xRow[p] * pTile[p];
                                    t1 += xRow[p] * pTile[GemmTileCols + p];
                                    t2 += xRow[p] * pTile[2L * GemmTileCols + p];
                                    t3 += xRow[p] * pTile[3L * GemmTileCols + p];
                                    t4 += xRow[p] * pTile[4L * GemmTileCols + p];
                                    t5 += xRow[p] * pTile[5L * GemmTileCols + p];
                                    t6 += xRow[p] * pTile[6L * GemmTileCols + p];
                                    t7 += xRow[p] * pTile[7L * GemmTileCols + p];
                                }
                            }

                            Avx.Store(pAcc + (0 * MaxBatch + b) * 8, c0);
                            Avx.Store(pAcc + (1 * MaxBatch + b) * 8, c1);
                            Avx.Store(pAcc + (2 * MaxBatch + b) * 8, c2);
                            Avx.Store(pAcc + (3 * MaxBatch + b) * 8, c3);
                            Avx.Store(pAcc + (4 * MaxBatch + b) * 8, c4);
                            Avx.Store(pAcc + (5 * MaxBatch + b) * 8, c5);
                            Avx.Store(pAcc + (6 * MaxBatch + b) * 8, c6);
                            Avx.Store(pAcc + (7 * MaxBatch + b) * 8, c7);

                            pTail[0 * MaxBatch + b] = t0;
                            pTail[1 * MaxBatch + b] = t1;
                            pTail[2 * MaxBatch + b] = t2;
                            pTail[3 * MaxBatch + b] = t3;
                            pTail[4 * MaxBatch + b] = t4;
                            pTail[5 * MaxBatch + b] = t5;
                            pTail[6 * MaxBatch + b] = t6;
                            pTail[7 * MaxBatch + b] = t7;
                        }
                    }
                    else
                    {
                        // Partial final tile group (fewer than eight rows): the row-major dot, the same per-chain walks
                        // the other formats' partial tiles run. Chain order and reduction are unchanged here too.
                        for (int r = 0; r < rn; r++)
                        {
                            float* tileRow = pTile + (long)r * GemmTileCols;

                            for (int b = 0; b < count; b++)
                            {
                                float* chain = pAcc + (r * MaxBatch + b) * 8;
                                float* tailSlot = pTail + r * MaxBatch + b;
                                float* xRow = inputs + (long)b * cols + cb;

                                Vector256<float> accV = Avx.LoadVector256(chain);
                                float tailAcc = *tailSlot;

                                for (int j = 0; j < 8; j++)
                                {
                                    int subStart = cb + j * 32;
                                    if (subStart >= cols)
                                    {
                                        break;
                                    }

                                    int limit = Math.Min(32, cols - subStart);

                                    for (int c = 0; c + 8 <= limit; c += 8)
                                    {
                                        accV = Fma.MultiplyAdd(Avx.LoadVector256(xRow + j * 32 + c), Avx.LoadVector256(tileRow + j * 32 + c), accV);
                                    }

                                    for (int k = (limit / 8) * 8; k < limit; k++)
                                    {
                                        tailAcc += xRow[j * 32 + k] * tileRow[j * 32 + k];
                                    }
                                }

                                Avx.Store(chain, accV);
                                *tailSlot = tailAcc;
                            }
                        }
                    }
                }

                FinishGemmTile(pAcc, pTail, rn, count, rows, rb, outBase);
            }
        }
    }

    // Q4_K scalar tile: nibble = q * scale - min * m per group, one accumulator advanced in ascending column order
    // exactly as RowDotQ4KScalar advances its own.
    private static void GemmQ4KScalar(byte* bytes, int stride, int rowFrom, int rowTo, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> tile = stackalloc float[GemmTileRows * GemmTileCols];

        fixed (float* pTile = tile)
        {
            for (int rb = rowFrom; rb < rowTo; rb += GemmTileRows)
            {
                int rn = Math.Min(GemmTileRows, rowTo - rb);

                // The scalar route carries each output element's running sum in its own output slot: opened here once
                // per tile group, then advanced by the ascending column walk below and written nowhere else.
                for (int r = 0; r < rn; r++)
                {
                    for (int b = 0; b < count; b++)
                    {
                        outBase[(long)b * rows + rb + r] = 0f;
                    }
                }

                for (int cb = 0; cb < cols; cb += GemmTileCols)
                {
                    int cn = Math.Min(GemmTileCols, cols - cb);

                    for (int r = 0; r < rn; r++)
                    {
                        byte* block = bytes + (long)(rb + r) * stride + (long)(cb / QkK) * Q4KBlockBytes;
                        float* tileRow = pTile + (long)r * GemmTileCols;

                        float d = ReadHalfLE(block);
                        float min = ReadHalfLE(block + 2);

                        for (int j = 0; j < 8; j++)
                        {
                            int subStart = cb + j * 32;
                            if (subStart >= cols)
                            {
                                break;
                            }

                            GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                            float scale = d * sc;
                            float shift = min * m;

                            byte* q = block + 16 + 32 * (j / 2);
                            int limit = Math.Min(32, cols - subStart);

                            for (int l = 0; l < limit; l++)
                            {
                                tileRow[j * 32 + l] = ((q[l] >> (4 * (j & 1))) & 0xF) * scale - shift;
                            }
                        }
                    }

                    for (int r = 0; r < rn; r++)
                    {
                        float* tileRow = pTile + (long)r * GemmTileCols;

                        for (int b = 0; b < count; b++)
                        {
                            float* xRow = inputs + (long)b * cols + cb;
                            float* outSlot = outBase + (long)b * rows + rb + r;
                            float acc = *outSlot;

                            for (int l = 0; l < cn; l++)
                            {
                                acc += xRow[l] * tileRow[l];
                            }

                            *outSlot = acc;
                        }
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------- Q6_K

    // Q6_K tile: w = d * sc_g * (q - 32) per group, materialized with the same 6-bit chunk unpack and fused scale the
    // reference kernels build, and with the same scalar arithmetic for a group's partial final chunk that those kernels
    // leave to their scalar tails. The column block is one 256-element super-block wide, so the tile walk visits the
    // same sixteen 16-element groups and the same eight-element chunks the vector kernels walk for a row.
    //
    // The dot is batch-major: one activation-vector load per chunk feeds all eight tile rows' accumulator chains, where
    // the batched-row scratch route re-reads the activation batch once per chain group. Each chain still advances
    // through the same ascending group and chunk positions with the same FMA association and the same scalar tail
    // cleanup, so every output lands bit-identical to that route's chain for its row — grouping here only chooses
    // which tile rows share one activation pass, never a chain's element order or its reduction.
    private static void GemmQ6KAvx2(byte* bytes, int stride, int rowFrom, int rowTo, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> tile = stackalloc float[GemmTileRows * GemmTileCols];
        Span<float> acc = stackalloc float[GemmAccLanes];
        Span<float> tail = stackalloc float[GemmAccTails];

        fixed (float* pTile = tile)
        fixed (float* pAcc = acc)
        fixed (float* pTail = tail)
        {
            for (int rb = rowFrom; rb < rowTo; rb += GemmTileRows)
            {
                int rn = Math.Min(GemmTileRows, rowTo - rb);
                acc.Slice(0, rn * MaxBatch * 8).Clear();
                tail.Slice(0, rn * MaxBatch).Clear();

                for (int cb = 0; cb < cols; cb += GemmTileCols)
                {
                    for (int r = 0; r < rn; r++)
                    {
                        byte* block = bytes + (long)(rb + r) * stride + (long)(cb / QkK) * Q6KBlockBytes;
                        float* tileRow = pTile + (long)r * GemmTileCols;

                        float dHalf = ReadHalfLE(block + 208);

                        for (int g = 0; g < 16; g++)
                        {
                            int subStart = cb + g * 16;
                            if (subStart >= cols)
                            {
                                break; // zero-padded tail block: no elements left to consume.
                            }

                            int h = g >> 3; // 128-element half of the super-block
                            int k = g & 7; // 16-element group inside the half
                            int quarter = k >> 1; // 0..3: which 32-element quarter of the half the group starts in
                            byte* q = block + 64 * h + 16 * (k & 3); // ql window: bytes [q, q+16) always inside the block
                            byte* qh = block + 128 + 32 * h + 16 * (k & 1); // 2-bit fields for the same elements, shift 2*quarter
                            float scale = dHalf * (sbyte)block[192 + g];
                            Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                            int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.

                            for (int c = 0; c + 8 <= limit; c += 8)
                            {
                                Vector256<int> six = LoadQ6KChunk(q, qh, quarter, c); // dequantization unpack, materialized once per chunk
                                Avx.Store(tileRow + g * 16 + c, Avx.Multiply(scaleV, Avx.ConvertToVector256Single(six))); // the same fused scale the dot chains consume
                            }

                            for (int e = (limit / 8) * 8; e < limit; e++)
                            {
                                // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                                int six = ((q[e] >> (4 * (quarter >> 1))) & 0xF) | (((qh[e] >> (2 * quarter)) & 3) << 4);
                                tileRow[g * 16 + e] = (six - 32) * scale;
                            }
                        }
                    }

                    if (rn == GemmTileRows)
                    {
                        // Full tile: the batch-major register route. Eight chains (one per tile row) stay live across the
                        // whole column block beside the activation vector; the tile rows are the FMA memory operands.
                        for (int b = 0; b < count; b++)
                        {
                            float* xRow = inputs + (long)b * cols + cb;

                            Vector256<float> c0 = Avx.LoadVector256(pAcc + (0 * MaxBatch + b) * 8);
                            Vector256<float> c1 = Avx.LoadVector256(pAcc + (1 * MaxBatch + b) * 8);
                            Vector256<float> c2 = Avx.LoadVector256(pAcc + (2 * MaxBatch + b) * 8);
                            Vector256<float> c3 = Avx.LoadVector256(pAcc + (3 * MaxBatch + b) * 8);
                            Vector256<float> c4 = Avx.LoadVector256(pAcc + (4 * MaxBatch + b) * 8);
                            Vector256<float> c5 = Avx.LoadVector256(pAcc + (5 * MaxBatch + b) * 8);
                            Vector256<float> c6 = Avx.LoadVector256(pAcc + (6 * MaxBatch + b) * 8);
                            Vector256<float> c7 = Avx.LoadVector256(pAcc + (7 * MaxBatch + b) * 8);

                            float t0 = pTail[0 * MaxBatch + b];
                            float t1 = pTail[1 * MaxBatch + b];
                            float t2 = pTail[2 * MaxBatch + b];
                            float t3 = pTail[3 * MaxBatch + b];
                            float t4 = pTail[4 * MaxBatch + b];
                            float t5 = pTail[5 * MaxBatch + b];
                            float t6 = pTail[6 * MaxBatch + b];
                            float t7 = pTail[7 * MaxBatch + b];

                            for (int g = 0; g < 16; g++)
                            {
                                int subStart = cb + g * 16;
                                if (subStart >= cols)
                                {
                                    break;
                                }

                                int limit = Math.Min(16, cols - subStart);
                                int lane = g * 16;

                                for (int c = 0; c + 8 <= limit; c += 8)
                                {
                                    Vector256<float> xv = Avx.LoadVector256(xRow + lane + c); // one activation load feeds all eight tile rows' chains.
                                    c0 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + lane + c), c0);
                                    c1 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + GemmTileCols + lane + c), c1);
                                    c2 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + 2L * GemmTileCols + lane + c), c2);
                                    c3 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + 3L * GemmTileCols + lane + c), c3);
                                    c4 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + 4L * GemmTileCols + lane + c), c4);
                                    c5 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + 5L * GemmTileCols + lane + c), c5);
                                    c6 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + 6L * GemmTileCols + lane + c), c6);
                                    c7 = Fma.MultiplyAdd(xv, Avx.LoadVector256(pTile + 7L * GemmTileCols + lane + c), c7);
                                }

                                for (int e = (limit / 8) * 8; e < limit; e++)
                                {
                                    int p = lane + e;
                                    t0 += xRow[p] * pTile[p];
                                    t1 += xRow[p] * pTile[GemmTileCols + p];
                                    t2 += xRow[p] * pTile[2L * GemmTileCols + p];
                                    t3 += xRow[p] * pTile[3L * GemmTileCols + p];
                                    t4 += xRow[p] * pTile[4L * GemmTileCols + p];
                                    t5 += xRow[p] * pTile[5L * GemmTileCols + p];
                                    t6 += xRow[p] * pTile[6L * GemmTileCols + p];
                                    t7 += xRow[p] * pTile[7L * GemmTileCols + p];
                                }
                            }

                            Avx.Store(pAcc + (0 * MaxBatch + b) * 8, c0);
                            Avx.Store(pAcc + (1 * MaxBatch + b) * 8, c1);
                            Avx.Store(pAcc + (2 * MaxBatch + b) * 8, c2);
                            Avx.Store(pAcc + (3 * MaxBatch + b) * 8, c3);
                            Avx.Store(pAcc + (4 * MaxBatch + b) * 8, c4);
                            Avx.Store(pAcc + (5 * MaxBatch + b) * 8, c5);
                            Avx.Store(pAcc + (6 * MaxBatch + b) * 8, c6);
                            Avx.Store(pAcc + (7 * MaxBatch + b) * 8, c7);

                            pTail[0 * MaxBatch + b] = t0;
                            pTail[1 * MaxBatch + b] = t1;
                            pTail[2 * MaxBatch + b] = t2;
                            pTail[3 * MaxBatch + b] = t3;
                            pTail[4 * MaxBatch + b] = t4;
                            pTail[5 * MaxBatch + b] = t5;
                            pTail[6 * MaxBatch + b] = t6;
                            pTail[7 * MaxBatch + b] = t7;
                        }
                    }
                    else
                    {
                        // Partial final tile group (fewer than eight rows): the row-major dot, the same per-chain walks
                        // the other formats' tile kernels run. Chain order and reduction are unchanged here too.
                        for (int r = 0; r < rn; r++)
                        {
                            float* tileRow = pTile + (long)r * GemmTileCols;

                            for (int b = 0; b < count; b++)
                            {
                                float* chain = pAcc + (r * MaxBatch + b) * 8;
                                float* tailSlot = pTail + r * MaxBatch + b;
                                float* xRow = inputs + (long)b * cols + cb;

                                Vector256<float> accV = Avx.LoadVector256(chain);
                                float tailAcc = *tailSlot;

                                for (int g = 0; g < 16; g++)
                                {
                                    int subStart = cb + g * 16;
                                    if (subStart >= cols)
                                    {
                                        break;
                                    }

                                    int limit = Math.Min(16, cols - subStart);

                                    for (int c = 0; c + 8 <= limit; c += 8)
                                    {
                                        accV = Fma.MultiplyAdd(Avx.LoadVector256(xRow + g * 16 + c), Avx.LoadVector256(tileRow + g * 16 + c), accV);
                                    }

                                    for (int e = (limit / 8) * 8; e < limit; e++)
                                    {
                                        tailAcc += xRow[g * 16 + e] * tileRow[g * 16 + e];
                                    }
                                }

                                Avx.Store(chain, accV);
                                *tailSlot = tailAcc;
                            }
                        }
                    }
                }

                FinishGemmTile(pAcc, pTail, rn, count, rows, rb, outBase);
            }
        }
    }

    // Q6_K scalar tile: w = d * sc_g * (q - 32) per group, one accumulator advanced in ascending column order exactly
    // as RowDotQ6KScalar advances its own.
    private static void GemmQ6KScalar(byte* bytes, int stride, int rowFrom, int rowTo, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> tile = stackalloc float[GemmTileRows * GemmTileCols];

        fixed (float* pTile = tile)
        {
            for (int rb = rowFrom; rb < rowTo; rb += GemmTileRows)
            {
                int rn = Math.Min(GemmTileRows, rowTo - rb);

                // The scalar route carries each output element's running sum in its own output slot: opened here once
                // per tile group, then advanced by the ascending column walk below and written nowhere else.
                for (int r = 0; r < rn; r++)
                {
                    for (int b = 0; b < count; b++)
                    {
                        outBase[(long)b * rows + rb + r] = 0f;
                    }
                }

                for (int cb = 0; cb < cols; cb += GemmTileCols)
                {
                    int cn = Math.Min(GemmTileCols, cols - cb);

                    for (int r = 0; r < rn; r++)
                    {
                        byte* block = bytes + (long)(rb + r) * stride + (long)(cb / QkK) * Q6KBlockBytes;
                        float* tileRow = pTile + (long)r * GemmTileCols;

                        float d = ReadHalfLE(block + 208);

                        for (int g = 0; g < 16; g++)
                        {
                            int subStart = cb + g * 16;
                            if (subStart >= cols)
                            {
                                break;
                            }

                            int h = g >> 3;
                            int k = g & 7;
                            int quarter = k >> 1;
                            byte* q = block + 64 * h + 16 * (k & 3);
                            byte* qh = block + 128 + 32 * h + 16 * (k & 1);
                            float scale = d * (sbyte)block[192 + g];

                            int limit = Math.Min(16, cols - subStart);

                            for (int l = 0; l < limit; l++)
                            {
                                int six = ((q[l] >> (4 * (quarter >> 1))) & 0xF) | (((qh[l] >> (2 * quarter)) & 3) << 4);
                                tileRow[g * 16 + l] = (six - 32) * scale;
                            }
                        }
                    }

                    for (int r = 0; r < rn; r++)
                    {
                        float* tileRow = pTile + (long)r * GemmTileCols;

                        for (int b = 0; b < count; b++)
                        {
                            float* xRow = inputs + (long)b * cols + cb;
                            float* outSlot = outBase + (long)b * rows + rb + r;
                            float acc = *outSlot;

                            for (int l = 0; l < cn; l++)
                            {
                                acc += xRow[l] * tileRow[l];
                            }

                            *outSlot = acc;
                        }
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------- IQ4_XS

    // IQ4_XS tile: w = d * (ls - 32) * kvalue[q] per sub-block, built by the same table lookup and scale multiply the
    // reference chunk builds.
    private static void GemmIQ4XAvx2(byte* bytes, int stride, int rowFrom, int rowTo, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> tile = stackalloc float[GemmTileRows * GemmTileCols];
        Span<float> acc = stackalloc float[GemmAccLanes];
        Span<float> tail = stackalloc float[GemmAccTails];

        fixed (float* pTile = tile)
        fixed (float* pAcc = acc)
        fixed (float* pTail = tail)
        {
            for (int rb = rowFrom; rb < rowTo; rb += GemmTileRows)
            {
                int rn = Math.Min(GemmTileRows, rowTo - rb);
                acc.Slice(0, rn * MaxBatch * 8).Clear();
                tail.Slice(0, rn * MaxBatch).Clear();

                for (int cb = 0; cb < cols; cb += GemmTileCols)
                {
                    for (int r = 0; r < rn; r++)
                    {
                        byte* block = bytes + (long)(rb + r) * stride + (long)(cb / QkK) * IQ4XsBlockBytes;
                        float* tileRow = pTile + (long)r * GemmTileCols;

                        float dHalf = ReadHalfLE(block);
                        ushort scalesH = ReadScalesHLE(block);

                        for (int ib = 0; ib < 8; ib++)
                        {
                            int subStart = cb + ib * 32;
                            if (subStart >= cols)
                            {
                                break; // zero-padded tail block: no elements left to consume.
                            }

                            float scale = dHalf * (GetIQ4XScale(ib, block + 4, scalesH) - 32);
                            Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);

                            byte* q = block + 8 + 16 * ib; // sub-block window: bytes [q, q+16) always inside the block
                            int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.

                            for (int c = 0; c + 8 <= limit; c += 8)
                            {
                                Avx.Store(tileRow + ib * 32 + c, LoadIQ4Chunk(q, c, scaleV));
                            }

                            for (int k = (limit / 8) * 8; k < limit; k++)
                            {
                                tileRow[ib * 32 + k] = scale * KValueIQ4(((q[k & 15] >> (4 * (k >> 4))) & 0xF));
                            }
                        }
                    }

                    for (int r = 0; r < rn; r++)
                    {
                        float* tileRow = pTile + (long)r * GemmTileCols;

                        for (int b = 0; b < count; b++)
                        {
                            float* chain = pAcc + (r * MaxBatch + b) * 8;
                            float* tailSlot = pTail + r * MaxBatch + b;
                            float* xRow = inputs + (long)b * cols + cb;

                            Vector256<float> accV = Avx.LoadVector256(chain);
                            float tailAcc = *tailSlot;

                            for (int ib = 0; ib < 8; ib++)
                            {
                                int subStart = cb + ib * 32;
                                if (subStart >= cols)
                                {
                                    break;
                                }

                                int limit = Math.Min(32, cols - subStart);

                                for (int c = 0; c + 8 <= limit; c += 8)
                                {
                                    accV = Fma.MultiplyAdd(Avx.LoadVector256(xRow + ib * 32 + c), Avx.LoadVector256(tileRow + ib * 32 + c), accV);
                                }

                                for (int k = (limit / 8) * 8; k < limit; k++)
                                {
                                    tailAcc += xRow[ib * 32 + k] * tileRow[ib * 32 + k];
                                }
                            }

                            Avx.Store(chain, accV);
                            *tailSlot = tailAcc;
                        }
                    }
                }

                FinishGemmTile(pAcc, pTail, rn, count, rows, rb, outBase);
            }
        }
    }

    // IQ4_XS scalar tile: w = d * (ls - 32) * kvalue[q] per sub-block, one accumulator advanced in ascending column
    // order exactly as RowDotIQ4XScalar advances its own.
    private static void GemmIQ4XScalar(byte* bytes, int stride, int rowFrom, int rowTo, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> tile = stackalloc float[GemmTileRows * GemmTileCols];

        fixed (float* pTile = tile)
        {
            for (int rb = rowFrom; rb < rowTo; rb += GemmTileRows)
            {
                int rn = Math.Min(GemmTileRows, rowTo - rb);

                // The scalar route carries each output element's running sum in its own output slot: opened here once
                // per tile group, then advanced by the ascending column walk below and written nowhere else.
                for (int r = 0; r < rn; r++)
                {
                    for (int b = 0; b < count; b++)
                    {
                        outBase[(long)b * rows + rb + r] = 0f;
                    }
                }

                for (int cb = 0; cb < cols; cb += GemmTileCols)
                {
                    int cn = Math.Min(GemmTileCols, cols - cb);

                    for (int r = 0; r < rn; r++)
                    {
                        byte* block = bytes + (long)(rb + r) * stride + (long)(cb / QkK) * IQ4XsBlockBytes;
                        float* tileRow = pTile + (long)r * GemmTileCols;

                        float d = ReadHalfLE(block);
                        ushort scalesH = ReadScalesHLE(block);

                        for (int ib = 0; ib < 8; ib++)
                        {
                            int subStart = cb + ib * 32;
                            if (subStart >= cols)
                            {
                                break;
                            }

                            float scale = d * (GetIQ4XScale(ib, block + 4, scalesH) - 32);
                            byte* q = block + 8 + 16 * ib;
                            int limit = Math.Min(32, cols - subStart);

                            for (int l = 0; l < limit; l++)
                            {
                                tileRow[ib * 32 + l] = scale * KValueIQ4(((q[l & 15] >> (4 * (l >> 4))) & 0xF));
                            }
                        }
                    }

                    for (int r = 0; r < rn; r++)
                    {
                        float* tileRow = pTile + (long)r * GemmTileCols;

                        for (int b = 0; b < count; b++)
                        {
                            float* xRow = inputs + (long)b * cols + cb;
                            float* outSlot = outBase + (long)b * rows + rb + r;
                            float acc = *outSlot;

                            for (int l = 0; l < cn; l++)
                            {
                                acc += xRow[l] * tileRow[l];
                            }

                            *outSlot = acc;
                        }
                    }
                }
            }
        }
    }

    // Tile-group finish: fold each chain's eight lanes with the same reduction the single-vector kernels use, add the
    // chain's scalar tail afterwards, and write outBase[b * rows + row] for every row of the tile group.
    private static void FinishGemmTile(float* pAcc, float* pTail, int tileRows, int count, int rows, int firstRow, float* outBase)
    {
        for (int r = 0; r < tileRows; r++)
        {
            for (int b = 0; b < count; b++)
            {
                outBase[(long)b * rows + firstRow + r] =
                    HSum(Avx.LoadVector256(pAcc + (r * MaxBatch + b) * 8)) + pTail[r * MaxBatch + b];
            }
        }
    }
}
