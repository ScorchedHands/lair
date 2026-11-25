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
        format is QuantFormat.Q8_0 or QuantFormat.Q4_K or QuantFormat.IQ4_XS;

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
                $"{QuantFormat.Q4_K}, and {QuantFormat.IQ4_XS} execute through the dense-matrix weight path.",
                nameof(w));
        }

        int stride = (int)strideBytes;
        QuantFormat format = w.Format;

        // Same raw-pointer job contract as GemvBatch: pinned spans stay stable for the duration of the call and every
        // claimed block finishes before WorkerPool.Run returns. A scheduling block owns whole weight rows and sweeps
        // every column for them, so each output element is produced by one uninterrupted ascending walk of its own
        // accumulator chain: partitioning rows across workers changes nothing about the bytes it writes.
        fixed (byte* pBytes = w.Bytes)
        fixed (float* pIn = inputs)
        fixed (float* pOut = outputs)
        {
            int workers = _effectiveThreads;

            // One tuning snapshot for the whole call: threshold, block count, and claimed-block width all read the same resolved posture.
            CpuKernelTuning tuning = _tuning;
            bool parallel = workers > 1 && rows >= tuning.ParallelRowThreshold;

            if (!parallel)
            {
                RunGemmRows(pBytes, pIn, pOut, stride, rows, cols, count, format, _useAvx2, 0, rows);
                return;
            }

            WorkerPool pool = GetOrCreatePool();
            pool.Run((rows + tuning.RowBlock - 1) / tuning.RowBlock, new GemmRowJob(pBytes, pIn, pOut, stride, rows, cols, count, format, _useAvx2, tuning));
        }
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
    // its scalar tail.
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
