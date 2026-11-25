using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Lair.Contracts;

namespace Lair.Backends.Cpu;

/// <summary>
/// Batched triple shared-input fusion: three dense weight matrices that consume one activation batch and share its column count are traversed
/// in one dispatch. Each row triple materializes its three packed rows side by side in a per-thread triple scratch with their formats' own
/// bit-exact dequant kernels, then walks the activation batch one time, loading each input vector once and feeding it to all three matrices'
/// accumulator chains — where three independent <see cref="IBackend.GemvBatch"/> calls walk that batch once per matrix instead. Every output
/// element matches what the independent call for that matrix writes, bit-for-bit at any thread count: the route reuses the scratch batch dot's
/// chain semantics exactly (ascending eight-lane chunks, one FMA per chain per chunk with the same operand order, the same <c>HSum</c>
/// reduction), and grouping only chooses which batch vectors share one pass over which rows — never a chain's element order or reduction.
/// Unlike the pair seam, the three matrices may have different row counts (an attention block's Q/K/V trio does): the fused executor walks the
/// union of the row spaces, and a matrix with no row at a position simply contributes nothing there, keeping its own rows' flat
/// <c>[batch * rows + row]</c> layout.
/// </summary>
public sealed unsafe partial class CpuBackend : IDenseWeightTripleFusion
{
    // Per-thread triple scratch: all three rows of one triple staged side by side so the triple dot can read them in the same traversal,
    // grown once to three times this engine's widest fused matrix and reused across every row triple of every call after that.
    [ThreadStatic]
    private static float[]? s_tripleScratch;

    /// <summary>Fusion covers the scratch batch-dot formats only; every other posture returns false and the caller keeps its three calls.</summary>
    public bool TryGemvBatchSharedInput(ReadOnlySpan<float> inputs, int count, WeightRef weightA, Span<float> outputsA, WeightRef weightB, Span<float> outputsB, WeightRef weightC, Span<float> outputsC)
    {
        ValidateFusableMatrix(weightA, nameof(weightA));
        ValidateFusableMatrix(weightB, nameof(weightB));
        ValidateFusableMatrix(weightC, nameof(weightC));

        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, MaxBatch);

        int cols = weightA.Shape.Dims[^1];
        if (weightB.Shape.Dims[^1] != cols || weightC.Shape.Dims[^1] != cols)
        {
            throw new ArgumentException(
                $"Shared-input fusion needs equal column counts; got {cols}, {weightB.Shape.Dims[^1]}, and {weightC.Shape.Dims[^1]}.",
                nameof(weightB));
        }

        QuantFormat formatA = weightA.Format;
        QuantFormat formatB = weightB.Format;
        QuantFormat formatC = weightC.Format;

        // One tuning snapshot for the whole call: eligibility, dispatch threshold, block width, and scratch grouping all read the same resolved posture.
        CpuKernelTuning tuning = _tuning;

        if (!ScratchBatchDotEligible(formatA, cols, count, tuning.ChainGroup, tuning.MaxScratchCols)
            || !ScratchBatchDotEligible(formatB, cols, count, tuning.ChainGroup, tuning.MaxScratchCols)
            || !ScratchBatchDotEligible(formatC, cols, count, tuning.ChainGroup, tuning.MaxScratchCols))
        {
            return false; // formats outside the scratch route (Q8_K i-quants included) fall back to independent GemvBatch calls.
        }

        int rowsA = weightA.Shape.Dims[0];
        int rowsB = weightB.Shape.Dims[0];
        int rowsC = weightC.Shape.Dims[0];

        if (inputs.Length != count * cols
            || (long)outputsA.Length < (long)count * rowsA
            || (long)outputsB.Length < (long)count * rowsB
            || (long)outputsC.Length < (long)count * rowsC)
        {
            throw new ArgumentException(
                $"Input length {inputs.Length} must equal the batch count times the shared column count and every output must cover its own matrix's count x rows.",
                nameof(inputs));
        }

        fixed (byte* bytesA = weightA.Bytes)
        fixed (byte* bytesB = weightB.Bytes)
        fixed (byte* bytesC = weightC.Bytes)
        fixed (float* pIn = inputs)
        fixed (float* outA = outputsA)
        fixed (float* outB = outputsB)
        fixed (float* outC = outputsC)
        {
            int strideA = (int)weightA.Shape.RowStrideBytes(formatA);
            int strideB = (int)weightB.Shape.RowStrideBytes(formatB);
            int strideC = (int)weightC.Shape.RowStrideBytes(formatC);

            // Same per-call participation sample as Gemv/GemvBatch: a quota applied on another thread takes effect from this kernel on.
            int workers = _effectiveThreads;

            // The union row space this dispatch walks: each matrix contributes exactly its own rows, so every matrix's rows are computed
            // exactly once no matter how the row counts differ across the three.
            int maxRows = Math.Max(rowsA, Math.Max(rowsB, rowsC));
            bool parallel = workers > 1 && maxRows >= tuning.ParallelRowThreshold;

            if (!parallel)
            {
                for (int row = 0; row < maxRows; row++)
                {
                    RunFusedTripleRow(bytesA, strideA, rowsA, formatA, bytesB, strideB, rowsB, formatB, bytesC, strideC, rowsC, formatC, row, cols, pIn, count, tuning.ChainGroup, tuning.PairChainGroup, tuning.TripleChainGroup, tuning.MaxScratchCols, outA, outB, outC);
                }

                return true;
            }

            WorkerPool pool = GetOrCreatePool();
            pool.Run((maxRows + tuning.RowBlock - 1) / tuning.RowBlock, new TripleBatchRowJob(
                bytesA, strideA, rowsA, formatA, outA,
                bytesB, strideB, rowsB, formatB, outB,
                bytesC, strideC, rowsC, formatC, outC,
                pIn, count, cols, tuning));

            return true;
        }
    }

    // One fused row triple. The row index selects each matrix's own row exactly as that matrix's own batch loop would, and a matrix with no
    // row at this index contributes nothing. With all three rows present and stageable — the triple scratch ceiling keeps three staged rows
    // cache-resident and the posture resolved a triple group — materialize each matrix's row into its half of the triple scratch with its own
    // bit-exact dequant kernel (the stored values are exactly the chunk vectors that kernel feeds its dot chains), then walk the hot activation
    // batch once through all three matrices' chain sets. Where only two matrices still hold a row the pair route runs for them when the two
    // share a row count (the pair executor walks one shared row range and writes both outputs with that stride), and where all three hold a
    // row but a triple of staged rows no longer fits the ceiling the pair route runs for two of them and the single-row scratch route for the
    // third — the same degraded routes the pair seam uses. Any pair whose two matrices differ in row count, and any lone row, takes the
    // single-row scratch route, the pair seam's own fallback. Every route takes its group width from the call's tuning snapshot and only
    // chooses which batch vectors share one pass — never a chain's order or reduction — so every route lands bit-identical to three
    // independent batch calls.
    private static void RunFusedTripleRow(
        byte* bytesA, int strideA, int rowsA, QuantFormat formatA,
        byte* bytesB, int strideB, int rowsB, QuantFormat formatB,
        byte* bytesC, int strideC, int rowsC, QuantFormat formatC,
        int row, int cols, float* inputs, int count, int chainGroup, int pairGroup, int tripleGroup, int maxScratchCols,
        float* outA, float* outB, float* outC)
    {
        bool hasA = row < rowsA;
        bool hasB = row < rowsB;
        bool hasC = row < rowsC;

        if (hasA && hasB && hasC)
        {
            if (tripleGroup >= CpuKernelTuning.ChainGroupMin && 3L * cols <= maxScratchCols)
            {
                RunFusedTripleRowTriple(bytesA + (long)row * strideA, bytesB + (long)row * strideB, bytesC + (long)row * strideC, formatA, formatB, formatC, cols, inputs, count, rowsA, rowsB, rowsC, tripleGroup, outA + row, outB + row, outC + row);
                return;
            }

            // Three staged rows no longer stay cache-resident beside their packed bytes: keep the fused call on the pair route for A and B
            // and the single-row scratch route for C — each bit-identical to its own matrix's independent batch call.
            RunFusedTriplePairOrSingles(bytesA + (long)row * strideA, strideA, rowsA, formatA, bytesB + (long)row * strideB, strideB, rowsB, formatB, cols, inputs, count, chainGroup, pairGroup, maxScratchCols, outA + row, outB + row);
            ScratchRowDot(bytesC + (long)row * strideC, formatC, cols, inputs, count, rowsC, chainGroup, maxScratchCols, outC + row);
            return;
        }

        if (hasA && hasB)
        {
            RunFusedTriplePairOrSingles(bytesA + (long)row * strideA, strideA, rowsA, formatA, bytesB + (long)row * strideB, strideB, rowsB, formatB, cols, inputs, count, chainGroup, pairGroup, maxScratchCols, outA + row, outB + row);
            return;
        }

        if (hasA && hasC)
        {
            RunFusedTriplePairOrSingles(bytesA + (long)row * strideA, strideA, rowsA, formatA, bytesC + (long)row * strideC, strideC, rowsC, formatC, cols, inputs, count, chainGroup, pairGroup, maxScratchCols, outA + row, outC + row);
            return;
        }

        if (hasB && hasC)
        {
            RunFusedTriplePairOrSingles(bytesB + (long)row * strideB, strideB, rowsB, formatB, bytesC + (long)row * strideC, strideC, rowsC, formatC, cols, inputs, count, chainGroup, pairGroup, maxScratchCols, outB + row, outC + row);
            return;
        }

        if (hasA)
        {
            ScratchRowDot(bytesA + (long)row * strideA, formatA, cols, inputs, count, rowsA, chainGroup, maxScratchCols, outA + row);
            return;
        }

        if (hasB)
        {
            ScratchRowDot(bytesB + (long)row * strideB, formatB, cols, inputs, count, rowsB, chainGroup, maxScratchCols, outB + row);
            return;
        }

        ScratchRowDot(bytesC + (long)row * strideC, formatC, cols, inputs, count, rowsC, chainGroup, maxScratchCols, outC + row);
    }

    // Two present matrices of a triple: the pair route when they share a row count (the pair executor walks one shared row range and writes
    // both outputs with that stride), otherwise two back-to-back single-row scratch passes — the pair seam's own fallback shape, bit-identical
    // to each matrix's independent batch call either way.
    private static void RunFusedTriplePairOrSingles(
        byte* rowX, int strideX, int rowsX, QuantFormat formatX,
        byte* rowY, int strideY, int rowsY, QuantFormat formatY,
        int cols, float* inputs, int count, int chainGroup, int pairGroup, int maxScratchCols,
        float* outX, float* outY)
    {
        if (rowsX == rowsY)
        {
            RunFusedPairRow(rowX, rowY, formatX, formatY, cols, inputs, count, rowsX, chainGroup, pairGroup, maxScratchCols, outX, outY);
            return;
        }

        ScratchRowDot(rowX, formatX, cols, inputs, count, rowsX, chainGroup, maxScratchCols, outX);
        ScratchRowDot(rowY, formatY, cols, inputs, count, rowsY, chainGroup, maxScratchCols, outY);
    }

    // The triple route for one row triple: all three rows materialized side by side in the triple scratch, then one shared traversal of the
    // activation batch updating all three matrices' accumulator chains.
    private static void RunFusedTripleRowTriple(byte* rowA, byte* rowB, byte* rowC, QuantFormat formatA, QuantFormat formatB, QuantFormat formatC, int cols, float* inputs, int count, int rowsA, int rowsB, int rowsC, int tripleGroup, float* outBaseA, float* outBaseB, float* outBaseC)
    {
        float[]? scratch = s_tripleScratch;
        if (scratch is null || scratch.Length < 3L * cols)
        {
            scratch = new float[3 * cols];
            s_tripleScratch = scratch;
        }

        fixed (float* pScratch = scratch)
        {
            MaterializeRow(rowA, formatA, pScratch, cols);
            MaterializeRow(rowB, formatB, pScratch + cols, cols);
            MaterializeRow(rowC, formatC, pScratch + 2L * cols, cols);

            TripleBatchDotFromScratch(pScratch, pScratch + cols, pScratch + 2L * cols, cols, inputs, count, rowsA, rowsB, rowsC, outBaseA, outBaseB, outBaseC, tripleGroup);
        }
    }

    // Register-blocked triple dot over one triple's three materialized rows: `tripleGroup` accumulator chains per matrix keep their
    // Vector256 lanes in registers across the whole triple, and each chain advances against its own matrix's chunk while the group's
    // input vector is loaded once and fed to all three matrices' FMAs — the activation batch is walked one time per row triple instead of
    // once per matrix. Chunk positions are 8-aligned exactly like the single-row scratch kernel's group windows (eligibility guarantees
    // aligned columns), so every lane holds the same elements in the same order and each output lands bit-identical to that kernel's chain,
    // down to the HSum fold an empty scalar tail adds — at any resolved triple group width, because grouping only chooses which batch vectors
    // share one pass over the triple and never reorders a chain or its reduction. Each matrix writes through its own row stride, so the
    // matrices' differing row counts never touch each other's output layout.
    private static void TripleBatchDotFromScratch(float* wA, float* wB, float* wC, int cols, float* inputs, int count, int rowsA, int rowsB, int rowsC, float* outA, float* outB, float* outC, int tripleGroup)
    {
        if (tripleGroup != CpuKernelTuning.TripleChainGroupMax)
        {
            TripleScratchGroupsOverStackChains(wA, wB, wC, cols, inputs, count, rowsA, rowsB, rowsC, outA, outB, outC, tripleGroup); // a pinned narrower triple group: the same chains in the same order, staged through stack lanes instead of live YMM registers.
            return;
        }

        int b = 0;

        for (; b + tripleGroup <= count; b += tripleGroup)
        {
            Vector256<float> a0 = Vector256<float>.Zero;
            Vector256<float> a1 = Vector256<float>.Zero;
            Vector256<float> a2 = Vector256<float>.Zero;
            Vector256<float> b0 = Vector256<float>.Zero;
            Vector256<float> b1 = Vector256<float>.Zero;
            Vector256<float> b2 = Vector256<float>.Zero;
            Vector256<float> c0 = Vector256<float>.Zero;
            Vector256<float> c1 = Vector256<float>.Zero;
            Vector256<float> c2 = Vector256<float>.Zero;

            float* in0 = inputs + (long)b * cols;
            for (int i = 0; i < cols; i += 8)
            {
                Vector256<float> wvA = Avx.LoadVector256(wA + i);
                Vector256<float> wvB = Avx.LoadVector256(wB + i);
                Vector256<float> wvC = Avx.LoadVector256(wC + i);

                Vector256<float> x0 = Avx.LoadVector256(in0); // one input vector load feeds all three matrices' chains.
                a0 = Fma.MultiplyAdd(x0, wvA, a0);
                b0 = Fma.MultiplyAdd(x0, wvB, b0);
                c0 = Fma.MultiplyAdd(x0, wvC, c0);
                Vector256<float> x1 = Avx.LoadVector256(in0 + cols);
                a1 = Fma.MultiplyAdd(x1, wvA, a1);
                b1 = Fma.MultiplyAdd(x1, wvB, b1);
                c1 = Fma.MultiplyAdd(x1, wvC, c1);
                Vector256<float> x2 = Avx.LoadVector256(in0 + 2L * cols);
                a2 = Fma.MultiplyAdd(x2, wvA, a2);
                b2 = Fma.MultiplyAdd(x2, wvB, b2);
                c2 = Fma.MultiplyAdd(x2, wvC, c2);

                in0 += 8;
            }

            outA[(long)b * rowsA] = HSum(a0);
            outA[(long)(b + 1) * rowsA] = HSum(a1);
            outA[(long)(b + 2) * rowsA] = HSum(a2);
            outB[(long)b * rowsB] = HSum(b0);
            outB[(long)(b + 1) * rowsB] = HSum(b1);
            outB[(long)(b + 2) * rowsB] = HSum(b2);
            outC[(long)b * rowsC] = HSum(c0);
            outC[(long)(b + 1) * rowsC] = HSum(c1);
            outC[(long)(b + 2) * rowsC] = HSum(c2);
        }

        // Ragged triple tail (count not a multiple of the triple group): one accumulator per remaining chain for each matrix, walking the
        // triple's rows in the same pass with the same chunk order and reduction as the full groups.
        for (; b < count; b++)
        {
            Vector256<float> accA = Vector256<float>.Zero;
            Vector256<float> accB = Vector256<float>.Zero;
            Vector256<float> accC = Vector256<float>.Zero;
            float* inB = inputs + (long)b * cols;
            for (int i = 0; i < cols; i += 8)
            {
                Vector256<float> xv = Avx.LoadVector256(inB + i);
                accA = Fma.MultiplyAdd(xv, Avx.LoadVector256(wA + i), accA);
                accB = Fma.MultiplyAdd(xv, Avx.LoadVector256(wB + i), accB);
                accC = Fma.MultiplyAdd(xv, Avx.LoadVector256(wC + i), accC);
            }

            outA[(long)b * rowsA] = HSum(accA);
            outB[(long)b * rowsB] = HSum(accB);
            outC[(long)b * rowsC] = HSum(accC);
        }
    }

    // A pinned triple group below the triple register budget: the same ascending per-chain walks for all three matrices, with each chain's
    // eight-lane accumulator staged in stack scratch (one group's worth per matrix, reused for every group and row triple) instead of live
    // YMM registers. Outputs are bit-identical to the register route and to three independent single-row passes because grouping never changes
    // a chain's element order or its reduction — only which batch vectors share one pass over the triple.
    private static void TripleScratchGroupsOverStackChains(float* wA, float* wB, float* wC, int cols, float* inputs, int count, int rowsA, int rowsB, int rowsC, float* outA, float* outB, float* outC, int tripleGroup)
    {
        Span<float> accA = stackalloc float[tripleGroup * 8];
        Span<float> accB = stackalloc float[tripleGroup * 8];
        Span<float> accC = stackalloc float[tripleGroup * 8];
        fixed (float* pA = accA)
        fixed (float* pB = accB)
        fixed (float* pC = accC)
        {
            int b0 = 0;
            for (; b0 + tripleGroup <= count; b0 += tripleGroup)
            {
                for (int lane = 0; lane < tripleGroup * 8; lane++)
                {
                    pA[lane] = 0f;
                    pB[lane] = 0f;
                    pC[lane] = 0f;
                }

                for (int i = 0; i < cols; i += 8)
                {
                    Vector256<float> wvA = Avx.LoadVector256(wA + i); // one hot-scratch read per matrix shared by the group's chains, as in the register route.
                    Vector256<float> wvB = Avx.LoadVector256(wB + i);
                    Vector256<float> wvC = Avx.LoadVector256(wC + i);
                    for (int j = 0; j < tripleGroup; j++)
                    {
                        Vector256<float> xv = Avx.LoadVector256(inputs + (long)(b0 + j) * cols + i); // one input vector load feeds all three matrices' chains.
                        float* chainA = pA + j * 8;
                        Avx.Store(chainA, Fma.MultiplyAdd(xv, wvA, Avx.LoadVector256(chainA)));
                        float* chainB = pB + j * 8;
                        Avx.Store(chainB, Fma.MultiplyAdd(xv, wvB, Avx.LoadVector256(chainB)));
                        float* chainC = pC + j * 8;
                        Avx.Store(chainC, Fma.MultiplyAdd(xv, wvC, Avx.LoadVector256(chainC)));
                    }
                }

                for (int j = 0; j < tripleGroup; j++)
                {
                    outA[(long)(b0 + j) * rowsA] = HSum(Avx.LoadVector256(pA + j * 8));
                    outB[(long)(b0 + j) * rowsB] = HSum(Avx.LoadVector256(pB + j * 8));
                    outC[(long)(b0 + j) * rowsC] = HSum(Avx.LoadVector256(pC + j * 8));
                }
            }

            // Ragged tail below one triple group: one accumulator per matrix walking its row exactly as each full-group chain does.
            for (; b0 < count; b0++)
            {
                Vector256<float> chainA = Vector256<float>.Zero;
                Vector256<float> chainB = Vector256<float>.Zero;
                Vector256<float> chainC = Vector256<float>.Zero;
                float* inB = inputs + (long)b0 * cols;
                for (int i = 0; i < cols; i += 8)
                {
                    Vector256<float> xv = Avx.LoadVector256(inB + i);
                    chainA = Fma.MultiplyAdd(xv, Avx.LoadVector256(wA + i), chainA);
                    chainB = Fma.MultiplyAdd(xv, Avx.LoadVector256(wB + i), chainB);
                    chainC = Fma.MultiplyAdd(xv, Avx.LoadVector256(wC + i), chainC);
                }

                outA[(long)b0 * rowsA] = HSum(chainA);
                outB[(long)b0 * rowsB] = HSum(chainB);
                outC[(long)b0 * rowsC] = HSum(chainC);
            }
        }
    }

    // Row-block executor for one fused triple call: each block walks the same slice of the union row space through whichever matrices still
    // hold a row there, mirroring how three independent GemvBatch calls each walk that matrix's own rows. Rows are independent, so a different
    // worker assignment than an independent trio of calls still lands bit-identical outputs.
    private sealed unsafe class TripleBatchRowJob : IGemvJob
    {
        private readonly byte* _bytesA;
        private readonly int _strideA;
        private readonly int _rowsA;
        private readonly QuantFormat _formatA;
        private readonly float* _outA;
        private readonly byte* _bytesB;
        private readonly int _strideB;
        private readonly int _rowsB;
        private readonly QuantFormat _formatB;
        private readonly float* _outB;
        private readonly byte* _bytesC;
        private readonly int _strideC;
        private readonly int _rowsC;
        private readonly QuantFormat _formatC;
        private readonly float* _outC;
        private readonly float* _inputs;
        private readonly int _count;
        private readonly int _cols;
        private readonly int _rowBlock; // captured from the dispatch snapshot so every claimed block uses the same width as Run's block count
        private readonly int _chainGroup;
        private readonly int _pairGroup;
        private readonly int _tripleGroup;
        private readonly int _maxScratchCols;

        public TripleBatchRowJob(
            byte* bytesA, int strideA, int rowsA, QuantFormat formatA, float* outA,
            byte* bytesB, int strideB, int rowsB, QuantFormat formatB, float* outB,
            byte* bytesC, int strideC, int rowsC, QuantFormat formatC, float* outC,
            float* inputs, int count, int cols, CpuKernelTuning tuning)
        {
            _bytesA = bytesA;
            _strideA = strideA;
            _rowsA = rowsA;
            _formatA = formatA;
            _outA = outA;
            _bytesB = bytesB;
            _strideB = strideB;
            _rowsB = rowsB;
            _formatB = formatB;
            _outB = outB;
            _bytesC = bytesC;
            _strideC = strideC;
            _rowsC = rowsC;
            _formatC = formatC;
            _outC = outC;
            _inputs = inputs;
            _count = count;
            _cols = cols;
            _rowBlock = tuning.RowBlock;
            _chainGroup = tuning.ChainGroup;
            _pairGroup = tuning.PairChainGroup;
            _tripleGroup = tuning.TripleChainGroup;
            _maxScratchCols = tuning.MaxScratchCols;
        }

        public void ProcessBlock(int block)
        {
            int firstRow = block * _rowBlock;
            int maxRows = Math.Max(_rowsA, Math.Max(_rowsB, _rowsC));
            int lastRow = Math.Min(firstRow + _rowBlock, maxRows);
            for (int row = firstRow; row < lastRow; row++)
            {
                RunFusedTripleRow(
                    _bytesA, _strideA, _rowsA, _formatA,
                    _bytesB, _strideB, _rowsB, _formatB,
                    _bytesC, _strideC, _rowsC, _formatC,
                    row, _cols, _inputs, _count, _chainGroup, _pairGroup, _tripleGroup, _maxScratchCols,
                    _outA, _outB, _outC);
            }
        }
    }
}
