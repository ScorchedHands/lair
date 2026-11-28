using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Lair.Contracts;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend : IDenseWeightPairFusion
{
    // Formats whose batch dot runs through a materialized row scratch: the dequant kernel stores exactly the chunk values that
    // kernel's dot chains consume, so a float staging buffer reproduces them bit-for-bit. Q8_K-quantized activation formats (the
    // sub-4-bit i-quants) stay on their integer-dot staging route and are excluded from scratch batching and fusion alike; every
    // other format keeps its existing streaming batch kernel unchanged. The scratch posture itself (chain grouping, width ceiling)
    // resolves per backend — see CpuKernelTuning — never fixed to one machine class here.
    private static readonly QuantFormat[] ScratchBatchDotFormats = [QuantFormat.Q4_K, QuantFormat.Q6_K];

    // Per-thread row scratch, grown once to this engine's widest fused matrix and reused across every row of every call after that.
    [ThreadStatic]
    private static float[]? s_rowScratch;

    // Per-thread pair scratch: both rows of one pair staged side by side so the pair dot can read them in the same traversal,
    // grown once to twice this engine's widest fused pair and reused across every row pair of every call after that.
    [ThreadStatic]
    private static float[]? s_pairScratch;

    // Scratch-route eligibility against the call's resolved tuning snapshot: a full chain group of batch vectors before materializing
    // (below one group the streaming kernels' in-register chunk feeds too few accumulator chains to pay for a row staging pass, so
    // small merged-decode batches keep their streaming route), the width ceiling that keeps a staged row cache-resident, and
    // block-aligned columns only: with a partial final group the streaming kernels split that group's ragged elements into scalar
    // chain adds at positions the scratch dot would fold into vector lanes, so unaligned widths keep the streaming route.
    private static bool ScratchBatchDotEligible(QuantFormat format, int cols, int count, int chainGroup, int maxScratchCols) =>
        Array.IndexOf(ScratchBatchDotFormats, format) >= 0
        && cols <= maxScratchCols
        && count >= chainGroup
        // Block-aligned columns only: with a partial final group the streaming kernels split that group's ragged elements into
        // scalar chain adds at positions the scratch dot would fold into vector lanes, so unaligned widths keep the streaming route.
        && (format != QuantFormat.Q4_K || cols % 32 == 0)
        && (format != QuantFormat.Q6_K || cols % 16 == 0);

    // Register-blocked batch dot over a materialized row: `chainGroup` accumulator chains per group keep their Vector256 lanes in
    // registers across the whole row instead of load/store round-trips through stack scratch, and each chain advances its own
    // input vector as one FMA against the weight chunk read from L1-hot scratch. Chunk positions are 8-aligned exactly like the
    // streaming kernels' group windows (eligibility guarantees aligned columns), so every lane holds the same elements in the same
    // order and each output lands bit-identical to the streaming route's chain, down to the HSum fold an empty scalar tail adds —
    // at any resolved group width, because grouping only chooses which batch vectors share one pass over the row and never reorders a chain.
    private static void BatchDotFromScratch(float* w, int cols, float* inputs, int count, int rows, float* outBase, int chainGroup)
    {
        if (chainGroup != CpuKernelTuning.ChainGroupMax)
        {
            ScratchGroupsOverStackChains(w, cols, inputs, count, rows, outBase, chainGroup); // a pinned narrower group: the same chains in the same order, staged through stack lanes instead of live YMM registers.
            return;
        }

        int b = 0;

        for (; b + chainGroup <= count; b += chainGroup)
        {
            Vector256<float> a0 = Vector256<float>.Zero;
            Vector256<float> a1 = Vector256<float>.Zero;
            Vector256<float> a2 = Vector256<float>.Zero;
            Vector256<float> a3 = Vector256<float>.Zero;
            Vector256<float> a4 = Vector256<float>.Zero;
            Vector256<float> a5 = Vector256<float>.Zero;
            Vector256<float> a6 = Vector256<float>.Zero;
            Vector256<float> a7 = Vector256<float>.Zero;

            float* in0 = inputs + (long)b * cols;
            for (int i = 0; i < cols; i += 8)
            {
                Vector256<float> wv = Avx.LoadVector256(w + i);
                a0 = Fma.MultiplyAdd(Avx.LoadVector256(in0), wv, a0);
                a1 = Fma.MultiplyAdd(Avx.LoadVector256(in0 + cols), wv, a1);
                a2 = Fma.MultiplyAdd(Avx.LoadVector256(in0 + 2L * cols), wv, a2);
                a3 = Fma.MultiplyAdd(Avx.LoadVector256(in0 + 3L * cols), wv, a3);
                a4 = Fma.MultiplyAdd(Avx.LoadVector256(in0 + 4L * cols), wv, a4);
                a5 = Fma.MultiplyAdd(Avx.LoadVector256(in0 + 5L * cols), wv, a5);
                a6 = Fma.MultiplyAdd(Avx.LoadVector256(in0 + 6L * cols), wv, a6);
                a7 = Fma.MultiplyAdd(Avx.LoadVector256(in0 + 7L * cols), wv, a7);
                in0 += 8;
            }

            outBase[(long)b * rows] = HSum(a0);
            outBase[(long)(b + 1) * rows] = HSum(a1);
            outBase[(long)(b + 2) * rows] = HSum(a2);
            outBase[(long)(b + 3) * rows] = HSum(a3);
            outBase[(long)(b + 4) * rows] = HSum(a4);
            outBase[(long)(b + 5) * rows] = HSum(a5);
            outBase[(long)(b + 6) * rows] = HSum(a6);
            outBase[(long)(b + 7) * rows] = HSum(a7);
        }

        // Ragged chain tail (count not a multiple of the group): one register accumulator per remaining chain, same chunk order and
        // reduction as the full groups. Prefill row widths are multiples of the group; merged decode batches under a full group run here.
        for (; b < count; b++)
        {
            Vector256<float> acc = Vector256<float>.Zero;
            float* inB = inputs + (long)b * cols;
            for (int i = 0; i < cols; i += 8)
            {
                acc = Fma.MultiplyAdd(Avx.LoadVector256(inB + i), Avx.LoadVector256(w + i), acc);
            }

            outBase[(long)b * rows] = HSum(acc);
        }
    }

    // A pinned group width below the register budget: the same ascending per-chain walks with each chain's eight-lane accumulator staged in
    // stack scratch (one group's worth, reused for every group and row) instead of live YMM registers. Outputs are bit-identical to the
    // register route because grouping never changes a chain's element order or its reduction — only which batch vectors share one pass.
    private static void ScratchGroupsOverStackChains(float* w, int cols, float* inputs, int count, int rows, float* outBase, int chainGroup)
    {
        Span<float> acc = stackalloc float[chainGroup * 8];
        fixed (float* pAcc = acc)
        {
            int b0 = 0;
            for (; b0 + chainGroup <= count; b0 += chainGroup)
            {
                for (int lane = 0; lane < chainGroup * 8; lane++)
                {
                    pAcc[lane] = 0f;
                }

                for (int i = 0; i < cols; i += 8)
                {
                    Vector256<float> wv = Avx.LoadVector256(w + i); // one hot-scratch weight read shared by the group's chains, as in the register route.
                    for (int j = 0; j < chainGroup; j++)
                    {
                        float* chain = pAcc + j * 8;
                        Avx.Store(chain, Fma.MultiplyAdd(Avx.LoadVector256(inputs + (long)(b0 + j) * cols + i), wv, Avx.LoadVector256(chain)));
                    }
                }

                for (int j = 0; j < chainGroup; j++)
                {
                    outBase[(long)(b0 + j) * rows] = HSum(Avx.LoadVector256(pAcc + j * 8));
                }
            }

            // Ragged tail below one group: a single accumulator walking the row exactly as each full-group chain does.
            for (; b0 < count; b0++)
            {
                Vector256<float> chain = Vector256<float>.Zero;
                float* inB = inputs + (long)b0 * cols;
                for (int i = 0; i < cols; i += 8)
                {
                    chain = Fma.MultiplyAdd(Avx.LoadVector256(inB + i), Avx.LoadVector256(w + i), chain);
                }

                outBase[(long)b0 * rows] = HSum(chain);
            }
        }
    }

    // Register-blocked pair dot over one pair's two materialized rows: `pairGroup` accumulator chains per matrix keep their
    // Vector256 lanes in registers across the whole pair, and each chain advances against its own matrix's chunk while the group's
    // input vector is loaded once and fed to both matrices' FMAs — the activation batch is walked one time per row pair instead of
    // once per matrix. Chunk positions are 8-aligned exactly like the single-row scratch kernel's group windows (eligibility
    // guarantees aligned columns), so every lane holds the same elements in the same order and each output lands bit-identical to
    // that kernel's chain, down to the HSum fold an empty scalar tail adds — at any resolved pair group width, because grouping
    // only chooses which batch vectors share one pass over the pair and never reorders a chain or its reduction.
    private static void PairBatchDotFromScratch(float* wA, float* wB, int cols, float* inputs, int count, int rows, float* outA, float* outB, int pairGroup)
    {
        if (pairGroup != CpuKernelTuning.PairChainGroupMax)
        {
            PairScratchGroupsOverStackChains(wA, wB, cols, inputs, count, rows, outA, outB, pairGroup); // a pinned narrower pair group: the same chains in the same order, staged through stack lanes instead of live YMM registers.
            return;
        }

        int b = 0;

        for (; b + pairGroup <= count; b += pairGroup)
        {
            Vector256<float> a0 = Vector256<float>.Zero;
            Vector256<float> a1 = Vector256<float>.Zero;
            Vector256<float> a2 = Vector256<float>.Zero;
            Vector256<float> a3 = Vector256<float>.Zero;
            Vector256<float> b0 = Vector256<float>.Zero;
            Vector256<float> b1 = Vector256<float>.Zero;
            Vector256<float> b2 = Vector256<float>.Zero;
            Vector256<float> b3 = Vector256<float>.Zero;

            float* in0 = inputs + (long)b * cols;
            for (int i = 0; i < cols; i += 8)
            {
                Vector256<float> wvA = Avx.LoadVector256(wA + i);
                Vector256<float> wvB = Avx.LoadVector256(wB + i);

                Vector256<float> x0 = Avx.LoadVector256(in0); // one input vector load feeds both matrices' chains.
                a0 = Fma.MultiplyAdd(x0, wvA, a0);
                b0 = Fma.MultiplyAdd(x0, wvB, b0);
                Vector256<float> x1 = Avx.LoadVector256(in0 + cols);
                a1 = Fma.MultiplyAdd(x1, wvA, a1);
                b1 = Fma.MultiplyAdd(x1, wvB, b1);
                Vector256<float> x2 = Avx.LoadVector256(in0 + 2L * cols);
                a2 = Fma.MultiplyAdd(x2, wvA, a2);
                b2 = Fma.MultiplyAdd(x2, wvB, b2);
                Vector256<float> x3 = Avx.LoadVector256(in0 + 3L * cols);
                a3 = Fma.MultiplyAdd(x3, wvA, a3);
                b3 = Fma.MultiplyAdd(x3, wvB, b3);

                in0 += 8;
            }

            outA[(long)b * rows] = HSum(a0);
            outA[(long)(b + 1) * rows] = HSum(a1);
            outA[(long)(b + 2) * rows] = HSum(a2);
            outA[(long)(b + 3) * rows] = HSum(a3);
            outB[(long)b * rows] = HSum(b0);
            outB[(long)(b + 1) * rows] = HSum(b1);
            outB[(long)(b + 2) * rows] = HSum(b2);
            outB[(long)(b + 3) * rows] = HSum(b3);
        }

        // Ragged pair tail (count not a multiple of the pair group): one accumulator per remaining chain for each matrix, walking
        // the pair's rows in the same pass with the same chunk order and reduction as the full groups.
        for (; b < count; b++)
        {
            Vector256<float> accA = Vector256<float>.Zero;
            Vector256<float> accB = Vector256<float>.Zero;
            float* inB = inputs + (long)b * cols;
            for (int i = 0; i < cols; i += 8)
            {
                Vector256<float> xv = Avx.LoadVector256(inB + i);
                accA = Fma.MultiplyAdd(xv, Avx.LoadVector256(wA + i), accA);
                accB = Fma.MultiplyAdd(xv, Avx.LoadVector256(wB + i), accB);
            }

            outA[(long)b * rows] = HSum(accA);
            outB[(long)b * rows] = HSum(accB);
        }
    }

    // A pinned pair group below the pair register budget: the same ascending per-chain walks for both matrices, with each chain's
    // eight-lane accumulator staged in stack scratch (one group's worth per matrix, reused for every group and row pair) instead of
    // live YMM registers. Outputs are bit-identical to the register route and to two independent single-row passes because grouping
    // never changes a chain's element order or its reduction — only which batch vectors share one pass over the pair.
    private static void PairScratchGroupsOverStackChains(float* wA, float* wB, int cols, float* inputs, int count, int rows, float* outA, float* outB, int pairGroup)
    {
        Span<float> accA = stackalloc float[pairGroup * 8];
        Span<float> accB = stackalloc float[pairGroup * 8];
        fixed (float* pA = accA)
        fixed (float* pB = accB)
        {
            int b0 = 0;
            for (; b0 + pairGroup <= count; b0 += pairGroup)
            {
                for (int lane = 0; lane < pairGroup * 8; lane++)
                {
                    pA[lane] = 0f;
                    pB[lane] = 0f;
                }

                for (int i = 0; i < cols; i += 8)
                {
                    Vector256<float> wvA = Avx.LoadVector256(wA + i); // one hot-scratch read per matrix shared by the group's chains, as in the register route.
                    Vector256<float> wvB = Avx.LoadVector256(wB + i);
                    for (int j = 0; j < pairGroup; j++)
                    {
                        Vector256<float> xv = Avx.LoadVector256(inputs + (long)(b0 + j) * cols + i); // one input vector load feeds both matrices' chains.
                        float* chainA = pA + j * 8;
                        Avx.Store(chainA, Fma.MultiplyAdd(xv, wvA, Avx.LoadVector256(chainA)));
                        float* chainB = pB + j * 8;
                        Avx.Store(chainB, Fma.MultiplyAdd(xv, wvB, Avx.LoadVector256(chainB)));
                    }
                }

                for (int j = 0; j < pairGroup; j++)
                {
                    outA[(long)(b0 + j) * rows] = HSum(Avx.LoadVector256(pA + j * 8));
                    outB[(long)(b0 + j) * rows] = HSum(Avx.LoadVector256(pB + j * 8));
                }
            }

            // Ragged tail below one pair group: one accumulator per matrix walking its row exactly as each full-group chain does.
            for (; b0 < count; b0++)
            {
                Vector256<float> chainA = Vector256<float>.Zero;
                Vector256<float> chainB = Vector256<float>.Zero;
                float* inB = inputs + (long)b0 * cols;
                for (int i = 0; i < cols; i += 8)
                {
                    Vector256<float> xv = Avx.LoadVector256(inB + i);
                    chainA = Fma.MultiplyAdd(xv, Avx.LoadVector256(wA + i), chainA);
                    chainB = Fma.MultiplyAdd(xv, Avx.LoadVector256(wB + i), chainB);
                }

                outA[(long)b0 * rows] = HSum(chainA);
                outB[(long)b0 * rows] = HSum(chainB);
            }
        }
    }

    // Materialize one packed row into the thread's scratch with its format's own dequant kernel, then run the blocked dot over it.
    // Returns false when the format/column posture is outside the scratch route so callers keep their streaming kernels. The grouping
    // and width ceiling arrive as plain locals from the call's resolved tuning snapshot — no allocation on this route.
    private static bool ScratchRowDot(byte* rowBytes, QuantFormat format, int cols, float* inputs, int count, int rows, int chainGroup, int maxScratchCols, float* outBase)
    {
        if (!ScratchBatchDotEligible(format, cols, count, chainGroup, maxScratchCols))
        {
            return false;
        }

        float[]? scratch = s_rowScratch;
        if (scratch is null || scratch.Length < cols)
        {
            scratch = new float[cols];
            s_rowScratch = scratch;
        }

        fixed (float* pScratch = scratch)
        {
            switch (format)
            {
                case QuantFormat.Q4_K:
                    DequantQ4KAvx2(rowBytes, pScratch, cols);
                    break;
                default: // Q6_K — the only other member of ScratchBatchDotFormats.
                    DequantQ6KAvx2(rowBytes, pScratch, cols);
                    break;
            }

            BatchDotFromScratch(pScratch, cols, inputs, count, rows, outBase, chainGroup);
        }

        return true;
    }

    /// <summary>Fusion covers the scratch batch-dot formats only; every other posture returns false and the caller keeps its calls. A fused call that takes fusion may still choose its executor: with a stated pair crossover width (<see cref="CpuKernelTuning.PairMatmulRouteBatchWidth"/>, 0 at every resolved posture) each matrix's rows run through the blocked matmul tile executor instead of this file's staged-pair traversal, byte-identical either way.</summary>
    public bool TryGemvBatchSharedInput(ReadOnlySpan<float> inputs, int count, WeightRef weightA, Span<float> outputsA, WeightRef weightB, Span<float> outputsB)
    {
        ValidateFusableMatrix(weightA, nameof(weightA));
        ValidateFusableMatrix(weightB, nameof(weightB));

        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, MaxBatch);

        int cols = weightA.Shape.Dims[^1];
        if (weightB.Shape.Dims[^1] != cols)
        {
            throw new ArgumentException(
                $"Shared-input fusion needs equal column counts; got {cols} and {weightB.Shape.Dims[^1]}.",
                nameof(weightB));
        }

        QuantFormat formatA = weightA.Format;
        QuantFormat formatB = weightB.Format;

        // One tuning snapshot for the whole call: eligibility, dispatch threshold, block width, and scratch grouping all read the same resolved posture.
        CpuKernelTuning tuning = _tuning;

        // The hybrid dense-matrix route is a statement about single-matrix batched passes only (see GemvBatch): it chooses which executor walks
        // one matrix's rows when no shared-input fusion opportunity exists. This route already shares one activation walk across both matrices'
        // accumulator chains, so a stated crossover width for either format says nothing here — the pair stays on its own measured posture and
        // every scratch-eligible pair fuses exactly as it does before that seam is stated at all. Formats outside the scratch route decline for
        // their own reason below; neither this fusion decision nor its bit-parity contract depends on a matmul-route width.
        if (!ScratchBatchDotEligible(formatA, cols, count, tuning.ChainGroup, tuning.MaxScratchCols) || !ScratchBatchDotEligible(formatB, cols, count, tuning.ChainGroup, tuning.MaxScratchCols))
        {
            return false; // formats outside the scratch route (Q8_K i-quants included) fall back to independent GemvBatch calls.
        }

        int rows = weightA.Shape.Dims[0];
        if (weightB.Shape.Dims[0] != rows)
        {
            return false; // this fused executor walks one shared row range, so unequal-row pairs fall back to independent calls.
        }

        if (inputs.Length != count * cols || (long)outputsA.Length < (long)count * rows || (long)outputsB.Length < (long)count * rows)
        {
            throw new ArgumentException($"Input length {inputs.Length} must equal the batch count times the shared column count and both outputs must cover count x {rows}.", nameof(inputs));
        }

        fixed (byte* bytesA = weightA.Bytes)
        fixed (byte* bytesB = weightB.Bytes)
        fixed (float* pIn = inputs)
        fixed (float* outA = outputsA)
        fixed (float* outB = outputsB)
        {
            int strideA = (int)weightA.Shape.RowStrideBytes(formatA);
            int strideB = (int)weightB.Shape.RowStrideBytes(formatB);

            // A stated pair crossover hands each matrix's rows to the blocked matmul tile executor instead of this seam's staged-row traversal: where a fused pair re-reads its two staged rows once per chain group across every weight row, the tile executor amortizes one activation-vector load over a tile group's eight rows' chains. No resolved posture states a width here (see CpuKernelTuning), so an auto open runs exactly the traversal below. Each matrix takes that format's own tile walk — same ascending chunk order, FMA association, and scalar-tail cleanup as its batched-row kernel — so both sides land byte-identical to two independent GemvBatch calls at any thread count.
            if (tuning.PairMatmulRouteBatchWidth is > 0 && count >= tuning.PairMatmulRouteBatchWidth && SupportsGemm(formatA) && SupportsGemm(formatB))
            {
                // The static contiguous row partition never reaches a fused pair at any stated width: these two per-matrix tile dispatches run one after
                // another on the same backend, and pairing their schedules is not part of what this seam measures (see CpuKernelTuning). Both keep dynamic claiming.
                RunMatmulDispatch(bytesA, strideA, pIn, outA, rows, cols, count, formatA, staticRowPartition: false);
                RunMatmulDispatch(bytesB, strideB, pIn, outB, rows, cols, count, formatB, staticRowPartition: false);

                return true;
            }

            // Same per-call participation sample as Gemv/GemvBatch: a quota applied on another thread takes effect from this kernel on.
            int workers = _effectiveThreads;
            bool parallel = workers > 1 && rows >= tuning.ParallelRowThreshold;

            if (!parallel)
            {
                for (int row = 0; row < rows; row++)
                {
                    RunFusedPairRow(bytesA + (long)row * strideA, bytesB + (long)row * strideB, formatA, formatB, cols, pIn, count, rows, tuning.ChainGroup, tuning.PairChainGroup, tuning.MaxScratchCols, outA + row, outB + row);
                }

                return true;
            }

            WorkerPool pool = GetOrCreatePool();
            pool.Run((rows + tuning.RowBlock - 1) / tuning.RowBlock, new PairBatchRowJob(bytesA, bytesB, strideA, strideB, formatA, formatB, cols, pIn, count, rows, outA, outB, tuning));

            return true;
        }
    }

    // One fused row pair. With both rows stageable — the pair scratch ceiling keeps two staged rows cache-resident and the posture
    // resolved a pair group — materialize each matrix's row into its half of the pair scratch with its own bit-exact dequant kernel
    // (the stored values are exactly the chunk vectors that kernel feeds its dot chains), then walk the hot activation batch once
    // through both matrices' chain sets. Without a pair group the existing back-to-back single-row route runs instead: it stages
    // one row at a time and walks the batch once per matrix, which is the fallback for postures the pair scratch ceiling excludes.
    // Both routes take their group width from the call's tuning snapshot and only choose which batch vectors share one pass —
    // never a chain's order or reduction — so every route lands bit-identical to two independent batch calls.
    private static void RunFusedPairRow(byte* rowA, byte* rowB, QuantFormat formatA, QuantFormat formatB, int cols, float* inputs, int count, int rows, int chainGroup, int pairGroup, int maxScratchCols, float* outBaseA, float* outBaseB)
    {
        if (pairGroup >= CpuKernelTuning.ChainGroupMin && 2L * cols <= maxScratchCols)
        {
            RunFusedPairRowPaired(rowA, rowB, formatA, formatB, cols, inputs, count, rows, pairGroup, outBaseA, outBaseB);
            return;
        }

        float[]? scratch = s_rowScratch;
        if (scratch is null || scratch.Length < cols)
        {
            scratch = new float[cols];
            s_rowScratch = scratch;
        }

        fixed (float* pScratch = scratch)
        {
            MaterializeRow(rowA, formatA, pScratch, cols);
            BatchDotFromScratch(pScratch, cols, inputs, count, rows, outBaseA, chainGroup);

            MaterializeRow(rowB, formatB, pScratch, cols);
            BatchDotFromScratch(pScratch, cols, inputs, count, rows, outBaseB, chainGroup);
        }
    }

    // The pair route for one row pair: both rows materialized side by side in the pair scratch, then one shared traversal of the
    // activation batch updating both matrices' accumulator chains.
    private static void RunFusedPairRowPaired(byte* rowA, byte* rowB, QuantFormat formatA, QuantFormat formatB, int cols, float* inputs, int count, int rows, int pairGroup, float* outBaseA, float* outBaseB)
    {
        float[]? scratch = s_pairScratch;
        if (scratch is null || scratch.Length < 2L * cols)
        {
            scratch = new float[2 * cols];
            s_pairScratch = scratch;
        }

        fixed (float* pScratch = scratch)
        {
            MaterializeRow(rowA, formatA, pScratch, cols);
            MaterializeRow(rowB, formatB, pScratch + cols, cols);

            PairBatchDotFromScratch(pScratch, pScratch + cols, cols, inputs, count, rows, outBaseA, outBaseB, pairGroup);
        }
    }

    private static void MaterializeRow(byte* rowBytes, QuantFormat format, float* scratch, int cols)
    {
        switch (format)
        {
            case QuantFormat.Q4_K:
                DequantQ4KAvx2(rowBytes, scratch, cols);
                break;
            default: // Q6_K — the only other member of ScratchBatchDotFormats.
                DequantQ6KAvx2(rowBytes, scratch, cols);
                break;
        }
    }

    private static void ValidateFusableMatrix(WeightRef w, string paramName)
    {
        int[] dims = w.Shape.Dims;
        if (dims.Length < 2 || dims[0] <= 0 || dims[^1] <= 0)
        {
            throw new ArgumentException("Weight shape must be a matrix with positive row and column counts.", paramName);
        }

        long strideBytes = w.Shape.RowStrideBytes(w.Format);
        if (strideBytes > int.MaxValue)
        {
            throw new ArgumentException("Row byte size exceeds the addressable span limit.", paramName);
        }

        if ((long)w.Bytes.Length != dims[0] * strideBytes)
        {
            throw new ArgumentException(
                $"WeightRef byte length is {w.Bytes.Length} but shape {dims[0]}x{dims[^1]} in format {w.Format} requires {dims[0] * strideBytes} bytes.",
                paramName);
        }
    }

    // Row-block executor for one fused pair call: each block walks the same row range through both matrices, mirroring how two
    // independent GemvBatch calls partition that matrix's rows. Rows are independent, so a different worker assignment than an
    // independent pair of calls still lands bit-identical outputs.
    private sealed unsafe class PairBatchRowJob : IGemvJob
    {
        private readonly byte* _bytesA;
        private readonly byte* _bytesB;
        private readonly int _strideA;
        private readonly int _strideB;
        private readonly QuantFormat _formatA;
        private readonly QuantFormat _formatB;
        private readonly int _cols;
        private readonly float* _inputs;
        private readonly int _count;
        private readonly int _rows;
        private readonly float* _outA;
        private readonly float* _outB;
        private readonly int _rowBlock; // captured from the dispatch snapshot so every claimed block uses the same width as Run's block count
        private readonly int _chainGroup;
        private readonly int _pairGroup;
        private readonly int _maxScratchCols;

        public PairBatchRowJob(byte* bytesA, byte* bytesB, int strideA, int strideB, QuantFormat formatA, QuantFormat formatB, int cols, float* inputs, int count, int rows, float* outA, float* outB, CpuKernelTuning tuning)
        {
            _bytesA = bytesA;
            _bytesB = bytesB;
            _strideA = strideA;
            _strideB = strideB;
            _formatA = formatA;
            _formatB = formatB;
            _cols = cols;
            _inputs = inputs;
            _count = count;
            _rows = rows;
            _outA = outA;
            _outB = outB;
            _rowBlock = tuning.RowBlock;
            _chainGroup = tuning.ChainGroup;
            _pairGroup = tuning.PairChainGroup;
            _maxScratchCols = tuning.MaxScratchCols;
        }

        public void ProcessBlock(int block)
        {
            int firstRow = block * _rowBlock;
            int lastRow = Math.Min(firstRow + _rowBlock, _rows);
            for (int row = firstRow; row < lastRow; row++)
            {
                RunFusedPairRow(_bytesA + (long)row * _strideA, _bytesB + (long)row * _strideB, _formatA, _formatB, _cols, _inputs, _count, _rows, _chainGroup, _pairGroup, _maxScratchCols, _outA + row, _outB + row);
            }
        }
    }
}
