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

    /// <summary>Fusion covers the scratch batch-dot formats only; every other posture returns false and the caller keeps its calls.</summary>
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

            // Same per-call participation sample as Gemv/GemvBatch: a quota applied on another thread takes effect from this kernel on.
            int workers = _effectiveThreads;
            bool parallel = workers > 1 && rows >= tuning.ParallelRowThreshold;

            if (!parallel)
            {
                for (int row = 0; row < rows; row++)
                {
                    RunFusedPairRow(bytesA + (long)row * strideA, bytesB + (long)row * strideB, formatA, formatB, cols, pIn, count, rows, tuning.ChainGroup, outA + row, outB + row);
                }

                return true;
            }

            WorkerPool pool = GetOrCreatePool();
            pool.Run((rows + tuning.RowBlock - 1) / tuning.RowBlock, new PairBatchRowJob(bytesA, bytesB, strideA, strideB, formatA, formatB, cols, pIn, count, rows, outA, outB, tuning));

            return true;
        }
    }

    // One fused row pair: materialize each matrix's row into the shared thread scratch with its own bit-exact dequant kernel (the
    // stored values are exactly the chunk vectors that kernel feeds its dot chains), then run both blocked chain sets over the hot
    // activation batch back to back, so the second traversal reads input vectors this call already pulled through cache. The group
    // width arrives from the call's tuning snapshot and only chooses which batch vectors share one pass — never a chain's order.
    private static void RunFusedPairRow(byte* rowA, byte* rowB, QuantFormat formatA, QuantFormat formatB, int cols, float* inputs, int count, int rows, int chainGroup, float* outBaseA, float* outBaseB)
    {
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
        }

        public void ProcessBlock(int block)
        {
            int firstRow = block * _rowBlock;
            int lastRow = Math.Min(firstRow + _rowBlock, _rows);
            for (int row = firstRow; row < lastRow; row++)
            {
                RunFusedPairRow(_bytesA + (long)row * _strideA, _bytesB + (long)row * _strideB, _formatA, _formatB, _cols, _inputs, _count, _rows, _chainGroup, _outA + row, _outB + row);
            }
        }
    }
}
