using System.Buffers;
using Lair.Contracts;

namespace Lair.Backends.Cpu;

/// <summary>
/// Single-vector shared-input fusion: one activation vector traversed through two or three dense weight matrices in one
/// dispatch. Every row is computed by the backend's own row evaluator with the same operands, the same accumulator order,
/// and the same row-block partitioning an independent <see cref="CpuBackend.Gemv"/> call would use, so each output element
/// is bit-identical to the independent call — including under threading and at any worker count. The win is dispatch and
/// cache locality: one pool hand-off per fused group instead of one per matrix, and the activation vector stays hot in
/// L1 across every traversal of the call.
/// </summary>
public sealed unsafe partial class CpuBackend : IDenseWeightSharedInputFusion
{
    public bool TryGemvSharedInput(ReadOnlySpan<float> x, WeightRef weightA, Span<float> outputA, WeightRef weightB, Span<float> outputB)
        => TryGemvSharedInputCore(x, weightA, outputA, weightB, outputB, hasC: false, default, default);

    public bool TryGemvSharedInput(ReadOnlySpan<float> x, WeightRef weightA, Span<float> outputA, WeightRef weightB, Span<float> outputB, WeightRef weightC, Span<float> outputC)
        => TryGemvSharedInputCore(x, weightA, outputA, weightB, outputB, hasC: true, weightC, outputC);

    private bool TryGemvSharedInputCore(
        ReadOnlySpan<float> x,
        WeightRef weightA,
        Span<float> outputA,
        WeightRef weightB,
        Span<float> outputB,
        bool hasC,
        WeightRef weightC,
        Span<float> outputC)
    {
        ValidateFusableMatrix(weightA, nameof(weightA));
        ValidateFusableMatrix(weightB, nameof(weightB));

        int cols = weightA.Shape.Dims[^1];
        if (weightB.Shape.Dims[^1] != cols)
        {
            return false; // a shared activation vector only reaches matrices with the same column count.
        }

        if (hasC)
        {
            ValidateFusableMatrix(weightC, nameof(weightC));
            if (weightC.Shape.Dims[^1] != cols)
            {
                return false;
            }
        }

        if (x.Length != cols)
        {
            throw new ArgumentException($"Input vector length {x.Length} must equal the shared column count {cols}.", nameof(x));
        }

        int rowsA = weightA.Shape.Dims[0];
        int rowsB = weightB.Shape.Dims[0];
        int rowsC = hasC ? weightC.Shape.Dims[0] : 0;

        if (outputA.Length < rowsA || outputB.Length < rowsB || (hasC && outputC.Length < rowsC))
        {
            throw new ArgumentException("Each output span must cover its own weight's row count.", nameof(outputB));
        }

        QuantFormat formatA = weightA.Format;
        QuantFormat formatB = weightB.Format;
        QuantFormat formatC = hasC ? weightC.Format : QuantFormat.F32;

        ReadOnlySpan<byte> cBytes = hasC ? weightC.Bytes : default;

        // One shared Q8_K quantization of the activation for the whole call, exactly as each independent Gemv would build
        // it: every sub-4-bit row dot consumes identical qa/da operands whichever call produced them.
        bool usesQ8K = UsesQ8KActivation(formatA) || UsesQ8KActivation(formatB) || (hasC && UsesQ8KActivation(formatC));
        int q8Blocks = Q8KBlockCount(cols);
        byte[]? qaRent = null;
        float[]? daRent = null;
        if (usesQ8K)
        {
            qaRent = ArrayPool<byte>.Shared.Rent(q8Blocks * QkK);
            daRent = ArrayPool<float>.Shared.Rent(q8Blocks);
        }

        fixed (byte* bytesA = weightA.Bytes)
        fixed (byte* bytesB = weightB.Bytes)
        fixed (byte* bytesC = cBytes)
        fixed (float* pX = x)
        fixed (float* outA = outputA)
        fixed (float* outB = outputB)
        fixed (float* outC = outputC)
        fixed (byte* qa = qaRent)
        fixed (float* da = daRent)
        {
            if (usesQ8K)
            {
                QuantizeActivationToQ8K(pX, cols, qa, da);
            }

            Q8KKernelOptions opts = new(_integerDot, _grid, _prefetchWeights);

            int strideA = (int)weightA.Shape.RowStrideBytes(formatA);
            int strideB = (int)weightB.Shape.RowStrideBytes(formatB);
            int strideC = hasC ? (int)weightC.Shape.RowStrideBytes(formatC) : 0;

            int workers = _effectiveThreads;

            // One tuning snapshot for the whole fused call: threshold, block count, and claimed-block width all read the same resolved posture.
            CpuKernelTuning tuning = _tuning;
            int totalRows = rowsA + rowsB + rowsC;
            bool parallel = workers > 1 && totalRows >= tuning.ParallelRowThreshold;

            if (!parallel)
            {
                for (int row = 0; row < totalRows; row++)
                {
                    RunSharedInputRow(bytesA, strideA, rowsA, formatA, outA, bytesB, strideB, rowsB, formatB, outB, bytesC, strideC, rowsC, formatC, outC, row, pX, qa, da, usesQ8K, cols, opts, _useAvx2);
                }

                return true;
            }

            WorkerPool pool = GetOrCreatePool();
            pool.Run((totalRows + tuning.RowBlock - 1) / tuning.RowBlock, new SharedInputRowJob(
                bytesA, strideA, rowsA, formatA, outA,
                bytesB, strideB, rowsB, formatB, outB,
                bytesC, strideC, rowsC, formatC, outC,
                pX, qa, da, usesQ8K ? 1 : 0, cols, opts, _useAvx2, tuning));

            return true;
        }
    }

    // One fused row: the row index selects its matrix exactly as an independent per-matrix loop would, and the evaluator is
    // the same RowDot/RowDotQ8K call with the same operands, so the written value is bit-identical to that call's.
    private static void RunSharedInputRow(
        byte* bytesA, int strideA, int rowsA, QuantFormat formatA, float* outA,
        byte* bytesB, int strideB, int rowsB, QuantFormat formatB, float* outB,
        byte* bytesC, int strideC, int rowsC, QuantFormat formatC, float* outC,
        int row, float* pX, byte* qa, float* da, bool usesQ8K, int cols, Q8KKernelOptions opts, bool avx2)
    {
        if (row < rowsA)
        {
            PrefetchRowWindow(bytesA, strideA, rowsA, row, opts.Prefetch);

            byte* rowBytes = bytesA + (long)row * strideA;
            outA[row] = UsesQ8KActivation(formatA)
                ? RowDotQ8K(rowBytes, qa, da, cols, formatA, opts)
                : RowDot(rowBytes, pX, cols, formatA, avx2);
            return;
        }

        if (row < rowsA + rowsB)
        {
            int local = row - rowsA;
            PrefetchRowWindow(bytesB, strideB, rowsB, local, opts.Prefetch);

            byte* rowBytes = bytesB + (long)local * strideB;
            outB[local] = UsesQ8KActivation(formatB)
                ? RowDotQ8K(rowBytes, qa, da, cols, formatB, opts)
                : RowDot(rowBytes, pX, cols, formatB, avx2);
            return;
        }

        int localC = row - rowsA - rowsB;
        PrefetchRowWindow(bytesC, strideC, rowsC, localC, opts.Prefetch);

        byte* rowBytesC = bytesC + (long)localC * strideC;
        outC[localC] = UsesQ8KActivation(formatC)
            ? RowDotQ8K(rowBytesC, qa, da, cols, formatC, opts)
            : RowDot(rowBytesC, pX, cols, formatC, avx2);
    }

    // Row-block executor for one fused call: the block walks one contiguous slice of the concatenated row space, each row
    // evaluated by its own matrix's evaluator exactly like that matrix's own threaded GEMV would. Rows are independent, so
    // any partition lands bit-identical to independent per-matrix calls.
    private sealed unsafe class SharedInputRowJob : IGemvJob
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
        private readonly float* _x;
        private readonly byte* _qa;
        private readonly float* _da;
        private readonly int _q8k;
        private readonly int _cols;
        private readonly Q8KKernelOptions _opts;
        private readonly bool _avx2;
        private readonly int _rowBlock; // captured from the dispatch snapshot so every claimed block uses the same width as Run's block count

        public SharedInputRowJob(
            byte* bytesA, int strideA, int rowsA, QuantFormat formatA, float* outA,
            byte* bytesB, int strideB, int rowsB, QuantFormat formatB, float* outB,
            byte* bytesC, int strideC, int rowsC, QuantFormat formatC, float* outC,
            float* x, byte* qa, float* da, int q8k, int cols, Q8KKernelOptions opts, bool avx2, CpuKernelTuning tuning)
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
            _x = x;
            _qa = qa;
            _da = da;
            _q8k = q8k;
            _cols = cols;
            _opts = opts;
            _avx2 = avx2;
            _rowBlock = tuning.RowBlock;
        }

        public void ProcessBlock(int block)
        {
            int firstRow = block * _rowBlock;
            int lastRow = Math.Min(firstRow + _rowBlock, _rowsA + _rowsB + _rowsC);
            for (int row = firstRow; row < lastRow; row++)
            {
                RunSharedInputRow(
                    _bytesA, _strideA, _rowsA, _formatA, _outA,
                    _bytesB, _strideB, _rowsB, _formatB, _outB,
                    _bytesC, _strideC, _rowsC, _formatC, _outC,
                    row, _x, _qa, _da, _q8k != 0, _cols, _opts, _avx2);
            }
        }
    }
}
