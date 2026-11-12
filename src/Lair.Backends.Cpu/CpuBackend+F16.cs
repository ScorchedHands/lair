using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
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
}
