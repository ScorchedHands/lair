using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
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
}
