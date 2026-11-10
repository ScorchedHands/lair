using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
    // Little-endian bf16 load widened to float: the 16 bits land in the f32 high half, so subnormals, zero, and
    // -0.0 pass through unchanged.
    private static float ReadBf16LE(byte* p) => BitConverter.UInt32BitsToSingle((uint)((p[0] | (p[1] << 8)) << 16));

    // BF16 rows are raw little-endian 16-bit bf16 patterns; the f32 value is the bits widened into the high half.
    private static float RowDotBF16Scalar(byte* row, float* x, int cols)
    {
        float acc = 0f;
        for (int i = 0; i < cols; i++)
        {
            acc += ReadBf16LE(row + 2 * i) * x[i];
        }

        return acc;
    }

    // AVX2: eight bf16 patterns per step widened to f32 by a left shift of 16 (the exact bf16-to-f32 conversion:
    // subnormals, zero, and -0.0 pass through unchanged) with the F16 path's eight-lane accumulator, chunk order,
    // scalar tail, and lane reduction.
    private static float RowDotBF16Avx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;

        int i = 0;
        for (; i + 8 <= cols; i += 8)
        {
            // pmovzxwd converts eight bf16 patterns to dwords; the left shift lands them in the f32 high half.
            Vector256<int> bits = Vector256.Create(
                Sse41.ConvertToVector128Int32((ushort*)(row + 2L * i)),
                Sse41.ConvertToVector128Int32((ushort*)(row + 2L * i) + 4));
            Vector256<float> w = Avx2.ShiftLeftLogical(bits, 16).AsSingle(); // vpmovzxwd widens eight bf16 patterns, then the bits land in the f32 high half
            Vector256<float> xv = Avx.LoadVector256(x + i); // safe: the row covers a full chunk of bf16 values
            acc = Fma.MultiplyAdd(w, xv, acc);
        }

        float tail = 0f;
        for (; i < cols; i++)
        {
            tail += ReadBf16LE(row + 2 * i) * x[i];
        }

        return HSum(acc) + tail;
    }

    // Batched BF16 row: one 16-bit-to-f32 widening per eight-element chunk feeds all accumulator chains with the
    // same FMA operand order as RowDotBF16Avx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchBF16Avx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            int i = 0;
            for (; i + 8 <= cols; i += 8)
            {
                // pmovzxwd converts eight bf16 patterns to dwords; the left shift lands them in the f32 high half.
                Vector256<int> bits = Vector256.Create(
                    Sse41.ConvertToVector128Int32((ushort*)(row + 2L * i)),
                    Sse41.ConvertToVector128Int32((ushort*)(row + 2L * i) + 4));
                Vector256<float> w = Avx2.ShiftLeftLogical(bits, 16).AsSingle(); // bf16 bits land in the f32 high half: subnormals, zero, and -0.0 pass through unchanged
                for (int b = 0; b < count; b++)
                {
                    float* chain = pAcc + b * 8;
                    Avx.Store(chain, Fma.MultiplyAdd(w, Avx.LoadVector256(inputs + (long)b * cols + i), Avx.LoadVector256(chain)));
                }
            }

            for (; i < cols; i++)
            {
                float w = ReadBf16LE(row + 2 * i);
                for (int b = 0; b < count; b++)
                {
                    tail[b] += w * inputs[(long)b * cols + i];
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // BF16 row expansion: eight bf16 patterns per chunk widened to f32 by a left shift of 16; odd tails scalar.
    // Same conversion as RowDotBF16Avx2.
    private static void DequantBF16Avx2(byte* row, float* outPtr, int cols)
    {
        int i = 0;
        for (; i + 8 <= cols; i += 8)
        {
            Vector256<int> bits = Vector256.Create(
                Sse41.ConvertToVector128Int32((ushort*)(row + 2L * i)),
                Sse41.ConvertToVector128Int32((ushort*)(row + 2L * i) + 4));
            Avx.Store(outPtr + i, Avx2.ShiftLeftLogical(bits, 16).AsSingle());
        }

        for (; i < cols; i++)
        {
            outPtr[i] = ReadBf16LE(row + 2 * i);
        }
    }
}
