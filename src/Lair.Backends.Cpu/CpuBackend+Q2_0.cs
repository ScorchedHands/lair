using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
    // Q2_0 blocks hold 64 two-bit quants with one half-float block scale per 64 elements (18 bytes per block):
    // half-float d at byte 0 and 16 little-endian quant bytes at bytes 2..17. Element j is the 2-bit code at bit
    // shift 2*(j % 4) of qs[j / 4], read as 00 = -1, 01 = 0, 10 = +1, 11 = +2. w = (q - 1) * d with q in [0, 3].
    private const int Qk20 = 64;

    private const int Q20BlockBytes = 18;

    // Constant per-lane masks for Q2_0 2-bit extraction from a broadcast 16-bit window: lane i tests bits 2i and
    // 2i+1, the two bits of element i's code.
    private static Vector256<int> GetQ20LowBitMasks() => Vector256.Create(1, 4, 16, 64, 256, 1024, 4096, 16384);

    private static Vector256<int> GetQ20HighBitMasks() => Vector256.Create(2, 8, 32, 128, 512, 2048, 8192, 32768);

    // Q2_0 chunk loader for eight elements starting at c in {0, 8, ..., 56}: the chunk's four 2-bit codes live in
    // the two little-endian bytes at qs[c/4] and qs[c/4 + 1], so one 16-bit window load broadcasts the word to
    // every dword lane. Each element's two bits are tested with a constant per-lane mask, exactly like the Q5_0
    // 5th-bit test: the low bit contributes 1 and the high bit 2, so the codes land in [0, 3] and subtracting 1
    // centers them to {-1, 0, 1, 2} in element order. The two-byte window load stays inside the block for every
    // chunk offset.
    private static Vector256<int> LoadQ20Chunk(byte* block, int c)
    {
        int word = block[2 + (c >> 2)] | (block[2 + (c >> 2) + 1] << 8); // little-endian pair for elements c..c+7
        Vector256<int> wv = Vector256.Create(word); // broadcast the 16-bit window to every dword lane

        Vector256<int> low = Avx2.And(
            Avx2.CompareGreaterThan(Avx2.And(wv, GetQ20LowBitMasks()), Vector256<int>.Zero),
            Vector256.Create(1)); // each element's low code bit
        Vector256<int> high = Avx2.And(
            Avx2.CompareGreaterThan(Avx2.And(wv, GetQ20HighBitMasks()), Vector256<int>.Zero),
            Vector256.Create(2)); // each element's high code bit

        return Avx2.Subtract(Avx2.Add(low, high), Vector256.Create(1)); // signed 2-bit values in element order
    }

    // Q2_0: each 18-byte block covers 64 two-bit quants: half-float d at byte 0 and 16 little-endian quant bytes
    // at bytes 2..17. Element j is the 2-bit code at bit shift 2*(j % 4) of qs[j / 4], read as 00 = -1, 01 = 0,
    // 10 = +1, 11 = +2. w = (q - 1) * d.
    private static float RowDotQ20Scalar(byte* row, float* x, int cols)
    {
        float acc = 0f;
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk20)
        {
            byte* block = row + (long)(baseIdx / Qk20) * Q20BlockBytes;
            float d = ReadHalfLE(block);

            int limit = Math.Min(Qk20, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int l = 0; l < limit; l++)
            {
                int two = (block[2 + (l >> 2)] >> (2 * (l & 3))) & 0x3;
                acc += x[baseIdx + l] * ((two - 1) * d);
            }
        }

        return acc;
    }

    // AVX2: broadcast the block scale once and unpack eight 2-bit quants per step from the two little-endian
    // bytes covering the chunk (broadcast the 16-bit word, then per-lane mask and variable shift), then the same
    // fused (x * q) * d FMA chain as RowDotQ5_0Avx2. The chunk window load stays inside the block. The padded
    // tail of a partial final block falls to scalar accumulation.
    private static float RowDotQ20Avx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float tail = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk20)
        {
            byte* block = row + (long)(baseIdx / Qk20) * Q20BlockBytes;
            float dHalf = ReadHalfLE(block);

            int limit = Math.Min(Qk20, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int c = 0; c + 8 <= limit; c += 8)
            {
                Vector256<int> two = LoadQ20Chunk(block, c);
                Vector256<float> qf = Avx.ConvertToVector256Single(two); // signed 2-bit values in element order
                Vector256<float> xv = Avx.LoadVector256(x + baseIdx + c); // safe: these columns are real and block-covered

                float dLocal = dHalf;
                acc = Fma.MultiplyAdd(Avx.Multiply(xv, qf), Avx.BroadcastScalarToVector256(&dLocal), acc);
            }

            for (int l = (limit / 8) * 8; l < limit; l++)
            {
                int two = (block[2 + (l >> 2)] >> (2 * (l & 3))) & 0x3;
                tail += x[baseIdx + l] * ((two - 1) * dHalf); // partial final block: at most seven elements per row
            }
        }

        return HSum(acc) + tail;
    }

    // Batched Q2_0 row: one 2-bit unpack per eight-element chunk feeds all accumulator chains with the same fused
    // (x * q) * d association as RowDotQ20Avx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchQ20Avx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk20)
            {
                byte* block = row + (long)(baseIdx / Qk20) * Q20BlockBytes;
                float dHalf = ReadHalfLE(block);

                int limit = Math.Min(Qk20, cols - baseIdx); // padded quants are zero: stop at the real column count.
                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<float> qf = Avx.ConvertToVector256Single(LoadQ20Chunk(block, c)); // dequantization unpack, materialized once per chunk
                    float dLocal = dHalf;
                    Vector256<float> dv = Avx.BroadcastScalarToVector256(&dLocal);
                    for (int b = 0; b < count; b++)
                    {
                        float* chain = pAcc + b * 8;
                        Avx.Store(chain, Fma.MultiplyAdd(Avx.Multiply(Avx.LoadVector256(inputs + (long)b * cols + baseIdx + c), qf), dv, Avx.LoadVector256(chain)));
                    }
                }

                for (int l = (limit / 8) * 8; l < limit; l++)
                {
                    int two = (block[2 + (l >> 2)] >> (2 * (l & 3))) & 0x3;
                    float w = (two - 1) * dHalf; // partial final block: at most seven elements per row
                    for (int b = 0; b < count; b++)
                    {
                        tail[b] += w * inputs[(long)b * cols + baseIdx + l];
                    }
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // Q2_0 row expansion: w = (q - 1) * d per block. Same 2-bit unpacking as RowDotQ20Avx2, storing instead of
    // dotting.
    private static void DequantQ20Avx2(byte* row, float* outPtr, int cols)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk20)
        {
            byte* block = row + (long)(baseIdx / Qk20) * Q20BlockBytes;
            float dHalf = ReadHalfLE(block);

            int limit = Math.Min(Qk20, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int l = 0; l < limit; l += 8)
            {
                if (l + 8 > limit)
                {
                    break;
                }

                Vector256<int> two = LoadQ20Chunk(block, l);
                float dLocal = dHalf;
                Avx.Store(outPtr + baseIdx + l, Avx.Multiply(Avx.ConvertToVector256Single(two), Avx.BroadcastScalarToVector256(&dLocal)));
            }

            for (int k = (limit / 8) * 8; k < limit; k++)
            {
                int two = (block[2 + (k >> 2)] >> (2 * (k & 3))) & 0x3;
                outPtr[baseIdx + k] = (two - 1) * dHalf; // partial final block: at most seven elements per row
            }
        }
    }
}
