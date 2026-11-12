using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
    // Q5_0 blocks hold 32 five-bit quants in 22 bytes: half-float d at byte 0, four little-endian bytes of 5th
    // quant bits at bytes 2..5 (element i takes bit i of the 32-bit qh word), and 16 nibble bytes at bytes 6..21:
    // element i < 16 is the low nibble of qs[i/2], element i >= 16 the high nibble of qs[(i-16)/2].
    // w = (q - 16) * d with q in [0, 31].
    private const int Q5_0BlockBytes = 22;

    // Little-endian 32-bit load of the Q5_0 5th-bit field at block bytes 2..5.
    private static uint ReadQhLE(byte* block) => (uint)(block[2] | (block[3] << 8) | (block[4] << 16) | (block[5] << 24));

    // Constant per-lane bit masks for Q5_0 5th-bit extraction: lane i tests bit i of the broadcast qh field byte.
    private static Vector256<int> GetQ5_0BitMasks() => Vector256.Create(1, 2, 4, 8, 16, 32, 64, 128);

    // Q5_0 chunk loader for eight elements starting at c in {0, 8, 16, 24}: element j < 16 is the low nibble of
    // qs[j] and element 16+j the high nibble of the same byte, each OR-ed with bit i of the qh dword. The window
    // always loads the full 16-byte qs region (the only 16-byte read that stays inside the 22-byte block) and
    // slides the high half down for c in {8, 24}.
    private static Vector256<int> LoadQ5_0Chunk(byte* block, uint qh, int c)
    {
        byte* qs = block + 6;
        Vector128<byte> window = Sse2.LoadVector128(qs); // full nibble region, always inside the block
        if ((c & 8) != 0)
        {
            window = Ssse3.Shuffle(window, GetSlideHighBytesShuffle()); // qs bytes 8..15 land in lanes 0..7 for c in {8, 24}
        }

        if (c >= 16)
        {
            // High nibbles per byte lane: srli_epi16 then a 0x0F byte mask (the classic SSE2 pattern).
            window = Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte();
        }

        Vector128<byte> masked = Sse2.And(window, GetNibbleMask());

        Vector128<int> lo = Sse41.ConvertToVector128Int32(masked); // bytes 0..3 -> dword lanes (pmovzxbd)
        Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes

        byte qhByte = (byte)(c >> 3); // all eight 5th bits of this chunk live in one qh field byte
        Vector128<byte> replicated = Ssse3.Shuffle(Sse2.LoadVector128(block + 2), Vector128.Create(qhByte));
        Vector128<int> qhDwords = Sse41.ConvertToVector128Int32(replicated); // four dword lanes, each the qh field byte value
        Vector256<int> highBit = Avx2.And(
            Avx2.CompareGreaterThan(Avx2.And(Vector256.Create(qhDwords, qhDwords), GetQ5_0BitMasks()), Vector256<int>.Zero),
            Vector256.Create(16));

        return Avx2.Subtract(Avx2.Or(Vector256.Create(lo, hi), highBit), Vector256.Create(16)); // signed 5-bit values in element order
    }

    // Q5_0: each 22-byte block covers 32 five-bit quants: half-float d at byte 0, the little-endian 32-bit qh
    // word at bytes 2..5, and 16 nibble bytes at bytes 6..21. Element j < 16 is the low nibble of qs[j] and
    // element 16+j the high nibble of the same byte; bit i of qh supplies the 5th bit. w = (q - 16) * d.
    private static float RowDotQ5_0Scalar(byte* row, float* x, int cols)
    {
        float acc = 0f;
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk8)
        {
            byte* block = row + (long)(baseIdx / Qk8) * Q5_0BlockBytes;
            float d = ReadHalfLE(block);
            uint qh = ReadQhLE(block);

            int limit = Math.Min(Qk8, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int l = 0; l < limit; l++)
            {
                int five = ((block[6 + (l & 15)] >> (4 * (l >> 4))) & 0xF) | ((int)(qh >> l) & 1) << 4;
                acc += x[baseIdx + l] * ((five - 16) * d);
            }
        }

        return acc;
    }

    // AVX2: broadcast the block scale once and unpack eight five-bit quants per step from the 16-byte nibble
    // window (low nibbles for elements 0..15, high nibbles for 16..31, each OR-ed with its 5th bit from the
    // qh dword), then the same fused (x * q) * d FMA chain as RowDotQ8Avx2. Window loads stay inside the block.
    // The padded tail of a partial final block falls to scalar accumulation.
    private static float RowDotQ5_0Avx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float tail = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk8)
        {
            byte* block = row + (long)(baseIdx / Qk8) * Q5_0BlockBytes;
            float dHalf = ReadHalfLE(block);
            uint qh = ReadQhLE(block);

            int limit = Math.Min(Qk8, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int l = 0; l < limit; l += 8)
            {
                if (l + 8 > limit)
                {
                    break;
                }

                Vector256<int> five = LoadQ5_0Chunk(block, qh, l);
                Vector256<float> qf = Avx.ConvertToVector256Single(five); // signed 5-bit values in element order
                Vector256<float> xv = Avx.LoadVector256(x + baseIdx + l); // safe: these columns are real and block-covered

                float dLocal = dHalf;
                acc = Fma.MultiplyAdd(Avx.Multiply(xv, qf), Avx.BroadcastScalarToVector256(&dLocal), acc);
            }

            for (int k = (limit / 8) * 8; k < limit; k++)
            {
                int five = ((block[6 + (k & 15)] >> (4 * (k >> 4))) & 0xF) | ((int)(qh >> k) & 1) << 4;
                tail += x[baseIdx + k] * ((five - 16) * dHalf); // partial final block: at most seven elements per row
            }
        }

        return HSum(acc) + tail;
    }

    // Batched Q5_0 row: one 5-bit unpack per eight-element chunk feeds all accumulator chains with the same
    // fused (x * q) * d association as RowDotQ5_0Avx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchQ5_0Avx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk8)
            {
                byte* block = row + (long)(baseIdx / Qk8) * Q5_0BlockBytes;
                float dHalf = ReadHalfLE(block);
                uint qh = ReadQhLE(block);

                int limit = Math.Min(Qk8, cols - baseIdx); // padded quants are zero: stop at the real column count.
                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<float> qf = Avx.ConvertToVector256Single(LoadQ5_0Chunk(block, qh, c)); // dequantization unpack, materialized once per chunk
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
                    int five = ((block[6 + (l & 15)] >> (4 * (l >> 4))) & 0xF) | ((int)(qh >> l) & 1) << 4;
                    float w = (five - 16) * dHalf; // partial final block: at most seven elements per row
                    for (int b = 0; b < count; b++)
                    {
                        tail[b] += w * inputs[(long)b * cols + baseIdx + l];
                    }
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // Q5_0 row expansion: w = (q - 16) * d per block. Same 5-bit unpacking as RowDotQ5_0Avx2, storing instead of dotting.
    private static void DequantQ5_0Avx2(byte* row, float* outPtr, int cols)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk8)
        {
            byte* block = row + (long)(baseIdx / Qk8) * Q5_0BlockBytes;
            float dHalf = ReadHalfLE(block);
            uint qh = ReadQhLE(block);

            int limit = Math.Min(Qk8, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int l = 0; l < limit; l += 8)
            {
                if (l + 8 > limit)
                {
                    break;
                }

                Vector256<int> five = LoadQ5_0Chunk(block, qh, l);
                float dLocal = dHalf;
                Avx.Store(outPtr + baseIdx + l, Avx.Multiply(Avx.ConvertToVector256Single(five), Avx.BroadcastScalarToVector256(&dLocal)));
            }

            for (int k = (limit / 8) * 8; k < limit; k++)
            {
                int five = ((block[6 + (k & 15)] >> (4 * (k >> 4))) & 0xF) | ((int)(qh >> k) & 1) << 4;
                outPtr[baseIdx + k] = (five - 16) * dHalf; // partial final block: at most seven elements per row
            }
        }
    }
}
