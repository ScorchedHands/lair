using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
    // Q5_K super-blocks hold 256 elements in 176 bytes: the Q4_K scale layout at bytes 0..15 (half d, half dmin,
    // 12 packed 6-bit group scales/mins, see QkK), 32 bytes of 5th quant bits at bytes 16..47 (group j reads bit
    // j of the window byte), and 128 nibble bytes at bytes 48..175 in the same group-pair windows as Q4_K.
    // w = d * sc_j * q - dmin * m_j with q in [0, 31].
    private const int Q5KBlockBytes = 176;

    // Q5_K chunk loader for eight window bytes at offset c: low nibbles for even groups, high nibbles for odd,
    // each OR-ed with bit j of the matching 5th-bit byte. Window loads are clamped so they never pass the end
    // of the block, exactly like LoadGroupBytes.
    private static Vector256<int> LoadQ5KChunk(byte* q, byte* qh, byte* blockEnd, int j, int c)
    {
        Vector128<byte> window = LoadGroupBytes(q, blockEnd, c);

        if ((j & 1) != 0)
        {
            // High nibbles per byte lane: srli_epi16 then a 0x0F byte mask (the classic SSE2 pattern).
            window = Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte();
        }

        Vector128<byte> masked = Sse2.And(window, GetNibbleMask());

        Vector128<int> lo = Sse41.ConvertToVector128Int32(masked); // bytes 0..3 -> dword lanes (pmovzxbd)
        Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes

        // Widen the eight 5th-bit bytes to dwords and test bit j of each with a constant broadcast mask.
        Vector128<byte> qhWindow = Sse2.LoadVector128(qh + c); // reads [qh+c, qh+c+16), clamped reads stay inside the block
        Vector128<int> qhLo = Sse41.ConvertToVector128Int32(qhWindow); // bytes 0..3 -> dword lanes (pmovzxbd)
        Vector128<int> qhHi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(qhWindow, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes
        Vector256<int> highBit = Avx2.And(
            Avx2.CompareGreaterThan(Avx2.And(Vector256.Create(qhLo, qhHi), Vector256.Create(1 << j)), Vector256<int>.Zero),
            Vector256.Create(16));

        return Avx2.Or(Vector256.Create(lo, hi), highBit); // unsigned 5-bit values in element order
    }

    // Q5_K: each 176-byte block covers 256 elements in eight 32-element groups j (0..7) with the Q4_K scale
    // layout at bytes 0..15. Group j uses the low nibbles of qs[32*(j/2) .. +31] (qs at byte 48) when j is even
    // and the high nibbles when odd; bit j of qh[32*(j/2) + l] (qh at byte 16) supplies the 5th bit.
    // w = d * sc_j * q - dmin * m_j with q in [0, 31].
    private static float RowDotQ5KScalar(byte* row, float* x, int cols)
    {
        float acc = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q5KBlockBytes;
            float d = ReadHalfLE(block);
            float min = ReadHalfLE(block + 2);

            for (int j = 0; j < 8; j++)
            {
                int subStart = baseIdx + j * 32;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                float scale = d * sc;
                float shift = min * m;

                byte* q = block + 48 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                byte* qh = block + 16 + 32 * (j / 2); // 5th bits for the same window bytes, bit j per byte
                int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.
                for (int l = 0; l < limit; l++)
                {
                    int five = ((q[l] >> (4 * (j & 1))) & 0xF) | (((qh[l] >> j) & 1) << 4);
                    acc += x[subStart + l] * (five * scale - shift);
                }
            }
        }

        return acc;
    }

    // AVX2: per group broadcast the fused scale and negated min-shift, unpack eight 5-bit quants at a time from
    // the nibble window (same clamped window loads as Q4_K) plus the 5th-bit byte, then two FMAs for the
    // four+four dword lanes. Partial chunks fall to scalar accumulation.
    private static float RowDotQ5KAvx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float tail = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q5KBlockBytes;
            byte* blockEnd = block + Q5KBlockBytes;
            float dHalf = ReadHalfLE(block);
            float minHalf = ReadHalfLE(block + 2);

            for (int j = 0; j < 8; j++)
            {
                int subStart = baseIdx + j * 32;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                float scale = dHalf * sc;
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                float negShift = -(minHalf * m);
                Vector256<float> shiftV = Avx.BroadcastScalarToVector256(&negShift);

                byte* q = block + 48 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                byte* qh = block + 16 + 32 * (j / 2); // 5th bits for the same window bytes, bit j per byte
                int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<int> five = LoadQ5KChunk(q, qh, blockEnd, j, c);

                    Vector256<float> qf = Avx.ConvertToVector256Single(five); // eight 5-bit values in element order
                    Vector256<float> xv = Avx.LoadVector256(x + subStart + c); // safe: these eight columns are real and block-covered

                    acc = Fma.MultiplyAdd(xv, Fma.MultiplyAdd(scaleV, qf, shiftV), acc);
                }

                for (int l = (limit / 8) * 8; l < limit; l++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    int five = ((q[l] >> (4 * (j & 1))) & 0xF) | (((qh[l] >> j) & 1) << 4);
                    tail += x[subStart + l] * (five * scale - minHalf * m);
                }
            }
        }

        return HSum(acc) + tail;
    }

    // Batched Q5_K row: one nibble-plus-5th-bit unpack and fused scale/shift per chunk feeds all accumulator
    // chains with the same FMA shape as RowDotQ5KAvx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchQ5KAvx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
            {
                byte* block = row + (long)(baseIdx / QkK) * Q5KBlockBytes;
                byte* blockEnd = block + Q5KBlockBytes;
                float dHalf = ReadHalfLE(block);
                float minHalf = ReadHalfLE(block + 2);

                for (int j = 0; j < 8; j++)
                {
                    int subStart = baseIdx + j * 32;
                    if (subStart >= cols)
                    {
                        break; // zero-padded tail block: no elements left to consume.
                    }

                    GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                    float scale = dHalf * sc;
                    Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                    float negShift = -(minHalf * m);
                    Vector256<float> shiftV = Avx.BroadcastScalarToVector256(&negShift);

                    byte* q = block + 48 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                    byte* qh = block + 16 + 32 * (j / 2); // 5th bits for the same window bytes, bit j per byte
                    int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.

                    for (int c = 0; c + 8 <= limit; c += 8)
                    {
                        Vector256<int> five = LoadQ5KChunk(q, qh, blockEnd, j, c); // dequantization unpack, materialized once per chunk
                        Vector256<float> wchunk = Fma.MultiplyAdd(scaleV, Avx.ConvertToVector256Single(five), shiftV);
                        for (int b = 0; b < count; b++)
                        {
                            float* chain = pAcc + b * 8;
                            Avx.Store(chain, Fma.MultiplyAdd(Avx.LoadVector256(inputs + (long)b * cols + subStart + c), wchunk, Avx.LoadVector256(chain)));
                        }
                    }

                    for (int l = (limit / 8) * 8; l < limit; l++)
                    {
                        // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                        int five = ((q[l] >> (4 * (j & 1))) & 0xF) | (((qh[l] >> j) & 1) << 4);
                        float value = five * scale - minHalf * m;
                        for (int b = 0; b < count; b++)
                        {
                            tail[b] += value * inputs[(long)b * cols + subStart + l];
                        }
                    }
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // Q5_K row expansion: w = d*sc_j*q - dmin*m_j per group. Same nibble-plus-5th-bit unpack and fused scale/shift
    // as RowDotQ5KAvx2.
    private static void DequantQ5KAvx2(byte* row, float* outPtr, int cols)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q5KBlockBytes;
            byte* blockEnd = block + Q5KBlockBytes;
            float dHalf = ReadHalfLE(block);
            float minHalf = ReadHalfLE(block + 2);

            for (int j = 0; j < 8; j++)
            {
                int subStart = baseIdx + j * 32;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                GetScaleMinK4(j, block + 4, out byte sc, out byte m);
                float scale = dHalf * sc;
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                float negShift = -(minHalf * m);
                Vector256<float> shiftV = Avx.BroadcastScalarToVector256(&negShift);

                byte* q = block + 48 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                byte* qh = block + 16 + 32 * (j / 2); // 5th bits for the same window bytes, bit j per byte
                int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<int> five = LoadQ5KChunk(q, qh, blockEnd, j, c);
                    Vector256<float> qf = Avx.ConvertToVector256Single(five); // eight 5-bit values in element order
                    Avx.Store(outPtr + subStart + c, Fma.MultiplyAdd(scaleV, qf, shiftV));
                }

                for (int l = (limit / 8) * 8; l < limit; l++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    int five = ((q[l] >> (4 * (j & 1))) & 0xF) | (((qh[l] >> j) & 1) << 4);
                    outPtr[subStart + l] = five * scale - minHalf * m;
                }
            }
        }
    }
}
