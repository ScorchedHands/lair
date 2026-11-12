using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
    // Q6_K super-blocks hold 256 elements in 210 bytes: 128 low-nibble bytes at 0..127, 64 bytes carrying the
    // high 2 bits of six-bit quants at 128..191, 16 signed int8 group scales at 192..207, and half-float d at
    // bytes 208..209. Sixteen 16-element groups: group g covers elements [16g, 16g+16) with scale scales[g];
    // w = d * sc_g * (q - 32) with q in [0, 63].
    private const int Q6KBlockBytes = 210;

    // Q6_K chunk loader for eight window bytes at offset c in {0, 8}: low nibbles for quarters 0..1, high
    // nibbles for quarters 2..3, each OR-ed with the 2-bit field at bit shift 2*quarter of the matching qh
    // byte. The conversion widens with sign, so the values land pre-centered at [-32, 31].
    private static Vector256<int> LoadQ6KChunk(byte* q, byte* qh, int quarter, int c)
    {
        Vector128<byte> window = Sse2.LoadVector128(q + c); // reads [q+c, q+c+16), always inside the 128-byte ql region

        if ((quarter >> 1) != 0)
        {
            // High nibbles per byte lane: srli_epi16 then a 0x0F byte mask (the classic SSE2 pattern).
            window = Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte();
        }

        Vector128<byte> masked = Sse2.And(window, GetNibbleMask());

        Vector128<int> lo = Sse41.ConvertToVector128Int32(masked); // bytes 0..3 -> dword lanes (pmovzxbd)
        Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes

        // Widen the eight high-bit bytes to dwords and extract the 2-bit field at bit shift 2*quarter with
        // constant immediates (the quarter loop variable selects the arm at compile time per unrolled group).
        Vector128<byte> qhWindow = Sse2.LoadVector128(qh + c); // reads [qh+c, qh+c+16), clamped reads stay inside the block
        Vector128<int> qhLo = Sse41.ConvertToVector128Int32(qhWindow); // bytes 0..3 -> dword lanes (pmovzxbd)
        Vector128<int> qhHi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(qhWindow, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes
        Vector256<int> qhDwords = Vector256.Create(qhLo, qhHi);
        Vector256<int> highBit = Avx2.ShiftLeftLogical(quarter switch
        {
            0 => Avx2.And(qhDwords, Vector256.Create(3)),
            1 => Avx2.And(Avx2.ShiftRightLogical(qhDwords, 2), Vector256.Create(3)),
            2 => Avx2.And(Avx2.ShiftRightLogical(qhDwords, 4), Vector256.Create(3)),
            _ => Avx2.And(Avx2.ShiftRightLogical(qhDwords, 6), Vector256.Create(3)),
        }, 4);

        return Avx2.Subtract(Avx2.Or(Vector256.Create(lo, hi), highBit), Vector256.Create(32));
    }

    // Q6_K: each 210-byte block covers 256 elements in sixteen 16-element groups g (0..15). Group g lives in
    // half h = g/8 of the block, quarter k/2 of that half, at ql window 64h + 16*(k&3): low nibbles for
    // quarters 0..1, high nibbles for quarters 2..3; qh byte 128 + 32h + 16*(k&1) + e carries the high 2 bits at
    // bit shift 2*quarter. w = d * sc_g * (q - 32) with q in [0, 63] and sc_g the signed int8 scale at byte
    // 192 + g.
    private static float RowDotQ6KScalar(byte* row, float* x, int cols)
    {
        float acc = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q6KBlockBytes;
            float d = ReadHalfLE(block + 208);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                int quarter = k >> 1; // 0..3: which 32-element quarter of the half the group starts in
                byte* q = block + 64 * h + 16 * (k & 3); // ql window: bytes [q, q+16) always inside the block
                byte* qh = block + 128 + 32 * h + 16 * (k & 1); // 2-bit fields for the same elements, shift 2*quarter
                float scale = d * (sbyte)block[192 + g];
                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.
                for (int e = 0; e < limit; e++)
                {
                    int six = ((q[e] >> (4 * (quarter >> 1))) & 0xF) | (((qh[e] >> (2 * quarter)) & 3) << 4);
                    acc += x[subStart + e] * ((six - 32) * scale);
                }
            }
        }

        return acc;
    }

    // AVX2: per group broadcast the signed group scale, unpack eight 6-bit quants at a time from the nibble
    // window plus the 2-bit high-byte field (signed conversion absorbs the -32 centering), then the fused
    // scale and the dot FMA in the same chain as RowDotQ4KAvx2. Partial chunks fall to scalar accumulation.
    private static float RowDotQ6KAvx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float tail = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q6KBlockBytes;
            float dHalf = ReadHalfLE(block + 208);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                int quarter = k >> 1; // 0..3: which 32-element quarter of the half the group starts in
                byte* q = block + 64 * h + 16 * (k & 3); // ql window: bytes [q, q+16) always inside the block
                byte* qh = block + 128 + 32 * h + 16 * (k & 1); // 2-bit fields for the same elements, shift 2*quarter
                float scale = dHalf * (sbyte)block[192 + g];
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<int> six = LoadQ6KChunk(q, qh, quarter, c);

                    Vector256<float> qf = Avx.ConvertToVector256Single(six); // signed 6-bit values in element order
                    Vector256<float> xv = Avx.LoadVector256(x + subStart + c); // safe: these eight columns are real and block-covered

                    acc = Fma.MultiplyAdd(xv, Avx.Multiply(scaleV, qf), acc);
                }

                for (int e = (limit / 8) * 8; e < limit; e++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    int six = ((q[e] >> (4 * (quarter >> 1))) & 0xF) | (((qh[e] >> (2 * quarter)) & 3) << 4);
                    tail += x[subStart + e] * ((six - 32) * scale);
                }
            }
        }

        return HSum(acc) + tail;
    }

    // Batched Q6_K row: one 6-bit unpack and fused scale per chunk feeds all accumulator chains with the same
    // FMA shape as RowDotQ6KAvx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchQ6KAvx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
            {
                byte* block = row + (long)(baseIdx / QkK) * Q6KBlockBytes;
                float dHalf = ReadHalfLE(block + 208);

                for (int g = 0; g < 16; g++)
                {
                    int subStart = baseIdx + g * 16;
                    if (subStart >= cols)
                    {
                        break; // zero-padded tail block: no elements left to consume.
                    }

                    int h = g >> 3; // 128-element half of the super-block
                    int k = g & 7; // 16-element group inside the half
                    int quarter = k >> 1; // 0..3: which 32-element quarter of the half the group starts in
                    byte* q = block + 64 * h + 16 * (k & 3); // ql window: bytes [q, q+16) always inside the block
                    byte* qh = block + 128 + 32 * h + 16 * (k & 1); // 2-bit fields for the same elements, shift 2*quarter
                    float scale = dHalf * (sbyte)block[192 + g];
                    Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                    int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.

                    for (int c = 0; c + 8 <= limit; c += 8)
                    {
                        Vector256<int> six = LoadQ6KChunk(q, qh, quarter, c); // dequantization unpack, materialized once per chunk
                        Vector256<float> wchunk = Avx.Multiply(scaleV, Avx.ConvertToVector256Single(six)); // fused scale stays in the dot FMA chain
                        for (int b = 0; b < count; b++)
                        {
                            float* chain = pAcc + b * 8;
                            Avx.Store(chain, Fma.MultiplyAdd(Avx.LoadVector256(inputs + (long)b * cols + subStart + c), wchunk, Avx.LoadVector256(chain)));
                        }
                    }

                    for (int e = (limit / 8) * 8; e < limit; e++)
                    {
                        // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                        int six = ((q[e] >> (4 * (quarter >> 1))) & 0xF) | (((qh[e] >> (2 * quarter)) & 3) << 4);
                        float w = (six - 32) * scale;
                        for (int b = 0; b < count; b++)
                        {
                            tail[b] += w * inputs[(long)b * cols + subStart + e];
                        }
                    }
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // Q6_K row expansion: w = d * sc_g * (q - 32) per group. Same 6-bit unpack and signed conversion as
    // RowDotQ6KAvx2.
    private static void DequantQ6KAvx2(byte* row, float* outPtr, int cols)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q6KBlockBytes;
            float dHalf = ReadHalfLE(block + 208);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                int quarter = k >> 1; // 0..3: which 32-element quarter of the half the group starts in
                byte* q = block + 64 * h + 16 * (k & 3); // ql window: bytes [q, q+16) always inside the block
                byte* qh = block + 128 + 32 * h + 16 * (k & 1); // 2-bit fields for the same elements, shift 2*quarter
                float scale = dHalf * (sbyte)block[192 + g];
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<int> six = LoadQ6KChunk(q, qh, quarter, c);
                    Avx.Store(outPtr + subStart + c, Avx.Multiply(scaleV, Avx.ConvertToVector256Single(six))); // signed 6-bit values, pre-centered
                }

                for (int e = (limit / 8) * 8; e < limit; e++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    int six = ((q[e] >> (4 * (quarter >> 1))) & 0xF) | (((qh[e] >> (2 * quarter)) & 3) << 4);
                    outPtr[subStart + e] = (six - 32) * scale;
                }
            }
        }
    }
}
