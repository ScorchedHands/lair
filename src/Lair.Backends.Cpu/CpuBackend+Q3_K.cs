using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
    // Q3_K super-blocks hold 256 elements in 110 bytes: 32 bytes carrying the high bit of every three-bit quant
    // at 0..31 (element e keeps its high bit at byte e % 32, bit e / 32), 64 two-bit low-quant bytes at 32..95 in
    // the same group-pair windows as Q2_K, twelve bytes packing sixteen six-bit group scales at 96..107, and
    // half-float d at bytes 108..109. w = d * (sc - 32) * q with sc in [0, 63] and q in [0, 7].
    private const int Q3KBlockBytes = 110;

    // Unpack the six-bit group scale for Q3_K group g out of the twelve packed bytes: groups 0..7 keep their low
    // nibble in scales[g], groups 8..15 in the high nibble of scales[g-8]; the four high bits of every scale live
    // in scales[8 + g%4] at bit shift 2*(g/4).
    private static byte GetQ3KScale(int g, byte* scales)
    {
        int low = g < 8 ? scales[g] & 0xF : scales[g - 8] >> 4;
        int high = (scales[8 + (g & 3)] >> (2 * (g >> 2))) & 0x3;
        return (byte)(low | (high << 4));
    }

    // Q3_K chunk loader for eight window bytes at offset c in {0, 8}: low 2 bits at bit shift 2*(k >> 1) of the
    // window byte, plus the quant's high bit read from bit m of hmask byte 16*(k&1) + e. The bias keeps a set
    // high bit at the plain low value and drops a clear one by four - the same ternary form the scalar path uses,
    // so both paths land on identical values.
    private static Vector256<int> LoadQ3KChunk(byte* q, byte* hm, int k, int c, int m)
    {
        Vector128<byte> window = Sse2.LoadVector128(q + c); // reads [q+c, q+c+16), always inside the 64-byte qs region

        // Per-byte right shift via srli_epi16 (the classic SSE2 pattern): bits borrowed from the adjacent byte land
        // above bit 1 of each lane and are dropped by the mask.
        Vector128<byte> shifted = (k >> 1) switch
        {
            0 => window,
            1 => Sse2.ShiftRightLogical(window.AsUInt16(), 2).AsByte(),
            2 => Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte(),
            _ => Sse2.ShiftRightLogical(window.AsUInt16(), 6).AsByte(),
        };

        Vector128<byte> masked = Sse2.And(shifted, GetTwoBitMask());

        Vector128<int> lo = Sse41.ConvertToVector128Int32(masked); // bytes 0..3 -> dword lanes (pmovzxbd)
        Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes

        // Widen the eight high-bit bytes to dwords and test bit m of each with a constant broadcast mask.
        Vector128<byte> hmWindow = Sse2.LoadVector128(hm + c); // reads [hm+c, hm+c+16), always inside the 32-byte hmask region
        Vector128<int> hmLo = Sse41.ConvertToVector128Int32(hmWindow); // bytes 0..3 -> dword lanes (pmovzxbd)
        Vector128<int> hmHi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(hmWindow, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes
        Vector256<int> bias = Avx2.AndNot(
            Avx2.CompareGreaterThan(Avx2.And(Vector256.Create(hmLo, hmHi), Vector256.Create(m)), Vector256<int>.Zero),
            Vector256.Create(-4)); // set high bit adds nothing, a clear one subtracts four

        return Avx2.Add(Vector256.Create(lo, hi), bias); // signed 3-bit values in element order
    }

    // Q3_K: each 110-byte block covers 256 elements in sixteen 16-element groups g (0..15) with the same group-pair
    // windows as Q2_K, qs at byte 32. The group's six-bit scale is unpacked from the twelve bytes at byte 96 and
    // read as (sc - 32); the quant's high bit lives at hmask byte 16*(k & 1) + e, bit 4h + (k >> 1), and a set
    // high bit keeps the low 2-bit value while a clear one subtracts four. w = d * (sc - 32) * q with q in [0, 7].
    private static float RowDotQ3KScalar(byte* row, float* x, int cols)
    {
        float acc = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q3KBlockBytes;
            float d = ReadHalfLE(block + 108);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                float scale = d * (GetQ3KScale(g, block + 96) - 32);

                byte* q = block + 32 + 32 * h + 16 * (k & 1); // low-2-bit window: bytes [q, q+16) always inside the block
                byte* hm = block + 16 * (k & 1); // element e keeps its high bit at hmask byte 16*(k&1) + e
                int m = 1 << (4 * h + (k >> 1)); // bit selector: the four groups of half h use bits 4h..4h+3
                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.
                for (int e = 0; e < limit; e++)
                {
                    int three = ((q[e] >> (2 * (k >> 1))) & 3) - ((hm[e] & m) != 0 ? 0 : 4);
                    acc += x[subStart + e] * (three * scale);
                }
            }
        }

        return acc;
    }

    // AVX2: per group broadcast the group scale, unpack eight 3-bit quants at a time from the window plus the
    // quarter's hmask byte, then the fused scale and the dot FMA in the same chain as RowDotQ6KAvx2. Partial
    // chunks fall to scalar accumulation.
    private static float RowDotQ3KAvx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float tail = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q3KBlockBytes;
            float dHalf = ReadHalfLE(block + 108);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                float scale = dHalf * (GetQ3KScale(g, block + 96) - 32);
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);

                byte* q = block + 32 + 32 * h + 16 * (k & 1); // low-2-bit window: bytes [q, q+16) always inside the block
                byte* hm = block + 16 * (k & 1); // element e keeps its high bit at hmask byte 16*(k&1) + e
                int m = 1 << (4 * h + (k >> 1)); // bit selector: the four groups of half h use bits 4h..4h+3
                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<int> three = LoadQ3KChunk(q, hm, k, c, m);

                    Vector256<float> qf = Avx.ConvertToVector256Single(three); // signed 3-bit values in element order
                    Vector256<float> xv = Avx.LoadVector256(x + subStart + c); // safe: these eight columns are real and block-covered

                    acc = Fma.MultiplyAdd(xv, Avx.Multiply(scaleV, qf), acc);
                }

                for (int e = (limit / 8) * 8; e < limit; e++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    int three = ((q[e] >> (2 * (k >> 1))) & 3) - ((hm[e] & m) != 0 ? 0 : 4);
                    tail += x[subStart + e] * (three * scale);
                }
            }
        }

        return HSum(acc) + tail;
    }

    // Batched Q3_K row: one 3-bit unpack and fused group scale per chunk feeds all accumulator chains with the
    // same FMA shape as RowDotQ3KAvx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchQ3KAvx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
            {
                byte* block = row + (long)(baseIdx / QkK) * Q3KBlockBytes;
                float dHalf = ReadHalfLE(block + 108);

                for (int g = 0; g < 16; g++)
                {
                    int subStart = baseIdx + g * 16;
                    if (subStart >= cols)
                    {
                        break; // zero-padded tail block: no elements left to consume.
                    }

                    int h = g >> 3; // 128-element half of the super-block
                    int k = g & 7; // 16-element group inside the half
                    float scale = dHalf * (GetQ3KScale(g, block + 96) - 32);
                    Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);

                    byte* q = block + 32 + 32 * h + 16 * (k & 1); // low-2-bit window: bytes [q, q+16) always inside the block
                    byte* hm = block + 16 * (k & 1); // element e keeps its high bit at hmask byte 16*(k&1) + e
                    int m = 1 << (4 * h + (k >> 1)); // bit selector: the four groups of half h use bits 4h..4h+3
                    int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.

                    for (int c = 0; c + 8 <= limit; c += 8)
                    {
                        Vector256<int> three = LoadQ3KChunk(q, hm, k, c, m); // dequantization unpack, materialized once per chunk
                        Vector256<float> wchunk = Avx.Multiply(scaleV, Avx.ConvertToVector256Single(three)); // signed 3-bit values, materialized once
                        for (int b = 0; b < count; b++)
                        {
                            float* chain = pAcc + b * 8;
                            Avx.Store(chain, Fma.MultiplyAdd(Avx.LoadVector256(inputs + (long)b * cols + subStart + c), wchunk, Avx.LoadVector256(chain)));
                        }
                    }

                    for (int e = (limit / 8) * 8; e < limit; e++)
                    {
                        // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                        int three = ((q[e] >> (2 * (k >> 1))) & 3) - ((hm[e] & m) != 0 ? 0 : 4);
                        float w = three * scale;
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

    // Q3_K row expansion: w = d * (sc - 32) * q per group. Same 3-bit unpack and hmask bias as RowDotQ3KAvx2.
    private static void DequantQ3KAvx2(byte* row, float* outPtr, int cols)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q3KBlockBytes;
            float dHalf = ReadHalfLE(block + 108);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                float scale = dHalf * (GetQ3KScale(g, block + 96) - 32);
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);

                byte* q = block + 32 + 32 * h + 16 * (k & 1); // low-2-bit window: bytes [q, q+16) always inside the block
                byte* hm = block + 16 * (k & 1); // element e keeps its high bit at hmask byte 16*(k&1) + e
                int m = 1 << (4 * h + (k >> 1)); // bit selector: the four groups of half h use bits 4h..4h+3
                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<int> three = LoadQ3KChunk(q, hm, k, c, m);
                    Avx.Store(outPtr + subStart + c, Avx.Multiply(scaleV, Avx.ConvertToVector256Single(three))); // signed 3-bit values
                }

                for (int e = (limit / 8) * 8; e < limit; e++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    int three = ((q[e] >> (2 * (k >> 1))) & 3) - ((hm[e] & m) != 0 ? 0 : 4);
                    outPtr[subStart + e] = three * scale;
                }
            }
        }
    }
}
