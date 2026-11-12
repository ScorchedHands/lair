using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
    // Q2_K super-blocks hold 256 elements in 84 bytes: sixteen packed scale/min pairs at bytes 0..15 (low nibble
    // is the group's 4-bit scale, high nibble its 4-bit min), 64 two-bit quant bytes at 16..79, and half-floats
    // d/dmin at bytes 80..83. Sixteen 16-element groups: group g covers elements [16g, 16g+16) with scale byte
    // scales[g]; inside a 128-element half h the quant window is 16 + 32h + 16*(g & 1) read at bit shift
    // 2*(g >> 1), so one window byte carries four successive groups of the same quarter. w = d * (sc & 0xF) * q
    // - dmin * (sc >> 4) with q in [0, 3].
    private const int Q2KBlockBytes = 84;

    // Q2_K chunk loader for eight window bytes at offset c in {0, 8}: the group's 2-bit field lives at bit shift
    // 2*(k >> 1) of each window byte, so one constant per-group shift and a 0x3 byte mask extract eight quants
    // from a single 16-byte window load.
    private static Vector256<int> LoadQ2KChunk(byte* q, int k, int c)
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

        Vector128<byte> masked = Sse2.And(shifted, GetTwoBitMask()); // eight 2-bit quants in each byte lane's low bits

        Vector128<int> lo = Sse41.ConvertToVector128Int32(masked); // bytes 0..3 -> dword lanes (pmovzxbd)
        Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes

        return Vector256.Create(lo, hi); // unsigned 2-bit values in element order
    }

    // Q2_K: each 84-byte block covers 256 elements in sixteen 16-element groups g (0..15), group g inside half
    // h = g/8 at quarter k = g%8. Scale byte g holds the group's 4-bit scale in its low nibble and its 4-bit min
    // in its high nibble; the group's quants live at window byte 16 + 32h + 16*(k & 1) + e read at bit shift
    // 2*(k >> 1). w = d * (sc & 0xF) * q - dmin * (sc >> 4) with q in [0, 3].
    private static float RowDotQ2KScalar(byte* row, float* x, int cols)
    {
        float acc = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q2KBlockBytes;
            float d = ReadHalfLE(block + 80);
            float min = ReadHalfLE(block + 82);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                byte sc = block[g];
                float scale = d * (sc & 0xF);
                float shift = min * (sc >> 4);

                byte* q = block + 16 + 32 * h + 16 * (k & 1); // group-pair window: bytes [q, q+16) always inside the block
                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.
                for (int e = 0; e < limit; e++)
                {
                    float twoBit = ((q[e] >> (2 * (k >> 1))) & 3) * scale - shift;
                    acc += x[subStart + e] * twoBit;
                }
            }
        }

        return acc;
    }

    // AVX2: per group broadcast the fused scale and negated min-shift, unpack eight 2-bit quants at a time from one
    // 16-byte window load with a constant per-group bit shift, then two FMAs for the four+four dword lanes.
    // Partial chunks fall to scalar accumulation.
    private static float RowDotQ2KAvx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float tail = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q2KBlockBytes;
            float dHalf = ReadHalfLE(block + 80);
            float minHalf = ReadHalfLE(block + 82);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                byte sc = block[g];
                float scale = dHalf * (sc & 0xF);
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                float negShift = -(minHalf * (sc >> 4));
                Vector256<float> shiftV = Avx.BroadcastScalarToVector256(&negShift);

                byte* q = block + 16 + 32 * h + 16 * (k & 1); // group-pair window: bytes [q, q+16) always inside the block
                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<int> two = LoadQ2KChunk(q, k, c);
                    Vector256<float> qf = Avx.ConvertToVector256Single(two); // eight 2-bit values in element order
                    Vector256<float> xv = Avx.LoadVector256(x + subStart + c); // safe: these eight columns are real and block-covered

                    acc = Fma.MultiplyAdd(xv, Fma.MultiplyAdd(scaleV, qf, shiftV), acc);
                }

                for (int e = (limit / 8) * 8; e < limit; e++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    float twoBit = ((q[e] >> (2 * (k >> 1))) & 3) * scale - minHalf * (sc >> 4);
                    tail += x[subStart + e] * twoBit;
                }
            }
        }

        return HSum(acc) + tail;
    }

    // Batched Q2_K row: one 2-bit unpack and fused scale/shift per chunk feeds all accumulator chains with the
    // same FMA shape as RowDotQ2KAvx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchQ2KAvx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
            {
                byte* block = row + (long)(baseIdx / QkK) * Q2KBlockBytes;
                float dHalf = ReadHalfLE(block + 80);
                float minHalf = ReadHalfLE(block + 82);

                for (int g = 0; g < 16; g++)
                {
                    int subStart = baseIdx + g * 16;
                    if (subStart >= cols)
                    {
                        break; // zero-padded tail block: no elements left to consume.
                    }

                    int h = g >> 3; // 128-element half of the super-block
                    int k = g & 7; // 16-element group inside the half
                    byte sc = block[g];
                    float scale = dHalf * (sc & 0xF);
                    Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                    float negShift = -(minHalf * (sc >> 4));
                    Vector256<float> shiftV = Avx.BroadcastScalarToVector256(&negShift);

                    byte* q = block + 16 + 32 * h + 16 * (k & 1); // group-pair window: bytes [q, q+16) always inside the block
                    int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.

                    for (int c = 0; c + 8 <= limit; c += 8)
                    {
                        Vector256<int> two = LoadQ2KChunk(q, k, c); // dequantization unpack, materialized once per chunk
                        Vector256<float> wchunk = Fma.MultiplyAdd(scaleV, Avx.ConvertToVector256Single(two), shiftV);
                        for (int b = 0; b < count; b++)
                        {
                            float* chain = pAcc + b * 8;
                            Avx.Store(chain, Fma.MultiplyAdd(Avx.LoadVector256(inputs + (long)b * cols + subStart + c), wchunk, Avx.LoadVector256(chain)));
                        }
                    }

                    for (int e = (limit / 8) * 8; e < limit; e++)
                    {
                        // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                        float value = ((q[e] >> (2 * (k >> 1))) & 3) * scale - minHalf * (sc >> 4);
                        for (int b = 0; b < count; b++)
                        {
                            tail[b] += value * inputs[(long)b * cols + subStart + e];
                        }
                    }
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // Q2_K row expansion: w = d * (sc & 0xF) * q - dmin * (sc >> 4) per group. Same 2-bit unpack as RowDotQ2KAvx2,
    // but multiply and subtract stay unfused so the expansion is bit-identical to the scalar path.
    private static void DequantQ2KAvx2(byte* row, float* outPtr, int cols)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q2KBlockBytes;
            float dHalf = ReadHalfLE(block + 80);
            float minHalf = ReadHalfLE(block + 82);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                byte sc = block[g];
                float scale = dHalf * (sc & 0xF);
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);
                float shift = minHalf * (sc >> 4);
                Vector256<float> shiftV = Avx.BroadcastScalarToVector256(&shift);

                byte* q = block + 16 + 32 * h + 16 * (k & 1); // group-pair window: bytes [q, q+16) always inside the block
                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<int> two = LoadQ2KChunk(q, k, c);
                    Avx.Store(outPtr + subStart + c, Avx.Subtract(Avx.Multiply(scaleV, Avx.ConvertToVector256Single(two)), shiftV));
                }

                for (int e = (limit / 8) * 8; e < limit; e++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    outPtr[subStart + e] = ((q[e] >> (2 * (k >> 1))) & 3) * scale - shift;
                }
            }
        }
    }
}
