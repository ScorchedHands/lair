using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
    private const int Q4KBlockBytes = 144;

    // Q4_K: each 144-byte block covers 256 elements in eight 32-element groups j (0..7). Group j uses the low
    // nibbles of qs[32*(j/2) .. +31] when j is even and the high nibbles of those same bytes when odd;
    // w = d * sc_j * q - dmin * m_j. Scale/min unpacking follows the byte layout documented at QkK (scales at
    // block+4, quants at block+16).
    private static float RowDotQ4KScalar(byte* row, float* x, int cols)
    {
        float acc = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q4KBlockBytes;
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

                byte* q = block + 16 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.
                for (int l = 0; l < limit; l++)
                {
                    float nibble = ((q[l] >> (4 * (j & 1))) & 0xF) * scale - shift;
                    acc += x[subStart + l] * nibble;
                }
            }
        }

        return acc;
    }

    // AVX2: per group broadcast the fused scale and negated min-shift, unpack eight nibbles at a time from one
    // 16-byte window load (masked low nibbles or srli-per-word high nibbles), then two FMAs for the four+four
    // dword lanes. Window loads are clamped so they never pass the end of the block. Live YMM: accumulator,
    // scaleV, shiftV plus short-lived unpack temporaries - at most seven simultaneously, so no spills occur.
    // Partial chunks fall to scalar accumulation.
    private static float RowDotQ4KAvx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float tail = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q4KBlockBytes;
            byte* blockEnd = block + Q4KBlockBytes;
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

                byte* q = block + 16 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector128<byte> window = LoadGroupBytes(q, blockEnd, c);

                    if ((j & 1) != 0)
                    {
                        // High nibbles per byte lane: srli_epi16 then a 0x0F byte mask (the classic SSE2 pattern).
                        window = Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte();
                    }

                    Vector128<byte> masked = Sse2.And(window, GetNibbleMask()); // needed nibbles now in each byte lane's low bits

                    Vector128<int> lo = Sse41.ConvertToVector128Int32(masked); // bytes 0..3 -> dword lanes (pmovzxbd)
                    Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes

                    Vector256<float> qf = Avx.ConvertToVector256Single(Vector256.Create(lo, hi)); // eight nibble values in element order
                    Vector256<float> xv = Avx.LoadVector256(x + subStart + c); // safe: these eight columns are real and block-covered

                    acc = Fma.MultiplyAdd(xv, Fma.MultiplyAdd(scaleV, qf, shiftV), acc);
                }

                for (int l = (limit / 8) * 8; l < limit; l++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    float nibble = ((q[l] >> (4 * (j & 1))) & 0xF) * scale - minHalf * m;
                    tail += x[subStart + l] * nibble;
                }
            }
        }

        return HSum(acc) + tail;
    }

    // Batched Q4_K row: one nibble unpack and fused scale/shift per chunk feeds all accumulator chains with the
    // same FMA shape as RowDotQ4KAvx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchQ4KAvx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
            {
                byte* block = row + (long)(baseIdx / QkK) * Q4KBlockBytes;
                byte* blockEnd = block + Q4KBlockBytes;
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

                    byte* q = block + 16 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                    int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.

                    for (int c = 0; c + 8 <= limit; c += 8)
                    {
                        Vector128<byte> window = LoadGroupBytes(q, blockEnd, c);

                        if ((j & 1) != 0)
                        {
                            // High nibbles per byte lane: srli_epi16 then a 0x0F byte mask (the classic SSE2 pattern).
                            window = Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte();
                        }

                        Vector128<byte> masked = Sse2.And(window, GetNibbleMask()); // needed nibbles now in each byte lane's low bits

                        Vector128<int> lo = Sse41.ConvertToVector128Int32(masked); // bytes 0..3 -> dword lanes (pmovzxbd)
                        Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes

                        Vector256<float> wchunk = Fma.MultiplyAdd(scaleV, Avx.ConvertToVector256Single(Vector256.Create(lo, hi)), shiftV); // dequantized chunk, materialized once
                        for (int b = 0; b < count; b++)
                        {
                            float* chain = pAcc + b * 8;
                            Avx.Store(chain, Fma.MultiplyAdd(Avx.LoadVector256(inputs + (long)b * cols + subStart + c), wchunk, Avx.LoadVector256(chain)));
                        }
                    }

                    for (int l = (limit / 8) * 8; l < limit; l++)
                    {
                        // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                        float nibble = ((q[l] >> (4 * (j & 1))) & 0xF) * scale - minHalf * m;
                        for (int b = 0; b < count; b++)
                        {
                            tail[b] += nibble * inputs[(long)b * cols + subStart + l];
                        }
                    }
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // Q4_K row expansion: w = d*sc_j*q - dmin*m_j per group. Same nibble unpack and fused scale/shift as RowDotQ4KAvx2.
    private static void DequantQ4KAvx2(byte* row, float* outPtr, int cols)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * Q4KBlockBytes;
            byte* blockEnd = block + Q4KBlockBytes;
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

                byte* q = block + 16 + 32 * (j / 2); // group-pair window: bytes [q, q+32) always inside the block
                int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector128<byte> window = LoadGroupBytes(q, blockEnd, c);

                    if ((j & 1) != 0)
                    {
                        // High nibbles per byte lane: srli_epi16 then a 0x0F byte mask (the classic SSE2 pattern).
                        window = Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte();
                    }

                    Vector128<byte> masked = Sse2.And(window, GetNibbleMask()); // needed nibbles now in each byte lane's low bits

                    Vector128<int> lo = Sse41.ConvertToVector128Int32(masked); // bytes 0..3 -> dword lanes (pmovzxbd)
                    Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(masked, GetUpperBytesShuffle())); // bytes 4..7 -> dword lanes

                    Vector256<float> qf = Avx.ConvertToVector256Single(Vector256.Create(lo, hi)); // eight nibble values in element order
                    Avx.Store(outPtr + subStart + c, Fma.MultiplyAdd(scaleV, qf, shiftV));
                }

                for (int l = (limit / 8) * 8; l < limit; l++)
                {
                    // Partial final chunk of the group: scalar cleanup, at most seven elements per group.
                    outPtr[subStart + l] = ((q[l] >> (4 * (j & 1))) & 0xF) * scale - minHalf * m;
                }
            }
        }
    }
}
