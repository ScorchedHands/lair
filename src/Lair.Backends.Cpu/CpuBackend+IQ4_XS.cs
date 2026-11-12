using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
    // IQ4_XS super-blocks hold 256 elements in 136 bytes: half-float d at byte 0, sixteen 2-bit scale high bits at
    // bytes 2..3 (sub-block ib keeps its pair at bit 2*ib), four bytes packing eight 4-bit scale low nibbles at
    // bytes 4..7, and 128 nibble bytes at bytes 8..135. Eight 32-value sub-blocks: sub-block ib covers elements
    // [32ib, 32ib+32) from nibble bytes [16ib, 16ib+16), its six-bit scale read back as (ls - 32).
    // w = d * (ls - 32) * kvalue[q] with kvalue the shared IQ4 value table.
    private const int IQ4XsBlockBytes = 136;

    // Unpack the six-bit sub-block scale for IQ4_XS sub-block ib out of the packed bytes: the low nibble lives in
    // scales_l[ib/2] at bit shift 4*(ib%2) and the two high bits in scales_h at bit shift 2*ib.
    private static int GetIQ4XScale(int ib, byte* scalesL, ushort scalesH)
    {
        int low = (scalesL[ib / 2] >> (4 * (ib % 2))) & 0xF;
        int high = (scalesH >> (2 * ib)) & 0x3;
        return low | (high << 4);
    }

    // Little-endian 16-bit load of the IQ4_XS scale-high field at block bytes 2..3.
    private static ushort ReadScalesHLE(byte* block) => (ushort)(block[2] | (block[3] << 8));

    // IQ4_XS: each 136-byte block covers 256 elements in eight 32-element sub-blocks ib (0..7). Sub-block ib reads
    // its 16 nibble bytes at block + 8 + 16*ib with the same low/high nibble element order as IQ4_NL; its six-bit
    // scale is unpacked from the packed bytes (low nibble in scales_l[ib/2], two high bits in scales_h) and read
    // as (ls - 32). w = d * (ls - 32) * kvalue[q].
    private static float RowDotIQ4XScalar(byte* row, float* x, int cols)
    {
        float acc = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * IQ4XsBlockBytes;
            float d = ReadHalfLE(block);
            ushort scalesH = ReadScalesHLE(block);

            for (int ib = 0; ib < 8; ib++)
            {
                int subStart = baseIdx + ib * 32;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                float scale = d * (GetIQ4XScale(ib, block + 4, scalesH) - 32);
                byte* q = block + 8 + 16 * ib; // sub-block window: bytes [q, q+16) always inside the block
                int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.
                for (int l = 0; l < limit; l++)
                {
                    acc += x[subStart + l] * (scale * KValueIQ4(((q[l & 15] >> (4 * (l >> 4))) & 0xF)));
                }
            }
        }

        return acc;
    }

    // AVX2: per sub-block broadcast the fused scale (it changes every 32 values) and unpack eight table values at
    // a time from the 16-byte nibble window, then the fused (x * w) FMA chain as in RowDotIQ4NAvx2. Partial
    // chunks fall to scalar accumulation.
    private static float RowDotIQ4XAvx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float tail = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * IQ4XsBlockBytes;
            float dHalf = ReadHalfLE(block);
            ushort scalesH = ReadScalesHLE(block);

            for (int ib = 0; ib < 8; ib++)
            {
                int subStart = baseIdx + ib * 32;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                float scale = dHalf * (GetIQ4XScale(ib, block + 4, scalesH) - 32);
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);

                byte* q = block + 8 + 16 * ib; // sub-block window: bytes [q, q+16) always inside the block
                int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.
                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<float> wchunk = LoadIQ4Chunk(q, c, scaleV); // table lookup, materialized once per chunk
                    Vector256<float> xv = Avx.LoadVector256(x + subStart + c); // safe: these columns are real and block-covered

                    acc = Fma.MultiplyAdd(xv, wchunk, acc);
                }

                for (int l = (limit / 8) * 8; l < limit; l++)
                {
                    // Partial final chunk of the sub-block: scalar cleanup, at most seven elements per sub-block.
                    tail += x[subStart + l] * (scale * KValueIQ4(((q[l & 15] >> (4 * (l >> 4))) & 0xF)));
                }
            }
        }

        return HSum(acc) + tail;
    }

    // Batched IQ4_XS row: one table lookup plus the sub-block scale per chunk feeds all accumulator chains with the
    // same fused (x * w) FMA shape as RowDotIQ4XAvx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchIQ4X_Avx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
            {
                byte* block = row + (long)(baseIdx / QkK) * IQ4XsBlockBytes;
                float dHalf = ReadHalfLE(block);
                ushort scalesH = ReadScalesHLE(block);

                for (int ib = 0; ib < 8; ib++)
                {
                    int subStart = baseIdx + ib * 32;
                    if (subStart >= cols)
                    {
                        break; // zero-padded tail block: no elements left to consume.
                    }

                    float scale = dHalf * (GetIQ4XScale(ib, block + 4, scalesH) - 32);
                    Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);

                    byte* q = block + 8 + 16 * ib; // sub-block window: bytes [q, q+16) always inside the block
                    int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.

                    for (int c = 0; c + 8 <= limit; c += 8)
                    {
                        Vector256<float> wchunk = LoadIQ4Chunk(q, c, scaleV); // table lookup, materialized once per chunk
                        for (int b = 0; b < count; b++)
                        {
                            float* chain = pAcc + b * 8;
                            Avx.Store(chain, Fma.MultiplyAdd(Avx.LoadVector256(inputs + (long)b * cols + subStart + c), wchunk, Avx.LoadVector256(chain)));
                        }
                    }

                    for (int l = (limit / 8) * 8; l < limit; l++)
                    {
                        // Partial final chunk of the sub-block: scalar cleanup, at most seven elements per sub-block.
                        float w = scale * KValueIQ4(((q[l & 15] >> (4 * (l >> 4))) & 0xF));
                        for (int b = 0; b < count; b++)
                        {
                            tail[b] += w * inputs[(long)b * cols + subStart + l];
                        }
                    }
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // IQ4_XS row expansion: w = d * (ls - 32) * kvalue[q] per sub-block. Same nibble unpack and table lookup as
    // RowDotIQ4XAvx2, storing instead of dotting.
    private static void DequantIQ4XAvx2(byte* row, float* outPtr, int cols)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += QkK)
        {
            byte* block = row + (long)(baseIdx / QkK) * IQ4XsBlockBytes;
            float dHalf = ReadHalfLE(block);
            ushort scalesH = ReadScalesHLE(block);

            for (int ib = 0; ib < 8; ib++)
            {
                int subStart = baseIdx + ib * 32;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                float scale = dHalf * (GetIQ4XScale(ib, block + 4, scalesH) - 32);
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&scale);

                byte* q = block + 8 + 16 * ib; // sub-block window: bytes [q, q+16) always inside the block
                int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Avx.Store(outPtr + subStart + c, LoadIQ4Chunk(q, c, scaleV)); // table lookup, materialized once per chunk
                }

                for (int l = (limit / 8) * 8; l < limit; l++)
                {
                    // Partial final chunk of the sub-block: scalar cleanup, at most seven elements per sub-block.
                    outPtr[subStart + l] = scale * KValueIQ4(((q[l & 15] >> (4 * (l >> 4))) & 0xF));
                }
            }
        }
    }
}
