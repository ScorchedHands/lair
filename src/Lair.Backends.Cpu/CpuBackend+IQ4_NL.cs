using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
    // IQ4_NL blocks hold 32 elements in 18 bytes: half-float d at byte 0 and 16 nibble bytes at bytes 2..17.
    // Element j < 16 is the low nibble of qs[j] and element 16+j the high nibble of the same byte; every nibble is
    // a code into the shared 16-value IQ4 table. w = d * kvalue[q].
    private const int Qk4Nl = 32;

    private const int IQ4NlBlockBytes = 18;

    // IQ4_NL: each 18-byte block covers 32 elements: half-float d at byte 0 and 16 nibble bytes at bytes 2..17.
    // Element j < 16 is the low nibble of qs[j] and element 16+j the high nibble of the same byte; every nibble is
    // a code into the shared 16-value IQ4 table. w = d * kvalue[q].
    private static float RowDotIQ4NScalar(byte* row, float* x, int cols)
    {
        float acc = 0f;
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk4Nl)
        {
            byte* block = row + (long)(baseIdx / Qk4Nl) * IQ4NlBlockBytes;
            float d = ReadHalfLE(block);

            int limit = Math.Min(Qk4Nl, cols - baseIdx); // padded nibbles are zero: stop at the real column count.
            for (int l = 0; l < limit; l++)
            {
                acc += x[baseIdx + l] * (d * KValueIQ4(((block[2 + (l & 15)] >> (4 * (l >> 4))) & 0xF)));
            }
        }

        return acc;
    }

    // AVX2: broadcast the block scale once and unpack eight table values per step from the 16-byte nibble window
    // (low nibbles for elements 0..15, high nibbles for 16..31, each looked up in the IQ4 table with pshufb),
    // then the fused (x * w) FMA chain. Window loads stay inside the block. The padded tail of a partial final
    // block falls to scalar accumulation.
    private static float RowDotIQ4NAvx2(byte* row, float* x, int cols)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float tail = 0f;

        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk4Nl)
        {
            byte* block = row + (long)(baseIdx / Qk4Nl) * IQ4NlBlockBytes;
            float dHalf = ReadHalfLE(block);
            Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&dHalf);

            byte* q = block + 2; // nibble region: bytes [q, q+16) always inside the block
            int limit = Math.Min(Qk4Nl, cols - baseIdx); // padded nibbles are zero: stop at the real column count.
            for (int c = 0; c + 8 <= limit; c += 8)
            {
                Vector256<float> wchunk = LoadIQ4Chunk(q, c, scaleV); // table lookup, materialized once per chunk
                Vector256<float> xv = Avx.LoadVector256(x + baseIdx + c); // safe: these columns are real and block-covered

                acc = Fma.MultiplyAdd(xv, wchunk, acc);
            }

            for (int l = (limit / 8) * 8; l < limit; l++)
            {
                tail += x[baseIdx + l] * (dHalf * KValueIQ4(((q[l & 15] >> (4 * (l >> 4))) & 0xF))); // partial final block: at most seven elements per row
            }
        }

        return HSum(acc) + tail;
    }

    // Batched IQ4_NL row: one table lookup per eight-element chunk feeds all accumulator chains with the same
    // fused (x * w) FMA shape as RowDotIQ4NAvx2, so the packed bytes are streamed once for the whole batch.
    private static void BatchIQ4N_Avx2(byte* row, float* inputs, int count, int rows, int cols, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch * 8];
        Span<float> tail = stackalloc float[MaxBatch];
        BatchInit(acc, tail, count);

        fixed (float* pAcc = acc)
        {
            for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk4Nl)
            {
                byte* block = row + (long)(baseIdx / Qk4Nl) * IQ4NlBlockBytes;
                float dHalf = ReadHalfLE(block);
                Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&dHalf);

                byte* q = block + 2; // nibble region: bytes [q, q+16) always inside the block
                int limit = Math.Min(Qk4Nl, cols - baseIdx); // padded nibbles are zero: stop at the real column count.

                for (int c = 0; c + 8 <= limit; c += 8)
                {
                    Vector256<float> wchunk = LoadIQ4Chunk(q, c, scaleV); // table lookup, materialized once per chunk
                    for (int b = 0; b < count; b++)
                    {
                        float* chain = pAcc + b * 8;
                        Avx.Store(chain, Fma.MultiplyAdd(Avx.LoadVector256(inputs + (long)b * cols + baseIdx + c), wchunk, Avx.LoadVector256(chain)));
                    }
                }

                for (int l = (limit / 8) * 8; l < limit; l++)
                {
                    float w = dHalf * KValueIQ4(((q[l & 15] >> (4 * (l >> 4))) & 0xF)); // partial final block: at most seven elements per row
                    for (int b = 0; b < count; b++)
                    {
                        tail[b] += w * inputs[(long)b * cols + baseIdx + l];
                    }
                }
            }

            BatchFinish(pAcc, tail, count, rows, outBase);
        }
    }

    // IQ4_NL row expansion: w = d * kvalue[q] per block. Same nibble unpack and table lookup as RowDotIQ4NAvx2,
    // storing instead of dotting.
    private static void DequantIQ4NAvx2(byte* row, float* outPtr, int cols)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Qk4Nl)
        {
            byte* block = row + (long)(baseIdx / Qk4Nl) * IQ4NlBlockBytes;
            float dHalf = ReadHalfLE(block);
            Vector256<float> scaleV = Avx.BroadcastScalarToVector256(&dHalf);

            byte* q = block + 2; // nibble region: bytes [q, q+16) always inside the block
            int limit = Math.Min(Qk4Nl, cols - baseIdx); // padded nibbles are zero: stop at the real column count.

            for (int c = 0; c + 8 <= limit; c += 8)
            {
                Avx.Store(outPtr + baseIdx + c, LoadIQ4Chunk(q, c, scaleV)); // table lookup, materialized once per chunk
            }

            for (int l = (limit / 8) * 8; l < limit; l++)
            {
                outPtr[baseIdx + l] = dHalf * KValueIQ4(((q[l & 15] >> (4 * (l >> 4))) & 0xF)); // partial final block: at most seven elements per row
            }
        }
    }
}
