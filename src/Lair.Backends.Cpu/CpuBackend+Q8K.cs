using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Lair.Contracts;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
    // Q8_K activation blocks consumed by the sub-4-bit i-quant integer dots: one float block scale per 256-value
    // super-block followed by 256 signed quants, the reference block_q8_K layout without the bsums the dots never
    // read. The quantizer mirrors quantize_row_q8_K_ref exactly - the scale comes from the signed value carrying the
    // block's largest magnitude, iscale = -127/max negates it so every stored quant lands in [-127, 127], and rounding
    // matches nearest_int. Negating an activation byte during sign folding therefore cannot overflow, every grid
    // magnitude is below 128 by table construction, and the whole dot stays exact inside int16 pair sums and int32
    // lane accumulators. That exactness is what lets the scalar and AVX2 paths differ in lane grouping while staying
    // bit-identical, and what makes row partitioning and batch order irrelevant.
    private const int Q8KMaxQuant = 127;

    /// <summary>The five sub-4-bit i-quants dot against a Q8_K quantization of the activation instead of its FP values.</summary>
    private static bool UsesQ8KActivation(QuantFormat format) =>
        format is QuantFormat.IQ2_XXS or QuantFormat.IQ2_XS or QuantFormat.IQ2_S or QuantFormat.IQ3_XXS or QuantFormat.IQ3_S;

    // Whole Q8_K super-blocks covering one GEMV width. Weight rows are already padded to whole super-blocks and columns
    // past the real width quantize as zero quants whose scale still comes only from the real values, so the padding
    // contributes exactly nothing while every kernel path sees identical operands.
    private static int Q8KBlockCount(int cols) => (cols + QkK - 1) / QkK;

    /// <summary>Quantize one FP activation vector into Q8_K quants plus per-super-block scales, once per GEMV call.</summary>
    private static void QuantizeActivationToQ8K(float* x, int cols, byte* qa, float* da)
    {
        int blocks = Q8KBlockCount(cols);

        for (int i = 0; i < blocks; i++)
        {
            int start = i * QkK;
            int limit = Math.Min(QkK, cols - start); // never negative: blocks covers exactly the real columns' super-blocks.
            byte* q = qa + start;

            float amax = 0f;
            float max = 0f;
            for (int j = 0; j < limit; j++)
            {
                float v = x[start + j];
                float ax = Math.Abs(v);
                if (ax > amax)
                {
                    amax = ax;
                    max = v; // the reference keeps the first strict improvement, sign included.
                }
            }

            if (amax == 0f)
            {
                da[i] = 0f; // all-zero block: scale zero, quants zero, and the dot contributes exactly nothing.
                for (int j = 0; j < QkK; j++)
                {
                    q[j] = 0;
                }

                continue;
            }

            // The reference negates the block scale so the block maximum quantizes to -127 and no quant reaches -128.
            float iscale = -127f / max;
            for (int j = 0; j < limit; j++)
            {
                q[j] = (byte)(sbyte)Math.Min(Q8KMaxQuant, NearestInt(iscale * x[start + j]));
            }

            for (int j = limit; j < QkK; j++)
            {
                q[j] = 0; // padding past the real width: zero quants under the real block scale.
            }

            da[i] = 1f / iscale;
        }
    }

    // Q8_K activation payload bytes per batch vector: whole super-blocks, so a sub-block window never runs off the end.
    private static int Q8KActivationBytes(int cols) => Q8KBlockCount(cols) * QkK;

    /// <summary>Quantize every batch vector into its own contiguous Q8_K region, once per batched GEMV call.</summary>
    private static void QuantizeActivationsToQ8K(float* x, int count, int cols, byte* qa, float* da)
    {
        for (int b = 0; b < count; b++)
        {
            QuantizeActivationToQ8K(x + (long)b * cols, cols, qa + (long)b * Q8KActivationBytes(cols), da + (long)b * Q8KBlockCount(cols));
        }
    }

    // Row evaluator for the sub-4-bit formats: same shape as RowDot but over a quantized activation. A row is still a
    // pure function of its own packed bytes plus the shared activation blocks, so any row partition stays bit-identical.
    private static float RowDotQ8K(byte* row, byte* qa, float* da, int cols, QuantFormat format, bool avx2)
    {
        return format switch
        {
            QuantFormat.IQ2_XXS => avx2 ? RowDotIQ2XXSQ8Avx2(row, qa, da, cols) : RowDotIQ2XXSQ8Scalar(row, qa, da, cols),
            QuantFormat.IQ2_XS => avx2 ? RowDotIQ2XSQ8Avx2(row, qa, da, cols) : RowDotIQ2XSQ8Scalar(row, qa, da, cols),
            QuantFormat.IQ2_S => avx2 ? RowDotIQ2SQ8Avx2(row, qa, da, cols) : RowDotIQ2SQ8Scalar(row, qa, da, cols),
            QuantFormat.IQ3_XXS => avx2 ? RowDotIQ3XQ8Avx2(row, qa, da, cols) : RowDotIQ3XQ8Scalar(row, qa, da, cols),
            QuantFormat.IQ3_S => avx2 ? RowDotIQ3SQ8Avx2(row, qa, da, cols) : RowDotIQ3SQ8Scalar(row, qa, da, cols),
            _ => throw new ArgumentOutOfRangeException(nameof(format), format, "Format does not dot against a Q8_K activation."),
        };
    }

    // Batched row evaluator over quantized activations. Each vector keeps its own accumulator chain advancing exactly
    // like its single-vector GEMV - exact int32 sub-block totals under ls factors, then d * bsum per super-block in
    // block order, then the format's final factor once - so every batch output is bit-identical to a solo row. The
    // weight bytes are traversed once and each decoded sub-block feeds all vectors' integer dots.
    private static void RowBatchDotQ8K(byte* rowBytes, byte* qa, float* da, int blocks, int count, int rows, QuantFormat format, bool avx2, float* outBase)
    {
        if (!avx2)
        {
            for (int b = 0; b < count; b++)
            {
                // Non-AVX2 fallback: the same scalar kernels a solo GEMV would run, one accumulator chain per vector.
                outBase[(long)b * rows] = RowDotQ8K(rowBytes, qa + (long)b * blocks * QkK, da + (long)b * blocks, blocks * QkK, format, false);
            }

            return;
        }

        switch (format)
        {
            case QuantFormat.IQ2_XXS:
                BatchIQ2XXQ8_Avx2(rowBytes, qa, da, blocks, count, rows, outBase);
                break;
            case QuantFormat.IQ2_XS:
                BatchIQ2XSQ8_Avx2(rowBytes, qa, da, blocks, count, rows, outBase);
                break;
            case QuantFormat.IQ2_S:
                BatchIQ2SQ8_Avx2(rowBytes, qa, da, blocks, count, rows, outBase);
                break;
            case QuantFormat.IQ3_XXS:
                BatchIQ3XQ8_Avx2(rowBytes, qa, da, blocks, count, rows, outBase);
                break;
            default:
                BatchIQ3SQ8_Avx2(rowBytes, qa, da, blocks, count, rows, outBase);
                break;
        }
    }

    // Reference nearest_int (the 2^23 bias trick): round to nearest with ties to even. One helper, so both the scalar
    // and any future SIMD quantizer path round identically.
    private static int NearestInt(float v) => (int)Math.Round(v, MidpointRounding.ToEven);

    // Exact integer dot over one 32-value sub-block against Q8_K quants. The staged magnitudes stay unsigned and each
    // group's sign byte folds onto the activation instead of the weight, matching the reference's choice to negate
    // quants. The two sixteen-element halves come back separately so each can carry its own sub-block scale exactly as
    // the reference splits sumi. Scalar port of the same arithmetic.
    private static void SubBlockDotQ8KScalar(byte* mag, uint signs, sbyte* act, out int loSum, out int hiSum)
    {
        int lo = 0;
        int hi = 0;

        for (int e = 0; e < 32; e++)
        {
            int a = act[e];
            if (((signs >> (8 * (e >> 3))) & KMaskIQ2XS(e & 7)) != 0)
            {
                a = -a;
            }

            int product = mag[e] * a;
            if (e < 16)
            {
                lo += product;
            }
            else
            {
                hi += product;
            }
        }

        loSum = lo;
        hiSum = hi;
    }

    // AVX2 form of the same sub-block dot: each group's sign byte is broadcast over its eight activation lanes with a
    // lane mask, a byte-wise AND with the kmask table turns every set sign bit into a nonzero test lane, psubb negates
    // exactly the tested lanes, and pmaddubsw then pmaddwd reproduce the reference's pair-then-quadruple integer grouping. Pair
    // sums are bounded by 2 * 127 * 127 < 32767, so no pmaddubsw lane can saturate, and every lane accumulator and
    // lane sum is an exact int32 - so the lane grouping here and the sequential order of SubBlockDotQ8KScalar are two
    // paths over the same integer total.
    private static void SubBlockDotQ8KAvx2(byte* mag, uint signs, sbyte* act, out int loSum, out int hiSum)
    {
        Vector256<byte> magnitudes = Avx.LoadVector256(mag); // 32 unsigned magnitudes, all below 128
        Vector256<sbyte> activation = Avx.LoadVector256(act); // 32 Q8_K quants, all in [-127, 127]

        // Each group's sign byte is broadcast over exactly its eight activation lanes with a lane mask, then AND-ed with
        // the kmask bit table leaves a nonzero lane exactly where the sign is set, and the byte compare turns that into an
        // all-ones mask over the lanes left positive. A packed dword cannot be spread with one shuffle or broadcast: those
        // intrinsics are element-typed, so broadcasting the four sign bytes broadcasts only their first byte.
        Vector256<byte> groupedSigns = GroupedSignBytes(signs);
        Vector256<byte> positive = Avx2.CompareEqual(Avx2.And(groupedSigns, GetSignBitTable()), Vector256<byte>.Zero);
        Vector256<byte> folded = Avx2.Or(
            Avx2.And(activation.AsByte(), positive),
            Avx2.AndNot(positive, Avx2.Subtract(Vector256<sbyte>.Zero, activation).AsByte()));

        Vector256<short> pairs = Avx2.MultiplyAddAdjacent(magnitudes, folded.AsSByte()); // pmaddubsw: adjacent products summed in int16 lanes
        Vector128<short> ones = Vector128.Create((short)1);

        loSum = HSumInt32(Sse2.MultiplyAddAdjacent(pairs.GetLower(), ones)); // pmaddwd against ones: quadruple sums in int32 lanes
        hiSum = HSumInt32(Sse2.MultiplyAddAdjacent(pairs.GetUpper(), ones));
    }

    // Integer lane reduction: every partial is an exact int32, so lane order is irrelevant and a plain fold is enough.
    private static int HSumInt32(Vector128<int> v) => v.GetElement(0) + v.GetElement(1) + v.GetElement(2) + v.GetElement(3);

    // Broadcasts each packed sign byte over exactly the eight activation lanes of its own value group, so a sub-block's
    // four groups line up with the kmask bit table below. Four scalar broadcasts and lane masks beat any shuffle trick:
    // pshufb indexes bytes but broadcast intrinsics replicate whole elements, which is not how the packed signs spread.
    private static Vector256<byte> GroupedSignBytes(uint signs) => Avx2.Or(
        Avx2.And(Vector256.Create((byte)(signs & 0xFF)), GroupLaneMask(0)),
        Avx2.Or(
            Avx2.And(Vector256.Create((byte)((signs >> 8) & 0xFF)), GroupLaneMask(1)),
            Avx2.Or(
                Avx2.And(Vector256.Create((byte)((signs >> 16) & 0xFF)), GroupLaneMask(2)),
                Avx2.And(Vector256.Create((byte)(signs >> 24)), GroupLaneMask(3)))));

    // All-ones over the eight lanes of value group g inside a 32-byte sub-block, zero elsewhere.
    private static Vector256<byte> GroupLaneMask(int group) => group switch
    {
        0 => Vector256.Create(0xFFFF_FFFF_FFFF_FFFFUL, 0UL, 0UL, 0UL).AsByte(),
        1 => Vector256.Create(0UL, 0xFFFF_FFFF_FFFF_FFFFUL, 0UL, 0UL).AsByte(),
        2 => Vector256.Create(0UL, 0UL, 0xFFFF_FFFF_FFFF_FFFFUL, 0UL).AsByte(),
        _ => Vector256.Create(0UL, 0UL, 0UL, 0xFFFF_FFFF_FFFF_FFFFUL).AsByte(),
    };

    // The kmask_iq2xs bit table as byte lanes: bit i of a group's sign byte tests value i of the eight-value group.
    private static Vector256<byte> GetSignBitTable() => Vector256.Create(
        0x8040_2010_0804_0201UL,
        0x8040_2010_0804_0201UL,
        0x8040_2010_0804_0201UL,
        0x8040_2010_0804_0201UL).AsByte();
}
