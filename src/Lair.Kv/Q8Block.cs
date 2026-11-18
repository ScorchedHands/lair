using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Lair.Backends.Cpu;

namespace Lair.Kv;

/// <summary>
/// GGML q8_0 storage element: exactly 34 bytes — one half-precision group scale followed by 32 signed-byte quants. The scale
/// and payload live in one fixed-size struct so a token region stays contiguous at block granularity and a block read is a
/// copy-less typed span over the arena. As an attention side this format serves both roles: as the dot side it scores stored
/// groups with exact integer dots against a Q8_K-prepared query (the signed quants unpack into magnitudes plus a sign mask,
/// so every magnitude stays below 128 and the group's signs fold onto the activation exactly like the shipped sub-4-bit dots);
/// as the value side it dequantizes each stored quant under its own group scale for weighted accumulation.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct Q8Block : IKvFormat<Q8Block>, IKvDotSide<Q8Block>, IKvValueSide<Q8Block>
{
    /// <summary>Values one q8_0 group covers.</summary>
    public const int Values = 32;

    /// <summary>Raw IEEE-754 binary16 bits for this group's scale d.</summary>
    public ushort ScaleBits;

    /// <summary>The 32 signed-byte quants this group stores under its scale.</summary>
    public fixed sbyte Quants[Values];

    public static int ValuesPerBlock => Values;

    public static int BlockBytes => Unsafe.SizeOf<Q8Block>(); // the footprint axis a pool sizes blocks from: 34

    /// <summary>
    /// Quantizes one side region into q8_0 blocks, whole 32-value groups only. Per group the scale is the largest magnitude
    /// divided by 127, stored as a half; a zero scale leaves every quant zero, and otherwise each value rounds away from zero
    /// against the stored half scale and clamps into the signed-byte range.
    /// </summary>
    public static void WriteRegion(ReadOnlySpan<float> values, Span<Q8Block> blocks)
    {
        if (values.Length != blocks.Length * Values)
        {
            throw new ArgumentException(
                $"q8_0 blocks cover whole {Values}-value groups: got {values.Length} values for {blocks.Length} blocks.",
                nameof(values));
        }

        for (int blockIndex = 0; blockIndex < blocks.Length; blockIndex++)
        {
            int start = blockIndex * Values;

            float amax = 0f; // largest magnitude seen in the group (absolute value): q8_0 carries an unsigned scale
            for (int i = 0; i < Values; i++)
            {
                float ax = Math.Abs(values[start + i]);
                if (ax > amax)
                {
                    amax = ax;
                }
            }

            Half dHalf = (Half)(amax / 127f);
            ref Q8Block block = ref blocks[blockIndex];
            block.ScaleBits = BitConverter.HalfToUInt16Bits(dHalf);

            float d = (float)dHalf; // quantize against the stored half scale, exactly as the dequant reads it back
            if (d == 0f)
            {
                for (int i = 0; i < Values; i++)
                {
                    block.Quants[i] = 0; // a zero scale keeps every quant zero: the group stores nothing
                }

                continue;
            }

            for (int i = 0; i < Values; i++)
            {
                double q = Math.Round((double)values[start + i] / d, MidpointRounding.AwayFromZero);
                if (q < -127.0)
                {
                    q = -127.0;
                }
                else if (q > 127.0)
                {
                    q = 127.0;
                }

                block.Quants[i] = (sbyte)q;
            }
        }
    }

    /// <summary>A prepared query head occupies the whole Q8_K super-blocks covering its <c>headDim</c> values, exactly as one GEMV activation does.</summary>
    public static int DotScratchBytesPerHead(int headDim) => CpuBackend.KvQ8KActivationBytes(headDim);

    /// <summary>A prepared query head carries one activation scale per whole Q8_K super-block covering its region.</summary>
    public static int DotScaleCount(int headDim) => CpuBackend.KvQ8KBlockCount(headDim);

    /// <summary>
    /// Prepares the query side once per attention call: each head's region goes through the shipped activation quantizer, so
    /// every stored scale follows the same signed-max rule and nearest rounding a sub-4-bit GEMV stages against. Scratch is
    /// laid out as one contiguous super-block run per head; scales follow it in the same order. An all-zero head lands on zero
    /// quants under a zero scale, and its dot contributes exactly nothing.
    /// </summary>
    public static void PrepareQuery(ReadOnlySpan<float> qHeads, int headCount, int headDim, Span<byte> scratch, Span<float> scales)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);

        int bytesPerHead = DotScratchBytesPerHead(headDim);
        int scaleCount = DotScaleCount(headDim);
        if (scratch.Length < headCount * bytesPerHead || scales.Length < headCount * scaleCount)
        {
            throw new ArgumentException(
                $"prepared-query operands must cover every head's whole super-block run ({bytesPerHead} bytes and {scaleCount} scales per head).",
                nameof(scratch));
        }

        fixed (float* pQ = qHeads)
        fixed (byte* pScratch = scratch)
        fixed (float* pScales = scales)
        {
            for (int h = 0; h < headCount; h++)
            {
                CpuBackend.KvQuantizeActivationToQ8K(pQ + (long)h * headDim, headDim, pScratch + (long)h * bytesPerHead, pScales + (long)h * scaleCount);
            }
        }
    }

    /// <summary>
    /// The K-dot of one scored position's kv-head region against this call's prepared query head: each stored group unpacks
    /// into magnitudes plus a sign mask — the committed writer clamps every quant to [-127, 127], so no magnitude reaches 128
    /// and the fold onto the activation cannot overflow — then runs through one shipped integer sub-block dot under the
    /// resolved encoding. Group totals are exact int32 sums; each is scaled once by this group's stored half scale times the
    /// activation super-block scale its values live in (a whole 32-value group never straddles a super-block boundary).
    /// </summary>
    public static float DotHead(ReadOnlySpan<Q8Block> tokenRegion, int headBlockOffset, ReadOnlySpan<float> qh, ReadOnlySpan<byte> scratch, ReadOnlySpan<float> scales, int headSlot, IntegerDotPath dot)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(headBlockOffset);

        int headBlocks = qh.Length / Values;
        if (tokenRegion.Length < headBlockOffset + headBlocks)
        {
            throw new ArgumentException(
                $"the kv-head region needs whole {Values}-value groups from block {headBlockOffset} for a {qh.Length}-value head, " +
                $"got a region of {tokenRegion.Length} blocks.",
                nameof(tokenRegion));
        }

        int scaleCount = DotScaleCount(qh.Length);
        if (scratch.Length < (long)headSlot * DotScratchBytesPerHead(qh.Length) + DotScratchBytesPerHead(qh.Length)
            || scales.Length < headSlot * scaleCount + scaleCount)
        {
            throw new ArgumentException("prepared-query operands must cover the dotted head.", nameof(scratch));
        }

        fixed (byte* pScratch = scratch)
        {
            sbyte* pAct = (sbyte*)pScratch + (long)headSlot * DotScratchBytesPerHead(qh.Length); // this call's quantized query head
            byte* mag = stackalloc byte[Values];

            float acc = 0f;
            for (int blockIndex = 0; blockIndex < headBlocks; blockIndex++)
            {
                ref readonly Q8Block block = ref Unsafe.Add(ref MemoryMarshal.GetReference(tokenRegion), headBlockOffset + blockIndex); // contiguous region: the head's groups reached by offset, copy-less

                uint signs = 0u;
                for (int i = 0; i < Values; i++)
                {
                    sbyte q = block.Quants[i];
                    if (q < 0)
                    {
                        signs |= 1u << i; // one sign bit per value, the shape both shipped dot encodings fold from
                        mag[i] = (byte)(-q); // magnitude below 128 by construction: the sign belongs to the activation fold
                    }
                    else
                    {
                        mag[i] = (byte)q;
                    }
                }

                CpuBackend.KvSubBlockDotQ8K(mag, signs, pAct + blockIndex * Values, out int loSum, out int hiSum, dot); // exact integer totals over the 32 values

                float storedScale = (float)BitConverter.UInt16BitsToHalf(block.ScaleBits);
                acc += (float)(loSum + hiSum) * storedScale * scales[headSlot * scaleCount + blockIndex / SuperValues]; // group total under both side scales, applied once per group
            }

            return acc;
        }
    }

    /// <summary>
    /// Adds <c>weight * quant * d</c> for every stored value of one scored position's kv-head region: each group decodes under
    /// its own half scale, exactly the pair the committed writer quantized against. A zero-scale group contributes zero to every
    /// accumulator lane it covers.
    /// </summary>
    public static void AccumulateHead(Span<float> outHead, ReadOnlySpan<Q8Block> tokenRegion, int headBlockOffset, float weight)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(headBlockOffset);

        int blocks = outHead.Length / Values;
        if (outHead.Length % Values != 0 || tokenRegion.Length < headBlockOffset + blocks)
        {
            throw new ArgumentException(
                $"the kv-head region needs whole {Values}-value groups from block {headBlockOffset} for a {outHead.Length}-value head, " +
                $"got a region of {tokenRegion.Length} blocks.",
                nameof(tokenRegion));
        }

        for (int blockIndex = 0; blockIndex < blocks; blockIndex++)
        {
            ref readonly Q8Block block = ref Unsafe.Add(ref MemoryMarshal.GetReference(tokenRegion), headBlockOffset + blockIndex);
            float d = (float)BitConverter.UInt16BitsToHalf(block.ScaleBits); // the stored scale, read back exactly as the writer left it

            for (int i = 0; i < Values; i++)
            {
                outHead[blockIndex * Values + i] += weight * ((float)block.Quants[i] * d);
            }
        }
    }

    private const int SuperValues = 256 / Values; // whole 32-value groups per Q8_K super-block: a group maps to exactly one activation scale
}
