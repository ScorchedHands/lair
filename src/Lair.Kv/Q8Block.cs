using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Lair.Kv;

/// <summary>
/// GGML q8_0 storage element: exactly 34 bytes — one half-precision group scale followed by 32 signed-byte quants. The scale
/// and payload live in one fixed-size struct so a token region stays contiguous at block granularity and a block read is a
/// copy-less typed span over the arena.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct Q8Block : IKvFormat<Q8Block>
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
}
