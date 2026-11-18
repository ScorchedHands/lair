using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Lair.Kv;

/// <summary>
/// GGML q4_0 storage element: exactly 18 bytes — one half-precision signed group scale followed by 16 nibble bytes that
/// cover 32 values. Byte j holds element start + j in its low nibble and element start + 16 + j in its high nibble, the
/// reference cross-half pairing rather than adjacent pairs. Scale and payload share one fixed-size struct so a token region
/// stays contiguous at block granularity and a block read is a copy-less typed span.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct Q4Block : IKvFormat<Q4Block>
{
    /// <summary>Values one q4_0 group covers.</summary>
    public const int Values = 32;

    private const int NibbleBytes = Values / 2; // one byte per pair: low nibble for element j, high nibble for element 16 + j

    /// <summary>Raw IEEE-754 binary16 bits for this group's signed scale d.</summary>
    public ushort ScaleBits;

    /// <summary>The 16 nibble-packed bytes covering this group's 32 values.</summary>
    public fixed byte Nibbles[NibbleBytes];

    public static int ValuesPerBlock => Values;

    public static int BlockBytes => Unsafe.SizeOf<Q4Block>(); // the footprint axis a pool sizes blocks from: 18

    /// <summary>
    /// Quantizes one side region into q4_0 blocks, whole 32-value groups only, faithful to the committed q4_0 reference
    /// writer: signed-max selection under strict comparison, signed half scale d = max / -8, an FP32 reciprocal id, C-style
    /// truncate-after-offset code rounding clamped to [0, 15], and the cross-half nibble pairing. An all-zero group lands on
    /// d = -0 (half 0x8000), id 0, and every nibble 0x8.
    /// </summary>
    public static void WriteRegion(ReadOnlySpan<float> values, Span<Q4Block> blocks)
    {
        if (values.Length != blocks.Length * Values)
        {
            throw new ArgumentException(
                $"q4_0 blocks cover whole {Values}-value groups: got {values.Length} values for {blocks.Length} blocks.",
                nameof(values));
        }

        for (int blockIndex = 0; blockIndex < blocks.Length; blockIndex++)
        {
            int start = blockIndex * Values;

            float amax = 0f; // magnitude seen so far...
            float max = 0f; // ...and the signed value that produced it: strict '<' keeps the first element on ties
            for (int i = 0; i < Values; i++)
            {
                float v = values[start + i];
                if (amax < Math.Abs(v))
                {
                    amax = Math.Abs(v);
                    max = v;
                }
            }

            // The reference scale is signed: d = max / -8, so the block side that carries the magnitude keeps full range. A
            // group of exact zeros lands on d = -0 (half 0x8000) and id = 0 under the same formula; every value * 0 + 8.5
            // truncates to code 8, which dequantizes back through the zero scale exactly as the reference does.
            float d = max / -8f;
            ref Q4Block block = ref blocks[blockIndex];
            block.ScaleBits = BitConverter.HalfToUInt16Bits((Half)d);

            float id = d != 0f ? 1f / d : 0f; // reciprocal of the FP32 scale, not of the stored half

            for (int j = 0; j < NibbleBytes; j++)
            {
                int low = QuantizeCode(values[start + j], id); // element start + j lives in this byte's low nibble...
                int high = QuantizeCode(values[start + Values / 2 + j], id); // ...element start + 16 + j in its high nibble

                block.Nibbles[j] = (byte)(low | (high << 4)); // reference packing: half-region pairing, not adjacent pairs
            }
        }
    }

    private static int QuantizeCode(float value, float reciprocalScale)
    {
        // z + 8.5f in FP32 first, then truncation toward zero — the C posture of MIN(15, (int8_t)(x0 + 8.5f)), not a
        // round-to-even helper: for finite block values |z| <= 8 so the offset lands in [0.5, 16.5] and only the top clamp is reachable.
        float z = value * reciprocalScale + 8.5f;

        if (!float.IsFinite(z))
        {
            return z > 0f ? 15 : 0; // guard for pathological non-finite inputs only
        }

        int truncated = (int)z; // C-equivalent truncation toward zero for finite in-range values

        if (truncated > 15)
        {
            return 15; // reference top clamp: MIN(15, ...) after the cast
        }

        return truncated < 0 ? 0 : truncated; // kept so a code never leaves [0, 15] on hostile input
    }
}
