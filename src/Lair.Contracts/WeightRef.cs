namespace Lair.Contracts;

/// <summary>
/// A resolved weight tensor: packed row bytes plus the metadata needed to interpret them. Bytes holds one
/// packed row after another and must be exactly Shape.Rows * Shape.RowStrideBytes(Format) long. Row packing
/// per format:
///   F32   4 bytes per element, no padding.
///   F16   2 bytes per element, no padding.
///   Q8_0  whole blocks of 34 bytes (half-float scale + 32 int8 quants), last block zero-padded to cols.
///   Q4_K  whole blocks of 144 bytes (two half scales, 12 packed 6-bit group scales/mins, 128 bytes of
///         4-bit quants), last block zero-padded to cols.
///   Q5_0  whole blocks of 22 bytes (half-float scale, 4 bytes of 5th quant bits, 16 bytes of nibbles),
///         last block zero-padded to cols.
///   Q5_K  whole blocks of 176 bytes (two half scales, 12 packed 6-bit group scales/mins, 32 bytes of
///         5th quant bits, 128 bytes of 4-bit quants), last block zero-padded to cols.
///   Q6_K  whole blocks of 210 bytes (128 bytes of low nibbles, 64 bytes of high 2 bits, 16 signed 8-bit
///         group scales, half-float scale), last block zero-padded to cols.
///   Q2_K  whole blocks of 84 bytes (16 bytes of 4-bit scale/min pairs, 64 bytes of 2-bit quants,
///         two half-float scales), last block zero-padded to cols.
///   Q3_K  whole blocks of 110 bytes (32 bytes of quant high bits, 64 bytes of 2-bit quant low bits,
///         12 bytes packing sixteen 6-bit group scales, half-float scale), last block zero-padded to cols.
///   IQ2_XXS whole blocks of 66 bytes (half-float scale, 32 words of 8-element 2-bit grid indices), last
///         block zero-padded to cols.
///   IQ2_XS  whole blocks of 74 bytes (half-float scale, 32 words of 8-element 2-bit grid indices, 8 bytes
///         of 6-bit group scales), last block zero-padded to cols.
///   IQ2_S   whole blocks of 82 bytes (half-float scale, 64 bytes of 2-bit quants, 8 bytes of quant high
///         bits, 8 bytes of 6-bit group scales), last block zero-padded to cols.
///   IQ3_XXS whole blocks of 98 bytes (half-float scale, 96 bytes of 3-bit quants), last block
///         zero-padded to cols.
///   IQ3_S   whole blocks of 110 bytes (half-float scale, 64 bytes of 3-bit quants, 8 bytes of quant high
///         bits, 32 bytes of signs, 4 bytes of 5-bit group scales), last block zero-padded to cols.
///   IQ4_NL  whole blocks of 18 bytes (half-float scale, 16 bytes of 4-bit quants), last block zero-padded
///         to cols.
///   IQ4_XS  whole blocks of 136 bytes (half-float scale, 2 bytes of high scale bits, 4 bytes of low
///         scale bits, 128 bytes of 4-bit quants), last block zero-padded to cols.
///   Q2_0    whole blocks of 18 bytes (half-float scale, 16 bytes of 2-bit quants), last block zero-padded
///         to cols.
///   BF16    2 bytes per element, no padding and no block structure.
/// A span member forces this to be a ref struct: WeightRef is stack-only and must not outlive the memory it
/// borrows (typically an IWeightSource or arena that stays alive for the whole inference run).
/// </summary>
public readonly ref partial struct WeightRef
{
    public WeightRef(ReadOnlySpan<byte> bytes, TensorShape shape, QuantFormat format)
    {
        Bytes = bytes;
        Shape = shape;
        Format = format;
    }

    public ReadOnlySpan<byte> Bytes { get; }

    public TensorShape Shape { get; }

    public QuantFormat Format { get; }
}
