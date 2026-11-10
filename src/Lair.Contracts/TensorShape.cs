namespace Lair.Contracts;

public sealed record TensorShape(int[] Dims)
{
    public long ElementCount
    {
        get
        {
            long count = 1;
            foreach (int dim in Dims)
            {
                count *= dim;
            }

            return count;
        }
    }

    public int Rows => Dims.Length > 0 ? Dims[0] : 0;

    // Packed row size in bytes for a row of Dims[^1] elements:
    //   F32   4 bytes per element, stored exactly with no padding.
    //   F16   2 bytes per element, stored exactly with no padding.
    //   Q8_0  whole 34-byte blocks (int16 scale + 32 int8 quants); the final block is
    //         zero-padded to cover a partial run of up to 32 elements.
    //   Q4_K  whole 144-byte blocks (half d, half dmin, 12 packed 6-bit group scales/mins,
    //         128 bytes of 4-bit quants); the final block is zero-padded to cover a partial
    //         run of up to 256 elements.
    //   Q5_0  whole 22-byte blocks (half d, 4 bytes of 5th quant bits, 16 bytes of nibbles);
    //         the final block is zero-padded to cover a partial run of up to 32 elements.
    //   Q5_K  whole 176-byte blocks (half d, half dmin, 12 packed 6-bit group scales/mins,
    //         32 bytes of 5th quant bits, 128 bytes of 4-bit quants); the final block is
    //         zero-padded to cover a partial run of up to 256 elements.
    //   Q6_K  whole 210-byte blocks (128 bytes of low nibbles, 64 bytes of high 2 bits,
    //         16 signed 8-bit group scales, half d); the final block is zero-padded to cover
    //         a partial run of up to 256 elements.
    //   Q2_K  whole 84-byte blocks (16 bytes of 4-bit scale/min pairs, 64 bytes of 2-bit quants,
    //         half d, half dmin); the final block is zero-padded to cover a partial run of up
    //         to 256 elements.
    //   Q3_K  whole 110-byte blocks (32 bytes of quant high bits, 64 bytes of 2-bit quant low bits,
    //         12 bytes packing sixteen 6-bit group scales, half d); the final block is zero-padded to
    //         cover a partial run of up to 256 elements.
    //   IQ2_XXS whole 66-byte blocks (half d, 32 words of 8-element 2-bit grid indices); the final block is
    //         zero-padded to cover a partial run of up to 256 elements.
    //   IQ2_XS  whole 74-byte blocks (half d, 32 words of 8-element 2-bit grid indices, 8 bytes of 6-bit
    //         group scales); the final block is zero-padded to cover a partial run of up to 256 elements.
    //   IQ2_S   whole 82-byte blocks (half d, 64 bytes of 2-bit quants, 8 bytes of quant high bits,
    //         8 bytes of 6-bit group scales); the final block is zero-padded to cover a partial run of
    //         up to 256 elements.
    //   IQ3_XXS whole 98-byte blocks (half d, 96 bytes of 3-bit quants); the final block is zero-padded
    //         to cover a partial run of up to 256 elements.
    //   IQ3_S   whole 110-byte blocks (half d, 64 bytes of 3-bit quants, 8 bytes of quant high bits,
    //         32 bytes of signs, 4 bytes of 5-bit group scales); the final block is zero-padded to cover
    //         a partial run of up to 256 elements.
    //   IQ4_NL  whole 18-byte blocks (half d, 16 bytes of 4-bit quants); the final block is zero-padded
    //         to cover a partial run of up to 32 elements.
    //   IQ4_XS  whole 136-byte blocks (half d, 2 bytes of high scale bits, 4 bytes of low scale bits,
    //         128 bytes of 4-bit quants); the final block is zero-padded to cover a partial run of up
    //         to 256 elements.
    //   Q2_0  whole 18-byte blocks (half d, 16 bytes of 2-bit quants); the final block is zero-padded to
    //         cover a partial run of up to 64 elements.
    //   BF16  2 bytes per element, stored exactly with no padding and no block structure.
    public long RowStrideBytes(QuantFormat fmt)
    {
        long elements = Dims.Length > 0 ? Dims[^1] : 0;
        return fmt switch
        {
            QuantFormat.F32 => elements * 4,
            QuantFormat.F16 => elements * 2,
            QuantFormat.Q8_0 => (elements + 31) / 32 * 34,
            QuantFormat.Q4_K => (elements + 255) / 256 * 144,
            QuantFormat.Q5_0 => (elements + 31) / 32 * 22,
            QuantFormat.Q5_K => (elements + 255) / 256 * 176,
            QuantFormat.Q6_K => (elements + 255) / 256 * 210,
            QuantFormat.Q2_K => (elements + 255) / 256 * 84,
            QuantFormat.Q3_K => (elements + 255) / 256 * 110,
            QuantFormat.IQ2_XXS => (elements + 255) / 256 * 66,
            QuantFormat.IQ2_XS => (elements + 255) / 256 * 74,
            QuantFormat.IQ2_S => (elements + 255) / 256 * 82,
            QuantFormat.IQ3_XXS => (elements + 255) / 256 * 98,
            QuantFormat.IQ3_S => (elements + 255) / 256 * 110,
            QuantFormat.IQ4_NL => (elements + 31) / 32 * 18,
            QuantFormat.IQ4_XS => (elements + 255) / 256 * 136,
            QuantFormat.Q2_0 => (elements + 63) / 64 * 18,
            QuantFormat.BF16 => elements * 2,
            _ => throw new ArgumentOutOfRangeException(nameof(fmt), fmt, "Unknown quant format."),
        };
    }
}
