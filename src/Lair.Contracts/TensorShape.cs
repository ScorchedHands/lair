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
            _ => throw new ArgumentOutOfRangeException(nameof(fmt), fmt, "Unknown quant format."),
        };
    }
}
