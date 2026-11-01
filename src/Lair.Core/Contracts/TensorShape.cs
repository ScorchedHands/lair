namespace Lair.Core.Contracts;

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
    public long RowStrideBytes(QuantFormat fmt)
    {
        long elements = Dims.Length > 0 ? Dims[^1] : 0;
        return fmt switch
        {
            QuantFormat.F32 => elements * 4,
            QuantFormat.F16 => elements * 2,
            QuantFormat.Q8_0 => (elements + 31) / 32 * 34,
            QuantFormat.Q4_K => (elements + 255) / 256 * 144,
            _ => throw new ArgumentOutOfRangeException(nameof(fmt), fmt, "Unknown quant format."),
        };
    }
}
