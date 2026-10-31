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
    //   F16   2 bytes per element, stored exactly with no padding.
    //   Q8_0  1 byte per element; quantized payloads live in whole 32-byte blocks,
    //         so a row is rounded up to a 32-byte multiple.
    //   Q4_K  4-bit nibbles packed 2 per byte (0.5 bytes per element); an odd element
    //         count leaves a half byte, so the row is rounded up to a 32-byte multiple.
    public long RowStrideBytes(QuantFormat fmt)
    {
        long elements = Dims.Length > 0 ? Dims[^1] : 0;
        long raw = fmt switch
        {
            QuantFormat.F16 => elements * 2,
            QuantFormat.Q8_0 => elements,
            QuantFormat.Q4_K => (elements + 1) / 2,
            _ => throw new ArgumentOutOfRangeException(nameof(fmt), fmt, "Unknown quant format."),
        };

        return fmt == QuantFormat.F16 ? raw : (raw + 31) / 32 * 32;
    }
}
