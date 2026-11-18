using System.Linq;

namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // IQ2_XXS: each 66-byte block covers 256 elements in eight 32-value sub-blocks ib, exactly as the oracle reads them. Sub-block
    // ib carries two packed aux words at bytes 2 + 8*ib (four grid index bytes) and 6 + 8*ib (four seven-bit sign-group codes plus
    // the scale code at bits 28..31). ls = 2 * (signAux >> 28) + 1 covers the whole sub-block; magnitudes stay unsigned, each group's
    // signs fold onto one operand of an exact mad chain, and bsum accumulates (loSum + hiSum) * ls per sub-block before d multiplies
    // it once. The kernel mirrors RowDotIQ2XXSQ8Scalar step for step: same aux windows, same eight-byte entry pairing at lane offsets
    // zero and four, same integer accumulation order, 0.125f landing once after the last block.
    private const int IQ2XxsBlockBytes = 66;

    // IQ2_XXS row expansion: host-side scalar reference math per sub-block reading the same packed bytes the kernel consumes -
    // db = d * (0.5f + scale code) * 0.25f fused once, magnitudes widened from their grid entry, signs flipped through ksigns rows.
    private static void DequantIQ2XXSRow(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int bi = 0; bi * IQ2XxsBlockBytes < rowBytes.Length && bi * 256 < cols; bi++)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice(bi * IQ2XxsBlockBytes, IQ2XxsBlockBytes); // whole blocks: the stride pads to exactly this width.
            float dHalf = ReadHalfLE(block[..2]);

            for (int ib = 0; ib < 8 && ib * 32 < cols - bi * 256; ib++)
            {
                uint gridAux = (uint)(block[2 + 8 * ib] | (block[3 + 8 * ib] << 8) | (block[4 + 8 * ib] << 16) | (block[5 + 8 * ib] << 24));
                uint signAux = (uint)(block[6 + 8 * ib] | (block[7 + 8 * ib] << 8) | (block[8 + 8 * ib] << 16) | (block[9 + 8 * ib] << 24));
                float db = dHalf * (0.5f + (signAux >> 28)) * 0.25f;

                for (int l = 0; l < 4; l++)
                {
                    ReadOnlySpan<byte> entry = GetIq2XXSGrid().Slice(checked((int)(gridAux >> (8 * l) & 0xFF)) * 8, 8); // one whole eight-value grid row in element order.
                    byte signRow = GetKSignsIQ2XS()[(int)((signAux >> (7 * l)) & 127)];

                    for (int b = 0; b < 8; b++)
                    {
                        int c = bi * 256 + ib * 32 + l * 8 + b; // the element this magnitude byte belongs to.
                        if (c >= cols)
                        {
                            break; // zero-padded tail block: stop at the real column count, as every reference path does.
                        }

                        float v = entry[b] * db; // magnitudes widen unsigned and carry no sign of their own ...
                        output[c] = ((signRow >> b) & 1) != 0 ? -v : v; // ... the group's ksigns row applies each value's sign, bit i for element i.
                    }
                }
            }
        }
    }

    private static string BuildGemvIQ2XXSPtx() =>
        BuildQ8KGemvPtx(
            entryName: "gemv_iq2xxs",
            blockBytes: IQ2XxsBlockBytes,
            subBlockArms: [.. Enumerable.Range(0, 8).Select(PtxIQ2XXSSubBlock)],
            finalFactor: "0f3E000000"); // 0.125f lands once at the end, exactly as `return 0.125f * sumf` reads it.

    // One unrolled sub-block arm: assemble both aux words from their constant windows, then walk the four groups - fold each
    // group's ksigns row into sg, take this group's grid index out of the packed aux byte, and dot its two activation words against
    // the entry's halves at lane offsets zero and four. The scale factor rides in lsA after both aux fields have been consumed for
    // signs and indices, so nothing live across dots gets clobbered by the tail math.
    private static string PtxIQ2XXSSubBlock(int ib)
    {
        List<string> lines = [.. PtxOpenSubBlock()];

        lines.AddRange(PtxAssembleFieldLE("lsA", 2 + 8 * ib)); // gridAux: four index bytes packed little-endian ...
        lines.AddRange(PtxAssembleFieldLE("lsB", 6 + 8 * ib)); // signAux: four seven-bit codes plus the scale nibble above them.

        for (int l = 0; l < 4; l++)
        {
            string into = l < 2 ? "loU" : "hiU"; // groups 0..1 own elements 0..15 and groups 2..3 own 16..31, as the reference halves split them.

            lines.AddRange(PtxSignGroup("lsB", l)); // fold this group's ksigns row into sg at its byte position ...
            lines.Add($"    shr.u32 gi, lsA, {8 * l};"); // ... then take this group's grid index out of the aux word's packed bytes ...
            lines.Add("    and.b32 gi, gi, 255;"); // ... masked to its own eight bits.

            for (int half = 0; half < 2; half++)
            {
                int j = 2 * l + half; // the two activation words this group owns inside the sub-block.
                lines.AddRange(PtxLoadActWord(32 * ib + 4 * j)); // its four quants, contiguous in the window ...
                lines.AddRange(PtxLoadEntryDword("gi", 3, half == 0 ? 0 : 4)); // ... against this entry's low or high magnitude half.
                lines.AddRange(PtxWordDot(j, into)); // exact folded mad chain over the word.
            }
        }

        lines.AddRange(PtxLsFromAuxTopNibble("lsB", "lsA")); // ls = 2 * (signAux >> 28) + 1, read after every sign code and index is consumed ...
        lines.AddRange(PtxBsumWhole("lsA")); // bsum += (loSum + hiSum) * ls: one exact integer step per sub-block.

        return string.Join("\n", [.. lines]);
    }
}
