using System.Linq;

namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // IQ3_XXS: each 98-byte block covers 256 elements in eight 32-value sub-blocks ib. Sub-block ib reads its aux word at bytes
    // 66 + 4*ib - bits 0..27 pack four seven-bit sign-group codes selecting ksigns rows, and bits 28..31 carry the whole-sub-block
    // scale read back as ls = 2 * (aux >> 28) + 1. Each of the four groups takes two consecutive grid bytes at qs + 8*ib: entry A
    // covers values 0..3 and entry B values 4..7, each a four-byte little-endian magnitude row looked up by its own byte index. The
    // kernel mirrors RowDotIQ3XQ8Scalar step for step, bsum += (loSum + hiSum) * ls per sub-block, d multiplying the exact block total
    // once, 0.25f landing after the last block.
    private const int IQ3XxsBlockBytes = 98;

    // IQ3_XXS row expansion: host-side reference math over the same packed bytes - db fused from d and this aux word's top nibble,
    // magnitudes widened unsigned straight out of their grid entries, signs flipped through ksigns rows.
    private static void DequantIQ3XRow(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int bi = 0; bi * IQ3XxsBlockBytes < rowBytes.Length && bi * 256 < cols; bi++)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice(bi * IQ3XxsBlockBytes, IQ3XxsBlockBytes); // whole blocks: the stride pads to exactly this width.
            float dHalf = ReadHalfLE(block[..2]);

            for (int ib = 0; ib < 8 && ib * 32 < cols - bi * 256; ib++)
            {
                uint aux = (uint)(block[66 + 4 * ib] | (block[67 + 4 * ib] << 8) | (block[68 + 4 * ib] << 16) | (block[69 + 4 * ib] << 24));
                float db = dHalf * (0.5f + (aux >> 28)) * 0.5f; // one factor covers the whole sub-block, fused with d in reference order.

                for (int l = 0; l < 4; l++)
                {
                    byte signRow = GetKSignsIQ2XS()[(int)((aux >> (7 * l)) & 127)]; // this group's seven-bit code selects a shared sign-table row.

                    for (int b = 0; b < 8; b++)
                    {
                        int c = bi * 256 + ib * 32 + l * 8 + b; // the element this magnitude byte belongs to.
                        if (c >= cols)
                        {
                            break; // zero-padded tail block: stop at the real column count, as every reference path does.
                        }

                        int indexByte = block[2 + 8 * ib + 2 * l + (b >> 2)]; // entry A carries values 0..3 and entry B values 4..7 of this group.
                        ReadOnlySpan<byte> entry = GetIq3XXSGrid().Slice(indexByte * 4, 4); // one four-byte magnitude row in element order.

                        float v = entry[b & 3] * db; // magnitudes widen unsigned and carry no sign of their own ...
                        output[c] = ((signRow >> b) & 1) != 0 ? -v : v; // ... the group's ksigns row applies each value's sign, bit i for element i.
                    }
                }
            }
        }
    }

    private static string BuildGemvIQ3XPtx() =>
        BuildQ8KGemvPtx(
            entryName: "gemv_iq3xxs",
            blockBytes: IQ3XxsBlockBytes,
            subBlockArms: [.. Enumerable.Range(0, 8).Select(PtxIQ3XSubBlock)],
            finalFactor: "0f3E800000"); // 0.25f lands once at the end, exactly as `return 0.25f * sumf` reads it.

    // One unrolled sub-block arm: assemble this aux word into lsB where it stays live across every dot (the arms never touch it), fold
    // each group's ksigns row into sg, and walk the two grid bytes that own this group's words - each byte indexes its own four-byte
    // entry against one activation word. The tail reads the scale nibble out of the aux word only after every sign code is consumed.
    private static string PtxIQ3XSubBlock(int ib)
    {
        List<string> lines = [.. PtxOpenSubBlock()];

        lines.AddRange(PtxAssembleFieldLE("lsB", 66 + 4 * ib)); // the aux word: four sign codes under one scale nibble, live until the tail.

        for (int l = 0; l < 4; l++)
        {
            string into = l < 2 ? "loU" : "hiU"; // groups 0..1 own elements 0..15 and groups 2..3 own 16..31, as the reference halves split them.

            lines.AddRange(PtxSignGroup("lsB", l)); // fold this group's ksigns row into sg at its byte position ...

            for (int half = 0; half < 2; half++)
            {
                int j = 2 * l + half; // the two activation words this group owns inside the sub-block.
                lines.AddRange(PtxLoadBlockByte(2 + 8 * ib + j, "gi")); // ... each fed by its own consecutive grid byte against a four-byte entry ...

                lines.AddRange(PtxLoadActWord(32 * ib + 4 * j)); // this word's four quants, contiguous in the window ...
                lines.AddRange(PtxLoadEntryDword("gi", 2, 0)); // ... read as one whole magnitude row.
                lines.AddRange(PtxWordDot(j, into)); // exact folded mad chain over the word.
            }
        }

        lines.AddRange(PtxLsFromAuxTopNibble("lsB", "lsA")); // ls = 2 * (aux >> 28) + 1 covers the whole sub-block ...
        lines.AddRange(PtxBsumWhole("lsA")); // bsum += (loSum + hiSum) * ls: one exact integer step per sub-block.

        return string.Join("\n", [.. lines]);
    }
}
