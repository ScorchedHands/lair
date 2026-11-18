using System.Linq;

namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // IQ3_S: each 110-byte block covers 256 elements in eight 32-value sub-blocks ib. Sub-block ib reads its eight grid index bytes at
    // bytes 2 + 8*ib, each nine-bit index taking bit eight from position j of the qh byte at block byte 66 + ib; raw sign bytes live at
    // block byte 74 + 4*ib per group; and one scale code rides in a nibble of block byte 106 + (ib >> 1) - even sub-blocks read that
    // byte's low half, odd its high - read back as ls = 1 + 2 * code covering the whole sub-block. The kernel mirrors RowDotIQ3SQ8Scalar
    // step for step: same index/qh pairing, raw sign rows folded the same way, bsum += (loSum + hiSum) * ls per sub-block, d multiplying
    // the exact block total once - and no final factor at all, because this format carries its scale inside ls itself.
    private const int IQ3SBlockBytes = 110;

    // IQ3_S row expansion: host-side reference math over the same packed bytes - db fused from d and this sub-block's scale code alone,
    // magnitudes widened unsigned straight out of their grid entries, signs applied through the raw sign plane.
    private static void DequantIQ3SRow(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int bi = 0; bi * IQ3SBlockBytes < rowBytes.Length && bi * 256 < cols; bi++)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice(bi * IQ3SBlockBytes, IQ3SBlockBytes); // whole blocks: the stride pads to exactly this width.
            float dHalf = ReadHalfLE(block[..2]);

            for (int ib = 0; ib < 8 && ib * 32 < cols - bi * 256; ib++)
            {
                byte qh = block[66 + ib]; // one byte carries the extension bit of every index in this sub-block.
                int ls = 1 + 2 * ((block[106 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF); // the scale code half this sub-block reads, doubled into its factor ...
                float db = dHalf * ls; // ... and fused with d in reference order: no further scalar factor belongs to this format.

                for (int l = 0; l < 4; l++)
                {
                    byte signRow = block[74 + 4 * ib + l]; // the group's raw sign byte: no table lookup anywhere on this path.

                    for (int b = 0; b < 8; b++)
                    {
                        int c = bi * 256 + ib * 32 + l * 8 + b; // the element this magnitude byte belongs to.
                        if (c >= cols)
                        {
                            break; // zero-padded tail block: stop at the real column count, as every reference path does.
                        }

                        int j = 2 * l + (b >> 2); // entry A carries values 0..3 and entry B values 4..7 of this group ...
                        int gi = block[2 + 8 * ib + j] | (((qh >> j) & 1) << 8); // ... each a nine-bit index widened by qh bit j.

                        ReadOnlySpan<byte> entry = GetIq3SGrid().Slice(gi * 4, 4); // one four-byte magnitude row in element order.
                        float v = entry[b & 3] * db; // magnitudes widen unsigned and carry no sign of their own ...
                        output[c] = ((signRow >> b) & 1) != 0 ? -v : v; // ... the raw plane applies each value's sign, bit i for element i.
                    }
                }
            }
        }
    }

    private static string BuildGemvIQ3SPtx() =>
        BuildQ8KGemvPtx(
            entryName: "gemv_iq3s",
            blockBytes: IQ3SBlockBytes,
            subBlockArms: [.. Enumerable.Range(0, 8).Select(PtxIQ3SSubBlock)],
            finalFactor: null); // the reference applies no final factor here: sumf lands as-is.

    // One unrolled sub-block arm: hold this qh byte in lsA while every group consumes its extension bit (lsA doubles as the scale
    // destination only after that), fold each raw sign row straight from the plane into sg, and walk the two grid bytes owning this
    // group's words - each widened with its own qh bit against a four-byte entry. The tail reads the scale nibble once nothing else
    // needs lsA anymore.
    private static string PtxIQ3SSubBlock(int ib)
    {
        List<string> lines = [.. PtxOpenSubBlock()];

        lines.AddRange(PtxLoadBlockByte(66 + ib, "lsA")); // qh: one extension bit per index byte lives in this single byte ...

        for (int l = 0; l < 4; l++)
        {
            string into = l < 2 ? "loU" : "hiU"; // groups 0..1 own elements 0..15 and groups 2..3 own 16..31, as the reference halves split them.

            lines.AddRange(PtxLoadBlockByte(74 + 4 * ib + l, "t1")); // this group's raw sign byte rides straight in the plane ...
            lines.Add($"    shl.b32 t1, t1, {8 * l};"); // ... folded into sg at its byte position with no table lookup ...
            lines.Add("    or.b32 sg, sg, t1;");

            for (int half = 0; half < 2; half++)
            {
                int j = 2 * l + half; // the two activation words this group owns inside the sub-block.
                lines.AddRange(PtxLoadBlockByte(2 + 8 * ib + j, "gi")); // ... each fed by its own consecutive grid byte ...

                lines.AddRange(PtxExtendIndexQhBit("lsA", j)); // ... widened with qh bit j at position eight against a four-byte entry.

                lines.AddRange(PtxLoadActWord(32 * ib + 4 * j)); // this word's four quants, contiguous in the window ...
                lines.AddRange(PtxLoadEntryDword("gi", 2, 0)); // ... read as one whole magnitude row.
                lines.AddRange(PtxWordDot(j, into)); // exact folded mad chain over the word.
            }
        }

        lines.AddRange(PtxLsFromScaleByte("lsA", 106 + (ib >> 1), highNibble: (ib & 1) != 0)); // ls = 1 + 2 * code covers the whole sub-block ...
        lines.AddRange(PtxBsumWhole("lsA")); // bsum += (loSum + hiSum) * ls: one exact integer step per sub-block.

        return string.Join("\n", [.. lines]);
    }
}
