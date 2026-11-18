using System.Linq;

namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // IQ2_S: each 82-byte block covers 256 elements in eight 32-value sub-blocks ib. Sub-block ib carries four grid index bytes at
    // bytes 2 + 4*ib, whose ten-bit-capable indices take bits eight and nine from the qh byte at block byte 66 + ib (pair per group),
    // raw sign bytes straight at block byte 34 + 4*ib inside the region the reference reaches through qs + QK_K/8 - a sign plane rather
    // than ksigns lookups - and one scale byte at block byte 74 + ib whose nibbles give ls1 over groups 0..1 and ls2 over groups 2..3.
    // The kernel mirrors RowDotIQ2SQ8Scalar step for step: same index/qh pairing, raw sign rows folded the same way, bsum += lo * ls1
    // then bsum += hi * ls2 per sub-block, d multiplying the exact block total once, 0.125f landing after the last block.
    private const int IQ2SBlockBytes = 82;

    // IQ2_S row expansion: host-side reference math over the same packed bytes - dbLo/dbHi fused per half from d and this byte's two
    // nibbles, magnitudes widened unsigned straight out of their grid entry, signs applied through the raw sign plane.
    private static void DequantIQ2SRow(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int bi = 0; bi * IQ2SBlockBytes < rowBytes.Length && bi * 256 < cols; bi++)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice(bi * IQ2SBlockBytes, IQ2SBlockBytes); // whole blocks: the stride pads to exactly this width.
            float dHalf = ReadHalfLE(block[..2]);

            for (int ib = 0; ib < 8 && ib * 32 < cols - bi * 256; ib++)
            {
                byte qh = block[66 + ib]; // one byte carries the two extension bits of every group's index.
                byte scaleByte = block[74 + ib]; // and another carries this sub-block's two scale codes.
                float dbLo = dHalf * (0.5f + (scaleByte & 0xF)) * 0.25f; // groups 0..1 read their factor from the low nibble ...
                float dbHi = dHalf * (0.5f + (scaleByte >> 4)) * 0.25f; // ... and groups 2..3 from the high one, each fused with d in reference order.

                for (int l = 0; l < 4; l++)
                {
                    int gi = block[2 + 4 * ib + l] | (((qh >> (2 * l)) & 1) << 8) | (((qh >> (2 * l + 1)) & 1) << 9); // nine-bit grid index.
                    ReadOnlySpan<byte> entry = GetIq2SGrid().Slice(gi * 8, 8); // one whole eight-value row of magnitudes in element order.
                    byte signRow = block[34 + 4 * ib + l]; // the group's raw sign byte: no table lookup anywhere on this path.

                    for (int b = 0; b < 8; b++)
                    {
                        int c = bi * 256 + ib * 32 + l * 8 + b; // the element this magnitude byte belongs to.
                        if (c >= cols)
                        {
                            break; // zero-padded tail block: stop at the real column count, as every reference path does.
                        }

                        float v = entry[b] * (l < 2 ? dbLo : dbHi); // magnitudes widen unsigned and carry no sign of their own ...
                        output[c] = ((signRow >> b) & 1) != 0 ? -v : v; // ... the raw plane applies each value's sign, bit i for element i.
                    }
                }
            }
        }
    }

    private static string BuildGemvIQ2SPtx() =>
        BuildQ8KGemvPtx(
            entryName: "gemv_iq2s",
            blockBytes: IQ2SBlockBytes,
            subBlockArms: [.. Enumerable.Range(0, 8).Select(PtxIQ2SSubBlock)],
            finalFactor: "0f3E000000"); // 0.125f lands once at the end, exactly as `return 0.125f * sumf` reads it.

    // One unrolled sub-block arm: hold this sub-block's qh byte in lsA while both scale nibbles wait for the tail (lsA doubles as a
    // scale destination only after every group has consumed its extension bits), then walk the four groups - load each index byte,
    // OR in this group's pair of qh bits at positions eight and nine, fold the raw sign row straight from the plane into sg, and dot
    // this group's activation words against the entry's halves. The tail applies each half under its own factor.
    private static string PtxIQ2SSubBlock(int ib)
    {
        List<string> lines = [.. PtxOpenSubBlock()];

        lines.AddRange(PtxLoadBlockByte(66 + ib, "lsA")); // qh: the two extension bits of every group live in this one byte ...

        for (int l = 0; l < 4; l++)
        {
            string into = l < 2 ? "loU" : "hiU"; // groups 0..1 own elements 0..15 and groups 2..3 own 16..31, as the reference halves split them.

            lines.AddRange(PtxLoadBlockByte(2 + 4 * ib + l, "gi")); // this group's index byte ...
            lines.AddRange(PtxExtendIndexQhPair("lsA", l)); // ... widened with bits 2l and 2l+1 of qh at positions eight and nine.

            lines.AddRange(PtxLoadBlockByte(34 + 4 * ib + l, "t1")); // the group's raw sign byte rides straight in the plane ...
            lines.Add($"    shl.b32 t1, t1, {8 * l};"); // ... folded into sg at this group's byte position with no table lookup ...
            lines.Add("    or.b32 sg, sg, t1;");

            for (int half = 0; half < 2; half++)
            {
                int j = 2 * l + half; // the two activation words this group owns inside the sub-block.
                lines.AddRange(PtxLoadActWord(32 * ib + 4 * j)); // its four quants, contiguous in the window ...
                lines.AddRange(PtxLoadEntryDword("gi", 3, half == 0 ? 0 : 4)); // ... against this entry's low or high magnitude half.
                lines.AddRange(PtxWordDot(j, into)); // exact folded mad chain over the word.
            }
        }

        lines.AddRange(PtxLsFromScaleByte("lsA", 74 + ib, highNibble: false)); // ls1 = 1 + 2 * (scale & 0xF) covers groups 0..1 ...
        lines.AddRange(PtxLsFromScaleByte("lsB", 74 + ib, highNibble: true)); // ... and ls2 = 1 + 2 * (scale >> 4) covers groups 2..3.
        lines.AddRange(PtxBsumSplit()); // bsum += lo * ls1 then bsum += hi * ls2: two exact integer steps per sub-block.

        return string.Join("\n", [.. lines]);
    }
}
