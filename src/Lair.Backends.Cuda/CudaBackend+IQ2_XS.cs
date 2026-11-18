using System.Linq;

namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // IQ2_XS: each 74-byte block covers 256 elements in eight 32-value sub-blocks ib. Sub-block ib carries four sixteen-bit grid
    // words at bytes 2 + 8*ib (the nine low bits index an entry, the seven high bits select a ksigns row) and one scale byte at
    // block byte 66 + ib whose nibbles give ls1 over groups 0..1 and ls2 over groups 2..3. The kernel mirrors RowDotIQ2XSQ8Scalar
    // step for step: same word windows, same entry pairing at lane offsets zero and four, bsum += lo * ls1 then bsum += hi * ls2 per
    // sub-block, d multiplying the exact block total once, 0.125f landing after the last block.
    private const int IQ2XsBlockBytes = 74;

    // IQ2_XS row expansion: host-side reference math over the same packed bytes - dbLo/dbHi fused per half from d and this byte's
    // two nibbles, magnitudes widened unsigned straight out of their grid entry, signs flipped through ksigns rows.
    private static void DequantIQ2XSRow(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int bi = 0; bi * IQ2XsBlockBytes < rowBytes.Length && bi * 256 < cols; bi++)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice(bi * IQ2XsBlockBytes, IQ2XsBlockBytes); // whole blocks: the stride pads to exactly this width.
            float dHalf = ReadHalfLE(block[..2]);

            for (int ib = 0; ib < 8 && ib * 32 < cols - bi * 256; ib++)
            {
                byte scaleByte = block[66 + ib]; // one byte carries the sub-block's two scale codes.
                float dbLo = dHalf * (0.5f + (scaleByte & 0xF)) * 0.25f; // groups 0..1 read their factor from the low nibble ...
                float dbHi = dHalf * (0.5f + (scaleByte >> 4)) * 0.25f; // ... and groups 2..3 from the high one, each fused with d in reference order.

                for (int l = 0; l < 4; l++)
                {
                    ushort word = (ushort)(block[2 + 8 * ib + 2 * l] | (block[3 + 8 * ib + 2 * l] << 8)); // one grid word carries one whole eight-value group.
                    ReadOnlySpan<byte> entry = GetIq2XSGrid().Slice((word & 511) * 8, 8); // the nine low bits index an eight-byte row in element order.
                    byte signRow = GetKSignsIQ2XS()[word >> 9]; // and its seven high bits select a shared sign-table row.

                    for (int b = 0; b < 8; b++)
                    {
                        int c = bi * 256 + ib * 32 + l * 8 + b; // the element this magnitude byte belongs to.
                        if (c >= cols)
                        {
                            break; // zero-padded tail block: stop at the real column count, as every reference path does.
                        }

                        float v = entry[b] * (l < 2 ? dbLo : dbHi); // magnitudes widen unsigned and carry no sign of their own ...
                        output[c] = ((signRow >> b) & 1) != 0 ? -v : v; // ... the group's ksigns row applies each value's sign, bit i for element i.
                    }
                }
            }
        }
    }

    private static string BuildGemvIQ2XSPtx() =>
        BuildQ8KGemvPtx(
            entryName: "gemv_iq2xs",
            blockBytes: IQ2XsBlockBytes,
            subBlockArms: [.. Enumerable.Range(0, 8).Select(PtxIQ2XSSubBlock)],
            finalFactor: "0f3E000000"); // 0.125f lands once at the end, exactly as `return 0.125f * sumf` reads it.

    // One unrolled sub-block arm: read both scale nibbles into lsA/lsB up front (nothing live across dots depends on them), then
    // walk the four groups - assemble each grid word from its two bytes, split off the seven-bit sign code before masking the index
    // to nine bits, and dot this group's activation words against the entry's halves. The tail applies each half under its own factor.
    private static string PtxIQ2XSSubBlock(int ib)
    {
        List<string> lines = [.. PtxOpenSubBlock()];

        lines.AddRange(PtxLsFromScaleByte("lsA", 66 + ib, highNibble: false)); // ls1 = 2 * (scale & 0xF) + 1 covers groups 0..1 ...
        lines.AddRange(PtxLsFromScaleByte("lsB", 66 + ib, highNibble: true)); // ... and ls2 = 2 * (scale >> 4) + 1 covers groups 2..3.

        for (int l = 0; l < 4; l++)
        {
            string into = l < 2 ? "loU" : "hiU"; // groups 0..1 own elements 0..15 and groups 2..3 own 16..31, as the reference halves split them.

            lines.AddRange(PtxAssembleFieldLE16("gi", 2 + 8 * ib + 2 * l)); // one sixteen-bit grid word per group ...
            lines.Add($"    shr.u32 mvv, gi, 9;"); // ... its seven high bits select a ksigns row before the index mask eats them ...
            lines.Add("    and.b32 mvv, mvv, 127;");
            lines.AddRange(PtxLoadKSignsByte("mvv", "t1"));
            lines.Add($"    shl.b32 t1, t1, {8 * l};"); // ... folded into sg at this group's byte position ...
            lines.Add("    or.b32 sg, sg, t1;");
            lines.Add("    and.b32 gi, gi, 511;"); // ... the nine low bits index an entry of the 512-row table.

            for (int half = 0; half < 2; half++)
            {
                int j = 2 * l + half; // the two activation words this group owns inside the sub-block.
                lines.AddRange(PtxLoadActWord(32 * ib + 4 * j)); // its four quants, contiguous in the window ...
                lines.AddRange(PtxLoadEntryDword("gi", 3, half == 0 ? 0 : 4)); // ... against this entry's low or high magnitude half.
                lines.AddRange(PtxWordDot(j, into)); // exact folded mad chain over the word.
            }
        }

        lines.AddRange(PtxBsumSplit()); // bsum += lo * ls1 then bsum += hi * ls2: two exact integer steps per sub-block.

        return string.Join("\n", [.. lines]);
    }
}
