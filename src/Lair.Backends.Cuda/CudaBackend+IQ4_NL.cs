using System.Linq;

namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // IQ4_NL: each 18-byte block covers 32 elements - half-float d at bytes 0..1 and sixteen nibble bytes at 2..17. Element j < 16 is
    // the low nibble of qs[j] and element 16+j the high nibble of the same byte; every nibble indexes the shared IQ4 value table, which
    // lives in device scratch so one lookup lands per element with no staging anywhere. The kernel mirrors RowDotIQ4NScalar step for
    // step: ascending elements across both nibble halves under their own bounds, w = d * kvalue fused before the activation multiplies in.
    private const int IQ4NlBlockBytes = 18;

    // IQ4_NL row expansion: host-side scalar reference math per block reading the same packed bytes - each nibble looked up in the
    // mirrored value table and scaled by d, exactly as the oracle's dequant path expands it.
    private static void DequantIQ4NlRow(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += 32)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice((baseIdx / 32) * IQ4NlBlockBytes, IQ4NlBlockBytes); // whole blocks: the stride pads to exactly this width.
            float d = ReadHalfLE(block[..2]);

            int limit = Math.Min(32, cols - baseIdx); // padded nibbles are zero: stop at the real column count.
            for (int l = 0; l < limit; l++)
            {
                int code = (block[2 + (l & 15)] >> (4 * (l >> 4))) & 0xF; // low nibble below sixteen, high nibble from the same byte above it.
                output[baseIdx + l] = d * KValueIQ4(code); // w = d * kvalue[q], table widened with sign exactly as the reference reads it.
            }
        }
    }

    private static string BuildGemvIQ4NPtx() =>
        BuildQuantGemvPtx(
            entryName: "gemv_iq4nl",
            blockElements: 32,
            blockBytes: IQ4NlBlockBytes,
            extraU32Regs: ", kv, hb, sc", // looked-up table byte, widening temp, and the nibble holder every arm reads through.
            headerLines: PtxWidenHalfLE(offset: 0, dst: "dv", lbl: "_hd"),
            groupArms: [.. Enumerable.Range(0, 2).Select(PtxIQ4NGroup)],
            tabLoad: true); // the shared value table rides in scratch beside this kernel's other device operands.

    private static string PtxIQ4NGroup(int g) =>
        PtxIQ4Core(
            groupStart: 16 * g, // elements this half owns inside its block window ...
            nibbleByteBase: 2, // ... read from the block's single sixteen-byte nibble region at constant byte offsets.
            nibbleShift: 4 * g, // low nibbles below element sixteen, high nibbles above it - same bytes either way.
            scaleReg: "dv", // IQ4_NL scales by its half-float header alone ...
            headLines: [], // ... with no per-arm setup beyond the widened header itself.
            armLabel: g == 0 ? "_iq4n0" : "_iq4n1",
            nextLabel: g == 0 ? "_iq4n1" : "_nextBlock");

    // Shared IQ4 nibble-half walk for both IQ4 layouts: each element loads its nibble byte at a constant window offset, extracts the
    // half this arm owns, looks that code up in the scratch table widened through the same two's-complement integer trick the Q6_K group
    // scales already use on this seam, fuses (scale * kvalue) exactly as the reference associates it per element - d alone for IQ4_NL, a
    // sub-block scale fused with d by headLines in arm order for IQ4_XS - and multiplies the activation in ascending column order. The
    // group's own bound plus the real column count guard every element exactly like the host limit does; padded nibbles are zero codes.
    private static string PtxIQ4Core(
        int groupStart,
        int nibbleByteBase,
        int nibbleShift,
        string scaleReg,
        IEnumerable<string> headLines,
        string armLabel,
        string nextLabel)
    {
        ArgumentNullException.ThrowIfNull(headLines);

        List<string> lines = new()
        {
            $"{armLabel}:",
            $"    add.u32 t0, base, {groupStart};", // group start: a tail block whose real columns end before it stops here ...
            "    setp.ge.u32 p, t0, cols_;",
            $"    @p bra {nextLabel};",
        };

        lines.AddRange(headLines); // per-arm scale fusion lands before the element walk, as each layout reads it.

        lines.Add("    mov.u32 col, 0;");
        lines.Add($"_lc_{armLabel}:");

        lines.AddRange(
        [
            "    setp.ge.u32 p, col, 16;", // the group's own bound ...
            $"    @p bra {nextLabel};",
            $"    add.u32 cabs, base, {groupStart};", // ... plus the real column count.
            "    add.u32 cabs, cabs, col;",
            "    setp.ge.u32 p, cabs, cols_;",
            $"    @p bra {nextLabel};",

            $"    add.u32 t0, col, {nibbleByteBase};", // this element's nibble byte at its constant window offset ...
            "    cvt.u64.u32 addr, t0;",
            "    add.u64 addr, addr, blockAddr;",
            "    ld.global.u8 sc, [addr];",

            (nibbleShift == 0 ? null! : $"    shr.u32 sc, sc, {nibbleShift};"), // hmm: conditional emission handled below instead.

        ]);

        lines.RemoveAt(lines.Count - 1);

        if (nibbleShift != 0)
        {
            lines.Add($"    shr.u32 sc, sc, {nibbleShift};"); // the high half shifts its nibble down before masking ...
        }

        lines.AddRange(
        [
            "    and.b32 sc, sc, 15;", // ... either way four bits carry one table code.
            "    cvt.u64.u32 addr, sc;", // the scratch value table answers this nibble directly: no staging anywhere on either side.
            "    add.u64 addr, tab, addr;",
            "    ld.global.u8 kv, [addr];",

            .. PtxWidenInt8("kv", "mv"), // widen with sign: negative table entries keep their value through the two's-complement trick.

            $"    mul.f32 mv, {scaleReg}, mv;", // (d * kvalue) or (sub-block scale * d *? see headLines): fused in reference order ...
            "    mad.lo.u32 t0, bb, cols_, cabs;", // this element's activation slot: batch-major like every kernel in the family.
            "    cvt.u64.u32 addr, t0;",
            "    shl.b64 addr, addr, 2;",
            "    add.u64 addr, xin, addr;",
            "    ld.global.f32 xv, [addr];",
            "    mul.f32 wv, xv, mv;", // x * (scale * kvalue): the reference association, then a plain ascending accumulation.
            "    add.f32 acc, acc, wv;",
            "    add.u32 col, col, 1;",
            $"    bra _lc_{armLabel};",
        ]);

        return string.Join("\n", [.. lines]);
    }
}
