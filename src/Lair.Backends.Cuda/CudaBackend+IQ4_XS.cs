using System.Linq;

namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // IQ4_XS: each 136-byte block covers 256 elements in eight 32-element sub-blocks ib. Half-float d sits at bytes 0..1, the sixteen-bit
    // scale-high field at 2..3 and twelve packed low-nibble bytes at 4..15; sub-block ib reads its sixteen nibble bytes at block + 8 + 16*ib
    // with IQ4_NL's own low/high element order. Its six-bit code unpacks as the reference does - low nibble in scales_l[ib / 2] at bit shift
    // 4 * (ib % 2), two high bits from scales_h at shift 2 * ib - and reads back as scale = d * (code - 32). The kernel mirrors RowDotIQ4XScalar
    // step for step: sub-block order outer, nibble halves inner, each element looked up in the shared scratch table with the same fused chain.
    private const int IQ4XsBlockBytes = 136;

    // Unpack the six-bit sub-block scale exactly as GetIQ4XScale on the oracle side does it: low nibble from its packed byte, high bits from
    // the shared sixteen-bit field, joined without an intermediate rounding step.
    private static int GetIQ4XScale(int ib, ReadOnlySpan<byte> scalesL, ushort scalesH)
    {
        int low = (scalesL[ib / 2] >> (4 * (ib % 2))) & 0xF;
        int high = (scalesH >> (2 * ib)) & 0x3;
        return low | (high << 4);
    }

    // IQ4_XS row expansion: host-side scalar reference math per sub-block reading the same packed bytes the kernel consumes.
    private static void DequantIQ4XRow(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += 256)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice((baseIdx / 256) * IQ4XsBlockBytes, IQ4XsBlockBytes); // whole blocks: the stride pads to exactly this width.
            float d = ReadHalfLE(block[..2]);
            ushort scalesH = (ushort)(block[2] | (block[3] << 8));

            for (int ib = 0; ib < 8 && ib * 32 < cols - baseIdx; ib++)
            {
                float scale = d * (GetIQ4XScale(ib, block[4..16], scalesH) - 32); // the sub-block's fused scale, in reference association.

                int limit = Math.Min(32, cols - baseIdx - ib * 32); // padded nibbles are zero: stop at the real column count.
                for (int l = 0; l < limit; l++)
                {
                    int code = (block[8 + 16 * ib + (l & 15)] >> (4 * (l >> 4))) & 0xF; // low nibble below sixteen, high nibble from the same byte above it.
                    output[baseIdx + ib * 32 + l] = scale * KValueIQ4(code); // w = scale * kvalue[q], table widened with sign exactly as the reference reads it.
                }
            }
        }
    }

    private static string BuildGemvIQ4XPtx() =>
        BuildQuantGemvPtx(
            entryName: "gemv_iq4xs",
            blockElements: 256, // whole super-blocks: the same ceil(cols / 256) geometry both sides of the seam use.
            blockBytes: IQ4XsBlockBytes,
            extraU32Regs: ", kv, hb, sc, sh, hi", // looked-up table byte, widening temp, nibble holder, and the two scale assembly temps.
            headerLines: [.. PtxWidenHalfLE(offset: 0, dst: "dv", lbl: "_hd"), .. PtxIQ4XHeader()],
            groupArms: IQ4XArmChain(), // sub-block order outer, nibble halves inner - the reference traversal exactly.
            tabLoad: true);

    // Assembles this block's sixteen-bit scale-high field from its two bytes once per super-block walk; every arm reads its own bits out of it.
    private static IEnumerable<string> PtxIQ4XHeader() =>
        [
            "    mov.u32 t0, 2;", // low byte first ...
            "    cvt.u64.u32 addr, t0;",
            "    add.u64 addr, addr, blockAddr;",
            "    ld.global.u8 sh, [addr];",
            "    mov.u32 t0, 3;", // ... and the high byte shifts into place beside it.
            "    cvt.u64.u32 addr, t0;",
            "    add.u64 addr, addr, blockAddr;",
            "    ld.global.u8 hi, [addr];",
            "    shl.b32 hi, hi, 8;",
            "    or.b32 sh, sh, hi;",
        ];

    // Builds the sixteen arms in reference order and chains each one's forward jump to its successor: sub-blocks outer, nibble halves inner,
    // the last arm falling through to the block advance like every other kernel in this family.
    private static IEnumerable<string> IQ4XArmChain()
    {
        List<string> arms = [];

        for (int ib = 0; ib < 8; ib++)
        {
            for (int g = 0; g < 2; g++)
            {
                int flat = ib * 2 + g; // arm order index across the whole block.
                string nextLabel = flat == 15 ? "_nextBlock" : $"_x{flat + 1}";

                arms.Add(PtxIQ4Core(
                    groupStart: 32 * ib + 16 * g, // elements this half owns inside its sub-block window ...
                    nibbleByteBase: 8 + 16 * ib, // ... read from the sixteen bytes this sub-block's nibbles live in.
                    nibbleShift: 4 * g, // low nibbles below element sixteen of the window, high nibbles above it.
                    scaleReg: "scl_", // each sub-block carries its own fused scale; headLines compute it before the walk starts.
                    headLines: PtxIQ4XScaleHead(ib),
                    armLabel: $"_x{flat}",
                    nextLabel: nextLabel));
            }
        }

        return arms;
    }

    // One sub-block's scale fusion, in reference order: unpack the six-bit code from its packed low nibble byte plus this block's shared high
    // field, center it on 32 as a float step exactly like every grouped format does here, then fuse d - giving (d * (code - 32)) before any
    // element multiplies kvalue in. The scale changes once per sub-block and never mid-window, so one head sequence covers all sixteen elements.
    private static IEnumerable<string> PtxIQ4XScaleHead(int ib)
    {
        List<string> lines = new()
        {
            $"    mov.u32 t0, {4 + ib / 2};", // this sub-block's packed low-nibble byte ...
            "    cvt.u64.u32 addr, t0;",
            "    add.u64 addr, addr, blockAddr;",
            "    ld.global.u8 sc, [addr];",

        };

        if ((ib & 1) != 0)
        {
            lines.Add("    shr.u32 sc, sc, 4;"); // odd sub-blocks read their code from this byte's high half ...
        }

        lines.AddRange(
        [
            "    and.b32 sc, sc, 15;", // ... even ones straight from the low half.
            $"    shr.u32 hi, sh, {2 * ib};", // the two high bits ride in this block's shared sixteen-bit field at their own shift ...
            "    and.b32 hi, hi, 3;",
            "    shl.b32 hi, hi, 4;",
            "    or.b32 sc, sc, hi;", // six-bit scale code in [0, 63].
            "    cvt.rn.f32.u32 scl_, sc;", // centering on 32 stays a float step exactly as the grouped formats read it ...
            "    sub.f32 scl_, scl_, 0f42000000;",
            "    mul.f32 scl_, scl_, dv;", // ... and fuses with d once per sub-block, before any element multiplies kvalue in.

        ]);

        return lines;
    }
}
