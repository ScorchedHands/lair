namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // Q5_0: each 22-byte block covers 32 elements as a half-float scale d at bytes 0..1, the little-endian 32-bit
    // fifth-bit field qh at bytes 2..5 (element l keeps bit l of that word), and 16 nibble bytes at bytes 6..21 — element
    // l < 16 is the low nibble of qs[l], element 16 + l the high nibble of the same byte. w = (five - 16) * d with five in
    // [0, 31], centered and scaled in that order exactly as the host scalar reference associates it before the activation
    // multiplies in. The fifth-bit field is assembled from four scalar byte loads because a packed row is not
    // dword-aligned and this seam answers only computed addresses; its per-element bit comes out of a register shift by
    // the block-relative element index, which the host chain expresses as (qh >> l) & 1.
    private const int Q5_0BlockBytes = 22;
    private const int Q5_0BlockElements = 32;

    // Q5_0 row expansion: host-side scalar reference math per block, reading the same packed bytes the kernel consumes.
    private static void DequantQ5_0Row(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Q5_0BlockElements)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice((baseIdx / Q5_0BlockElements) * Q5_0BlockBytes, Q5_0BlockBytes);
            float d = ReadHalfLE(block[..2]);
            uint qh = (uint)(block[2] | (block[3] << 8) | (block[4] << 16) | (block[5] << 24));

            int limit = Math.Min(Q5_0BlockElements, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int l = 0; l < limit; l++)
            {
                int five = ((block[6 + (l & 15)] >> (4 * (l >> 4))) & 0xF) | (((int)(qh >> l) & 1) << 4);
                output[baseIdx + l] = (five - 16) * d;
            }
        }
    }

    // Q5_0 GEMV kernel: one row per thread, packed bytes read directly with scalar byte loads. The block walks two 16-column
    // arms — the low-nibble half of the nibble region and its high-nibble half — each carrying its own group bound plus a
    // per-column check against the real column count, so a partial tail block stops exactly where the host limit
    // min(32, cols - baseIdx) stops it. Every arm reads its 5th bit from the assembled qh word at the same element index.
    private static string BuildGemvQ5_0Ptx() => BuildQuantGemvPtx(
        entryName: "gemv_q5_0",
        blockElements: Q5_0BlockElements,
        blockBytes: Q5_0BlockBytes,
        extraU32Regs: ", qv, hi, el, qhw, b1",
        headerLines: [.. PtxWidenHalfLE(offset: 0, dst: "dv", lbl: "_hd"), .. PtxLoadFieldLE(2, "qhw")],
        groupArms: [PtxQ5_0Group(0), PtxQ5_0Group(1)]);

    // One unrolled nibble arm: window byte 6 + col for both halves (the high half reads the same bytes at bit shift 4),
    // and the element's own qh bit selected by shifting the assembled word right by start + col.
    private static string PtxQ5_0Group(int j)
    {
        int groupStart = 16 * j;
        string nextArm = j == 1 ? "_nextBlock" : $"_g{j + 1}";

        List<string> lines = new()
        {
            $"_g{j}:",
            $"    add.u32 t0, base, {groupStart};", // group start: a tail block whose real columns end before it stops here
            "    setp.ge.u32 p, t0, cols_;",
            $"    @p bra _nextBlock;",
            "    mov.u32 col, 0;",
            $"_c{j}:",
            "    setp.ge.u32 p, col, 16;", // the group's own bound ...
            $"    @p bra {nextArm};",
            $"    add.u32 cabs, base, {groupStart};", // ... plus the real column count
            "    add.u32 cabs, cabs, col;",
            "    setp.ge.u32 p, cabs, cols_;",
            $"    @p bra {nextArm};",
        };

        if (j == 1)
        {
            lines.AddRange(
            [
                "    add.u32 t0, col, 6;", // same nibble window as the low arm, read at bit shift 4
                "    cvt.u64.u32 addr, t0;",
                "    add.u64 addr, addr, blockAddr;",
                "    ld.global.u8 qv, [addr];",
                "    shr.u32 qv, qv, 4;",
            ]);
        }
        else
        {
            lines.AddRange(
            [
                "    add.u32 t0, col, 6;", // low nibble of the window byte for elements 0..15
                "    cvt.u64.u32 addr, t0;",
                "    add.u64 addr, addr, blockAddr;",
                "    ld.global.u8 qv, [addr];",
            ]);
        }

        lines.AddRange(
        [
            "    and.b32 qv, qv, 15;", // four low bits of the five-bit code
            $"    add.u32 el, col, {groupStart};", // block-relative element index whose qh bit supplies the fifth bit
            "    shr.u32 el, qhw, el;",
            "    and.b32 hi, el, 1;",
            "    shl.b32 hi, hi, 4;",
            "    or.b32 qv, qv, hi;", // five-bit code in [0, 31]
            "    cvt.rn.f32.u32 wv, qv;",
            "    sub.f32 wv, wv, 0f41800000;", // five - 16: exact in f32 either way
            "    mul.f32 wv, wv, dv;", // (five - 16) * d: the host scalar association before the activation multiplies in
            "    mad.lo.u32 t0, bb, cols_, cabs;",
            "    cvt.u64.u32 addr, t0;",
            "    shl.b64 addr, addr, 2;",
            "    add.u64 addr, xin, addr;",
            "    ld.global.f32 xv, [addr];",
            "    mul.f32 wv, xv, wv;",
            "    add.f32 acc, acc, wv;",
            "    add.u32 col, col, 1;",
            $"    bra _c{j};",
        ]);

        return string.Join("\n", [.. lines]);
    }
}
