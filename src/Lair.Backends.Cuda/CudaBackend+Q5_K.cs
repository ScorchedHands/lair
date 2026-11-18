namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // Q5_K: each 176-byte block covers 256 elements in eight 32-element groups j (0..7) with the Q4_K scale layout at bytes
    // 0..15 — half-floats d and min first, then twelve packed six-bit group scales/mins. Group j uses the low nibbles of
    // qs[48 + 32*(j/2) .. +31] when j is even and the high nibbles of those same bytes when odd; a single fixed 32-byte
    // plane at bytes 16..47 supplies every group's fifth bit, so element l reads byte l of that one plane at bit j.
    // w = d * sc_j * q - dmin * m_j with q in [0, 31], fused and subtracted in the same order the host scalar reference uses
    // before the activation multiplies in. Scale/min unpacking mirrors the kernel's existing Q4_K arms byte for byte: two
    // half-floats at block+0..3, twelve packed six-bit codes at block+4..15 (groups 4-7 split their high bits across two
    // header words), and nibbles from block+48.
    private const int Q5KBlockBytes = 176;
    private const int Q5KBlockElements = 256;

    // Q5_K row expansion: host-side scalar reference math per block and group, reading the same packed bytes the kernel consumes.
    private static void DequantQ5KRow(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Q5KBlockElements)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice((baseIdx / Q5KBlockElements) * Q5KBlockBytes, Q5KBlockBytes);
            float d = ReadHalfLE(block[..2]);
            float min = ReadHalfLE(block[2..4]);

            for (int j = 0; j < 8; j++)
            {
                int subStart = baseIdx + j * 32;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                GetScaleMinQ4K(j, block[4..16], out byte sc, out byte m);
                float scale = d * sc;
                float shift = min * m;

                ReadOnlySpan<byte> q = block[(48 + 32 * (j / 2))..]; // group-pair window: bytes [q, q+32) always inside the block
                int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.
                for (int l = 0; l < limit; l++)
                {
                    output[subStart + l] = (((q[l] >> (4 * (j & 1))) & 0xF) | (((block[16 + l] >> j) & 1) << 4)) * scale - shift;
                }
            }
        }
    }

    // Q5_K GEMV kernel: one row per thread, packed bytes read directly with scalar byte loads. The eight groups are fully
    // unrolled straight-line code so every group carries its own explicit 32-column bound (the host limit is
    // min(32, cols - subStart); a shared column loop bounded only by the real column count ran past the group and read
    // across later windows) and uses immediate byte offsets for the scale, min, nibble window, and fifth-bit plane loads.
    // Both half-float headers widen through integer bit math into a per-thread local slot because cvt.f16 forms are rejected
    // on this seam. Accumulation is plain mul then add in ascending column order, matching the host scalar reference chain.
    private static string BuildGemvQ5KPtx()
    {
        List<string> arms = [];
        for (int j = 0; j < 8; j++)
        {
            arms.Add(PtxQ5KGroup(j));
        }

        return BuildQuantGemvPtx(
            entryName: "gemv_q5k",
            blockElements: Q5KBlockElements,
            blockBytes: Q5KBlockBytes,
            extraU32Regs: ", sc, mn, hi, qv",
            headerLines: [.. PtxWidenHalfLE(offset: 0, dst: "dv", lbl: "_hd"), .. PtxWidenHalfLE(offset: 2, dst: "mv", lbl: "_hm")],
            groupArms: arms);
    }

    // One unrolled group arm: bound the group start and every column against the real column count, load this group's six-bit
    // scale/min codes with immediate header offsets (groups 4-7 reassemble their split high bits), fuse them with the block
    // headers d and min, then walk the group reading its nibble window plus byte l of the one fixed fifth-bit plane.
    private static string PtxQ5KGroup(int j)
    {
        int groupStart = 32 * j;
        int windowBase = 48 + 32 * (j / 2); // group-pair window inside this block
        string nextArm = j == 7 ? "_nextBlock" : $"_g{j + 1}";

        List<string> lines = new()
        {
            $"_g{j}:",
            $"    add.u32 t0, base, {groupStart};", // group start: a tail block whose real columns end before it stops here
            "    setp.ge.u32 p, t0, cols_;",
            "    @p bra _nextBlock;",
        };

        if (j < 4)
        {
            lines.AddRange(
            [
                $"    mov.u32 t0, {4 + j};", // this group's six-bit scale code
                "    cvt.u64.u32 addr, t0;",
                "    add.u64 addr, addr, blockAddr;",
                "    ld.global.u8 sc, [addr];",
                "    and.b32 sc, sc, 63;",
                $"    mov.u32 t0, {8 + j};", // this group's six-bit min code
                "    cvt.u64.u32 addr, t0;",
                "    add.u64 addr, addr, blockAddr;",
                "    ld.global.u8 mn, [addr];",
                "    and.b32 mn, mn, 63;",
            ]);
        }
        else
        {
            lines.AddRange(
            [
                $"    mov.u32 t0, {8 + j};", // one word carries both low nibbles for this pair of groups
                "    cvt.u64.u32 addr, t0;",
                "    add.u64 addr, addr, blockAddr;",
                "    ld.global.u8 sc, [addr];",
                "    and.b32 sc, sc, 15;",
                "    ld.global.u8 mn, [addr];", // the same word's high nibble is this group's min low bits
                "    shr.u32 mn, mn, 4;",
                "    and.b32 mn, mn, 15;",
                $"    mov.u32 t0, {j};", // split high bits of the scale code
                "    cvt.u64.u32 addr, t0;",
                "    add.u64 addr, addr, blockAddr;",
                "    ld.global.u8 hi, [addr];",
                "    shr.u32 hi, hi, 6;",
                "    and.b32 hi, hi, 3;",
                "    shl.b32 hi, hi, 4;",
                "    or.b32 sc, sc, hi;",
                $"    mov.u32 t0, {4 + j};", // split high bits of the min code
                "    cvt.u64.u32 addr, t0;",
                "    add.u64 addr, addr, blockAddr;",
                "    ld.global.u8 hi, [addr];",
                "    shr.u32 hi, hi, 6;",
                "    and.b32 hi, hi, 3;",
                "    shl.b32 hi, hi, 4;",
                "    or.b32 mn, mn, hi;",
            ]);
        }

        lines.AddRange(
        [
            "    cvt.rn.f32.u32 scl_, sc;", // d * sc and min * m: the same fused pair the host chain carries per group
            "    mul.f32 scl_, scl_, dv;",
            "    cvt.rn.f32.u32 sft_, mn;",
            "    mul.f32 sft_, sft_, mv;",
            "    mov.u32 col, 0;",
            $"_c{j}:",
            "    setp.ge.u32 p, col, 32;", // the group's own bound ...
            $"    @p bra {nextArm};",
            $"    add.u32 cabs, base, {groupStart};", // ... plus the real column count
            "    add.u32 cabs, cabs, col;",
            "    setp.ge.u32 p, cabs, cols_;",
            $"    @p bra {nextArm};",
            $"    add.u32 t0, col, {windowBase};", // nibble window byte for this column
            "    cvt.u64.u32 addr, t0;",
            "    add.u64 addr, addr, blockAddr;",
            "    ld.global.u8 qv, [addr];",
        ]);

        if ((j & 1) != 0)
        {
            lines.Add("    shr.u32 qv, qv, 4;"); // odd groups read the high nibble of the shared pair window
        }

        lines.AddRange(
        [
            "    and.b32 qv, qv, 15;", // four low bits of the five-bit code
            "    add.u32 t0, col, 16;", // byte l of the one fixed fifth-bit plane at bytes 16..47 ...
            "    cvt.u64.u32 addr, t0;",
            "    add.u64 addr, addr, blockAddr;",
            "    ld.global.u8 hi, [addr];", // ... whose bit j is this group's fifth quant bit
            $"    shr.u32 hi, hi, {j};",
            "    and.b32 hi, hi, 1;",
            "    shl.b32 hi, hi, 4;",
            "    or.b32 qv, qv, hi;", // five-bit code in [0, 31]
            "    cvt.rn.f32.u32 wv, qv;",
            "    mul.f32 wv, wv, scl_;", // q * (d * sc) ...
            "    sub.f32 wv, wv, sft_;", // ... minus min * m: the host scalar association before the activation multiplies in
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
