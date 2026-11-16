namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // Q6_K: each 210-byte block covers 256 elements in sixteen 16-element groups g (0..15). Group g lives in half
    // h = g/8 of the super-block, quarter k/2 of that half, at ql window 64h + 16*(k&3): low nibbles for quarters
    // 0..1, high nibbles for 2..3; qh byte 128 + 32h + 16*(k&1) + e carries the high 2 bits at bit shift 2*quarter.
    // w = d * sc_g * (q - 32) with q in [0, 63] and sc_g the signed int8 scale at byte 192 + g; half-float d sits at
    // bytes 208..209. The kernel below mirrors the scalar reference math column for column: six-bit unpack, centering,
    // fused group scale, then a plain mul by the activation and add into acc in ascending column order, so one device
    // result matches one host dot bit for bit.
    private const int Q6KBlockBytes = 210;
    private const int Q6KBlockElements = 256;

    // Q6_K row expansion: host-side scalar reference math per block and group, reading the same packed bytes the kernel consumes.
    private static void DequantQ6KRow(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Q6KBlockElements)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice((baseIdx / Q6KBlockElements) * Q6KBlockBytes, Q6KBlockBytes);
            float d = ReadHalfLE(block[208..]);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                int quarter = k >> 1; // 0..3: which 32-element quarter of the half the group starts in

                ReadOnlySpan<byte> q = block.Slice(64 * h + 16 * (k & 3), 16); // ql window: always inside the block
                ReadOnlySpan<byte> qh = block.Slice(128 + 32 * h + 16 * (k & 1), 16); // 2-bit fields for the same elements
                float scale = d * (sbyte)block[192 + g];

                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.
                for (int e = 0; e < limit; e++)
                {
                    int six = ((q[e] >> (4 * (quarter >> 1))) & 0xF) | (((qh[e] >> (2 * quarter)) & 3) << 4);
                    output[subStart + e] = (six - 32) * scale;
                }
            }
        }
    }

    // Q6_K GEMV kernel: one row per thread, packed bytes read directly with scalar byte loads. The sixteen groups are
    // fully unrolled straight-line code so every group carries its own explicit column bound (the host limit is
    // min(16, cols - subStart); a shared column loop bounded only by the real column count ran past the group and read
    // across later windows) and uses immediate byte offsets for the ql window, qh field, and scale loads instead of
    // dynamic group-loop address arithmetic. The signed int8 group scale widens through u32 bit math — subtracting 256
    // when its high bit is set — because every value that produces stays exactly representable in f32; the half-float
    // header widens through integer bit math into a per-thread local slot reloaded as f32 because cvt.f16 forms are
    // rejected on this seam. Accumulation is plain mul then add in ascending column order, matching the host scalar
    // reference chain exactly. The group arms below are emitted by PtxQ6KGroup so each arm's window offsets stay a pure
    // function of its group index.
    private const string GemvQ6KPtxHeader =
        ".visible .entry gemv_q6k(\n" +
        "    .param .u64 w_ptr,\n" +
        "    .param .u64 x_ptr,\n" +
        "    .param .u64 out_ptr,\n" +
        "    .param .u32 rows_arg,\n" +
        "    .param .u32 cols_arg,\n" +
        "    .param .u32 count_arg\n" +
        ") {\n" +
        "    .reg .u64 w, xin, o, blockAddr, addr;\n" +
        "    .reg .u32 row, bb, rows_, cols_, cnt_, tid, t0, strideBlocks, strideBytes, rowBase, base, col, cabs, scb, hb, qv, hi;\n" +
        "    .reg .u32 bits, payload, exp, mant, sign, fbits;\n" +
        "    .reg .b16 hw;\n" +
        "    .reg .pred p;\n" +
        "    .reg .f32 acc, dv, sfu, hf, scl_, wv, xv, sub_;\n" +
        "    .local .align 4 .b8 slot[4];\n" +
        "\n" +
        "    ld.param.u64 w, [w_ptr];\n" +
        "    ld.param.u64 xin, [x_ptr];\n" +
        "    ld.param.u64 o, [out_ptr];\n" +
        "    ld.param.u32 rows_, [rows_arg];\n" +
        "    ld.param.u32 cols_, [cols_arg];\n" +
        "    ld.param.u32 cnt_, [count_arg];\n" +
        "\n" +
        "    mov.u32 tid, %ctaid.x;\n" +
        "    mov.u32 t0, %ntid.x;\n" +
        "    mul.lo.u32 tid, tid, t0;\n" +
        "    mov.u32 t0, %tid.x;\n" +
        "    add.u32 row, tid, t0;\n" +
        "    setp.ge.u32 p, row, rows_;\n" +
        "    @p bra done;\n" +
        "\n" +
        "    add.u32 strideBlocks, cols_, 255;\n" +
        "    shr.u32 strideBlocks, strideBlocks, 8;\n" +
        "    mul.lo.u32 strideBytes, strideBlocks, 210;\n" +
        "    mul.lo.u32 rowBase, row, strideBytes;\n" +
        "\n" +
        "    mov.u32 bb, 0;\n" +
        "batch:\n" +
        "    setp.ge.u32 p, bb, cnt_;\n" +
        "    @p bra done;\n" +
        "    mov.f32 acc, 0f00000000;\n" +
        "    cvt.u64.u32 blockAddr, rowBase;\n" +
        "    add.u64 blockAddr, w, blockAddr;\n" +
        "\n" +
        "    mov.u32 base, 0;\n" +
        "_block:\n" +
        "    setp.ge.u32 p, base, cols_;\n" +
        "    @p bra _store;\n" +
        "\n" +
        // Half-float d at block+208 widened through integer bit math into the per-thread local slot. The sign stays
        // unshifted until each arm needs it, because the subnormal arm tests the half's own sign-bit position while the
        // normal arm folds it into the f32 high half.
        "    mov.u32 t0, 208;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.b16 hw, [addr];\n" +
        "    mov.b32 bits, {hw, 0};\n" +
        "    and.b32 sign, bits, 32768;\n" +
        "    and.b32 payload, bits, 32767;\n" +
        "    shr.u32 exp, payload, 10;\n" +
        "    and.b32 mant, payload, 1023;\n" +
        "    setp.le.u32 p, exp, 0;\n" +
        "    @!p bra _d_norm;\n" +
        "_d_sub:\n" +
        "    cvt.rn.f32.u32 sub_, mant;\n" +
        "    mul.f32 dv, sub_, 0f33800000;\n" +
        "    setp.eq.u32 p, sign, 32768;\n" +
        "    @!p bra _d_done;\n" +
        "    mul.f32 dv, dv, 0fBF800000;\n" + // -1.0f: the subnormal magnitude carries its sign through a real negation
        "    bra _d_done;\n" +
        "_d_norm:\n" +
        "    shl.b32 sign, sign, 16;\n" +
        "    add.u32 fbits, exp, 112;\n" +
        "    shl.b32 fbits, fbits, 23;\n" +
        "    shl.b32 mant, mant, 13;\n" +
        "    or.b32 fbits, fbits, mant;\n" +
        "    xor.b32 fbits, fbits, sign;\n" +
        "    st.local.u32 [slot], fbits;\n" +
        "    ld.local.f32 dv, [slot];\n" +
        "_d_done:\n";

    // One unrolled group arm: bound the group start against the real column count, unpack this group's signed scale byte
    // (subtracting 256 when its high bit is set keeps every value exactly representable in f32), fuse it with the block
    // header d, then walk sixteen columns re-checking each absolute column bound so a partial tail group stops exactly
    // where the host limit min(16, cols - subStart) stops it.
    private static string PtxQ6KGroup(int g)
    {
        int h = g >> 3;
        int k = g & 7;
        int quarter = k >> 1;

        int groupStart = g * 16;
        int scaleOffset = 192 + g;
        int qlWindow = 64 * h + 16 * (k & 3);
        int qhWindow = 128 + 32 * h + 16 * (k & 1);

        string nextGroup = g == 15 ? "_nextBlock" : $"_g{g + 1}";

        List<string> lines = [];

        lines.Add($"_g{g}:");
        lines.Add($"    add.u32 t0, base, {groupStart};");
        lines.Add("    setp.ge.u32 p, t0, cols_;");
        lines.Add("    @p bra _nextBlock;");
        lines.Add($"    mov.u32 t0, {scaleOffset};");
        lines.Add("    cvt.u64.u32 addr, t0;");
        lines.Add("    add.u64 addr, addr, blockAddr;");
        lines.Add("    ld.global.u8 scb, [addr];");
        lines.Add("    and.b32 scb, scb, 255;");
        lines.Add("    shr.u32 hb, scb, 7;");
        lines.Add("    shl.b32 hb, hb, 8;");
        lines.Add("    cvt.rn.f32.u32 sfu, scb;");
        lines.Add("    cvt.rn.f32.u32 hf, hb;");
        lines.Add("    sub.f32 sfu, sfu, hf;");
        lines.Add("    mul.f32 scl_, dv, sfu;");

        lines.Add("    mov.u32 col, 0;");
        lines.Add($"_c{g}:");
        lines.Add("    setp.ge.u32 p, col, 16;");
        lines.Add($"    @p bra {nextGroup};");
        lines.Add($"    add.u32 cabs, base, {groupStart};");
        lines.Add("    add.u32 cabs, cabs, col;");
        lines.Add("    setp.ge.u32 p, cabs, cols_;");
        lines.Add($"    @p bra {nextGroup};");

        // ql nibble: low nibble for quarters 0..1, high nibble for 2..3 (the shift is an immediate per unrolled group).
        lines.Add($"    add.u32 t0, col, {qlWindow};");
        lines.Add("    cvt.u64.u32 addr, t0;");
        lines.Add("    add.u64 addr, addr, blockAddr;");
        lines.Add("    ld.global.u8 qv, [addr];");
        if (quarter >= 2)
        {
            lines.Add("    shr.u32 qv, qv, 4;");
        }

        lines.Add("    and.b32 qv, qv, 15;");

        // qh field: the same element's high 2 bits at bit shift 2*quarter of its window byte.
        lines.Add($"    add.u32 t0, col, {qhWindow};");
        lines.Add("    cvt.u64.u32 addr, t0;");
        lines.Add("    add.u64 addr, addr, blockAddr;");
        lines.Add("    ld.global.u8 hi, [addr];");
        if (quarter != 0)
        {
            lines.Add($"    shr.u32 hi, hi, {2 * quarter};");
        }

        lines.Add("    and.b32 hi, hi, 3;");
        lines.Add("    shl.b32 hi, hi, 4;");
        lines.Add("    or.b32 qv, qv, hi;"); // six-bit code in [0, 63]

        // Centering and fused scale stay in the same chain order the host scalar reference uses.
        lines.Add("    cvt.rn.f32.u32 wv, qv;");
        lines.Add("    sub.f32 wv, wv, 0f42000000;"); // six - 32: exact in f32 either way
        lines.Add("    mul.f32 wv, wv, scl_;");

        lines.Add("    mad.lo.u32 t0, bb, cols_, cabs;");
        lines.Add("    cvt.u64.u32 addr, t0;");
        lines.Add("    shl.b64 addr, addr, 2;");
        lines.Add("    add.u64 addr, xin, addr;");
        lines.Add("    ld.global.f32 xv, [addr];");
        lines.Add("    mul.f32 wv, xv, wv;");
        lines.Add("    add.f32 acc, acc, wv;");

        lines.Add("    add.u32 col, col, 1;");
        lines.Add($"    bra _c{g};");
        lines.Add("");

        return string.Join("\n", [.. lines]);
    }

    // The kernel's block walk plus all sixteen group arms and the store epilogue, assembled from PtxQ6KGroup so every
    // arm's window offsets stay a pure function of its group index.
    private static string BuildGemvQ6KPtx()
    {
        List<string> groups = [];
        for (int g = 0; g < 16; g++)
        {
            groups.Add(PtxQ6KGroup(g));
        }

        return GemvQ6KPtxHeader + string.Join("\n", [.. groups]) +
            "_nextBlock:\n" +
            "    add.u32 base, base, 256;\n" +
            "    add.u64 blockAddr, blockAddr, 210;\n" +
            "    bra _block;\n" +
            "_store:\n" +
            "    mad.lo.u32 t0, bb, rows_, row;\n" +
            "    cvt.u64.u32 addr, t0;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, o, addr;\n" +
            "    st.global.f32 [addr], acc;\n" +
            "    add.u32 bb, bb, 1;\n" +
            "    bra batch;\n" +
            "done:\n" +
            "    ret;\n" +
            "}\n";
    }
}
