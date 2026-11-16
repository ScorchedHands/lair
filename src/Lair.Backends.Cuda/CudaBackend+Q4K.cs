namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // Q4_K: each 144-byte block covers 256 elements in eight 32-element groups j (0..7). Group j uses the low
    // nibbles of qs[32*(j/2) .. +31] when j is even and the high nibbles of those same bytes when odd;
    // w = d * sc_j * q - dmin * m_j. Scale/min unpacking follows the byte layout documented at QkK: two half-floats
    // (d, min) in the block first four bytes, twelve packed six-bit group scales/mins at block+4..15, and 128 bytes of
    // quants from block+16. The kernel below mirrors the scalar reference math column for column: nibble * scale -
    // shift with a plain mul then add into acc in ascending column order, so one device result matches one host dot bit for bit.
    private const int Q4KBlockBytes = 144;
    private const int Q4KBlockElements = 256;

    // Unpack the 6-bit scale/min pair for Q4_K group j from the twelve packed bytes (groups 4-7 split their high bits across two words).
    private static void GetScaleMinQ4K(int j, ReadOnlySpan<byte> scales, out byte sc, out byte m)
    {
        if (j < 4)
        {
            sc = (byte)(scales[j] & 0x3F);
            m = (byte)(scales[j + 4] & 0x3F);
        }
        else
        {
            sc = (byte)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
            m = (byte)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
        }
    }

    // Little-endian half-float load from a span, widened to float (the pointer overload lives with the F16 kernel).
    private static float ReadHalfLE(ReadOnlySpan<byte> b) => (float)BitConverter.Int16BitsToHalf((short)(b[0] | (b[1] << 8)));

    // Q4_K row expansion: host-side scalar reference math per block and group, reading the same packed bytes the kernel consumes.
    private static void DequantQ4KRow(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Q4KBlockElements)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice((baseIdx / Q4KBlockElements) * Q4KBlockBytes, Q4KBlockBytes);
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

                ReadOnlySpan<byte> q = block[(16 + 32 * (j / 2))..]; // group-pair window: bytes [q, q+32) always inside the block
                int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.
                for (int l = 0; l < limit; l++)
                {
                    output[subStart + l] = ((q[l] >> (4 * (j & 1))) & 0xF) * scale - shift;
                }
            }
        }
    }

    // Q4_K GEMV kernel: one row per thread, packed bytes read directly with scalar byte loads. The eight groups are
    // fully unrolled straight-line code so every group carries its own explicit 32-column bound (the host limit is
    // min(32, cols - subStart); a shared column loop bounded only by the real column count ran past the group and
    // read across later windows) and uses immediate byte offsets for the scale, min, and pair-window loads instead of
    // dynamic group-loop address arithmetic. Half-float headers widen through integer bit math into a per-thread local
    // slot reloaded as f32 because cvt.f16 forms are rejected on this seam. Accumulation is plain mul then add in
    // ascending column order, matching the host scalar reference chain exactly.
    private const string GemvQ4KPtx =
        ".visible .entry gemv_q4k(\n" +
        "    .param .u64 w_ptr,\n" +
        "    .param .u64 x_ptr,\n" +
        "    .param .u64 out_ptr,\n" +
        "    .param .u32 rows_arg,\n" +
        "    .param .u32 cols_arg,\n" +
        "    .param .u32 count_arg\n" +
        ") {\n" +
        "    .reg .u64 w, xin, o, blockAddr, addr;\n" +
        "    .reg .u32 row, bb, rows_, cols_, cnt_, tid, t0, strideBlocks, strideBytes, rowBase, base, col, cabs, sc, mn, hi, qv;\n" +
        "    .reg .u32 bits, payload, exp, mant, sign, fbits;\n" +
        "    .reg .b16 hw;\n" +
        "    .reg .pred p;\n" +
        "    .reg .f32 acc, dv, mv, scl_, sft_, sub_, wv, xv;\n" +
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
        "    mul.lo.u32 strideBytes, strideBlocks, 144;\n" +
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
        "    ld.global.b16 hw, [blockAddr];\n" +
        "    mov.b32 bits, {hw, 0};\n" +
        "    and.b32 sign, bits, 32768;\n" +
        "    and.b32 payload, bits, 32767;\n" +
        "    shr.u32 exp, payload, 10;\n" +
        "    and.b32 mant, payload, 1023;\n" +
        "    shl.b32 sign, sign, 16;\n" +
        "    setp.le.u32 p, exp, 0;\n" +
        "    @p bra _d_sub;\n" +
        "    add.u32 fbits, exp, 112;\n" +
        "    shl.b32 fbits, fbits, 23;\n" +
        "    shl.b32 mant, mant, 13;\n" +
        "    or.b32 fbits, fbits, mant;\n" +
        "    xor.b32 fbits, fbits, sign;\n" +
        "    st.local.u32 [slot], fbits;\n" +
        "    ld.local.f32 dv, [slot];\n" +
        "    bra _d_done;\n" +
        "_d_sub:\n" +
        "    cvt.rn.f32.u32 sub_, mant;\n" +
        "    mul.f32 dv, sub_, 0f33800000;\n" +
        "    setp.eq.u32 p, sign, 32768;\n" +
        "    @!p bra _d_done;\n" +
        "    mul.f32 dv, dv, 0f80000000;\n" +
        "_d_done:\n" +
        "\n" +
        "    add.u64 addr, blockAddr, 2;\n" +
        "    ld.global.b16 hw, [addr];\n" +
        "    mov.b32 bits, {hw, 0};\n" +
        "    and.b32 sign, bits, 32768;\n" +
        "    and.b32 payload, bits, 32767;\n" +
        "    shr.u32 exp, payload, 10;\n" +
        "    and.b32 mant, payload, 1023;\n" +
        "    shl.b32 sign, sign, 16;\n" +
        "    setp.le.u32 p, exp, 0;\n" +
        "    @p bra _m_sub;\n" +
        "    add.u32 fbits, exp, 112;\n" +
        "    shl.b32 fbits, fbits, 23;\n" +
        "    shl.b32 mant, mant, 13;\n" +
        "    or.b32 fbits, fbits, mant;\n" +
        "    xor.b32 fbits, fbits, sign;\n" +
        "    st.local.u32 [slot], fbits;\n" +
        "    ld.local.f32 mv, [slot];\n" +
        "    bra _m_done;\n" +
        "_m_sub:\n" +
        "    cvt.rn.f32.u32 sub_, mant;\n" +
        "    mul.f32 mv, sub_, 0f33800000;\n" +
        "    setp.eq.u32 p, sign, 32768;\n" +
        "    @!p bra _m_done;\n" +
        "    mul.f32 mv, mv, 0f80000000;\n" +
        "_m_done:\n" +
        "\n" +
        "_g0:\n" +
        "    add.u32 t0, base, 0;\n" +
        "    setp.ge.u32 p, t0, cols_;\n" +
        "    @p bra _nextBlock;\n" +
        "    mov.u32 t0, 4;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 sc, [addr];\n" +
        "    and.b32 sc, sc, 63;\n" +
        "    mov.u32 t0, 8;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 mn, [addr];\n" +
        "    and.b32 mn, mn, 63;\n" +
        "    cvt.rn.f32.u32 scl_, sc;\n" +
        "    mul.f32 scl_, scl_, dv;\n" +
        "    cvt.rn.f32.u32 sft_, mn;\n" +
        "    mul.f32 sft_, sft_, mv;\n" +
        "    mov.u32 col, 0;\n" +
        "_c0:\n" +
        "    setp.ge.u32 p, col, 32;\n" +
        "    @p bra _g1;\n" +
        "    add.u32 cabs, base, 0;\n" +
        "    add.u32 cabs, cabs, col;\n" +
        "    setp.ge.u32 p, cabs, cols_;\n" +
        "    @p bra _g1;\n" +
        "    add.u32 t0, col, 16;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 qv, [addr];\n" +
        "    and.b32 qv, qv, 15;\n" +
        "    cvt.rn.f32.u32 xv, qv;\n" +
        "    mul.f32 wv, xv, scl_;\n" +
        "    sub.f32 wv, wv, sft_;\n" +
        "    mad.lo.u32 t0, bb, cols_, cabs;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, xin, addr;\n" +
        "    ld.global.f32 xv, [addr];\n" +
        "    mul.f32 wv, xv, wv;\n" +
        "    add.f32 acc, acc, wv;\n" +
        "    add.u32 col, col, 1;\n" +
        "    bra _c0;\n" +
        "\n" +
        "_g1:\n" +
        "    add.u32 t0, base, 32;\n" +
        "    setp.ge.u32 p, t0, cols_;\n" +
        "    @p bra _nextBlock;\n" +
        "    mov.u32 t0, 5;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 sc, [addr];\n" +
        "    and.b32 sc, sc, 63;\n" +
        "    mov.u32 t0, 9;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 mn, [addr];\n" +
        "    and.b32 mn, mn, 63;\n" +
        "    cvt.rn.f32.u32 scl_, sc;\n" +
        "    mul.f32 scl_, scl_, dv;\n" +
        "    cvt.rn.f32.u32 sft_, mn;\n" +
        "    mul.f32 sft_, sft_, mv;\n" +
        "    mov.u32 col, 0;\n" +
        "_c1:\n" +
        "    setp.ge.u32 p, col, 32;\n" +
        "    @p bra _g2;\n" +
        "    add.u32 cabs, base, 32;\n" +
        "    add.u32 cabs, cabs, col;\n" +
        "    setp.ge.u32 p, cabs, cols_;\n" +
        "    @p bra _g2;\n" +
        "    add.u32 t0, col, 16;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 qv, [addr];\n" +
        "    shr.u32 qv, qv, 4;\n" +
        "    and.b32 qv, qv, 15;\n" +
        "    cvt.rn.f32.u32 xv, qv;\n" +
        "    mul.f32 wv, xv, scl_;\n" +
        "    sub.f32 wv, wv, sft_;\n" +
        "    mad.lo.u32 t0, bb, cols_, cabs;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, xin, addr;\n" +
        "    ld.global.f32 xv, [addr];\n" +
        "    mul.f32 wv, xv, wv;\n" +
        "    add.f32 acc, acc, wv;\n" +
        "    add.u32 col, col, 1;\n" +
        "    bra _c1;\n" +
        "\n" +
        "_g2:\n" +
        "    add.u32 t0, base, 64;\n" +
        "    setp.ge.u32 p, t0, cols_;\n" +
        "    @p bra _nextBlock;\n" +
        "    mov.u32 t0, 6;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 sc, [addr];\n" +
        "    and.b32 sc, sc, 63;\n" +
        "    mov.u32 t0, 10;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 mn, [addr];\n" +
        "    and.b32 mn, mn, 63;\n" +
        "    cvt.rn.f32.u32 scl_, sc;\n" +
        "    mul.f32 scl_, scl_, dv;\n" +
        "    cvt.rn.f32.u32 sft_, mn;\n" +
        "    mul.f32 sft_, sft_, mv;\n" +
        "    mov.u32 col, 0;\n" +
        "_c2:\n" +
        "    setp.ge.u32 p, col, 32;\n" +
        "    @p bra _g3;\n" +
        "    add.u32 cabs, base, 64;\n" +
        "    add.u32 cabs, cabs, col;\n" +
        "    setp.ge.u32 p, cabs, cols_;\n" +
        "    @p bra _g3;\n" +
        "    add.u32 t0, col, 48;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 qv, [addr];\n" +
        "    and.b32 qv, qv, 15;\n" +
        "    cvt.rn.f32.u32 xv, qv;\n" +
        "    mul.f32 wv, xv, scl_;\n" +
        "    sub.f32 wv, wv, sft_;\n" +
        "    mad.lo.u32 t0, bb, cols_, cabs;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, xin, addr;\n" +
        "    ld.global.f32 xv, [addr];\n" +
        "    mul.f32 wv, xv, wv;\n" +
        "    add.f32 acc, acc, wv;\n" +
        "    add.u32 col, col, 1;\n" +
        "    bra _c2;\n" +
        "\n" +
        "_g3:\n" +
        "    add.u32 t0, base, 96;\n" +
        "    setp.ge.u32 p, t0, cols_;\n" +
        "    @p bra _nextBlock;\n" +
        "    mov.u32 t0, 7;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 sc, [addr];\n" +
        "    and.b32 sc, sc, 63;\n" +
        "    mov.u32 t0, 11;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 mn, [addr];\n" +
        "    and.b32 mn, mn, 63;\n" +
        "    cvt.rn.f32.u32 scl_, sc;\n" +
        "    mul.f32 scl_, scl_, dv;\n" +
        "    cvt.rn.f32.u32 sft_, mn;\n" +
        "    mul.f32 sft_, sft_, mv;\n" +
        "    mov.u32 col, 0;\n" +
        "_c3:\n" +
        "    setp.ge.u32 p, col, 32;\n" +
        "    @p bra _g4;\n" +
        "    add.u32 cabs, base, 96;\n" +
        "    add.u32 cabs, cabs, col;\n" +
        "    setp.ge.u32 p, cabs, cols_;\n" +
        "    @p bra _g4;\n" +
        "    add.u32 t0, col, 48;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 qv, [addr];\n" +
        "    shr.u32 qv, qv, 4;\n" +
        "    and.b32 qv, qv, 15;\n" +
        "    cvt.rn.f32.u32 xv, qv;\n" +
        "    mul.f32 wv, xv, scl_;\n" +
        "    sub.f32 wv, wv, sft_;\n" +
        "    mad.lo.u32 t0, bb, cols_, cabs;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, xin, addr;\n" +
        "    ld.global.f32 xv, [addr];\n" +
        "    mul.f32 wv, xv, wv;\n" +
        "    add.f32 acc, acc, wv;\n" +
        "    add.u32 col, col, 1;\n" +
        "    bra _c3;\n" +
        "\n" +
        "_g4:\n" +
        "    add.u32 t0, base, 128;\n" +
        "    setp.ge.u32 p, t0, cols_;\n" +
        "    @p bra _nextBlock;\n" +
        "    mov.u32 t0, 12;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 sc, [addr];\n" +
        "    and.b32 sc, sc, 15;\n" +
        "    ld.global.u8 mn, [addr];\n" +
        "    shr.u32 mn, mn, 4;\n" +
        "    and.b32 mn, mn, 15;\n" +
        "    mov.u32 t0, 4;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 hi, [addr];\n" +
        "    shr.u32 hi, hi, 6;\n" +
        "    and.b32 hi, hi, 3;\n" +
        "    shl.b32 hi, hi, 4;\n" +
        "    or.b32 sc, sc, hi;\n" +
        "    mov.u32 t0, 8;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 hi, [addr];\n" +
        "    shr.u32 hi, hi, 6;\n" +
        "    and.b32 hi, hi, 3;\n" +
        "    shl.b32 hi, hi, 4;\n" +
        "    or.b32 mn, mn, hi;\n" +
        "    cvt.rn.f32.u32 scl_, sc;\n" +
        "    mul.f32 scl_, scl_, dv;\n" +
        "    cvt.rn.f32.u32 sft_, mn;\n" +
        "    mul.f32 sft_, sft_, mv;\n" +
        "    mov.u32 col, 0;\n" +
        "_c4:\n" +
        "    setp.ge.u32 p, col, 32;\n" +
        "    @p bra _g5;\n" +
        "    add.u32 cabs, base, 128;\n" +
        "    add.u32 cabs, cabs, col;\n" +
        "    setp.ge.u32 p, cabs, cols_;\n" +
        "    @p bra _g5;\n" +
        "    add.u32 t0, col, 80;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 qv, [addr];\n" +
        "    and.b32 qv, qv, 15;\n" +
        "    cvt.rn.f32.u32 xv, qv;\n" +
        "    mul.f32 wv, xv, scl_;\n" +
        "    sub.f32 wv, wv, sft_;\n" +
        "    mad.lo.u32 t0, bb, cols_, cabs;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, xin, addr;\n" +
        "    ld.global.f32 xv, [addr];\n" +
        "    mul.f32 wv, xv, wv;\n" +
        "    add.f32 acc, acc, wv;\n" +
        "    add.u32 col, col, 1;\n" +
        "    bra _c4;\n" +
        "\n" +
        "_g5:\n" +
        "    add.u32 t0, base, 160;\n" +
        "    setp.ge.u32 p, t0, cols_;\n" +
        "    @p bra _nextBlock;\n" +
        "    mov.u32 t0, 13;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 sc, [addr];\n" +
        "    and.b32 sc, sc, 15;\n" +
        "    ld.global.u8 mn, [addr];\n" +
        "    shr.u32 mn, mn, 4;\n" +
        "    and.b32 mn, mn, 15;\n" +
        "    mov.u32 t0, 5;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 hi, [addr];\n" +
        "    shr.u32 hi, hi, 6;\n" +
        "    and.b32 hi, hi, 3;\n" +
        "    shl.b32 hi, hi, 4;\n" +
        "    or.b32 sc, sc, hi;\n" +
        "    mov.u32 t0, 9;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 hi, [addr];\n" +
        "    shr.u32 hi, hi, 6;\n" +
        "    and.b32 hi, hi, 3;\n" +
        "    shl.b32 hi, hi, 4;\n" +
        "    or.b32 mn, mn, hi;\n" +
        "    cvt.rn.f32.u32 scl_, sc;\n" +
        "    mul.f32 scl_, scl_, dv;\n" +
        "    cvt.rn.f32.u32 sft_, mn;\n" +
        "    mul.f32 sft_, sft_, mv;\n" +
        "    mov.u32 col, 0;\n" +
        "_c5:\n" +
        "    setp.ge.u32 p, col, 32;\n" +
        "    @p bra _g6;\n" +
        "    add.u32 cabs, base, 160;\n" +
        "    add.u32 cabs, cabs, col;\n" +
        "    setp.ge.u32 p, cabs, cols_;\n" +
        "    @p bra _g6;\n" +
        "    add.u32 t0, col, 80;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 qv, [addr];\n" +
        "    shr.u32 qv, qv, 4;\n" +
        "    and.b32 qv, qv, 15;\n" +
        "    cvt.rn.f32.u32 xv, qv;\n" +
        "    mul.f32 wv, xv, scl_;\n" +
        "    sub.f32 wv, wv, sft_;\n" +
        "    mad.lo.u32 t0, bb, cols_, cabs;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, xin, addr;\n" +
        "    ld.global.f32 xv, [addr];\n" +
        "    mul.f32 wv, xv, wv;\n" +
        "    add.f32 acc, acc, wv;\n" +
        "    add.u32 col, col, 1;\n" +
        "    bra _c5;\n" +
        "\n" +
        "_g6:\n" +
        "    add.u32 t0, base, 192;\n" +
        "    setp.ge.u32 p, t0, cols_;\n" +
        "    @p bra _nextBlock;\n" +
        "    mov.u32 t0, 14;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 sc, [addr];\n" +
        "    and.b32 sc, sc, 15;\n" +
        "    ld.global.u8 mn, [addr];\n" +
        "    shr.u32 mn, mn, 4;\n" +
        "    and.b32 mn, mn, 15;\n" +
        "    mov.u32 t0, 6;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 hi, [addr];\n" +
        "    shr.u32 hi, hi, 6;\n" +
        "    and.b32 hi, hi, 3;\n" +
        "    shl.b32 hi, hi, 4;\n" +
        "    or.b32 sc, sc, hi;\n" +
        "    mov.u32 t0, 10;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 hi, [addr];\n" +
        "    shr.u32 hi, hi, 6;\n" +
        "    and.b32 hi, hi, 3;\n" +
        "    shl.b32 hi, hi, 4;\n" +
        "    or.b32 mn, mn, hi;\n" +
        "    cvt.rn.f32.u32 scl_, sc;\n" +
        "    mul.f32 scl_, scl_, dv;\n" +
        "    cvt.rn.f32.u32 sft_, mn;\n" +
        "    mul.f32 sft_, sft_, mv;\n" +
        "    mov.u32 col, 0;\n" +
        "_c6:\n" +
        "    setp.ge.u32 p, col, 32;\n" +
        "    @p bra _g7;\n" +
        "    add.u32 cabs, base, 192;\n" +
        "    add.u32 cabs, cabs, col;\n" +
        "    setp.ge.u32 p, cabs, cols_;\n" +
        "    @p bra _g7;\n" +
        "    add.u32 t0, col, 112;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 qv, [addr];\n" +
        "    and.b32 qv, qv, 15;\n" +
        "    cvt.rn.f32.u32 xv, qv;\n" +
        "    mul.f32 wv, xv, scl_;\n" +
        "    sub.f32 wv, wv, sft_;\n" +
        "    mad.lo.u32 t0, bb, cols_, cabs;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, xin, addr;\n" +
        "    ld.global.f32 xv, [addr];\n" +
        "    mul.f32 wv, xv, wv;\n" +
        "    add.f32 acc, acc, wv;\n" +
        "    add.u32 col, col, 1;\n" +
        "    bra _c6;\n" +
        "\n" +
        "_g7:\n" +
        "    add.u32 t0, base, 224;\n" +
        "    setp.ge.u32 p, t0, cols_;\n" +
        "    @p bra _nextBlock;\n" +
        "    mov.u32 t0, 15;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 sc, [addr];\n" +
        "    and.b32 sc, sc, 15;\n" +
        "    ld.global.u8 mn, [addr];\n" +
        "    shr.u32 mn, mn, 4;\n" +
        "    and.b32 mn, mn, 15;\n" +
        "    mov.u32 t0, 7;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 hi, [addr];\n" +
        "    shr.u32 hi, hi, 6;\n" +
        "    and.b32 hi, hi, 3;\n" +
        "    shl.b32 hi, hi, 4;\n" +
        "    or.b32 sc, sc, hi;\n" +
        "    mov.u32 t0, 11;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 hi, [addr];\n" +
        "    shr.u32 hi, hi, 6;\n" +
        "    and.b32 hi, hi, 3;\n" +
        "    shl.b32 hi, hi, 4;\n" +
        "    or.b32 mn, mn, hi;\n" +
        "    cvt.rn.f32.u32 scl_, sc;\n" +
        "    mul.f32 scl_, scl_, dv;\n" +
        "    cvt.rn.f32.u32 sft_, mn;\n" +
        "    mul.f32 sft_, sft_, mv;\n" +
        "    mov.u32 col, 0;\n" +
        "_c7:\n" +
        "    setp.ge.u32 p, col, 32;\n" +
        "    @p bra _nextBlock;\n" +
        "    add.u32 cabs, base, 224;\n" +
        "    add.u32 cabs, cabs, col;\n" +
        "    setp.ge.u32 p, cabs, cols_;\n" +
        "    @p bra _nextBlock;\n" +
        "    add.u32 t0, col, 112;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    add.u64 addr, addr, blockAddr;\n" +
        "    ld.global.u8 qv, [addr];\n" +
        "    shr.u32 qv, qv, 4;\n" +
        "    and.b32 qv, qv, 15;\n" +
        "    cvt.rn.f32.u32 xv, qv;\n" +
        "    mul.f32 wv, xv, scl_;\n" +
        "    sub.f32 wv, wv, sft_;\n" +
        "    mad.lo.u32 t0, bb, cols_, cabs;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, xin, addr;\n" +
        "    ld.global.f32 xv, [addr];\n" +
        "    mul.f32 wv, xv, wv;\n" +
        "    add.f32 acc, acc, wv;\n" +
        "    add.u32 col, col, 1;\n" +
        "    bra _c7;\n" +
        "\n" +
        "_nextBlock:\n" +
        "    add.u32 base, base, 256;\n" +
        "    add.u64 blockAddr, blockAddr, 144;\n" +
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
