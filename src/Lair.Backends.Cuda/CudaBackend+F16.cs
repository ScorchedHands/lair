namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // F16 GEMV kernel: half weights widen through integer bit math into a per-thread local slot reloaded as f32;
    // cvt.f16 forms are rejected on this seam, so the subnormal branch scales the mantissa by 2^-24 instead.
    private const string GemvF16Ptx =
        ".visible .entry gemv_f16(\n" +
        "    .param .u64 w_ptr,\n" +
        "    .param .u64 x_ptr,\n" +
        "    .param .u64 out_ptr,\n" +
        "    .param .u32 rows_arg,\n" +
        "    .param .u32 cols_arg,\n" +
        "    .param .u32 count_arg\n" +
        ") {\n" +
        "    .reg .u64 w, xin, o, addr;\n" +
        "    .reg .u32 row, col, bb, rows_, cols_, cnt_, tid, t0;\n" +
        "    .reg .u32 bits, payload, exp, mant, sign, fbits;\n" +
        "    .reg .b16 hw;\n" +
        "    .reg .pred p;\n" +
        "    .reg .f32 acc, wv, xv, subv;\n" +
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
        "    mov.u32 bb, 0;\n" +
        "batch:\n" +
        "    setp.ge.u32 p, bb, cnt_;\n" +
        "    @p bra done;\n" +
        "    mov.f32 acc, 0f00000000;\n" +
        "    mov.u32 col, 0;\n" +
        "_loop_col:\n" +
        "    setp.ge.u32 p, col, cols_;\n" +
        "    @p bra store;\n" +
        "    mad.lo.u32 t0, row, cols_, col;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 1;\n" +
        "    add.u64 addr, w, addr;\n" +
        "    ld.global.b16 hw, [addr];\n" +
        "    mov.b32 bits, {hw, 0};\n" +
        "    and.b32 sign, bits, 32768;\n" +
        "    and.b32 payload, bits, 32767;\n" +
        "    shr.u32 exp, payload, 10;\n" +
        "    and.b32 mant, payload, 1023;\n" +
        "    shl.b32 sign, sign, 16;\n" +
        "    setp.le.u32 p, exp, 0;\n" +
        "    @p bra widen_sub;\n" +
        "    add.u32 fbits, exp, 112;\n" +
        "    shl.b32 fbits, fbits, 23;\n" +
        "    shl.b32 mant, mant, 13;\n" +
        "    or.b32 fbits, fbits, mant;\n" +
        "    xor.b32 fbits, fbits, sign;\n" +
        "    st.local.u32 [slot], fbits;\n" +
        "    ld.local.f32 wv, [slot];\n" +
        "    bra widen_done;\n" +
        "widen_sub:\n" +
        "    cvt.rn.f32.u32 subv, mant;\n" +
        "    mul.f32 wv, subv, 0f33800000;\n" +
        "    setp.eq.u32 p, sign, 32768;\n" +
        "    @!p bra widen_done;\n" +
        "    mul.f32 wv, wv, 0f80000000;\n" +
        "widen_done:\n" +
        "    mad.lo.u32 t0, bb, cols_, col;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, xin, addr;\n" +
        "    ld.global.f32 xv, [addr];\n" +
        "    mul.f32 wv, wv, xv;\n" +
        "    add.f32 acc, acc, wv;\n" +
        "    add.u32 col, col, 1;\n" +
        "    bra _loop_col;\n" +
        "store:\n" +
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

    // F16 row expansion: one little-endian half widened to float per column.
    private static unsafe void DequantF16Row(byte* rowPtr, int cols, Span<float> output)
    {
        for (int i = 0; i < cols; i++)
        {
            output[i] = ReadHalfLE(rowPtr + 2 * i);
        }
    }

    // Little-endian half-float load widened to float.
    private static unsafe float ReadHalfLE(byte* p) => (float)BitConverter.Int16BitsToHalf((short)(*p | (p[1] << 8)));
}
