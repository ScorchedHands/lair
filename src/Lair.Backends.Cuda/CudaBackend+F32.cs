namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // F32 GEMV kernel: one row per thread, plain f32 loads, mul then add in ascending column order.
    private const string GemvF32Ptx =
        ".visible .entry gemv_f32(\n" +
        "    .param .u64 w_ptr,\n" +
        "    .param .u64 x_ptr,\n" +
        "    .param .u64 out_ptr,\n" +
        "    .param .u32 rows_arg,\n" +
        "    .param .u32 cols_arg,\n" +
        "    .param .u32 count_arg\n" +
        ") {\n" +
        "    .reg .u64 w, xin, o, addr;\n" +
        "    .reg .u32 row, col, bb, rows_, cols_, cnt_, tid, t0;\n" +
        "    .reg .pred p;\n" +
        "    .reg .f32 acc, wv, xv;\n" +
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
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, w, addr;\n" +
        "    ld.global.f32 wv, [addr];\n" +
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

    // F32 row expansion is a straight copy of the row's floats.
    private static unsafe void DequantF32Row(byte* rowPtr, int cols, Span<float> output) =>
        new ReadOnlySpan<float>(rowPtr, cols).CopyTo(output);
}
