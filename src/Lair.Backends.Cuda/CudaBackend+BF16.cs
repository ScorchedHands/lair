namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // BF16 GEMV kernel: the 16 weight bits land in the f32 high half through a shift into the per-thread local slot.
    private const string GemvBf16Ptx =
        ".visible .entry gemv_bf16(\n" +
        "    .param .u64 w_ptr,\n" +
        "    .param .u64 x_ptr,\n" +
        "    .param .u64 out_ptr,\n" +
        "    .param .u32 rows_arg,\n" +
        "    .param .u32 cols_arg,\n" +
        "    .param .u32 count_arg\n" +
        ") {\n" +
        "    .reg .u64 w, xin, o, addr;\n" +
        "    .reg .u32 row, col, bb, rows_, cols_, cnt_, tid, t0;\n" +
        "    .reg .u32 bits;\n" +
        "    .reg .b16 hw;\n" +
        "    .reg .pred p;\n" +
        "    .reg .f32 acc, wv, xv;\n" +
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
        "    shl.b32 t0, bits, 16;\n" +
        "    st.local.u32 [slot], t0;\n" +
        "    ld.local.f32 wv, [slot];\n" +
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

    // BF16 row expansion: one little-endian bf16 widened to float per column.
    private static unsafe void DequantBf16Row(byte* rowPtr, int cols, Span<float> output)
    {
        for (int i = 0; i < cols; i++)
        {
            output[i] = ReadBf16LE(rowPtr + 2 * i);
        }
    }

    // Little-endian bf16 load widened to float: the 16 bits land in the f32 high half.
    private static unsafe float ReadBf16LE(byte* p) => BitConverter.UInt32BitsToSingle((uint)((p[0] | (p[1] << 8)) << 16));
}
