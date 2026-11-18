using System.Numerics;
using System.Text;

namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // Address arithmetic stays in computed-register form throughout: this driver seam rejects [base+reg] load forms, so
    // each element address is mad'd from its index, widened, shifted by the element size, and added to the base. Half
    // and bf16 weights widen through integer bit math into a per-thread local slot reloaded as f32; cvt.f16 forms are
    // rejected outright (CUDA_ERROR_INVALID_PTX) on this seam. Accumulation is plain mul then add in ascending column
    // order - the same chain the scalar CPU reference uses, so one device result matches one host dot bit for bit.
    private const string PtxHeader =
        ".version 7.0\n" +
        ".target sm_50\n" +
        ".address_size 64\n";

    // Shared register set for the packed-quant GEMV family: a row guard, a batch loop, and one walk of `blockBytes`-wide
    // blocks covering `blockElements` columns each. The unpack scratch (bits/payload/exp/mant/sign/fbits/sub_/hw plus the
    // per-thread local slot) belongs to every kernel in the family because every block header is a half-float widened
    // through integer bit math, and the scale pair (dv/min mv with fused scl_/sft_) belongs to every grouped format.
    private static string BuildQuantGemvPtx(
        string entryName,
        int blockElements,
        int blockBytes,
        string extraU32Regs,
        IEnumerable<string> headerLines,
        IEnumerable<string> groupArms)
    {
        ArgumentNullException.ThrowIfNull(headerLines);
        ArgumentNullException.ThrowIfNull(groupArms);

        List<string> lines =
        [
            $".visible .entry {entryName}(",
            "    .param .u64 w_ptr,",
            "    .param .u64 x_ptr,",
            "    .param .u64 out_ptr,",
            "    .param .u32 rows_arg,",
            "    .param .u32 cols_arg,",
            "    .param .u32 count_arg",
            ") {",
            "    .reg .u64 w, xin, o, blockAddr, addr;",
                        // `col`/`cabs` are the group-relative and row-absolute column cursors every arm in this family runs on.
            $"    .reg .u32 row, bb, rows_, cols_, cnt_, tid, t0, strideBlocks, strideBytes, rowBase, base, col, cabs{extraU32Regs};",

            "    .reg .u32 bits, payload, exp, mant, sign, fbits;",
            "    .reg .b16 hw;",
            "    .reg .pred p;",
            "    .reg .f32 acc, dv, mv, scl_, sft_, wv, xv, sub_, hf;",
            "    .local .align 4 .b8 slot[4];",
            "",
            "    ld.param.u64 w, [w_ptr];",
            "    ld.param.u64 xin, [x_ptr];",
            "    ld.param.u64 o, [out_ptr];",
            "    ld.param.u32 rows_, [rows_arg];",
            "    ld.param.u32 cols_, [cols_arg];",
            "    ld.param.u32 cnt_, [count_arg];",
            "",
            "    mov.u32 tid, %ctaid.x;",
            "    mov.u32 t0, %ntid.x;",
            "    mul.lo.u32 tid, tid, t0;",
            "    mov.u32 t0, %tid.x;",
            "    add.u32 row, tid, t0;",
            "    setp.ge.u32 p, row, rows_;",
            "    @p bra done;",
            "",
            // Row stride in bytes: ceil(cols / blockElements) whole blocks, each blockBytes wide — the same zero-padded
            // tail-block geometry TensorShape.RowStrideBytes reports to ValidateShape.
            $"    add.u32 strideBlocks, cols_, {blockElements - 1};",
            $"    shr.u32 strideBlocks, strideBlocks, {BitOperations.Log2((uint)blockElements)};",
            $"    mul.lo.u32 strideBytes, strideBlocks, {blockBytes};",
            "    mul.lo.u32 rowBase, row, strideBytes;",
            "",
            // One batch iteration re-walks the whole block chain for its own activation vector and stores one output.
            "    mov.u32 bb, 0;",
            "batch:",
            "    setp.ge.u32 p, bb, cnt_;",
            "    @p bra done;",
            "    mov.f32 acc, 0f00000000;",
            "    cvt.u64.u32 blockAddr, rowBase;",
            "    add.u64 blockAddr, w, blockAddr;",
            "",
            "    mov.u32 base, 0;",
            "_block:",
            "    setp.ge.u32 p, base, cols_;",
            "    @p bra _store;",
        ];

        lines.AddRange(headerLines);
        lines.AddRange(groupArms);

        lines.AddRange(
        [
            "",
            "_nextBlock:",
            $"    add.u32 base, base, {blockElements};",
            $"    add.u64 blockAddr, blockAddr, {blockBytes};",
            "    bra _block;",
            "_store:",
            "    mad.lo.u32 t0, bb, rows_, row;",
            "    cvt.u64.u32 addr, t0;",
            "    shl.b64 addr, addr, 2;",
            "    add.u64 addr, o, addr;",
            "    st.global.f32 [addr], acc;",
            "    add.u32 bb, bb, 1;",
            "    bra batch;",
            "done:",
            "    ret;",
            "}",
        ]);

        return string.Join("\n", [.. lines]) + "\n";
    }

    // Half-float block header at `offset` inside the current block widened to f32 in `dst`, through integer bit math and a
    // per-thread local slot because cvt.f16 forms are rejected on this seam. The subnormal branch scales the mantissa by
    // 2^-24 and carries its sign with a real negation; the normal branch folds the half exponent into the f32 biased
    // exponent (offset 112) and shifts the mantissa left 13, exactly as the Q6_K header widening does. `lbl` keeps every
    // label unique when one block carries two headers (the d/min pairs).
    private static IEnumerable<string> PtxWidenHalfLE(int offset, string dst, string lbl) =>
    [
        $"    mov.u32 t0, {offset};",
        "    cvt.u64.u32 addr, t0;",
        "    add.u64 addr, addr, blockAddr;",
        "    ld.global.b16 hw, [addr];",
        "    mov.b32 bits, {hw, 0};",
        "    and.b32 sign, bits, 32768;",
        "    and.b32 payload, bits, 32767;",
        "    shr.u32 exp, payload, 10;",
        "    and.b32 mant, payload, 1023;",
        "    setp.le.u32 p, exp, 0;",
        $"    @!p bra {lbl}_norm;",
        $"{lbl}_sub:",
        "    cvt.rn.f32.u32 sub_, mant;",
        $"    mul.f32 {dst}, sub_, 0f33800000;", // 2^-24: the half subnormal magnitude, exactly representable in f32
        "    setp.eq.u32 p, sign, 32768;",
        $"    @!p bra {lbl}_done;",
        $"    mul.f32 {dst}, {dst}, 0fBF800000;", // -1.0f so the subnormal magnitude keeps its sign through a real negation
        $"    bra {lbl}_done;",
        $"{lbl}_norm:",
        "    shl.b32 sign, sign, 16;",
        "    add.u32 fbits, exp, 112;",
        "    shl.b32 fbits, fbits, 23;",
        "    shl.b32 mant, mant, 13;",
        "    or.b32 fbits, fbits, mant;",
        "    xor.b32 fbits, fbits, sign;",
        "    st.local.u32 [slot], fbits;",
        $"    ld.local.f32 {dst}, [slot];",
        $"{lbl}_done:",
    ];

    // Unsigned byte held in a u32 register widened to its two's-complement value as f32: subtract 256 when bit 7 is set.
    // Every value that produces stays exactly representable in f32, which is how the Q6_K group scales widen today.
    private static IEnumerable<string> PtxWidenInt8(string srcReg, string dstF) =>
    [
        $"    and.b32 {srcReg}, {srcReg}, 255;",
        "    shr.u32 hb, " + srcReg + ", 7;",
        "    shl.b32 hb, hb, 8;",
        $"    cvt.rn.f32.u32 {dstF}, {srcReg};",
        "    cvt.rn.f32.u32 hf, hb;",
        $"    sub.f32 {dstF}, {dstF}, hf;",
    ];

    // Little-endian 32-bit field assembled from four scalar byte loads at `offset`: the seam answers only computed
    // addresses and a packed row is not dword-aligned, so a wide load of the field itself would be misaligned.
    private static IEnumerable<string> PtxLoadFieldLE(int offset, string dstReg) =>
    [
        $"    mov.u32 t0, {offset};",
        "    cvt.u64.u32 addr, t0;",
        "    add.u64 addr, addr, blockAddr;",
        $"    ld.global.u8 {dstReg}, [addr];",
        $"    mov.u32 t0, {offset + 1};",
        "    cvt.u64.u32 addr, t0;",
        "    add.u64 addr, addr, blockAddr;",
        "    ld.global.u8 b1, [addr];",
        "    shl.b32 b1, b1, 8;",
        "    or.b32 " + dstReg + ", " + dstReg + ", b1;",
        $"    mov.u32 t0, {offset + 2};",
        "    cvt.u64.u32 addr, t0;",
        "    add.u64 addr, addr, blockAddr;",
        "    ld.global.u8 b1, [addr];",
        "    shl.b32 b1, b1, 16;",
        "    or.b32 " + dstReg + ", " + dstReg + ", b1;",
        $"    mov.u32 t0, {offset + 3};",
        "    cvt.u64.u32 addr, t0;",
        "    add.u64 addr, addr, blockAddr;",
        "    ld.global.u8 b1, [addr];",
        "    shl.b32 b1, b1, 24;",
        "    or.b32 " + dstReg + ", " + dstReg + ", b1;",
    ];

    private static readonly byte[] PtxBytes = Encoding.UTF8.GetBytes(
        PtxHeader + "\n" + GemvF32Ptx + "\n" + GemvF16Ptx + "\n" + GemvBf16Ptx + "\n" + GemvQ4KPtx + "\n" + BuildGemvQ6KPtx() +
        "\n" + BuildGemvQ8_0Ptx() + "\n" + BuildGemvQ5_0Ptx() + "\n" + BuildGemvQ2_0Ptx() +
        "\n" + BuildGemvQ5KPtx() + "\n" + BuildGemvQ2KPtx() + "\n" + BuildGemvQ3KPtx() + "\0");
}
