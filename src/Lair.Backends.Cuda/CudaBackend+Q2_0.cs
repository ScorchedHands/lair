namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // Q2_0: each 18-byte block covers 64 elements as a half-float scale d at bytes 0..1 plus 16 little-endian quant bytes
    // at bytes 2..17, element l carrying its two-bit code at bit shift 2 * (l % 4) of qs[l / 4]. w = (two - 1) * d with two
    // in [0, 3], centered and scaled in that order exactly as the host scalar reference associates it before the activation
    // multiplies in. The window byte index and the bit shift both come out of the column counter — l / 4 and 2 * (l & 3) —
    // because four elements share one byte, so this format keeps a register-computed shift instead of an immediate per arm.
    private const int Q2_0BlockBytes = 18;
    private const int Q2_0BlockElements = 64;

    // Q2_0 row expansion: host-side scalar reference math per block, reading the same packed bytes the kernel consumes.
    private static void DequantQ2_0Row(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Q2_0BlockElements)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice((baseIdx / Q2_0BlockElements) * Q2_0BlockBytes, Q2_0BlockBytes);
            float d = ReadHalfLE(block[..2]);

            int limit = Math.Min(Q2_0BlockElements, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int l = 0; l < limit; l++)
            {
                int two = (block[2 + (l >> 2)] >> (2 * (l & 3))) & 0x3;
                output[baseIdx + l] = (two - 1) * d;
            }
        }
    }

    // Q2_0 GEMV kernel: one row per thread, quant bytes read directly with scalar byte loads. The block's single group is
    // bounded twice — once at the group start and again per column against the real column count — so a partial tail block
    // stops exactly where the host limit min(64, cols - baseIdx) stops it.
    private static string BuildGemvQ2_0Ptx() => BuildQuantGemvPtx(
        entryName: "gemv_q2_0",
        blockElements: Q2_0BlockElements,
        blockBytes: Q2_0BlockBytes,
        extraU32Regs: ", qv, wi, sh",
        headerLines: PtxWidenHalfLE(offset: 0, dst: "dv", lbl: "_hd"),
        groupArms: [PtxQ2_0Group()]);

    private static string PtxQ2_0Group()
    {
        List<string> lines = new()
        {
            "_g0:",
            "    setp.ge.u32 p, base, cols_;", // group start: a tail block with no real columns at its start stops here
            "    @p bra _nextBlock;",
            "    mov.u32 col, 0;",
            "_c0:",
            "    setp.ge.u32 p, col, 64;", // the group's own bound ...
            "    @p bra _nextBlock;",
            "    add.u32 cabs, base, col;", // ... plus the real column count
            "    setp.ge.u32 p, cabs, cols_;",
            "    @p bra _nextBlock;",
            "    shr.u32 wi, col, 2;", // window byte: block + 2 + (column / 4), four elements per byte ...
            "    shl.b32 sh, col, 1;", // ... and its bit shift at 2 * (column & 3) — the low two bits doubled
            "    and.b32 sh, sh, 6;",
            "    add.u32 t0, wi, 2;",
            "    cvt.u64.u32 addr, t0;",
            "    add.u64 addr, addr, blockAddr;",
            "    ld.global.u8 qv, [addr];",
        };

        lines.AddRange(
        [
            "    shr.u32 qv, qv, sh;", // register shift: the element's own two-bit code
            "    and.b32 qv, qv, 3;",
            "    cvt.rn.f32.u32 wv, qv;",
            "    sub.f32 wv, wv, 0f3F800000;", // two - 1: exact in f32 either way
            "    mul.f32 wv, wv, dv;", // (two - 1) * d: the host scalar association before the activation multiplies in
            "    mad.lo.u32 t0, bb, cols_, cabs;",
            "    cvt.u64.u32 addr, t0;",
            "    shl.b64 addr, addr, 2;",
            "    add.u64 addr, xin, addr;",
            "    ld.global.f32 xv, [addr];",
            "    mul.f32 wv, xv, wv;",
            "    add.f32 acc, acc, wv;",
            "    add.u32 col, col, 1;",
            "    bra _c0;",
        ]);

        return string.Join("\n", [.. lines]);
    }
}
