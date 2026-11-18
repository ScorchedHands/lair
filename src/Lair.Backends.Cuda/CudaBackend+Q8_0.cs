namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // Q8_0: each 34-byte block covers 32 elements as a half-float scale d at bytes 0..1 followed by 32 int8 quants.
    // w = (float)q * d, the same association the host scalar reference uses per element before it multiplies in the
    // activation and adds into acc in ascending column order. The signed quant widens through u32 bit math — subtracting
    // 256 when its high bit is set keeps every value exactly representable in f32, which is how Q6_K group scales widen
    // on this seam; the half-float header widens through integer bit math into a per-thread local slot because cvt.f16
    // forms are rejected here.
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0BlockElements = 32;

    // Q8_0 row expansion: host-side scalar reference math per block, reading the same packed bytes the kernel consumes.
    private static void DequantQ8_0Row(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Q8_0BlockElements)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice((baseIdx / Q8_0BlockElements) * Q8_0BlockBytes, Q8_0BlockBytes);
            float d = ReadHalfLE(block[..2]);

            int limit = Math.Min(Q8_0BlockElements, cols - baseIdx); // padded quants are zero: stop at the real column count.
            for (int l = 0; l < limit; l++)
            {
                output[baseIdx + l] = (float)(sbyte)block[2 + l] * d;
            }
        }
    }

    // Q8_0 GEMV kernel: one row per thread, quant bytes read directly with scalar byte loads. The block's single group is
    // bounded twice — once at the group start and again per column against the real column count — so a partial tail block
    // stops exactly where the host limit min(32, cols - baseIdx) stops it.
    private static string BuildGemvQ8_0Ptx() => BuildQuantGemvPtx(
        entryName: "gemv_q8_0",
        blockElements: Q8_0BlockElements,
        blockBytes: Q8_0BlockBytes,
        extraU32Regs: ", qv, hb",
        headerLines: PtxWidenHalfLE(offset: 0, dst: "dv", lbl: "_hd"),
        groupArms: [PtxQ8_0Group()]);

    private static string PtxQ8_0Group()
    {
        List<string> lines = new()
        {
            "_g0:",
            "    setp.ge.u32 p, base, cols_;", // group start: a tail block with no real columns at its start stops here
            "    @p bra _nextBlock;",
            "    mov.u32 col, 0;",
            "_c0:",
            "    setp.ge.u32 p, col, 32;", // the group's own bound ...
            "    @p bra _nextBlock;",
            "    add.u32 cabs, base, col;", // ... plus the real column count, so a partial tail block stops where the host limit min(32, cols - baseIdx) stops it
            "    setp.ge.u32 p, cabs, cols_;",
            "    @p bra _nextBlock;",
            "    add.u32 t0, col, 2;", // quant byte: block + 2 + column inside the block
            "    cvt.u64.u32 addr, t0;",
            "    add.u64 addr, addr, blockAddr;",
            "    ld.global.u8 qv, [addr];",
        };

        lines.AddRange(PtxWidenInt8("qv", "wv")); // signed int8 quant as f32

        lines.AddRange(
        [
            "    mul.f32 wv, wv, dv;", // (float)q * d: the host scalar association before the activation multiplies in
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
