namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // Q3_K: each 110-byte block covers 256 elements in sixteen 16-element groups g (0..15) with the same group windows as
    // Q2_K, quants at byte 32. A quant's high bit lives at hmask byte 16*(k & 1) + e, bit 4h + (k >> 1), so a set high bit
    // keeps the low two-bit value and a clear one subtracts four — the same ternary form the host scalar path uses. Twelve
    // bytes at byte 96 pack sixteen six-bit group scales: groups 0..7 keep their low nibble in scales[g], groups 8..15 take
    // the high nibble of scales[g-8]; the four high bits of every scale sit in scales[8 + g%4] at bit shift 2*(g/4). Half-float
    // d sits at bytes 108..109. w = d * (sc - 32) * q with sc in [0, 63] and q in [-4, 3], centered and scaled in that order
    // exactly as the host scalar reference associates it before the activation multiplies in.
    private const int Q3KBlockBytes = 110;
    private const int Q3KBlockElements = 256;

    // Unpack the six-bit group scale for group g out of the twelve packed bytes at block+96, matching GetQ3KScale on the host side.
    private static byte GetQ3KScale(int g, ReadOnlySpan<byte> scales)
    {
        int low = g < 8 ? scales[g] & 0xF : scales[g - 8] >> 4;
        int high = (scales[8 + (g & 3)] >> (2 * (g >> 2))) & 0x3;
        return (byte)(low | (high << 4));
    }

    // Q3_K row expansion: host-side scalar reference math per block and group, reading the same packed bytes the kernel consumes.
    private static void DequantQ3KRow(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Q3KBlockElements)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice((baseIdx / Q3KBlockElements) * Q3KBlockBytes, Q3KBlockBytes);
            float d = ReadHalfLE(block[108..]);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                float scale = d * (GetQ3KScale(g, block[96..]) - 32);

                ReadOnlySpan<byte> q = block[(32 + 32 * h + 16 * (k & 1))..]; // low-2-bit window: bytes [q, q+16) always inside the block
                int m = 4 * h + (k >> 1); // bit selector inside this quarter's hmask byte
                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.
                for (int e = 0; e < limit; e++)
                {
                    int three = ((q[e] >> (2 * (k >> 1))) & 3) - ((block[16 * (k & 1) + e] & (1 << m)) != 0 ? 0 : 4);
                    output[subStart + e] = three * scale;
                }
            }
        }
    }

    // Q3_K GEMV kernel: one row per thread, packed bytes read directly with scalar byte loads. The sixteen groups are fully
    // unrolled straight-line code so every group carries its own explicit 16-column bound (the host limit is
    // min(16, cols - subStart); a shared column loop bounded only by the real column count ran past the group and read across
    // later windows) and uses immediate byte offsets for its scale bytes, quant window, bit shift, and hmask field. The
    // half-float header widens through integer bit math into a per-thread local slot because cvt.f16 forms are rejected on this
    // seam; the high-bit ternary lands as one predicated subtract of four so a set bit leaves the low value alone exactly as
    // the host chain reads it. Accumulation is plain mul then add in ascending column order, matching the host scalar reference
    // chain. The group arms are emitted by PtxQ3KGroup so each arm's window offsets stay a pure function of its group index.
    private static string BuildGemvQ3KPtx()
    {
        List<string> arms = [];
        for (int g = 0; g < 16; g++)
        {
            arms.Add(PtxQ3KGroup(g));
        }

        return BuildQuantGemvPtx(
            entryName: "gemv_q3k",
            blockElements: Q3KBlockElements,
            blockBytes: Q3KBlockBytes,
            extraU32Regs: ", sc, hi, hb, qv",
            headerLines: PtxWidenHalfLE(offset: 108, dst: "dv", lbl: "_hd"),
            groupArms: arms);
    }

    // One unrolled group arm: bound the group start and every column against the real column count, reassemble this group's
    // six-bit scale out of its low-nibble byte plus its split high bits, center it on 32 and fuse with d, then walk sixteen
    // columns reading the group's two-bit field at its own constant bit shift together with that element's hmask bit.
    private static string PtxQ3KGroup(int g)
    {
        int h = g >> 3;
        int k = g & 7;

        int groupStart = g * 16;
        int windowBase = 32 + 32 * h + 16 * (k & 1); // low-2-bit window inside this block
        int bitShift = 2 * (k >> 1);
        int maskBit = 4 * h + (k >> 1); // the bit every element of this quarter carries in its hmask byte
        string nextArm = g == 15 ? "_nextBlock" : $"_g{g + 1}";

        List<string> lines = new()
        {
            $"_g{g}:",
            $"    add.u32 t0, base, {groupStart};", // group start: a tail block whose real columns end before it stops here
            "    setp.ge.u32 p, t0, cols_;",
            "    @p bra _nextBlock;",
        };

        if (g < 8)
        {
            lines.AddRange(
            [
                $"    mov.u32 t0, {96 + g};", // this group's scale low nibble ...
                "    cvt.u64.u32 addr, t0;",
                "    add.u64 addr, addr, blockAddr;",
                "    ld.global.u8 sc, [addr];",
                "    and.b32 sc, sc, 15;",
            ]);
        }
        else
        {
            lines.AddRange(
            [
                $"    mov.u32 t0, {96 + g - 8};", // ... sits in the high half of a shared byte for groups 8..15
                "    cvt.u64.u32 addr, t0;",
                "    add.u64 addr, addr, blockAddr;",
                "    ld.global.u8 sc, [addr];",
                "    shr.u32 sc, sc, 4;",
                "    and.b32 sc, sc, 15;",
            ]);
        }

        lines.AddRange(
        [
            $"    mov.u32 t0, {96 + 8 + (g & 3)};", // ... plus the four high bits at this group's constant bit shift
            "    cvt.u64.u32 addr, t0;",
            "    add.u64 addr, addr, blockAddr;",
            "    ld.global.u8 hi, [addr];",
            $"    shr.u32 hi, hi, {2 * (g >> 2)};",
            "    and.b32 hi, hi, 3;",
            "    shl.b32 hi, hi, 4;",
            "    or.b32 sc, sc, hi;", // six-bit scale code in [0, 63]
            "    cvt.rn.f32.u32 scl_, sc;",
            "    sub.f32 scl_, scl_, 0f42000000;", // sc - 32: exact in f32 either way ...
            "    mul.f32 scl_, scl_, dv;", // ... fused with d the same way the host chain carries it per group
            "    mov.u32 col, 0;",
            $"_c{g}:",
            "    setp.ge.u32 p, col, 16;", // the group's own bound ...
            $"    @p bra {nextArm};",
            $"    add.u32 cabs, base, {groupStart};", // ... plus the real column count
            "    add.u32 cabs, cabs, col;",
            "    setp.ge.u32 p, cabs, cols_;",
            $"    @p bra {nextArm};",
            $"    add.u32 t0, col, {windowBase};", // this group's window byte at its own constant bit shift
            "    cvt.u64.u32 addr, t0;",
            "    add.u64 addr, addr, blockAddr;",
            "    ld.global.u8 qv, [addr];",
            $"    shr.u32 qv, qv, {bitShift};",
            "    and.b32 qv, qv, 3;", // the low two bits of this element's three-bit code
        ]);

        lines.AddRange(
        [
            $"    add.u32 t0, col, {16 * (k & 1)};", // this element's hmask byte carries its high bit at bit 4h + (k >> 1) ...
            "    cvt.u64.u32 addr, t0;",
            "    add.u64 addr, addr, blockAddr;",
            "    ld.global.u8 hb, [addr];",
            $"    and.b32 hb, hb, {1 << maskBit};",
            "    setp.eq.u32 p, hb, 0;", // ... a clear high bit drops the low value by four, exactly as the host ternary reads it
            "    cvt.rn.f32.u32 wv, qv;",
            $"    @p sub.f32 wv, wv, {PtxFour};",
        ]);

        lines.AddRange(
        [
            "    mul.f32 wv, wv, scl_;", // three * (d * (sc - 32)): the host scalar association before the activation multiplies in
            "    mad.lo.u32 t0, bb, cols_, cabs;",
            "    cvt.u64.u32 addr, t0;",
            "    shl.b64 addr, addr, 2;",
            "    add.u64 addr, xin, addr;",
            "    ld.global.f32 xv, [addr];",
            "    mul.f32 wv, xv, wv;",
            "    add.f32 acc, acc, wv;",
            "    add.u32 col, col, 1;",
            $"    bra _c{g};",
        ]);

        return string.Join("\n", [.. lines]);
    }

    // PTX literal for the value four: a clear quant high bit subtracts it from the low two-bit value.
    private const string PtxFour = "0f40800000";
}
