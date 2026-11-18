namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // Q2_K: each 84-byte block covers 256 elements in sixteen 16-element groups g (0..15). Scale byte g at block+g carries
    // the group's four-bit scale in its low nibble and its four-bit min in its high nibble; inside a 128-element half
    // h = g/8 the quant window is block + 16 + 32h + 16*(k & 1) read at bit shift 2*(k >> 1), so one window byte carries
    // four successive groups of the same quarter. Half-floats d and min sit at bytes 80..83. w = d * (sc & 0xF) * q -
    // dmin * (sc >> 4) with q in [0, 3], fused and subtracted in that order exactly as the host scalar reference associates
    // it before the activation multiplies in.
    private const int Q2KBlockBytes = 84;
    private const int Q2KBlockElements = 256;

    // Q2_K row expansion: host-side scalar reference math per block and group, reading the same packed bytes the kernel consumes.
    private static void DequantQ2KRow(ReadOnlySpan<byte> rowBytes, int cols, Span<float> output)
    {
        for (int baseIdx = 0; baseIdx < cols; baseIdx += Q2KBlockElements)
        {
            ReadOnlySpan<byte> block = rowBytes.Slice((baseIdx / Q2KBlockElements) * Q2KBlockBytes, Q2KBlockBytes);
            float d = ReadHalfLE(block[80..]);
            float min = ReadHalfLE(block[82..]);

            for (int g = 0; g < 16; g++)
            {
                int subStart = baseIdx + g * 16;
                if (subStart >= cols)
                {
                    break; // zero-padded tail block: no elements left to consume.
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                byte scByte = block[g];
                float scale = d * (scByte & 0xF);
                float shift = min * (scByte >> 4);

                ReadOnlySpan<byte> q = block[(16 + 32 * h + 16 * (k & 1))..]; // group window: bytes [q, q+16) always inside the block
                int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.
                for (int e = 0; e < limit; e++)
                {
                    output[subStart + e] = ((q[e] >> (2 * (k >> 1))) & 3) * scale - shift;
                }
            }
        }
    }

    // Q2_K GEMV kernel: one row per thread, packed bytes read directly with scalar byte loads. The sixteen groups are fully
    // unrolled straight-line code so every group carries its own explicit 16-column bound (the host limit is
    // min(16, cols - subStart); a shared column loop bounded only by the real column count ran past the group and read
    // across later windows) and uses immediate byte offsets for its scale byte, quant window, and bit shift. Both half-float
    // headers widen through integer bit math into a per-thread local slot because cvt.f16 forms are rejected on this seam.
    // Accumulation is plain mul then add in ascending column order, matching the host scalar reference chain exactly. The
    // group arms below are emitted by PtxQ2KGroup so each arm's window offsets stay a pure function of its group index.
    private static string BuildGemvQ2KPtx()
    {
        List<string> arms = [];
        for (int g = 0; g < 16; g++)
        {
            arms.Add(PtxQ2KGroup(g));
        }

        return BuildQuantGemvPtx(
            entryName: "gemv_q2k",
            blockElements: Q2KBlockElements,
            blockBytes: Q2KBlockBytes,
            extraU32Regs: ", sc, mn, qv",
            headerLines: [.. PtxWidenHalfLE(offset: 80, dst: "dv", lbl: "_hd"), .. PtxWidenHalfLE(offset: 82, dst: "mv", lbl: "_hm")],
            groupArms: arms);
    }

    // One unrolled group arm: bound the group start and every column against the real column count, unpack this group's
    // four-bit scale and min out of one byte (the host chain reads that same byte twice), fuse them with the block headers d
    // and min, then walk sixteen columns reading the group's two-bit field at its own constant bit shift.
    private static string PtxQ2KGroup(int g)
    {
        int h = g >> 3;
        int k = g & 7;

        int groupStart = g * 16;
        int windowBase = 16 + 32 * h + 16 * (k & 1);
        int bitShift = 2 * (k >> 1);
        string nextArm = g == 15 ? "_nextBlock" : $"_g{g + 1}";

        List<string> lines = new()
        {
            $"_g{g}:",
            $"    add.u32 t0, base, {groupStart};", // group start: a tail block whose real columns end before it stops here
            "    setp.ge.u32 p, t0, cols_;",
            "    @p bra _nextBlock;",
            $"    mov.u32 t0, {g};", // this group's packed scale/min byte at block+g
            "    cvt.u64.u32 addr, t0;",
            "    add.u64 addr, addr, blockAddr;",
            "    ld.global.u8 sc, [addr];",
            "    mov.b32 mn, sc;", // the host chain reads this one byte twice: low nibble scale, high nibble min
            "    and.b32 sc, sc, 15;",
            "    shr.u32 mn, mn, 4;",
            "    and.b32 mn, mn, 15;",
            "    cvt.rn.f32.u32 scl_, sc;", // d * (sc & 0xF) ...
            "    mul.f32 scl_, scl_, dv;",
            "    cvt.rn.f32.u32 sft_, mn;", // ... and min * (sc >> 4): the same fused pair the host chain carries per group
            "    mul.f32 sft_, sft_, mv;",
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
            "    and.b32 qv, qv, 3;", // two-bit code in [0, 3]
        };

        lines.AddRange(
        [
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
            $"    bra _c{g};",
        ]);

        return string.Join("\n", [.. lines]);
    }
}
