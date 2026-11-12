using Lair.Backends.Cpu;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// Encoding-layout guard for the K-format super-blocks: every byte below is a hand-written constant derived from the
/// pinned llama.cpp b12032 reference formula, not from any LAIR-side or test-side quantizer writer. A self-consistent
/// test (LAIR's writer feeding LAIR's reader) can only prove round-trip symmetry, never that the byte layout itself is
/// the one the reference dequantizers read; these blocks pin the layout against expectations computed here from the C
/// source alone. Measurement methodology: element-by-element comparison of DequantRow output against double-precision
/// values decoded independently in this file (no LAIR code involved), plus scalar-vs-AVX2 bit equality, and hand
/// derived anchor values that are exactly representable so any layout error shows up as a huge delta rather than as
/// rounding noise. Measured margins for the Q5_K fifth-bit plane: reading it per nibble window instead of from one
/// fixed region changes block A by up to 2016 and block B by up to 188 absolute, against an assertion band of 1e-6.
/// </summary>
[Trait("Category", "Kernel")]
public class QuantLayoutTests
{
    // Reference semantics (pinned b12032), ggml-quants.c:
    //
    //   void dequantize_row_q5_K(const block_q5_K *x, float *y, int64_t k) {
    //       const uint8_t *ql = x[i].qs;
    //       const uint8_t *qh = x[i].qh;                       // one fixed plane: no window advance
    //       ...
    //       uint8_t u1 = 1, u2 = 2;
    //       for (int j = 0; j < QK_K; j += 64) {
    //           get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
    //           const float d1 = d * sc; const float m1 = min * m;
    //           get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
    //           const float d2 = d * sc; const float m2 = min * m;
    //           for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
    //           for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
    //           ql += 32; is += 2;                             // only the nibble window advances
    //           u1 <<= 2; u2 <<= 2;                           // the plane stays put, its bit mask moves
    //       }
    //   }
    //
    // and ggml-common.h:
    //
    //   typedef struct { union { ggml_half d; ggml_half dmin; }; uint8_t scales[K_SCALE_SIZE]; 
    //                    uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; } block_q5_K;   // 176 bytes
    //   typedef struct { union { ggml_half d; ggml_half dmin; }; uint8_t scales[K_SCALE_SIZE];
    //                    uint8_t qs[QK_K/2]; } block_q4_K;                       // 144 bytes
    //
    // With QK_K = 256 and K_SCALE_SIZE = 12 that fixes the byte map: d at 0..1, dmin at 2..3, twelve scale/min code
    // bytes at 4..15, the single 32-byte fifth-bit plane at 16..47 (Q5_K only), and 128 nibble bytes starting at 48
    // for Q5_K / 16 for Q4_K. Element e = j*32 + l keeps its low four bits in qs byte 48 + 32*(j/2) + l (low nibble
    // when j is even, high nibble when odd) and its fifth bit at bit j of plane byte 16 + l, so each plane byte
    // carries eight different groups' bits. Scale codes are six-bit and unpack as:
    //
    //   static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    //       if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    //       else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
    //              *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4); }
    //   }
    private const int Q5KBlockBytes = 176;
    private const int Q4KBlockBytes = 144;

    // Block A: d = half(2.0), dmin = half(0.5). Group scale/min codes (six-bit) are sc = {0, 63, 32, 5, 15, 62, 8, 63}
    // and m = {0, 1, 47, 63, 32, 16, 60, 63}, which covers all-clear (0), all-set (63) and top-bits-only (32) codes in
    // both nibble halves of the packed table: groups 0..3 read their whole code from bytes 4..7 / 8..11 while groups
    // 4..7 splice four bits out of bytes 12..15 with two bits taken from bytes 4..7 (scales) and 8..11 (mins).
    // Fifth-bit plane: l = 0..7 all set, l = 8..15 all clear, l = 16..23 alternating per-group-parity masks 0x55/0xAA,
    // l = 24..31 a single-bit ladder so every bit position is exercised at a distinct byte. Nibble codes vary across
    // groups and positions (9/6, 5/1/14, 3, 7, l&7, 12 then 1..4, 0/5/10/15, 15) so no two groups share a value.
    private static readonly byte[] Q5KBlockA = new byte[Q5KBlockBytes]
    {
            // d little-endian (half 2.0), dmin (half 0.5), then the twelve packed code bytes
            0x00, 0x40, 0x00, 0x38, 0x00, 0xFF, 0x20, 0xC5, 0x80, 0x41, 0xEF, 0xFF, 0x0F, 0x0E, 0xC8, 0xFF,
            // fifth-bit plane bytes 16..47: all-set region, all-clear region, alternating masks, single-bit ladder
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            // nibble windows at bytes 48..: group pair (0,1) then (2,3), low nibble even / high nibble odd
            0x59, 0x59, 0x59, 0x59, 0x59, 0x59, 0x59, 0x59, 0x56, 0x56, 0x56, 0x56, 0x56, 0x56, 0x56, 0x56,
            0x16, 0x16, 0x16, 0x16, 0x16, 0x16, 0x16, 0x16, 0xE6, 0xE6, 0xE6, 0xE6, 0xE6, 0xE6, 0xE6, 0xE6,
            0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73,
            0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73,
            // group pair (4,5): codes l&7 in the all-set plane region, then 12 / 1..4 with varied ladder bits
            0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
            0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
            // group pair (6,7): codes 0/5/10/15 cycling against the ladder bits, and code 15 throughout
            0xF0, 0xF5, 0xFA, 0xFF, 0xF0, 0xF5, 0xFA, 0xFF, 0xF0, 0xF5, 0xFA, 0xFF, 0xF0, 0xF5, 0xFA, 0xFF,
            0xF0, 0xF5, 0xFA, 0xFF, 0xF0, 0xF5, 0xFA, 0xFF, 0xF0, 0xF5, 0xFA, 0xFF, 0xF0, 0xF5, 0xFA, 0xFF
    };

    // Block B: d = half(-0.25), dmin = half(8.0) - negative super-block scale and a min twelve times larger than A's,
    // so the sign of the group scale and the magnitude of the subtracted shift both carry through. Group codes are
    // sc = {63, 1, 47, 32, 62, 5, 16, 0} / m = {1, 63, 16, 60, 32, 47, 8, 0}: the same value set as A but assigned to
    // different groups and different nibble halves, plus a group whose scale code is zero (group 7), which must make
    // every one of its weights equal minus its shift no matter what fifth bits are stored. Plane regions mirror A:
    // all-clear at l = 0..7, all-set at l = 8..15, rotating single bit at l = 16..23, and paired masks {3,7}/{0,4} at
    // l = 24..31. Groups j >= 2 therefore disagree with groups 0-1 in both nibbles and plane bits: a fifth-bit plane
    // read that advances per nibble window moves this block by up to 188 absolute (block A by up to 2016).
    private static readonly byte[] Q5KBlockB = new byte[Q5KBlockBytes]
    {
            0x00, 0xB4, 0x00, 0x48, 0xFF, 0x01, 0x6F, 0x20, 0x81, 0xBF, 0x10, 0x3C, 0x0E, 0xF5, 0x80, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x88, 0x11, 0x88, 0x11, 0x88, 0x11, 0x88, 0x11,
            0x1F, 0x2F, 0x3F, 0x4F, 0x5F, 0x6F, 0x1F, 0x2F, 0x32, 0x42, 0x52, 0x62, 0x13, 0x23, 0x33, 0x43,
            0x50, 0x60, 0x10, 0x20, 0x31, 0x41, 0x51, 0x61, 0x12, 0x22, 0x32, 0x42, 0x53, 0x63, 0x13, 0x23,
            0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4,
            0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4, 0xA4,
            0x08, 0x28, 0x08, 0x28, 0x08, 0x28, 0x08, 0x28, 0x08, 0x28, 0x08, 0x28, 0x08, 0x28, 0x08, 0x28,
            0x12, 0x32, 0x12, 0x32, 0x12, 0x32, 0x12, 0x32, 0x12, 0x32, 0x12, 0x32, 0x12, 0x32, 0x12, 0x32,
            0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D,
            0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D
    };

    // Q4_K block C: d = half(1.0), dmin = half(2.0), sc = {63, 0, 31, 7, 58, 16, 4, 63} / m = {0, 63, 9, 45, 32, 1,
    // 48, 15}. Same twelve-byte code table as Q5_K (bytes 4..15) but the nibble region starts at byte 16 with no
    // fifth-bit plane in front of it, so a layout mix-up here shifts every window by 32 bytes. Codes cover all sixteen
    // four-bit values in both nibble halves and include an all-zero group whose weights must land on minus its shift.
    private static readonly byte[] Q4KBlockC = new byte[Q4KBlockBytes]
    {
            0x00, 0x3C, 0x00, 0x40, 0xFF, 0x40, 0x1F, 0xC7, 0x80, 0x3F, 0xC9, 0x2D, 0x0A, 0x10, 0x04, 0xFF,
            0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87, 0x70, 0xF1, 0xE2, 0xD3, 0xC4, 0xB5, 0xA6, 0x97,
            0x80, 0x71, 0xF2, 0xE3, 0xD4, 0xC5, 0xB6, 0xA7, 0x90, 0x81, 0x72, 0xF3, 0xE4, 0xD5, 0xC6, 0xB7,
            0x80, 0x91, 0xA2, 0xB3, 0xC0, 0xD1, 0xE2, 0xF3, 0x80, 0x91, 0xA2, 0xB3, 0xC0, 0xD1, 0xE2, 0xF3,
            0x80, 0x91, 0xA2, 0xB3, 0xC0, 0xD1, 0xE2, 0xF3, 0x80, 0x91, 0xA2, 0xB3, 0xC0, 0xD1, 0xE2, 0xF3,
            0xB0, 0xB0, 0xB0, 0xB0, 0xB0, 0xB0, 0xB0, 0xB0, 0xB0, 0xB0, 0xB0, 0xB0, 0xB0, 0xB0, 0xB0, 0xB0,
            0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
            0xE0, 0xE1, 0xE0, 0xE1, 0xE0, 0xE1, 0xE0, 0xE1, 0xE2, 0xE3, 0xE2, 0xE3, 0xE2, 0xE3, 0xE2, 0xE3,
            0xE4, 0xE5, 0xE4, 0xE5, 0xE4, 0xE5, 0xE4, 0xE5, 0xE6, 0xE7, 0xE6, 0xE7, 0xE6, 0xE7, 0xE6, 0xE7
    };

    [Fact]
    public void DequantRow_Q5K_ReferenceEncodedSuperBlocks_BothPathsMatchReferenceDecode()
    {
        byte[] packed = new byte[2 * Q5KBlockBytes];
        Q5KBlockA.CopyTo(packed, 0);
        Q5KBlockB.CopyTo(packed, Q5KBlockBytes);

        TensorShape shape = MatrixShape(2, 256); // one reference block per row: the stride math is part of the guard
        using CpuBackend scalar = new(threads: 1, allowAvx2: false);
        using CpuBackend avx2 = new(threads: 1, allowAvx2: true);
        WeightRef w = new(packed, shape, QuantFormat.Q5_K);

        double[] expectedA = ReferenceQ5KSuperBlock(Q5KBlockA);
        double[] expectedB = ReferenceQ5KSuperBlock(Q5KBlockB);

        AssertGroupCodesMatchDocumentedTable(expectedA, Q5KBlockA, sc: [0, 63, 32, 5, 15, 62, 8, 63], m: [0, 1, 47, 63, 32, 16, 60, 63]);
        AssertGroupCodesMatchDocumentedTable(expectedB, Q5KBlockB, sc: [63, 1, 47, 32, 62, 5, 16, 0], m: [1, 63, 16, 60, 32, 47, 8, 0]);

        float[] actualScalar = new float[256];
        float[] actualAvx2 = new float[256];

        for (int row = 0; row < 2; row++)
        {
            double[] expected = row == 0 ? expectedA : expectedB;
            scalar.DequantRow(w, row, actualScalar);
            avx2.DequantRow(w, row, actualAvx2);

            // Every value in these blocks is dyadic (d, dmin and the products all land on exact binary fractions), so
            // the reference decode reproduces the float result to well under a ulp. The band below is three orders of
            // magnitude tighter than any layout error: advancing the fifth-bit plane per nibble window moves block A
            // by up to 2016 and block B by up to 188 absolute, groups j >= 2 onward.
            AssertExact(expected, actualScalar, $"scalar Q5_K reference block {row}");
            AssertExact(expected, actualAvx2, $"avx2 Q5_K reference block {row}");
            Assert.True(actualScalar.AsSpan().SequenceEqual(actualAvx2.AsSpan()), $"Q5_K reference block {row}: avx2 DequantRow diverges from scalar");
        }

        // A row of two blocks in one super-block-sized pair must also expand correctly when both blocks share a row.
        TensorShape wide = MatrixShape(1, 512);
        float[] wideOut = new float[512];
        avx2.DequantRow(new WeightRef(packed, wide, QuantFormat.Q5_K), 0, wideOut);
        AssertExact(expectedA, wideOut.AsSpan(0, 256), "Q5_K block A inside a two-block row");
        AssertExact(expectedB, wideOut.AsSpan(256, 256), "Q5_K block B inside a two-block row");
    }

    [Fact]
    public void DequantRow_Q5K_HandComputedAnchorValues_AreBitExact()
    {
        // Anchors derived by hand from the reference formula, one per interesting case:
        //   A[0]:   group 0 has scale code 0, so five = 9 + 16 (plane byte 0xFF sets bit 0) still yields exactly 0.
        //   A[39]:  group 1 (odd nibble), l = 7: high nibble of window byte 48+7 = 0x59 is 5, plane index 7 (block byte
        //           23) is 0xFF so bit 1 is set -> five = 21; d*sc = 2*63 = 126, min*m = 0.5*1 -> 126*21 - 0.5.
        //   A[99]:  group 3 (odd), l = 3: nibble 7 from window byte 48+32+3 = 0x73, plane index 3 = 0xFF sets bit 3 ->
        //           five = 23; d*sc = 2*5 = 10, min*m = 0.5*63 = 31.5 -> 230 - 31.5.
        //   A[239]: group 7, l = 15: nibble 15 from window byte 48+96+15 = 0xFF, plane index 15 = 0x00 clears bit 7 ->
        //           five = 15; d*sc = 2*63 = 126, min*m = 31.5 -> 1890 - 31.5 (this is exactly the value a windowed
        //           plane read produces for element 255 instead).
        //   A[255]: group 7, l = 31: nibble 15 from window byte 48+96+31, plane index 31 = 0x80 sets bit 7 -> five = 31;
        //           d*sc = 126, min*m = 31.5 -> 126*31 - 31.5 (the block maximum).
        //   A[191]: group 5, l = 31: nibble 4 from window byte 48+64+31 = 0x47, plane index 31 = 0x80 leaves bit 5
        //           clear -> five = 4; d*sc = 2*62 = 124, min*m = 0.5*16 = 8 -> 496 - 8 (the all-clear-bit case).
        //   B[0]:   group 0 with negative super-block scale: nibble 15 from window byte 48+0 = 0x1F, plane index 0 =
        //           0x00 clears bit 0 -> five = 15; d*sc = -0.25*63 = -15.75, min*m = 8*1 = 8 -> -236.25 - 8.
        //   B[8]:   group 0, l = 8: the stored low four bits are (8 >> 2) & 3 = 2, read from window byte 48+8 = block
        //           byte 56 = 0x32; plane index 8 = 0xFF sets bit 0 -> five = 18; d*sc = -15.75, min*m = 8 -> -291.5.
        //   B[192]: group 6 whose scale code is 16 and whose fifth bit is clear: nibble 13 from window byte 48+96,
        //           plane index 0 = 0x00 -> five = 13; d*sc = -4, min*m = 8*8 = 64 -> -52 - 64.
        (int Index, float Expected)[] anchorsA = [(0, 0f), (39, 2645.5f), (99, 198.5f), (191, 488f), (239, 1858.5f), (255, 3874.5f)];
        (int Index, float Expected)[] anchorsB = [(0, -244.25f), (8, -291.5f), (192, -116f)];

        byte[] packedA = Q5KBlockA;
        byte[] packedB = Q5KBlockB;

        using CpuBackend scalar = new(threads: 1, allowAvx2: false);
        using CpuBackend avx2 = new(threads: 1, allowAvx2: true);

        float[] outScalar = new float[256];
        float[] outAvx2 = new float[256];

        scalar.DequantRow(new WeightRef(packedA, MatrixShape(1, 256), QuantFormat.Q5_K), 0, outScalar);
        avx2.DequantRow(new WeightRef(packedA, MatrixShape(1, 256), QuantFormat.Q5_K), 0, outAvx2);

        foreach ((int index, float expected) in anchorsA)
        {
            Assert.Equal(expected, outScalar[index]);
            Assert.Equal(expected, outAvx2[index]);
        }

        scalar.DequantRow(new WeightRef(packedB, MatrixShape(1, 256), QuantFormat.Q5_K), 0, outScalar);
        avx2.DequantRow(new WeightRef(packedB, MatrixShape(1, 256), QuantFormat.Q5_K), 0, outAvx2);

        foreach ((int index, float expected) in anchorsB)
        {
            Assert.Equal(expected, outScalar[index]);
            Assert.Equal(expected, outAvx2[index]);
        }
    }

    [Fact]
    public void DequantRow_Q4K_ReferenceEncodedSuperBlock_BothPathsMatchReferenceDecode()
    {
        TensorShape shape = MatrixShape(1, 256);
        using CpuBackend scalar = new(threads: 1, allowAvx2: false);
        using CpuBackend avx2 = new(threads: 1, allowAvx2: true);
        WeightRef w = new(Q4KBlockC, shape, QuantFormat.Q4_K);

        double[] expected = ReferenceQ4KSuperBlock(Q4KBlockC);
        AssertGroupCodesMatchDocumentedTable(expected, Q4KBlockC, sc: [63, 0, 31, 7, 58, 16, 4, 63], m: [0, 63, 9, 45, 32, 1, 48, 15]);

        float[] actualScalar = new float[256];
        float[] actualAvx2 = new float[256];
        scalar.DequantRow(w, 0, actualScalar);
        avx2.DequantRow(w, 0, actualAvx2);

        AssertExact(expected, actualScalar, "scalar Q4_K reference block");
        AssertExact(expected, actualAvx2, "avx2 Q4_K reference block");
        Assert.True(actualScalar.AsSpan().SequenceEqual(actualAvx2.AsSpan()), "Q4_K reference block: avx2 DequantRow diverges from scalar");

        // Hand anchors for the same reason as above: C[0] = 1*63*0 - 2*0 (code (l&7)|((l>>2)&8) is 0 at l = 0),
        // C[15] = 1*63*7 - 2*0 with code 7|0, C[32] = group 1 whose scale code is zero -> exactly minus its shift
        // -(2*63) whatever nibble is stored.
        Assert.Equal(0f, actualScalar[0]);
        Assert.Equal(63f, actualScalar[1]);
        Assert.Equal(441f, actualScalar[15]);
        Assert.Equal(-126f, actualScalar[32]);
    }

    private static void AssertGroupCodesMatchDocumentedTable(double[] expectedValues, ReadOnlySpan<byte> block, ReadOnlySpan<int> sc, ReadOnlySpan<int> m)
    {
        // Self-validation of the written constants: unpacking them with get_scale_min_k4 semantics must reproduce the
        // documented code table, otherwise a mistyped constant would silently redefine what the test asserts.
        for (int j = 0; j < 8; j++)
        {
            (byte scaleCode, byte minCode) = GetScaleMinK4(j, block.Slice(4, 12));
            Assert.Equal(sc[j], (int)scaleCode);
            Assert.Equal(m[j], (int)minCode);

            // A group whose scale code and min code are both zero must dequantize to exactly zero for all 32 elements,
            // whatever nibbles and fifth bits its bytes hold.
            if (sc[j] == 0 && m[j] == 0)
            {
                for (int l = 0; l < 32; l++)
                {
                    Assert.Equal(0d, expectedValues[j * 32 + l]);
                }
            }
        }
    }

    // get_scale_min_k4 from the pinned reference: six-bit codes for groups 0..3 sit whole in bytes j and j+4; groups
    // 4..7 splice four bits out of byte j+4 with two bits taken from byte j-4 (scale) and byte j (min).
    private static (byte Scale, byte Min) GetScaleMinK4(int j, ReadOnlySpan<byte> scales)
    {
        if (j < 4)
        {
            return ((byte)(scales[j] & 63), (byte)(scales[j + 4] & 63));
        }

        byte scale = (byte)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
        byte min = (byte)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
        return (scale, min);
    }

    private static double HalfLE(ReadOnlySpan<byte> b, int offset) =>
        (float)BitConverter.Int16BitsToHalf((short)(b[offset] | (b[offset + 1] << 8)));

    // Q5_K row expansion transcribed from dequantize_row_q5_K: ql advances with the nibble window, qh never does and
    // only u1/u2 move. Written against struct offsets (qh at 16, qs at 48) rather than LAIR's kernel code.
    private static double[] ReferenceQ5KSuperBlock(ReadOnlySpan<byte> block)
    {
        if (block.Length != Q5KBlockBytes)
        {
            throw new ArgumentException("Q5_K super-block must be 176 bytes.", nameof(block));
        }

        double d = HalfLE(block, 0);
        double min = HalfLE(block, 2);
        ReadOnlySpan<byte> scales = block.Slice(4, 12);
        ReadOnlySpan<byte> qh = block.Slice(16, QK_K / 8); // the one fixed fifth-bit plane
        ReadOnlySpan<byte> qs = block.Slice(48, QK_K / 2);

        double[] values = new double[QK_K];
        int groupIndex = 0;
        byte u1 = 1, u2 = 2;

        for (int jBase = 0; jBase < QK_K; jBase += 64)
        {
            (byte scLo, byte mLo) = GetScaleMinK4(groupIndex, scales);
            (byte scHi, byte mHi) = GetScaleMinK4(groupIndex + 1, scales);
            double d1 = d * scLo, m1 = min * mLo;
            double d2 = d * scHi, m2 = min * mHi;

            int window = 32 * (jBase / 64); // ql += 32 per iteration in the reference loop

            for (int l = 0; l < 32; ++l)
            {
                values[jBase + l] = d1 * ((qs[window + l] & 0xF) + ((qh[l] & u1) != 0 ? 16 : 0)) - m1;
            }

            for (int l = 0; l < 32; ++l)
            {
                values[jBase + 32 + l] = d2 * ((qs[window + l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) - m2;
            }

            groupIndex += 2;
            u1 <<= 2;
            u2 <<= 2;
        }

        return values;
    }

    // Q4_K row expansion transcribed from dequantize_row_q4_K: same scale/min unpacking, nibble windows at qs + 32*p.
    private static double[] ReferenceQ4KSuperBlock(ReadOnlySpan<byte> block)
    {
        if (block.Length != Q4KBlockBytes)
        {
            throw new ArgumentException("Q4_K super-block must be 144 bytes.", nameof(block));
        }

        double d = HalfLE(block, 0);
        double min = HalfLE(block, 2);
        ReadOnlySpan<byte> scales = block.Slice(4, 12);
        ReadOnlySpan<byte> qs = block.Slice(16, QK_K / 2); // no fifth-bit plane in this format

        double[] values = new double[QK_K];
        int groupIndex = 0;

        for (int jBase = 0; jBase < QK_K; jBase += 64)
        {
            (byte scLo, byte mLo) = GetScaleMinK4(groupIndex, scales);
            (byte scHi, byte mHi) = GetScaleMinK4(groupIndex + 1, scales);

            for (int l = 0; l < 32; ++l)
            {
                values[jBase + l] = d * scLo * (qs[32 * (jBase / 64) + l] & 0xF) - min * mLo;
            }

            for (int l = 0; l < 32; ++l)
            {
                values[jBase + 32 + l] = d * scHi * (qs[32 * (jBase / 64) + l] >> 4) - min * mHi;
            }

            groupIndex += 2;
        }

        return values;
    }

    private static void AssertExact(ReadOnlySpan<double> expected, ReadOnlySpan<float> actual, string context)
    {
        for (int i = 0; i < actual.Length; i++)
        {
            double diff = Math.Abs(actual[i] - expected[i]);

            // The blocks are built so every weight is an exact binary fraction: the kernels and this reference decode
            // compute identical values, so any real difference can only come from a layout error.
            Assert.True(diff <= 1e-6, $"{context}: output[{i}] differs by {diff:G9} (expected {expected[i]:G9}, got {actual[i]:G9})");
        }
    }

    private const int QK_K = 256;

    private static TensorShape MatrixShape(int rows, int cols) => new([rows, cols]);
}
