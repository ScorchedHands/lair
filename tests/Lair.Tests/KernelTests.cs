using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using Lair.Backends.Cpu;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// Dense CPU kernel tests: test-only block quantizers feed scalar, AVX2, and threaded GEMV paths. A naive
/// double-precision dequant-and-dot oracle checks the float kernels; parallel runs must be bit-identical to a
/// single-threaded run because every output row is computed independently.
/// </summary>
[Trait("Category", "Kernel")]
public class KernelTests
{
    private static readonly (int Rows, int Cols)[] MatrixSizes = [(100, 137), (256, 256), (512, 1024)];

    [Theory]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ4_XS)]
    public void Gemv_Scalar_MatchesDoubleOracle(QuantFormat format)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            (float[] x, byte[] weights) = CreateCase(rows, cols, format, seed: 1234);

            using CpuBackend backend = new(threads: 1, allowAvx2: false);
            float[] output = new float[rows];
            backend.Gemv(x, new WeightRef(weights, MatrixShape(rows, cols), format), output);

            (double[] expected, double[] magnitudes) = OracleDot(x, weights, rows, cols, format);
            AssertClose(output, expected, magnitudes, relativeTolerance: 1e-4, $"scalar {format} GEMV at {rows}x{cols}");
        }
    }

    [Theory]
    [InlineData(QuantFormat.F16)]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ4_XS)]
    public void Gemv_Avx2_MatchesScalarWithinTightTolerance(QuantFormat format)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            (float[] x, byte[] weights) = CreateCase(rows, cols, format, seed: 987);

            using CpuBackend scalar = new(threads: 1, allowAvx2: false);
            using CpuBackend avx2 = new(threads: 1, allowAvx2: true);

            float[] expected = new float[rows];
            float[] actual = new float[rows];
            WeightRef w = new(weights, MatrixShape(rows, cols), format);
            scalar.Gemv(x, w, expected);
            avx2.Gemv(x, w, actual);

            // Bound the difference by eps-like accumulation scale: dot-product reassociation noise grows with sum|x*w|,
            // not with a near-zero output value. The oracle magnitudes provide that scale without sharing kernel code.
            (double[] _, double[] magnitudes) = OracleDot(x, weights, rows, cols, format);
            AssertClose(actual, expected, relativeTolerance: 1e-5, $"AVX2 vs scalar {format} GEMV at {rows}x{cols}", magnitudes);

            static void AssertClose(ReadOnlySpan<float> a, ReadOnlySpan<float> b, double relativeTolerance, string context, ReadOnlySpan<double> scale)
            {
                for (int i = 0; i < a.Length; i++)
                {
                    double diff = Math.Abs(a[i] - b[i]);
                    Assert.True(diff <= relativeTolerance * Math.Max(1e-6, scale[i]), $"{context}: output[{i}] differs by {diff:G6} (scalar {b[i]:G9}, avx2 {a[i]:G9})");
                }
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.F16)]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ4_XS)]
    public void Gemv_Parallel_IsBitIdenticalToSingleThreaded(QuantFormat format)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            (float[] x, byte[] weights) = CreateCase(rows, cols, format, seed: 55);

            using CpuBackend single = new(threads: 1);
            using CpuBackend parallel = new(threads: 4);

            float[] expected = new float[rows];
            float[] actual = new float[rows];
            WeightRef w = new(weights, MatrixShape(rows, cols), format);
            single.Gemv(x, w, expected);
            parallel.Gemv(x, w, actual);

            Assert.True(expected.AsSpan().SequenceEqual(actual.AsSpan()), $"threaded {format} GEMV at {rows}x{cols} must be bit-identical");
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ4_XS)]
    public void Gemv_Avx2_MatchesDoubleOracle(QuantFormat format)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            (float[] x, byte[] weights) = CreateCase(rows, cols, format, seed: 7);

            using CpuBackend backend = new(threads: 1, allowAvx2: true);
            float[] output = new float[rows];
            backend.Gemv(x, new WeightRef(weights, MatrixShape(rows, cols), format), output);

            (double[] expected, double[] magnitudes) = OracleDot(x, weights, rows, cols, format);
            AssertClose(output, expected, magnitudes, relativeTolerance: 1e-4, $"AVX2 {format} GEMV at {rows}x{cols}");
        }
    }

    [Fact]
    public void Dot_MatchesNaiveLoop()
    {
        Random rng = new(31);
        foreach (int count in new[] { 0, 7, 8, 9, 64, 257 })
        {
            float[] a = NextFloats(rng, count);
            float[] b = NextFloats(rng, count);

            using CpuBackend scalar = new(threads: 1, allowAvx2: false);
            using CpuBackend avx2 = new(threads: 1, allowAvx2: true);

            double expected = 0;
            double magnitude = 0;
            for (int i = 0; i < count; i++)
            {
                expected += (double)a[i] * b[i];
                magnitude += Math.Abs((double)a[i] * b[i]);
            }

            AssertClose([scalar.Dot(a, b)], [expected], [magnitude], relativeTolerance: 1e-5, $"Dot({count}) scalar");
            AssertClose([avx2.Dot(a, b)], [expected], [magnitude], relativeTolerance: 1e-5, $"Dot({count}) avx2");

            Assert.Throws<ArgumentException>(() => avx2.Dot(a, new float[count + 1]));
        }
    }

    [Fact]
    public void Axpy_Fill_Scale_MatchNaiveLoops()
    {
        Random rng = new(47);
        foreach (int count in new[] { 0, 3, 8, 9, 128 })
        {
            float[] x = NextFloats(rng, count);

            using CpuBackend scalar = new(threads: 1, allowAvx2: false);
            using CpuBackend avx2 = new(threads: 1, allowAvx2: true);

            foreach (bool useAvx in new[] { false, true })
            {
                CpuBackend backend = useAvx ? avx2 : scalar;

                float[] yNaive = NextFloats(rng, count);
                float[] yKernel = new float[count];
                yNaive.CopyTo(yKernel, 0);

                float alpha = -1.5f;
                backend.Axpy(alpha, x, yKernel);
                for (int i = 0; i < count; i++)
                {
                    yNaive[i] += alpha * x[i];
                }

                if (!useAvx)
                {
                    Assert.True(yNaive.AsSpan().SequenceEqual(yKernel.AsSpan()), $"Axpy({count}) scalar must be bit-identical to naive loop");
                }
                else
                {
                    // The AVX2 path fuses multiply and add in one rounding step, so compare with a tiny tolerance.
                    for (int i = 0; i < count; i++)
                    {
                        Assert.True(Math.Abs(yKernel[i] - yNaive[i]) <= 1e-6 * Math.Max(1.0, Math.Abs(yNaive[i])), $"Axpy({count}) avx2 drifts at [{i}]");
                    }

                    // Re-seed the kernel buffer from naive results so Fill/Scale run on identical inputs for both paths.
                    yNaive.CopyTo(yKernel, 0);
                }

                backend.Fill(yKernel, 3.25f);
                foreach (float v in yKernel)
                {
                    Assert.Equal(3.25f, v);
                }

                float s = 0.75f;
                backend.Scale(yKernel, s);
                for (int i = 0; i < count; i++)
                {
                    Assert.Equal(3.25f * s, yKernel[i]); // single multiply on both paths: bit-identical
                }

                Assert.Throws<ArgumentException>(() => backend.Axpy(1f, x, new float[count + 1]));
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.F16)]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ4_XS)]
    public void WeightRef_WrongByteLength_ThrowsClearMessage(QuantFormat format)
    {
        TensorShape shape = MatrixShape(4, 70);
        long expected = 4 * shape.RowStrideBytes(format);

        using CpuBackend backend = new();
        byte[] tooShort = new byte[expected - 1];
        float[] x = new float[70];
        float[] output = new float[4];

        ArgumentException ex = Assert.Throws<ArgumentException>(() => backend.Gemv(x, new WeightRef(tooShort, shape, format), output));
        Assert.Contains("requires", ex.Message);
        Assert.Contains(expected.ToString(), ex.Message);

        byte[] tooLong = new byte[expected + 1];
        Assert.Throws<ArgumentException>(() => backend.Gemv(x, new WeightRef(tooLong, shape, format), output));

        // x length and output length are validated against the matrix dimensions as well.
        byte[] exact = new byte[expected];
        Assert.Throws<ArgumentException>(() => backend.Gemv(new float[69], new WeightRef(exact, shape, format), output));
        Assert.Throws<ArgumentException>(() => backend.Gemv(x, new WeightRef(exact, shape, format), new float[3]));
    }

    [Theory]
    [InlineData(QuantFormat.F16)]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ4_XS)]
    public void DequantRow_MatchesDoubleOracleAcrossFormats(QuantFormat format)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            (float[] _, byte[] weights) = CreateCase(rows, cols, format, seed: 4242);

            using CpuBackend scalar = new(threads: 1, allowAvx2: false);
            using CpuBackend avx2 = new(threads: 1, allowAvx2: true);
            WeightRef w = new(weights, MatrixShape(rows, cols), format);

            float[] actualScalar = new float[cols];
            float[] actualAvx2 = new float[cols];

            foreach (int row in new[] { 0, rows / 2, rows - 1 })
            {
                double[] expected = OracleDequantRow(weights, row, cols, format);
                scalar.DequantRow(w, row, actualScalar);
                avx2.DequantRow(w, row, actualAvx2);

                AssertValuesClose(actualScalar, expected, relativeTolerance: 1e-4, $"scalar {format} DequantRow({row}) at {rows}x{cols}");
                AssertValuesClose(actualAvx2, expected, relativeTolerance: 1e-4, $"avx2 {format} DequantRow({row}) at {rows}x{cols}");

                // The AVX2 fused scale/shift is bit-compatible with the scalar unpack except for fma rounding on
                // Q4_K group values; keep both paths within a tiny band of each other.
                Assert.True(actualScalar.AsSpan().SequenceEqual(actualAvx2.AsSpan()) || MaxAbsDiff(actualScalar, actualAvx2) <= 1e-6 * MaxAbs(expected), $"DequantRow({row}) {format}: avx2 diverges from scalar");
            }

            // Dequantized rows must feed GEMV identically to the packed path (embedding lookups rely on this).
            foreach ((float[] x, byte[] weightsG) in new[] { CreateCase(rows, cols, format, seed: 4242) })
            {
                Assert.Equal(weights, weightsG); // same seed must reproduce the same packed image

                float[] gemvOut = new float[rows];
                avx2.Gemv(x, w, gemvOut);

                (double[] oracle, double[] magnitudes) = OracleDot(x, weights, rows, cols, format);
                AssertClose(gemvOut, oracle, magnitudes, relativeTolerance: 1e-4, $"packed vs {format} GEMV at {rows}x{cols}");

                float[] row0 = new float[cols];
                avx2.DequantRow(w, 0, row0);
                double manualDot = 0;
                for (int i = 0; i < cols; i++)
                {
                    manualDot += (double)x[i] * row0[i];
                }

                Assert.True(Math.Abs(gemvOut[0] - manualDot) <= 1e-4 * Math.Max(1e-6, magnitudes[0]), $"{format}: Gemv(row0) vs dequant-row dot mismatch");
            }
        }
    }

    [Fact]
    public void DequantRow_F32_CopiesExactFloats()
    {
        const int rows = 5;
        const int cols = 71; // odd tail: no padding absorbs a wide load in any vector path
        TensorShape shape = MatrixShape(rows, cols);
        byte[] packed = new byte[rows * (long)shape.RowStrideBytes(QuantFormat.F32)];

        Random rng = new(808);
        float[] source = NextFloats(rng, rows * cols);
        source.CopyTo(MemoryMarshal.Cast<byte, float>(packed.AsSpan()));

        using CpuBackend backend = new();
        WeightRef w = new(packed, shape, QuantFormat.F32);
        float[] outRow = new float[cols];

        for (int row = 0; row < rows; row++)
        {
            backend.DequantRow(w, row, outRow);
            Assert.True(outRow.AsSpan().SequenceEqual(source.AsSpan(row * cols, cols)), $"F32 DequantRow({row}) must be bit-exact");
        }

        // 1-D tensors (norm weights) are addressed as a single-row matrix.
        float[] norm = NextFloats(rng, cols);
        byte[] packedNorm = new byte[cols * 4];
        norm.CopyTo(MemoryMarshal.Cast<byte, float>(packedNorm.AsSpan()));
        WeightRef w1 = new(packedNorm, MatrixShape(1, cols), QuantFormat.F32);

        float[] outNorm = new float[cols];
        backend.DequantRow(w1, 0, outNorm);
        Assert.True(outNorm.AsSpan().SequenceEqual(norm.AsSpan()), "F32 single-row DequantRow must be bit-exact");
    }

    [Theory]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ3_XXS)]
    public void DequantRow_SyntheticSuperBlocks_BothPathsMatchHandWrittenBytes(QuantFormat format)
    {
        const int cols = 512; // two full super-blocks in one row: no padded tail absorbs a wide load
        byte[] packed = BuildSyntheticSuperBlocks(format);
        TensorShape shape = MatrixShape(1, cols);

        using CpuBackend scalar = new(threads: 1, allowAvx2: false);
        using CpuBackend avx2 = new(threads: 1, allowAvx2: true);
        WeightRef w = new(packed, shape, format);

        float[] actualScalar = new float[cols];
        float[] actualAvx2 = new float[cols];
        scalar.DequantRow(w, 0, actualScalar);
        avx2.DequantRow(w, 0, actualAvx2);

        double[] expected = OracleDequantRow(packed, 0, cols, format);
        AssertValuesClose(actualScalar, expected, relativeTolerance: 1e-6, $"scalar {format} synthetic super-blocks");
        AssertValuesClose(actualAvx2, expected, relativeTolerance: 1e-6, $"avx2 {format} synthetic super-blocks");

        // Both paths unpack the same bytes with the same arithmetic order, so the expansion must match bit-for-bit.
        Assert.True(actualScalar.AsSpan().SequenceEqual(actualAvx2.AsSpan()), $"{format}: avx2 DequantRow diverges from scalar on synthetic blocks");
    }

    // Hand-written two-block images: zero and maximum group scales/mins, every 2-bit and 3-bit code value, both
    // hmask states, and scale codes spanning the whole six-bit range, so group windows, scale unpacking, and the
    // Q3_K bias arithmetic all take part.
    private static byte[] BuildSyntheticSuperBlocks(QuantFormat format)
    {
        if (format == QuantFormat.Q2_K)
        {
            byte[] packed = new byte[2 * 84];

            // Block 0: d = 1.0, dmin = 0.5. Scale bytes cover zero/zero, max/zero, zero/max, max/max, then varied.
            WriteLE(packed, 80, (ushort)BitConverter.HalfToInt16Bits((Half)1f));
            WriteLE(packed, 82, (ushort)BitConverter.HalfToInt16Bits((Half)0.5f));
            for (int g = 0; g < 16; g++)
            {
                int sc = g switch { 0 => 0, 1 => 15, 2 => 0, 3 => 15, _ => (g * 5) & 0xF };
                int m = g switch { 0 => 0, 1 => 0, 2 => 15, 3 => 15, _ => (g * 11) & 0xF };
                packed[g] = (byte)(sc | (m << 4));
            }

            // Four 16-byte group-pair windows: all codes 3, all codes 0, all codes 1, and a per-shift mix.
            for (int e = 0; e < 16; e++)
            {
                packed[16 + e] = 0xFF; // 2-bit codes 3 for the four even-k groups of half 0
                packed[32 + e] = 0x00; // codes 0 for the four odd-k groups of half 0
                packed[48 + e] = 0x55; // codes 1 for the four even-k groups of half 1
                packed[64 + e] = 0xA5; // codes 1, 0, 2, 2 across the shifts of half 1
            }

            // Block 1: d = 0 makes every weight zero whatever the scale nibble; dmin nonzero must not leak in.
            WriteLE(packed, 84 + 80, (ushort)BitConverter.HalfToInt16Bits((Half)0f));
            WriteLE(packed, 84 + 82, (ushort)BitConverter.HalfToInt16Bits((Half)3f));
            for (int g = 0; g < 16; g++)
            {
                packed[84 + g] = 0xFF; // max scale and max min with d = 0: all weights must dequantize to zero
            }

            for (int e = 0; e < 64; e++)
            {
                packed[84 + 16 + e] = (byte)(e & 0xFF); // varied 2-bit codes across every shift position
            }

            return packed;
        }

        if (format == QuantFormat.IQ2_XXS)
        {
            byte[] packed = new byte[2 * 66];

            // Block 0: d = 1.0. Each sub-block's four-bit scale walks the whole range and its four seven-bit sign-group
            // codes select varied rows of the shared sign table (including code 0 at ib = 0). The four grid indices of a
            // sub-block land on different entries so lookups cover the whole table.
            WriteLE(packed, 0, (ushort)BitConverter.HalfToInt16Bits((Half)1f));
            for (int ib = 0; ib < 8; ib++)
            {
                uint grid = 0;
                uint signsAux = (uint)(ib * 2 & 0xF) << 28; // scale codes 0, 2, ... 14 across the eight sub-blocks
                for (int l = 0; l < 4; l++)
                {
                    grid |= (uint)((ib * 61 + l * 37) & 0xFF) << (8 * l); // varied grid indices across the whole entry range
                    signsAux |= (uint)((l + 4 * ib) & 127) << (7 * l); // sign-group codes cover varied table rows, including code 0 at ib = 0, l = 0
                }

                WriteLE(packed, 2 + 8 * ib, (ushort)(grid & 0xFFFF)); // words 0..1 of the aux region carry the four grid indices
                WriteLE(packed, 2 + 8 * ib + 2, (ushort)(grid >> 16));
                WriteLE(packed, 2 + 8 * ib + 4, (ushort)(signsAux & 0xFFFF)); // words 2..3 carry the scale code and sign codes
                WriteLE(packed, 2 + 8 * ib + 6, (ushort)(signsAux >> 16));
            }

            // Block 1: d = 0 makes every weight zero whatever the scale, sign, or grid bits; maxed-out bits must not leak
            // in. Grid index 255 is the last entry of the table.
            WriteLE(packed, 66, (ushort)BitConverter.HalfToInt16Bits((Half)0f));
            for (int ib = 0; ib < 8; ib++)
            {
                WriteLE(packed, 66 + 2 + 8 * ib, 0xFFFF); // every group reads the last grid entry
                WriteLE(packed, 66 + 2 + 8 * ib + 2, 0xFFFF);
                WriteLE(packed, 66 + 2 + 8 * ib + 4, 0xFFFF); // max scale and all-ones sign codes
                WriteLE(packed, 66 + 2 + 8 * ib + 6, 0xFFFF);
            }

            return packed;
        }

        if (format == QuantFormat.IQ2_XS)
        {
            byte[] packed = new byte[2 * 74];

            // Block 0: d = 1.0. Scale bytes pair a low-nibble code with a high-nibble code so both sub-block scales take
            // part for every sub-block, covering the whole four-bit range; grid indices vary across the whole 512-entry
            // range and the seven-bit sign codes select varied table rows (including code 0 at ib = 0, l = 0).
            WriteLE(packed, 0, (ushort)BitConverter.HalfToInt16Bits((Half)1f));
            for (int ib = 0; ib < 8; ib++)
            {
                packed[66 + ib] = (byte)(((ib * 3) & 0xF) | (((15 - ib) & 0xF) << 4)); // low nibble 0, 3, ... and high nibble 15, 14, ...
                for (int l = 0; l < 4; l++)
                {
                    int gi = (ib * 128 + l * 37) & 511; // varied grid indices across the whole entry range
                    int signCode = (l + 4 * ib) & 127; // varied sign-group codes, including code 0 at ib = 0, l = 0
                    WriteLE(packed, 2 + 8 * ib + 2 * l, (ushort)(gi | (signCode << 9)));
                }
            }

            // Block 1: d = 0 makes every weight zero whatever the scale, sign, or grid bits; maxed-out bits must not leak
            // in. Grid index 511 is the last entry and sign index 127 the last row of the shared sign table.
            WriteLE(packed, 74, (ushort)BitConverter.HalfToInt16Bits((Half)0f));
            for (int ib = 0; ib < 8; ib++)
            {
                packed[74 + 66 + ib] = 0xFF; // max scale codes in both nibbles
                for (int l = 0; l < 4; l++)
                {
                    WriteLE(packed, 74 + 2 + 8 * ib + 2 * l, (ushort)(511 | (127 << 9))); // grid entry 511 with sign row 127
                }
            }

            return packed;
        }

        if (format == QuantFormat.IQ3_XXS)
        {
            byte[] packed = new byte[2 * 98];

            // Block 0: d = 1.0. Each sub-block's four-bit scale walks the whole range, and its four seven-bit sign-group
            // codes select varied rows of the shared sign table (including code 0 at ib = 0). Grid indices vary across
            // the full entry range so lookups land on different entries for every group.
            WriteLE(packed, 0, (ushort)BitConverter.HalfToInt16Bits((Half)1f));
            for (int ib = 0; ib < 8; ib++)
            {
                uint aux = ((uint)(ib * 2 & 0xF) << 28); // scale codes 0, 2, ... 14 across the eight sub-blocks
                for (int l = 0; l < 4; l++)
                {
                    aux |= (uint)((l + 4 * ib) & 127) << (7 * l); // sign-group codes cover varied table rows, including code 0 at ib = 0, l = 0
                }

                packed[66 + 4 * ib] = (byte)aux;
                packed[67 + 4 * ib] = (byte)(aux >> 8);
                packed[68 + 4 * ib] = (byte)(aux >> 16);
                packed[69 + 4 * ib] = (byte)(aux >> 24);

                for (int g = 0; g < 8; g++)
                {
                    packed[2 + 8 * ib + g] = (byte)((ib * 37 + g * 13) & 0xFF); // varied grid indices across the whole entry range
                }
            }

            // Block 1: d = 0 makes every weight zero whatever the scale, sign, or grid bits; maxed-out bits must not leak in.
            WriteLE(packed, 98, (ushort)BitConverter.HalfToInt16Bits((Half)0f));
            for (int ib = 0; ib < 8; ib++)
            {
                uint aux = ((uint)15 << 28) | 0x0FFFFFFF; // max scale and all-ones sign codes
                packed[98 + 66 + 4 * ib] = (byte)aux;
                packed[98 + 67 + 4 * ib] = (byte)(aux >> 8);
                packed[98 + 68 + 4 * ib] = (byte)(aux >> 16);
                packed[98 + 69 + 4 * ib] = (byte)(aux >> 24);

                for (int g = 0; g < 8; g++)
                {
                    packed[98 + 2 + 8 * ib + g] = 255; // every group reads the last grid entry
                }
            }

            return packed;
        }

        byte[] q3 = new byte[2 * 110];

        // Block 0: d = 1.0. hmask all-set for the first 16 bytes, all-clear for the second 16, so both bias
        // branches run for every group.
        WriteLE(q3, 108, (ushort)BitConverter.HalfToInt16Bits((Half)1f));
        for (int e = 0; e < 16; e++)
        {
            q3[e] = 0xFF; // high bit set for every bit position of both halves
            q3[16 + e] = 0x00; // high bit clear for the odd-k groups
        }

        // Four 16-byte group-pair windows: all low codes 3, all low codes 0, all low codes 1, and a per-shift mix.
        for (int e = 0; e < 16; e++)
        {
            q3[32 + e] = 0xFF; // low 2 bits 3 for the four even-k groups of half 0
            q3[48 + e] = 0x00; // low 2 bits 0 for the four odd-k groups of half 0
            q3[64 + e] = 0x55; // low 2 bits 1 for the four even-k groups of half 1
            q3[80 + e] = 0xA5; // low 2 bits 1, 0, 2, 2 across the shifts of half 1
        }

        // Six-bit scale codes spanning the whole range: groups 0..7 keep the low nibble, groups 8..15 the high
        // nibble, and the two high bits of every code land in scales[8 + g%4] at bit 2*(g/4).
        for (int g = 0; g < 16; g++)
        {
            int sc = (g * 7) & 0x3F; // 0, 7, 14, ... covering both nibble halves and both high-bit fields
            if (g < 8)
            {
                q3[96 + g] |= (byte)(sc & 0xF);
            }
            else
            {
                q3[96 + (g - 8)] |= (byte)((sc & 0xF) << 4);
            }

            q3[96 + 8 + (g & 3)] |= (byte)((sc >> 4) << (2 * (g >> 2)));
        }

        // Block 1: d = 0.5 with scale code 0 (read back as -32) and mixed hmask, exercising negative group scales.
        WriteLE(q3, 110 + 108, (ushort)BitConverter.HalfToInt16Bits((Half)0.5f));
        for (int g = 0; g < 16; g++)
        {
            int sc = g == 0 ? 0 : (g * 3) & 0x3F;
            if (g < 8)
            {
                q3[110 + 96 + g] |= (byte)(sc & 0xF);
            }
            else
            {
                q3[110 + 96 + (g - 8)] |= (byte)((sc & 0xF) << 4);
            }

            q3[110 + 96 + 8 + (g & 3)] |= (byte)((sc >> 4) << (2 * (g >> 2)));
        }

        for (int e = 0; e < 32; e++)
        {
            q3[110 + e] = (byte)(e * 5); // hmask bytes mix set and clear bits across all eight bit positions
        }

        for (int e = 0; e < 64; e++)
        {
            q3[110 + 32 + e] = (byte)(e * 3); // varied low 2-bit codes across every shift position
        }

        return q3;
    }

    [Theory]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ4_XS)]
    public void DequantRow_SyntheticBlocks_BothPathsMatchHandWrittenBytes(QuantFormat format)
    {
        int cols = format == QuantFormat.IQ4_NL ? 96 : 512; // three IQ4_NL blocks, or two IQ4_XS super-blocks in one row
        byte[] packed = BuildSyntheticIq4Blocks(format);
        TensorShape shape = MatrixShape(1, cols);

        using CpuBackend scalar = new(threads: 1, allowAvx2: false);
        using CpuBackend avx2 = new(threads: 1, allowAvx2: true);
        WeightRef w = new(packed, shape, format);

        float[] actualScalar = new float[cols];
        float[] actualAvx2 = new float[cols];
        scalar.DequantRow(w, 0, actualScalar);
        avx2.DequantRow(w, 0, actualAvx2);

        double[] expected = OracleDequantRow(packed, 0, cols, format);
        AssertValuesClose(actualScalar, expected, relativeTolerance: 1e-6, $"scalar {format} synthetic blocks");
        AssertValuesClose(actualAvx2, expected, relativeTolerance: 1e-6, $"avx2 {format} synthetic blocks");

        // Both paths unpack the same bytes with the same arithmetic order, so the expansion must match bit-for-bit.
        Assert.True(actualScalar.AsSpan().SequenceEqual(actualAvx2.AsSpan()), $"{format}: avx2 DequantRow diverges from scalar on synthetic blocks");
    }

    // Hand-written IQ4 images: every nibble code 0..15 appears in both nibble halves, d covers zero and negative
    // values, and for IQ4_XS the eight sub-block scales span the whole six-bit range so negative (ls - 32) scales
    // and both scale fields take part alongside the shared table lookup.
    private static byte[] BuildSyntheticIq4Blocks(QuantFormat format)
    {
        if (format == QuantFormat.IQ4_NL)
        {
            byte[] packed = new byte[3 * 18];

            // Block 0: d = 1.0. Byte j keeps code j in its low nibble and code (j + 8) % 16 in its high nibble, so
            // all sixteen table entries dequantize in both nibble halves.
            WriteLE(packed, 0, (ushort)BitConverter.HalfToInt16Bits((Half)1f));
            for (int j = 0; j < 16; j++)
            {
                packed[2 + j] = (byte)(j | (((j + 8) % 16) << 4));
            }

            // Block 1: d = 0 makes every weight zero whatever the code; code 15 (table value 113) must not leak in.
            WriteLE(packed, 18, (ushort)BitConverter.HalfToInt16Bits((Half)0f));
            for (int j = 0; j < 16; j++)
            {
                packed[18 + 2 + j] = 0xFF;
            }

            // Block 2: d = -0.5 with varied codes so the negative scale and the negative table entries both run.
            WriteLE(packed, 36, (ushort)BitConverter.HalfToInt16Bits((Half)(-0.5f)));
            for (int j = 0; j < 16; j++)
            {
                packed[36 + 2 + j] = (byte)(((j * 5) % 16) | (j << 4));
            }

            return packed;
        }

        byte[] xs = new byte[2 * 136];

        // Block 0: d = 1.0. Sub-block scales vary across the whole six-bit range; the low nibble lands in
        // scales_l[ib/2] at bit 4*(ib%2) and the two high bits in scales_h at bit 2*ib.
        WriteLE(xs, 0, (ushort)BitConverter.HalfToInt16Bits((Half)1f));
        ushort scalesH0 = 0;
        for (int ib = 0; ib < 8; ib++)
        {
            int ls = (ib * 9) & 0x3F; // 0, 9, 18, ... covering both nibble fields and both scale halves
            xs[4 + ib / 2] |= (byte)((ls & 0xF) << (4 * (ib % 2)));
            scalesH0 |= (ushort)(((ls >> 4) & 0x3) << (2 * ib));
        }

        WriteLE(xs, 2, scalesH0);
        for (int ib = 0; ib < 8; ib++)
        {
            for (int j = 0; j < 16; j++)
            {
                xs[8 + 16 * ib + j] = (byte)(((ib + j) % 16) | ((15 - j) << 4)); // every table entry in both nibble halves
            }
        }

        // Block 1: d = 0.25 with scale code 0 (read back as -32) and code 63 (read back as +31) side by side, so
        // negative and positive sub-block scales dequantize through the same lookup.
        WriteLE(xs, 136, (ushort)BitConverter.HalfToInt16Bits((Half)0.25f));
        ushort scalesH1 = 0;
        for (int ib = 0; ib < 8; ib++)
        {
            int ls = ib < 4 ? 0 : 63;
            xs[136 + 4 + ib / 2] |= (byte)((ls & 0xF) << (4 * (ib % 2)));
            scalesH1 |= (ushort)(((ls >> 4) & 0x3) << (2 * ib));
        }

        WriteLE(xs, 136 + 2, scalesH1);
        for (int e = 0; e < 128; e++)
        {
            xs[136 + 8 + e] = (byte)(e & 0xFF); // varied codes across every nibble position
        }

        return xs;
    }

    [Theory]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ4_XS)]
    public void Gemv_PartialFinalBlock_BothPathsMatchDoubleOracle(QuantFormat format)
    {
        // 100 and 137 columns are neither multiples of 32 nor of 256: the final block is zero-padded and only the
        // real columns may contribute, in both the fused and the scalar cleanup tail.
        foreach ((int rows, int cols) in new[] { (300, 100), (300, 137) })
        {
            (float[] x, byte[] weights) = CreateCase(rows, cols, format, seed: 4242);

            using CpuBackend scalar = new(threads: 1, allowAvx2: false);
            using CpuBackend avx2 = new(threads: 1, allowAvx2: true);
            WeightRef w = new(weights, MatrixShape(rows, cols), format);

            float[] actualScalar = new float[rows];
            float[] actualAvx2 = new float[rows];
            scalar.Gemv(x, w, actualScalar);
            avx2.Gemv(x, w, actualAvx2);

            (double[] expected, double[] magnitudes) = OracleDot(x, weights, rows, cols, format);
            AssertClose(actualScalar, expected, magnitudes, relativeTolerance: 1e-4, $"scalar {format} partial final block at {rows}x{cols}");
            AssertClose(actualAvx2, expected, magnitudes, relativeTolerance: 1e-4, $"avx2 {format} partial final block at {rows}x{cols}");

            // The dequant expansion of the same padded row must agree bit-for-bit across paths.
            float[] rowScalar = new float[cols];
            float[] rowAvx2 = new float[cols];
            scalar.DequantRow(w, 0, rowScalar);
            avx2.DequantRow(w, 0, rowAvx2);
            Assert.True(rowScalar.AsSpan().SequenceEqual(rowAvx2.AsSpan()), $"{format}: avx2 DequantRow diverges from scalar on a partial final block");
        }
    }

    [Theory]
    [InlineData(QuantFormat.F32)]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ4_XS)]
    public void DequantRow_BadRowIndexOrOutputLength_ThrowsClearMessage(QuantFormat format)
    {
        TensorShape shape = MatrixShape(4, 70);
        byte[] packed = new byte[4 * (long)new TensorShape([1, 70]).RowStrideBytes(format)];

        using CpuBackend backend = new();

        // WeightRef is a ref struct: construct one inside each lambda instead of capturing a local.
        ArgumentOutOfRangeException rowEx = Assert.Throws<ArgumentOutOfRangeException>(
            () => backend.DequantRow(new WeightRef(packed, shape, format), 4, new float[70]));
        Assert.Contains("outside the weight", rowEx.Message);
        Assert.Throws<ArgumentOutOfRangeException>(
            () => backend.DequantRow(new WeightRef(packed, shape, format), -1, new float[70]));

        ArgumentException outEx = Assert.Throws<ArgumentException>(
            () => backend.DequantRow(new WeightRef(packed, shape, format), 0, new float[69]));
        Assert.Contains("shorter than the weight column count", outEx.Message);

        byte[] tooShort = new byte[packed.Length - 1];
        Assert.Throws<ArgumentException>(
            () => backend.DequantRow(new WeightRef(tooShort, shape, format), 0, new float[70]));
    }

    private static TensorShape MatrixShape(int rows, int cols) => new([rows, cols]);

    private static (float[] X, byte[] Weights) CreateCase(int rows, int cols, QuantFormat format, int seed)
    {
        Random rng = new(seed);
        float[] x = NextFloats(rng, cols);
        float[] weightsFloat = NextFloats(rng, rows * cols);

        long stride = MatrixShape(rows, cols).RowStrideBytes(format);
        byte[] packed = new byte[rows * stride];

        for (int row = 0; row < rows; row++)
        {
            ReadOnlySpan<float> src = weightsFloat.AsSpan(row * cols, cols);
            Span<byte> dst = packed.AsSpan((int)(row * stride), (int)stride);
            switch (format)
            {
                case QuantFormat.F16:
                    QuantizeF16(src, dst);
                    break;
                case QuantFormat.Q8_0:
                    QuantizeQ8_0(src, dst);
                    break;
                case QuantFormat.Q5_0:
                    QuantizeQ5_0(src, dst);
                    break;
                case QuantFormat.Q5_K:
                    QuantizeQ5K(src, dst);
                    break;
                case QuantFormat.Q6_K:
                    QuantizeQ6K(src, dst);
                    break;
                case QuantFormat.Q2_K:
                    QuantizeQ2K(src, dst);
                    break;
                case QuantFormat.Q3_K:
                    QuantizeQ3K(src, dst);
                    break;
                case QuantFormat.IQ2_XXS:
                    QuantizeIQ2Xxs(src, dst);
                    break;
                case QuantFormat.IQ2_XS:
                    QuantizeIQ2Xs(src, dst);
                    break;
                case QuantFormat.IQ3_XXS:
                    QuantizeIQ3Xxs(src, dst);
                    break;
                case QuantFormat.IQ4_NL:
                    QuantizeIQ4N(src, dst);
                    break;
                case QuantFormat.IQ4_XS:
                    QuantizeIQ4X(src, dst);
                    break;
                default:
                    QuantizeQ4K(src, dst);
                    break;
            }
        }

        return (x, packed);
    }

    private static float[] NextFloats(Random rng, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        return values;
    }

    // Test-only F16 writer: round-to-nearest half-floats, little-endian.
    private static void QuantizeF16(ReadOnlySpan<float> src, Span<byte> dst)
    {
        for (int i = 0; i < src.Length; i++)
        {
            ushort bits = (ushort)BitConverter.HalfToInt16Bits((Half)src[i]);
            WriteLE(dst, 2 * i, bits);
        }
    }

    // Test-only Q8_0 writer: per 32-element block, d = amax/127 stored as half-float, quants clamped to [-127, 127].
    private static void QuantizeQ8_0(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int blocks = (src.Length + 31) / 32;
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> block = src.Slice(b * 32, Math.Min(32, src.Length - b * 32));

            float amax = 0f;
            foreach (float v in block)
            {
                amax = Math.Max(amax, Math.Abs(v));
            }

            Half dHalf = (Half)(amax / 127f);
            WriteLE(dst, b * 34, (ushort)BitConverter.HalfToInt16Bits(dHalf));

            if (dHalf == (Half)0f)
            {
                continue; // all-zero block: quants stay zero, matching the reference behavior
            }

            for (int l = 0; l < block.Length; l++)
            {
                int q = (int)Math.Round((double)block[l] / (float)dHalf, MidpointRounding.AwayFromZero);
                q = Math.Clamp(q, -127, 127);
                dst[b * 34 + 2 + l] = (byte)(sbyte)q; // padded quants stay zero
            }
        }
    }

    // Test-only Q5_0 writer: per 32-element block, d = amax/16 stored as half-float, five-bit quants centered at
    // q - 16 with q in [0, 31]; the 5th bit lives in the little-endian qh dword at block bytes 2..5.
    private static void QuantizeQ5_0(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int blocks = (src.Length + 31) / 32;
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> block = src.Slice(b * 32, Math.Min(32, src.Length - b * 32));

            float amax = 0f;
            foreach (float v in block)
            {
                amax = Math.Max(amax, Math.Abs(v));
            }

            Half dHalf = (Half)(amax / 16f);
            WriteLE(dst, b * 22, (ushort)BitConverter.HalfToInt16Bits(dHalf));

            if (dHalf == (Half)0f)
            {
                continue; // all-zero block: quants stay zero, matching the reference behavior
            }

            uint qh = 0;
            for (int l = 0; l < block.Length; l++)
            {
                int five = Math.Clamp((int)Math.Round((double)block[l] / (float)dHalf, MidpointRounding.AwayFromZero) + 16, 0, 31);
                int j = l & 15;
                ref byte target = ref dst[b * 22 + 6 + j];
                if (l < 16)
                {
                    target = (byte)((target & 0xF0) | (five & 0xF));
                }
                else
                {
                    target = (byte)((target & 0x0F) | (five << 4));
                }

                qh |= (uint)((five >> 4) & 1) << l;
            }

            dst[b * 22 + 2] = (byte)qh;
            dst[b * 22 + 3] = (byte)(qh >> 8);
            dst[b * 22 + 4] = (byte)(qh >> 16);
            dst[b * 22 + 5] = (byte)(qh >> 24);
        }
    }

    // Test-only Q5_K writer mirroring the reference layout: Q4_K-style block/group scales and mins (see
    // QuantizeQ4K), then 5-bit codes q in [0, 31] against dGroup = d*sc and dmGroup = dmin*m. Nibbles live in
    // the 128-byte qs region at block+48 (group-pair windows, low nibble even groups, high odd); the 5th bit
    // lives at bit j of qh[32*(j/2) + l] in the 32-byte qh region at block+16. Padded elements produce zero blocks.
    private static void QuantizeQ5K(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int blocks = (src.Length + 255) / 256;
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> srcBlock = src.Slice(b * 256, Math.Min(256, src.Length - b * 256));

            float[] scales = new float[8];
            float[] mins = new float[8]; // positive offsets subtracted from weights: w = d*sc*q - dmin*m
            for (int j = 0; j < 8; j++)
            {
                int start = j * 32;
                if (start >= srcBlock.Length)
                {
                    break; // padded region contributes nothing
                }

                ReadOnlySpan<float> group = srcBlock.Slice(start, Math.Min(32, srcBlock.Length - start));
                float amax = 0f;
                float vmin = 0f;
                foreach (float v in group)
                {
                    amax = Math.Max(amax, Math.Abs(v));
                    vmin = Math.Min(vmin, v);
                }

                scales[j] = amax / 31f;
                mins[j] = Math.Max(0f, -vmin);
            }

            float dBlock = 0f;
            float mBlock = 0f;
            for (int j = 0; j < 8; j++)
            {
                dBlock = Math.Max(dBlock, scales[j]);
                mBlock = Math.Max(mBlock, mins[j]);
            }

            Half dHalf = (Half)(dBlock / 63f);
            Half mHalf = (Half)(mBlock / 63f);
            WriteLE(dst, b * 176, (ushort)BitConverter.HalfToInt16Bits(dHalf));
            WriteLE(dst, b * 176 + 2, (ushort)BitConverter.HalfToInt16Bits(mHalf));

            // Quantize group codes against the half-floats exactly as they will be read back.
            float dBack = (float)dHalf;
            float mBack = (float)mHalf;
            byte[] scBits = new byte[8];
            byte[] minBits = new byte[8];
            for (int j = 0; j < 8; j++)
            {
                int ls = dBack > 0f ? Math.Clamp((int)Math.Round(scales[j] / dBack, MidpointRounding.AwayFromZero), 0, 63) : 0;
                int lm = mBack > 0f ? Math.Clamp((int)Math.Round(mins[j] / mBack, MidpointRounding.AwayFromZero), 0, 63) : 0;
                scBits[j] = (byte)ls;
                minBits[j] = (byte)lm;
            }

            // Pack the six-bit group codes: bytes j/4+j hold low 6 bits of scale_j/min_j and receive high 2 bits of group 4+.
            for (int j = 0; j < 8; j++)
            {
                if (j < 4)
                {
                    dst[b * 176 + 4 + j] |= scBits[j];
                    dst[b * 176 + 8 + j] |= minBits[j];
                }
                else
                {
                    dst[b * 176 + 12 + (j - 4)] |= (byte)(scBits[j] & 0xF);
                    dst[b * 176 + 12 + (j - 4)] |= (byte)((minBits[j] & 0xF) << 4);
                    dst[b * 176 + 4 + (j - 4)] |= (byte)((scBits[j] >> 4) << 6);
                    dst[b * 176 + 8 + (j - 4)] |= (byte)((minBits[j] >> 4) << 6);
                }
            }

            for (int e = 0; e < srcBlock.Length; e++)
            {
                int j = e / 32;
                float dGroup = dBack * scBits[j];
                if (dGroup == 0f)
                {
                    continue; // all-zero group: quants stay zero, matching the reference behavior
                }

                float dmGroup = mBack * minBits[j];
                int five = Math.Clamp((int)Math.Round(((double)srcBlock[e] + dmGroup) / dGroup, MidpointRounding.AwayFromZero), 0, 31);

                // Element e lives in group j at qs window byte 32*(j/2) + (e % 32): low nibble for even groups, high for odd.
                ref byte target = ref dst[b * 176 + 48 + 32 * (j / 2) + (e % 32)];
                if ((j & 1) == 0)
                {
                    target = (byte)((target & 0xF0) | (five & 0xF));
                }
                else
                {
                    target = (byte)((target & 0x0F) | (five << 4));
                }

                // The 5th bit lives at bit j of the matching qh byte.
                dst[b * 176 + 16 + 32 * (j / 2) + (e % 32)] |= (byte)(((five >> 4) & 1) << j);
            }
        }
    }

    // Test-only Q6_K writer: per 16-element group, amax = max|v|, signed 8-bit scale s = round(amax / (31 * d))
    // clamped to [-127, 127] against the block half-float d = amax(block) / 31, and six-bit codes q in [0, 63]
    // centered at q - 32. ql nibbles at block+0 (low nibbles for quarters 0..1, high for 2..3), 2-bit fields at
    // bit 2*quarter of qh bytes block+128, scales at block+192, d at block+208. Padded elements produce zero blocks.
    private static void QuantizeQ6K(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int blocks = (src.Length + 255) / 256;
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> srcBlock = src.Slice(b * 256, Math.Min(256, src.Length - b * 256));

            float amaxBlock = 0f;
            foreach (float v in srcBlock)
            {
                amaxBlock = Math.Max(amaxBlock, Math.Abs(v));
            }

            Half dHalf = (Half)(amaxBlock / 31f);
            WriteLE(dst, b * 210 + 208, (ushort)BitConverter.HalfToInt16Bits(dHalf));

            float dBack = (float)dHalf;
            for (int g = 0; g < 16; g++)
            {
                int start = g * 16;
                if (start >= srcBlock.Length)
                {
                    break; // padded region contributes nothing
                }

                ReadOnlySpan<float> group = srcBlock.Slice(start, Math.Min(16, srcBlock.Length - start));
                float amax = 0f;
                foreach (float v in group)
                {
                    amax = Math.Max(amax, Math.Abs(v));
                }

                sbyte sc = dBack > 0f
                    ? (sbyte)Math.Clamp((int)Math.Round((double)amax / (31.0 * dBack), MidpointRounding.AwayFromZero), -127, 127)
                    : (sbyte)0;
                dst[b * 210 + 192 + g] = (byte)sc;

                if (sc == 0)
                {
                    continue; // all-zero group: quants stay zero, matching the reference behavior
                }

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                int quarter = k >> 1; // 0..3: which 32-element quarter of the half the group starts in
                for (int e = 0; e < group.Length; e++)
                {
                    int six = Math.Clamp((int)Math.Round((double)group[e] / ((float)sc * dBack), MidpointRounding.AwayFromZero) + 32, 0, 63);

                    // Element e lives at ql byte 64h + 16*(k&3) + e: low nibble for quarters 0..1, high for 2..3.
                    ref byte target = ref dst[b * 210 + 64 * h + 16 * (k & 3) + e];
                    if ((quarter >> 1) == 0)
                    {
                        target = (byte)((target & 0xF0) | (six & 0xF));
                    }
                    else
                    {
                        target = (byte)((target & 0x0F) | (six << 4));
                    }

                    // The high 2 bits live at bit 2*quarter of qh byte 128 + 32h + 16*(k&1) + e.
                    dst[b * 210 + 128 + 32 * h + 16 * (k & 1) + e] |= (byte)((six >> 4) << (2 * quarter));
                }
            }
        }
    }

    // Test-only Q2_K writer mirroring the reference layout: sixteen 16-element groups per 256-element block, each
    // with amax/3 scale and max(0, -min) offset; block d/dmin = group-max / 15 stored as half-floats at bytes 80
    // and 82. Scale byte g packs the four-bit group scale into its low nibble and the four-bit min into its high
    // nibble; two-bit codes live in the 64-byte qs region at block+16, window byte 32h + 16*(k&1) read at bit
    // shift 2*(k>>1), so four group-pairs share one window byte. Padded elements produce zero blocks.
    private static void QuantizeQ2K(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int blocks = (src.Length + 255) / 256;
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> srcBlock = src.Slice(b * 256, Math.Min(256, src.Length - b * 256));

            float[] scales = new float[16];
            float[] mins = new float[16]; // positive offsets subtracted from weights: w = d*sc*q - dmin*m
            for (int g = 0; g < 16; g++)
            {
                int start = g * 16;
                if (start >= srcBlock.Length)
                {
                    break; // padded region contributes nothing
                }

                ReadOnlySpan<float> group = srcBlock.Slice(start, Math.Min(16, srcBlock.Length - start));
                float amax = 0f;
                float vmin = 0f;
                foreach (float v in group)
                {
                    amax = Math.Max(amax, Math.Abs(v));
                    vmin = Math.Min(vmin, v);
                }

                scales[g] = amax / 3f;
                mins[g] = Math.Max(0f, -vmin);
            }

            float dBlock = 0f;
            float mBlock = 0f;
            for (int g = 0; g < 16; g++)
            {
                dBlock = Math.Max(dBlock, scales[g]);
                mBlock = Math.Max(mBlock, mins[g]);
            }

            Half dHalf = (Half)(dBlock / 15f);
            Half mHalf = (Half)(mBlock / 15f);
            WriteLE(dst, b * 84 + 80, (ushort)BitConverter.HalfToInt16Bits(dHalf));
            WriteLE(dst, b * 84 + 82, (ushort)BitConverter.HalfToInt16Bits(mHalf));

            // Quantize group codes against the half-floats exactly as they will be read back.
            float dBack = (float)dHalf;
            float mBack = (float)mHalf;
            byte[] scBits = new byte[16];
            byte[] minBits = new byte[16];
            for (int g = 0; g < 16; g++)
            {
                scBits[g] = dBack > 0f ? (byte)Math.Clamp((int)Math.Round(scales[g] / dBack, MidpointRounding.AwayFromZero), 0, 15) : (byte)0;
                minBits[g] = mBack > 0f ? (byte)Math.Clamp((int)Math.Round(mins[g] / mBack, MidpointRounding.AwayFromZero), 0, 15) : (byte)0;
                dst[b * 84 + g] = (byte)(scBits[g] | (minBits[g] << 4));
            }

            for (int e = 0; e < srcBlock.Length; e++)
            {
                int g = e / 16;
                float dGroup = dBack * scBits[g];
                if (dGroup == 0f)
                {
                    continue; // all-zero group: quants stay zero, matching the reference behavior
                }

                float dmGroup = mBack * minBits[g];
                int two = Math.Clamp((int)Math.Round(((double)srcBlock[e] + dmGroup) / dGroup, MidpointRounding.AwayFromZero), 0, 3);

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                ref byte target = ref dst[b * 84 + 16 + 32 * h + 16 * (k & 1) + (e % 16)];
                target = (byte)(target | (two << (2 * (k >> 1)))); // groups k, k+2, k+4, k+6 share one window byte
            }
        }
    }

    // Test-only Q3_K writer mirroring the reference layout: sixteen 16-element groups per 256-element block with
    // amax/4 group scales; block d = group-max / 32 stored as a half-float at byte 108, and each group's six-bit
    // scale code sc in [32, 63] is read back as (sc - 32) >= 0. Packing: groups 0..7 keep the low nibble of sc in
    // scales[g], groups 8..15 in the high nibble of scales[g-8], and the two high bits of every scale in
    // scales[8 + g%4] at bit 2*(g/4). Three-bit codes live as the low 2 bits at qs window byte 32 + 32h + 16*(k&1),
    // bit 2*(k>>1), plus the high bit at hmask byte 16*(k&1) + e, bit 4h + (k>>1). Padded elements produce zero blocks.
    private static void QuantizeQ3K(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int blocks = (src.Length + 255) / 256;
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> srcBlock = src.Slice(b * 256, Math.Min(256, src.Length - b * 256));

            float[] scales = new float[16];
            for (int g = 0; g < 16; g++)
            {
                int start = g * 16;
                if (start >= srcBlock.Length)
                {
                    break; // padded region contributes nothing
                }

                ReadOnlySpan<float> group = srcBlock.Slice(start, Math.Min(16, srcBlock.Length - start));
                float amax = 0f;
                foreach (float v in group)
                {
                    amax = Math.Max(amax, Math.Abs(v));
                }

                scales[g] = amax / 4f;
            }

            float dBlock = 0f;
            for (int g = 0; g < 16; g++)
            {
                dBlock = Math.Max(dBlock, scales[g]);
            }

            Half dHalf = (Half)(dBlock / 32f);
            WriteLE(dst, b * 110 + 108, (ushort)BitConverter.HalfToInt16Bits(dHalf));

            // Quantize group codes against the half-floats exactly as they will be read back.
            float dBack = (float)dHalf;
            int[] scBits = new int[16];
            for (int g = 0; g < 16; g++)
            {
                scBits[g] = dBack > 0f ? Math.Clamp((int)Math.Round(scales[g] / dBack, MidpointRounding.AwayFromZero), 0, 31) + 32 : 32;
                int low = scBits[g] & 0xF;
                int high = (scBits[g] >> 4) & 0x3; // the two bits above the stored nibble

                if (g < 8)
                {
                    dst[b * 110 + 96 + g] |= (byte)low;
                }
                else
                {
                    dst[b * 110 + 96 + (g - 8)] |= (byte)(low << 4);
                }

                dst[b * 110 + 96 + 8 + (g & 3)] |= (byte)(high << (2 * (g >> 2))); // groups g and g+8 share a bit field
            }

            for (int e = 0; e < srcBlock.Length; e++)
            {
                int g = e / 16;
                float dGroup = dBack * (scBits[g] - 32);
                if (dGroup == 0f)
                {
                    continue; // all-zero group: quants stay zero, matching the reference behavior
                }

                int three = Math.Clamp((int)Math.Round((double)srcBlock[e] / dGroup, MidpointRounding.AwayFromZero), -4, 3) + 4; // stored code in [0, 7]

                int h = g >> 3; // 128-element half of the super-block
                int k = g & 7; // 16-element group inside the half
                ref byte q = ref dst[b * 110 + 32 + 32 * h + 16 * (k & 1) + (e % 16)];
                q = (byte)(q | ((three & 3) << (2 * (k >> 1)))); // low 2 bits: groups k, k+2, k+4, k+6 share one byte

                if (three > 3)
                {
                    // The high bit of the 3-bit code lives at hmask byte 16*(k&1) + e, bit 4h + (k>>1).
                    dst[b * 110 + 16 * (k & 1) + (e % 16)] |= (byte)(1 << (4 * h + (k >> 1)));
                }
            }
        }
    }

    // The shared 16-value IQ4 table, pinned to the reference tag: byte n holds the value a nibble code n selects.
    private static readonly sbyte[] Iq4KValues = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

    // Test-only IQ4_NL writer: per 32-element block, d = amax / 113 stored as a half-float at byte 0, and each
    // element takes the nearest of the sixteen table values scaled by d. Element j < 16 keeps its code in the low
    // nibble of qs[j] (block bytes 2..17) and element 16+j in the high nibble of the same byte. Padded elements
    // keep zero quants, which the kernels never read past the real column count.
    private static void QuantizeIQ4N(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int blocks = (src.Length + 31) / 32;
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> block = src.Slice(b * 32, Math.Min(32, src.Length - b * 32));

            float amax = 0f;
            foreach (float v in block)
            {
                amax = Math.Max(amax, Math.Abs(v));
            }

            Half dHalf = (Half)(amax / 113f);
            WriteLE(dst, b * 18, (ushort)BitConverter.HalfToInt16Bits(dHalf));

            if (dHalf == (Half)0f)
            {
                continue; // all-zero block: quants stay zero, matching the reference behavior
            }

            float dBack = (float)dHalf;
            for (int l = 0; l < block.Length; l++)
            {
                int code = NearestIQ4Code((double)block[l] / dBack);
                int j = l & 15;
                ref byte target = ref dst[b * 18 + 2 + j];
                if (l < 16)
                {
                    target = (byte)((target & 0xF0) | code);
                }
                else
                {
                    target = (byte)((target & 0x0F) | (code << 4));
                }
            }
        }
    }

    // Test-only IQ4_XS writer mirroring the reference layout: eight 32-value sub-blocks per 256-element block.
    // Block d = amax(block) / 113 stored as a half-float at byte 0; each sub-block's six-bit scale is
    // round(amax(sub-block) / (32 * d)) clamped to [0, 63] and read back as (ls - 32), packing its low nibble in
    // scales_l[ib/2] at bit 4*(ib%2) (block bytes 4..7) and its two high bits in scales_h at bit 2*ib (bytes 2..3).
    // Codes quantize against dl = d * (ls - 32) and live in the 128-byte qs region at block+8 with the same
    // low/high nibble element order as IQ4_NL. Padded elements keep zero quants.
    private static void QuantizeIQ4X(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int blocks = (src.Length + 255) / 256;
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> srcBlock = src.Slice(b * 256, Math.Min(256, src.Length - b * 256));

            float amaxBlock = 0f;
            foreach (float v in srcBlock)
            {
                amaxBlock = Math.Max(amaxBlock, Math.Abs(v));
            }

            Half dHalf = (Half)(amaxBlock / 113f);
            WriteLE(dst, b * 136, (ushort)BitConverter.HalfToInt16Bits(dHalf));
            float dBack = (float)dHalf;

            ushort scalesH = 0;
            for (int ib = 0; ib < 8; ib++)
            {
                int start = ib * 32;
                if (start >= srcBlock.Length)
                {
                    break; // padded region contributes nothing
                }

                ReadOnlySpan<float> sub = srcBlock.Slice(start, Math.Min(32, srcBlock.Length - start));
                float amax = 0f;
                foreach (float v in sub)
                {
                    amax = Math.Max(amax, Math.Abs(v));
                }

                int ls = dBack > 0f ? Math.Clamp((int)Math.Round((double)amax / (32.0 * dBack), MidpointRounding.AwayFromZero), 0, 63) : 0;
                dst[b * 136 + 4 + ib / 2] |= (byte)((ls & 0xF) << (4 * (ib % 2)));
                scalesH |= (ushort)(((ls >> 4) & 0x3) << (2 * ib));

                float dl = dBack * (ls - 32);
                if (dl == 0f)
                {
                    continue; // zero sub-block scale: quants stay zero, matching the reference behavior
                }

                for (int e = 0; e < sub.Length; e++)
                {
                    int code = NearestIQ4Code((double)sub[e] / dl);
                    ref byte target = ref dst[b * 136 + 8 + 16 * ib + (e & 15)];
                    if (e < 16)
                    {
                        target = (byte)((target & 0xF0) | code);
                    }
                    else
                    {
                        target = (byte)((target & 0x0F) | (code << 4));
                    }
                }
            }

            WriteLE(dst, b * 136 + 2, scalesH);
        }
    }

    // Test-only nearest-code lookup: pick the table entry closest to the scaled value, clamped to the table range.
    private static int NearestIQ4Code(double scaled)
    {
        int best = 0;
        double bestDiff = double.MaxValue;
        for (int n = 0; n < Iq4KValues.Length; n++)
        {
            double diff = Math.Abs(scaled - Iq4KValues[n]);
            if (diff < bestDiff)
            {
                bestDiff = diff;
                best = n;
            }
        }

        return best;
    }

    // Test-local copy of the shared IQ2_XXS grid as 256 uint64 magnitudes: entry i's little-endian bytes are the
    // eight magnitude bytes the kernels read at grid[i * 8 + n].
    private static readonly ulong[] Iq2XXSGridU64 =
    [
        0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08, 0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
        0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819, 0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
        0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b, 0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
        0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
        0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819, 0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
        0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
        0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808, 0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
        0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
        0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08, 0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
        0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819, 0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
        0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808, 0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
        0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808, 0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
        0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
        0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819, 0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
        0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19, 0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
        0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b, 0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
        0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908, 0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
        0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08, 0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
        0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819, 0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
        0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808, 0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
        0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819, 0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
        0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b, 0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
        0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808, 0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
        0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b, 0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
        0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
        0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808, 0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
        0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
        0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
        0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908, 0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
        0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808, 0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
        0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b, 0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
        0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19, 0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908
    ];

    // Test-local copy of the shared IQ2_XS grid as 512 uint64 magnitudes: entry i's little-endian bytes are the
    // eight magnitude bytes the kernels read at grid[i * 8 + n].
    private static readonly ulong[] Iq2XSGridU64 =
    [
        0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08, 0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
        0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919, 0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
        0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919, 0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
        0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
        0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
        0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
        0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908, 0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
        0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808, 0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
        0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
        0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
        0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
        0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819, 0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
        0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819, 0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
        0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b, 0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
        0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819, 0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
        0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908, 0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
        0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b, 0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
        0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
        0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
        0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
        0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808, 0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
        0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808, 0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
        0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808, 0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
        0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908, 0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
        0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808, 0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
        0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819, 0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
        0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908, 0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
        0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
        0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
        0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b, 0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
        0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808, 0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
        0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
        0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
        0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
        0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808, 0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
        0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819, 0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
        0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808, 0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
        0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819, 0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
        0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908, 0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
        0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b, 0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
        0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908, 0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
        0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808, 0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
        0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
        0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808, 0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
        0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819, 0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
        0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08, 0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
        0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19, 0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
        0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
        0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b, 0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
        0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b, 0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
        0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
        0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808, 0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
        0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808, 0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
        0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08, 0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
        0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19, 0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
        0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919, 0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
        0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08, 0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
        0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908, 0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
        0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908, 0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
        0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b, 0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
        0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808, 0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
        0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
        0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b, 0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
        0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b
    ];


    // Test-local copy of the shared IQ3_XXS grid as 256 uint32 magnitudes: entry i's little-endian bytes are the four
    // magnitude bytes the kernels read at grid[i * 4 + n].
    private static readonly uint[] Iq3XXSGridU32 =
    [
        0x04040404, 0x04040414, 0x04040424, 0x04040C0C, 0x04040C1C, 0x04040C3E, 0x04041404, 0x04041414,
        0x04041C0C, 0x04042414, 0x04043E1C, 0x04043E2C, 0x040C040C, 0x040C041C, 0x040C0C04, 0x040C0C14,
        0x040C140C, 0x040C142C, 0x040C1C04, 0x040C1C14, 0x040C240C, 0x040C2C24, 0x040C3E04, 0x04140404,
        0x04140414, 0x04140424, 0x04140C0C, 0x04141404, 0x04141414, 0x04141C0C, 0x04141C1C, 0x04141C3E,
        0x04142C0C, 0x04142C3E, 0x04143E2C, 0x041C040C, 0x041C043E, 0x041C0C04, 0x041C0C14, 0x041C142C,
        0x041C3E04, 0x04240C1C, 0x04241C3E, 0x04242424, 0x04242C3E, 0x04243E1C, 0x04243E2C, 0x042C040C,
        0x042C043E, 0x042C1C14, 0x042C2C14, 0x04341C2C, 0x04343424, 0x043E0C04, 0x043E0C24, 0x043E0C34,
        0x043E241C, 0x043E340C, 0x0C04040C, 0x0C04041C, 0x0C040C04, 0x0C040C14, 0x0C04140C, 0x0C04141C,
        0x0C041C04, 0x0C041C14, 0x0C041C24, 0x0C04243E, 0x0C042C04, 0x0C0C0404, 0x0C0C0414, 0x0C0C0C0C,
        0x0C0C1404, 0x0C0C1414, 0x0C14040C, 0x0C14041C, 0x0C140C04, 0x0C140C14, 0x0C14140C, 0x0C141C04,
        0x0C143E14, 0x0C1C0404, 0x0C1C0414, 0x0C1C1404, 0x0C1C1C0C, 0x0C1C2434, 0x0C1C3434, 0x0C24040C,
        0x0C24042C, 0x0C242C04, 0x0C2C1404, 0x0C2C1424, 0x0C2C2434, 0x0C2C3E0C, 0x0C34042C, 0x0C3E1414,
        0x0C3E2404, 0x14040404, 0x14040414, 0x14040C0C, 0x14040C1C, 0x14041404, 0x14041414, 0x14041434,
        0x14041C0C, 0x14042414, 0x140C040C, 0x140C041C, 0x140C042C, 0x140C0C04, 0x140C0C14, 0x140C140C,
        0x140C1C04, 0x140C341C, 0x140C343E, 0x140C3E04, 0x14140404, 0x14140414, 0x14140C0C, 0x14140C3E,
        0x14141404, 0x14141414, 0x14141C3E, 0x14142404, 0x14142C2C, 0x141C040C, 0x141C0C04, 0x141C0C24,
        0x141C3E04, 0x141C3E24, 0x14241C2C, 0x14242C1C, 0x142C041C, 0x142C143E, 0x142C240C, 0x142C3E24,
        0x143E040C, 0x143E041C, 0x143E0C34, 0x143E242C, 0x1C04040C, 0x1C040C04, 0x1C040C14, 0x1C04140C,
        0x1C04141C, 0x1C042C04, 0x1C04342C, 0x1C043E14, 0x1C0C0404, 0x1C0C0414, 0x1C0C1404, 0x1C0C1C0C,
        0x1C0C2424, 0x1C0C2434, 0x1C14040C, 0x1C14041C, 0x1C140C04, 0x1C14142C, 0x1C142C14, 0x1C143E14,
        0x1C1C0C0C, 0x1C1C1C1C, 0x1C241C04, 0x1C24243E, 0x1C243E14, 0x1C2C0404, 0x1C2C0434, 0x1C2C1414,
        0x1C2C2C2C, 0x1C340C24, 0x1C341C34, 0x1C34341C, 0x1C3E1C1C, 0x1C3E3404, 0x24040424, 0x24040C3E,
        0x24041C2C, 0x24041C3E, 0x24042C1C, 0x24042C3E, 0x240C3E24, 0x24141404, 0x24141C3E, 0x24142404,
        0x24143404, 0x24143434, 0x241C043E, 0x241C242C, 0x24240424, 0x24242C0C, 0x24243424, 0x242C142C,
        0x242C241C, 0x242C3E04, 0x243E042C, 0x243E0C04, 0x243E0C14, 0x243E1C04, 0x2C040C14, 0x2C04240C,
        0x2C043E04, 0x2C0C0404, 0x2C0C0434, 0x2C0C1434, 0x2C0C2C2C, 0x2C140C24, 0x2C141C14, 0x2C143E14,
        0x2C1C0414, 0x2C1C2C1C, 0x2C240C04, 0x2C24141C, 0x2C24143E, 0x2C243E14, 0x2C2C0414, 0x2C2C1C0C,
        0x2C342C04, 0x2C3E1424, 0x2C3E2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340C140C,
        0x340C340C, 0x34140C3E, 0x34143424, 0x341C1C04, 0x341C1C34, 0x34242424, 0x342C042C, 0x342C2C14,
        0x34341C1C, 0x343E041C, 0x343E140C, 0x3E04041C, 0x3E04042C, 0x3E04043E, 0x3E040C04, 0x3E041C14,
        0x3E042C14, 0x3E0C1434, 0x3E0C2404, 0x3E140C14, 0x3E14242C, 0x3E142C14, 0x3E1C0404, 0x3E1C0C2C,
        0x3E1C1C1C, 0x3E1C3404, 0x3E24140C, 0x3E24240C, 0x3E2C0404, 0x3E2C0414, 0x3E2C1424, 0x3E341C04
    ];

    // Test-local copy of the shared sign table: each row is an eight-bit sign pattern with an even number of set bits.
    private static readonly int[] KSignsIQ2Xs =
    [
        0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
        144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
        160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
        48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
        192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
        80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
        96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
        240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
    ];

    // Test-only IQ2_XXS writer mirroring the reference layout: eight 32-value sub-blocks per 256-element block, each
    // sub-block carrying four eight-value grid indices in the first two words of its aux region and a four-bit
    // sub-block scale (read back as db = d * (0.5 + sc) * 0.25) plus four seven-bit sign-group codes in the last two.
    // One grid entry supplies the eight magnitudes of a group; the parity rule mirrors the reference, flipping the sign
    // of the smallest-magnitude value when the chosen negative-sign set has odd parity so the row stays inside the
    // even-parity table. Block d = amax / 666 stored as a half-float at byte 0. Padded elements keep zero quants.
    private static void QuantizeIQ2Xxs(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int blocks = (src.Length + 255) / 256;
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> srcBlock = src.Slice(b * 256, Math.Min(256, src.Length - b * 256));

            float amaxBlock = 0f;
            foreach (float v in srcBlock)
            {
                amaxBlock = Math.Max(amaxBlock, Math.Abs(v));
            }

            Half dHalf = (Half)(amaxBlock / 666f);
            WriteLE(dst, b * 66, (ushort)BitConverter.HalfToInt16Bits(dHalf));
            float dBack = (float)dHalf;

            for (int ib = 0; ib < 8; ib++)
            {
                int start = ib * 32;
                if (start >= srcBlock.Length)
                {
                    break; // padded region contributes nothing
                }

                ReadOnlySpan<float> sub = srcBlock.Slice(start, Math.Min(32, srcBlock.Length - start));
                float amax = 0f;
                foreach (float v in sub)
                {
                    amax = Math.Max(amax, Math.Abs(v));
                }

                int sc = dBack > 0f && amax > 0f ? Math.Clamp((int)Math.Round((double)amax / (43.0 * dBack) - 0.5, MidpointRounding.AwayFromZero), 0, 15) : 0;
                double db = dBack * (0.5 + sc) * 0.25;

                uint gridAux = 0;
                uint signAux = ((uint)sc << 28);

                for (int l = 0; l < 4; l++)
                {
                    float[] group = sub.Slice(l * 8, Math.Min(8, sub.Length - l * 8)).ToArray();
                    if (group.Length < 8)
                    {
                        break; // padded group: quants stay zero
                    }

                    // Pick the grid entry whose eight magnitude bytes minimize the group's squared error, then the sign
                    // row matching the values' signs under the reference parity rule.
                    int bestEntry = 0;
                    double bestCost = double.MaxValue;
                    for (int e = 0; e < 256; e++)
                    {
                        double cost = 0;
                        for (int p = 0; p < 8; p++)
                        {
                            double magnitude = (Iq2XXSGridU64[e] >> (8 * p)) & 0xFF;
                            double diff = group[p] - db * magnitude;
                            cost += diff * diff;
                        }

                        if (cost < bestCost)
                        {
                            bestCost = cost;
                            bestEntry = e;
                        }
                    }

                    gridAux |= (uint)bestEntry << (8 * l);

                    int signs = 0;
                    int smallest = 0;
                    double smallestAbs = double.MaxValue;
                    for (int p = 0; p < 8; p++)
                    {
                        if (group[p] < 0f)
                        {
                            signs |= 1 << p;
                        }

                        if (Math.Abs(group[p]) < smallestAbs)
                        {
                            smallestAbs = Math.Abs(group[p]);
                            smallest = p;
                        }
                    }

                    if (BitOperations.PopCount((uint)signs) % 2 != 0)
                    {
                        signs ^= 1 << smallest; // odd parity: flip the smallest-magnitude value to land on a table row
                    }

                    signAux |= (uint)FindKSignCode(signs) << (7 * l);
                }

                WriteLE(dst, b * 66 + 2 + 8 * ib, (ushort)(gridAux & 0xFFFF));
                WriteLE(dst, b * 66 + 2 + 8 * ib + 2, (ushort)(gridAux >> 16));
                WriteLE(dst, b * 66 + 2 + 8 * ib + 4, (ushort)(signAux & 0xFFFF));
                WriteLE(dst, b * 66 + 2 + 8 * ib + 6, (ushort)(signAux >> 16));
            }
        }
    }

    // Test-only IQ2_XS writer mirroring the reference layout: eight 32-value sub-blocks per 256-element block, each
    // grid word packing a nine-bit grid index and the seven-bit sign-group code of its eight-value group, and each
    // sub-block scale byte packing the two four-bit scale codes of its nibble pair (read back as
    // db = d * (0.5 + nibble) * 0.25). The same even-parity sign rule as the IQ2_XXS writer applies per group. Block
    // d = amax / 666 stored as a half-float at byte 0. Padded elements keep zero quants.
    private static void QuantizeIQ2Xs(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int blocks = (src.Length + 255) / 256;
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> srcBlock = src.Slice(b * 256, Math.Min(256, src.Length - b * 256));

            float amaxBlock = 0f;
            foreach (float v in srcBlock)
            {
                amaxBlock = Math.Max(amaxBlock, Math.Abs(v));
            }

            Half dHalf = (Half)(amaxBlock / 666f);
            WriteLE(dst, b * 74, (ushort)BitConverter.HalfToInt16Bits(dHalf));
            float dBack = (float)dHalf;

            for (int ib = 0; ib < 8; ib++)
            {
                int start = ib * 32;
                if (start >= srcBlock.Length)
                {
                    break; // padded region contributes nothing
                }

                ReadOnlySpan<float> sub = srcBlock.Slice(start, Math.Min(32, srcBlock.Length - start));

                // Each scale nibble covers two groups: pick its code against the pair's largest magnitude.
                int[] codes = new int[2];
                double[] dbs = new double[2];
                for (int half = 0; half < 2; half++)
                {
                    float amax = 0f;
                    for (int p = half * 16; p < Math.Min(half * 16 + 16, sub.Length); p++)
                    {
                        amax = Math.Max(amax, Math.Abs(sub[p]));
                    }

                    codes[half] = dBack > 0f && amax > 0f ? Math.Clamp((int)Math.Round((double)amax / (43.0 * dBack) - 0.5, MidpointRounding.AwayFromZero), 0, 15) : 0;
                    dbs[half] = dBack * (0.5 + codes[half]) * 0.25;
                }

                dst[b * 74 + 66 + ib] = (byte)(codes[0] | (codes[1] << 4));

                for (int l = 0; l < 4; l++)
                {
                    float[] group = sub.Slice(l * 8, Math.Min(8, sub.Length - l * 8)).ToArray();
                    if (group.Length < 8)
                    {
                        break; // padded group: quants stay zero
                    }

                    int bestEntry = 0;
                    double bestCost = double.MaxValue;
                    for (int e = 0; e < 512; e++)
                    {
                        double cost = 0;
                        for (int p = 0; p < 8; p++)
                        {
                            double magnitude = (Iq2XSGridU64[e] >> (8 * p)) & 0xFF;
                            double diff = group[p] - dbs[l / 2] * magnitude;
                            cost += diff * diff;
                        }

                        if (cost < bestCost)
                        {
                            bestCost = cost;
                            bestEntry = e;
                        }
                    }

                    int signs = 0;
                    int smallest = 0;
                    double smallestAbs = double.MaxValue;
                    for (int p = 0; p < 8; p++)
                    {
                        if (group[p] < 0f)
                        {
                            signs |= 1 << p;
                        }

                        if (Math.Abs(group[p]) < smallestAbs)
                        {
                            smallestAbs = Math.Abs(group[p]);
                            smallest = p;
                        }
                    }

                    if (BitOperations.PopCount((uint)signs) % 2 != 0)
                    {
                        signs ^= 1 << smallest; // odd parity: flip the smallest-magnitude value to land on a table row
                    }

                    ushort word = (ushort)(bestEntry | (FindKSignCode(signs) << 9));
                    WriteLE(dst, b * 74 + 2 + 8 * ib + 2 * l, word);
                }
            }
        }
    }

    // Test-only IQ3_XXS writer mirroring the reference layout: eight 32-value sub-blocks per 256-element block, each
    // aux word packing a four-bit sub-block scale (read back as db = d * (0.5 + sc) * 0.5) and four seven-bit
    // sign-group codes selecting rows of the shared sign table. Each eight-value group stores two grid indices whose
    // little-endian bytes supply the magnitudes; the parity rule mirrors the reference, flipping the sign of the
    // smallest-magnitude value when the chosen negative-sign set has odd parity so the row stays inside the
    // even-parity table. Block d = amax / 32 stored as a half-float at byte 0. Padded elements keep zero quants.
    private static void QuantizeIQ3Xxs(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int blocks = (src.Length + 255) / 256;
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> srcBlock = src.Slice(b * 256, Math.Min(256, src.Length - b * 256));

            float amaxBlock = 0f;
            foreach (float v in srcBlock)
            {
                amaxBlock = Math.Max(amaxBlock, Math.Abs(v));
            }

            Half dHalf = (Half)(amaxBlock / 32f);
            WriteLE(dst, b * 98, (ushort)BitConverter.HalfToInt16Bits(dHalf));
            float dBack = (float)dHalf;

            for (int ib = 0; ib < 8; ib++)
            {
                int start = ib * 32;
                if (start >= srcBlock.Length)
                {
                    break; // padded region contributes nothing
                }

                ReadOnlySpan<float> sub = srcBlock.Slice(start, Math.Min(32, srcBlock.Length - start));
                float amax = 0f;
                foreach (float v in sub)
                {
                    amax = Math.Max(amax, Math.Abs(v));
                }

                int sc = dBack > 0f ? Math.Clamp((int)Math.Round((double)amax / (62.0 * dBack) - 0.5, MidpointRounding.AwayFromZero), 0, 15) : 0;
                double db = dBack * (0.5 + sc) * 0.5;

                int qsBase = b * 98 + 2 + 8 * ib;
                uint aux = ((uint)sc << 28);

                for (int l = 0; l < 4; l++)
                {
                    float[] group = sub.Slice(l * 8, Math.Min(8, sub.Length - l * 8)).ToArray();
                    if (group.Length < 8)
                    {
                        break; // padded group: quants stay zero
                    }

                    // Pick the grid entry whose four magnitude bytes minimize the half-group's squared error, then the
                    // sign row matching the values' signs under the reference parity rule.
                    int[] entry = new int[2];
                    int[] signByte = new int[2];
                    for (int h = 0; h < 2; h++)
                    {
                        int bestEntry = 0;
                        double bestCost = double.MaxValue;
                        for (int e = 0; e < 256; e++)
                        {
                            double cost = 0;
                            for (int p = 0; p < 4; p++)
                            {
                                double magnitude = (Iq3XXSGridU32[e] >> (8 * p)) & 0xFF;
                                double diff = group[h * 4 + p] - db * magnitude;
                                cost += diff * diff;
                            }

                            if (cost < bestCost)
                            {
                                bestCost = cost;
                                bestEntry = e;
                            }
                        }

                        entry[h] = bestEntry;

                        int signs = 0;
                        int smallest = 0;
                        double smallestAbs = double.MaxValue;
                        for (int p = 0; p < 4; p++)
                        {
                            if (group[h * 4 + p] < 0f)
                            {
                                signs |= 1 << (h * 4 + p);
                            }

                            if (Math.Abs(group[h * 4 + p]) < smallestAbs)
                            {
                                smallestAbs = Math.Abs(group[h * 4 + p]);
                                smallest = h * 4 + p;
                            }
                        }

                        if (BitOperations.PopCount((uint)signs) % 2 != 0)
                        {
                            signs ^= 1 << smallest; // odd parity: flip the smallest-magnitude value to land on a table row
                        }

                        signByte[h] = signs;
                    }

                    dst[qsBase + 2 * l] = (byte)entry[0];
                    dst[qsBase + 2 * l + 1] = (byte)entry[1];

                    int code = FindKSignCode(signByte[0] | signByte[1]);
                    aux |= (uint)code << (7 * l);
                }

                dst[b * 98 + 66 + 4 * ib] = (byte)aux;
                dst[b * 98 + 67 + 4 * ib] = (byte)(aux >> 8);
                dst[b * 98 + 68 + 4 * ib] = (byte)(aux >> 16);
                dst[b * 98 + 69 + 4 * ib] = (byte)(aux >> 24);
            }
        }
    }

    // Test-only sign-code lookup: find the table row whose byte carries the group's eight sign bits.
    private static int FindKSignCode(int signs)
    {
        for (int c = 0; c < 128; c++)
        {
            if (KSignsIQ2Xs[c] == signs)
            {
                return c;
            }
        }

        throw new InvalidOperationException("No sign-table row carries the chosen signs.");
    }

    // Test-only Q4_K writer mirroring the reference layout: per-group amax/15 scales and max(0, -min) offsets,
    // block-level d/dmin = group-max / 63 stored as half-floats, then nibbles quantized against exactly those
    // stored values so dequantization is self-consistent with the kernels. Padded elements produce zero blocks.
    private static unsafe void QuantizeQ4K(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int blocks = (src.Length + 255) / 256;

        fixed (byte* pDst = dst)
        {
            for (int b = 0; b < blocks; b++)
            {
                byte* block = pDst + b * 144;
                ReadOnlySpan<float> srcBlock = src.Slice(b * 256, Math.Min(256, src.Length - b * 256));

                float[] scales = new float[8];
                float[] mins = new float[8]; // positive offsets subtracted from weights: w = d*sc*q - mmin*m
                for (int j = 0; j < 8; j++)
                {
                    int start = j * 32;
                    if (start >= srcBlock.Length)
                    {
                        break; // padded region contributes nothing
                    }

                    ReadOnlySpan<float> group = srcBlock.Slice(start, Math.Min(32, srcBlock.Length - start));
                    float amax = 0f;
                    float vmin = 0f;
                    foreach (float v in group)
                    {
                        amax = Math.Max(amax, Math.Abs(v));
                        vmin = Math.Min(vmin, v);
                    }

                    scales[j] = amax / 15f;
                    mins[j] = Math.Max(0f, -vmin);
                }

                float dBlock = 0f;
                float mBlock = 0f;
                for (int j = 0; j < 8; j++)
                {
                    dBlock = Math.Max(dBlock, scales[j]);
                    mBlock = Math.Max(mBlock, mins[j]);
                }

                Half dHalf = (Half)(dBlock / 63f);
                Half mHalf = (Half)(mBlock / 63f);
                WriteLE(dst, b * 144, (ushort)BitConverter.HalfToInt16Bits(dHalf));
                WriteLE(dst, b * 144 + 2, (ushort)BitConverter.HalfToInt16Bits(mHalf));

                // Quantize group codes against the half-floats exactly as they will be read back.
                float dBack = (float)dHalf;
                float mBack = (float)mHalf;
                byte[] scBits = new byte[8];
                byte[] minBits = new byte[8];
                for (int j = 0; j < 8; j++)
                {
                    int ls = dBack > 0f ? Math.Clamp((int)Math.Round(scales[j] / dBack, MidpointRounding.AwayFromZero), 0, 63) : 0;
                    int lm = mBack > 0f ? Math.Clamp((int)Math.Round(mins[j] / mBack, MidpointRounding.AwayFromZero), 0, 63) : 0;
                    scBits[j] = (byte)ls;
                    minBits[j] = (byte)lm;
                }

                // Pack the six-bit group codes: bytes j/4+j hold low 6 bits of scale_j/min_j and receive high 2 bits of group 4+.
                for (int j = 0; j < 8; j++)
                {
                    if (j < 4)
                    {
                        block[4 + j] |= scBits[j];
                        block[8 + j] |= minBits[j];
                    }
                    else
                    {
                        block[12 + (j - 4)] |= (byte)(scBits[j] & 0xF); // low nibble: scale high bits? see layout below
                        block[12 + (j - 4)] |= (byte)((minBits[j] & 0xF) << 4);
                        block[4 + (j - 4)] |= (byte)((scBits[j] >> 4) << 6);
                        block[8 + (j - 4)] |= (byte)((minBits[j] >> 4) << 6);
                    }
                }

                for (int e = 0; e < srcBlock.Length; e++)
                {
                    int j = e / 32;
                    float dGroup = dBack * scBits[j];
                    if (dGroup == 0f)
                    {
                        continue; // all-zero group: nibbles stay zero, matching the reference behavior
                    }

                    float dmGroup = mBack * minBits[j];
                    int q = Math.Clamp((int)Math.Round(((double)srcBlock[e] + dmGroup) / dGroup, MidpointRounding.AwayFromZero), 0, 15);

                    // Element e lives in group j at window byte 32*(j/2) + (e % 32): low nibble for even groups, high for odd.
                    ref byte target = ref block[16 + 32 * (j / 2) + (e % 32)];
                    if ((j & 1) == 0)
                    {
                        target = (byte)((target & 0xF0) | q);
                    }
                    else
                    {
                        target = (byte)((target & 0x0F) | (q << 4));
                    }
                }
            }
        }
    }

    // Independent oracle: dequantize every stored block to double and dot, with no shared code with the kernels.
    // Also returns per-row sum(|x_i * w_i|), which is the natural scale for float accumulation error bounds.
    private static unsafe (double[] Values, double[] Magnitudes) OracleDot(float[] x, byte[] weights, int rows, int cols, QuantFormat format)
    {
        long stride = MatrixShape(rows, cols).RowStrideBytes(format);
        double[] expected = new double[rows];
        double[] magnitudes = new double[rows];

        fixed (byte* pW = weights)
        fixed (float* pX = x)
        {
            for (int row = 0; row < rows; row++)
            {
                byte* r = pW + row * stride;
                double acc = 0;
                double mag = 0;

                if (format == QuantFormat.F16)
                {
                    for (int i = 0; i < cols; i++)
                    {
                        ushort bits = ReadLE(r, 2 * i);
                        double w = (float)BitConverter.Int16BitsToHalf((short)bits);
                        acc += w * pX[i];
                        mag += Math.Abs(w * pX[i]);
                    }
                }
                else if (format == QuantFormat.Q8_0)
                {
                    int blocks = (cols + 31) / 32;
                    for (int b = 0; b < blocks; b++)
                    {
                        double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(r, b * 34));
                        int limit = Math.Min(32, cols - b * 32);
                        for (int l = 0; l < limit; l++)
                        {
                            double w = (sbyte)r[b * 34 + 2 + l] * d;
                            acc += pX[b * 32 + l] * w;
                            mag += Math.Abs(pX[b * 32 + l] * w);
                        }
                    }
                }
                else if (format == QuantFormat.Q5_0)
                {
                    int blocks = (cols + 31) / 32;
                    for (int b = 0; b < blocks; b++)
                    {
                        byte* block = r + b * 22;
                        double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));
                        uint qh = (uint)(block[2] | (block[3] << 8) | (block[4] << 16) | (block[5] << 24));
                        int limit = Math.Min(32, cols - b * 32);
                        for (int l = 0; l < limit; l++)
                        {
                            int five = ((block[6 + (l & 15)] >> (4 * (l >> 4))) & 0xF) | ((int)(qh >> l) & 1) << 4;
                            double w = (five - 16) * d;
                            acc += pX[b * 32 + l] * w;
                            mag += Math.Abs(pX[b * 32 + l] * w);
                        }
                    }
                }
                else if (format == QuantFormat.Q5_K)
                {
                    int blocks = (cols + 255) / 256;
                    for (int b = 0; b < blocks; b++)
                    {
                        byte* block = r + b * 176;
                        double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));
                        double min = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 2));

                        for (int j = 0; j < 8; j++)
                        {
                            int subStart = b * 256 + j * 32;
                            if (subStart >= cols)
                            {
                                break; // zero-padded tail block carries no real columns
                            }

                            double sc, m;
                            if (j < 4)
                            {
                                sc = block[4 + j] & 0x3F;
                                m = block[8 + j] & 0x3F;
                            }
                            else
                            {
                                sc = (block[12 + (j - 4)] & 0xF) | ((block[4 + (j - 4)] >> 6) << 4);
                                m = (block[12 + (j - 4)] >> 4) | ((block[8 + (j - 4)] >> 6) << 4);
                            }

                            int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.
                            for (int l = 0; l < limit; l++)
                            {
                                byte qByte = block[48 + 32 * (j / 2) + l];
                                int five = ((qByte >> (4 * (j & 1))) & 0xF) | (((block[16 + 32 * (j / 2) + l] >> j) & 1) << 4);
                                double w = (d * sc * five) - (min * m);
                                acc += pX[subStart + l] * w;
                                mag += Math.Abs(pX[subStart + l] * w);
                            }
                        }
                    }
                }
                else if (format == QuantFormat.Q6_K)
                {
                    int blocks = (cols + 255) / 256;
                    for (int b = 0; b < blocks; b++)
                    {
                        byte* block = r + b * 210;
                        double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 208));

                        for (int g = 0; g < 16; g++)
                        {
                            int subStart = b * 256 + g * 16;
                            if (subStart >= cols)
                            {
                                break; // zero-padded tail block carries no real columns
                            }

                            int h = g >> 3; // 128-element half of the super-block
                            int k = g & 7; // 16-element group inside the half
                            int quarter = k >> 1; // 0..3: which 32-element quarter of the half the group starts in
                            double scale = d * (sbyte)block[192 + g];
                            int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.
                            for (int e = 0; e < limit; e++)
                            {
                                int six = ((block[64 * h + 16 * (k & 3) + e] >> (4 * (quarter >> 1))) & 0xF)
                                    | (((block[128 + 32 * h + 16 * (k & 1) + e] >> (2 * quarter)) & 3) << 4);
                                double w = (six - 32) * scale;
                                acc += pX[subStart + e] * w;
                                mag += Math.Abs(pX[subStart + e] * w);
                            }
                        }
                    }
                }
                else if (format == QuantFormat.Q2_K)
                {
                    int blocks = (cols + 255) / 256;
                    for (int b = 0; b < blocks; b++)
                    {
                        byte* block = r + b * 84;
                        double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 80));
                        double min = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 82));

                        for (int g = 0; g < 16; g++)
                        {
                            int subStart = b * 256 + g * 16;
                            if (subStart >= cols)
                            {
                                break; // zero-padded tail block carries no real columns
                            }

                            int h = g >> 3; // 128-element half of the super-block
                            int k = g & 7; // 16-element group inside the half
                            byte sc = block[g];
                            double scale = d * (sc & 0xF);
                            double shift = min * (sc >> 4);
                            int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.
                            for (int e = 0; e < limit; e++)
                            {
                                int two = (block[16 + 32 * h + 16 * (k & 1) + e] >> (2 * (k >> 1))) & 3;
                                double w = scale * two - shift;
                                acc += pX[subStart + e] * w;
                                mag += Math.Abs(pX[subStart + e] * w);
                            }
                        }
                    }
                }
                else if (format == QuantFormat.Q3_K)
                {
                    int blocks = (cols + 255) / 256;
                    for (int b = 0; b < blocks; b++)
                    {
                        byte* block = r + b * 110;
                        double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 108));

                        for (int g = 0; g < 16; g++)
                        {
                            int subStart = b * 256 + g * 16;
                            if (subStart >= cols)
                            {
                                break; // zero-padded tail block carries no real columns
                            }

                            int h = g >> 3; // 128-element half of the super-block
                            int k = g & 7; // 16-element group inside the half
                            int low = g < 8 ? block[96 + g] & 0xF : block[96 + (g - 8)] >> 4;
                            int high = (block[96 + 8 + (g & 3)] >> (2 * (g >> 2))) & 3;
                            double scale = d * ((low | (high << 4)) - 32);
                            int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.
                            for (int e = 0; e < limit; e++)
                            {
                                int two = (block[32 + 32 * h + 16 * (k & 1) + e] >> (2 * (k >> 1))) & 3;
                                int highBit = (block[16 * (k & 1) + e] >> (4 * h + (k >> 1))) & 1;
                                double w = scale * (two + (highBit << 2) - 4);
                                acc += pX[subStart + e] * w;
                                mag += Math.Abs(pX[subStart + e] * w);
                            }
                        }
                    }
                }
                else if (format == QuantFormat.IQ2_XXS)
                {
                    int blocks = (cols + 255) / 256;
                    for (int b = 0; b < blocks; b++)
                    {
                        byte* block = r + b * 66;
                        double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));

                        for (int ib = 0; ib < 8; ib++)
                        {
                            int subStart = b * 256 + ib * 32;
                            if (subStart >= cols)
                            {
                                break; // zero-padded tail block carries no real columns
                            }

                            uint gridAux = (uint)(block[2 + 8 * ib] | (block[3 + 8 * ib] << 8) | (block[4 + 8 * ib] << 16) | (block[5 + 8 * ib] << 24));
                            uint signAux = (uint)(block[6 + 8 * ib] | (block[7 + 8 * ib] << 8) | (block[8 + 8 * ib] << 16) | (block[9 + 8 * ib] << 24));
                            double db = d * (0.5 + (signAux >> 28)) * 0.25;
                            int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.
                            for (int e = 0; e < limit; e++)
                            {
                                int signs = KSignsIQ2Xs[(signAux >> (7 * (e >> 3))) & 127]; // seven-bit sign-group code of the group holding element e
                                int gi = (int)((gridAux >> (8 * (e >> 3))) & 0xFF); // grid index: one entry covers one whole eight-value group
                                int magnitude = (int)((Iq2XXSGridU64[gi] >> (8 * (e & 7))) & 0xFF);
                                double w = db * magnitude * ((signs & (1 << (e & 7))) != 0 ? -1 : 1);
                                acc += pX[subStart + e] * w;
                                mag += Math.Abs(pX[subStart + e] * w);
                            }
                        }
                    }
                }
                else if (format == QuantFormat.IQ2_XS)
                {
                    int blocks = (cols + 255) / 256;
                    for (int b = 0; b < blocks; b++)
                    {
                        byte* block = r + b * 74;
                        double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));

                        for (int ib = 0; ib < 8; ib++)
                        {
                            int subStart = b * 256 + ib * 32;
                            if (subStart >= cols)
                            {
                                break; // zero-padded tail block carries no real columns
                            }

                            byte scaleByte = block[66 + ib]; // one byte carries the sub-block's two scale codes
                            double dbLo = d * (0.5 + (scaleByte & 0xF)) * 0.25;
                            double dbHi = d * (0.5 + (scaleByte >> 4)) * 0.25;
                            int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.
                            for (int e = 0; e < limit; e++)
                            {
                                int word = block[2 + 8 * ib + 2 * (e >> 3)] | (block[3 + 8 * ib + 2 * (e >> 3)] << 8); // one grid word carries one whole eight-value group
                                int signs = KSignsIQ2Xs[word >> 9]; // seven-bit sign-group code of the group holding element e
                                int magnitude = (int)((Iq2XSGridU64[word & 511] >> (8 * (e & 7))) & 0xFF);
                                double w = (e < 16 ? dbLo : dbHi) * magnitude * ((signs & (1 << (e & 7))) != 0 ? -1 : 1);
                                acc += pX[subStart + e] * w;
                                mag += Math.Abs(pX[subStart + e] * w);
                            }
                        }
                    }
                }
                else if (format == QuantFormat.IQ3_XXS)
                {
                    int blocks = (cols + 255) / 256;
                    for (int b = 0; b < blocks; b++)
                    {
                        byte* block = r + b * 98;
                        double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));

                        for (int ib = 0; ib < 8; ib++)
                        {
                            int subStart = b * 256 + ib * 32;
                            if (subStart >= cols)
                            {
                                break; // zero-padded tail block carries no real columns
                            }

                            uint aux = (uint)(block[66 + 4 * ib] | (block[67 + 4 * ib] << 8) | (block[68 + 4 * ib] << 16) | (block[69 + 4 * ib] << 24));
                            double db = d * (0.5 + (aux >> 28)) * 0.5;
                            int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.
                            for (int e = 0; e < limit; e++)
                            {
                                int signs = KSignsIQ2Xs[(aux >> (7 * (e >> 3))) & 127]; // seven-bit sign-group code of the group holding element e
                                int gi = block[2 + 8 * ib + 2 * (e >> 3) + ((e >> 2) & 1)]; // grid index: low four values from the first entry, high four from the second
                                int magnitude = (int)((Iq3XXSGridU32[gi] >> (8 * (e & 3))) & 0xFF);
                                double w = db * magnitude * ((signs & (1 << (e & 7))) != 0 ? -1 : 1);
                                acc += pX[subStart + e] * w;
                                mag += Math.Abs(pX[subStart + e] * w);
                            }
                        }
                    }
                }
                else if (format == QuantFormat.IQ4_NL)
                {
                    int blocks = (cols + 31) / 32;
                    for (int b = 0; b < blocks; b++)
                    {
                        byte* block = r + b * 18;
                        double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));
                        int limit = Math.Min(32, cols - b * 32); // padded nibbles are zero: stop at the real column count.
                        for (int l = 0; l < limit; l++)
                        {
                            int code = (block[2 + (l & 15)] >> (4 * (l >> 4))) & 0xF;
                            double w = d * Iq4KValues[code];
                            acc += pX[b * 32 + l] * w;
                            mag += Math.Abs(pX[b * 32 + l] * w);
                        }
                    }
                }
                else if (format == QuantFormat.IQ4_XS)
                {
                    int blocks = (cols + 255) / 256;
                    for (int b = 0; b < blocks; b++)
                    {
                        byte* block = r + b * 136;
                        double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));
                        ushort scalesH = (ushort)(block[2] | (block[3] << 8));

                        for (int ib = 0; ib < 8; ib++)
                        {
                            int subStart = b * 256 + ib * 32;
                            if (subStart >= cols)
                            {
                                break; // zero-padded tail block carries no real columns
                            }

                            int ls = ((block[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scalesH >> (2 * ib)) & 3) << 4);
                            double scale = d * (ls - 32);
                            int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.
                            for (int l = 0; l < limit; l++)
                            {
                                int code = (block[8 + 16 * ib + (l & 15)] >> (4 * (l >> 4))) & 0xF;
                                double w = scale * Iq4KValues[code];
                                acc += pX[subStart + l] * w;
                                mag += Math.Abs(pX[subStart + l] * w);
                            }
                        }
                    }
                }
                else
                {
                    int blocks = (cols + 255) / 256;
                    for (int b = 0; b < blocks; b++)
                    {
                        byte* block = r + b * 144;
                        double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));
                        double min = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 2));

                        for (int j = 0; j < 8; j++)
                        {
                            int subStart = b * 256 + j * 32;
                            if (subStart >= cols)
                            {
                                break; // zero-padded tail block carries no real columns
                            }

                            double sc, m;
                            if (j < 4)
                            {
                                sc = block[4 + j] & 0x3F;
                                m = block[8 + j] & 0x3F;
                            }
                            else
                            {
                                sc = (block[12 + (j - 4)] & 0xF) | ((block[4 + (j - 4)] >> 6) << 4);
                                m = (block[12 + (j - 4)] >> 4) | ((block[8 + (j - 4)] >> 6) << 4);
                            }

                            int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.
                            for (int l = 0; l < limit; l++)
                            {
                                byte qByte = block[16 + 32 * (j / 2) + l];
                                int q = (qByte >> (4 * (j & 1))) & 0xF;
                                double w = (d * sc * q) - (min * m);
                                acc += pX[subStart + l] * w;
                                mag += Math.Abs(pX[subStart + l] * w);
                            }
                        }
                    }
                }

                expected[row] = acc;
                magnitudes[row] = mag;
            }
        }

        return (expected, magnitudes);
    }

    private static void AssertClose(ReadOnlySpan<float> actual, ReadOnlySpan<double> expected, ReadOnlySpan<double> scale, double relativeTolerance, string context)
    {
        for (int i = 0; i < actual.Length; i++)
        {
            double diff = Math.Abs(actual[i] - expected[i]);

            // Error of a float dot product accumulates like eps * sum|x*w|: bound it against that scale.
            Assert.True(diff <= relativeTolerance * Math.Max(1e-6, scale[i]), $"{context}: output[{i}] differs by {diff:G6} (expected {expected[i]:G9}, got {actual[i]:G9})");
        }
    }

    private static void WriteLE(Span<byte> dst, int offset, ushort value)
    {
        dst[offset] = (byte)value;
        dst[offset + 1] = (byte)(value >> 8);
    }

    // Independent row oracle: decode one packed row to double with naive block math, sharing no code with the kernels.
    private static unsafe double[] OracleDequantRow(byte[] weights, int row, int cols, QuantFormat format)
    {
        long stride = MatrixShape(1, cols).RowStrideBytes(format);

        fixed (byte* pW = weights)
        {
            byte* r = pW + row * stride;
            double[] values = new double[cols];

            if (format == QuantFormat.F16)
            {
                for (int i = 0; i < cols; i++)
                {
                    values[i] = (float)BitConverter.Int16BitsToHalf((short)ReadLE(r, 2 * i));
                }

                return values;
            }

            if (format == QuantFormat.Q8_0)
            {
                int blocks = (cols + 31) / 32;
                for (int b = 0; b < blocks; b++)
                {
                    double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(r, b * 34));
                    int limit = Math.Min(32, cols - b * 32);
                    for (int l = 0; l < limit; l++)
                    {
                        values[b * 32 + l] = (sbyte)r[b * 34 + 2 + l] * d;
                    }
                }

                return values;
            }

            if (format == QuantFormat.Q5_0)
            {
                int blocks = (cols + 31) / 32;
                for (int b = 0; b < blocks; b++)
                {
                    byte* block = r + b * 22;
                    double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));
                    uint qh = (uint)(block[2] | (block[3] << 8) | (block[4] << 16) | (block[5] << 24));
                    int limit = Math.Min(32, cols - b * 32);
                    for (int l = 0; l < limit; l++)
                    {
                        int five = ((block[6 + (l & 15)] >> (4 * (l >> 4))) & 0xF) | ((int)(qh >> l) & 1) << 4;
                        values[b * 32 + l] = (five - 16) * d;
                    }
                }

                return values;
            }

            if (format == QuantFormat.Q5_K)
            {
                int blocks = (cols + 255) / 256;
                for (int b = 0; b < blocks; b++)
                {
                    byte* block = r + b * 176;
                    double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));
                    double min = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 2));

                    for (int j = 0; j < 8; j++)
                    {
                        int subStart = b * 256 + j * 32;
                        if (subStart >= cols)
                        {
                            break; // zero-padded tail block carries no real columns
                        }

                        double sc, m;
                        if (j < 4)
                        {
                            sc = block[4 + j] & 0x3F;
                            m = block[8 + j] & 0x3F;
                        }
                        else
                        {
                            sc = (block[12 + (j - 4)] & 0xF) | ((block[4 + (j - 4)] >> 6) << 4);
                            m = (block[12 + (j - 4)] >> 4) | ((block[8 + (j - 4)] >> 6) << 4);
                        }

                        int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.
                        for (int l = 0; l < limit; l++)
                        {
                            byte qByte = block[48 + 32 * (j / 2) + l];
                            int five = ((qByte >> (4 * (j & 1))) & 0xF) | (((block[16 + 32 * (j / 2) + l] >> j) & 1) << 4);
                            values[subStart + l] = (d * sc * five) - (min * m);
                        }
                    }
                }

                return values;
            }

            if (format == QuantFormat.Q6_K)
            {
                int blocks = (cols + 255) / 256;
                for (int b = 0; b < blocks; b++)
                {
                    byte* block = r + b * 210;
                    double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 208));

                    for (int g = 0; g < 16; g++)
                    {
                        int subStart = b * 256 + g * 16;
                        if (subStart >= cols)
                        {
                            break; // zero-padded tail block carries no real columns
                        }

                        int h = g >> 3; // 128-element half of the super-block
                        int k = g & 7; // 16-element group inside the half
                        int quarter = k >> 1; // 0..3: which 32-element quarter of the half the group starts in
                        double scale = d * (sbyte)block[192 + g];
                        int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.
                        for (int e = 0; e < limit; e++)
                        {
                            int six = ((block[64 * h + 16 * (k & 3) + e] >> (4 * (quarter >> 1))) & 0xF)
                                | (((block[128 + 32 * h + 16 * (k & 1) + e] >> (2 * quarter)) & 3) << 4);
                            values[subStart + e] = (six - 32) * scale;
                        }
                    }
                }

                return values;
            }

            if (format == QuantFormat.Q2_K)
            {
                int blocks = (cols + 255) / 256;
                for (int b = 0; b < blocks; b++)
                {
                    byte* block = r + b * 84;
                    double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 80));
                    double min = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 82));

                    for (int g = 0; g < 16; g++)
                    {
                        int subStart = b * 256 + g * 16;
                        if (subStart >= cols)
                        {
                            break; // zero-padded tail block carries no real columns
                        }

                        int h = g >> 3; // 128-element half of the super-block
                        int k = g & 7; // 16-element group inside the half
                        byte sc = block[g];
                        double scale = d * (sc & 0xF);
                        double shift = min * (sc >> 4);
                        int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.
                        for (int e = 0; e < limit; e++)
                        {
                            int two = (block[16 + 32 * h + 16 * (k & 1) + e] >> (2 * (k >> 1))) & 3;
                            values[subStart + e] = scale * two - shift;
                        }
                    }
                }

                return values;
            }

            if (format == QuantFormat.Q3_K)
            {
                int blocks = (cols + 255) / 256;
                for (int b = 0; b < blocks; b++)
                {
                    byte* block = r + b * 110;
                    double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 108));

                    for (int g = 0; g < 16; g++)
                    {
                        int subStart = b * 256 + g * 16;
                        if (subStart >= cols)
                        {
                            break; // zero-padded tail block carries no real columns
                        }

                        int h = g >> 3; // 128-element half of the super-block
                        int k = g & 7; // 16-element group inside the half
                        int low = g < 8 ? block[96 + g] & 0xF : block[96 + (g - 8)] >> 4;
                        int high = (block[96 + 8 + (g & 3)] >> (2 * (g >> 2))) & 3;
                        double scale = d * ((low | (high << 4)) - 32);
                        int limit = Math.Min(16, cols - subStart); // padded quants are zero: stop at the real column count.
                        for (int e = 0; e < limit; e++)
                        {
                            int two = (block[32 + 32 * h + 16 * (k & 1) + e] >> (2 * (k >> 1))) & 3;
                            int highBit = (block[16 * (k & 1) + e] >> (4 * h + (k >> 1))) & 1;
                            values[subStart + e] = scale * (two + (highBit << 2) - 4);
                        }
                    }
                }

                return values;
            }

            if (format == QuantFormat.IQ2_XXS)
            {
                int blocks = (cols + 255) / 256;
                for (int b = 0; b < blocks; b++)
                {
                    byte* block = r + b * 66;
                    double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));

                    for (int ib = 0; ib < 8; ib++)
                    {
                        int subStart = b * 256 + ib * 32;
                        if (subStart >= cols)
                        {
                            break; // zero-padded tail block carries no real columns
                        }

                        uint gridAux = (uint)(block[2 + 8 * ib] | (block[3 + 8 * ib] << 8) | (block[4 + 8 * ib] << 16) | (block[5 + 8 * ib] << 24));
                        uint signAux = (uint)(block[6 + 8 * ib] | (block[7 + 8 * ib] << 8) | (block[8 + 8 * ib] << 16) | (block[9 + 8 * ib] << 24));
                        double db = d * (0.5 + (signAux >> 28)) * 0.25;
                        int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.
                        for (int e = 0; e < limit; e++)
                        {
                            int signs = KSignsIQ2Xs[(signAux >> (7 * (e >> 3))) & 127]; // seven-bit sign-group code of the group holding element e
                            int gi = (int)((gridAux >> (8 * (e >> 3))) & 0xFF); // grid index: one entry covers one whole eight-value group
                            int magnitude = (int)((Iq2XXSGridU64[gi] >> (8 * (e & 7))) & 0xFF);
                            values[subStart + e] = db * magnitude * ((signs & (1 << (e & 7))) != 0 ? -1 : 1);
                        }
                    }
                }

                return values;
            }

            if (format == QuantFormat.IQ2_XS)
            {
                int blocks = (cols + 255) / 256;
                for (int b = 0; b < blocks; b++)
                {
                    byte* block = r + b * 74;
                    double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));

                    for (int ib = 0; ib < 8; ib++)
                    {
                        int subStart = b * 256 + ib * 32;
                        if (subStart >= cols)
                        {
                            break; // zero-padded tail block carries no real columns
                        }

                        byte scaleByte = block[66 + ib]; // one byte carries the sub-block's two scale codes
                        double dbLo = d * (0.5 + (scaleByte & 0xF)) * 0.25;
                        double dbHi = d * (0.5 + (scaleByte >> 4)) * 0.25;
                        int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.
                        for (int e = 0; e < limit; e++)
                        {
                            int word = block[2 + 8 * ib + 2 * (e >> 3)] | (block[3 + 8 * ib + 2 * (e >> 3)] << 8); // one grid word carries one whole eight-value group
                            int signs = KSignsIQ2Xs[word >> 9]; // seven-bit sign-group code of the group holding element e
                            int magnitude = (int)((Iq2XSGridU64[word & 511] >> (8 * (e & 7))) & 0xFF);
                            values[subStart + e] = (e < 16 ? dbLo : dbHi) * magnitude * ((signs & (1 << (e & 7))) != 0 ? -1 : 1);
                        }
                    }
                }

                return values;
            }

            if (format == QuantFormat.IQ3_XXS)
            {
                int blocks = (cols + 255) / 256;
                for (int b = 0; b < blocks; b++)
                {
                    byte* block = r + b * 98;
                    double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));

                    for (int ib = 0; ib < 8; ib++)
                    {
                        int subStart = b * 256 + ib * 32;
                        if (subStart >= cols)
                        {
                            break; // zero-padded tail block carries no real columns
                        }

                        uint aux = (uint)(block[66 + 4 * ib] | (block[67 + 4 * ib] << 8) | (block[68 + 4 * ib] << 16) | (block[69 + 4 * ib] << 24));
                        double db = d * (0.5 + (aux >> 28)) * 0.5;
                        int limit = Math.Min(32, cols - subStart); // padded quants are zero: stop at the real column count.
                        for (int e = 0; e < limit; e++)
                        {
                            int signs = KSignsIQ2Xs[(aux >> (7 * (e >> 3))) & 127]; // seven-bit sign-group code of the group holding element e
                            int gi = block[2 + 8 * ib + 2 * (e >> 3) + ((e >> 2) & 1)]; // grid index: low four values from the first entry, high four from the second
                            int magnitude = (int)((Iq3XXSGridU32[gi] >> (8 * (e & 3))) & 0xFF);
                            values[subStart + e] = db * magnitude * ((signs & (1 << (e & 7))) != 0 ? -1 : 1);
                        }
                    }
                }

                return values;
            }

            if (format == QuantFormat.IQ4_NL)
            {
                int blocks = (cols + 31) / 32;
                for (int b = 0; b < blocks; b++)
                {
                    byte* block = r + b * 18;
                    double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));
                    int limit = Math.Min(32, cols - b * 32); // padded nibbles are zero: stop at the real column count.
                    for (int l = 0; l < limit; l++)
                    {
                        values[b * 32 + l] = d * Iq4KValues[(block[2 + (l & 15)] >> (4 * (l >> 4))) & 0xF];
                    }
                }

                return values;
            }

            if (format == QuantFormat.IQ4_XS)
            {
                int blocks = (cols + 255) / 256;
                for (int b = 0; b < blocks; b++)
                {
                    byte* block = r + b * 136;
                    double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));
                    ushort scalesH = (ushort)(block[2] | (block[3] << 8));

                    for (int ib = 0; ib < 8; ib++)
                    {
                        int subStart = b * 256 + ib * 32;
                        if (subStart >= cols)
                        {
                            break; // zero-padded tail block carries no real columns
                        }

                        int ls = ((block[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scalesH >> (2 * ib)) & 3) << 4);
                        double scale = d * (ls - 32);
                        int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.
                        for (int l = 0; l < limit; l++)
                        {
                            values[subStart + l] = scale * Iq4KValues[(block[8 + 16 * ib + (l & 15)] >> (4 * (l >> 4))) & 0xF];
                        }
                    }
                }

                return values;
            }

            int qBlocks = (cols + 255) / 256;
            for (int b = 0; b < qBlocks; b++)
            {
                byte* block = r + b * 144;
                double d = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 0));
                double min = (float)BitConverter.Int16BitsToHalf((short)ReadLE(block, 2));

                for (int j = 0; j < 8; j++)
                {
                    int subStart = b * 256 + j * 32;
                    if (subStart >= cols)
                    {
                        break; // zero-padded tail block carries no real columns
                    }

                    double sc, m;
                    if (j < 4)
                    {
                        sc = block[4 + j] & 0x3F;
                        m = block[8 + j] & 0x3F;
                    }
                    else
                    {
                        sc = (block[12 + (j - 4)] & 0xF) | ((block[4 + (j - 4)] >> 6) << 4);
                        m = (block[12 + (j - 4)] >> 4) | ((block[8 + (j - 4)] >> 6) << 4);
                    }

                    int limit = Math.Min(32, cols - subStart); // padded nibbles are zero: stop at the real column count.
                    for (int l = 0; l < limit; l++)
                    {
                        byte qByte = block[16 + 32 * (j / 2) + l];
                        int q = (qByte >> (4 * (j & 1))) & 0xF;
                        values[subStart + l] = (d * sc * q) - (min * m);
                    }
                }
            }

            return values;
        }
    }

    private static void AssertValuesClose(ReadOnlySpan<float> actual, ReadOnlySpan<double> expected, double relativeTolerance, string context)
    {
        for (int i = 0; i < actual.Length; i++)
        {
            double diff = Math.Abs(actual[i] - expected[i]);

            // Per-element dequant error is bounded by eps * |w|: bound it against the value magnitude.
            Assert.True(diff <= relativeTolerance * Math.Max(1e-6, Math.Abs(expected[i])), $"{context}: [{i}] differs by {diff:G6} (expected {expected[i]:G9}, got {actual[i]:G9})");
        }
    }

    private static double MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double max = 0;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs(a[i] - b[i]));
        }

        return max;
    }

    private static double MaxAbs(ReadOnlySpan<double> v)
    {
        double max = 0;
        foreach (double d in v)
        {
            max = Math.Max(max, Math.Abs(d));
        }

        return max;
    }

    private static unsafe ushort ReadLE(byte* p, int offset) => (ushort)(p[offset] | (p[offset + 1] << 8));
}
