using System.Buffers;
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
    [InlineData(QuantFormat.F32)]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q6_K)]
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
