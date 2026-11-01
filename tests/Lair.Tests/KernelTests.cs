using System.Buffers;
using Lair.Backends.Cpu;
using Lair.Core.Contracts;

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

    private static unsafe ushort ReadLE(byte* p, int offset) => (ushort)(p[offset] | (p[offset + 1] << 8));
}
