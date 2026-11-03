using System.Runtime.InteropServices;
using Lair.Backends.Cpu;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// Batched GEMV tests: for every quant family the batched path must produce results bit-identical to N
/// independent single-vector GEMV calls, across batch sizes, thread counts, and the scalar and AVX2 paths.
/// Comparisons run on the raw bits of the floats, so NaN payloads and signed zeros compare too.
/// </summary>
[Trait("Category", "Kernel")]
public class BatchKernelTests
{
    private static readonly (int Rows, int Cols)[] MatrixSizes = [(100, 137), (256, 256), (512, 1024)];

    private static readonly int[] BatchCounts = [1, 2, 8, 16];

    [Theory]
    [InlineData(QuantFormat.F32)]
    [InlineData(QuantFormat.F16)]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q4_K)]
    public void GemvBatch_IsBitIdenticalToIndependentSingleVectorCalls(QuantFormat format)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            foreach (int count in BatchCounts)
            {
                (float[] inputs, byte[] weights) = CreateCase(rows, cols, format, seed: 1234 + count, vectors: count);

                using CpuBackend single = new(threads: 1);
                using CpuBackend batch = new(threads: 1);

                float[] expected = new float[count * rows];
                for (int b = 0; b < count; b++)
                {
                    single.Gemv(inputs.AsSpan(b * cols, cols), new WeightRef(weights, MatrixShape(rows, cols), format), expected.AsSpan(b * rows, rows));
                }

                float[] actual = new float[count * rows];
                batch.GemvBatch(inputs, count, new WeightRef(weights, MatrixShape(rows, cols), format), actual);

                AssertBitsEqual(expected, actual, $"{format} batch({count}) at {rows}x{cols}");
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.F32)]
    [InlineData(QuantFormat.F16)]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q4_K)]
    public void GemvBatch_ScalarPath_IsBitIdenticalToScalarSingleVectorCalls(QuantFormat format)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            foreach (int count in new[] { 1, 8 })
            {
                (float[] inputs, byte[] weights) = CreateCase(rows, cols, format, seed: 77 + count, vectors: count);

                using CpuBackend single = new(threads: 1, allowAvx2: false);
                using CpuBackend batch = new(threads: 1, allowAvx2: false);

                float[] expected = new float[count * rows];
                for (int b = 0; b < count; b++)
                {
                    single.Gemv(inputs.AsSpan(b * cols, cols), new WeightRef(weights, MatrixShape(rows, cols), format), expected.AsSpan(b * rows, rows));
                }

                float[] actual = new float[count * rows];
                batch.GemvBatch(inputs, count, new WeightRef(weights, MatrixShape(rows, cols), format), actual);

                AssertBitsEqual(expected, actual, $"scalar {format} batch({count}) at {rows}x{cols}");
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.F32)]
    [InlineData(QuantFormat.F16)]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q4_K)]
    public void GemvBatch_Threaded_IsBitIdenticalToSingleThreaded(QuantFormat format)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            const int count = 8;
            (float[] inputs, byte[] weights) = CreateCase(rows, cols, format, seed: 55, vectors: count);

            using CpuBackend single = new(threads: 1);
            using CpuBackend parallel = new(threads: 4);

            float[] expected = new float[count * rows];
            float[] actual = new float[count * rows];
            single.GemvBatch(inputs, count, new WeightRef(weights, MatrixShape(rows, cols), format), expected);
            parallel.GemvBatch(inputs, count, new WeightRef(weights, MatrixShape(rows, cols), format), actual);

            AssertBitsEqual(expected, actual, $"threaded {format} batch({count}) at {rows}x{cols}");
        }
    }

    [Theory]
    [InlineData(QuantFormat.F16)]
    [InlineData(QuantFormat.Q4_K)]
    public void GemvBatch_BadCountOrSpanLengths_ThrowsClearMessage(QuantFormat format)
    {
        TensorShape shape = MatrixShape(4, 70);
        long expectedBytes = 4 * shape.RowStrideBytes(format);
        byte[] packed = new byte[expectedBytes];
        float[] inputs = new float[2 * 70];

        using CpuBackend backend = new();

        ArgumentOutOfRangeException countLow = Assert.Throws<ArgumentOutOfRangeException>(
            () => backend.GemvBatch(inputs, 0, new WeightRef(packed, shape, format), new float[2 * 4]));
        Assert.Contains("count", countLow.ParamName);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => backend.GemvBatch(inputs, 17, new WeightRef(packed, shape, format), new float[17 * 4]));

        ArgumentException inputEx = Assert.Throws<ArgumentException>(
            () => backend.GemvBatch(new float[2 * 69], 2, new WeightRef(packed, shape, format), new float[2 * 4]));
        Assert.Contains("count", inputEx.Message);

        Assert.Throws<ArgumentException>(
            () => backend.GemvBatch(inputs, 2, new WeightRef(packed, shape, format), new float[2 * 4 - 1]));

        Assert.Throws<ArgumentException>(
            () => backend.GemvBatch(inputs, 2, new WeightRef(new byte[expectedBytes - 1], shape, format), new float[2 * 4]));
    }

    private static TensorShape MatrixShape(int rows, int cols) => new([rows, cols]);

    // Bit-exact comparison over the raw bits of the outputs: NaN payloads and signed zeros must match too.
    private static void AssertBitsEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string context)
    {
        ReadOnlySpan<int> expectedBits = MemoryMarshal.Cast<float, int>(expected);
        ReadOnlySpan<int> actualBits = MemoryMarshal.Cast<float, int>(actual);

        for (int i = 0; i < expectedBits.Length; i++)
        {
            Assert.True(
                expectedBits[i] == actualBits[i],
                $"{context}: output[{i}] differs: expected {expected[i]:G9} (bits {expectedBits[i]:X8}), got {actual[i]:G9} (bits {actualBits[i]:X8})");
        }
    }

    private static (float[] Inputs, byte[] Weights) CreateCase(int rows, int cols, QuantFormat format, int seed, int vectors)
    {
        Random rng = new(seed);
        float[] inputs = NextFloats(rng, vectors * cols);
        float[] weightsFloat = NextFloats(rng, rows * cols);

        long stride = MatrixShape(rows, cols).RowStrideBytes(format);
        byte[] packed = new byte[rows * stride];

        for (int row = 0; row < rows; row++)
        {
            ReadOnlySpan<float> src = weightsFloat.AsSpan(row * cols, cols);
            Span<byte> dst = packed.AsSpan((int)(row * stride), (int)stride);
            switch (format)
            {
                case QuantFormat.F32:
                    src.CopyTo(MemoryMarshal.Cast<byte, float>(dst));
                    break;
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

        return (inputs, packed);
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
                        block[12 + (j - 4)] |= (byte)(scBits[j] & 0xF);
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

    private static void WriteLE(Span<byte> dst, int offset, ushort value)
    {
        dst[offset] = (byte)value;
        dst[offset + 1] = (byte)(value >> 8);
    }
}
