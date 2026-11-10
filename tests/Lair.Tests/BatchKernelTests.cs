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
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ4_XS)]
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
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ4_XS)]
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
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ4_XS)]
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
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ4_XS)]
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

    // Test-only Q5_K writer mirroring the reference layout: Q4_K-style block/group scales and mins, then 5-bit
    // codes q in [0, 31] against dGroup = d*sc and dmGroup = dmin*m. Nibbles live in the 128-byte qs region at
    // block+48 (group-pair windows, low nibble even groups, high odd); the 5th bit lives at bit j of
    // qh[32*(j/2) + l] in the 32-byte qh region at block+16. Padded elements produce zero blocks.
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
