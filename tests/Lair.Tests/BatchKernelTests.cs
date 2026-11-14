using System.Numerics;
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

    // Identity coverage runs to the backend's batch ceiling: what a decode batch advances and every prompt row width a forward engine may resolve.
    private static readonly int[] BatchCounts = [1, 2, 8, 16, 32, 64];

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
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ2_S)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ3_S)]
    [InlineData(QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.Q2_0)]
    [InlineData(QuantFormat.BF16)]
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
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ2_S)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ3_S)]
    [InlineData(QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.Q2_0)]
    [InlineData(QuantFormat.BF16)]
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
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ2_S)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ3_S)]
    [InlineData(QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.Q2_0)]
    [InlineData(QuantFormat.BF16)]
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
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ2_S)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ3_S)]
    [InlineData(QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.Q2_0)]
    [InlineData(QuantFormat.BF16)]
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
            () => backend.GemvBatch(inputs, 65, new WeightRef(packed, shape, format), new float[65 * 4])); // one past the batch ceiling every kernel's scratch is sized for

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
    internal static void AssertBitsEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string context)
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

    internal static (float[] Inputs, byte[] Weights) CreateCase(int rows, int cols, QuantFormat format, int seed, int vectors)
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
                case QuantFormat.IQ2_XXS:
                    QuantizeIQ2Xxs(src, dst);
                    break;
                case QuantFormat.IQ2_XS:
                    QuantizeIQ2Xs(src, dst);
                    break;
                case QuantFormat.IQ2_S:
                    QuantizeIQ2S(src, dst);
                    break;
                case QuantFormat.IQ3_XXS:
                    QuantizeIQ3Xxs(src, dst);
                    break;
                case QuantFormat.IQ3_S:
                    QuantizeIQ3S(src, dst);
                    break;
                case QuantFormat.IQ4_NL:
                    QuantizeIQ4N(src, dst);
                    break;
                case QuantFormat.IQ4_XS:
                    QuantizeIQ4X(src, dst);
                    break;
                case QuantFormat.Q2_0:
                    QuantizeQ20(src, dst);
                    break;
                case QuantFormat.BF16:
                    QuantizeBF16(src, dst);
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

    // Test-only Q2_0 writer: per 64-element block, d = amax/2 stored as half-float at byte 0, and each element
    // takes the nearest of the four code values (00 = -1, 01 = 0, 10 = +1, 11 = +2 scaled by d). Element j lives
    // at the 2-bit field 2*(j % 4) of qs[j / 4] (block bytes 2..17). Padded elements keep zero quants.
    private static void QuantizeQ20(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int blocks = (src.Length + 63) / 64;
        for (int b = 0; b < blocks; b++)
        {
            ReadOnlySpan<float> block = src.Slice(b * 64, Math.Min(64, src.Length - b * 64));

            float amax = 0f;
            foreach (float v in block)
            {
                amax = Math.Max(amax, Math.Abs(v));
            }

            Half dHalf = (Half)(amax / 2f);
            WriteLE(dst, b * 18, (ushort)BitConverter.HalfToInt16Bits(dHalf));

            if (dHalf == (Half)0f)
            {
                continue; // all-zero block: quants stay zero, matching the reference behavior
            }

            for (int l = 0; l < block.Length; l++)
            {
                int two = Math.Clamp((int)Math.Round((double)block[l] / (float)dHalf, MidpointRounding.AwayFromZero) + 1, 0, 3);
                ref byte target = ref dst[b * 18 + 2 + (l >> 2)];
                int shift = 2 * (l & 3);
                target = (byte)((target & ~(3 << shift)) | (two << shift)); // padded quants stay zero
            }
        }
    }

    // Test-only BF16 writer: round-to-nearest-even f32-to-bf16 (rounding bias and carry as in the reference
    // conversion), stored little-endian with the surviving 16 bits as the f32 high half.
    private static void QuantizeBF16(ReadOnlySpan<float> src, Span<byte> dst)
    {
        for (int i = 0; i < src.Length; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(src[i]);
            ushort code = (ushort)((bits + (0x7FFF + ((bits >> 16) & 1))) >> 16);
            WriteLE(dst, 2 * i, code);
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

                // The 5th bit lives at bit j of plane byte (e % 32) in the fixed qh region: every group writes into
                // the same 32 bytes, so one byte carries eight groups' bits.
                dst[b * 176 + 16 + (e % 32)] |= (byte)(((five >> 4) & 1) << j);
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

    // Test-local copy of the shared IQ2_S grid as 1024 uint64 magnitudes: entry i's little-endian bytes are
    // the eight magnitude bytes the kernels read at grid[i * 8 + n].
    private static readonly ulong[] Iq2SGridU64 =
    [
        0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08, 0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
        0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919, 0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
        0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919, 0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
        0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
        0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
        0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
        0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19, 0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
        0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919, 0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
        0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
        0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919, 0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
        0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
        0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b, 0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
        0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
        0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b, 0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
        0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
        0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808, 0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
        0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b, 0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
        0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08, 0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
        0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
        0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819, 0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
        0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908, 0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
        0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b, 0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
        0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808, 0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
        0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919, 0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
        0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808, 0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
        0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b, 0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
        0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
        0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
        0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919, 0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
        0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819, 0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
        0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908, 0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
        0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908, 0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
        0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908, 0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
        0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908, 0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
        0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819, 0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
        0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
        0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819, 0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
        0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819, 0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
        0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808, 0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
        0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08, 0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
        0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908, 0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
        0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819, 0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
        0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819, 0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
        0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808, 0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
        0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908, 0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
        0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819, 0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
        0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
        0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19, 0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
        0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919, 0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
        0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808, 0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
        0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908, 0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
        0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08, 0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
        0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908, 0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
        0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808, 0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
        0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908, 0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
        0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819, 0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
        0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808, 0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
        0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808, 0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
        0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b, 0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
        0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
        0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
        0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19, 0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
        0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08, 0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
        0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908, 0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
        0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b, 0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
        0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808, 0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
        0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
        0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08, 0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
        0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
        0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b, 0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
        0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b, 0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
        0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08, 0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
        0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808, 0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
        0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919, 0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
        0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808, 0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
        0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819, 0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
        0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808, 0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
        0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919, 0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
        0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b, 0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
        0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808, 0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
        0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
        0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b, 0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
        0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b, 0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
        0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08, 0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
        0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b, 0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
        0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08, 0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
        0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919, 0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
        0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19, 0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
        0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919, 0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
        0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819, 0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
        0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919, 0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
        0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808, 0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
        0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919, 0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
        0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b, 0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
        0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919, 0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
        0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08, 0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
        0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808, 0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
        0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908, 0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
        0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808, 0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
        0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908, 0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
        0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808, 0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
        0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908, 0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
        0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
        0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919, 0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
        0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808, 0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
        0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b, 0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
        0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08, 0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
        0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919, 0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
        0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908, 0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
        0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808, 0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
        0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908, 0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
        0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808, 0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
        0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808, 0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
        0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908, 0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
        0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08, 0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
        0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b, 0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
        0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819, 0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
        0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819, 0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
        0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808, 0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
        0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908, 0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
        0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908, 0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
        0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819, 0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
        0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b, 0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
        0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819, 0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
        0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b, 0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
        0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819, 0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
        0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819, 0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
        0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808, 0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
    ];

    // Test-local copy of the shared IQ3_S grid as 512 uint32 magnitudes: entry i's little-endian bytes are
    // the four magnitude bytes the kernels read at grid[i * 4 + n].
    private static readonly uint[] Iq3SGridU32 =
    [
        0x01010101, 0x01010103, 0x01010105, 0x0101010B, 0x0101010F, 0x01010301, 0x01010303, 0x01010305,
        0x01010309, 0x0101030D, 0x01010501, 0x01010503, 0x0101050B, 0x01010707, 0x01010901, 0x01010905,
        0x0101090B, 0x0101090F, 0x01010B03, 0x01010B07, 0x01010D01, 0x01010D05, 0x01010F03, 0x01010F09,
        0x01010F0F, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030B,
        0x01030501, 0x01030507, 0x0103050F, 0x01030703, 0x0103070B, 0x01030909, 0x01030D03, 0x01030D0B,
        0x01030F05, 0x01050101, 0x01050103, 0x0105010B, 0x0105010F, 0x01050301, 0x01050307, 0x0105030D,
        0x01050503, 0x0105050B, 0x01050701, 0x01050709, 0x01050905, 0x0105090B, 0x0105090F, 0x01050B03,
        0x01050B07, 0x01050F01, 0x01050F07, 0x01070107, 0x01070303, 0x0107030B, 0x01070501, 0x01070505,
        0x01070703, 0x01070707, 0x0107070D, 0x01070909, 0x01070B01, 0x01070B05, 0x01070D0F, 0x01070F03,
        0x01070F0B, 0x01090101, 0x01090307, 0x0109030F, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
        0x01090907, 0x01090B03, 0x01090F01, 0x010B0105, 0x010B0109, 0x010B0501, 0x010B0505, 0x010B050D,
        0x010B0707, 0x010B0903, 0x010B090B, 0x010B090F, 0x010B0D0D, 0x010B0F07, 0x010D010D, 0x010D0303,
        0x010D0307, 0x010D0703, 0x010D0B05, 0x010D0F03, 0x010F0101, 0x010F0105, 0x010F0109, 0x010F0501,
        0x010F0505, 0x010F050D, 0x010F0707, 0x010F0B01, 0x010F0B09, 0x03010101, 0x03010103, 0x03010105,
        0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030B, 0x0301030F, 0x03010501, 0x03010505,
        0x03010703, 0x03010709, 0x0301070D, 0x03010B09, 0x03010B0D, 0x03010D03, 0x03010F05, 0x03030101,
        0x03030103, 0x03030107, 0x0303010D, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
        0x03030903, 0x03030B01, 0x03030B05, 0x03030F01, 0x03030F0D, 0x03050101, 0x03050305, 0x0305030B,
        0x0305030F, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050B0B, 0x03050D01,
        0x03050F05, 0x03070103, 0x03070109, 0x0307010F, 0x03070301, 0x03070307, 0x03070503, 0x0307050F,
        0x03070701, 0x03070709, 0x03070903, 0x03070D05, 0x03070F01, 0x03090107, 0x0309010B, 0x03090305,
        0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090D, 0x03090B01, 0x03090B09, 0x030B0103,
        0x030B0301, 0x030B0307, 0x030B0503, 0x030B0701, 0x030B0705, 0x030B0B03, 0x030D0501, 0x030D0509,
        0x030D050F, 0x030D0909, 0x030D090D, 0x030F0103, 0x030F0107, 0x030F0301, 0x030F0305, 0x030F0503,
        0x030F070B, 0x030F0903, 0x030F0D05, 0x030F0F01, 0x05010101, 0x05010103, 0x05010107, 0x0501010B,
        0x0501010F, 0x05010301, 0x05010305, 0x05010309, 0x0501030D, 0x05010503, 0x05010507, 0x0501050F,
        0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090B, 0x05010B01, 0x05010B05, 0x05010D0F,
        0x05010F01, 0x05010F07, 0x05010F0B, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030F,
        0x05030505, 0x0503050B, 0x05030703, 0x05030709, 0x05030905, 0x05030B03, 0x05050103, 0x05050109,
        0x0505010F, 0x05050503, 0x05050507, 0x05050701, 0x0505070F, 0x05050903, 0x05050B07, 0x05050B0F,
        0x05050F03, 0x05050F09, 0x05070101, 0x05070105, 0x0507010B, 0x05070303, 0x05070505, 0x05070509,
        0x05070703, 0x05070707, 0x05070905, 0x05070B01, 0x05070D0D, 0x05090103, 0x0509010F, 0x05090501,
        0x05090507, 0x05090705, 0x0509070B, 0x05090903, 0x05090F05, 0x05090F0B, 0x050B0109, 0x050B0303,
        0x050B0505, 0x050B070F, 0x050B0901, 0x050B0B07, 0x050B0F01, 0x050D0101, 0x050D0105, 0x050D010F,
        0x050D0503, 0x050D0B0B, 0x050D0D03, 0x050F010B, 0x050F0303, 0x050F050D, 0x050F0701, 0x050F0907,
        0x050F0B01, 0x07010105, 0x07010303, 0x07010307, 0x0701030B, 0x0701030F, 0x07010505, 0x07010703,
        0x07010707, 0x0701070B, 0x07010905, 0x07010909, 0x0701090F, 0x07010B03, 0x07010D07, 0x07010F03,
        0x07030103, 0x07030107, 0x0703010B, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030D01,
        0x07030F05, 0x07030F0D, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050B01,
        0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050F, 0x07070701, 0x07070903,
        0x07070907, 0x0707090F, 0x07070B0B, 0x07070F07, 0x07090107, 0x07090303, 0x0709030D, 0x07090505,
        0x07090703, 0x07090B05, 0x07090D01, 0x07090D09, 0x070B0103, 0x070B0301, 0x070B0305, 0x070B050B,
        0x070B0705, 0x070B0909, 0x070B0B0D, 0x070B0F07, 0x070D030D, 0x070D0903, 0x070F0103, 0x070F0107,
        0x070F0501, 0x070F0505, 0x070F070B, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
        0x0901050F, 0x09010705, 0x09010903, 0x09010B01, 0x09010F01, 0x09030105, 0x0903010F, 0x09030303,
        0x09030307, 0x09030505, 0x09030701, 0x0903070B, 0x09030907, 0x09030B03, 0x09030B0B, 0x09050103,
        0x09050107, 0x09050301, 0x0905030B, 0x09050503, 0x09050707, 0x09050901, 0x09050B0F, 0x09050D05,
        0x09050F01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070B,
        0x09090101, 0x09090105, 0x09090509, 0x0909070F, 0x09090901, 0x09090F03, 0x090B010B, 0x090B010F,
        0x090B0503, 0x090B0D05, 0x090D0307, 0x090D0709, 0x090D0D01, 0x090F0301, 0x090F030B, 0x090F0701,
        0x090F0907, 0x090F0B03, 0x0B010105, 0x0B010301, 0x0B010309, 0x0B010505, 0x0B010901, 0x0B010909,
        0x0B01090F, 0x0B010B05, 0x0B010D0D, 0x0B010F09, 0x0B030103, 0x0B030107, 0x0B03010B, 0x0B030305,
        0x0B030503, 0x0B030705, 0x0B030F05, 0x0B050101, 0x0B050303, 0x0B050507, 0x0B050701, 0x0B05070D,
        0x0B050B07, 0x0B070105, 0x0B07010F, 0x0B070301, 0x0B07050F, 0x0B070909, 0x0B070B03, 0x0B070D0B,
        0x0B070F07, 0x0B090103, 0x0B090109, 0x0B090501, 0x0B090705, 0x0B09090D, 0x0B0B0305, 0x0B0B050D,
        0x0B0B0B03, 0x0B0B0B07, 0x0B0D0905, 0x0B0F0105, 0x0B0F0109, 0x0B0F0505, 0x0D010303, 0x0D010307,
        0x0D01030B, 0x0D010703, 0x0D010707, 0x0D010D01, 0x0D030101, 0x0D030501, 0x0D03050F, 0x0D030D09,
        0x0D050305, 0x0D050709, 0x0D050905, 0x0D050B0B, 0x0D050D05, 0x0D050F01, 0x0D070101, 0x0D070309,
        0x0D070503, 0x0D070901, 0x0D09050B, 0x0D090907, 0x0D090D05, 0x0D0B0101, 0x0D0B0107, 0x0D0B0709,
        0x0D0B0D01, 0x0D0D010B, 0x0D0D0901, 0x0D0F0303, 0x0D0F0307, 0x0F010101, 0x0F010109, 0x0F01010F,
        0x0F010501, 0x0F010505, 0x0F01070D, 0x0F010901, 0x0F010B09, 0x0F010D05, 0x0F030105, 0x0F030303,
        0x0F030509, 0x0F030907, 0x0F03090B, 0x0F050103, 0x0F050109, 0x0F050301, 0x0F05030D, 0x0F050503,
        0x0F050701, 0x0F050B03, 0x0F070105, 0x0F070705, 0x0F07070B, 0x0F070B07, 0x0F090103, 0x0F09010B,
        0x0F090307, 0x0F090501, 0x0F090B01, 0x0F0B0505, 0x0F0B0905, 0x0F0D0105, 0x0F0D0703, 0x0F0F0101,
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

    // Test-only IQ2_S writer mirroring the reference layout: eight 32-value sub-blocks per 256-element block, each
    // grid index packing the eight low bits of its nine-bit entry plus two qh bits (bit 2l at index bit 8, bit 2l+1
    // at index bit 9), and each sub-block scale byte packing the two four-bit scale codes of its nibble pair (read
    // back as db = d * (0.5 + nibble) * 0.25). The group's eight sign bits live raw in one sign byte inside the qs
    // region, so no parity rule is needed. Block d = amax / 666 stored as a half-float at byte 0. Padded elements
    // keep zero quants.
    private static void QuantizeIQ2S(ReadOnlySpan<float> src, Span<byte> dst)
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
            WriteLE(dst, b * 82, (ushort)BitConverter.HalfToInt16Bits(dHalf));
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

                dst[b * 82 + 74 + ib] = (byte)(codes[0] | (codes[1] << 4));

                byte qh = 0;
                for (int l = 0; l < 4; l++)
                {
                    float[] group = sub.Slice(l * 8, Math.Min(8, sub.Length - l * 8)).ToArray();
                    if (group.Length < 8)
                    {
                        break; // padded group: quants stay zero
                    }

                    int bestEntry = 0;
                    double bestCost = double.MaxValue;
                    for (int e = 0; e < 1024; e++)
                    {
                        double cost = 0;
                        for (int p = 0; p < 8; p++)
                        {
                            double magnitude = (Iq2SGridU64[e] >> (8 * p)) & 0xFF;
                            double diff = group[p] - dbs[l / 2] * magnitude;
                            cost += diff * diff;
                        }

                        if (cost < bestCost)
                        {
                            bestCost = cost;
                            bestEntry = e;
                        }
                    }

                    dst[b * 82 + 2 + 4 * ib + l] = (byte)(bestEntry & 0xFF);
                    qh |= (byte)(((bestEntry >> 8) & 1) << (2 * l));
                    qh |= (byte)(((bestEntry >> 9) & 1) << (2 * l + 1));

                    int signs = 0;
                    for (int p = 0; p < 8; p++)
                    {
                        if (group[p] < 0f)
                        {
                            signs |= 1 << p;
                        }
                    }

                    dst[b * 82 + 34 + 4 * ib + l] = (byte)signs; // raw sign bits: the reference reads signs[l] directly
                }

                dst[b * 82 + 66 + ib] = qh;
            }
        }
    }

    // Test-only IQ3_S writer mirroring the reference layout: eight 32-value sub-blocks per 256-element block, each
    // grid index packing the eight low bits of its nine-bit entry plus one qh bit (bit 2l for the low four values
    // of a group, bit 2l+1 for the high four), and each scale byte packing the codes of two sub-blocks in its two
    // nibbles (read back as db = d * (1 + 2 * code)). The group's eight sign bits live raw in one sign byte, so no
    // parity rule is needed. Block d = amax / 465 stored as a half-float at byte 0. Padded elements keep zero quants.
    private static void QuantizeIQ3S(ReadOnlySpan<float> src, Span<byte> dst)
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

            Half dHalf = (Half)(amaxBlock / 465f);
            WriteLE(dst, b * 110, (ushort)BitConverter.HalfToInt16Bits(dHalf));
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

                int code = dBack > 0f && amax > 0f ? Math.Clamp((int)Math.Round(((double)amax / (15.0 * dBack) - 1.0) / 2.0, MidpointRounding.AwayFromZero), 0, 15) : 0;
                double db = dBack * (1 + 2 * code);
                dst[b * 110 + 106 + (ib >> 1)] |= (byte)(code << (4 * (ib & 1))); // two sub-blocks share one scale byte

                byte qh = 0;
                for (int l = 0; l < 4; l++)
                {
                    float[] group = sub.Slice(l * 8, Math.Min(8, sub.Length - l * 8)).ToArray();
                    if (group.Length < 8)
                    {
                        break; // padded group: quants stay zero
                    }

                    int signs = 0;
                    for (int p = 0; p < 8; p++)
                    {
                        if (group[p] < 0f)
                        {
                            signs |= 1 << p;
                        }
                    }

                    dst[b * 110 + 74 + 4 * ib + l] = (byte)signs; // raw sign bits: the reference reads signs[l] directly

                    for (int h = 0; h < 2; h++)
                    {
                        int bestEntry = 0;
                        double bestCost = double.MaxValue;
                        for (int e = 0; e < 512; e++)
                        {
                            double cost = 0;
                            for (int p = 0; p < 4; p++)
                            {
                                double magnitude = (Iq3SGridU32[e] >> (8 * p)) & 0xFF;
                                double diff = group[h * 4 + p] - db * magnitude;
                                cost += diff * diff;
                            }

                            if (cost < bestCost)
                            {
                                bestCost = cost;
                                bestEntry = e;
                            }
                        }

                        dst[b * 110 + 2 + 8 * ib + 2 * l + h] = (byte)(bestEntry & 0xFF);
                        qh |= (byte)(((bestEntry >> 8) & 1) << (2 * l + h));
                    }
                }

                dst[b * 110 + 66 + ib] = qh;
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
