using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// Blocked-matmul coverage for the formats its dense weight path executes: every output element of a tile walk must
/// carry exactly the bytes an independent single-vector GEMV wrote for it, because the matmul accumulates that
/// element over the same successive quant chunks, in the same order, with the same operand association and the same
/// scalar cleanup as the vector kernel — not merely close to it. Shapes straddle tile boundaries (a column count
/// beyond one 256-element column block, a partial final quant block) so a walk that stopped at a tile edge or read a
/// padded lane would show up here rather than pass silently. Comparisons run on the raw bits of the floats: NaN
/// payloads and signed zeros compare too. Formats with no dense weight path are refused naming themselves instead of
/// being served by some other kernel, and the threaded partition stays byte-identical to a single-threaded walk.
/// </summary>
[Trait("Category", "Kernel")]
public sealed class GemmKernelTests
{
    // Shapes that straddle the blocked matmul's column boundary and its partial final quant block: one 256-element
    // column block with a padded tail, two column blocks whose second is partial, exact powers, and a wide matrix.
    private static readonly (int Rows, int Cols)[] MatrixSizes = [(100, 137), (130, 300), (256, 256), (9, 264), (512, 1024)];

    // Identity coverage runs to the backend's batch ceiling: what a decode batch advances and every prompt row width
    // a forward engine may resolve.
    private static readonly int[] BatchCounts = [1, 2, 8, 16, 32, 64];

    /// <summary>The formats the dense weight path executes.</summary>
    public static TheoryData<QuantFormat> SupportedFormats() => [QuantFormat.Q8_0, QuantFormat.Q4_K, QuantFormat.IQ4_XS];

    [Theory]
    [MemberData(nameof(SupportedFormats))]
    public void GemmBatch_IsRawByteIdenticalToIndependentSingleVectorGemv(QuantFormat format)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            foreach (int count in BatchCounts)
            {
                (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, format, seed: 4217 + count, vectors: count);

                using CpuBackend single = new(threads: 1);
                using CpuBackend matmul = new(threads: 1);

                float[] expected = new float[count * rows];
                for (int b = 0; b < count; b++)
                {
                    single.Gemv(inputs.AsSpan(b * cols, cols), new WeightRef(weights, MatrixShape(rows, cols), format), expected.AsSpan(b * rows, rows));
                }

                float[] actual = new float[count * rows];
                matmul.GemmBatch(inputs, count, new WeightRef(weights, MatrixShape(rows, cols), format), actual);

                AssertBitsEqual(expected, actual, $"{format} gemm({count}) at {rows}x{cols}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(SupportedFormats))]
    public void GemmBatch_IsRawByteIdenticalToBatchedGemv(QuantFormat format)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            foreach (int count in BatchCounts)
            {
                (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, format, seed: 90210 + count, vectors: count);

                using CpuBackend backend = new(threads: 4);

                float[] gemvBatch = new float[count * rows];
                backend.GemvBatch(inputs, count, new WeightRef(weights, MatrixShape(rows, cols), format), gemvBatch);

                float[] gemm = new float[count * rows];
                backend.GemmBatch(inputs, count, new WeightRef(weights, MatrixShape(rows, cols), format), gemm);

                AssertBitsEqual(gemvBatch, gemm, $"{format} gemm({count}) against the batched GEMV at {rows}x{cols}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(SupportedFormats))]
    public void GemmBatch_ScalarPath_IsRawByteIdenticalToScalarSingleVectorGemv(QuantFormat format)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            foreach (int count in new[] { 1, 8 })
            {
                (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, format, seed: 31 + count, vectors: count);

                using CpuBackend single = new(threads: 1, allowAvx2: false);
                using CpuBackend matmul = new(threads: 1, allowAvx2: false);

                float[] expected = new float[count * rows];
                for (int b = 0; b < count; b++)
                {
                    single.Gemv(inputs.AsSpan(b * cols, cols), new WeightRef(weights, MatrixShape(rows, cols), format), expected.AsSpan(b * rows, rows));
                }

                float[] actual = new float[count * rows];
                matmul.GemmBatch(inputs, count, new WeightRef(weights, MatrixShape(rows, cols), format), actual);

                AssertBitsEqual(expected, actual, $"scalar {format} gemm({count}) at {rows}x{cols}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(SupportedFormats))]
    public void GemmBatch_Threaded_IsRawByteIdenticalToSingleThreaded(QuantFormat format)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            const int count = 8;
            (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, format, seed: 778_8, vectors: count);

            using CpuBackend single = new(threads: 1);
            using CpuBackend parallel = new(threads: 4);

            float[] expected = new float[count * rows];
            single.GemmBatch(inputs, count, new WeightRef(weights, MatrixShape(rows, cols), format), expected);

            float[] actual = new float[count * rows];
            parallel.GemmBatch(inputs, count, new WeightRef(weights, MatrixShape(rows, cols), format), actual);

            AssertBitsEqual(expected, actual, $"threaded {format} gemm({count}) at {rows}x{cols}");
        }
    }

    [Theory]
    [InlineData(QuantFormat.F32)]
    [InlineData(QuantFormat.F16)]
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
    [InlineData(QuantFormat.Q2_0)]
    [InlineData(QuantFormat.BF16)]
    public void GemmBatch_FormatWithoutDenseWeightPath_FailsNamingFormat(QuantFormat format)
    {
        TensorShape shape = MatrixShape(4, 70);
        byte[] packed = new byte[4 * shape.RowStrideBytes(format)];

        using CpuBackend backend = new();

        Assert.False(backend.SupportsGemm(format), $"the dense weight path claims a kernel for {format}");

        ArgumentException error = Assert.Throws<ArgumentException>(
            () => backend.GemmBatch(new float[2 * 70], 2, new WeightRef(packed, shape, format), new float[2 * 4]));
        Assert.Contains(format.ToString(), error.Message); // the refusal names the format it refuses
        Assert.Contains("no blocked matmul exists in this build", error.Message);
    }

    [Fact]
    public void GemmBatch_BadCountOrSpanLengths_ThrowsClearMessage()
    {
        QuantFormat format = QuantFormat.Q4_K;
        TensorShape shape = MatrixShape(4, 70);
        long expectedBytes = 4 * shape.RowStrideBytes(format);
        byte[] packed = new byte[expectedBytes];
        float[] inputs = new float[2 * 70];

        using CpuBackend backend = new();

        ArgumentOutOfRangeException countLow = Assert.Throws<ArgumentOutOfRangeException>(
            () => backend.GemmBatch(inputs, 0, new WeightRef(packed, shape, format), new float[2 * 4]));
        Assert.Contains("count", countLow.ParamName);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => backend.GemmBatch(inputs, 65, new WeightRef(packed, shape, format), new float[65 * 4])); // one past the batch ceiling every kernel's scratch is sized for

        ArgumentException inputEx = Assert.Throws<ArgumentException>(
            () => backend.GemmBatch(new float[2 * 69], 2, new WeightRef(packed, shape, format), new float[2 * 4]));
        Assert.Contains("count", inputEx.Message);

        Assert.Throws<ArgumentException>(
            () => backend.GemmBatch(inputs, 2, new WeightRef(packed, shape, format), new float[2 * 4 - 1]));

        Assert.Throws<ArgumentException>(
            () => backend.GemmBatch(inputs, 2, new WeightRef(new byte[expectedBytes - 1], shape, format), new float[2 * 4]));
    }

    private static TensorShape MatrixShape(int rows, int cols) => new([rows, cols]);

    // Bit-exact comparison over the raw bits of the outputs: NaN payloads and signed zeros must match too.
    private static void AssertBitsEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        ReadOnlySpan<byte> expectedBytes = MemoryMarshal.AsBytes(expected);
        ReadOnlySpan<byte> actualBytes = MemoryMarshal.AsBytes(actual);

        for (int i = 0; i < expectedBytes.Length; i++)
        {
            Assert.True(
                expectedBytes[i] == actualBytes[i],
                $"{context}: first differing byte at {i} (float {i / 4}): expected 0x{expectedBytes[i]:X2}, got 0x{actualBytes[i]:X2}");
        }
    }
}
