using System.Runtime.InteropServices;
using Lair.Backends.Cpu;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// Single-vector shared-input fusion (the CPU backend's IDenseWeightSharedInputFusion seam used by the decode forward): every fused
/// output element must match what the independent per-matrix Gemv calls write, bit-for-bit, single-threaded and threaded. Each fused
/// row runs the identical RowDot/RowDotQ8K evaluator with identical operands, so accumulator order and row independence are preserved
/// across the fused dispatch exactly as across the independent calls.
/// </summary>
[Trait("Category", "Kernel")]
public class SharedInputGemvTests
{
    // The engine's own dense matrix shapes (block-aligned columns) plus a ragged-column case, where the fused rows still run the
    // same scalar-tail evaluators the independent Gemv calls use.
    private static readonly (int Rows, int Cols)[] MatrixSizes = [(256, 384), (1024, 512), (100, 137)];

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q6_K, QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K)] // the q4_k_m decode posture: Q4_K attention matrices over a Q6_K value matrix.
    [InlineData(QuantFormat.Q8_0, QuantFormat.Q6_K)]
    [InlineData(QuantFormat.IQ3_S, QuantFormat.Q6_K)] // the shared qa/da quantization must serve the i-quant rows exactly as each independent call's own quantization did.
    [InlineData(QuantFormat.F32, QuantFormat.F16)]
    public void FusedPair_SingleVector_IsBitIdenticalToIndependentGemvCalls(QuantFormat formatA, QuantFormat formatB)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            using CpuBackend backend = new(threads: 1);

            (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 1234 + rows, vectors: 1);
            (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 9021 + rows, vectors: 1);

            TensorShape shapeA = new([rows, cols]);
            TensorShape shapeB = new([rows, cols]);
            WeightRef weightA = new(weightsA, shapeA, formatA);
            WeightRef weightB = new(weightsB, shapeB, formatB);

            float[] x = inputs.AsSpan(0, cols).ToArray();

            float[] expectedA = new float[rows];
            float[] expectedB = new float[rows];
            backend.Gemv(x, weightA, expectedA);
            backend.Gemv(x, weightB, expectedB);

            float[] actualA = new float[rows];
            float[] actualB = new float[rows];

            IDenseWeightSharedInputFusion fusion = backend;
            Assert.True(fusion.TryGemvSharedInput(x, weightA, actualA, weightB, actualB), $"{formatA}+{formatB} at {rows}x{cols} must fuse");

            BatchKernelTests.AssertBitsEqual(expectedA, actualA, $"fused pair A {formatA}/{formatB} at {rows}x{cols}");
            BatchKernelTests.AssertBitsEqual(expectedB, actualB, $"fused pair B {formatA}/{formatB} at {rows}x{cols}");
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K, QuantFormat.Q6_K)] // the decode attention posture: Q, K, and V over one normed vector.
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K, QuantFormat.Q8_0)]
    [InlineData(QuantFormat.IQ3_S, QuantFormat.IQ3_S, QuantFormat.Q4_K)]
    public void FusedTriple_SingleVector_IsBitIdenticalToIndependentGemvCalls(QuantFormat formatA, QuantFormat formatB, QuantFormat formatC)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            using CpuBackend backend = new(threads: 1);

            (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 5678 + rows, vectors: 1);
            (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 6789 + rows, vectors: 1);
            (_, byte[] weightsC) = BatchKernelTests.CreateCase(rows, cols, formatC, seed: 7890 + rows, vectors: 1);

            TensorShape shape = new([rows, cols]);
            WeightRef weightA = new(weightsA, shape, formatA);
            WeightRef weightB = new(weightsB, shape, formatB);
            WeightRef weightC = new(weightsC, shape, formatC);

            float[] x = inputs.AsSpan(0, cols).ToArray();

            float[] expectedA = new float[rows];
            float[] expectedB = new float[rows];
            float[] expectedC = new float[rows];
            backend.Gemv(x, weightA, expectedA);
            backend.Gemv(x, weightB, expectedB);
            backend.Gemv(x, weightC, expectedC);

            float[] actualA = new float[rows];
            float[] actualB = new float[rows];
            float[] actualC = new float[rows];

            IDenseWeightSharedInputFusion fusion = backend;
            Assert.True(fusion.TryGemvSharedInput(x, weightA, actualA, weightB, actualB, weightC, actualC), $"{formatA}+{formatB}+{formatC} at {rows}x{cols} must fuse");

            BatchKernelTests.AssertBitsEqual(expectedA, actualA, $"fused triple A {formatA}/{formatB}/{formatC} at {rows}x{cols}");
            BatchKernelTests.AssertBitsEqual(expectedB, actualB, $"fused triple B {formatA}/{formatB}/{formatC} at {rows}x{cols}");
            BatchKernelTests.AssertBitsEqual(expectedC, actualC, $"fused triple C {formatA}/{formatB}/{formatC} at {rows}x{cols}");
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q6_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.IQ3_S, QuantFormat.Q4_K)]
    public void FusedSharedInput_Threaded_IsBitIdenticalToSingleThreadedAndIndependentCalls(QuantFormat formatA, QuantFormat formatB)
    {
        // Row counts past the backend's parallel threshold so both routes really thread; row independence must keep every element
        // identical no matter which worker claims which block of the combined row range.
        using CpuBackend single = new(threads: 1);
        using CpuBackend threaded = new(threads: 4);

        int rows = 512;
        int cols = 768;

        (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 30_000 + rows, vectors: 1);
        (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 41_000 + rows, vectors: 1);

        TensorShape shape = new([rows, cols]);
        WeightRef weightA = new(weightsA, shape, formatA);
        WeightRef weightB = new(weightsB, shape, formatB);

        float[] x = inputs.AsSpan(0, cols).ToArray();

        float[] expectedSingleA = new float[rows];
        float[] expectedSingleB = new float[rows];
        single.Gemv(x, weightA, expectedSingleA);
        single.Gemv(x, weightB, expectedSingleB);

        float[] threadedIndependentA = new float[rows];
        float[] threadedIndependentB = new float[rows];
        threaded.Gemv(x, weightA, threadedIndependentA);
        threaded.Gemv(x, weightB, threadedIndependentB);

        float[] fusedSingleA = new float[rows];
        float[] fusedSingleB = new float[rows];
        Assert.True(single.TryGemvSharedInput(x, weightA, fusedSingleA, weightB, fusedSingleB), "single-thread fusion refused");

        float[] fusedThreadedA = new float[rows];
        float[] fusedThreadedB = new float[rows];
        Assert.True(threaded.TryGemvSharedInput(x, weightA, fusedThreadedA, weightB, fusedThreadedB), "threaded fusion refused");

        BatchKernelTests.AssertBitsEqual(expectedSingleA, fusedSingleA, $"single-thread fused A {formatA}/{formatB}");
        BatchKernelTests.AssertBitsEqual(expectedSingleB, fusedSingleB, $"single-thread fused B {formatA}/{formatB}");
        BatchKernelTests.AssertBitsEqual(expectedSingleA, threadedIndependentA, $"threaded independent A {formatA}/{formatB}");
        BatchKernelTests.AssertBitsEqual(expectedSingleA, fusedThreadedA, $"threaded fused A {formatA}/{formatB}");
        BatchKernelTests.AssertBitsEqual(expectedSingleB, fusedThreadedB, $"threaded fused B {formatA}/{formatB}");
    }

    [Fact]
    public void SharedInput_MismatchedColumnCounts_RefusesWithoutTouchingOutputs()
    {
        using CpuBackend backend = new(threads: 1);

        (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(256, 384, QuantFormat.Q4_K, seed: 7, vectors: 1);
        (_, byte[] weightsB) = BatchKernelTests.CreateCase(256, 512, QuantFormat.Q6_K, seed: 8, vectors: 1);

        float[] x = inputs.AsSpan(0, 384).ToArray();

        float[] outputsA = [.. Enumerable.Repeat(float.NaN, 256)];
        float[] outputsB = [.. Enumerable.Repeat(float.NaN, 256)];

        IDenseWeightSharedInputFusion fusion = backend;
        Assert.False(fusion.TryGemvSharedInput(x, new WeightRef(weightsA, new TensorShape([256, 384]), QuantFormat.Q4_K), outputsA, new WeightRef(weightsB, new TensorShape([256, 512]), QuantFormat.Q6_K), outputsB));

        Assert.All(outputsA, value => Assert.True(float.IsNaN(value)));
        Assert.All(outputsB, value => Assert.True(float.IsNaN(value)));
    }

    [Fact]
    public void SharedInput_RaggedColumnCounts_StayBitIdenticalAcrossFusedAndIndependentRoutes()
    {
        // Ragged widths run the scalar-tail evaluators on both routes; the fused combined dispatch must still land element-for-element.
        int rows = 131;
        int cols = 137;

        using CpuBackend backend = new(threads: 1);

        (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 42, vectors: 1);
        (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q6_K, seed: 43, vectors: 1);

        TensorShape shape = new([rows, cols]);
        WeightRef weightA = new(weightsA, shape, QuantFormat.Q4_K);
        WeightRef weightB = new(weightsB, shape, QuantFormat.Q6_K);

        float[] x = inputs.AsSpan(0, cols).ToArray();

        float[] expectedA = new float[rows];
        float[] expectedB = new float[rows];
        backend.Gemv(x, weightA, expectedA);
        backend.Gemv(x, weightB, expectedB);

        float[] fusedA = new float[rows];
        float[] fusedB = new float[rows];
        Assert.True(backend.TryGemvSharedInput(x, weightA, fusedA, weightB, fusedB));

        BatchKernelTests.AssertBitsEqual(expectedA, fusedA, $"ragged fused A at {rows}x{cols}");
        BatchKernelTests.AssertBitsEqual(expectedB, fusedB, $"ragged fused B at {rows}x{cols}");
    }
}
