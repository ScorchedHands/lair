using Lair.Backends.Cpu;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// Shared-input dense pair fusion (the CPU backend's IDenseWeightPairFusion seam): every fused output element must match the value
/// an independent GemvBatch call for that matrix writes, bit-for-bit, single-threaded and threaded — same accumulator order per
/// chain, same row partitioning under threading. Postures outside the fused format set refuse with false and leave outputs untouched,
/// so callers keep their independent calls there.
/// </summary>
[Trait("Category", "Kernel")]
public class DenseWeightFusionTests
{
    // The engine's own dense matrix shapes (block-aligned columns) plus a ragged-column case that must refuse fusion outright.
    private static readonly (int Rows, int Cols)[] MatrixSizes = [(256, 384), (1024, 512)];

    private static readonly int[] BatchCounts = [1, 2, 7, 8, 9, 16, 32, 64];

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q6_K, QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K)] // the q4_k_m feed-forward posture: Q4_K gate over a Q6_K up.
    public void FusedPair_IsBitIdenticalToIndependentGemvBatchCalls(QuantFormat formatA, QuantFormat formatB)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            foreach (int count in BatchCounts)
            {
                using CpuBackend backend = new(threads: 1);

                (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 1234 + count, vectors: count);
                (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 9021 + count, vectors: count);

                TensorShape shapeA = new([rows, cols]);
                TensorShape shapeB = new([rows, cols]);

                float[] expectedA = new float[count * rows];
                float[] expectedB = new float[count * rows];
                backend.GemvBatch(inputs, count, new WeightRef(weightsA, shapeA, formatA), expectedA);
                backend.GemvBatch(inputs, count, new WeightRef(weightsB, shapeB, formatB), expectedB);

                float[] actualA = new float[count * rows];
                float[] actualB = new float[count * rows];

                IDenseWeightPairFusion fusion = backend;
                bool ran = fusion.TryGemvBatchSharedInput(inputs, count, new WeightRef(weightsA, shapeA, formatA), actualA, new WeightRef(weightsB, shapeB, formatB), actualB);

                // Fusion engages from one full register group of chains up; below that the backend declines and the caller keeps its
                // independent calls. Either route must land bit-identical to those independent calls.
                Assert.Equal(count >= 8, ran);
                if (!ran)
                {
                    continue; // refused postures leave the outputs untouched — the caller's independent GemvBatch calls are what run there.
                }

                BatchKernelTests.AssertBitsEqual(expectedA, actualA, $"fused A {formatA}/{formatB} batch({count}) at {rows}x{cols}");
                BatchKernelTests.AssertBitsEqual(expectedB, actualB, $"fused B {formatA}/{formatB} batch({count}) at {rows}x{cols}");
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q6_K, QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K)]
    public void FusedPair_Threaded_IsBitIdenticalToSingleThreadedAndIndependentCalls(QuantFormat formatA, QuantFormat formatB)
    {
        // Row counts past the backend's parallel threshold so both routes really thread; row independence must keep every element
        // identical no matter which worker claims which block.
        foreach (int count in new[] { 8, 32 })
        {
            using CpuBackend single = new(threads: 1);
            using CpuBackend threaded = new(threads: 4);

            int rows = 512;
            int cols = 768;

            (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 30_000 + count, vectors: count);
            (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 41_000 + count, vectors: count);

            TensorShape shape = new([rows, cols]);
            WeightRef weightA = new(weightsA, shape, formatA);
            WeightRef weightB = new(weightsB, shape, formatB);

            float[] expectedSingleA = new float[count * rows];
            float[] expectedSingleB = new float[count * rows];
            single.GemvBatch(inputs, count, weightA, expectedSingleA);
            single.GemvBatch(inputs, count, weightB, expectedSingleB);

            float[] threadedIndependentA = new float[count * rows];
            float[] threadedIndependentB = new float[count * rows];
            threaded.GemvBatch(inputs, count, weightA, threadedIndependentA);
            threaded.GemvBatch(inputs, count, weightB, threadedIndependentB);

            float[] fusedSingleA = new float[count * rows];
            float[] fusedSingleB = new float[count * rows];
            Assert.True(single.TryGemvBatchSharedInput(inputs, count, weightA, fusedSingleA, weightB, fusedSingleB));

            float[] fusedThreadedA = new float[count * rows];
            float[] fusedThreadedB = new float[count * rows];
            Assert.True(threaded.TryGemvBatchSharedInput(inputs, count, weightA, fusedThreadedA, weightB, fusedThreadedB), "threaded fusion refused");

            BatchKernelTests.AssertBitsEqual(expectedSingleA, fusedSingleA, $"single-thread fused A {formatA}/{formatB} batch({count})");
            BatchKernelTests.AssertBitsEqual(expectedSingleB, fusedSingleB, $"single-thread fused B {formatA}/{formatB} batch({count})");
            BatchKernelTests.AssertBitsEqual(expectedSingleA, threadedIndependentA, $"threaded independent A {formatA}/{formatB} batch({count})");
            BatchKernelTests.AssertBitsEqual(expectedSingleA, fusedThreadedA, $"threaded fused A {formatA}/{formatB} batch({count})");
            BatchKernelTests.AssertBitsEqual(expectedSingleB, fusedThreadedB, $"threaded fused B {formatA}/{formatB} batch({count})");
        }
    }

    [Theory]
    // Formats outside the scratch route (integer-dot i-quants included), and a block-unaligned column count on an eligible format.
    [InlineData(QuantFormat.Q8_0, QuantFormat.Q8_0)]
    [InlineData(QuantFormat.IQ4_XS, QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K)]
    public void FusedPair_OutsideFusedPostures_RefusesWithoutTouchingOutputs(QuantFormat formatA, QuantFormat formatB)
    {
        int rows = 100;
        int cols = 137; // ragged for every block layout: the fused route must decline rather than approximate.

        foreach (int count in new[] { 8 })
        {
            using CpuBackend backend = new(threads: 1);

            (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 555 + count, vectors: count);
            (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 666 + count, vectors: count);

            TensorShape shape = new([rows, cols]);

            float[] outputsA = [.. Enumerable.Repeat(float.NaN, count * rows)];
            float[] outputsB = [.. Enumerable.Repeat(float.NaN, count * rows)];

            bool ran = backend.TryGemvBatchSharedInput(inputs, count, new WeightRef(weightsA, shape, formatA), outputsA, new WeightRef(weightsB, shape, formatB), outputsB);

            Assert.False(ran, $"{formatA}+{formatB} at {rows}x{cols} must refuse shared-input fusion");
            Assert.All(outputsA, value => Assert.True(float.IsNaN(value)));
            Assert.All(outputsB, value => Assert.True(float.IsNaN(value)));
        }
    }

    [Fact]
    public void FusedPair_MismatchedColumnCounts_ThrowsClearMessage()
    {
        using CpuBackend backend = new(threads: 1);

        (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(256, 384, QuantFormat.Q4_K, seed: 7, vectors: 8);
        (_, byte[] weightsB) = BatchKernelTests.CreateCase(256, 512, QuantFormat.Q6_K, seed: 8, vectors: 8);

        float[] outputsA = new float[8 * 256];
        float[] outputsB = new float[8 * 256];

        ArgumentException error = Assert.Throws<ArgumentException>(() => backend.TryGemvBatchSharedInput(
            inputs, 8, new WeightRef(weightsA, new TensorShape([256, 384]), QuantFormat.Q4_K), outputsA,
            new WeightRef(weightsB, new TensorShape([256, 512]), QuantFormat.Q6_K), outputsB));

        Assert.Contains("equal column counts", error.Message);
    }
}
