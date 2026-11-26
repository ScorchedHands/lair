using Lair.Backends.Cpu;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// The hybrid dense-matrix route's contract: a Q6_K batched pass at or above the resolved matmul-route batch width runs
/// its rows through the blocked matmul tile executor instead of the per-row scratch batch dot, and every output element
/// lands bit-for-bit on what the batched-row route writes for it — the same ascending group and chunk walk, the same
/// FMA association, and the same scalar-tail cleanup, because the tile executor only chooses which tile rows share one
/// activation pass and never reorders a chain. The pin is a bench/test seam (the posture-resolved default flows from the
/// same tuning snapshot every kernel reads); formats outside the blocked matmul keep their batched-row kernels unchanged
/// whatever the pin says, and a width below the batch count leaves the batched pass on its own route.
/// </summary>
[Trait("Category", "Kernel")]
public sealed class Q6KGemmRouteTests
{
    // Shapes that straddle the tile boundary and the ragged column tails the hybrid route must keep byte-identical at:
    // a partial final quant block, a column count beyond one 256-element column block, and the Qwen-shaped down-matrix
    // geometry the route is measured against (2048 rows x 3584 columns).
    private static readonly (int Rows, int Cols)[] MatrixSizes = [(100, 137), (130, 300), (9, 264), (512, 1024), (2048, 3584)];

    // Counts at and above the crossover width, plus ragged counts that exercise the tile kernel's chain tails.
    private static readonly int[] RouteCounts = [8, 9, 16, 32, 64];

    [Theory]
    [MemberData(nameof(MatrixSizesData))]
    public void GemvBatch_Q6KMatmulRoute_IsBitIdenticalToTheBatchedRowRoute(int rows, int cols)
    {
        foreach (int count in RouteCounts)
        {
            (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q6_K, seed: 51_000 + count, vectors: count);
            WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q6_K);

            using CpuBackend rowRoute = new(threads: 1, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidth = 0 });
            float[] expected = new float[count * rows];
            rowRoute.GemvBatch(inputs, count, weight, expected);

            using CpuBackend hybrid = new(threads: 1, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidth = 8 });
            float[] actual = new float[count * rows];
            hybrid.GemvBatch(inputs, count, weight, actual);

            BatchKernelTests.AssertBitsEqual(expected, actual, $"Q6_K hybrid route at {rows}x{cols} count {count}");
        }
    }

    public static TheoryData<int, int> MatrixSizesData()
    {
        TheoryData<int, int> data = [];
        foreach ((int rows, int cols) in MatrixSizes)
        {
            data.Add(rows, cols);
        }

        return data;
    }

    [Theory]
    [InlineData(7)] // one below the crossover width: the batched-row scratch route stays in force
    [InlineData(1)]
    [InlineData(3)]
    public void GemvBatch_Q6KBelowRouteWidth_KeepsBatchedRowRoute_BitIdentical(int count)
    {
        const int rows = 300;
        const int cols = 300;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q6_K, seed: 52_000 + count, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q6_K);

        using CpuBackend reference = new(threads: 1);
        float[] expected = new float[count * rows];
        reference.GemvBatch(inputs, count, weight, expected);

        using CpuBackend pinned = new(threads: 1, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidth = 8 });
        float[] actual = new float[count * rows];
        pinned.GemvBatch(inputs, count, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, $"Q6_K batch({count}) below the route width at {rows}x{cols}");
    }

    [Fact]
    public void GemvBatch_Q6KMatmulRoute_Threaded_IsBitIdenticalToSingleThreaded()
    {
        const int rows = 512;
        const int cols = 1024;
        const int count = 32;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q6_K, seed: 53_000, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q6_K);

        using CpuBackend single = new(threads: 1, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidth = 8 });
        float[] expected = new float[count * rows];
        single.GemvBatch(inputs, count, weight, expected);

        // Forced tiny row blocks past the threshold hand every row to a worker: the same tile walks, partitioned.
        using CpuBackend threaded = new(threads: 8, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidth = 8, RowBlock = 3, ParallelRowThreshold = 1 });
        float[] actual = new float[count * rows];
        threaded.GemvBatch(inputs, count, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, $"threaded Q6_K hybrid route at {rows}x{cols}");
    }

    [Fact]
    public void GemvBatch_Q6KMatmulRoute_ScalarPosture_IsBitIdenticalToScalarBatchedRowRoute()
    {
        const int rows = 130;
        const int cols = 300;
        const int count = 8;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q6_K, seed: 54_000, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q6_K);

        using CpuBackend reference = new(threads: 1, allowAvx2: false, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidth = 0 });
        float[] expected = new float[count * rows];
        reference.GemvBatch(inputs, count, weight, expected);

        using CpuBackend hybrid = new(threads: 1, allowAvx2: false, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidth = 8 });
        float[] actual = new float[count * rows];
        hybrid.GemvBatch(inputs, count, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, $"scalar Q6_K hybrid route at {rows}x{cols}");
    }

    [Theory]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.IQ4_XS)]
    public void GemvBatch_FormatsOutsideTheHybridFormatKeepTheirBatchedRowRoute_UnderTheRoutePin(QuantFormat format)
    {
        const int rows = 300;
        const int cols = 300;
        const int count = 16;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, format, seed: 55_000, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), format);

        using CpuBackend reference = new(threads: 1);
        float[] expected = new float[count * rows];
        reference.GemvBatch(inputs, count, weight, expected);

        using CpuBackend pinned = new(threads: 1, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidth = 8 });
        float[] actual = new float[count * rows];
        pinned.GemvBatch(inputs, count, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, $"{format} batch({count}) unchanged by the Q6_K route pin at {rows}x{cols}");
    }

    // ---------------------------------------------------------------- posture-resolved width

    [Fact]
    public void Resolve_MatmulRouteBatchWidthFollowsTheVectorPosture()
    {
        Assert.Equal(8, CpuKernelTuning.Resolve(32, vectorPosture: true, CpuKernelTuningOverrides.Auto).MatmulRouteBatchWidth); // the full chain-group width: the measured crossover where tile reuse starts paying
        Assert.Equal(0, CpuKernelTuning.Resolve(32, vectorPosture: false, CpuKernelTuningOverrides.Auto).MatmulRouteBatchWidth); // without a vector posture the tile kernels never run
    }

    [Theory]
    [InlineData(0)] // 0 is a stated width: every batched pass keeps its batched-row kernel
    [InlineData(8)]
    [InlineData(64)]
    public void Resolve_MatmulRouteBatchWidthPinsResolveExactly(int width)
    {
        CpuKernelTuning tuning = CpuKernelTuning.Resolve(32, vectorPosture: true, new CpuKernelTuningOverrides { MatmulRouteBatchWidth = width });

        Assert.Equal(width, tuning.MatmulRouteBatchWidth);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(65)] // past the batch ceiling every dense kernel's scratch is sized for
    public void Resolve_MatmulRouteBatchWidthPinOutsideItsSpan_RefusesNamingQualifiedKeyAndValue(int value)
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(
            () => CpuKernelTuning.Resolve(8, vectorPosture: true, new CpuKernelTuningOverrides { MatmulRouteBatchWidth = value }));

        Assert.Equal("engine.cpu.tuning.matmul_route_batch_width", error.ParamName);
        Assert.Contains($"engine.cpu.tuning.matmul_route_batch_width {value}", error.Message);
    }

    [Fact]
    public void BackendOpen_AutoPostureResolvesTheMatmulRouteWidthAtTheBenchShape()
    {
        using CpuBackend backend = new(threads: 32, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false));

        Assert.Equal(8, backend.ResolvedTuning.MatmulRouteBatchWidth);

        using CpuBackend scalar = new(threads: 32, cpuFeatures: new CpuFeaturePosture(Avx2Available: false, AvxVnniAvailable: false));

        Assert.Equal(0, scalar.ResolvedTuning.MatmulRouteBatchWidth);
    }
}
