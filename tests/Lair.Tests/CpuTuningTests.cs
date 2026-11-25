using Lair.Backends.Cpu;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// Detection-driven CPU kernel tuning end to end below model open: the posture-resolved defaults come from the opened backend's
/// effective worker count and vector posture — never from one machine class — and explicit pins (the bench/test seam that host
/// configuration also flows through) resolve field by field beside them. A pin outside its accepted span refuses naming the
/// qualified key rather than clamping quietly. Because every resolved value only partitions independent work — which rows a worker
/// claims, when threading pays off, which batch vectors share one scratch pass, and where staging stops being cache-safe — every
/// posture here must produce byte-identical outputs, asserted against the same backend's own independent calls.
/// </summary>
[Trait("Category", "Kernel")]
public sealed class CpuTuningTests
{
    // ---------------------------------------------------------------- posture-resolved resolution

    [Theory]
    [InlineData(1, 64)]
    [InlineData(4, 64)]
    [InlineData(8, 64)]
    [InlineData(16, 32)]
    [InlineData(32, 16)] // the measured optimum at this box's 32-worker posture lands exactly here
    [InlineData(64, 8)]
    [InlineData(128, 8)] // finer than a 1/64 page per worker stops paying off against per-row dispatch overhead
    public void Resolve_AutoRowBlockScalesWithEffectiveWorkerCount(int workers, int expectedRowBlock)
    {
        CpuKernelTuning tuning = CpuKernelTuning.Resolve(workers, vectorPosture: true, CpuKernelTuningOverrides.Auto);

        Assert.Equal(expectedRowBlock, tuning.RowBlock);
    }

    [Theory]
    [InlineData(1, 128)]
    [InlineData(4, 128)]
    [InlineData(8, 256)]
    [InlineData(32, 256)] // the measured optimum at 32 workers with a 16-row block lands exactly here
    public void Resolve_AutoDispatchThresholdTracksRowBlockAndWorkerCountTogether(int workers, int expectedThreshold)
    {
        CpuKernelTuning tuning = CpuKernelTuning.Resolve(workers, vectorPosture: true, CpuKernelTuningOverrides.Auto);

        Assert.Equal(expectedThreshold, tuning.ParallelRowThreshold);
        Assert.True(tuning.ParallelRowThreshold >= 2 * tuning.RowBlock); // threading never starts below two blocks of work
    }

    [Fact]
    public void Resolve_ChainGroupFollowsTheVectorPostureAlone()
    {
        // Eight chains plus one weight chunk fit the vector posture's register budget; without that posture no scratch route runs at all.
        Assert.Equal(CpuKernelTuning.ChainGroupMax, CpuKernelTuning.Resolve(32, vectorPosture: true, CpuKernelTuningOverrides.Auto).ChainGroup);
        Assert.Equal(CpuKernelTuning.ChainGroupMin, CpuKernelTuning.Resolve(32, vectorPosture: false, CpuKernelTuningOverrides.Auto).ChainGroup);
    }

    [Fact]
    public void Resolve_AutoScratchCeilingStaysCacheSafeOnVectorPostures()
    {
        Assert.Equal(16_384, CpuKernelTuning.Resolve(1, vectorPosture: true, CpuKernelTuningOverrides.Auto).MaxScratchCols);
        Assert.Equal(16_384, CpuKernelTuning.Resolve(128, vectorPosture: false, CpuKernelTuningOverrides.Auto).MaxScratchCols);
    }

    [Fact]
    public void Resolve_EveryFieldPinnedResolvesToExactlyTheStatedValues()
    {
        CpuKernelTuning tuning = CpuKernelTuning.Resolve(32, vectorPosture: true, new CpuKernelTuningOverrides
        {
            RowBlock = 4,
            ParallelRowThreshold = 10,
            ChainGroup = 2,
            MaxScratchCols = 512,
        });

        Assert.Equal(4, tuning.RowBlock);
        Assert.Equal(10, tuning.ParallelRowThreshold);
        Assert.Equal(2, tuning.ChainGroup);
        Assert.Equal(512, tuning.MaxScratchCols);
    }

    [Fact]
    public void Resolve_AutoDispatchThresholdFollowsAPinnedRowBlock()
    {
        // A pinned row block feeds the auto threshold formula too: the threshold tracks the posture the open actually runs, not the one the formula would have picked on its own.
        CpuKernelTuning tuning = CpuKernelTuning.Resolve(8, vectorPosture: true, new CpuKernelTuningOverrides { RowBlock = 16 });

        Assert.Equal(16, tuning.RowBlock);
        Assert.Equal(16 * 8 / 2, tuning.ParallelRowThreshold);
    }

    [Theory]
    [InlineData(0, "engine.cpu.tuning.row_block")]
    [InlineData(65, "engine.cpu.tuning.row_block")]
    [InlineData(9, "engine.cpu.tuning.chain_group")]
    [InlineData(256, "engine.cpu.tuning.max_scratch_cols")]
    [InlineData(70_000, "engine.cpu.tuning.max_scratch_cols")]
    [InlineData(0, "engine.cpu.tuning.parallel_row_threshold")]
    public void Resolve_PinOutsideItsSpan_RefusesNamingQualifiedKeyAndValue(int value, string qualifiedKey)
    {
        CpuKernelTuningOverrides overrides = qualifiedKey.EndsWith("row_block", StringComparison.Ordinal)
            ? new CpuKernelTuningOverrides { RowBlock = value }
            : qualifiedKey.EndsWith("chain_group", StringComparison.Ordinal)
                ? new CpuKernelTuningOverrides { ChainGroup = value }
                : qualifiedKey.EndsWith("max_scratch_cols", StringComparison.Ordinal)
                    ? new CpuKernelTuningOverrides { MaxScratchCols = value }
                    : new CpuKernelTuningOverrides { ParallelRowThreshold = value };

        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() => CpuKernelTuning.Resolve(8, vectorPosture: true, overrides));

        Assert.Equal(qualifiedKey, error.ParamName); // the refusal names the host key a bench/test seam mirrors, not an internal field
        Assert.Contains($"{qualifiedKey} {value}", error.Message);
    }

    [Fact]
    public void Resolve_ChainGroupAboveRegisterBudget_StatesWhyItRefuses()
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(
            () => CpuKernelTuning.Resolve(8, vectorPosture: true, new CpuKernelTuningOverrides { ChainGroup = 9 }));

        Assert.Contains("register budget", error.Message); // wider groups would spill the very accumulators grouping exists to keep live
    }

    // ---------------------------------------------------------------- what an opened backend resolves

    [Fact]
    public void BackendOpen_AutoPostureResolvesTheMeasuredOptimaAtTheBenchShape()
    {
        // Stating a vector posture keeps this assertion host-independent: the values are the committed dense-traversal measurement at a 32-worker posture, now derived rather than hardcoded.
        using CpuBackend backend = new(threads: 32, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false));

        Assert.Equal(16, backend.ResolvedTuning.RowBlock);
        Assert.Equal(256, backend.ResolvedTuning.ParallelRowThreshold);
        Assert.Equal(8, backend.ResolvedTuning.ChainGroup);
        Assert.Equal(16_384, backend.ResolvedTuning.MaxScratchCols);
    }

    [Fact]
    public void BackendOpen_ScalarPostureResolvesChainGroupToOne()
    {
        using CpuBackend backend = new(threads: 32, cpuFeatures: new CpuFeaturePosture(Avx2Available: false, AvxVnniAvailable: false));

        Assert.Equal(CpuKernelTuning.ChainGroupMin, backend.ResolvedTuning.ChainGroup);
        Assert.Equal(16, backend.ResolvedTuning.RowBlock); // worker-scaled knobs resolve the same way; only the register-budget knob follows the posture
    }

    [Fact]
    public void BackendOpen_QuotaChangeReResolvesWorkerScaledKnobs()
    {
        using CpuBackend backend = new(threads: 32, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false));

        Assert.Equal(16, backend.ResolvedTuning.RowBlock);
        Assert.Equal(256, backend.ResolvedTuning.ParallelRowThreshold);

        Assert.Equal(4, backend.ApplyThreadQuota(4)); // participation capped to four workers
        Assert.Equal(64, backend.ResolvedTuning.RowBlock); // coarser blocks with fewer participants, resolved from the new posture
        Assert.Equal(128, backend.ResolvedTuning.ParallelRowThreshold);

        backend.ApplyThreadQuota(null); // restoring the uncapped choice resolves the measured posture back
        Assert.Equal(16, backend.ResolvedTuning.RowBlock);
        Assert.Equal(256, backend.ResolvedTuning.ParallelRowThreshold);
    }

    [Fact]
    public void BackendOpen_QuotaChangeKeepsPinsAndReResolvesAutoSiblings()
    {
        using CpuBackend backend = new(threads: 8, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false), tuning: new CpuKernelTuningOverrides { RowBlock = 3 });

        backend.ApplyThreadQuota(2);

        Assert.Equal(3, backend.ResolvedTuning.RowBlock); // a stated value pins across quota moves exactly like an ISA does
        Assert.Equal(CpuKernelTuning.Resolve(2, vectorPosture: true, new CpuKernelTuningOverrides { RowBlock = 3 }).ParallelRowThreshold, backend.ResolvedTuning.ParallelRowThreshold);
    }

    [Fact]
    public void BackendOpen_PinOutsideItsSpan_RefusesTheOpen()
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(
            () => new CpuBackend(threads: 1, tuning: new CpuKernelTuningOverrides { ChainGroup = 9 }));

        Assert.Equal("engine.cpu.tuning.chain_group", error.ParamName);
    }

    // ---------------------------------------------------------------- every posture is byte-identical

    private const int ParityRows = 512;
    private const int ParityCols = 768; // block-aligned and under the auto scratch ceiling, so the scratch route runs at the default posture

    [Theory]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q6_K)]
    public void GemvBatch_EveryTuningPosture_IsBitIdenticalToTheDefaultPosture(QuantFormat format)
    {
        foreach (int count in new[] { 7, 8, 9 }) // ragged counts exercise group tails in both the unrolled and the grouped scratch kernels
        {
            (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(ParityRows, ParityCols, format, seed: 41_000 + count, vectors: count);
            WeightRef weight = new(weights, new TensorShape([ParityRows, ParityCols]), format);

            using CpuBackend reference = new(threads: 1);
            float[] expected = new float[ParityRows * count];
            reference.GemvBatch(inputs, count, weight, expected);

            foreach (CpuKernelTuningOverrides posture in TuningPostures())
            {
                using CpuBackend variant = new(threads: 8, tuning: posture);
                float[] actual = new float[ParityRows * count];
                variant.GemvBatch(inputs, count, weight, actual);

                BatchKernelTests.AssertBitsEqual(expected, actual, $"{format} batch({count}) at {ParityRows}x{ParityCols} under {posture}");
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q6_K)]
    public void Gemv_ForceThreadedSmallBlocks_IsBitIdenticalToSingleThreaded(QuantFormat format)
    {
        // Row blocks are pure partitioning: forcing tiny blocks past the threshold hands every row to a worker and must still land element-for-element.
        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(ParityRows, ParityCols, format, seed: 42_000, vectors: 1);
        WeightRef weight = new(weights, new TensorShape([ParityRows, ParityCols]), format);

        using CpuBackend reference = new(threads: 1);
        float[] expected = new float[ParityRows];
        reference.Gemv(inputs, weight, expected);

        using CpuBackend threaded = new(threads: 8, tuning: new CpuKernelTuningOverrides { RowBlock = 3, ParallelRowThreshold = 1 });
        float[] actual = new float[ParityRows];
        threaded.Gemv(inputs, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, $"{format} forced-block GEMV at {ParityRows}x{ParityCols}");
    }

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q6_K, QuantFormat.Q4_K)]
    public void FusedBatchSharedInput_PinnedChainGroup_IsBitIdenticalToIndependentBatchCalls(QuantFormat formatA, QuantFormat formatB)
    {
        // The fused pair runs its scratch groups through the grouped kernel at a pinned width; the independent calls run the unrolled default posture. Row and chain semantics must still match.
        int count = 8;
        (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(ParityRows, ParityCols, formatA, seed: 43_000, vectors: count);
        (_, byte[] weightsB) = BatchKernelTests.CreateCase(ParityRows, ParityCols, formatB, seed: 44_000, vectors: count);

        WeightRef weightA = new(weightsA, new TensorShape([ParityRows, ParityCols]), formatA);
        WeightRef weightB = new(weightsB, new TensorShape([ParityRows, ParityCols]), formatB);

        using CpuBackend independent = new(threads: 8);
        float[] expectedA = new float[ParityRows * count];
        float[] expectedB = new float[ParityRows * count];
        independent.GemvBatch(inputs, count, weightA, expectedA);
        independent.GemvBatch(inputs, count, weightB, expectedB);

        using CpuBackend fused = new(threads: 8, tuning: new CpuKernelTuningOverrides { ChainGroup = 3 });
        float[] fusedA = new float[ParityRows * count];
        float[] fusedB = new float[ParityRows * count];
        Assert.True(fused.TryGemvBatchSharedInput(inputs, count, weightA, fusedA, weightB, fusedB), $"{formatA}+{formatB} batch fusion refused at a pinned chain group");

        BatchKernelTests.AssertBitsEqual(expectedA, fusedA, $"fused batch A {formatA}/{formatB} at pinned chain group");
        BatchKernelTests.AssertBitsEqual(expectedB, fusedB, $"fused batch B {formatA}/{formatB} at pinned chain group");
    }

    // A pinned scratch ceiling below the matrix width demotes the batch route to the streaming kernels — a route change, not a value change, and the bytes must not move.
    private static IEnumerable<CpuKernelTuningOverrides> TuningPostures()
    {
        yield return new CpuKernelTuningOverrides { RowBlock = 3, ParallelRowThreshold = 1 };
        yield return new CpuKernelTuningOverrides { ChainGroup = 3 };
        yield return new CpuKernelTuningOverrides { MaxScratchCols = CpuKernelTuning.MaxScratchColsMin }; // 768-column matrices fall out of the scratch route here
        yield return new CpuKernelTuningOverrides { RowBlock = 1, ParallelRowThreshold = 1 };
    }
}
