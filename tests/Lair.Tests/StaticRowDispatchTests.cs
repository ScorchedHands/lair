using Lair.Backends.Cpu;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// The unpromoted static contiguous row partition for threaded CPU dense dispatch (internal measurement seam): what a posture may and may not
/// run with it, how its schedule is shaped, and the parity contract that keeps it a scheduling choice rather than an arithmetic one. Today every
/// resolved posture states no crossover, so dynamic block claiming stays in force; only a stated width at or above a call's batch count gives each
/// participant one contiguous range of canonical rows instead. Rows are independent either way, so batched equals solo equals threaded byte-for-byte
/// across counts 8..64, ragged tails, multi-block columns, and every format these dispatches run — the seam moves who walks a row, never how it is walked.
/// Single-vector GEMV decode, fused pair/triple traversals, shared-operand dot batching, and an unthreaded call never consult this seam at any value.
/// </summary>
[Trait("Category", "Kernel")]
[Trait("Category", "CpuTuning")]
public sealed class StaticRowDispatchTests
{
    // Feed-forward-shaped geometries with ragged column tails: a partial final quant block (137, 264), columns past several 256-element blocks,
    // and row counts that do not divide evenly into any scheduling-block width or participant count used below.
    private static readonly (int Rows, int Cols)[] WideShapes = [(100, 137), (9, 264), (130, 300)];

    // Counts at and above the stated crossover plus a ragged one across chain-group boundaries.
    private static readonly int[] PartitionCounts = [8, 17, 32, 64];

    // The thread postures these tests partition against: small quotas for empty-range participation and an eight-worker quota where ranges dominate the schedule.
    private static readonly int[] ThreadPostures = [2, 5, 8];

    // ---------------------------------------------------------------- parity across schedules, thread counts, formats, and widths

    [Theory]
    [MemberData(nameof(MatrixSizesAndFormatsData))]
    public void GemvBatch_StaticRowPartition_BitIdenticalToDynamicAtEveryThreadCount(int rows, int cols, QuantFormat format)
    {
        foreach (int count in PartitionCounts)
        {
            (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, format, seed: 81_000 + count * 31 + rows, vectors: count);
            WeightRef weight = new(weights, new TensorShape([rows, cols]), format);

            // Reference arm: today's posture on one thread — what every resolved posture runs and the byte contract every other arm must match.
            using CpuBackend reference = new(threads: 1);
            Assert.Equal(0, reference.ResolvedTuning.StaticRowPartitionBatchWidth); // no auto posture states a crossover for this seam.

            float[] expected = new float[count * rows];
            reference.GemvBatch(inputs, count, weight, expected);

            foreach (int threads in ThreadPostures)
            {
                // Dynamic arm: threaded claiming with the dispatch threshold pinned low so these shapes actually thread at every quota.
                using CpuBackend dynamicArm = new(threads: threads, tuning: new CpuKernelTuningOverrides { ParallelRowThreshold = 1 });
                Assert.Equal(0, dynamicArm.ResolvedTuning.StaticRowPartitionBatchWidth);

                float[] actualDynamic = new float[count * rows];
                dynamicArm.GemvBatch(inputs, count, weight, actualDynamic);
                BatchKernelTests.AssertBitsEqual(expected, actualDynamic, $"threaded dynamic claiming at {threads} workers for {format} at {rows}x{cols} count {count}");

                // Static arm: contiguous per-participant ranges over the same rows, with a stated crossover covering this batch. One engaged dispatch on this fresh backend.
                using CpuBackend staticArm = new(threads: threads, tuning: new CpuKernelTuningOverrides { ParallelRowThreshold = 1, StaticRowPartitionBatchWidth = 8 });
                Assert.Equal(8, staticArm.ResolvedTuning.StaticRowPartitionBatchWidth);

                float[] actualStatic = new float[count * rows];
                staticArm.GemvBatch(inputs, count, weight, actualStatic);
                BatchKernelTests.AssertBitsEqual(expected, actualStatic, $"static contiguous partition at {threads} workers for {format} at {rows}x{cols} count {count}");

                Assert.Equal(1L, staticArm.StaticRowPartitionDispatchCount); // the seam actually claimed this call's schedule rather than falling back to claiming.
            }
        }
    }

    /// <summary>
    /// Ragged scheduling-block tails under a stated partition: a row block that does not divide the matrix, participants wider than the block count
    /// (many empty contiguous ranges), and counts where the last participant's range is shorter than the others. Output bytes never move.
    /// </summary>
    [Theory]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.IQ3_XXS)]
    public void GemvBatch_StaticRowPartition_RaggedBlocksAndWiderQuotas_KeepEveryByte(QuantFormat format)
    {
        const int count = 16;

        foreach ((int rows, int cols) in new[] { (50, 264), (37, 137), (9, 300) })
        {
            (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, format, seed: 82_000 + rows * 7 + cols, vectors: count);
            WeightRef weight = new(weights, new TensorShape([rows, cols]), format);

            using CpuBackend reference = new(threads: 1);
            float[] expected = new float[count * rows];
            reference.GemvBatch(inputs, count, weight, expected);

            foreach (int threads in new[] { 2, 3, 8 })
            {
                // RowBlock pinned past the row count for some quotas: block counts of one or two against eight participants leave most ranges empty.
                foreach (int rowBlock in new[] { 1, 7, 64 })
                {
                    using CpuBackend staticArm = new(threads: threads, tuning: new CpuKernelTuningOverrides { ParallelRowThreshold = 1, RowBlock = rowBlock, StaticRowPartitionBatchWidth = 8 });

                    float[] actualStatic = new float[count * rows];
                    staticArm.GemvBatch(inputs, count, weight, actualStatic);
                    BatchKernelTests.AssertBitsEqual(expected, actualStatic, $"static partition with {rowBlock}-row blocks at {threads} workers for {format} at {rows}x{cols}");

                    using CpuBackend dynamicArm = new(threads: threads, tuning: new CpuKernelTuningOverrides { ParallelRowThreshold = 1, RowBlock = rowBlock });
                    float[] actualDynamic = new float[count * rows];
                    dynamicArm.GemvBatch(inputs, count, weight, actualDynamic);
                    BatchKernelTests.AssertBitsEqual(expected, actualDynamic, $"dynamic claiming with {rowBlock}-row blocks at {threads} workers for {format} at {rows}x{cols}");

                    Assert.Equal(1L, staticArm.StaticRowPartitionDispatchCount); // every one of these shapes threaded on contiguous ranges.
                }
            }
        }
    }

    /// <summary>The blocked-matmul tile dispatch honors the same seam: a stated crossover gives contiguous ranges there too, byte-for-byte against both its own dynamic schedule and single-thread reference — covering the public dense-matrix path as well as the hybrid batched route.</summary>
    [Theory]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.Q6_K)]
    public void GemmBatch_BlockedMatmulDispatch_StaticRowPartition_BitIdenticalAcrossThreadCounts(QuantFormat format)
    {
        const int rows = 300; // three hundred weight rows across a handful of scheduling blocks at these quotas.
        const int cols = 1_024;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, format, seed: 83_000 + rows, vectors: 16);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), format);

        using CpuBackend reference = new(threads: 1);
        Assert.True(((IBackend)reference).SupportsGemm(format)); // these formats carry a blocked matmul in this build.

        float[] expected = new float[16 * rows];
        ((IBackend)reference).GemmBatch(inputs, 16, weight, expected);

        foreach (int threads in new[] { 2, 7 })
        {
            using CpuBackend dynamicArm = new(threads: threads, tuning: new CpuKernelTuningOverrides { ParallelRowThreshold = 1 });
            float[] actualDynamic = new float[16 * rows];
            ((IBackend)dynamicArm).GemmBatch(inputs, 16, weight, actualDynamic);
            BatchKernelTests.AssertBitsEqual(expected, actualDynamic, $"blocked matmul with dynamic claiming at {threads} workers for {format}");
            Assert.Equal(0L, dynamicArm.StaticRowPartitionDispatchCount); // no width stated: the claim loop ran.

            using CpuBackend staticArm = new(threads: threads, tuning: new CpuKernelTuningOverrides { ParallelRowThreshold = 1, StaticRowPartitionBatchWidth = 8 });
            float[] actualStatic = new float[16 * rows];
            ((IBackend)staticArm).GemmBatch(inputs, 16, weight, actualStatic);
            BatchKernelTests.AssertBitsEqual(expected, actualStatic, $"blocked matmul on contiguous static ranges at {threads} workers for {format}");

            Assert.Equal(1L, staticArm.StaticRowPartitionDispatchCount); // the dense-matrix path consulted the same seam.
        }
    }

    /// <summary>The hybrid Q4_K tile crossover and the static partition are independent statements over the same rows: pinning both keeps every byte of each unpromoted contract — including the integer executor's own determinism across schedules (AVX2 rung forced so resolution is deterministic across machines).</summary>
    [Fact]
    public void GemvBatch_Q4KTileAndIntegerSeam_StaticRowPartition_KeepsBothUnpromotedContracts()
    {
        const int rows = 512;
        const int cols = 1_024;

        CpuFeaturePosture avx2Only = new(Avx2Available: true, AvxVnniAvailable: false);
        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 84_000, vectors: 16);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend reference = new(threads: 1, cpuFeatures: avx2Only);
        float[] expectedFloat = new float[16 * rows];
        reference.GemvBatch(inputs, 16, weight, expectedFloat); // today's traversal on one thread.

        foreach (int threads in new[] { 4, 8 })
        {
            using CpuBackend tileDynamic = new(threads: threads, cpuFeatures: avx2Only, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 8 });
            float[] actualTileDynamic = new float[16 * rows];
            tileDynamic.GemvBatch(inputs, 16, weight, actualTileDynamic);
            BatchKernelTests.AssertBitsEqual(expectedFloat, actualTileDynamic, $"stated Q4_K tile crossover with dynamic claiming at {threads} workers");

            using CpuBackend tileStatic = new(threads: threads, cpuFeatures: avx2Only, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 8, StaticRowPartitionBatchWidth = 8 });
            float[] actualTileStatic = new float[16 * rows];
            tileStatic.GemvBatch(inputs, 16, weight, actualTileStatic);
            BatchKernelTests.AssertBitsEqual(expectedFloat, actualTileStatic, $"stated Q4_K tile crossover on contiguous static ranges at {threads} workers");

            // The integer executor claims the pass before any float traversal when both seams are stated; its own determinism contract then runs across schedules.
            using CpuBackend integerDynamic = new(threads: threads, cpuFeatures: avx2Only, tuning: new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8 });
            float[] actualIntegerDynamic = new float[16 * rows];
            integerDynamic.GemvBatch(inputs, 16, weight, actualIntegerDynamic);

            using CpuBackend integerStatic = new(threads: threads, cpuFeatures: avx2Only, tuning: new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8, StaticRowPartitionBatchWidth = 8 });
            float[] actualIntegerStatic = new float[16 * rows];
            integerStatic.GemvBatch(inputs, 16, weight, actualIntegerStatic);

            BatchKernelTests.AssertBitsEqual(actualIntegerDynamic, actualIntegerStatic, $"integer Q4_K executor on contiguous static ranges at {threads} workers"); // the same staged operands and chain order every schedule.
        }
    }

    // ---------------------------------------------------------------- no posture promotes it; only a stated width engages

    [Fact]
    public void Resolve_StaticRowPartitionStaysUnpromotedAtEveryPosture()
    {
        foreach (int workers in new[] { 1, 8, 32 })
        {
            Assert.Equal(0, CpuKernelTuning.Resolve(workers, vectorPosture: true, CpuKernelTuningOverrides.Auto).StaticRowPartitionBatchWidth); // no auto posture states a crossover for it...
            Assert.Equal(0, CpuKernelTuning.Resolve(workers, vectorPosture: false, CpuKernelTuningOverrides.Auto).StaticRowPartitionBatchWidth); // ...and threading posture alone never opens it either.
        }
    }

    [Fact]
    public void Resolve_StaticRowPartitionPinResolvesExactly()
    {
        foreach (int width in new[] { 0, 1, 8, 64 })
        {
            CpuKernelTuning tuning = CpuKernelTuning.Resolve(32, vectorPosture: true, new CpuKernelTuningOverrides { StaticRowPartitionBatchWidth = width });

            Assert.Equal(width, tuning.StaticRowPartitionBatchWidth);
        }
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(65)] // past the batch ceiling every dense kernel's scratch is sized for
    public void Resolve_StaticRowPartitionPinOutsideItsSpan_RefusesNamingQualifiedKeyAndValue(int value)
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(
            () => CpuKernelTuning.Resolve(8, vectorPosture: true, new CpuKernelTuningOverrides { StaticRowPartitionBatchWidth = value }));

        Assert.Equal("engine.cpu.tuning.static_row_partition_batch_width", error.ParamName);
        Assert.Contains($"engine.cpu.tuning.static_row_partition_batch_width {value}", error.Message);
    }

    /// <summary>The gate is a batch count: below the stated width every call keeps dynamic claiming, single-vector GEMV decode never engages this seam at any value, and a call that does not thread has no participants to partition.</summary>
    [Fact]
    public void StaticRowPartitionGate_KeepsCallsBelowItsWidthAndSingleVectorDecodeOnDynamicClaiming()
    {
        const int rows = 300;
        const int cols = 1_024;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 85_000, vectors: 64);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend pinned = new(threads: 8, tuning: new CpuKernelTuningOverrides { StaticRowPartitionBatchWidth = 16 });

        for (int count = 1; count <= 15; count++)
        {
            float[] slice = inputs.AsSpan(0, count * cols).ToArray();
            pinned.GemvBatch(slice, count, weight, new float[count * rows]); // below the stated width: dynamic claiming.

            Assert.Equal(0L, pinned.StaticRowPartitionDispatchCount);
        }

        for (int call = 0; call < 3; call++)
        {
            pinned.Gemv(new float[cols], weight, new float[rows]); // single-vector GEMV: this seam never reaches a decode call.
        }

        Assert.Equal(0L, pinned.StaticRowPartitionDispatchCount);

        pinned.GemvBatch(inputs.AsSpan(0, 16 * cols).ToArray(), 16, weight, new float[16 * rows]); // at the width with threading in force: now it engages.
        Assert.Equal(1L, pinned.StaticRowPartitionDispatchCount);

        using CpuBackend unthreaded = new(threads: 1, tuning: new CpuKernelTuningOverrides { StaticRowPartitionBatchWidth = 8 });
        float[] solo = new float[64 * rows];
        unthreaded.GemvBatch(inputs, 64, weight, solo); // a call that does not thread at all partitions nothing.

        Assert.Equal(0L, unthreaded.StaticRowPartitionDispatchCount);
    }

    /// <summary>A fused pair's staged traversal and its two per-matrix tile dispatches are never reached by this seam at any stated width: pairs keep dynamic claiming byte-for-byte.</summary>
    [Fact]
    public void TryGemvBatchSharedInput_FusedPairNeverEngagesStaticRowPartition()
    {
        const int rows = 256;
        const int cols = 1_024;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 86_000, vectors: 32);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend defaultPosture = new(threads: 8);
        float[] expectedA = new float[32 * rows];
        float[] expectedB = new float[32 * rows];
        Assert.True(((IDenseWeightPairFusion)defaultPosture).TryGemvBatchSharedInput(inputs, 32, weight, expectedA, weight, expectedB));

        using CpuBackend statedWidths = new(threads: 8, tuning: new CpuKernelTuningOverrides { StaticRowPartitionBatchWidth = 8, PairMatmulRouteBatchWidth = 8 });
        float[] outA = new float[32 * rows];
        float[] outB = new float[32 * rows];

        Assert.True(((IDenseWeightPairFusion)statedWidths).TryGemvBatchSharedInput(inputs, 32, weight, outA, weight, outB)); // both the staged pair traversal and its tile crossover stay off this seam.
        BatchKernelTests.AssertBitsEqual(expectedA, outA, $"gate side at {rows}x{cols} under a stated static-partition width");
        BatchKernelTests.AssertBitsEqual(expectedB, outB, $"up side at {rows}x{cols} under a stated static-partition width");

        Assert.Equal(0L, statedWidths.StaticRowPartitionDispatchCount); // no pair dispatch consulted the partition.

        ((IBackend)statedWidths).GemmBatch(inputs, 32, weight, new float[32 * rows]); // a single-matrix dense pass does engage it...
        Assert.Equal(1L, statedWidths.StaticRowPartitionDispatchCount);
    }

    /// <summary>Participation resolved from the live quota: lowering it mid-life re-partitions against fewer participants and keeps every byte identical, including across formats whose batched route stages a quantized activation.</summary>
    [Fact]
    public void ApplyThreadQuota_StaticRowPartitionReResolvesFromLiveQuota_KeepsEveryByte()
    {
        const int rows = 300;
        const int cols = 1_024;

        foreach (QuantFormat format in new[] { QuantFormat.Q4_K, QuantFormat.IQ4_XS })
        {
            (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, format, seed: 87_000 + rows, vectors: 32);
            WeightRef weight = new(weights, new TensorShape([rows, cols]), format);

            using CpuBackend reference = new(threads: 1);
            float[] expected = new float[32 * rows];
            reference.GemvBatch(inputs, 32, weight, expected);

            using CpuBackend backend = new(threads: 8, tuning: new CpuKernelTuningOverrides { ParallelRowThreshold = 1, StaticRowPartitionBatchWidth = 8 });

            foreach (int quota in new[] { 8, 5, 2 })
            {
                int effective = backend.ApplyThreadQuota(quota);
                Assert.Equal(quota, effective); // the partition resolves against this call's own participant count.

                float[] actual = new float[32 * rows];
                backend.GemvBatch(inputs, 32, weight, actual);
                BatchKernelTests.AssertBitsEqual(expected, actual, $"static contiguous partition re-resolved at quota {effective} for {format}");
            }

            Assert.Equal(3L, backend.StaticRowPartitionDispatchCount); // one engaged dispatch per quota posture.
        }
    }

    /// <summary>The schedule itself is deterministic: the same posture run repeatedly writes identical bits, and a static arm matches its dynamic neighbour at the same quota byte-for-byte.</summary>
    [Fact]
    public void StaticRowPartition_RepeatedRunsAndScheduleEquality_WriteIdenticalBits()
    {
        const int rows = 512;
        const int cols = 300;
        const int count = 64;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q6_K, seed: 88_000, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q6_K); // Q6_K takes the blocked-matmul route at this count under an auto posture.

        using CpuBackend staticArm = new(threads: 8, tuning: new CpuKernelTuningOverrides { StaticRowPartitionBatchWidth = 8 });
        float[] firstRun = new float[count * rows];
        staticArm.GemvBatch(inputs, count, weight, firstRun);

        for (int repeat = 0; repeat < 3; repeat++)
        {
            float[] again = new float[count * rows];
            staticArm.GemvBatch(inputs, count, weight, again); // the same contiguous schedule every time: identical bytes.
            BatchKernelTests.AssertBitsEqual(firstRun, again, $"static partition repeat {repeat}");
        }

        using CpuBackend dynamicArm = new(threads: 8);
        float[] dynamicRun = new float[count * rows];
        dynamicArm.GemvBatch(inputs, count, weight, dynamicRun); // today's claiming over the same fixture.
        BatchKernelTests.AssertBitsEqual(firstRun, dynamicRun, "static partition against its dynamic neighbour at the same quota");

        Assert.Equal(4L, staticArm.StaticRowPartitionDispatchCount);
        Assert.Equal(0L, dynamicArm.StaticRowPartitionDispatchCount); // an auto posture never engages this seam.
    }

    // ---------------------------------------------------------------- schedule shape: contiguous ranges over canonical order

    public static TheoryData<int, int> RangeShapeData()
    {
        TheoryData<int, int> data = [];
        foreach (int participants in new[] { 2, 3, 5, 7, 16, 32 })
        {
            foreach (int blocks in new[] { 1, 2, 3, 4, 7, 8, 9, 17, 33, 689 })
            {
                data.Add(participants, blocks);
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(RangeShapeData))]
    public void StaticPartitionRange_ContiguousDisjointAndCompleteOverCanonicalOrder(int participants, int totalBlocks)
    {
        bool[] claimed = new bool[totalBlocks];
        int previousFrom = 0;

        for (int participant = 0; participant < participants; participant++)
        {
            (int from, int to) = CpuBackend.StaticPartitionRange(participant, participants, totalBlocks);

            Assert.True(from <= to, $"participant {participant} of {participants} over {totalBlocks} blocks got an inverted range [{from}, {to})"); // every range is contiguous or empty.
            Assert.Equal(previousFrom, from); // ranges abut in canonical order: no gap between neighbours' pieces.

            for (int block = from; block < to; block++)
            {
                Assert.False(claimed[block], $"block {block} belongs to more than one participant of {participants} over {totalBlocks} blocks");
                claimed[block] = true;
            }

            previousFrom = to; // the next participant starts where this one ended.
        }

        for (int block = 0; block < totalBlocks; block++)
        {
            Assert.True(claimed[block], $"block {block} of {totalBlocks} was left unpartitioned across {participants} participants");
        }
    }

    [Fact]
    public void StaticPartitionRange_RefusesWhenThereIsNothingToPartition()
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(
            () => CpuBackend.StaticPartitionRange(0, 1, 8)); // one participant has no range to hand out.

        Assert.Equal("participant", error.ParamName);
        Assert.Contains("1", error.Message);
    }

    [Fact]
    public void StaticPartitionRange_RefusesOutOfRangeParticipantAndNegativeBlockCount()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => CpuBackend.StaticPartitionRange(4, 4, 8)); // a participant index at the participants...
        Assert.Throws<ArgumentOutOfRangeException>(() => CpuBackend.StaticPartitionRange(9, 4, 8)); // ...and past them are both refusals.
        Assert.Throws<ArgumentOutOfRangeException>(() => CpuBackend.StaticPartitionRange(2, 3, -1)); // a negative block count is nothing to schedule at all.

        (int from, int to) = CpuBackend.StaticPartitionRange(0, 3, 0); // zero blocks partitions into empty ranges rather than inventing work.
        Assert.Equal(0, from);
        Assert.Equal(to, from);
    }

    public static TheoryData<int, int, QuantFormat> MatrixSizesAndFormatsData()
    {
        TheoryData<int, int, QuantFormat> data = [];
        foreach ((int rows, int cols) in WideShapes)
        {
            foreach (QuantFormat format in new[] { QuantFormat.F32, QuantFormat.Q8_0, QuantFormat.Q4_K, QuantFormat.Q5_K, QuantFormat.IQ4_XS, QuantFormat.IQ3_XXS })
            {
                data.Add(rows, cols, format);
            }
        }

        return data;
    }
}
