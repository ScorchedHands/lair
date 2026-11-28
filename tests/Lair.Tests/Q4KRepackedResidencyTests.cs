using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Models;

namespace Lair.Tests;

/// <summary>
/// The budget-gated repacked Q4_K residency probe (internal, unpromoted): allocation gated by a stated weights-budget cap, selection gated by a
/// stated batch crossover on an AVX2-resolved backend, refusal semantics that name budget vocabulary and both byte costs instead of narrowing or
/// falling back, and the route's own determinism contract. Quantizing FP32 activations to signed bytes loses precision against the float kernels, so
/// outputs are NOT expected bit-identical to them — what this variant guarantees instead is ascending super-block walks in fixed integer lane order
/// with a fixed per-block fold sequence, so batched equals solo equals threaded at any worker count and any chain grouping. No resolved posture states
/// the crossover and no default engine posture engages ordinary reads of prepared copies: every test here opens preparation or measurement engagement
/// through internal seams explicitly, which is also how default-posture preservation gets asserted — an auto backend with a prepared copy resident still runs today's float traversals byte-for-byte.
/// </summary>
[Trait("Category", "Kernel")]
[Trait("Category", "CpuTuning")]
public sealed class Q4KRepackedResidencyTests
{
    // Feed-forward-shaped geometries with ragged column tails: a partial final quant block (137, 264), columns past several 256-element blocks,
    // and row counts that do not divide evenly into any scheduling-block width or participant count used below.
    private static readonly (int Rows, int Cols)[] MatrixSizes = [(100, 137), (9, 264), (130, 300), (512, 1024)];

    // Counts at and above the stated width, ragged counts across chain-group boundaries, and single-vector calls below it.
    private static readonly int[] RepackedCounts = [8, 9, 16, 20, 32, 63, 64];

    // AVX2-only posture: the one resolution this variant is stated for. (This machine's auto lands on Vnni.)
    private static CpuFeaturePosture Avx2Only => new(Avx2Available: true, AvxVnniAvailable: false);

    private static ResolvedBudgetChain WeightsChain(long? dramCap) => new()
    {
        Resource = BudgetResource.Weights,
        Tiers =
        [
            new ResolvedBudgetTier { Tier = BudgetTier.Dram, IsOriginStore = false, Cap = dramCap },
            new ResolvedBudgetTier { Tier = BudgetTier.Disk, IsOriginStore = true }, // the model file's own packed set: bounds nothing of ours.
        ],
    };

    private static CpuBackend OpenAvx2Only(int threads = 1, int repackedWidth = 8) =>
        new(threads: threads, cpuFeatures: Avx2Only, tuning: new CpuKernelTuningOverrides { Q4KRepackedResidencyBatchWidth = repackedWidth });

    // ---------------------------------------------------------------- the budget gate: ceilings decide what a resident copy may hold

    [Fact]
    public void Gate_CeilingThatFits_StatesNoClamp_TodaysShapeUnchanged()
    {
        int cols = 1024;
        long rows = 512; // repacked cost: 4 blocks x 272 bytes per row -> 557,056 bytes total.

        Assert.Null(Q4KRepackedResidencyGate.Compute(WeightsChain(dramCap: 4), cols, requestedRows: rows));
        Assert.Null(Q4KRepackedResidencyGate.Compute(WeightsChain(dramCap: null), cols, requestedRows: long.MaxValue / 2)); // a chain with no ceiling bounds nothing.
    }

    [Fact]
    public void Gate_OverDeclaredCeiling_ClampNamesTierCapAndBothRowCosts()
    {
        int cols = 1024;
        long rows = 3000; // repacked cost: 4 blocks x 272 bytes per row -> over three megabytes against a one-megabyte ceiling.

        Q4KRepackedRowClamp? clamp = Q4KRepackedResidencyGate.Compute(WeightsChain(dramCap: 1), cols, requestedRows: rows);

        Assert.NotNull(clamp);
        Assert.True(clamp!.AllowsPositiveRows); // 1 MiB admits whole repacked rows here; the narrowing is what a diagnostic reports.
        Assert.Equal(BudgetTier.Dram, clamp.Tier);
        Assert.Equal(1, clamp.CapMegabytes);
        Assert.Equal(Q4KResidencyLayout.PackedRowBytes(cols), clamp.PackedRowBytes); // 576 packed vs 1088 repacked per row at this width.
        Assert.Equal(Q4KResidencyLayout.RepackedRowBytes(cols), clamp.RepackedRowBytes);
        Assert.True(clamp.AllowedRows < rows && clamp.AllowedRows == (1L * 1024 * 1024) / Q4KResidencyLayout.RepackedRowBytes(cols));

        string described = clamp.Describe();
        Assert.Contains("'dram' caps weights residency at 1 megabytes", described);
        Assert.Contains($"adds {Q4KResidencyLayout.RepackedRowBytes(cols)} resident bytes", described);
        Assert.Contains("narrow from 3000 to ", described); // requested against allowed, in whole rows — never a silent narrowing.
    }

    [Fact]
    public void Gate_CeilingBelowOneRepackedRow_StillReportsTheUnfitClamp()
    {
        int cols = 4 * 1024 * 1024; // one repacked row costs over four megabytes: a 1 MiB ceiling admits zero rows.

        Q4KRepackedRowClamp? clamp = Q4KRepackedResidencyGate.Compute(WeightsChain(dramCap: 1), cols, requestedRows: 1);

        Assert.NotNull(clamp);
        Assert.False(clamp!.AllowsPositiveRows); // the posture reads off diagnostics rather than vanishing into a silently narrowed residency.
        Assert.Equal(0, clamp.AllowedRows);
    }

    [Fact]
    public void Gate_ChainOfAnotherResource_RefusesNamingBudgetVocabulary()
    {
        ResolvedBudgetChain kvChain = new()
        {
            Resource = BudgetResource.Kv,
            Tiers = [new ResolvedBudgetTier { Tier = BudgetTier.Dram, IsOriginStore = false, Cap = 1024 }],
        };

        ModelException error = Assert.Throws<ModelException>(() => Q4KRepackedResidencyGate.Compute(kvChain, cols: 256, requestedRows: 8));
        Assert.Contains("'kv' is not 'weights'", error.Message); // repacked weight copies bound against exactly one resource.
    }

    [Theory]
    [InlineData(137)]
    [InlineData(256)]
    [InlineData(1024)]
    [InlineData(11_008)]
    public void Layout_PackedCostMatchesTheFormatsOwnStride_AndRepackedCostIsHeaderPlusLanes(int cols)
    {
        TensorShape shape = new([2, cols]);

        Assert.Equal(shape.RowStrideBytes(QuantFormat.Q4_K), Q4KResidencyLayout.PackedRowBytes(cols));
        Assert.Equal(Q4KResidencyLayout.BlocksPerRow(cols) * (Q4KResidencyLayout.MetadataBytes + 256), Q4KResidencyLayout.RepackedRowBytes(cols));
    }

    // ---------------------------------------------------------------- preparation: the stated cap admits or refuses, naming its numbers

    [Fact]
    public void Prepare_CapThatFits_AllocatesTheRepackedCopyAndReportsItsCost()
    {
        const int rows = 300;
        const int cols = 1024;

        (float[] _, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 91_000, vectors: 8);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend backend = OpenAvx2Only();
        long capMegabytes = Q4KResidencyLayout.RepackedRowBytes(cols) * rows / (1024 * 1024) + 1; // the ceiling this matrix's own cost needs.

        using Q4KRepackedMatrix repacked = backend.PrepareDenseQ4KResidency(weight, capMegabytes);

        Assert.True(repacked.IsPrepared);
        Assert.Equal(rows, repacked.Rows);
        Assert.Equal(cols, repacked.Cols);
        Assert.Equal((long)rows * Q4KResidencyLayout.PackedRowBytes(cols), repacked.PackedBytes); // 144-byte super-blocks...
        Assert.Equal((long)rows * Q4KResidencyLayout.RepackedRowBytes(cols), repacked.RepackedBytes); // ...against this copy's 272-byte ones.
        Assert.Equal(repacked.RepackedBytes, backend.PreparedQ4KResidencyBytes);
    }

    [Fact]
    public void Prepare_OverStatedCeiling_RefusesNamingBudgetFormatAndNumbers_AndAllocatesNothing()
    {
        const int rows = 3000; // repacked cost: 12 blocks x 272 per row -> over three megabytes against a one-megabyte ceiling.
        const int cols = 1024;

        (float[] _, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 92_000, vectors: 8);

        using CpuBackend backend = OpenAvx2Only();

        // WeightRef is a ref struct, so the refused matrix gets built inside each throwing call rather than captured as a local.
        Q4KRepackedResidencyException error = Assert.IsType<Q4KRepackedResidencyException>(Record.Exception(() =>
            backend.PrepareDenseQ4KResidency(new WeightRef(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K), weightsBudgetCapMegabytes: 1)));

        long packedBytes = (long)rows * Q4KResidencyLayout.PackedRowBytes(cols);
        long repackedBytes = (long)rows * Q4KResidencyLayout.RepackedRowBytes(cols);

        Assert.Contains("repacked Q4_K residency refused", error.Message); // the refusal is a whole-request answer, not a narrowing...
        Assert.Contains("'weights.caps.dram_mb' ceiling of 1 megabytes", error.Message); // ...it names the budget key and cap number...
        Assert.Contains($"format Q4_K dense matrix {rows}x{cols}", error.Message); // ...the format...
        Assert.Contains($"{packedBytes} packed bytes", error.Message); // ...both totals, per-row costs, and the comparison itself.
        Assert.Contains($"{repackedBytes} bytes", error.Message); // ...both totals...
        Assert.Contains($"{Q4KResidencyLayout.RepackedBlockBytes} per 256-element super-block, {Q4KResidencyLayout.RepackedRowBytes(cols)} per row", error.Message); // ...and the comparison's own per-row costs.
        Assert.Equal(0, backend.PreparedQ4KResidencyBytes); // nothing falls back to a narrower copy.
    }

    [Fact]
    public void Prepare_VnniResolvedBackend_RefusesNamingTheRungItNeverRuns_AndAllocatesNothing()
    {
        const int rows = 300;
        const int cols = 1024;

        (float[] _, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 93_000, vectors: 8);

        using CpuBackend vnni = new(threads: 1, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: true));
        Assert.Equal(IntegerDotPath.Vnni, vnni.ResolvedIntegerDotPath); // this machine's auto lands here today.

        Q4KRepackedResidencyException error = Assert.IsType<Q4KRepackedResidencyException>(Record.Exception(() =>
            vnni.PrepareDenseQ4KResidency(new WeightRef(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K), weightsBudgetCapMegabytes: 1024)));

        Assert.Contains("AVX2 integer-dot rung only", error.Message); // the posture names what it resolved instead of borrowing another ladder.
        Assert.Equal(0, vnni.PreparedQ4KResidencyBytes);
    }

    [Fact]
    public void Prepare_ScalarPosture_Refuses_AndAllocatesNothing()
    {
        const int rows = 300;
        const int cols = 1024;

        (float[] _, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 94_000, vectors: 8);

        using CpuBackend scalar = new(threads: 1, allowAvx2: false);

        Q4KRepackedResidencyException error = Assert.IsType<Q4KRepackedResidencyException>(Record.Exception(() =>
            scalar.PrepareDenseQ4KResidency(new WeightRef(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K), weightsBudgetCapMegabytes: 1024)));

        Assert.Contains("no vector kernel", error.Message);
        Assert.Equal(0, scalar.PreparedQ4KResidencyBytes);
    }

    [Fact]
    public void Prepare_FormatOutsideQ4K_RefusesNamingFormat_AndAllocatesNothing()
    {
        const int rows = 300;
        const int cols = 1024;

        (float[] _, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q6_K, seed: 95_000, vectors: 8);

        using CpuBackend backend = OpenAvx2Only();

        Q4KRepackedResidencyException error = Assert.IsType<Q4KRepackedResidencyException>(Record.Exception(() =>
            backend.PrepareDenseQ4KResidency(new WeightRef(weights, new TensorShape([rows, cols]), QuantFormat.Q6_K), weightsBudgetCapMegabytes: 1024)));

        Assert.Contains("stated for format Q4_K only", error.Message); // no other format borrows this layout's cost numbers.
        Assert.Equal(0, backend.PreparedQ4KResidencyBytes);
    }

    [Fact]
    public void Prepare_DisposedCopy_ReleasesItsBlock_AndFurtherReadsRefuse()
    {
        const int rows = 128;
        const int cols = 512;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 96_000, vectors: 8);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend backend = OpenAvx2Only();
        Q4KRepackedMatrix repacked = backend.PrepareDenseQ4KResidency(weight, weightsBudgetCapMegabytes: 1024);
        float[] outputs = new float[8 * rows];

        backend.GemvBatch(inputs, 8, weight, repacked, outputs); // the copy serves its reads while prepared...
        Assert.False(float.IsNaN(outputs[0]));

        repacked.Dispose();
        Assert.False(repacked.IsPrepared);

        Assert.IsType<ObjectDisposedException>(Record.Exception(() => backend.GemvBatch(inputs, 8, new WeightRef(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K), repacked, outputs))); // ...and refuses once released rather than reading freed memory.
    }

    // ---------------------------------------------------------------- selection: only a stated crossover on an AVX2-resolved backend reads the copy back

    [Fact]
    public void GemvBatchRepacked_UnstatedWidth_RefusesNamingQualifiedKeyAndCount()
    {
        const int rows = 300;
        const int cols = 1024;
        const int count = 8;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 97_000, vectors: count);

        using CpuBackend unstated = OpenAvx2Only(repackedWidth: 0); // the seam opened without its crossover.
        using Q4KRepackedMatrix repacked = unstated.PrepareDenseQ4KResidency(new WeightRef(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K), weightsBudgetCapMegabytes: 1024);

        float[] outputs = new float[count * rows];

        ArgumentOutOfRangeException error = Assert.IsType<ArgumentOutOfRangeException>(Record.Exception(() =>
            unstated.GemvBatch(inputs, count, new WeightRef(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K), repacked, outputs)));

        Assert.Equal("engine.cpu.tuning.q4k_repacked_residency_batch_width", error.ParamName); // a prepared copy never changes what an unstated pass runs.
        Assert.Contains("resolved 0 against count 8", error.Message);
    }

    [Fact]
    public void GemvBatchRepacked_CountBelowStatedWidth_RefusesNamingResolvedAgainstRequested()
    {
        const int rows = 300;
        const int cols = 1024;
        const int count = 7; // one below the stated crossover.

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 98_000, vectors: count);

        using CpuBackend backend = OpenAvx2Only();
        using Q4KRepackedMatrix repacked = backend.PrepareDenseQ4KResidency(new WeightRef(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K), weightsBudgetCapMegabytes: 1024);

        float[] outputs = new float[count * rows];

        ArgumentOutOfRangeException error = Assert.IsType<ArgumentOutOfRangeException>(Record.Exception(() =>
            backend.GemvBatch(inputs, count, new WeightRef(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K), repacked, outputs)));

        Assert.Equal("engine.cpu.tuning.q4k_repacked_residency_batch_width", error.ParamName);
        Assert.Contains("resolved 8 against count 7", error.Message); // the refusal compares this call's own numbers.
    }

    [Fact]
    public void GemvBatchRepacked_WrongMatrixPairing_RefusesNamingBothShapes()
    {
        const int rows = 300;
        const int cols = 1024;
        const int count = 8;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 99_000, vectors: count);
        // The wrong matrix is built inside the throwing call; `other` was never a separate ref-struct local.

        using CpuBackend backend = OpenAvx2Only();
        using Q4KRepackedMatrix repacked = backend.PrepareDenseQ4KResidency(new WeightRef(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K), weightsBudgetCapMegabytes: 1024);

        float[] outputs = new float[count * rows];

        ArgumentException error = Assert.IsType<ArgumentException>(Record.Exception(() =>
            backend.GemvBatch(inputs, count, new WeightRef(weights, new TensorShape([rows, 137]), QuantFormat.Q4_K), repacked, outputs)));

        Assert.Contains("300x137", error.Message); // a resident read names exactly the matrix that was prepared.
    }

    [Fact]
    public void GemvBatchRepacked_VnniResolvedBackend_RunsNoResidentRead()
    {
        const int rows = 300;
        const int cols = 1024;
        const int count = 8;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 97_500, vectors: count);

        using CpuBackend vnniPinned = new(threads: 1, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: true), tuning: new CpuKernelTuningOverrides { Q4KRepackedResidencyBatchWidth = 8 });
        Assert.Equal(IntegerDotPath.Vnni, vnniPinned.ResolvedIntegerDotPath); // a stated width is resolved...

        float[] outputs = new float[count * rows];

        using CpuBackend avx2Owner = OpenAvx2Only();
        using Q4KRepackedMatrix preparedElsewhere = avx2Owner.PrepareDenseQ4KResidency(new WeightRef(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K), weightsBudgetCapMegabytes: 1024);
        Assert.ThrowsAny<Exception>(() => vnniPinned.GemvBatch(inputs, count, new WeightRef(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K), preparedElsewhere, outputs)); // ...but only the AVX2 rung may read a copy back.
    }

    // ---------------------------------------------------------------- determinism contract and default-posture preservation

    [Theory]
    [MemberData(nameof(MatrixSizesData))]
    public void GemvBatchRepacked_AtEveryCount_IsBitIdenticalToItselfAcrossGroupingsAndThreads(int rows, int cols)
    {
        foreach (int count in RepackedCounts)
        {
            (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 101_000 + count, vectors: count);
            WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

            // Reference arm: the default resolved group width (8) with a stated residency crossover.
            using CpuBackend reference = OpenAvx2Only();
            Assert.Equal(IntegerDotPath.Avx2, reference.ResolvedIntegerDotPath); // the posture this variant is stated for, forced so it holds on any machine

            using Q4KRepackedMatrix repacked = reference.PrepareDenseQ4KResidency(weight, weightsBudgetCapMegabytes: 1024);
            float[] expected = new float[count * rows];
            reference.GemvBatch(inputs, count, weight, repacked, expected);

            // Ragged chain grouping: every stated group splits the batch into consecutive groups plus a tail — same chains, same order, identical bits whatever width walked them.
            foreach (int group in new[] { 2, 5 })
            {
                using CpuBackend grouped = new(threads: 1, cpuFeatures: Avx2Only, tuning: new CpuKernelTuningOverrides { Q4KRepackedResidencyBatchWidth = 8, ChainGroup = group });
                using Q4KRepackedMatrix groupedCopy = grouped.PrepareDenseQ4KResidency(weight, weightsBudgetCapMegabytes: 1024);

                float[] actualGrouped = new float[count * rows];
                grouped.GemvBatch(inputs, count, weight, groupedCopy, actualGrouped);

                BatchKernelTests.AssertBitsEqual(expected, actualGrouped, $"repacked Q4_K read at {rows}x{cols} count {count} under chain group {group}");
            }

            // Threaded arm: forced tiny row blocks hand every row to a worker — the same per-row walks of one shared copy, partitioned.
            using CpuBackend threaded = new(threads: 8, cpuFeatures: Avx2Only, tuning: new CpuKernelTuningOverrides { Q4KRepackedResidencyBatchWidth = 8, RowBlock = 1, ParallelRowThreshold = 1 });
            using Q4KRepackedMatrix threadedCopy = threaded.PrepareDenseQ4KResidency(weight, weightsBudgetCapMegabytes: 1024);

            float[] actualThreaded = new float[count * rows];
            threaded.GemvBatch(inputs, count, weight, threadedCopy, actualThreaded);

            BatchKernelTests.AssertBitsEqual(expected, actualThreaded, $"threaded repacked Q4_K read at {rows}x{cols} count {count}");
        }
    }

    // When activations quantize exactly — constant vectors whose staged quants dequantize losslessly — the resident read must land on what the float
    // kernels compute for the same dot within pure summation-order rounding: this is the numeric contract that separates real arithmetic from
    // activation-quantization loss. (Against ordinary seeded activations the two traversals differ by quantization noise, which is expected and not asserted here.)
    [Theory]
    [MemberData(nameof(MatrixSizesData))]
    public void GemvBatchRepacked_ExactQuantizedActivations_MatchFloatRouteWithinSummationRounding(int rows, int cols)
    {
        const int count = 16;

        (_, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 102_000 + cols, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        List<float> inputList = [];
        for (int v = 0; v < count; v++)
        {
            float constant = v % 2 == 0 ? 0.5f : -0.25f; // each vector is one constant: its quantization is lossless, dequantization exact.

            inputList.AddRange(Enumerable.Repeat(constant, cols));
        }

        float[] inputs = [.. inputList];

        using CpuBackend floatRoute = new(threads: 1, cpuFeatures: Avx2Only); // no width stated anywhere: today's traversal.
        Assert.Equal(IntegerDotPath.Avx2, floatRoute.ResolvedIntegerDotPath);
        float[] expected = new float[count * rows];
        floatRoute.GemvBatch(inputs, count, weight, expected);

        using CpuBackend residentRead = OpenAvx2Only();
        using Q4KRepackedMatrix repacked = residentRead.PrepareDenseQ4KResidency(weight, weightsBudgetCapMegabytes: 1024);
        float[] actual = new float[count * rows];
        residentRead.GemvBatch(inputs, count, weight, repacked, actual);

        double maxAbsDiff = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            // Bit-equality is possible here and not required: with exact activation quantization both traversals sum the same terms, so agreement to summation rounding — not divergence per element — is what this contract states.
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs((double)actual[i] - expected[i]));
        }

        double scaleRef = Math.Max(1, Enumerable.Range(0, expected.Length).Max(i => Math.Abs(expected[i])));
        Assert.True(maxAbsDiff <= 2e-3 * scaleRef, $"repacked Q4_K read at {rows}x{cols} with exact activation quantization drifted {maxAbsDiff:E4} absolute against the float route (values up to {scaleRef:F1})");
    }

    // Default posture stays exactly today's: an auto open resolves no residency width, and a backend holding a prepared copy still runs its ordinary
    // batched passes on their current traversals — byte-for-byte identical to a backend that never prepared one. Nothing about preparation reaches a call that does not name the copy.
    [Fact]
    public void AutoPosture_ResolvesNoResidencyWidth_AndPlainGemvBatchIgnoresAPreparedCopy_BitForBit()
    {
        Assert.Equal(0, CpuKernelTuning.Resolve(32, vectorPosture: true, CpuKernelTuningOverrides.Auto).Q4KRepackedResidencyBatchWidth); // no auto posture states a width for it...
        Assert.Equal(0, CpuKernelTuning.Resolve(32, vectorPosture: false, CpuKernelTuningOverrides.Auto).Q4KRepackedResidencyBatchWidth); // ...and without a vector posture nothing runs anyway.

        const int rows = 300;
        const int cols = 1024;
        const int count = 16;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 103_000, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend reference = new(threads: 1, cpuFeatures: Avx2Only); // no seam stated at all: today's posture.
        float[] expected = new float[count * rows];
        reference.GemvBatch(inputs, count, weight, expected);

        using CpuBackend preparedButUnstated = OpenAvx2Only(repackedWidth: 0); // a copy resident in the backend...
        using Q4KRepackedMatrix repacked = preparedButUnstated.PrepareDenseQ4KResidency(weight, weightsBudgetCapMegabytes: 1024);

        float[] actual = new float[count * rows];
        preparedButUnstated.GemvBatch(inputs, count, weight, actual); // ...through the ordinary overload changes nothing.

        BatchKernelTests.AssertBitsEqual(expected, actual, "a resident copy beside an unstated pass leaves today's traversal in force");
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(65)] // past the batch ceiling every dense kernel's scratch is sized for
    public void Resolve_Q4KRepackedResidencyPinOutsideItsSpan_RefusesNamingQualifiedKeyAndValue(int value)
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(
            () => CpuKernelTuning.Resolve(8, vectorPosture: true, new CpuKernelTuningOverrides { Q4KRepackedResidencyBatchWidth = value }));

        Assert.Equal("engine.cpu.tuning.q4k_repacked_residency_batch_width", error.ParamName);
        Assert.Contains($"engine.cpu.tuning.q4k_repacked_residency_batch_width {value}", error.Message);
    }

    [Fact]
    public void Resolve_Q4KRepackedResidencyPinResolvesExactly()
    {
        foreach (int width in new[] { 0, 1, 8, 64 })
        {
            CpuKernelTuning tuning = CpuKernelTuning.Resolve(32, vectorPosture: true, new CpuKernelTuningOverrides { Q4KRepackedResidencyBatchWidth = width });

            Assert.Equal(width, tuning.Q4KRepackedResidencyBatchWidth);
        }
    }

    // The seam is its own gate: a stated residency crossover says nothing about the landed integer-dot executor or any float route, and a stated
    // Q6_K crossover never opens a resident read. Each seam selects only what it was named for.
    [Fact]
    public void GemvBatch_PinsOfOtherSeams_KeepFloatTraversal_AtStatedResidencyWidth()
    {
        const int rows = 300;
        const int cols = 1024;
        const int count = 16;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 104_000, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend reference = new(threads: 1, cpuFeatures: Avx2Only);
        float[] expected = new float[count * rows];
        reference.GemvBatch(inputs, count, weight, expected);

        using CpuBackend residencyPinnedPlainCall = OpenAvx2Only(); // crossover stated, no copy named: the ordinary overload stays on its measured traversal.
        float[] actual = new float[count * rows];
        residencyPinnedPlainCall.GemvBatch(inputs, count, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, "a stated residency width alone opens no route an ordinary call did not already run");
    }

    // The engine-open seam is separate from preparation and selection: an ordinary dense batched pass reads a prepared copy only when this backend was explicitly engaged, even though the crossover itself may already be stated for measurement. Default posture stays byte-identical; engagement changes only which overload's route runs.
    [Fact]
    public void EngineOpenMeasurementEngagement_OrdinaryGemvBatchReadsPreparedCopy_BitIdenticalToExplicitSeam()
    {
        const int rows = 300;
        const int cols = 1024;
        const int count = 16;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 105_000, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend backend = OpenAvx2Only(); // crossover stated, but engagement is not.
        Assert.False(backend.Q4KMeasurementEngaged);

        using Q4KRepackedMatrix copy = backend.PrepareDenseQ4KResidency(weight, weightsBudgetCapMegabytes: 1024);

        float[] explicitRead = new float[count * rows];
        backend.GemvBatch(inputs, count, weight, copy, explicitRead); // the named-copy seam answers first.

        backend.EnableQ4KRepackedResidencyMeasurements(1024); // engagement states no allocation: it only admits ordinary reads of copies already prepared here.
        Assert.True(backend.Q4KMeasurementEngaged);

        float[] engagedOrdinary = new float[count * rows];
        backend.GemvBatch(inputs, count, weight, engagedOrdinary); // the same call shape now resolves to that exact copy through lookup by source identity.

        BatchKernelTests.AssertBitsEqual(explicitRead, engagedOrdinary, "an engine-open repacked read must answer exactly what its explicit seam answers");
    }

    [Fact]
    public void StatedResidencyWidthAndPreparedCopy_WithoutEngagement_KeepsFloatTraversal_BitForBit()
    {
        const int rows = 300;
        const int cols = 1024;
        const int count = 16;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 106_000, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend reference = new(threads: 1, cpuFeatures: Avx2Only); // today's posture.
        float[] expected = new float[count * rows];
        reference.GemvBatch(inputs, count, weight, expected);

        using CpuBackend statedButUnengaged = OpenAvx2Only(); // crossover stated and a copy prepared...
        Assert.False(statedButUnengaged.Q4KMeasurementEngaged);
        using Q4KRepackedMatrix copy = statedButUnengaged.PrepareDenseQ4KResidency(weight, weightsBudgetCapMegabytes: 1024);

        float[] actual = new float[count * rows];
        statedButUnengaged.GemvBatch(inputs, count, weight, actual); // ...but without the explicit engine-open engagement call today's route stays in force.

        BatchKernelTests.AssertBitsEqual(expected, actual, "a prepared copy beside an unengaged backend must not change ordinary dense passes");
    }

    [Fact]
    public void EnableQ4KRepackedResidencyMeasurements_OnVnniRung_RefusesNamingMissingAvx2ReadPath()
    {
        using CpuBackend vnni = new(threads: 1, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: true), tuning: new CpuKernelTuningOverrides { Q4KRepackedResidencyBatchWidth = 8 });

        Q4KRepackedResidencyException error = Assert.Throws<Q4KRepackedResidencyException>(() => vnni.EnableQ4KRepackedResidencyMeasurements(1024));
        Assert.Contains("AVX2 integer-dot rung", error.Message); // a VNNI-resolved backend has no resident AVX2 copy to read back, so engagement refuses rather than opening some other kernel.
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
}
