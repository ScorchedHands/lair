using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Bit-parity tests for the shared-operand dot batch: every output of a batched FP32 dot against one common vector must land on the exact
/// bits its own individual <see cref="IBackend.Dot"/> call produces, at any chain-group width and worker posture (batching covers independent
/// accumulator chains only), in both postures — AVX2 chunk-and-tail and scalar fallback. The causal attention kernels are then compared
/// route-off against route-on on synthetic fixtures across row counts, grouped-query widths, windows that leave partial blocks at either edge,
/// softcap on and off, ragged head dimensions, and groups wider than one batch: batching changes only how a score is produced, so every output
/// must match bit for bit — no tolerance anywhere in this suite.
/// </summary>
[Trait("Category", "Attention")]
public sealed class AttentionSharedDotTests
{
    // Ragged common lengths: exact eight-lane groups, pure scalar tails, mixed tails, and the production head dimension.
    private static readonly int[] CommonLengths = [8, 9, 31, 32, 64, 127, 128];

    // Batch counts that straddle one chain group: a single dot, just over one group, exactly one group, and several groups.
    private static readonly int[] BatchCounts = [2, 3, 4, 5, 8, 9, 16, 20, 32];

    // Seeds for the deterministic fixture values: magnitudes with signs and fractional parts in play so a bit comparison is meaningful.
    private static readonly int[] Seeds = [17, 31, 4_201_955];

    // Head dimensions that straddle the eight-lane boundary: pure tail, exact groups, mixed tails, and the production width.
    private static readonly int[] HeadDims = [1, 2, 7, 8, 9, 13, 32, 64, 128];

    // ---------------------------------------------------------------- backend dot parity

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void DotBatchSharedInput_MatchesIndividualDots_BitForBit(bool vectorPosture)
    {
        foreach (int commonLength in CommonLengths)
        {
            float[] common = Build(commonLength, Seeds[1]);

            foreach (int count in BatchCounts)
            {
                // A scalar posture resolves the shared-dot route off by default, so the parity case pins it on to prove the batch kernel's fallback lands on DotScalar exactly.
                using CpuBackend backend = new(
                    threads: 1,
                    allowAvx2: vectorPosture,
                    tuning: vectorPosture ? null : new CpuKernelTuningOverrides { SharedInputDotChainGroup = CpuKernelTuning.SharedInputDotChainGroupMax });

                float[] inputs = Build(count * commonLength, Seeds[0]);
                float[] batched = new float[count];

                Assert.True(
                    ((IDenseSharedInputDot)backend).TryDotBatchSharedInput(inputs, count, common, batched),
                    $"batch={count} common length={commonLength} vectorPosture={vectorPosture}: a stated chain group must run the shared-dot route.");

                float[] expected = new float[count];
                for (int i = 0; i < count; i++)
                {
                    expected[i] = backend.Dot(inputs.AsSpan(i * commonLength, commonLength), common);
                }

                AssertBitsEqual(expected, batched, $"batch={count} common length={commonLength} vectorPosture={vectorPosture}");
            }
        }
    }

    [Theory]
    [InlineData(1)] // one chain per pass: the shared-dot route degrades to per-chain walks rather than refusing
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)] // a full group against an eight-head batch splits into consecutive groups of it
    public void DotBatchSharedInput_ChainGroupWidthChangesOnlyDispatch_BitForBit(int chainGroup)
    {
        const int CommonLength = 128; // headDim-shaped, so the chunk walk covers whole groups and the tail is empty

        foreach (int count in new[] { 2, 7, 9, 16 })
        {
            float[] common = Build(CommonLength, Seeds[1]);

            using CpuBackend reference = new(threads: 1, allowAvx2: true);
            float[] inputs = Build(count * CommonLength, Seeds[0]);

            float[] expected = new float[count];
            for (int i = 0; i < count; i++)
            {
                expected[i] = reference.Dot(inputs.AsSpan(i * CommonLength, CommonLength), common);
            }

            foreach (int pinned in new[] { chainGroup, Math.Min(chainGroup + 1, CpuKernelTuning.SharedInputDotChainGroupMax) })
            {
                using CpuBackend batched = new(threads: 4, allowAvx2: true, tuning: new CpuKernelTuningOverrides { SharedInputDotChainGroup = pinned });

                float[] outputs = new float[count];
                Assert.True(((IDenseSharedInputDot)batched).TryDotBatchSharedInput(inputs, count, common, outputs));

                AssertBitsEqual(expected, outputs, $"chain group={pinned} batch={count} against individual dots");
            }
        }
    }

    [Fact]
    public void DotBatchSharedInput_RouteOffKeepsIndividualDotsAndTouchesNoOutputs()
    {
        float[] untouched = [7.5f, -3.25f];

        using CpuBackend off = new(threads: 1, allowAvx2: true, tuning: new CpuKernelTuningOverrides { SharedInputDotChainGroup = 0 });

        Assert.False(((IDenseSharedInputDot)off).TryDotBatchSharedInput(Build(8, Seeds[0]), 1, Build(4, Seeds[1]), untouched)); // a single dot has no operand to share with anything
        Assert.Equal(new float[] { 7.5f, -3.25f }, untouched);

        Assert.False(((IDenseSharedInputDot)off).TryDotBatchSharedInput(Build(8, Seeds[0]), 2, Build(4, Seeds[1]), untouched)); // the pinned posture keeps shared-operand dots individual
        Assert.Equal(new float[] { 7.5f, -3.25f }, untouched);

        using CpuBackend scalarPosture = new(threads: 1, allowAvx2: false); // auto resolves to off without a vector posture
        Assert.False(((IDenseSharedInputDot)scalarPosture).TryDotBatchSharedInput(Build(8, Seeds[0]), 4, Build(4, Seeds[1]), untouched));

        using CpuBackend scalarPinned = new(threads: 1, allowAvx2: false, tuning: new CpuKernelTuningOverrides { SharedInputDotChainGroup = 4 });
        float[] scalarInputs = Build(16, Seeds[0]);
        float[] scalarCommon = Build(4, Seeds[1]);
        float[] scalarOutputs = new float[4];

        Assert.True(((IDenseSharedInputDot)scalarPinned).TryDotBatchSharedInput(scalarInputs, 4, scalarCommon, scalarOutputs)); // a pinned group on a scalar host still runs, one reference dot per output

        float[] scalarExpected = new float[4];
        for (int i = 0; i < 4; i++)
        {
            scalarExpected[i] = scalarPinned.Dot(scalarInputs.AsSpan(i * 4, 4), scalarCommon);
        }

        AssertBitsEqual(scalarExpected, scalarOutputs, "scalar-posture shared-dot batch against its own reference dots");

        using CpuBackend threaded = new(threads: 8, allowAvx2: true, tuning: new CpuKernelTuningOverrides { SharedInputDotChainGroup = 4 });
        float[] threadInputs = Build(32, Seeds[0]);
        float[] threadCommon = Build(4, Seeds[1]);

        float[] threadedOutputs = new float[8];
        Assert.True(((IDenseSharedInputDot)threaded).TryDotBatchSharedInput(threadInputs, 8, threadCommon, threadedOutputs));

        using CpuBackend singleThread = new(threads: 1, allowAvx2: true);
        float[] singleOutputs = new float[8];
        Assert.True(((IDenseSharedInputDot)singleThread).TryDotBatchSharedInput(threadInputs, 8, threadCommon, singleOutputs));

        AssertBitsEqual(singleOutputs, threadedOutputs, "threaded shared-dot batch against the same call on one worker");
    }

    [Fact]
    public void DotBatchSharedInput_OutsideItsContract_ThrowsOrRefuses()
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        Assert.Throws<ArgumentException>(() => ((IDenseSharedInputDot)backend).TryDotBatchSharedInput(new float[3], 2, [1f, 2f, 3f], new float[2])); // inputs must cover every vector of the batch
        Assert.Throws<ArgumentException>(() => ((IDenseSharedInputDot)backend).TryDotBatchSharedInput(Build(6, Seeds[0]), 2, [1f, 2f, 3f], new float[1])); // one output slot per dotted input vector

        Assert.False(((IDenseSharedInputDot)backend).TryDotBatchSharedInput(new float[8], 4, [], new float[4])); // a zero-wide shared operand is the caller's own degenerate case
    }

    [Fact]
    public void Resolve_SharedInputDotChainGroupFollowsTheVectorPosture()
    {
        Assert.Equal(CpuKernelTuning.ChainGroupMax, CpuKernelTuning.Resolve(32, vectorPosture: true, CpuKernelTuningOverrides.Auto).SharedInputDotChainGroup); // a full chain group shares one pass over the shared operand
        Assert.Equal(0, CpuKernelTuning.Resolve(32, vectorPosture: false, CpuKernelTuningOverrides.Auto).SharedInputDotChainGroup); // without a vector posture every such dot stays individual

        using CpuBackend vectorized = new(threads: 4, allowAvx2: true);
        Assert.Equal(CpuKernelTuning.ChainGroupMax, vectorized.ResolvedTuning.SharedInputDotChainGroup);

        using CpuBackend scalar = new(threads: 4, allowAvx2: false);
        Assert.Equal(0, scalar.ResolvedTuning.SharedInputDotChainGroup);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(9)] // wider than the vector posture's chain-group register budget
    public void Resolve_SharedInputDotChainGroupPinOutsideItsSpan_RefusesNamingQualifiedKeyAndValue(int value)
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(
            () => CpuKernelTuning.Resolve(8, vectorPosture: true, new CpuKernelTuningOverrides { SharedInputDotChainGroup = value }));

        Assert.Equal("engine.cpu.tuning.shared_input_dot_chain_group", error.ParamName);
        Assert.Contains($"engine.cpu.tuning.shared_input_dot_chain_group {value}", error.Message);
    }

    // ---------------------------------------------------------------- causal attention route parity

    [Theory]
    [InlineData(1, 2, 5)] // one head per kv-head group: the seam's own floor leaves nothing batched
    [InlineData(4, 2, 8)] // two heads per group
    [InlineData(6, 3, 9)] // ragged groups under a non-dividing grouped-query mapping
    [InlineData(16, 2, 7)] // eight-head production-shaped groups over a window that leaves partial blocks at both edges
    public void CausalAttention_RouteOn_MatchesRouteOffBitForBit(int headCount, int kvHeadCount, int positions)
    {
        foreach (int headDim in HeadDims)
        {
            foreach ((float softcap, int window) in new[] { (0f, 0), (4f, 3), (0f, 5), (8.5f, 2) })
            {
                using CpuBackend off = new(threads: 1, allowAvx2: true, tuning: new CpuKernelTuningOverrides { SharedInputDotChainGroup = 0 });
                using CpuBackend on = new(threads: 4, allowAvx2: true); // the posture-resolved full chain group

                using AttentionFixture fixture = AttentionFixture.Build(headCount, kvHeadCount, headDim, positions, rows: 1);
                int region = headCount * headDim;

                float[] outOff = new float[region];
                LayerOps.CausalAttention(off, fixture.QRows.AsSpan(0, region), headCount, kvHeadCount, headDim, fixture.Kv, 0, positions - 1, 0.125f, softcap, window, outOff);

                float[] outOn = new float[region];
                LayerOps.CausalAttention(on, fixture.QRows.AsSpan(0, region), headCount, kvHeadCount, headDim, fixture.Kv, 0, positions - 1, 0.125f, softcap, window, outOn);

                AssertBitsEqual(outOff, outOn, $"CausalAttention route parity heads={headCount}/{kvHeadCount} headDim={headDim} softcap={softcap} window={window}");
            }
        }
    }

    [Theory]
    [InlineData(16, 2)] // eight-head groups walked as consecutive narrower batches
    [InlineData(4, 3)] // a mapping where one group is shorter than the batch cap
    public void CausalAttention_GroupWiderThanOneBatch_MatchesRouteOffBitForBit(int headCount, int kvHeadCount)
    {
        const int HeadDim = 13; // ragged: the shared-operand walk carries a scalar tail at every position
        const int Positions = 7;

        using CpuBackend off = new(threads: 1, allowAvx2: true, tuning: new CpuKernelTuningOverrides { SharedInputDotChainGroup = 0 });
        using CpuBackend narrow = new(threads: 4, allowAvx2: true, tuning: new CpuKernelTuningOverrides { SharedInputDotChainGroup = 3 });

        using AttentionFixture fixture = AttentionFixture.Build(headCount, kvHeadCount, HeadDim, Positions, rows: 1);
        int region = headCount * HeadDim;

        float[] outOff = new float[region];
        LayerOps.CausalAttention(off, fixture.QRows.AsSpan(0, region), headCount, kvHeadCount, HeadDim, fixture.Kv, 0, Positions - 1, 0.125f, 4f, 6, outOff);

        float[] outNarrow = new float[region];
        LayerOps.CausalAttention(narrow, fixture.QRows.AsSpan(0, region), headCount, kvHeadCount, HeadDim, fixture.Kv, 0, Positions - 1, 0.125f, 4f, 6, outNarrow);

        AssertBitsEqual(outOff, outNarrow, $"CausalAttention against consecutive narrower shared-dot batches at heads={headCount}/{kvHeadCount}");
    }

    [Theory]
    [InlineData(3)] // a single row's worth of sub-batch
    [InlineData(5)]
    [InlineData(17)] // more rows than one chain group: every position's K read serves many rows and heads at once
    public void CausalAttentionTiledRows_RouteOn_MatchesRouteOffBitForBit(int rows)
    {
        foreach (int headDim in HeadDims)
        {
            foreach ((float softcap, int window) in new[] { (0f, 0), (4f, 5), (16f, 3), (0f, 7) })
            {
                const int HeadCount = 8;
                const int KvHeadCount = 2; // four-head groups
                const int Positions = 21;

                using CpuBackend off = new(threads: 1, allowAvx2: true, tuning: new CpuKernelTuningOverrides { SharedInputDotChainGroup = 0 });
                using CpuBackend on = new(threads: 4, allowAvx2: true);

                using AttentionFixture fixture = AttentionFixture.Build(HeadCount, KvHeadCount, headDim, Positions, rows);
                int region = HeadCount * headDim;

                float[] outOff = new float[rows * region];
                LayerOps.CausalAttentionTiledRows(off, fixture.QRows, 4, rows, HeadCount, KvHeadCount, headDim, fixture.Kv, 0, 0.125f, softcap, window, new float[rows * HeadCount], new float[rows * HeadCount], outOff);

                float[] outOn = new float[rows * region];
                LayerOps.CausalAttentionTiledRows(on, fixture.QRows, 4, rows, HeadCount, KvHeadCount, headDim, fixture.Kv, 0, 0.125f, softcap, window, new float[rows * HeadCount], new float[rows * HeadCount], outOn);

                AssertBitsEqual(outOff, outOn, $"tiled route parity rows={rows} headDim={headDim} softcap={softcap} window={window}");
            }
        }
    }

    [Theory]
    [InlineData(6)] // a sub-batch whose selection crosses the block boundary at both edges
    [InlineData(13)]
    public void CausalAttentionTiledRows_WindowedRaggedRange_MatchesPerRowKernelAndRouteOffBitForBit(int rows)
    {
        const int HeadCount = 8;
        const int KvHeadCount = 2;
        const int HeadDim = 96; // whole eight-lane groups at a production-shaped width
        const int Positions = 34;

        using CpuBackend off = new(threads: 1, allowAvx2: true, tuning: new CpuKernelTuningOverrides { SharedInputDotChainGroup = 0 });
        using CpuBackend on = new(threads: 4, allowAvx2: true);

        using AttentionFixture fixture = AttentionFixture.Build(HeadCount, KvHeadCount, HeadDim, Positions, rows);
        int region = HeadCount * HeadDim;

        foreach (bool routeOn in new[] { false, true })
        {
            CpuBackend backend = routeOn ? on : off;

            float[] tiledOut = new float[rows * region];
            LayerOps.CausalAttentionTiledRows(backend, fixture.QRows, 9, rows, HeadCount, KvHeadCount, HeadDim, fixture.Kv, 0, 0.125f, 4f, 6, new float[rows * HeadCount], new float[rows * HeadCount], tiledOut);

            // The same equality the engine's prefill identity suite relies on: each tiled row lands bit-for-bit on that row's own causal pass.
            for (int r = 0; r < rows; r++)
            {
                float[] perRowOut = new float[region];
                LayerOps.CausalAttention(backend, fixture.QRows.AsSpan(r * region, region), HeadCount, KvHeadCount, HeadDim, fixture.Kv, 0, 9 + r, 0.125f, 4f, 6, perRowOut);

                AssertBitsEqual(
                    perRowOut,
                    tiledOut.AsSpan(r * region, region),
                    $"tiled row {r} against its own causal pass with the shared-dot route {(routeOn ? "on" : "off")}");
            }
        }
    }

    // ---------------------------------------------------------------- fixtures and helpers

    private sealed class AttentionFixture : IDisposable
    {
        private readonly CpuBackend _backend;

        public BlockPool Pool { get; }
        public SequenceKv Kv { get; }
        public float[] QRows { get; }

        private AttentionFixture(CpuBackend backend, BlockPool pool, SequenceKv kv, float[] qRows)
        {
            _backend = backend;
            Pool = pool;
            Kv = kv;
            QRows = qRows;
        }

        // Synthetic queries plus written K/V for every position, on a four-token block table so partial blocks sit at both edges of any selection.
        public static AttentionFixture Build(int headCount, int kvHeadCount, int headDim, int positions, int rows)
        {
            CpuBackend backend = new(threads: 1, allowAvx2: true);

            BlockPool pool = new(blockTokens: 4, layerCount: 1, kvHeads: kvHeadCount, headDim, capacityBlocks: Math.Max(8, positions / 4 + 4), backend);
            SequenceKv kv = new(pool);

            for (int p = 0; p < positions; p++)
            {
                float[] kVector = AttentionSharedDotTests.Build(kvHeadCount * headDim, Seeds[2] + p);
                float[] vVector = AttentionSharedDotTests.Build(kvHeadCount * headDim, Seeds[2] + p + 977);

                kv.Append();
                kv.WriteK(p, 0, kVector);
                kv.WriteV(p, 0, vVector);
            }

            return new AttentionFixture(backend, pool, kv, AttentionSharedDotTests.Build(rows * headCount * headDim, Seeds[0] + positions));
        }

        public void Dispose()
        {
            Kv.Dispose();
            Pool.Dispose();
            _backend.Dispose();
        }
    }

    // Deterministic pseudo-random magnitudes with signs and fractional parts in play: the same fixture values every run, so a parity failure is reproducible.
    private static float[] Build(int length, int seed)
    {
        float[] values = new float[length];
        uint state = (uint)seed | 1u;

        for (int i = 0; i < length; i++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;

            values[i] = ((state & 0xFFFF) / 8_192f - 4f) * (i % 7 == 0 ? -1.0000001f : 1.0000001f);
        }

        return values;
    }

    private static void AssertBitsEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (int i = 0; i < expected.Length; i++)
        {
            uint expectedBits = BitsOf(expected[i]);
            uint actualBits = BitsOf(actual[i]);

            Assert.True(expectedBits == actualBits, $"{context}: element {i} differs — expected bits 0x{expectedBits:X8}, got 0x{actualBits:X8}.");
        }
    }

    private static uint BitsOf(float value) => BitConverter.SingleToUInt32Bits(value);
}
