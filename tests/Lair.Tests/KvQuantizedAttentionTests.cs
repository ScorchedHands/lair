using System.Diagnostics;
using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// In-kernel quantized-dot CPU attention over the committed q8_0/q4_0 block storage: single-row decode parity of the
/// generic quantized attention against a naive dequantized reference (including zero-scale groups and negative values),
/// exact match between the scalar and vector integer-dot encodings, windowed causal reads that never address positions
/// outside the selection, prepared-query reuse that leaks no state between heads or calls, q4 nibble dequantization
/// faithful to the committed writer, and loose agreement with the production FP32 attention on the same synthetic values.
/// </summary>
[Trait("Category", "Attention")]
[Trait("Category", "Kv")]
public class KvQuantizedAttentionTests
{
    // One synthetic geometry shared by every case: 4 query heads over 2 kv-heads (a 2:1 grouped-query mapping), 64-dim
    // heads so each head region covers exactly two whole 32-value groups, and a 4-token block size so the sequence
    // crosses block boundaries and exercises partial blocks at both edges.
    private const int BlockTokens = 4;
    private const int Layers = 1;
    private const int HeadCount = 4;
    private const int KvHeadCount = 2;
    private const int HeadDim = 64;
    private const int KvDim = KvHeadCount * HeadDim;

    private const int Seed = 20_261_118;

    // Tolerances: the quantized kernel and the dequantized reference see the same stored values, so their only differences
    // are the query-side Q8_K quantization and accumulation grouping. Against the production FP32 attention the storage
    // quantization itself enters, so those bounds are looser and sized to the format's own group scale.
    private const float Q8ReferenceTolerance = 0.01f;
    private const float K8V4ReferenceTolerance = 0.01f;
    private const float Q8FloatParityTolerance = 0.05f;
    private const float K8V4FloatParityTolerance = 0.30f;

    // ---- q8_0/q8_0 parity against the naive dequantized reference ----

    [Fact]
    public void Q8Q8Attention_MatchesNaiveDequantizedReference()
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        IntegerDotPath dot = backend.ResolvedIntegerDotPath;

        Fixture fixture = Fixture.Build(Seed);

        using BlockPool<Q8Block, Q8Block> pool = new(BlockTokens, Layers, KvHeadCount, HeadDim, capacityBlocks: 4, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0));
        using SequenceKv<Q8Block, Q8Block> kv = new(pool);
        FillQuantized(kv, fixture);

        byte[] scratch = new byte[Q8Block.DotScratchBytesPerHead(HeadDim) * HeadCount];
        float[] scales = new float[Q8Block.DotScaleCount(HeadDim) * HeadCount];
        float[] outConcat = new float[HeadCount * HeadDim];

        LayerOps.CausalAttention<Q8Block, Q8Block>(
            fixture.QHeads, HeadCount, KvHeadCount, HeadDim, kv, layer: 0, position: fixture.Positions - 1,
            fixture.AttnScale, softcap: 0f, window: 0, dot, scratch, scales, outConcat);

        float[] reference = new float[HeadCount * HeadDim];
        NaiveAttention(fixture.QHeads, DequantAll<Q8Block>(fixture.KValues, DequantQ8), DequantAll<Q8Block>(fixture.VValues, DequantQ8),
            fixture.Positions - 1, fixture.AttnScale, softcap: 0f, window: 0, reference);

        AssertWithin(reference, outConcat, Q8ReferenceTolerance);
        Assert.All(outConcat, v => Assert.True(float.IsFinite(v)));
    }

    // ---- q8_0 K / q4_0 V parity against the naive dequantized reference ----

    [Fact]
    public void K8V4Attention_MatchesNaiveDequantizedReference()
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        IntegerDotPath dot = backend.ResolvedIntegerDotPath;

        Fixture fixture = Fixture.Build(Seed);

        using BlockPool<Q8Block, Q4Block> pool = new(BlockTokens, Layers, KvHeadCount, HeadDim, capacityBlocks: 4, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q4_0));
        using SequenceKv<Q8Block, Q4Block> kv = new(pool);
        FillQuantized(kv, fixture);

        byte[] scratch = new byte[Q8Block.DotScratchBytesPerHead(HeadDim) * HeadCount];
        float[] scales = new float[Q8Block.DotScaleCount(HeadDim) * HeadCount];
        float[] outConcat = new float[HeadCount * HeadDim];

        LayerOps.CausalAttention<Q8Block, Q4Block>(
            fixture.QHeads, HeadCount, KvHeadCount, HeadDim, kv, layer: 0, position: fixture.Positions - 1,
            fixture.AttnScale, softcap: 0f, window: 0, dot, scratch, scales, outConcat);

        float[] reference = new float[HeadCount * HeadDim];
        NaiveAttention(fixture.QHeads, DequantAll<Q8Block>(fixture.KValues, DequantQ8), DequantAll<Q4Block>(fixture.VValues, DequantQ4),
            fixture.Positions - 1, fixture.AttnScale, softcap: 0f, window: 0, reference);

        AssertWithin(reference, outConcat, K8V4ReferenceTolerance);
        Assert.All(outConcat, v => Assert.True(float.IsFinite(v)));
    }

    // ---- window + softcap semantics, and boundedness against poisoned out-of-window storage ----

    [Fact]
    public void Q8Q8Attention_WindowAndSoftcap_MatchReference_AndNeverReadOutsideSelection()
    {
        const int Window = 3;
        const float Softcap = 30f;
        int queryPosition = 6; // positions 0..2 are outside the window; they get poisoned after the reference values are recorded

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        IntegerDotPath dot = backend.ResolvedIntegerDotPath;

        Fixture fixture = Fixture.Build(Seed);

        using BlockPool<Q8Block, Q8Block> pool = new(BlockTokens, Layers, KvHeadCount, HeadDim, capacityBlocks: 4, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0));
        using SequenceKv<Q8Block, Q8Block> kv = new(pool);
        FillQuantized(kv, fixture);

        // Poison every out-of-window position with huge magnitudes: any read that reached them would move the output far
        // beyond the reference, so a match proves the selection bounded the walk rather than a loop bound merely skipping them.
        for (int pos = 0; pos < queryPosition - Window + 1; pos++)
        {
            kv.WriteK(pos, 0, Filled(KvDim, 4096f));
            kv.WriteV(pos, 0, Filled(KvDim, -4096f));
        }

        byte[] scratch = new byte[Q8Block.DotScratchBytesPerHead(HeadDim) * HeadCount];
        float[] scales = new float[Q8Block.DotScaleCount(HeadDim) * HeadCount];
        float[] outConcat = new float[HeadCount * HeadDim];

        LayerOps.CausalAttention<Q8Block, Q8Block>(
            fixture.QHeads, HeadCount, KvHeadCount, HeadDim, kv, layer: 0, position: queryPosition,
            fixture.AttnScale, Softcap, Window, dot, scratch, scales, outConcat);

        // The reference runs over the same windowed range with the original (unpoisoned) values, applying the softcap
        // after scaling and before the online max, exactly as the float kernel documents.
        float[] reference = new float[HeadCount * HeadDim];
        NaiveAttention(fixture.QHeads, DequantAll<Q8Block>(fixture.KValues, DequantQ8), DequantAll<Q8Block>(fixture.VValues, DequantQ8),
            queryPosition, fixture.AttnScale, Softcap, Window, reference);

        AssertWithin(reference, outConcat, Q8ReferenceTolerance);
    }

    // ---- scalar and vector integer-dot encodings land on identical outputs ----

    [Fact]
    public void Q8Q8Attention_ScalarDotEncoding_MatchesVectorDotEncoding_BitForBit()
    {
        using CpuBackend vectorBackend = new(threads: 1, allowAvx2: true);
        using CpuBackend scalarBackend = new(threads: 1, allowAvx2: false);
        Assert.Equal(IntegerDotPath.Scalar, scalarBackend.ResolvedIntegerDotPath); // the no-vector posture forces the reference dot

        Fixture fixture = Fixture.Build(Seed);

        using BlockPool<Q8Block, Q8Block> pool = new(BlockTokens, Layers, KvHeadCount, HeadDim, capacityBlocks: 4, vectorBackend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0));
        using SequenceKv<Q8Block, Q8Block> kv = new(pool);
        FillQuantized(kv, fixture);

        byte[] scratch = new byte[Q8Block.DotScratchBytesPerHead(HeadDim) * HeadCount];
        float[] scales = new float[Q8Block.DotScaleCount(HeadDim) * HeadCount];
        float[] vectorOut = new float[HeadCount * HeadDim];
        float[] scalarOut = new float[HeadCount * HeadDim];

        LayerOps.CausalAttention<Q8Block, Q8Block>(
            fixture.QHeads, HeadCount, KvHeadCount, HeadDim, kv, layer: 0, position: fixture.Positions - 1,
            fixture.AttnScale, softcap: 0f, window: 0, vectorBackend.ResolvedIntegerDotPath, scratch, scales, vectorOut);

        LayerOps.CausalAttention<Q8Block, Q8Block>(
            fixture.QHeads, HeadCount, KvHeadCount, HeadDim, kv, layer: 0, position: fixture.Positions - 1,
            fixture.AttnScale, softcap: 0f, window: 0, scalarBackend.ResolvedIntegerDotPath, scratch, scales, scalarOut);

        Assert.Equal(vectorOut, scalarOut); // exact int32 sub-block totals under the same scales: the encoding cannot move a bit
    }

    // ---- prepared-query reuse across calls leaks no state between heads or calls ----

    [Fact]
    public void Q8Q8Attention_ReusedQueryScratch_ProducesBitIdenticalResultsAcrossCalls_AndHeadsStayIndependent()
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        IntegerDotPath dot = backend.ResolvedIntegerDotPath;

        Fixture fixture = Fixture.Build(Seed);

        using BlockPool<Q8Block, Q8Block> pool = new(BlockTokens, Layers, KvHeadCount, HeadDim, capacityBlocks: 4, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0));
        using SequenceKv<Q8Block, Q8Block> kv = new(pool);
        FillQuantized(kv, fixture);

        byte[] scratch = new byte[Q8Block.DotScratchBytesPerHead(HeadDim) * HeadCount];
        float[] scales = new float[Q8Block.DotScaleCount(HeadDim) * HeadCount];

        float[] firstOut = new float[HeadCount * HeadDim];
        LayerOps.CausalAttention<Q8Block, Q8Block>(
            fixture.QHeads, HeadCount, KvHeadCount, HeadDim, kv, layer: 0, position: fixture.Positions - 1,
            fixture.AttnScale, softcap: 0f, window: 0, dot, scratch, scales, firstOut);

        // A different query set through the same buffers must not inherit residue from the first preparation.
        float[] swappedQ = (float[])fixture.QHeads.Clone();
        for (int j = 0; j < HeadDim; j++)
        {
            (swappedQ[j], swappedQ[HeadDim + j]) = (swappedQ[HeadDim + j], swappedQ[j]); // swap head 0's whole region with head 1's
        }

        float[] swappedOut = new float[HeadCount * HeadDim];
        LayerOps.CausalAttention<Q8Block, Q8Block>(
            swappedQ, HeadCount, KvHeadCount, HeadDim, kv, layer: 0, position: fixture.Positions - 1,
            fixture.AttnScale, softcap: 0f, window: 0, dot, scratch, scales, swappedOut);

        // Head independence: swapping two heads' queries must swap exactly those two heads' outputs, bit-for-bit.
        Assert.Equal(firstOut.AsSpan(0, HeadDim).ToArray(), swappedOut.AsSpan(HeadDim, HeadDim).ToArray());
        Assert.Equal(firstOut.AsSpan(HeadDim, HeadDim).ToArray(), swappedOut.AsSpan(0, HeadDim).ToArray());
        for (int h = 2; h < HeadCount; h++)
        {
            Assert.Equal(
                firstOut.AsSpan(h * HeadDim, HeadDim).ToArray(),
                swappedOut.AsSpan(h * HeadDim, HeadDim).ToArray());
        }

        // Re-running the original queries through the same buffers reproduces the first run exactly.
        float[] rereadOut = new float[HeadCount * HeadDim];
        LayerOps.CausalAttention<Q8Block, Q8Block>(
            fixture.QHeads, HeadCount, KvHeadCount, HeadDim, kv, layer: 0, position: fixture.Positions - 1,
            fixture.AttnScale, softcap: 0f, window: 0, dot, scratch, scales, rereadOut);

        Assert.Equal(firstOut, rereadOut);
    }

    // ---- zero-scale groups score and contribute exactly zero ----

    [Fact]
    public void Q8Q8Attention_ZeroScaleGroups_ScoreAndContributeExactlyZero()
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        IntegerDotPath dot = backend.ResolvedIntegerDotPath;

        Fixture fixture = Fixture.Build(Seed);
        fixture.KValues[1] = new float[KvDim]; // position 1's whole K region is zero: its group scales land on zero
        fixture.VValues[2] = new float[KvDim]; // position 2's whole V region is zero: it contributes nothing

        using BlockPool<Q8Block, Q8Block> pool = new(BlockTokens, Layers, KvHeadCount, HeadDim, capacityBlocks: 4, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0));
        using SequenceKv<Q8Block, Q8Block> kv = new(pool);
        FillQuantized(kv, fixture);

        byte[] scratch = new byte[Q8Block.DotScratchBytesPerHead(HeadDim) * HeadCount];
        float[] scales = new float[Q8Block.DotScaleCount(HeadDim) * HeadCount];
        float[] outConcat = new float[HeadCount * HeadDim];

        LayerOps.CausalAttention<Q8Block, Q8Block>(
            fixture.QHeads, HeadCount, KvHeadCount, HeadDim, kv, layer: 0, position: fixture.Positions - 1,
            fixture.AttnScale, softcap: 0f, window: 0, dot, scratch, scales, outConcat);

        float[] reference = new float[HeadCount * HeadDim];
        NaiveAttention(fixture.QHeads, DequantAll<Q8Block>(fixture.KValues, DequantQ8), DequantAll<Q8Block>(fixture.VValues, DequantQ8),
            fixture.Positions - 1, fixture.AttnScale, softcap: 0f, window: 0, reference);

        AssertWithin(reference, outConcat, Q8ReferenceTolerance);
    }

    // ---- q4_0 nibble dequantization is faithful to the committed writer ----

    [Fact]
    public void Q4Dequant_RoundTripsWriterValues_WithinGroupScaleGranularity()
    {
        Random rng = new(Seed);
        float[] values = new float[Q4Block.Values];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)(rng.NextDouble() * 3.0 - 1.5);
        }

        byte[] arena = new byte[Q4Block.BlockBytes];
        Q4Block.WriteRegion(values, System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Q4Block>(arena.AsSpan()));

        float[] dequantized = DequantQ4(System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Q4Block>(arena.AsSpan()));

        float amax = values.Max(Math.Abs);
        for (int i = 0; i < values.Length; i++)
        {
            // q4_0 granularity: a value lands on the nearest of 16 codes under d = max / -8, so the round trip may drift
            // by at most one code step — the group scale — from the original.
            Assert.True(
                Math.Abs(dequantized[i] - values[i]) <= amax / 8f + 1e-6f,
                $"q4 round trip drifted further than one code step at element {i}: {values[i]} -> {dequantized[i]}");
        }

        // The all-zero group keeps its code-8 through-zero-scale posture: every decoded value is exactly zero.
        byte[] zeroArena = new byte[Q4Block.BlockBytes];
        Q4Block.WriteRegion(new float[Q4Block.Values], System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Q4Block>(zeroArena.AsSpan()));
        Assert.All(DequantQ4(System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Q4Block>(zeroArena.AsSpan())), v => Assert.Equal(0f, v));
    }

    // ---- generic selection mirrors the non-generic selection and bounds its reads ----

    [Fact]
    public void GenericSelection_MirrorsNonGenericSelection_AndReadsOutsideTheSelectionRefuse()
    {
        const int Positions = 12; // three 4-token blocks

        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        using BlockPool<Q8Block, Q4Block> quantPool = new(BlockTokens, Layers, KvHeadCount, HeadDim, capacityBlocks: 4, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q4_0));
        using SequenceKv<Q8Block, Q4Block> quantKv = new(quantPool);
        for (int pos = 0; pos < Positions; pos++)
        {
            quantKv.Append();
            quantKv.WriteK(pos, 0, Filled(KvDim, 1f));
            quantKv.WriteV(pos, 0, Filled(KvDim, 1f));
        }

        using BlockPool floatPool = new(BlockTokens, Layers, KvHeadCount, HeadDim, capacityBlocks: 4, backend);
        using SequenceKv floatKv = new(floatPool);
        for (int pos = 0; pos < Positions; pos++)
        {
            floatKv.Append();
            floatKv.WriteK(pos, 0, Filled(KvDim, 1f));
            floatKv.WriteV(pos, 0, Filled(KvDim, 1f));
        }

        BlockSelection quantCausal = quantKv.SelectCausalRead(queryPosition: 10, layerWindow: 5); // [6..10]: blocks 1..2, block 0 skipped
        BlockSelection floatCausal = floatKv.SelectCausalRead(queryPosition: 10, layerWindow: 5);
        Assert.Equal(floatCausal.FirstPosition, quantCausal.FirstPosition);
        Assert.Equal(floatCausal.LastPosition, quantCausal.LastPosition);
        Assert.Equal(floatCausal.FirstBlockIndex, quantCausal.FirstBlockIndex);
        Assert.Equal(floatCausal.SelectedBlocks, quantCausal.SelectedBlocks);
        Assert.Equal(floatCausal.TotalBlocks, quantCausal.TotalBlocks);
        Assert.Equal(1, quantCausal.SkippedBlocks);
        Assert.False(quantCausal.CoversWholeTable);

        BlockSelection quantRanged = quantKv.SelectRangeRead(1, 5);
        BlockSelection floatRanged = floatKv.SelectRangeRead(1, 5);
        Assert.Equal(floatRanged.FirstBlockIndex, quantRanged.FirstBlockIndex);
        Assert.Equal(floatRanged.SelectedBlocks, quantRanged.SelectedBlocks);

        // A read outside the selection is a failure, not a silent read: the windowed pass cannot reach past its window.
        ArgumentOutOfRangeException outsideK = Assert.Throws<ArgumentOutOfRangeException>(() => quantKv.ReadK(quantCausal, logicalPos: 5, layer: 0));
        Assert.Contains("outside the selected read", outsideK.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentOutOfRangeException>(() => quantKv.ReadV(quantCausal, logicalPos: 11, layer: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => quantKv.SelectCausalRead(queryPosition: 12, layerWindow: 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => quantKv.SelectRangeRead(0, 12));

        // Inside the selection the reads address the backing blocks normally.
        Assert.Equal(4, quantKv.ReadK(quantCausal, logicalPos: 6, layer: 0).Length); // KvDim 128 over q8_0 groups = 4 blocks
        Assert.Equal(4, quantKv.ReadV(quantCausal, logicalPos: 10, layer: 0).Length); // KvDim 128 over q4_0 groups = 4 blocks
    }

    // ---- loose agreement with the production FP32 attention on identical synthetic values ----

    [Fact]
    public void QuantizedAttention_ApproximatesProductionFloatAttention_WithinStorageQuantizationError()
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        IntegerDotPath dot = backend.ResolvedIntegerDotPath;

        Fixture fixture = Fixture.Build(Seed);

        using BlockPool floatPool = new(BlockTokens, Layers, KvHeadCount, HeadDim, capacityBlocks: 4, backend);
        using SequenceKv floatKv = new(floatPool);
        for (int pos = 0; pos < fixture.Positions; pos++)
        {
            floatKv.Append();
            floatKv.WriteK(pos, 0, fixture.KValues[pos]);
            floatKv.WriteV(pos, 0, fixture.VValues[pos]);
        }

        float[] floatOut = new float[HeadCount * HeadDim];
        LayerOps.CausalAttention(
            backend, fixture.QHeads, HeadCount, KvHeadCount, HeadDim, floatKv, layer: 0, position: fixture.Positions - 1,
            fixture.AttnScale, softcap: 0f, window: 0, floatOut);

        byte[] scratch = new byte[Q8Block.DotScratchBytesPerHead(HeadDim) * HeadCount];
        float[] scales = new float[Q8Block.DotScaleCount(HeadDim) * HeadCount];

        using (BlockPool<Q8Block, Q8Block> q8Pool = new(BlockTokens, Layers, KvHeadCount, HeadDim, capacityBlocks: 4, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0)))
        using (SequenceKv<Q8Block, Q8Block> q8Kv = new(q8Pool))
        {
            FillQuantized(q8Kv, fixture);
            float[] q8Out = new float[HeadCount * HeadDim];
            LayerOps.CausalAttention<Q8Block, Q8Block>(
                fixture.QHeads, HeadCount, KvHeadCount, HeadDim, q8Kv, layer: 0, position: fixture.Positions - 1,
                fixture.AttnScale, softcap: 0f, window: 0, dot, scratch, scales, q8Out);
            AssertWithin(floatOut, q8Out, Q8FloatParityTolerance);
        }

        using (BlockPool<Q8Block, Q4Block> k8v4Pool = new(BlockTokens, Layers, KvHeadCount, HeadDim, capacityBlocks: 4, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q4_0)))
        using (SequenceKv<Q8Block, Q4Block> k8v4Kv = new(k8v4Pool))
        {
            FillQuantized(k8v4Kv, fixture);
            float[] k8v4Out = new float[HeadCount * HeadDim];
            LayerOps.CausalAttention<Q8Block, Q4Block>(
                fixture.QHeads, HeadCount, KvHeadCount, HeadDim, k8v4Kv, layer: 0, position: fixture.Positions - 1,
                fixture.AttnScale, softcap: 0f, window: 0, dot, scratch, scales, k8v4Out);
            AssertWithin(floatOut, k8v4Out, K8V4FloatParityTolerance);
        }
    }

    // ---- speed sanity against the production FP32 attention: reported only, never asserted ----

    [Fact]
    public void QuantizedAttention_SpeedSanityAgainstFloatAttention_ReportsWithoutGating()
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        IntegerDotPath dot = backend.ResolvedIntegerDotPath;

        Fixture fixture = Fixture.Build(Seed);
        const int Walks = 200;

        using BlockPool floatPool = new(BlockTokens, Layers, KvHeadCount, HeadDim, capacityBlocks: 4, backend);
        using SequenceKv floatKv = new(floatPool);
        for (int pos = 0; pos < fixture.Positions; pos++)
        {
            floatKv.Append();
            floatKv.WriteK(pos, 0, fixture.KValues[pos]);
            floatKv.WriteV(pos, 0, fixture.VValues[pos]);
        }

        float[] floatOut = new float[HeadCount * HeadDim];
        for (int warm = 0; warm < 20; warm++)
        {
            LayerOps.CausalAttention(backend, fixture.QHeads, HeadCount, KvHeadCount, HeadDim, floatKv, 0, fixture.Positions - 1, fixture.AttnScale, 0f, 0, floatOut);
        }

        Stopwatch floatWatch = Stopwatch.StartNew();
        for (int i = 0; i < Walks; i++)
        {
            LayerOps.CausalAttention(backend, fixture.QHeads, HeadCount, KvHeadCount, HeadDim, floatKv, 0, fixture.Positions - 1, fixture.AttnScale, 0f, 0, floatOut);
        }
        double floatUs = floatWatch.Elapsed.TotalMilliseconds * 1000.0 / Walks;

        byte[] scratch = new byte[Q8Block.DotScratchBytesPerHead(HeadDim) * HeadCount];
        float[] scales = new float[Q8Block.DotScaleCount(HeadDim) * HeadCount];

        using BlockPool<Q8Block, Q8Block> q8Pool = new(BlockTokens, Layers, KvHeadCount, HeadDim, capacityBlocks: 4, backend, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0));
        using SequenceKv<Q8Block, Q8Block> q8Kv = new(q8Pool);
        FillQuantized(q8Kv, fixture);

        float[] q8Out = new float[HeadCount * HeadDim];
        for (int warm = 0; warm < 20; warm++)
        {
            LayerOps.CausalAttention<Q8Block, Q8Block>(fixture.QHeads, HeadCount, KvHeadCount, HeadDim, q8Kv, 0, fixture.Positions - 1, fixture.AttnScale, 0f, 0, dot, scratch, scales, q8Out);
        }

        Stopwatch quantWatch = Stopwatch.StartNew();
        for (int i = 0; i < Walks; i++)
        {
            LayerOps.CausalAttention<Q8Block, Q8Block>(fixture.QHeads, HeadCount, KvHeadCount, HeadDim, q8Kv, 0, fixture.Positions - 1, fixture.AttnScale, 0f, 0, dot, scratch, scales, q8Out);
        }
        double quantUs = quantWatch.Elapsed.TotalMilliseconds * 1000.0 / Walks;

        // Sanity only: a quantized attention that is pathologically slower than the float path would show up here,
        // but no timing value gates this test.
        Assert.True(quantUs < floatUs * 20.0, $"quantized attention per-call {quantUs:F2}us vs float {floatUs:F2}us");
    }

    // ---- fixture and reference helpers ----

    private sealed class Fixture
    {
        public int Positions { get; private set; }
        public float[] QHeads { get; private set; } = [];
        public float[][] KValues { get; private set; } = [];
        public float[][] VValues { get; private set; } = [];
        public float AttnScale { get; private set; }

        public static Fixture Build(int seed)
        {
            Random rng = new(seed);
            Fixture fixture = new()
            {
                Positions = 7, // two blocks, second partially filled: partial blocks at both edges
                QHeads = new float[HeadCount * HeadDim],
                AttnScale = 1f / MathF.Sqrt(HeadDim),
            };

            for (int i = 0; i < fixture.QHeads.Length; i++)
            {
                fixture.QHeads[i] = (float)(rng.NextDouble() * 3.0 - 1.5);
            }

            fixture.KValues = new float[fixture.Positions][];
            fixture.VValues = new float[fixture.Positions][];
            for (int pos = 0; pos < fixture.Positions; pos++)
            {
                fixture.KValues[pos] = new float[KvDim];
                fixture.VValues[pos] = new float[KvDim];
                for (int i = 0; i < KvDim; i++)
                {
                    fixture.KValues[pos][i] = (float)(rng.NextDouble() * 3.0 - 1.5);
                    fixture.VValues[pos][i] = (float)(rng.NextDouble() * 3.0 - 1.5);
                }

                // Deliberate exact zeros and negatives: zero magnitudes must land on zero scales, and negative values
                // must survive the signed-byte and nibble encodings.
                fixture.KValues[pos][0] = -fixture.KValues[pos][0];
                fixture.KValues[pos][1] = 0f;
                fixture.VValues[pos][2] = 0f;
                fixture.VValues[pos][31] = -fixture.VValues[pos][31];
            }

            return fixture;
        }
    }

    private static void FillQuantized<TK, TV>(SequenceKv<TK, TV> kv, Fixture fixture)
        where TK : struct, IKvFormat<TK>
        where TV : struct, IKvFormat<TV>
    {
        for (int pos = 0; pos < fixture.Positions; pos++)
        {
            kv.Append();
            kv.WriteK(pos, 0, fixture.KValues[pos]);
            kv.WriteV(pos, 0, fixture.VValues[pos]);
        }
    }

    private static float[][] DequantAll<T>(float[][] storedValues, Func<ReadOnlySpan<T>, float[]> dequant)
        where T : struct, IKvFormat<T>
    {
        // The reference sees exactly what the pool stored: re-quantizing the fixture values through the committed
        // writer and decoding them back is what both the kernel and the reference attend over.
        float[][] result = new float[storedValues.Length][];
        for (int pos = 0; pos < storedValues.Length; pos++)
        {
            byte[] arena = new byte[storedValues[pos].Length / 32 * SizeOfBlock<T>()];
            WriteRegionOf<T>(storedValues[pos], arena);
            result[pos] = dequant(System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(arena.AsSpan()));
        }

        return result;
    }

    private static int SizeOfBlock<T>() where T : struct
        => System.Runtime.CompilerServices.Unsafe.SizeOf<T>();

    private static void WriteRegionOf<T>(float[] values, byte[] arena) where T : struct, IKvFormat<T>
    {
        T.WriteRegion(values, System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(arena.AsSpan()));
    }

    private static unsafe float[] DequantQ8(ReadOnlySpan<Q8Block> blocks)
    {
        float[] values = new float[blocks.Length * Q8Block.Values];
        for (int b = 0; b < blocks.Length; b++)
        {
            ref readonly Q8Block block = ref System.Runtime.CompilerServices.Unsafe.Add(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(blocks), b);
            float d = (float)BitConverter.UInt16BitsToHalf(block.ScaleBits);
            for (int i = 0; i < Q8Block.Values; i++)
            {
                values[b * Q8Block.Values + i] = (float)block.Quants[i] * d;
            }
        }

        return values;
    }

    private static unsafe float[] DequantQ4(ReadOnlySpan<Q4Block> blocks)
    {
        float[] values = new float[blocks.Length * Q4Block.Values];
        for (int b = 0; b < blocks.Length; b++)
        {
            ref readonly Q4Block block = ref System.Runtime.CompilerServices.Unsafe.Add(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(blocks), b);
            float d = (float)BitConverter.UInt16BitsToHalf(block.ScaleBits);
            for (int j = 0; j < Q4Block.Values / 2; j++)
            {
                byte packed = block.Nibbles[j];
                values[b * Q4Block.Values + j] = ((packed & 0x0F) - 8) * d;
                values[b * Q4Block.Values + Q4Block.Values / 2 + j] = ((packed >> 4) - 8) * d;
            }
        }

        return values;
    }

    /// <summary>
    /// The online-softmax semantics of the float kernel — grouped-query mapping, scaled-then-capped scores, the
    /// rescale-on-new-max rule, and the final 1f / sumExp normalization — run over dequantized stored values with the
    /// FP32 queries, one position at a time in ascending order.
    /// </summary>
    private static void NaiveAttention(
        float[] qHeads, float[][] kByPos, float[][] vByPos, int position, float attnScale, float softcap, int window, float[] outConcat)
    {
        int firstVisible = window > 0 ? Math.Max(0, position - window + 1) : 0;

        for (int h = 0; h < HeadCount; h++)
        {
            int kvHead = h * KvHeadCount / HeadCount;
            float runningMax = float.NegativeInfinity;
            float sumExp = 0f;
            Span<float> outHead = outConcat.AsSpan(h * HeadDim, HeadDim);
            outHead.Clear();

            for (int t = firstVisible; t <= position; t++)
            {
                float score = 0f;
                for (int j = 0; j < HeadDim; j++)
                {
                    score += kByPos[t][kvHead * HeadDim + j] * qHeads[h * HeadDim + j];
                }

                score *= attnScale;
                if (softcap > 0f)
                {
                    score = (float)(softcap * Math.Tanh((double)score / softcap));
                }

                if (score > runningMax)
                {
                    if (!float.IsNegativeInfinity(runningMax))
                    {
                        float rescale = MathF.Exp(runningMax - score);
                        sumExp *= rescale;
                        for (int j = 0; j < HeadDim; j++)
                        {
                            outHead[j] *= rescale;
                        }
                    }

                    runningMax = score;
                }

                float weight = MathF.Exp(score - runningMax);
                sumExp += weight;

                for (int j = 0; j < HeadDim; j++)
                {
                    outHead[j] += weight * vByPos[t][kvHead * HeadDim + j];
                }
            }

            float invSum = 1f / sumExp;
            for (int j = 0; j < HeadDim; j++)
            {
                outHead[j] *= invSum;
            }
        }
    }

    private static void AssertWithin(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);

        float maxAbs = 0f;
        int worstIndex = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = Math.Abs(expected[i] - actual[i]);
            if (diff > maxAbs)
            {
                maxAbs = diff;
                worstIndex = i;
            }
        }

        Assert.True(
            maxAbs <= tolerance,
            $"max abs diff {maxAbs:F6} at element {worstIndex} (reference {expected[worstIndex]:F6}, actual {actual[worstIndex]:F6}) exceeds tolerance {tolerance:F6}");
    }

    private static float[] Filled(int count, float value)
    {
        float[] values = new float[count];
        Array.Fill(values, value);
        return values;
    }
}
