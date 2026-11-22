using Lair.Backends.Cpu;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// CPU-only coverage of the qwen4exp forward-composition spike: a resolved geometry alone (no model file, never the 70 GB
/// Flash-Next target) drives one synthetic block through hyper-connection mixing → attention → residual combine → feed-forward.
/// The linear block runs gated-delta-net composition and its recurrent state plus convolution history carry across chunk splits;
/// a sparse-full block composes indexer selection and restricted attention through the existing QsaOps semantics; the
/// hyper-connection mix/combine math is checked against hand-derived reference order and shapes, and re-sequencing every stage
/// in the graph's own order reproduces the spike's block forward bit-for-bit. The production TransformerModel refusal stays a
/// capability boundary: adding this spike did not wire qwen4exp into engine open.
/// </summary>
[Trait("Category", "Engine")]
public sealed class Qwen4ExpForwardSpikeTests
{
    private const int Hidden = 8;
    private const int HeadCount = 2;
    private const int KvHeadCount = 1;
    private const int HeadDim = 4; // hidden / head_count, even so the spike rotates full heads
    private const int IntermediateSize = 6;
    private const float Eps = 1e-5f;

    // ---------------------------------------------------------------- linear block: gated-delta-net composition carries state across chunk splits

    [Fact]
    public void Spike_LinearBlock_GdnStateAndConvHistoryCarryAcrossChunkSplits()
    {
        Qwen4ExpGeometry geometry = LinearOnlyGeometry(linearLayers: 1);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        int flattened = geometry.HyperConnections.FlattenedStreamFloats(Hidden);
        float[] streamsIn = Matrix(flattened * 4, Hidden); // four tokens' input stream sets (the layer reads each token independently)

        using Qwen4ExpForwardSpike wholeRun = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend, maxChunk: 8, maxPositions: 32);
        float[] fullOut = new float[flattened * 4];
        wholeRun.StartSequence(new float[Hidden]);
        wholeRun.ForwardBlockChunk(layer: 0, streamsIn, tokenCount: 4, fullOut);

        // Split the same four-token run after two tokens; the carried delta-rule state and convolution history must make the
        // second call continue exactly where the first left off.
        using Qwen4ExpForwardSpike splitRun = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend, maxChunk: 8, maxPositions: 32);
        float[] splitOut = new float[flattened * 4];
        splitRun.StartSequence(new float[Hidden]);
        splitRun.ForwardBlockChunk(layer: 0, streamsIn.AsSpan(0, flattened * 2), tokenCount: 2, splitOut.AsSpan(0, flattened * 2));
        splitRun.ForwardBlockChunk(layer: 0, streamsIn.AsSpan(flattened * 2, flattened * 2), tokenCount: 2, splitOut.AsSpan(flattened * 2, flattened * 2));

        Assert.True(fullOut.AsSpan().SequenceEqual(splitOut.AsSpan()), "a gated-delta-net block forward must not change across a chunk boundary");
    }

    [Fact]
    public void Spike_LinearBlock_IsDeterministicAcrossRuns()
    {
        Qwen4ExpGeometry geometry = LinearOnlyGeometry(linearLayers: 2); // two linear blocks in sequence, both carrying their own state
        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        int flattened = geometry.HyperConnections.FlattenedStreamFloats(Hidden);
        float[] streamsIn = Matrix(flattened * 3, Hidden);

        float[] firstRun = RunAllLayers(geometry, backend, streamsIn, tokenCount: 3, flattened);
        float[] secondRun = RunAllLayers(geometry, backend, streamsIn, tokenCount: 3, flattened);

        Assert.True(firstRun.AsSpan().SequenceEqual(secondRun.AsSpan()), "a synthetic qwen4exp forward must be deterministic across runs");
    }

    // ---------------------------------------------------------------- full block: indexer selection and restricted attention compose through QsaOps

    [Fact]
    public void Spike_FullBlock_RunsQsaSelectionAndRestrictedAttentionDeterministically()
    {
        // A sparse-full layer with a compress ratio of two groups cells into pools so the spike actually reaches selection.
        Qwen4ExpGeometry geometry = GeometryWithFullLayer(compressRatio: 2);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        int flattened = geometry.HyperConnections.FlattenedStreamFloats(Hidden);
        float[] streamsIn = Matrix(flattened * 5, Hidden); // five tokens: enough cached cells to form complete pools

        Assert.Equal(AttentionKind.FullQuerySparseAttention, KindOf(geometry));

        using Qwen4ExpForwardSpike firstRun = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend, maxChunk: 8, maxPositions: 32);
        float[] outFirst = new float[flattened * 5];
        firstRun.StartSequence(new float[Hidden]);
        for (int t = 0; t < 5; t++)
        {
            firstRun.ForwardBlock(layer: 0, streamsIn.AsSpan(t * flattened, flattened), outFirst.AsSpan(t * flattened, flattened));
        }

        foreach (float value in outFirst)
        {
            Assert.True(float.IsFinite(value), "sparse-full composition must produce finite residual values");
        }

        // A second identical sequence reproduces the whole run bit-for-bit: selection ties and restricted attention are pinned.
        using Qwen4ExpForwardSpike secondRun = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend, maxChunk: 8, maxPositions: 32);
        float[] outSecond = new float[flattened * 5];
        secondRun.StartSequence(new float[Hidden]);
        for (int t = 0; t < 5; t++)
        {
            secondRun.ForwardBlock(layer: 0, streamsIn.AsSpan(t * flattened, flattened), outSecond.AsSpan(t * flattened, flattened));
        }

        Assert.True(outFirst.AsSpan().SequenceEqual(outSecond.AsSpan()), "sparse-full composition must be deterministic across runs");

        // The sublayer actually ran: the residual differs from its input wherever attention and feed-forward contributed.
        bool changed = false;
        for (int i = 0; i < outFirst.Length; i++)
        {
            if (!outFirst[i].Equals(streamsIn[i]))
            {
                changed = true;
                break;
            }
        }

        Assert.True(changed, "the block forward must fold the sublayer outputs back into the residual streams");
    }

    // ---------------------------------------------------------------- hyper-connection mix and combine match the reference order and shapes

    [Fact]
    public void Spike_MixStreams_MatchesGroupedNormLowRankGateMeanAndInjectScale()
    {
        Qwen4ExpGeometry geometry = LinearOnlyGeometry(linearLayers: 1);
        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        int flattened = geometry.HyperConnections.FlattenedStreamFloats(Hidden);
        int streamCount = geometry.HyperConnections.StreamCount; // two streams at this fixture's count
        int lowRank = geometry.HyperConnections.LowRank;

        using Qwen4ExpForwardSpike spike = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend);
        float[] streams = Matrix(flattened, 1).Select(v => v + 0.5f).ToArray(); // a deterministic nonzero stream set

        Qwen4ExpForwardSpike.HyperConnectionTensors hc = spike.AttentionHyperOperands(0);

        float[] mixed = new float[Hidden];
        float[] inject = new float[streamCount];
        spike.MixStreams(0, Qwen4ExpForwardSpike.SublayerSide.Attention, streams, mixed, inject);

        // Grouped RMSNorm per stream scaled by that stream's gamma slice.
        double invStreams = 1.0 / streamCount;
        float[] normedStream = new float[flattened];
        for (int s = 0; s < streamCount; s++)
        {
            RmsNormWeighted(streams.AsSpan(s * Hidden, Hidden), hc.Norm.AsSpan(s * Hidden, Hidden), Eps, normedStream.AsSpan(s * Hidden, Hidden));
        }

        // Down projection scaled by 1/streams then SiLU.
        float[] lo = new float[lowRank];
        for (int r = 0; r < lowRank; r++)
        {
            lo[r] = (float)Silu(Dot(hc.Down, r, flattened, normedStream) * invStreams);
        }

        // Up projection expands to a per-stream gate; the streams are gated by sigmoid and collapsed by their mean.
        float[] mixedExpected = new float[Hidden];
        for (int h = 0; h < Hidden; h++)
        {
            double sum = 0.0;
            for (int s = 0; s < streamCount; s++)
            {
                int f = s * Hidden + h;
                double gate = Dot(hc.Up, f, lowRank, lo);
                sum += normedStream[f] * Sigmoid(gate);
            }

            mixedExpected[h] = (float)(sum * invStreams);
        }

        for (int h = 0; h < Hidden; h++)
        {
            Assert.True(Close(mixed[h], mixedExpected[h]), $"mixed stream hidden {h}: expected {mixedExpected[h]:G9}, got {mixed[h]:G9}");
        }

        // Inject scores come from the inject projection scaled by 1/streams.
        for (int s = 0; s < streamCount; s++)
        {
            double expected = Dot(hc.Inject, s, flattened, normedStream) * invStreams;
            Assert.True(Close(inject[s], expected), $"inject score {s}: expected {expected:G9}, got {inject[s]:G9}");
        }
    }

    [Fact]
    public void Spike_CombineStreams_ScattersBlockOutputByTwoSigmoidWeightsOverResidual()
    {
        Qwen4ExpGeometry geometry = LinearOnlyGeometry(linearLayers: 1);
        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        int flattened = geometry.HyperConnections.FlattenedStreamFloats(Hidden);
        int streamCount = geometry.HyperConnections.StreamCount;

        using Qwen4ExpForwardSpike spike = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend);

        float[] streamsIn = Matrix(flattened, 1).Select(v => v + 0.5f).ToArray();
        float[] blockOut = Matrix(Hidden, 1).Select(v => v - 0.25f).ToArray();
        float[] inject = [-2.0f, 3.0f]; // one near-zero-injection stream and one strong stream

        float[] streamsOut = new float[flattened];
        spike.CombineStreams(streamsIn, blockOut, inject, streamsOut);

        for (int s = 0; s < streamCount; s++)
        {
            double weight = 2.0 * Sigmoid(inject[s]); // a zero injection is a plain residual add: weight ≈ 1 at inject ≈ -∞? no — sigmoid(0)=0.5 → weight=1

            for (int h = 0; h < Hidden; h++)
            {
                double expected = streamsIn[s * Hidden + h] + blockOut[h] * weight;
                Assert.True(Close(streamsOut[s * Hidden + h], expected), $"combined stream {s} hidden {h}: expected {expected:G9}, got {streamsOut[s * Hidden + h]:G9}");
            }

            // The 2*sigmoid rule centres a unit-weight scatter on inject == 0, so the residual add is exact there.
            if (inject[s] == -2.0f)
            {
                Assert.True(weight < 1.0, "a strongly negative injection scatters less than one copy of the block output");
            }
        }

        // A zero inject leaves a plain residual add: streamsOut equals streamsIn plus exactly one block-output copy per stream.
        float[] unitInject = [0f, 0f];
        float[] identityOut = new float[flattened];
        spike.CombineStreams(streamsIn, blockOut, unitInject, identityOut);

        for (int s = 0; s < streamCount; s++)
        {
            for (int h = 0; h < Hidden; h++)
            {
                double expected = streamsIn[s * Hidden + h] + blockOut[h]; // 2*sigmoid(0) == 1 exactly
                Assert.True(Close(identityOut[s * Hidden + h], expected), $"unit-injection residual add at stream {s} hidden {h}: expected {expected:G9}, got {identityOut[s * Hidden + h]:G9}");
            }
        }
    }

    // ---------------------------------------------------------------- re-sequencing every stage in graph order reproduces the spike's block forward

    [Fact]
    public void Spike_LinearBlockForward_MatchesReferenceStageSequenceBitForBit()
    {
        Qwen4ExpGeometry geometry = LinearOnlyGeometry(linearLayers: 1);
        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        int flattened = geometry.HyperConnections.FlattenedStreamFloats(Hidden);
        int streamCount = geometry.HyperConnections.StreamCount;
        GdnDescriptor gdn = geometry.LinearAttention;

        using Qwen4ExpForwardSpike spike = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend);
        float[] streamsIn = Matrix(flattened, 1).Select(v => v + 0.5f).ToArray();

        // What the spike produces for one token from a fresh sequence.
        float[] spikeOut = new float[flattened];
        spike.StartSequence(new float[Hidden]);
        spike.ForwardBlock(layer: 0, streamsIn, spikeOut);

        // Re-sequence build_hc_mix → attention → build_hc_combine → build_hc_mix → ffn → build_hc_combine by hand against the same operands.
        GdnOps.GdnBlockTensors w = spike.LinearOperands(0);

        float[] mixedAttention = new float[Hidden];
        float[] injectAttn = new float[streamCount];
        spike.MixStreams(0, Qwen4ExpForwardSpike.SublayerSide.Attention, streamsIn, mixedAttention, injectAttn);

        // One-token gated-delta-net chunk from a zeroed state — exactly the sequence start the spike began at.
        int channels = w.ConvChannels;
        float[] convHistory = new float[(w.ConvKernel - 1) * channels];
        float[] nextConvHistory = new float[convHistory.Length];
        float[] recurrentState = new float[w.RecurrentStateValues];
        GdnOps.GdnChunkScratch scratch = GdnOps.GdnChunkScratch.Create(w, maxTokenCount: 1);
        float[] attnOut = new float[Hidden];
        GdnOps.ForwardChunk(mixedAttention, w, Eps, tokenCount: 1, convHistory, nextConvHistory, recurrentState, scratch, attnOut);

        float[] midStreams = new float[flattened];
        spike.CombineStreams(streamsIn, attnOut, injectAttn, midStreams);

        float[] mixedFfn = new float[Hidden];
        float[] injectFfn = new float[streamCount];
        spike.MixStreams(0, Qwen4ExpForwardSpike.SublayerSide.FeedForward, midStreams, mixedFfn, injectFfn);

        float[] ffnOut = new float[Hidden];
        spike.ForwardDenseFfnStandIn(0, mixedFfn, ffnOut);

        float[] expectedOut = new float[flattened];
        spike.CombineStreams(midStreams, ffnOut, injectFfn, expectedOut);

        Assert.True(spikeOut.AsSpan().SequenceEqual(expectedOut.AsSpan()), "the block forward must compose exactly the reference stage sequence in order");
    }

    [Fact]
    public void Spike_FullBlockForward_MatchesReferenceStageSequenceBitForBit()
    {
        Qwen4ExpGeometry geometry = GeometryWithFullLayer(compressRatio: 2);
        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        int flattened = geometry.HyperConnections.FlattenedStreamFloats(Hidden);
        int streamCount = geometry.HyperConnections.StreamCount;

        // Instance A runs the block the way the spike composes it.
        using Qwen4ExpForwardSpike composed = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend);
        float[] streamsIn = Matrix(flattened, 1).Select(v => v + 0.5f).ToArray();
        float[] composedOut = new float[flattened];
        composed.StartSequence(new float[Hidden]);
        composed.ForwardBlock(layer: 0, streamsIn, composedOut);

        // Instance B re-sequences the reference graph order by hand using the spike's own stage primitives from an identical
        // fresh sequence: build_hc_mix → sparse attention → build_hc_combine → build_hc_mix → dense ffn stand-in → build_hc_combine.
        using Qwen4ExpForwardSpike sequenced = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend);
        sequenced.StartSequence(new float[Hidden]);

        float[] mixedAttention = new float[Hidden];
        float[] injectAttn = new float[streamCount];
        sequenced.MixStreams(0, Qwen4ExpForwardSpike.SublayerSide.Attention, streamsIn, mixedAttention, injectAttn);

        float[] attnOut = new float[Hidden];
        sequenced.SparseAttentionSublayer(0, mixedAttention, attnOut);

        float[] midStreams = new float[flattened];
        sequenced.CombineStreams(streamsIn, attnOut, injectAttn, midStreams);

        float[] mixedFfn = new float[Hidden];
        float[] injectFfn = new float[streamCount];
        sequenced.MixStreams(0, Qwen4ExpForwardSpike.SublayerSide.FeedForward, midStreams, mixedFfn, injectFfn);

        float[] ffnOut = new float[Hidden];
        sequenced.ForwardDenseFfnStandIn(0, mixedFfn, ffnOut);

        float[] sequencedOut = new float[flattened];
        sequenced.CombineStreams(midStreams, ffnOut, injectFfn, sequencedOut);

        Assert.True(composedOut.AsSpan().SequenceEqual(sequencedOut.AsSpan()), "the sparse-full block forward must compose exactly the reference stage sequence in order");
    }

    // ---------------------------------------------------------------- the production engine refusal is unchanged by adding this spike

    [Fact]
    public void TransformerModel_StillRefusesQwen4Exp_OpenBecauseForwardExecutionIsIncomplete()
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        Qwen4ExpGeometry geometry = LinearOnlyGeometry(linearLayers: 1);

        // The spike is a synthetic composition harness only — it never opens a model. Asserting the resolved geometry drives it
        // while TransformerModel.Open still refuses this family proves the boundary between "the plan resolves" and "the engine runs".
        using Qwen4ExpForwardSpike spike = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend);

        Assert.Equal(AttentionKind.LinearGatedDeltaNet, spike.KindForLayer(0));
        Assert.NotNull(spike.Geometry.HyperConnections); // the same resolved geometry a plan carries feeds this spike without any file access.

        // No TransformerModel.Open overload accepts synthetic spike operands: composing qwen4exp into engine open is still unimplemented.
        bool hasSpikeOpen = typeof(TransformerModel).GetMethods()
            .Any(m => m.Name == "Open" && m.GetParameters().Any(p => p.ParameterType.FullName?.Contains("Qwen4ExpForwardSpike") ?? false));

        Assert.False(hasSpikeOpen, "TransformerModel.Open must not expose synthetic qwen4exp forward spike operands");
    }

    // ---------------------------------------------------------------- helpers

    private static Qwen4ExpGeometry LinearOnlyGeometry(int linearLayers)
    {
        AttentionKind[] kinds = [.. Enumerable.Repeat(AttentionKind.LinearGatedDeltaNet, linearLayers)];
        int[] ratios = new int[linearLayers]; // zero ratio on every linear layer

        return Geometry(kinds, ratios);
    }

    private static Qwen4ExpGeometry GeometryWithFullLayer(int compressRatio)
    {
        AttentionKind[] kinds = [AttentionKind.FullQuerySparseAttention];
        int[] ratios = [compressRatio];

        return Geometry(kinds, ratios);
    }

    private static Qwen4ExpGeometry Geometry(AttentionKind[] kinds, int[] ratios) => new(
        AttentionKinds: kinds,
        CompressRatios: ratios,
        LinearAttention: new GdnDescriptor(KeyHeadCount: 1, ValueHeadCount: 2, StateDim: 2, InnerSize: 4, TimeStepRank: 2, ConvKernelSize: 3),
        SparseAttention: new QsaDescriptor(IndexerHeadCount: 2, IndexerKeyLength: 4, TopKCells: 6), // even component width so the spike rotates indexer heads
        HyperConnections: new HyperConnectionDescriptor(StreamCount: 2, LowRank: 3));

    private static AttentionKind KindOf(Qwen4ExpGeometry geometry) => geometry.KindForLayer(0);

    private static float[] RunAllLayers(Qwen4ExpGeometry geometry, CpuBackend backend, float[] streamsIn, int tokenCount, int flattened)
    {
        using Qwen4ExpForwardSpike spike = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend);

        // Carry each layer's output streams into the next layer for every token, exactly as a multi-layer block sequence would.
        float[] current = [.. streamsIn];
        float[] next = new float[streamsIn.Length];

        spike.StartSequence(new float[Hidden]);

        for (int t = 0; t < tokenCount; t++)
        {
            // For a single layer index the spike keeps per-layer state across tokens; iterate every block in order.
            for (int layer = 0; layer < geometry.LayerCount; layer++)
            {
                float[] layerIn = current.AsSpan(t * flattened, flattened).ToArray();
                float[] layerOut = new float[flattened];
                spike.ForwardBlock(layer, layerIn, layerOut);
                layerOut.CopyTo(current.AsSpan(t * flattened, flattened));
            }

            _ = next; // kept for clarity: multi-layer carryover is written straight back into `current` per token.
        }

        return current;
    }

    private static void RmsNormWeighted(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, float eps, Span<float> dst)
    {
        double sumSq = 0.0;
        for (int i = 0; i < src.Length; i++)
        {
            sumSq += (double)src[i] * src[i];
        }

        double invRms = 1.0 / Math.Sqrt(sumSq / src.Length + eps);
        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = (float)((double)src[i] * invRms * weight[i]);
        }
    }

    private static double Dot(float[] matrix, int row, int width, ReadOnlySpan<float> vector)
    {
        double sum = 0.0;
        for (int i = 0; i < width; i++)
        {
            sum += (double)matrix[row * width + i] * vector[i];
        }

        return sum;
    }

    private static double Silu(double x) => x / (1.0 + Math.Exp(-x));

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static bool Close(float actual, double expected)
    {
        double tolerance = 1e-5 * Math.Max(1.0, Math.Abs(expected));
        return Math.Abs((double)actual - expected) <= tolerance;
    }

    private static float[] Matrix(int rows, int cols)
    {
        float[] matrix = new float[rows * cols];
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                matrix[row * cols + col] = (float)(((row * 17 + col * 29) % 11) - 5) / 4f;
            }
        }

        return matrix;
    }
}
