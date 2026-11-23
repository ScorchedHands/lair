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
/// in the graph's own order reproduces the spike's block forward bit-for-bit. The spike stays a synthetic composition harness:
/// engine open wires this family through its own bounded engine seam over real GGUF tensors, and no Open overload ever accepts
/// the spike's synthetic operands.
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

    // ---------------------------------------------------------------- sparse MoE: router, top-k, gather and expert execution compose in graph order

    [Fact]
    public void Spike_SparseMoE_FullBlock_ComposesHyperConnectionsQsaAndSparseMoEInGraphOrder()
    {
        Qwen4ExpGeometry geometry = GeometryWithFullLayer(compressRatio: 2);
        Qwen4ExpForwardSpike.SparseMoEConfig cfg = new(ExpertCount: 4, RoutedExpertsPerToken: 2, HasSharedExpert: true);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        int flattened = geometry.HyperConnections.FlattenedStreamFloats(Hidden);
        int streamCount = geometry.HyperConnections.StreamCount;

        // Instance A runs the block the way a sparse-configured spike composes it.
        using Qwen4ExpForwardSpike composed = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend, sparseMoE: cfg);
        float[] streamsIn = Matrix(flattened, 1).Select(v => v + 0.5f).ToArray();
        float[] composedOut = new float[flattened];
        composed.StartSequence(new float[Hidden]);
        composed.ForwardBlock(layer: 0, streamsIn, composedOut);

        foreach (float value in composedOut)
        {
            Assert.True(float.IsFinite(value), "sparse MoE composition must produce finite residual values");
        }

        // Instance B re-sequences the reference graph order by hand from an identical fresh sequence: build_hc_mix → sparse
        // attention → build_hc_combine → build_hc_mix → sparse MoE (router/gather/experts) → build_hc_combine.
        using Qwen4ExpForwardSpike sequenced = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend, sparseMoE: cfg);
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
        sequenced.ForwardSparseMoE(0, mixedFfn, ffnOut);

        float[] sequencedOut = new float[flattened];
        sequenced.CombineStreams(midStreams, ffnOut, injectFfn, sequencedOut);

        Assert.True(composedOut.AsSpan().SequenceEqual(sequencedOut.AsSpan()), "the sparse-full block forward must compose hyper-connections, QSA attention and sparse MoE in the reference stage order");
    }

    [Fact]
    public void Spike_SparseMoE_TopKSelection_IsDeterministicAndRespectsConfiguredCounts()
    {
        Qwen4ExpGeometry geometry = LinearOnlyGeometry(linearLayers: 1);
        Qwen4ExpForwardSpike.SparseMoEConfig cfg = new(ExpertCount: 6, RoutedExpertsPerToken: 3, HasSharedExpert: false);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using Qwen4ExpForwardSpike spike = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend, sparseMoE: cfg);

        float[] mixed = HiddenRow(1);

        int[] idsA = new int[3];
        float[] weightsA = new float[3];
        spike.RouteSparseMoE(0, mixed, idsA, weightsA);

        int[] idsB = new int[3];
        float[] weightsB = new float[3];
        spike.RouteSparseMoE(0, mixed, idsB, weightsB);

        Assert.True(idsA.SequenceEqual(idsB), "routing the same row twice must select the same experts");
        Assert.True(weightsA.AsSpan().SequenceEqual(weightsB.AsSpan()), "routing the same row twice must assign the same weights");

        Assert.Equal(3, idsA.Length); // exactly the configured routing count
        for (int i = 1; i < idsA.Length; i++)
        {
            Assert.True(idsA[i - 1] < idsA[i], $"selection must run in ascending expert-index order: {idsA[i - 1]} then {idsA[i]}");
        }

        Assert.All(idsA, id => Assert.True(id >= 0 && id < cfg.ExpertCount, $"selected expert {id} outside the declared expert count"));

        // qwen4exp routes with norm = true, so the selected weights renormalize to sum one.
        double sum = 0.0;
        foreach (float weight in weightsA)
        {
            sum += weight;
        }

        Assert.True(Math.Abs(sum - 1.0) < 1e-6, $"normalized routing weights must sum to 1, got {sum:G9}");

        // The selection matches an independent double-precision reimplementation over the spike's own router operands.
        Qwen4ExpForwardSpike.SparseFfnOperands moe = spike.SparseMoEOperands(0);
        (int[] refIds, double[] refWeights) = RouteReference(moe.Router, cfg.ExpertCount, cfg.RoutedExpertsPerToken, cfg.NormalizeExpertWeights, cfg.ExpertWeightsScale, mixed);
        Assert.True(refIds.SequenceEqual(idsA), "selection must match the independent softmax-then-top-k reference");

        for (int i = 0; i < refWeights.Length; i++)
        {
            Assert.True(Close(weightsA[i], refWeights[i]), $"routing weight {i}: expected {refWeights[i]:G9}, got {weightsA[i]:G9}");
        }
    }

    [Fact]
    public void Spike_SparseMoE_EqualRouterScores_PickLowerExpertIndex()
    {
        // Experts 0 and 3 carry an identical, clearly-highest router row, so every token ties them for the top slot. With a
        // routing count of one the strict-greater ascending scan must land on expert 0, not 3 — the same tie rule the
        // production route and the qwen2moe reference apply.
        Qwen4ExpGeometry geometry = LinearOnlyGeometry(linearLayers: 1);
        Qwen4ExpForwardSpike.SparseMoEConfig cfg = new(ExpertCount: 4, RoutedExpertsPerToken: 1, HasSharedExpert: false);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using Qwen4ExpForwardSpike spike = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend, sparseMoE: cfg);

        Qwen4ExpForwardSpike.SparseFfnOperands moe = spike.SparseMoEOperands(0);
        float[] tieRow = [2f, 1.5f, 1f, 0.5f, 0f, -0.5f, -1f, -1.5f]; // a fixed direction that scores above the other rows
        for (int c = 0; c < Hidden; c++)
        {
            moe.Router[0 * Hidden + c] = tieRow[c];
            moe.Router[3 * Hidden + c] = tieRow[c];
            moe.Router[1 * Hidden + c] = tieRow[c] * 0.1f; // strictly weaker, so experts 0 and 3 are the only top candidates
            moe.Router[2 * Hidden + c] = tieRow[c] * 0.1f;
        }

        int[] ids = new int[1];
        float[] weights = new float[1];
        spike.RouteSparseMoE(0, HiddenRow(1), ids, weights);

        Assert.Equal(0, ids[0]); // the tie resolves to the lower expert index
    }

    [Fact]
    public void Spike_SparseMoE_RoutedExpertOutput_MatchesHandComputedReferenceSelectionAndWeightSum()
    {
        Qwen4ExpGeometry geometry = LinearOnlyGeometry(linearLayers: 1);
        Qwen4ExpForwardSpike.SparseMoEConfig cfg = new(ExpertCount: 5, RoutedExpertsPerToken: 2, HasSharedExpert: true);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using Qwen4ExpForwardSpike spike = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend, sparseMoE: cfg);
        Qwen4ExpForwardSpike.SparseFfnOperands moe = spike.SparseMoEOperands(0);

        foreach (int seed in new[] { 1, 2, 3 })
        {
            float[] mixed = HiddenRow(seed);

            float[] actual = new float[Hidden];
            spike.ForwardSparseMoE(0, mixed, actual);

            double[] expected = ReferenceSparseFfn(moe, cfg, mixed);

            for (int h = 0; h < Hidden; h++)
            {
                Assert.True(Close(actual[h], expected[h]), $"seed {seed} sparse output {h}: expected {expected[h]:G9}, got {actual[h]:G9}");
            }
        }
    }

    [Fact]
    public void Spike_SparseMoE_SharedExpertContribution_IsIncludedAndGatedByPlainSigmoid()
    {
        Qwen4ExpGeometry geometry = LinearOnlyGeometry(linearLayers: 1);
        var withShared = new Qwen4ExpForwardSpike.SparseMoEConfig(ExpertCount: 4, RoutedExpertsPerToken: 2, HasSharedExpert: true);
        var withoutShared = new Qwen4ExpForwardSpike.SparseMoEConfig(ExpertCount: 4, RoutedExpertsPerToken: 2, HasSharedExpert: false);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using Qwen4ExpForwardSpike shared = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend, sparseMoE: withShared);
        using Qwen4ExpForwardSpike plain = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend, sparseMoE: withoutShared);

        Qwen4ExpForwardSpike.SparseFfnOperands moe = shared.SparseMoEOperands(0);
        float[] mixed = HiddenRow(1);

        float[] sharedOut = new float[Hidden];
        float[] plainOut = new float[Hidden];
        shared.ForwardSparseMoE(0, mixed, sharedOut);
        plain.ForwardSparseMoE(0, mixed, plainOut);

        // The shared expert must visibly change the feed-forward output.
        bool changed = false;
        for (int h = 0; h < Hidden; h++)
        {
            if (!sharedOut[h].Equals(plainOut[h]))
            {
                changed = true;
                break;
            }
        }

        Assert.True(changed, "a declared shared expert must change the sparse feed-forward output");

        // The shared contribution is the dense shared block scaled by sigmoid(score), where score is the selector dot — the
        // gate the qwen4exp reference builds.
        double score = 0.0;
        for (int c = 0; c < Hidden; c++)
        {
            score += (double)moe.SharedGateVector[c] * mixed[c];
        }

        double sigmoidGate = Sigmoid(score);
        double[] routedOnly = ReferenceRoutedSum(moe, withoutShared, mixed);
        double[] sharedBlock = SharedBlockContribution(moe, mixed);

        for (int h = 0; h < Hidden; h++)
        {
            double expectedSigmoid = routedOnly[h] + sigmoidGate * sharedBlock[h];
            Assert.True(Close(sharedOut[h], expectedSigmoid), $"shared output {h} must equal routed + sigmoid(score)*shared: expected {expectedSigmoid:G9}, got {sharedOut[h]:G9}");
        }

        // The qwen4exp gate is a plain sigmoid, not qwen2moe's silu(score) / score. The two forms agree for any nonzero score
        // (silu(x)/x == sigmoid(x)), so the only place they differ is a zero score: qwen4exp lands on sigmoid(0) = 0.5, while
        // qwen2moe divides 0 by 0. A zeroed selector vector forces that score to exactly zero, and the spike stays finite at
        // 0.5 — proving the plain-sigmoid form the reference uses.
        for (int c = 0; c < Hidden; c++)
        {
            moe.SharedGateVector[c] = 0f;
        }

        float[] zeroScoreOut = new float[Hidden];
        shared.ForwardSparseMoE(0, mixed, zeroScoreOut);

        foreach (float value in zeroScoreOut)
        {
            Assert.True(float.IsFinite(value), "a zero shared-expert score must stay finite under the qwen4exp sigmoid gate, not divide 0 by 0");
        }

        for (int h = 0; h < Hidden; h++)
        {
            double expectedHalfShared = routedOnly[h] + 0.5 * sharedBlock[h]; // sigmoid(0) == 0.5 exactly
            Assert.True(Close(zeroScoreOut[h], expectedHalfShared), $"zero-score shared output {h}: expected {expectedHalfShared:G9}, got {zeroScoreOut[h]:G9}");
        }
    }

    [Fact]
    public void Spike_SparseMoE_IsDeterministicAcrossRuns_AndAcrossChunkSplits()
    {
        Qwen4ExpGeometry geometry = GeometryWithFullLayer(compressRatio: 2);
        Qwen4ExpForwardSpike.SparseMoEConfig cfg = new(ExpertCount: 4, RoutedExpertsPerToken: 2, HasSharedExpert: true);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        int flattened = geometry.HyperConnections.FlattenedStreamFloats(Hidden);
        float[] streamsIn = Matrix(flattened * 4, Hidden);

        using Qwen4ExpForwardSpike firstRun = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend, maxChunk: 8, maxPositions: 32, sparseMoE: cfg);
        float[] outFirst = new float[flattened * 4];
        firstRun.StartSequence(new float[Hidden]);
        firstRun.ForwardBlockChunk(layer: 0, streamsIn, tokenCount: 4, outFirst);

        using Qwen4ExpForwardSpike secondRun = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend, maxChunk: 8, maxPositions: 32, sparseMoE: cfg);
        float[] outSecond = new float[flattened * 4];
        secondRun.StartSequence(new float[Hidden]);
        secondRun.ForwardBlockChunk(layer: 0, streamsIn, tokenCount: 4, outSecond);

        Assert.True(outFirst.AsSpan().SequenceEqual(outSecond.AsSpan()), "a sparse MoE block forward must be deterministic across runs");

        // Splitting the same four-token run after two tokens: sparse MoE is per-token with no carried state, and the QSA
        // attention state advances across the boundary, so the split run lands bit-for-bit on the unsplit run.
        using Qwen4ExpForwardSpike splitRun = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend, maxChunk: 8, maxPositions: 32, sparseMoE: cfg);
        float[] outSplit = new float[flattened * 4];
        splitRun.StartSequence(new float[Hidden]);
        splitRun.ForwardBlockChunk(layer: 0, streamsIn.AsSpan(0, flattened * 2), tokenCount: 2, outSplit.AsSpan(0, flattened * 2));
        splitRun.ForwardBlockChunk(layer: 0, streamsIn.AsSpan(flattened * 2, flattened * 2), tokenCount: 2, outSplit.AsSpan(flattened * 2, flattened * 2));

        Assert.True(outFirst.AsSpan().SequenceEqual(outSplit.AsSpan()), "a sparse MoE block forward must not change across a chunk boundary");
    }

    // ---------------------------------------------------------------- the spike never becomes a public engine open surface

    [Fact]
    public void TransformerModel_OpenNeverExposesSyntheticQwen4ExpSpikeOperands()
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        Qwen4ExpGeometry geometry = LinearOnlyGeometry(linearLayers: 1);

        // The spike is a synthetic composition harness only — it never opens a model. Asserting the resolved geometry drives it
        // while no Open overload takes its operand types proves the boundary between the spike and the engine seam.
        using Qwen4ExpForwardSpike spike = new(geometry, Hidden, HeadCount, KvHeadCount, HeadDim, IntermediateSize, Eps, backend);

        Assert.Equal(AttentionKind.LinearGatedDeltaNet, spike.KindForLayer(0));
        Assert.NotNull(spike.Geometry.HyperConnections); // the same resolved geometry a plan carries feeds this spike without any file access.

        // No TransformerModel.Open overload accepts synthetic spike operands or the sparse-FFN config: the engine seam wires
        // real GGUF tensors through its own internal engine, and the spike's operand types stay spike-local.
        bool hasSpikeOpen = typeof(TransformerModel).GetMethods()
            .Any(m => m.Name == "Open" && m.GetParameters().Any(p =>
                (p.ParameterType.FullName?.Contains("Qwen4ExpForwardSpike") ?? false)
                || (p.ParameterType.FullName?.Contains("SparseMoEConfig") ?? false)));

        Assert.False(hasSpikeOpen, "TransformerModel.Open must not expose synthetic qwen4exp forward spike operands");
    }

    // ---------------------------------------------------------------- sparse MoE reference helpers

    // Independent double-precision sparse-FFN forward over the spike's own operands: the same all-expert softmax, the same
    // ascending strict-greater top-k with the lower-index tie rule, the same ascending accumulation, and the shared block
    // gated by plain sigmoid(score). This is the hand-computed reference the sparse assertions compare against.
    private static double[] ReferenceSparseFfn(Qwen4ExpForwardSpike.SparseFfnOperands moe, Qwen4ExpForwardSpike.SparseMoEConfig cfg, float[] mixed)
    {
        (int[] ids, double[] weights) = RouteReference(moe.Router, cfg.ExpertCount, cfg.RoutedExpertsPerToken, cfg.NormalizeExpertWeights, cfg.ExpertWeightsScale, mixed);

        double[] outRow = new double[Hidden];

        for (int slot = 0; slot < ids.Length; slot++)
        {
            double[] expertOut = ExpertContribution(moe, ids[slot], mixed);
            for (int h = 0; h < Hidden; h++)
            {
                outRow[h] += weights[slot] * expertOut[h];
            }
        }

        if (cfg.HasSharedExpert)
        {
            double score = 0.0;
            for (int c = 0; c < Hidden; c++)
            {
                score += (double)moe.SharedGateVector[c] * mixed[c];
            }

            double gate = Sigmoid(score);
            double[] shared = SharedBlockContribution(moe, mixed);
            for (int h = 0; h < Hidden; h++)
            {
                outRow[h] += gate * shared[h];
            }
        }

        return outRow;
    }

    private static double[] ReferenceRoutedSum(Qwen4ExpForwardSpike.SparseFfnOperands moe, Qwen4ExpForwardSpike.SparseMoEConfig cfg, float[] mixed)
    {
        (int[] ids, double[] weights) = RouteReference(moe.Router, cfg.ExpertCount, cfg.RoutedExpertsPerToken, cfg.NormalizeExpertWeights, cfg.ExpertWeightsScale, mixed);

        double[] outRow = new double[Hidden];
        for (int slot = 0; slot < ids.Length; slot++)
        {
            double[] expertOut = ExpertContribution(moe, ids[slot], mixed);
            for (int h = 0; h < Hidden; h++)
            {
                outRow[h] += weights[slot] * expertOut[h];
            }
        }

        return outRow;
    }

    private static (int[] Ids, double[] Weights) RouteReference(float[] router, int expertCount, int routed, bool normalize, float scale, float[] mixed)
    {
        // Mirror the spike's own arithmetic exactly: a dot summed in double then narrowed to float, an all-expert softmax on
        // floats with MathF.Exp and a max-subtraction guard, ascending strict-greater picks that poison the chosen slot, then
        // insertion-sort into ascending expert-index order. Matching the engine's float path bit-for-bit is what keeps a
        // near-tie from resolving differently between this reference and the spike.
        float[] scores = new float[expertCount];
        for (int e = 0; e < expertCount; e++)
        {
            scores[e] = (float)Dot(router, e, Hidden, mixed);
        }

        float maxScore = float.NegativeInfinity;
        for (int e = 0; e < expertCount; e++)
        {
            if (scores[e] > maxScore)
            {
                maxScore = scores[e];
            }
        }

        float allSum = 0f;
        float[] probs = new float[expertCount];
        for (int e = 0; e < expertCount; e++)
        {
            probs[e] = MathF.Exp(scores[e] - maxScore);
            allSum += probs[e];
        }

        for (int e = 0; e < expertCount; e++)
        {
            probs[e] /= allSum;
        }

        int[] ids = new int[routed];
        float[] weightsF = new float[routed];
        double selectedSum = 0.0; // the reference sums in double only to compare weight magnitudes loosely; selection order is pinned by the float scores above
        for (int pick = 0; pick < routed; pick++)
        {
            int best = -1;
            float bestScore = float.NegativeInfinity;
            for (int e = 0; e < expertCount; e++)
            {
                if (scores[e] > bestScore)
                {
                    best = e;
                    bestScore = scores[e];
                }

                // ties keep the earlier index, exactly as the spike's scan does.
            }

            ids[pick] = best;
            weightsF[pick] = probs[best];
            selectedSum += probs[best];
            scores[best] = float.NegativeInfinity;
        }

        for (int i = 1; i < routed; i++)
        {
            int id = ids[i];
            float weight = weightsF[i];
            int j = i - 1;
            while (j >= 0 && ids[j] > id)
            {
                ids[j + 1] = ids[j];
                weightsF[j + 1] = weightsF[j];
                j--;
            }

            ids[j + 1] = id;
            weightsF[j + 1] = weight;
        }

        if (normalize && selectedSum > 0.0)
        {
            for (int i = 0; i < routed; i++)
            {
                weightsF[i] /= (float)selectedSum;
            }
        }

        if (scale != 0f)
        {
            for (int i = 0; i < routed; i++)
            {
                weightsF[i] *= scale;
            }
        }

        double[] weights = new double[routed];
        for (int i = 0; i < routed; i++)
        {
            weights[i] = weightsF[i];
        }

        return (ids, weights);
    }

    private static double[] ExpertContribution(Qwen4ExpForwardSpike.SparseFfnOperands moe, int expert, float[] mixed)
    {
        // Mirror the spike's expert chain: a float gate/up dot narrowed from the double sum, a float SwiGLU activation, then a
        // down projection summed in double over that float activation — the same shapes and order the spike walks.
        float[] act = new float[IntermediateSize];
        for (int i = 0; i < IntermediateSize; i++)
        {
            int row = expert * IntermediateSize + i;
            float gate = (float)Dot(moe.ExpertGate, row, Hidden, mixed);
            float up = (float)Dot(moe.ExpertUp, row, Hidden, mixed);
            act[i] = (float)(Silu(gate) * up);
        }

        double[] outRow = new double[Hidden];
        for (int h = 0; h < Hidden; h++)
        {
            outRow[h] = DotRowFlat(moe.ExpertDown, expert * Hidden + h, IntermediateSize, act);
        }

        return outRow;
    }

    private static double[] SharedBlockContribution(Qwen4ExpForwardSpike.SparseFfnOperands moe, float[] mixed)
    {
        float[] act = new float[IntermediateSize];
        for (int i = 0; i < IntermediateSize; i++)
        {
            float gate = (float)Dot(moe.SharedGate, i, Hidden, mixed);
            float up = (float)Dot(moe.SharedUp, i, Hidden, mixed);
            act[i] = (float)(Silu(gate) * up);
        }

        double[] outRow = new double[Hidden];
        for (int h = 0; h < Hidden; h++)
        {
            outRow[h] = DotRowFlat(moe.SharedDown, h, IntermediateSize, act);
        }

        return outRow;
    }

    private static double DotRowFlat(float[] matrix, int row, int width, float[] vector)
    {
        double sum = 0.0;
        for (int i = 0; i < width; i++)
        {
            sum += (double)matrix[row * width + i] * vector[i];
        }

        return sum;
    }

    private static float[] HiddenRow(int seed)
    {
        float[] row = new float[Hidden];
        for (int c = 0; c < Hidden; c++)
        {
            row[c] = (float)(((c * 7 + seed * 5) % 9) - 4) / 3f + 0.1f * seed;
        }

        return row;
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
