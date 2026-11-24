using System.Buffers;
using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Layer-level parity fixtures for the gated-delta-net and query-sparse-attention CPU semantics: conv history carries across chunk
/// splits, the delta rule lands on a hand-derived recurrence written in the reference's transposed slot layout, selection follows the
/// reference's pool-and-tail visibility with a deterministic tie rule, and restricted attention is bit-identical to the pinned causal
/// kernel whenever its selection covers the whole causal prefix.
/// </summary>
[Trait("Category", "Layers")]
public sealed class GdnQsaLayerTests
{
    [Fact]
    public void GdnOps_L2NormHeads_MatchesHandWholeHeadSum()
    {
        float[] src = [0.5f, -1f, 2f, 0f, 1f, 0f]; // two head regions of three dimensions
        float[] dst = new float[src.Length];
        const float Eps = 1e-6f;

        GdnOps.L2NormHeads(src, headDim: 3, Eps, dst);

        for (int region = 0; region < src.Length; region += 3)
        {
            double sumSq = 0.0;
            for (int i = 0; i < 3; i++)
            {
                sumSq += (double)src[region + i] * src[region + i];
            }

            double inv = 1.0 / Math.Sqrt(sumSq + Eps);
            for (int i = 0; i < 3; i++)
            {
                double expected = src[region + i] * inv;
                Assert.True(
                    Close(dst[region + i], expected),
                    $"L2 norm head {region / 3} dim {i}: expected {expected:G9}, got {dst[region + i]:G9}");
            }
        }

        // eps keeps an all-zero head finite instead of dividing by zero.
        float[] zeroDst = new float[3];
        GdnOps.L2NormHeads(new float[3], headDim: 3, Eps, zeroDst);
        foreach (float v in zeroDst)
        {
            Assert.Equal(0f, v);
        }
    }

    [Fact]
    public void GdnOps_DepthwiseConv_HistoryCarriesAcrossChunks_AndStoresRawTail()
    {
        const int Channels = 2;
        const int Kernel = 3;

        float[] history = [0.5f, 1.5f, -2f, 0.25f]; // two stored rows, oldest first
        float[] tokens = [1f, -0.5f, 0.25f, 2f, -1f, 0.75f, 0f, 1f, 2f, -2f]; // five token rows of two channels
        float[] kernel = [0.5f, -0.25f, 1f, 2f, 0f, -1f]; // one contiguous tap row per channel

        float[] outTokens = new float[tokens.Length];
        float[] newHistory = new float[history.Length];
        GdnOps.DepthwiseCausalConvSilu(history, tokens, Channels, tokenCount: 5, Kernel, kernel, outTokens, newHistory);

        float[] virtualInput = [.. history, .. tokens]; // the reference's virtual concatenation
        for (int t = 0; t < 5; t++)
        {
            for (int c = 0; c < Channels; c++)
            {
                double sum = 0.0;
                for (int j = 0; j < Kernel; j++)
                {
                    sum += (double)kernel[c * Kernel + j] * virtualInput[(t + j) * Channels + c]; // virtual row t + j indexes the history rows first, then this token's row
                }

                double expected = Silu(sum);
                Assert.True(
                    Close(outTokens[t * Channels + c], expected),
                    $"conv[{t}, {c}]: expected {expected:G9}, got {outTokens[t * Channels + c]:G9}");
            }
        }

        // The recurrent update stores the last kernel-minus-1 raw rows of the virtual concatenation, not convolution outputs.
        Assert.True(newHistory.AsSpan(0, history.Length).SequenceEqual(virtualInput.AsSpan(virtualInput.Length - history.Length, history.Length)));

        // Splitting the same five-token run after three tokens must leave every convolved row bit-identical.
        float[] splitOut = new float[tokens.Length];
        float[] firstHistory = [.. history]; // the same starting convolution state as the single-run reference
        float[] firstNew = new float[history.Length];
        GdnOps.DepthwiseCausalConvSilu(firstHistory, tokens.AsSpan(0, 3 * Channels), Channels, tokenCount: 3, Kernel, kernel, splitOut.AsSpan(0, 3 * Channels), firstNew);
        GdnOps.DepthwiseCausalConvSilu(firstNew, tokens.AsSpan(3 * Channels, 2 * Channels), Channels, tokenCount: 2, Kernel, kernel, splitOut.AsSpan(3 * Channels, 2 * Channels), firstHistory);

        Assert.True(outTokens.AsSpan().SequenceEqual(splitOut.AsSpan()), "convolution results must not change across a chunk boundary");
    }

    [Fact]
    public void GdnOps_GatedDeltaNet_MatchesHandRecurrence_AndCarriesStateAcrossChunks()
    {
        const int StateDim = 3;
        const int KeyHeads = 1;
        const int ValueHeads = 2;
        const int Tokens = 4;

        float[] q = [1f, 2f, 0f, -0.5f, 0.25f, 1f, 0f, 1f, 2f, 1f, -1f, 0.5f, 0f, 0.25f, 1f]; // token-major: one q head region per token (then padded to the shared head width)
        float[] k = [0.5f, -1f, 2f, 1f, 0f, -0.25f, 0.75f, 1.5f, -1f, 0.25f, 0f, 1f, 1f, 0.5f, -2f];
        float[] v = [0.5f, -1f, 2f, 0f, 1f, 0.5f, 1f, 0.25f, -0.5f, 2f, 0f, 1f, -1f, 0.75f, 0.5f, 0f, 1f, -0.25f, 2f, 1.5f, 0f, -1f, 0.25f, 1f, 0.5f, -0.5f, 1f, 2f, 0f, 1f, 1.5f, -1f, 0.25f, 0.5f, 0f, 1f]; // token-major value-head rows

        float[] gate = [-0.2f, 0.1f, 0.05f, -0.3f, -0.15f, 0.4f, 0.2f, -0.05f]; // token-major per value head
        float[] beta = [0.75f, 0.25f, 0.5f, 0.125f, 0.9f, 0.3f, 0.4f, 0.6f];

        float[] state = new float[ValueHeads * StateDim * StateDim];
        float[] outFull = new float[Tokens * ValueHeads * StateDim];

        GdnOps.GatedDeltaNetChunk(q, k, v, gate, beta, KeyHeads, ValueHeads, StateDim, Tokens, state, outFull);

        // Hand recurrence in the logical S[i][j] layout: the engine stores the transpose, so the oracle reads logical rows and the kernel reads transposed slots.
        double[][] logical = [new double[StateDim * StateDim], new double[StateDim * StateDim]];
        double outputScale = 1.0 / Math.Sqrt(StateDim);

        int outIndex = 0;
        for (int t = 0; t < Tokens; t++)
        {
            for (int h = 0; h < ValueHeads; h++)
            {
                double decay = Math.Exp(gate[t * ValueHeads + h]);
                double b = beta[t * ValueHeads + h];

                for (int i = 0; i < StateDim; i++)
                {
                    for (int j = 0; j < StateDim; j++)
                    {
                        logical[h][i * StateDim + j] *= decay;
                    }
                }

                double[] delta = new double[StateDim];
                for (int j = 0; j < StateDim; j++)
                {
                    double scaledSum = 0.0;
                    for (int i = 0; i < StateDim; i++)
                    {
                        scaledSum += logical[h][i * StateDim + j] * k[t * StateDim + i];
                    }

                    delta[j] = ((double)v[t * ValueHeads * StateDim + h * StateDim + j] - scaledSum) * b;
                }

                for (int i = 0; i < StateDim; i++)
                {
                    for (int j = 0; j < StateDim; j++)
                    {
                        logical[h][i * StateDim + j] += k[t * StateDim + i] * delta[j];
                    }
                }

                for (int j = 0; j < StateDim; j++)
                {
                    double sum = 0.0;
                    for (int i = 0; i < StateDim; i++)
                    {
                        sum += logical[h][i * StateDim + j] * q[t * StateDim + i];
                    }

                    Assert.True(
                        Close(outFull[outIndex++], sum * outputScale),
                        $"delta rule token {t} head {h} output {j}: expected {sum * outputScale:G9}, got {outFull[outIndex - 1]:G9}");
                }
            }
        }

        // The same four-token recurrence split into two chunks must produce bit-identical outputs and final state.
        float[] splitState = new float[ValueHeads * StateDim * StateDim];
        float[] splitOut = new float[Tokens * ValueHeads * StateDim];

        GdnOps.GatedDeltaNetChunk(q.AsSpan(0, 3 * StateDim), k.AsSpan(0, 3 * StateDim), v.AsSpan(0, 3 * ValueHeads * StateDim), gate.AsSpan(0, 3 * ValueHeads), beta.AsSpan(0, 3 * ValueHeads), KeyHeads, ValueHeads, StateDim, 3, splitState, splitOut.AsSpan(0, 3 * ValueHeads * StateDim));
        GdnOps.GatedDeltaNetChunk(q.AsSpan(3 * StateDim, 1 * StateDim), k.AsSpan(3 * StateDim, 1 * StateDim), v.AsSpan(3 * ValueHeads * StateDim, 1 * ValueHeads * StateDim), gate.AsSpan(3 * ValueHeads, 1 * ValueHeads), beta.AsSpan(3 * ValueHeads, 1 * ValueHeads), KeyHeads, ValueHeads, StateDim, 1, splitState, splitOut.AsSpan(3 * ValueHeads * StateDim, 1 * ValueHeads * StateDim));

        Assert.True(outFull.AsSpan().SequenceEqual(splitOut.AsSpan()), "delta-rule outputs must not change across a chunk boundary");
        Assert.True(state.AsSpan().SequenceEqual(splitState.AsSpan()), "delta-rule state must carry unchanged across a chunk boundary");
    }

    [Fact]
    public void GdnOps_ForwardChunk_IsBitIdenticalAcrossChunkSplits_AndStableAcrossRuns()
    {
        GdnOps.GdnBlockTensors w = new()
        {
            HiddenSize = 3,
            KeyHeads = 1,
            ValueHeads = 2,
            StateDim = 2,
            ConvKernel = 4,
            Qkv = Matrix(8, 3),
            Gate = Matrix(4, 3),
            BetaAlphaCat = Matrix(4, 3),
            SsmA = [-0.7f, -0.3f],
            DtBias = [0.15f, -0.25f],
            Conv1d = Matrix(8 * 4, 1),
            NormWeight = [1.25f, -0.75f],
            Output = Matrix(3, 4),
        };

        float[] hidden = Matrix(4, 3);
        float[] fullOut = new float[4 * 3];
        float[] state = new float[w.RecurrentStateValues];
        float[] history = new float[w.ConvHistoryValues];
        float[] newHistory = new float[w.ConvHistoryValues];
        GdnOps.GdnChunkScratch scratch = GdnOps.GdnChunkScratch.Create(w, maxTokenCount: 4);

        GdnOps.ForwardChunk(hidden, w, eps: 1e-6f, tokenCount: 4, history, newHistory, state, scratch, fullOut);

        // Split after two tokens: every later stage reads exactly the carried conv history and recurrent state, so the result must be bit-identical.
        float[] splitOut = new float[4 * 3];
        float[] splitState = new float[w.RecurrentStateValues];
        float[] splitHistory = new float[w.ConvHistoryValues];
        float[] splitNew = new float[w.ConvHistoryValues];
        GdnOps.GdnChunkScratch splitScratch = GdnOps.GdnChunkScratch.Create(w, maxTokenCount: 4);

        GdnOps.ForwardChunk(hidden.AsSpan(0, 2 * 3), w, eps: 1e-6f, tokenCount: 2, splitHistory, splitNew, splitState, splitScratch, splitOut.AsSpan(0, 2 * 3));
        GdnOps.ForwardChunk(hidden.AsSpan(2 * 3, 2 * 3), w, eps: 1e-6f, tokenCount: 2, splitNew, splitHistory, splitState, splitScratch, splitOut.AsSpan(2 * 3, 2 * 3));

        Assert.True(fullOut.AsSpan().SequenceEqual(splitOut.AsSpan()), "a gated-delta-net forward must not change across a chunk boundary");

        // A fresh run from zeroed state reproduces the same activation exactly.
        float[] rerunOut = new float[4 * 3];
        float[] rerunState = new float[w.RecurrentStateValues];
        float[] rerunHistory = new float[w.ConvHistoryValues];
        float[] rerunNew = new float[w.ConvHistoryValues];
        GdnOps.GdnChunkScratch rerunScratch = GdnOps.GdnChunkScratch.Create(w, maxTokenCount: 4);

        GdnOps.ForwardChunk(hidden, w, eps: 1e-6f, tokenCount: 4, rerunHistory, rerunNew, rerunState, rerunScratch, rerunOut);
        Assert.True(fullOut.AsSpan().SequenceEqual(rerunOut.AsSpan()), "gated-delta-net forward must be deterministic across runs");
    }

    [Fact]
    public void GdnOps_ForwardChunkKernel_BatchedProjectionsMatchPerTokenSoloBitForBit()
    {
        GdnOps.GdnKernelTensors w = KernelTensors();
        float[] hidden = Matrix(4, w.HiddenSize);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using ArrayGdnWeightSource weights = new(w);

        // Batched multi-token forward: every projection role runs as one batched pass per window.
        float[] batchOut = new float[4 * w.HiddenSize];
        float[] batchState = new float[RecurrentValues(w)];
        float[] batchHistory = new float[ConvHistoryValues(w)];
        float[] batchNew = new float[ConvHistoryValues(w)];
        GdnOps.GdnChunkScratch batchScratch = GdnOps.GdnChunkScratch.Create(w.KeyHeads, w.ValueHeads, w.StateDim, w.ConvKernel, maxTokenCount: 4);
        GdnOps.ForwardChunkKernel(backend, weights, w, eps: 1e-6f, 4, hidden, batchHistory, batchNew, batchState, batchScratch, batchOut);

        // Solo per-token forward over the same operands: the batched projections and the per-row solo dots must land identically,
        // and the convolution history and recurrent state must advance the same way across the token-by-token carry.
        float[] soloOut = new float[4 * w.HiddenSize];
        float[] soloState = new float[RecurrentValues(w)];
        float[] soloHistory = new float[ConvHistoryValues(w)];
        float[] soloNew = new float[ConvHistoryValues(w)];
        GdnOps.GdnChunkScratch soloScratch = GdnOps.GdnChunkScratch.Create(w.KeyHeads, w.ValueHeads, w.StateDim, w.ConvKernel, maxTokenCount: 4);
        for (int t = 0; t < 4; t++)
        {
            GdnOps.ForwardChunkKernel(backend, weights, w, eps: 1e-6f, 1, hidden.AsSpan(t * w.HiddenSize, w.HiddenSize), soloHistory, soloNew, soloState, soloScratch, soloOut.AsSpan(t * w.HiddenSize, w.HiddenSize));
            (soloHistory, soloNew) = (soloNew, soloHistory);
        }

        Assert.True(batchOut.AsSpan().SequenceEqual(soloOut.AsSpan()), "a batched multi-token GDN projection must land bit-for-bit on the per-token solo forward");
        Assert.True(batchState.AsSpan().SequenceEqual(soloState.AsSpan()), "the recurrent state must advance identically under batched projections");

        // After the solo loop's final rotation the produced conv tail sits in the history slot (what the next chunk would read),
        // which is exactly the tail the single batched call left in its new-history buffer.
        Assert.True(batchNew.AsSpan().SequenceEqual(soloHistory.AsSpan()), "the convolution history tail must carry identically under batched projections");
    }

    [Fact]
    public void GdnOps_ForwardChunkKernel_SingleTokenRunsSolo_MultiTokenRunsBatched()
    {
        GdnOps.GdnKernelTensors w = KernelTensors();
        float[] hidden = Matrix(4, w.HiddenSize);

        using CpuBackend inner = new(threads: 1, allowAvx2: true);
        using CountingBackend backend = new(inner);
        using ArrayGdnWeightSource weights = new(w);

        // Single token: the decode posture — every role a solo GEMV, no batched kernel call.
        float[] soloOut = new float[w.HiddenSize];
        float[] soloState = new float[RecurrentValues(w)];
        float[] soloHistory = new float[ConvHistoryValues(w)];
        float[] soloNew = new float[ConvHistoryValues(w)];
        GdnOps.GdnChunkScratch soloScratch = GdnOps.GdnChunkScratch.Create(w.KeyHeads, w.ValueHeads, w.StateDim, w.ConvKernel, maxTokenCount: 1);
        GdnOps.ForwardChunkKernel(backend, weights, w, eps: 1e-6f, 1, hidden.AsSpan(0, w.HiddenSize), soloHistory, soloNew, soloState, soloScratch, soloOut);
        Assert.Equal(5, backend.GemvCalls); // qkv, gate, beta, alpha, output — each one solo GEMV
        Assert.Equal(0, backend.GemvBatchCalls);

        // Multi-token: every role one batched pass per window, no solo GEMV left.
        backend.GemvCalls = 0;
        backend.GemvBatchCalls = 0;
        float[] batchOut = new float[4 * w.HiddenSize];
        float[] batchState = new float[RecurrentValues(w)];
        float[] batchHistory = new float[ConvHistoryValues(w)];
        float[] batchNew = new float[ConvHistoryValues(w)];
        GdnOps.GdnChunkScratch batchScratch = GdnOps.GdnChunkScratch.Create(w.KeyHeads, w.ValueHeads, w.StateDim, w.ConvKernel, maxTokenCount: 4);
        GdnOps.ForwardChunkKernel(backend, weights, w, eps: 1e-6f, 4, hidden, batchHistory, batchNew, batchState, batchScratch, batchOut);
        Assert.Equal(0, backend.GemvCalls);
        Assert.Equal(5, backend.GemvBatchCalls); // qkv, gate, beta, alpha, output — one windowed batched pass each

        // A chunk wider than the batch ceiling still lands on the same per-token results: windowing never moves a token's arithmetic.
        backend.GemvCalls = 0;
        backend.GemvBatchCalls = 0;
        float[] wideHidden = Matrix(70, w.HiddenSize);
        float[] wideOut = new float[70 * w.HiddenSize];
        float[] wideState = new float[RecurrentValues(w)];
        float[] wideHistory = new float[ConvHistoryValues(w)];
        float[] wideNew = new float[ConvHistoryValues(w)];
        GdnOps.GdnChunkScratch wideScratch = GdnOps.GdnChunkScratch.Create(w.KeyHeads, w.ValueHeads, w.StateDim, w.ConvKernel, maxTokenCount: 70);
        GdnOps.ForwardChunkKernel(backend, weights, w, eps: 1e-6f, 70, wideHidden, wideHistory, wideNew, wideState, wideScratch, wideOut);
        Assert.Equal(0, backend.GemvCalls);
        Assert.Equal(10, backend.GemvBatchCalls); // five roles split into two windows (64 + 6) each
    }

    [Fact]
    public void QsaOps_IndexerScores_SumsRectifiedHeadDotsWithOneScale()
    {
        const int Heads = 2;
        const int Dim = 4;

        float[] q = [1f, 0f, 2f, 0f, -1f, -1f, 0f, 0f]; // two indexer heads
        float[] pooled = [1f, 1f, 0f, 0f, -1f, 0f, 0f, 0f]; // two pooled keys

        float[] scores = new float[2];
        QsaOps.IndexerScores(q, pooled, Heads, Dim, poolCount: 2, scores);

        double scale = 1.0 / Math.Sqrt(Dim);
        Assert.True(Close(scores[0], (1.0 + 0.0) * scale), $"pool 0 score: expected {(1.0 + 0.0) * scale:G9}, got {scores[0]:G9}"); // head 0 contributes 1, head 1 is rectified to zero
        Assert.True(Close(scores[1], (0.0 + 1.0) * scale), $"pool 1 score: expected {(0.0 + 1.0) * scale:G9}, got {scores[1]:G9}"); // head 0 is rectified, head 1 contributes 1
    }

    [Fact]
    public void QsaOps_PoolKeys_AppliesMeanBeforeNormAndLeavesTailRaw()
    {
        QsaOps.QsaBlockTensors w = SparseTensors(compressRatio: 3, topKCells: 6);
        float[] raw = Matrix(5, w.IndexerDim);
        float[] pooled = new float[w.IndexerDim]; // only one complete pool exists for five cells at ratio three

        QsaOps.PoolKeys(raw, cacheCells: 5, w, normEps: 1e-5f, pooled);

        double[] mean = new double[w.IndexerDim];
        for (int row = 0; row < 3; row++)
        {
            for (int i = 0; i < w.IndexerDim; i++)
            {
                mean[i] += raw[row * w.IndexerDim + i];
            }
        }

        for (int i = 0; i < w.IndexerDim; i++)
        {
            mean[i] /= 3.0; // pooling averages the member keys component-wise before normalization
        }

        double sumSq = 0.0;
        for (int i = 0; i < w.IndexerDim; i++)
        {
            sumSq += mean[i] * mean[i]; // RMSNorm runs over the pooled vector, not over the original member rows
        }

        double invRms = 1.0 / Math.Sqrt(sumSq / w.IndexerDim + 1e-5);
        for (int i = 0; i < w.IndexerDim; i++)
        {
            double expected = mean[i] * invRms * w.IndexerKeyNorm[i];
            Assert.True(Close(pooled[i], expected), $"pooled key {i}: expected {expected:G9}, got {pooled[i]:G9}");
        }
    }

    [Fact]
    public void QsaOps_SelectCells_KeepsTopPoolsTailAndForcesNewestBlock()
    {
        QsaOps.QsaBlockTensors w = SparseTensors(compressRatio: 2, topKCells: 6); // three complete pools may be kept

        int[] selected = [.. QsaOps.SelectCells([0.5f, -1f, 0.7f, 0.3f], cacheCells: 9, w)];
        Assert.Equal([0, 1, 4, 5, 6, 7, 8], selected); // pools 2, 0, and 3 survive the budget; the incomplete tail cell 8 is always visible

        QsaOps.QsaBlockTensors tie = SparseTensors(compressRatio: 2, topKCells: 2); // one pool of budget, plus the raw tail
        int[] tieSelected = [.. QsaOps.SelectCells([0.9f, 0.9f], cacheCells: 5, tie)];
        Assert.Equal([0, 1, 4], tieSelected); // equal scores resolve to the lower block index deterministically

        QsaOps.QsaBlockTensors forced = SparseTensors(compressRatio: 2, topKCells: 6); // newest block is a complete pool and consumes budget
        int[] forcedSelected = [.. QsaOps.SelectCells([0.5f, -1f, 0.7f, 0.3f], cacheCells: 8, forced)];
        Assert.Equal([0, 1, 4, 5, 6, 7], forcedSelected); // pool 3 is forced, leaving only pools 2 and 0 from the top-k ranking
    }

    [Fact]
    public void QsaOps_RestrictedAttention_FullSelectionIsBitIdenticalToCausalKernel()
    {
        const int HeadDim = 2;
        const int Heads = 4;
        const int KvHeads = 2;
        const int Positions = 4;

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using BlockPool pool = new(blockTokens: 2, layerCount: 1, kvHeads: KvHeads, headDim: HeadDim, capacityBlocks: 2, backend);
        using SequenceKv kv = new(pool);

        float[][] kRows = [[0.5f, -0.25f, 1f, 0f], [-1f, 0.25f, 0.5f, 2f], [0f, -0.5f, 1.5f, -1f], [2f, 1f, -0.5f, 0.75f]];
        float[][] vRows = [[0.25f, 0.75f, -0.5f, 1f], [-1f, 0.125f, 0.5f, 0f], [1.5f, -0.25f, 0.75f, -1f], [0f, 2f, 1f, -0.5f]];

        float[] keyCache = new float[Positions * KvHeads * HeadDim];
        float[] valueCache = new float[Positions * KvHeads * HeadDim];

        for (int t = 0; t < Positions; t++)
        {
            kv.Append();
            kRows[t].CopyTo(keyCache.AsSpan(t * KvHeads * HeadDim, KvHeads * HeadDim));
            vRows[t].CopyTo(valueCache.AsSpan(t * KvHeads * HeadDim, KvHeads * HeadDim));
            kv.WriteK(t, 0, kRows[t]);
            kv.WriteV(t, 0, vRows[t]);
        }

        float[] q = [1f, 2f, 0f, 1f, -0.5f, 0.25f, 1f, 0f];
        float scale = 1f / MathF.Sqrt(HeadDim);

        float[] causal = new float[Heads * HeadDim];
        LayerOps.CausalAttention(backend, q, Heads, KvHeads, HeadDim, kv, layer: 0, position: Positions - 1, scale, softcap: 0f, window: 0, causal);

        QsaOps.QsaBlockTensors w = SparseTensors(compressRatio: 2, topKCells: 8);
        w.HeadCount = Heads;
        w.KvHeadCount = KvHeads;
        w.HeadDim = HeadDim;

        float[] restricted = new float[Heads * HeadDim];
        QsaOps.RestrictedAttention(backend, q, keyCache, valueCache, [0, 1, 2, 3], Positions, w, scale, restricted);

        Assert.True(causal.AsSpan().SequenceEqual(restricted.AsSpan()), "a full causal selection must land bit-for-bit on the pinned attention kernel");
    }

    [Fact]
    public void QsaOps_RestrictedAttention_DroppedCellsNeverTouchSoftmaxState()
    {
        const int HeadDim = 2;
        const int Heads = 1;
        const int KvHeads = 1;

        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        float[] q = [10f, 0f];
        float[] keyCache = [10f, 0f, 0.5f, -0.25f, 1f, 0.75f]; // cell 0 carries the huge score that the sparse selection excludes
        float[] valueCache = [100f, 0f, 0.25f, 0.75f, -1f, 0.5f];

        QsaOps.QsaBlockTensors w = SparseTensors(compressRatio: 2, topKCells: 4);
        w.HeadCount = Heads;
        w.KvHeadCount = KvHeads;
        w.HeadDim = HeadDim;

        float scale = 1f / MathF.Sqrt(HeadDim);
        List<int> selected = [1, 2];

        float[] sparse = new float[Heads * HeadDim];
        QsaOps.RestrictedAttention(backend, q, keyCache, valueCache, selected, cacheCells: 3, w, scale, sparse);

        double s1 = backend.Dot(q, keyCache.AsSpan(1 * HeadDim, HeadDim)) * scale;
        double s2 = backend.Dot(q, keyCache.AsSpan(2 * HeadDim, HeadDim)) * scale;
        double max = Math.Max(s1, s2);
        double w1 = Math.Exp(s1 - max);
        double w2 = Math.Exp(s2 - max);
        double sum = w1 + w2;

        for (int j = 0; j < HeadDim; j++)
        {
            double expected = (w1 * valueCache[1 * HeadDim + j] + w2 * valueCache[2 * HeadDim + j]) / sum;
            Assert.True(Close(sparse[j], expected), $"sparse output {j}: expected {expected:G9}, got {sparse[j]:G9}");
        }

        float[] full = new float[Heads * HeadDim];
        QsaOps.RestrictedAttention(backend, q, keyCache, valueCache, [0, 1, 2], cacheCells: 3, w, scale, full);
        Assert.True(Math.Abs(full[0] - sparse[0]) > 1e-3f, "the excluded high-scoring cell must change the full-selection result");
    }

    private static QsaOps.QsaBlockTensors SparseTensors(int compressRatio, int topKCells)
    {
        return new QsaOps.QsaBlockTensors
        {
            HiddenSize = 4,
            HeadCount = 2,
            KvHeadCount = 1,
            HeadDim = 3,
            IndexerHeads = 2,
            IndexerDim = 4,
            TopKCells = topKCells,
            CompressRatio = compressRatio,
            QueryGate = Matrix(2 * 2 * 3, 4),
            Key = Matrix(1 * 3, 4),
            Value = Matrix(1 * 3, 4),
            Output = Matrix(4, 2 * 3),
            IndexerQueryProj = Matrix(2 * 4, 4),
            IndexerKeyProj = Matrix(4, 4),
            IndexerQueryNorm = [1f, 0.5f, -0.5f, 2f],
            IndexerKeyNorm = [2f, 1f, 0.5f, -1f],
            QueryHeadNorm = [1f, 1f, 1f],
            KeyHeadNorm = [1f, 1f, 1f],
        };
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

    private static double Silu(double x) => x / (1.0 + Math.Exp(-x));

    private static bool Close(float actual, double expected)
    {
        double tolerance = 1e-5 * Math.Max(1.0, Math.Abs(expected));
        return Math.Abs((double)actual - expected) <= tolerance;
    }

    // A kernel-fed block's geometry, mirroring the synthetic dense fixture above: every projection role ships as an F32
    // LogicalTensor the array weight source answers with deterministic rows, and the conv taps / decay / dt bias / head norm
    // stay materialized exactly as the kernel seam carries them.
    private static GdnOps.GdnKernelTensors KernelTensors() => new()
    {
        HiddenSize = 3,
        KeyHeads = 1,
        ValueHeads = 2,
        StateDim = 2,
        ConvKernel = 4,
        Qkv = new LogicalTensor("gdn.qkv", new TensorShape([8, 3]), QuantFormat.F32, 0, 8 * 3 * 4L),
        Gate = new LogicalTensor("gdn.gate", new TensorShape([4, 3]), QuantFormat.F32, 0, 4 * 3 * 4L),
        Beta = new LogicalTensor("gdn.beta", new TensorShape([2, 3]), QuantFormat.F32, 0, 2 * 3 * 4L),
        Alpha = new LogicalTensor("gdn.alpha", new TensorShape([2, 3]), QuantFormat.F32, 0, 2 * 3 * 4L),
        Output = new LogicalTensor("gdn.out", new TensorShape([3, 4]), QuantFormat.F32, 0, 3 * 4 * 4L),
        Conv1d = Matrix(8 * 4, 1),
        SsmA = [-0.7f, -0.3f],
        DtBias = [0.15f, -0.25f],
        NormWeight = [1.25f, -0.75f],
    };

    private static int RecurrentValues(GdnOps.GdnKernelTensors w) => w.ValueHeads * w.StateDim * w.StateDim;

    private static int ConvHistoryValues(GdnOps.GdnKernelTensors w) => (w.ConvKernel - 1) * w.ConvChannels;

    /// <summary>An array-backed whole-tensor weight source that answers each GDN projection role with deterministic F32 rows,
    /// so a batched and a solo forward read the very same bytes.</summary>
    private sealed class ArrayGdnWeightSource : IWeightSource, IDisposable
    {
        private readonly Dictionary<string, byte[]> _payloads = new(StringComparer.Ordinal);

        public ArrayGdnWeightSource(GdnOps.GdnKernelTensors w)
        {
            Add(w.Qkv);
            Add(w.Gate);
            Add(w.Beta);
            Add(w.Alpha);
            Add(w.Output);
        }

        private void Add(LogicalTensor tensor)
        {
            int rows = tensor.Shape.Dims[0];
            int cols = tensor.Shape.Dims[^1];
            float[] matrix = Matrix(rows, cols);
            byte[] bytes = new byte[matrix.Length * 4];
            System.Buffer.BlockCopy(matrix, 0, bytes, 0, bytes.Length);
            _payloads[tensor.Name] = bytes;
        }

        public ReadOnlySpan<byte> Resolve(LogicalTensor tensor) =>
            _payloads.TryGetValue(tensor.Name, out byte[]? bytes)
                ? bytes
                : throw new InvalidOperationException($"GDN weight source has no payload for tensor '{tensor.Name}'.");

        public void Dispose()
        {
        }
    }

    /// <summary>Wraps a real CPU backend and tallies which dense kernel each projection took, so a test can assert the solo
    /// versus batched posture without reaching into the engine. All compute delegates to the wrapped backend unchanged.</summary>
    private sealed class CountingBackend : IBackend
    {
        private readonly CpuBackend _inner;

        public CountingBackend(CpuBackend inner) => _inner = inner;

        public int GemvCalls { get; set; }

        public int GemvBatchCalls { get; set; }

        public string DeviceId => _inner.DeviceId;

        public WeightTier WeightTier => _inner.WeightTier;

        public MemoryHandle AllocateAligned(long bytes, int alignment) => _inner.AllocateAligned(bytes, alignment);

        public void FreeAligned(MemoryHandle handle) => _inner.FreeAligned(handle);

        public IKvDeviceMirror OpenKvMirror(KvMirrorRequest request) => _inner.OpenKvMirror(request);

        public void Gemv(ReadOnlySpan<float> x, WeightRef w, Span<float> output)
        {
            GemvCalls++;
            _inner.Gemv(x, w, output);
        }

        public void GemvBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs)
        {
            GemvBatchCalls++;
            _inner.GemvBatch(inputs, count, w, outputs);
        }

        public bool SupportsGemv(QuantFormat format) => _inner.SupportsGemv(format);

        public void GemmBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs) => _inner.GemmBatch(inputs, count, w, outputs);

        public bool SupportsGemm(QuantFormat format) => _inner.SupportsGemm(format);

        public void DequantRow(WeightRef w, int row, Span<float> output) => _inner.DequantRow(w, row, output);

        public float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b) => _inner.Dot(a, b);

        public void Axpy(float alpha, ReadOnlySpan<float> x, Span<float> y) => _inner.Axpy(alpha, x, y);

        public void Fill(Span<float> y, float v) => _inner.Fill(y, v);

        public void Scale(Span<float> y, float s) => _inner.Scale(y, s);

        public void Dispose() => _inner.Dispose();
    }
}
