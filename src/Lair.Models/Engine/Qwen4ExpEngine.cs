using Lair.Contracts;
using Lair.Loader;

namespace Lair.Models.Engine;

/// <summary>
/// Bounded CPU-only forward engine for a small synthetic qwen4exp fixture: it loads the file's own tensors through the model
/// plan's tensor map, sequences every block in the reference order — hyper-connection attention mix, gated-delta-net linear
/// attention or indexer-selected restricted full attention per resolved layer kind, hyper-attention combine, hyper-connection
/// feed-forward mix, routed sparse feed-forward with a plain-sigmoid shared expert (or the dense gate/up/down triplet when the
/// file declares no experts), and the hyper-feed-forward combine — then collapses every sequence's residual streams through the
/// global output hyper-connection trio before the output head produces logits. It is the engine seam behind
/// <see cref="TransformerModel"/> for this family: no operand type here is public engine surface, nothing loads the 70 GB
/// Flash-Next target, and every cache is sized from the opened context ceiling rather than a serving pool.
///
/// The composition mirrors the committed spike's stage math exactly — the same grouped RMS norms, low-rank SiLU gate pair, mean
/// collapse of gated streams, 2·sigmoid scatter on the combine side, all-expert router softmax with ascending strict-greater
/// top-k and lower-index tie resolution, SwiGLU expert chains accumulated in ascending expert-index order — against real GGUF rows
/// instead of synthetic fills. State is explicit per sequence: each linear block's delta-rule slot arrays plus raw convolution
/// history advance across chunk splits through <see cref="GdnOps.ForwardChunk"/>, and each sparse-full block's whole-sequence KV
/// rows, raw indexer-key cache and pooled selection run through the existing <see cref="QsaOps"/> semantics unchanged, so a
/// sequence forwarded as consecutive chunks lands bit-for-bit on one unsplit pass. The shared expert uses the qwen4exp reference
/// gate — plain <c>sigmoid(dot(ffn_gate_inp_shexp, mixed))</c>, finite at a zero score where qwen2moe's
/// <c>silu(score) / score</c> divides 0 by 0 — rather than qwen2moe's posture, which stays exactly as production applies it for
/// that family.
/// </summary>
internal sealed class Qwen4ExpEngine : IDisposable
{
    private readonly ModelPlan _plan;
    private readonly Qwen4ExpGeometry _geometry;
    private readonly IBackend _backend;
    private readonly IWeightSource _weights;
    private readonly LogicalTensorTable _fileTensors;
    private readonly int _hiddenSize;
    private readonly int _headCount;
    private readonly int _kvHeadCount;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly float _eps;
    private readonly int _streamCount;
    private readonly int _lowRank;
    private readonly int _flattened;
    private readonly int _maxPositions;

    // Per-block operands loaded once from the file: one hyper-connection trio per sublayer side, either a gated-delta-net operand
    // set (linear layers) or sparse-full attention operands plus the indexer quartet (full layers), and the block's routed expert
    // stacks with their router row — every role read as contiguous float rows because this bounded path runs F32 fixtures only.
    private readonly LayerOperands[] _layers;

    // The global output hyper-connection trio: one flattened norm vector plus a final down/up pair, no inject row at this scope.
    private readonly OutputHyperConnectionTensors _outputHc;

    // Logit head rows and the embedding table — untied <c>output.weight</c> when the file ships it, tied token-embedding rows otherwise.
    private readonly float[] _embeddingRows; // vocabSize x hidden
    private readonly float[] _logitHead;     // vocabSize x hidden (untied), or the same rows when tied

    // Rotary tables shared across every block: one position-major cos/sin run for the q/k head dimension and one for the indexer
    // component width, built once at open so no forward rebuilds either.
    private readonly float[] _cosHead;
    private readonly float[] _sinHead;
    private readonly float[] _cosIndexer;
    private readonly float[] _sinIndexer;
    private readonly int _ropeHalfHead;
    private readonly int _ropeHalfIndexer;

    public Qwen4ExpEngine(ModelPlan plan, LogicalTensorTable tensors, IWeightSource weights, IBackend backend, int maxPositions)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(tensors);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(backend);

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxPositions);

        if (plan.Qwen4Exp is not { } geometry)
        {
            throw new ArgumentException("A qwen4exp engine needs a plan whose family geometry resolved.", nameof(plan));
        }

        _plan = plan;
        _geometry = geometry;
        _backend = backend;
        _weights = weights;
        _fileTensors = tensors;
        _hiddenSize = plan.HiddenSize;
        _headCount = plan.HeadCount;
        _kvHeadCount = plan.KvHeadCount;
        _headDim = plan.HeadDim;
        _intermediateSize = plan.IntermediateSize;
        _eps = plan.EpsRmsNorm;
        _maxPositions = maxPositions;

        if (_headDim % 2 != 0)
        {
            throw new ModelException($"qwen4exp head dimension {_headDim} is odd; RoPE pairs require an even head dimension");
        }

        if (geometry.FullLayerCount > 0 && geometry.SparseAttention.IndexerKeyLength % 2 != 0)
        {
            // The spike's own posture: indexer heads rotate in pairs, so a sparse-full block needs an even component width.
            throw new ModelException($"qwen4exp indexer key length {geometry.SparseAttention.IndexerKeyLength} is odd; RoPE pairs require an even indexer key length on sparse-full blocks");
        }

        _streamCount = geometry.HyperConnections.StreamCount;
        _lowRank = geometry.HyperConnections.LowRank;
        _flattened = geometry.HyperConnections.FlattenedStreamFloats(_hiddenSize);

        if (_streamCount <= 1)
        {
            throw new ModelException($"qwen4exp hyper_connection.count is {_streamCount}; mixing needs at least two residual streams");
        }

        TensorMap map = plan.Tensors;

        _layers = new LayerOperands[geometry.LayerCount];
        for (int layer = 0; layer < geometry.LayerCount; layer++)
        {
            _layers[layer] = LoadLayer(layer, map);
        }

        _outputHc = new OutputHyperConnectionTensors
        {
            Norm = LoadVector(RequireTensor(map[TensorRole.OutputStreamNorm], tensors), "output hyper-connection norm", _flattened),
            Down = LoadMatrix(RequireTensor(map[TensorRole.OutputStreamDown], tensors), "output hyper-connection down", [_lowRank, _flattened]),
            Up = LoadMatrix(RequireTensor(map[TensorRole.OutputStreamUp], tensors), "output hyper-connection up", [_flattened, _lowRank]),
        };

        LogicalTensor tokenEmbedding = RequireTensor(map[TensorRole.TokenEmbedding], tensors);
        _embeddingRows = LoadMatrix(tokenEmbedding, "token embedding", [plan.VocabSize, _hiddenSize]);

        LogicalTensor? outputHead = map[TensorRole.OutputHead].Resolve(tensors);
        _logitHead = outputHead is not null
            ? LoadMatrix(outputHead, "output head", [plan.VocabSize, _hiddenSize])
            : _embeddingRows; // tied posture: the logits read the same rows the embeddings do

        _ropeHalfHead = _headDim / 2;
        _ropeHalfIndexer = geometry.SparseAttention.IndexerKeyLength / 2;

        _cosHead = new float[maxPositions * _ropeHalfHead];
        _sinHead = new float[maxPositions * _ropeHalfHead];
        LayerOps.BuildRopeTables(plan.RopeTheta, _headDim, maxPositions, _cosHead, _sinHead);

        _cosIndexer = new float[maxPositions * Math.Max(1, _ropeHalfIndexer)];
        _sinIndexer = new float[maxPositions * Math.Max(1, _ropeHalfIndexer)];
        if (_ropeHalfIndexer > 0)
        {
            LayerOps.BuildRopeTables(plan.RopeTheta, geometry.SparseAttention.IndexerKeyLength, maxPositions, _cosIndexer, _sinIndexer);
        }
    }

    /// <summary>The resolved family geometry this engine sequences against.</summary>
    public Qwen4ExpGeometry Geometry => _geometry;

    /// <summary>
    /// Fresh per-sequence state: each linear block's delta-rule slot arrays plus convolution history, and each sparse-full block's
    /// whole-sequence KV rows, raw indexer-key cache, pooled-key scratch and head scratch — all sized from the opened context
    /// ceiling. This is the engine-local bounded cache path: it carries what a synthetic sequence needs across chunk splits, not a
    /// production serving pool.
    /// </summary>
    public Qwen4ExpSequence NewSequence() => new(_geometry, _headCount, _kvHeadCount, _headDim, _maxPositions);

    /// <summary>
    /// Forward one chunk of one sequence: every token seeds its residual streams from its embedding row (the reference's
    /// <c>hc_init</c> repeats that row across every stream), then each block runs over the whole chunk — attention-side
    /// hyper-connection mix, a gated-delta-net chunk or per-token indexer-selected restricted attention, attention combine,
    /// feed-forward-side mix, the block's feed-forward, and the final combine. The carried recurrent state, convolution history and
    /// synthetic caches advance in place across calls, so a sequence split at any chunk boundary lands bit-for-bit on one unsplit
    /// pass. Returns every token's logits for this chunk; the final row is what <see cref="LastLogits"/> reports beside it.
    /// </summary>
    public Span<float> ForwardChunk(Qwen4ExpSequence sequence, ReadOnlySpan<int> tokens)
    {
        ArgumentNullException.ThrowIfNull(sequence);

        int tokenCount = tokens.Length;
        if (tokenCount == 0)
        {
            throw new ArgumentException("A qwen4exp chunk forward needs at least one token.", nameof(tokens));
        }

        if (sequence.Positions + tokenCount > _maxPositions)
        {
            throw new InvalidOperationException(
                $"a qwen4exp sequence of {sequence.Positions} positions plus a further {tokenCount} tokens passes this engine's opened context ceiling of {_maxPositions}; the bounded synthetic path sizes its caches at open and never grows them.");
        }

        int streamsPerToken = _flattened;
        float[] streams = new float[tokenCount * streamsPerToken];
        for (int t = 0; t < tokenCount; t++)
        {
            SeedStreams(tokens[t], streams.AsSpan(t * streamsPerToken, streamsPerToken));
        }

        float[] mixedRows = new float[tokenCount * _hiddenSize];
        float[] sublayerRows = new float[tokenCount * _hiddenSize];
        float[] midStreams = new float[tokenCount * streamsPerToken];
        // The attention side mixes every token's streams before the block runs over the whole chunk, so each token's inject
        // scores must survive until that token's own combine — one row per token, exactly as the spike carries them.
        float[] attnInject = new float[tokenCount * _streamCount];
        float[] injectScratch = new float[_streamCount];
        float[] ffnScratch = new float[_intermediateSize];

        for (int layer = 0; layer < _geometry.LayerCount; layer++)
        {
            LayerOperands w = _layers[layer];
            AttentionKind kind = _geometry.KindForLayer(layer);

            for (int t = 0; t < tokenCount; t++)
            {
                MixStreams(w.AttnHyper, streams.AsSpan(t * streamsPerToken, streamsPerToken), mixedRows.AsSpan(t * _hiddenSize, _hiddenSize), attnInject.AsSpan(t * _streamCount, _streamCount));
            }

            if (kind == AttentionKind.LinearGatedDeltaNet)
            {
                GdnOps.ForwardChunk(
                    mixedRows.AsSpan(0, tokenCount * _hiddenSize),
                    w.Linear,
                    _eps,
                    tokenCount,
                    sequence.ConvHistory(layer),
                    sequence.NextConvHistory(layer),
                    sequence.RecurrentState(layer),
                    sequence.GdnScratch(layer),
                    sublayerRows.AsSpan(0, tokenCount * _hiddenSize));

                // The convolution history update writes the new tail while the previous one is still read; rotate so the next chunk sees exactly what this one left behind.
                sequence.RotateConvHistory(layer);
            }
            else
            {
                QsaOps.QsaBlockTensors sparseTensors = w.SparseFull ?? throw new InvalidOperationException($"qwen4exp layer {layer} resolved as a sparse-full block but its attention operands were never loaded.");

                for (int t = 0; t < tokenCount; t++)
                {
                    SparseAttention(sparseTensors, sequence.Qsa(layer), mixedRows.AsSpan(t * _hiddenSize, _hiddenSize), sublayerRows.AsSpan(t * _hiddenSize, _hiddenSize));
                }
            }

            for (int t = 0; t < tokenCount; t++)
            {
                CombineStreams(streams.AsSpan(t * streamsPerToken, streamsPerToken), sublayerRows.AsSpan(t * _hiddenSize, _hiddenSize), attnInject.AsSpan(t * _streamCount, _streamCount), midStreams.AsSpan(t * streamsPerToken, streamsPerToken));

                MixStreams(w.FfnHyper, midStreams.AsSpan(t * streamsPerToken, streamsPerToken), mixedRows.AsSpan(t * _hiddenSize, _hiddenSize), injectScratch);

                ForwardFeedForward(w, mixedRows.AsSpan(t * _hiddenSize, _hiddenSize), sublayerRows.AsSpan(t * _hiddenSize, _hiddenSize), ffnScratch);

                CombineStreams(midStreams.AsSpan(t * streamsPerToken, streamsPerToken), sublayerRows.AsSpan(t * _hiddenSize, _hiddenSize), injectScratch, streams.AsSpan(t * streamsPerToken, streamsPerToken));
            }
        }

        sequence.Positions += tokenCount;

        float[] logits = new float[tokenCount * _plan.VocabSize];
        for (int t = 0; t < tokenCount; t++)
        {
            MixOutputStreams(streams.AsSpan(t * streamsPerToken, streamsPerToken), mixedRows.AsSpan(t * _hiddenSize, _hiddenSize));

            for (int v = 0; v < _plan.VocabSize; v++)
            {
                logits[t * _plan.VocabSize + v] = DotRow(_logitHead, v, _hiddenSize, mixedRows.AsSpan(t * _hiddenSize, _hiddenSize));
            }

            if (_plan.FinalLogitSoftcap > 0f)
            {
                for (int v = 0; v < _plan.VocabSize; v++)
                {
                    logits[t * _plan.VocabSize + v] = LayerOps.Softcap(logits[t * _plan.VocabSize + v], _plan.FinalLogitSoftcap);
                }
            }
        }

        if (sequence.FinalLogits.Length != _plan.VocabSize)
        {
            sequence.FinalLogits = new float[_plan.VocabSize]; // the seam's last-logits read is one vocab-sized row, allocated on the first forward
        }

        logits.AsSpan((tokenCount - 1) * _plan.VocabSize, _plan.VocabSize).CopyTo(sequence.FinalLogits);
        return logits.AsSpan();
    }

    /// <summary>The last chunk's final-row logits this sequence produced.</summary>
    public Span<float> LastLogits(Qwen4ExpSequence sequence)
    {
        ArgumentNullException.ThrowIfNull(sequence);
        return sequence.FinalLogits;
    }

    // One sparse-full block's attention for one token against the block's carried caches: project and rotate the block q/k heads,
    // append K/V, project/rotate the indexer query at this position, append the raw indexer key, pool complete blocks of ratio
    // cells rotated to their first member, score pools, select visible cells through QsaOps.SelectCells, run dense grouped-query
    // attention restricted to that selection — then gate each attended head by sigmoid of its gate slice and project. The
    // query-and-gate layout alternates a query slice and its gate slice per head exactly as the reference's interleaved rows carry
    // them.
    private void SparseAttention(QsaOps.QsaBlockTensors w, Qwen4ExpSequence.SequenceQsaState state, ReadOnlySpan<float> mixed, Span<float> attnOut)
    {
        int position = state.CacheCells;
        if (position >= _maxPositions)
        {
            throw new InvalidOperationException($"a qwen4exp sparse block reached cache cell {position}, past this engine's opened context ceiling of {_maxPositions}.");
        }

        int qRegion = _headCount * _headDim;
        int kvRegion = _kvHeadCount * _headDim;
        int idxDim = w.IndexerDim;

        for (int h = 0; h < _headCount; h++)
        {
            int qRow = (2 * h) * _headDim;
            int gateRow = (2 * h + 1) * _headDim;

            for (int j = 0; j < _headDim; j++)
            {
                state.QueryHead[j] = DotRow(w.QueryGate, qRow + j, _hiddenSize, mixed);
                state.GateHead[j] = DotRow(w.QueryGate, gateRow + j, _hiddenSize, mixed);
            }

            LayerOps.RmsNormWeightedHeads(state.QueryHead.AsSpan(0, _headDim), w.QueryHeadNorm, _headDim, _eps, state.QueryHead.AsSpan(0, _headDim));
            QsaOps.RotateAtPosition(state.QueryHead.AsSpan(0, _headDim), _ropeHalfHead, _cosHead, _sinHead, position);
            state.QueryHead.AsSpan(0, _headDim).CopyTo(state.RotatedQuery.AsSpan(h * _headDim, _headDim));
        }

        for (int h = 0; h < _kvHeadCount; h++)
        {
            for (int j = 0; j < _headDim; j++)
            {
                state.KHead[j] = DotRow(w.Key, h * _headDim + j, _hiddenSize, mixed);
                state.VHead[j] = DotRow(w.Value, h * _headDim + j, _hiddenSize, mixed);
            }

            LayerOps.RmsNormWeightedHeads(state.KHead.AsSpan(0, _headDim), w.KeyHeadNorm, _headDim, _eps, state.KHead.AsSpan(0, _headDim));
            QsaOps.RotateAtPosition(state.KHead.AsSpan(0, _headDim), _ropeHalfHead, _cosHead, _sinHead, position);

            state.KHead.AsSpan(0, _headDim).CopyTo(state.KeyCache.AsSpan(position * kvRegion + h * _headDim, _headDim));
            state.VHead.AsSpan(0, _headDim).CopyTo(state.ValueCache.AsSpan(position * kvRegion + h * _headDim, _headDim));
        }

        for (int h = 0; h < w.IndexerHeads; h++)
        {
            for (int j = 0; j < idxDim; j++)
            {
                state.IndexerQuery[h * idxDim + j] = DotRow(w.IndexerQueryProj, h * idxDim + j, _hiddenSize, mixed);
            }
        }

        LayerOps.RmsNormWeightedHeads(state.IndexerQuery.AsSpan(0, w.IndexerHeads * idxDim), w.IndexerQueryNorm, idxDim, _eps, state.IndexerQuery.AsSpan(0, w.IndexerHeads * idxDim));

        for (int h = 0; h < w.IndexerHeads; h++)
        {
            QsaOps.RotateAtPosition(state.IndexerQuery.AsSpan(h * idxDim, idxDim), _ropeHalfIndexer, _cosIndexer, _sinIndexer, position);
        }

        for (int j = 0; j < idxDim; j++)
        {
            state.RawIndexerKeys[position * idxDim + j] = DotRow(w.IndexerKeyProj, j, _hiddenSize, mixed);
        }

        state.CacheCells = position + 1;
        int cacheCells = state.CacheCells;
        float attnScale = 1f / MathF.Sqrt((float)_headDim);

        if (w.CompressRatio < 2 || cacheCells / w.CompressRatio == 0)
        {
            // A ratio that groups fewer than two cells runs dense and never reaches selection; a cache shorter than one complete pool has nothing to rank. Either way every cached cell stays visible.
            List<int> visible = [];
            for (int cell = 0; cell < cacheCells; cell++)
            {
                visible.Add(cell);
            }

            QsaOps.RestrictedAttention(_backend, state.RotatedQuery, state.KeyCache, state.ValueCache, visible, cacheCells, w, attnScale, state.AttentionConcat);
        }
        else
        {
            int fullPools = cacheCells / w.CompressRatio;
            QsaOps.PoolKeys(state.RawIndexerKeys.AsSpan(0, cacheCells * idxDim), cacheCells, w, _eps, state.PooledKeys.AsSpan(0, fullPools * idxDim));

            for (int p = 0; p < fullPools; p++)
            {
                QsaOps.RotateAtPosition(state.PooledKeys.AsSpan(p * idxDim, idxDim), _ropeHalfIndexer, _cosIndexer, _sinIndexer, p * w.CompressRatio);
            }

            QsaOps.IndexerScores(state.IndexerQuery.AsSpan(0, w.IndexerHeads * idxDim), state.PooledKeys.AsSpan(0, fullPools * idxDim), w.IndexerHeads, idxDim, fullPools, state.Scores.AsSpan(0, fullPools));
            List<int> selected = QsaOps.SelectCells(state.Scores.AsSpan(0, fullPools), cacheCells, w);
            QsaOps.RestrictedAttention(_backend, state.RotatedQuery, state.KeyCache, state.ValueCache, selected, cacheCells, w, attnScale, state.AttentionConcat);
        }

        for (int h = 0; h < _headCount; h++)
        {
            for (int j = 0; j < _headDim; j++)
            {
                state.AttentionConcat[h * _headDim + j] *= GdnOps.Sigmoid(state.GateHead[j]);
            }
        }

        for (int o = 0; o < _hiddenSize; o++)
        {
            attnOut[o] = DotRow(w.Output, o, qRegion, state.AttentionConcat);
        }
    }

    // The reference build_hc_mix: grouped RMS normalization per stream scaled by that stream's slice of the flattened norm vector,
    // the flattened streams through the low-rank down projection scaled by 1/streams and SiLU, expansion through the up projection
    // to a per-stream gate, gating the normalized streams by sigmoid of that gate, and collapse by their mean to the hidden row the
    // sublayer consumes. The same normalized streams feed the inject projection the combine weights the residual scatter with.
    private void MixStreams(HyperConnectionTensors hc, ReadOnlySpan<float> streams, Span<float> mixed, Span<float> inject)
    {
        float[] normed = new float[_flattened];
        for (int s = 0; s < _streamCount; s++)
        {
            int baseOffset = s * _hiddenSize;
            LayerOps.RmsNormWeighted(streams.Slice(baseOffset, _hiddenSize), hc.Norm.AsSpan(baseOffset, _hiddenSize), _eps, normed.AsSpan(baseOffset, _hiddenSize));
        }

        float invStreams = 1f / _streamCount;

        float[] lo = new float[_lowRank];
        for (int r = 0; r < _lowRank; r++)
        {
            lo[r] = LayerOps.Silu(DotRow(hc.Down, r, _flattened, normed) * invStreams);
        }

        float[] gated = new float[_flattened];
        for (int f = 0; f < _flattened; f++)
        {
            float gate = DotRow(hc.Up, f, _lowRank, lo);
            gated[f] = normed[f] * GdnOps.Sigmoid(gate);
        }

        for (int h = 0; h < _hiddenSize; h++)
        {
            double sum = 0.0;
            for (int s = 0; s < _streamCount; s++)
            {
                sum += gated[s * _hiddenSize + h];
            }

            mixed[h] = (float)(sum * invStreams);
        }

        for (int s = 0; s < _streamCount; s++)
        {
            inject[s] = DotRow(hc.Inject, s, _flattened, normed) * invStreams;
        }
    }

    // The global output-side mix: the same grouped norm / low-rank SiLU / sigmoid gate / mean collapse as every block-side mix, but
    // over the output_hc_* trio and with no inject row — the conversion ships only that trio globally.
    private void MixOutputStreams(ReadOnlySpan<float> streams, Span<float> mixed)
    {
        float[] normed = new float[_flattened];
        for (int s = 0; s < _streamCount; s++)
        {
            int baseOffset = s * _hiddenSize;
            LayerOps.RmsNormWeighted(streams.Slice(baseOffset, _hiddenSize), _outputHc.Norm.AsSpan(baseOffset, _hiddenSize), _eps, normed.AsSpan(baseOffset, _hiddenSize));
        }

        float invStreams = 1f / _streamCount;

        float[] lo = new float[_lowRank];
        for (int r = 0; r < _lowRank; r++)
        {
            lo[r] = LayerOps.Silu(DotRow(_outputHc.Down, r, _flattened, normed) * invStreams);
        }

        for (int h = 0; h < _hiddenSize; h++)
        {
            double sum = 0.0;
            for (int s = 0; s < _streamCount; s++)
            {
                int f = s * _hiddenSize + h;
                float gate = DotRow(_outputHc.Up, f, _lowRank, lo);
                sum += normed[f] * GdnOps.Sigmoid(gate);
            }

            mixed[h] = (float)(sum * invStreams);
        }
    }

    // The reference build_hc_combine: scatter one sublayer's hidden output back across every residual stream weighted by 2·sigmoid
    // of that stream's inject score, so a zero injection leaves a plain residual add.
    private void CombineStreams(ReadOnlySpan<float> streamsIn, ReadOnlySpan<float> blockOut, ReadOnlySpan<float> inject, Span<float> streamsOut)
    {
        for (int s = 0; s < _streamCount; s++)
        {
            float weight = 2f * GdnOps.Sigmoid(inject[s]);
            int baseOffset = s * _hiddenSize;

            for (int h = 0; h < _hiddenSize; h++)
            {
                streamsOut[baseOffset + h] = streamsIn[baseOffset + h] + blockOut[h] * weight;
            }
        }
    }

    // The block's feed-forward side. A resolved sparse descriptor routes every token through the same all-expert softmax, ascending
    // strict-greater top-k (lower index wins ties), renormalization and scale the production route applies, then each selected
    // expert's SwiGLU chain accumulates in ascending expert-index order scaled by its routing weight; a declared shared expert runs
    // once under the plain sigmoid gate over its hidden-sized selector vector — the qwen4exp reference gate, finite at a zero score
    // where qwen2moe's silu(score)/score divides 0 by 0. A file declaring no experts keeps the dense gate/up/down triplet.
    // Selection carries no cross-token state, so splitting a chunk across calls reproduces the unsplit run exactly.
    private void ForwardFeedForward(LayerOperands w, ReadOnlySpan<float> mixed, Span<float> ffnOut, float[] scratch)
    {
        if (w.Sparse is not { } sparse)
        {
            for (int i = 0; i < _intermediateSize; i++)
            {
                float gate = DotRow(w.DenseGate!, i, _hiddenSize, mixed);
                float up = DotRow(w.DenseUp!, i, _hiddenSize, mixed);
                scratch[i] = LayerOps.Silu(gate) * up;
            }

            for (int h = 0; h < _hiddenSize; h++)
            {
                ffnOut[h] = DotRow(w.DenseDown!, h, _intermediateSize, scratch);
            }

            return;
        }

        int expertCount = sparse.ExpertCount;
        int routed = sparse.RoutedExpertsPerToken;

        float[] scores = new float[expertCount];
        for (int e = 0; e < expertCount; e++)
        {
            scores[e] = DotRow(w.Router!, e, _hiddenSize, mixed);
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
        for (int e = 0; e < expertCount; e++)
        {
            scores[e] = MathF.Exp(scores[e] - maxScore);
            allSum += scores[e];
        }

        for (int e = 0; e < expertCount; e++)
        {
            scores[e] /= allSum;
        }

        int[] ids = new int[routed];
        float[] weights = new float[routed];
        float selectedSum = 0f;
        for (int pick = 0; pick < routed; pick++)
        {
            int best = -1;
            float bestScore = float.NegativeInfinity;
            for (int e = 0; e < expertCount; e++)
            {
                if (scores[e] > bestScore) // strict '>' over ascending indexes: a tie keeps the earlier expert
                {
                    best = e;
                    bestScore = scores[e];
                }
            }

            ids[pick] = best;
            weights[pick] = bestScore;
            selectedSum += bestScore;
            scores[best] = float.NegativeInfinity; // consumed: later passes cannot re-pick this expert
        }

        for (int i = 1; i < routed; i++)
        {
            int id = ids[i];
            float weight = weights[i];
            int j = i - 1;
            while (j >= 0 && ids[j] > id)
            {
                ids[j + 1] = ids[j];
                weights[j + 1] = weights[j];
                j--;
            }

            ids[j + 1] = id;
            weights[j + 1] = weight;
        }

        if (sparse.NormalizeExpertWeights && selectedSum > 0f)
        {
            for (int i = 0; i < routed; i++)
            {
                weights[i] /= selectedSum;
            }
        }

        if (sparse.ExpertWeightsScale != 0f)
        {
            for (int i = 0; i < routed; i++)
            {
                weights[i] *= sparse.ExpertWeightsScale;
            }
        }

        ffnOut.Clear(); // the routed and shared contributions accumulate from zero, mirroring the production sparse posture

        for (int slot = 0; slot < routed; slot++)
        {
            int expert = ids[slot];
            float weight = weights[slot];

            for (int i = 0; i < _intermediateSize; i++)
            {
                int row = expert * _intermediateSize + i;
                float gate = DotRow(w.ExpertGate!, row, _hiddenSize, mixed);
                float up = DotRow(w.ExpertUp!, row, _hiddenSize, mixed);
                scratch[i] = LayerOps.Silu(gate) * up;
            }

            for (int h = 0; h < _hiddenSize; h++)
            {
                ffnOut[h] += weight * DotRow(w.ExpertDown!, expert * _hiddenSize + h, _intermediateSize, scratch);
            }
        }

        if (sparse.HasSharedExpert)
        {
            float score = 0f; // the qwen4exp gate is plain sigmoid(score): finite at a zero score where qwen2moe's silu/score divides 0 by 0
            for (int c = 0; c < _hiddenSize; c++)
            {
                score += w.SharedGateVector![c] * mixed[c];
            }

            float sharedWeight = GdnOps.Sigmoid(score);

            for (int i = 0; i < _intermediateSize; i++)
            {
                float gate = DotRow(w.SharedGate!, i, _hiddenSize, mixed);
                float up = DotRow(w.SharedUp!, i, _hiddenSize, mixed);
                scratch[i] = LayerOps.Silu(gate) * up;
            }

            for (int h = 0; h < _hiddenSize; h++)
            {
                ffnOut[h] += sharedWeight * DotRow(w.SharedDown!, h, _intermediateSize, scratch);
            }
        }
    }

    private void SeedStreams(int tokenId, Span<float> streams)
    {
        if (tokenId < 0 || tokenId >= _plan.VocabSize)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenId), $"token id {tokenId} is outside the vocabulary of {_plan.VocabSize}.");
        }

        _embeddingRows.AsSpan(tokenId * _hiddenSize, _hiddenSize).CopyTo(streams[.._hiddenSize]);
        for (int s = 1; s < _streamCount; s++)
        {
            streams[.._hiddenSize].CopyTo(streams.Slice(s * _hiddenSize, _hiddenSize)); // hc_init repeats the embedding row across every stream
        }

        if (_plan.InputEmbeddingScale != 0f)
        {
            for (int i = 0; i < streams.Length; i++)
            {
                streams[i] *= _plan.InputEmbeddingScale;
            }
        }
    }

    private LayerOperands LoadLayer(int layer, TensorMap map)
    {
        LogicalTensorTable tensors = _fileTensors;
        AttentionKind kind = _geometry.KindForLayer(layer);

        var operands = new LayerOperands
        {
            AttnHyper = LoadHyper(map[TensorRole.AttentionStreamInject], map[TensorRole.AttentionStreamDown], map[TensorRole.AttentionStreamUp], map[TensorRole.AttentionStreamNorm], tensors, layer),
            FfnHyper = LoadHyper(map[TensorRole.FeedForwardStreamInject], map[TensorRole.FeedForwardStreamDown], map[TensorRole.FeedForwardStreamUp], map[TensorRole.FeedForwardStreamNorm], tensors, layer),
        };

        if (kind == AttentionKind.LinearGatedDeltaNet)
        {
            GdnDescriptor gdn = _geometry.LinearAttention;
            operands.Linear = new GdnOps.GdnBlockTensors
            {
                HiddenSize = _hiddenSize,
                KeyHeads = gdn.KeyHeadCount,
                ValueHeads = gdn.ValueHeadCount,
                StateDim = gdn.StateDim,
                ConvKernel = gdn.ConvKernelSize,
                Qkv = LoadMatrix(RequireTensor(map[TensorRole.AttentionQkv], tensors, layer), "fused qkv buffer", [gdn.ConvChannels, _hiddenSize]),
                Gate = LoadMatrix(RequireTensor(map[TensorRole.LinearAttentionGate], tensors, layer), "linear attention gate", [gdn.InnerSize, _hiddenSize]),
                BetaAlphaCat = ConcatRows(
                    LoadMatrix(RequireTensor(map[TensorRole.SsmBetaProjection], tensors, layer), "ssm beta projection", [gdn.TimeStepRank, _hiddenSize]),
                    LoadMatrix(RequireTensor(map[TensorRole.SsmAlphaProjection], tensors, layer), "ssm alpha projection", [gdn.TimeStepRank, _hiddenSize])),
                SsmA = LoadVector(RequireTensor(map[TensorRole.SsmDecayWeight], tensors, layer), "decay weights", gdn.ValueHeadCount),
                DtBias = LoadVector(RequireTensor(map[TensorRole.SsmTimeStepBias], tensors, layer), "dt bias", gdn.ValueHeadCount),
                Conv1d = LoadMatrix(RequireTensor(map[TensorRole.SsmConvolutionKernel], tensors, layer), "conv kernel", [gdn.ConvChannels, gdn.ConvKernelSize]),
                NormWeight = LoadVector(RequireTensor(map[TensorRole.SsmHeadNorm], tensors, layer), "ssm head norm", gdn.StateDim),
                Output = LoadMatrix(RequireTensor(map[TensorRole.SsmOutputProjection], tensors, layer), "ssm output projection", [_hiddenSize, gdn.InnerSize]),
            };
        }
        else
        {
            QsaDescriptor qsa = _geometry.SparseAttention;
            operands.SparseFull = new QsaOps.QsaBlockTensors
            {
                HiddenSize = _hiddenSize,
                HeadCount = _headCount,
                KvHeadCount = _kvHeadCount,
                HeadDim = _headDim,
                IndexerHeads = qsa.IndexerHeadCount,
                IndexerDim = qsa.IndexerKeyLength,
                TopKCells = qsa.TopKCells,
                CompressRatio = _geometry.CompressRatioForLayer(layer),
                QueryGate = LoadMatrix(RequireTensor(map[TensorRole.AttentionQuery], tensors, layer), "query-and-gate rows", [2 * _headCount * _headDim, _hiddenSize]),
                Key = LoadMatrix(RequireTensor(map[TensorRole.AttentionKey], tensors, layer), "key projections", [_kvHeadCount * _headDim, _hiddenSize]),
                Value = LoadMatrix(RequireTensor(map[TensorRole.AttentionValue], tensors, layer), "value projections", [_kvHeadCount * _headDim, _hiddenSize]),
                Output = LoadMatrix(RequireTensor(map[TensorRole.AttentionOutput], tensors, layer), "attention output projection", [_hiddenSize, _headCount * _headDim]),
                IndexerQueryProj = LoadMatrix(RequireTensor(map[TensorRole.IndexerQueryProjection], tensors, layer), "indexer query projections", [qsa.IndexerHeadCount * qsa.IndexerKeyLength, _hiddenSize]),
                IndexerKeyProj = LoadMatrix(RequireTensor(map[TensorRole.IndexerKeyProjection], tensors, layer), "indexer key projection", [qsa.IndexerKeyLength, _hiddenSize]),
                IndexerQueryNorm = LoadVector(RequireTensor(map[TensorRole.IndexerQueryNorm], tensors, layer), "indexer query norm", qsa.IndexerKeyLength),
                IndexerKeyNorm = LoadVector(RequireTensor(map[TensorRole.IndexerKeyNorm], tensors, layer), "indexer key norm", qsa.IndexerKeyLength),
                // The target ships no per-head block norm tensors for this family's split layers, and the plan gate names none: a
                // uniform weight of ones reads as identity, matching what the spike carries when no norm tensor is declared.
                QueryHeadNorm = Ones(_headDim),
                KeyHeadNorm = Ones(_headDim),
            };
        }

        if (_plan.SparseFfn is { } declared)
        {
            int experts = declared.ExpertCount;
            operands.Router = LoadMatrix(RequireTensor(map[declared.RouterRole], tensors, layer), "router matrix", [experts, _hiddenSize]);
            operands.ExpertGate = LoadStack(RequireTensor(map[declared.ExpertGateRole], tensors, layer), "expert gate stack", experts);
            operands.ExpertUp = LoadStack(RequireTensor(map[declared.ExpertUpRole], tensors, layer), "expert up stack", experts);
            operands.ExpertDown = LoadStack(RequireTensor(map[declared.ExpertDownRole], tensors, layer), "expert down stack", experts);

            if (declared.HasSharedExpert)
            {
                TensorDescriptor gateDesc = map[RequiredSparseRole(declared.SharedGateRole, "gate")];
                TensorDescriptor upDesc = map[RequiredSparseRole(declared.SharedUpRole, "up")];
                TensorDescriptor downDesc = map[RequiredSparseRole(declared.SharedDownRole, "down")];

                operands.SharedGate = LoadMatrix(RequireTensor(gateDesc, tensors, layer), "shared expert gate", [_intermediateSize, _hiddenSize]);
                operands.SharedUp = LoadMatrix(RequireTensor(upDesc, tensors, layer), "shared expert up", [_intermediateSize, _hiddenSize]);
                operands.SharedDown = LoadMatrix(RequireTensor(downDesc, tensors, layer), "shared expert down", [_hiddenSize, _intermediateSize]);

                LogicalTensor? selector = declared.SharedInputGateRole is { } gateRole ? map[gateRole].Resolve(tensors, layer) : null;
                operands.SharedGateVector = selector is not null
                    ? LoadVector(selector, "shared expert selector vector", _hiddenSize)
                    : new float[_hiddenSize]; // no selector shipped for this layer: the shared block adds ungated, exactly as the production posture reads it
            }

            operands.Sparse = declared;
        }
        else
        {
            operands.DenseGate = LoadMatrix(RequireTensor(map[TensorRole.FeedForwardGate], tensors, layer), "feed-forward gate", [_intermediateSize, _hiddenSize]);
            operands.DenseUp = LoadMatrix(RequireTensor(map[TensorRole.FeedForwardUp], tensors, layer), "feed-forward up", [_intermediateSize, _hiddenSize]);
            operands.DenseDown = LoadMatrix(RequireTensor(map[TensorRole.FeedForwardDown], tensors, layer), "feed-forward down", [_hiddenSize, _intermediateSize]);
        }

        return operands;
    }

    private HyperConnectionTensors LoadHyper(TensorDescriptor inject, TensorDescriptor down, TensorDescriptor up, TensorDescriptor norm, LogicalTensorTable tensors, int layer) => new()
    {
        Norm = LoadVector(RequireTensor(norm, tensors, layer), "hyper-connection norm", _flattened),
        Down = LoadMatrix(RequireTensor(down, tensors, layer), "hyper-connection down", [_lowRank, _flattened]),
        Up = LoadMatrix(RequireTensor(up, tensors, layer), "hyper-connection up", [_flattened, _lowRank]),
        Inject = LoadMatrix(RequireTensor(inject, tensors, layer), "hyper-connection inject", [_streamCount, _flattened]),
    };

    private static LogicalTensor RequireTensor(TensorDescriptor descriptor, LogicalTensorTable tensors, int layer = -1) =>
        descriptor.Resolve(tensors, layer) ?? throw new ModelException($"tensor '{descriptor.NameFor(layer)}' is missing: the bounded qwen4exp engine cannot run without it");

    private static TensorRole RequiredSparseRole(TensorRole? role, string side) =>
        role ?? throw new ModelException($"the sparse-FFN descriptor declares a shared expert without naming its {side} matrix");

    private static float[] ConcatRows(float[] first, float[] second)
    {
        float[] joined = new float[first.Length + second.Length];
        first.CopyTo(joined, 0);
        second.CopyTo(joined, first.Length);
        return joined;
    }

    private static float[] Ones(int length)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = 1f;
        }

        return values;
    }

    // Reads one F32 tensor's rows as contiguous floats through the backend's row dequantization — this bounded path refuses any other format by name.
    private void RequireF32(LogicalTensor tensor)
    {
        if (tensor.Format != QuantFormat.F32)
        {
            throw new ModelException(
                $"the bounded qwen4exp engine runs F32 fixtures only: tensor '{tensor.Name}' carries format " +
                $"'{tensor.Format.ToString().ToLowerInvariant()}' — a quantized posture needs the production weight views and kernels this synthetic path does not carry.");
        }
    }

    private ReadOnlySpan<byte> ResolveBytes(LogicalTensor tensor) => _weights.Resolve(tensor);

    private float[] LoadVector(LogicalTensor tensor, string what, int expectedLength)
    {
        RequireF32(tensor);
        if (tensor.Shape.Dims.Length != 1 || tensor.Shape.Dims[0] != expectedLength)
        {
            throw new ModelException($"tensor '{tensor.Name}' ({what}) declares shape [{string.Join(", ", tensor.Shape.Dims)}] but the qwen4exp geometry needs a vector of length {expectedLength}");
        }

        ReadOnlySpan<byte> bytes = ResolveBytes(tensor);
        float[] values = new float[expectedLength];
        _backend.DequantRow(new WeightRef(bytes, new TensorShape([1, expectedLength]), tensor.Format), 0, values);
        return values;
    }

    private float[] LoadMatrix(LogicalTensor tensor, string what, int[] logicalRowsCols)
    {
        RequireF32(tensor);
        if (tensor.Shape.Dims.Length != 2 || tensor.Shape.Dims[0] != logicalRowsCols[0] || tensor.Shape.Dims[^1] != logicalRowsCols[1])
        {
            throw new ModelException($"tensor '{tensor.Name}' ({what}) declares shape [{string.Join(", ", tensor.Shape.Dims)}] but the qwen4exp geometry needs [{logicalRowsCols[0]}, {logicalRowsCols[1]}]");
        }

        int rows = logicalRowsCols[0];
        int cols = logicalRowsCols[1];
        ReadOnlySpan<byte> bytes = ResolveBytes(tensor);
        float[] values = new float[checked(rows * (long)cols)];
        TensorShape shape2D = new([rows, cols]);
        for (int r = 0; r < rows; r++)
        {
            _backend.DequantRow(new WeightRef(bytes, shape2D, tensor.Format), r, values.AsSpan(r * cols, cols));
        }

        return values;
    }

    private float[] LoadStack(LogicalTensor tensor, string what, int experts)
    {
        RequireF32(tensor);
        if (tensor.Shape.Dims.Length != 3 || tensor.Shape.Dims[0] != experts)
        {
            throw new ModelException($"tensor '{tensor.Name}' ({what}) must be a three-dimensional expert stack of {experts} experts, got [{string.Join(", ", tensor.Shape.Dims)}]");
        }

        int totalRows = checked(tensor.Shape.Dims[0] * tensor.Shape.Dims[1]);
        int cols = tensor.Shape.Dims[^1];
        ReadOnlySpan<byte> bytes = ResolveBytes(tensor);
        float[] values = new float[checked(totalRows * (long)cols)];
        TensorShape shape2D = new([totalRows, cols]);
        for (int r = 0; r < totalRows; r++)
        {
            _backend.DequantRow(new WeightRef(bytes, shape2D, tensor.Format), r, values.AsSpan(r * cols, cols));
        }

        return values;
    }

    private static float DotRow(float[] matrix, int row, int width, ReadOnlySpan<float> vector)
    {
        double sum = 0.0;
        for (int i = 0; i < width; i++)
        {
            sum += (double)matrix[row * width + i] * vector[i];
        }

        return (float)sum;
    }

    public void Dispose()
    {
        // No unmanaged resources: weight views were read at open and the backend is caller-owned.
    }

    private sealed class LayerOperands
    {
        public HyperConnectionTensors AttnHyper = new();
        public HyperConnectionTensors FfnHyper = new();

        // Exactly one of these attention operand sets is filled, chosen by the layer's resolved kind; a linear block leaves SparseFull
        // empty and a sparse-full block leaves Linear empty. A forward reads only the side its kind selects.
        public GdnOps.GdnBlockTensors Linear;
        public QsaOps.QsaBlockTensors? SparseFull;

        public float[]? Router;
        public float[]? ExpertGate;
        public float[]? ExpertUp;
        public float[]? ExpertDown;
        public float[]? SharedGate;
        public float[]? SharedUp;
        public float[]? SharedDown;
        public float[]? SharedGateVector;
        public SparseFfnDescriptor? Sparse;

        public float[]? DenseGate;
        public float[]? DenseUp;
        public float[]? DenseDown;
    }

    private sealed class HyperConnectionTensors
    {
        public float[] Norm = []; // flattened: each stream's gamma slice
        public float[] Down = []; // lowRank rows x flattened
        public float[] Up = []; // flattened rows x lowRank
        public float[] Inject = []; // streamCount rows x flattened
    }

    private sealed class OutputHyperConnectionTensors
    {
        public float[] Norm = []; // flattened: the final norm vector over the whole flattened width
        public float[] Down = []; // lowRank rows x flattened
        public float[] Up = []; // flattened rows x lowRank
    }
}

/// <summary>
/// One sequence's carried qwen4exp state inside the bounded engine: every linear block's delta-rule slot arrays plus its raw
/// convolution history (with the rotated next-chunk tail), and every sparse-full block's whole-sequence KV rows, raw indexer-key
/// cache, pooled-key scratch, per-pool scores and head scratch. Sized once from the opened context ceiling — this is the
/// engine-local synthetic cache path, not a production serving pool: persistent serving caches, block-pool KV residency and
/// batched prefill windowing remain target work.
/// </summary>
internal sealed class Qwen4ExpSequence
{
    private readonly SequenceGdnState?[] _gdn;
    private readonly SequenceQsaState?[] _qsa;

    internal Qwen4ExpSequence(Qwen4ExpGeometry geometry, int headCount, int kvHeadCount, int headDim, int maxPositions)
    {
        _gdn = new SequenceGdnState?[geometry.LayerCount];
        _qsa = new SequenceQsaState?[geometry.LayerCount];

        for (int layer = 0; layer < geometry.LayerCount; layer++)
        {
            if (geometry.KindForLayer(layer) == AttentionKind.LinearGatedDeltaNet)
            {
                _gdn[layer] = new SequenceGdnState(geometry.LinearAttention, maxPositions);
            }
            else
            {
                _qsa[layer] = new SequenceQsaState(geometry.SparseAttention, kvHeadCount, headDim, headCount, maxPositions);
            }
        }
    }

    /// <summary>Tokens this sequence has already advanced through — the position the next chunk continues at.</summary>
    public int Positions { get; internal set; }

    /// <summary>The final-row logits the last forward produced; sized once the first forward runs.</summary>
    public float[] FinalLogits { get; internal set; } = [];

    internal ReadOnlySpan<float> ConvHistory(int layer) => _gdn[layer]!.ConvHistory;
    internal Span<float> NextConvHistory(int layer) => _gdn[layer]!.NextConvHistory;
    internal Span<float> RecurrentState(int layer) => _gdn[layer]!.Recurrent;
    internal GdnOps.GdnChunkScratch GdnScratch(int layer) => _gdn[layer]!.Scratch;
    internal void RotateConvHistory(int layer) => _gdn[layer]!.RotateConvHistory();
    internal SequenceQsaState Qsa(int layer) => _qsa[layer]!;

    internal sealed class SequenceGdnState
    {
        public SequenceGdnState(GdnDescriptor gdn, int maxTokenCapacity)
        {
            Recurrent = new float[gdn.RecurrentStateValuesPerSequence];
            ConvHistory = new float[gdn.ConvHistoryValuesPerSequence];
            NextConvHistory = new float[gdn.ConvHistoryValuesPerSequence];
            Scratch = GdnOps.GdnChunkScratch.Create(ToBlockTensors(gdn), maxTokenCapacity);
        }

        public float[] Recurrent { get; }
        public float[] ConvHistory { get; private set; }
        public float[] NextConvHistory { get; private set; }
        public GdnOps.GdnChunkScratch Scratch { get; }

        public void RotateConvHistory() => (ConvHistory, NextConvHistory) = (NextConvHistory, ConvHistory);

        private static GdnOps.GdnBlockTensors ToBlockTensors(GdnDescriptor gdn) => new()
        {
            HiddenSize = 0,
            KeyHeads = gdn.KeyHeadCount,
            ValueHeads = gdn.ValueHeadCount,
            StateDim = gdn.StateDim,
            ConvKernel = gdn.ConvKernelSize,
        };
    }

    internal sealed class SequenceQsaState
    {
        public SequenceQsaState(QsaDescriptor qsa, int kvHeadCount, int headDim, int headCount, int capacity)
        {
            int kvRegion = kvHeadCount * headDim;
            KeyCache = new float[capacity * kvRegion];
            ValueCache = new float[capacity * kvRegion];
            RawIndexerKeys = new float[capacity * qsa.IndexerKeyLength];
            PooledKeys = new float[capacity * qsa.IndexerKeyLength];
            Scores = new float[capacity];
            IndexerQuery = new float[qsa.IndexerHeadCount * qsa.IndexerKeyLength];
            RotatedQuery = new float[headCount * headDim];
            AttentionConcat = new float[headCount * headDim];
            QueryHead = new float[headDim];
            KHead = new float[headDim];
            VHead = new float[headDim];
            GateHead = new float[headDim];
        }

        public float[] KeyCache { get; }
        public float[] ValueCache { get; }
        public float[] RawIndexerKeys { get; }
        public float[] PooledKeys { get; }
        public float[] Scores { get; }
        public float[] IndexerQuery { get; }
        public float[] RotatedQuery { get; }
        public float[] QueryHead { get; }
        public float[] KHead { get; }
        public float[] VHead { get; }
        public float[] GateHead { get; }
        public float[] AttentionConcat { get; }
        public int CacheCells { get; set; }
    }
}
