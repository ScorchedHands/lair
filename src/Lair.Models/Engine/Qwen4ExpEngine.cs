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
/// history advance across chunk splits through <see cref="GdnOps.ForwardChunkKernel"/>, and each sparse-full block's whole-sequence KV
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
    private readonly string _modelDisplay = "";

    // Per-block operands captured once at open: one hyper-connection trio per sublayer side, either a gated-delta-net weight-view
    // set (linear layers) or sparse-full attention views plus the indexer quartet (full layers), and the block's routed expert
    // stacks with their router row and per-expert slab geometry. A forward resolves each matrix role's bytes from the weight
    // source and dots them through the backend kernels; only operands with no dense-GEMV shape — norms, biases, decay weights,
    // conv taps, the shared selector vector — stay materialized as small float rows the way production materializes its own.
    private readonly LayerOperands[] _layers;

    // The global output hyper-connection trio: one flattened norm vector plus a final down/up pair, no inject row at this scope.
    private readonly OutputHyperConnectionTensors _outputHc;

    // Logit head rows and the embedding table — untied <c>output.weight</c> when the file ships it, tied token-embedding rows otherwise.
    private readonly LogicalTensor _tokenEmbedding = null!;   // vocabSize x hidden, read through DequantRow per lookup
    private readonly LogicalTensor _logitHead = null!;        // vocabSize x hidden (untied), or the embedding tensor itself when tied

    // The slab seam every routed expert chain runs through: the weight source's own slab views when it answers slab resolutions,
    // a contiguous-slab view of it otherwise — the same posture the production sparse path takes.
    private readonly IExpertSlabSource _expertSlabs = null!;
    private readonly IGatherKernel _gatherKernel = new CpuGatherKernel();

    // Per-token feed-forward scratch sized once from the plan's own widths: the activation row a gathered chain reads, the
    // gate/up/down intermediates, the router scores, and this token's selection.
    private float[] _ffnInput = [];
    private float[] _ffnScratch = [];
    private float[] _ffnGateOut = [];
    private float[] _ffnUpOut = [];
    private float[] _ffnDownOut = [];
    private float[] _routerScores = [];
    private int[] _moeIds = [];
    private float[] _moeWeights = [];
    private float[] _selectedExpertWeights = []; // routing weight by expert index for the current token's gathered chain; only selected entries are read

    // The two-dimensional shape a GEMV over one expert slice carries, shared by every layer of a role the way the production
    // sparse path shares it: gate and up emit intermediate values from hidden inputs, down emits hidden values from intermediates.
    private TensorShape _expertGateShape = new([]);
    private TensorShape _expertUpShape = new([]);
    private TensorShape _expertDownShape = new([]);

    // Reused scratch for the hyper-connection mix and the sparse attention projections: sized once from plan widths so a forward
    // does not allocate per token. The accumulator span borrows whichever output row the current gathered chain writes into; it is
    // assigned immediately before each synchronous gather call, never stored across calls.
    private float[] _hcNormed = [];
    private float[] _hcLowRank = [];
    private float[] _hcGated = [];
    private float[] _queryGateRows = [];
    private float[] _keyProjectionRows = [];
    private float[] _valueProjectionRows = [];
    private float[] _indexerQueryRows = [];
    private float[] _ffnOutputScratch = []; // sparse routed/shared contributions accumulate here, then copy to the caller's output row — a class cannot hold a caller-owned Span field.

    // Rotary tables shared across every block: one position-major cos/sin run for the q/k head dimension and one for the indexer
    // component width, built once at open so no forward rebuilds either.
    private readonly float[] _cosHead;
    private readonly float[] _sinHead;
    private readonly float[] _cosIndexer;
    private readonly float[] _sinIndexer;
    private readonly int _ropeHalfHead;
    private readonly int _ropeHalfIndexer;

    public Qwen4ExpEngine(ModelPlan plan, LogicalTensorTable tensors, IWeightSource weights, IBackend backend, int maxPositions, string modelDisplay)
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

        _modelDisplay = modelDisplay;
        ValidateForwardPosture(plan, tensors);
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

        _expertSlabs = weights as IExpertSlabSource ?? new ExpertSlabWeightSource(weights);

        _ffnInput = new float[_hiddenSize];
        _ffnScratch = new float[_intermediateSize];
        _ffnGateOut = new float[_intermediateSize];
        _ffnUpOut = new float[_intermediateSize];
        _ffnDownOut = new float[_hiddenSize];
        _ffnOutputScratch = new float[_hiddenSize];

        _hcNormed = new float[_flattened];
        _hcLowRank = new float[_lowRank];
        _hcGated = new float[_flattened];

        if (geometry.FullLayerCount > 0)
        {
            int qRegion = _headCount * _headDim;
            int kvRegion = _kvHeadCount * _headDim;
            int indexerWidth = geometry.SparseAttention.IndexerHeadCount * geometry.SparseAttention.IndexerKeyLength;

            _queryGateRows = new float[2 * qRegion];
            _keyProjectionRows = new float[kvRegion];
            _valueProjectionRows = new float[kvRegion];
            _indexerQueryRows = new float[indexerWidth];
        }

        if (plan.SparseFfn is { } sparseDeclared)
        {
            int expertCount = sparseDeclared.ExpertCount;
            int routed = sparseDeclared.RoutedExpertsPerToken;

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(expertCount);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(routed);
            if (routed > expertCount)
            {
                throw new ModelException($"qwen4exp declares {expertCount} experts but routes each token through {routed}; the router cannot select more than it ships");
            }

            _routerScores = new float[expertCount];
            _moeIds = new int[routed];
            _moeWeights = new float[routed];
            _selectedExpertWeights = new float[expertCount];

            _expertGateShape = new TensorShape([_intermediateSize, _hiddenSize]);
            _expertUpShape = new TensorShape([_intermediateSize, _hiddenSize]);
            _expertDownShape = new TensorShape([_hiddenSize, _intermediateSize]);
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
            Down = MatrixRole(RequireTensor(map[TensorRole.OutputStreamDown], tensors), "output hyper-connection down", [_lowRank, _flattened]),
            Up = MatrixRole(RequireTensor(map[TensorRole.OutputStreamUp], tensors), "output hyper-connection up", [_flattened, _lowRank]),
        };

        _tokenEmbedding = RequireTensor(map[TensorRole.TokenEmbedding], tensors);
        ValidateMatrixShape(_tokenEmbedding, "token embedding", [plan.VocabSize, _hiddenSize]);

        LogicalTensor? outputHead = map[TensorRole.OutputHead].Resolve(tensors);
        if (outputHead is not null)
        {
            ValidateMatrixShape(outputHead, "output head", [plan.VocabSize, _hiddenSize]);
        }

        _logitHead = outputHead ?? _tokenEmbedding; // tied posture: the logits read the same rows the embeddings do

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
                GdnOps.ForwardChunkKernel(
                    _backend,
                    _weights,
                    w.KernelLinear,
                    _eps,
                    tokenCount,
                    mixedRows.AsSpan(0, tokenCount * _hiddenSize),
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
                QsaKernelTensors sparseTensors = w.SparseFull ?? throw new InvalidOperationException($"qwen4exp layer {layer} resolved as a sparse-full block but its attention operands were never loaded.");

                for (int t = 0; t < tokenCount; t++)
                {
                    SparseAttention(sparseTensors, sequence.Qsa(layer), mixedRows.AsSpan(t * _hiddenSize, _hiddenSize), sublayerRows.AsSpan(t * _hiddenSize, _hiddenSize));
                }
            }

            for (int t = 0; t < tokenCount; t++)
            {
                CombineStreams(streams.AsSpan(t * streamsPerToken, streamsPerToken), sublayerRows.AsSpan(t * _hiddenSize, _hiddenSize), attnInject.AsSpan(t * _streamCount, _streamCount), midStreams.AsSpan(t * streamsPerToken, streamsPerToken));

                MixStreams(w.FfnHyper, midStreams.AsSpan(t * streamsPerToken, streamsPerToken), mixedRows.AsSpan(t * _hiddenSize, _hiddenSize), injectScratch);

                ForwardFeedForward(w, mixedRows.AsSpan(t * _hiddenSize, _hiddenSize), sublayerRows.AsSpan(t * _hiddenSize, _hiddenSize));

                CombineStreams(midStreams.AsSpan(t * streamsPerToken, streamsPerToken), sublayerRows.AsSpan(t * _hiddenSize, _hiddenSize), injectScratch, streams.AsSpan(t * streamsPerToken, streamsPerToken));
            }
        }

        sequence.Positions += tokenCount;

        float[] logits = new float[tokenCount * _plan.VocabSize];
        for (int t = 0; t < tokenCount; t++)
        {
            MixOutputStreams(streams.AsSpan(t * streamsPerToken, streamsPerToken), mixedRows.AsSpan(t * _hiddenSize, _hiddenSize));
        }

        _backend.GemvBatch(mixedRows.AsSpan(0, tokenCount * _hiddenSize), tokenCount, Ref(_logitHead), logits);

        if (_plan.FinalLogitSoftcap > 0f)
        {
            for (int i = 0; i < logits.Length; i++)
            {
                logits[i] = LayerOps.Softcap(logits[i], _plan.FinalLogitSoftcap);
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
    private void SparseAttention(QsaKernelTensors w, Qwen4ExpSequence.SequenceQsaState state, ReadOnlySpan<float> mixed, Span<float> attnOut)
    {
        QsaOps.QsaBlockTensors geometry = w.Geometry;
        int position = state.CacheCells;
        if (position >= _maxPositions)
        {
            throw new InvalidOperationException($"a qwen4exp sparse block reached cache cell {position}, past this engine's opened context ceiling of {_maxPositions}.");
        }

        int kvRegion = _kvHeadCount * _headDim;
        int idxDim = geometry.IndexerDim;

        // The whole query-and-gate matrix lands in one GEMV pass, exactly as the production dense projections read it: row 2h is head h's
        // query slice and row 2h + 1 its gate slice. Each head then normalizes/rotates only its own query half while its gate stays raw.
        _backend.Gemv(mixed, Ref(w.QueryGate), _queryGateRows);

        for (int h = 0; h < _headCount; h++)
        {
            int qRow = (2 * h) * _headDim;
            int gateRow = (2 * h + 1) * _headDim;

            _queryGateRows.AsSpan(qRow, _headDim).CopyTo(state.QueryHead);
            _queryGateRows.AsSpan(gateRow, _headDim).CopyTo(state.GateHead);

            LayerOps.RmsNormWeightedHeads(state.QueryHead.AsSpan(0, _headDim), geometry.QueryHeadNorm, _headDim, _eps, state.QueryHead.AsSpan(0, _headDim));
            QsaOps.RotateAtPosition(state.QueryHead.AsSpan(0, _headDim), _ropeHalfHead, _cosHead, _sinHead, position);
            state.QueryHead.AsSpan(0, _headDim).CopyTo(state.RotatedQuery.AsSpan(h * _headDim, _headDim));
        }

        _backend.Gemv(mixed, Ref(w.Key), _keyProjectionRows);
        _backend.Gemv(mixed, Ref(w.Value), _valueProjectionRows);

        for (int h = 0; h < _kvHeadCount; h++)
        {
            int rowOffset = h * _headDim;

            _keyProjectionRows.AsSpan(rowOffset, _headDim).CopyTo(state.KHead);
            _valueProjectionRows.AsSpan(rowOffset, _headDim).CopyTo(state.VHead);

            LayerOps.RmsNormWeightedHeads(state.KHead.AsSpan(0, _headDim), geometry.KeyHeadNorm, _headDim, _eps, state.KHead.AsSpan(0, _headDim));
            QsaOps.RotateAtPosition(state.KHead.AsSpan(0, _headDim), _ropeHalfHead, _cosHead, _sinHead, position);

            state.KHead.AsSpan(0, _headDim).CopyTo(state.KeyCache.AsSpan(position * kvRegion + rowOffset, _headDim));
            state.VHead.AsSpan(0, _headDim).CopyTo(state.ValueCache.AsSpan(position * kvRegion + rowOffset, _headDim));
        }

        int indexerWidth = geometry.IndexerHeads * idxDim;
        if (indexerWidth > 0)
        {
            _backend.Gemv(mixed, Ref(w.IndexerQueryProj), _indexerQueryRows);
            _indexerQueryRows.AsSpan(0, indexerWidth).CopyTo(state.IndexerQuery.AsSpan(0, indexerWidth));

            LayerOps.RmsNormWeightedHeads(state.IndexerQuery.AsSpan(0, indexerWidth), geometry.IndexerQueryNorm, idxDim, _eps, state.IndexerQuery.AsSpan(0, indexerWidth));

            for (int h = 0; h < geometry.IndexerHeads; h++)
            {
                QsaOps.RotateAtPosition(state.IndexerQuery.AsSpan(h * idxDim, idxDim), _ropeHalfIndexer, _cosIndexer, _sinIndexer, position);
            }
        }

        if (idxDim > 0)
        {
            _backend.Gemv(mixed, Ref(w.IndexerKeyProj), state.RawIndexerKeys.AsSpan(position * idxDim, idxDim));
        }

        state.CacheCells = position + 1;
        int cacheCells = state.CacheCells;
        float attnScale = 1f / MathF.Sqrt((float)_headDim);

        if (geometry.CompressRatio < 2 || cacheCells / geometry.CompressRatio == 0)
        {
            // A ratio that groups fewer than two cells runs dense and never reaches selection; a cache shorter than one complete pool has nothing to rank. Either way every cached cell stays visible.
            List<int> visible = [];
            for (int cell = 0; cell < cacheCells; cell++)
            {
                visible.Add(cell);
            }

            QsaOps.RestrictedAttention(_backend, state.RotatedQuery, state.KeyCache, state.ValueCache, visible, cacheCells, geometry, attnScale, state.AttentionConcat);
        }
        else
        {
            int fullPools = cacheCells / geometry.CompressRatio;
            QsaOps.PoolKeys(state.RawIndexerKeys.AsSpan(0, cacheCells * idxDim), cacheCells, geometry, _eps, state.PooledKeys.AsSpan(0, fullPools * idxDim));

            for (int p = 0; p < fullPools; p++)
            {
                QsaOps.RotateAtPosition(state.PooledKeys.AsSpan(p * idxDim, idxDim), _ropeHalfIndexer, _cosIndexer, _sinIndexer, p * geometry.CompressRatio);
            }

            QsaOps.IndexerScores(state.IndexerQuery.AsSpan(0, indexerWidth), state.PooledKeys.AsSpan(0, fullPools * idxDim), geometry.IndexerHeads, idxDim, fullPools, state.Scores.AsSpan(0, fullPools));
            List<int> selected = QsaOps.SelectCells(state.Scores.AsSpan(0, fullPools), cacheCells, geometry);
            QsaOps.RestrictedAttention(_backend, state.RotatedQuery, state.KeyCache, state.ValueCache, selected, cacheCells, geometry, attnScale, state.AttentionConcat);
        }

        for (int h = 0; h < _headCount; h++)
        {
            for (int j = 0; j < _headDim; j++)
            {
                state.AttentionConcat[h * _headDim + j] *= GdnOps.Sigmoid(state.GateHead[j]);
            }
        }

        _backend.Gemv(state.AttentionConcat, Ref(w.Output), attnOut);
    }

    // The reference build_hc_mix: grouped RMS normalization per stream scaled by that stream's slice of the flattened norm vector,
    // the flattened streams through the low-rank down projection scaled by 1/streams and SiLU, expansion through the up projection
    // to a per-stream gate, gating the normalized streams by sigmoid of that gate, and collapse by their mean to the hidden row the
    // sublayer consumes. The same normalized streams feed the inject projection the combine weights the residual scatter with.
    private void MixStreams(HyperConnectionTensors hc, ReadOnlySpan<float> streams, Span<float> mixed, Span<float> inject)
    {
        for (int s = 0; s < _streamCount; s++)
        {
            int baseOffset = s * _hiddenSize;
            LayerOps.RmsNormWeighted(streams.Slice(baseOffset, _hiddenSize), hc.Norm.AsSpan(baseOffset, _hiddenSize), _eps, _hcNormed.AsSpan(baseOffset, _hiddenSize));
        }

        float invStreams = 1f / _streamCount;

        _backend.Gemv(_hcNormed, Ref(hc.Down), _hcLowRank);
        for (int r = 0; r < _lowRank; r++)
        {
            _hcLowRank[r] = LayerOps.Silu(_hcLowRank[r] * invStreams);
        }

        _backend.Gemv(_hcLowRank, Ref(hc.Up), _hcGated);
        for (int f = 0; f < _flattened; f++)
        {
            float gate = _hcGated[f];
            _hcGated[f] = _hcNormed[f] * GdnOps.Sigmoid(gate);
        }

        for (int h = 0; h < _hiddenSize; h++)
        {
            double sum = 0.0;
            for (int s = 0; s < _streamCount; s++)
            {
                sum += _hcGated[s * _hiddenSize + h];
            }

            mixed[h] = (float)(sum * invStreams);
        }

        _backend.Gemv(_hcNormed, Ref(hc.Inject), inject);
        for (int s = 0; s < _streamCount; s++)
        {
            inject[s] *= invStreams; // the reference scales each stream's low-rank expansion by 1/streams before its combine scatter weights it with 2·sigmoid
        }
    }

    // The global output-side mix: the same grouped norm / low-rank SiLU / sigmoid gate / mean collapse as every block-side mix, but
    // over the output_hc_* trio and with no inject row — the conversion ships only that trio globally.
    private void MixOutputStreams(ReadOnlySpan<float> streams, Span<float> mixed)
    {
        for (int s = 0; s < _streamCount; s++)
        {
            int baseOffset = s * _hiddenSize;
            LayerOps.RmsNormWeighted(streams.Slice(baseOffset, _hiddenSize), _outputHc.Norm.AsSpan(baseOffset, _hiddenSize), _eps, _hcNormed.AsSpan(baseOffset, _hiddenSize));
        }

        float invStreams = 1f / _streamCount;

        _backend.Gemv(_hcNormed, Ref(_outputHc.Down), _hcLowRank);
        for (int r = 0; r < _lowRank; r++)
        {
            _hcLowRank[r] = LayerOps.Silu(_hcLowRank[r] * invStreams);
        }

        _backend.Gemv(_hcLowRank, Ref(_outputHc.Up), _hcGated);

        for (int h = 0; h < _hiddenSize; h++)
        {
            double sum = 0.0;
            for (int s = 0; s < _streamCount; s++)
            {
                int f = s * _hiddenSize + h;
                float gate = _hcGated[f];
                sum += _hcNormed[f] * GdnOps.Sigmoid(gate);
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
    private void ForwardFeedForward(LayerOperands w, ReadOnlySpan<float> mixed, Span<float> ffnOut)
    {
        if (w.Sparse is not { } sparse)
        {
            _backend.Gemv(mixed, Ref(w.DenseGate!), _ffnGateOut);
            _backend.Gemv(mixed, Ref(w.DenseUp!), _ffnUpOut);

            for (int i = 0; i < _intermediateSize; i++)
            {
                _ffnScratch[i] = LayerOps.Silu(_ffnGateOut[i]) * _ffnUpOut[i];
            }

            _backend.Gemv(_ffnScratch, Ref(w.DenseDown!), ffnOut);
            return;
        }

        int expertCount = sparse.ExpertCount;
        int routed = sparse.RoutedExpertsPerToken;

        mixed.CopyTo(_ffnInput); // the gathered chain cannot capture a caller-owned span across its callback boundary: this token's row is copied into reusable engine scratch first.

        _backend.Gemv(_ffnInput, Ref(w.Router!), _routerScores.AsSpan(0, expertCount));

        float maxScore = float.NegativeInfinity;
        for (int e = 0; e < expertCount; e++)
        {
            if (_routerScores[e] > maxScore)
            {
                maxScore = _routerScores[e];
            }
        }

        float allSum = 0f;
        for (int e = 0; e < expertCount; e++)
        {
            _routerScores[e] = MathF.Exp(_routerScores[e] - maxScore);
            allSum += _routerScores[e];
        }

        for (int e = 0; e < expertCount; e++)
        {
            _routerScores[e] /= allSum;
        }

        float selectedSum = 0f;
        for (int pick = 0; pick < routed; pick++)
        {
            int best = -1;
            float bestScore = float.NegativeInfinity;
            for (int e = 0; e < expertCount; e++)
            {
                if (_routerScores[e] > bestScore) // strict '>' over ascending indexes: a tie keeps the earlier expert
                {
                    best = e;
                    bestScore = _routerScores[e];
                }
            }

            _moeIds[pick] = best;
            _moeWeights[pick] = bestScore;
            selectedSum += bestScore;
            _routerScores[best] = float.NegativeInfinity; // consumed: later passes cannot re-pick this expert
        }

        for (int i = 1; i < routed; i++)
        {
            int id = _moeIds[i];
            float weight = _moeWeights[i];
            int j = i - 1;
            while (j >= 0 && _moeIds[j] > id)
            {
                _moeIds[j + 1] = _moeIds[j];
                _moeWeights[j + 1] = _moeWeights[j];
                j--;
            }

            _moeIds[j + 1] = id;
            _moeWeights[j + 1] = weight;
        }

        if (sparse.NormalizeExpertWeights && selectedSum > 0f)
        {
            for (int i = 0; i < routed; i++)
            {
                _moeWeights[i] /= selectedSum;
            }
        }

        if (sparse.ExpertWeightsScale != 0f)
        {
            for (int i = 0; i < routed; i++)
            {
                _moeWeights[i] *= sparse.ExpertWeightsScale;
            }
        }

        ExpertSlabRole gate = w.GateRole;
        ExpertSlabRole up = w.UpRole;
        ExpertSlabRole down = w.DownRole;

        for (int i = 0; i < routed; i++)
        {
            _selectedExpertWeights[_moeIds[i]] = _moeWeights[i]; // the gather callback visits expert indexes, not selection slots: this maps each selected slab back to its routing weight.
        }

        _ffnOutputScratch.AsSpan().Clear(); // the routed and shared contributions accumulate from zero, mirroring the production sparse posture

        _gatherKernel.Gather(
            _expertSlabs,
            gate,
            up,
            down,
            _moeIds.AsSpan(0, routed),
            (int expertIndex, ReadOnlySpan<byte> gateSlab, ReadOnlySpan<byte> upSlab, ReadOnlySpan<byte> downSlab) =>
            {
                float weight = _selectedExpertWeights[expertIndex];

                _backend.Gemv(_ffnInput, new WeightRef(gateSlab, gate.ExpertShape, gate.Stack.Format), _ffnGateOut);
                _backend.Gemv(_ffnInput, new WeightRef(upSlab, up.ExpertShape, up.Stack.Format), _ffnUpOut);

                for (int i = 0; i < _intermediateSize; i++)
                {
                    _ffnScratch[i] = LayerOps.Silu(_ffnGateOut[i]) * _ffnUpOut[i];
                }

                _backend.Gemv(_ffnScratch, new WeightRef(downSlab, down.ExpertShape, down.Stack.Format), _ffnDownOut);

                for (int h = 0; h < _hiddenSize; h++)
                {
                    _ffnOutputScratch[h] += weight * _ffnDownOut[h]; // ascending expert-index order: the gather kernel's supplied selection is this token's sorted routing set.
                }
            });

        if (sparse.HasSharedExpert)
        {
            float score = 0f; // the qwen4exp gate is plain sigmoid(score): finite at a zero score where qwen2moe's silu/score divides 0 by 0
            for (int c = 0; c < _hiddenSize; c++)
            {
                score += w.SharedGateVector![c] * mixed[c];
            }

            float sharedWeight = GdnOps.Sigmoid(score);

            _backend.Gemv(mixed, Ref(w.SharedGate!), _ffnGateOut);
            _backend.Gemv(mixed, Ref(w.SharedUp!), _ffnUpOut);

            for (int i = 0; i < _intermediateSize; i++)
            {
                _ffnScratch[i] = LayerOps.Silu(_ffnGateOut[i]) * _ffnUpOut[i];
            }

            _backend.Gemv(_ffnScratch, Ref(w.SharedDown!), _ffnDownOut);

            for (int h = 0; h < _hiddenSize; h++)
            {
                _ffnOutputScratch[h] += sharedWeight * _ffnDownOut[h];
            }
        }

        _ffnOutputScratch.AsSpan(0, _hiddenSize).CopyTo(ffnOut); // the caller's sublayer row receives the accumulated sparse contribution once the gather callback is done with its scratch.
    }

    private void SeedStreams(int tokenId, Span<float> streams)
    {
        if (tokenId < 0 || tokenId >= _plan.VocabSize)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenId), $"token id {tokenId} is outside the vocabulary of {_plan.VocabSize}.");
        }

        _backend.DequantRow(Ref(_tokenEmbedding), tokenId, streams[.._hiddenSize]); // the embedding row is one packed table row read through the backend's own dequantization, not a resident F32 copy of the whole table.
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
            LogicalTensor qkv = MatrixRole(RequireTensor(map[TensorRole.AttentionQkv], tensors, layer), "fused qkv buffer", [gdn.ConvChannels, _hiddenSize]);
            LogicalTensor gate = MatrixRole(RequireTensor(map[TensorRole.LinearAttentionGate], tensors, layer), "linear attention gate", [gdn.InnerSize, _hiddenSize]);
            LogicalTensor beta = MatrixRole(RequireTensor(map[TensorRole.SsmBetaProjection], tensors, layer), "ssm beta projection", [gdn.TimeStepRank, _hiddenSize]);
            LogicalTensor alpha = MatrixRole(RequireTensor(map[TensorRole.SsmAlphaProjection], tensors, layer), "ssm alpha projection", [gdn.TimeStepRank, _hiddenSize]);
            LogicalTensor output = MatrixRole(RequireTensor(map[TensorRole.SsmOutputProjection], tensors, layer), "ssm output projection", [_hiddenSize, gdn.InnerSize]);

            operands.KernelLinear = new GdnOps.GdnKernelTensors
            {
                HiddenSize = _hiddenSize,
                KeyHeads = gdn.KeyHeadCount,
                ValueHeads = gdn.ValueHeadCount,
                StateDim = gdn.StateDim,
                ConvKernel = gdn.ConvKernelSize,
                Qkv = qkv,
                Gate = gate,
                Beta = beta,
                Alpha = alpha,
                Output = output,
                Conv1d = LoadMatrixValues(RequireTensor(map[TensorRole.SsmConvolutionKernel], tensors, layer), "conv kernel", [gdn.ConvChannels, gdn.ConvKernelSize]),
                SsmA = LoadVector(RequireTensor(map[TensorRole.SsmDecayWeight], tensors, layer), "decay weights", gdn.ValueHeadCount),
                DtBias = LoadVector(RequireTensor(map[TensorRole.SsmTimeStepBias], tensors, layer), "dt bias", gdn.ValueHeadCount),
                NormWeight = LoadVector(RequireTensor(map[TensorRole.SsmHeadNorm], tensors, layer), "ssm head norm", gdn.StateDim),
            };
        }
        else
        {
            QsaDescriptor qsa = _geometry.SparseAttention;
            LogicalTensor queryGate = MatrixRole(RequireTensor(map[TensorRole.AttentionQuery], tensors, layer), "query-and-gate rows", [2 * _headCount * _headDim, _hiddenSize]);
            LogicalTensor key = MatrixRole(RequireTensor(map[TensorRole.AttentionKey], tensors, layer), "key projections", [_kvHeadCount * _headDim, _hiddenSize]);
            LogicalTensor value = MatrixRole(RequireTensor(map[TensorRole.AttentionValue], tensors, layer), "value projections", [_kvHeadCount * _headDim, _hiddenSize]);
            LogicalTensor attnOutput = MatrixRole(RequireTensor(map[TensorRole.AttentionOutput], tensors, layer), "attention output projection", [_hiddenSize, _headCount * _headDim]);
            LogicalTensor indexerQuery = MatrixRole(RequireTensor(map[TensorRole.IndexerQueryProjection], tensors, layer), "indexer query projections", [qsa.IndexerHeadCount * qsa.IndexerKeyLength, _hiddenSize]);
            LogicalTensor indexerKey = MatrixRole(RequireTensor(map[TensorRole.IndexerKeyProjection], tensors, layer), "indexer key projection", [qsa.IndexerKeyLength, _hiddenSize]);

            operands.SparseFull = new QsaKernelTensors
            {
                Geometry = new QsaOps.QsaBlockTensors
                {
                    HiddenSize = _hiddenSize,
                    HeadCount = _headCount,
                    KvHeadCount = _kvHeadCount,
                    HeadDim = _headDim,
                    IndexerHeads = qsa.IndexerHeadCount,
                    IndexerDim = qsa.IndexerKeyLength,
                    TopKCells = qsa.TopKCells,
                    CompressRatio = _geometry.CompressRatioForLayer(layer),
                    IndexerQueryNorm = LoadVector(RequireTensor(map[TensorRole.IndexerQueryNorm], tensors, layer), "indexer query norm", qsa.IndexerKeyLength),
                    IndexerKeyNorm = LoadVector(RequireTensor(map[TensorRole.IndexerKeyNorm], tensors, layer), "indexer key norm", qsa.IndexerKeyLength),
                    // The target ships no per-head block norm tensors for this family's split layers, and the plan gate names none: a
                    // uniform weight of ones reads as identity, matching what the spike carries when no norm tensor is declared.
                    QueryHeadNorm = Ones(_headDim),
                    KeyHeadNorm = Ones(_headDim),
                },
                QueryGate = queryGate,
                Key = key,
                Value = value,
                Output = attnOutput,
                IndexerQueryProj = indexerQuery,
                IndexerKeyProj = indexerKey,
            };
        }

        if (_plan.SparseFfn is { } declared)
        {
            int experts = declared.ExpertCount;
            operands.Router = MatrixRole(RequireTensor(map[declared.RouterRole], tensors, layer), "router matrix", [experts, _hiddenSize]);

            LogicalTensor gateStack = RequireTensor(map[declared.ExpertGateRole], tensors, layer);
            LogicalTensor upStack = RequireTensor(map[declared.ExpertUpRole], tensors, layer);
            LogicalTensor downStack = RequireTensor(map[declared.ExpertDownRole], tensors, layer);

            ValidateStackShape(gateStack, "expert gate stack", experts, _intermediateSize, _hiddenSize);
            ValidateStackShape(upStack, "expert up stack", experts, _intermediateSize, _hiddenSize);
            ValidateStackShape(downStack, "expert down stack", experts, _hiddenSize, _intermediateSize);

            operands.GateRole = new ExpertSlabRole(gateStack, _expertGateShape, ExpertSlabBytes(gateStack, _intermediateSize, experts));
            operands.UpRole = new ExpertSlabRole(upStack, _expertUpShape, ExpertSlabBytes(upStack, _intermediateSize, experts));
            operands.DownRole = new ExpertSlabRole(downStack, _expertDownShape, ExpertSlabBytes(downStack, _hiddenSize, experts));

            if (declared.HasSharedExpert)
            {
                TensorDescriptor gateDesc = map[RequiredSparseRole(declared.SharedGateRole, "gate")];
                TensorDescriptor upDesc = map[RequiredSparseRole(declared.SharedUpRole, "up")];
                TensorDescriptor downDesc = map[RequiredSparseRole(declared.SharedDownRole, "down")];

                operands.SharedGate = MatrixRole(RequireTensor(gateDesc, tensors, layer), "shared expert gate", [_intermediateSize, _hiddenSize]);
                operands.SharedUp = MatrixRole(RequireTensor(upDesc, tensors, layer), "shared expert up", [_intermediateSize, _hiddenSize]);
                operands.SharedDown = MatrixRole(RequireTensor(downDesc, tensors, layer), "shared expert down", [_hiddenSize, _intermediateSize]);

                LogicalTensor? selector = declared.SharedInputGateRole is { } gateRole ? map[gateRole].Resolve(tensors, layer) : null;
                operands.SharedGateVector = selector is not null
                    ? LoadVector(selector, "shared expert selector vector", _hiddenSize)
                    : new float[_hiddenSize]; // no selector shipped for this layer: the shared block adds ungated, exactly as the production posture reads it
            }

            operands.Sparse = declared;
        }
        else
        {
            operands.DenseGate = MatrixRole(RequireTensor(map[TensorRole.FeedForwardGate], tensors, layer), "feed-forward gate", [_intermediateSize, _hiddenSize]);
            operands.DenseUp = MatrixRole(RequireTensor(map[TensorRole.FeedForwardUp], tensors, layer), "feed-forward up", [_intermediateSize, _hiddenSize]);
            operands.DenseDown = MatrixRole(RequireTensor(map[TensorRole.FeedForwardDown], tensors, layer), "feed-forward down", [_hiddenSize, _intermediateSize]);
        }

        return operands;
    }

    private HyperConnectionTensors LoadHyper(TensorDescriptor inject, TensorDescriptor down, TensorDescriptor up, TensorDescriptor norm, LogicalTensorTable tensors, int layer) => new()
    {
        Norm = LoadVector(RequireTensor(norm, tensors, layer), "hyper-connection norm", _flattened),
        Down = MatrixRole(RequireTensor(down, tensors, layer), "hyper-connection down", [_lowRank, _flattened]),
        Up = MatrixRole(RequireTensor(up, tensors, layer), "hyper-connection up", [_flattened, _lowRank]),
        Inject = MatrixRole(RequireTensor(inject, tensors, layer), "hyper-connection inject", [_streamCount, _flattened]),
    };

    private static LogicalTensor RequireTensor(TensorDescriptor descriptor, LogicalTensorTable tensors, int layer = -1) =>
        descriptor.Resolve(tensors, layer) ?? throw new ModelException($"tensor '{descriptor.NameFor(layer)}' is missing: the bounded qwen4exp engine cannot run without it");

    private static TensorRole RequiredSparseRole(TensorRole? role, string side) =>
        role ?? throw new ModelException($"the sparse-FFN descriptor declares a shared expert without naming its {side} matrix");

    private void ValidateForwardPosture(ModelPlan plan, LogicalTensorTable tensors)
    {
        if (plan.RopeScaling is { } scaling)
        {
            throw PostureRefusal($"rope scaling variant '{scaling.Variant}' with multiplier {scaling.Multiplier} and attention factor {scaling.AttentionFactor}");
        }

        if (plan.RopeRotaryDimensions != 0)
        {
            throw PostureRefusal($"partial rotary dimensions {plan.RopeRotaryDimensions}; this seam rotates each head in full");
        }

        if (plan.AttentionWindow > 0)
        {
            throw PostureRefusal($"attention window {plan.AttentionWindow} under interleave '{plan.WindowInterleave}'");
        }

        if (plan.AttentionSoftcap > 0f)
        {
            throw PostureRefusal($"attention softcap {plan.AttentionSoftcap}");
        }

        if (plan.UseAttentionBias)
        {
            throw PostureRefusal("attention projection biases");
        }

        if (plan.UseNormBias)
        {
            throw PostureRefusal("norm bias vectors");
        }

        if (plan.QkNorm is QkNormMode.PerHeadWeighted or QkNormMode.StreamWeighted)
        {
            throw PostureRefusal($"weighted QK norm mode '{plan.QkNorm}'; this seam carries only unweighted head norms");
        }

        if (plan.SparseFfn is { } sparse)
        {
            if (sparse.Layout != SparseFfnLayout.StackedExperts)
            {
                throw PostureRefusal($"sparse-FFN expert layout '{sparse.Layout}'");
            }

            if (plan.SparseFfnStrategy is { } strategy && strategy != SparseFfnStrategy.DenseResident)
            {
                throw PostureRefusal($"sparse-FFN strategy '{strategy}'");
            }
        }

        TensorRole[] pleRoles = new TensorRole[]
        {
            TensorRole.PleConvolutionKernel,
            TensorRole.PleKeyProjection,
            TensorRole.PleValueProjection,
            TensorRole.PleConvolutionNorm,
            TensorRole.PleKeyNorm,
            TensorRole.PleQueryNorm,
        };

        TensorRole[] attentionBiasRoles = new TensorRole[]
        {
            TensorRole.AttentionQueryBias,
            TensorRole.AttentionKeyBias,
            TensorRole.AttentionValueBias,
            TensorRole.AttentionOutputBias,
        };

        TensorRole[] normBiasRoles = new TensorRole[]
        {
            TensorRole.AttentionNormBias,
            TensorRole.FeedForwardNormBias,
        };

        for (int layer = 0; layer < plan.LayerCount; layer++)
        {
            foreach (TensorRole role in pleRoles)
            {
                if (HasRole(plan.Tensors, tensors, role, layer))
                {
                    throw PostureRefusal($"PLE n-gram module tensor '{plan.Tensors[role].NameFor(layer)}'");
                }
            }

            foreach (TensorRole role in attentionBiasRoles)
            {
                if (HasRole(plan.Tensors, tensors, role, layer))
                {
                    throw PostureRefusal($"attention bias tensor '{plan.Tensors[role].NameFor(layer)}'");
                }
            }

            foreach (TensorRole role in normBiasRoles)
            {
                if (HasRole(plan.Tensors, tensors, role, layer))
                {
                    throw PostureRefusal($"norm bias tensor '{plan.Tensors[role].NameFor(layer)}'");
                }
            }
        }

        if (HasRole(plan.Tensors, tensors, TensorRole.OutputNormBias, -1))
        {
            throw PostureRefusal($"output norm bias tensor '{plan.Tensors[TensorRole.OutputNormBias].NameFor(-1)}'");
        }
    }

    private ModelException PostureRefusal(string posture) => new(
        $"model '{_modelDisplay}' architecture 'qwen4exp' bounded forward has no composition over {posture}: run a fixture that omits it, or resolve a family whose forward composes that posture");

    private static bool HasRole(TensorMap map, LogicalTensorTable tensors, TensorRole role, int layer) =>
        map[role].Resolve(tensors, layer) is not null;

    private static float[] Ones(int length)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = 1f;
        }

        return values;
    }

    private ReadOnlySpan<byte> ResolveBytes(LogicalTensor tensor) => _weights.Resolve(tensor);

    private WeightRef Ref(LogicalTensor tensor) => new(ResolveBytes(tensor), tensor.Shape, tensor.Format);

    private LogicalTensor MatrixRole(LogicalTensor tensor, string what, int[] logicalRowsCols)
    {
        ValidateMatrixShape(tensor, what, logicalRowsCols);
        return tensor;
    }

    private static void ValidateMatrixShape(LogicalTensor tensor, string what, int[] logicalRowsCols)
    {
        if (tensor.Shape.Dims.Length != 2 || tensor.Shape.Dims[0] != logicalRowsCols[0] || tensor.Shape.Dims[^1] != logicalRowsCols[1])
        {
            throw new ModelException($"tensor '{tensor.Name}' ({what}) declares shape [{string.Join(", ", tensor.Shape.Dims)}] but the qwen4exp geometry needs [{logicalRowsCols[0]}, {logicalRowsCols[1]}]");
        }
    }

    private static void ValidateStackShape(LogicalTensor tensor, string what, int experts, int rows, int cols)
    {
        if (tensor.Shape.Dims.Length != 3 || tensor.Shape.Dims[0] != experts || tensor.Shape.Dims[1] != rows || tensor.Shape.Dims[2] != cols)
        {
            throw new ModelException($"tensor '{tensor.Name}' ({what}) must be a three-dimensional expert stack of {experts} experts shaped [{experts}, {rows}, {cols}], got [{string.Join(", ", tensor.Shape.Dims)}]");
        }
    }

    private static long ExpertSlabBytes(LogicalTensor tensor, int rows, int experts)
    {
        long expertBytes = checked((long)rows * tensor.Shape.RowStrideBytes(tensor.Format));
        long stackBytes = checked((long)experts * expertBytes);
        if (expertBytes <= 0 || tensor.SizeBytes != stackBytes)
        {
            throw new ModelException($"tensor '{tensor.Name}' declares {tensor.SizeBytes} bytes for {experts} expert slabs of {expertBytes} bytes each; the bounded qwen4exp slab seam needs contiguous stacked experts");
        }

        return expertBytes;
    }

    private float[] LoadVector(LogicalTensor tensor, string what, int expectedLength)
    {
        if (tensor.Shape.Dims.Length != 1 || tensor.Shape.Dims[0] != expectedLength)
        {
            throw new ModelException($"tensor '{tensor.Name}' ({what}) declares shape [{string.Join(", ", tensor.Shape.Dims)}] but the qwen4exp geometry needs a vector of length {expectedLength}");
        }

        ReadOnlySpan<byte> bytes = ResolveBytes(tensor);
        float[] values = new float[expectedLength];
        _backend.DequantRow(new WeightRef(bytes, new TensorShape([1, expectedLength]), tensor.Format), 0, values);
        return values;
    }

    private float[] LoadMatrixValues(LogicalTensor tensor, string what, int[] logicalRowsCols)
    {
        ValidateMatrixShape(tensor, what, logicalRowsCols);

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

    public void Dispose()
    {
        // No unmanaged resources: weight views were read at open and the backend is caller-owned.
    }

    private sealed class LayerOperands
    {
        public HyperConnectionTensors AttnHyper = new();
        public HyperConnectionTensors FfnHyper = new();

        // Exactly one of these attention operand sets is filled, chosen by the layer's resolved kind; a linear block carries its GDN
        // weight views and a sparse-full block carries its QSA weight views plus the geometry/norms QsaOps consumes. A forward reads
        // only the side its kind selects.
        public GdnOps.GdnKernelTensors KernelLinear;
        public QsaKernelTensors? SparseFull;

        public LogicalTensor? Router;
        public ExpertSlabRole GateRole;
        public ExpertSlabRole UpRole;
        public ExpertSlabRole DownRole;
        public LogicalTensor? SharedGate;
        public LogicalTensor? SharedUp;
        public LogicalTensor? SharedDown;
        public float[]? SharedGateVector;
        public SparseFfnDescriptor? Sparse;

        public LogicalTensor? DenseGate;
        public LogicalTensor? DenseUp;
        public LogicalTensor? DenseDown;
    }

    private struct QsaKernelTensors
    {
        public QsaKernelTensors()
        {
        }

        public QsaOps.QsaBlockTensors Geometry; // the scalar/norm side QsaOps consumes; its projection arrays stay empty because this engine projects through the backend instead

        public LogicalTensor QueryGate = null!; // 2 * HeadCount * HeadDim rows x HiddenSize: every head's query slice then its gate slice, alternating per head
        public LogicalTensor Key = null!; // KvHeadCount * HeadDim rows x HiddenSize
        public LogicalTensor Value = null!; // same shape as Key
        public LogicalTensor Output = null!; // HiddenSize rows x (HeadCount * HeadDim)
        public LogicalTensor IndexerQueryProj = null!; // IndexerHeads * IndexerDim rows x HiddenSize
        public LogicalTensor IndexerKeyProj = null!; // IndexerDim rows x HiddenSize
    }

    private sealed class HyperConnectionTensors
    {
        public float[] Norm = []; // flattened: each stream's gamma slice, materialized because RMS scaling has no dense-GEMV shape
        public LogicalTensor Down = null!; // lowRank rows x flattened
        public LogicalTensor Up = null!; // flattened rows x lowRank
        public LogicalTensor Inject = null!; // streamCount rows x flattened
    }

    private sealed class OutputHyperConnectionTensors
    {
        public float[] Norm = []; // flattened: the final norm vector over the whole flattened width
        public LogicalTensor Down = null!; // lowRank rows x flattened
        public LogicalTensor Up = null!; // flattened rows x lowRank
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
