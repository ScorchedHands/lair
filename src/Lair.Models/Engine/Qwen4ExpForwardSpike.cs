using Lair.Contracts;

namespace Lair.Models.Engine;

/// <summary>
/// Bounded CPU-only spike that sequences one synthetic qwen4exp block forward from a resolved <see cref="Qwen4ExpGeometry"/>
/// alone: hyper-connection stream mixing before the attention sublayer, the layer's own attention (a gated-delta-net chunk
/// through <see cref="GdnOps"/> on a linear block, or indexer-selected restricted attention through <see cref="QsaOps"/> on a
/// sparse-full block), the attention output projection, hyper-connection residual mixing after the sublayer, then the same
/// hyper-connection pair around a clearly-labelled dense SwiGLU stand-in for the block's sparse MoE. It composes exactly the
/// reference graph's per-block sequence — <c>build_hc_mix → attention → build_hc_combine → build_hc_mix → ffn →
/// build_hc_combine</c> — over synthetic operands sized straight from the geometry, never a model file, so it proves the
/// resolved interleave, recurrent-state geometry, indexer budgets and hyper-connection shape can drive a real block forward
/// without opening the 70 GB Flash-Next target. This is a composition spike, not target serving: the sparse router, gather
/// and expert residency are replaced by a dense stand-in, and the production <see cref="TransformerModel"/> open refusal
/// stays in place because nothing here loads GGUF tensors or claims full-family parity.
///
/// The hyper-connection math mirrors the reference exactly. <c>build_hc_mix</c> groups each residual stream through its own
/// RMS normalization scaled by that stream's slice of the flattened norm vector, runs the flattened stream through the
/// low-rank down projection scaled by 1/streams and SiLU, expands it through the up projection to a per-stream gate, gates
/// the normalized streams by sigmoid of that gate, and collapses the streams by their mean to the single hidden row the
/// attention or feed-forward sublayer consumes; the same down matmul also yields the per-stream inject scores the combine
/// step needs. <c>build_hc_combine</c> scatters one block output back across every stream weighted by 2·sigmoid of that
/// stream's inject, so a zero injection is a plain residual add. State is explicit: each linear block's delta-rule slot
/// arrays and convolution history advance across calls, and each sparse block's synthetic whole-sequence KV and raw
/// indexer-key caches grow across calls, so a run split at any chunk boundary lands bit-for-bit on the unsplit run.
/// </summary>
internal sealed class Qwen4ExpForwardSpike : IDisposable
{
    /// <summary>Which sublayer side a hyper-connection pair wraps: the attention side or the feed-forward side.</summary>
    internal enum SublayerSide
    {
        Attention,
        FeedForward,
    }

    /// <summary>
    /// Declared sparse-FFN shape the spike sequences for a qwen4exp block: the expert count the router scores and the
    /// routing count each token selects, whether the block adds one shared expert with its own hidden-sized selector
    /// vector, and the two routing-weight knobs <c>build_moe_ffn</c> carries. These mirror the reference descriptor's
    /// <c>n_expert</c>/<c>n_expert_used</c>, the shared-expert declaration and the declared
    /// <c>expert_weights_scale</c>/<c>expert_weights_norm</c>; qwen4exp always routes through a softmax gate with the
    /// selected weights renormalized, so <see cref="NormalizeExpertWeights"/> defaults to true.
    /// </summary>
    internal sealed record SparseMoEConfig(
        int ExpertCount,
        int RoutedExpertsPerToken,
        bool HasSharedExpert = true,
        float ExpertWeightsScale = 0f,
        bool NormalizeExpertWeights = true);

    private readonly Qwen4ExpGeometry _geometry;
    private readonly int _hiddenSize;
    private readonly int _headCount;
    private readonly int _kvHeadCount;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly float _eps;
    private readonly float _attnScale;
    private readonly double _ropeBase;
    private readonly int _streamCount;
    private readonly int _lowRank;
    private readonly int _flattened;
    private readonly int _maxChunk;
    private readonly int _maxPositions;
    private readonly IBackend _backend;

    // Per-block synthetic operands, sized from the geometry: the hyper-connection pair on each sublayer side, the block's
    // own attention operands (gated-delta-net tensors on a linear layer, sparse-attention tensors on a full layer), and the
    // dense SwiGLU gate/up/down stand-in that replaces the block's routed experts in this spike.
    private readonly LayerOperands[] _layers;

    // Per-block single-sequence state that advances across calls. Linear blocks carry the delta-rule slot arrays plus the raw
    // convolution history; sparse blocks carry the whole-sequence KV rows and the raw indexer-key rows the pools are built
    // from. A ResetSequence zeroes every one of them, so a fresh sequence starts from a clean slate exactly as the engine would.
    private readonly GdnState[] _gdnStates;
    private readonly QsaState[] _qsaStates;

    // Rotary tables shared across every block: the block q/k head dimension and the indexer component width each get their
    // own position-major cos/sin run, built once at construction so a forward never rebuilds them.
    private readonly float[] _cosHead;
    private readonly float[] _sinHead;
    private readonly float[] _cosIndexer;
    private readonly float[] _sinIndexer;
    private readonly int _ropeHalfHead;
    private readonly int _ropeHalfIndexer;

    // Reused single-token scratch for the hyper-connection mix and combine, plus the chunk-major mixed/attention rows a
    // chunk forward walks. Sized once at construction; a forward allocates nothing.
    private readonly float[] _normedStream;   // flattened: grouped-normalized, gamma-scaled streams before the low-rank pair
    private readonly float[] _lowRankScratch; // lowRank: the SiLU'd down projection
    private readonly float[] _gateScratch;    // flattened: the per-stream gate before sigmoid
    private readonly float[] _gatedScratch;   // flattened: the normalized streams after the sigmoid gate
    private readonly float[] _injectScratch;  // streamCount x maxChunk: each token's per-stream inject scores
    private readonly float[] _mixedRows;      // maxChunk * hidden: the collapsed hidden rows the sublayer consumes
    private readonly float[] _sublayerRows;   // maxChunk * hidden: attention or feed-forward outputs before the combine
    private readonly float[] _midStreams;     // maxChunk * flattened: streams after the attention combine, before the ffn mix
    private readonly float[] _ffnScratch;     // intermediateSize: SwiGLU gate/up accumulator

    // Sparse-FFN scratch and declared shape. Present only when the spike is built with a <see cref="SparseMoEConfig"/>;
    // a dense stand-in fixture carries none and never touches these arrays. The router scores one row per expert, then top-k
    // selection lands into ids/weights in ascending expert-index order exactly as the production route does, and the routed
    // chain reuses _ffnScratch for each selected expert's SwiGLU activation while _expertOut holds that expert's down result.
    private readonly SparseMoEConfig? _sparseMoE;
    private float[] _routerScores = [];   // ExpertCount: softmax probabilities over every declared expert, poisoned as picked
    private int[] _selectedIds = [];      // RoutedExpertsPerToken: chosen experts in ascending index order
    private float[] _selectedWeights = []; // RoutedExpertsPerToken: each chosen expert's routing weight beside its id
    private float[] _expertOut = [];      // hiddenSize: one selected expert's down output before it folds into the feed-forward sum

    public Qwen4ExpForwardSpike(
        Qwen4ExpGeometry geometry,
        int hiddenSize,
        int headCount,
        int kvHeadCount,
        int headDim,
        int intermediateSize,
        float eps,
        IBackend backend,
        int maxChunk = 8,
        int maxPositions = 32,
        double ropeBase = 10000.0,
        SparseMoEConfig? sparseMoE = null)
    {
        ArgumentNullException.ThrowIfNull(geometry);
        ArgumentNullException.ThrowIfNull(backend);

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvHeadCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(intermediateSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxChunk);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxPositions);

        if (headDim % 2 != 0)
        {
            throw new ArgumentException("The synthetic spike rotates full q/k heads, so the head dimension must be even.", nameof(headDim));
        }

        _geometry = geometry;
        _hiddenSize = hiddenSize;
        _headCount = headCount;
        _kvHeadCount = kvHeadCount;
        _headDim = headDim;
        _intermediateSize = intermediateSize;
        _eps = eps;
        _attnScale = 1f / MathF.Sqrt((float)headDim);
        _ropeBase = ropeBase;
        _maxChunk = maxChunk;
        _maxPositions = maxPositions;
        _backend = backend;
        _sparseMoE = sparseMoE;

        if (sparseMoE is not null)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sparseMoE.ExpertCount);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sparseMoE.RoutedExpertsPerToken);
            if (sparseMoE.RoutedExpertsPerToken > sparseMoE.ExpertCount)
            {
                throw new ArgumentException(
                    $"a token cannot route to more experts than the block declares: {sparseMoE.RoutedExpertsPerToken} routed out of {sparseMoE.ExpertCount} experts.",
                    nameof(sparseMoE));
            }
        }

        _streamCount = geometry.HyperConnections.StreamCount;
        _lowRank = geometry.HyperConnections.LowRank;
        _flattened = geometry.HyperConnections.FlattenedStreamFloats(hiddenSize);

        if (_streamCount <= 1)
        {
            throw new ArgumentException("Hyper connections mix at least two residual streams; a single stream has nothing to mix.", nameof(geometry));
        }

        _layers = new LayerOperands[geometry.LayerCount];
        for (int layer = 0; layer < geometry.LayerCount; layer++)
        {
            _layers[layer] = BuildSyntheticLayer(layer);
        }

        _gdnStates = new GdnState[geometry.LayerCount];
        for (int layer = 0; layer < geometry.LayerCount; layer++)
        {
            if (geometry.KindForLayer(layer) == AttentionKind.LinearGatedDeltaNet)
            {
                _gdnStates[layer] = new GdnState(geometry.LinearAttention, maxChunk);
            }
        }

        _qsaStates = new QsaState[geometry.LayerCount];
        for (int layer = 0; layer < geometry.LayerCount; layer++)
        {
            if (geometry.KindForLayer(layer) == AttentionKind.FullQuerySparseAttention)
            {
                _qsaStates[layer] = new QsaState(geometry.SparseAttention, geometry.CompressRatioForLayer(layer), headCount, kvHeadCount, headDim, maxChunk);
            }
        }

        _ropeHalfHead = headDim / 2;
        _ropeHalfIndexer = geometry.SparseAttention.IndexerKeyLength / 2;

        _cosHead = new float[maxPositions * _ropeHalfHead];
        _sinHead = new float[maxPositions * _ropeHalfHead];
        LayerOps.BuildRopeTables(ropeBase, headDim, maxPositions, _cosHead, _sinHead);

        _cosIndexer = new float[maxPositions * _ropeHalfIndexer];
        _sinIndexer = new float[maxPositions * _ropeHalfIndexer];
        if (_ropeHalfIndexer > 0)
        {
            LayerOps.BuildRopeTables(ropeBase, geometry.SparseAttention.IndexerKeyLength, maxPositions, _cosIndexer, _sinIndexer);
        }

        _normedStream = new float[_flattened];
        _lowRankScratch = new float[_lowRank];
        _gateScratch = new float[_flattened];
        _gatedScratch = new float[_flattened];
        _injectScratch = new float[_streamCount * maxChunk];
        _mixedRows = new float[maxChunk * hiddenSize];
        _sublayerRows = new float[maxChunk * hiddenSize];
        _midStreams = new float[maxChunk * _flattened];
        _ffnScratch = new float[intermediateSize];

        if (_sparseMoE is not null)
        {
            _routerScores = new float[_sparseMoE.ExpertCount];
            _selectedIds = new int[_sparseMoE.RoutedExpertsPerToken];
            _selectedWeights = new float[_sparseMoE.RoutedExpertsPerToken];
            _expertOut = new float[hiddenSize];
        }
    }

    /// <summary>The resolved geometry this spike sequences against.</summary>
    public Qwen4ExpGeometry Geometry => _geometry;

    /// <summary>The hidden width every stream and sublayer row carries.</summary>
    public int HiddenSize => _hiddenSize;

    /// <summary>Residual streams the hyper connections mix.</summary>
    public int StreamCount => _streamCount;

    /// <summary>Low rank the per-sublayer down/up pair compresses the streams through.</summary>
    public int LowRank => _lowRank;

    /// <summary>Flattened stream width: streams x hidden — the row count of every stream tensor this spike carries.</summary>
    public int FlattenedStreamFloats => _flattened;

    /// <summary>Which attention one layer sequences through, straight from the geometry's resolved interleave.</summary>
    public AttentionKind KindForLayer(int layer) => _geometry.KindForLayer(layer);

    // Operand accessors for a spike test that must re-sequence the exact stages in order to prove composition: a linear block's
    // gated-delta-net tensors, a sparse-full block's attention tensors, and the dense feed-forward stand-in's gate/up/down rows.
    internal GdnOps.GdnBlockTensors LinearOperands(int layer) => _layers[layer].Linear;

    internal QsaOps.QsaBlockTensors SparseOperands(int layer) => _layers[layer].Sparse;

    internal HyperConnectionTensors AttentionHyperOperands(int layer) => _layers[layer].AttnHyper;

    internal HyperConnectionTensors FeedForwardHyperOperands(int layer) => _layers[layer].FfnHyper;

    /// <summary>The sparse-FFN operands a sparse-configured spike built for this layer (router, stacked experts, shared block and selector), or an empty set for a dense stand-in fixture.</summary>
    internal SparseFfnOperands SparseMoEOperands(int layer) => _layers[layer].MoE;

    /// <summary>Clear every block's carried state so a fresh sequence starts from a clean slate exactly as the engine would. The residual streams themselves are supplied to each <see cref="ForwardBlockChunk"/> call — the reference's <c>hc_init</c> repeats the embedding across every stream, and a spike test feeds that repeated stream set directly; this method only validates the seed width and zeroes the recurrent, convolution-history and synthetic caches.</summary>
    public void StartSequence(ReadOnlySpan<float> embeddingRow)
    {
        if (embeddingRow.Length < _hiddenSize)
        {
            throw new ArgumentException("The seed embedding row must cover the hidden width.", nameof(embeddingRow));
        }

        for (int layer = 0; layer < _geometry.LayerCount; layer++)
        {
            _gdnStates[layer]?.Reset();
            _qsaStates[layer]?.Reset();
        }
    }

    /// <summary>
    /// Run one block forward over a chunk of tokens for one sequence: hyper-connection attention-side mixing collapses each
    /// token's streams to a hidden row, the block's attention runs (a single gated-delta-net chunk over every mixed row on a
    /// linear layer, or per-token indexer selection plus restricted attention on a sparse-full layer), the attention combine
    /// scatters each token's attention output back across its streams weighted by 2·sigmoid of its inject, the feed-forward
    /// side mixes again, the dense SwiGLU stand-in runs, and its combine finishes the block. The carried recurrent state,
    /// convolution history and synthetic caches advance in place, so splitting the same token run across two calls lands
    /// bit-for-bit on the unsplit run.
    /// </summary>
    public void ForwardBlockChunk(int layer, ReadOnlySpan<float> streamsIn, int tokenCount, Span<float> streamsOut)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tokenCount);
        if (tokenCount > _maxChunk)
        {
            throw new ArgumentException($"This spike sizes its chunk scratch for at most {_maxChunk} tokens per call.", nameof(tokenCount));
        }

        int streamsPerToken = _flattened;
        if (streamsIn.Length < tokenCount * streamsPerToken || streamsOut.Length < tokenCount * streamsPerToken)
        {
            throw new ArgumentException("A block chunk forward needs one flattened stream set in and out per token.", nameof(streamsIn));
        }

        LayerOperands w = _layers[layer];
        AttentionKind kind = _geometry.KindForLayer(layer);

        // Attention-side hyper-connection mix per token: collapse the streams to the hidden row the attention consumes and
        // record each token's per-stream inject for the combine that follows.
        for (int t = 0; t < tokenCount; t++)
        {
            MixStreams(layer, SublayerSide.Attention, streamsIn.Slice(t * streamsPerToken, streamsPerToken), _mixedRows.AsSpan(t * _hiddenSize, _hiddenSize), _injectScratch.AsSpan(t * _streamCount, _streamCount));
        }

        if (kind == AttentionKind.LinearGatedDeltaNet)
        {
            GdnState state = _gdnStates[layer]!;
            GdnOps.ForwardChunk(
                _mixedRows.AsSpan(0, tokenCount * _hiddenSize),
                w.Linear,
                _eps,
                tokenCount,
                state.ConvHistory,
                state.NextConvHistory,
                state.Recurrent,
                state.Scratch,
                _sublayerRows.AsSpan(0, tokenCount * _hiddenSize));

            // The convolution history update writes the new tail into NextConvHistory while ConvHistory still holds the
            // previous one; swap them so the next chunk reads exactly what this one left behind.
            state.RotateConvHistory();
        }
        else
        {
            for (int t = 0; t < tokenCount; t++)
            {
                SparseAttention(layer, _mixedRows.AsSpan(t * _hiddenSize, _hiddenSize), _sublayerRows.AsSpan(t * _hiddenSize, _hiddenSize));
            }
        }

        // Attention combine: scatter each token's attention output back across its streams, weighted by 2·sigmoid of that
        // stream's inject, producing the streams the feed-forward side consumes.
        for (int t = 0; t < tokenCount; t++)
        {
            CombineStreams(streamsIn.Slice(t * streamsPerToken, streamsPerToken), _sublayerRows.AsSpan(t * _hiddenSize, _hiddenSize), _injectScratch.AsSpan(t * _streamCount, _streamCount), _midStreams.AsSpan(t * streamsPerToken, streamsPerToken));
        }

        // Feed-forward-side hyper-connection mix, then either the sparse router/gather/expert chain (when the spike was
        // built with a SparseMoEConfig) or the dense SwiGLU stand-in (for a linear-only fixture), then the final combine.
        for (int t = 0; t < tokenCount; t++)
        {
            MixStreams(layer, SublayerSide.FeedForward, _midStreams.AsSpan(t * streamsPerToken, streamsPerToken), _mixedRows.AsSpan(t * _hiddenSize, _hiddenSize), _injectScratch.AsSpan(t * _streamCount, _streamCount));

            if (_sparseMoE is not null)
            {
                ForwardSparseMoE(layer, _mixedRows.AsSpan(t * _hiddenSize, _hiddenSize), _sublayerRows.AsSpan(t * _hiddenSize, _hiddenSize));
            }
            else
            {
                ForwardDenseFfnStandIn(layer, _mixedRows.AsSpan(t * _hiddenSize, _hiddenSize), _sublayerRows.AsSpan(t * _hiddenSize, _hiddenSize));
            }

            CombineStreams(_midStreams.AsSpan(t * streamsPerToken, streamsPerToken), _sublayerRows.AsSpan(t * _hiddenSize, _hiddenSize), _injectScratch.AsSpan(t * _streamCount, _streamCount), streamsOut.Slice(t * streamsPerToken, streamsPerToken));
        }
    }

    /// <summary>Run one block forward for a single token; a convenience wrapper over <see cref="ForwardBlockChunk"/> at one token.</summary>
    public void ForwardBlock(int layer, ReadOnlySpan<float> streamsIn, Span<float> streamsOut)
    {
        ForwardBlockChunk(layer, streamsIn, tokenCount: 1, streamsOut);
    }

    /// <summary>
    /// The reference <c>build_hc_mix</c>: group each residual stream through its own RMS normalization scaled by that stream's
    /// slice of the flattened norm vector, flatten, run the low-rank down projection scaled by 1/streams and SiLU, expand it
    /// through the up projection to a per-stream gate, gate the normalized streams by sigmoid of that gate, and collapse the
    /// streams by their mean into <paramref name="mixed"/>. The same normalized streams also feed the inject projection
    /// (scaled by 1/streams) that the combine step weights the residual scatter with.
    /// </summary>
    internal void MixStreams(int layer, SublayerSide side, ReadOnlySpan<float> streams, Span<float> mixed, Span<float> inject)
    {
        if (streams.Length < _flattened || mixed.Length < _hiddenSize || inject.Length < _streamCount)
        {
            throw new ArgumentException("Hyper-connection mixing needs one flattened stream set and room for the mixed row and every inject score.", nameof(streams));
        }

        HyperConnectionTensors hc = _layers[layer].Hyper(side);

        // Grouped RMSNorm: each stream normalizes over its own hidden slice, then scales by that stream's gamma slice.
        for (int s = 0; s < _streamCount; s++)
        {
            int baseOffset = s * _hiddenSize;
            LayerOps.RmsNormWeighted(streams.Slice(baseOffset, _hiddenSize), hc.Norm.AsSpan(baseOffset, _hiddenSize), _eps, _normedStream.AsSpan(baseOffset, _hiddenSize));
        }

        float invStreams = 1f / _streamCount;

        // Low-rank down projection over the flattened streams, scaled by 1/streams, then SiLU.
        for (int r = 0; r < _lowRank; r++)
        {
            _lowRankScratch[r] = LayerOps.Silu(DotRow(hc.Down, r, _flattened, _normedStream) * invStreams);
        }

        // Up projection expands the low-rank vector back to a per-stream gate across the flattened width.
        for (int f = 0; f < _flattened; f++)
        {
            _gateScratch[f] = DotRow(hc.Up, f, _lowRank, _lowRankScratch);
        }

        // Gate the normalized streams by sigmoid of the gate; the mean over streams is the single hidden row the sublayer runs.
        for (int f = 0; f < _flattened; f++)
        {
            _gatedScratch[f] = _normedStream[f] * GdnOps.Sigmoid(_gateScratch[f]);
        }

        for (int h = 0; h < _hiddenSize; h++)
        {
            double sum = 0.0;
            for (int s = 0; s < _streamCount; s++)
            {
                sum += _gatedScratch[s * _hiddenSize + h];
            }

            mixed[h] = (float)(sum * invStreams);
        }

        for (int s = 0; s < _streamCount; s++)
        {
            inject[s] = DotRow(hc.Inject, s, _flattened, _normedStream) * invStreams;
        }
    }

    /// <summary>
    /// The reference <c>build_hc_combine</c>: scatter one sublayer's hidden output back across every residual stream weighted
    /// by 2·sigmoid of that stream's inject score, so a zero injection leaves a plain residual add and a unit injection
    /// doubles the contribution.
    /// </summary>
    internal void CombineStreams(ReadOnlySpan<float> streamsIn, ReadOnlySpan<float> blockOut, ReadOnlySpan<float> inject, Span<float> streamsOut)
    {
        if (streamsIn.Length < _flattened || blockOut.Length < _hiddenSize || inject.Length < _streamCount || streamsOut.Length < _flattened)
        {
            throw new ArgumentException("Hyper-connection combine needs the input streams, one hidden block output, every inject score and room for the output streams.", nameof(streamsIn));
        }

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

    /// <summary>
    /// The dense SwiGLU gate/up/down block that stands in for this family's routed experts in the spike: gate and up project
    /// the mixed hidden row, SiLU gates the gate and scales it by the up result, and down projects back to the hidden width.
    /// This is explicitly a stand-in — the sparse router, top-k selection, gather and expert residency are not composed here.
    /// </summary>
    internal void ForwardDenseFfnStandIn(int layer, ReadOnlySpan<float> mixed, Span<float> ffnOut)
    {
        if (mixed.Length < _hiddenSize || ffnOut.Length < _hiddenSize)
        {
            throw new ArgumentException("The dense feed-forward stand-in needs one hidden row in and out.", nameof(mixed));
        }

        DenseFfnTensors ffn = _layers[layer].Ffn;

        for (int i = 0; i < _intermediateSize; i++)
        {
            float gate = DotRow(ffn.Gate, i, _hiddenSize, mixed);
            float up = DotRow(ffn.Up, i, _hiddenSize, mixed);
            _ffnScratch[i] = LayerOps.Silu(gate) * up;
        }

        for (int h = 0; h < _hiddenSize; h++)
        {
            ffnOut[h] = DotRow(ffn.Down, h, _intermediateSize, _ffnScratch);
        }
    }

    /// <summary>
    /// The reference <c>build_moe_ffn</c> routing for one hidden row, mirroring the production <c>RouteSparseExperts</c>
    /// exactly so a spike selection is bit-identical to the engine's: the router emits one score per declared expert, every
    /// score is softmaxed together (qwen4exp routes through a softmax gate), then <c>RoutedExpertsPerToken</c> picks run as
    /// ascending-index scans under strict greater-than so equal scores resolve to the lower expert index; a picked slot is
    /// poisoned so a later pass cannot re-pick it. The chosen slots are insertion-sorted into ascending expert-index order
    /// beside their weights, then — because qwen4exp renormalizes (norm = true) — divided by the selected probability sum,
    /// and finally scaled by <c>ExpertWeightsScale</c> when nonzero. Selection carries no cross-token state, so routing the
    /// same row twice, or across a chunk split, lands on the same experts in the same order.
    /// </summary>
    internal void RouteSparseMoE(int layer, ReadOnlySpan<float> mixed, Span<int> ids, Span<float> weights)
    {
        SparseMoEConfig cfg = _sparseMoE!;
        SparseFfnOperands moe = _layers[layer].MoE;
        int expertCount = cfg.ExpertCount;
        int routed = cfg.RoutedExpertsPerToken;

        if (mixed.Length < _hiddenSize || ids.Length < routed || weights.Length < routed)
        {
            throw new ArgumentException("Sparse routing needs one hidden row and room for every routed expert and weight.", nameof(mixed));
        }

        // Router scores over every declared expert, then an all-expert softmax in place with a max-subtraction guard.
        for (int e = 0; e < expertCount; e++)
        {
            _routerScores[e] = DotRow(moe.Router, e, _hiddenSize, mixed);
        }

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

        // Ascending-index strict-greater picks: a tie keeps the earlier expert regardless of buffer reuse.
        float selectedSum = 0f;
        for (int pick = 0; pick < routed; pick++)
        {
            int best = -1;
            float bestScore = float.NegativeInfinity;
            for (int e = 0; e < expertCount; e++)
            {
                if (_routerScores[e] > bestScore)
                {
                    best = e;
                    bestScore = _routerScores[e];
                }
            }

            ids[pick] = best;
            weights[pick] = bestScore;
            selectedSum += bestScore;
            _routerScores[best] = float.NegativeInfinity; // consumed: later passes cannot re-pick this expert
        }

        // Sort the chosen slots into ascending expert-index execution order beside their weights.
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

        if (cfg.NormalizeExpertWeights && selectedSum > 0f)
        {
            for (int i = 0; i < routed; i++)
            {
                weights[i] /= selectedSum;
            }
        }

        if (cfg.ExpertWeightsScale != 0f)
        {
            for (int i = 0; i < routed; i++)
            {
                weights[i] *= cfg.ExpertWeightsScale;
            }
        }
    }

    /// <summary>
    /// One block's sparse feed-forward for a single hidden row: route, then run each selected expert's gate/up/down chain
    /// in ascending expert-index order — a spike-local gather that mirrors <see cref="CpuGatherKernel"/>'s contract (visit
    /// the selection once, ascending, each expert's slab read as a contiguous view of its stacked role) over synthetic
    /// operands instead of GGUF byte slabs — accumulating each expert's down output into <paramref name="ffnOut"/> scaled by
    /// its routing weight. A declared shared expert then runs its own dense-shaped SwiGLU block once, gated by
    /// <c>sigmoid(dot(ffn_gate_inp_shexp, mixed))</c> — the plain sigmoid the qwen4exp reference applies, not qwen2moe's
    /// <c>silu(score) / score</c> — and folds into the same sum. The result is the hidden row the hyper-connection combine
    /// consumes. This path carries no cross-token state, so splitting a chunk across calls reproduces the unsplit run.
    /// </summary>
    internal void ForwardSparseMoE(int layer, ReadOnlySpan<float> mixed, Span<float> ffnOut)
    {
        SparseMoEConfig cfg = _sparseMoE!;
        SparseFfnOperands moe = _layers[layer].MoE;
        int routed = cfg.RoutedExpertsPerToken;

        if (mixed.Length < _hiddenSize || ffnOut.Length < _hiddenSize)
        {
            throw new ArgumentException("The sparse feed-forward needs one hidden row in and out.", nameof(mixed));
        }

        RouteSparseMoE(layer, mixed, _selectedIds.AsSpan(0, routed), _selectedWeights.AsSpan(0, routed));

        // Clear only this row's feed-forward region before accumulating: the caller may alias a shared scratch buffer whose
        // other rows belong to neighbouring tokens and must stay untouched. The routed and shared contributions build from zero.
        for (int h = 0; h < _hiddenSize; h++)
        {
            ffnOut[h] = 0f;
        }

        // Routed experts, ascending: each selected expert's SwiGLU chain reads its slab as a contiguous view of the stacked role.
        for (int slot = 0; slot < routed; slot++)
        {
            int expert = _selectedIds[slot];
            float weight = _selectedWeights[slot];

            for (int i = 0; i < _intermediateSize; i++)
            {
                int row = expert * _intermediateSize + i;
                float gate = DotRow(moe.ExpertGate, row, _hiddenSize, mixed);
                float up = DotRow(moe.ExpertUp, row, _hiddenSize, mixed);
                _ffnScratch[i] = LayerOps.Silu(gate) * up;
            }

            for (int h = 0; h < _hiddenSize; h++)
            {
                _expertOut[h] = DotRow(moe.ExpertDown, expert * _hiddenSize + h, _intermediateSize, _ffnScratch);
            }

            for (int h = 0; h < _hiddenSize; h++)
            {
                ffnOut[h] += weight * _expertOut[h];
            }
        }

        // Shared expert once, under its own sigmoid gate over the hidden-sized selector vector (ffn_gate_inp_shexp).
        if (cfg.HasSharedExpert)
        {
            float score = 0f;
            for (int c = 0; c < _hiddenSize; c++)
            {
                score += moe.SharedGateVector[c] * mixed[c];
            }

            float sharedWeight = GdnOps.Sigmoid(score);

            for (int i = 0; i < _intermediateSize; i++)
            {
                float gate = DotRow(moe.SharedGate, i, _hiddenSize, mixed);
                float up = DotRow(moe.SharedUp, i, _hiddenSize, mixed);
                _ffnScratch[i] = LayerOps.Silu(gate) * up;
            }

            for (int h = 0; h < _hiddenSize; h++)
            {
                ffnOut[h] += sharedWeight * DotRow(moe.SharedDown, h, _intermediateSize, _ffnScratch);
            }
        }
    }

    /// <summary>
    /// One sparse-full block's attention for a single token against the block's synthetic whole-sequence caches: project and
    /// rotate the block q/k heads, append the token's K/V, project the indexer query and rotate it at this position, append
    /// the raw indexer key, pool every complete block of ratio cells and rotate each pooled key to its first member, score
    /// the pools, select the visible cells through <see cref="QsaOps.SelectCells"/>, and run dense grouped-query attention
    /// restricted to that selection through <see cref="QsaOps.RestrictedAttention"/>. The query-and-gate layout alternates a
    /// query slice and its gate slice per head, so the attention output is gated element-wise by sigmoid of each head's gate
    /// before the output projection — the same gating the reference's interleaved query/gate rows carry.
    /// </summary>
    internal void SparseAttentionSublayer(int layer, ReadOnlySpan<float> mixed, Span<float> attnOut)
    {
        SparseAttention(layer, mixed, attnOut);
    }

    private void SparseAttention(int layer, ReadOnlySpan<float> mixed, Span<float> attnOut)
    {
        QsaOps.QsaBlockTensors w = _layers[layer].Sparse;
        QsaState state = _qsaStates[layer]!;
        int position = state.CacheCells;

        if (position >= _maxPositions)
        {
            throw new ArgumentException($"This spike sized its rotary tables for {_maxPositions} positions per sequence.", nameof(mixed));
        }

        int qRegion = _headCount * _headDim;
        int kvRegion = _kvHeadCount * _headDim;

        // Block q heads with their per-head gate slices: query slice then gate slice, alternating per head.
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

        // Block k heads normalized and rotated, value heads raw, both appended to the synthetic KV cache this token will attend.
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

        // Indexer query: one component region per scoring head, normalized and rotated at this token's position.
        int idxDim = w.IndexerDim;
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

        // Raw indexer key for this cell: stored unnormalized and unrotated, exactly as the reference caches the raw key side.
        for (int j = 0; j < idxDim; j++)
        {
            state.RawIndexerKeys[position * idxDim + j] = DotRow(w.IndexerKeyProj, j, _hiddenSize, mixed);
        }

        state.CacheCells = position + 1;
        int cacheCells = state.CacheCells;

        if (w.CompressRatio < 2 || cacheCells / w.CompressRatio == 0)
        {
            // A ratio that groups fewer than two cells runs dense and never reaches selection, exactly as the reference's
            // layer-level rule states; a cache shorter than one complete pool has nothing to rank either. Either way every
            // cached cell stays visible and attention walks the whole causal prefix directly.
            List<int> visible = [];
            for (int cell = 0; cell < cacheCells; cell++)
            {
                visible.Add(cell);
            }

            QsaOps.RestrictedAttention(_backend, state.RotatedQuery, state.KeyCache, state.ValueCache, visible, cacheCells, w, _attnScale, state.AttentionConcat);
        }
        else
        {
            // Pool every complete block of ratio cells (mean then RMS-normalize), then rotate each pooled key to its first
            // member's position — pooling precedes norm and rotation, so the rotation is a separate pass over the pooled run.
            int fullPools = cacheCells / w.CompressRatio;
            QsaOps.PoolKeys(state.RawIndexerKeys.AsSpan(0, cacheCells * idxDim), cacheCells, w, _eps, state.PooledKeys.AsSpan(0, fullPools * idxDim));

            for (int p = 0; p < fullPools; p++)
            {
                QsaOps.RotateAtPosition(state.PooledKeys.AsSpan(p * idxDim, idxDim), _ropeHalfIndexer, _cosIndexer, _sinIndexer, p * w.CompressRatio);
            }

            QsaOps.IndexerScores(state.IndexerQuery.AsSpan(0, w.IndexerHeads * idxDim), state.PooledKeys.AsSpan(0, fullPools * idxDim), w.IndexerHeads, idxDim, fullPools, state.Scores.AsSpan(0, fullPools));
            List<int> selected = QsaOps.SelectCells(state.Scores.AsSpan(0, fullPools), cacheCells, w);
            QsaOps.RestrictedAttention(_backend, state.RotatedQuery, state.KeyCache, state.ValueCache, selected, cacheCells, w, _attnScale, state.AttentionConcat);
        }

        // Gate each attended head by sigmoid of its gate slice, then project the concatenated heads through the block output.
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

    private LayerOperands BuildSyntheticLayer(int layer)
    {
        var operands = new LayerOperands
        {
            AttnHyper = BuildHyper(),
            FfnHyper = BuildHyper(),
            Ffn = new DenseFfnTensors
            {
                Gate = SyntheticMatrix(_intermediateSize, _hiddenSize),
                Up = SyntheticMatrix(_intermediateSize, _hiddenSize),
                Down = SyntheticMatrix(_hiddenSize, _intermediateSize),
            },
        };

        if (_sparseMoE is not null)
        {
            operands.MoE = BuildSparseMoEOperands();
        }

        if (_geometry.KindForLayer(layer) == AttentionKind.LinearGatedDeltaNet)
        {
            GdnDescriptor gdn = _geometry.LinearAttention;
            int valueDim = gdn.ValueHeadCount * gdn.StateDim;

            operands.Linear = new GdnOps.GdnBlockTensors
            {
                HiddenSize = _hiddenSize,
                KeyHeads = gdn.KeyHeadCount,
                ValueHeads = gdn.ValueHeadCount,
                StateDim = gdn.StateDim,
                ConvKernel = gdn.ConvKernelSize,
                Qkv = SyntheticMatrix(gdn.ConvChannels, _hiddenSize),
                Gate = SyntheticMatrix(valueDim, _hiddenSize),
                BetaAlphaCat = SyntheticMatrix(2 * gdn.ValueHeadCount, _hiddenSize),
                SsmA = SyntheticVector(gdn.ValueHeadCount, -1.0),
                DtBias = SyntheticVector(gdn.ValueHeadCount, 0.0),
                Conv1d = SyntheticMatrix(gdn.ConvChannels * gdn.ConvKernelSize, 1),
                NormWeight = SyntheticVector(gdn.StateDim, 1.0),
                Output = SyntheticMatrix(_hiddenSize, valueDim),
            };
        }
        else
        {
            QsaDescriptor qsa = _geometry.SparseAttention;

            operands.Sparse = new QsaOps.QsaBlockTensors
            {
                HiddenSize = _hiddenSize,
                HeadCount = _headCount,
                KvHeadCount = _kvHeadCount,
                HeadDim = _headDim,
                IndexerHeads = qsa.IndexerHeadCount,
                IndexerDim = qsa.IndexerKeyLength,
                TopKCells = qsa.TopKCells,
                CompressRatio = _geometry.CompressRatioForLayer(layer),
                QueryGate = SyntheticMatrix(2 * _headCount * _headDim, _hiddenSize),
                Key = SyntheticMatrix(_kvHeadCount * _headDim, _hiddenSize),
                Value = SyntheticMatrix(_kvHeadCount * _headDim, _hiddenSize),
                Output = SyntheticMatrix(_hiddenSize, _headCount * _headDim),
                IndexerQueryProj = SyntheticMatrix(qsa.IndexerHeadCount * qsa.IndexerKeyLength, _hiddenSize),
                IndexerKeyProj = SyntheticMatrix(qsa.IndexerKeyLength, _hiddenSize),
                IndexerQueryNorm = SyntheticVector(qsa.IndexerKeyLength, 1.0),
                IndexerKeyNorm = SyntheticVector(qsa.IndexerKeyLength, 1.0),
                QueryHeadNorm = SyntheticVector(_headDim, 1.0),
                KeyHeadNorm = SyntheticVector(_headDim, 1.0),
            };
        }

        return operands;
    }

    private HyperConnectionTensors BuildHyper() => new()
    {
        Norm = SyntheticVector(_flattened, 1.0),
        Down = SyntheticMatrix(_lowRank, _flattened),
        Up = SyntheticMatrix(_flattened, _lowRank),
        Inject = SyntheticMatrix(_streamCount, _flattened),
    };

    // Synthetic sparse-FFN operands sized straight from the declared shape and the block's hidden/intermediate widths: a
    // router matrix over the experts, the stacked per-expert gate/up/down triplets (gate/up are intermediate x hidden and
    // down is hidden x intermediate, exactly the reference's *_exps stacks), and the shared expert's own gate/up/down plus
    // its hidden-sized selector vector. Every fill is the same deterministic pattern the rest of the spike uses, so a
    // sparse run is reproducible with no model file.
    private SparseFfnOperands BuildSparseMoEOperands()
    {
        SparseMoEConfig cfg = _sparseMoE!;
        int experts = cfg.ExpertCount;

        return new SparseFfnOperands
        {
            Router = SyntheticMatrix(experts, _hiddenSize),
            ExpertGate = SyntheticMatrix(experts * _intermediateSize, _hiddenSize),
            ExpertUp = SyntheticMatrix(experts * _intermediateSize, _hiddenSize),
            ExpertDown = SyntheticMatrix(experts * _hiddenSize, _intermediateSize),
            SharedGate = SyntheticMatrix(_intermediateSize, _hiddenSize),
            SharedUp = SyntheticMatrix(_intermediateSize, _hiddenSize),
            SharedDown = SyntheticMatrix(_hiddenSize, _intermediateSize),
            SharedGateVector = SyntheticVector(_hiddenSize, 0.5), // clearly nonzero selector so the shared gate is a real score
        };
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

    // Deterministic synthetic fills so a spike run is reproducible without a model file: matrices follow a fixed index
    // pattern, norm vectors stay clearly nonzero, and decay weights land negative as the reference stores them.
    private static float[] SyntheticMatrix(int rows, int cols)
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

    private static float[] SyntheticVector(int length, double bias)
    {
        float[] vector = new float[length];
        for (int i = 0; i < length; i++)
        {
            double value = (i % 3) * 0.25 - 0.25 + bias;
            vector[i] = (float)value;
        }

        return vector;
    }

    public void Dispose()
    {
        // No unmanaged resources: the backend is caller-owned and the spike releases nothing it did not allocate as arrays.
    }

    private sealed class LayerOperands
    {
        public HyperConnectionTensors AttnHyper = new();
        public HyperConnectionTensors FfnHyper = new();
        public DenseFfnTensors Ffn = new();

        // Sparse-FFN operands, present only when the spike was built with a SparseMoEConfig; a dense stand-in fixture leaves
        // this at its empty default and never reads it. The forward chooses sparse or dense from whether the config is set.
        public SparseFfnOperands MoE = new();

        // Exactly one of these attention operand sets is filled, chosen by the layer's resolved kind; a linear block leaves
        // Sparse at its empty default and a sparse-full block leaves Linear empty. A forward reads only the side its kind selects.
        public GdnOps.GdnBlockTensors Linear;
        public QsaOps.QsaBlockTensors Sparse;

        public HyperConnectionTensors Hyper(SublayerSide side) => side == SublayerSide.Attention ? AttnHyper : FfnHyper;
    }

    internal sealed class HyperConnectionTensors
    {
        public float[] Norm = []; // flattened: each stream's gamma slice
        public float[] Down = []; // lowRank rows x flattened
        public float[] Up = []; // flattened rows x lowRank
        public float[] Inject = []; // streamCount rows x flattened
    }

    private sealed class DenseFfnTensors
    {
        public float[] Gate = []; // intermediateSize rows x hidden
        public float[] Up = []; // intermediateSize rows x hidden
        public float[] Down = []; // hidden rows x intermediateSize
    }

    internal sealed class SparseFfnOperands
    {
        public float[] Router = []; // ExpertCount rows x hidden — the ffn_gate_inp router matrix
        public float[] ExpertGate = []; // (ExpertCount * intermediate) rows x hidden — stacked ffn_gate_exps slab views
        public float[] ExpertUp = []; // (ExpertCount * intermediate) rows x hidden — stacked ffn_up_exps slab views
        public float[] ExpertDown = []; // (ExpertCount * hidden) rows x intermediate — stacked ffn_down_exps slab views
        public float[] SharedGate = []; // intermediateSize rows x hidden — the shared expert's gate matrix
        public float[] SharedUp = []; // intermediateSize rows x hidden — the shared expert's up matrix
        public float[] SharedDown = []; // hiddenSize rows x intermediateSize — the shared expert's down matrix
        public float[] SharedGateVector = []; // hiddenSize: the ffn_gate_inp_shexp selector row whose score gates the shared block
    }

    private sealed class GdnState
    {
        public GdnState(GdnDescriptor gdn, int maxChunk)
        {
            Recurrent = new float[gdn.RecurrentStateValuesPerSequence];
            ConvHistory = new float[gdn.ConvHistoryValuesPerSequence];
            NextConvHistory = new float[gdn.ConvHistoryValuesPerSequence];
            Scratch = GdnOps.GdnChunkScratch.Create(ToBlockTensors(gdn), maxChunk);
        }

        public float[] Recurrent { get; }
        public float[] ConvHistory { get; private set; }
        public float[] NextConvHistory { get; private set; }
        public GdnOps.GdnChunkScratch Scratch { get; }

        public void Reset()
        {
            Array.Clear(Recurrent);
            Array.Clear(ConvHistory);
            Array.Clear(NextConvHistory);
        }

        public void RotateConvHistory()
        {
            (ConvHistory, NextConvHistory) = (NextConvHistory, ConvHistory);
        }

        private static GdnOps.GdnBlockTensors ToBlockTensors(GdnDescriptor gdn) => new()
        {
            HiddenSize = 0,
            KeyHeads = gdn.KeyHeadCount,
            ValueHeads = gdn.ValueHeadCount,
            StateDim = gdn.StateDim,
            ConvKernel = gdn.ConvKernelSize,
        };
    }

    private sealed class QsaState
    {
        public QsaState(QsaDescriptor qsa, int compressRatio, int headCount, int kvHeadCount, int headDim, int capacity)
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

        public void Reset()
        {
            CacheCells = 0;
        }
    }
}
