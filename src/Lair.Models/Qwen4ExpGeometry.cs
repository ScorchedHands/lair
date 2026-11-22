namespace Lair.Models;

/// <summary>
/// Which attention a block of this family runs, resolved per layer from the file's declared interleave rather than
/// from tensor presence alone: a gated-delta-net linear-attention block (its fused qkv buffer, its gate matrix and
/// its recurrent-state tensors) or a full-attention block that scores through the sparse indexer. The two kinds share
/// one profile because every layer of this family carries the same hyper-connection pair around whichever sublayer it
/// runs; which side a layer takes is what its declared compress ratio says, not where its tensors happen to live.
/// </summary>
public enum AttentionKind
{
    /// <summary>The block's attention is gated-delta-net linear attention: constant-size recurrent state per sequence, no KV rows.</summary>
    LinearGatedDeltaNet,

    /// <summary>The block runs dense grouped-query attention restricted to indexer-selected cache cells over a whole-sequence KV pool.</summary>
    FullQuerySparseAttention,
}

/// <summary>
/// Geometry one gated-delta-net linear-attention block of this family resolves from the file's declared
/// <c>&lt;prefix&gt;ssm.*</c> group: key-head groups (<c>group_count</c>), shared head dimension
/// (<c>state_size</c>), concatenated value width (<c>inner_size</c>) and its gate rows (<c>time_step_rank</c>), plus
/// the causal depthwise convolution's tap count (<c>conv_kernel</c>). The derived <see cref="ValueHeadCount"/> is the
/// declared inner size divided by the head dimension, which is exactly how many value-head slices every token carries.
/// </summary>
public sealed record GdnDescriptor(
    int KeyHeadCount,
    int ValueHeadCount,
    int StateDim,
    int InnerSize,
    int TimeStepRank,
    int ConvKernelSize)
{
    /// <summary>One token's q region (and its duplicated k region) inside the fused projection row: key-head groups × head dimension.</summary>
    public int QkRegionFloats => KeyHeadCount * StateDim;

    /// <summary>Fused-projection rows and convolution channels one linear block runs over: two q/k regions plus the value width — exactly the declared <c>attn_qkv</c>/<c>ssm_conv1d</c> row count.</summary>
    public int ConvChannels => 2 * QkRegionFloats + InnerSize;

    /// <summary>Floating-point slots one sequence holds as this block's recurrent delta-rule state: each value head's full slot matrix, matching the layer-level kernel's footprint exactly.</summary>
    public long RecurrentStateValuesPerSequence => (long)ValueHeadCount * StateDim * StateDim;

    /// <summary>Floating-point slots one sequence carries as this block's raw qkv convolution history: every channel of <c>conv_kernel - 1</c> stored rows.</summary>
    public long ConvHistoryValuesPerSequence => (long)(ConvKernelSize - 1) * ConvChannels;

    /// <summary>Floating-point slots one sequence holds across this block's recurrent storage: the delta-rule state plus the convolution history.</summary>
    public long StateSlotsPerSequence => RecurrentStateValuesPerSequence + ConvHistoryValuesPerSequence;
}

/// <summary>
/// Indexer geometry a full-attention sparse block of this family resolves from the file's declared
/// <c>&lt;prefix&gt;attention.indexer.*</c> group: scoring heads (<c>head_count</c>), their shared component width
/// (<c>key_length</c>) and the per-token selection budget in cache cells (<c>top_k</c>). The pool rule is the layer-level
/// kernel's own: a token keeps at most <c>top_k / ratio</c> complete pools, where the ratio comes from that layer's
/// declared compress ratio.
/// </summary>
public sealed record QsaDescriptor(
    int IndexerHeadCount,
    int IndexerKeyLength,
    int TopKCells)
{
    /// <summary>Floating-point slots one cached cell carries in this indexer's cache: every scoring head's component region.</summary>
    public long IndexerCacheValuesPerCachedCell => (long)IndexerHeadCount * IndexerKeyLength;

    /// <summary>Complete pool blocks the selection budget keeps for a layer with this compress ratio — the same floor rule the layer-level selection applies.</summary>
    public int PoolBudgetForRatio(int compressRatio) => TopKCells / Math.Max(1, compressRatio);
}

/// <summary>
/// Hyper-connection shape one block of this family resolves from the file's declared
/// <c>&lt;prefix&gt;hyper_connection.*</c> group: how many residual streams the block mixes (<c>count</c>) and the low
/// rank its per-sublayer down/up pair compresses them through (<c>low_rank</c>). The family ships each stream tensor
/// flattened to <see cref="FlattenedStreamFloats"/> rows or columns, so every hyper-connection matrix shape is this
/// record's two numbers crossed with the hidden size.
/// </summary>
public sealed record HyperConnectionDescriptor(
    int StreamCount,
    int LowRank)
{
    /// <summary>The flattened residual width the family's stream tensors carry: streams × hidden size — the row count of every inject tensor and down projection, and the column count of every up projection.</summary>
    public int FlattenedStreamFloats(int hiddenSize) => checked(StreamCount * hiddenSize);
}

/// <summary>
/// Resolved qwen4exp execution geometry a plan carries for a file that declares this family's metadata groups: the
/// per-layer attention interleave (<see cref="AttentionKinds"/>) together with the declared compress ratios it was
/// resolved from, the shared gated-delta-net geometry every linear block runs, the indexer geometry and selection
/// budget every full block scores through, and the hyper-connection shape wrapping both sublayers. The budgets the
/// residency machinery seats against derive here: <see cref="LinearLayerCount"/>/<see cref="FullLayerCount"/>, the
/// per-sequence recurrent-state slots across all linear blocks (<see cref="RecurrentStateSlotsPerSequence"/>) and the
/// indexer cache footprint of a sequence window (<see cref="IndexerCacheValuesForWindow"/>). A plan resolved from a
/// file that declares none of these groups carries <c>null</c> instead.
/// </summary>
public sealed record Qwen4ExpGeometry(
    IReadOnlyList<AttentionKind> AttentionKinds,
    IReadOnlyList<int> CompressRatios,
    GdnDescriptor LinearAttention,
    QsaDescriptor SparseAttention,
    HyperConnectionDescriptor HyperConnections)
{
    /// <summary>The block count this geometry covers — equal to both per-layer lists' length.</summary>
    public int LayerCount => AttentionKinds.Count;

    /// <summary>Which attention one layer runs: linear blocks carry zero declared ratio, full sparse blocks a positive one.</summary>
    public AttentionKind KindForLayer(int layer) => AttentionKinds[layer];

    /// <summary>The compress-ratio rule that made this decision for the layer. Linear layers return 0 and run no selection at all; an out-of-range layer fails naming it rather than defaulting to some other kind.</summary>
    public int CompressRatioForLayer(int layer) => CompressRatios[layer];

    /// <summary>Blocks running gated-delta-net linear attention — the layers whose state, not KV rows, bounds a sequence's footprint.</summary>
    public int LinearLayerCount { get; } = AttentionKinds.Count(k => k == AttentionKind.LinearGatedDeltaNet);

    /// <summary>Blocks running full sparse attention — each holding its own whole-sequence indexer cache.</summary>
    public int FullLayerCount { get; } = AttentionKinds.Count(k => k == AttentionKind.FullQuerySparseAttention);

    /// <summary>Floating-point slots one sequence holds as recurrent state across every linear block of this model: the delta-rule slot matrices plus each block's convolution history — what a residency tier must seat before demoting anything to disk.</summary>
    public long RecurrentStateSlotsPerSequence => (long)LinearLayerCount * LinearAttention.StateSlotsPerSequence;

    /// <summary>Floating-point slots one sequence window of this width carries in every full block's indexer cache: cached cells × per-cell footprint × full-block count — the sparse-attention counterpart to KV blocks for budget accounting.</summary>
    public long IndexerCacheValuesForWindow(int contextTokens) => (long)FullLayerCount * SparseAttention.IndexerCacheValuesPerCachedCell * contextTokens;

    /// <summary>Floating-point slots one sequence window holds across both attention families: recurrent state on the linear side, indexer cache on the full side. KV rows for the full layers stay counted where block pools already count them.</summary>
    public long RecurrentAndIndexerSlotsForWindow(int contextTokens) => RecurrentStateSlotsPerSequence + IndexerCacheValuesForWindow(contextTokens);

    /// <summary>Floating-point slots this model's declared hyper-connection tensors carry across all blocks: each block ships an inject, a down and up pair and a norm vector on both its sublayer sides — declared-shape accounting only, since no block executes yet.</summary>
    public long HyperConnectionResidentFloats(int hiddenSize) => (long)LayerCount * 2L * PerSublayerHyperConnectionFloats(hiddenSize);

    /// <summary>Floating-point slots one sublayer side's hyper-connection tensors carry: the flattened norm vector, each stream's inject row, and the low-rank down/up pair.</summary>
    public long PerSublayerHyperConnectionFloats(int hiddenSize)
    {
        int flattened = HyperConnections.FlattenedStreamFloats(hiddenSize);
        return (long)flattened // this side's norm vector
            + (long)HyperConnections.StreamCount * flattened // the inject scores, one row per stream
            + 2L * flattened * HyperConnections.LowRank; // down and up projections across the low rank
    }

    /// <summary>Floating-point slots the model's global output hyper-connection tensors carry: one norm vector plus a final down/up pair mixing the streams into the logits' hidden side — no inject scores, because this conversion ships only that trio globally.</summary>
    public long OutputHyperConnectionFloats(int hiddenSize)
    {
        int flattened = HyperConnections.FlattenedStreamFloats(hiddenSize);
        return (long)flattened // the final norm vector over the whole flattened width
            + 2L * flattened * HyperConnections.LowRank; // one down/up pair across the low rank, no per-stream inject row at this scope
    }

    /// <summary>Floating-point slots every full block's indexer projection and norm tensors carry: one key region per scoring head, one shared k-projection row set, and two component-width norms — the resident operand footprint of the selection machinery.</summary>
    public long IndexerOperandFloats() => SparseAttention.IndexerHeadCount * SparseAttention.IndexerKeyLength // q projections across heads
        + SparseAttention.IndexerKeyLength // the single key projection's rows (IndexerDim x hidden counted at its own width here)
        + 2L * SparseAttention.IndexerKeyLength; // query and key norm vectors
}
