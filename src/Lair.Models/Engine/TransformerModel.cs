using System.Buffers;
using System.Globalization;

using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;

namespace Lair.Models.Engine;

/// <summary>
/// Forward engine for supported GGUF architectures (llama, qwen2): embedding lookup, per-layer norm / GQA
/// attention with RoPE and paged KV via <see cref="SequenceKv"/>, gated feed-forward, final norm, and an untied
/// output matrix or tied token-embedding rows for logits. The graph mirrors the reference structure: each layer
    /// normalizes its input stream (norm x weight under the plan's resolved norm strategy) before the QKV GEMVs; RoPE
    /// rotates q/k pairs at the absolute position with theta_i = base^(-2i/headDim), pairing adjacent dimensions
    /// (2i, 2i + 1) under the llama-family strategy or the head halves (i, i + headDim/2) under the qwen3 strategy;
    /// attention scores scale by 1/sqrt(headDim), pass through the plan's attention-score softcap when one is resolved, and
/// softmax runs online over positions 0..pos, narrowed per layer to <c>[max(0, pos - window + 1), pos]</c> on the
/// layers the plan's resolved window interleave masks and left unbounded on the layers it leaves alone. The window
/// masks scores only: every position's K/V is still written and stays readable. The residual added before the
/// FFN norm is the pre-attention stream value, or that stream plus a normalized attention output under a placement that
/// norms sublayer outputs — either way matching the reference graph's ffn_inp add; rope scaling variants
/// are not supported. FP32 buffers are allocated once at Open and reused across calls; packed matrix rows are
/// consumed through <see cref="WeightRef"/> without being materialized, except for embedding lookups and small
/// norm/bias vectors preloaded at Open. Attention-bias tensors follow the plan's resolved policy: under
/// <see cref="BiasPolicy.Declared"/> the resolved <c>use_bias</c> flag — a declaration or the profile's feature
/// default — makes the q/k/v bias vectors required or leaves every attention bias unloaded and unapplied, with one
/// structured log line naming the ignored tensors, while an output-projection bias is loaded whenever the file
/// carries one; the plan's resolved <see cref="QkNormMode"/> normalizes each q and k head between its projection
/// (plus any bias) and its rotation — per head scaled by the layer's head-dimension vector, over each side's whole
/// projected stream scaled by that side's full-width vector, normalized with the epsilon alone, or not at all. Under
/// <see cref="NormPlacement.PostSublayerOutputs"/> every sublayer output is normalized once more with its own vector before
/// it joins the residual stream, which makes each layer's post-sublayer norm tensors required; a pre-only placement reads
/// none and rejects a file that carries one. A resolved
/// final-logit softcap saturates every logit vector — untied output head or tied token-embedding rows — after
/// the full vector is computed and before any softmax, sampling, or copy-out. A resolved input-embedding gain
/// multiplies every freshly dequantized embedding row by its value before the first layer on every forward path.
/// A resolved sparse-FFN descriptor replaces this block's dense gate/up/down triplet with an expert-routed block
/// whose every expert is resident: the router matrix scores all experts, top-k selection keeps the
/// <c>&lt;prefix&gt;expert_used_count</c> best-scoring experts — equal scores resolve to the lower expert index — and
/// renormalizes the selected routing weights by their own sum when the descriptor declares normalization; each
/// selected expert's gate/up/down GEMVs then run over the same normalized stream in ascending expert-index order and
/// accumulate into the residual scaled by its weight. A declared shared expert adds one dense gate/up/down block to
/// that accumulation once per token, an undeclared shared expert reads no sparse role tensor at all, and a file
/// declaring none keeps the dense triplet path unchanged. Sparse routing runs on the prefill/decode token path and on
/// <see cref="DecodeBatch"/>; <see cref="Encode"/> forwards dense blocks only and fails naming the descriptor when a
/// model carries one.
/// </summary>

public sealed class TransformerModel : IDisposable
{
    /// <summary>Largest number of sequences a single <see cref="DecodeBatch"/> call may advance.</summary>
    public const int MaxBatchSequences = 16;

    private readonly IBackend _backend;
    private readonly IWeightSource _weights;

    // Only a source this instance created is its own to release: a caller-supplied source stays under the caller's
    // lifetime rules and must outlive the model.
    private readonly bool _ownsWeights;

    // Kept for tensor name lookups only: the mmap source owns payload access.
    private readonly LogicalTensorTable _fileTensors;

    // Fused qkv tensors the split role map resolved past, and the companion scales of nameable tensors, are
    // recorded at Open: the fused ones are reported as one structured line, the companion ones validated.
    private readonly List<string> _ignoredFusedTensors = [];
    private HashSet<string>? _nameableTensors;

    // Per-layer packed matrix tensors, resolved from the mmap view on each use so no dequantized copy is kept.
    private readonly LogicalTensor[] _wq;
    private readonly LogicalTensor[] _wk;
    private readonly LogicalTensor[] _wv;
    private readonly LogicalTensor[] _wo;
    private LogicalTensor[] _ffnGate = []; // empty when a sparse descriptor replaces the dense block
    private LogicalTensor[] _ffnUp = []; // empty when a sparse descriptor replaces the dense block
    private LogicalTensor[] _ffnDown = []; // empty when a sparse descriptor replaces the dense block

    // Sparse-FFN block state, resolved once at Open from the plan's descriptor; empty arrays and zero geometry mean
    // the dense gate/up/down triplet runs. The router tensor is [experts x hidden]; each expert matrix role is one
    // stacked tensor whose logical shape carries the expert count as its outermost dimension, so expert e's block is
    // the contiguous row range [e * rows .. (e + 1) * rows) and a per-expert GEMV reads it through a slice of that
    // role's payload with the fixed two-dimensional shape below. Every expert stays resident: routing selects among
    // these tensors instead of fetching them.
    private readonly SparseFfnDescriptor? _sparseFfn;
    private LogicalTensor[] _moeRouter = [];
    private LogicalTensor[] _moeExpertGate = [];
    private LogicalTensor[] _moeExpertUp = [];
    private LogicalTensor[] _moeExpertDown = [];
    private LogicalTensor[] _sharedGate = [];
    private LogicalTensor[] _sharedUp = [];
    private LogicalTensor[] _sharedDown = [];

    // The shared expert's per-token selector rows, dequantized once at load exactly like the norm and bias vectors.
    // One row of the hidden size per layer when the file ships them for every layer; an empty array when it ships
    // none, which is what leaves the shared contribution ungated.
    private float[][] _sharedInputGate = [];

    // Per-expert slice geometry: one expert's rows and cols as the shape a GEMV over its slice carries, plus that
    // slice's packed byte block per role — indexed by layer, because a converted file is free to hold different quant
    // tiers in the same role at different layers. Built once at Open so routing and expert execution allocate nothing
    // per token; empty means no sparse block runs.
    private TensorShape _moeExpertGateShape = new([0, 0]);
    private TensorShape _moeExpertUpShape = new([0, 0]);
    private TensorShape _moeExpertDownShape = new([0, 0]);
    private long[] _moeExpertGateBytes = [];
    private long[] _moeExpertUpBytes = [];
    private long[] _moeExpertDownBytes = [];

    // Routing scratch, sized at Open for descriptor-bearing models only. The token path consumes one
    // routing at a time; the batch path stores every sequence's selection flat, sequence by sequence, so
    // each expert group can pack its member rows into one batched gate/up/down pass instead of paying a
    // batch call per member. Selection order is ascending expert index, which is the order the batch path
    // groups by.

    private float[] _moeLogits = [];      // experts: router scores for one sequence, consumed during selection
    private int[] _moeIds = [];           // routedExpertsPerToken: selected expert indexes, ascending order
    private float[] _moeWeights = [];     // routedExpertsPerToken: the weight of each selected expert
    private int[] _batchMoeIds = [];      // MaxBatchSequences * routedExpertsPerToken: per-sequence selections
    private float[] _batchMoeWeights = [];// same length, aligned with _batchMoeIds
    private int[] _expertMembers = []; // MaxBatchSequences: member rows of the expert under execution
    private float[] _moeSubsetIn = []; // MaxBatchSequences * hidden: one expert group's activations and down results, packed in place

    // Small FP32 vectors preloaded at Open: per-layer norm weights and any biases present in the file. Null
    // entries mean the tensor is absent (llama files carry no attention bias, qwen2 files may).
    private readonly float[][] _attnNormWeight;
    private readonly float[][] _ffnNormWeight;
    private readonly float[]?[] _biasQ;
    private readonly float[]?[] _biasK;
    private readonly float[]?[] _biasV;
    private readonly float[]?[] _biasO;

    // Per-head QK-norm weight vectors (one headDim vector per layer, shared by every head of its side),
    // preloaded only when the plan resolves a weighted mode: every head normalizes its own headDim slice.
    private readonly float[][] _attnQNorm;
    private readonly float[][] _attnKNorm;

    // Post-sublayer norm weight vectors, preloaded only when the plan places norms on sublayer outputs: each holds one
    // hidden-sized vector per layer and normalizes that sublayer's own output before its residual add. A pre-only
    // placement loads neither and rejects a file carrying either tensor.
    private readonly float[][] _postAttnNorm;
    private readonly float[][] _postFfnNorm;

    // Norm-bias vectors preloaded only when the plan declares them: the LayerNorm path adds them after the
    // weighted normalization. Null entries mean the layer's tensor is absent.
    private readonly float[]?[] _attnNormBias;
    private readonly float[]?[] _ffnNormBias;
    private readonly float[]? _outputNormBias;

    private readonly LogicalTensor _tokenEmbd;
    private readonly LogicalTensor? _output;
    private readonly float[] _outputNormWeight;
    private readonly float[]? _outputBias;

    // FP32 buffers reused across every forward call.
    private readonly float[] _cur;     // hidden: layer input stream (residual carrier) and FFN output accumulator
    private readonly float[] _normed;  // hidden: weighted norm before the QKV / FFN GEMVs
    private readonly float[] _q;       // headCount * headDim
    private readonly float[] _k;       // kvHeads * headDim (also the KV slot write source)
    private readonly float[] _v;       // kvHeads * headDim
    private readonly float[] _attnOut; // headCount * headDim: heads concatenated in order for wo GEMV
    private readonly float[] _proj;    // hidden: attn_output / ffn_down GEMV target before the residual add
    private readonly float[] _gateUp;  // intermediateSize: gate result, then activation(gate) * up written back into it
    private readonly float[] _up;      // intermediateSize: up GEMV result
    private readonly float[] _logits;  // vocabSize

    // Batched decode buffers: sequence-major blocks of MaxBatchSequences copies of the single-token streams,
    // allocated once at Open so a batched step adds no per-step heap pressure. Each sequence's block mirrors the
    // same-sized buffer in its own single-token forward.
    private readonly float[] _batchCur;     // MaxBatchSequences * hidden
    private readonly float[] _batchNormed;  // MaxBatchSequences * hidden
    private readonly float[] _batchQ;       // MaxBatchSequences * headCount * headDim
    private readonly float[] _batchK;       // MaxBatchSequences * kvHeads * headDim
    private readonly float[] _batchV;       // MaxBatchSequences * kvHeads * headDim
    private readonly float[] _batchAttnOut; // MaxBatchSequences * headCount * headDim
    private readonly float[] _batchProj;    // MaxBatchSequences * hidden
    private readonly float[] _batchGateUp;  // MaxBatchSequences * intermediateSize
    private readonly float[] _batchUp;      // MaxBatchSequences * intermediateSize
    private readonly float[] _batchLogits;  // MaxBatchSequences * vocabSize

    private readonly float[] _cos;          // maxContext * rope pair count
    private readonly float[] _sin;
    private readonly int _ropeHalf;         // pairs the rotation covers: the declared slice, or the full head
    private readonly float _attnScale;      // head-dimension score gain times the variant's attention factor

    /// <summary>The resolved model plan this engine forwards under: geometry, profile tensor map, and strategy values.</summary>
    public ModelPlan Plan { get; }

    /// <summary>Flat scalar view of <see cref="Plan"/>, kept as the terse read surface for geometry consumers.</summary>
    public ModelDesc Desc { get; }

    /// <summary>Largest sequence length this model can forward: the extent of its precomputed RoPE tables.</summary>
    public int MaxContextTokens { get; }

    /// <summary>Opens the engine over a memory-mapped view of the file's own tensor payloads.</summary>
    public static TransformerModel Open(GgufFile file, IBackend backend, int maxContextTokens) =>
        Open(file, backend, maxContextTokens, logLine: null);

    /// <summary>
    /// Opens the engine over a memory-mapped view of the file's own tensor payloads and reports profile-
    /// resolution events (the generic llama-shaped fallback applied to an unknown architecture) through
    /// <paramref name="logLine"/>.
    /// </summary>
    public static TransformerModel Open(GgufFile file, IBackend backend, int maxContextTokens, Action<string>? logLine) =>
        new(file, backend, maxContextTokens, null, logLine);

    /// <summary>
    /// Opens the engine with weight payloads supplied by an alternate <see cref="IWeightSource"/>: every tensor read
    /// goes through that source instead of a mmap view. Tensor metadata (names, shapes, formats, offsets) still comes
    /// from <paramref name="file"/>, and the caller keeps ownership of the source — it must outlive the model and is
    /// disposed by the caller, not here.
    /// </summary>
    public static TransformerModel Open(GgufFile file, IWeightSource weights, IBackend backend, int maxContextTokens) =>
        Open(file, weights, backend, maxContextTokens, logLine: null);

    /// <summary>
    /// Opens the engine with an alternate <see cref="IWeightSource"/> and reports profile-resolution events (the
    /// generic llama-shaped fallback applied to an unknown architecture) through <paramref name="logLine"/>.
    /// </summary>
    public static TransformerModel Open(GgufFile file, IWeightSource weights, IBackend backend, int maxContextTokens, Action<string>? logLine) =>
        new(file, backend, maxContextTokens, weights, logLine);

    /// <summary>
    /// Opens the engine with an explicitly supplied <see cref="ModelProfile"/> resolving the plan instead of the
    /// registry selection: the profile's prefix supplies every metadata key and its strategy values drive the
    /// forward, with profile- and tensor-resolution events reported through <paramref name="logLine"/>. Assembly-internal
    /// so no host configuration path can reach it.
    /// </summary>
    internal static TransformerModel Open(GgufFile file, IBackend backend, int maxContextTokens, ModelProfile profile, Action<string>? logLine = null) =>
        new(file, backend, maxContextTokens, weights: null, logLine, profile);

    private TransformerModel(GgufFile file, IBackend backend, int maxContextTokens, IWeightSource? weights = null, Action<string>? logLine = null, ModelProfile? profileOverride = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxContextTokens);

        _backend = backend;
        Plan = profileOverride is null ? ModelPlans.Resolve(file, logLine) : ModelPlans.Resolve(file, profileOverride, logLine);
        Desc = Plan.ToDesc();
        MaxContextTokens = maxContextTokens;
        _ownsWeights = weights is null;
        _weights = weights ?? new GgufMmapWeightSource(file);
        _fileTensors = file.Tensors;

        if (Plan.HeadDim % 2 != 0)
        {
            throw new ModelException($"head dimension {Plan.HeadDim} is odd; RoPE pairs require an even head dimension");
        }

        // A resolved sparse-FFN descriptor replaces this block's dense gate/up/down triplet with expert-routed
        // execution: the dense feed-forward roles are neither required nor loaded, and any dense feed-forward tensor
        // such a file still ships is reported as ignored. Every declared expert loads here — all experts of every
        // layer stay resident in the mmap view for the model's lifetime, so forward-time routing selects among them
        // instead of fetching them, and no residency machinery runs on this path. The dense gate/up/down block stays
        // exactly as it was for a file that declares no expert count.
        _sparseFfn = Plan.SparseFfn;

        // A resolved strategy other than the dense-resident variant is a shape this engine has no kernel for: the plan
        // can represent it, so the load records it and reports here rather than running some other variant behind the
        // selection the file declared. The message names the strategy, what the descriptor carries for it, and the one
        // value that runs today — nothing falls back silently.
        if (Plan.SparseFfnStrategy is { } resolvedStrategy && resolvedStrategy != SparseFfnStrategy.DenseResident)
        {
            throw new ModelException(
                $"sparse-FFN strategy '{resolvedStrategy}' is not implemented by the forward engine: architecture " +
                $"'{Plan.Architecture}' declares a sparse-FFN block with {_sparseFfn!.ExpertCount} experts routing " +
                $"{_sparseFfn.RoutedExpertsPerToken} per token over a shared basis at rank {_sparseFfn.Rank?.ToString(CultureInfo.InvariantCulture) ?? "an undeclared rank"}: " +
                $"only {SparseFfnStrategy.DenseResident} executes today — declare {Plan.Prefix}sparse_ffn_strategy 'dense_resident' to run this file");
        }

        int layers = Plan.LayerCount;

        _wq = LoadMatrices(layers, TensorRole.AttentionQuery, FusedQkvName(TensorRole.AttentionQuery));
        _wk = LoadMatrices(layers, TensorRole.AttentionKey, FusedQkvName(TensorRole.AttentionKey));
        _wv = LoadMatrices(layers, TensorRole.AttentionValue, FusedQkvName(TensorRole.AttentionValue));
        _wo = LoadMatrices(layers, TensorRole.AttentionOutput);
        if (_sparseFfn is null)
        {
            _ffnGate = LoadMatrices(layers, TensorRole.FeedForwardGate);
            _ffnUp = LoadMatrices(layers, TensorRole.FeedForwardUp);
            _ffnDown = LoadMatrices(layers, TensorRole.FeedForwardDown);
        }
        else
        {
            int experts = _sparseFfn.ExpertCount;

            // The router emits one score per declared expert from the hidden stream. Each stacked role is one tensor
            // whose logical shape carries that expert count as its outermost dimension followed by this matrix's rows
            // and columns, so a GEMV over one expert reads a contiguous row range of the stack: gate/up experts emit
            // intermediateSize values from hidden inputs, down experts emit hidden values from intermediate inputs.
            _moeRouter = LoadMatrices(layers, _sparseFfn.RouterRole);
            RequireShape(_moeRouter, experts, Plan.HiddenSize);

            LogicalTensor[] expertGate = LoadExpertStacks(layers, _sparseFfn.ExpertGateRole, experts, Plan.IntermediateSize, Plan.HiddenSize);
            LogicalTensor[] expertUp = LoadExpertStacks(layers, _sparseFfn.ExpertUpRole, experts, Plan.IntermediateSize, Plan.HiddenSize);
            LogicalTensor[] expertDown = LoadExpertStacks(layers, _sparseFfn.ExpertDownRole, experts, Plan.HiddenSize, Plan.IntermediateSize);

            // Per-expert slice geometry for these roles: one expert's packed block and the two-dimensional shape a
            // GEMV over that slice carries, per layer — each role sizes its slice from that layer's own tensor format
            // and row stride, because a converted file is free to hold different quant tiers in one role at different
            // layers (the reference converter does exactly that), while LoadExpertStacks has already pinned every
            // layer of a role to the same rows and columns.
            _moeExpertGate = expertGate;
            _moeExpertUp = expertUp;
            _moeExpertDown = expertDown;
            long[] gateBytes = new long[layers];
            long[] upBytes = new long[layers];
            long[] downBytes = new long[layers];
            for (int l = 0; l < layers; l++)
            {
                gateBytes[l] = Plan.IntermediateSize * expertGate[l].Shape.RowStrideBytes(expertGate[l].Format);
                upBytes[l] = Plan.IntermediateSize * expertUp[l].Shape.RowStrideBytes(expertUp[l].Format);
                downBytes[l] = Plan.HiddenSize * expertDown[l].Shape.RowStrideBytes(expertDown[l].Format);
            }

            _moeExpertGateShape = new TensorShape([Plan.IntermediateSize, Plan.HiddenSize]);
            _moeExpertUpShape = new TensorShape([Plan.IntermediateSize, Plan.HiddenSize]);
            _moeExpertDownShape = new TensorShape([Plan.HiddenSize, Plan.IntermediateSize]);
            _moeExpertGateBytes = gateBytes;
            _moeExpertUpBytes = upBytes;
            _moeExpertDownBytes = downBytes;

            // Routing scratch and packed-expert storage sized from the descriptor's concrete counts: one selection at
            // a time for routing plus per-sequence copies of every batched selection, member-row indexes for one
            // expert group, and one hidden-sized buffer carrying each group's activations and down results in place.
            _moeLogits = new float[_sparseFfn.ExpertCount];
            _moeIds = new int[_sparseFfn.RoutedExpertsPerToken];
            _moeWeights = new float[_sparseFfn.RoutedExpertsPerToken];
            _batchMoeIds = new int[MaxBatchSequences * _sparseFfn.RoutedExpertsPerToken];
            _batchMoeWeights = new float[MaxBatchSequences * _sparseFfn.RoutedExpertsPerToken];
            _expertMembers = new int[MaxBatchSequences];
            _moeSubsetIn = new float[MaxBatchSequences * Plan.HiddenSize];

            // A declared shared expert is one dense-shaped matrix triplet the routing adds to every token's output,
            // plus — for a family whose reference gates it — one selector vector per layer. An undeclared shared
            // block loads and reads nothing here at all.
            if (_sparseFfn.HasSharedExpert)
            {
                TensorRole gateRole = RequiredSparseRole(_sparseFfn.SharedGateRole, "gate");
                TensorRole upRole = RequiredSparseRole(_sparseFfn.SharedUpRole, "up");
                TensorRole downRole = RequiredSparseRole(_sparseFfn.SharedDownRole, "down");

                _sharedGate = LoadMatrices(layers, gateRole);
                _sharedUp = LoadMatrices(layers, upRole);
                _sharedDown = LoadMatrices(layers, downRole);
                RequireShape(_sharedGate, Plan.IntermediateSize, Plan.HiddenSize);
                RequireShape(_sharedUp, Plan.IntermediateSize, Plan.HiddenSize);
                RequireShape(_sharedDown, Plan.HiddenSize, Plan.IntermediateSize);

                // The reference also ships one selector vector of the hidden size per layer for the shared expert and
                // takes this token's score from it. A file that carries none of them stays ungated; a table carrying
                // them for only part of the block fails here rather than gating some layers and not others.
                if (_sparseFfn.SharedInputGateRole is { } inputGateRole)
                {
                    _sharedInputGate = LoadSharedExpertSelectors(layers, Plan.Tensors[inputGateRole]);
                }
            }

            if (logLine is not null)
            {
                ReportIgnoredTensors(logLine, "dense_ffn_ignored", file, [TensorRole.FeedForwardGate, TensorRole.FeedForwardUp, TensorRole.FeedForwardDown]);
            }
        }

        // Matrix shapes must agree with the declared dimensions: rows are the output side, columns the input side.
        RequireShape(_wq, (long)Plan.HeadCount * Plan.HeadDim, Plan.HiddenSize); // q emits headCount * head_dim
        RequireShape(_wk, (long)Plan.KvHeadCount * Plan.HeadDim, Plan.HiddenSize);
        RequireShape(_wv, (long)Plan.KvHeadCount * Plan.HeadDim, Plan.HiddenSize);
        RequireShape(_wo, Plan.HiddenSize, (long)Plan.HeadCount * Plan.HeadDim);
        if (_sparseFfn is null)
        {
            RequireShape(_ffnGate, Plan.IntermediateSize, Plan.HiddenSize);
            RequireShape(_ffnUp, Plan.IntermediateSize, Plan.HiddenSize);
            RequireShape(_ffnDown, Plan.HiddenSize, Plan.IntermediateSize);
        }

        // Quantization companions of nameable tensors are validated before anything loads, and fused qkv
        // tensors the split map resolved past are reported once, in the style of the ignored-tensor lines.
        ValidateCompanionTensors();

        if (logLine is not null && _ignoredFusedTensors.Count > 0)
        {
            GgufMetadata metadata = file.Metadata;
            string modelName = metadata.Has("general.name") ? metadata.Get("general.name").AsString() : "";
            logLine(
                $"tensors_ignored model='{(modelName.Length > 0 ? modelName : file.Path)}' " +
                $"architecture='{Plan.Architecture}' tensors={string.Join(",", _ignoredFusedTensors)}");
        }

        _attnNormWeight = LoadLayerVectors(layers, TensorRole.AttentionNorm, Plan.HiddenSize);
        _ffnNormWeight = LoadLayerVectors(layers, TensorRole.FeedForwardNorm, Plan.HiddenSize);

        if (Plan.Bias == BiasPolicy.Declared && !Plan.UseAttentionBias)
        {
            // Declared false: present attention-bias tensors are neither loaded nor applied, and the file is
            // reported once naming them.
            _biasQ = new float[layers][];
            _biasK = new float[layers][];
            _biasV = new float[layers][];
            _biasO = new float[layers][];
            ReportIgnoredTensors(
                logLine,
                "bias_ignored",
                file,
                [TensorRole.AttentionQueryBias, TensorRole.AttentionKeyBias, TensorRole.AttentionValueBias, TensorRole.AttentionOutputBias]);
        }
        else
        {
            // Resolved on: the q/k/v bias vectors are required — a declared use_bias or a profile default that
            // turns the feature on makes them part of the file's graph, and the failure names the first one the
            // file lacks. The output-projection bias is used when present but demanded by no resolved value:
            // only some conversions ship one. ByTensorPresence keeps the presence rule for the q/k/v side too.
            bool required = Plan.Bias == BiasPolicy.Declared;
            _biasQ = LoadAttentionBias(layers, TensorRole.AttentionQueryBias, (long)Plan.HeadCount * Plan.HeadDim, required);
            _biasK = LoadAttentionBias(layers, TensorRole.AttentionKeyBias, (long)Plan.KvHeadCount * Plan.HeadDim, required);
            _biasV = LoadAttentionBias(layers, TensorRole.AttentionValueBias, (long)Plan.KvHeadCount * Plan.HeadDim, required);
            _biasO = LoadAttentionBias(layers, TensorRole.AttentionOutputBias, Plan.HiddenSize, required: false);
        }

        if (Plan.UseNormBias)
        {
            _attnNormBias = LoadLayerVectors(layers, TensorRole.AttentionNormBias, Plan.HiddenSize);
            _ffnNormBias = LoadLayerVectors(layers, TensorRole.FeedForwardNormBias, Plan.HiddenSize);
        }
        else
        {
            _attnNormBias = new float[layers][];
            _ffnNormBias = new float[layers][];
            ReportIgnoredTensors(logLine, "norm_bias_ignored", file, [TensorRole.AttentionNormBias, TensorRole.FeedForwardNormBias]);
        }

        switch (Plan.QkNorm)
        {
            case QkNormMode.PerHeadWeighted:
                _attnQNorm = LoadLayerVectors(layers, TensorRole.AttentionQueryNorm, Plan.HeadDim);
                _attnKNorm = LoadLayerVectors(layers, TensorRole.AttentionKeyNorm, Plan.HeadDim);
                break;

            case QkNormMode.StreamWeighted:
                // The whole-stream form normalizes each side's projected width at once, before any head split, so the
                // file's vector must cover that side's full projection — head_count × head_dim for q and kv_head_count ×
                // head_dim for k — rather than one head. A per-head-sized vector fails the load naming its shape.
                _attnQNorm = LoadLayerVectors(layers, TensorRole.AttentionQueryNorm, Plan.HeadCount * Plan.HeadDim);
                _attnKNorm = LoadLayerVectors(layers, TensorRole.AttentionKeyNorm, Plan.KvHeadCount * Plan.HeadDim);
                break;

            case QkNormMode.Off:
                // Resolved off: present per-head vectors are neither loaded nor applied, and the file is reported once naming them.
                _attnQNorm = new float[layers][];
                _attnKNorm = new float[layers][];
                ReportIgnoredTensors(logLine, "qk_norm_ignored", file, [TensorRole.AttentionQueryNorm, TensorRole.AttentionKeyNorm]);
                break;

            case QkNormMode.PerHeadUnweighted:
                // The unweighted form normalizes with the plan epsilon alone and demands no tensor at all. Present
                // per-head vectors are reported as unused rather than mistaken for a requirement this mode cannot express.
                _attnQNorm = new float[layers][];
                _attnKNorm = new float[layers][];
                ReportIgnoredTensors(logLine, "qk_norm_vectors_unused", file, [TensorRole.AttentionQueryNorm, TensorRole.AttentionKeyNorm]);
                break;

            default:
                throw new ModelException($"QK-norm mode '{Plan.QkNorm}' is not implemented by the forward engine");
        }

        if (Plan.NormPlacement == NormPlacement.PostSublayerOutputs)
        {
            _postAttnNorm = LoadPostOutputNorms(layers, TensorRole.PostAttentionNorm);
            _postFfnNorm = LoadPostOutputNorms(layers, TensorRole.PostFeedForwardNorm);
        }
        else
        {
            // A pre-only placement reads no post-sublayer norm vector: a file carrying one describes a graph this
            // profile does not implement, so it fails naming every such tensor instead of leaving them unread.
            RejectUnconsumedPostNorms(layers);
            _postAttnNorm = new float[layers][];
            _postFfnNorm = new float[layers][];
        }

        LogicalTensor tokenEmbd = RequireTensor(TensorRole.TokenEmbedding);
        RequireSingle(tokenEmbd, Plan.VocabSize, Plan.HiddenSize);
        _tokenEmbd = tokenEmbd;

        LogicalTensor outputNorm = RequireTensor(TensorRole.OutputNorm);
        _outputNormWeight = LoadVector(outputNorm, Plan.HiddenSize);
        _outputNormBias = Plan.UseNormBias
            ? LoadOptionalVector(Plan.Tensors[TensorRole.OutputNormBias].Resolve(_fileTensors), Plan.HiddenSize)
            : null;

        LogicalTensor? output = Plan.Tensors[TensorRole.OutputHead].Resolve(_fileTensors);
        if (output is not null)
        {
            RequireSingle(output, Plan.VocabSize, Plan.HiddenSize);
        }

        _output = output;
        _outputBias = LoadOptionalVector(Plan.Tensors[TensorRole.OutputBias].Resolve(_fileTensors), Plan.VocabSize);

        int hidden = Plan.HiddenSize;
        _cur = new float[hidden];
        _normed = new float[hidden];
        _q = new float[Plan.HeadCount * Plan.HeadDim];
        _k = new float[Plan.KvHeadCount * Plan.HeadDim];
        _v = new float[Plan.KvHeadCount * Plan.HeadDim];
        _attnOut = new float[Plan.HeadCount * Plan.HeadDim];
        _proj = new float[hidden];
        _gateUp = new float[Plan.IntermediateSize];
        _up = new float[Plan.IntermediateSize];
        _logits = new float[Plan.VocabSize];

        _batchCur = new float[MaxBatchSequences * hidden];
        _batchNormed = new float[MaxBatchSequences * hidden];
        _batchQ = new float[MaxBatchSequences * Plan.HeadCount * Plan.HeadDim];
        _batchK = new float[MaxBatchSequences * Plan.KvHeadCount * Plan.HeadDim];
        _batchV = new float[MaxBatchSequences * Plan.KvHeadCount * Plan.HeadDim];
        _batchAttnOut = new float[MaxBatchSequences * Plan.HeadCount * Plan.HeadDim];
        _batchProj = new float[MaxBatchSequences * hidden];
        _batchGateUp = new float[MaxBatchSequences * Plan.IntermediateSize];
        _batchUp = new float[MaxBatchSequences * Plan.IntermediateSize];
        _batchLogits = new float[MaxBatchSequences * Plan.VocabSize];

        // The rotation covers the declared rotary slice, or the full head dimension when the file declares none.
        _ropeHalf = (Plan.RopeRotaryDimensions > 0 ? Plan.RopeRotaryDimensions : Plan.HeadDim) / 2;

        // The score gain is the head-dimension scale times the variant's attention factor: the scaled angle
        // tables carry their own magnitude, so the score path carries the factor exactly once. An unscaled plan
        // multiplies by one and the gain is the plain head-dimension scale.
        _attnScale = (Plan.RopeScaling?.AttentionFactor ?? 1f) / MathF.Sqrt(Plan.HeadDim);

        // Precompute cos/sin per position and pair up to the maximum context so decode is table-driven: identical
        // inputs always reproduce bit-identical rotations regardless of call order or buffer reuse. A resolved
        // scaling plan builds the variant's tables over the slice instead of the unscaled full-head tables.
        _cos = new float[maxContextTokens * _ropeHalf];
        _sin = new float[maxContextTokens * _ropeHalf];
        if (Plan.RopeScaling is { } scaling)
        {
            LayerOps.BuildRopeScalingTables(
                Plan.RopeTheta,
                Plan.HeadDim,
                2 * _ropeHalf,
                maxContextTokens,
                scaling.Multiplier,
                scaling.OriginalContextLength,
                scaling.AttentionFactor,
                scaling.CorrectionScale,
                scaling.BetaFast,
                scaling.BetaSlow,
                _cos,
                _sin);
        }
        else
        {
            LayerOps.BuildRopeTables(Plan.RopeTheta, Plan.HeadDim, maxContextTokens, _cos, _sin);
        }

        // A positional plan that is not the plain full-head rotation is announced once at Open, in the style of the
        // tensor and profile lines: the variant, its multiplier and original context length, the attention factor
        // carried into the score path, and the dimension count the rotation actually covers. The plain shape stays
        // unannounced so the line means the same thing on every host.
        if (logLine is not null && (Plan.RopeScaling is not null || Plan.RopeRotaryDimensions > 0))
        {
            GgufMetadata metadata = file.Metadata;
            string modelName = metadata.Has("general.name") ? metadata.Get("general.name").AsString() : "";
            RopeScalingPlan? plan = Plan.RopeScaling;
            logLine(
                $"rope_scaling model='{(modelName.Length > 0 ? modelName : file.Path)}' " +
                $"architecture='{Plan.Architecture}' variant={(plan?.Variant ?? "none")} " +
                $"multiplier={(plan?.Multiplier ?? 1f).ToString(CultureInfo.InvariantCulture)} " +
                $"original_context_length={(plan?.OriginalContextLength ?? 0)} " +
                $"attention_factor={(plan?.AttentionFactor ?? 1f).ToString(CultureInfo.InvariantCulture)} " +
                $"rotary_dimensions={(Plan.RopeRotaryDimensions > 0 ? Plan.RopeRotaryDimensions : Plan.HeadDim)}");
        }
    }

    /// <summary>Forwards each token in order, appending its K/V to the sequence's paged KV cache.</summary>
    public void Prefill(IReadOnlyList<int> tokens, SequenceKv kv)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(kv);

        foreach (int tokenId in tokens)
        {
            ForwardToken(tokenId, kv.Length, kv);
        }
    }

    /// <summary>Forwards one token at the next free position and returns its vocab-sized logits.</summary>
    public Span<float> DecodeNext(int tokenId, SequenceKv kv)
    {
        ArgumentNullException.ThrowIfNull(kv);

        ForwardToken(tokenId, kv.Length, kv);
        return _logits.AsSpan();
    }

    /// <summary>
    /// Logits from the most recent forward: after a prefill these are already the next-token distribution for the
    /// sequence's final token — sampling them avoids re-forwarding (and double-appending) that token.
    /// </summary>
    public Span<float> LastLogits() => _logits.AsSpan();

    /// <summary>
    /// Advances up to <see cref="MaxBatchSequences"/> sequences by one token each in a single step: every dense
    /// weight matrix (attention q/k/v/o, feed-forward gate/up/down, lm_head) is traversed once for the whole batch
    /// through <see cref="IBackend.GemvBatch"/>, while norms, RoPE, causal attention, and the gated activation stay
    /// per-sequence because each sequence attends over its own KV history. Each sequence appends at its own <see cref="SequenceKv.Length"/>
    /// position, exactly as <see cref="DecodeNext"/> would, and every batched result is bit-identical to stepping that
    /// sequence alone. Sequences must belong to the paged KV pool this engine's forwards write into. The returned
    /// logits are owned by the model — valid until the next forward call — laid out as
    /// <c>logits[sequence * vocabSize + token]</c>.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// The two lists differ in length, fewer than one or more than <see cref="MaxBatchSequences"/> sequences were
    /// given, or a list entry is null.
    /// </exception>
    public Span<float> DecodeBatch(IReadOnlyList<int> tokenIds, IReadOnlyList<SequenceKv> kvs)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        ArgumentNullException.ThrowIfNull(kvs);

        int n = tokenIds.Count;
        if (n != kvs.Count)
        {
            throw new ArgumentException($"tokenIds has {tokenIds.Count} entries but kvs has {kvs.Count}: one token per sequence is required.", nameof(kvs));
        }

        if (n == 0 || n > MaxBatchSequences)
        {
            throw new ArgumentException($"a decode batch must hold between 1 and {MaxBatchSequences} sequences, got {n}.", nameof(tokenIds));
        }

        int[] positions = new int[n];
        for (int s = 0; s < n; s++)
        {
            if (tokenIds[s] < 0 || tokenIds[s] >= Plan.VocabSize)
            {
                throw new ArgumentException($"token id {tokenIds[s]} at sequence {s} is outside the vocabulary of {Plan.VocabSize}.", nameof(tokenIds));
            }

            SequenceKv kv = kvs[s] ?? throw new ArgumentException($"kvs entry {s} is null.", nameof(kvs));
            if (kv.Length >= MaxContextTokens)
            {
                throw new InvalidOperationException(
                    $"sequence {s} position {kv.Length} exceeds this model's max context of {MaxContextTokens} tokens.");
            }

            positions[s] = kv.Length;
        }

        // Embedding lookup: each token's dequantized token_embd row seeds its sequence's input stream.
        ReadOnlySpan<byte> embBytes = _weights.Resolve(_tokenEmbd);
        WeightRef embRef = new(embBytes, _tokenEmbd.Shape, _tokenEmbd.Format);
        for (int s = 0; s < n; s++)
        {
            _backend.DequantRow(embRef, tokenIds[s], _batchCur.AsSpan(s * Plan.HiddenSize, Plan.HiddenSize));
        }

        ApplyInputEmbeddingScale(_batchCur.AsSpan(0, n * Plan.HiddenSize));

        // Reserve this step's slot in every sequence before any layer writes, mirroring ForwardToken's append.
        for (int s = 0; s < n; s++)
        {
            kvs[s].Append();
        }

        int hidden = Plan.HiddenSize;
        int qkvRegion = Plan.HeadCount * Plan.HeadDim;
        int kvRegion = Plan.KvHeadCount * Plan.HeadDim;
        int inter = Plan.IntermediateSize;
        float attnScale = _attnScale;

        for (int l = 0; l < Plan.LayerCount; l++)
        {
            int layerWindow = LayerAttentionWindow(l);

            // Pre-attention norm per sequence: identical op to the single-token forward.
            for (int s = 0; s < n; s++)
            {
                ApplyNorm(_batchCur.AsSpan(s * hidden, hidden), _attnNormWeight[l], _attnNormBias[l], _batchNormed.AsSpan(s * hidden, hidden));
            }

            // One weight pass per matrix for the whole batch; each sequence's row result matches its single-vector GEMV.
            ReadOnlySpan<byte> bytes = _weights.Resolve(_wq[l]);
            _backend.GemvBatch(_batchNormed.AsSpan(0, n * hidden), n, new WeightRef(bytes, _wq[l].Shape, _wq[l].Format), _batchQ.AsSpan(0, n * qkvRegion));

            bytes = _weights.Resolve(_wk[l]);
            _backend.GemvBatch(_batchNormed.AsSpan(0, n * hidden), n, new WeightRef(bytes, _wk[l].Shape, _wk[l].Format), _batchK.AsSpan(0, n * kvRegion));

            bytes = _weights.Resolve(_wv[l]);
            _backend.GemvBatch(_batchNormed.AsSpan(0, n * hidden), n, new WeightRef(bytes, _wv[l].Shape, _wv[l].Format), _batchV.AsSpan(0, n * kvRegion));

            for (int s = 0; s < n; s++)
            {
                AddBias(_batchQ.AsSpan(s * qkvRegion, qkvRegion), _biasQ[l]);
                AddBias(_batchK.AsSpan(s * kvRegion, kvRegion), _biasK[l]);
                AddBias(_batchV.AsSpan(s * kvRegion, kvRegion), _biasV[l]);

                ApplyQkNorm(_batchQ.AsSpan(s * qkvRegion, qkvRegion), _attnQNorm[l]);
                ApplyQkNorm(_batchK.AsSpan(s * kvRegion, kvRegion), _attnKNorm[l]);

                ApplyRopeHeads(_batchQ.AsSpan(s * qkvRegion, qkvRegion), Plan.HeadCount, positions[s]);
                ApplyRopeHeads(_batchK.AsSpan(s * kvRegion, kvRegion), Plan.KvHeadCount, positions[s]);

                kvs[s].WriteK(positions[s], l, _batchK.AsSpan(s * kvRegion, kvRegion));
                kvs[s].WriteV(positions[s], l, _batchV.AsSpan(s * kvRegion, kvRegion));

                LayerOps.CausalAttention(_backend, _batchQ.AsSpan(s * qkvRegion, qkvRegion), Plan.HeadCount, Plan.KvHeadCount, Plan.HeadDim, kvs[s], l, positions[s], attnScale, Plan.AttentionSoftcap, layerWindow, _batchAttnOut.AsSpan(s * qkvRegion, qkvRegion));
            }

            bytes = _weights.Resolve(_wo[l]);
            _backend.GemvBatch(_batchAttnOut.AsSpan(0, n * qkvRegion), n, new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), _batchProj.AsSpan(0, n * hidden));

            for (int s = 0; s < n; s++)
            {
                AddBias(_batchProj.AsSpan(s * hidden, hidden), _biasO[l]);

                // Post-sublayer placement: this sequence's attention output joins its stream only after the post-attention norm.
                ApplyPostSublayerNorm(_batchProj.AsSpan(s * hidden, hidden), _postAttnNorm[l]);

                // ffn_inp add on the sequence's own stream, same order as the single-token forward.
                Span<float> cur = _batchCur.AsSpan(s * hidden, hidden);
                ReadOnlySpan<float> proj = _batchProj.AsSpan(s * hidden, hidden);
                for (int i = 0; i < hidden; i++)
                {
                    cur[i] += proj[i];
                }

                ApplyNorm(cur, _ffnNormWeight[l], _ffnNormBias[l], _batchNormed.AsSpan(s * hidden, hidden));
            }

            if (_sparseFfn is { } routing)
            {
                ApplySparseFeedForwardBatch(routing, l, n);
            }
            else
            {
                bytes = _weights.Resolve(_ffnGate[l]);
                _backend.GemvBatch(_batchNormed.AsSpan(0, n * hidden), n, new WeightRef(bytes, _ffnGate[l].Shape, _ffnGate[l].Format), _batchGateUp.AsSpan(0, n * inter));

                bytes = _weights.Resolve(_ffnUp[l]);
                _backend.GemvBatch(_batchNormed.AsSpan(0, n * hidden), n, new WeightRef(bytes, _ffnUp[l].Shape, _ffnUp[l].Format), _batchUp.AsSpan(0, n * inter));

                // act = activation(gate) * up in place on each sequence's gate buffer.
                for (int s = 0; s < n; s++)
                {
                    ApplyActivation(_batchGateUp.AsSpan(s * inter, inter), _batchUp.AsSpan(s * inter, inter));
                }

                bytes = _weights.Resolve(_ffnDown[l]);
                _backend.GemvBatch(_batchGateUp.AsSpan(0, n * inter), n, new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), _batchProj.AsSpan(0, n * hidden));
            }

            for (int s = 0; s < n; s++)
            {
                // Post-sublayer placement: this sequence's feed-forward output joins its stream only after the post-feed-forward norm.
                ApplyPostSublayerNorm(_batchProj.AsSpan(s * hidden, hidden), _postFfnNorm[l]);

                // ffn_out + ffn_inp: the input stream for this sequence's next layer.
                Span<float> cur = _batchCur.AsSpan(s * hidden, hidden);
                ReadOnlySpan<float> proj = _batchProj.AsSpan(s * hidden, hidden);
                for (int i = 0; i < hidden; i++)
                {
                    cur[i] += proj[i];
                }
            }
        }

        for (int s = 0; s < n; s++)
        {
            ApplyNorm(_batchCur.AsSpan(s * hidden, hidden), _outputNormWeight, _outputNormBias, _batchNormed.AsSpan(s * hidden, hidden));
        }

        LogicalTensor lmHead = _output ?? _tokenEmbd; // tied weights reuse the token-embedding rows as logits
        ReadOnlySpan<byte> headBytes = _weights.Resolve(lmHead);
        _backend.GemvBatch(_batchNormed.AsSpan(0, n * hidden), n, new WeightRef(headBytes, lmHead.Shape, lmHead.Format), _batchLogits.AsSpan(0, n * Plan.VocabSize));

        if (_outputBias is not null)
        {
            for (int s = 0; s < n; s++)
            {
                AddBias(_batchLogits.AsSpan(s * Plan.VocabSize, Plan.VocabSize), _outputBias);
            }
        }

        ApplyFinalLogitSoftcap(_batchLogits.AsSpan(0, n * Plan.VocabSize));

        return _batchLogits.AsSpan(0, n * Plan.VocabSize);
    }

    /// <summary>Token positions one encoder pass advances through the dense matrices per batched weight traversal.</summary>
    public const int EncodeChunkTokens = 8;

    /// <summary>Largest token count a single <see cref="Encode"/> call may forward.</summary>
    public const int MaxEncodeTokens = 256;

    /// <summary>
    /// Encoder-only forward: runs the whole token sequence through every layer with full-sequence (bidirectional)
    /// attention — each query position scores against all positions of its own sequence's KV contents instead of a
    /// causal prefix, narrowed per layer to the plan's resolved attention window around the query's own position on
    /// the layers its resolved window interleave masks — and returns the final hidden states after the output norm,
    /// without lm_head logits. Per-position norms, RoPE at absolute positions, biases, residuals, and the gated
    /// activation are exactly the decode-path
    /// operations; K/V for
    /// every position is appended to <paramref name="kv"/> as usual (positions start at <see cref="SequenceKv.Length"/>,
    /// so encoding a fresh sequence keeps each token attending only to its own sequence). Dense matrices are traversed
    /// once per chunk of up to <see cref="EncodeChunkTokens"/> positions via <see cref="IBackend.GemvBatch"/>; every
    /// position then attends over the whole finished KV span, which is what distinguishes this path from incremental
    /// causal decode. All queries — whichever chunk they belong to — see exactly every position of their sequence after
    /// appending (0..<see cref="SequenceKv.Length"/>) in ascending order — within the resolved window around its own
    /// position when one is declared — so results are independent of chunking.
    /// This establishes the dense-LLM architecture running in encoder mode as groundwork for embedding extraction;
    /// dedicated encoder-native architectures (ModernBERT-class) require their own tensor layouts and arrive with the
    /// embeddings endpoint itself. The returned array is caller-owned: <c>hidden[position * HiddenSize + i]</c>, length
    /// <c>tokens.Count * HiddenSize</c>.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// The token list is empty or larger than <see cref="MaxEncodeTokens"/>, a token id is outside the vocabulary, or
    /// appending the sequence would exceed this model's max context.
    /// </exception>
    public float[] Encode(IReadOnlyList<int> tokens, SequenceKv kv)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(kv);

        int n = tokens.Count;
        if (n == 0 || n > MaxEncodeTokens)
        {
            throw new ArgumentException(
                $"an encoder forward must hold between 1 and {MaxEncodeTokens} tokens, got {n}.", nameof(tokens));
        }

        int hidden = Plan.HiddenSize;
        int basePosition = kv.Length;
        if (basePosition + n > MaxContextTokens)
        {
            throw new ArgumentException(
                $"{n} encoder tokens starting at position {basePosition} exceed this model's max context of {MaxContextTokens}.",
                nameof(tokens));
        }

        if (_sparseFfn is { } encoderSparse)
        {
            throw new ModelException(
                $"architecture '{Plan.Architecture}' declares a sparse-FFN block with {encoderSparse.ExpertCount} experts " +
                $"routing {encoderSparse.RoutedExpertsPerToken} per token: the encoder forward runs dense gate/up/down blocks only");
        }

        int qkvRegion = Plan.HeadCount * Plan.HeadDim;
        int kvRegion = Plan.KvHeadCount * Plan.HeadDim;
        int inter = Plan.IntermediateSize;
        float attnScale = _attnScale;

        ArrayPool<float> poolBuffers = ArrayPool<float>.Shared;

        // Position-major working streams for the whole sequence plus chunk-local scratch for batched GEMVs. The rented
        // arrays are returned before any throw past this point, so a mid-call failure cannot leak pool entries.
        float[] cur = poolBuffers.Rent(n * hidden);          // residual carrier per position; becomes the final stream
        float[] normed = poolBuffers.Rent(n * hidden);       // rmsnorm result feeding the QKV / FFN GEMVs
        float[] qAll = poolBuffers.Rent(n * qkvRegion);      // every position's rotated queries, kept for attention
        float[] attnOut = poolBuffers.Rent(n * qkvRegion);   // concatenated heads per position before wo
        float[] chunkNormed = poolBuffers.Rent(EncodeChunkTokens * hidden);
        float[] chunkQ = poolBuffers.Rent(EncodeChunkTokens * qkvRegion);
        float[] chunkK = poolBuffers.Rent(EncodeChunkTokens * kvRegion);
        float[] chunkV = poolBuffers.Rent(EncodeChunkTokens * kvRegion);
        float[] chunkProj = poolBuffers.Rent(EncodeChunkTokens * hidden);
        float[] chunkGateUp = poolBuffers.Rent(EncodeChunkTokens * inter);
        float[] chunkUp = poolBuffers.Rent(EncodeChunkTokens * inter);

        try
        {
            // Embedding lookup: each token's dequantized token_embd row seeds that position's input stream.
            ReadOnlySpan<byte> embBytes = _weights.Resolve(_tokenEmbd);
            WeightRef embRef = new(embBytes, _tokenEmbd.Shape, _tokenEmbd.Format);
            for (int i = 0; i < n; i++)
            {
                if ((uint)tokens[i] >= (uint)Plan.VocabSize)
                {
                    throw new ArgumentException($"token id {tokens[i]} at index {i} is outside the vocabulary of {Plan.VocabSize}.", nameof(tokens));
                }

                _backend.DequantRow(embRef, tokens[i], cur.AsSpan(i * hidden, hidden));
            }

            ApplyInputEmbeddingScale(cur.AsSpan(0, n * hidden));

            // Reserve every position's KV slot up front so later layer passes address them by absolute position.
            for (int i = 0; i < n; i++)
            {
                kv.Append();
            }

            int chunkCount = (n + EncodeChunkTokens - 1) / EncodeChunkTokens;

            for (int l = 0; l < Plan.LayerCount; l++)
            {
                int layerWindow = LayerAttentionWindow(l);

                // Phase 1: pre-attention norm, QKV GEMVs (+bias), RoPE at absolute positions, and the K/V writes — all
                // per chunk. Queries stay in qAll so no position's attention runs before every position's K/V exists.
                for (int c = 0; c < chunkCount; c++)
                {
                    int firstPos = basePosition + c * EncodeChunkTokens;
                    int m = Math.Min(EncodeChunkTokens, n - c * EncodeChunkTokens);

                    for (int i = 0; i < m; i++)
                    {
                        ApplyNorm(cur.AsSpan((c * EncodeChunkTokens + i) * hidden, hidden), _attnNormWeight[l], _attnNormBias[l], chunkNormed.AsSpan(i * hidden, hidden));
                    }

                    ReadOnlySpan<byte> bytes = _weights.Resolve(_wq[l]);
                    _backend.GemvBatch(chunkNormed.AsSpan(0, m * hidden), m, new WeightRef(bytes, _wq[l].Shape, _wq[l].Format), chunkQ.AsSpan(0, m * qkvRegion));

                    bytes = _weights.Resolve(_wk[l]);
                    _backend.GemvBatch(chunkNormed.AsSpan(0, m * hidden), m, new WeightRef(bytes, _wk[l].Shape, _wk[l].Format), chunkK.AsSpan(0, m * kvRegion));

                    bytes = _weights.Resolve(_wv[l]);
                    _backend.GemvBatch(chunkNormed.AsSpan(0, m * hidden), m, new WeightRef(bytes, _wv[l].Shape, _wv[l].Format), chunkV.AsSpan(0, m * kvRegion));

                    for (int i = 0; i < m; i++)
                    {
                        AddBias(chunkQ.AsSpan(i * qkvRegion, qkvRegion), _biasQ[l]);
                        AddBias(chunkK.AsSpan(i * kvRegion, kvRegion), _biasK[l]);
                        AddBias(chunkV.AsSpan(i * kvRegion, kvRegion), _biasV[l]);

                        ApplyQkNorm(chunkQ.AsSpan(i * qkvRegion, qkvRegion), _attnQNorm[l]);
                        ApplyQkNorm(chunkK.AsSpan(i * kvRegion, kvRegion), _attnKNorm[l]);

                        ApplyRopeHeads(chunkQ.AsSpan(i * qkvRegion, qkvRegion), Plan.HeadCount, firstPos + i);
                        ApplyRopeHeads(chunkK.AsSpan(i * kvRegion, kvRegion), Plan.KvHeadCount, firstPos + i);

                        int position = c * EncodeChunkTokens + i; // sequence-relative index into the streams
                        kv.WriteK(basePosition + position, l, chunkK.AsSpan(i * kvRegion, kvRegion));
                        kv.WriteV(basePosition + position, l, chunkV.AsSpan(i * kvRegion, kvRegion));

                        chunkQ.AsSpan(i * qkvRegion, qkvRegion).CopyTo(qAll.AsSpan(position * qkvRegion, qkvRegion));
                    }
                }

                // Phase 2: bidirectional attention per chunk — every query scores against all n positions of this
                // sequence's finished KV. Chunk-local outputs land at each position's slot in attnOut.
                for (int c = 0; c < chunkCount; c++)
                {
                    int m = Math.Min(EncodeChunkTokens, n - c * EncodeChunkTokens);

                    LayerOps.BidirectionalAttention(
                        _backend,
                        qAll.AsSpan(c * EncodeChunkTokens * qkvRegion, m * qkvRegion),
                        queryCount: m,
                        queryPositionOffset: basePosition + c * EncodeChunkTokens,
                        fromPosition: 0,
                        toPosition: basePosition + n - 1,
                        Plan.HeadCount,
                        Plan.KvHeadCount,
                        Plan.HeadDim,
                        kv,
                        l,
                        attnScale,
                        Plan.AttentionSoftcap,
                        layerWindow,
                        attnOut.AsSpan(c * EncodeChunkTokens * qkvRegion, m * qkvRegion));
                }

                // Phase 3: output projection (+bias) and the ffn_inp residual add on each position's own stream.
                for (int c = 0; c < chunkCount; c++)
                {
                    int m = Math.Min(EncodeChunkTokens, n - c * EncodeChunkTokens);

                    ReadOnlySpan<byte> bytes = _weights.Resolve(_wo[l]);
                    _backend.GemvBatch(attnOut.AsSpan(c * EncodeChunkTokens * qkvRegion, m * qkvRegion), m, new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), chunkProj.AsSpan(0, m * hidden));

                    for (int i = 0; i < m; i++)
                    {
                        AddBias(chunkProj.AsSpan(i * hidden, hidden), _biasO[l]);

                        // Post-sublayer placement: this position's attention output joins its stream only after the post-attention norm.
                        ApplyPostSublayerNorm(chunkProj.AsSpan(i * hidden, hidden), _postAttnNorm[l]);

                        Span<float> posCur = cur.AsSpan((c * EncodeChunkTokens + i) * hidden, hidden);
                        ReadOnlySpan<float> proj = chunkProj.AsSpan(i * hidden, hidden);
                        for (int x = 0; x < hidden; x++)
                        {
                            posCur[x] += proj[x];
                        }

                        ApplyNorm(posCur, _ffnNormWeight[l], _ffnNormBias[l], normed.AsSpan((c * EncodeChunkTokens + i) * hidden, hidden));
                    }
                }

                // Phase 4: gate/up GEMVs and the gated activation per chunk.
                for (int c = 0; c < chunkCount; c++)
                {
                    int m = Math.Min(EncodeChunkTokens, n - c * EncodeChunkTokens);

                    ReadOnlySpan<byte> bytes = _weights.Resolve(_ffnGate[l]);
                    _backend.GemvBatch(normed.AsSpan(c * EncodeChunkTokens * hidden, m * hidden), m, new WeightRef(bytes, _ffnGate[l].Shape, _ffnGate[l].Format), chunkGateUp.AsSpan(0, m * inter));

                    bytes = _weights.Resolve(_ffnUp[l]);
                    _backend.GemvBatch(normed.AsSpan(c * EncodeChunkTokens * hidden, m * hidden), m, new WeightRef(bytes, _ffnUp[l].Shape, _ffnUp[l].Format), chunkUp.AsSpan(0, m * inter));

                    for (int i = 0; i < m; i++)
                    {
                        ApplyActivation(chunkGateUp.AsSpan(i * inter, inter), chunkUp.AsSpan(i * inter, inter));
                    }

                    // Phase 5: down GEMV and the ffn_out residual add, completing this layer for every position.
                    bytes = _weights.Resolve(_ffnDown[l]);
                    _backend.GemvBatch(chunkGateUp.AsSpan(0, m * inter), m, new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), chunkProj.AsSpan(0, m * hidden));

                    for (int i = 0; i < m; i++)
                    {
                        // Post-sublayer placement: this position's feed-forward output joins its stream only after the post-feed-forward norm.
                        ApplyPostSublayerNorm(chunkProj.AsSpan(i * hidden, hidden), _postFfnNorm[l]);

                        Span<float> posCur = cur.AsSpan((c * EncodeChunkTokens + i) * hidden, hidden);
                        ReadOnlySpan<float> proj = chunkProj.AsSpan(i * hidden, hidden);
                        for (int x = 0; x < hidden; x++)
                        {
                            posCur[x] += proj[x]; // ffn_out + ffn_inp: this position's input stream for the next layer
                        }
                    }
                }
            }

            float[] encoded = new float[n * hidden]; // caller-owned result, exact size — pooling helpers consume it row-wise
            for (int i = 0; i < n; i++)
            {
                ApplyNorm(cur.AsSpan(i * hidden, hidden), _outputNormWeight, _outputNormBias, encoded.AsSpan(i * hidden, hidden));
            }

            return encoded;
        }
        finally
        {
            poolBuffers.Return(cur, clearArray: false);
            poolBuffers.Return(normed, clearArray: false);
            poolBuffers.Return(qAll, clearArray: false);
            poolBuffers.Return(attnOut, clearArray: false);
            poolBuffers.Return(chunkNormed, clearArray: false);
            poolBuffers.Return(chunkQ, clearArray: false);
            poolBuffers.Return(chunkK, clearArray: false);
            poolBuffers.Return(chunkV, clearArray: false);
            poolBuffers.Return(chunkProj, clearArray: false);
            poolBuffers.Return(chunkGateUp, clearArray: false);
            poolBuffers.Return(chunkUp, clearArray: false);
        }
    }

    public void Dispose()
    {
        _weights.Dispose();
    }

    private void ForwardToken(int tokenId, int position, SequenceKv kv)
    {
        if ((uint)tokenId >= (uint)Plan.VocabSize)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenId), $"token id {tokenId} is outside the vocabulary of {Plan.VocabSize}.");
        }

        if (position >= MaxContextTokens)
        {
            throw new InvalidOperationException(
                $"sequence position {position} exceeds this model's max context of {MaxContextTokens} tokens.");
        }

        // Embedding lookup: one dequantized row of token_embd becomes the layer input stream.
        ReadOnlySpan<byte> embBytes = _weights.Resolve(_tokenEmbd);
        _backend.DequantRow(new WeightRef(embBytes, _tokenEmbd.Shape, _tokenEmbd.Format), tokenId, _cur.AsSpan(0, Plan.HiddenSize));

        ApplyInputEmbeddingScale(_cur.AsSpan(0, Plan.HiddenSize));

        kv.Append(); // reserves this position's slot in every layer; WriteK/WriteV address it by logical position

        int hidden = Plan.HiddenSize;
        float attnScale = _attnScale;

        for (int l = 0; l < Plan.LayerCount; l++)
        {
            int layerWindow = LayerAttentionWindow(l);

            // Pre-attention norm on the stream: normed = rmsnorm(cur) * attn_norm weight.
            ApplyNorm(_cur.AsSpan(0, hidden), _attnNormWeight[l], _attnNormBias[l], _normed);

            ReadOnlySpan<byte> bytes = _weights.Resolve(_wq[l]);
            _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(bytes, _wq[l].Shape, _wq[l].Format), _q);
            AddBias(_q, _biasQ[l]);

            bytes = _weights.Resolve(_wk[l]);
            _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(bytes, _wk[l].Shape, _wk[l].Format), _k);
            AddBias(_k, _biasK[l]);

            bytes = _weights.Resolve(_wv[l]);
            _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(bytes, _wv[l].Shape, _wv[l].Format), _v);
            AddBias(_v, _biasV[l]);

            // Resolved QK-norm mode: each side's projected stream is normalized after the projection and bias and
            // before RoPE, either per head in its own head-dimension slice or across all its heads under one mean.
            ApplyQkNorm(_q, _attnQNorm[l]);
            ApplyQkNorm(_k, _attnKNorm[l]);

            ApplyRopeHeads(_q, Plan.HeadCount, position);
            ApplyRopeHeads(_k, Plan.KvHeadCount, position);

            kv.WriteK(position, l, _k);
            kv.WriteV(position, l, _v);

            LayerOps.CausalAttention(_backend, _q, Plan.HeadCount, Plan.KvHeadCount, Plan.HeadDim, kv, l, position, attnScale, Plan.AttentionSoftcap, layerWindow, _attnOut);

            // Output projection (+bias) into scratch; the residual added before the FFN norm is the pre-attention
            // stream value (ffn_inp = attn_out + inpSA), matching the reference graph. cur held that stream through
            // attention untouched, so adding proj in place turns it into ffn_inp. Under a post-sublayer-output
            // placement the attention output — including any output bias — is normalized with its own vector first.
            bytes = _weights.Resolve(_wo[l]);
            _backend.Gemv(_attnOut.AsSpan(0, Plan.HeadCount * Plan.HeadDim), new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), _proj);
            AddBias(_proj, _biasO[l]);

            ApplyPostSublayerNorm(_proj, _postAttnNorm[l]);

            for (int i = 0; i < hidden; i++)
            {
                _cur[i] += _proj[i];
            }

            ApplyNorm(_cur.AsSpan(0, hidden), _ffnNormWeight[l], _ffnNormBias[l], _normed);

            if (_sparseFfn is { } routing)
            {
                ApplySparseFeedForward(routing, l);
            }
            else
            {
                bytes = _weights.Resolve(_ffnGate[l]);
                _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(bytes, _ffnGate[l].Shape, _ffnGate[l].Format), _gateUp);

                bytes = _weights.Resolve(_ffnUp[l]);
                _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(bytes, _ffnUp[l].Shape, _ffnUp[l].Format), _up);

                // act = activation(gate) * up with the plan's resolved activation, applied in place on the gate buffer.
                ApplyActivation(_gateUp.AsSpan(0, Plan.IntermediateSize), _up.AsSpan(0, Plan.IntermediateSize));

                bytes = _weights.Resolve(_ffnDown[l]);
                _backend.Gemv(_gateUp.AsSpan(0, Plan.IntermediateSize), new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), _proj);
            }

            ApplyPostSublayerNorm(_proj, _postFfnNorm[l]);

            for (int i = 0; i < hidden; i++)
            {
                _cur[i] += _proj[i]; // ffn_out + ffn_inp: the input stream for the next layer
            }
        }

        ApplyNorm(_cur.AsSpan(0, hidden), _outputNormWeight, _outputNormBias, _normed);

        LogicalTensor lmHead = _output ?? _tokenEmbd; // tied weights reuse the token-embedding rows as logits
        ReadOnlySpan<byte> headBytes = _weights.Resolve(lmHead);
        _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(headBytes, lmHead.Shape, lmHead.Format), _logits);

        if (_outputBias is not null)
        {
            AddBias(_logits, _outputBias);
        }

        ApplyFinalLogitSoftcap(_logits);
    }

    /// <summary>Rotates every head region of data in place at the given position using the precomputed tables, with the pairing the plan's resolved RoPE strategy selects.</summary>
    private void ApplyRopeHeads(Span<float> data, int headCount, int position)
    {
        ReadOnlySpan<float> cosAtPosition = _cos.AsSpan(position * _ropeHalf, _ropeHalf);
        ReadOnlySpan<float> sinAtPosition = _sin.AsSpan(position * _ropeHalf, _ropeHalf);

        for (int h = 0; h < headCount; h++)
        {
            // Pairs are relative to each head's own region.
            Span<float> region = data.Slice(h * Plan.HeadDim, Plan.HeadDim);
            switch (Plan.Rope)
            {
                case RopeStrategy.StandardFullHead:
                    LayerOps.ApplyRopeToHead(region, _ropeHalf, cosAtPosition, sinAtPosition);
                    break;
                case RopeStrategy.RotateHalfFullHead:
                    LayerOps.ApplyRopeToSplitHalfHead(region, _ropeHalf, cosAtPosition, sinAtPosition);
                    break;
                default:
                    throw new ModelException($"rope strategy '{Plan.Rope}' is not implemented by the forward engine");
            }
        }
    }

    private static void AddBias(Span<float> values, float[]? bias)
    {
        if (bias is null)
        {
            return;
        }

        for (int i = 0; i < values.Length; i++)
        {
            values[i] += bias[i];
        }
    }

    // Norm and activation run as a direct call chosen from the plan's strategy values: no interface dispatch and
    // no allocation anywhere in the forward, so every path — single token, batched decode, and encode — applies
    // the resolved strategy identically. The bias operand is the layer's declared norm-bias vector when one was
    // loaded (an empty span otherwise) and is consumed only by the LayerNorm strategy, which takes its epsilon from
    // the plan's block-norm epsilon while the RMS strategy keeps the plan's RMS epsilon.
    private void ApplyNorm(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias, Span<float> dst)
    {
        switch (Plan.Norm)
        {
            case NormStrategy.RmsNorm:
                LayerOps.RmsNormWeighted(src, weight, Plan.EpsRmsNorm, dst);
                break;
            case NormStrategy.LayerNorm:
                LayerOps.LayerNormWeighted(src, weight, bias, Plan.EpsLayerNorm, dst);
                break;
            default:
                throw new ModelException($"norm strategy '{Plan.Norm}' is not implemented by the forward engine");
        }
    }

    // The resolved QK-norm form runs as a direct call chosen from the plan's mode, exactly as the norm and activation
    // strategies do: no interface dispatch and no allocation anywhere in the forward, so every path — single token,
    // batched decode, and encode — applies the resolved form identically between each head's projection (plus any bias)
    // and its rotation. The per-head weighted variant consumes the layer's head-dimension vector over each head region;
    // the stream-weighted variant consumes one whole-projection vector and normalizes every head of q (or of k) under a
    // single mean across that side's full width, which is what the reference does before splitting its projection into
    // heads. The unweighted variant normalizes with the plan epsilon alone, which is numerically identical to a weight
    // of ones and reads nothing from the file; off leaves every head region untouched. Each form normalizes in place: it finishes reading a value before writing back at that index.
    private void ApplyQkNorm(Span<float> heads, ReadOnlySpan<float> weight)
    {
        switch (Plan.QkNorm)
        {
            case QkNormMode.PerHeadWeighted:
                LayerOps.RmsNormWeightedHeads(heads, weight, Plan.HeadDim, Plan.EpsRmsNorm, heads);
                break;

            case QkNormMode.StreamWeighted:
                // One mean over the whole q (or k) projection and one element-by-element vector covering that same
                // width: every head is scaled by its own weights but normalized jointly with the other heads.
                LayerOps.RmsNormWeighted(heads, weight, Plan.EpsRmsNorm, heads);
                break;

            case QkNormMode.PerHeadUnweighted:
                LayerOps.RmsNormUnweightedHeads(heads, Plan.HeadDim, Plan.EpsRmsNorm, heads);
                break;

            case QkNormMode.Off:
                return;

            default:
                throw new ModelException($"QK-norm mode '{Plan.QkNorm}' is not implemented by the forward engine");
        }
    }

    // The second norm a sublayer output may receive runs as a direct branch on the plan's resolved placement, the way the
    // softcaps do. Under PreOnly nothing here reads anything and each output joins the residual stream unchanged; under
    // PostSublayerOutputs that sublayer's own norm vector normalizes its output in place — the attention output after its
    // projection and any output bias, or the feed-forward output after the down GEMV — before its residual add, through
    // the family's resolved norm kind with no bias operand, since no post-sublayer bias tensor exists in the canonical map.
    // In place is safe for both norm kinds: each finishes reading a value before writing back at that index.
    // The input-embedding gain runs as a direct branch on the plan's resolved value, the same way the norm and
    // activation strategies do: a resolved gain of zero leaves every embedding row untouched. A non-zero gain is
    // the sqrt(hidden size) the gemma-family reference graphs multiply each embedding row by before the first
    // layer, applied in place to the freshly dequantized rows on every forward path — single token, batched
    // decode, and encode — so each path seeds its residual stream identically.
    private void ApplyInputEmbeddingScale(Span<float> embeddings)
    {
        float scale = Plan.InputEmbeddingScale;
        if (scale == 0f)
        {
            return;
        }

        for (int i = 0; i < embeddings.Length; i++)
        {
            embeddings[i] *= scale;
        }
    }

    private void ApplyPostSublayerNorm(Span<float> output, ReadOnlySpan<float> weight)
    {
        switch (Plan.NormPlacement)
        {
            case NormPlacement.PreOnly:
                return;

            case NormPlacement.PostSublayerOutputs:
                ApplyNorm(output, weight, ReadOnlySpan<float>.Empty, output);
                break;

            default:
                throw new ModelException($"norm placement '{Plan.NormPlacement}' is not implemented by the forward engine");
        }
    }

    // The score window one layer narrows to runs as a direct branch on the plan's resolved interleave, the way the
    // norm, activation, and softcap switches do: the declared window masks every layer under AllLayers, and only the
    // even-indexed layers under AlternatingLocalFirst, where the odd-indexed layers attend the whole range. A
    // declared window of zero is unbounded whatever the pattern. The value is resolved once per layer — never per
    // token or per head — and bounds scores only: every position's K/V is still written and stays readable.
    private int LayerAttentionWindow(int layer)
    {
        switch (Plan.WindowInterleave)
        {
            case WindowInterleave.AllLayers:
                return Plan.AttentionWindow;

            case WindowInterleave.AlternatingLocalFirst:
                return layer % 2 == 0 ? Plan.AttentionWindow : 0;

            default:
                throw new ModelException($"window interleave '{Plan.WindowInterleave}' is not implemented by the forward engine");
        }
    }

    private void ApplyActivation(Span<float> gate, ReadOnlySpan<float> up)
    {
        switch (Plan.Activation)
        {
            case ActivationStrategy.SwiGLU:
                for (int i = 0; i < gate.Length; i++)
                {
                    gate[i] = LayerOps.Silu(gate[i]) * up[i];
                }

                break;
            case ActivationStrategy.GeGlu:
                for (int i = 0; i < gate.Length; i++)
                {
                    gate[i] = LayerOps.GeluTanh(gate[i]) * up[i];
                }

                break;
            case ActivationStrategy.GeGluExact:
                for (int i = 0; i < gate.Length; i++)
                {
                    gate[i] = LayerOps.Gelu(gate[i]) * up[i];
                }

                break;
            default:
                throw new ModelException($"activation strategy '{Plan.Activation}' is not implemented by the forward engine");
        }
    }

    // The final-logit softcap runs as a direct branch on the plan's resolved value, the same way the norm and
    // activation strategies do: a resolved cap of zero is disabled and leaves the vector untouched. It is applied
    // to every produced logit vector — untied output head or tied token-embedding rows, single-token or batched —
    // after the full vector including any output bias is computed and before any softmax, sampling, or copy-out.
    private void ApplyFinalLogitSoftcap(Span<float> logits)
    {
        float cap = Plan.FinalLogitSoftcap;
        if (cap <= 0f)
        {
            return;
        }

        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] = LayerOps.Softcap(logits[i], cap);
        }
    }

    // Loads one per-layer matrix role. When a fused qkv template is given for a q/k/v role and the layer
    // carries the fused tensor alongside the resolved split tensor, the fused layout is not consumed — the
    // split map already fills the role — and the name is recorded for the one structured tensors_ignored line.
    private LogicalTensor[] LoadMatrices(int layers, TensorRole role, string? fusedTemplate = null)
    {
        LogicalTensor[] tensors = new LogicalTensor[layers];
        for (int l = 0; l < layers; l++)
        {
            tensors[l] = RequireTensor(role, l);

            if (fusedTemplate is not null)
            {
                string fused = FusedName(fusedTemplate, l);
                if (_fileTensors.Find(fused) is not null)
                {
                    _ignoredFusedTensors.Add(fused);
                }
            }
        }

        return tensors;
    }

    private float[][] LoadLayerVectors(int layers, TensorRole role, int expectedLength)
    {
        float[][] vectors = new float[layers][];
        for (int l = 0; l < layers; l++)
        {
            LogicalTensor tensor = RequireTensor(role, l);
            vectors[l] = LoadVector(tensor, expectedLength);
        }

        return vectors;
    }

    // Norm and bias tensors are one-dimensional in GGUF. Addressing them as a single-row matrix lets the backend's
    // row dequantization handle any stored format (F32 for most conversions, F16 elsewhere) with no extra API.
    private float[] LoadVector(LogicalTensor tensor, int expectedLength)
    {
        if (tensor.Shape.Dims.Length != 1 || tensor.Shape.Dims[0] != expectedLength)
        {
            throw new ModelException(
                $"tensor '{tensor.Name}' must have shape [{expectedLength}], got [{string.Join(", ", tensor.Shape.Dims)}]");
        }

        TensorShape row = new([1, expectedLength]);
        ReadOnlySpan<byte> bytes = _weights.Resolve(tensor);
        float[] values = new float[expectedLength];
        _backend.DequantRow(new WeightRef(bytes, row, tensor.Format), 0, values);
        return values;
    }

    // Loads one per-layer stacked expert role and validates it against the descriptor's counts: the logical shape
    // carries the declared expert count as its outermost dimension followed by this matrix's rows and columns, and
    // the payload holds exactly that many packed rows so a single expert's slice stays inside the tensor. A stack
    // whose per-expert block is too large for one addressable slice fails here rather than mis-slicing at forward time.
    private LogicalTensor[] LoadExpertStacks(int layers, TensorRole role, int experts, long rows, long cols)
    {
        LogicalTensor[] tensors = new LogicalTensor[layers];

        for (int l = 0; l < layers; l++)
        {
            LogicalTensor tensor = RequireTensor(role, l);
            int[] dims = tensor.Shape.Dims;
            if (dims.Length != 3 || dims[0] != experts || dims[1] != rows || dims[^1] != cols)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' must have shape [{experts}, {rows}, {cols}], got [{string.Join(", ", dims)}]");
            }

            long stride = tensor.Shape.RowStrideBytes(tensor.Format);
            long payload = (long)dims[0] * rows * stride;
            if (payload != tensor.SizeBytes)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' declares {tensor.SizeBytes} payload bytes but its shape and format require {payload}");
            }

            long expertBlock = rows * stride;
            if (expertBlock > int.MaxValue)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' has an expert block of {expertBlock} bytes, beyond a single addressable slice");
            }

            tensors[l] = tensor;
        }

        return tensors;
    }

    // A descriptor that declares a shared expert names all three matrices together at plan resolution. This reads the
    // declared role and fails on any half-set left unset, so a malformed descriptor never reaches execution; an
    // undeclared shared expert loads nothing here at all.
    private static TensorRole RequiredSparseRole(TensorRole? role, string side)
    {
        if (role is not { } resolved)
        {
            throw new ModelException($"the sparse-FFN descriptor declares a shared expert without naming its {side} matrix");
        }

        return resolved;
    }

    // Loads every layer's shared-expert selector vector, in the presence discipline the optional tensors use: a file
    // shipping none of them loads nothing and the shared block keeps its ungated accumulation, a file shipping one for
    // some layers only fails naming the first missing layer instead of gating half the block, and a complete table is
    // dequantized once here so forward time reads plain floats like every other vector role.
    private float[][] LoadSharedExpertSelectors(int layers, TensorDescriptor descriptor)
    {
        LogicalTensor?[] present = new LogicalTensor?[layers];
        bool anyPresent = false;
        int firstMissingLayer = -1;

        for (int l = 0; l < layers; l++)
        {
            present[l] = descriptor.Resolve(_fileTensors, l);
            if (present[l] is null)
            {
                if (firstMissingLayer < 0)
                {
                    firstMissingLayer = l;
                }
            }
            else
            {
                anyPresent = true;
            }
        }

        if (!anyPresent)
        {
            return [];
        }

        if (firstMissingLayer >= 0)
        {
            throw new ModelException(
                $"tensor '{descriptor.NameFor(firstMissingLayer)}' is missing: architecture '{Plan.Architecture}' " +
                "ships the shared-expert selector vector for some layers only");
        }

        float[][] vectors = new float[layers][];
        for (int l = 0; l < layers; l++)
        {
            vectors[l] = LoadVector(present[l]!, Plan.HiddenSize);
        }

        return vectors;
    }

    // A slice of one expert's resident stack as the weight operand a GEMV over that expert consumes: the contiguous
    // row range at this role's per-expert byte geometry, carrying the two-dimensional shape resolved for it at Open.
    private static WeightRef ExpertWeightRef(ReadOnlySpan<byte> payload, long expertBytes, int expertIndex, TensorShape shape, QuantFormat format) =>
        new(payload.Slice((int)(expertIndex * expertBytes), (int)expertBytes), shape, format);

    // Router scoring and top-k selection for one normalized activation vector, shared by both forward paths so a
    // batched step reproduces single-sequence routing exactly. The router emits one score per declared expert; every
    // score is softmaxed together because the reference routes over all experts, then RoutedExpertsPerToken picks run
    // as ascending-index scans using strict greater-than comparison — equal scores resolve to the lower expert index
    // deterministically, with no sort or dictionary stability assumption. A picked slot is poisoned so later passes
    // cannot re-pick it; selected slots are insertion-sorted into ascending expert-index execution order alongside
    // their weights. Normalization divides by the selected probability sum when declared, and a nonzero
    // ExpertWeightsScale multiplies every weight after that division. Scratch holds one routing at a time: batched
    // callers copy each selection before selecting the next sequence.
    private void RouteSparseExperts(SparseFfnDescriptor sparse, int layer, ReadOnlySpan<float> input)
    {
        LogicalTensor router = _moeRouter[layer];
        ReadOnlySpan<byte> bytes = _weights.Resolve(router);
        Span<float> scores = _moeLogits.AsSpan(0, sparse.ExpertCount);

        // The router emits one score per declared expert before any selection or softmax runs.
        _backend.Gemv(input, new WeightRef(bytes, router.Shape, router.Format), scores);

        float maxScore = float.NegativeInfinity;
        for (int e = 0; e < scores.Length; e++)
        {
            if (scores[e] > maxScore)
            {
                maxScore = scores[e];
            }
        }

        // All-expert softmax probabilities in place, summed and divided in ascending expert order: the reference
        // routes over every declared expert before selecting any of them. The max-subtraction guard keeps every
        // exponent in range.
        float allSum = 0f;
        for (int e = 0; e < scores.Length; e++)
        {
            scores[e] = MathF.Exp(scores[e] - maxScore);
            allSum += scores[e];
        }

        for (int e = 0; e < scores.Length; e++)
        {
            scores[e] /= allSum;
        }

        int routed = sparse.RoutedExpertsPerToken;
        float selectedSum = 0f;
        for (int pick = 0; pick < routed; pick++)
        {
            int best = -1;
            float bestScore = float.NegativeInfinity;
            for (int e = 0; e < scores.Length; e++)
            {
                // Strict '>' over ascending indexes: a tie keeps the earlier expert, so equal scores resolve to the
                // lower index regardless of call order or buffer reuse.
                if (scores[e] > bestScore)
                {
                    best = e;
                    bestScore = scores[e];
                }
            }

            _moeIds[pick] = best;
            _moeWeights[pick] = bestScore;
            selectedSum += bestScore;
            scores[best] = float.NegativeInfinity; // consumed: later passes cannot re-pick this expert
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
    }

    // Adds weight times source into target element-wise, in ascending order: the accumulation both forward paths use
    // for every selected expert's down output, so a batched step replays the single-token arithmetic exactly.
    private static void AccumulateScaled(Span<float> target, ReadOnlySpan<float> source, float weight)
    {
        for (int i = 0; i < source.Length; i++)
        {
            target[i] += weight * source[i];
        }
    }

    // The weight a declared shared expert contributes with. A block whose profile names no selector role, or whose
    // file ships none of those vectors, gets exactly 1f: the ungated accumulation this path has always used. When the
    // vectors are present the score is taken from them against this token's normalized stream and applied as
    // silu(score) / score to the shared output — the pinned qwen2moe graph builds it that way, as
    // `ggml_div(ggml_silu(cur_gate_inp), cur_gate_inp)` over a GEMV of ffn_gate_inp_shexp, which is sigmoid's value
    // read through its own silu rather than a separate node. The dot runs in ascending column order and both forward
    // paths reach it through this method with the same per-token stream, so batched steps stay bit-identical to
    // single-sequence ones. A score of exactly zero divides 0 by 0 as the reference graph does.
    private float SharedExpertWeight(int layer, ReadOnlySpan<float> normed)
    {
        if (_sharedInputGate.Length == 0)
        {
            return 1f; // no selector vector in this file: the shared block adds ungated, exactly as before
        }

        float[] selector = _sharedInputGate[layer];
        float score = 0f;
        for (int c = 0; c < selector.Length; c++)
        {
            score += selector[c] * normed[c];
        }

        return LayerOps.Silu(score) / score;
    }

    // Sparse feed-forward for the current token: routing selects, each selected expert's gate/up/down chain runs over
    // the normalized stream in ascending expert-index order, and every output accumulates into _proj scaled by its
    // routing weight — a declared shared expert contributing once under SharedExpertWeight, which is 1f unless the
    // file ships that block's selector vectors. The caller's post-feed-forward norm
    // and residual add then run exactly as they do on the dense path.
    private void ApplySparseFeedForward(SparseFfnDescriptor sparse, int layer)
    {
        RouteSparseExperts(sparse, layer, _normed.AsSpan(0, Plan.HiddenSize));

        int hidden = Plan.HiddenSize;
        Span<float> proj = _proj;
        proj.Clear(); // the routed and shared contributions accumulate from zero below

        for (int slot = 0; slot < sparse.RoutedExpertsPerToken; slot++)
        {
            ExecuteRoutedExpert(layer, _moeIds[slot], _moeSubsetIn.AsSpan(0, hidden));
            AccumulateScaled(proj, _moeSubsetIn.AsSpan(0, hidden), _moeWeights[slot]);
        }

        if (sparse.HasSharedExpert)
        {
            RunSharedExpertToken(layer, _moeSubsetIn.AsSpan(0, hidden));
            AccumulateScaled(proj, _moeSubsetIn.AsSpan(0, hidden), SharedExpertWeight(layer, _normed.AsSpan(0, hidden)));
        }
    }

    // One routed expert's gate/up/down chain over a single activation vector. The gate and up intermediates reuse the
    // dense token buffers; each GEMV reads this expert's contiguous row range of its resident stacked role at the
    // per-expert byte geometry resolved at Open, so no copy of any expert tensor exists anywhere.
    private void ExecuteRoutedExpert(int layer, int expertIndex, Span<float> output)
    {
        LogicalTensor gate = _moeExpertGate[layer];
        ReadOnlySpan<byte> payload = _weights.Resolve(gate);
        _backend.Gemv(_normed.AsSpan(0, Plan.HiddenSize), ExpertWeightRef(payload, _moeExpertGateBytes[layer], expertIndex, _moeExpertGateShape, gate.Format), _gateUp);

        LogicalTensor up = _moeExpertUp[layer];
        payload = _weights.Resolve(up);
        _backend.Gemv(_normed.AsSpan(0, Plan.HiddenSize), ExpertWeightRef(payload, _moeExpertUpBytes[layer], expertIndex, _moeExpertUpShape, up.Format), _up);

        ApplyActivation(_gateUp.AsSpan(0, Plan.IntermediateSize), _up.AsSpan(0, Plan.IntermediateSize));

        LogicalTensor down = _moeExpertDown[layer];
        payload = _weights.Resolve(down);
        _backend.Gemv(_gateUp.AsSpan(0, Plan.IntermediateSize), ExpertWeightRef(payload, _moeExpertDownBytes[layer], expertIndex, _moeExpertDownShape, down.Format), output);
    }

    // The shared expert's dense-shaped triplet over a single activation vector: the same gate/up/down GEMVs the dense
    // block runs, reading its own per-layer tensors and writing this token's contribution into the accumulation slot.
    private void RunSharedExpertToken(int layer, Span<float> output)
    {
        LogicalTensor gate = _sharedGate[layer];
        ReadOnlySpan<byte> bytes = _weights.Resolve(gate);
        _backend.Gemv(_normed.AsSpan(0, Plan.HiddenSize), new WeightRef(bytes, gate.Shape, gate.Format), _gateUp);

        LogicalTensor up = _sharedUp[layer];
        bytes = _weights.Resolve(up);
        _backend.Gemv(_normed.AsSpan(0, Plan.HiddenSize), new WeightRef(bytes, up.Shape, up.Format), _up);

        ApplyActivation(_gateUp.AsSpan(0, Plan.IntermediateSize), _up.AsSpan(0, Plan.IntermediateSize));

        LogicalTensor down = _sharedDown[layer];
        bytes = _weights.Resolve(down);
        _backend.Gemv(_gateUp.AsSpan(0, Plan.IntermediateSize), new WeightRef(bytes, down.Shape, down.Format), output);
    }

    // Sparse feed-forward for one batched step: every sequence routes first (one selection at a time in scratch,
    // copied into the per-sequence arrays before the next selection runs). Execution then walks selection slots in
    // ascending order and experts by index within each slot, packing that group's member rows of _batchNormed for one
    // GemvBatch pass through the expert's resident stack; results accumulate straight into each sequence's projection
    // stream under its routing weight. A declared shared expert runs last over all sequences, every sequence scaled by
    // its own SharedExpertWeight. The packed subset buffer carries activations and down results in place, so no copy of any expert tensor or whole-batch
    // output exists — and because every sequence accumulates its own contributions in the same slot order a single
    // token does, each batched result is bit-identical to stepping that sequence alone. The caller's post-feed-forward
    // norm and residual add then run exactly as they do on the dense path.
    private void ApplySparseFeedForwardBatch(SparseFfnDescriptor sparse, int layer, int n)
    {
        int hidden = Plan.HiddenSize;
        int routed = sparse.RoutedExpertsPerToken;

        for (int s = 0; s < n; s++)
        {
            RouteSparseExperts(sparse, layer, _batchNormed.AsSpan(s * hidden, hidden));

            for (int pick = 0; pick < routed; pick++)
            {
                _batchMoeIds[s * routed + pick] = _moeIds[pick];
                _batchMoeWeights[s * routed + pick] = _moeWeights[pick];
            }
        }

        Span<float> batchProj = _batchProj.AsSpan(0, n * hidden);
        batchProj.Clear(); // the routed and shared contributions accumulate from zero below

        for (int slot = 0; slot < routed; slot++)
        {
            for (int expertIndex = 0; expertIndex < sparse.ExpertCount; expertIndex++)
            {
                int count = 0;
                for (int s = 0; s < n; s++)
                {
                    if (_batchMoeIds[s * routed + slot] == expertIndex)
                    {
                        _expertMembers[count++] = s;
                    }
                }

                ExecuteRoutedExpertBatch(layer, expertIndex, count, slot, routed);
            }
        }

        if (sparse.HasSharedExpert)
        {
            RunSharedExpertBatch(layer, n, hidden);
        }
    }

    // One routed expert's gate/up/down chain over the packed member rows of one selection slot: a single batched pass
    // through its resident stack, bit-identical per row to the token path, with each member's down result accumulated
    // into that sequence's projection stream under its routing weight as soon as the group completes. _moeSubsetIn
    // carries this group's activations and then its contiguous down results in place; gate/up intermediates reuse the
    // dense batch buffers. A slot no sequence selected runs nothing at all.
    private void ExecuteRoutedExpertBatch(int layer, int expertIndex, int count, int slot, int routed)
    {
        if (count == 0)
        {
            return;
        }

        int hidden = Plan.HiddenSize;
        int inter = Plan.IntermediateSize;
        Span<float> subsetIn = _moeSubsetIn.AsSpan(0, count * hidden);

        for (int m = 0; m < count; m++)
        {
            _batchNormed.AsSpan(_expertMembers[m] * hidden, hidden).CopyTo(subsetIn.Slice(m * hidden, hidden));
        }

        LogicalTensor gate = _moeExpertGate[layer];
        ReadOnlySpan<byte> payload = _weights.Resolve(gate);
        _backend.GemvBatch(subsetIn, count, ExpertWeightRef(payload, _moeExpertGateBytes[layer], expertIndex, _moeExpertGateShape, gate.Format), _batchGateUp.AsSpan(0, count * inter));

        LogicalTensor up = _moeExpertUp[layer];
        payload = _weights.Resolve(up);
        _backend.GemvBatch(subsetIn, count, ExpertWeightRef(payload, _moeExpertUpBytes[layer], expertIndex, _moeExpertUpShape, up.Format), _batchUp.AsSpan(0, count * inter));

        for (int m = 0; m < count; m++)
        {
            ApplyActivation(_batchGateUp.AsSpan(m * inter, inter), _batchUp.AsSpan(m * inter, inter));
        }

        LogicalTensor down = _moeExpertDown[layer];
        payload = _weights.Resolve(down);
        _backend.GemvBatch(_batchGateUp.AsSpan(0, count * inter), count, ExpertWeightRef(payload, _moeExpertDownBytes[layer], expertIndex, _moeExpertDownShape, down.Format), subsetIn);

        for (int m = 0; m < count; m++)
        {
            int s = _expertMembers[m];
            AccumulateScaled(_batchProj.AsSpan(s * hidden, hidden), subsetIn.Slice(m * hidden, hidden), _batchMoeWeights[s * routed + slot]);
        }
    }

    // The shared expert's dense-shaped triplet over every sequence in one batched pass: the same gate/up/down
    // GemvBatch traversal the dense block runs, accumulating each sequence's output into its projection stream under
    // that sequence's SharedExpertWeight — 1f for a file shipping no selector vectors. An undeclared shared expert
    // never reaches this path and reads none of these roles at all.
    private void RunSharedExpertBatch(int layer, int n, int hidden)
    {
        LogicalTensor gate = _sharedGate[layer];
        ReadOnlySpan<byte> bytes = _weights.Resolve(gate);

        for (int s = 0; s < n; s++)
        {
            _batchNormed.AsSpan(s * hidden, hidden).CopyTo(_moeSubsetIn.AsSpan(s * hidden, hidden));
        }

        Span<float> subsetIn = _moeSubsetIn.AsSpan(0, n * hidden);
        int inter = Plan.IntermediateSize;

        _backend.GemvBatch(subsetIn, n, new WeightRef(bytes, gate.Shape, gate.Format), _batchGateUp.AsSpan(0, n * inter));

        LogicalTensor up = _sharedUp[layer];
        bytes = _weights.Resolve(up);
        _backend.GemvBatch(subsetIn, n, new WeightRef(bytes, up.Shape, up.Format), _batchUp.AsSpan(0, n * inter));

        for (int s = 0; s < n; s++)
        {
            ApplyActivation(_batchGateUp.AsSpan(s * inter, inter), _batchUp.AsSpan(s * inter, inter));
        }

        LogicalTensor down = _sharedDown[layer];
        bytes = _weights.Resolve(down);
        _backend.GemvBatch(_batchGateUp.AsSpan(0, n * inter), n, new WeightRef(bytes, down.Shape, down.Format), subsetIn);

        for (int s = 0; s < n; s++)
        {
            AccumulateScaled(_batchProj.AsSpan(s * hidden, hidden), _moeSubsetIn.AsSpan(s * hidden, hidden), SharedExpertWeight(layer, _batchNormed.AsSpan(s * hidden, hidden)));
        }
    }


    private static void RequireSingle(LogicalTensor tensor, int rows, long cols)
    {
        RequireShape([tensor], rows, cols);
    }

    private static void RequireShape(LogicalTensor[] tensors, long rows, long cols) => RequireShape(tensors.AsSpan(), rows, cols);

    private static void RequireShape(ReadOnlySpan<LogicalTensor> tensors, long rows, long cols)
    {
        foreach (LogicalTensor tensor in tensors)
        {
            int[] dims = tensor.Shape.Dims;
            if (dims.Length != 2 || dims[0] != rows || dims[^1] != cols)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' must have shape [{rows}, {cols}], got [{string.Join(", ", dims)}]");
            }

            long stride = tensor.Shape.RowStrideBytes(tensor.Format);
            if (dims[0] * stride != tensor.SizeBytes)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' declares {tensor.SizeBytes} payload bytes but its shape and format require {dims[0] * stride}");
            }
        }
    }

    private LogicalTensor RequireTensor(TensorRole role, int layer = -1)
    {
        TensorDescriptor descriptor = Plan.Tensors[role];
        LogicalTensor? hit = descriptor.Resolve(_fileTensors, layer);
        if (hit is not null)
        {
            return hit;
        }

        // A required split role tensor absent while its fused qkv sibling is present: the file ships the fused
        // layout, which this engine does not de-interleave. The failure names both sides instead of only the
        // missing name.
        if (FusedQkvName(role) is string fusedTemplate)
        {
            string fused = FusedName(fusedTemplate, layer);
            if (_fileTensors.Find(fused) is not null)
            {
                throw new ModelException(
                    $"tensor '{descriptor.NameFor(layer)}' is missing and '{fused}' is present: the fused qkv " +
                    $"layout is not de-interleaved for architecture '{Plan.Architecture}'");
            }
        }

        throw new ModelException(
            $"tensor '{descriptor.NameFor(layer)}' is missing: required by architecture '{Plan.Architecture}'");
    }

    // The fused qkv name template behind a q/k/v role, or null for any other role. The tensor map resolves the
    // split names; the fused layout is recognized only to report it (alongside the split tensors) or to reject
    // a file that carries it as the only source of a role.
    private static string? FusedQkvName(TensorRole role) =>
        role is TensorRole.AttentionQuery or TensorRole.AttentionKey or TensorRole.AttentionValue
            ? "blk.{layer}.attn_qkv.weight"
            : null;

    private static string FusedName(string template, int layer) =>
        template.Replace("{layer}", layer.ToString(CultureInfo.InvariantCulture));

    // Quantization companion entries — a <tensor>.scale block or a <tensor>.input_scale scalar — of a tensor
    // the plan's map can name are known-optional: every format the dequantizers implement today carries its
    // own scales inside the block, so a companion carries no meaning for this engine and never triggers a
    // table rejection. A companion that is not a plain float scale would have to be decoded to be interpreted,
    // which is a load failure naming it rather than a guess. A companion of a tensor the map does not name is
    // simply not part of the forward.
    private void ValidateCompanionTensors()
    {
        foreach (LogicalTensor tensor in _fileTensors.Tensors)
        {
            if (!TryCompanionBase(tensor.Name, out string baseName) || !IsNameableTensorName(baseName))
            {
                continue;
            }

            if (tensor.Format is not (QuantFormat.F32 or QuantFormat.F16))
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' is a quantization companion of '{baseName}' in format {tensor.Format}: " +
                    $"companion-carried scales are not decoded for architecture '{Plan.Architecture}'");
            }
        }
    }

    // Splits a companion tensor name into the name of the tensor it accompanies: the recognized companion
    // suffixes are .scale (a per-block scale array) and .input_scale (a tensor-wide scalar).
    private static bool TryCompanionBase(string name, out string baseName)
    {
        if (name.EndsWith(".input_scale", StringComparison.Ordinal))
        {
            baseName = name[..^".input_scale".Length];
            return true;
        }

        if (name.EndsWith(".scale", StringComparison.Ordinal))
        {
            baseName = name[..^".scale".Length];
            return true;
        }

        baseName = string.Empty;
        return false;
    }

    // Whether the plan's tensor map can name this tensor: a primary or alias name of any descriptor, for a
    // global role or for any layer the plan covers. Built once and cached.
    private bool IsNameableTensorName(string name)
    {
        _nameableTensors ??= BuildNameableTensorNames();
        return _nameableTensors.Contains(name);
    }

    private HashSet<string> BuildNameableTensorNames()
    {
        HashSet<string> names = new(StringComparer.Ordinal);
        foreach (TensorDescriptor descriptor in Plan.Tensors.Descriptors)
        {
            if (descriptor.Scope == TensorScope.PerLayer)
            {
                for (int l = 0; l < Plan.LayerCount; l++)
                {
                    names.Add(descriptor.NameFor(l));
                    foreach (string alias in descriptor.Aliases)
                    {
                        names.Add(alias.Replace("{layer}", l.ToString(CultureInfo.InvariantCulture)));
                    }
                }
            }
            else
            {
                names.Add(descriptor.Name);
                foreach (string alias in descriptor.Aliases)
                {
                    names.Add(alias);
                }
            }
        }

        return names;
    }

    private float[]?[] LoadOptionalBias(int layers, TensorRole role, long expectedLength)
    {
        TensorDescriptor descriptor = Plan.Tensors[role];
        float[]?[] biases = new float[layers][]; // null entries for absent bias tensors
        for (int l = 0; l < layers; l++)
        {
            LogicalTensor? tensor = descriptor.Resolve(_fileTensors, l);
            if (tensor is not null)
            {
                biases[l] = LoadVector(tensor, checked((int)expectedLength));
            }
        }

        return biases;
    }

    // Attention-bias loading under the plan's bias policy: ByTensorPresence keeps the presence rule, while a
    // resolved-on attention bias makes every layer's q/k/v bias tensors required and names the first one the file
    // lacks — whether the resolution came from a declared use_bias or from the profile's feature default. The
    // output-projection bias is never demanded: it loads when present, under any policy.
    private float[]?[] LoadAttentionBias(int layers, TensorRole role, long expectedLength, bool required)
    {
        if (!required)
        {
            return LoadOptionalBias(layers, role, expectedLength);
        }

        TensorDescriptor descriptor = Plan.Tensors[role];
        float[]?[] biases = new float[layers][];
        for (int l = 0; l < layers; l++)
        {
            LogicalTensor? tensor = descriptor.Resolve(_fileTensors, l);
            if (tensor is null)
            {
                throw new ModelException(
                    $"tensor '{descriptor.NameFor(l)}' is missing: attention biases are resolved on for architecture " +
                    $"'{Plan.Architecture}' (a declared {Plan.Prefix}attention.use_bias or the profile default) and " +
                    "this tensor is required in every layer");
            }

            biases[l] = LoadVector(tensor, checked((int)expectedLength));
        }

        return biases;
    }

    // Post-sublayer norm vectors under a placement that consumes them: every layer needs both, and the failure names the
    // first one the file lacks together with the resolved placement. The map declares these roles optional because a
    // pre-only profile must accept files without them; only the plan decides which side of that rule applies.
    private float[][] LoadPostOutputNorms(int layers, TensorRole role)
    {
        TensorDescriptor descriptor = Plan.Tensors[role];
        float[][] vectors = new float[layers][];
        for (int l = 0; l < layers; l++)
        {
            LogicalTensor? tensor = descriptor.Resolve(_fileTensors, l);
            if (tensor is null)
            {
                throw new ModelException(
                    $"tensor '{descriptor.NameFor(l)}' is missing: norm placement '{Plan.NormPlacement}' needs a post-sublayer output norm in every layer of architecture '{Plan.Architecture}'");
            }

            vectors[l] = LoadVector(tensor, Plan.HiddenSize);
        }

        return vectors;
    }

    // The other side of the same rule: a pre-only placement reads no post-sublayer norm vector at all, and those tensors
    // are not ignorable material — carrying one describes a graph this profile does not implement. Every such tensor in
    // the table is named before the load fails.
    private void RejectUnconsumedPostNorms(int layers)
    {
        List<string> present = [];
        foreach (TensorRole role in new TensorRole[] { TensorRole.PostAttentionNorm, TensorRole.PostFeedForwardNorm })
        {
            TensorDescriptor descriptor = Plan.Tensors[role];
            for (int l = 0; l < layers; l++)
            {
                if (descriptor.Resolve(_fileTensors, l) is not null)
                {
                    present.Add(descriptor.NameFor(l));
                }
            }
        }

        if (present.Count == 0)
        {
            return;
        }

        throw new ModelException(
            $"tensor{(present.Count > 1 ? "s" : "")} '{string.Join("', '", present)}' " +
            $"{(present.Count > 1 ? "are" : "is")} present but norm placement '{Plan.NormPlacement}' reads no " +
            $"post-sublayer output norms for architecture '{Plan.Architecture}'");
    }

    // One structured line naming the model, the architecture, and every tensor of the given roles the file
    // carries although the resolved plan consumes none of them, in the same style as the tokenizer profile defaults
    // and the profile substitution line. No line when no such tensor is present.
    private void ReportIgnoredTensors(Action<string>? logLine, string prefix, GgufFile file, IReadOnlyList<TensorRole> roles)
    {
        if (logLine is null)
        {
            return;
        }

        List<string> names = [];
        foreach (TensorRole role in roles)
        {
            TensorDescriptor descriptor = Plan.Tensors[role];
            if (descriptor.Scope == TensorScope.PerLayer)
            {
                for (int l = 0; l < Plan.LayerCount; l++)
                {
                    if (descriptor.Resolve(_fileTensors, l) is not null)
                    {
                        names.Add(descriptor.NameFor(l));
                    }
                }
            }
            else if (descriptor.Resolve(_fileTensors) is not null)
            {
                names.Add(descriptor.NameFor(-1));
            }
        }

        if (names.Count == 0)
        {
            return;
        }

        GgufMetadata metadata = file.Metadata;
        string modelName = metadata.Has("general.name") ? metadata.Get("general.name").AsString() : "";
        logLine(
            $"{prefix} model='{(modelName.Length > 0 ? modelName : file.Path)}' " +
            $"architecture='{Plan.Architecture}' tensors={string.Join(",", names)}");
    }

    private float[]? LoadOptionalVector(LogicalTensor? tensor, int expectedLength) =>
        tensor is null ? null : LoadVector(tensor, expectedLength);
}
