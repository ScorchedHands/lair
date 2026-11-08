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
/// softmax runs online over positions 0..pos, narrowed to <c>[max(0, pos - window + 1), pos]</c> when the plan
/// resolves an attention window. The window masks scores only: every position's K/V is still written and stays
/// readable. The residual added before the
/// FFN norm is the pre-attention stream value, matching the reference graph's ffn_inp add; rope scaling variants
/// are not supported. FP32 buffers are allocated once at Open and reused across calls; packed matrix rows are
/// consumed through <see cref="WeightRef"/> without being materialized, except for embedding lookups and small
/// norm/bias vectors preloaded at Open. Attention-bias tensors follow the plan's declared policy: under
/// <see cref="BiasPolicy.Declared"/> the declared <c>use_bias</c> flag makes them required or leaves them
/// unloaded and unapplied, with one structured log line naming the ignored tensors; a declared per-head QK-norm
/// normalizes each q and k head with its own weight vector after the projection and before RoPE. A resolved
/// final-logit softcap saturates every logit vector — untied output head or tied token-embedding rows — after
/// the full vector is computed and before any softmax, sampling, or copy-out.
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
    private readonly LogicalTensor[] _ffnGate;
    private readonly LogicalTensor[] _ffnUp;
    private readonly LogicalTensor[] _ffnDown;

    // Small FP32 vectors preloaded at Open: per-layer norm weights and any biases present in the file. Null
    // entries mean the tensor is absent (llama files carry no attention bias, qwen2 files may).
    private readonly float[][] _attnNormWeight;
    private readonly float[][] _ffnNormWeight;
    private readonly float[]?[] _biasQ;
    private readonly float[]?[] _biasK;
    private readonly float[]?[] _biasV;
    private readonly float[]?[] _biasO;

    // Per-head QK-norm weight vectors (one headDim vector per layer, shared by every head of its side),
    // preloaded only when the plan declares the feature: every head normalizes its own headDim slice.
    private readonly float[][] _attnQNorm;
    private readonly float[][] _attnKNorm;

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

    private readonly float[] _cos;          // maxContext * (headDim / 2)
    private readonly float[] _sin;
    private readonly int _ropeHalf;

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
    /// forward. Assembly-internal so no host configuration path can reach it.
    /// </summary>
    internal static TransformerModel Open(GgufFile file, IBackend backend, int maxContextTokens, ModelProfile profile) =>
        new(file, backend, maxContextTokens, weights: null, logLine: null, profile);

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

        int layers = Plan.LayerCount;

        _wq = LoadMatrices(layers, TensorRole.AttentionQuery, FusedQkvName(TensorRole.AttentionQuery));
        _wk = LoadMatrices(layers, TensorRole.AttentionKey, FusedQkvName(TensorRole.AttentionKey));
        _wv = LoadMatrices(layers, TensorRole.AttentionValue, FusedQkvName(TensorRole.AttentionValue));
        _wo = LoadMatrices(layers, TensorRole.AttentionOutput);
        _ffnGate = LoadMatrices(layers, TensorRole.FeedForwardGate);
        _ffnUp = LoadMatrices(layers, TensorRole.FeedForwardUp);
        _ffnDown = LoadMatrices(layers, TensorRole.FeedForwardDown);

        // Matrix shapes must agree with the declared dimensions: rows are the output side, columns the input side.
        RequireShape(_wq, (long)Plan.HeadCount * Plan.HeadDim, Plan.HiddenSize); // q emits headCount * head_dim
        RequireShape(_wk, (long)Plan.KvHeadCount * Plan.HeadDim, Plan.HiddenSize);
        RequireShape(_wv, (long)Plan.KvHeadCount * Plan.HeadDim, Plan.HiddenSize);
        RequireShape(_wo, Plan.HiddenSize, (long)Plan.HeadCount * Plan.HeadDim);
        RequireShape(_ffnGate, Plan.IntermediateSize, Plan.HiddenSize);
        RequireShape(_ffnUp, Plan.IntermediateSize, Plan.HiddenSize);
        RequireShape(_ffnDown, Plan.HiddenSize, Plan.IntermediateSize);

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
            // Declared true makes the tensors required; ByTensorPresence keeps the presence rule.
            bool required = Plan.Bias == BiasPolicy.Declared;
            _biasQ = LoadAttentionBias(layers, TensorRole.AttentionQueryBias, (long)Plan.HeadCount * Plan.HeadDim, required);
            _biasK = LoadAttentionBias(layers, TensorRole.AttentionKeyBias, (long)Plan.KvHeadCount * Plan.HeadDim, required);
            _biasV = LoadAttentionBias(layers, TensorRole.AttentionValueBias, (long)Plan.KvHeadCount * Plan.HeadDim, required);
            _biasO = LoadAttentionBias(layers, TensorRole.AttentionOutputBias, Plan.HiddenSize, required);
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

        if (Plan.HasQkNorm)
        {
            _attnQNorm = LoadLayerVectors(layers, TensorRole.AttentionQueryNorm, Plan.HeadDim);
            _attnKNorm = LoadLayerVectors(layers, TensorRole.AttentionKeyNorm, Plan.HeadDim);
        }
        else
        {
            _attnQNorm = new float[layers][];
            _attnKNorm = new float[layers][];
            ReportIgnoredTensors(logLine, "qk_norm_ignored", file, [TensorRole.AttentionQueryNorm, TensorRole.AttentionKeyNorm]);
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

        _ropeHalf = Plan.HeadDim / 2;

        // Precompute cos/sin per position and pair up to the maximum context so decode is table-driven: identical
        // inputs always reproduce bit-identical rotations regardless of call order or buffer reuse.
        _cos = new float[maxContextTokens * _ropeHalf];
        _sin = new float[maxContextTokens * _ropeHalf];
        LayerOps.BuildRopeTables(Plan.RopeTheta, Plan.HeadDim, maxContextTokens, _cos, _sin);
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

        // Reserve this step's slot in every sequence before any layer writes, mirroring ForwardToken's append.
        for (int s = 0; s < n; s++)
        {
            kvs[s].Append();
        }

        int hidden = Plan.HiddenSize;
        int qkvRegion = Plan.HeadCount * Plan.HeadDim;
        int kvRegion = Plan.KvHeadCount * Plan.HeadDim;
        int inter = Plan.IntermediateSize;
        float attnScale = 1f / MathF.Sqrt(Plan.HeadDim);

        for (int l = 0; l < Plan.LayerCount; l++)
        {
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

                if (Plan.HasQkNorm)
                {
                    LayerOps.RmsNormWeightedHeads(_batchQ.AsSpan(s * qkvRegion, qkvRegion), _attnQNorm[l], Plan.HeadDim, Plan.EpsRmsNorm, _batchQ.AsSpan(s * qkvRegion, qkvRegion));
                    LayerOps.RmsNormWeightedHeads(_batchK.AsSpan(s * kvRegion, kvRegion), _attnKNorm[l], Plan.HeadDim, Plan.EpsRmsNorm, _batchK.AsSpan(s * kvRegion, kvRegion));
                }

                ApplyRopeHeads(_batchQ.AsSpan(s * qkvRegion, qkvRegion), Plan.HeadCount, positions[s]);
                ApplyRopeHeads(_batchK.AsSpan(s * kvRegion, kvRegion), Plan.KvHeadCount, positions[s]);

                kvs[s].WriteK(positions[s], l, _batchK.AsSpan(s * kvRegion, kvRegion));
                kvs[s].WriteV(positions[s], l, _batchV.AsSpan(s * kvRegion, kvRegion));

                LayerOps.CausalAttention(_backend, _batchQ.AsSpan(s * qkvRegion, qkvRegion), Plan.HeadCount, Plan.KvHeadCount, Plan.HeadDim, kvs[s], l, positions[s], attnScale, Plan.AttentionSoftcap, Plan.AttentionWindow, _batchAttnOut.AsSpan(s * qkvRegion, qkvRegion));
            }

            bytes = _weights.Resolve(_wo[l]);
            _backend.GemvBatch(_batchAttnOut.AsSpan(0, n * qkvRegion), n, new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), _batchProj.AsSpan(0, n * hidden));

            for (int s = 0; s < n; s++)
            {
                AddBias(_batchProj.AsSpan(s * hidden, hidden), _biasO[l]);

                // ffn_inp add on the sequence's own stream, same order as the single-token forward.
                Span<float> cur = _batchCur.AsSpan(s * hidden, hidden);
                ReadOnlySpan<float> proj = _batchProj.AsSpan(s * hidden, hidden);
                for (int i = 0; i < hidden; i++)
                {
                    cur[i] += proj[i];
                }

                ApplyNorm(cur, _ffnNormWeight[l], _ffnNormBias[l], _batchNormed.AsSpan(s * hidden, hidden));
            }

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

            for (int s = 0; s < n; s++)
            {
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
    /// causal prefix, narrowed to the plan's resolved attention window around the query's own position when one is
    /// declared — and returns the final hidden states after the output norm, without lm_head logits. Per-position
    /// norms, RoPE at absolute positions, biases, residuals, and the gated activation are exactly the decode-path
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

        int qkvRegion = Plan.HeadCount * Plan.HeadDim;
        int kvRegion = Plan.KvHeadCount * Plan.HeadDim;
        int inter = Plan.IntermediateSize;
        float attnScale = 1f / MathF.Sqrt(Plan.HeadDim);

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

            // Reserve every position's KV slot up front so later layer passes address them by absolute position.
            for (int i = 0; i < n; i++)
            {
                kv.Append();
            }

            int chunkCount = (n + EncodeChunkTokens - 1) / EncodeChunkTokens;

            for (int l = 0; l < Plan.LayerCount; l++)
            {
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

                        if (Plan.HasQkNorm)
                        {
                            LayerOps.RmsNormWeightedHeads(chunkQ.AsSpan(i * qkvRegion, qkvRegion), _attnQNorm[l], Plan.HeadDim, Plan.EpsRmsNorm, chunkQ.AsSpan(i * qkvRegion, qkvRegion));
                            LayerOps.RmsNormWeightedHeads(chunkK.AsSpan(i * kvRegion, kvRegion), _attnKNorm[l], Plan.HeadDim, Plan.EpsRmsNorm, chunkK.AsSpan(i * kvRegion, kvRegion));
                        }

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
                        Plan.AttentionWindow,
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

        kv.Append(); // reserves this position's slot in every layer; WriteK/WriteV address it by logical position

        int hidden = Plan.HiddenSize;
        float attnScale = 1f / MathF.Sqrt(Plan.HeadDim);

        for (int l = 0; l < Plan.LayerCount; l++)
        {
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

            // Declared QK-norm: each q and k head is RMS-normalized with the head-dimension weight vector,
            // after the projection and bias and before RoPE.
            if (Plan.HasQkNorm)
            {
                LayerOps.RmsNormWeightedHeads(_q, _attnQNorm[l], Plan.HeadDim, Plan.EpsRmsNorm, _q);
                LayerOps.RmsNormWeightedHeads(_k, _attnKNorm[l], Plan.HeadDim, Plan.EpsRmsNorm, _k);
            }

            ApplyRopeHeads(_q, Plan.HeadCount, position);
            ApplyRopeHeads(_k, Plan.KvHeadCount, position);

            kv.WriteK(position, l, _k);
            kv.WriteV(position, l, _v);

            LayerOps.CausalAttention(_backend, _q, Plan.HeadCount, Plan.KvHeadCount, Plan.HeadDim, kv, l, position, attnScale, Plan.AttentionSoftcap, Plan.AttentionWindow, _attnOut);

            // Output projection (+bias) into scratch; the residual added before the FFN norm is the pre-attention
            // stream value (ffn_inp = attn_out + inpSA), matching the reference graph. cur held that stream through
            // attention untouched, so adding proj in place turns it into ffn_inp.
            bytes = _weights.Resolve(_wo[l]);
            _backend.Gemv(_attnOut.AsSpan(0, Plan.HeadCount * Plan.HeadDim), new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), _proj);
            AddBias(_proj, _biasO[l]);

            for (int i = 0; i < hidden; i++)
            {
                _cur[i] += _proj[i];
            }

            ApplyNorm(_cur.AsSpan(0, hidden), _ffnNormWeight[l], _ffnNormBias[l], _normed);

            bytes = _weights.Resolve(_ffnGate[l]);
            _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(bytes, _ffnGate[l].Shape, _ffnGate[l].Format), _gateUp);

            bytes = _weights.Resolve(_ffnUp[l]);
            _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(bytes, _ffnUp[l].Shape, _ffnUp[l].Format), _up);

            // act = activation(gate) * up with the plan's resolved activation, applied in place on the gate buffer.
            ApplyActivation(_gateUp.AsSpan(0, Plan.IntermediateSize), _up.AsSpan(0, Plan.IntermediateSize));

            bytes = _weights.Resolve(_ffnDown[l]);
            _backend.Gemv(_gateUp.AsSpan(0, Plan.IntermediateSize), new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), _proj);

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
    // loaded (an empty span otherwise) and is consumed only by the LayerNorm strategy.
    private void ApplyNorm(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias, Span<float> dst)
    {
        switch (Plan.Norm)
        {
            case NormStrategy.RmsNorm:
                LayerOps.RmsNormWeighted(src, weight, Plan.EpsRmsNorm, dst);
                break;
            case NormStrategy.LayerNorm:
                LayerOps.LayerNormWeighted(src, weight, bias, Plan.EpsRmsNorm, dst);
                break;
            default:
                throw new ModelException($"norm strategy '{Plan.Norm}' is not implemented by the forward engine");
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
    // declared use_bias makes every attention-bias tensor required and names the first one the file lacks.
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
                    $"tensor '{descriptor.NameFor(l)}' is missing: {Plan.Prefix}attention.use_bias declares attention biases required for architecture '{Plan.Architecture}'");
            }

            biases[l] = LoadVector(tensor, checked((int)expectedLength));
        }

        return biases;
    }

    // One structured line naming the model, the architecture, and every tensor of the given roles the file
    // carries although the plan declares the feature off, in the same style as the tokenizer profile defaults
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
