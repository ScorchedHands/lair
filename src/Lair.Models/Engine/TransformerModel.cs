using Lair.Core.Contracts;
using Lair.Kv;
using Lair.Loader;

namespace Lair.Models.Engine;

/// <summary>
/// Forward engine for supported GGUF architectures (llama, qwen2): embedding lookup, per-layer RMSNorm / GQA
/// attention with RoPE and paged KV via <see cref="SequenceKv"/>, SwiGLU feed-forward, final norm, and an untied
/// output matrix or tied token-embedding rows for logits. The graph mirrors the reference structure: each layer
/// normalizes its input stream (rmsnorm x weight) before the QKV GEMVs; RoPE rotates q/k pairs (i, i+headDim/2) at
/// the absolute position with theta_i = base^(-2i/headDim); attention scores scale by 1/sqrt(headDim) and softmax
/// runs online over positions 0..pos. The residual added before the FFN norm is the pre-attention stream value,
/// matching the reference graph's ffn_inp add; rope scaling variants are not supported. FP32 buffers are allocated
/// once at Open and reused across calls; packed matrix rows are consumed through <see cref="WeightRef"/> without
/// being materialized, except for embedding lookups and small norm/bias vectors preloaded at Open.
/// </summary>
public sealed class TransformerModel : IDisposable
{
    private readonly IBackend _backend;
    private readonly GgufMmapWeightSource _weights;

    // Kept for tensor name lookups only: the mmap source owns payload access.
    private readonly LogicalTensorTable _fileTensors;

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
    private readonly LogicalTensor _tokenEmbd;
    private readonly LogicalTensor? _output;
    private readonly float[] _outputNormWeight;
    private readonly float[]? _outputBias;

    // FP32 buffers reused across every forward call.
    private readonly float[] _cur;     // hidden: layer input stream (residual carrier) and FFN output accumulator
    private readonly float[] _normed;  // hidden: weighted rmsnorm before the QKV / FFN GEMVs
    private readonly float[] _q;       // headCount * headDim
    private readonly float[] _k;       // kvHeads * headDim (also the KV slot write source)
    private readonly float[] _v;       // kvHeads * headDim
    private readonly float[] _attnOut; // headCount * headDim: heads concatenated in order for wo GEMV
    private readonly float[] _proj;    // hidden: attn_output / ffn_down GEMV target before the residual add
    private readonly float[] _gateUp;  // intermediateSize: gate result, then silu(gate) * up written back into it
    private readonly float[] _up;      // intermediateSize: up GEMV result
    private readonly float[] _logits;  // vocabSize

    private readonly float[] _cos;          // maxContext * (headDim / 2)
    private readonly float[] _sin;
    private readonly int _ropeHalf;

    public ModelDesc Desc { get; }

    /// <summary>Largest sequence length this model can forward: the extent of its precomputed RoPE tables.</summary>
    public int MaxContextTokens { get; }

    public static TransformerModel Open(GgufFile file, IBackend backend, int maxContextTokens) => new(file, backend, maxContextTokens);

    private TransformerModel(GgufFile file, IBackend backend, int maxContextTokens)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxContextTokens);

        _backend = backend;
        Desc = ModelDesc.FromGguf(file);
        MaxContextTokens = maxContextTokens;
        _weights = new GgufMmapWeightSource(file);
        _fileTensors = file.Tensors;

        if (Desc.HeadDim % 2 != 0)
        {
            throw new ModelException($"head dimension {Desc.HeadDim} is odd; RoPE pairs require an even head dimension");
        }

        int layers = Desc.LayerCount;
        _wq = LoadMatrices(layers, "attn_q.weight");
        _wk = LoadMatrices(layers, "attn_k.weight");
        _wv = LoadMatrices(layers, "attn_v.weight");
        _wo = LoadMatrices(layers, "attn_output.weight");
        _ffnGate = LoadMatrices(layers, "ffn_gate.weight");
        _ffnUp = LoadMatrices(layers, "ffn_up.weight");
        _ffnDown = LoadMatrices(layers, "ffn_down.weight");

        // Matrix shapes must agree with the declared dimensions: rows are the output side, columns the input side.
        RequireShape(_wq, Desc.HiddenSize, Desc.HiddenSize);          // q emits headCount * headDim == hidden
        RequireShape(_wk, (long)Desc.KvHeadCount * Desc.HeadDim, Desc.HiddenSize);
        RequireShape(_wv, (long)Desc.KvHeadCount * Desc.HeadDim, Desc.HiddenSize);
        RequireShape(_wo, Desc.HiddenSize, (long)Desc.HeadCount * Desc.HeadDim);
        RequireShape(_ffnGate, Desc.IntermediateSize, Desc.HiddenSize);
        RequireShape(_ffnUp, Desc.IntermediateSize, Desc.HiddenSize);
        RequireShape(_ffnDown, Desc.HiddenSize, Desc.IntermediateSize);

        _attnNormWeight = LoadLayerVectors(layers, "attn_norm.weight", Desc.HiddenSize);
        _ffnNormWeight = LoadLayerVectors(layers, "ffn_norm.weight", Desc.HiddenSize);

        _biasQ = LoadOptionalBias(layers, "attn_q.bias", (long)Desc.HeadCount * Desc.HeadDim);
        _biasK = LoadOptionalBias(layers, "attn_k.bias", (long)Desc.KvHeadCount * Desc.HeadDim);
        _biasV = LoadOptionalBias(layers, "attn_v.bias", (long)Desc.KvHeadCount * Desc.HeadDim);
        _biasO = LoadOptionalBias(layers, "attn_output.bias", Desc.HiddenSize);

        LogicalTensor? tokenEmbd = file.Tensors.Find("token_embd.weight");
        if (tokenEmbd is null)
        {
            throw new ModelException($"tensor 'token_embd.weight' is missing: required by architecture '{Desc.Architecture}'");
        }

        RequireSingle(tokenEmbd, Desc.VocabSize, Desc.HiddenSize);
        _tokenEmbd = tokenEmbd;

        LogicalTensor? outputNorm = file.Tensors.Find("output_norm.weight");
        if (outputNorm is null)
        {
            throw new ModelException($"tensor 'output_norm.weight' is missing: required by architecture '{Desc.Architecture}'");
        }

        _outputNormWeight = LoadVector(outputNorm, Desc.HiddenSize);

        LogicalTensor? output = file.Tensors.Find("output.weight");
        if (output is not null)
        {
            RequireSingle(output, Desc.VocabSize, Desc.HiddenSize);
        }

        _output = output;
        _outputBias = LoadOptionalVector(file.Tensors.Find("output.bias"), Desc.VocabSize);

        int hidden = Desc.HiddenSize;
        _cur = new float[hidden];
        _normed = new float[hidden];
        _q = new float[Desc.HeadCount * Desc.HeadDim];
        _k = new float[Desc.KvHeadCount * Desc.HeadDim];
        _v = new float[Desc.KvHeadCount * Desc.HeadDim];
        _attnOut = new float[Desc.HeadCount * Desc.HeadDim];
        _proj = new float[hidden];
        _gateUp = new float[Desc.IntermediateSize];
        _up = new float[Desc.IntermediateSize];
        _logits = new float[Desc.VocabSize];

        _ropeHalf = Desc.HeadDim / 2;

        // Precompute cos/sin per position and pair up to the maximum context so decode is table-driven: identical
        // inputs always reproduce bit-identical rotations regardless of call order or buffer reuse.
        _cos = new float[maxContextTokens * _ropeHalf];
        _sin = new float[maxContextTokens * _ropeHalf];
        LayerOps.BuildRopeTables(Desc.RopeTheta, Desc.HeadDim, maxContextTokens, _cos, _sin);
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

    public void Dispose()
    {
        _weights.Dispose();
    }

    private void ForwardToken(int tokenId, int position, SequenceKv kv)
    {
        if ((uint)tokenId >= (uint)Desc.VocabSize)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenId), $"token id {tokenId} is outside the vocabulary of {Desc.VocabSize}.");
        }

        if (position >= MaxContextTokens)
        {
            throw new InvalidOperationException(
                $"sequence position {position} exceeds this model's max context of {MaxContextTokens} tokens.");
        }

        // Embedding lookup: one dequantized row of token_embd becomes the layer input stream.
        ReadOnlySpan<byte> embBytes = _weights.Resolve(_tokenEmbd);
        _backend.DequantRow(new WeightRef(embBytes, _tokenEmbd.Shape, _tokenEmbd.Format), tokenId, _cur.AsSpan(0, Desc.HiddenSize));

        kv.Append(); // reserves this position's slot in every layer; WriteK/WriteV address it by logical position

        int hidden = Desc.HiddenSize;
        float attnScale = 1f / MathF.Sqrt(Desc.HeadDim);

        for (int l = 0; l < Desc.LayerCount; l++)
        {
            // Pre-attention norm on the stream: normed = rmsnorm(cur) * attn_norm weight.
            LayerOps.RmsNormWeighted(_cur.AsSpan(0, hidden), _attnNormWeight[l], Desc.EpsRmsNorm, _normed);

            ReadOnlySpan<byte> bytes = _weights.Resolve(_wq[l]);
            _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(bytes, _wq[l].Shape, _wq[l].Format), _q);
            AddBias(_q, _biasQ[l]);

            bytes = _weights.Resolve(_wk[l]);
            _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(bytes, _wk[l].Shape, _wk[l].Format), _k);
            AddBias(_k, _biasK[l]);

            bytes = _weights.Resolve(_wv[l]);
            _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(bytes, _wv[l].Shape, _wv[l].Format), _v);
            AddBias(_v, _biasV[l]);

            ApplyRopeHeads(_q, Desc.HeadCount, position);
            ApplyRopeHeads(_k, Desc.KvHeadCount, position);

            kv.WriteK(position, l, _k);
            kv.WriteV(position, l, _v);

            LayerOps.CausalAttention(_backend, _q, Desc.HeadCount, Desc.KvHeadCount, Desc.HeadDim, kv, l, position, attnScale, _attnOut);

            // Output projection (+bias) into scratch; the residual added before the FFN norm is the pre-attention
            // stream value (ffn_inp = attn_out + inpSA), matching the reference graph. cur held that stream through
            // attention untouched, so adding proj in place turns it into ffn_inp.
            bytes = _weights.Resolve(_wo[l]);
            _backend.Gemv(_attnOut.AsSpan(0, Desc.HeadCount * Desc.HeadDim), new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), _proj);
            AddBias(_proj, _biasO[l]);

            for (int i = 0; i < hidden; i++)
            {
                _cur[i] += _proj[i];
            }

            LayerOps.RmsNormWeighted(_cur.AsSpan(0, hidden), _ffnNormWeight[l], Desc.EpsRmsNorm, _normed);

            bytes = _weights.Resolve(_ffnGate[l]);
            _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(bytes, _ffnGate[l].Shape, _ffnGate[l].Format), _gateUp);

            bytes = _weights.Resolve(_ffnUp[l]);
            _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(bytes, _ffnUp[l].Shape, _ffnUp[l].Format), _up);

            // act = silu(gate) * up with silu(x) = x / (1 + exp(-x)), applied in place on the gate buffer.
            for (int i = 0; i < Desc.IntermediateSize; i++)
            {
                _gateUp[i] = LayerOps.Silu(_gateUp[i]) * _up[i];
            }

            bytes = _weights.Resolve(_ffnDown[l]);
            _backend.Gemv(_gateUp.AsSpan(0, Desc.IntermediateSize), new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), _proj);

            for (int i = 0; i < hidden; i++)
            {
                _cur[i] += _proj[i]; // ffn_out + ffn_inp: the input stream for the next layer
            }
        }

        LayerOps.RmsNormWeighted(_cur.AsSpan(0, hidden), _outputNormWeight, Desc.EpsRmsNorm, _normed);

        LogicalTensor lmHead = _output ?? _tokenEmbd; // tied weights reuse the token-embedding rows as logits
        ReadOnlySpan<byte> headBytes = _weights.Resolve(lmHead);
        _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(headBytes, lmHead.Shape, lmHead.Format), _logits);

        if (_outputBias is not null)
        {
            AddBias(_logits, _outputBias);
        }
    }

    /// <summary>Rotates every head region of data in place at the given position using the precomputed tables.</summary>
    private void ApplyRopeHeads(float[] data, int headCount, int position)
    {
        ReadOnlySpan<float> cosAtPosition = _cos.AsSpan(position * _ropeHalf, _ropeHalf);
        ReadOnlySpan<float> sinAtPosition = _sin.AsSpan(position * _ropeHalf, _ropeHalf);

        for (int h = 0; h < headCount; h++)
        {
            // Pairs are relative to each head's own region.
            LayerOps.ApplyRopeToHead(data.AsSpan(h * Desc.HeadDim, Desc.HeadDim), _ropeHalf, cosAtPosition, sinAtPosition);
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

    private LogicalTensor[] LoadMatrices(int layers, string suffix)
    {
        LogicalTensor[] tensors = new LogicalTensor[layers];
        for (int l = 0; l < layers; l++)
        {
            tensors[l] = RequireTensor(suffix, $"blk.{l}.{suffix}");
        }

        return tensors;
    }

    private float[][] LoadLayerVectors(int layers, string suffix, int expectedLength)
    {
        float[][] vectors = new float[layers][];
        for (int l = 0; l < layers; l++)
        {
            LogicalTensor tensor = RequireTensor(suffix, $"blk.{l}.{suffix}");
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

    private LogicalTensor RequireTensor(string suffix, string name)
    {
        return _fileTensors.Find(name) ?? throw new ModelException(
            $"tensor '{name}' is missing: required by architecture '{Desc.Architecture}'");
    }

    private float[]?[] LoadOptionalBias(int layers, string suffix, long expectedLength)
    {
        float[]?[] biases = new float[layers][]; // null entries for absent bias tensors
        for (int l = 0; l < layers; l++)
        {
            LogicalTensor? tensor = _fileTensors.Find($"blk.{l}.{suffix}");
            if (tensor is not null)
            {
                biases[l] = LoadVector(tensor, checked((int)expectedLength));
            }
        }

        return biases;
    }

    private float[]? LoadOptionalVector(LogicalTensor? tensor, int expectedLength) =>
        tensor is null ? null : LoadVector(tensor, expectedLength);
}
