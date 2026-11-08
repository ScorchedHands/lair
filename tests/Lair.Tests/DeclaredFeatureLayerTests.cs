using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Forward-level goldens for the declared feature flags: a model whose head dimension is declared
/// independently of hidden / head_count (the gemma-shaped geometry, with grouped-query attention) against a
/// double-precision scalar reference forward; per-head QK-norm on and off against reference forwards that
/// prove the norm changes the logits and lands between the projection and RoPE; declared norm-bias vectors
/// applied through the LayerNorm strategy end to end; and a declared <c>use_bias=false</c> with bias tensors
/// present producing logits bit-identical to the same tensors without them, while a declared <c>true</c>
/// applies them. Engine-to-engine comparisons are asserted bit-for-bit; reference comparisons use the
/// double-precision tolerance shared by the strategy goldens.
/// </summary>
[Trait("Category", "Layers")]
public sealed class DeclaredFeatureLayerTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
            }
        }

        _tempFiles.Clear();
    }

    // ---------------------------------------------------------------- independent head dimension

    [Fact]
    public void Forward_IndependentHeadDim_MatchesScalarReferenceForward_InDecodeBatchAndEncode()
    {
        FeatureModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors()));

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        Assert.Equal(4, model.Plan.HeadDim); // declared via head_length: 2 x 4 != 6
        Assert.Equal(6, model.Plan.HiddenSize);
        Assert.Equal(1, model.Plan.KvHeadCount); // grouped-query mapping: both heads share the single kv-head

        FeatureReference reference = new(spec);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        Assert.True(
            Closest(model.DecodeNext(1, kv), reference.LogitsAt([], 1)),
            "decode logits at position 0 must match the reference forward");

        Assert.True(
            Closest(model.DecodeNext(4, kv), reference.LogitsAt([1], 4)),
            "decode logits at position 1 must match the reference forward");

        // Batched decode: two fresh sequences stepped in one batched call.
        using SequenceKv batchA = new(pool);
        using SequenceKv batchB = new(pool);
        Span<float> batchLogits = model.DecodeBatch([1, 4], [batchA, batchB]);

        Assert.True(
            Closest(batchLogits.Slice(0, model.Plan.VocabSize), reference.LogitsAt([], 1)),
            "batched logits for sequence A must match the reference forward");
        Assert.True(
            Closest(batchLogits.Slice(model.Plan.VocabSize, model.Plan.VocabSize), reference.LogitsAt([], 4)),
            "batched logits for sequence B must match the reference forward");

        // Encode: bidirectional attention over a fresh three-token sequence.
        using SequenceKv encodeKv = new(pool);
        float[] encoded = model.Encode([1, 4, 2], encodeKv);

        Assert.Equal(3 * model.Plan.HiddenSize, encoded.Length);
        double[][] expectedEncoded = reference.Encoded([1, 4, 2]);
        for (int p = 0; p < 3; p++)
        {
            for (int i = 0; i < model.Plan.HiddenSize; i++)
            {
                float actual = encoded[p * model.Plan.HiddenSize + i];
                double expected = expectedEncoded[p][i];
                Assert.True(
                    Math.Abs(actual - expected) <= 1e-5 * Math.Max(1.0, Math.Abs(expected)),
                    $"encode hidden[{p}][{i}]: expected {expected:G9}, got {actual:G9}");
            }
        }
    }

    [Fact]
    public void Forward_IndependentHeadDim_RejectsQProjectionWithWrongRowCount()
    {
        FeatureModelSpec spec = new();
        List<TensorDef> tensors = spec.Tensors();
        int idx = tensors.FindIndex(t => t.Name == "blk.0.attn_q.weight");
        tensors[idx] = new TensorDef("blk.0.attn_q.weight", [6, 6], [.. spec.AttnQ[..36]]); // must be [head_count * head_dim, hidden] = [8, 6]

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(spec.MetadataPairs(), [.. tensors])));
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));
        Assert.Contains("blk.0.attn_q.weight", ex.Message);
        Assert.Contains("[8, 6]", ex.Message);
    }

    // ---------------------------------------------------------------- QK-norm

    [Fact]
    public void Forward_QkNormOnOff_MatchesReferenceForward_AndNormChangesLogits_BeforeRope()
    {
        FeatureModelSpec spec = new();
        string offPath = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors()));
        string onPath = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.attention.use_qk_norm", true)],
            [.. spec.Tensors(), new TensorDef("blk.0.attn_q_norm.weight", [FeatureModelSpec.HeadDim], spec.QkNorm), new TensorDef("blk.0.attn_k_norm.weight", [FeatureModelSpec.HeadDim], spec.KNorm)]));

        FeatureReference reference = new(spec);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        float[] onDecode0;
        float[] onDecode1;
        float[] onEncoded;
        {
            using GgufFile gguf = GgufFile.Open(onPath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

            Assert.Equal(QkNormMode.PerHeadWeighted, model.Plan.QkNorm);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
            using SequenceKv kv = new(pool);

            onDecode0 = model.DecodeNext(1, kv).ToArray();
            onDecode1 = model.DecodeNext(4, kv).ToArray();

            using SequenceKv encodeKv = new(pool);
            onEncoded = model.Encode([1, 4, 2], encodeKv);
        }

        Assert.True(
            Closest(onDecode0, reference.LogitsAt([], 1, qkNorm: true)),
            "QK-norm decode logits at position 0 must match the reference forward");
        Assert.True(
            Closest(onDecode1, reference.LogitsAt([1], 4, qkNorm: true)),
            "QK-norm decode logits at position 1 must match the reference forward");

        // The norm genuinely changes the logits: the same tensors without the norm give different values.
        // (Position 0 alone cannot show this: a single-position softmax weights 1 regardless of the scores,
        // so the position-0 logits are norm-invariant by construction — the position-1 pair carries the proof.)
        Assert.False(
            Closest(onDecode1, reference.LogitsAt([1], 4, qkNorm: false)),
            "QK-norm on must not leave the logits equal to the no-norm reference");

        // The norm lands between the projection and RoPE: with the deliberately uneven norm weights, rotating
        // first and normalizing after is a different forward, and the engine matches the before-RoPE reference.
        Assert.False(
            Closest(onDecode1, reference.LogitsAt([1], 4, qkNorm: true, qkNormAfterRope: true)),
            "QK-norm applied after RoPE would produce different logits; the engine must apply it before");

        double[][] expectedEncoded = reference.Encoded([1, 4, 2], qkNorm: true);
        for (int p = 0; p < 3; p++)
        {
            for (int i = 0; i < FeatureModelSpec.Hidden; i++)
            {
                float actual = onEncoded[p * FeatureModelSpec.Hidden + i];
                double expected = expectedEncoded[p][i];
                Assert.True(
                    Math.Abs(actual - expected) <= 1e-5 * Math.Max(1.0, Math.Abs(expected)),
                    $"encode hidden[{p}][{i}] with QK-norm: expected {expected:G9}, got {actual:G9}");
            }
        }

        // Batched decode with the norm must be bit-identical to stepping the same sequence alone.
        {
            using GgufFile gguf = GgufFile.Open(onPath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
            using SequenceKv single = new(pool);
            model.Prefill([1, 4], single);
            float[] singleLogits = model.LastLogits().ToArray();

            using SequenceKv batched = new(pool);
            model.DecodeBatch([1], [batched]);
            Span<float> batchLogits = model.DecodeBatch([4], [batched]);
            Assert.True(
                batchLogits.SequenceEqual(singleLogits.AsSpan()),
                "a token batched through DecodeBatch with QK-norm must be bit-identical to stepping the same sequence alone");
        }

        // Off again: the same tensors without the declaration match the no-norm reference exactly.
        {
            using GgufFile gguf = GgufFile.Open(offPath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

            Assert.Equal(QkNormMode.Off, model.Plan.QkNorm);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
            using SequenceKv kv = new(pool);
            Assert.True(
                Closest(model.DecodeNext(1, kv), reference.LogitsAt([], 1, qkNorm: false)),
                "QK-norm off must match the no-norm reference forward");
        }
    }

    // ---------------------------------------------------------------- norm bias through LayerNorm

    [Fact]
    public void Forward_NormBiasTensors_AreAppliedThroughLayerNorm_InDecodeAndEncode()
    {
        FeatureModelSpec spec = new();
        string path = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.attention.norm_bias", true)],
            [.. spec.Tensors(), new TensorDef("blk.0.attn_norm.bias", [FeatureModelSpec.Hidden], spec.AttnNormBias), new TensorDef("blk.0.ffn_norm.bias", [FeatureModelSpec.Hidden], spec.FfnNormBias), new TensorDef("output_norm.bias", [FeatureModelSpec.Hidden], spec.OutputNormBias)]));

        ModelProfile layerNormProfile = ModelProfiles.Resolve("llama") with { Norm = NormStrategy.LayerNorm };

        FeatureReference reference = new(spec);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, layerNormProfile);

        Assert.Equal(NormStrategy.LayerNorm, model.Plan.Norm);
        Assert.True(model.Plan.UseNormBias);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        Assert.True(
            Closest(model.DecodeNext(1, kv), reference.LogitsAt([], 1, norm: NormStrategy.LayerNorm, normBias: true)),
            "decode logits at position 0 must match the LayerNorm+bias reference forward");
        Assert.True(
            Closest(model.DecodeNext(4, kv), reference.LogitsAt([1], 4, norm: NormStrategy.LayerNorm, normBias: true)),
            "decode logits at position 1 must match the LayerNorm+bias reference forward");

        // The bias genuinely shifts the stream: the same tensors without the declared bias differ.
        Assert.False(
            Closest(model.LastLogits(), reference.LogitsAt([1], 4, norm: NormStrategy.LayerNorm, normBias: false)),
            "declared norm-bias must change the logits against the no-bias reference");

        using SequenceKv encodeKv = new(pool);
        float[] encoded = model.Encode([1, 4, 2], encodeKv);

        double[][] expectedEncoded = reference.Encoded([1, 4, 2], norm: NormStrategy.LayerNorm, normBias: true);
        for (int p = 0; p < 3; p++)
        {
            for (int i = 0; i < FeatureModelSpec.Hidden; i++)
            {
                float actual = encoded[p * FeatureModelSpec.Hidden + i];
                double expected = expectedEncoded[p][i];
                Assert.True(
                    Math.Abs(actual - expected) <= 1e-5 * Math.Max(1.0, Math.Abs(expected)),
                    $"encode hidden[{p}][{i}] with norm bias: expected {expected:G9}, got {actual:G9}");
            }
        }
    }

    // ---------------------------------------------------------------- declared use_bias

    [Fact]
    public void Forward_DeclaredUseBiasFalse_WithBiasTensorsPresent_MatchesTheTensorFreeModelBitForBit()
    {
        FeatureModelSpec spec = new();
        List<TensorDef> withBiases =
        [
            .. spec.Tensors(),
            new TensorDef("blk.0.attn_q.bias", [FeatureModelSpec.Heads * FeatureModelSpec.HeadDim], spec.AttnQBias),
            new TensorDef("blk.0.attn_k.bias", [FeatureModelSpec.KvHeads * FeatureModelSpec.HeadDim], spec.AttnKBias),
            new TensorDef("blk.0.attn_v.bias", [FeatureModelSpec.KvHeads * FeatureModelSpec.HeadDim], spec.AttnVBias),
            new TensorDef("blk.0.attn_output.bias", [FeatureModelSpec.Hidden], spec.AttnOBias),
        ];

        string offPath = WriteTemp(BuildGguf([.. spec.MetadataPairs(), ("llama.attention.use_bias", false)], [.. withBiases]));
        string onPath = WriteTemp(BuildGguf([.. spec.MetadataPairs(), ("llama.attention.use_bias", true)], [.. withBiases]));
        string nonePath = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors())); // same tensors, no bias tensors at all

        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        float[] offLogits;
        {
            using GgufFile gguf = GgufFile.Open(offPath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

            Assert.False(model.Plan.UseAttentionBias);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
            using SequenceKv kv = new(pool);
            model.DecodeNext(1, kv);
            offLogits = model.DecodeNext(4, kv).ToArray();
        }

        float[] noneLogits;
        {
            using GgufFile gguf = GgufFile.Open(nonePath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
            using SequenceKv kv = new(pool);
            model.DecodeNext(1, kv);
            noneLogits = model.DecodeNext(4, kv).ToArray();
        }

        Assert.True(
            offLogits.AsSpan().SequenceEqual(noneLogits.AsSpan()),
            "declared use_bias=false with bias tensors present must produce logits identical to the same model without them");

        // Declared true with the same tensors applies them: the logits differ from both no-bias runs.
        float[] onLogits;
        {
            using GgufFile gguf = GgufFile.Open(onPath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

            Assert.True(model.Plan.UseAttentionBias);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
            using SequenceKv kv = new(pool);
            model.DecodeNext(1, kv);
            onLogits = model.DecodeNext(4, kv).ToArray();
        }

        Assert.False(
            onLogits.AsSpan().SequenceEqual(noneLogits.AsSpan()),
            "declared use_bias=true with bias tensors present must apply them");

        FeatureReference reference = new(spec);
        Assert.True(
            Closest(onLogits, reference.LogitsAt([1], 4, attentionBias: true)),
            "declared use_bias=true logits must match the biased reference forward");
    }

    // ---------------------------------------------------------------- reference oracle

    private static bool Closest(ReadOnlySpan<float> actual, double[] expected)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            if (Math.Abs(actual[i] - expected[i]) > 1e-5 * Math.Max(1.0, Math.Abs(expected[i])))
            {
                return false;
            }
        }

        return true;
    }

    // A complete single-layer model with hidden 6, heads 2, kv-heads 1, a declared head dim of 4 (2 x 4 != 6),
    // ffn 8, vocab 6 — all F32 tensors with hand-picked payloads the reference forward reads directly.
    private sealed class FeatureModelSpec
    {
        public const int Hidden = 6;
        public const int Heads = 2;
        public const int KvHeads = 1;
        public const int HeadDim = 4;
        public const int Intermediate = 8;
        public const int Vocab = 6;
        public const float Eps = 1e-5f;
        public const double RopeTheta = 10_000.0;

        public List<(string Key, object Value)> MetadataPairs() =>
        [
            ("general.architecture", "llama"),
            ("llama.embedding_length", Hidden),
            ("llama.block_count", 1),
            ("llama.attention.head_count", Heads),
            ("llama.attention.head_count_kv", KvHeads),
            ("llama.feed_forward_length", Intermediate),
            ("llama.attention.head_length", HeadDim),
            ("llama.attention.layer_norm_rms_epsilon", Eps),
            ("llama.attention.layer_norm_epsilon", Eps),
            ("llama.vocab_size", Vocab),
        ];

        public List<TensorDef> Tensors() =>
        [
            new("token_embd.weight", [Vocab, Hidden], Embeddings),
            new("blk.0.attn_q.weight", [Heads * HeadDim, Hidden], AttnQ),
            new("blk.0.attn_k.weight", [KvHeads * HeadDim, Hidden], AttnK),
            new("blk.0.attn_v.weight", [KvHeads * HeadDim, Hidden], AttnV),
            new("blk.0.attn_output.weight", [Hidden, Heads * HeadDim], AttnO),
            new("blk.0.ffn_gate.weight", [Intermediate, Hidden], FfnGate),
            new("blk.0.ffn_up.weight", [Intermediate, Hidden], FfnUp),
            new("blk.0.ffn_down.weight", [Hidden, Intermediate], FfnDown),
            new("blk.0.attn_norm.weight", [Hidden], AttnNorm),
            new("blk.0.ffn_norm.weight", [Hidden], FfnNorm),
            new("output_norm.weight", [Hidden], OutputNorm),
        ];

        // Rows with distinct directions and magnitudes: the K vectors then differ per position, so a
        // magnitude-only op like QK-norm moves the attention weights by a detectable amount.
        public readonly float[] Embeddings =
        [
            0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f,
            0.11f, -0.21f, 0.31f, -0.41f, 0.51f, -0.61f,
            0.32f, 0.22f, -0.32f, 0.42f, -0.52f, 0.62f,
            -0.13f, 0.23f, 0.33f, -0.43f, 0.53f, 0.63f,
            0.94f, -0.84f, 0.74f, -0.64f, 0.54f, -0.44f,
            0.15f, 0.25f, 0.35f, 0.45f, 0.55f, -0.65f,
        ];

        public readonly float[] AttnQ =
        [
            0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f,
            0.06f, 0.11f, 0.16f, 0.21f, 0.26f, 0.31f,
            0.07f, 0.12f, 0.17f, 0.22f, 0.27f, 0.32f,
            0.08f, 0.13f, 0.18f, 0.23f, 0.28f, 0.33f,
            0.09f, 0.14f, 0.19f, 0.24f, 0.29f, 0.34f,
            0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.35f,
            0.11f, 0.16f, 0.21f, 0.26f, 0.31f, 0.36f,
            0.12f, 0.17f, 0.22f, 0.27f, 0.32f, 0.37f,
        ];

        public readonly float[] AttnK =
        [
            0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f,
            0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f,
            0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f,
            0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 0.09f,
        ];

        // Distinct V rows per position: attention weight changes then move the output by a detectable amount.
        public readonly float[] AttnV =
        [
            0.09f, 0.08f, 0.07f, 0.06f, 0.05f, 0.04f,
            -0.45f, 0.38f, -0.31f, 0.24f, -0.17f, 0.10f,
            0.30f, -0.22f, 0.14f, -0.06f, 0.42f, -0.34f,
            0.06f, 0.05f, 0.04f, 0.03f, 0.02f, 0.01f,
        ];

        public readonly float[] AttnO =
        [
            0.10f, 0.00f, 0.20f, 0.00f, 0.30f, 0.00f, 0.15f, 0.00f,
            0.00f, 0.30f, 0.00f, 0.40f, 0.00f, 0.50f, 0.00f, 0.25f,
            0.15f, 0.00f, 0.25f, 0.00f, 0.35f, 0.00f, 0.20f, 0.00f,
            0.00f, 0.35f, 0.00f, 0.45f, 0.00f, 0.55f, 0.00f, 0.30f,
            0.10f, 0.00f, 0.15f, 0.00f, 0.25f, 0.00f, 0.35f, 0.00f,
            0.00f, 0.20f, 0.00f, 0.30f, 0.00f, 0.40f, 0.00f, 0.45f,
        ];

        public readonly float[] FfnGate =
        [
            0.10f, 0.11f, 0.12f, 0.13f, 0.14f, 0.15f,
            0.16f, 0.17f, 0.18f, 0.19f, 0.20f, 0.21f,
            0.22f, 0.23f, 0.24f, 0.25f, 0.26f, 0.27f,
            0.28f, 0.29f, 0.30f, 0.31f, 0.32f, 0.33f,
            0.34f, 0.35f, 0.36f, 0.37f, 0.38f, 0.39f,
            0.40f, 0.41f, 0.42f, 0.43f, 0.44f, 0.45f,
            0.46f, 0.47f, 0.48f, 0.49f, 0.50f, 0.51f,
            0.52f, 0.53f, 0.54f, 0.55f, 0.56f, 0.57f,
        ];

        public readonly float[] FfnUp =
        [
            0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f,
            0.07f, 0.08f, 0.09f, 0.10f, 0.11f, 0.12f,
            0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f,
            0.19f, 0.20f, 0.21f, 0.22f, 0.23f, 0.24f,
            0.25f, 0.26f, 0.27f, 0.28f, 0.29f, 0.30f,
            0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f,
            0.37f, 0.38f, 0.39f, 0.40f, 0.41f, 0.42f,
            0.43f, 0.44f, 0.45f, 0.46f, 0.47f, 0.48f,
        ];

        public readonly float[] FfnDown =
        [
            0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.10f, 0.11f, 0.12f,
            0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.20f,
            0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f, 0.27f, 0.28f,
            0.29f, 0.30f, 0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f,
            0.37f, 0.38f, 0.39f, 0.40f, 0.41f, 0.42f, 0.43f, 0.44f,
            0.45f, 0.46f, 0.47f, 0.48f, 0.49f, 0.50f, 0.51f, 0.52f,
        ];

        public readonly float[] AttnNorm = [1.10f, 0.90f, 1.05f, 0.95f, 1.00f, 1.15f];
        public readonly float[] FfnNorm = [0.90f, 1.10f, 0.95f, 1.05f, 1.00f, 0.85f];
        public readonly float[] OutputNorm = [1.00f, 1.02f, 0.98f, 1.01f, 0.99f, 1.03f];

        // Deliberately uneven per-head weights: with them, normalizing before RoPE and normalizing after are
        // different forwards, so the goldens pin the order.
        public readonly float[] QkNorm = [1.30f, 0.80f, 1.10f, 0.90f];
        public readonly float[] KNorm = [0.85f, 1.20f, 0.95f, 1.05f];

        public readonly float[] AttnQBias = [0.10f, -0.20f, 0.30f, -0.40f, 0.50f, -0.60f, 0.15f, -0.25f];
        public readonly float[] AttnKBias = [0.05f, -0.10f, 0.15f, -0.20f];
        public readonly float[] AttnVBias = [-0.05f, 0.10f, -0.15f, 0.20f];
        public readonly float[] AttnOBias = [0.01f, 0.02f, -0.03f, 0.04f, -0.05f, 0.06f];

        public readonly float[] AttnNormBias = [0.10f, -0.10f, 0.20f, -0.20f, 0.30f, -0.30f];
        public readonly float[] FfnNormBias = [-0.15f, 0.15f, -0.25f, 0.25f, -0.35f, 0.35f];
        public readonly float[] OutputNormBias = [0.05f, -0.05f, 0.10f, -0.10f, 0.15f, -0.15f];
    }

    // A scalar reference forward over the spec's hand-picked tensors: RMSNorm or LayerNorm (optionally biased),
    // optional attention biases, optional per-head QK-norm before (or, for the ordering proof, after) RoPE,
    // standard full-head RoPE, causal or bidirectional attention with an explicit softmax, and silu(gate) * up.
    // Computed in double and compared against the engine's float logits with the shared tolerance.
    private sealed class FeatureReference
    {
        private readonly FeatureModelSpec _spec;

        public FeatureReference(FeatureModelSpec spec) => _spec = spec;

        /// <summary>Logits for a query token forwarded at position historyTokens.Length after replaying the sequence's earlier tokens through the same layer.</summary>
        public double[] LogitsAt(
            int[] historyTokens,
            int queryToken,
            bool qkNorm = false,
            bool qkNormAfterRope = false,
            NormStrategy norm = NormStrategy.RmsNorm,
            bool normBias = false,
            bool attentionBias = false)
        {
            double[][] kvK = new double[historyTokens.Length + 1][];
            double[][] kvV = new double[historyTokens.Length + 1][];

            for (int p = 0; p < historyTokens.Length; p++)
            {
                Advance(Row(_spec.Embeddings, historyTokens[p], FeatureModelSpec.Hidden), p, kvK, kvV, qkNorm, qkNormAfterRope, norm, normBias, attentionBias);
            }

            double[] cur = Row(_spec.Embeddings, queryToken, FeatureModelSpec.Hidden);
            Advance(cur, historyTokens.Length, kvK, kvV, qkNorm, qkNormAfterRope, norm, normBias, attentionBias);

            return OutputLogits(cur, norm, normBias);
        }

        public double[][] Encoded(
            int[] tokens,
            bool qkNorm = false,
            NormStrategy norm = NormStrategy.RmsNorm,
            bool normBias = false)
        {
            int n = tokens.Length;
            double[][] cur = new double[n][];
            for (int i = 0; i < n; i++)
            {
                cur[i] = Row(_spec.Embeddings, tokens[i], FeatureModelSpec.Hidden);
            }

            double[][] kvK = new double[n][];
            double[][] kvV = new double[n][];

            // Phase 1: pre-attention norm, QKV, QK-norm, RoPE, K/V writes for every position.
            double[][] qAll = new double[n][];
            for (int p = 0; p < n; p++)
            {
                double[] normed = Norm(cur[p], _spec.AttnNorm, normBias ? _spec.AttnNormBias : null, norm);
                double[] q = Gemv(_spec.AttnQ, FeatureModelSpec.Heads * FeatureModelSpec.HeadDim, FeatureModelSpec.Hidden, normed);
                double[] k = Gemv(_spec.AttnK, FeatureModelSpec.KvHeads * FeatureModelSpec.HeadDim, FeatureModelSpec.Hidden, normed);
                double[] v = Gemv(_spec.AttnV, FeatureModelSpec.KvHeads * FeatureModelSpec.HeadDim, FeatureModelSpec.Hidden, normed);

                if (qkNorm)
                {
                    QkNormHeads(q, _spec.QkNorm);
                    QkNormHeads(k, _spec.KNorm);
                }

                Rope(q, p);
                Rope(k, p);

                qAll[p] = q;
                kvK[p] = k;
                kvV[p] = v;
            }

            // Phase 2: every query attends over the whole finished KV span.
            double[] attnOut = new double[n * FeatureModelSpec.Heads * FeatureModelSpec.HeadDim];
            for (int p = 0; p < n; p++)
            {
                Attention(qAll[p], kvK, kvV, 0, n - 1, attnOut.AsSpan(p * FeatureModelSpec.Heads * FeatureModelSpec.HeadDim));
            }

            // Phase 3: output projection and the ffn_inp residual add.
            for (int p = 0; p < n; p++)
            {
                double[] proj = Gemv(_spec.AttnO, FeatureModelSpec.Hidden, FeatureModelSpec.Heads * FeatureModelSpec.HeadDim, attnOut.AsSpan(p * FeatureModelSpec.Heads * FeatureModelSpec.HeadDim).ToArray());
                for (int i = 0; i < FeatureModelSpec.Hidden; i++)
                {
                    cur[p][i] += proj[i];
                }
            }

            // Phases 4 and 5: gate/up GEMVs, silu gating, down GEMV, and the ffn_out residual add.
            for (int p = 0; p < n; p++)
            {
                double[] normed = Norm(cur[p], _spec.FfnNorm, normBias ? _spec.FfnNormBias : null, norm);
                double[] gate = Gemv(_spec.FfnGate, FeatureModelSpec.Intermediate, FeatureModelSpec.Hidden, normed);
                double[] up = Gemv(_spec.FfnUp, FeatureModelSpec.Intermediate, FeatureModelSpec.Hidden, normed);
                for (int i = 0; i < FeatureModelSpec.Intermediate; i++)
                {
                    gate[i] = Silu(gate[i]) * up[i];
                }

                double[] proj = Gemv(_spec.FfnDown, FeatureModelSpec.Hidden, FeatureModelSpec.Intermediate, gate);
                for (int i = 0; i < FeatureModelSpec.Hidden; i++)
                {
                    cur[p][i] += proj[i];
                }
            }

            double[][] encoded = new double[n][];
            for (int i = 0; i < n; i++)
            {
                encoded[i] = Norm(cur[i], _spec.OutputNorm, normBias ? _spec.OutputNormBias : null, norm);
            }

            return encoded;
        }

        // One causal decode layer: norm, QKV, QK-norm, RoPE, KV write at the position, causal attention,
        // o-projection, residual, FFN norm, gate/up, silu gating, down, residual — the stream advances in place.
        private void Advance(
            double[] cur,
            int position,
            double[][] kvK,
            double[][] kvV,
            bool qkNorm,
            bool qkNormAfterRope,
            NormStrategy norm,
            bool normBias,
            bool attentionBias)
        {
            double[] normed = Norm(cur, _spec.AttnNorm, normBias ? _spec.AttnNormBias : null, norm);
            double[] q = Gemv(_spec.AttnQ, FeatureModelSpec.Heads * FeatureModelSpec.HeadDim, FeatureModelSpec.Hidden, normed);
            double[] k = Gemv(_spec.AttnK, FeatureModelSpec.KvHeads * FeatureModelSpec.HeadDim, FeatureModelSpec.Hidden, normed);
            double[] v = Gemv(_spec.AttnV, FeatureModelSpec.KvHeads * FeatureModelSpec.HeadDim, FeatureModelSpec.Hidden, normed);

            if (attentionBias)
            {
                AddBias(q, _spec.AttnQBias);
                AddBias(k, _spec.AttnKBias);
                AddBias(v, _spec.AttnVBias);
            }

            if (qkNorm && !qkNormAfterRope)
            {
                QkNormHeads(q, _spec.QkNorm);
                QkNormHeads(k, _spec.KNorm);
            }

            Rope(q, position);
            Rope(k, position);

            if (qkNorm && qkNormAfterRope)
            {
                QkNormHeads(q, _spec.QkNorm);
                QkNormHeads(k, _spec.KNorm);
            }

            kvK[position] = k;
            kvV[position] = v;

            double[] attnOut = new double[FeatureModelSpec.Heads * FeatureModelSpec.HeadDim];
            Attention(q, kvK, kvV, 0, position, attnOut);

            double[] proj = Gemv(_spec.AttnO, FeatureModelSpec.Hidden, FeatureModelSpec.Heads * FeatureModelSpec.HeadDim, attnOut);
            if (attentionBias)
            {
                AddBias(proj, _spec.AttnOBias);
            }

            for (int i = 0; i < FeatureModelSpec.Hidden; i++)
            {
                cur[i] += proj[i];
            }

            double[] ffnNormed = Norm(cur, _spec.FfnNorm, normBias ? _spec.FfnNormBias : null, norm);
            double[] gate = Gemv(_spec.FfnGate, FeatureModelSpec.Intermediate, FeatureModelSpec.Hidden, ffnNormed);
            double[] up = Gemv(_spec.FfnUp, FeatureModelSpec.Intermediate, FeatureModelSpec.Hidden, ffnNormed);
            for (int i = 0; i < FeatureModelSpec.Intermediate; i++)
            {
                gate[i] = Silu(gate[i]) * up[i];
            }

            double[] down = Gemv(_spec.FfnDown, FeatureModelSpec.Hidden, FeatureModelSpec.Intermediate, gate);
            for (int i = 0; i < FeatureModelSpec.Hidden; i++)
            {
                cur[i] += down[i];
            }
        }

        private double[] OutputLogits(double[] cur, NormStrategy norm, bool normBias)
        {
            double[] normed = Norm(cur, _spec.OutputNorm, normBias ? _spec.OutputNormBias : null, norm);
            return Gemv(_spec.Embeddings, FeatureModelSpec.Vocab, FeatureModelSpec.Hidden, normed); // tied lm_head
        }

        private static void Attention(double[] q, double[][] kvK, double[][] kvV, int from, int to, Span<double> outHead)
        {
            const int headDim = FeatureModelSpec.HeadDim;
            double scale = 1.0 / Math.Sqrt(headDim);

            // heads 2, kv-heads 1: every head reads kv-head (h * 1 / 2) == 0.
            for (int h = 0; h < FeatureModelSpec.Heads; h++)
            {
                int kvHead = h * FeatureModelSpec.KvHeads / FeatureModelSpec.Heads;

                double[] scores = new double[to - from + 1];
                double max = double.NegativeInfinity;
                for (int t = from; t <= to; t++)
                {
                    double dot = 0.0;
                    for (int j = 0; j < headDim; j++)
                    {
                        dot += q[h * headDim + j] * kvK[t][kvHead * headDim + j];
                    }

                    scores[t - from] = dot * scale;
                    max = Math.Max(max, scores[t - from]);
                }

                double sum = 0.0;
                for (int t = from; t <= to; t++)
                {
                    scores[t - from] = Math.Exp(scores[t - from] - max);
                    sum += scores[t - from];
                }

                for (int j = 0; j < headDim; j++)
                {
                    double acc = 0.0;
                    for (int t = from; t <= to; t++)
                    {
                        acc += scores[t - from] / sum * kvV[t][kvHead * headDim + j];
                    }

                    outHead[h * headDim + j] = acc;
                }
            }
        }

        private static void Rope(double[] headRegion, int position)
        {
            for (int h = 0; h < headRegion.Length / FeatureModelSpec.HeadDim; h++)
            {
                for (int i = 0; i < FeatureModelSpec.HeadDim / 2; i++)
                {
                    double invFreq = Math.Pow(FeatureModelSpec.RopeTheta, -2.0 * i / FeatureModelSpec.HeadDim);
                    double angle = position * invFreq;
                    double c = Math.Cos(angle);
                    double s = Math.Sin(angle);

                    double lo = headRegion[h * FeatureModelSpec.HeadDim + 2 * i];
                    double hi = headRegion[h * FeatureModelSpec.HeadDim + 2 * i + 1];
                    headRegion[h * FeatureModelSpec.HeadDim + 2 * i] = lo * c - hi * s;
                    headRegion[h * FeatureModelSpec.HeadDim + 2 * i + 1] = hi * c + lo * s;
                }
            }
        }

        // Per-head RMS norm with the head-dimension weight vector: each head normalizes its own slice, in place.
        private static void QkNormHeads(double[] heads, float[] weight)
        {
            for (int region = 0; region < heads.Length; region += FeatureModelSpec.HeadDim)
            {
                double sumSq = 0.0;
                for (int i = 0; i < FeatureModelSpec.HeadDim; i++)
                {
                    sumSq += heads[region + i] * heads[region + i];
                }

                double invRms = 1.0 / Math.Sqrt(sumSq / FeatureModelSpec.HeadDim + FeatureModelSpec.Eps);
                for (int i = 0; i < FeatureModelSpec.HeadDim; i++)
                {
                    heads[region + i] = heads[region + i] * invRms * weight[i];
                }
            }
        }

        private static double[] Norm(double[] x, float[] weight, float[]? bias, NormStrategy strategy)
        {
            int n = x.Length;
            double[] result = new double[n];

            if (strategy == NormStrategy.RmsNorm)
            {
                double sumSq = 0.0;
                for (int i = 0; i < n; i++)
                {
                    sumSq += x[i] * x[i];
                }

                double invRms = 1.0 / Math.Sqrt(sumSq / n + FeatureModelSpec.Eps);
                for (int i = 0; i < n; i++)
                {
                    result[i] = x[i] * invRms * weight[i];
                }

                return result;
            }

            double mean = 0.0;
            for (int i = 0; i < n; i++)
            {
                mean += x[i];
            }

            mean /= n;

            double var = 0.0;
            for (int i = 0; i < n; i++)
            {
                var += (x[i] - mean) * (x[i] - mean);
            }

            var /= n;
            double invStd = 1.0 / Math.Sqrt(var + FeatureModelSpec.Eps);

            for (int i = 0; i < n; i++)
            {
                double value = (x[i] - mean) * invStd * weight[i];
                if (bias is not null)
                {
                    value += bias[i];
                }

                result[i] = value;
            }

            return result;
        }

        private static void AddBias(double[] values, float[] bias)
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] += bias[i];
            }
        }

        private static double Silu(double x) => x / (1.0 + Math.Exp(-x));

        private static double[] Gemv(float[] rows, int outCount, int inCount, double[] input)
        {
            double[] output = new double[outCount];
            for (int r = 0; r < outCount; r++)
            {
                double acc = 0.0;
                for (int c = 0; c < inCount; c++)
                {
                    acc += rows[r * inCount + c] * input[c];
                }

                output[r] = acc;
            }

            return output;
        }

        private static double[] Row(float[] data, int row, int length)
        {
            double[] result = new double[length];
            for (int i = 0; i < length; i++)
            {
                result[i] = data[row * length + i];
            }

            return result;
        }
    }

    private sealed record TensorDef(string Name, int[] Dims, float[] Payload); // dims in logical (row-major) order; vectors are one-dimensional

    /// <summary>Builds a GGUF v3 image from KV pairs and F32 tensors with the exact payloads the reference forward reads.</summary>
    private static byte[] BuildGguf(IReadOnlyList<(string Key, object Value)> kvPairs, IReadOnlyList<TensorDef> tensors)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(GgufMagic);
        writer.Write(3u); // version
        writer.Write((ulong)tensors.Count);
        writer.Write((ulong)kvPairs.Count);

        foreach ((string key, object value) in kvPairs)
        {
            WriteGgufString(writer, key);
            switch (value)
            {
                case string s:
                    writer.Write(8u); // STRING
                    WriteGgufString(writer, s);
                    break;
                case int i:
                    writer.Write(5u); // INT32
                    writer.Write(i);
                    break;
                case float f:
                    writer.Write(6u); // FLOAT32
                    writer.Write(f);
                    break;
                case bool b:
                    writer.Write(7u); // BOOL
                    writer.Write((byte)(b ? 1 : 0));
                    break;
                default:
                    throw new ArgumentException($"Unsupported fixture value type {value.GetType().Name}.", nameof(kvPairs));
            }
        }

        long offset = 0;
        foreach (TensorDef tensor in tensors)
        {
            // GGUF stores dims fastest-varying first: the logical row-major order is written reversed.
            int[] stored = new int[tensor.Dims.Length];
            for (int d = 0; d < tensor.Dims.Length; d++)
            {
                stored[d] = tensor.Dims[tensor.Dims.Length - 1 - d];
            }

            WriteGgufString(writer, tensor.Name);
            writer.Write((uint)stored.Length);
            foreach (int dim in stored)
            {
                writer.Write((ulong)dim);
            }

            writer.Write(0u); // ggml type F32
            writer.Write((ulong)offset);

            long elements = 1;
            foreach (int dim in tensor.Dims)
            {
                elements *= dim;
            }

            offset += elements * 4;
        }

        long headerLength = stream.Position;
        long dataStart = (headerLength + 31) / 32 * 32; // GGUF default alignment is 32
        for (long i = headerLength; i < dataStart; i++)
        {
            writer.Write((byte)0);
        }

        foreach (TensorDef tensor in tensors)
        {
            long elements = 1;
            foreach (int dim in tensor.Dims)
            {
                elements *= dim;
            }

            if (tensor.Payload.Length != elements)
            {
                throw new ArgumentException($"tensor '{tensor.Name}' payload has {tensor.Payload.Length} entries, expected {elements}.", nameof(tensors));
            }

            foreach (float value in tensor.Payload)
            {
                writer.Write(value);
            }
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-declared-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }
}
