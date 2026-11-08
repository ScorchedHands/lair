using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Forward-level goldens for norms placed on sublayer outputs and for the unweighted per-head QK-norm form. A synthetic
/// model whose plan resolves LayerNorm, GeGlu, and post-sublayer output norms — the graph that normalizes each sublayer
/// output once more before its residual add — is forwarded against a double-precision scalar reference implementing that
/// order, and the same weights without the post-norm tensors are shown to give different logits under a pre-only
/// placement. Batched decode is bit-identical to single-token decode under the post-norms, the encoder path is golden
/// under them, and each post-norm vector is shown to move the logits rather than sit unread. The unweighted per-head
/// form is shown to match the weighted form with an all-ones vector bit-for-bit, at the operation level and through the
/// whole forward, and to differ from no normalization at a position where attention is non-degenerate. Engine-to-engine
/// comparisons are asserted bit-for-bit; reference comparisons use the double-precision tolerance shared by the strategy
/// goldens.
/// </summary>
[Trait("Category", "Layers")]
public sealed class ResidualNormLayerTests : IDisposable
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

    // ---------------------------------------------------------------- post-sublayer output norms

    [Fact]
    public void Forward_PostSublayerOutputNorms_MatchScalarReference_AndDifferFromAPreOnlyPlacement()
    {
        PostNormModelSpec spec = new();
        string postPath = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors(withPostNorms: true)));
        string prePath = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors(withPostNorms: false)));

        ModelProfile postProfile = ModelProfiles.Resolve("llama") with
        {
            Norm = NormStrategy.LayerNorm,
            Activation = ActivationStrategy.GeGlu,
            NormPlacement = NormPlacement.PostSublayerOutputs,
        };
        ModelProfile preProfile = ModelProfiles.Resolve("llama") with
        {
            Norm = NormStrategy.LayerNorm,
            Activation = ActivationStrategy.GeGlu,
        };

        PostNormReference reference = new(spec);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        float[] decode0;
        float[] decode1;
        {
            using GgufFile gguf = GgufFile.Open(postPath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, postProfile);

            Assert.Equal(NormPlacement.PostSublayerOutputs, model.Plan.NormPlacement);
            Assert.Equal(NormStrategy.LayerNorm, model.Plan.Norm);
            Assert.Equal(ActivationStrategy.GeGlu, model.Plan.Activation);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
            using SequenceKv kv = new(pool);

            decode0 = model.DecodeNext(1, kv).ToArray();
            decode1 = model.DecodeNext(4, kv).ToArray();
        }

        Assert.True(
            Closest(decode0, reference.LogitsAt([], 1, postNorms: true)),
            "post-norm decode logits at position 0 must match the reference forward");
        Assert.True(
            Closest(decode1, reference.LogitsAt([1], 4, postNorms: true)),
            "post-norm decode logits at position 1 must match the reference forward");

        // The same weights under a pre-only placement give a different forward: the post-norms are part of the graph
        // this placement selects, not decoration the engine happens to read.
        float[] preDecode1;
        {
            using GgufFile gguf = GgufFile.Open(prePath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, preProfile);

            Assert.Equal(NormPlacement.PreOnly, model.Plan.NormPlacement);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
            using SequenceKv kv = new(pool);

            model.DecodeNext(1, kv);
            preDecode1 = model.DecodeNext(4, kv).ToArray();
        }

        Assert.True(
            Closest(preDecode1, reference.LogitsAt([1], 4, postNorms: false)),
            "the pre-only placement must match the reference forward without the post-norms");
        Assert.True(
            decode1.Zip(preDecode1).Any(pair => Math.Abs(pair.First - pair.Second) > 1e-6f * Math.Max(1.0f, Math.Abs(pair.Second))),
            "post-sublayer output norms must move the logits against the same tensors under a pre-only placement");
    }

    [Fact]
    public void Forward_PostSublayerOutputNorms_DecodeBatchIsBitIdenticalToSingleTokenDecode()
    {
        PostNormModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors(withPostNorms: true)));

        ModelProfile postProfile = ModelProfiles.Resolve("llama") with
        {
            Norm = NormStrategy.LayerNorm,
            Activation = ActivationStrategy.GeGlu,
            NormPlacement = NormPlacement.PostSublayerOutputs,
        };

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, postProfile);

        int[] tokens = [1, 4, 2];

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 8, backend);

        // Single-token path over the three tokens: the last forward's logits are the reference.
        float[] singleLogits;
        {
            using SequenceKv single = new(pool);
            model.Prefill(tokens, single);
            singleLogits = model.LastLogits().ToArray();
        }

        // Batched path: the same three tokens, one per batch step across a fresh sequence.
        float[] batchLogits;
        {
            using SequenceKv batched = new(pool);
            model.DecodeBatch([tokens[0]], [batched]);
            model.DecodeBatch([tokens[1]], [batched]);
            batchLogits = model.DecodeBatch([tokens[2]], [batched]).ToArray();
        }

        Assert.True(
            batchLogits.AsSpan().SequenceEqual(singleLogits.AsSpan()),
            "a token batched through DecodeBatch under post-norms must be bit-identical to stepping the same sequence alone");

        // Two sequences in one batched call: each matches its own single-token forward at the same position.
        {
            using SequenceKv seqA = new(pool);
            using SequenceKv seqB = new(pool);
            Span<float> pairLogits = model.DecodeBatch([1, 4], [seqA, seqB]);
            float[] firstOfPair = pairLogits.Slice(0, model.Plan.VocabSize).ToArray();
            float[] secondOfPair = pairLogits.Slice(model.Plan.VocabSize, model.Plan.VocabSize).ToArray();

            using SequenceKv soloA = new(pool);
            float[] soloALogits = model.DecodeNext(1, soloA).ToArray();
            Assert.True(
                firstOfPair.AsSpan().SequenceEqual(soloALogits.AsSpan()),
                "the batched first sequence must match its own single-token forward");

            using SequenceKv soloB = new(pool);
            float[] soloBLogits = model.DecodeNext(4, soloB).ToArray();
            Assert.True(
                secondOfPair.AsSpan().SequenceEqual(soloBLogits.AsSpan()),
                "the batched second sequence must match its own single-token forward");
        }
    }

    [Fact]
    public void Forward_PostSublayerOutputNorms_EncodeMatchesReferenceForward()
    {
        PostNormModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors(withPostNorms: true)));

        ModelProfile postProfile = ModelProfiles.Resolve("llama") with
        {
            Norm = NormStrategy.LayerNorm,
            Activation = ActivationStrategy.GeGlu,
            NormPlacement = NormPlacement.PostSublayerOutputs,
        };

        PostNormReference reference = new(spec);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, postProfile);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
        using SequenceKv encodeKv = new(pool);
        float[] encoded = model.Encode([1, 4, 2], encodeKv);

        Assert.Equal(3 * model.Plan.HiddenSize, encoded.Length);
        double[][] expectedEncoded = reference.Encoded([1, 4, 2], postNorms: true);
        for (int p = 0; p < 3; p++)
        {
            for (int i = 0; i < model.Plan.HiddenSize; i++)
            {
                float actual = encoded[p * model.Plan.HiddenSize + i];
                double expected = expectedEncoded[p][i];
                Assert.True(
                    Math.Abs(actual - expected) <= 1e-5 * Math.Max(1.0, Math.Abs(expected)),
                    $"encode hidden[{p}][{i}] under post-norms: expected {expected:G9}, got {actual:G9}");
            }
        }

        // The encoder path also distinguishes the placement: the same tensors without the post-norms encode differently.
        using GgufFile preGguf = GgufFile.Open(WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors(withPostNorms: false))));
        using TransformerModel preModel = TransformerModel.Open(preGguf, backend, maxContextTokens: 8, ModelProfiles.Resolve("llama") with { Norm = NormStrategy.LayerNorm, Activation = ActivationStrategy.GeGlu });
        using SequenceKv preKv = new(pool);
        float[] preEncoded = preModel.Encode([1, 4, 2], preKv);

        Assert.True(
            encoded.Zip(preEncoded).Any(pair => Math.Abs(pair.First - pair.Second) > 1e-5 * Math.Max(1.0, Math.Abs(pair.Second))),
            "the encoder forward under post-norms must differ from the same tensors under a pre-only placement");
    }

    [Fact]
    public void Forward_PostSublayerNormVectors_AreConsumedAndNotMerelyLoaded()
    {
        PostNormModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors(withPostNorms: true)));
        string changedAttentionPath = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors(
            withPostNorms: true,
            postAttentionNorm: [0.55f, 1.35f, 0.85f, 1.15f, 1.25f, 0.75f])));
        string changedFeedForwardPath = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors(
            withPostNorms: true,
            postFeedForwardNorm: [1.30f, 0.70f, 1.20f, 0.80f, 0.95f, 1.10f])));

        ModelProfile postProfile = ModelProfiles.Resolve("llama") with
        {
            Norm = NormStrategy.LayerNorm,
            Activation = ActivationStrategy.GeGlu,
            NormPlacement = NormPlacement.PostSublayerOutputs,
        };

        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        float[] baseline = ForwardTwoTokens(path, postProfile, backend);
        float[] withAttentionVectorChanged = ForwardTwoTokens(changedAttentionPath, postProfile, backend);
        float[] withFeedForwardVectorChanged = ForwardTwoTokens(changedFeedForwardPath, postProfile, backend);

        Assert.True(
            baseline.Zip(withAttentionVectorChanged).Any(pair => Math.Abs(pair.First - pair.Second) > 1e-6 * Math.Max(1.0, Math.Abs(pair.Second))),
            "changing the post-attention norm vector must change the logits");
        Assert.True(
            baseline.Zip(withFeedForwardVectorChanged).Any(pair => Math.Abs(pair.First - pair.Second) > 1e-6 * Math.Max(1.0, Math.Abs(pair.Second))),
            "changing the post-feed-forward norm vector must change the logits");
    }

    // ---------------------------------------------------------------- unweighted per-head QK-norm

    [Fact]
    public void RmsNormUnweightedHeads_MatchesTheWeightedFormWithAnAllOnesVector_BitForBit()
    {
        float[] src = [0.30f, -1.20f, 2.50f, 0.75f, -0.40f, 0.90f, -1.70f, 3.10f, 0.15f, -0.55f, 1.05f, 0.65f];
        float[] ones = [1f, 1f, 1f, 1f]; // four heads of four dimensions each
        float eps = 1e-5f;

        float[] unweighted = new float[src.Length];
        float[] weighted = new float[src.Length];
        LayerOps.RmsNormUnweightedHeads(src, 4, eps, unweighted);
        LayerOps.RmsNormWeightedHeads(src, ones, 4, eps, weighted);

        Assert.True(
            unweighted.AsSpan().SequenceEqual(weighted.AsSpan()),
            "the unweighted per-head form must be bit-for-bit what the weighted form produces for an all-ones vector");

        // It genuinely normalizes: the result differs from leaving each head untouched, head by head.
        Assert.True(
            unweighted.Zip(src).Any(pair => Math.Abs(pair.First - pair.Second) > 1e-6f),
            "per-head normalization must change a head whose magnitude is not already unit root-mean-square");

        // Each head is normalized on its own: after one pass its mean square sits one epsilon below 1 — the epsilon
        // under the root is what keeps the divisor above the true RMS — and a second pass leaves the head where it is.
        float[] head = [0.30f, -1.20f, 2.50f, 0.75f];
        float[] normalizedHead = new float[4];
        LayerOps.RmsNormUnweightedHeads(head, 4, eps, normalizedHead);

        double sumSq = 0.0;
        foreach (float value in normalizedHead)
        {
            sumSq += (double)value * value;
        }

        Assert.True(Math.Abs(sumSq / head.Length - 1.0) <= 1e-4, $"normalized head mean square {sumSq / head.Length:G9}");

        float[] twiceNormalized = new float[4];
        LayerOps.RmsNormUnweightedHeads(normalizedHead, 4, eps, twiceNormalized);
        for (int i = 0; i < normalizedHead.Length; i++)
        {
            Assert.True(
                Math.Abs(twiceNormalized[i] - normalizedHead[i]) <= 1e-5 * Math.Max(1.0, Math.Abs(normalizedHead[i])),
                $"second normalization pass[{i}]: expected {normalizedHead[i]:G9}, got {twiceNormalized[i]:G9}");
        }

        Assert.Throws<ArgumentException>(() => LayerOps.RmsNormUnweightedHeads(new float[6], 4, eps, new float[6]));
        Assert.Throws<ArgumentOutOfRangeException>(() => LayerOps.RmsNormUnweightedHeads(src, 0, eps, new float[src.Length]));
    }

    [Fact]
    public void Forward_UnweightedQkNorm_MatchesTheWeightedFormWithOnesBitForBit_AndDiffersFromNoNorm()
    {
        PostNormModelSpec spec = new();

        // No per-head norm tensors at all: the unweighted mode demands none, so the file loads and forwards.
        string unweightedPath = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors(withPostNorms: false, withQkNormTensors: false)));

        // The same weights with all-ones per-head vectors: the weighted form must produce exactly the same logits.
        string weightedOnesPath = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.attention.use_qk_norm", true)],
            [
                .. spec.Tensors(withPostNorms: false, withQkNormTensors: false),
                new TensorDef("blk.0.attn_q_norm.weight", [PostNormModelSpec.HeadDim], [1f, 1f, 1f, 1f]),
                new TensorDef("blk.0.attn_k_norm.weight", [PostNormModelSpec.HeadDim], [1f, 1f, 1f, 1f]),
            ]));

        // Same tensors, normalization off: the position-1 logits must move, where attention is non-degenerate.
        string offPath = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.attention.use_qk_norm", false)],
            spec.Tensors(withPostNorms: false, withQkNormTensors: false)));

        ModelProfile unweightedProfile = ModelProfiles.Resolve("llama") with { DefaultQkNorm = QkNormMode.PerHeadUnweighted };

        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        float[] unweightedLogits;
        {
            using GgufFile gguf = GgufFile.Open(unweightedPath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, unweightedProfile);

            Assert.Equal(QkNormMode.PerHeadUnweighted, model.Plan.QkNorm);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
            using SequenceKv kv = new(pool);
            model.DecodeNext(1, kv);
            unweightedLogits = model.DecodeNext(4, kv).ToArray();
        }

        float[] weightedOnesLogits;
        {
            using GgufFile gguf = GgufFile.Open(weightedOnesPath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

            Assert.Equal(QkNormMode.PerHeadWeighted, model.Plan.QkNorm);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
            using SequenceKv kv = new(pool);
            model.DecodeNext(1, kv);
            weightedOnesLogits = model.DecodeNext(4, kv).ToArray();
        }

        Assert.True(
            unweightedLogits.AsSpan().SequenceEqual(weightedOnesLogits.AsSpan()),
            "the unweighted per-head form must produce bit-identical logits to the weighted form with an all-ones vector");

        float[] offLogits;
        {
            using GgufFile gguf = GgufFile.Open(offPath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

            Assert.Equal(QkNormMode.Off, model.Plan.QkNorm);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
            using SequenceKv kv = new(pool);
            model.DecodeNext(1, kv);
            offLogits = model.DecodeNext(4, kv).ToArray();
        }

        Assert.True(
            unweightedLogits.Zip(offLogits).Any(pair => Math.Abs(pair.First - pair.Second) > 1e-6f * Math.Max(1.0, Math.Abs(pair.Second))),
            "per-head QK-norm must move the logits at a position where attention is non-degenerate");
    }

    // ---------------------------------------------------------------- engine helpers

    private static float[] ForwardTwoTokens(string path, ModelProfile profile, CpuBackend backend)
    {
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, profile);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        model.DecodeNext(1, kv);
        return model.DecodeNext(4, kv).ToArray();
    }

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

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-residual-norm-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // A complete single-layer model with hidden 6, heads 2, kv-heads 1, head dim 4 declared by key_length /
    // value_length (2 x 4 != 6), ffn 8, vocab 6 — all F32 tensors with hand-picked payloads the reference forward
    // reads directly. The post-sublayer norm vectors are deliberately uneven so consuming them is visible.
    private sealed class PostNormModelSpec
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
            ("llama.attention.key_length", HeadDim),
            ("llama.attention.value_length", HeadDim),
            ("llama.attention.layer_norm_rms_epsilon", Eps),
            ("llama.vocab_size", Vocab),
        ];

        public List<TensorDef> Tensors(
            bool withPostNorms,
            float[]? postAttentionNorm = null,
            float[]? postFeedForwardNorm = null,
            bool withQkNormTensors = true)
        {
            List<TensorDef> tensors =
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

            if (withPostNorms)
            {
                tensors.Add(new("blk.0.post_attention_norm.weight", [Hidden], postAttentionNorm ?? PostAttentionNorm));
                tensors.Add(new("blk.0.post_ffw_norm.weight", [Hidden], postFeedForwardNorm ?? PostFeedForwardNorm));
            }

            if (withQkNormTensors)
            {
                tensors.Add(new("blk.0.attn_q_norm.weight", [HeadDim], QkNorm));
                tensors.Add(new("blk.0.attn_k_norm.weight", [HeadDim], KNorm));
            }

            return tensors;
        }

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

        public readonly float[] PostAttentionNorm = [1.20f, 0.80f, 1.10f, 0.90f, 1.05f, 0.95f];
        public readonly float[] PostFeedForwardNorm = [0.85f, 1.15f, 0.92f, 1.08f, 1.02f, 1.18f];

        public readonly float[] QkNorm = [1.30f, 0.80f, 1.10f, 0.90f];
        public readonly float[] KNorm = [0.85f, 1.20f, 0.95f, 1.05f];
    }

    // A scalar reference forward over the spec's hand-picked tensors: LayerNorm, standard full-head RoPE with adjacent
    // pairs, causal or bidirectional attention with an explicit softmax, gelu_tanh(gate) * up, and — when the placement
    // puts norms on sublayer outputs — a LayerNorm over the attention output after its projection and over the
    // feed-forward output after the down projection, each before its residual add. Computed in double and compared
    // against the engine's float logits.
    private sealed class PostNormReference
    {
        private readonly PostNormModelSpec _spec;

        public PostNormReference(PostNormModelSpec spec) => _spec = spec;

        /// <summary>Logits for a query token forwarded at position historyTokens.Length after replaying the sequence's earlier tokens through the same layer.</summary>
        public double[] LogitsAt(int[] historyTokens, int queryToken, bool postNorms)
        {
            double[][] kvK = new double[historyTokens.Length + 1][];
            double[][] kvV = new double[historyTokens.Length + 1][];

            for (int p = 0; p < historyTokens.Length; p++)
            {
                Advance(Row(_spec.Embeddings, historyTokens[p], PostNormModelSpec.Hidden), p, kvK, kvV, postNorms);
            }

            double[] cur = Row(_spec.Embeddings, queryToken, PostNormModelSpec.Hidden);
            Advance(cur, historyTokens.Length, kvK, kvV, postNorms);

            double[] normed = LayerNorm(cur, _spec.OutputNorm);
            return Gemv(_spec.Embeddings, PostNormModelSpec.Vocab, PostNormModelSpec.Hidden, normed); // tied lm_head
        }

        public double[][] Encoded(int[] tokens, bool postNorms)
        {
            int n = tokens.Length;
            double[][] cur = new double[n][];
            for (int i = 0; i < n; i++)
            {
                cur[i] = Row(_spec.Embeddings, tokens[i], PostNormModelSpec.Hidden);
            }

            double[][] kvK = new double[n][];
            double[][] kvV = new double[n][];

            // Phase 1: pre-attention norm, QKV, RoPE, K/V writes for every position.
            double[][] qAll = new double[n][];
            for (int p = 0; p < n; p++)
            {
                double[] normed = LayerNorm(cur[p], _spec.AttnNorm);
                double[] q = Gemv(_spec.AttnQ, PostNormModelSpec.Heads * PostNormModelSpec.HeadDim, PostNormModelSpec.Hidden, normed);
                double[] k = Gemv(_spec.AttnK, PostNormModelSpec.KvHeads * PostNormModelSpec.HeadDim, PostNormModelSpec.Hidden, normed);
                double[] v = Gemv(_spec.AttnV, PostNormModelSpec.KvHeads * PostNormModelSpec.HeadDim, PostNormModelSpec.Hidden, normed);

                Rope(q, p);
                Rope(k, p);

                qAll[p] = q;
                kvK[p] = k;
                kvV[p] = v;
            }

            // Phase 2: every query attends over the whole finished KV span.
            double[] attnOut = new double[n * PostNormModelSpec.Heads * PostNormModelSpec.HeadDim];
            for (int p = 0; p < n; p++)
            {
                Attention(qAll[p], kvK, kvV, 0, n - 1, attnOut.AsSpan(p * PostNormModelSpec.Heads * PostNormModelSpec.HeadDim));
            }

            // Phase 3: output projection, the post-attention norm over that output, and the ffn_inp residual add.
            for (int p = 0; p < n; p++)
            {
                double[] proj = Gemv(_spec.AttnO, PostNormModelSpec.Hidden, PostNormModelSpec.Heads * PostNormModelSpec.HeadDim, attnOut.AsSpan(p * PostNormModelSpec.Heads * PostNormModelSpec.HeadDim).ToArray());
                if (postNorms)
                {
                    proj = LayerNorm(proj, _spec.PostAttentionNorm);
                }

                for (int i = 0; i < PostNormModelSpec.Hidden; i++)
                {
                    cur[p][i] += proj[i];
                }
            }

            // Phases 4 and 5: gate/up GEMVs, gelu_tanh gating, down GEMV, the post-feed-forward norm over that output,
            // and the ffn_out residual add.
            for (int p = 0; p < n; p++)
            {
                double[] normed = LayerNorm(cur[p], _spec.FfnNorm);
                double[] gate = Gemv(_spec.FfnGate, PostNormModelSpec.Intermediate, PostNormModelSpec.Hidden, normed);
                double[] up = Gemv(_spec.FfnUp, PostNormModelSpec.Intermediate, PostNormModelSpec.Hidden, normed);
                for (int i = 0; i < PostNormModelSpec.Intermediate; i++)
                {
                    gate[i] = GeluTanh(gate[i]) * up[i];
                }

                double[] proj = Gemv(_spec.FfnDown, PostNormModelSpec.Hidden, PostNormModelSpec.Intermediate, gate);
                if (postNorms)
                {
                    proj = LayerNorm(proj, _spec.PostFeedForwardNorm);
                }

                for (int i = 0; i < PostNormModelSpec.Hidden; i++)
                {
                    cur[p][i] += proj[i];
                }
            }

            double[][] encoded = new double[n][];
            for (int i = 0; i < n; i++)
            {
                encoded[i] = LayerNorm(cur[i], _spec.OutputNorm);
            }

            return encoded;
        }

        // One causal decode layer: norm, QKV + RoPE, KV write at the position, causal attention, o-projection, the
        // post-attention norm, residual, FFN norm, gate/up, gelu_tanh gating, down, the post-feed-forward norm, residual.
        private void Advance(double[] cur, int position, double[][] kvK, double[][] kvV, bool postNorms)
        {
            double[] normed = LayerNorm(cur, _spec.AttnNorm);
            double[] q = Gemv(_spec.AttnQ, PostNormModelSpec.Heads * PostNormModelSpec.HeadDim, PostNormModelSpec.Hidden, normed);
            double[] k = Gemv(_spec.AttnK, PostNormModelSpec.KvHeads * PostNormModelSpec.HeadDim, PostNormModelSpec.Hidden, normed);
            double[] v = Gemv(_spec.AttnV, PostNormModelSpec.KvHeads * PostNormModelSpec.HeadDim, PostNormModelSpec.Hidden, normed);

            Rope(q, position);
            Rope(k, position);

            kvK[position] = k;
            kvV[position] = v;

            double[] attnOut = new double[PostNormModelSpec.Heads * PostNormModelSpec.HeadDim];
            Attention(q, kvK, kvV, 0, position, attnOut);

            double[] proj = Gemv(_spec.AttnO, PostNormModelSpec.Hidden, PostNormModelSpec.Heads * PostNormModelSpec.HeadDim, attnOut);
            if (postNorms)
            {
                proj = LayerNorm(proj, _spec.PostAttentionNorm);
            }

            for (int i = 0; i < PostNormModelSpec.Hidden; i++)
            {
                cur[i] += proj[i];
            }

            double[] ffnNormed = LayerNorm(cur, _spec.FfnNorm);
            double[] gate = Gemv(_spec.FfnGate, PostNormModelSpec.Intermediate, PostNormModelSpec.Hidden, ffnNormed);
            double[] up = Gemv(_spec.FfnUp, PostNormModelSpec.Intermediate, PostNormModelSpec.Hidden, ffnNormed);
            for (int i = 0; i < PostNormModelSpec.Intermediate; i++)
            {
                gate[i] = GeluTanh(gate[i]) * up[i];
            }

            double[] down = Gemv(_spec.FfnDown, PostNormModelSpec.Hidden, PostNormModelSpec.Intermediate, gate);
            if (postNorms)
            {
                down = LayerNorm(down, _spec.PostFeedForwardNorm);
            }

            for (int i = 0; i < PostNormModelSpec.Hidden; i++)
            {
                cur[i] += down[i];
            }
        }

        private static void Attention(double[] q, double[][] kvK, double[][] kvV, int from, int to, Span<double> outHead)
        {
            const int headDim = PostNormModelSpec.HeadDim;
            double scale = 1.0 / Math.Sqrt(headDim);

            // heads 2, kv-heads 1: every head reads kv-head (h * 1 / 2) == 0.
            for (int h = 0; h < PostNormModelSpec.Heads; h++)
            {
                double[] scores = new double[to - from + 1];
                double max = double.NegativeInfinity;
                for (int t = from; t <= to; t++)
                {
                    double dot = 0.0;
                    for (int j = 0; j < headDim; j++)
                    {
                        dot += q[h * headDim + j] * kvK[t][j];
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
                        acc += scores[t - from] / sum * kvV[t][j];
                    }

                    outHead[h * headDim + j] = acc;
                }
            }
        }

        // Standard full-head RoPE with adjacent pairs (2i, 2i + 1): the pairing the llama-shaped profile selects.
        private static void Rope(double[] headRegion, int position)
        {
            for (int h = 0; h < headRegion.Length / PostNormModelSpec.HeadDim; h++)
            {
                for (int i = 0; i < PostNormModelSpec.HeadDim / 2; i++)
                {
                    double invFreq = Math.Pow(PostNormModelSpec.RopeTheta, -2.0 * i / PostNormModelSpec.HeadDim);
                    double angle = position * invFreq;
                    double c = Math.Cos(angle);
                    double s = Math.Sin(angle);

                    double lo = headRegion[h * PostNormModelSpec.HeadDim + 2 * i];
                    double hi = headRegion[h * PostNormModelSpec.HeadDim + 2 * i + 1];
                    headRegion[h * PostNormModelSpec.HeadDim + 2 * i] = lo * c - hi * s;
                    headRegion[h * PostNormModelSpec.HeadDim + 2 * i + 1] = hi * c + lo * s;
                }
            }
        }

        private static double[] LayerNorm(double[] x, float[] weight)
        {
            int n = x.Length;
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
            double invStd = 1.0 / Math.Sqrt(var + PostNormModelSpec.Eps);

            double[] result = new double[n];
            for (int i = 0; i < n; i++)
            {
                result[i] = (x[i] - mean) * invStd * weight[i];
            }

            return result;
        }

        private static double GeluTanh(double x) =>
            0.5 * x * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));

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
}
