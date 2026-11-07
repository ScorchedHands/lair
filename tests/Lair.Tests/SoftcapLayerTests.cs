using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Forward-level goldens for the two logit caps: the scalar softcap against a double-precision
/// <c>cap * tanh(x / cap)</c> reference across small, exact-<c>±cap</c>, and saturating magnitudes; a model
/// declaring the attention-score softcap against a reference forward that caps every score after scaling and
/// before the softmax, with the capped and uncapped logits demonstrably different where attention is
/// non-degenerate; a model declaring the final-logit softcap against a reference forward that caps the full
/// logit vector, with the capped logits provably inside the cap; batched and single-token decode
/// bit-identical under both caps; and the tied token-embedding output path honoring the final-logit cap.
/// Engine-to-engine comparisons are asserted bit-for-bit; reference comparisons use the double-precision
/// tolerance shared by the strategy goldens.
/// </summary>
[Trait("Category", "Layers")]
public sealed class SoftcapLayerTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

    // Caps the synthetic models declare: the q/k projections put scores at the unit scale and the attention
    // cap at a quarter of it, so capping bends the softmax weights by a detectable amount; the final cap sits
    // below the fixture's logit magnitudes so at least one logit saturates.
    private const float AttentionCap = 0.25f;
    private const float FinalCap = 0.3f;

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

    // ---------------------------------------------------------------- scalar golden

    [Theory]
    // Small magnitudes: the cap is the identity to float precision.
    [InlineData(30f, 0f)]
    [InlineData(30f, 1e-6f)]
    [InlineData(30f, -1e-6f)]
    [InlineData(50f, 1e-4f)]
    // Mid magnitudes: the curvature is visible but far from saturation.
    [InlineData(30f, 1f)]
    [InlineData(30f, -1f)]
    [InlineData(30f, 10f)]
    [InlineData(30f, -10f)]
    [InlineData(1f, 0.5f)]
    [InlineData(1f, -0.5f)]
    // Exact ±cap: cap * tanh(±1), strictly inside the cap.
    [InlineData(30f, 30f)]
    [InlineData(30f, -30f)]
    [InlineData(50f, 50f)]
    [InlineData(50f, -50f)]
    // Large magnitudes: saturation toward ±cap is visible.
    [InlineData(30f, 100f)]
    [InlineData(30f, -100f)]
    [InlineData(30f, 1000f)]
    [InlineData(30f, -1000f)]
    [InlineData(50f, 1000f)]
    [InlineData(50f, -1000f)]
    [InlineData(1f, 1000f)]
    public void Softcap_MatchesDoubleReference_AcrossSmallExactCapAndSaturatingMagnitudes(float cap, float x)
    {
        double expected = (double)cap * Math.Tanh((double)x / (double)cap);
        float actual = LayerOps.Softcap(x, cap);

        Assert.True(
            Math.Abs(actual - expected) <= 1e-6 * Math.Max(1.0, Math.Abs(expected)),
            $"softcap({x:G9}, cap {cap:G9}): expected {expected:G9}, got {actual:G9}");
        Assert.True(Math.Abs(actual) <= cap, $"softcap({x:G9}, cap {cap:G9}) escaped the cap: {actual:G9}");
    }

    [Fact]
    public void Softcap_IsIdentityForSmallInputs_AndSaturatesExactlyForHugeOnes()
    {
        // Small inputs: the cubic correction is far below float resolution, so the value passes through unchanged.
        Assert.Equal(1e-6f, LayerOps.Softcap(1e-6f, 30f));
        Assert.Equal(-1e-6f, LayerOps.Softcap(-1e-6f, 30f));

        // Huge inputs: tanh saturates to exactly 1 in double, so the result is exactly ±cap.
        Assert.Equal(30f, LayerOps.Softcap(1000f, 30f));
        Assert.Equal(-30f, LayerOps.Softcap(-1000f, 30f));

        // Exact ±cap lands strictly inside the cap: cap * tanh(1) ≈ 0.7616 * cap.
        float atCap = LayerOps.Softcap(30f, 30f);
        Assert.True(atCap > 0f && atCap < 30f, $"softcap(30, 30) = {atCap:G9} must lie strictly inside (0, 30)");
        Assert.Equal(-atCap, LayerOps.Softcap(-30f, 30f));
    }

    // ---------------------------------------------------------------- attention-score softcap

    [Fact]
    public void Forward_AttentionSoftcap_MatchesCappedReference_DiffersFromUncapped_AndBatchMatchesSingle()
    {
        SoftcapModelSpec spec = new();
        string path = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.attn_logit_softcapping", AttentionCap)],
            spec.Tensors()));

        SoftcapReference reference = new(spec);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        Assert.Equal(AttentionCap, model.Plan.AttentionSoftcap);
        Assert.Equal(0f, model.Plan.FinalLogitSoftcap); // only the attention cap is declared

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        Assert.True(
            Closest(model.DecodeNext(1, kv), reference.LogitsAt([], 1, attentionCap: AttentionCap)),
            "capped decode logits at position 0 must match the capped reference forward");

        float[] cappedDecode1 = model.DecodeNext(4, kv).ToArray();
        Assert.True(
            Closest(cappedDecode1, reference.LogitsAt([1], 4, attentionCap: AttentionCap)),
            "capped decode logits at position 1 must match the capped reference forward");

        // The cap genuinely reshapes the attention: at position 1 the attention is non-degenerate, and the
        // uncapped reference gives demonstrably different logits.
        Assert.False(
            Closest(cappedDecode1, reference.LogitsAt([1], 4, attentionCap: 0f)),
            "attention softcap on must not leave the logits equal to the uncapped reference");

        // Encode: bidirectional attention caps its scores too — every position against the capped reference.
        using SequenceKv encodeKv = new(pool);
        float[] encoded = model.Encode([1, 4, 2], encodeKv);

        Assert.Equal(3 * model.Plan.HiddenSize, encoded.Length);
        double[][] expectedEncoded = reference.Encoded([1, 4, 2], attentionCap: AttentionCap);
        for (int p = 0; p < 3; p++)
        {
            for (int i = 0; i < model.Plan.HiddenSize; i++)
            {
                float actual = encoded[p * model.Plan.HiddenSize + i];
                double expected = expectedEncoded[p][i];
                Assert.True(
                    Math.Abs(actual - expected) <= 1e-5 * Math.Max(1.0, Math.Abs(expected)),
                    $"encode hidden[{p}][{i}] with attention softcap: expected {expected:G9}, got {actual:G9}");
            }
        }

        // Batched decode must be bit-identical to single-token decode under the cap.
        using SequenceKv single = new(pool);
        model.Prefill([1, 4], single);
        float[] singleLogits = model.LastLogits().ToArray();

        using SequenceKv batched = new(pool);
        model.DecodeBatch([1], [batched]);
        Span<float> batchLogits = model.DecodeBatch([4], [batched]);
        Assert.True(
            batchLogits.SequenceEqual(singleLogits.AsSpan()),
            "a token batched through DecodeBatch under the attention softcap must be bit-identical to stepping the same sequence alone");
    }

    // ---------------------------------------------------------------- final-logit softcap

    [Fact]
    public void Forward_FinalLogitSoftcap_MatchesCappedReference_BoundsLogits_AndDiffersFromUncapped()
    {
        SoftcapModelSpec spec = new();
        string path = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.final_logit_softcapping", FinalCap)],
            spec.Tensors()));

        SoftcapReference reference = new(spec);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        Assert.Equal(0f, model.Plan.AttentionSoftcap); // only the final-logit cap is declared
        Assert.Equal(FinalCap, model.Plan.FinalLogitSoftcap);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        Assert.True(
            Closest(model.DecodeNext(1, kv), reference.LogitsAt([], 1, finalCap: FinalCap)),
            "capped decode logits at position 0 must match the capped reference forward");

        float[] cappedDecode1 = model.DecodeNext(4, kv).ToArray();
        Assert.True(
            Closest(cappedDecode1, reference.LogitsAt([1], 4, finalCap: FinalCap)),
            "capped decode logits at position 1 must match the capped reference forward");

        // The cap genuinely reshapes the logits: the uncapped reference gives demonstrably different values —
        // the tanh curvature alone moves them by far more than the comparison tolerance.
        Assert.False(
            Closest(cappedDecode1, reference.LogitsAt([1], 4, finalCap: 0f)),
            "final-logit softcap on must not leave the logits equal to the uncapped reference");

        // Every capped logit provably sits inside the cap: the reference forward saturates at least one logit
        // past the cap, and the engine's cap holds the line.
        double[] uncapped = reference.LogitsAt([1], 4, finalCap: 0f);
        Assert.True(
            Array.Exists(uncapped, value => Math.Abs(value) > FinalCap),
            "the uncapped reference must saturate at least one logit past the cap for the bound to be meaningful");
        foreach (float logit in cappedDecode1)
        {
            Assert.True(
                Math.Abs(logit) <= FinalCap,
                $"capped logit {logit:G9} escaped the cap of {FinalCap:G9}");
        }

        // Batched decode must be bit-identical to single-token decode under the cap.
        using SequenceKv single = new(pool);
        model.Prefill([1, 4], single);
        float[] singleLogits = model.LastLogits().ToArray();

        using SequenceKv batched = new(pool);
        model.DecodeBatch([1], [batched]);
        Span<float> batchLogits = model.DecodeBatch([4], [batched]);
        Assert.True(
            batchLogits.SequenceEqual(singleLogits.AsSpan()),
            "a token batched through DecodeBatch under the final-logit softcap must be bit-identical to stepping the same sequence alone");
    }

    [Fact]
    public void Forward_TiedEmbeddingHead_HonorsFinalLogitSoftcap()
    {
        // The fixture has no output.weight: every logit here comes from the tied token-embedding rows, so the
        // capped engine forward can only match a reference that caps the tied product.
        SoftcapModelSpec spec = new();
        string path = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.final_logit_softcapping", FinalCap)],
            spec.Tensors()));

        SoftcapReference reference = new(spec);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        Assert.False(model.Plan.HasOutputWeights); // tied path confirmed
        Assert.Equal(FinalCap, model.Plan.FinalLogitSoftcap);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        float[] logits = model.DecodeNext(1, kv).ToArray();
        Assert.True(
            Closest(logits, reference.LogitsAt([], 1, finalCap: FinalCap)),
            "tied-head logits must match the capped tied reference forward");
        foreach (float logit in logits)
        {
            Assert.True(Math.Abs(logit) <= FinalCap, $"tied capped logit {logit:G9} escaped the cap of {FinalCap:G9}");
        }
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

    // A complete single-layer model with hidden 6, heads 2, kv-heads 1, a declared head dim of 4, ffn 8, vocab 6
    // — all F32 tensors with hand-picked payloads the reference forward reads directly. No output.weight: the
    // logits come from the tied token-embedding rows.
    private sealed class SoftcapModelSpec
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

        public readonly float[] Embeddings =
        [
            0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f,
            0.11f, -0.21f, 0.31f, -0.41f, 0.51f, -0.61f,
            0.32f, 0.22f, -0.32f, 0.42f, -0.52f, 0.62f,
            -0.13f, 0.23f, 0.33f, -0.43f, 0.53f, 0.63f,
            0.94f, -0.84f, 0.74f, -0.64f, 0.54f, -0.44f,
            0.15f, 0.25f, 0.35f, 0.45f, 0.55f, -0.65f,
        ];

        // Scaled-up q/k rows put the attention scores at the unit scale, where a cap bends the softmax
        // weights by a detectable amount.
        public readonly float[] AttnQ =
        [
            0.50f, 1.00f, 1.50f, 2.00f, 2.50f, 3.00f,
            0.60f, 1.10f, 1.60f, 2.10f, 2.60f, 3.10f,
            0.70f, 1.20f, 1.70f, 2.20f, 2.70f, 3.20f,
            0.80f, 1.30f, 1.80f, 2.30f, 2.80f, 3.30f,
            0.90f, 1.40f, 1.90f, 2.40f, 2.90f, 3.40f,
            1.00f, 1.50f, 2.00f, 2.50f, 3.00f, 3.50f,
            1.10f, 1.60f, 2.10f, 2.60f, 3.10f, 3.60f,
            1.20f, 1.70f, 2.20f, 2.70f, 3.20f, 3.70f,
        ];

        public readonly float[] AttnK =
        [
            0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f,
            0.20f, 0.30f, 0.40f, 0.50f, 0.60f, 0.70f,
            0.30f, 0.40f, 0.50f, 0.60f, 0.70f, 0.80f,
            0.40f, 0.50f, 0.60f, 0.70f, 0.80f, 0.90f,
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
    }

    // A scalar reference forward over the spec's hand-picked tensors: RMSNorm, standard full-head RoPE, causal
    // or bidirectional attention with an explicit softmax — with the attention-score softcap applied to every
    // score after scaling and before the softmax when a cap is given — silu(gate) * up, and the tied
    // token-embedding logits, capped by the final-logit softcap when a cap is given. Computed in double and
    // compared against the engine's float logits with the shared tolerance.
    private sealed class SoftcapReference
    {
        private readonly SoftcapModelSpec _spec;

        public SoftcapReference(SoftcapModelSpec spec) => _spec = spec;

        /// <summary>Logits for a query token forwarded at position historyTokens.Length after replaying the sequence's earlier tokens through the same layer.</summary>
        public double[] LogitsAt(int[] historyTokens, int queryToken, float attentionCap = 0f, float finalCap = 0f)
        {
            double[][] kvK = new double[historyTokens.Length + 1][];
            double[][] kvV = new double[historyTokens.Length + 1][];

            for (int p = 0; p < historyTokens.Length; p++)
            {
                Advance(Row(_spec.Embeddings, historyTokens[p], SoftcapModelSpec.Hidden), p, kvK, kvV, attentionCap);
            }

            double[] cur = Row(_spec.Embeddings, queryToken, SoftcapModelSpec.Hidden);
            Advance(cur, historyTokens.Length, kvK, kvV, attentionCap);

            double[] normed = Norm(cur, _spec.OutputNorm);
            double[] logits = Gemv(_spec.Embeddings, SoftcapModelSpec.Vocab, SoftcapModelSpec.Hidden, normed); // tied lm_head
            if (finalCap > 0f)
            {
                for (int i = 0; i < logits.Length; i++)
                {
                    logits[i] = (double)finalCap * Math.Tanh(logits[i] / (double)finalCap);
                }
            }

            return logits;
        }

        public double[][] Encoded(int[] tokens, float attentionCap = 0f)
        {
            int n = tokens.Length;
            double[][] cur = new double[n][];
            for (int i = 0; i < n; i++)
            {
                cur[i] = Row(_spec.Embeddings, tokens[i], SoftcapModelSpec.Hidden);
            }

            double[][] kvK = new double[n][];
            double[][] kvV = new double[n][];

            double[][] qAll = new double[n][];
            for (int p = 0; p < n; p++)
            {
                double[] normed = Norm(cur[p], _spec.AttnNorm);
                double[] q = Gemv(_spec.AttnQ, SoftcapModelSpec.Heads * SoftcapModelSpec.HeadDim, SoftcapModelSpec.Hidden, normed);
                double[] k = Gemv(_spec.AttnK, SoftcapModelSpec.KvHeads * SoftcapModelSpec.HeadDim, SoftcapModelSpec.Hidden, normed);
                double[] v = Gemv(_spec.AttnV, SoftcapModelSpec.KvHeads * SoftcapModelSpec.HeadDim, SoftcapModelSpec.Hidden, normed);

                Rope(q, p);
                Rope(k, p);

                qAll[p] = q;
                kvK[p] = k;
                kvV[p] = v;
            }

            double[] attnOut = new double[n * SoftcapModelSpec.Heads * SoftcapModelSpec.HeadDim];
            for (int p = 0; p < n; p++)
            {
                Attention(qAll[p], kvK, kvV, 0, n - 1, attentionCap, attnOut.AsSpan(p * SoftcapModelSpec.Heads * SoftcapModelSpec.HeadDim));
            }

            for (int p = 0; p < n; p++)
            {
                double[] proj = Gemv(_spec.AttnO, SoftcapModelSpec.Hidden, SoftcapModelSpec.Heads * SoftcapModelSpec.HeadDim, attnOut.AsSpan(p * SoftcapModelSpec.Heads * SoftcapModelSpec.HeadDim).ToArray());
                for (int i = 0; i < SoftcapModelSpec.Hidden; i++)
                {
                    cur[p][i] += proj[i];
                }
            }

            for (int p = 0; p < n; p++)
            {
                double[] normed = Norm(cur[p], _spec.FfnNorm);
                double[] gate = Gemv(_spec.FfnGate, SoftcapModelSpec.Intermediate, SoftcapModelSpec.Hidden, normed);
                double[] up = Gemv(_spec.FfnUp, SoftcapModelSpec.Intermediate, SoftcapModelSpec.Hidden, normed);
                for (int i = 0; i < SoftcapModelSpec.Intermediate; i++)
                {
                    gate[i] = Silu(gate[i]) * up[i];
                }

                double[] proj = Gemv(_spec.FfnDown, SoftcapModelSpec.Hidden, SoftcapModelSpec.Intermediate, gate);
                for (int i = 0; i < SoftcapModelSpec.Hidden; i++)
                {
                    cur[p][i] += proj[i];
                }
            }

            double[][] encoded = new double[n][];
            for (int i = 0; i < n; i++)
            {
                encoded[i] = Norm(cur[i], _spec.OutputNorm);
            }

            return encoded;
        }

        private void Advance(double[] cur, int position, double[][] kvK, double[][] kvV, float attentionCap)
        {
            double[] normed = Norm(cur, _spec.AttnNorm);
            double[] q = Gemv(_spec.AttnQ, SoftcapModelSpec.Heads * SoftcapModelSpec.HeadDim, SoftcapModelSpec.Hidden, normed);
            double[] k = Gemv(_spec.AttnK, SoftcapModelSpec.KvHeads * SoftcapModelSpec.HeadDim, SoftcapModelSpec.Hidden, normed);
            double[] v = Gemv(_spec.AttnV, SoftcapModelSpec.KvHeads * SoftcapModelSpec.HeadDim, SoftcapModelSpec.Hidden, normed);

            Rope(q, position);
            Rope(k, position);

            kvK[position] = k;
            kvV[position] = v;

            double[] attnOut = new double[SoftcapModelSpec.Heads * SoftcapModelSpec.HeadDim];
            Attention(q, kvK, kvV, 0, position, attentionCap, attnOut);

            double[] proj = Gemv(_spec.AttnO, SoftcapModelSpec.Hidden, SoftcapModelSpec.Heads * SoftcapModelSpec.HeadDim, attnOut);
            for (int i = 0; i < SoftcapModelSpec.Hidden; i++)
            {
                cur[i] += proj[i];
            }

            double[] ffnNormed = Norm(cur, _spec.FfnNorm);
            double[] gate = Gemv(_spec.FfnGate, SoftcapModelSpec.Intermediate, SoftcapModelSpec.Hidden, ffnNormed);
            double[] up = Gemv(_spec.FfnUp, SoftcapModelSpec.Intermediate, SoftcapModelSpec.Hidden, ffnNormed);
            for (int i = 0; i < SoftcapModelSpec.Intermediate; i++)
            {
                gate[i] = Silu(gate[i]) * up[i];
            }

            double[] down = Gemv(_spec.FfnDown, SoftcapModelSpec.Hidden, SoftcapModelSpec.Intermediate, gate);
            for (int i = 0; i < SoftcapModelSpec.Hidden; i++)
            {
                cur[i] += down[i];
            }
        }

        private static void Attention(double[] q, double[][] kvK, double[][] kvV, int from, int to, float attentionCap, Span<double> outHead)
        {
            const int headDim = SoftcapModelSpec.HeadDim;
            double scale = 1.0 / Math.Sqrt(headDim);

            // heads 2, kv-heads 1: every head reads kv-head (h * 1 / 2) == 0.
            for (int h = 0; h < SoftcapModelSpec.Heads; h++)
            {
                int kvHead = h * SoftcapModelSpec.KvHeads / SoftcapModelSpec.Heads;

                double[] scores = new double[to - from + 1];
                double max = double.NegativeInfinity;
                for (int t = from; t <= to; t++)
                {
                    double dot = 0.0;
                    for (int j = 0; j < headDim; j++)
                    {
                        dot += q[h * headDim + j] * kvK[t][kvHead * headDim + j];
                    }

                    double score = dot * scale;
                    if (attentionCap > 0f)
                    {
                        score = (double)attentionCap * Math.Tanh(score / (double)attentionCap); // after scaling, before the softmax
                    }

                    scores[t - from] = score;
                    max = Math.Max(max, score);
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
            for (int h = 0; h < headRegion.Length / SoftcapModelSpec.HeadDim; h++)
            {
                for (int i = 0; i < SoftcapModelSpec.HeadDim / 2; i++)
                {
                    double invFreq = Math.Pow(SoftcapModelSpec.RopeTheta, -2.0 * i / SoftcapModelSpec.HeadDim);
                    double angle = position * invFreq;
                    double c = Math.Cos(angle);
                    double s = Math.Sin(angle);

                    double lo = headRegion[h * SoftcapModelSpec.HeadDim + 2 * i];
                    double hi = headRegion[h * SoftcapModelSpec.HeadDim + 2 * i + 1];
                    headRegion[h * SoftcapModelSpec.HeadDim + 2 * i] = lo * c - hi * s;
                    headRegion[h * SoftcapModelSpec.HeadDim + 2 * i + 1] = hi * c + lo * s;
                }
            }
        }

        private static double[] Norm(double[] x, float[] weight)
        {
            int n = x.Length;
            double sumSq = 0.0;
            for (int i = 0; i < n; i++)
            {
                sumSq += x[i] * x[i];
            }

            double invRms = 1.0 / Math.Sqrt(sumSq / n + SoftcapModelSpec.Eps);
            double[] result = new double[n];
            for (int i = 0; i < n; i++)
            {
                result[i] = x[i] * invRms * weight[i];
            }

            return result;
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
                case double d:
                    writer.Write(12u); // FLOAT64
                    writer.Write(d);
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
        string path = Path.Combine(Path.GetTempPath(), $"lair-softcap-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }
}
