using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Forward-level goldens for the gemma2 shape: a synthetic gemma2 file whose head dimension is declared by
/// <c>head_length</c> independently of hidden and head_count (2 x 3 != 4), with grouped query attention, both
/// logit caps declared, and the whole block carried through the reference graph's norm layout — every norm an
/// RMS norm, each sublayer output normalized with its own vector before its residual add, the token-embedding
/// row scaled by sqrt(hidden) before the first layer, and the feed-forward gated by the exact GELU. The goldens
/// match a double-precision scalar reference across decode, batched decode, and encode, and each feature of the
/// graph is pinned by a demonstrably different counterfactual reference: no embedding gain, unweighted per-head
/// QK-norm, the tanh-approximated gate, pre-only placement, and an uncapped attention. Engine-to-engine
/// comparisons are bit-for-bit; reference comparisons use the double-precision tolerance shared by the strategy
/// goldens.
/// </summary>
[Trait("Category", "Layers")]
public sealed class Gemma2LayerTests : IDisposable
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

    [Fact]
    public void Forward_Gemma2Profile_MatchesReferenceGraph_InDecodeBatchAndEncode()
    {
        Gemma2ModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors()));

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        // Nothing declares the features: the gemma2 profile resolved the whole graph — RMS norms, the exact-GELU
        // gate, rotate-half rotary rows, post-sublayer norms, the embedding gain, and both caps from the file's
        // declarations — with QK-norm and attention biases left off.
        Assert.Equal(NormStrategy.RmsNorm, model.Plan.Norm);
        Assert.Equal(ActivationStrategy.GeGluExact, model.Plan.Activation);
        Assert.Equal(RopeStrategy.RotateHalfFullHead, model.Plan.Rope);
        Assert.Equal(NormPlacement.PostSublayerOutputs, model.Plan.NormPlacement);
        Assert.Equal(QkNormMode.Off, model.Plan.QkNorm);
        Assert.False(model.Plan.UseAttentionBias);
        Assert.True(
            Math.Abs(model.Plan.InputEmbeddingScale - MathF.Sqrt(Gemma2ModelSpec.Hidden)) <= 1e-6f,
            $"input embedding scale {model.Plan.InputEmbeddingScale:G9} must be sqrt(hidden)");
        Assert.Equal(Gemma2ModelSpec.HeadDim, model.Plan.HeadDim); // declared by head_length (2 x 3 != 4)
        Assert.False(model.Plan.HasOutputWeights); // tied token-embedding head

        Gemma2Reference reference = new(spec);

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
    public void Forward_Gemma2GraphFeatures_AreEachDemonstrablyLoad_BearingAgainstCounterfactualReferences()
    {
        Gemma2ModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors()));

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
        Gemma2Reference reference = new(spec);

        // The embedding gain is load-bearing: the same tensors forwarded without the sqrt(hidden) scaling give
        // a demonstrably different forward. The comparison runs on the encoder hidden states — the block's RMS
        // norms make the gain invisible at the logits, but it survives to the output norm untouched.
        using (SequenceKv encodeKv = new(pool))
        {
            float[] encoded = model.Encode([1, 4, 2], encodeKv);

            Assert.False(
                ClosestAny(encoded, Flatten(reference.Encoded([1, 4, 2], embedScale: false))),
                "the gemma2 profile must scale each embedding row by sqrt(hidden); the unscaled reference differs");

            Assert.False(
                ClosestAny(encoded, Flatten(reference.Encoded([1, 4, 2], qkNorm: true))),
                "gemma2 normalizes no q or k head; the QK-normed reference must differ");

            Assert.False(
                ClosestAny(encoded, Flatten(reference.Encoded([1, 4, 2], postNorms: false))),
                "gemma2 normalizes each sublayer output before its residual add; the pre-only reference must differ");

            Assert.False(
                ClosestAny(encoded, Flatten(reference.Encoded([1, 4, 2], attentionCap: 0f))),
                "the declared attention-score cap must not leave the hidden states equal to the uncapped reference");
        }

        // The gate is the exact GELU: the tanh approximation the other dense families use gives a different
        // forward. The comparison runs with the final cap lifted — the cap would clip the small gelu-versus-
        // tanh difference below the tolerance — and the engine's uncapped logits are compared against the
        // reference's uncapped ones.
        using (SequenceKv kv = new(pool))
        {
            model.DecodeNext(1, kv);
            float[] logitsAt1 = model.LastLogits().ToArray();

            Assert.False(
                ClosestAny(logitsAt1, reference.LogitsAt([1], 4, geluTanh: true, finalCap: 0f)),
                "gemma2 gates with gelu(x) * up; the gelu_tanh reference must differ");
        }

        // The final cap holds the line: the uncapped reference carries logits past the cap, and every engine
        // logit — produced under the declared cap — sits inside it.
        using (SequenceKv kv = new(pool))
        {
            model.DecodeNext(1, kv);
            float[] logitsAt1 = model.LastLogits().ToArray();

            double[] uncapped = reference.LogitsAt([1], 4, finalCap: 0f);
            Assert.True(
                Array.Exists(uncapped, value => Math.Abs(value) > Gemma2ModelSpec.FinalCap),
                "the fixture must drive at least one uncapped logit past the final cap for this check to mean anything");
            Assert.True(
                logitsAt1.All(value => Math.Abs(value) <= Gemma2ModelSpec.FinalCap),
                "every capped logit must sit inside the final-logit cap");
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

    // True when every value of a flat engine result sits within tolerance of the matching reference value.
    private static bool ClosestAny(ReadOnlySpan<float> actual, double[] expected)
    {
        if (actual.Length != expected.Length)
        {
            return false;
        }

        return Closest(actual, expected);
    }

    // Flatten a reference encoder result (position-major hidden states) into one flat array.
    private static double[] Flatten(double[][] rows)
    {
        double[] flat = new double[rows.Length * rows[0].Length];
        for (int p = 0; p < rows.Length; p++)
        {
            for (int i = 0; i < rows[p].Length; i++)
            {
                flat[p * rows[p].Length + i] = rows[p][i];
            }
        }

        return flat;
    }

    // A complete single-layer gemma2-shaped model: hidden 4, heads 2, kv-heads 1, head dim 4 declared by
    // head_length (2 x 4 = 8 q rows against a hidden of 4), ffn 8, vocab 6 — all F32 tensors with hand-picked payloads the reference forward
    // reads directly. Both caps are declared at magnitudes the fixture demonstrably exceeds, and there is no
    // output.weight: the logits come from the tied token-embedding rows.
    private sealed class Gemma2ModelSpec
    {
        public const int Hidden = 4;
        public const int Heads = 2;
        public const int KvHeads = 1;
        public const int HeadDim = 4;
        public const int Intermediate = 8;
        public const int Vocab = 6;
        public const float Eps = 1e-5f;
        public const double RopeTheta = 10_000.0;
        // The attention cap sits at the unit scale the fixture's scores reach, so it demonstrably bends the
        // softmax. The final cap sits below the fixture's logit magnitudes: it binds without saturating, so
        // even the small exact-GELU-versus-tanh difference survives it.
        public const float AttentionCap = 0.5f;
        public const float FinalCap = 0.05f;

        public List<(string Key, object Value)> MetadataPairs() =>
        [
            ("general.architecture", "gemma2"),
            ("gemma2.embedding_length", Hidden),
            ("gemma2.block_count", 1),
            ("gemma2.attention.head_count", Heads),
            ("gemma2.attention.head_count_kv", KvHeads),
            ("gemma2.feed_forward_length", Intermediate),
            ("gemma2.attention.head_length", HeadDim),
            ("gemma2.attention.key_length", HeadDim),
            ("gemma2.attention.value_length", HeadDim),
            ("gemma2.attention.layer_norm_rms_epsilon", Eps),
            ("gemma2.attn_logit_softcapping", AttentionCap),
            ("gemma2.final_logit_softcapping", FinalCap),
            ("gemma2.vocab_size", Vocab),
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
            new("blk.0.post_attention_norm.weight", [Hidden], PostAttnNorm),
            new("blk.0.post_ffw_norm.weight", [Hidden], PostFfnNorm),
            new("output_norm.weight", [Hidden], OutputNorm),
        ];

        // Rows with distinct directions and magnitudes: the K vectors then differ per position, so the
        // attention weights — and every op downstream of them — move by a detectable amount. The embedding
        // rows carry unit RMS so the attention norm contributes no gain and the FFN gate magnitudes stay
        // predictable through the block.
        public readonly float[] Embeddings =
        [
            0.50f, 0.50f, 0.50f, 0.50f,
            0.50f, -0.50f, 0.50f, -0.50f,
            0.50f, 0.50f, -0.50f, 0.50f,
            -0.50f, 0.50f, 0.50f, -0.50f,
            0.50f, -0.50f, -0.50f, 0.50f,
            0.50f, 0.50f, 0.50f, -0.50f,
        ];

        public readonly float[] AttnQ =
        [
            0.05f, 0.10f, 0.15f, 0.20f,
            0.06f, 0.11f, 0.16f, 0.21f,
            0.07f, 0.12f, 0.17f, 0.22f,
            0.08f, 0.13f, 0.18f, 0.23f,
            0.09f, 0.14f, 0.19f, 0.24f,
            0.10f, 0.15f, 0.20f, 0.25f,
            0.11f, 0.16f, 0.21f, 0.26f,
            0.12f, 0.17f, 0.22f, 0.27f,
        ];

        public readonly float[] AttnK =
        [
            0.01f, 0.02f, 0.03f, 0.04f,
            0.02f, 0.03f, 0.04f, 0.05f,
            0.03f, 0.04f, 0.05f, 0.06f,
            0.04f, 0.05f, 0.06f, 0.07f,
        ];

        // Distinct V rows per position: attention weight changes then move the output by a detectable amount.
        public readonly float[] AttnV =
        [
            0.09f, 0.08f, 0.07f, 0.06f,
            -0.45f, 0.38f, -0.31f, 0.24f,
            0.30f, -0.22f, 0.14f, -0.06f,
            0.06f, 0.05f, 0.04f, 0.03f,
        ];

        public readonly float[] AttnO =
        [
            0.10f, 0.00f, 0.20f, 0.00f, 0.30f, 0.00f, 0.15f, 0.00f,
            0.00f, 0.30f, 0.00f, 0.40f, 0.00f, 0.50f, 0.00f, 0.25f,
            0.15f, 0.00f, 0.25f, 0.00f, 0.35f, 0.00f, 0.20f, 0.00f,
            0.00f, 0.35f, 0.00f, 0.45f, 0.00f, 0.55f, 0.00f, 0.30f,
        ];

        // Gate and up rows carry magnitudes around 1000: even after every RMS compression in the block the
        // post-norm gates land well past |1|, where the exact GELU and its tanh approximation differ by far
        // more than the comparison tolerance, and the down rows carry magnitudes around 0.1 so the gated
        // product survives the output RMS compression and actually moves the logits.
        public readonly float[] FfnGate =
        [
            1000.10f, 1000.11f, 1000.12f, 1000.13f,
            1000.16f, 1000.17f, 1000.18f, 1000.19f,
            1000.22f, 1000.23f, 1000.24f, 1000.25f,
            1000.28f, 1000.29f, 1000.30f, 1000.31f,
            -1000.34f, -1000.35f, -1000.36f, -1000.37f,
            0.40f, 0.41f, 0.42f, 0.43f,
            -1000.46f, -1000.47f, -1000.48f, -1000.49f,
            1000.52f, 1000.53f, 1000.54f, 1000.55f,
        ];

        public readonly float[] FfnUp =
        [
            100.11f, 100.12f, 100.13f, 100.14f,
            100.17f, 100.18f, 100.19f, 100.20f,
            100.23f, 100.24f, 100.25f, 100.26f,
            100.29f, 100.30f, 100.31f, 100.32f,
            -100.35f, 100.36f, -100.37f, 100.38f,
            100.41f, -100.42f, 100.43f, -100.44f,
            100.47f, 100.48f, 100.49f, 100.50f,
            100.53f, 100.54f, 100.55f, 100.56f,
        ];

        public readonly float[] FfnDown =
        [
            0.010f, 0.012f, 0.014f, 0.016f, 0.018f, 0.020f, 0.022f, 0.024f,
            0.011f, 0.013f, 0.015f, 0.017f, 0.019f, 0.021f, 0.023f, 0.025f,
            -0.012f, 0.014f, -0.016f, 0.018f, -0.020f, 0.022f, -0.024f, 0.026f,
            0.013f, -0.015f, 0.017f, -0.019f, 0.021f, -0.023f, 0.025f, -0.027f,
        ];

        public readonly float[] AttnNorm = [1.10f, 0.90f, 1.05f, 0.95f];
        public readonly float[] FfnNorm = [0.90f, 1.10f, 0.95f, 1.05f];
        public readonly float[] PostAttnNorm = [1.05f, 0.95f, 1.15f, 0.85f];
        public readonly float[] PostFfnNorm = [0.95f, 1.15f, 0.85f, 1.05f];
        public readonly float[] OutputNorm = [1.00f, 1.02f, 0.98f, 1.01f];
    }

    // A scalar reference forward over the spec's hand-picked tensors, mirroring the reference gemma2 graph:
    // the embedding row scaled by sqrt(hidden); every norm — pre-sublayer, post-sublayer, and output — an RMS
    // norm with the file's epsilon; no per-head QK-norm; rotate-half full-head RoPE; causal or bidirectional
    // attention with scores scaled by 1/sqrt(headDim), capped by cap * tanh(score / cap), and an explicit
    // softmax; each sublayer output normalized with its own vector before its residual add; gelu(gate) * up;
    // and the tied token-embedding logits capped by the final-logit softcap. Computed in double and compared
    // against the engine's float logits with the shared tolerance. The counterfactual switches (embedScale,
    // qkNorm, geluTanh, postNorms, caps) each describe a different graph, used only to prove the engine's
    // forward genuinely runs the gemma2 one.
    private sealed class Gemma2Reference
    {
        private readonly Gemma2ModelSpec _spec;

        public Gemma2Reference(Gemma2ModelSpec spec) => _spec = spec;

        /// <summary>Logits for a query token forwarded at position historyTokens.Length after replaying the sequence's earlier tokens through the same layer.</summary>
        public double[] LogitsAt(
            int[] historyTokens,
            int queryToken,
            bool embedScale = true,
            bool qkNorm = false,
            bool geluTanh = false,
            bool postNorms = true,
            float attentionCap = Gemma2ModelSpec.AttentionCap,
            float finalCap = Gemma2ModelSpec.FinalCap)
        {
            double[][] kvK = new double[historyTokens.Length + 1][];
            double[][] kvV = new double[historyTokens.Length + 1][];

            for (int p = 0; p < historyTokens.Length; p++)
            {
                Advance(Embed(historyTokens[p], embedScale), p, kvK, kvV, qkNorm, geluTanh, postNorms, attentionCap);
            }

            double[] cur = Embed(queryToken, embedScale);
            Advance(cur, historyTokens.Length, kvK, kvV, qkNorm, geluTanh, postNorms, attentionCap);

            double[] normed = Norm(cur, _spec.OutputNorm);
            double[] logits = Gemv(_spec.Embeddings, Gemma2ModelSpec.Vocab, Gemma2ModelSpec.Hidden, normed); // tied lm_head
            if (finalCap > 0f)
            {
                for (int i = 0; i < logits.Length; i++)
                {
                    logits[i] = (double)finalCap * Math.Tanh(logits[i] / (double)finalCap);
                }
            }

            return logits;
        }

        public double[][] Encoded(
            int[] tokens,
            bool embedScale = true,
            bool qkNorm = false,
            bool postNorms = true,
            float attentionCap = Gemma2ModelSpec.AttentionCap)
        {
            int n = tokens.Length;
            double[][] cur = new double[n][];
            for (int i = 0; i < n; i++)
            {
                cur[i] = Embed(tokens[i], embedScale);
            }

            double[][] kvK = new double[n][];
            double[][] kvV = new double[n][];

            // Phase 1: pre-attention norm, QKV, RoPE, K/V writes for every position.
            double[][] qAll = new double[n][];
            for (int p = 0; p < n; p++)
            {
                double[] normed = Norm(cur[p], _spec.AttnNorm);
                double[] q = Gemv(_spec.AttnQ, Gemma2ModelSpec.Heads * Gemma2ModelSpec.HeadDim, Gemma2ModelSpec.Hidden, normed);
                double[] k = Gemv(_spec.AttnK, Gemma2ModelSpec.KvHeads * Gemma2ModelSpec.HeadDim, Gemma2ModelSpec.Hidden, normed);
                double[] v = Gemv(_spec.AttnV, Gemma2ModelSpec.KvHeads * Gemma2ModelSpec.HeadDim, Gemma2ModelSpec.Hidden, normed);

                if (qkNorm)
                {
                    QkNormHeads(q);
                    QkNormHeads(k);
                }

                Rope(q, p);
                Rope(k, p);

                qAll[p] = q;
                kvK[p] = k;
                kvV[p] = v;
            }

            // Phase 2: every query attends over the whole finished KV span.
            double[] attnOut = new double[n * Gemma2ModelSpec.Heads * Gemma2ModelSpec.HeadDim];
            for (int p = 0; p < n; p++)
            {
                Attention(qAll[p], kvK, kvV, 0, n - 1, attentionCap, attnOut.AsSpan(p * Gemma2ModelSpec.Heads * Gemma2ModelSpec.HeadDim));
            }

            // Phase 3: output projection, the post-attention norm on the output, and the ffn_inp residual add.
            for (int p = 0; p < n; p++)
            {
                double[] proj = Gemv(_spec.AttnO, Gemma2ModelSpec.Hidden, Gemma2ModelSpec.Heads * Gemma2ModelSpec.HeadDim, attnOut.AsSpan(p * Gemma2ModelSpec.Heads * Gemma2ModelSpec.HeadDim).ToArray());
                if (postNorms)
                {
                    proj = Norm(proj, _spec.PostAttnNorm);
                }

                for (int i = 0; i < Gemma2ModelSpec.Hidden; i++)
                {
                    cur[p][i] += proj[i];
                }
            }

            // Phases 4 and 5: gate/up GEMVs, the exact-GELU gate, the down GEMV, the post-feed-forward norm on
            // its output, and the ffn_out residual add.
            for (int p = 0; p < n; p++)
            {
                double[] normed = Norm(cur[p], _spec.FfnNorm);
                double[] gate = Gemv(_spec.FfnGate, Gemma2ModelSpec.Intermediate, Gemma2ModelSpec.Hidden, normed);
                double[] up = Gemv(_spec.FfnUp, Gemma2ModelSpec.Intermediate, Gemma2ModelSpec.Hidden, normed);
                for (int i = 0; i < Gemma2ModelSpec.Intermediate; i++)
                {
                    gate[i] = Gelu(gate[i]) * up[i];
                }

                double[] proj = Gemv(_spec.FfnDown, Gemma2ModelSpec.Hidden, Gemma2ModelSpec.Intermediate, gate);
                if (postNorms)
                {
                    proj = Norm(proj, _spec.PostFfnNorm);
                }

                for (int i = 0; i < Gemma2ModelSpec.Hidden; i++)
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

        // The Encoded path always runs the reference placement; the switch is fixed here rather than threaded
        // through its signature because only the decode goldens exercise the pre-only counterfactual.
        private const bool postNormsSwitch = true;

        // One causal decode layer: embedding gain already applied; norm, QKV, optional counterfactual QK-norm,
        // RoPE, KV write at the position, causal attention with the capped scores, o-projection, the
        // post-attention norm, residual, FFN norm, gate/up, the gated activation, down, the post-feed-forward
        // norm, residual — the stream advances in place.
        private void Advance(
            double[] cur,
            int position,
            double[][] kvK,
            double[][] kvV,
            bool qkNorm,
            bool geluTanh,
            bool postNorms,
            float attentionCap)
        {
            double[] normed = Norm(cur, _spec.AttnNorm);
            double[] q = Gemv(_spec.AttnQ, Gemma2ModelSpec.Heads * Gemma2ModelSpec.HeadDim, Gemma2ModelSpec.Hidden, normed);
            double[] k = Gemv(_spec.AttnK, Gemma2ModelSpec.KvHeads * Gemma2ModelSpec.HeadDim, Gemma2ModelSpec.Hidden, normed);
            double[] v = Gemv(_spec.AttnV, Gemma2ModelSpec.KvHeads * Gemma2ModelSpec.HeadDim, Gemma2ModelSpec.Hidden, normed);

            if (qkNorm)
            {
                QkNormHeads(q);
                QkNormHeads(k);
            }

            Rope(q, position);
            Rope(k, position);

            kvK[position] = k;
            kvV[position] = v;

            double[] attnOut = new double[Gemma2ModelSpec.Heads * Gemma2ModelSpec.HeadDim];
            Attention(q, kvK, kvV, 0, position, attentionCap, attnOut);

            double[] proj = Gemv(_spec.AttnO, Gemma2ModelSpec.Hidden, Gemma2ModelSpec.Heads * Gemma2ModelSpec.HeadDim, attnOut);
            if (postNorms)
            {
                proj = Norm(proj, _spec.PostAttnNorm);
            }

            for (int i = 0; i < Gemma2ModelSpec.Hidden; i++)
            {
                cur[i] += proj[i];
            }

            double[] ffnNormed = Norm(cur, _spec.FfnNorm);
            double[] gate = Gemv(_spec.FfnGate, Gemma2ModelSpec.Intermediate, Gemma2ModelSpec.Hidden, ffnNormed);
            double[] up = Gemv(_spec.FfnUp, Gemma2ModelSpec.Intermediate, Gemma2ModelSpec.Hidden, ffnNormed);
            for (int i = 0; i < Gemma2ModelSpec.Intermediate; i++)
            {
                gate[i] = (geluTanh ? GeluTanh(gate[i]) : Gelu(gate[i])) * up[i];
            }

            double[] down = Gemv(_spec.FfnDown, Gemma2ModelSpec.Hidden, Gemma2ModelSpec.Intermediate, gate);
            if (postNorms)
            {
                down = Norm(down, _spec.PostFfnNorm);
            }

            for (int i = 0; i < Gemma2ModelSpec.Hidden; i++)
            {
                cur[i] += down[i];
            }
        }

        // The token-embedding row, scaled by sqrt(hidden) the way the reference graph seeds its residual stream.
        private double[] Embed(int token, bool embedScale)
        {
            double[] row = Row(_spec.Embeddings, token, Gemma2ModelSpec.Hidden);
            if (embedScale)
            {
                double scale = Math.Sqrt(Gemma2ModelSpec.Hidden);
                for (int i = 0; i < row.Length; i++)
                {
                    row[i] *= scale;
                }
            }

            return row;
        }

        private static void Attention(double[] q, double[][] kvK, double[][] kvV, int from, int to, float attentionCap, Span<double> outHead)
        {
            const int headDim = Gemma2ModelSpec.HeadDim;
            double scale = 1.0 / Math.Sqrt(headDim);

            // heads 2, kv-heads 1: every head reads kv-head (h * 1 / 2) == 0.
            for (int h = 0; h < Gemma2ModelSpec.Heads; h++)
            {
                int kvHead = h * Gemma2ModelSpec.KvHeads / Gemma2ModelSpec.Heads;

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
                        score = (double)attentionCap * Math.Tanh(score / (double)attentionCap);
                    }

                    scores[t - from] = score;
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

        // Rotate-half RoPE (the gemma2 pairing): pair i occupies head dimensions i and i + headDim/2.
        private static void Rope(double[] headRegion, int position)
        {
            for (int h = 0; h < headRegion.Length / Gemma2ModelSpec.HeadDim; h++)
            {
                for (int i = 0; i < Gemma2ModelSpec.HeadDim / 2; i++)
                {
                    double invFreq = Math.Pow(Gemma2ModelSpec.RopeTheta, -2.0 * i / Gemma2ModelSpec.HeadDim);
                    double angle = position * invFreq;
                    double c = Math.Cos(angle);
                    double s = Math.Sin(angle);

                    double lo = headRegion[h * Gemma2ModelSpec.HeadDim + i];
                    double hi = headRegion[h * Gemma2ModelSpec.HeadDim + i + Gemma2ModelSpec.HeadDim / 2];
                    headRegion[h * Gemma2ModelSpec.HeadDim + i] = lo * c - hi * s;
                    headRegion[h * Gemma2ModelSpec.HeadDim + i + Gemma2ModelSpec.HeadDim / 2] = hi * c + lo * s;
                }
            }
        }

        // The counterfactual unweighted per-head RMS norm: each head normalizes its own slice with the epsilon
        // alone. The gemma2 graph runs no such norm; this exists only to prove the engine does not.
        private static void QkNormHeads(double[] heads)
        {
            for (int region = 0; region < heads.Length; region += Gemma2ModelSpec.HeadDim)
            {
                double sumSq = 0.0;
                for (int i = 0; i < Gemma2ModelSpec.HeadDim; i++)
                {
                    sumSq += heads[region + i] * heads[region + i];
                }

                double invRms = 1.0 / Math.Sqrt(sumSq / Gemma2ModelSpec.HeadDim + Gemma2ModelSpec.Eps);
                for (int i = 0; i < Gemma2ModelSpec.HeadDim; i++)
                {
                    heads[region + i] *= invRms;
                }
            }
        }

        private static double[] Norm(double[] x, float[] weight)
        {
            int n = x.Length;
            double[] result = new double[n];

            double sumSq = 0.0;
            for (int i = 0; i < n; i++)
            {
                sumSq += x[i] * x[i];
            }

            double invRms = 1.0 / Math.Sqrt(sumSq / n + Gemma2ModelSpec.Eps);
            for (int i = 0; i < n; i++)
            {
                result[i] = x[i] * invRms * weight[i];
            }

            return result;
        }

        private static double Gelu(double x) => 0.5 * x * (1.0 + Erf(x / Math.Sqrt(2.0)));

        private static double GeluTanh(double x) =>
            0.5 * x * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));

        // Abramowitz & Stegun 7.1.26 error function: enough precision for a double reference at these magnitudes.
        private static double Erf(double x)
        {
            double sign = x < 0 ? -1.0 : 1.0;
            double ax = Math.Abs(x);
            double t = 1.0 / (1.0 + 0.3275911 * ax);
            double poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
            return sign * (1.0 - poly * Math.Exp(-ax * ax));
        }

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
        string path = Path.Combine(Path.GetTempPath(), $"lair-gemma2-layer-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }
}
