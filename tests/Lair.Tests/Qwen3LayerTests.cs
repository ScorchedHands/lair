using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Forward-level goldens for the qwen3 shape: a synthetic qwen3 file whose head dimension is declared by
/// <c>key_length</c> / <c>value_length</c> independently of hidden and head_count (2 x 4 != 6), with grouped
/// query attention and per-head q/k norm vectors, forwarded against a double-precision scalar reference. The
/// norm runs with no declaration at all — the qwen3 profile default turns it on — and the goldens pin the norm
/// between the projection and RoPE; a declared <c>use_qk_norm=false</c> overrides the default and must match
/// the no-norm reference instead. Engine-to-engine comparisons are bit-for-bit; reference comparisons use the
/// double-precision tolerance shared by the strategy goldens.
/// </summary>
[Trait("Category", "Layers")]
public sealed class Qwen3LayerTests : IDisposable
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
    public void Forward_Qwen3ProfileDefaultQkNorm_MatchesScalarReference_InDecodeBatchAndEncode()
    {
        Qwen3ModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors()));

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        // Nothing declares the feature: the qwen3 profile default resolved it on, and the head dimension came
        // from key_length / value_length (2 x 4 != 6) with grouped-query attention.
        Assert.Equal(QkNormMode.PerHeadWeighted, model.Plan.QkNorm);
        Assert.Equal(4, model.Plan.HeadDim);
        Assert.Equal(6, model.Plan.HiddenSize);
        Assert.Equal(1, model.Plan.KvHeadCount);

        Qwen3Reference reference = new(spec);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        Assert.True(
            Closest(model.DecodeNext(1, kv), reference.LogitsAt([], 1)),
            "decode logits at position 0 must match the reference forward");

        Assert.True(
            Closest(model.DecodeNext(4, kv), reference.LogitsAt([1], 4)),
            "decode logits at position 1 must match the reference forward");

        // The norm genuinely changes the logits: the same tensors without the norm give different values.
        Assert.False(
            Closest(model.LastLogits(), reference.LogitsAt([1], 4, qkNorm: false)),
            "the qwen3 profile default must not leave the logits equal to the no-norm reference");

        // The norm lands between the projection and RoPE: with the deliberately uneven norm weights, rotating
        // first and normalizing after is a different forward, and the engine matches the before-RoPE reference.
        Assert.False(
            Closest(model.LastLogits(), reference.LogitsAt([1], 4, qkNorm: true, qkNormAfterRope: true)),
            "QK-norm applied after RoPE would produce different logits; the engine must apply it before");

        // The pairing is rotate-half (i with i + headDim/2), not adjacent dimensions: at position 1 an interleaved
        // rotation gives a different forward, and the engine matches the head-halves reference. The two pairings are
        // genuinely distinct on this fixture, so the positive match above pins which one the qwen3 profile selects.
        double[] rotateHalfLogits = reference.LogitsAt([1], 4);
        double[] interleavedLogits = reference.LogitsAt([1], 4, interleavedRope: true);
        Assert.True(
            rotateHalfLogits.Zip(interleavedLogits).Any(pair => Math.Abs(pair.First - pair.Second) > 1e-5 * Math.Max(1.0, Math.Abs(pair.Second))),
            "the rotate-half and adjacent-dimension RoPE pairings must differ at position 1 for this fixture");

        Assert.False(
            Closest(model.LastLogits(), reference.LogitsAt([1], 4, interleavedRope: true)),
            "qwen3 rotates each head's halves (i, i + headDim/2), not adjacent dimensions (2i, 2i + 1)");

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
    public void Forward_Qwen3DeclaredQkNormFalse_OverridesTheDefault_AndMatchesTheNoNormReference()
    {
        Qwen3ModelSpec spec = new();
        string path = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("qwen3.attention.use_qk_norm", false)],
            spec.Tensors()));

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        Assert.Equal(QkNormMode.Off, model.Plan.QkNorm); // the declaration overrides the profile default

        Qwen3Reference reference = new(spec);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        Assert.True(
            Closest(model.DecodeNext(1, kv), reference.LogitsAt([], 1, qkNorm: false)),
            "declared-off decode logits must match the no-norm reference forward");
        Assert.True(
            Closest(model.DecodeNext(4, kv), reference.LogitsAt([1], 4, qkNorm: false)),
            "declared-off logits at position 1 must match the no-norm reference forward");

        // Batched decode and encode follow the same declared-off reference.
        using SequenceKv batchKv = new(pool);
        Span<float> batchLogits = model.DecodeBatch([1], [batchKv]);
        Assert.True(
            Closest(batchLogits.Slice(0, model.Plan.VocabSize), reference.LogitsAt([], 1, qkNorm: false)),
            "batched declared-off logits must match the no-norm reference forward");

        using SequenceKv encodeKv = new(pool);
        float[] encoded = model.Encode([1, 4, 2], encodeKv);
        double[][] expectedEncoded = reference.Encoded([1, 4, 2], qkNorm: false);
        for (int p = 0; p < 3; p++)
        {
            for (int i = 0; i < model.Plan.HiddenSize; i++)
            {
                float actual = encoded[p * model.Plan.HiddenSize + i];
                double expected = expectedEncoded[p][i];
                Assert.True(
                    Math.Abs(actual - expected) <= 1e-5 * Math.Max(1.0, Math.Abs(expected)),
                    $"encode hidden[{p}][{i}] declared-off: expected {expected:G9}, got {actual:G9}");
            }
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

    // A complete single-layer qwen3-shaped model: hidden 6, heads 2, kv-heads 1, head dim 4 declared by
    // key_length / value_length (2 x 4 != 6), ffn 8, vocab 6 — all F32 tensors with hand-picked payloads the
    // reference forward reads directly, including the per-head q/k norm vectors.
    private sealed class Qwen3ModelSpec
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
            ("general.architecture", "qwen3"),
            ("qwen3.embedding_length", Hidden),
            ("qwen3.block_count", 1),
            ("qwen3.attention.head_count", Heads),
            ("qwen3.attention.head_count_kv", KvHeads),
            ("qwen3.feed_forward_length", Intermediate),
            ("qwen3.attention.key_length", HeadDim),
            ("qwen3.attention.value_length", HeadDim),
            ("qwen3.attention.layer_norm_rms_epsilon", Eps),
            ("qwen3.vocab_size", Vocab),
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
            new("blk.0.attn_q_norm.weight", [HeadDim], QkNorm),
            new("blk.0.attn_k_norm.weight", [HeadDim], KNorm),
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
    }

    // A scalar reference forward over the spec's hand-picked tensors: RMSNorm, rotate-half full-head RoPE (the
    // pairing the qwen3 profile selects), causal or bidirectional attention with an explicit softmax, silu(gate) * up,
    // and per-head QK-norm before (or, for the ordering proof, after) RoPE. Computed in double and compared against
    // the engine's float logits with the shared tolerance.
    private sealed class Qwen3Reference
    {
        private readonly Qwen3ModelSpec _spec;

        public Qwen3Reference(Qwen3ModelSpec spec) => _spec = spec;

        /// <summary>Logits for a query token forwarded at position historyTokens.Length after replaying the sequence's earlier tokens through the same layer.</summary>
        public double[] LogitsAt(int[] historyTokens, int queryToken, bool qkNorm = true, bool qkNormAfterRope = false, bool interleavedRope = false)
        {
            double[][] kvK = new double[historyTokens.Length + 1][];
            double[][] kvV = new double[historyTokens.Length + 1][];

            for (int p = 0; p < historyTokens.Length; p++)
            {
                Advance(Row(_spec.Embeddings, historyTokens[p], Qwen3ModelSpec.Hidden), p, kvK, kvV, qkNorm, qkNormAfterRope, interleavedRope);
            }

            double[] cur = Row(_spec.Embeddings, queryToken, Qwen3ModelSpec.Hidden);
            Advance(cur, historyTokens.Length, kvK, kvV, qkNorm, qkNormAfterRope, interleavedRope);

            double[] normed = Norm(cur, _spec.OutputNorm);
            return Gemv(_spec.Embeddings, Qwen3ModelSpec.Vocab, Qwen3ModelSpec.Hidden, normed); // tied lm_head
        }

        public double[][] Encoded(int[] tokens, bool qkNorm = true)
        {
            int n = tokens.Length;
            double[][] cur = new double[n][];
            for (int i = 0; i < n; i++)
            {
                cur[i] = Row(_spec.Embeddings, tokens[i], Qwen3ModelSpec.Hidden);
            }

            double[][] kvK = new double[n][];
            double[][] kvV = new double[n][];

            // Phase 1: pre-attention norm, QKV, QK-norm, RoPE, K/V writes for every position.
            double[][] qAll = new double[n][];
            for (int p = 0; p < n; p++)
            {
                double[] normed = Norm(cur[p], _spec.AttnNorm);
                double[] q = Gemv(_spec.AttnQ, Qwen3ModelSpec.Heads * Qwen3ModelSpec.HeadDim, Qwen3ModelSpec.Hidden, normed);
                double[] k = Gemv(_spec.AttnK, Qwen3ModelSpec.KvHeads * Qwen3ModelSpec.HeadDim, Qwen3ModelSpec.Hidden, normed);
                double[] v = Gemv(_spec.AttnV, Qwen3ModelSpec.KvHeads * Qwen3ModelSpec.HeadDim, Qwen3ModelSpec.Hidden, normed);

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
            double[] attnOut = new double[n * Qwen3ModelSpec.Heads * Qwen3ModelSpec.HeadDim];
            for (int p = 0; p < n; p++)
            {
                Attention(qAll[p], kvK, kvV, 0, n - 1, attnOut.AsSpan(p * Qwen3ModelSpec.Heads * Qwen3ModelSpec.HeadDim));
            }

            // Phase 3: output projection and the ffn_inp residual add.
            for (int p = 0; p < n; p++)
            {
                double[] proj = Gemv(_spec.AttnO, Qwen3ModelSpec.Hidden, Qwen3ModelSpec.Heads * Qwen3ModelSpec.HeadDim, attnOut.AsSpan(p * Qwen3ModelSpec.Heads * Qwen3ModelSpec.HeadDim).ToArray());
                for (int i = 0; i < Qwen3ModelSpec.Hidden; i++)
                {
                    cur[p][i] += proj[i];
                }
            }

            // Phases 4 and 5: gate/up GEMVs, silu gating, down GEMV, and the ffn_out residual add.
            for (int p = 0; p < n; p++)
            {
                double[] normed = Norm(cur[p], _spec.FfnNorm);
                double[] gate = Gemv(_spec.FfnGate, Qwen3ModelSpec.Intermediate, Qwen3ModelSpec.Hidden, normed);
                double[] up = Gemv(_spec.FfnUp, Qwen3ModelSpec.Intermediate, Qwen3ModelSpec.Hidden, normed);
                for (int i = 0; i < Qwen3ModelSpec.Intermediate; i++)
                {
                    gate[i] = Silu(gate[i]) * up[i];
                }

                double[] proj = Gemv(_spec.FfnDown, Qwen3ModelSpec.Hidden, Qwen3ModelSpec.Intermediate, gate);
                for (int i = 0; i < Qwen3ModelSpec.Hidden; i++)
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

        // One causal decode layer: norm, QKV, QK-norm, RoPE, KV write at the position, causal attention,
        // o-projection, residual, FFN norm, gate/up, silu gating, down, residual — the stream advances in place.
        private void Advance(double[] cur, int position, double[][] kvK, double[][] kvV, bool qkNorm, bool qkNormAfterRope, bool interleavedRope)
        {
            double[] normed = Norm(cur, _spec.AttnNorm);
            double[] q = Gemv(_spec.AttnQ, Qwen3ModelSpec.Heads * Qwen3ModelSpec.HeadDim, Qwen3ModelSpec.Hidden, normed);
            double[] k = Gemv(_spec.AttnK, Qwen3ModelSpec.KvHeads * Qwen3ModelSpec.HeadDim, Qwen3ModelSpec.Hidden, normed);
            double[] v = Gemv(_spec.AttnV, Qwen3ModelSpec.KvHeads * Qwen3ModelSpec.HeadDim, Qwen3ModelSpec.Hidden, normed);

            if (qkNorm && !qkNormAfterRope)
            {
                QkNormHeads(q, _spec.QkNorm);
                QkNormHeads(k, _spec.KNorm);
            }

            Rope(q, position, interleavedRope);
            Rope(k, position, interleavedRope);

            if (qkNorm && qkNormAfterRope)
            {
                QkNormHeads(q, _spec.QkNorm);
                QkNormHeads(k, _spec.KNorm);
            }

            kvK[position] = k;
            kvV[position] = v;

            double[] attnOut = new double[Qwen3ModelSpec.Heads * Qwen3ModelSpec.HeadDim];
            Attention(q, kvK, kvV, 0, position, attnOut);

            double[] proj = Gemv(_spec.AttnO, Qwen3ModelSpec.Hidden, Qwen3ModelSpec.Heads * Qwen3ModelSpec.HeadDim, attnOut);
            for (int i = 0; i < Qwen3ModelSpec.Hidden; i++)
            {
                cur[i] += proj[i];
            }

            double[] ffnNormed = Norm(cur, _spec.FfnNorm);
            double[] gate = Gemv(_spec.FfnGate, Qwen3ModelSpec.Intermediate, Qwen3ModelSpec.Hidden, ffnNormed);
            double[] up = Gemv(_spec.FfnUp, Qwen3ModelSpec.Intermediate, Qwen3ModelSpec.Hidden, ffnNormed);
            for (int i = 0; i < Qwen3ModelSpec.Intermediate; i++)
            {
                gate[i] = Silu(gate[i]) * up[i];
            }

            double[] down = Gemv(_spec.FfnDown, Qwen3ModelSpec.Hidden, Qwen3ModelSpec.Intermediate, gate);
            for (int i = 0; i < Qwen3ModelSpec.Hidden; i++)
            {
                cur[i] += down[i];
            }
        }

        private static void Attention(double[] q, double[][] kvK, double[][] kvV, int from, int to, Span<double> outHead)
        {
            const int headDim = Qwen3ModelSpec.HeadDim;
            double scale = 1.0 / Math.Sqrt(headDim);

            // heads 2, kv-heads 1: every head reads kv-head (h * 1 / 2) == 0.
            for (int h = 0; h < Qwen3ModelSpec.Heads; h++)
            {
                int kvHead = h * Qwen3ModelSpec.KvHeads / Qwen3ModelSpec.Heads;

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

        // Rotate-half RoPE (the qwen3 pairing): pair i occupies head dimensions i and i + headDim/2. The
        // interleaved variant pairs adjacent dimensions 2i, 2i + 1 instead — the llama-family layout used only to
        // prove the engine does not rotate this architecture that way.
        private static void Rope(double[] headRegion, int position, bool interleaved = false)
        {
            for (int h = 0; h < headRegion.Length / Qwen3ModelSpec.HeadDim; h++)
            {
                for (int i = 0; i < Qwen3ModelSpec.HeadDim / 2; i++)
                {
                    double invFreq = Math.Pow(Qwen3ModelSpec.RopeTheta, -2.0 * i / Qwen3ModelSpec.HeadDim);
                    double angle = position * invFreq;
                    double c = Math.Cos(angle);
                    double s = Math.Sin(angle);

                    int lowIndex = interleaved ? h * Qwen3ModelSpec.HeadDim + 2 * i : h * Qwen3ModelSpec.HeadDim + i;
                    int highIndex = interleaved ? h * Qwen3ModelSpec.HeadDim + 2 * i + 1 : h * Qwen3ModelSpec.HeadDim + i + Qwen3ModelSpec.HeadDim / 2;

                    double lo = headRegion[lowIndex];
                    double hi = headRegion[highIndex];
                    headRegion[lowIndex] = lo * c - hi * s;
                    headRegion[highIndex] = hi * c + lo * s;
                }
            }
        }

        // Per-head RMS norm with the head-dimension weight vector: each head normalizes its own slice, in place.
        private static void QkNormHeads(double[] heads, float[] weight)
        {
            for (int region = 0; region < heads.Length; region += Qwen3ModelSpec.HeadDim)
            {
                double sumSq = 0.0;
                for (int i = 0; i < Qwen3ModelSpec.HeadDim; i++)
                {
                    sumSq += heads[region + i] * heads[region + i];
                }

                double invRms = 1.0 / Math.Sqrt(sumSq / Qwen3ModelSpec.HeadDim + Qwen3ModelSpec.Eps);
                for (int i = 0; i < Qwen3ModelSpec.HeadDim; i++)
                {
                    heads[region + i] = heads[region + i] * invRms * weight[i];
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

            double invRms = 1.0 / Math.Sqrt(sumSq / n + Qwen3ModelSpec.Eps);
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
        string path = Path.Combine(Path.GetTempPath(), $"lair-qwen3-layer-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }
}
