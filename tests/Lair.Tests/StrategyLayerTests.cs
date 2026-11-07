using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Norm and activation strategy units (LayerNorm, exact GELU, tanh-approximated GELU, GeGlu composition) against
/// double-precision references computed in this file, and a forward-level check that a plan resolved to LayerNorm
/// and GeGlu runs every forward path — single token, batched decode, and encode — identically to a hand-written
/// scalar reference forward over the same tensors. The plan override used to reach those strategies is
/// assembly-internal: no host configuration path resolves a profile from outside the profile registry.
/// </summary>
[Trait("Category", "Layers")]
public sealed class StrategyLayerTests : IDisposable
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

    // ---------------------------------------------------------------- norm units

    [Fact]
    public void LayerNormWeighted_MatchesHandComputedVector_WithBias_ZeroVariance_AndEpsilon()
    {
        float[] src = [1f, 2f, 3f, 4f];
        float[] weight = [2f, 1f, 0.5f, 0.25f];

        // Hand arithmetic: mean = 10/4 = 2.5; population variance = 5/4 = 1.25; std = sqrt(1.25).
        double mean = (1.0 + 2.0 + 3.0 + 4.0) / 4.0;
        double variance = ((1.0 - mean) * (1.0 - mean) + (2.0 - mean) * (2.0 - mean)
            + (3.0 - mean) * (3.0 - mean) + (4.0 - mean) * (4.0 - mean)) / 4.0;
        Assert.Equal(2.5, mean, 12);
        Assert.Equal(1.25, variance, 12);
        double invStd = 1.0 / Math.Sqrt(variance); // eps = 0: the divisor is exactly the standard deviation

        double[] expectedNoBias =
        [
            (1.0 - mean) * invStd * 2.0,
            (2.0 - mean) * invStd * 1.0,
            (3.0 - mean) * invStd * 0.5,
            (4.0 - mean) * invStd * 0.25,
        ];

        float[] dst = new float[4];
        LayerOps.LayerNormWeighted(src, weight, ReadOnlySpan<float>.Empty, 0f, dst);
        for (int i = 0; i < 4; i++)
        {
            Assert.True(
                Math.Abs(dst[i] - expectedNoBias[i]) <= 1e-6 * Math.Max(1.0, Math.Abs(expectedNoBias[i])),
                $"layernorm[{i}]: expected {expectedNoBias[i]:G9}, got {dst[i]:G9}");
        }

        // A supplied bias shifts every normalized entry.
        float[] bias = [0.1f, -0.2f, 0.3f, -0.4f];
        LayerOps.LayerNormWeighted(src, weight, bias, 0f, dst);
        for (int i = 0; i < 4; i++)
        {
            double expected = expectedNoBias[i] + bias[i];
            Assert.True(
                Math.Abs(dst[i] - expected) <= 1e-6 * Math.Max(1.0, Math.Abs(expected)),
                $"layernorm+bias[{i}]: expected {expected:G9}, got {dst[i]:G9}");
        }

        // Zero variance: the epsilon under the root keeps the result finite, and every entry collapses to the bias.
        LayerOps.LayerNormWeighted(new float[4], weight, bias, 1e-5f, dst);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(bias[i], dst[i]);
        }

        // eps enters only under the square root: with variance 1 the divisor moves from 1 to sqrt(1 + 1e-3).
        float[] alternating = [1f, -1f, 1f, -1f];
        float[] unit = [1f, 1f, 1f, 1f];
        double epsDivisor = Math.Sqrt(1.0 + 1e-3);
        LayerOps.LayerNormWeighted(alternating, unit, ReadOnlySpan<float>.Empty, 1e-3f, dst);
        for (int i = 0; i < 4; i++)
        {
            double expected = alternating[i] / epsDivisor;
            Assert.True(
                Math.Abs(dst[i] - expected) <= 1e-6 * Math.Max(1.0, Math.Abs(expected)),
                $"layernorm eps[{i}]: expected {expected:G9}, got {dst[i]:G9}");
        }

        Assert.Throws<ArgumentException>(() => LayerOps.LayerNormWeighted(src, new float[2], ReadOnlySpan<float>.Empty, 0f, dst));
        Assert.Throws<ArgumentException>(() => LayerOps.LayerNormWeighted(src, weight, new float[2], 0f, dst));
    }

    // ---------------------------------------------------------------- scalar activations

    [Theory]
    [InlineData(-8f)]
    [InlineData(-3.7f)]
    [InlineData(-1.5f)]
    [InlineData(-0.5f)]
    [InlineData(0f)]
    [InlineData(0.3f)]
    [InlineData(1f)]
    [InlineData(2.5f)]
    [InlineData(6f)]
    [InlineData(10f)]
    public void Gelu_MatchesSeriesComputedErfReference(float x)
    {
        double d = x;
        double expected = 0.5 * d * (1.0 + ReferenceErf(d / Math.Sqrt(2.0)));

        float actual = LayerOps.Gelu(x);
        Assert.True(
            Math.Abs(actual - expected) <= 1e-6 * Math.Max(1.0, Math.Abs(expected)),
            $"gelu({x}): expected {expected:G9}, got {actual:G9}");
    }

    [Theory]
    [InlineData(-8f)]
    [InlineData(-3.7f)]
    [InlineData(-1.5f)]
    [InlineData(-0.5f)]
    [InlineData(0f)]
    [InlineData(0.3f)]
    [InlineData(1f)]
    [InlineData(2.5f)]
    [InlineData(6f)]
    [InlineData(10f)]
    public void GeluTanh_MatchesDirectFormulaEvaluation(float x)
    {
        double d = x;
        double expected = 0.5 * d * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (d + 0.044715 * d * d * d)));

        float actual = LayerOps.GeluTanh(x);
        Assert.True(
            Math.Abs(actual - expected) <= 1e-6 * Math.Max(1.0, Math.Abs(expected)),
            $"gelu_tanh({x}): expected {expected:G9}, got {actual:G9}");
    }

    [Fact]
    public void GeGlu_Composition_MatchesHandComputedVector()
    {
        float[] gate = [-2f, -0.5f, 0f, 0.75f, 3f];
        float[] up = [1.5f, -1f, 2f, 0.25f, -0.5f];

        // Hand oracle: gelu_tanh over each gate entry times the matching up entry, all in double.
        double[] expected = new double[gate.Length];
        for (int i = 0; i < gate.Length; i++)
        {
            double g = gate[i];
            double act = 0.5 * g * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (g + 0.044715 * g * g * g)));
            expected[i] = act * up[i];
        }

        float[] dst = [.. gate];
        for (int i = 0; i < gate.Length; i++)
        {
            dst[i] = LayerOps.GeluTanh(gate[i]) * up[i];
        }

        for (int i = 0; i < gate.Length; i++)
        {
            Assert.True(
                Math.Abs(dst[i] - expected[i]) <= 1e-6 * Math.Max(1.0, Math.Abs(expected[i])),
                $"geglu[{i}]: expected {expected[i]:G9}, got {dst[i]:G9}");
        }
    }

    // ---------------------------------------------------------------- forward-level strategy execution

    [Fact]
    public void Forward_LayerNormGeGluPlan_MatchesReferenceForward_InDecodeBatchAndEncode()
    {
        StrategyModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.TensorDefs()));

        ModelProfile strategyProfile = ModelProfiles.Resolve("llama") with { Norm = NormStrategy.LayerNorm, Activation = ActivationStrategy.GeGlu };

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, strategyProfile);

        Assert.Equal(NormStrategy.LayerNorm, model.Plan.Norm);
        Assert.Equal(ActivationStrategy.GeGlu, model.Plan.Activation);
        Assert.Equal("llama", model.Plan.Architecture); // the declared name is recorded even under an explicit profile

        ReferenceForward reference = new(spec);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        // Single-token decode at positions 0 and 1.
        Span<float> logits0 = model.DecodeNext(1, kv);
        Assert.True(
            Closest(logits0, reference.LogitsAt([], 1)),
            "decode logits at position 0 must match the reference forward");

        Span<float> logits1 = model.DecodeNext(4, kv);
        Assert.True(
            Closest(logits1, reference.LogitsAt([1], 4)),
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
    public void Forward_LayerNormGeGluPlan_DecodeBatchAndEncodeMatchTheSingleTokenPath()
    {
        StrategyModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.TensorDefs()));

        ModelProfile strategyProfile = ModelProfiles.Resolve("llama") with { Norm = NormStrategy.LayerNorm, Activation = ActivationStrategy.GeGlu };

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, strategyProfile);

        int[] tokens = [1, 4, 2];

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);

        // Single-token path over the three tokens: the last forward's logits are the reference.
        using SequenceKv seqA = new(pool);
        model.Prefill(tokens, seqA);
        float[] singleLogits = model.LastLogits().ToArray();

        // Batched path: the same three tokens, one per batch step across a fresh sequence.
        using SequenceKv seqB = new(pool);
        model.DecodeBatch([tokens[0]], [seqB]);
        model.DecodeBatch([tokens[1]], [seqB]);
        Span<float> batchLogits = model.DecodeBatch([tokens[2]], [seqB]);
        Assert.True(
            batchLogits.SequenceEqual(singleLogits.AsSpan()),
            "a token batched through DecodeBatch must be bit-identical to stepping the same sequence alone");

        // Encode over a fresh sequence must stay finite and non-degenerate.
        using SequenceKv seqD = new(pool);
        float[] encoded = model.Encode(tokens, seqD);
        foreach (float v in encoded)
        {
            Assert.True(float.IsFinite(v), $"encoded value {v} is not finite");
        }
    }

    // ---------------------------------------------------------------- reference oracle helpers

    // Independent double-precision erf for the exact-GELU oracle: the Maclaurin series
    // erf(t) = 2/sqrt(pi) * sum_k (-1)^k t^(2k+1) / (k! (2k+1)) while its terms stay well inside the double
    // range, and the Laplace continued fraction erfc(t) = exp(-t^2)/sqrt(pi) * 1/(t + 1/(2t + 2/(t + 3/(2t + ...))))
    // beyond |t| = 4, where the series would cancel catastrophically.
    private static double ReferenceErf(double t)
    {
        double absT = Math.Abs(t);

        if (absT <= 4.0)
        {
            double term = t;
            double sum = term;
            for (int k = 1; k <= 40; k++)
            {
                term *= -t * t / k;
                double contribution = term / (2 * k + 1);
                sum += contribution;
                if (Math.Abs(contribution) <= 1e-18 * Math.Abs(sum))
                {
                    break;
                }
            }

            return 2.0 / Math.Sqrt(Math.PI) * sum;
        }

        if (absT >= 26.0)
        {
            return t > 0.0 ? 1.0 : -1.0;
        }

        // Modified Lentz evaluation of the continued fraction denominator.
        const double tiny = 1e-300;
        double b = t;
        double c = t / tiny;
        double d = 1.0 / t;
        double f = d;
        for (int i = 1; i <= 200; i++)
        {
            b = (i % 2 == 1 ? 2.0 : 1.0) * t + b;
            d = 1.0 / (b + d);
            c = b + 1.0 / c;
            double delta = c * d;
            f *= delta;
            if (Math.Abs(delta - 1.0) <= 1e-16)
            {
                break;
            }
        }

        double erfc = Math.Exp(-t * t) / Math.Sqrt(Math.PI) * f;
        return t > 0.0 ? 1.0 - erfc : -1.0 + erfc;
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
        string path = Path.Combine(Path.GetTempPath(), $"lair-strategy-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // A complete single-layer model with hidden 4, heads 2, kv-heads 2, head dim 2, ffn 8, vocab 6 — all F32
    // tensors with hand-picked payloads the reference forward reads directly.
    private sealed class StrategyModelSpec
    {
        public const int Hidden = 4;
        public const int Heads = 2;
        public const int KvHeads = 2;
        public const int HeadDim = 2;
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
            ("llama.attention.layer_norm_rms_epsilon", Eps),
            ("llama.vocab_size", Vocab),
        ];

        public List<TensorDef> TensorDefs() =>
        [
            new("token_embd.weight", [Vocab, Hidden], Embeddings),
            new("blk.0.attn_q.weight", [Hidden, Hidden], AttnQ),
            new("blk.0.attn_k.weight", [Hidden, Hidden], AttnK),
            new("blk.0.attn_v.weight", [Hidden, Hidden], AttnV),
            new("blk.0.attn_output.weight", [Hidden, Hidden], AttnO),
            new("blk.0.ffn_gate.weight", [Intermediate, Hidden], FfnGate),
            new("blk.0.ffn_up.weight", [Intermediate, Hidden], FfnUp),
            new("blk.0.ffn_down.weight", [Hidden, Intermediate], FfnDown),
            new("blk.0.attn_norm.weight", [Hidden], AttnNorm),
            new("blk.0.ffn_norm.weight", [Hidden], FfnNorm),
            new("output_norm.weight", [Hidden], OutputNorm),
        ];

        public readonly float[] Embeddings =
        [
            0.10f, 0.20f, 0.30f, 0.40f,
            0.11f, 0.21f, 0.31f, 0.41f,
            0.12f, 0.22f, 0.32f, 0.42f,
            0.13f, 0.23f, 0.33f, 0.43f,
            0.14f, 0.24f, 0.34f, 0.44f,
            0.15f, 0.25f, 0.35f, 0.45f,
        ];

        public readonly float[] AttnQ =
        [
            0.05f, 0.10f, 0.15f, 0.20f,
            0.06f, 0.11f, 0.16f, 0.21f,
            0.07f, 0.12f, 0.17f, 0.22f,
            0.08f, 0.13f, 0.18f, 0.23f,
        ];

        public readonly float[] AttnK =
        [
            0.01f, 0.02f, 0.03f, 0.04f,
            0.02f, 0.03f, 0.04f, 0.05f,
            0.03f, 0.04f, 0.05f, 0.06f,
            0.04f, 0.05f, 0.06f, 0.07f,
        ];

        public readonly float[] AttnV =
        [
            0.09f, 0.08f, 0.07f, 0.06f,
            0.08f, 0.07f, 0.06f, 0.05f,
            0.07f, 0.06f, 0.05f, 0.04f,
            0.06f, 0.05f, 0.04f, 0.03f,
        ];

        public readonly float[] AttnO =
        [
            0.10f, 0.00f, 0.20f, 0.00f,
            0.00f, 0.30f, 0.00f, 0.40f,
            0.15f, 0.00f, 0.25f, 0.00f,
            0.00f, 0.35f, 0.00f, 0.45f,
        ];

        public readonly float[] FfnGate =
        [
            0.10f, 0.11f, 0.12f, 0.13f,
            0.14f, 0.15f, 0.16f, 0.17f,
            0.18f, 0.19f, 0.20f, 0.21f,
            0.22f, 0.23f, 0.24f, 0.25f,
            0.26f, 0.27f, 0.28f, 0.29f,
            0.30f, 0.31f, 0.32f, 0.33f,
            0.34f, 0.35f, 0.36f, 0.37f,
            0.38f, 0.39f, 0.40f, 0.41f,
        ];

        public readonly float[] FfnUp =
        [
            0.01f, 0.02f, 0.03f, 0.04f,
            0.05f, 0.06f, 0.07f, 0.08f,
            0.09f, 0.10f, 0.11f, 0.12f,
            0.13f, 0.14f, 0.15f, 0.16f,
            0.17f, 0.18f, 0.19f, 0.20f,
            0.21f, 0.22f, 0.23f, 0.24f,
            0.25f, 0.26f, 0.27f, 0.28f,
            0.29f, 0.30f, 0.31f, 0.32f,
        ];

        public readonly float[] FfnDown =
        [
            0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.10f, 0.11f, 0.12f,
            0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.20f,
            0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f, 0.27f, 0.28f,
            0.29f, 0.30f, 0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f,
        ];

        public readonly float[] AttnNorm = [1.10f, 0.90f, 1.05f, 0.95f];
        public readonly float[] FfnNorm = [0.90f, 1.10f, 0.95f, 1.05f];
        public readonly float[] OutputNorm = [1.00f, 1.02f, 0.98f, 1.01f];
    }

    // A scalar reference forward over the spec's hand-picked tensors: LayerNorm (no bias), standard full-head
    // RoPE, causal or bidirectional attention with an explicit softmax, and gelu_tanh(gate) * up — computed in
    // double and compared against the engine's float logits.
    private sealed class ReferenceForward
    {
        private readonly StrategyModelSpec _spec;

        public ReferenceForward(StrategyModelSpec spec) => _spec = spec;

        /// <summary>Logits for a query token forwarded at position historyTokens.Length after replaying the sequence's earlier tokens through the same layer.</summary>
        public double[] LogitsAt(int[] historyTokens, int queryToken)
        {
            double[][] kvK = new double[historyTokens.Length + 1][];
            double[][] kvV = new double[historyTokens.Length + 1][];

            for (int p = 0; p < historyTokens.Length; p++)
            {
                Advance(Row(_spec.Embeddings, historyTokens[p], StrategyModelSpec.Hidden), p, kvK, kvV);
            }

            double[] cur = Row(_spec.Embeddings, queryToken, StrategyModelSpec.Hidden);
            Advance(cur, historyTokens.Length, kvK, kvV);

            return OutputLogits(cur);
        }

        public double[][] Encoded(int[] tokens)
        {
            int n = tokens.Length;
            double[][] cur = new double[n][];
            for (int i = 0; i < n; i++)
            {
                cur[i] = Row(_spec.Embeddings, tokens[i], StrategyModelSpec.Hidden);
            }

            double[][] kvK = new double[n][];
            double[][] kvV = new double[n][];

            // Phase 1: pre-attention norm, QKV, RoPE, K/V writes for every position.
            double[][] qAll = new double[n][];
            for (int p = 0; p < n; p++)
            {
                double[] normed = LayerNorm(cur[p], _spec.AttnNorm);
                double[] q = Gemv(_spec.AttnQ, StrategyModelSpec.Hidden, StrategyModelSpec.Hidden, normed);
                double[] k = Gemv(_spec.AttnK, StrategyModelSpec.Hidden, StrategyModelSpec.Hidden, normed);
                double[] v = Gemv(_spec.AttnV, StrategyModelSpec.Hidden, StrategyModelSpec.Hidden, normed);

                Rope(q, p);
                Rope(k, p);

                qAll[p] = q;
                kvK[p] = k;
                kvV[p] = v;
            }

            // Phase 2: every query attends over the whole finished KV span.
            double[] attnOut = new double[n * StrategyModelSpec.Hidden];
            for (int p = 0; p < n; p++)
            {
                Attention(qAll[p], kvK, kvV, 0, n - 1, attnOut.AsSpan(p * StrategyModelSpec.Hidden));
            }

            // Phase 3: output projection and the ffn_inp residual add.
            for (int p = 0; p < n; p++)
            {
                double[] proj = Gemv(_spec.AttnO, StrategyModelSpec.Hidden, StrategyModelSpec.Hidden, attnOut.AsSpan(p * StrategyModelSpec.Hidden).ToArray());
                for (int i = 0; i < StrategyModelSpec.Hidden; i++)
                {
                    cur[p][i] += proj[i];
                }
            }

            // Phases 4 and 5: gate/up GEMVs, gelu_tanh gating, down GEMV, and the ffn_out residual add.
            for (int p = 0; p < n; p++)
            {
                double[] normed = LayerNorm(cur[p], _spec.FfnNorm);
                double[] gate = Gemv(_spec.FfnGate, StrategyModelSpec.Intermediate, StrategyModelSpec.Hidden, normed);
                double[] up = Gemv(_spec.FfnUp, StrategyModelSpec.Intermediate, StrategyModelSpec.Hidden, normed);
                for (int i = 0; i < StrategyModelSpec.Intermediate; i++)
                {
                    gate[i] = GeluTanh(gate[i]) * up[i];
                }

                double[] proj = Gemv(_spec.FfnDown, StrategyModelSpec.Hidden, StrategyModelSpec.Intermediate, gate);
                for (int i = 0; i < StrategyModelSpec.Hidden; i++)
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

        // One causal decode layer: norm, QKV + RoPE, KV write at the position, causal attention, o-projection,
        // residual, FFN norm, gate/up, gelu_tanh gating, down, residual — the stream is advanced in place.
        private void Advance(double[] cur, int position, double[][] kvK, double[][] kvV)
        {
            double[] normed = LayerNorm(cur, _spec.AttnNorm);
            double[] q = Gemv(_spec.AttnQ, StrategyModelSpec.Hidden, StrategyModelSpec.Hidden, normed);
            double[] k = Gemv(_spec.AttnK, StrategyModelSpec.Hidden, StrategyModelSpec.Hidden, normed);
            double[] v = Gemv(_spec.AttnV, StrategyModelSpec.Hidden, StrategyModelSpec.Hidden, normed);

            Rope(q, position);
            Rope(k, position);

            kvK[position] = k;
            kvV[position] = v;

            double[] attnOut = new double[StrategyModelSpec.Hidden];
            Attention(q, kvK, kvV, 0, position, attnOut);

            double[] proj = Gemv(_spec.AttnO, StrategyModelSpec.Hidden, StrategyModelSpec.Hidden, attnOut);
            for (int i = 0; i < StrategyModelSpec.Hidden; i++)
            {
                cur[i] += proj[i];
            }

            double[] ffnNormed = LayerNorm(cur, _spec.FfnNorm);
            double[] gate = Gemv(_spec.FfnGate, StrategyModelSpec.Intermediate, StrategyModelSpec.Hidden, ffnNormed);
            double[] up = Gemv(_spec.FfnUp, StrategyModelSpec.Intermediate, StrategyModelSpec.Hidden, ffnNormed);
            for (int i = 0; i < StrategyModelSpec.Intermediate; i++)
            {
                gate[i] = GeluTanh(gate[i]) * up[i];
            }

            double[] down = Gemv(_spec.FfnDown, StrategyModelSpec.Hidden, StrategyModelSpec.Intermediate, gate);
            for (int i = 0; i < StrategyModelSpec.Hidden; i++)
            {
                cur[i] += down[i];
            }
        }

        private double[] OutputLogits(double[] cur)
        {
            double[] normed = LayerNorm(cur, _spec.OutputNorm);
            return Gemv(_spec.Embeddings, StrategyModelSpec.Vocab, StrategyModelSpec.Hidden, normed); // tied lm_head
        }

        private static void Attention(double[] q, double[][] kvK, double[][] kvV, int from, int to, Span<double> outHead)
        {
            double scale = 1.0 / Math.Sqrt(StrategyModelSpec.HeadDim);
            int headDim = StrategyModelSpec.HeadDim;

            // heads == kv-heads here: head h reads kv-head h.
            for (int h = 0; h < StrategyModelSpec.Heads; h++)
            {
                double[] scores = new double[to - from + 1];
                double max = double.NegativeInfinity;
                for (int t = from; t <= to; t++)
                {
                    double dot = 0.0;
                    for (int j = 0; j < headDim; j++)
                    {
                        dot += q[h * headDim + j] * kvK[t][h * headDim + j];
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
                        acc += scores[t - from] / sum * kvV[t][h * headDim + j];
                    }

                    outHead[h * headDim + j] = acc;
                }
            }
        }

        private static void Rope(double[] headRegion, int position)
        {
            for (int i = 0; i < StrategyModelSpec.HeadDim / 2; i++)
            {
                double invFreq = Math.Pow(StrategyModelSpec.RopeTheta, -2.0 * i / StrategyModelSpec.HeadDim);
                double angle = position * invFreq;
                double c = Math.Cos(angle);
                double s = Math.Sin(angle);

                double lo = headRegion[2 * i];
                double hi = headRegion[2 * i + 1];
                headRegion[2 * i] = lo * c - hi * s;
                headRegion[2 * i + 1] = hi * c + lo * s;
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
            double invStd = 1.0 / Math.Sqrt(var + StrategyModelSpec.Eps);

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
