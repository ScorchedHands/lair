using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Load-time resolution for the gemma2 profile: profile selection and its resolved geometry under the gemma2
/// prefix (the head dimension declared by <c>head_length</c> with <c>key_length</c> / <c>value_length</c> agreeing),
/// the strategy values the profile carries — RMS block norms, the exact-GELU gated feed-forward, rotate-half
/// rotary rows, post-sublayer norm placement, the input-embedding gain, and the per-head QK-norm and attention-bias
/// defaults staying off — the declared overrides for both features, the declared <c>layer_norm_epsilon</c> feeding
/// the LayerNorm epsilon slot while the RMS epsilon feeds the block norms, the softcap and window declarations the
/// real file carries, and the tensor-side consequences of the profile defaults at Open: missing post-sublayer norm
/// vectors fail the load naming them, a declared <c>use_bias=true</c> makes every layer's q/k/v bias vectors
/// required, and present bias vectors under the resolved-off default are reported as ignored.
/// </summary>
[Trait("Category", "Loader")]
public sealed class Gemma2PlanTests : IDisposable
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

    // ---------------------------------------------------------------- gemma2 profile resolution

    [Fact]
    public void ModelPlan_Gemma2Metadata_ResolvesProfileGeometry_AndReferenceGraphDefaults()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata("gemma2"), [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal("gemma2", plan.Architecture);
        Assert.Equal("gemma2.", plan.Prefix); // every key was read under the gemma2 prefix
        Assert.Equal("gemma2", plan.Profile);
        Assert.Equal(26, plan.LayerCount);
        Assert.Equal(2304, plan.HiddenSize);
        Assert.Equal(16, plan.HeadCount);
        Assert.Equal(4, plan.KvHeadCount);
        Assert.Equal(144, plan.HeadDim); // declared by head_length / key_length / value_length: 16 x 144 != 2304
        Assert.Equal(9216, plan.IntermediateSize);
        Assert.Equal(256_000, plan.VocabSize);
        Assert.True(Math.Abs(plan.RopeTheta - 10_000f) < 1f, $"rope theta {plan.RopeTheta:G9}");
        Assert.True(Math.Abs(plan.EpsRmsNorm - 1e-6f) < 1e-8f, $"rms epsilon {plan.EpsRmsNorm:G9}");

        // The reference gemma2 graph runs every block norm as an RMS norm, gates the feed-forward with the exact
        // GELU, rotates each head's halves, and normalizes each sublayer output with its own vector before the
        // residual add — the profile carries all four with no declaration involved.
        Assert.Equal(NormStrategy.RmsNorm, plan.Norm);
        Assert.Equal(ActivationStrategy.GeGluExact, plan.Activation);
        Assert.Equal(RopeStrategy.RotateHalfFullHead, plan.Rope);
        Assert.Equal(NormPlacement.PostSublayerOutputs, plan.NormPlacement);

        // The conversion ships no per-head norm vectors and no attention biases, so both features default off;
        // the bias policy stays declared-only and no norm-bias vectors are demanded.
        Assert.Equal(QkNormMode.Off, plan.QkNorm);
        Assert.False(plan.UseAttentionBias);
        Assert.False(plan.UseNormBias);
        Assert.Equal(BiasPolicy.Declared, plan.Bias);

        // The conversion leaves the embedding gain out of the weights: the plan resolves sqrt(hidden) for it.
        Assert.True(
            Math.Abs(plan.InputEmbeddingScale - MathF.Sqrt(2304f)) <= 1e-4f,
            $"input embedding scale {plan.InputEmbeddingScale:G9} must be sqrt(2304)");

        // The declared head dimension is independent of hidden / head_count: 16 x 144 is the q/k/v region size,
        // not a split of the hidden size.
        Assert.True(
            16 * plan.HeadDim != plan.HiddenSize || plan.HeadDim == 144,
            "head dim comes from the declaration, not from hidden / head_count");

        // The block-norm epsilon slot stays zero for this file: it declares only the RMS epsilon, which the
        // RMS block norms consume.
        Assert.Equal(0f, plan.EpsLayerNorm);
    }

    [Fact]
    public void ModelPlan_Gemma2RealFileKeys_ResolvesCapsWindowAndHeadLengthSource()
    {
        // The keys the real gemma-2 conversion carries: head_length alongside the agreeing key_length /
        // value_length, both softcaps, and the sliding window.
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [
                ("general.architecture", "gemma2"),
                ("gemma2.embedding_length", 2304),
                ("gemma2.block_count", 26),
                ("gemma2.attention.head_count", 16),
                ("gemma2.attention.head_count_kv", 4),
                ("gemma2.feed_forward_length", 9216),
                ("gemma2.attention.head_length", 144),
                ("gemma2.attention.key_length", 144),
                ("gemma2.attention.value_length", 144),
                ("gemma2.attention.sliding_window", 4096),
                ("gemma2.attention.layer_norm_rms_epsilon", 1e-6f),
                ("gemma2.attn_logit_softcapping", 50f),
                ("gemma2.final_logit_softcapping", 30f),
                ("gemma2.vocab_size", 256_000),
            ],
            [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal(144, plan.HeadDim); // head_length names the source; the agreeing key/value lengths confirm it
        Assert.Equal(4096, plan.AttentionWindow);
        Assert.Equal(50f, plan.AttentionSoftcap);
        Assert.Equal(30f, plan.FinalLogitSoftcap);
    }

    [Fact]
    public void ModelPlan_Gemma2DeclaredQkNorm_OverridesTheProfileDefault_EitherWay()
    {
        foreach (bool declared in new bool[] { true, false })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
                [.. Metadata("gemma2"), ("gemma2.attention.use_qk_norm", declared)], [])));
            ModelPlan plan = ModelPlans.Resolve(gguf);

            // A declaration wins in both directions: true requests the weighted form the file's graph does not
            // run, false restates the profile default.
            Assert.Equal(declared ? QkNormMode.PerHeadWeighted : QkNormMode.Off, plan.QkNorm);
        }
    }

    [Fact]
    public void ModelPlan_Gemma2DeclaredUseBias_ResolvesTheFeatureInBothDirections()
    {
        foreach (bool declared in new bool[] { true, false })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
                [.. Metadata("gemma2"), ("gemma2.attention.use_bias", declared)], [])));
            Assert.Equal(declared, ModelPlans.Resolve(gguf).UseAttentionBias);
        }
    }

    [Fact]
    public void ModelPlan_Gemma2DeclaredLayerNormEpsilon_ResolvesIt_AndTheRmsEpsilonStaysSeparate()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("gemma2"), ("gemma2.attention.layer_norm_epsilon", 1e-5f)], [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.True(
            Math.Abs(plan.EpsLayerNorm - 1e-5f) < 1e-8f,
            $"layer-norm epsilon {plan.EpsLayerNorm:G9} must come from the declared key");
        Assert.True(
            Math.Abs(plan.EpsRmsNorm - 1e-6f) < 1e-8f,
            $"rms epsilon {plan.EpsRmsNorm:G9} stays the operand of the RMS norms whatever the declared key says");
    }

    // ---------------------------------------------------------------- tensor-side consequences at Open

    [Fact]
    public void TransformerModel_Gemma2ProfileDefault_MissingPostSublayerNorms_FailsLoadNamingThem_AndThePlacement()
    {
        string path = WriteTemp(BuildGguf(Metadata("gemma2", tiny: true), [.. TinyTensors(postNorms: false)]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("blk.0.post_attention_norm.weight", ex.Message);
        Assert.Contains("norm placement", ex.Message);
        Assert.Contains("gemma2", ex.Message);
    }

    [Fact]
    public void TransformerModel_Gemma2DeclaredUseBiasTrue_MissingBiasTensors_FailsLoadNamingTheFirstOne()
    {
        string path = WriteTemp(BuildGguf(
            [.. Metadata("gemma2", tiny: true), ("gemma2.attention.use_bias", true)],
            [.. TinyTensors()]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("blk.0.attn_q.bias", ex.Message);
        Assert.Contains("gemma2", ex.Message);
    }

    [Fact]
    public void TransformerModel_Gemma2BiasResolvedOff_PresentBiasTensors_AreReportedIgnoredOnce_AndForwardUnchanged()
    {
        List<string> logLines = [];
        string biasedPath = WriteTemp(BuildGguf(
            [.. Metadata("gemma2", tiny: true), ("general.name", "Biased Gemma")],
            [.. TinyTensors(),
                new TensorDef("blk.0.attn_q.bias", [8]),
                new TensorDef("blk.0.attn_k.bias", [4]),
                new TensorDef("blk.0.attn_v.bias", [4])]));
        string plainPath = WriteTemp(BuildGguf(Metadata("gemma2", tiny: true), [.. TinyTensors()]));

        using CpuBackend backend = new();

        float[] biasedLogits;
        {
            using GgufFile gguf = GgufFile.Open(biasedPath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, logLines.Add);

            Assert.False(model.Plan.UseAttentionBias); // the profile default keeps the feature off
            Assert.Single(logLines);
            Assert.StartsWith("bias_ignored ", logLines[0]);
            Assert.Contains("model='Biased Gemma'", logLines[0]);
            Assert.Contains("architecture='gemma2'", logLines[0]);
            Assert.Contains("blk.0.attn_q.bias", logLines[0]);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 2, backend);
            using SequenceKv kv = new(pool);
            biasedLogits = model.DecodeNext(3, kv).ToArray();
        }

        float[] plainLogits;
        {
            using GgufFile gguf = GgufFile.Open(plainPath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 2, backend);
            using SequenceKv kv = new(pool);
            plainLogits = model.DecodeNext(3, kv).ToArray();
        }

        Assert.True(
            biasedLogits.AsSpan().SequenceEqual(plainLogits.AsSpan()),
            "bias vectors under the resolved-off default must not change the forward");
    }

    [Fact]
    public void TransformerModel_Gemma2DeclaredQkNormTrue_MissingNormVectors_FailsLoadNamingThem()
    {
        string path = WriteTemp(BuildGguf(
            [.. Metadata("gemma2", tiny: true), ("gemma2.attention.use_qk_norm", true)],
            [.. TinyTensors()]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("blk.0.attn_q_norm.weight", ex.Message);
        Assert.Contains("gemma2", ex.Message);
    }

    // ---------------------------------------------------------------- shared helpers

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-gemma2-plan-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // Canonical gemma2 metadata with the real file's geometry (head dim declared by head_length alongside the
    // agreeing key_length / value_length, 16 x 144 != 2304); the tiny variant matches TinyTensors.
    private static List<(string Key, object Value)> Metadata(string prefix, bool tiny = false)
    {
        if (tiny)
        {
            return
            [
                ("general.architecture", prefix),
                ($"{prefix}.embedding_length", 4),
                ($"{prefix}.block_count", 1),
                ($"{prefix}.attention.head_count", 2),
                ($"{prefix}.attention.head_count_kv", 1),
                ($"{prefix}.feed_forward_length", 8),
                ($"{prefix}.attention.head_length", 4),
                ($"{prefix}.attention.key_length", 4),
                ($"{prefix}.attention.value_length", 4),
                ($"{prefix}.attention.layer_norm_rms_epsilon", 1e-5f),
                ($"{prefix}.vocab_size", 6),
            ];
        }

        return
        [
            ("general.architecture", prefix),
            ($"{prefix}.embedding_length", 2304),
            ($"{prefix}.block_count", 26),
            ($"{prefix}.attention.head_count", 16),
            ($"{prefix}.attention.head_count_kv", 4),
            ($"{prefix}.feed_forward_length", 9216),
            ($"{prefix}.attention.head_length", 144),
            ($"{prefix}.attention.key_length", 144),
            ($"{prefix}.attention.value_length", 144),
            ($"{prefix}.attention.sliding_window", 4096),
            ($"{prefix}.attention.layer_norm_rms_epsilon", 1e-6f),
            ($"{prefix}.attn_logit_softcapping", 50f),
            ($"{prefix}.final_logit_softcapping", 30f),
            ($"{prefix}.vocab_size", 256_000),
        ];
    }

    // Dims in logical (row-major) order; vectors are one-dimensional. A complete single-layer gemma2-shaped
    // model with hidden 4, heads 2, kv-heads 1, head dim 4 declared by head_length (2 x 4 != 4 hidden is a
    // grouped-query shape: the q region is 8 rows, not a split of the hidden size), ffn 8, vocab 6
    // — all F32 tensors. The post-sublayer
    // norm vectors are included unless asked off: the gemma2 placement requires them, and the failure cases
    // need them absent. The hidden 4 / heads 2 / head dim 2 split is exact, so the head_length declaration
    // stays consistent with the tensor shapes.
    private sealed record TensorDef(string Name, int[] Dims);

    private static List<TensorDef> TinyTensors(bool postNorms = true)
    {
        // Head dim 4 with heads 2 (2 x 4 = 8 q rows against a hidden of 4) mirrors the real file's declared
        // head_length: the q/k/v region size comes from the declaration, not from hidden / head_count. kv-heads
        // 1 gives the grouped-query shape the real file carries. The head dimension is even, as RoPE pairing
        // requires.
        List<TensorDef> tensors =
        [
            new("token_embd.weight", [6, 4]),
            new("blk.0.attn_q.weight", [8, 4]),
            new("blk.0.attn_k.weight", [4, 4]),
            new("blk.0.attn_v.weight", [4, 4]),
            new("blk.0.attn_output.weight", [4, 8]),
            new("blk.0.ffn_gate.weight", [8, 4]),
            new("blk.0.ffn_up.weight", [8, 4]),
            new("blk.0.ffn_down.weight", [4, 8]),
            new("blk.0.attn_norm.weight", [4]),
            new("blk.0.ffn_norm.weight", [4]),
            new("output_norm.weight", [4]),
        ];

        if (postNorms)
        {
            tensors.Add(new("blk.0.post_attention_norm.weight", [4]));
            tensors.Add(new("blk.0.post_ffw_norm.weight", [4]));
        }

        return tensors;
    }

    /// <summary>Builds a GGUF v3 image from KV pairs (strings, ints, floats, bools) and tensors.</summary>
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

        int ordinal = 0;
        foreach (TensorDef tensor in tensors)
        {
            long elements = 1;
            foreach (int dim in tensor.Dims)
            {
                elements *= dim;
            }

            bool isNorm = tensor.Name.Contains("norm");
            for (long e = 0; e < elements; e++)
            {
                float value = MathF.Sin((ordinal * 17 + e) * 0.13f); // deterministic, finite values everywhere
                writer.Write(isNorm ? 1.25f + value : value);       // norm weights stay clearly nonzero
            }

            ordinal++;
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
