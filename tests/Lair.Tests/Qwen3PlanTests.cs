using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Load-time resolution for the qwen3 profile and the vocabulary-size fallback: qwen3 profile selection and
/// its resolved geometry under the qwen3 prefix, the per-head QK-norm profile default (on for qwen3, off for
/// the other profiles, always overridable by a declaration), the missing-norm-tensor failure the default
/// produces at Open, vocabulary resolution from <c>&lt;prefix&gt;vocab_size</c> or the tokenizer vocabulary
/// length with its structured fallback and disagreement lines, and the fused qkv and quantization-companion
/// tensor rules — ignored with one structured line when the split map already fills every role, rejected when
/// a fused tensor is the only source of a role or a companion is not a plain float scale.
/// </summary>
[Trait("Category", "Loader")]
public sealed class Qwen3PlanTests : IDisposable
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

    // ---------------------------------------------------------------- qwen3 profile resolution

    [Fact]
    public void ModelPlan_Qwen3Metadata_ResolvesProfileGeometry_AndQkNormDefaultOn()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata("qwen3"), [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal("qwen3", plan.Architecture);
        Assert.Equal("qwen3.", plan.Prefix); // every key was read under the qwen3 prefix
        Assert.Equal(28, plan.LayerCount);
        Assert.Equal(1024, plan.HiddenSize);
        Assert.Equal(16, plan.HeadCount);
        Assert.Equal(8, plan.KvHeadCount);
        Assert.Equal(128, plan.HeadDim); // declared by key_length / value_length: 16 x 128 != 1024
        Assert.Equal(3072, plan.IntermediateSize);
        Assert.Equal(151_936, plan.VocabSize);
        Assert.True(Math.Abs(plan.RopeTheta - 1e6f) < 1f, $"rope theta {plan.RopeTheta:G9}");
        Assert.True(Math.Abs(plan.EpsRmsNorm - 1e-6f) < 1e-8f, $"rms epsilon {plan.EpsRmsNorm:G9}");

        Assert.Equal(NormStrategy.RmsNorm, plan.Norm);
        Assert.Equal(ActivationStrategy.SwiGLU, plan.Activation);
        Assert.Equal(BiasPolicy.Declared, plan.Bias); // bias stays declared-only for qwen3
        Assert.Equal(RopeStrategy.RotateHalfFullHead, plan.Rope); // qwen3 rotary rows pair the head halves

        // The profile default turns QK-norm on without any declaration; attention bias still defaults false, and no
        // registered profile places norms on sublayer outputs.
        Assert.Equal(QkNormMode.PerHeadWeighted, plan.QkNorm);
        Assert.Equal(NormPlacement.PreOnly, plan.NormPlacement);
        Assert.False(plan.UseAttentionBias);
        Assert.False(plan.UseNormBias);
    }

    [Fact]
    public void ModelPlan_Qwen3DeclaredQkNorm_OverridesTheProfileDefault_EitherWay()
    {
        foreach (bool declared in new bool[] { true, false })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
                [.. Metadata("qwen3"), ("qwen3.attention.use_qk_norm", declared)], [])));
            ModelPlan plan = ModelPlans.Resolve(gguf);

            Assert.Equal(declared ? QkNormMode.PerHeadWeighted : QkNormMode.Off, plan.QkNorm); // the declaration wins over the profile default in both directions
        }
    }

    [Fact]
    public void ModelPlan_QkNormProfileDefault_OffForLlamaQwen2_AndTheGenericFallback()
    {
        foreach (string architecture in new string[] { "llama", "qwen2" })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata(architecture, tiny: true), [])));
            Assert.Equal(QkNormMode.Off, ModelPlans.Resolve(gguf).QkNorm); // unchanged behavior for the existing profiles
            Assert.Equal(NormPlacement.PreOnly, ModelPlans.Resolve(gguf).NormPlacement); // and their pre-block-only norms
            Assert.Equal(RopeStrategy.StandardFullHead, ModelPlans.Resolve(gguf).Rope); // adjacent-pair RoPE stays theirs

            using GgufFile declared = GgufFile.Open(WriteTemp(BuildGguf(
                [.. Metadata(architecture, tiny: true), ($"{architecture}.attention.use_qk_norm", true)], [])));
            Assert.Equal(QkNormMode.PerHeadWeighted, ModelPlans.Resolve(declared).QkNorm); // a declaration still turns it on there too
        }

        using GgufFile fallback = GgufFile.Open(WriteTemp(BuildGguf(
            Metadata("myarch", tiny: true), [.. TinyTensors()])));
        Assert.Equal(QkNormMode.Off, ModelPlans.Resolve(fallback).QkNorm); // the generic llama-shaped profile keeps its current defaults
        Assert.Equal(RopeStrategy.StandardFullHead, ModelPlans.Resolve(fallback).Rope); // and its adjacent-pair RoPE
    }

    [Fact]
    public void ModelPlan_Qwen3QkNormDefault_MissingNormTensors_FailsLoadNamingThem_AndTheArchitecture()
    {
        string path = WriteTemp(BuildGguf(Metadata("qwen3", tiny: true), [.. TinyTensors(withQkNorm: false)]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("blk.0.attn_q_norm.weight", ex.Message); // the profile default makes the vectors required at Open
        Assert.Contains("qwen3", ex.Message);
    }

    [Fact]
    public void ModelPlan_Qwen3DeclaredQkNormFalse_WithNormTensorsPresent_LogsTheIgnoredVectorsOnce()
    {
        List<string> logLines = [];
        string path = WriteTemp(BuildGguf(
            [.. Metadata("qwen3", tiny: true), ("general.name", "No Norm"), ("qwen3.attention.use_qk_norm", false)],
            [.. TinyTensors()]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, logLines.Add);

        Assert.Equal(QkNormMode.Off, model.Plan.QkNorm);
        Assert.Single(logLines);
        Assert.StartsWith("qk_norm_ignored ", logLines[0]);
        Assert.Contains("model='No Norm'", logLines[0]);
        Assert.Contains("architecture='qwen3'", logLines[0]);
        Assert.Contains("blk.0.attn_q_norm.weight", logLines[0]);
        Assert.Contains("blk.0.attn_k_norm.weight", logLines[0]);
    }

    // ---------------------------------------------------------------- vocabulary-size fallback

    [Fact]
    public void ModelPlan_VocabSizeDeclared_ResolvesIt_AndLogsNothing_EvenWhenTokensAgree()
    {
        List<string> logLines = [];
        using GgufFile declaredOnly = GgufFile.Open(WriteTemp(BuildGguf(Metadata("qwen3", tiny: true), [])));
        Assert.Equal(6, ModelPlans.Resolve(declaredOnly, logLines.Add).VocabSize);
        Assert.Empty(logLines); // a declared key needs no fallback line

        // Declared and token list present and equal: the declared key is used, nothing to report.
        using GgufFile agreeing = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("qwen3", tiny: true), TokensPair(6)], [])));
        Assert.Equal(6, ModelPlans.Resolve(agreeing, logLines.Add).VocabSize);
        Assert.Empty(logLines);
    }

    [Fact]
    public void ModelPlan_VocabSizeMissing_FallsBackToTokenCount_AndLogsOneStructuredLine()
    {
        List<string> logLines = [];
        List<(string Key, object Value)> pairs = [.. Metadata("qwen3", tiny: true), ("general.name", "Fallback Model")];
        pairs.RemoveAll(p => p.Key == "qwen3.vocab_size");

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf([.. pairs, TokensPair(9)], [])));
        ModelPlan plan = ModelPlans.Resolve(gguf, logLines.Add);

        Assert.Equal(9, plan.VocabSize); // the length of tokenizer.ggml.tokens
        Assert.Single(logLines);
        Assert.Equal("vocab_fallback model='Fallback Model' source=tokenizer.ggml.tokens count=9", logLines[0]);
    }

    [Fact]
    public void ModelPlan_VocabSizeDeclaredAndTokensDisagree_DeclaredWinsAndTheLineNamesBothSides()
    {
        List<string> logLines = [];
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("qwen3", tiny: true), ("general.name", "Disagreeing File"), TokensPair(9)], [])));
        ModelPlan plan = ModelPlans.Resolve(gguf, logLines.Add);

        Assert.Equal(6, plan.VocabSize); // the declared key wins
        Assert.Single(logLines);
        Assert.Equal("vocab_disagree model='Disagreeing File' declared=6 source=tokenizer.ggml.tokens count=9", logLines[0]);
    }

    [Fact]
    public void ModelPlan_VocabSizeAndTokensMissing_FailsNamingBothKeys()
    {
        List<(string Key, object Value)> pairs = [.. Metadata("qwen3", tiny: true)];
        pairs.RemoveAll(p => p.Key == "qwen3.vocab_size");

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf([.. pairs], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("qwen3.vocab_size", ex.Message);
        Assert.Contains("tokenizer.ggml.tokens", ex.Message);
    }

    // ---------------------------------------------------------------- fused qkv tensors

    [Fact]
    public void TransformerModel_FusedQkvAlongsideSplitTensors_IsIgnoredWithOneStructuredLine_AndForwardUnchanged()
    {
        List<string> logLines = [];
        string fusedPath = WriteTemp(BuildGguf(
            [.. Metadata("qwen3", tiny: true), ("general.name", "Fused Sibling")],
            [.. TinyTensors(), new TensorDef("blk.0.attn_qkv.weight", [12, 4])]));
        string plainPath = WriteTemp(BuildGguf(Metadata("qwen3", tiny: true), [.. TinyTensors()]));

        using CpuBackend backend = new();

        float[] fusedLogits;
        {
            using GgufFile gguf = GgufFile.Open(fusedPath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, logLines.Add);

            Assert.Single(logLines);
            Assert.StartsWith("tensors_ignored ", logLines[0]);
            Assert.Contains("model='Fused Sibling'", logLines[0]);
            Assert.Contains("architecture='qwen3'", logLines[0]);
            Assert.Contains("blk.0.attn_qkv.weight", logLines[0]);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 2, backend);
            using SequenceKv kv = new(pool);
            fusedLogits = model.DecodeNext(3, kv).ToArray();
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
            fusedLogits.AsSpan().SequenceEqual(plainLogits.AsSpan()),
            "a fused qkv tensor alongside the split tensors must not change the forward");
    }

    [Fact]
    public void TransformerModel_FusedQkvAsOnlySourceOfARole_FailsActionablyNamingBothSides()
    {
        List<TensorDef> tensors = [.. TinyTensors()];
        tensors.RemoveAll(t => t.Name == "blk.0.attn_q.weight"); // the fused tensor is the only q source now

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            Metadata("qwen3", tiny: true), [.. tensors, new TensorDef("blk.0.attn_qkv.weight", [12, 4])])));
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("blk.0.attn_q.weight", ex.Message);
        Assert.Contains("blk.0.attn_qkv.weight", ex.Message);
        Assert.Contains("fused", ex.Message);
    }

    // ---------------------------------------------------------------- quantization companions

    [Fact]
    public void TransformerModel_ScaleCompanionsOfNameableTensors_AreRecognized_AndForwardUnchanged()
    {
        string companionPath = WriteTemp(BuildGguf(Metadata("qwen3", tiny: true), [.. TinyTensors(),
            new TensorDef("blk.0.attn_q.weight.scale", [4]),
            new TensorDef("blk.0.attn_k.weight.scale", [4]),
            new TensorDef("blk.0.ffn_gate.weight.input_scale", [1])]));
        string plainPath = WriteTemp(BuildGguf(Metadata("qwen3", tiny: true), [.. TinyTensors()]));

        using CpuBackend backend = new();

        float[] companionLogits;
        {
            using GgufFile gguf = GgufFile.Open(companionPath);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 2, backend);
            using SequenceKv kv = new(pool);
            companionLogits = model.DecodeNext(3, kv).ToArray();
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
            companionLogits.AsSpan().SequenceEqual(plainLogits.AsSpan()),
            "float companions of nameable tensors are known-optional entries and must not change the forward");
    }

    [Fact]
    public void TransformerModel_ScaleCompanionInQuantizedFormat_FailsActionablyNamingIt()
    {
        // A .scale entry that is itself quantized would have to be decoded to be interpreted: a load failure, not a guess.
        byte[] opaquePayload = new byte[34];
        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata("qwen3", tiny: true), [.. TinyTensors(),
            new TensorDef("blk.0.attn_q.weight.scale", [32], GgmlTypeIdQ8_0, opaquePayload)])));
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("blk.0.attn_q.weight.scale", ex.Message);
        Assert.Contains("companion", ex.Message);
    }

    [Fact]
    public void ModelPlan_FallbackTable_ScaleCompanionOfNameableTensor_IsNotUnexpected_ButOfAnUnknownTensorIs()
    {
        // The llama-shaped fallback gate treats a companion of a tensor it can name as known-optional...
        using GgufFile known = GgufFile.Open(WriteTemp(BuildGguf(
            Metadata("myarch", tiny: true), [.. TinyTensors(withQkNorm: false), new TensorDef("blk.0.attn_q.weight.scale", [4])])));
        ModelPlan plan = ModelPlans.Resolve(known);
        Assert.Equal(ModelProfiles.GenericArchitecture, plan.Profile);

        // ...but a companion of a tensor the map cannot name is still an unexpected table entry.
        using GgufFile unknown = GgufFile.Open(WriteTemp(BuildGguf(
            Metadata("myarch", tiny: true), [.. TinyTensors(withQkNorm: false), new TensorDef("blk.0.v_proj.weight.scale", [4])])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(unknown));
        Assert.Contains("unexpected tensors: blk.0.v_proj.weight.scale", ex.Message);
    }

    // ---------------------------------------------------------------- shared helpers

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-qwen3-plan-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // Canonical qwen3 metadata with the real file's geometry (head dim declared by key_length / value_length,
    // 16 x 128 != 1024); the tiny variant matches TinyTensors.
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
                ($"{prefix}.attention.head_count_kv", 2),
                ($"{prefix}.feed_forward_length", 8),
                ($"{prefix}.attention.key_length", 2),
                ($"{prefix}.attention.value_length", 2),
                ($"{prefix}.attention.layer_norm_rms_epsilon", 1e-5f),
                ($"{prefix}.vocab_size", 6),
            ];
        }

        return
        [
            ("general.architecture", prefix),
            ($"{prefix}.embedding_length", 1024),
            ($"{prefix}.block_count", 28),
            ($"{prefix}.attention.head_count", 16),
            ($"{prefix}.attention.head_count_kv", 8),
            ($"{prefix}.feed_forward_length", 3072),
            ($"{prefix}.attention.key_length", 128),
            ($"{prefix}.attention.value_length", 128),
            ($"{prefix}.attention.layer_norm_rms_epsilon", 1e-6f),
            ($"{prefix}.rope.freq_base", 1e6f),
            ($"{prefix}.vocab_size", 151_936),
        ];
    }

    // A tokenizer.ggml.tokens array of the given length, for the vocabulary-fallback cases.
    private static (string Key, object Value) TokensPair(int count)
    {
        string[] tokens = new string[count];
        for (int i = 0; i < count; i++)
        {
            tokens[i] = $"tok{i}";
        }

        return ("tokenizer.ggml.tokens", tokens);
    }

    private const uint GgmlTypeIdQ8_0 = 8;

    // Dims in logical (row-major) order; vectors are one-dimensional. A non-F32 tensor carries its own payload bytes.
    private sealed record TensorDef(string Name, int[] Dims, uint GgmlType = 0, byte[]? RawPayload = null);

    // A complete single-layer model with hidden 4, heads 2, kv-heads 2, head dim 2, ffn 8, vocab 6 — all F32
    // tensors, the per-head QK-norm vectors included unless asked off.
    private static List<TensorDef> TinyTensors(bool withQkNorm = true)
    {
        List<TensorDef> tensors =
        [
            new("token_embd.weight", [6, 4]),
            new("blk.0.attn_q.weight", [4, 4]),
            new("blk.0.attn_k.weight", [4, 4]),
            new("blk.0.attn_v.weight", [4, 4]),
            new("blk.0.attn_output.weight", [4, 4]),
            new("blk.0.ffn_gate.weight", [8, 4]),
            new("blk.0.ffn_up.weight", [8, 4]),
            new("blk.0.ffn_down.weight", [4, 8]),
            new("blk.0.attn_norm.weight", [4]),
            new("blk.0.ffn_norm.weight", [4]),
            new("output_norm.weight", [4]),
        ];

        if (withQkNorm)
        {
            tensors.Add(new("blk.0.attn_q_norm.weight", [2]));
            tensors.Add(new("blk.0.attn_k_norm.weight", [2]));
        }

        return tensors;
    }

    /// <summary>Builds a GGUF v3 image from KV pairs (strings, ints, floats, bools, string arrays) and tensors.</summary>
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
                case string[] array:
                    writer.Write(9u); // ARRAY
                    writer.Write(8u); // element type STRING
                    writer.Write((ulong)array.Length);
                    foreach (string element in array)
                    {
                        WriteGgufString(writer, element);
                    }

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

            writer.Write(tensor.GgmlType);
            writer.Write((ulong)offset);

            long elements = 1;
            foreach (int dim in tensor.Dims)
            {
                elements *= dim;
            }

            long payloadBytes = tensor.GgmlType == 0 ? elements * 4 : (tensor.RawPayload?.Length ?? 0);
            offset += payloadBytes;
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
            if (tensor.GgmlType != 0)
            {
                writer.Write(tensor.RawPayload ?? []);
                ordinal++;
                continue;
            }

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
