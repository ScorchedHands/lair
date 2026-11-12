using System.Text;

using Lair.Loader;
using Lair.Models;

namespace Lair.Tests;

/// <summary>
/// Load-time resolution of the declared rope scaling and rotary slice: an absent <c>rope.scaling.type</c> keeps
/// the unscaled plan (<see cref="ModelPlan.RopeScaling"/> null, <see cref="ModelPlan.RopeRotaryDimensions"/> zero)
/// for every profile including the generic llama-shaped fallback, a declared <c>yarn</c> variant resolves the
/// multiplier, original context length, correction scale, beta pair, and attention factor exactly — with the
/// reference defaults standing in when the operand keys are absent and the reference's magnitude cancellation
/// holding when no log multiplier is declared — while any other variant name, a scaling operand without a variant
/// declaration, an invalid factor or magnitude, and a missing original context length each fail the load naming
/// the key and the declared value. The slice resolves <c>rope.dimension_count</c>: absent means zero (full head),
/// a positive even count not exceeding the head dimension lands in the plan, and an odd, zero, negative,
/// too-large, or text declaration fails naming the key and the value. Keys under another profile's prefix are
/// not the plan's keys.
/// </summary>
[Trait("Category", "Loader")]
public sealed class RopeScalingPlanTests : IDisposable
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
    public void ModelPlan_AbsentScalingKeys_ResolvesUnscaled_ForEveryProfile()
    {
        foreach (string architecture in new string[] { "llama", "qwen2", "qwen3" })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata(architecture), [])));
            ModelPlan plan = ModelPlans.Resolve(gguf);

            Assert.Null(plan.RopeScaling);
            Assert.Equal(0, plan.RopeRotaryDimensions);
        }

        // The generic llama-shaped fallback resolves the same absence under the declared architecture's prefix.
        using GgufFile fallback = GgufFile.Open(WriteTemp(BuildGguf(Metadata("myarch", tiny: true), TinyTensors())));
        ModelPlan fallbackPlan = ModelPlans.Resolve(fallback);
        Assert.Equal(ModelProfiles.GenericArchitecture, fallbackPlan.Profile);
        Assert.Null(fallbackPlan.RopeScaling);
    }

    [Fact]
    public void ModelPlan_DeclaredYarnVariant_ResolvesTheReferenceDefaults_UnderTheProfilePrefix()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.rope.scaling.type", "yarn"), ("llama.rope.scaling.factor", 4f), ("llama.rope.scaling.original_context_length", 8192)], [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.NotNull(plan.RopeScaling);
        Assert.Equal("yarn", plan.RopeScaling.Variant);
        Assert.Equal(4f, plan.RopeScaling.Multiplier);
        Assert.Equal(8192, plan.RopeScaling.OriginalContextLength);
        Assert.Equal(1f, plan.RopeScaling.CorrectionScale); // the reference's YaRN default
        Assert.Equal(32f, plan.RopeScaling.BetaFast);
        Assert.Equal(1f, plan.RopeScaling.BetaSlow);

        // With no declared log multiplier the variant's magnitude scale cancels exactly against the one the
        // angle tables fold in, so the resolved attention factor equals the default declared factor.
        Assert.True(Math.Abs(plan.RopeScaling.AttentionFactor - 1f) <= 1e-6, $"resolved factor {plan.RopeScaling.AttentionFactor:G9}");

        // The fallback profile resolves the declaration under the declared architecture's own prefix.
        using GgufFile fallback = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("myarch", tiny: true), ("myarch.rope.scaling.type", "yarn"), ("myarch.rope.scaling.original_context_length", 2048)], TinyTensors())));
        ModelPlan fallbackPlan = ModelPlans.Resolve(fallback);
        Assert.Equal(ModelProfiles.GenericArchitecture, fallbackPlan.Profile);
        Assert.NotNull(fallbackPlan.RopeScaling);
        Assert.Equal(1f, fallbackPlan.RopeScaling.Multiplier); // an absent factor key means the unscaled multiplier
        Assert.Equal(2048, fallbackPlan.RopeScaling.OriginalContextLength);
    }

    [Fact]
    public void ModelPlan_DeclaredYarnOperands_ResolveTheReferenceAttentionFactor()
    {
        const double Multiplier = 4.0;
        const float DeclaredAttention = 1.5f;
        const double LogMultiplier = 0.7;

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.rope.scaling.type", "yarn"), ("llama.rope.scaling.factor", (float)Multiplier), ("llama.rope.scaling.original_context_length", 8192), ("llama.rope.scaling.attn_factor", DeclaredAttention), ("llama.rope.scaling.yarn_log_multiplier", LogMultiplier), ("llama.rope.scaling.yarn_ext_factor", 0.5f), ("llama.rope.scaling.yarn_beta_fast", 16f), ("llama.rope.scaling.yarn_beta_slow", 2f)], [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.NotNull(plan.RopeScaling);
        Assert.Equal(0.5f, plan.RopeScaling.CorrectionScale);
        Assert.Equal(16f, plan.RopeScaling.BetaFast);
        Assert.Equal(2f, plan.RopeScaling.BetaSlow);

        // Reference formula: getMscale(m, 1) / getMscale(m, s), divided by the magnitude scale the tables fold
        // in (1 + 0.1 * ln m), then multiplied by the declared attention factor.
        double mscale(double scale, double magnitude) => scale <= 1.0 ? 1.0 : 0.1 * magnitude * Math.Log(scale) + 1.0;
        double expected = mscale(Multiplier, 1.0) / mscale(Multiplier, LogMultiplier) / (1.0 + 0.1 * Math.Log(Multiplier)) * DeclaredAttention;

        Assert.True(
            Math.Abs(plan.RopeScaling.AttentionFactor - expected) <= 1e-6 * Math.Max(1.0, expected),
            $"attention factor: expected {expected:G9}, got {plan.RopeScaling.AttentionFactor:G9}");
    }

    [Fact]
    public void ModelPlan_NoneVariant_ResolvesUnscaled_EvenWithOperandsPresent()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.rope.scaling.type", "none"), ("llama.rope.scaling.factor", 4f), ("llama.rope.scaling.original_context_length", 8192)], [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Null(plan.RopeScaling);
    }

    [Theory]
    [InlineData("linear")]
    [InlineData("longrope")]
    [InlineData("dynamic")]
    [InlineData("yarnx")]
    public void ModelPlan_UnsupportedVariantDeclaration_FailsLoadNamingKeyAndValue(string variant)
    {
        foreach (string prefix in new string[] { "llama", "qwen2", "qwen3" })
        {
            string key = prefix + ".rope.scaling.type";
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf([.. Metadata(prefix), (key, variant)], [])));
            ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

            Assert.Contains(key, ex.Message);
            Assert.Contains(variant, ex.Message);
        }
    }

    [Fact]
    public void ModelPlan_ScalingOperandWithoutVariant_FailsLoadNamingTheDeclaredKey()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.rope.scaling.factor", 4f)], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("llama.rope.scaling.factor", ex.Message);
    }

    [Fact]
    public void ModelPlan_YarnWithoutOriginalContextLength_FailsLoadNamingTheKey()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.rope.scaling.type", "yarn")], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("llama.rope.scaling.original_context_length", ex.Message);
    }

    [Theory]
    [InlineData(0, "0")]
    [InlineData(-2f, "-2")]
    public void ModelPlan_InvalidScalingFactor_FailsLoadNamingKeyAndValue(float declared, string valueText)
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.rope.scaling.type", "yarn"), ("llama.rope.scaling.original_context_length", 8192), ("llama.rope.scaling.factor", declared)], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("llama.rope.scaling.factor", ex.Message);
        Assert.Contains(valueText, ex.Message);
    }

    [Fact]
    public void ModelPlan_TextScalingFactor_FailsLoadNamingKeyAndValue()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.rope.scaling.type", "yarn"), ("llama.rope.scaling.original_context_length", 8192), ("llama.rope.scaling.factor", "big")], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("llama.rope.scaling.factor", ex.Message);
        Assert.Contains("big", ex.Message);
    }

    [Theory]
    [InlineData(-1f, "-1")]
    public void ModelPlan_NegativeYarnMagnitude_FailsLoadNamingKeyAndValue(float declared, string valueText)
    {
        foreach (string key in new string[]
        {
            "llama.rope.scaling.attn_factor",
            "llama.rope.scaling.yarn_log_multiplier",
            "llama.rope.scaling.yarn_ext_factor",
            "llama.rope.scaling.yarn_beta_fast",
            "llama.rope.scaling.yarn_beta_slow",
        })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
                [.. Metadata("llama"), ("llama.rope.scaling.type", "yarn"), ("llama.rope.scaling.original_context_length", 8192), (key, declared)], [])));
            ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

            Assert.Contains(key, ex.Message);
            Assert.Contains(valueText, ex.Message);
        }
    }

    [Fact]
    public void ModelPlan_ZeroCorrectionScale_KeepsTheDeclaredAttentionFactor()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.rope.scaling.type", "yarn"), ("llama.rope.scaling.original_context_length", 8192), ("llama.rope.scaling.factor", 4f), ("llama.rope.scaling.attn_factor", 1.5f), ("llama.rope.scaling.yarn_ext_factor", 0f)], [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.NotNull(plan.RopeScaling);
        Assert.Equal(0f, plan.RopeScaling.CorrectionScale);
        Assert.Equal(1.5f, plan.RopeScaling.AttentionFactor); // the variant's magnitude rebuild is skipped
    }

    [Fact]
    public void ModelPlan_RotarySlice_ResolvesUnderTheProfilePrefix()
    {
        // Absent means the full head; a declared even count not exceeding it lands in the plan, including one
        // exactly equal to the head dimension.
        using GgufFile absent = GgufFile.Open(WriteTemp(BuildGguf(Metadata("llama"), [])));
        Assert.Equal(0, ModelPlans.Resolve(absent).RopeRotaryDimensions);

        using GgufFile partial = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.rope.dimension_count", 32)], [])));
        Assert.Equal(32, ModelPlans.Resolve(partial).RopeRotaryDimensions);

        using GgufFile exact = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.rope.dimension_count", 64)], []))); // 64 == 576 / 9
        Assert.Equal(64, ModelPlans.Resolve(exact).RopeRotaryDimensions);

        // The key belongs to the prefix: the same declaration under the other profile's prefix is not seen.
        using GgufFile wrongPrefix = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("qwen2.rope.dimension_count", 32)], [])));
        Assert.Equal(0, ModelPlans.Resolve(wrongPrefix).RopeRotaryDimensions);

        // A scaling declaration under the other profile's prefix is equally not the plan's key.
        using GgufFile wrongScalingPrefix = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("qwen2.rope.scaling.type", "yarn")], [])));
        Assert.Null(ModelPlans.Resolve(wrongScalingPrefix).RopeScaling);
    }

    [Theory]
    [InlineData(0, "0")]
    [InlineData(-2, "-2")]
    [InlineData(31, "31")]        // an odd slice cannot form pairs
    [InlineData(66, "66")]        // a slice wider than the head dimension is out of range
    [InlineData(3000000000L, "3000000000")] // an int64 declaration past the int32 range cannot become a slice
    public void ModelPlan_InvalidRotarySlice_FailsLoadNamingKeyAndValue(object declared, string valueText)
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.rope.dimension_count", declared)], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("llama.rope.dimension_count", ex.Message);
        Assert.Contains(valueText, ex.Message);
    }

    [Fact]
    public void ModelPlan_TextRotarySlice_FailsLoadNamingKeyAndValue()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.rope.dimension_count", "half")], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("llama.rope.dimension_count", ex.Message);
        Assert.Contains("half", ex.Message);
    }

    // ---------------------------------------------------------------- shared helpers

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-rope-plan-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // Canonical metadata pairs per architecture (parse-only fixtures need no tensors); the tiny variant keeps
    // the geometry small for the fallback fixtures.
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
                ($"{prefix}.attention.layer_norm_rms_epsilon", 1e-5f),
                ($"{prefix}.vocab_size", 6),
            ];
        }

        return
        [
            ("general.architecture", prefix),
            ($"{prefix}.embedding_length", 576),
            ($"{prefix}.block_count", 30),
            ($"{prefix}.attention.head_count", 9),
            ($"{prefix}.attention.head_count_kv", 3),
            ($"{prefix}.feed_forward_length", 1536),
            ($"{prefix}.attention.layer_norm_rms_epsilon", 1e-5f),
            ($"{prefix}.vocab_size", 49_152),
        ];
    }

    // A complete single-layer llama-shaped tensor table the generic fallback gate accepts; payloads are
    // irrelevant to plan resolution.
    private static List<TensorDef> TinyTensors() =>
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

    /// <summary>Builds a GGUF v3 image from KV pairs; no tensors are needed for parse-only resolution.</summary>
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
                case uint u:
                    writer.Write(4u); // UINT32
                    writer.Write(u);
                    break;
                case long l:
                    writer.Write(11u); // INT64
                    writer.Write(l);
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

            for (long e = 0; e < elements; e++)
            {
                writer.Write(0f);
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

    private sealed record TensorDef(string Name, int[] Dims); // dims in logical (row-major) order; vectors are one-dimensional
}
