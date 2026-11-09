using System.Text;

using Lair.Loader;
using Lair.Models;

namespace Lair.Tests;

/// <summary>
/// Load-time resolution of the two logit caps: absent keys resolve to the disabled zero, declared values
/// resolve to the exact number through the shared numeric read (float32, float64, and integer declarations),
/// and a non-positive, NaN, infinite, or out-of-float-range declaration fails the load with a message naming
/// the key and the declared value. The registered profiles and the generic llama-shaped fallback all resolve
/// the caps from the profile's own prefix, and the flat scalar view passes both values through.
/// </summary>
[Trait("Category", "Loader")]
public sealed class SoftcapPlanTests : IDisposable
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
    public void ModelPlan_AbsentCapKeys_ResolveBothCapsDisabled_ForBothProfiles()
    {
        foreach (string architecture in new string[] { "llama", "qwen2" })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata(architecture), [])));
            ModelPlan plan = ModelPlans.Resolve(gguf);

            Assert.Equal(0f, plan.AttentionSoftcap);
            Assert.Equal(0f, plan.FinalLogitSoftcap);
        }
    }

    [Fact]
    public void ModelPlan_DeclaredCapKeys_ResolveTheExactNumber_UnderTheProfilePrefix()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.attn_logit_softcapping", 50f), ("llama.final_logit_softcapping", 30f)], [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal(50f, plan.AttentionSoftcap);
        Assert.Equal(30f, plan.FinalLogitSoftcap);

        // The keys belong to the prefix: the same declarations under the other profile's prefix are not seen.
        using GgufFile qwen = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("qwen2"), ("qwen2.attn_logit_softcapping", 50f), ("qwen2.final_logit_softcapping", 30f)], [])));
        ModelPlan qwenPlan = ModelPlans.Resolve(qwen);
        Assert.Equal(50f, qwenPlan.AttentionSoftcap);
        Assert.Equal(30f, qwenPlan.FinalLogitSoftcap);

        using GgufFile wrongPrefix = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("qwen2.attn_logit_softcapping", 50f)], [])));
        ModelPlan wrongPlan = ModelPlans.Resolve(wrongPrefix);
        Assert.Equal(0f, wrongPlan.AttentionSoftcap); // a foreign-prefix key is not the plan's key

        // Each cap resolves independently: declaring one leaves the other disabled.
        using GgufFile onlyAttention = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.attn_logit_softcapping", 50f)], [])));
        Assert.Equal(50f, ModelPlans.Resolve(onlyAttention).AttentionSoftcap);
        Assert.Equal(0f, ModelPlans.Resolve(onlyAttention).FinalLogitSoftcap);

        using GgufFile onlyFinal = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.final_logit_softcapping", 30f)], [])));
        Assert.Equal(0f, ModelPlans.Resolve(onlyFinal).AttentionSoftcap);
        Assert.Equal(30f, ModelPlans.Resolve(onlyFinal).FinalLogitSoftcap);
    }

    [Theory]
    [InlineData(50f)]
    [InlineData(30f)]
    [InlineData(0.25f)]
    [InlineData(1e-4f)]
    [InlineData(1e6f)]
    [InlineData(3.4028234663852886e+38)] // the largest float32 value: still a valid finite cap
    public void ModelPlan_DeclaredFloat32Cap_ResolvesExactly(float declared)
    {
        foreach (string key in new string[] { "llama.attn_logit_softcapping", "llama.final_logit_softcapping" })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf([.. Metadata("llama"), (key, declared)], [])));
            ModelPlan plan = ModelPlans.Resolve(gguf);

            float resolved = key.EndsWith("attn_logit_softcapping", StringComparison.Ordinal)
                ? plan.AttentionSoftcap
                : plan.FinalLogitSoftcap;
            Assert.Equal(declared, resolved);
        }
    }

    [Theory]
    [InlineData(20.5)] // float64 declaration
    [InlineData(40L)] // integer declarations resolve as their value
    [InlineData(1234567)]
    public void ModelPlan_NonFloat32NumericDeclarations_ResolveTheirValue(object declared)
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.attn_logit_softcapping", declared)], [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal((float)(double)Convert.ChangeType(declared, typeof(double)), plan.AttentionSoftcap);
    }

    [Theory]
    [InlineData(0f, "0")]
    [InlineData(-0f, "-0")]
    [InlineData(-1f, "-1")]
    [InlineData(-0.5f, "-0.5")]
    [InlineData(float.NaN, "NaN")]
    [InlineData(float.PositiveInfinity, "Infinity")]
    [InlineData(float.NegativeInfinity, "-Infinity")]
    public void ModelPlan_InvalidCapDeclaration_FailsLoadNamingKeyAndValue(float declared, string valueText)
    {
        foreach (string key in new string[] { "llama.attn_logit_softcapping", "llama.final_logit_softcapping" })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf([.. Metadata("llama"), (key, declared)], [])));
            ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

            Assert.Contains(key, ex.Message);
            Assert.Contains(valueText, ex.Message);
        }
    }

    [Fact]
    public void ModelPlan_OutOfFloatRangeCapDeclaration_FailsLoadNamingKeyAndValue()
    {
        // A float64 declaration that overflows the float32 cap to infinity is as unusable as an infinite one.
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.attn_logit_softcapping", 1e300)], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("llama.attn_logit_softcapping", ex.Message);
        Assert.Contains("1E+300", ex.Message);
    }

    [Fact]
    public void ModelPlan_NonNumericCapDeclaration_FailsLoadNamingTheKey()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.final_logit_softcapping", "thirty")], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("llama.final_logit_softcapping", ex.Message);
    }

    [Fact]
    public void ModelPlan_GenericFallback_ResolvesCapsUnderTheDeclaredPrefix_AndReportsDisabledWhenAbsent()
    {
        // Absent under the declared prefix: both caps disabled on a fallback plan.
        using (GgufFile absent = GgufFile.Open(WriteTemp(BuildGguf(Metadata("myarch", tiny: true), TinyTensors()))))
        {
            ModelPlan plan = ModelPlans.Resolve(absent);
            Assert.Equal(ModelProfiles.GenericArchitecture, plan.Profile);
            Assert.Equal(0f, plan.AttentionSoftcap);
            Assert.Equal(0f, plan.FinalLogitSoftcap);
        }

        // Declared under the declared prefix: the fallback plan resolves them like any other profile.
        using GgufFile declared = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("myarch", tiny: true), ("myarch.attn_logit_softcapping", 50f), ("myarch.final_logit_softcapping", 30f)], TinyTensors())));
        ModelPlan declaredPlan = ModelPlans.Resolve(declared);
        Assert.Equal(50f, declaredPlan.AttentionSoftcap);
        Assert.Equal(30f, declaredPlan.FinalLogitSoftcap);
    }

    [Fact]
    public void ModelDesc_PassesBothCapValuesThrough_AndMatchesThePlanView()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.attn_logit_softcapping", 50f), ("llama.final_logit_softcapping", 30f)], [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);
        ModelDesc desc = plan.ToDesc();

        Assert.Equal(50f, desc.AttentionSoftcap);
        Assert.Equal(30f, desc.FinalLogitSoftcap);
        Assert.Equal(plan.ToDesc(), ModelDesc.FromGguf(gguf));

        // Disabled caps surface as zero in the flat view too.
        using GgufFile absent = GgufFile.Open(WriteTemp(BuildGguf(Metadata("llama"), [])));
        ModelDesc absentDesc = ModelDesc.FromGguf(absent);
        Assert.Equal(0f, absentDesc.AttentionSoftcap);
        Assert.Equal(0f, absentDesc.FinalLogitSoftcap);
    }

    // ---------------------------------------------------------------- shared helpers

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-softcap-plan-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // Canonical llama/qwen2 metadata pairs (parse-only fixtures need no tensors); the tiny variant keeps the
    // geometry small for the fallback fixtures.
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

    // A complete single-layer llama-shaped tensor table (hidden 4, heads 2, kv-heads 2, ffn 8, vocab 6) the
    // generic fallback gate accepts; payloads are irrelevant to plan resolution.
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
