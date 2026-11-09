using System.Text;

using Lair.Backends.Cpu;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Load-time resolution for norm placement and the per-head QK-norm mode: every registered profile and the generic
/// llama-shaped fallback resolves a pre-only placement and keeps its current QK-norm mode (weighted for qwen3, off
/// elsewhere), a declared <c>attention.use_qk_norm=false</c> forces the mode off against a weighted profile default,
/// and the two post-sublayer norm tensors are load-time decisive in both directions — a file carrying one under a
/// pre-only placement fails naming every such tensor, and a placement that norms sublayer outputs fails naming the one
/// its layers lack. An unweighted per-head mode demands no norm tensor at all and reports a file's present vectors as
/// unused rather than treating them as a requirement.
/// </summary>
[Trait("Category", "Loader")]
public sealed class ResidualNormPlanTests : IDisposable
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

    // ---------------------------------------------------------------- placement and mode resolution

    [Fact]
    public void ModelPlan_EveryProfile_ResolvesAPreOnlyPlacement_AndKeepsItsCurrentQkNormMode()
    {
        foreach (string architecture in new string[] { "llama", "qwen2" })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata(architecture), [])));
            ModelPlan plan = ModelPlans.Resolve(gguf);

            Assert.Equal(NormPlacement.PreOnly, plan.NormPlacement);
            Assert.Equal(QkNormMode.Off, plan.QkNorm);
        }

        using (GgufFile qwen3 = GgufFile.Open(WriteTemp(BuildGguf(Metadata("qwen3"), []))))
        {
            ModelPlan plan = ModelPlans.Resolve(qwen3);

            Assert.Equal(NormPlacement.PreOnly, plan.NormPlacement);
            Assert.Equal(QkNormMode.PerHeadWeighted, plan.QkNorm); // the qwen3 profile default stays weighted
        }

        using GgufFile fallback = GgufFile.Open(WriteTemp(BuildGguf(
            Metadata("myarch"), [.. TinyTensors(withQkNorm: false)])));
        ModelPlan fallbackPlan = ModelPlans.Resolve(fallback);

        Assert.Equal(ModelProfiles.GenericArchitecture, fallbackPlan.Profile);
        Assert.Equal(NormPlacement.PreOnly, fallbackPlan.NormPlacement);
        Assert.Equal(QkNormMode.Off, fallbackPlan.QkNorm);
    }

    [Fact]
    public void ModelPlan_Qwen3DeclaredUseQkNormFalse_ForcesOff_AgainstTheWeightedProfileDefault()
    {
        using (GgufFile declaredOff = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("qwen3"), ("qwen3.attention.use_qk_norm", false)], []))))
        {
            ModelPlan plan = ModelPlans.Resolve(declaredOff);

            Assert.Equal(QkNormMode.Off, plan.QkNorm); // the declaration overrides the profile default
            Assert.Equal(NormPlacement.PreOnly, plan.NormPlacement); // and no declaration moves the placement
        }

        using (GgufFile declaredOn = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("qwen3"), ("qwen3.attention.use_qk_norm", true)], []))))
        {
            Assert.Equal(QkNormMode.PerHeadWeighted, ModelPlans.Resolve(declaredOn).QkNorm);
        }

        using GgufFile absent = GgufFile.Open(WriteTemp(BuildGguf(Metadata("qwen3"), [])));
        Assert.Equal(QkNormMode.PerHeadWeighted, ModelPlans.Resolve(absent).QkNorm); // the profile default stands
    }

    // ---------------------------------------------------------------- post-sublayer tensors under a pre-only placement

    [Fact]
    public void TransformerModel_PostNormTensorsUnderAPreOnlyProfile_FailLoadNamingEveryOneOfThem()
    {
        using CpuBackend backend = new();

        // Both tensors present: the load fails naming each one and the placement that reads neither.
        using (GgufFile both = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("general.name", "Post Norms")],
            [.. TinyTensors(withQkNorm: false), new TensorDef("blk.0.post_attention_norm.weight", [4]), new TensorDef("blk.0.post_ffw_norm.weight", [4])]))))
        {
            ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(both, backend, maxContextTokens: 8));

            Assert.Contains("blk.0.post_attention_norm.weight", ex.Message);
            Assert.Contains("blk.0.post_ffw_norm.weight", ex.Message);
            Assert.Contains("PreOnly", ex.Message);
            Assert.Contains("llama", ex.Message);
        }

        // Only one of them present: still a layout the profile does not implement, and the tensor is named.
        using GgufFile one = GgufFile.Open(WriteTemp(BuildGguf(
            Metadata("qwen2"),
            [.. TinyTensors(withQkNorm: false), new TensorDef("blk.0.post_ffw_norm.weight", [4])])));
        ModelException singleEx = Assert.Throws<ModelException>(() => TransformerModel.Open(one, backend, maxContextTokens: 8));

        Assert.Contains("blk.0.post_ffw_norm.weight", singleEx.Message);
        Assert.Contains("PreOnly", singleEx.Message);
        Assert.Contains("qwen2", singleEx.Message);
    }

    [Fact]
    public void TransformerModel_PostNormTensorsUnderTheGenericFallback_FailLoadNamingThem()
    {
        // The llama-shaped table gate can name these tensors, so acceptance proceeds to the engine — which reads none
        // of them under a pre-only placement and fails rather than forwarding a graph it cannot express.
        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            Metadata("myarch"),
            [.. TinyTensors(withQkNorm: false), new TensorDef("blk.0.post_attention_norm.weight", [4])])));

        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("blk.0.post_attention_norm.weight", ex.Message);
        Assert.Contains("PreOnly", ex.Message);
    }

    // ---------------------------------------------------------------- a placement that consumes them

    [Fact]
    public void TransformerModel_PostSublayerOutputPlacement_MissingPostNormTensor_FailsNamingIt()
    {
        ModelProfile postProfile = ModelProfiles.Resolve("llama") with { NormPlacement = NormPlacement.PostSublayerOutputs };

        using CpuBackend backend = new();

        // The feed-forward side is absent: the failure names that tensor and the placement that needs it.
        using (GgufFile missingFeedForward = GgufFile.Open(WriteTemp(BuildGguf(
            Metadata("llama"),
            [.. TinyTensors(withQkNorm: false), new TensorDef("blk.0.post_attention_norm.weight", [4])]))))
        {
            ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(missingFeedForward, backend, maxContextTokens: 8, postProfile));

            Assert.Contains("blk.0.post_ffw_norm.weight", ex.Message);
            Assert.DoesNotContain("blk.0.post_attention_norm.weight", ex.Message); // the tensor that is present is not the failure
            Assert.Contains("PostSublayerOutputs", ex.Message);
        }

        // The attention side is absent: the failure names that one instead.
        using GgufFile missingAttention = GgufFile.Open(WriteTemp(BuildGguf(
            Metadata("llama"),
            [.. TinyTensors(withQkNorm: false), new TensorDef("blk.0.post_ffw_norm.weight", [4])])));
        ModelException attentionEx = Assert.Throws<ModelException>(() => TransformerModel.Open(missingAttention, backend, maxContextTokens: 8, postProfile));

        Assert.Contains("blk.0.post_attention_norm.weight", attentionEx.Message);
        Assert.Contains("PostSublayerOutputs", attentionEx.Message);
    }

    [Fact]
    public void TransformerModel_PostSublayerOutputPlacement_WithEveryTensor_OpensWithThatPlacementResolved()
    {
        ModelProfile postProfile = ModelProfiles.Resolve("llama") with { NormPlacement = NormPlacement.PostSublayerOutputs };

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            Metadata("llama"),
            [.. TinyTensors(withQkNorm: false), new TensorDef("blk.0.post_attention_norm.weight", [4]), new TensorDef("blk.0.post_ffw_norm.weight", [4])])));
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, postProfile);

        Assert.Equal(NormPlacement.PostSublayerOutputs, model.Plan.NormPlacement);
        Assert.Equal(QkNormMode.Off, model.Plan.QkNorm); // the placement says nothing about the QK-norm mode
    }

    // ---------------------------------------------------------------- the unweighted mode demands nothing

    [Fact]
    public void TransformerModel_UnweightedQkNorm_LoadsWithoutAnyNormTensor_AndReportsPresentOnesAsUnused()
    {
        ModelProfile unweightedProfile = ModelProfiles.Resolve("llama") with { DefaultQkNorm = QkNormMode.PerHeadUnweighted };

        using CpuBackend backend = new();

        // No per-head norm tensors in the table: the mode needs none, so the file loads.
        using (GgufFile without = GgufFile.Open(WriteTemp(BuildGguf(Metadata("llama"), [.. TinyTensors(withQkNorm: false)]))))
        {
            List<string> quietLines = [];
            using TransformerModel model = TransformerModel.Open(without, backend, maxContextTokens: 8, unweightedProfile, quietLines.Add);

            Assert.Equal(QkNormMode.PerHeadUnweighted, model.Plan.QkNorm);
            Assert.Empty(quietLines); // nothing present, nothing to report
        }

        // Present vectors are not a requirement this mode can read: they are reported as unused, in the style of the
        // other ignored-tensor lines, and the model still loads.
        List<string> logLines = [];
        using GgufFile withVectors = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("general.name", "Unused Norms")],
            [.. TinyTensors(withQkNorm: true)])));
        using (TransformerModel model = TransformerModel.Open(withVectors, backend, maxContextTokens: 8, unweightedProfile, logLines.Add))
        {
            Assert.Equal(QkNormMode.PerHeadUnweighted, model.Plan.QkNorm);
            Assert.Single(logLines);
            Assert.StartsWith("qk_norm_vectors_unused ", logLines[0]);
            Assert.Contains("model='Unused Norms'", logLines[0]);
            Assert.Contains("architecture='llama'", logLines[0]);
            Assert.Contains("blk.0.attn_q_norm.weight", logLines[0]);
            Assert.Contains("blk.0.attn_k_norm.weight", logLines[0]);
        }
    }

    // ---------------------------------------------------------------- shared helpers

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-residual-norm-plan-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // Canonical metadata pairs (parse-only fixtures need no tensors) matching TinyTensors: hidden 4, heads 2,
    // kv-heads 2, head dim 2, ffn 8, vocab 6.
    private static List<(string Key, object Value)> Metadata(string prefix) =>
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

    // A complete single-layer model — all F32 tensors, the per-head QK-norm vectors included unless asked off.
    private static List<TensorDef> TinyTensors(bool withQkNorm)
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

    private sealed record TensorDef(string Name, int[] Dims, uint GgmlType = 0);

    /// <summary>Builds a GGUF v3 image from KV pairs (strings, ints, floats, bools) and F32 tensors with synthetic-but-nondegenerate payloads.</summary>
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

            writer.Write(tensor.GgmlType);
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
