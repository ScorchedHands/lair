using System.Globalization;
using System.Text;

using Lair.Backends.Cpu;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Load-time resolution and deferred execution for the sparse-FFN (MoE) slot: the registered MoE architectures
/// the pinned reference reports resolve a validated descriptor from the file's declared expert metadata
/// (<c>expert_count</c>, <c>expert_used_count</c>, <c>expert_shared_count</c>, <c>expert_weights_scale</c>, and
/// <c>expert_weights_norm</c>) with the router, per-expert, and shared-expert tensor roles the profile's map
/// names under the reference's names; a declared count that is present but not a sane positive integer fails
/// naming key and value; a recognized MoE file fails engine open with the deferral naming the architecture and
/// the expert count — never as a missing-tensor failure; a dense family file keeps a null descriptor; and an
/// unregistered architecture carrying MoE tensor names still fails through the tensor-table message.
/// </summary>
[Trait("Category", "Loader")]
public sealed class SparseFfnPlanTests : IDisposable
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

    // ---------------------------------------------------------------- descriptor resolution

    [Fact]
    public void ModelPlan_Qwen2MoeFile_ResolvesSparseFfnDescriptor_WithReferenceCountsRolesAndTensorNames()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(MoeMetadata("qwen2moe"), [.. MoeTensors()])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal("qwen2moe", plan.Architecture);
        Assert.Equal("qwen2moe.", plan.Prefix); // every key was read under the qwen2moe prefix

        SparseFfnDescriptor? sparse = plan.SparseFfn;
        Assert.NotNull(sparse);

        Assert.Equal(8, sparse!.ExpertCount);
        Assert.Equal(2, sparse.RoutedExpertsPerToken);
        Assert.True(sparse.HasSharedExpert);
        Assert.Equal(TensorRole.MoeRouter, sparse.RouterRole);
        Assert.Equal(TensorRole.MoeExpertGate, sparse.ExpertGateRole);
        Assert.Equal(TensorRole.MoeExpertUp, sparse.ExpertUpRole);
        Assert.Equal(TensorRole.MoeExpertDown, sparse.ExpertDownRole);
        Assert.Equal(TensorRole.MoeSharedExpertGate, sparse.SharedGateRole);
        Assert.Equal(TensorRole.MoeSharedExpertUp, sparse.SharedUpRole);
        Assert.Equal(TensorRole.MoeSharedExpertDown, sparse.SharedDownRole);
        Assert.True(
            Math.Abs(sparse.ExpertWeightsScale - 1f) < 1e-6f,
            $"expert weights scale {sparse.ExpertWeightsScale:G9} must come from the declared key");
        Assert.True(sparse.NormalizeExpertWeights);

        // The file's table is nameable through the plan's map under the reference's names, including the
        // shared-expert gate vector the qwen2moe conversion ships alongside the shared gate matrix.
        Assert.Equal("blk.0.ffn_gate_inp.weight", plan.Tensors[sparse.RouterRole].NameFor(0));
        Assert.Equal("blk.0.ffn_gate_exps.weight", plan.Tensors[sparse.ExpertGateRole].NameFor(0));
        Assert.Equal("blk.0.ffn_up_exps.weight", plan.Tensors[sparse.ExpertUpRole].NameFor(0));
        Assert.Equal("blk.0.ffn_down_exps.weight", plan.Tensors[sparse.ExpertDownRole].NameFor(0));
        Assert.Equal("blk.0.ffn_gate_shexp.weight", plan.Tensors[sparse.SharedGateRole!.Value].NameFor(0));
        Assert.NotNull(plan.Tensors[sparse.RouterRole].Resolve(gguf.Tensors, 0));
        Assert.NotNull(plan.Tensors[sparse.ExpertGateRole].Resolve(gguf.Tensors, 0));
        Assert.NotNull(plan.Tensors[sparse.ExpertUpRole].Resolve(gguf.Tensors, 0));
        Assert.NotNull(plan.Tensors[sparse.ExpertDownRole].Resolve(gguf.Tensors, 0));
        Assert.NotNull(plan.Tensors[sparse.SharedGateRole.Value].Resolve(gguf.Tensors, 0));
        Assert.NotNull(plan.Tensors[sparse.SharedUpRole!.Value].Resolve(gguf.Tensors, 0));
        Assert.NotNull(plan.Tensors[sparse.SharedDownRole!.Value].Resolve(gguf.Tensors, 0));
        Assert.Contains("blk.{layer}.ffn_gate_inp_shexp.weight", plan.Tensors[sparse.SharedGateRole.Value].Aliases);
    }

    [Fact]
    public void ModelPlan_Qwen3MoeFile_WithoutDeclaredSharedExpert_ResolvesSharedRolesNull()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            MoeMetadata("qwen3moe", sharedExpert: false), [.. MoeTensors()])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        SparseFfnDescriptor? sparse = plan.SparseFfn;
        Assert.NotNull(sparse);
        Assert.Equal(8, sparse!.ExpertCount);
        Assert.Equal(2, sparse.RoutedExpertsPerToken);
        Assert.False(sparse.HasSharedExpert);
        Assert.Null(sparse.SharedGateRole);
        Assert.Null(sparse.SharedUpRole);
        Assert.Null(sparse.SharedDownRole);
        Assert.True(Math.Abs(sparse.ExpertWeightsScale - 1f) < 1e-6f, "the declared scale must land in the descriptor");
        Assert.True(sparse.NormalizeExpertWeights);
    }

    [Fact]
    public void ModelPlan_MoEFile_WithoutRoutingWeightKeys_ResolvesTheReferenceDefaults()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(MoeMetadata("qwen2moe", weights: false), [])));
        SparseFfnDescriptor? sparse = ModelPlans.Resolve(gguf).SparseFfn;

        Assert.NotNull(sparse);
        Assert.Equal(0f, sparse!.ExpertWeightsScale); // the reference's disabled default: the file declares no scale
        Assert.False(sparse.NormalizeExpertWeights);  // and declares no normalization
    }

    [Fact]
    public void ModelPlan_MoEFile_RoutingCountDefaultsToExpertCount_WhenUsedCountAbsent()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen2moe", usedCount: false)], [])));
        SparseFfnDescriptor? sparse = ModelPlans.Resolve(gguf).SparseFfn;

        Assert.NotNull(sparse);
        Assert.Equal(8, sparse!.ExpertCount);
        Assert.Equal(8, sparse.RoutedExpertsPerToken); // the declared expert count is the routing count
    }

    // ---------------------------------------------------------------- declared MoE key validation

    [Theory]
    [InlineData("qwen2moe.expert_count", 0)]
    [InlineData("qwen2moe.expert_count", -3)]
    [InlineData("qwen2moe.expert_count", "wide")]
    public void ModelPlan_InvalidDeclaredExpertCount_FailsNamingKeyAndValue(string key, object declared)
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen2moe", expertKeys: false), (key, declared)], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains(key, ex.Message);
        Assert.Contains(
            declared is string text ? text : System.Convert.ToInt64(declared).ToString(CultureInfo.InvariantCulture),
            ex.Message);
        Assert.Contains("sparse-FFN count", ex.Message);
    }

    [Theory]
    [InlineData("qwen2moe.expert_used_count", 0)]
    [InlineData("qwen2moe.expert_used_count", -1)]
    [InlineData("qwen2moe.expert_used_count", "wide")]
    public void ModelPlan_InvalidDeclaredRoutingCount_FailsNamingKeyAndValue(string key, object declared)
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen2moe", expertKeys: false), ("qwen2moe.expert_count", 8), (key, declared)], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains(key, ex.Message);
        Assert.Contains(
            declared is string text ? text : System.Convert.ToInt64(declared).ToString(CultureInfo.InvariantCulture),
            ex.Message);
        Assert.Contains("sparse-FFN count", ex.Message);
    }

    [Fact]
    public void ModelPlan_RoutingCountAboveExpertCount_FailsNamingKeyAndValue()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen2moe", expertKeys: false), ("qwen2moe.expert_count", 8), ("qwen2moe.expert_used_count", 9)],
            [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("qwen2moe.expert_used_count", ex.Message);
        Assert.Contains("9", ex.Message);
    }

    [Fact]
    public void ModelPlan_InvalidDeclaredExpertWeightsScale_FailsNamingKeyAndValue()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen2moe"), ("qwen2moe.expert_weights_scale", -0.5f)], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("qwen2moe.expert_weights_scale", ex.Message);
        Assert.Contains("-0.5", ex.Message);
    }

    // ---------------------------------------------------------------- deferred execution at Open

    [Fact]
    public void TransformerModelOpen_Qwen2MoeFile_FailsWithDeferredExecutionMessage_NotAMissingTensorFailure()
    {
        string path = WriteTemp(BuildGguf(MoeMetadata("qwen2moe"), [.. MoeTensors()]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("qwen2moe", ex.Message);
        Assert.Contains("8 experts", ex.Message);
        Assert.Contains("sparse FFN execution is not implemented", ex.Message);
        Assert.DoesNotContain("is missing", ex.Message); // the deferral is not a missing-tensor complaint
        Assert.DoesNotContain("blk.0.ffn_gate.weight", ex.Message); // the dense FFN name the file lacks never surfaces
    }

    [Fact]
    public void TransformerModelOpen_Qwen3MoeFile_FailsWithDeferredExecutionMessage_NamingItsArchitectureAndCount()
    {
        string path = WriteTemp(BuildGguf(MoeMetadata("qwen3moe", sharedExpert: false), [.. MoeTensors()]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("qwen3moe", ex.Message);
        Assert.Contains("8 experts", ex.Message);
        Assert.Contains("sparse FFN execution is not implemented", ex.Message);
        Assert.DoesNotContain("is missing", ex.Message);
    }

    // ---------------------------------------------------------------- dense families and the fallback gate

    [Theory]
    [InlineData("llama")]
    [InlineData("qwen2")]
    [InlineData("qwen3")]
    [InlineData("gemma2")]
    public void ModelPlan_DenseFamilyWithoutExpertKeys_KeepsNullSparseFfnDescriptor_AndOpensNormally(string architecture)
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata(architecture, tiny: true), [.. DenseTensors(architecture)])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Null(plan.SparseFfn);
        Assert.NotNull(plan.Tensors[TensorRole.FeedForwardGate].Resolve(gguf.Tensors, 0)); // the dense block stays required

        using CpuBackend backend = new();
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);
        Assert.Null(model.Plan.SparseFfn); // the deferral never fires for a dense family file
    }

    [Fact]
    public void ModelPlan_UnregisteredArchitectureWithMoETensorNames_FailsThroughTensorTableMessage_NotTheDeferral()
    {
        List<TensorDef> tensors = [.. DenseTensors("llama")];
        tensors.Add(new("blk.0.ffn_gate_inp.weight", [4, 8]));    // the router matrix the llama map cannot name
        tensors.Add(new("blk.0.ffn_down_exps.weight", [4, 8, 8])); // a stacked expert triplet

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("myarch", tiny: true), ("myarch.expert_count", 8), ("myarch.expert_used_count", 2)],
            tensors)));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("llama-shaped fallback", ex.Message);
        Assert.Contains("unexpected tensors: blk.0.ffn_gate_inp.weight", ex.Message);
        Assert.DoesNotContain("sparse FFN execution", ex.Message); // the tensor table rejects it before any deferral
    }

    [Fact]
    public void ModelProfiles_MoEArchitecturesDeclareSparseFfnShapes_AndDenseProfilesDoNot()
    {
        Assert.Contains("qwen2moe", ModelProfiles.SupportedArchitectures);
        Assert.Contains("qwen3moe", ModelProfiles.SupportedArchitectures);

        SparseFfnDescriptor? qwen2MoeShape = ModelProfiles.Resolve("qwen2moe").SparseFfn;
        Assert.NotNull(qwen2MoeShape);
        Assert.True(qwen2MoeShape!.HasSharedExpert);
        Assert.Equal(TensorRole.MoeRouter, qwen2MoeShape.RouterRole);

        SparseFfnDescriptor? qwen3MoeShape = ModelProfiles.Resolve("qwen3moe").SparseFfn;
        Assert.NotNull(qwen3MoeShape);
        Assert.False(qwen3MoeShape!.HasSharedExpert);
        Assert.Null(qwen3MoeShape.SharedGateRole);

        foreach (string architecture in new string[] { "llama", "qwen2", "qwen3", "gemma2" })
        {
            Assert.Null(ModelProfiles.Resolve(architecture).SparseFfn);
        }
    }

    // ---------------------------------------------------------------- shared helpers

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-sparse-ffn-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // Canonical tiny geometry (hidden 4, one layer, heads 2, kv-heads 1, ffn 8, vocab 6) with the reference's
    // MoE keys: 8 experts, top-2 routing, one shared expert, the routing-weight scale and normalization the
    // reference reads. The expert-count keys can be dropped one at a time to declare invalid values instead.
    private static List<(string Key, object Value)> MoeMetadata(
        string prefix,
        bool expertKeys = true,
        bool usedCount = true,
        bool sharedExpert = true,
        bool weights = true)
    {
        List<(string Key, object Value)> metadata =
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

        if (expertKeys)
        {
            metadata.Add(($"{prefix}.expert_count", 8));
            if (usedCount)
            {
                metadata.Add(($"{prefix}.expert_used_count", 2));
            }

            if (sharedExpert)
            {
                metadata.Add(($"{prefix}.expert_shared_count", 1));
            }

            if (weights)
            {
                metadata.Add(($"{prefix}.expert_weights_scale", 1f));
                metadata.Add(($"{prefix}.expert_weights_norm", true));
            }
        }

        return metadata;
    }

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

    private sealed record TensorDef(string Name, int[] Dims); // dims in logical (row-major) order; vectors are one-dimensional

    // A complete single-layer MoE block with hidden 4, heads 2, kv-heads 2, head dim 2, ffn 8, vocab 6 — the
    // same geometry the dense fixtures use, with the dense gate/up/down names replaced by the MoE ones: the
    // router matrix [hidden, experts], the stacked per-expert gate/up/down triplets [intermediate, hidden,
    // experts] and [hidden, intermediate, experts], and the shared expert's dense triplet plus its gate vector.
    // The dense gate/up/down names stay absent: an expert-routed block ships none of them.
    private static List<TensorDef> MoeTensors() =>
        [
            new("token_embd.weight", [6, 4]),
            new("blk.0.attn_q.weight", [4, 4]),
            new("blk.0.attn_k.weight", [4, 4]),
            new("blk.0.attn_v.weight", [4, 4]),
            new("blk.0.attn_output.weight", [4, 4]),
            new("blk.0.attn_norm.weight", [4]),
            new("blk.0.ffn_norm.weight", [4]),
            new("output_norm.weight", [4]),
            new("blk.0.ffn_gate_inp.weight", [4, 8]),
            new("blk.0.ffn_gate_exps.weight", [8, 4, 8]),
            new("blk.0.ffn_up_exps.weight", [8, 4, 8]),
            new("blk.0.ffn_down_exps.weight", [4, 8, 8]),
            new("blk.0.ffn_gate_shexp.weight", [8, 4]),
            new("blk.0.ffn_up_shexp.weight", [8, 4]),
            new("blk.0.ffn_down_shexp.weight", [4, 8]),
            new("blk.0.ffn_gate_inp_shexp.weight", [1, 4]),
        ];

    // A complete single-layer dense block with the same tiny geometry: hidden 4, heads 2, kv-heads 1, head dim
    // 4 (2 x 4 against a hidden of 4 is the grouped-query shape the fixtures use), ffn 8, vocab 6 — all F32
    // tensors, plus whatever the family's resolved profile demands at Open: the qwen3 profile defaults the
    // per-head QK-norm on and so ships its vectors, and the gemma2 profile's post-sublayer placement ships its
    // vectors. The llama and qwen2 fixtures carry neither.
    private static List<TensorDef> DenseTensors(string architecture)
    {
        List<TensorDef> tensors =
        [
            new("token_embd.weight", [6, 4]),
            new("blk.0.attn_q.weight", [4, 4]), // head_count * head_dim == hidden == 4
            new("blk.0.attn_k.weight", [4, 4]), // kv_heads * head_dim == 4
            new("blk.0.attn_v.weight", [4, 4]),
            new("blk.0.attn_output.weight", [4, 4]),
            new("blk.0.ffn_gate.weight", [8, 4]),
            new("blk.0.ffn_up.weight", [8, 4]),
            new("blk.0.ffn_down.weight", [4, 8]),
            new("blk.0.attn_norm.weight", [4]),
            new("blk.0.ffn_norm.weight", [4]),
            new("output_norm.weight", [4]), // no output.weight: the tied path is exercised instead
        ];

        if (architecture == "qwen3")
        {
            tensors.Add(new("blk.0.attn_q_norm.weight", [2]));
            tensors.Add(new("blk.0.attn_k_norm.weight", [2]));
        }

        if (architecture == "gemma2")
        {
            tensors.Add(new("blk.0.post_attention_norm.weight", [4]));
            tensors.Add(new("blk.0.post_ffw_norm.weight", [4]));
        }

        return tensors;
    }

    /// <summary>Builds a GGUF v3 image from KV pairs and F32 tensors with synthetic-but-nondegenerate payloads.</summary>
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
