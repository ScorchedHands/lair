using System.Globalization;
using System.Text;

using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Load-time resolution and execution for the sparse-FFN (MoE) slot: the registered MoE architectures
/// the pinned reference reports resolve a validated descriptor from the file's declared expert metadata
/// (<c>expert_count</c>, <c>expert_used_count</c>, <c>expert_shared_count</c>, <c>expert_weights_scale</c>, and
/// <c>expert_weights_norm</c>) with the router, per-expert, and shared-expert tensor roles the profile's map names
/// under the reference's names; a declared count that is present but not a sane positive integer fails naming key and
/// value; a recognized MoE file opens into the engine with its descriptor and runs the routed experts instead of
/// deferring; a dense family file keeps a null descriptor; an unregistered architecture carrying MoE tensor names still
/// fails through the tensor-table message. The same slot also carries how a block stores its experts: the declared
/// <c>expert_layout</c> and <c>expert_rank</c> resolve into the descriptor's layout, rank, basis/coefficient/output-basis
/// roles (absent means stacked with no rank and none of those roles), and the declared <c>sparse_ffn_strategy</c>
/// resolves against every profile's dense-resident default — an unrecognized name fails naming key and value, a
/// shared-basis selection on a stacked declaration is refused rather than downgraded, and a strategy the forward engine
/// has no kernel for reports at engine open instead of running something else.
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
        // The shared-expert selector vector is its own tensor under a role of its own, and the shared gate matrix's
        // name carries no alias onto it: they are different tensors with different shapes.
        Assert.Equal(TensorRole.MoeSharedExpertInputGate, sparse.SharedInputGateRole);
        Assert.Equal("blk.0.ffn_gate_inp_shexp.weight", plan.Tensors[sparse.SharedInputGateRole!.Value].NameFor(0));
        Assert.NotNull(plan.Tensors[sparse.SharedInputGateRole.Value].Resolve(gguf.Tensors, 0));
        Assert.Empty(plan.Tensors[sparse.SharedGateRole.Value].Aliases);
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
        Assert.Null(sparse.SharedInputGateRole); // no shared expert, so nothing to gate it with
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

    // ---------------------------------------------------------------- descriptor-bearing files open and execute

    [Fact]
    public void TransformerModelOpen_Qwen2MoeFile_ResolvesTheDescriptorAndReplacesTheDenseFfnNames()
    {
        string path = WriteTemp(BuildGguf(MoeMetadata("qwen2moe"), [.. MoeTensors()]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        SparseFfnDescriptor sparse = model.Plan.SparseFfn!;
        Assert.Equal(8, sparse.ExpertCount);
        Assert.True(sparse.HasSharedExpert);
        Assert.Null(model.Plan.Tensors[TensorRole.FeedForwardGate].Resolve(gguf.Tensors, 0)); // the routed block replaces the dense FFN names
        Assert.Null(model.Plan.Tensors[TensorRole.FeedForwardUp].Resolve(gguf.Tensors, 0));
        Assert.Null(model.Plan.Tensors[TensorRole.FeedForwardDown].Resolve(gguf.Tensors, 0));
    }

    [Fact]
    public void TransformerModelOpen_Qwen3MoeFile_WithoutSharedExpert_ResolvesTheDescriptorAndExecutes()
    {
        string path = WriteTemp(BuildGguf(MoeMetadata("qwen3moe", sharedExpert: false), [.. MoeTensors("qwen3moe")]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        SparseFfnDescriptor sparse = model.Plan.SparseFfn!;
        Assert.Equal(8, sparse.ExpertCount);
        Assert.False(sparse.HasSharedExpert);
        Assert.Null(model.Plan.Tensors[TensorRole.FeedForwardGate].Resolve(gguf.Tensors, 0));
        Assert.Null(model.Plan.Tensors[TensorRole.FeedForwardUp].Resolve(gguf.Tensors, 0));
        Assert.Null(model.Plan.Tensors[TensorRole.FeedForwardDown].Resolve(gguf.Tensors, 0));
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
        Assert.Null(model.Plan.SparseFfn); // no expert count declared: the dense block stays the only one
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

        // The reference gates its shared expert per token with a vector of its own, so only qwen2moe names that role.
        Assert.Equal(TensorRole.MoeSharedExpertInputGate, qwen2MoeShape.SharedInputGateRole);

        SparseFfnDescriptor? qwen3MoeShape = ModelProfiles.Resolve("qwen3moe").SparseFfn;
        Assert.NotNull(qwen3MoeShape);
        Assert.False(qwen3MoeShape!.HasSharedExpert);
        Assert.Null(qwen3MoeShape.SharedGateRole);
        Assert.Null(qwen3MoeShape.SharedInputGateRole);

        foreach (string architecture in new string[] { "llama", "qwen2", "qwen3", "gemma2" })
        {
            Assert.Null(ModelProfiles.Resolve(architecture).SparseFfn);
        }
    }

    // ---------------------------------------------------------------- declared expert storage layout and rank

    [Fact]
    public void ModelPlan_Qwen2MoeFile_DeclaringSharedBasisLayout_ResolvesRankRolesAndTheLowRankStrategy()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen2moe"), ("qwen2moe.expert_layout", "shared_basis"), ("qwen2moe.expert_rank", 4), ("qwen2moe.sparse_ffn_strategy", "low_rank_shared_basis")],
            [.. MoeTensors(), .. FactorizedTensors(aliasSpellings: true)])));

        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.NotNull(plan.SparseFfn);
        SparseFfnDescriptor sparse = plan.SparseFfn!;

        Assert.Equal(SparseFfnLayout.FactorizedSharedBasis, sparse.Layout);
        Assert.Equal<int?>(4, sparse.Rank); // the declared rank lands in the descriptor as the coefficient-space width

        Assert.Equal(TensorRole.MoeExpertInputBasis, sparse.ExpertInputBasisRole);
        Assert.Equal(TensorRole.MoeExpertCoefficients, sparse.ExpertCoefficientRole);
        Assert.Equal(TensorRole.MoeExpertOutputBasis, sparse.ExpertOutputBasisRole);

        // Each factorized role resolves through the plan's map under this family's naming shape. The coefficient tensor
        // this fixture ships is spelled with the plural alias, and it resolves all the same: a sidecar exporter that
        // spells its stacked-coefficient name either way reaches the same role through the layer-substituted alias.
        var inputBasis = plan.Tensors[sparse.ExpertInputBasisRole!.Value].Resolve(gguf.Tensors, 0);
        var coefficient = plan.Tensors[sparse.ExpertCoefficientRole!.Value].Resolve(gguf.Tensors, 0);
        var outputBasis = plan.Tensors[sparse.ExpertOutputBasisRole!.Value].Resolve(gguf.Tensors, 0);

        Assert.NotNull(inputBasis);
        Assert.NotNull(coefficient); // found through the alias name
        Assert.NotNull(outputBasis);

        Assert.Equal("blk.0.ffn_exps_basis_in.weight", inputBasis!.Name);
        Assert.Equal("blk.0.ffn_experts_coefficient.weight", coefficient!.Name);
        Assert.Equal("blk.0.ffn_exps_basis_out.weight", outputBasis!.Name);

        // The declaration selects its strategy too: a factorized file naming its own variant has it recorded rather than
        // some other variant running behind its back.
        Assert.Equal(SparseFfnStrategy.LowRankSharedBasis, plan.SparseFfnStrategy);
    }

    [Fact]
    public void ModelPlan_MoEFile_WithoutLayoutDeclaration_ResolvesStackedLayoutNoRankAndNoFactorizedRoles()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(MoeMetadata("qwen2moe"), [.. MoeTensors()])));

        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.NotNull(plan.SparseFfn);
        SparseFfnDescriptor stacked = plan.SparseFfn!;

        Assert.Equal(SparseFfnLayout.StackedExperts, stacked.Layout); // every published conversion stores its experts this way
        Assert.Null(stacked.Rank);
        Assert.Null(stacked.ExpertInputBasisRole);
        Assert.Null(stacked.ExpertCoefficientRole);
        Assert.Null(stacked.ExpertOutputBasisRole);

        // A stacked declaration is an un-factorizable file: nothing in its descriptor points at shared-basis work, so no
        // low-rank variant is selected for it. Absent the strategy key, the resolved profile's default answers instead.
        Assert.Equal(SparseFfnStrategy.DenseResident, plan.SparseFfnStrategy);
    }

    [Fact]
    public void ModelPlan_ExplicitlyDeclaredDenseResidentStrategy_ResolvesItOverTheProfileDefault()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen3moe", sharedExpert: false), ("qwen3moe.sparse_ffn_strategy", "dense_resident")], [])));

        Assert.Equal(SparseFfnStrategy.DenseResident, ModelPlans.Resolve(gguf).SparseFfnStrategy);
    }

    [Fact]
    public void ModelPlan_DenseFamilyWithoutExperts_CarriesNoSparseFfnStrategyAtAll()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata("llama", tiny: true), [.. DenseTensors("llama")])));

        Assert.Null(ModelPlans.Resolve(gguf).SparseFfn);
        Assert.Null(ModelPlans.Resolve(gguf).SparseFfnStrategy); // nothing routes, so no variant is selected for it
    }

    [Fact]
    public void ModelProfiles_DefaultSparseFfnStrategy_IsDenseResidentForEveryRegisteredProfile()
    {
        foreach (string architecture in ModelProfiles.SupportedArchitectures)
        {
            Assert.Equal(SparseFfnStrategy.DenseResident, ModelProfiles.Resolve(architecture).DefaultSparseFfnStrategy);
        }

        // A factorized shape is what a file declares, never what a family prefers: the sparse shapes carry the stacked
        // layout with no rank while naming the basis/coefficient/output roles their maps can resolve.
        SparseFfnDescriptor qwen2MoeShape = ModelProfiles.Resolve("qwen2moe").SparseFfn!;

        Assert.Equal(SparseFfnLayout.StackedExperts, qwen2MoeShape.Layout);
        Assert.Null(qwen2MoeShape.Rank);
        Assert.NotNull(qwen2MoeShape.ExpertInputBasisRole);
    }

    // ---------------------------------------------------------------- declared layout and strategy validation

    [Theory]
    [InlineData("qwen2moe.sparse_ffn_strategy", "auto")]
    [InlineData("qwen2moe.sparse_ffn_strategy", "low_rank")]
    [InlineData("qwen2moe.sparse_ffn_strategy", "stacked")] // a layout value is not a strategy name
    public void ModelPlan_InvalidDeclaredSparseFfnStrategy_FailsNamingKeyAndValue(string key, string declared)
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen2moe"), (key, declared)], [])));

        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains(key, ex.Message);
        Assert.Contains(declared, ex.Message);
        Assert.Contains("sparse-FFN strategy", ex.Message);
        Assert.Contains("'dense_resident' and 'low_rank_shared_basis'", ex.Message); // the accepted set is stated, not guessed
    }

    [Fact]
    public void ModelPlan_InvalidDeclaredSparseFfnStrategyType_FailsNamingKey()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen2moe"), ("qwen2moe.sparse_ffn_strategy", 7)], [])));

        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("qwen2moe.sparse_ffn_strategy", ex.Message);
        Assert.Contains("must be a string", ex.Message);
    }

    [Fact]
    public void ModelPlan_LowRankStrategyOnAStackedFile_IsRefusedNotDowngraded_NamingBothSidesOfTheMismatch()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen2moe"), ("qwen2moe.expert_layout", "stacked"), ("qwen2moe.sparse_ffn_strategy", "low_rank_shared_basis")], [])));

        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf)); // no plan resolves: nothing runs dense math under a low-rank label

        Assert.Contains("qwen2moe.sparse_ffn_strategy", ex.Message);
        Assert.Contains("'low_rank_shared_basis'", ex.Message);
        Assert.Contains("stacked dense matrices with no rank", ex.Message);
        Assert.Contains("expert_layout 'shared_basis'", ex.Message); // the two declarations that would settle it are named
        Assert.Contains("dense_resident", ex.Message);

        // The same file without the selection still resolves cleanly, so the refusal is about the mismatch alone.
        using GgufFile plain = GgufFile.Open(WriteTemp(BuildGguf(MoeMetadata("qwen2moe"), [])));
        Assert.Equal(SparseFfnStrategy.DenseResident, ModelPlans.Resolve(plain).SparseFfnStrategy);
    }

    [Theory]
    [InlineData("qwen2moe.expert_layout", "lora")]
    [InlineData("qwen2moe.expert_layout", "factorized")] // the plan's own wording for it is not a declaration value
    public void ModelPlan_InvalidDeclaredExpertLayout_FailsNamingKeyAndValue(string key, string declared)
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf([.. MoeMetadata("qwen2moe"), (key, declared)], [])));

        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains(key, ex.Message);
        Assert.Contains(declared, ex.Message);
        Assert.Contains("'stacked' and 'shared_basis'", ex.Message);
    }

    [Fact]
    public void ModelPlan_SharedBasisLayoutWithoutARank_FailsNamingTheMissingRankKey()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen2moe"), ("qwen2moe.expert_layout", "shared_basis")], [])));

        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("qwen2moe.expert_rank", ex.Message);
        Assert.Contains("must declare the rank its coefficient tensors carry", ex.Message);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    [InlineData("wide")]
    public void ModelPlan_InvalidDeclaredRank_FailsNamingKeyAndValue(object declared)
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen2moe"), ("qwen2moe.expert_layout", "shared_basis"), ("qwen2moe.expert_rank", declared)], [])));

        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("qwen2moe.expert_rank", ex.Message);
        Assert.Contains(
            declared is string text ? text : System.Convert.ToInt64(declared).ToString(CultureInfo.InvariantCulture),
            ex.Message);
    }

    [Fact]
    public void ModelPlan_RankDeclaredOnAStackedFile_FailsAsAnInconsistentDeclaration()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen2moe"), ("qwen2moe.expert_rank", 4)], [])));

        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("qwen2moe.expert_rank", ex.Message);
        Assert.Contains("only a block whose qwen2moe.expert_layout is 'shared_basis'", ex.Message);
    }

    // ---------------------------------------------------------------- strategy execution and refusal

    [Fact]
    public void TransformerModelOpen_FactorizedFileSelectingLowRank_ReportsNotImplemented_NamingTheStrategyAndWhatRunsToday()
    {
        string path = WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen2moe"), ("qwen2moe.expert_layout", "shared_basis"), ("qwen2moe.expert_rank", 4), ("qwen2moe.sparse_ffn_strategy", "low_rank_shared_basis")],
            [.. MoeTensors(), .. FactorizedTensors()]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);

        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("sparse-FFN strategy 'LowRankSharedBasis' is not implemented by the forward engine", ex.Message);
        Assert.Contains("architecture 'qwen2moe'", ex.Message);
        Assert.Contains("8 experts routing 2 per token over a shared basis at rank 4", ex.Message);
        Assert.Contains("only DenseResident executes today", ex.Message);
        Assert.Contains("declare qwen2moe.sparse_ffn_strategy 'dense_resident' to run this file", ex.Message);

        // A strategy with no kernel reports as the missing variant, never as a tensor problem: the factorized tensors are
        // representable material and nothing here is absent.
        Assert.DoesNotContain("is missing", ex.Message);
    }

    [Fact]
    public void TransformerModelOpen_FactorizedFileWithoutASelection_ExecutesDenseResident_AndKeepsItsFactorizedDescriptor()
    {
        string path = WriteTemp(BuildGguf(
            [.. MoeMetadata("qwen2moe"), ("qwen2moe.expert_layout", "shared_basis"), ("qwen2moe.expert_rank", 4)],
            [.. MoeTensors(), .. FactorizedTensors()]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8); // the engine takes this factorized file on its resident dense path today

        Assert.Equal(SparseFfnLayout.FactorizedSharedBasis, model.Plan.SparseFfn!.Layout); // representable either way
        Assert.Equal<int?>(4, model.Plan.SparseFfn.Rank);                                 // the declared rank stays visible to callers
        Assert.Equal(TensorRole.MoeExpertCoefficients, model.Plan.SparseFfn.ExpertCoefficientRole);
        Assert.Equal(SparseFfnStrategy.DenseResident, model.Plan.SparseFfnStrategy);      // and the profile default runs it
    }

    [Fact]
    public void TransformerModelOpen_StackedFileWithNoFactorizedDeclaration_KeepsLowRankUnselected_AndExecutesDenseResident()
    {
        string path = WriteTemp(BuildGguf(MoeMetadata("qwen2moe"), [.. MoeTensors()]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8); // an un-factorizable file has no low-rank variant to select and runs the resident experts

        Assert.Equal(SparseFfnLayout.StackedExperts, model.Plan.SparseFfn!.Layout);
        Assert.Null(model.Plan.SparseFfn.Rank);
        Assert.NotEqual(SparseFfnStrategy.LowRankSharedBasis, model.Plan.SparseFfnStrategy!.Value);
        Assert.Equal(SparseFfnStrategy.DenseResident, model.Plan.SparseFfnStrategy.Value);
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
    private static List<TensorDef> MoeTensors(string architecture = "qwen2moe")
    {
        List<TensorDef> tensors =
            [
                new("token_embd.weight", [6, 4]),
                new("blk.0.attn_q.weight", [4, 4]),
                new("blk.0.attn_k.weight", [4, 4]),
                new("blk.0.attn_v.weight", [4, 4]),
                new("blk.0.attn_output.weight", [4, 4]),
                new("blk.0.attn_norm.weight", [4]),
                new("blk.0.ffn_norm.weight", [4]),
                new("output_norm.weight", [4]),
                new("blk.0.ffn_gate_inp.weight", [8, 4]),
                new("blk.0.ffn_gate_exps.weight", [8, 8, 4]),
                new("blk.0.ffn_up_exps.weight", [8, 8, 4]),
                new("blk.0.ffn_down_exps.weight", [8, 4, 8]),
            ];

        if (architecture == "qwen2moe")
        {
            tensors.Add(new("blk.0.ffn_gate_shexp.weight", [8, 4]));
            tensors.Add(new("blk.0.ffn_up_shexp.weight", [8, 4]));
            tensors.Add(new("blk.0.ffn_down_shexp.weight", [4, 8]));
            tensors.Add(new("blk.0.ffn_gate_inp_shexp.weight", [4])); // one-dimensional, the size of the hidden stream
        }

        if (architecture is "qwen3" or "qwen3moe")
        {
            tensors.Add(new("blk.0.attn_q_norm.weight", [2]));
            tensors.Add(new("blk.0.attn_k_norm.weight", [2]));
        }

        return tensors;
    }

    // The factorized layout's tensors at the same geometry (hidden 4, intermediate 8, declared rank 4): one shared input
    // basis mapping hidden to rank, one coefficient stack carrying a [rank] contribution per expert with the expert count
    // as its outermost dimension exactly like the dense-stacked roles do, and one shared output projection expanding rank
    // back to hidden. Nothing resident reads these three today — they exist so a factorized declaration resolves through
    // this family's map names. With aliasSpellings set the coefficient tensor is shipped under the plural alias spelling
    // instead of the primary name, which the same role must still resolve.
    private static List<TensorDef> FactorizedTensors(bool aliasSpellings = false) =>
        [
            new("blk.0.ffn_exps_basis_in.weight", [4, 4]),
            new(aliasSpellings ? "blk.0.ffn_experts_coefficient.weight" : "blk.0.ffn_exps_coefficient.weight", [8, 4]),
            new("blk.0.ffn_exps_basis_out.weight", [4, 4]),
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
