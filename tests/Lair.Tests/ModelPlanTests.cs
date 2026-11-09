using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Load-time plan resolution over synthetic GGUF fixtures: profile-registry selection and the rejection
/// wording it owns, the generic llama-shaped fallback for unknown architectures (exact tensor-table match,
/// its rejection wording and structured log line), resolved geometry for both supported profiles including
/// the head_length head-dimension source, the plan's tensor map (required versus optional roles, per-layer
/// name templates, alias candidates), the strategy values and declared feature flags the forward engine
/// switches on, and tensor lookup through the map at Open.
/// </summary>
[Trait("Category", "Loader")]
public sealed class ModelPlanTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

    // Roles a file may omit: attention biases (llama ships none, qwen2 may), the untied output head and its
    // bias, the per-head QK-norm vectors, the norm-bias vectors, and the post-sublayer norm vectors (no registered
    // profile places norms on sublayer outputs today) — every one of them exists only for files whose plan consumes
    // the matching feature.
    private static readonly TensorRole[] OptionalRoles =
    [
        TensorRole.AttentionQueryBias,
        TensorRole.AttentionKeyBias,
        TensorRole.AttentionValueBias,
        TensorRole.AttentionOutputBias,
        TensorRole.OutputHead,
        TensorRole.OutputBias,
        TensorRole.AttentionQueryNorm,
        TensorRole.AttentionKeyNorm,
        TensorRole.AttentionNormBias,
        TensorRole.FeedForwardNormBias,
        TensorRole.OutputNormBias,
        TensorRole.PostAttentionNorm,
        TensorRole.PostFeedForwardNorm,
    ];

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

    // ---------------------------------------------------------------- plan resolution

    [Fact]
    public void ModelPlan_LlamaMetadata_ResolvesProfileGeometry_AndFlatView()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata("llama"), [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal("llama", plan.Architecture);
        Assert.Equal("llama.", plan.Prefix);
        Assert.Equal(30, plan.LayerCount);
        Assert.Equal(576, plan.HiddenSize);
        Assert.Equal(9, plan.HeadCount);
        Assert.Equal(3, plan.KvHeadCount);
        Assert.Equal(64, plan.HeadDim);
        Assert.Equal(1536, plan.IntermediateSize);
        Assert.Equal(49_152, plan.VocabSize);
        Assert.Equal(10_000f, plan.RopeTheta);
        Assert.True(Math.Abs(plan.EpsRmsNorm - 1e-5f) < 1e-7f, $"rms epsilon {plan.EpsRmsNorm:G9}");
        Assert.False(plan.HasOutputWeights); // no output.weight in the tensor table: the tied path

        // The flat scalar view is exactly the plan's resolved geometry.
        Assert.Equal(plan.ToDesc(), ModelDesc.FromGguf(gguf));
    }

    [Fact]
    public void ModelPlan_Qwen2Metadata_ResolvesSameGeometryUnderTheQwenPrefix()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata("qwen2"), [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal("qwen2", plan.Architecture);
        Assert.Equal("qwen2.", plan.Prefix); // every key was read under the qwen2 prefix
        Assert.Equal(30, plan.LayerCount);
        Assert.Equal(576, plan.HiddenSize);
        Assert.Equal(9, plan.HeadCount);
        Assert.Equal(3, plan.KvHeadCount);
        Assert.Equal(64, plan.HeadDim);
        Assert.Equal(1536, plan.IntermediateSize);
        Assert.Equal(49_152, plan.VocabSize);
        Assert.Equal(10_000f, plan.RopeTheta);
        Assert.True(Math.Abs(plan.EpsRmsNorm - 1e-5f) < 1e-7f, $"rms epsilon {plan.EpsRmsNorm:G9}");
        Assert.False(plan.HasOutputWeights);
    }

    [Fact]
    public void ModelPlan_Strategies_AndSparseSlot_ResolveToTheImplementedSet_ForBothProfiles()
    {
        foreach (string architecture in new string[] { "llama", "qwen2" })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata(architecture), [])));
            ModelPlan plan = ModelPlans.Resolve(gguf);

            Assert.Equal(NormStrategy.RmsNorm, plan.Norm);
            Assert.Equal(ActivationStrategy.SwiGLU, plan.Activation);
            Assert.Equal(BiasPolicy.Declared, plan.Bias);
            Assert.Equal(RopeStrategy.StandardFullHead, plan.Rope);
            Assert.Null(plan.SparseFfn); // dense gate/up/down: no profile declares experts

            // The declared feature flags default to false: these fixtures declare none of the feature keys.
            Assert.False(plan.UseAttentionBias);
            Assert.False(plan.UseNormBias);
            Assert.Equal(QkNormMode.Off, plan.QkNorm);
            Assert.Equal(NormPlacement.PreOnly, plan.NormPlacement); // no registered profile norms sublayer outputs today
        }
    }

    [Fact]
    public void ModelPlan_UnsupportedArchitecture_MessageComesFromTheRegistry()
    {
        string registryList = string.Join(", ", ModelProfiles.SupportedArchitectures);

        // Zero tensors: the table matches no descriptor of the llama-shaped map, so the generic fallback is
        // rejected and the registry rejection wording surfaces, extended with the attempted-fallback detail.
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf([("general.architecture", "bert")], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        // The supported list in the rejection is the registry's own list, in registry order, and the plan's
        // rejection extends the profile lookup's registry wording with the attempted-fallback detail.
        Assert.Contains(
            $"model architecture 'bert' is not supported (supported architectures: {registryList})",
            ex.Message);
        Assert.Contains("llama-shaped fallback", ex.Message);
        Assert.True(ModelDesc.SupportedArchitectures.SequenceEqual(ModelProfiles.SupportedArchitectures));

        // The profile lookup keeps the plain registry wording; the plan's rejection carries it as its opening.
        Assert.StartsWith(Assert.Throws<ModelException>(() => ModelProfiles.Resolve("bert")).Message, ex.Message);

        // A missing architecture key names the key and the same registry list.
        using GgufFile missing = GgufFile.Open(WriteTemp(BuildGguf([("general.name", "mystery")], [])));
        ModelException missingEx = Assert.Throws<ModelException>(() => ModelPlans.Resolve(missing));
        Assert.Contains("general.architecture", missingEx.Message);
        Assert.Contains(registryList, missingEx.Message);
    }

    // ---------------------------------------------------------------- generic llama-shaped fallback

    [Fact]
    public void ModelPlan_UnknownArchitecture_LlamaShapedTable_ResolvesAsGenericFallback()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata("myarch", tiny: true), [.. TinyTensors(withOptional: false)])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal("myarch", plan.Architecture); // the declared name is recorded, not the fallback identity
        Assert.Equal(ModelProfiles.GenericArchitecture, plan.Profile);
        Assert.Equal("myarch.", plan.Prefix); // every key was read under the declared prefix
        Assert.Equal(1, plan.LayerCount);
        Assert.Equal(4, plan.HiddenSize);
        Assert.Equal(2, plan.HeadCount);
        Assert.Equal(2, plan.KvHeadCount);
        Assert.Equal(2, plan.HeadDim);
        Assert.Equal(8, plan.IntermediateSize);
        Assert.Equal(6, plan.VocabSize);
        Assert.Equal(10_000f, plan.RopeTheta);
        Assert.True(Math.Abs(plan.EpsRmsNorm - 1e-5f) < 1e-7f, $"rms epsilon {plan.EpsRmsNorm:G9}");
        Assert.False(plan.HasOutputWeights);

        Assert.Equal(NormStrategy.RmsNorm, plan.Norm);
        Assert.Equal(ActivationStrategy.SwiGLU, plan.Activation);
        Assert.Equal(BiasPolicy.ByTensorPresence, plan.Bias);
        Assert.Equal(RopeStrategy.StandardFullHead, plan.Rope);
        Assert.Null(plan.SparseFfn);

        // The fallback profile is never selectable by name and never joins the supported-architecture list.
        Assert.DoesNotContain(ModelProfiles.GenericArchitecture, ModelProfiles.SupportedArchitectures);
        Assert.Throws<ModelException>(() => ModelProfiles.Resolve(ModelProfiles.GenericArchitecture));
    }

    [Fact]
    public void ModelPlan_UnknownArchitecture_ForwardIsIdenticalToTheSameTensorsDeclaredAsLlama()
    {
        byte[] unknownImage = BuildGguf(Metadata("myarch", tiny: true), [.. TinyTensors(withOptional: false)]);
        byte[] llamaImage = BuildGguf(Metadata("llama", tiny: true), [.. TinyTensors(withOptional: false)]);

        using CpuBackend backend = new();

        float[] unknownLogits;
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(unknownImage));
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

            Assert.Equal("myarch", model.Plan.Architecture);
            Assert.Equal(ModelProfiles.GenericArchitecture, model.Plan.Profile);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 2, backend);
            using SequenceKv kv = new(pool);
            unknownLogits = model.DecodeNext(3, kv).ToArray();
        }

        float[] llamaLogits;
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(llamaImage));
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 2, backend);
            using SequenceKv kv = new(pool);
            llamaLogits = model.DecodeNext(3, kv).ToArray();
        }

        Assert.Equal(llamaLogits.Length, unknownLogits.Length);
        Assert.True(
            unknownLogits.AsSpan().SequenceEqual(llamaLogits.AsSpan()),
            "fallback forward must be bit-identical to the same tensors and prefix declared as llama");
        foreach (float v in unknownLogits)
        {
            Assert.True(float.IsFinite(v), $"logit {v} is not finite");
        }
    }

    [Fact]
    public void ModelPlan_UnknownArchitecture_RenamedRequiredTensor_RejectsNamingIt_AndTheAttemptedFallback()
    {
        List<TensorDef> tensors = [.. TinyTensors(withOptional: false)];
        tensors.RemoveAll(t => t.Name == "blk.0.attn_q.weight"); // rename it: the map can no longer name it

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            Metadata("myarch", tiny: true),
            [.. tensors, new TensorDef("blk.0.attn_query.weight", [4, 4])])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("model architecture 'myarch' is not supported", ex.Message);
        Assert.Contains("llama-shaped fallback profile was attempted", ex.Message);
        Assert.Contains("missing tensors: blk.0.attn_q.weight", ex.Message);
        Assert.Contains("unexpected tensors: blk.0.attn_query.weight", ex.Message);
    }

    [Fact]
    public void ModelPlan_UnknownArchitecture_UnexpectedTensor_RejectsNamingIt()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            Metadata("myarch", tiny: true),
            [.. TinyTensors(withOptional: false), new TensorDef("blk.0.attn_qkv.weight", [4, 4])])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("model architecture 'myarch' is not supported", ex.Message);
        Assert.Contains("llama-shaped fallback profile was attempted", ex.Message);
        Assert.Contains("unexpected tensors: blk.0.attn_qkv.weight", ex.Message);
    }

    [Fact]
    public void ModelPlan_UnknownArchitecture_MissingRequiredMetadataKey_RejectsNamingTheKey()
    {
        List<(string Key, object Value)> pairs = [.. Metadata("myarch", tiny: true)];
        pairs.RemoveAll(p => p.Key == "myarch.block_count"); // a llama-shaped table cannot rescue missing geometry

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf([.. pairs], [.. TinyTensors(withOptional: false)])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));
        Assert.Contains("GGUF metadata key 'myarch.block_count' is not present", ex.Message);
    }

    [Fact]
    public void ModelPlan_EmptyOrWhitespaceArchitecture_RejectsOutright_AndNeverReachesTheFallback()
    {
        foreach (string architecture in new string[] { "", "   " })
        {
            // A fully llama-shaped file whose declared architecture is blank: without the outright rejection it
            // would pass the fallback gate, resolve its geometry under the degenerate '.' prefix, and report the
            // fallback log line. It must fail with its own message before any of that happens.
            List<string> fallbackLines = [];
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
                Metadata(architecture, tiny: true),
                [.. TinyTensors(withOptional: false)])));

            ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf, fallbackLines.Add));

            Assert.Equal("GGUF metadata key 'general.architecture' is present but empty", ex.Message);
            Assert.DoesNotContain("is not present", ex.Message);        // distinct from the missing-key wording
            Assert.DoesNotContain("is not supported", ex.Message);      // distinct from the registry wording
            Assert.DoesNotContain("llama-shaped fallback", ex.Message); // the fallback gate never ran
            Assert.Empty(fallbackLines);                                // no model_profile line was emitted
        }
    }

    [Fact]
    public void ModelPlan_FallbackLogLine_IsEmittedWhenApplied_AndNeverForRegisteredProfiles()
    {
        List<string> fallbackLines = [];

        using (GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("myarch", tiny: true), ("general.name", "Mystery Model")],
            [.. TinyTensors(withOptional: false)]))))
        {
            ModelPlans.Resolve(gguf, fallbackLines.Add);
        }

        Assert.Single(fallbackLines);
        Assert.Equal("model_profile model='Mystery Model' architecture='myarch' profile=llama-generic", fallbackLines[0]);

        foreach (string architecture in new string[] { "llama", "qwen2" })
        {
            List<string> registeredLines = [];
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata(architecture, tiny: true), [.. TinyTensors(withOptional: false)])));
            ModelPlans.Resolve(gguf, registeredLines.Add);
            Assert.Empty(registeredLines); // a registered profile never reports the fallback
        }
    }

    // ---------------------------------------------------------------- tensor map

    [Fact]
    public void TensorMap_CanonicalProfiles_MarkOptionalRolesAndNameRequiredOnesWithTodaysNames()
    {
        foreach (string architecture in new string[] { "llama", "qwen2" })
        {
            TensorMap map = ModelProfiles.Resolve(architecture).Tensors;

            foreach (TensorDescriptor descriptor in map.Descriptors)
            {
                Assert.True(
                    descriptor.Required != Array.Exists(OptionalRoles, role => role == descriptor.Role),
                    $"{architecture} role {descriptor.Role} must be {(descriptor.Required ? "required" : "optional")}");
                Assert.Empty(descriptor.Aliases); // these two profiles resolve to the canonical names only
            }

            Assert.Equal("token_embd.weight", map[TensorRole.TokenEmbedding].NameFor(-1));
            Assert.Equal("blk.7.attn_q.weight", map[TensorRole.AttentionQuery].NameFor(7));
            Assert.Equal("blk.7.attn_k.weight", map[TensorRole.AttentionKey].NameFor(7));
            Assert.Equal("blk.7.attn_v.weight", map[TensorRole.AttentionValue].NameFor(7));
            Assert.Equal("blk.7.attn_output.weight", map[TensorRole.AttentionOutput].NameFor(7));
            Assert.Equal("blk.7.ffn_gate.weight", map[TensorRole.FeedForwardGate].NameFor(7));
            Assert.Equal("blk.7.ffn_up.weight", map[TensorRole.FeedForwardUp].NameFor(7));
            Assert.Equal("blk.7.ffn_down.weight", map[TensorRole.FeedForwardDown].NameFor(7));
            Assert.Equal("blk.7.attn_norm.weight", map[TensorRole.AttentionNorm].NameFor(7));
            Assert.Equal("blk.7.ffn_norm.weight", map[TensorRole.FeedForwardNorm].NameFor(7));
            Assert.Equal("blk.7.attn_q.bias", map[TensorRole.AttentionQueryBias].NameFor(7));
            Assert.Equal("blk.7.attn_output.bias", map[TensorRole.AttentionOutputBias].NameFor(7));
            Assert.Equal("output_norm.weight", map[TensorRole.OutputNorm].NameFor(-1));
            Assert.Equal("output.weight", map[TensorRole.OutputHead].NameFor(-1));
            Assert.Equal("output.bias", map[TensorRole.OutputBias].NameFor(-1));
            Assert.Equal("blk.7.attn_q_norm.weight", map[TensorRole.AttentionQueryNorm].NameFor(7));
            Assert.Equal("blk.7.attn_k_norm.weight", map[TensorRole.AttentionKeyNorm].NameFor(7));
            Assert.Equal("blk.7.attn_norm.bias", map[TensorRole.AttentionNormBias].NameFor(7));
            Assert.Equal("blk.7.ffn_norm.bias", map[TensorRole.FeedForwardNormBias].NameFor(7));
            Assert.Equal("output_norm.bias", map[TensorRole.OutputNormBias].NameFor(-1));
            Assert.Equal("blk.7.post_attention_norm.weight", map[TensorRole.PostAttentionNorm].NameFor(7));
            Assert.Equal("blk.7.post_ffw_norm.weight", map[TensorRole.PostFeedForwardNorm].NameFor(7));
        }
    }

    [Fact]
    public void TensorMap_ResolvesRequiredRoles_AndLeavesAbsentOptionalRolesNull()
    {
        TensorMap map = ModelProfiles.Resolve("llama").Tensors;

        LogicalTensorTable withoutOptional = Table(
            ("token_embd.weight", [6, 4]),
            ("blk.0.attn_q.weight", [4, 4]),
            ("blk.0.attn_k.weight", [4, 4]),
            ("blk.0.attn_v.weight", [4, 4]),
            ("blk.0.attn_output.weight", [4, 4]),
            ("blk.0.ffn_gate.weight", [8, 4]),
            ("blk.0.ffn_up.weight", [8, 4]),
            ("blk.0.ffn_down.weight", [4, 8]),
            ("blk.0.attn_norm.weight", [4]),
            ("blk.0.ffn_norm.weight", [4]),
            ("output_norm.weight", [4]));

        foreach (TensorDescriptor descriptor in map.Descriptors)
        {
            LogicalTensor? hit = descriptor.Resolve(withoutOptional, layer: 0);
            if (descriptor.Required)
            {
                Assert.True(hit is not null, $"required role {descriptor.Role} must resolve");
                Assert.Equal(descriptor.NameFor(descriptor.Scope == TensorScope.PerLayer ? 0 : -1), hit!.Name);
            }
            else
            {
                Assert.True(hit is null, $"optional role {descriptor.Role} must stay unresolved when absent");
            }
        }

        LogicalTensorTable withOptional = Table(
            ("token_embd.weight", [6, 4]),
            ("blk.0.attn_q.weight", [4, 4]),
            ("blk.0.attn_k.weight", [4, 4]),
            ("blk.0.attn_v.weight", [4, 4]),
            ("blk.0.attn_output.weight", [4, 4]),
            ("blk.0.ffn_gate.weight", [8, 4]),
            ("blk.0.ffn_up.weight", [8, 4]),
            ("blk.0.ffn_down.weight", [4, 8]),
            ("blk.0.attn_norm.weight", [4]),
            ("blk.0.ffn_norm.weight", [4]),
            ("blk.0.attn_q.bias", [4]),
            ("blk.0.attn_k.bias", [4]),
            ("blk.0.attn_v.bias", [4]),
            ("blk.0.attn_output.bias", [4]),
            ("output_norm.weight", [4]),
            ("output.weight", [6, 4]),
            ("output.bias", [6]),
            ("blk.0.attn_q_norm.weight", [4]),
            ("blk.0.attn_k_norm.weight", [4]),
            ("blk.0.attn_norm.bias", [4]),
            ("blk.0.ffn_norm.bias", [4]),
            ("output_norm.bias", [4]),
            ("blk.0.post_attention_norm.weight", [4]),
            ("blk.0.post_ffw_norm.weight", [4]));

        foreach (TensorDescriptor descriptor in map.Descriptors)
        {
            Assert.True(
                descriptor.Resolve(withOptional, layer: 0) is not null,
                $"role {descriptor.Role} must resolve when its tensor is present");
        }
    }

    [Fact]
    public void TensorMap_AliasCandidate_ResolvesOnlyWhenThePrimaryNameIsAbsent()
    {
        // A map that declares an alias for the untied output head: the primary name still wins, and the
        // alias fills in only when the primary is missing.
        TensorMap aliased = new([
            .. ModelProfiles.Resolve("llama").Tensors.Descriptors.Select(descriptor =>
                descriptor.Role == TensorRole.OutputHead ? descriptor with { Aliases = ["lm_head.weight"] } : descriptor),
        ]);

        LogicalTensorTable primaryOnly = Table(("output.weight", [6, 4]));
        LogicalTensorTable aliasOnly = Table(("lm_head.weight", [6, 4]));
        LogicalTensorTable both = Table(("output.weight", [6, 4]), ("lm_head.weight", [6, 4]));

        Assert.Equal("output.weight", aliased[TensorRole.OutputHead].Resolve(primaryOnly)?.Name);
        Assert.Equal("lm_head.weight", aliased[TensorRole.OutputHead].Resolve(aliasOnly)?.Name);
        Assert.Equal("output.weight", aliased[TensorRole.OutputHead].Resolve(both)?.Name);
    }

    [Fact]
    public void TensorMap_RejectsIncompleteOrDuplicatedRoleCoverage()
    {
        IReadOnlyList<TensorDescriptor> canonical = ModelProfiles.Resolve("llama").Tensors.Descriptors;

        Assert.Throws<ModelException>(() => new TensorMap([.. canonical.Where(d => d.Role != TensorRole.OutputNorm)]));
        Assert.Throws<ModelException>(() => new TensorMap([.. canonical, canonical[0]]));
    }

    // ---------------------------------------------------------------- tensor lookup through the plan at Open

    [Fact]
    public void TransformerModel_Open_ResolvesTensorsThroughThePlanMap_ForQwen2WithOptionalTensorsPresent()
    {
        string path = WriteTemp(BuildGguf(Metadata("qwen2", tiny: true), [.. TinyTensors(withOptional: true)]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        Assert.Equal("qwen2", model.Plan.Architecture);
        Assert.True(model.Plan.HasOutputWeights); // output.weight resolved through the map
        Assert.Equal(model.Plan.ToDesc(), model.Desc);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 2, backend);
        using SequenceKv kv = new(pool);
        Assert.Equal(6, model.DecodeNext(3, kv).Length); // untied head over output.weight: one logit per vocabulary entry
    }

    [Fact]
    public void TransformerModel_Open_MissingRequiredTensor_StillNamesTheMapsPrimaryName()
    {
        List<TensorDef> tensors = [.. TinyTensors(withOptional: false)];
        tensors.RemoveAll(t => t.Name == "blk.0.attn_output.weight");

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata("llama", tiny: true), [.. tensors])));
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));
        Assert.Contains("blk.0.attn_output.weight", ex.Message);
        Assert.Contains("llama", ex.Message);
    }

    // ---------------------------------------------------------------- declared feature flags and head-dimension sources

    [Fact]
    public void ModelPlan_UseBiasDeclared_ResolvesTheDeclaredFlag_ForLlamaAndQwen2()
    {
        foreach (string architecture in new string[] { "llama", "qwen2" })
        {
            foreach (bool declared in new bool[] { true, false })
            {
                using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
                    [.. Metadata(architecture), ($"{architecture}.attention.use_bias", declared)], [])));
                ModelPlan plan = ModelPlans.Resolve(gguf);

                Assert.Equal(BiasPolicy.Declared, plan.Bias);
                Assert.Equal(declared, plan.UseAttentionBias);
            }

            // An absent key defaults to false: the plan still resolves the declared policy.
            using GgufFile absent = GgufFile.Open(WriteTemp(BuildGguf(Metadata(architecture), [])));
            ModelPlan absentPlan = ModelPlans.Resolve(absent);
            Assert.Equal(BiasPolicy.Declared, absentPlan.Bias);
            Assert.False(absentPlan.UseAttentionBias);
        }
    }

    [Fact]
    public void ModelPlan_UseBiasDeclaredTrue_MissingBiasTensor_FailsLoadNamingIt_AndTheDeclaringKey()
    {
        string path = WriteTemp(BuildGguf(
            [.. Metadata("qwen2", tiny: true), ("qwen2.attention.use_bias", true)],
            [.. TinyTensors(withOptional: false)]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("blk.0.attn_q.bias", ex.Message);
        Assert.Contains("qwen2.attention.use_bias", ex.Message);
    }

    [Fact]
    public void ModelPlan_UseBiasDeclaredTrue_LoadsEveryDeclaredBiasTensor_AndForwards()
    {
        string path = WriteTemp(BuildGguf(
            [.. Metadata("qwen2", tiny: true), ("qwen2.attention.use_bias", true)],
            [.. TinyTensors(withOptional: true)]));

        List<string> logLines = [];
        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, logLines.Add);

        Assert.True(model.Plan.UseAttentionBias);
        Assert.Empty(logLines); // nothing declared off, nothing ignored

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 2, backend);
        using SequenceKv kv = new(pool);
        Assert.Equal(6, model.DecodeNext(3, kv).Length);
    }

    [Fact]
    public void ModelPlan_UseBiasDeclaredFalse_WithBiasTensorsPresent_LogsTheIgnoredTensorsOnce()
    {
        List<string> logLines = [];
        string path = WriteTemp(BuildGguf(
            [.. Metadata("qwen2", tiny: true), ("general.name", "Declared Off"), ("qwen2.attention.use_bias", false)],
            [.. TinyTensors(withOptional: true)]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, logLines.Add);

        Assert.False(model.Plan.UseAttentionBias);
        Assert.Single(logLines);
        Assert.StartsWith("bias_ignored ", logLines[0]);
        Assert.Contains("model='Declared Off'", logLines[0]);
        Assert.Contains("architecture='qwen2'", logLines[0]);
        foreach (string tensor in new string[] { "blk.0.attn_q.bias", "blk.0.attn_k.bias", "blk.0.attn_v.bias", "blk.0.attn_output.bias" })
        {
            Assert.Contains(tensor, logLines[0]);
        }

        // Declared false with no bias tensors present: nothing to ignore, so no line.
        List<string> quietLines = [];
        using GgufFile quiet = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("qwen2", tiny: true), ("qwen2.attention.use_bias", false)],
            [.. TinyTensors(withOptional: false)])));
        using TransformerModel quietModel = TransformerModel.Open(quiet, backend, maxContextTokens: 8, quietLines.Add);
        Assert.False(quietModel.Plan.UseAttentionBias);
        Assert.Empty(quietLines);
    }

    [Fact]
    public void ModelPlan_NormBiasAndQkNormDeclared_ResolveTheirFlags_AndMissingTensorsFailLoadNamingThem()
    {
        using (GgufFile flagsGguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama", tiny: true), ("llama.attention.norm_bias", true), ("llama.attention.use_qk_norm", true)], []))))
        {
            ModelPlan plan = ModelPlans.Resolve(flagsGguf);
            Assert.True(plan.UseNormBias);
            Assert.Equal(QkNormMode.PerHeadWeighted, plan.QkNorm);
        }

        string path = WriteTemp(BuildGguf(
            [.. Metadata("llama", tiny: true), ("llama.attention.norm_bias", true), ("llama.attention.use_qk_norm", true)],
            [.. TinyTensors(withOptional: false)]));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));
        Assert.Contains("blk.0.attn_norm.bias", ex.Message);

        // Without the norm-bias tensors but with use_qk_norm declared, the QK-norm tensors are named instead.
        string qkPath = WriteTemp(BuildGguf(
            [.. Metadata("llama", tiny: true), ("llama.attention.use_qk_norm", true)],
            [.. TinyTensors(withOptional: false)]));
        using GgufFile qkGguf = GgufFile.Open(qkPath);
        ModelException qkEx = Assert.Throws<ModelException>(() => TransformerModel.Open(qkGguf, backend, maxContextTokens: 8));
        Assert.Contains("blk.0.attn_q_norm.weight", qkEx.Message);

        // Both keys absent: both flags default to false.
        using GgufFile absent = GgufFile.Open(WriteTemp(BuildGguf(Metadata("llama", tiny: true), [])));
        ModelPlan absentPlan = ModelPlans.Resolve(absent);
        Assert.False(absentPlan.UseNormBias);
        Assert.Equal(QkNormMode.Off, absentPlan.QkNorm);
    }

    [Fact]
    public void ModelPlan_HeadLength_ResolvesAnIndependentHeadDim_GemmaShaped()
    {
        // A gemma-shaped geometry: head_count x head_dim (8 x 256) differs from embedding_length (2304).
        List<(string Key, object Value)> gemma =
        [
            ("general.architecture", "llama"),
            ("llama.embedding_length", 2304),
            ("llama.block_count", 26),
            ("llama.attention.head_count", 8),
            ("llama.attention.head_count_kv", 4),
            ("llama.feed_forward_length", 2304),
            ("llama.attention.head_length", 256),
            ("llama.attention.layer_norm_rms_epsilon", 1e-6f),
            ("llama.vocab_size", 256_000),
        ];

        using (GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(gemma, []))))
        {
            ModelPlan plan = ModelPlans.Resolve(gguf);
            Assert.Equal(256, plan.HeadDim); // the declared dimension is honored, not hidden / head_count
            Assert.Equal(2304, plan.HiddenSize);
            Assert.Equal(8, plan.HeadCount);
        }

        // head_length is the preferred source and must agree with a declared key_length / value_length.
        using (GgufFile agree = GgufFile.Open(WriteTemp(BuildGguf(
            [.. gemma, ("llama.attention.key_length", 256), ("llama.attention.value_length", 256)], []))))
        {
            Assert.Equal(256, ModelPlans.Resolve(agree).HeadDim);
        }

        using GgufFile disagree = GgufFile.Open(WriteTemp(BuildGguf(
            [.. gemma, ("llama.attention.key_length", 128), ("llama.attention.value_length", 256)], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(disagree));
        Assert.Contains("disagree", ex.Message);
        Assert.Contains("head_length", ex.Message);
    }

    // ---------------------------------------------------------------- shared helpers

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-plan-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // Canonical llama/qwen2 metadata pairs (parse-only fixtures need no tensors); the tiny variant matches TinyTensors.
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

    // A complete single-layer model with hidden 4, heads 2, kv-heads 2, ffn 8, vocab 6 — all F32 tensors.
    private static List<TensorDef> TinyTensors(bool withOptional)
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

        if (withOptional)
        {
            tensors.Add(new("blk.0.attn_q.bias", [4]));
            tensors.Add(new("blk.0.attn_k.bias", [4]));
            tensors.Add(new("blk.0.attn_v.bias", [4]));
            tensors.Add(new("blk.0.attn_output.bias", [4]));
            tensors.Add(new("output.weight", [6, 4]));
            tensors.Add(new("output.bias", [6]));
        }

        return tensors;
    }

    private static LogicalTensorTable Table(params (string Name, int[] Dims)[] tensors)
    {
        List<LogicalTensor> table = [];
        foreach ((string name, int[] dims) in tensors)
        {
            long elements = 1;
            foreach (int dim in dims)
            {
                elements *= dim;
            }

            table.Add(new LogicalTensor(name, new TensorShape(dims), QuantFormat.F32, 0, elements * 4));
        }

        return new LogicalTensorTable(table);
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
