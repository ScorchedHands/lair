using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Load-time resolution for the phi3 profile: the strategy values the registered profile carries — RMS block norms, the
/// SiLU-gated feed-forward, declared bias policy with QK-norm off unless the file says otherwise, pre-only norm placement,
/// one fused qkv buffer per layer read as three head-region slices, and the gate half of the gated pair living inside the
/// up tensor — the geometry the real file declares (32 heads over a 3072-wide hidden size with no key_length or
/// value_length anywhere, so the head dimension is derived at 96 and equals the declared rotary slice), the untied output
/// head, and the fused-layout discipline the forward engine holds: one buffer per layer whose query region, key region,
/// and value region are addressed as slices of that payload rather than split into three matrices. The counterfactuals
/// prove those slices carry meaning — reordering regions inside the same tensor with identical shape and format moves the
/// logits, so a wrong region offset cannot pass unnoticed. A file carrying this family's tensors under the split names is
/// refused naming both sides; a gate matrix or rope-factor vectors this block never reads are reported or ignored rather
/// than loaded. Declared values that cannot describe a rotary slice fail naming the key and the value. Against the pinned
/// fixture the case asserts the file's own keys, its exact 195-tensor inventory, and its formats. Cases skip when the
/// fixture file is absent.
/// </summary>
[Trait("Category", "Loader")]
[Collection("model-loaded")]
public sealed class Phi3PlanTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

    private const string FixtureRelativePath = "models/Phi-3-mini-4k-instruct-Q2_K.gguf";
    private const string FixtureSha256 = "72237d1e7c56abc777a2f4d1482f3eff4bcfc156566f226749bc47baa9608285";

    // The tiny fixture geometry every forward case below runs against: two heads of a 4-wide dimension over an
    // 8-wide hidden size (so the fused buffer is one 8-row query region followed by two 8-row kv regions), and a
    // 16-wide intermediate whose gate/up pair ships as one 32-row tensor.
    private const int TinyHidden = 8;
    private const int TinyHeads = 2;
    private const int TinyHeadDim = 4;
    private const int TinyKvHeads = 2;
    private const int TinyIntermediate = 16;
    private const int TinyLayers = 2;

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

    // ---------------------------------------------------------------- the registered profile itself

    [Fact]
    public void ModelProfiles_Phi3_ResolvesFusedQkvAndGateFreeFeedForward()
    {
        Assert.Contains("phi3", ModelProfiles.SupportedArchitectures);

        ModelProfile profile = ModelProfiles.Resolve("phi3");

        Assert.Equal("phi3.", profile.Prefix);
        Assert.Equal(NormStrategy.RmsNorm, profile.Norm);
        Assert.Equal(ActivationStrategy.SwiGLU, profile.Activation);
        Assert.Equal(BiasPolicy.Declared, profile.Bias);
        Assert.Equal(RopeStrategy.RotateHalfFullHead, profile.Rope);
        Assert.Equal(NormPlacement.PreOnly, profile.NormPlacement);

        // This family's conversion ships no per-head q/k norm vectors and no attention-bias vectors: both features stay
        // off unless the file declares them on.
        Assert.Equal(QkNormMode.Off, profile.DefaultQkNorm);
        Assert.False(profile.DefaultAttentionBias);

        // The two layout values this family exists for: one fused qkv buffer per layer, and a gate half that rides inside
        // the up tensor rather than naming its own matrix.
        Assert.Equal(QkvLayout.FusedBufferSlices, profile.AttentionWeights);
        Assert.Equal(FeedForwardGateSource.UpTensorHalves, profile.GateSource);

        // No embedding gain and no window stride: this family's reference graph runs neither.
        Assert.False(profile.ScaleInputEmbeddingsBySqrtHidden);
        Assert.Equal(WindowInterleave.AllLayers, profile.WindowInterleave);

        TensorMap tensors = profile.Tensors;

        // The fused buffer is the required projection tensor of every layer; the three split names stay declared so a file
        // carrying them is recognizable material this layout does not read rather than an unnameable table. The gate role
        // relaxes for the same reason: no conversion in this family ships it.
        Assert.Equal("blk.{layer}.attn_qkv.weight", tensors[TensorRole.AttentionQkv].Name);
        Assert.True(tensors[TensorRole.AttentionQkv].Required);
        Assert.False(tensors[TensorRole.AttentionQuery].Required);
        Assert.False(tensors[TensorRole.AttentionKey].Required);
        Assert.False(tensors[TensorRole.AttentionValue].Required);
        Assert.False(tensors[TensorRole.FeedForwardGate].Required);

        // A profile whose block projects through three matrices leaves the fused role out of its map entirely, which is why
        // a tensor under that name stays unexpected material for every other family.
        ModelProfile canonical = ModelProfiles.Resolve("llama");
        Assert.DoesNotContain(canonical.Tensors.Descriptors, descriptor => descriptor.Role == TensorRole.AttentionQkv);

        Assert.True(
            !TensorMap.IsProfileOptionalRole(TensorRole.FeedForwardGate) && TensorMap.IsProfileOptionalRole(TensorRole.AttentionQkv),
            "the fused qkv role is the only non-sparse role a profile may leave undeclared");

        // Both layout values default to what every other family runs, so registration cannot change any existing profile.
        Assert.Equal(QkvLayout.SeparateTensors, canonical.AttentionWeights);
        Assert.Equal(FeedForwardGateSource.SeparateGateMatrix, canonical.GateSource);
    }

    // ---------------------------------------------------------------- plan resolution on the file's own geometry

    [Fact]
    public void ModelPlan_Phi3DeclaredGeometry_ResolvesDerivedHeadDimAndLayoutValues()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata("phi3"), [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal("phi3", plan.Architecture);
        Assert.Equal("phi3.", plan.Prefix);
        Assert.Equal("phi3", plan.Profile);
        Assert.Equal(32, plan.LayerCount);
        Assert.Equal(3072, plan.HiddenSize);
        Assert.Equal(32, plan.HeadCount);
        Assert.Equal(32, plan.KvHeadCount);
        Assert.Equal(8192, plan.IntermediateSize);

        // No head_length, key_length or value_length is declared anywhere in this family's file: the dimension comes from
        // hidden / head_count and lands exactly on the rotary slice the file declares.
        Assert.Equal(96, plan.HeadDim); // hidden 3072 / heads 32; no key_length or value_length declares it here
        Assert.Equal(96, plan.RopeRotaryDimensions);
        Assert.True(plan.HeadDim == plan.RopeRotaryDimensions, "the declared slice covers this family's whole head");

        Assert.Equal(NormStrategy.RmsNorm, plan.Norm);
        Assert.Equal(ActivationStrategy.SwiGLU, plan.Activation);
        Assert.Equal(RopeStrategy.RotateHalfFullHead, plan.Rope);
        Assert.Equal(NormPlacement.PreOnly, plan.NormPlacement);
        Assert.Equal(QkvLayout.FusedBufferSlices, plan.AttentionWeights);
        Assert.Equal(FeedForwardGateSource.UpTensorHalves, plan.GateSource);

        // Feature flags this file declares nothing for: bias and QK-norm resolve off through the profile defaults, no norm
        // bias, no softcap, no window, one rotation base at the value a file declaring none is read at.
        Assert.False(plan.UseAttentionBias);
        Assert.False(plan.UseNormBias);
        Assert.Equal(QkNormMode.Off, plan.QkNorm);
        Assert.Equal(0f, plan.AttentionSoftcap);
        Assert.Equal(0f, plan.FinalLogitSoftcap);
        Assert.Equal(0, plan.AttentionWindow);
        Assert.True(Math.Abs(plan.RopeTheta - ModelPlan.DefaultRopeTheta) < 1e-6f, $"undeclared rotary base {plan.RopeTheta:G9}");
        Assert.Null(plan.SparseFfn);

        // The fused buffer's row count is the plan's own region arithmetic: every head of q, then k, then value.
        Assert.Equal(9216L, (long)plan.HeadCount * plan.HeadDim + 2L * plan.KvHeadCount * plan.HeadDim);

        // The gate-free block carries both halves in one tensor: the up matrix's rows are two intermediates wide.
        Assert.Equal(16384L, 2L * plan.IntermediateSize);
    }

    [Fact]
    public void ModelPlan_Phi3LayoutValuesComeFromTheProfile_NotTheFile()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata("phi3"), [])));

        ModelPlan resolved = ModelPlans.Resolve(gguf);
        ModelPlan split = ModelPlans.Resolve(gguf, ModelProfiles.Resolve("phi3") with { AttentionWeights = QkvLayout.SeparateTensors });
        ModelPlan gated = ModelPlans.Resolve(gguf, ModelProfiles.Resolve("phi3") with { GateSource = FeedForwardGateSource.SeparateGateMatrix });

        Assert.Equal(QkvLayout.FusedBufferSlices, resolved.AttentionWeights);
        Assert.Equal(FeedForwardGateSource.UpTensorHalves, resolved.GateSource);

        // Neither value has a metadata key: only an explicitly supplied profile changes either one.
        Assert.Equal(QkvLayout.SeparateTensors, split.AttentionWeights);
        Assert.Equal(FeedForwardGateSource.SeparateGateMatrix, gated.GateSource);
    }

    [Fact]
    public void ModelPlan_Phi3RopeScalingDeclaration_ResolvesTheDeclaredVariant()
    {
        // The 4k conversion declares no scaling variant and ships no rope-factor vectors; a file that does declare one
        // resolves it exactly as every other family's plan does, because this key is shared machinery.
        using GgufFile plain = GgufFile.Open(WriteTemp(BuildGguf(Metadata("phi3"), [])));
        Assert.Null(ModelPlans.Resolve(plain).RopeScaling);

        using GgufFile scaled = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("phi3"), ("phi3.rope.scaling.type", "yarn"), ("phi3.rope.scaling.factor", 2.0f), ("phi3.rope.scaling.original_context_length", 4096)], [])));

        ModelPlan plan = ModelPlans.Resolve(scaled);
        Assert.NotNull(plan.RopeScaling);
        Assert.Equal(2f, plan.RopeScaling!.Multiplier);
        Assert.Equal(4096, plan.RopeScaling.OriginalContextLength);
    }

    // ---------------------------------------------------------------- declared-value failures naming key and value

    [Fact]
    public void ModelPlan_Phi3InvalidRotarySlice_FailsNamingKeyAndValue()
    {
        foreach (object declared in new object[] { 95, 0, -2, 128 })
        {
            List<(string Key, object Value)> pairs = [.. Metadata("phi3")];
            pairs.RemoveAll(pair => pair.Key == "phi3.rope.dimension_count");
            pairs.Add(("phi3.rope.dimension_count", declared));

            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(pairs, [])));
            ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

            Assert.Contains("phi3.rope.dimension_count", ex.Message);
            Assert.Contains(declared.ToString()!, ex.Message);
        }
    }

    [Fact]
    public void ModelPlan_Phi3InvalidWindowAndSoftcap_FailNamingKeyAndValue()
    {
        foreach (object declared in new object[] { 0, -512 })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
                [.. Metadata("phi3"), ("phi3.attention.sliding_window", declared)], [])));

            ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));
            Assert.Contains("phi3.attention.sliding_window", ex.Message);
            Assert.Contains(declared.ToString()!, ex.Message);
        }

        foreach (object declared in new object[] { 0f, -4f, float.NaN })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
                [.. Metadata("phi3"), ("phi3.attn_logit_softcapping", declared)], [])));

            ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));
            Assert.Contains("phi3.attn_logit_softcapping", ex.Message);
        }
    }

    [Fact]
    public void ModelPlan_Phi3MissingRmsEpsilon_FailsNamingTheKey()
    {
        List<(string Key, object Value)> pairs = [.. Metadata("phi3")];
        pairs.RemoveAll(pair => pair.Key == "phi3.attention.layer_norm_rms_epsilon");

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(pairs, [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("phi3.attention.layer_norm_rms_epsilon", ex.Message);
    }

    // ---------------------------------------------------------------- the forward engine's side of the fused layout

    [Fact]
    public void TransformerModel_Phi3FusedBufferOnly_LoadsAndForwards_WithNoSplitTensorsPresent()
    {
        using CpuBackend backend = new();
        string path = WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors()));

        using GgufFile gguf = GgufFile.Open(path);
        LogicalTensorTable table = gguf.Tensors;

        Assert.DoesNotContain(table.Tensors, t => t.Name == "blk.0.attn_q.weight");
        Assert.DoesNotContain(table.Tensors, t => t.Name == "blk.0.attn_k.weight");
        Assert.DoesNotContain(table.Tensors, t => t.Name == "blk.0.attn_v.weight");
        Assert.Contains(table.Tensors, t => t.Name == "blk.0.attn_qkv.weight");

        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        Assert.Equal(QkvLayout.FusedBufferSlices, model.Plan.AttentionWeights);
        Assert.True(
            plan_rows(model) == TinyHeads * TinyHeadDim + 2L * TinyKvHeads * TinyHeadDim,
            "the fused buffer must carry the query region followed by both kv regions");

        float[] logits = ForwardTinyPhi3(backend, path);
        Assert.Equal(TinyVocabSize(), logits.Length);
        Assert.True(logits.All(float.IsFinite), "a tiny phi3 forward must produce finite logits throughout");
    }

    [Fact]
    public void TransformerModel_Phi3SplitTensorsInsteadOfFusedBuffer_FailNamingBothSides()
    {
        using CpuBackend backend = new();
        string path = WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors(splitQkv: true)));

        using GgufFile gguf = GgufFile.Open(path);
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("blk.0.attn_qkv.weight", ex.Message);
        Assert.Contains("blk.0.attn_q.weight", ex.Message);
        Assert.Contains("blk.0.attn_k.weight", ex.Message);
        Assert.Contains("blk.0.attn_v.weight", ex.Message);
        Assert.Contains("fused", ex.Message);
    }

    [Fact]
    public void TransformerModel_Phi3FusedBufferRowCountMismatch_FailsNamingRegionsAndShape()
    {
        using CpuBackend backend = new();
        string path = WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors(qkvRowsShort: true)));

        using GgufFile gguf = GgufFile.Open(path);
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        // The buffer is one query region short of the three this layout reads.
        Assert.Contains("blk.0.attn_qkv.weight", ex.Message);
        Assert.Contains("[24, 8]", ex.Message);
    }

    [Fact]
    public void TransformerModel_Phi3QkvRegionSliceOrder_IsWhatTheForwardReads_Counterfactually()
    {
        using CpuBackend backend = new();

        // Same file shape, same tensor names and formats, same bytes total: only the order of the three head regions inside
        // blk.0.attn_qkv.weight differs. The forward addresses each role by its own region offset, so a buffer whose
        // key/value halves are exchanged must move the logits — an engine that ignored slice offsets would forward these two
        // files identically and fail here.
        float[] natural = ForwardTinyPhi3(backend, WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors())));

        Assert.True(
            !natural.AsSpan().SequenceEqual(ForwardTinyPhi3(backend, WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors(), qkvSwapKeyValue: true)))),
            "exchanging the key and value regions of the fused buffer must change every position that reads them");

        Assert.True(
            !natural.AsSpan().SequenceEqual(ForwardTinyPhi3(backend, WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors(), qkvSwapQueryKey: true)))),
            "exchanging the query and key regions of the fused buffer must change the forward");

        // Rotating the whole buffer by one row keeps its shape and format intact but moves every region boundary, so no role
        // reads the rows it was resolved for.
        Assert.True(
            !natural.AsSpan().SequenceEqual(ForwardTinyPhi3(backend, WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors(), qkvRotateOneRow: true)))),
            "a one-row shift of the fused buffer moves every region offset and must change the forward");

        // The same file twice reproduces itself exactly: the differences above are the payload, not a nondeterministic read.
        float[] again = ForwardTinyPhi3(backend, WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors())));
        Assert.True(natural.AsSpan().SequenceEqual(again.AsSpan()), "the fused layout must resolve deterministically");
    }

    [Fact]
    public void TransformerModel_Phi3GateHalvesAreTheUpTensorRows_Counterfactually_AndAGateMatrixIsNeverRead()
    {
        using CpuBackend backend = new();

        float[] natural = ForwardTinyPhi3(backend, WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors())));

        // The two halves of blk.N.ffn_up.weight are the gate side and the up side. Swapping them keeps the tensor's shape
        // and every byte count identical while feeding the activation a different pair, so the logits must move.
        Assert.True(
            !natural.AsSpan().SequenceEqual(ForwardTinyPhi3(backend, WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors(), swapGateUpHalves: true)))),
            "the gate and up halves of blk.N.ffn_up.weight are distinct operands and must not be interchangeable");

        // A file that still ships a separate gate matrix is not consumed by this layout, and carrying it changes nothing.
        List<string> lines = [];
        string withGatePath = WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors(denseGate: true)));

        using (GgufFile gguf = GgufFile.Open(withGatePath))
        {
            Assert.Contains(gguf.Tensors.Tensors, t => t.Name == "blk.0.ffn_gate.weight");

            float[] withGate;
            using (TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, lines.Add))
            {
                Assert.Equal(FeedForwardGateSource.UpTensorHalves, model.Plan.GateSource);

                using BlockPool pool = new(2, model.Plan.LayerCount, model.Plan.KvHeadCount, model.Plan.HeadDim, 4, backend);
                using SequenceKv kv = new(pool);
                model.Prefill([1, 2, 3], kv);
                withGate = model.DecodeNext(4, kv).ToArray();
            }

            Assert.True(natural.AsSpan().SequenceEqual(withGate), "a gate matrix this layout does not read must not reach the forward");
        }

        List<string> plainLines = [];
        string plainPath = WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors()));
        using (GgufFile gguf = GgufFile.Open(plainPath))
        {
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, plainLines.Add);
        }

        Assert.DoesNotContain(plainLines, line => line.StartsWith("ffn_gate_ignored ", StringComparison.Ordinal));

        string ignoredLine = lines.Single(line => line.StartsWith("ffn_gate_ignored ", StringComparison.Ordinal));
        Assert.Contains("blk.0.ffn_gate.weight", ignoredLine);
        Assert.Contains("blk.1.ffn_gate.weight", ignoredLine);
        Assert.Contains("architecture='phi3'", ignoredLine);
    }

    [Fact]
    public void TransformerModel_Phi3RopeFactorVectors_AreNeitherRequiredNorRead()
    {
        using CpuBackend backend = new();

        float[] withoutFactors = ForwardTinyPhi3(backend, WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors())));

        // The long-context variants of this family ship one rotary-factor vector per block; the 4k file ships none at all.
        // Neither side is part of what this profile loads or forwards, so adding them changes no logit and omitting them
        // still opens.
        string withFactors = WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors(ropeFactors: true)));

        using (GgufFile gguf = GgufFile.Open(withFactors))
        {
            Assert.Contains(gguf.Tensors.Tensors, t => t.Name == "blk.0.rope_factors_long.weight");
            Assert.Contains(gguf.Tensors.Tensors, t => t.Name == "blk.0.rope_factors_short.weight");

            float[] logits;
            using (TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8))
            {
                Assert.Null(model.Plan.RopeScaling);

                using BlockPool pool = new(2, model.Plan.LayerCount, model.Plan.KvHeadCount, model.Plan.HeadDim, 4, backend);
                using SequenceKv kv = new(pool);
                model.Prefill([1, 2, 3], kv);
                logits = model.DecodeNext(4, kv).ToArray();
            }

            Assert.True(withoutFactors.AsSpan().SequenceEqual(logits), "rope-factor vectors must stay outside this file's forward");
        }
    }

    [Fact]
    public void TransformerModel_Phi3UntiedOutputHead_IsReadFromItsOwnTensor()
    {
        using CpuBackend backend = new();

        string untiedPath = WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors()));
        string tiedPath = WriteTemp(BuildGguf(Metadata("phi3", tiny: true), TinyTensors(tiedHead: true)));

        using GgufFile untied = GgufFile.Open(untiedPath);
        using GgufFile tied = GgufFile.Open(tiedPath);

        Assert.Contains(untied.Tensors.Tensors, t => t.Name == "output.weight");
        Assert.DoesNotContain(tied.Tensors.Tensors, t => t.Name == "output.weight");

        ModelPlan untiedPlan = ModelPlans.Resolve(untied);
        ModelPlan tiedPlan = ModelPlans.Resolve(tied);

        Assert.True(untiedPlan.HasOutputWeights, "a file carrying output.weight resolves an untied head");
        Assert.False(tiedPlan.HasOutputWeights, "a file without it ties the logits to the token-embedding rows");

        float[] untiedLogits = ForwardTinyPhi3(backend, untiedPath);
        float[] tiedLogits = ForwardTinyPhi3(backend, tiedPath);

        Assert.Equal(untiedLogits.Length, tiedLogits.Length);
        Assert.True(
            !untiedLogits.AsSpan().SequenceEqual(tiedLogits.AsSpan()),
            "the untied output matrix and the embedding rows are different weights and must not resolve to one another");
    }

    // ---------------------------------------------------------------- the pinned fixture itself

    [Fact]
    public void ModelPlan_RealPhi3Fixture_ResolvesDeclaredGeometryAndLayout()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(FixtureRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        AssertFixtureHash(modelPath);

        using GgufFile gguf = GgufFile.Open(modelPath);
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal("phi3", plan.Architecture);
        Assert.Equal("phi3", plan.Profile);
        Assert.Equal(32, plan.LayerCount);
        Assert.Equal(3072, plan.HiddenSize);
        Assert.Equal(32, plan.HeadCount);
        Assert.Equal(32, plan.KvHeadCount);
        Assert.Equal(8192, plan.IntermediateSize);

        // The head dimension is derived — this file declares no key_length, value_length or head_length anywhere — and it
        // equals the rotary slice the file does declare, so the rotation covers each head in full.
        Assert.DoesNotContain("phi3.attention.key_length", gguf.Metadata.Keys);
        Assert.DoesNotContain("phi3.attention.value_length", gguf.Metadata.Keys);
        Assert.DoesNotContain("phi3.attention.head_length", gguf.Metadata.Keys);
        Assert.Equal(96, plan.HeadDim);
        Assert.Equal(96, plan.RopeRotaryDimensions);

        // No rotary frequency is declared: the rotation runs at the base a file declaring none takes. No scaling variant
        // and no softcap key either, so nothing scaled or capped happens here whatever the profile says about them.
        Assert.DoesNotContain("phi3.rope.freq_base", gguf.Metadata.Keys);
        Assert.True(Math.Abs(plan.RopeTheta - ModelPlan.DefaultRopeTheta) < 1e-6f, $"undeclared rotary base {plan.RopeTheta:G9}");
        Assert.Null(plan.RopeScaling);
        Assert.Equal(0f, plan.AttentionSoftcap);
        Assert.Equal(0f, plan.FinalLogitSoftcap);

        // The vocabulary size has no declared key and falls back to the tokenizer's own length.
        Assert.DoesNotContain("phi3.vocab_size", gguf.Metadata.Keys);
        Assert.Equal(gguf.Metadata.GetTokens().Length, plan.VocabSize);
        Assert.Equal(32_064, plan.VocabSize);

        // Both layout values land in the plan from the profile.
        Assert.Equal(QkvLayout.FusedBufferSlices, plan.AttentionWeights);
        Assert.Equal(FeedForwardGateSource.UpTensorHalves, plan.GateSource);
    }

    [Fact]
    public void Phi3Fixture_TensorInventory_IsOneFusedQkvAndOneGateFreeUpPerLayer()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(FixtureRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        AssertFixtureHash(modelPath);

        LogicalTensorTable table = new GgufModelLoader().Load(modelPath);

        Dictionary<string, (QuantFormat Format, int Count)> expected = new(StringComparer.Ordinal)
        {
            ["attn_qkv.weight"] = (QuantFormat.Q2_K, 32),
            ["attn_output.weight"] = (QuantFormat.Q3_K, 32),
            ["attn_norm.weight"] = (QuantFormat.F32, 32),
            ["ffn_up.weight"] = (QuantFormat.Q2_K, 32),
            ["ffn_down.weight"] = (QuantFormat.Q3_K, 32),
            ["ffn_norm.weight"] = (QuantFormat.F32, 32),
            ["token_embd.weight"] = (QuantFormat.Q2_K, 1),
            ["output_norm.weight"] = (QuantFormat.F32, 1),
            ["output.weight"] = (QuantFormat.Q6_K, 1),
        };

        Dictionary<string, int> observedCounts = new(StringComparer.Ordinal);
        Dictionary<string, QuantFormat> observedFormats = new(StringComparer.Ordinal);

        foreach (LogicalTensor tensor in table.Tensors)
        {
            string suffix = tensor.Name.StartsWith("blk.", StringComparison.Ordinal)
                ? tensor.Name[(tensor.Name.IndexOf('.', 4) + 1)..]
                : tensor.Name;

            if (!observedCounts.TryAdd(suffix, 1))
            {
                observedCounts[suffix]++;
            }

            Assert.True(
                observedFormats.TryAdd(suffix, tensor.Format) || observedFormats[suffix] == tensor.Format,
                $"tensors under '{suffix}' mix quant formats");
        }

        // 32 layers x 6 tensors + the three global ones. No split q/k/v matrix, no gate matrix, and no rope-factor vector
        // anywhere in this file — every one of these 195 tensors is a role this profile reads or a norm vector.
        Assert.Equal(195, table.Tensors.Count);
        Assert.Equal(expected.Keys.OrderBy(k => k, StringComparer.Ordinal), observedCounts.Keys.OrderBy(k => k, StringComparer.Ordinal));

        foreach ((string suffix, (QuantFormat format, int count)) in expected)
        {
            Assert.True(observedCounts.TryGetValue(suffix, out int observed), $"no tensors found under '{suffix}'");
            Assert.True(observed == count, $"'{suffix}': expected {count} tensors of format {format}, got {observed}");
            Assert.Equal(format, observedFormats[suffix]);
        }

        Assert.DoesNotContain(table.Tensors, t => t.Name.EndsWith("attn_q.weight", StringComparison.Ordinal));
        Assert.DoesNotContain(table.Tensors, t => t.Name.EndsWith("attn_k.weight", StringComparison.Ordinal));
        Assert.DoesNotContain(table.Tensors, t => t.Name.EndsWith("attn_v.weight", StringComparison.Ordinal));
        Assert.DoesNotContain(table.Tensors, t => t.Name.Contains("ffn_gate", StringComparison.Ordinal));
        Assert.DoesNotContain(table.Tensors, t => t.Name.Contains("rope_factors", StringComparison.Ordinal));

        // Logical (row-major) shapes: the fused buffer is 32 heads of 96 wide for q plus two such regions for k and value,
        // over hidden columns; the gate-free up tensor carries both halves' rows over hidden columns; the down projection
        // reads the intermediate stream back into hidden; and the untied head is vocab rows over hidden columns.
        Assert.Equal([9216, 3072], table.Tensors.First(t => t.Name == "blk.0.attn_qkv.weight").Shape.Dims);
        Assert.Equal([16384, 3072], table.Tensors.First(t => t.Name == "blk.0.ffn_up.weight").Shape.Dims);
        Assert.Equal([3072, 8192], table.Tensors.First(t => t.Name == "blk.0.ffn_down.weight").Shape.Dims);
        Assert.Equal([3072, 3072], table.Tensors.First(t => t.Name == "blk.0.attn_output.weight").Shape.Dims);
        Assert.Equal([3072], table.Tensors.First(t => t.Name == "blk.0.attn_norm.weight").Shape.Dims);
        Assert.Equal([3072], table.Tensors.First(t => t.Name == "blk.0.ffn_norm.weight").Shape.Dims);
        Assert.Equal([32_064, 3072], table.Tensors.First(t => t.Name == "token_embd.weight").Shape.Dims);
        Assert.Equal([32_064, 3072], table.Tensors.First(t => t.Name == "output.weight").Shape.Dims);

        // The fused buffer's payload is exactly its rows at this format's packed row size: three regions of whole rows with
        // no padding between them, which is what makes each role addressable as a slice.
        LogicalTensor qkv = table.Tensors.First(t => t.Name == "blk.0.attn_qkv.weight");
        long stride = qkv.Shape.RowStrideBytes(qkv.Format);
        Assert.Equal(9216 * stride, qkv.SizeBytes);

        LogicalTensor up = table.Tensors.First(t => t.Name == "blk.0.ffn_up.weight");
        long upStride = up.Shape.RowStrideBytes(up.Format);
        Assert.Equal(16384 * upStride, up.SizeBytes);
    }

    // ---------------------------------------------------------------- helpers

    private static void AssertFixtureHash(string modelPath)
    {
        using System.Security.Cryptography.SHA256 hasher = System.Security.Cryptography.SHA256.Create();
        using FileStream stream = File.OpenRead(modelPath);
        string actual = Convert.ToHexString(hasher.ComputeHash(stream)).ToLowerInvariant();

        Assert.True(
            string.Equals(actual, FixtureSha256, StringComparison.Ordinal),
            $"phi3 loader guard: {FixtureRelativePath} hashes to {actual}, not the pinned {FixtureSha256}: " +
            "these assertions were written against that exact file.");
    }

    private static long plan_rows(TransformerModel model) =>
        (long)model.Plan.HeadCount * model.Plan.HeadDim + 2L * model.Plan.KvHeadCount * model.Plan.HeadDim;

    private static int TinyVocabSize() => 6;

    // One token forwarded through the tiny phi3-shaped file at a fixed path.
    private float[] ForwardTinyPhi3(CpuBackend backend, string path)
    {
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        Assert.Equal(QkvLayout.FusedBufferSlices, model.Plan.AttentionWeights);
        Assert.Equal(FeedForwardGateSource.UpTensorHalves, model.Plan.GateSource);

        using BlockPool pool = new(2, model.Plan.LayerCount, model.Plan.KvHeadCount, model.Plan.HeadDim, 4, backend);
        using SequenceKv kv = new(pool);
        model.Prefill([1, 2, 3], kv);

        return model.DecodeNext(4, kv).ToArray();
    }

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-phi3-plan-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // phi3 metadata: the real file's geometry (32 heads of a derived 96-wide dimension over a 3072 hidden size, one fused
    // qkv buffer and one gate-free up tensor per layer, the declared rotary slice and no frequency) for plan assertions,
    // and a tiny shape for the forward cases — hidden 8, two heads of 4 with two kv-heads so every region is 8 rows wide,
    // a 16-wide intermediate shipped as one 32-row gate/up tensor, and two layers.
    private static List<(string Key, object Value)> Metadata(string prefix, bool tiny = false)
    {
        if (tiny)
        {
            return
            [
                ("general.architecture", "phi3"),
                ("general.name", "Tiny Phi3"),
                ("phi3.embedding_length", TinyHidden),
                ("phi3.block_count", TinyLayers),
                ("phi3.attention.head_count", TinyHeads),
                ("phi3.attention.head_count_kv", TinyKvHeads),
                ("phi3.feed_forward_length", TinyIntermediate),
                ("phi3.rope.dimension_count", TinyHeadDim),
                ("phi3.attention.layer_norm_rms_epsilon", 1e-5f),
            ];
        }

        // A synthetic plan fixture declares its vocabulary because plan resolution needs either that key or a tokenizer;
        // the real-file case proves this family's file takes the fallback from `tokenizer.ggml.tokens` instead.
        return
        [
            ("general.architecture", "phi3"),
            ("phi3.embedding_length", 3072),
            ("phi3.block_count", 32),
            ("phi3.attention.head_count", 32),
            ("phi3.attention.head_count_kv", 32),
            ("phi3.feed_forward_length", 8192),
            ("phi3.rope.dimension_count", 96),
            ("phi3.attention.layer_norm_rms_epsilon", 1e-5f),
            ("phi3.vocab_size", 32_064),
        ];
    }

    private sealed record TensorDef(string Name, int[] Dims); // dims in logical (row-major) order; vectors are one-dimensional

    // A complete tiny phi3 model under the fused layout: one attn_qkv tensor per layer carrying the query region then both
    // kv regions, and one ffn_up tensor per layer carrying the gate half then the up half. Every switch below produces a
    // file with identical names for everything it reads — only the tensors this profile must not consume differ, or the
    // payload inside a fused buffer is reordered.
    private static List<TensorDef> TinyTensors(
        bool splitQkv = false,
        bool qkvRowsShort = false,
        bool denseGate = false,
        bool ropeFactors = false,
        bool tiedHead = false)
    {
        int fusedRows = TinyHeads * TinyHeadDim + 2 * TinyKvHeads * TinyHeadDim;

        List<TensorDef> tensors = [new("token_embd.weight", [TinyVocabSize(), TinyHidden])];

        for (int layer = 0; layer < TinyLayers; layer++)
        {
            if (!splitQkv)
            {
                tensors.Add(new($"blk.{layer}.attn_qkv.weight", [qkvRowsShort ? fusedRows - TinyKvHeads * TinyHeadDim : fusedRows, TinyHidden]));
            }

            if (splitQkv)
            {
                tensors.Add(new($"blk.{layer}.attn_q.weight", [TinyHeads * TinyHeadDim, TinyHidden]));
                tensors.Add(new($"blk.{layer}.attn_k.weight", [TinyKvHeads * TinyHeadDim, TinyHidden]));
                tensors.Add(new($"blk.{layer}.attn_v.weight", [TinyKvHeads * TinyHeadDim, TinyHidden]));
            }

            tensors.Add(new($"blk.{layer}.attn_output.weight", [TinyHidden, TinyHeads * TinyHeadDim]));
            tensors.Add(new($"blk.{layer}.ffn_up.weight", [2 * TinyIntermediate, TinyHidden]));
            tensors.Add(new($"blk.{layer}.ffn_down.weight", [TinyHidden, TinyIntermediate]));
            tensors.Add(new($"blk.{layer}.attn_norm.weight", [TinyHidden]));
            tensors.Add(new($"blk.{layer}.ffn_norm.weight", [TinyHidden]));

            if (denseGate)
            {
                tensors.Add(new($"blk.{layer}.ffn_gate.weight", [TinyIntermediate, TinyHidden]));
            }

            if (ropeFactors)
            {
                tensors.Add(new($"blk.{layer}.rope_factors_long.weight", [TinyHeadDim / 2]));
                tensors.Add(new($"blk.{layer}.rope_factors_short.weight", [TinyHeadDim / 2]));
            }
        }

        tensors.Add(new("output_norm.weight", [TinyHidden]));

        if (!tiedHead)
        {
            tensors.Add(new("output.weight", [TinyVocabSize(), TinyHidden]));
        }

        return tensors;
    }

    /// <summary>Builds a GGUF v3 image of F32 tensors from KV pairs and the tiny phi3 geometry. The permutation switches
    /// reorder rows inside one tensor — key/value regions exchanged, query/key regions exchanged, or the whole buffer
    /// rotated by a row for every layer's fused qkv tensor — while leaving names, shapes, formats and byte counts alone;
    /// the gate/up switch exchanges that tensor's two halves instead.</summary>
    private static byte[] BuildGguf(
        IReadOnlyList<(string Key, object Value)> kvPairs,
        IReadOnlyList<TensorDef> tensors,
        bool qkvSwapKeyValue = false,
        bool qkvSwapQueryKey = false,
        bool qkvRotateOneRow = false,
        bool swapGateUpHalves = false)
    {
        List<(string Key, object Value)> pairs = [.. kvPairs];

        if (tensors.Count > 0)
        {
            // The tiny forward fixtures carry their vocabulary as a tokenizer vocabulary rather than a declared size, which
            // is the fallback the real file uses too.
            pairs.RemoveAll(pair => pair.Key.StartsWith("tokenizer.", StringComparison.Ordinal));
            WriteTokenVocabulary(pairs);
        }

        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(GgufMagic);
        writer.Write(3u); // version
        writer.Write((ulong)tensors.Count);
        writer.Write((ulong)pairs.Count);

        foreach ((string key, object value) in pairs)
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

        // Payload values are keyed by tensor name, not table position: adding a tensor this layout ignores must leave the
        // bytes of every tensor it reads untouched, so those cases can compare forwards across files whose tables differ.
        foreach (TensorDef tensor in tensors)
        {
            long rows = tensor.Dims[0];
            long cols = tensor.Dims.Length > 1 ? tensor.Dims[1] : 1;
            bool isNorm = tensor.Name.Contains("norm");

            // Keep the per-name seed small enough that row offsets remain visible at float precision. A full hash would be
            // large enough for MathF.Sin to round nearby rows onto the same value, hiding exactly these reorderings.
            int tensorSeed = TensorSeed(tensor.Name) % 997;

            for (long e = 0; e < rows * cols; e++)
            {
                long row = cols == 1 ? 0 : e / cols;
                float value = MathF.Sin((tensorSeed * 17 + ReorderedRow(tensor, row, qkvSwapKeyValue, qkvSwapQueryKey, qkvRotateOneRow, swapGateUpHalves)) * 0.13f);
                writer.Write(isNorm ? 1.25f + value : value); // norm weights stay clearly nonzero
            }
        }

        writer.Flush();
        return stream.ToArray();
    }

    // Which source row a payload position copies: the identity for every tensor this does not touch, and one of three
    // region reorderings inside a fused qkv buffer or its gate/up counterpart. Every variant stays a permutation of the
    // same rows, so shapes, formats and byte counts are identical across variants by construction.
    private static long ReorderedRow(
        TensorDef tensor,
        long row,
        bool swapKeyValue,
        bool swapQueryKey,
        bool rotateOneRow,
        bool swapGateUpHalves)
    {
        if (tensor.Name.EndsWith("attn_qkv.weight", StringComparison.Ordinal))
        {
            int qRegion = TinyHeads * TinyHeadDim;
            int kvRegion = TinyKvHeads * TinyHeadDim;

            if (swapKeyValue && row >= qRegion)
            {
                long within = row - qRegion;
                return within < kvRegion ? row + kvRegion : row - kvRegion;
            }

            if (swapQueryKey && row < 2 * qRegion)
            {
                // The query region and the key region are the same width here, so exchanging them is a straight swap of
                // the first two blocks.
                return row < qRegion ? row + qRegion : row - qRegion;
            }

            if (rotateOneRow)
            {
                long total = tensor.Dims[0];
                return (row + 1) % total;
            }

            return row;
        }

        if (tensor.Name.EndsWith("ffn_up.weight", StringComparison.Ordinal) && swapGateUpHalves)
        {
            long half = TinyIntermediate;
            return row < half ? row + half : row - half;
        }

        return row;
    }

    // A stable per-name seed: every tensor keeps its payload across files whose table order or extra unread tensors differ.
    private static int TensorSeed(string name)
    {
        uint hash = 2166136261u;
        foreach (byte b in Encoding.UTF8.GetBytes(name))
        {
            hash ^= b;
            hash *= 16777619u;
        }

        return (int)(hash & 0x7fff_ffff);
    }

    private static void WriteTokenVocabulary(List<(string Key, object Value)> pairs)
    {
        pairs.Add(("tokenizer.ggml.model", "llama"));
        pairs.Add(("tokenizer.ggml.tokens", new string[] { "<pad>", "<bos>", "a", "b", "c", "<eos>" }));
        pairs.Add(("tokenizer.ggml.add_bos_token", false));
        pairs.Add(("tokenizer.ggml.bos_token_id", 1));
        pairs.Add(("tokenizer.ggml.eos_token_id", 5));
    }

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }
}
