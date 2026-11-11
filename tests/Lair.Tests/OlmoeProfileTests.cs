using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Loader-side coverage of the olmoe profile, against the real pinned Q2_K fixture where a file is needed
/// (skip-if-absent) and against the registry alone where it is not. The registered profile carries the sparse shape
/// its reference block runs — router plus stacked gate/up/down expert triplets and no shared expert at all — with the
/// whole-stream weighted QK-norm as its feature default, because that is the form the reference applies to q and k
/// before splitting them into heads. The fixture's own tensor table is asserted exactly: 195 tensors in the quantized
/// family it ships, each layer carrying hidden-sized (not head-dimension) F32 norm vectors under the names the sparse
/// map already knows, one stacked triplet per role with the declared expert count as its outermost dimension, and an
/// untied Q6_K output head. The plan resolves those declarations into a 64-expert block routing 8 experts per token
/// with no shared expert, no routing-weight gain and no renormalization — the keys are simply absent from this file —
/// and the forward engine opens it, which is what proves the stream-sized norm vectors load under that mode.
/// </summary>
[Trait("Category", "Loader")]
[Collection("model-loaded")]
public sealed class OlmoeProfileTests
{
    private const string FixtureRelativePath = "models/OLMoE-1B-7B-0924-Instruct-Q2_K.gguf";

    // ---------------------------------------------------------------- the registered profile itself

    [Fact]
    public void ModelProfiles_Olmoe_ResolvesASharedExpertFreeSparseShapeWithStreamWeightedQkNorm()
    {
        Assert.Contains("olmoe", ModelProfiles.SupportedArchitectures);

        ModelProfile profile = ModelProfiles.Resolve("olmoe");

        Assert.Equal("olmoe.", profile.Prefix);
        Assert.Equal(NormStrategy.RmsNorm, profile.Norm);
        Assert.Equal(ActivationStrategy.SwiGLU, profile.Activation);
        Assert.Equal(BiasPolicy.Declared, profile.Bias);
        // The file declares no rope pairing and the reference asserts a full-head rotation: adjacent pairs.
        Assert.Equal(RopeStrategy.StandardFullHead, profile.Rope);
        Assert.Equal(NormPlacement.PreOnly, profile.NormPlacement);

        // QK-norm defaults to the whole-stream weighted form this family's graph runs; nothing in the file can say so.
        Assert.Equal(QkNormMode.StreamWeighted, profile.DefaultQkNorm);

        SparseFfnDescriptor sparse = profile.SparseFfn!;
        Assert.False(sparse.HasSharedExpert);
        Assert.Null(sparse.SharedGateRole);
        Assert.Null(sparse.SharedUpRole);
        Assert.Null(sparse.SharedDownRole);
        Assert.Null(sparse.SharedInputGateRole);

        // The shape names the router and the stacked expert triplets under this family's own tensor names, carries no
        // counts of its own (the file declares them), and states neither a routing-weight gain nor renormalization.
        Assert.Equal(TensorRole.MoeRouter, sparse.RouterRole);
        Assert.Equal(TensorRole.MoeExpertGate, sparse.ExpertGateRole);
        Assert.Equal(TensorRole.MoeExpertUp, sparse.ExpertUpRole);
        Assert.Equal(TensorRole.MoeExpertDown, sparse.ExpertDownRole);
        Assert.Equal(SparseFfnLayout.StackedExperts, sparse.Layout);
        Assert.Null(sparse.Rank);
        Assert.Equal(0, sparse.ExpertCount);
        Assert.Equal(0, sparse.RoutedExpertsPerToken);
        Assert.Equal(0f, sparse.ExpertWeightsScale);
        Assert.False(sparse.NormalizeExpertWeights);

        // The map the profile carries is the MoE one: it names every sparse role, and its dense gate/up/down roles stay
        // optional because an expert-routed block ships none of them.
        TensorMap tensors = profile.Tensors;
        Assert.False(tensors[TensorRole.FeedForwardGate].Required);
        Assert.False(tensors[TensorRole.FeedForwardUp].Required);
        Assert.False(tensors[TensorRole.FeedForwardDown].Required);
        Assert.Equal("blk.{layer}.ffn_gate_inp.weight", tensors[TensorRole.MoeRouter].Name);
        Assert.Equal("blk.{layer}.ffn_gate_exps.weight", tensors[TensorRole.MoeExpertGate].Name);
        Assert.Equal("blk.{layer}.ffn_up_exps.weight", tensors[TensorRole.MoeExpertUp].Name);
        Assert.Equal("blk.{layer}.ffn_down_exps.weight", tensors[TensorRole.MoeExpertDown].Name);
    }

    // ---------------------------------------------------------------- the fixture's tensor table

    [Fact]
    public void GgufModelLoader_RealOlmoeFixture_ShipsStackedExpertsAndHiddenSizedQkNormVectors()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(FixtureRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        LogicalTensorTable table = new GgufModelLoader().Load(modelPath);

        // Exact inventory of the file as shipped: every layer carries the same formats, the expert triplets are stacked
        // with the declared expert count outermost, and each side of attention keeps its own hidden-sized F32 norm vector.
        Dictionary<string, (QuantFormat Format, int Count)> expected = new(StringComparer.Ordinal)
        {
            ["attn_q.weight"] = (QuantFormat.Q2_K, 16),
            ["attn_k.weight"] = (QuantFormat.Q2_K, 16),
            ["attn_v.weight"] = (QuantFormat.Q3_K, 16),
            ["attn_output.weight"] = (QuantFormat.Q3_K, 16),
            ["attn_norm.weight"] = (QuantFormat.F32, 16),
            ["ffn_norm.weight"] = (QuantFormat.F32, 16),
            ["attn_q_norm.weight"] = (QuantFormat.F32, 16),
            ["attn_k_norm.weight"] = (QuantFormat.F32, 16),
            ["ffn_gate_inp.weight"] = (QuantFormat.F32, 16),
            ["ffn_gate_exps.weight"] = (QuantFormat.Q2_K, 16),
            ["ffn_up_exps.weight"] = (QuantFormat.Q2_K, 16),
            ["ffn_down_exps.weight"] = (QuantFormat.Q3_K, 16),
            ["token_embd.weight"] = (QuantFormat.Q2_K, 1),
            ["output.weight"] = (QuantFormat.Q6_K, 1),
            ["output_norm.weight"] = (QuantFormat.F32, 1),
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

        Assert.Equal(195, table.Tensors.Count);
        Assert.Equal(expected.Keys.OrderBy(k => k, StringComparer.Ordinal), observedCounts.Keys.OrderBy(k => k, StringComparer.Ordinal));
        foreach ((string suffix, (QuantFormat format, int count)) in expected)
        {
            Assert.True(observedCounts.TryGetValue(suffix, out int observed), $"no tensors found under '{suffix}'");
            Assert.True(
                observed == count,
                $"'{suffix}': expected {count} tensors of format {format}, got {observed}");
            Assert.Equal(format, observedFormats[suffix]);
        }

        // Shapes: the router is [experts, hidden], each stacked triplet covers its side of the block per expert with
        // experts outermost (down reversed), and each norm vector spans the whole projection rather than one head.
        Assert.Equal(new int[] { 64, 2048 }, table.Tensors.First(t => t.Name == "blk.0.ffn_gate_inp.weight").Shape.Dims);
        Assert.Equal(new int[] { 64, 1024, 2048 }, table.Tensors.First(t => t.Name == "blk.0.ffn_gate_exps.weight").Shape.Dims);
        Assert.Equal(new int[] { 64, 1024, 2048 }, table.Tensors.First(t => t.Name == "blk.0.ffn_up_exps.weight").Shape.Dims);
        Assert.Equal(new int[] { 64, 2048, 1024 }, table.Tensors.First(t => t.Name == "blk.0.ffn_down_exps.weight").Shape.Dims);
        Assert.Equal(new int[] { 2048 }, table.Tensors.First(t => t.Name == "blk.0.attn_q_norm.weight").Shape.Dims);
        Assert.Equal(new int[] { 2048 }, table.Tensors.First(t => t.Name == "blk.0.attn_k_norm.weight").Shape.Dims);

        // No shared-expert tensor of any name is present: this family ships none, and the map's optional slots stay empty.
        Assert.DoesNotContain(table.Tensors, t => t.Name.Contains("shexp", StringComparison.Ordinal));
    }

    // ---------------------------------------------------------------- plan resolution from that file

    [Fact]
    public void ModelPlan_RealOlmoeFixture_ResolvesRegisteredProfileSparseCountsAndStreamWeightedQkNorm()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(FixtureRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile file = GgufFile.Open(modelPath);
        ModelPlan plan = ModelPlans.Resolve(file);

        Assert.Equal("olmoe", plan.Architecture);
        Assert.Equal("olmoe", plan.Profile); // registered: never the generic fallback identity
        Assert.Equal("olmoe.", plan.Prefix);

        Assert.Equal(16, plan.LayerCount);
        Assert.Equal(2048, plan.HiddenSize);
        Assert.Equal(16, plan.HeadCount);
        Assert.Equal(16, plan.KvHeadCount);
        // The file declares no head_length/key_length/value_length: the dimension is hidden size over head count.
        Assert.Equal(128, plan.HeadDim);
        Assert.Equal(1024, plan.IntermediateSize);
        Assert.Equal(10_000f, plan.RopeTheta);
        Assert.True(Math.Abs(plan.EpsRmsNorm - 9.999999747378752E-06f) < 1e-12f);

        // No declared vocab_size: the size is the length of tokenizer.ggml.tokens, matching token_embd's row count.
        Assert.Equal(50304, plan.VocabSize);
        Assert.True(plan.HasOutputWeights); // an untied Q6_K output head is present

        Assert.False(plan.UseAttentionBias);
        Assert.False(plan.UseNormBias);
        Assert.Equal(QkNormMode.StreamWeighted, plan.QkNorm); // the profile default stands: this file declares nothing
        Assert.Equal(RopeStrategy.StandardFullHead, plan.Rope);
        Assert.Equal(0f, plan.AttentionSoftcap);
        Assert.Equal(0f, plan.FinalLogitSoftcap);
        Assert.Equal(0, plan.AttentionWindow);

        SparseFfnDescriptor sparse = plan.SparseFfn!;
        Assert.Equal(64, sparse.ExpertCount); // declared olmoe.expert_count
        Assert.Equal(8, sparse.RoutedExpertsPerToken); // declared olmoe.expert_used_count
        Assert.False(sparse.HasSharedExpert); // no expert_shared_count in the file
        // The resolved shape is the profile's, with only its counts filled in from this file: stacked storage (no
        // expert_layout declared), no rank, and neither a routing-weight gain nor renormalization.
        SparseFfnDescriptor profileShape = ModelProfiles.Resolve("olmoe").SparseFfn!;
        Assert.Equal(profileShape.RouterRole, sparse.RouterRole);
        Assert.Equal(profileShape.ExpertGateRole, sparse.ExpertGateRole);
        Assert.Equal(profileShape.ExpertUpRole, sparse.ExpertUpRole);
        Assert.Equal(profileShape.ExpertDownRole, sparse.ExpertDownRole);
        Assert.Null(sparse.SharedInputGateRole);
        Assert.Equal(SparseFfnLayout.StackedExperts, sparse.Layout);
        Assert.Null(sparse.Rank);
        Assert.Equal(0f, sparse.ExpertWeightsScale);
        Assert.False(sparse.NormalizeExpertWeights);
        Assert.Equal(SparseFfnStrategy.DenseResident, plan.SparseFfnStrategy!.Value);
    }

    // ---------------------------------------------------------------- the engine loads what that mode needs

    [Fact]
    public void TransformerModel_RealOlmoeFixture_LoadsHiddenSizedQkNormVectorsAndReportsNothingUnused()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(FixtureRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile file = GgufFile.Open(modelPath);
        using CpuBackend backend = new();

        List<string> logLines = [];
        using TransformerModel model = TransformerModel.Open(file, backend, maxContextTokens: 8, logLines.Add);

        Assert.Equal(QkNormMode.StreamWeighted, model.Plan.QkNorm);
        Assert.False(model.Plan.SparseFfn!.HasSharedExpert);

        // Nothing was ignored: the architecture is registered, every norm vector the mode demands is present at the
        // width that mode names (head_count x head_dim for q and kv_head_count x head_dim for k), and no bias or
        // post-sublayer tensor exists in the file to report.
        // The only report is the vocabulary fallback this file legitimately needs: it declares no olmoe.vocab_size, so
        // the size comes from tokenizer.ggml.tokens. Nothing was ignored — every norm vector the mode demands is
        // present at the width that mode names (head_count x head_dim for q and kv_head_count x head_dim for k), and
        // the file carries no bias or post-sublayer tensor to report.
        Assert.Single(logLines);
        Assert.StartsWith("vocab_fallback ", logLines[0]);
        Assert.Contains("source=tokenizer.ggml.tokens count=50304", logLines[0]);
    }

    // ---------------------------------------------------------------- the routed experts actually execute

    [Fact]
    public void TransformerModel_RealOlmoeFixture_RoutesAndExecutesExpertStacksThatQuantAtDifferentTiers()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(FixtureRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile file = GgufFile.Open(modelPath);
        using CpuBackend backend = new();
        using TransformerModel model = TransformerModel.Open(file, backend, maxContextTokens: 16);

        LogicalTensor gate = file.Tensors.Tensors.First(t => t.Name == "blk.0.ffn_gate_exps.weight");
        LogicalTensor down = file.Tensors.Tensors.First(t => t.Name == "blk.0.ffn_down_exps.weight");
        Assert.Equal(QuantFormat.Q2_K, gate.Format);
        Assert.Equal(QuantFormat.Q3_K, down.Format);

        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(file);
        IReadOnlyList<int> ids = tokenizer.Encode("The capital of France is");

        using BlockPool pool = new(16, model.Desc.LayerCount, model.Desc.KvHeadCount, model.Desc.HeadDim, 2, backend);
        using SequenceKv kv = new(pool);
        model.Prefill(ids, kv);

        // A prefill and one decode step both walk the routed expert chain. The gate/up stack quantizes at Q2_K and the
        // down stack at Q3_K in this file, so every selected expert's slice must be sized from its own role's format:
        // a shared per-role stride taken from the wrong tensor fails inside these GEMVs rather than here.
        ReadOnlySpan<float> promptLogits = model.LastLogits();
        Assert.Equal(model.Plan.VocabSize, promptLogits.Length);

        float[] decodeLogits = [.. model.DecodeNext(ids[^1], kv)];
        Assert.Equal(model.Plan.VocabSize, decodeLogits.Length);
        foreach (float logit in decodeLogits)
        {
            Assert.True(float.IsFinite(logit), $"decode produced a non-finite logit {logit}");
        }

        double max = 0.0;
        for (int i = 1; i < decodeLogits.Length; i++)
        {
            if (Math.Abs(decodeLogits[i] - decodeLogits[0]) > max)
            {
                max = Math.Abs(decodeLogits[i] - decodeLogits[0]);
            }
        }

        Assert.True(max > 0.0, "every logit in the decode step is identical — no expert contribution reached the output");
    }
}
