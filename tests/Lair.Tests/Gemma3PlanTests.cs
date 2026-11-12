using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Load-time resolution for the gemma3 profile: the strategy values the registered profile carries — RMS block norms,
/// the exact-GELU gated feed-forward, rotate-half rotary rows, post-sublayer norm placement, the input-embedding gain,
/// and the per-head QK-norm defaulting on because this family's conversions ship the per-head vectors — the geometry
/// the real file declares (a 256-wide head from <c>key_length</c> / <c>value_length</c> with no <c>head_length</c>
/// anywhere, four heads of it against a hidden size they do not divide), the declared window and the stride the
/// profile applies it on, and the two rotary bases the plan resolves for the two kinds of layer: the declared
/// <c>rope.freq_base</c> for the layers the stride leaves unbounded and the declared <c>rope.freq_base_swa</c> — or the
/// base an undeclared frequency takes — for the layers it narrows. A declared second base that is not a positive
/// finite number fails naming the key and the value, and the forward engine consults it only where a layer really
/// carries a window: the same file with no window declared forwards identically whatever that key declares. Against the
/// pinned fixture the case asserts the file's own keys, its exact tensor inventory, and the per-layer pattern the
/// profile resolves from its single window declaration. Cases skip when the fixture file is absent.
/// </summary>
[Trait("Category", "Loader")]
[Collection("model-loaded")]
public sealed class Gemma3PlanTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

    private const string FixtureRelativePath = "models/gemma-3-1b-it-Q3_K_S.gguf";
    private const string FixtureSha256 = "0ffb15257ccee33bee17e732fb3db55a332801d187e87164488888022fe1634c";

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
    public void ModelProfiles_Gemma3_ResolvesReferenceGraphDefaultsWithPerHeadQkNormOn()
    {
        Assert.Contains("gemma3", ModelProfiles.SupportedArchitectures);

        ModelProfile profile = ModelProfiles.Resolve("gemma3");

        Assert.Equal("gemma3.", profile.Prefix);
        Assert.Equal(NormStrategy.RmsNorm, profile.Norm);
        Assert.Equal(ActivationStrategy.GeGluExact, profile.Activation);
        Assert.Equal(BiasPolicy.Declared, profile.Bias);
        Assert.Equal(RopeStrategy.RotateHalfFullHead, profile.Rope);
        Assert.Equal(NormPlacement.PostSublayerOutputs, profile.NormPlacement);

        // This family's conversion ships per-head q/k norm vectors for every layer, so the weighted mode is the default
        // rather than something the file has to declare; attention bias stays declared-only like every other profile.
        Assert.Equal(QkNormMode.PerHeadWeighted, profile.DefaultQkNorm);
        Assert.False(profile.DefaultAttentionBias);

        // The window stride this family's reference declares for itself: full attention on the last layer of each group
        // of six, the declared window on the rest.
        Assert.Equal(WindowInterleave.SixLayerGroupsLocalFirst, profile.WindowInterleave);

        // The conversion leaves the embedding gain out of the weights, so the profile carries it.
        Assert.True(profile.ScaleInputEmbeddingsBySqrtHidden);

        // The canonical map already names every tensor this family ships, including the per-head norm vectors and the
        // post-sublayer pair, all of them present in the pinned file.
        TensorMap tensors = profile.Tensors;
        Assert.Equal("blk.{layer}.attn_q_norm.weight", tensors[TensorRole.AttentionQueryNorm].Name);
        Assert.Equal("blk.{layer}.attn_k_norm.weight", tensors[TensorRole.AttentionKeyNorm].Name);
        Assert.Equal("blk.{layer}.post_attention_norm.weight", tensors[TensorRole.PostAttentionNorm].Name);
        Assert.Equal("blk.{layer}.post_ffw_norm.weight", tensors[TensorRole.PostFeedForwardNorm].Name);
    }

    // ---------------------------------------------------------------- plan resolution on the file's own geometry

    [Fact]
    public void ModelPlan_Gemma3DeclaredGeometry_ResolvesUnequalHeadRegionWindowAndStride()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata("gemma3"), [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal("gemma3", plan.Architecture);
        Assert.Equal("gemma3.", plan.Prefix);
        Assert.Equal("gemma3", plan.Profile);
        Assert.Equal(26, plan.LayerCount);
        Assert.Equal(1152, plan.HiddenSize);
        Assert.Equal(4, plan.HeadCount);
        Assert.Equal(1, plan.KvHeadCount);
        Assert.Equal(6912, plan.IntermediateSize);
        Assert.Equal(262_144, plan.VocabSize);

        // The head dimension comes from the agreeing key_length / value_length pair: four heads of 256 is 1024, which is
        // not a split of the 1152 hidden size this file runs.
        Assert.Equal(256, plan.HeadDim);
        Assert.True(
            plan.HeadCount * plan.HeadDim != plan.HiddenSize,
            $"gemma3's q region ({plan.HeadCount * plan.HeadDim}) must stay distinct from its hidden size ({plan.HiddenSize})");

        Assert.Equal(NormStrategy.RmsNorm, plan.Norm);
        Assert.Equal(ActivationStrategy.GeGluExact, plan.Activation);
        Assert.Equal(RopeStrategy.RotateHalfFullHead, plan.Rope);
        Assert.Equal(NormPlacement.PostSublayerOutputs, plan.NormPlacement);
        Assert.Equal(QkNormMode.PerHeadWeighted, plan.QkNorm);
        Assert.False(plan.UseAttentionBias);
        Assert.False(plan.UseNormBias);
        Assert.Equal(0f, plan.EpsLayerNorm); // this family declares only the RMS epsilon
        Assert.True(Math.Abs(plan.EpsRmsNorm - 1e-6f) < 1e-8f, $"rms epsilon {plan.EpsRmsNorm:G9}");
        Assert.True(
            Math.Abs(plan.InputEmbeddingScale - MathF.Sqrt(1152f)) <= 1e-4f,
            $"input embedding scale {plan.InputEmbeddingScale:G9} must be sqrt(1152)");

        // No softcap key is declared here, so both caps resolve disabled whatever the profile says about them.
        Assert.Equal(0f, plan.AttentionSoftcap);
        Assert.Equal(0f, plan.FinalLogitSoftcap);

        Assert.Equal(512, plan.AttentionWindow);
        Assert.Equal(WindowInterleave.SixLayerGroupsLocalFirst, plan.WindowInterleave);
    }

    [Fact]
    public void ModelPlan_Gemma3WindowStride_ResolvesWholeRangeEverySixthLayer_AndTwoRotationBases()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata("gemma3"), [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        List<int> unwindowed = [];

        for (int layer = 0; layer < plan.LayerCount; layer++)
        {
            int window = plan.AttentionWindowForLayer(layer);

            if (layer % 6 == 5)
            {
                Assert.True(window == 0, $"layer {layer} is this family's full-attention layer and must stay unbounded, resolved {window}");
                unwindowed.Add(layer);
            }
            else
            {
                Assert.True(window == 512, $"layer {layer} must read inside the declared 512-position window, resolved {window}");
            }

            // The rotation base follows the same per-layer resolution: the windowed layers rotate at the base a file
            // declaring no windowed frequency takes, and the unbounded ones at the declared global frequency.
            float baseForLayer = plan.RopeThetaForLayer(layer);

            if (window == 0)
            {
                Assert.True(
                    Math.Abs(baseForLayer - plan.RopeTheta) < 1f,
                    $"layer {layer} attends the whole range and must rotate at the declared {plan.RopeTheta:G9}, resolved {baseForLayer:G9}");
            }
            else
            {
                Assert.True(
                    Math.Abs(baseForLayer - ModelPlan.DefaultRopeTheta) < 1f,
                    $"layer {layer} carries the window and must rotate at {ModelPlan.DefaultRopeTheta:G9}, resolved {baseForLayer:G9}");
            }
        }

        Assert.Equal(new int[] { 5, 11, 17, 23 }, unwindowed.ToArray());

        // One window key and one global frequency produced every value above: this file declares no per-layer
        // declaration and no windowed frequency, which is what the profile's stride and the plan's fallback resolve from.
        Assert.True(Math.Abs(plan.RopeTheta - 1_000_000f) < 1f, $"declared rope theta {plan.RopeTheta:G9}");
        Assert.Equal(ModelPlan.DefaultRopeTheta, plan.RopeWindowedTheta);
    }

    [Fact]
    public void ModelPlan_Gemma3DeclaredWindowedRopeBase_ReachesOnlyWindowedLayers()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(
            BuildGguf([.. Metadata("gemma3"), ("gemma3.rope.freq_base_swa", 5_000f)], [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.True(Math.Abs(plan.RopeWindowedTheta - 5_000f) < 1f, $"declared windowed base {plan.RopeWindowedTheta:G9}");

        Assert.True(Math.Abs(plan.RopeThetaForLayer(0) - 5_000f) < 1f, "layer 0 carries the window");
        Assert.True(Math.Abs(plan.RopeThetaForLayer(4) - 5_000f) < 1f, "layer 4 carries the window");
        Assert.True(Math.Abs(plan.RopeThetaForLayer(5) - 1_000_000f) < 1f, "layer 5 is this family's full-attention layer");
        Assert.True(Math.Abs(plan.RopeThetaForLayer(23) - 1_000_000f) < 1f, "layer 23 is this family's full-attention layer");
    }

    [Fact]
    public void ModelPlan_Gemma3WithoutAWindow_NeverResolvesTheWindowedRotationBase()
    {
        // The declared second base reaches nothing when the plan narrows no layer: every layer attends the whole range
        // and rotates at the declared global frequency whatever the windowed key declares.
        List<(string Key, object Value)> pairs = [.. Metadata("gemma3")];
        pairs.RemoveAll(pair => pair.Key == "gemma3.attention.sliding_window");
        pairs.Add(("gemma3.rope.freq_base_swa", 5_000f));

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(pairs, [])));
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal(0, plan.AttentionWindow);

        for (int layer = 0; layer < plan.LayerCount; layer++)
        {
            Assert.True(
                plan.RopeThetaForLayer(layer) == plan.RopeTheta,
                $"layer {layer} of a windowless plan must rotate at {plan.RopeTheta:G9}, resolved {plan.RopeThetaForLayer(layer):G9}");
        }
    }

    [Fact]
    public void ModelPlan_Gemma3WindowInterleaveComesFromTheProfile_NotTheGeometry()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata("gemma3"), [])));

        ModelPlan resolved = ModelPlans.Resolve(gguf);
        ModelPlan everyLayer = ModelPlans.Resolve(gguf, ModelProfiles.Resolve("gemma3") with { WindowInterleave = WindowInterleave.AllLayers });
        ModelPlan alternating = ModelPlans.Resolve(gguf, ModelProfiles.Resolve("gemma3") with { WindowInterleave = WindowInterleave.AlternatingLocalFirst });

        Assert.Equal(WindowInterleave.SixLayerGroupsLocalFirst, resolved.WindowInterleave);

        // The stride-6 pattern is what makes layer 5 read the whole range; a pattern that masks every layer does not.
        Assert.Equal(0, resolved.AttentionWindowForLayer(5));
        Assert.Equal(512, everyLayer.AttentionWindowForLayer(5));

        // Nor is it the accident of an alternating pattern: layer 1 is windowed here and unbounded there.
        Assert.Equal(512, resolved.AttentionWindowForLayer(1));
        Assert.Equal(0, alternating.AttentionWindowForLayer(1));
    }

    // ---------------------------------------------------------------- declared-value failures

    [Fact]
    public void ModelPlan_Gemma3InvalidWindowedRopeBase_FailsNamingKeyAndValue()
    {
        foreach (object declared in new object[] { 0f, -25f, float.NaN, float.PositiveInfinity })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(
                BuildGguf([.. Metadata("gemma3"), ("gemma3.rope.freq_base_swa", declared)], [])));

            ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

            Assert.Contains("gemma3.rope.freq_base_swa", ex.Message);
            Assert.Contains("positive finite", ex.Message);

            if (declared is float number && float.IsFinite(number))
            {
                Assert.Contains(number.ToString(System.Globalization.CultureInfo.InvariantCulture), ex.Message);
            }
        }
    }

    [Fact]
    public void ModelPlan_Gemma3InvalidWindowDeclaration_FailsNamingKeyAndValue()
    {
        foreach (object declared in new object[] { 0, -512, "narrow" })
        {
            List<(string Key, object Value)> pairs = [.. Metadata("gemma3")];
            pairs.RemoveAll(pair => pair.Key == "gemma3.attention.sliding_window");
            pairs.Add(("gemma3.attention.sliding_window", declared));

            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(pairs, [])));
            ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

            Assert.Contains("gemma3.attention.sliding_window", ex.Message);
            Assert.Contains("positive integers", ex.Message);

            if (declared is not string)
            {
                Assert.Contains(declared.ToString()!, ex.Message);
            }
        }
    }

    [Fact]
    public void ModelPlan_Gemma3DisagreeingHeadLengths_FailNamingBothKeysAndValues()
    {
        List<(string Key, object Value)> pairs = [.. Metadata("gemma3")];
        pairs.RemoveAll(pair => pair.Key == "gemma3.attention.value_length");
        pairs.Add(("gemma3.attention.value_length", 128));

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(pairs, [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("gemma3.attention.key_length (256)", ex.Message);
        Assert.Contains("gemma3.attention.value_length (128)", ex.Message);
    }

    // ---------------------------------------------------------------- the forward engine's side of the profile

    [Fact]
    public void TransformerModel_Gemma3ProfileDefaultQkNorm_MissingPerHeadVectors_FailsLoadNamingThem()
    {
        string path = WriteTemp(BuildGguf(Metadata("gemma3", tiny: true), TinyTensors(qkNorms: false)));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("blk.0.attn_q_norm.weight", ex.Message);
        Assert.Contains("gemma3", ex.Message);
    }

    [Fact]
    public void TransformerModel_Gemma3MissingPostSublayerNorms_FailsLoadNamingThem_AndThePlacement()
    {
        string path = WriteTemp(BuildGguf(Metadata("gemma3", tiny: true), TinyTensors(postNorms: false)));

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(path);
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("blk.0.post_attention_norm.weight", ex.Message);
        Assert.Contains("norm placement", ex.Message);
        Assert.Contains("gemma3", ex.Message);
    }

    [Fact]
    public void TransformerModel_Gemma3WindowedRopeBase_ChangesWindowedLayersOnly_AcrossLoads()
    {
        using CpuBackend backend = new();

        float[] atStandardBase = ForwardTinyGemma3(backend, windowOverride: 4, windowedBase: 10_000f);
        float[] atOtherBase = ForwardTinyGemma3(backend, windowOverride: 4, windowedBase: 5_000f);

        Assert.True(
            !atStandardBase.AsSpan().SequenceEqual(atOtherBase.AsSpan()),
            "a declared windowed rotary base must reach the layers that carry the window");

        // With no window declared, no layer consults that base: the same two declarations forward identically.
        float[] unwindowedStandard = ForwardTinyGemma3(backend, windowOverride: -1, windowedBase: 10_000f);
        float[] unwindowedOther = ForwardTinyGemma3(backend, windowOverride: -1, windowedBase: 5_000f);

        Assert.True(
            unwindowedStandard.AsSpan().SequenceEqual(unwindowedOther.AsSpan()),
            "a windowless gemma3 plan must forward identically whatever its windowed rotary base declares");
    }

    [Fact]
    public void TransformerModel_Gemma3SplitRotationBases_AreAnnouncedOnceAtOpen()
    {
        using CpuBackend backend = new();

        List<string> splitLines = [];
        {
            string path = WriteTemp(BuildGguf(Metadata("gemma3", tiny: true, windowedBase: 5_000f), TinyTensors(), windowOverride: 4));
            using GgufFile gguf = GgufFile.Open(path);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, splitLines.Add);

            Assert.Equal(WindowInterleave.SixLayerGroupsLocalFirst, model.Plan.WindowInterleave);
            Assert.Single(splitLines, line => line.StartsWith("rope_bases ", StringComparison.Ordinal));

            string line = splitLines.First(l => l.StartsWith("rope_bases ", StringComparison.Ordinal));
            Assert.Contains("architecture='gemma3'", line);
            Assert.Contains("global_base=1000000", line);
            Assert.Contains("windowed_base=5000", line);
            Assert.Contains("windowed_layers=6", line); // seven tiny layers, the sixth of them full-attention
            Assert.Contains("unwindowed_layers=1", line);
        }

        List<string> unwindowedLines = [];
        {
            string path = WriteTemp(BuildGguf(Metadata("gemma3", tiny: true, windowedBase: 5_000f), TinyTensors(), windowOverride: -1));
            using GgufFile gguf = GgufFile.Open(path);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, unwindowedLines.Add);

            Assert.Equal(0, model.Plan.AttentionWindow);
            Assert.False(
                unwindowedLines.Any(line => line.StartsWith("rope_bases ", StringComparison.Ordinal)),
                "a plan whose every layer is unbounded must not announce a rotation-base split it never consults");
        }
    }

    // ---------------------------------------------------------------- the pinned fixture itself

    [Fact]
    public void ModelPlan_RealGemma3Fixture_ResolvesDeclaredWindowStrideAndRotationBase()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(FixtureRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        AssertFixtureHash(modelPath);

        using GgufFile gguf = GgufFile.Open(modelPath);
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal("gemma3", plan.Architecture);
        Assert.Equal("gemma3", plan.Profile);
        Assert.Equal(26, plan.LayerCount);
        Assert.Equal(1152, plan.HiddenSize);
        Assert.Equal(4, plan.HeadCount);
        Assert.Equal(1, plan.KvHeadCount);
        Assert.Equal(256, plan.HeadDim);
        Assert.Equal(6912, plan.IntermediateSize);

        Assert.Equal(512, plan.AttentionWindow); // what the file declares at gemma3.attention.sliding_window
        Assert.Equal(WindowInterleave.SixLayerGroupsLocalFirst, plan.WindowInterleave);

        for (int layer = 0; layer < plan.LayerCount; layer++)
        {
            int window = plan.AttentionWindowForLayer(layer);
            bool fullAttentionLayer = layer % 6 == 5;

            Assert.True(
                window == (fullAttentionLayer ? 0 : 512),
                $"layer {layer} must resolve {(fullAttentionLayer ? "the whole range" : "the declared 512-position window")}, resolved {window}");
        }

        // The file declares one window key, the stride-6 pattern behind it, and a single global rotation frequency:
        // nothing per-layer, and no windowed frequency for the plan to read.
        Assert.Single(gguf.Metadata.Keys, key => key.Contains("sliding_window", StringComparison.Ordinal));
        Assert.DoesNotContain("gemma3.rope.freq_base_swa", gguf.Metadata.Keys);
        Assert.True(Math.Abs(plan.RopeTheta - 1_000_000f) < 1f, $"declared rope theta {plan.RopeTheta:G9}");
        Assert.Equal(ModelPlan.DefaultRopeTheta, plan.RopeWindowedTheta);

        // No softcap key on either side: this file runs uncapped.
        Assert.Equal(0f, plan.AttentionSoftcap);
        Assert.Equal(0f, plan.FinalLogitSoftcap);

        // The head dimension is declared by the agreeing key_length / value_length pair, and no head_length key exists
        // for it to prefer; the vocabulary size has no declared key either and falls back to the tokenizer's length.
        Assert.DoesNotContain("gemma3.attention.head_length", gguf.Metadata.Keys);
        Assert.Contains("gemma3.attention.key_length", gguf.Metadata.Keys);
        Assert.DoesNotContain("gemma3.vocab_size", gguf.Metadata.Keys);
        Assert.Equal(gguf.Metadata.GetTokens().Length, plan.VocabSize);
        Assert.Equal(262_144, plan.VocabSize);
    }

    [Fact]
    public void Gemma3Fixture_TensorInventory_IsTheCanonicalMapInThisFamilyQuantization()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(FixtureRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        AssertFixtureHash(modelPath);

        LogicalTensorTable table = new GgufModelLoader().Load(modelPath);

        // Exact inventory of the file as shipped: every one of the 26 layers carries the same formats, the projections
        // are IQ4_NL with the output projections and down-projection at Q3_K, every norm vector is F32, the token
        // embedding is Q8_0, and no untied output head exists — the output head is tied to that embedding.
        Dictionary<string, (QuantFormat Format, int Count)> expected = new(StringComparer.Ordinal)
        {
            ["attn_q.weight"] = (QuantFormat.IQ4_NL, 26),
            ["attn_k.weight"] = (QuantFormat.IQ4_NL, 26),
            ["attn_v.weight"] = (QuantFormat.IQ4_NL, 26),
            ["attn_output.weight"] = (QuantFormat.Q3_K, 26),
            ["attn_norm.weight"] = (QuantFormat.F32, 26),
            ["attn_q_norm.weight"] = (QuantFormat.F32, 26),
            ["attn_k_norm.weight"] = (QuantFormat.F32, 26),
            ["post_attention_norm.weight"] = (QuantFormat.F32, 26),
            ["post_ffw_norm.weight"] = (QuantFormat.F32, 26),
            ["ffn_gate.weight"] = (QuantFormat.IQ4_NL, 26),
            ["ffn_up.weight"] = (QuantFormat.IQ4_NL, 26),
            ["ffn_down.weight"] = (QuantFormat.Q3_K, 26),
            ["ffn_norm.weight"] = (QuantFormat.F32, 26),
            ["token_embd.weight"] = (QuantFormat.Q8_0, 1),
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

        Assert.Equal(340, table.Tensors.Count);
        Assert.Equal(expected.Keys.OrderBy(k => k, StringComparer.Ordinal), observedCounts.Keys.OrderBy(k => k, StringComparer.Ordinal));

        foreach ((string suffix, (QuantFormat format, int count)) in expected)
        {
            Assert.True(observedCounts.TryGetValue(suffix, out int observed), $"no tensors found under '{suffix}'");
            Assert.True(
                observed == count,
                $"'{suffix}': expected {count} tensors of format {format}, got {observed}");
            Assert.Equal(format, observedFormats[suffix]);
        }

        Assert.DoesNotContain(table.Tensors, t => t.Name == "output.weight");

        // Shapes in the logical order LAIR reports them: each projection is [out, in], so the q side is four heads of
        // 256 rows against a 1152-wide hidden size, and every norm vector spans one head dimension for the per-head pair
        // and the hidden size for the block norms. The tied embedding is vocab-sized rows over hidden columns.
        Assert.Equal([1024, 1152], table.Tensors.First(t => t.Name == "blk.0.attn_q.weight").Shape.Dims);
        Assert.Equal([256, 1152], table.Tensors.First(t => t.Name == "blk.0.attn_k.weight").Shape.Dims);
        Assert.Equal([256, 1152], table.Tensors.First(t => t.Name == "blk.0.attn_v.weight").Shape.Dims);
        Assert.Equal([1152, 1024], table.Tensors.First(t => t.Name == "blk.0.attn_output.weight").Shape.Dims);
        Assert.Equal([256], table.Tensors.First(t => t.Name == "blk.0.attn_q_norm.weight").Shape.Dims);
        Assert.Equal([256], table.Tensors.First(t => t.Name == "blk.0.attn_k_norm.weight").Shape.Dims);
        Assert.Equal([1152], table.Tensors.First(t => t.Name == "blk.0.post_attention_norm.weight").Shape.Dims);
        Assert.Equal([262_144, 1152], table.Tensors.First(t => t.Name == "token_embd.weight").Shape.Dims);
    }

    // ---------------------------------------------------------------- helpers

    private static void AssertFixtureHash(string modelPath)
    {
        using System.Security.Cryptography.SHA256 hasher = System.Security.Cryptography.SHA256.Create();
        using FileStream stream = File.OpenRead(modelPath);
        string actual = Convert.ToHexString(hasher.ComputeHash(stream)).ToLowerInvariant();

        Assert.True(
            string.Equals(actual, FixtureSha256, StringComparison.Ordinal),
            $"gemma3 loader guard: {FixtureRelativePath} hashes to {actual}, not the pinned {FixtureSha256}: " +
            "these assertions were written against that exact file.");
    }

    // One token forwarded through a seven-layer gemma3-shaped file, with the window and windowed rotary base as the only
    // knobs: identical weights and ids across the two variants, so any difference in the logits is the rotation.
    private float[] ForwardTinyGemma3(CpuBackend backend, int windowOverride, float windowedBase)
    {
        string path = WriteTemp(BuildGguf(Metadata("gemma3", tiny: true, windowedBase: windowedBase), TinyTensors(), windowOverride));

        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        Assert.Equal(windowOverride < 0 ? 0 : windowOverride, model.Plan.AttentionWindow); // the knob landed in the plan it was meant to configure

        using BlockPool pool = new(
            blockTokens: 2,
            layerCount: model.Plan.LayerCount,
            kvHeads: model.Plan.KvHeadCount,
            headDim: model.Plan.HeadDim,
            capacityBlocks: 4,
            backend);

        using SequenceKv kv = new(pool);
        model.Prefill([1, 2, 3], kv);

        return model.DecodeNext(4, kv).ToArray();
    }

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-gemma3-plan-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // gemma3 metadata: the real file's geometry (a 256-wide head declared by key_length / value_length with no
    // head_length anywhere, four heads against a hidden size they do not divide, one kv-head, the 512-wide window and
    // the 1e6 global frequency) for plan assertions, and a tiny shape for the forward cases — hidden 8, heads 3,
    // head dim 4 declared by the agreeing key/value lengths (3 x 4 != 8), one kv-head, seven layers so the stride puts
    // exactly one of them on the whole range. The head dimension is wide enough that a rotary base change moves more
    // than the constant first frequency pair.
    private static List<(string Key, object Value)> Metadata(
        string prefix,
        bool tiny = false,
        float windowedBase = 0f)
    {
        if (tiny)
        {
            List<(string Key, object Value)> pairs =
            [
                ("general.architecture", "gemma3"),
                ("general.name", "Tiny Gemma3"),
                ("gemma3.embedding_length", 8),
                ("gemma3.block_count", 7),
                ("gemma3.attention.head_count", 3),
                ("gemma3.attention.head_count_kv", 1),
                ("gemma3.feed_forward_length", 16),
                ("gemma3.attention.key_length", 4),
                ("gemma3.attention.value_length", 4),
                ("gemma3.attention.sliding_window", 4),
                ("gemma3.attention.layer_norm_rms_epsilon", 1e-5f),
                ("gemma3.rope.freq_base", 1_000_000f),
            ];

            if (windowedBase > 0f)
            {
                pairs.Add(("gemma3.rope.freq_base_swa", windowedBase));
            }

            return pairs;
        }

        List<(string Key, object Value)> real =
        [
            ("general.architecture", "gemma3"),
            ("gemma3.embedding_length", 1152),
            ("gemma3.block_count", 26),
            ("gemma3.attention.head_count", 4),
            ("gemma3.attention.head_count_kv", 1),
            ("gemma3.feed_forward_length", 6912),
            ("gemma3.attention.key_length", 256),
            ("gemma3.attention.value_length", 256),
            ("gemma3.attention.sliding_window", 512),
            ("gemma3.attention.layer_norm_rms_epsilon", 1e-6f),
            ("gemma3.rope.freq_base", 1_000_000f),
            ("gemma3.vocab_size", 262_144),
        ];

        if (windowedBase > 0f)
        {
            real.Add(("gemma3.rope.freq_base_swa", windowedBase));
        }

        return real;
    }

    // A complete seven-layer gemma3-shaped model under the tiny geometry: the per-head q/k norm vectors span one head
    // dimension each and the block and post-sublayer norm vectors span the hidden size, all required by the profile's
    // QK-norm default and post-sublayer placement. Both sets can be switched off for the load-failure cases.
    private sealed record TensorDef(string Name, int[] Dims); // dims in logical (row-major) order; vectors are one-dimensional

    private static List<TensorDef> TinyTensors(bool qkNorms = true, bool postNorms = true)
    {
        List<TensorDef> tensors = [new("token_embd.weight", [6, 8])];

        for (int layer = 0; layer < 7; layer++)
        {
            tensors.Add(new($"blk.{layer}.attn_q.weight", [12, 8])); // 3 heads of 4 dimensions against an 8-wide hidden size
            tensors.Add(new($"blk.{layer}.attn_k.weight", [4, 8]));
            tensors.Add(new($"blk.{layer}.attn_v.weight", [4, 8]));
            tensors.Add(new($"blk.{layer}.attn_output.weight", [8, 12]));
            tensors.Add(new($"blk.{layer}.ffn_gate.weight", [16, 8]));
            tensors.Add(new($"blk.{layer}.ffn_up.weight", [16, 8]));
            tensors.Add(new($"blk.{layer}.ffn_down.weight", [8, 16]));
            tensors.Add(new($"blk.{layer}.attn_norm.weight", [8]));
            tensors.Add(new($"blk.{layer}.ffn_norm.weight", [8]));

            if (qkNorms)
            {
                tensors.Add(new($"blk.{layer}.attn_q_norm.weight", [4]));
                tensors.Add(new($"blk.{layer}.attn_k_norm.weight", [4]));
            }

            if (postNorms)
            {
                tensors.Add(new($"blk.{layer}.post_attention_norm.weight", [8]));
                tensors.Add(new($"blk.{layer}.post_ffw_norm.weight", [8]));
            }
        }

        tensors.Add(new("output_norm.weight", [8]));

        return tensors;
    }

    /// <summary>Builds a GGUF v3 image from KV pairs (strings, ints, longs, floats, bools, string arrays) and F32 tensors. A window override restates the sliding-window declaration; -1 removes it entirely.</summary>
    private static byte[] BuildGguf(
        IReadOnlyList<(string Key, object Value)> kvPairs,
        IReadOnlyList<TensorDef> tensors,
        int? windowOverride = null)
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

        List<(string Key, object Value)> pairs = [.. kvPairs];

        if (windowOverride is not null)
        {
            pairs.RemoveAll(pair => pair.Key == "gemma3.attention.sliding_window");

            if (windowOverride.Value > 0)
            {
                pairs.Add(("gemma3.attention.sliding_window", windowOverride.Value));
            }
        }

        if (tensors.Count > 0)
        {
            // The tiny forward fixtures carry their vocabulary as a tokenizer vocabulary rather than a declared size,
            // which is the fallback the real file uses too.
            pairs.RemoveAll(pair => pair.Key.StartsWith("tokenizer.", StringComparison.Ordinal));
            WriteTokenVocabulary(pairs);
        }

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
