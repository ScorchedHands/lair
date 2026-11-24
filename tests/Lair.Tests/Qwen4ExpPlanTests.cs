using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Loader-side coverage of the qwen4exp profile against synthetic fixtures only — never the 75 GB target pair, whose
/// header gate lives in Qwen4ExpLoaderTests unchanged. The registered profile resolves a file that declares its own
/// interleave: per-layer gated-delta-net vs sparse-full kinds from <c>attention.compress_ratios</c>, cross-checked with
/// an optional <c>full_attention_interval</c>; the recurrent-state, indexer and hyper-connection groups under their exact
/// shipped names; a feed-forward width that arrives only as <c>expert_feed_forward_length</c>. The plan then carries the
/// resolved geometry — kinds, ratios, derived value heads and budget facts residency accounting sizes against — and its
/// tensor gate refuses tables whose per-layer material contradicts those declarations by concrete name. A valid fixture no
/// longer refuses at engine open: it opens on the bounded synthetic qwen4exp forward seam, whose forward coverage lives in
/// Qwen4ExpEngineTests — this suite pins the resolution and the tensor gate, and the seam-open posture they land on.
/// </summary>
[Trait("Category", "Loader")]
public sealed class Qwen4ExpPlanTests : IDisposable
{
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
                // temp cleanup is best-effort
            }
        }
    }

    private const int Layers = 4;          // ratios [0,0,0,1]: three linear blocks, one sparse full block per group of four
    private const int Hidden = 8;
    private const int StateDim = 2;        // ssm.state_size — the shared head dimension every value head carries
    private const int InnerSize = 4;       // ssm.inner_size over state dim: two derived value heads per linear block
    private const int TimeStepRank = 2;    // must equal the derived value-head count exactly
    private const int GroupCount = 1;      // key-head groups → conv channels 2 * (1 x 2) + 4 = 8 == hidden
    private const int ConvKernel = 3;
    private const int IndexerHeads = 2;
    private const int IndexerKeyLength = 4; // even component width so the engine seam can rotate indexer heads in pairs
    private const int TopKCells = 6;
    private const int StreamCount = 2;     // flattened residual width: 2 x hidden = 16
    private const int LowRank = 3;
    private const int ExpertCount = 2;
    private const int ExpertsPerToken = 1;
    private const int IntermediateSize = 6; // declared only as qwen4exp.expert_feed_forward_length: no dense key exists

    private static readonly AttentionKind[] ExpectedKinds =
    [
        AttentionKind.LinearGatedDeltaNet,
        AttentionKind.LinearGatedDeltaNet,
        AttentionKind.LinearGatedDeltaNet,
        AttentionKind.FullQuerySparseAttention,
    ];

    // ---------------------------------------------------------------- the registered profile itself

    [Fact]
    public void ModelProfiles_Qwen4Exp_DeclaresFamilySemanticsAndNamesTheShippedTensorSpellings()
    {
        Assert.Contains("qwen4exp", ModelProfiles.SupportedArchitectures);

        ModelProfile profile = ModelProfiles.Resolve("qwen4exp");

        Assert.Equal("qwen4exp.", profile.Prefix);
        Assert.True(profile.DeclaresQwen4ExpSemantics); // the only registered profile that resolves family geometry at all
        Assert.Equal(NormStrategy.RmsNorm, profile.Norm);
        Assert.Equal(ActivationStrategy.SwiGLU, profile.Activation);

        // The sparse shape is qwen2moe's — router plus stacked expert triplets with a shared-expert side the selector
        // vector names — because this conversion writes its FFN material under exactly those *_exps/*_shexp names.
        SparseFfnDescriptor sparse = profile.SparseFfn!;
        Assert.True(sparse.HasSharedExpert);
        Assert.Equal(TensorRole.MoeSharedExpertInputGate, sparse.SharedInputGateRole);

        TensorMap tensors = profile.Tensors;

        // The plain sublayer/output norms and the split q/k/v/output quartet relax: which layers carry them follows that
        // layer's resolved kind, not a flat flag — only full blocks project through separate tensors.
        Assert.False(tensors[TensorRole.AttentionQuery].Required);
        Assert.False(tensors[TensorRole.FeedForwardGate].Required);
        Assert.False(tensors[TensorRole.OutputNorm].Required);

        // Family tensor names exactly as the target file spells them — ssm_a carries no .weight suffix, dt bias is a
        // bare .bias, hyper-connections live under hc_attn_/hc_ffn_ per sublayer side and output_hc_ globally.
        Assert.Equal("blk.{layer}.attn_qkv.weight", tensors[TensorRole.AttentionQkv].Name);
        Assert.Equal("blk.{layer}.attn_gate.weight", tensors[TensorRole.LinearAttentionGate].Name);
        Assert.Equal("blk.{layer}.ssm_a", tensors[TensorRole.SsmDecayWeight].Name);
        Assert.Equal("blk.{layer}.ssm_dt.bias", tensors[TensorRole.SsmTimeStepBias].Name);
        Assert.Equal("blk.{layer}.indexer.q_proj.weight", tensors[TensorRole.IndexerQueryProjection].Name);
        Assert.Equal("blk.{layer}.hc_ffn_up.weight", tensors[TensorRole.FeedForwardStreamUp].Name);
        Assert.Equal(TensorScope.Global, tensors[TensorRole.OutputStreamDown].Scope);
        Assert.Equal("output_hc_norm.weight", tensors[TensorRole.OutputStreamNorm].Name);

        // PLE names follow the same target file and stay optional: only layers its declared ple.layers list selects ship them.
        Assert.False(tensors[TensorRole.PleConvolutionKernel].Required);
        Assert.Equal("blk.{layer}.ple_norm_query.weight", tensors[TensorRole.PleQueryNorm].Name);
    }

    // ---------------------------------------------------------------- the synthetic plan resolves family geometry from its own declarations

    [Fact]
    public void ModelPlan_Qwen4ExpResolvesInterleaveStateGeometryAndBudgetFactsFromDeclaredMetadata()
    {
        using GgufFile file = OpenFixture(BuildGguf(MetadataPairs(), FixtureTensors()));

        List<string> logLines = [];
        ModelPlan plan = ModelPlans.Resolve(file, logLines.Add);

        Assert.Equal("qwen4exp", plan.Architecture);
        Assert.Equal("qwen4exp", plan.Profile); // registered: never the generic fallback identity
        Assert.NotNull(plan.Qwen4Exp); // a registered family-semantics profile resolves geometry; every other family carries null

        Qwen4ExpGeometry geometry = plan.Qwen4Exp!;

        // The interleave lands as declared ratios and resolved kinds, with the interval cross-check satisfied silently.
        Assert.Equal([0, 0, 0, 1], [.. geometry.CompressRatios]);
        Assert.Equal(ExpectedKinds, [.. geometry.AttentionKinds]);
        Assert.Equal(3, geometry.LinearLayerCount);
        Assert.Equal(1, geometry.FullLayerCount);

        // Derived value heads: inner_size / state_size = 2 — and the declared gate-row count covers exactly those heads.
        GdnDescriptor gdn = geometry.LinearAttention;
        Assert.Equal(GroupCount, gdn.KeyHeadCount);
        Assert.Equal(2, gdn.ValueHeadCount);
        Assert.Equal(StateDim, gdn.StateDim);
        Assert.Equal(InnerSize, gdn.InnerSize);
        Assert.Equal(TimeStepRank, gdn.TimeStepRank);

        // Fused-buffer rows and conv channels: two key-head regions plus the value width — 2 * (1 x 2) + 4 = 8.
        Assert.Equal(GroupCount * StateDim, gdn.QkRegionFloats);
        Assert.Equal(8, gdn.ConvChannels);

        // Recurrent footprint per linear block: each head's full slot matrix plus the raw convolution history —
        // (2 x 2 x 2) + ((3 - 1) x 8) = 24 slots; across this fixture's three linear blocks that is 72, and it is what a
        // residency tier must seat before demoting anything on the linear side.
        Assert.Equal(8L, gdn.RecurrentStateValuesPerSequence);
        Assert.Equal(16L, gdn.ConvHistoryValuesPerSequence);
        Assert.Equal(24L, gdn.StateSlotsPerSequence);
        Assert.Equal(72L, geometry.RecurrentStateSlotsPerSequence);

        // Indexer budgets: every cached cell carries both scoring heads' component regions (8 values), the window's full
        // block footprint is cells x per-cell width x one full layer — 4 positions = 32 slots here — and a token keeps at
        // most top_k / ratio complete pools, so this fixture's ratio-1 layer budgets all six while ratio 2 halves it.
        QsaDescriptor qsa = geometry.SparseAttention;
        Assert.Equal(8L, qsa.IndexerCacheValuesPerCachedCell);
        Assert.Equal(32L, geometry.IndexerCacheValuesForWindow(4));
        Assert.Equal(104L, geometry.RecurrentAndIndexerSlotsForWindow(4));
        Assert.Equal(TopKCells, qsa.PoolBudgetForRatio(1));
        Assert.Equal(TopKCells / 2, qsa.PoolBudgetForRatio(2));

        // Hyper-connection resident shape: each block ships a norm vector (flattened width), one inject row per stream and
        // the low-rank down/up pair on both sublayer sides — 16 + 32 + 96 = 144 floats per side, x8 across four blocks;
        // the global trio carries that same norm plus a final down/up pair without an inject row.
        HyperConnectionDescriptor hc = geometry.HyperConnections;
        Assert.Equal(16, hc.FlattenedStreamFloats(Hidden));
        Assert.Equal(144L, geometry.PerSublayerHyperConnectionFloats(Hidden));
        Assert.Equal(1152L, geometry.HyperConnectionResidentFloats(Hidden));
        Assert.Equal(16L + 2L * 16 * LowRank, geometry.OutputHyperConnectionFloats(Hidden));

        // The feed-forward width arrives only as the expert declaration — this family writes no dense key at all — so the
        // substitution lands in the plan and reports itself once beside the vocabulary fallback the fixture also needs.
        Assert.Equal(IntermediateSize, plan.IntermediateSize);
        string logs = string.Join("\n", logLines);
        Assert.Contains("ffn_fallback model='", logs);
        Assert.Contains("source=qwen4exp.expert_feed_forward_length value=6", logs);
        Assert.Contains("vocab_fallback ", logs);

        // The sparse counts resolve from the file like every MoE family's: declared experts and routing width, no shared
        // count key in this fixture (the target declares none either), resident dense strategy.
        SparseFfnDescriptor resolved = plan.SparseFfn!;
        Assert.Equal(ExpertCount, resolved.ExpertCount);
        Assert.Equal(ExpertsPerToken, resolved.RoutedExpertsPerToken);
        Assert.False(resolved.HasSharedExpert); // no qwen4exp.expert_shared_count declared here
        Assert.Equal(SparseFfnLayout.StackedExperts, resolved.Layout);
        Assert.Equal(SparseFfnStrategy.DenseResident, plan.SparseFfnStrategy!.Value);

        Assert.True(plan.HasOutputWeights); // the fixture ships an untied output head beside token_embd
    }

    [Fact]
    public void ModelPlan_OtherArchitectures_ResolveNoQwen4ExpGeometry()
    {
        // A llama-shaped file against its own profile resolves exactly as before — and carries nothing family-specific.
        List<(string Key, object Value)> pairs =
        [
            ("general.architecture", "llama"),
            ("llama.embedding_length", Hidden),
            ("llama.block_count", 1),
            ("llama.attention.head_count", 2),
            ("llama.attention.head_count_kv", 1),
            ("llama.feed_forward_length", IntermediateSize),
            ("llama.attention.layer_norm_rms_epsilon", 1e-5f),
            ("llama.vocab_size", 5),
        ];

        List<TensorDef> tensors =
        [
            new("token_embd.weight", [5, Hidden]),
            new("blk.0.attn_q.weight", [8, Hidden]),
            new("blk.0.attn_k.weight", [4, Hidden]),
            new("blk.0.attn_v.weight", [4, Hidden]),
            new("blk.0.attn_output.weight", [Hidden, 8]),
            new("blk.0.ffn_gate.weight", [IntermediateSize, Hidden]),
            new("blk.0.ffn_up.weight", [IntermediateSize, Hidden]),
            new("blk.0.ffn_down.weight", [Hidden, IntermediateSize]),
            new("blk.0.attn_norm.weight", [Hidden]),
            new("blk.0.ffn_norm.weight", [Hidden]),
            new("output_norm.weight", [Hidden]),
        ];

        using GgufFile file = OpenFixture(BuildGguf(pairs, tensors));
        ModelPlan plan = ModelPlans.Resolve(file);

        Assert.Null(plan.Qwen4Exp);
    }

    // ---------------------------------------------------------------- the interleave gate refuses declarations it cannot resolve

    [Fact]
    public void ResolveQwen4Exp_MissingOrContradictoryInterleaveDeclarations_NamedByTheirKeys()
    {
        List<(string Key, object Value)> missingRatios = [.. MetadataPairs()];
        missingRatios.RemoveAll(p => p.Key == "qwen4exp.attention.compress_ratios");

        using GgufFile noRatios = OpenFixture(BuildGguf(missingRatios, FixtureTensors()));
        ModelException ratiosEx = Assert.Throws<ModelException>(() => ModelPlans.Resolve(noRatios));
        Assert.Contains("qwen4exp.attention.compress_ratios", ratiosEx.Message);

        // Wrong array length names both sides: what was declared and the block_count it must cover.
        List<(string Key, object Value)> shortRatios = [.. MetadataPairs()];
        int ratioIndex = shortRatios.FindIndex(p => p.Key == "qwen4exp.attention.compress_ratios");
        shortRatios[ratioIndex] = ("qwen4exp.attention.compress_ratios", new int[] { 0, 0, 1 });

        using GgufFile mismatched = OpenFixture(BuildGguf(shortRatios, FixtureTensors()));
        ModelException lengthEx = Assert.Throws<ModelException>(() => ModelPlans.Resolve(mismatched));
        Assert.Contains("declares 3 entries but qwen4exp.block_count is 4", lengthEx.Message);

        // A declared interval whose group ends contradict the array fails naming both sides with the layer index — first a
        // positive ratio that lands mid-group, then its mirror: a group end that declares no full attention.
        List<(string Key, object Value)> earlyRatio = [.. MetadataPairs()];
        int sameIndex = earlyRatio.FindIndex(p => p.Key == "qwen4exp.attention.compress_ratios");
        earlyRatio[sameIndex] = ("qwen4exp.attention.compress_ratios", new int[] { 0, 1, 0, 0 });

        using GgufFile contradicts = OpenFixture(BuildGguf(earlyRatio, FixtureTensors()));
        ModelException intervalEx = Assert.Throws<ModelException>(() => ModelPlans.Resolve(contradicts));
        Assert.Contains("'qwen4exp.full_attention_interval' (4) contradicts 'qwen4exp.attention.compress_ratios'[1] = 1", intervalEx.Message);

        List<(string Key, object Value)> allLinear = [.. MetadataPairs()];
        int linearIndex = allLinear.FindIndex(p => p.Key == "qwen4exp.attention.compress_ratios");
        allLinear[linearIndex] = ("qwen4exp.attention.compress_ratios", new int[] { 0, 0, 0, 0 });

        using GgufFile noGroupEnds = OpenFixture(BuildGguf(allLinear, FixtureTensors()));
        ModelException mirrorEx = Assert.Throws<ModelException>(() => ModelPlans.Resolve(noGroupEnds));
        Assert.Contains("every last position of an interval group must declare a positive compress ratio", mirrorEx.Message);

        // Absent interval leaves the array sole authority: an all-linear declaration — ratios every zero, no group-end rule at
        // all — resolves cleanly against a table where every layer carries linear material.
        List<(string Key, object Value)> noInterval = [.. MetadataPairs()];
        int authorityIndex = noInterval.FindIndex(p => p.Key == "qwen4exp.attention.compress_ratios");
        noInterval[authorityIndex] = ("qwen4exp.attention.compress_ratios", new int[] { 0, 0, 0, 0 });
        noInterval.RemoveAll(p => p.Key == "qwen4exp.full_attention_interval");

        using GgufFile authoritative = OpenFixture(BuildGguf(noInterval, FixtureTensorsAllLinear()));
        ModelPlan plan = ModelPlans.Resolve(authoritative);
        Assert.Equal(0, plan.Qwen4Exp!.FullLayerCount); // ratios alone decided: no full layers this time
    }

    [Fact]
    public void ResolveQwen4Exp_StateGroupContradictions_NamedWithTheirValues()
    {
        // A value width that is not a whole number of head-dimension slices names both keys with their values.
        List<(string Key, object Value)> indivisible = [.. MetadataPairs()];
        int innerIndex = indivisible.FindIndex(p => p.Key == "qwen4exp.ssm.inner_size");
        indivisible[innerIndex] = ("qwen4exp.ssm.inner_size", 5);

        using GgufFile badInner = OpenFixture(BuildGguf(indivisible, FixtureTensors()));
        ModelException innerEx = Assert.Throws<ModelException>(() => ModelPlans.Resolve(badInner));
        Assert.Contains("ssm.inner_size (5) is not a whole number of value heads at qwen4exp.ssm.state_size (2)", innerEx.Message);

        // Declared gate rows that do not cover the derived value heads fail naming key + value against the derivation —
        // decay weights, dt biases and alpha/beta rows each carry exactly one entry per head.
        List<(string Key, object Value)> wrongRank = [.. MetadataPairs()];
        int rankIndex = wrongRank.FindIndex(p => p.Key == "qwen4exp.ssm.time_step_rank");
        wrongRank[rankIndex] = ("qwen4exp.ssm.time_step_rank", 3);

        using GgufFile badRank = OpenFixture(BuildGguf(wrongRank, FixtureTensors()));
        ModelException rankEx = Assert.Throws<ModelException>(() => ModelPlans.Resolve(badRank));
        Assert.Contains("ssm.time_step_rank (3) does not cover the derived value heads of qwen4exp.ssm.inner_size / qwen4exp.ssm.state_size = 2", rankEx.Message);

        // A required state-group key is named when absent.
        List<(string Key, object Value)> noStateSize = [.. MetadataPairs()];
        noStateSize.RemoveAll(p => p.Key == "qwen4exp.ssm.state_size");

        using GgufFile missingKey = OpenFixture(BuildGguf(noStateSize, FixtureTensors()));
        Assert.Contains("qwen4exp.ssm.state_size", Assert.Throws<ModelException>(() => ModelPlans.Resolve(missingKey)).Message);
    }

    // ---------------------------------------------------------------- the tensor gate refuses tables that contradict the interleave

    [Fact]
    public void ValidateQwen4ExpTable_MisplacedOrMissingFamilyTensors_NamedWithConcreteLayerIndexes()
    {
        using CpuBackend backend = new();

        // ssm_* material on a layer whose ratio says full sparse attention runs there: named as misplaced, not ignored.
        List<TensorDef> misplacedFull = [.. FixtureTensors()];
        misplacedFull.Add(new TensorDef("blk.3.ssm_norm.weight", [StateDim]));

        using GgufFile extraOnSparse = OpenFixture(BuildGguf(MetadataPairs(), misplacedFull));
        ModelException misEx = Assert.Throws<ModelException>(() => TransformerModel.Open(extraOnSparse, backend, maxContextTokens: 8));
        Assert.Contains("misplaced tensors: blk.3.ssm_norm.weight", misEx.Message);

        // Split q/k/v material on a linear layer is the mirror refusal — named by what the file actually carries.
        List<TensorDef> misplacedLinear = [.. FixtureTensors()];
        misplacedLinear.Add(new TensorDef("blk.0.attn_k.weight", [4, Hidden]));

        using GgufFile extraOnLinear = OpenFixture(BuildGguf(MetadataPairs(), misplacedLinear));
        ModelException mirrorEx = Assert.Throws<ModelException>(() => TransformerModel.Open(extraOnLinear, backend, maxContextTokens: 8));
        Assert.Contains("misplaced tensors: blk.0.attn_k.weight", mirrorEx.Message);

        // A required linear-block tensor that never arrives is named exactly the way the engine would have — through this
        // gate, before any payload work was spent on a model whose declared groups contradict its table.
        List<TensorDef> missingGate = [.. FixtureTensors()];
        missingGate.RemoveAll(t => t.Name == "blk.1.attn_gate.weight");

        using GgufFile noGate = OpenFixture(BuildGguf(MetadataPairs(), missingGate));
        ModelException gateEx = Assert.Throws<ModelException>(() => TransformerModel.Open(noGate, backend, maxContextTokens: 8));
        Assert.Contains("missing tensors: blk.1.attn_gate.weight", gateEx.Message);

        // The same refusal covers the sparse side's indexer quartet and the global hyper-connection trio by name.
        List<TensorDef> noIndexer = [.. FixtureTensors()];
        noIndexer.RemoveAll(t => t.Name == "blk.3.indexer.k_proj.weight");

        using GgufFile missingIndexer = OpenFixture(BuildGguf(MetadataPairs(), noIndexer));
        Assert.Contains("missing tensors: blk.3.indexer.k_proj.weight", Assert.Throws<ModelException>(() => TransformerModel.Open(missingIndexer, backend, maxContextTokens: 8)).Message);

        List<TensorDef> noOutputHc = [.. FixtureTensors()];
        noOutputHc.RemoveAll(t => t.Name == "output_hc_down.weight");

        using GgufFile missingGlobal = OpenFixture(BuildGguf(MetadataPairs(), noOutputHc));
        Assert.Contains("missing tensors: output_hc_down.weight", Assert.Throws<ModelException>(() => TransformerModel.Open(missingGlobal, backend, maxContextTokens: 8)).Message);
    }

    [Fact]
    public void ValidateQwen4ExpTable_HyperConnectionAndRecurrentShapes_CheckedAgainstDeclaredGeometry()
    {
        using CpuBackend backend = new();

        // Inject rows are stream count x flattened width — a fixture that widens them to four streams contradicts the
        // declared hyper_connection.count and is refused naming its own shape beside what it needs.
        List<TensorDef> badInject = [.. FixtureTensors()];
        int injectIndex = badInject.FindIndex(t => t.Name == "blk.2.hc_attn_inject.weight");
        badInject[injectIndex] = new TensorDef("blk.2.hc_attn_inject.weight", [4, StreamCount * Hidden]);

        using GgufFile mismatchedInject = OpenFixture(BuildGguf(MetadataPairs(), badInject));
        ModelException injectEx = Assert.Throws<ModelException>(() => TransformerModel.Open(mismatchedInject, backend, maxContextTokens: 8));
        Assert.Contains("blk.2.hc_attn_inject.weight", injectEx.Message);
        Assert.Contains("[4, 16]", injectEx.Message); // declared shape as the file spells it
        Assert.Contains("needs [2, 16]", injectEx.Message); // stream count x flattened width

        // Conv rows must cover every fused channel across kernel columns: 8 channels at this fixture's geometry, so a
        // shorter row set is refused naming what arrived against what the declared groups add up to.
        List<TensorDef> badConv = [.. FixtureTensors()];
        int convIndex = badConv.FindIndex(t => t.Name == "blk.0.ssm_conv1d.weight");
        badConv[convIndex] = new TensorDef("blk.0.ssm_conv1d.weight", [6, ConvKernel]);

        using GgufFile mismatchedConv = OpenFixture(BuildGguf(MetadataPairs(), badConv));
        ModelException convEx = Assert.Throws<ModelException>(() => TransformerModel.Open(mismatchedConv, backend, maxContextTokens: 8));
        Assert.Contains("blk.0.ssm_conv1d.weight", convEx.Message);
        Assert.Contains("[6, 3]", convEx.Message);

        // The decay weight vector must carry one entry per value head (2 here) — a longer vector is refused naming its length.
        List<TensorDef> badDecay = [.. FixtureTensors()];
        int decayIndex = badDecay.FindIndex(t => t.Name == "blk.0.ssm_a");
        badDecay[decayIndex] = new TensorDef("blk.0.ssm_a", [3]);

        using GgufFile mismatchedDecay = OpenFixture(BuildGguf(MetadataPairs(), badDecay));
        ModelException decayEx = Assert.Throws<ModelException>(() => TransformerModel.Open(mismatchedDecay, backend, maxContextTokens: 8));
        Assert.Contains("blk.0.ssm_a", decayEx.Message);
    }

    // ---------------------------------------------------------------- the resolved plan opens the bounded synthetic forward seam

    [Fact]
    public void TransformerModel_Qwen4ExpPlan_ResolvesGeometryAndOpensTheBoundedSyntheticForwardSeam()
    {
        using GgufFile file = OpenFixture(BuildGguf(MetadataPairs(), FixtureTensors()));
        using CpuBackend backend = new();

        using TransformerModel model = TransformerModel.Open(file, backend, maxContextTokens: 8);

        // The composition exists now: a valid fixture opens on the assembly-internal qwen4exp engine seam instead of the
        // standard dense loading, and the seam reports itself on the opened model. Forward coverage lives in Qwen4ExpEngineTests.
        Assert.True(model.HasQwen4ExpForward);

        // The plan itself resolved fully before the seam opened — this is the same file through ModelPlans with its geometry intact.
        ModelPlan plan = ModelPlans.Resolve(file);
        Assert.NotNull(plan.Qwen4Exp);
        Assert.Equal(3, plan.Qwen4Exp.LinearLayerCount);
    }

    // ---------------------------------------------------------------- the target's positional posture resolves under this family's prefix

    [Fact]
    public void ModelPlan_Qwen4ExpDeclaredHeadWidthAndPartialRotarySlice_ResolveForTheForwardToCompose()
    {
        // The pinned target declares its head/key/value width outright and narrows rotation with rope.dimension_count; the
        // plan must land both values rather than rejecting the partial posture, so the bounded forward composes over them.
        List<(string Key, object Value)> declared =
        [
            .. MetadataPairs(),
            ("qwen4exp.attention.head_length", 4),
            ("qwen4exp.rope.dimension_count", 2),
        ];

        using GgufFile file = OpenFixture(BuildGguf(declared, FixtureTensors()));
        ModelPlan plan = ModelPlans.Resolve(file);

        Assert.Equal(4, plan.HeadDim); // declared width rather than the hidden / head_count derivation
        Assert.Equal(2, plan.RopeRotaryDimensions); // a leading slice: the rest of each region passes through unchanged

        // A slice wider than the heads it must pair inside still fails naming the key and the declared value.
        List<(string Key, object Value)> tooWide =
        [
            .. MetadataPairs(),
            ("qwen4exp.attention.head_length", 4),
            ("qwen4exp.rope.dimension_count", 6),
        ];

        using GgufFile oversized = OpenFixture(BuildGguf(tooWide, FixtureTensors()));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(oversized));
        Assert.Contains("qwen4exp.rope.dimension_count", ex.Message);
        Assert.Contains("6", ex.Message);
    }

    // ---------------------------------------------------------------- fixture builders (synthetic GGUF v3 images; array metadata supported)

    private static List<(string Key, object Value)> MetadataPairs() =>
        [
            ("general.architecture", "qwen4exp"),
            ("qwen4exp.embedding_length", Hidden),
            ("qwen4exp.block_count", Layers),
            ("qwen4exp.attention.head_count", 2),
            ("qwen4exp.attention.head_count_kv", 1),
            // No qwen4exp.feed_forward_length: this family declares its FFN width only as the expert one, exactly as the target file does.
            ("qwen4exp.expert_feed_forward_length", IntermediateSize),
            ("qwen4exp.attention.layer_norm_rms_epsilon", 1e-5f),
            ("qwen4exp.ssm.group_count", GroupCount),
            ("qwen4exp.ssm.state_size", StateDim),
            ("qwen4exp.ssm.inner_size", InnerSize),
            ("qwen4exp.ssm.time_step_rank", TimeStepRank),
            ("qwen4exp.ssm.conv_kernel", ConvKernel),
            ("qwen4exp.attention.compress_ratios", new int[] { 0, 0, 0, 1 }),
            ("qwen4exp.full_attention_interval", 4),
            ("qwen4exp.attention.indexer.head_count", IndexerHeads),
            ("qwen4exp.attention.indexer.key_length", IndexerKeyLength),
            ("qwen4exp.attention.indexer.top_k", TopKCells),
            ("qwen4exp.hyper_connection.count", StreamCount),
            ("qwen4exp.hyper_connection.low_rank", LowRank),
            ("qwen4exp.expert_count", ExpertCount),
            ("qwen4exp.expert_used_count", ExpertsPerToken),
            // No qwen4exp.vocab_size either: the vocabulary resolves from tokenizer.ggml.tokens and reports its fallback.
            ("tokenizer.ggml.tokens", new string[] { "a", "b", "c", "d", "e" }),
        ];

    private static List<TensorDef> FixtureTensors()
    {
        List<TensorDef> tensors = GlobalDefs();

        for (int layer = 0; layer < Layers; layer++)
        {
            tensors.AddRange(LayerCommonDefs(layer));
            tensors.AddRange(ExpectedKinds[layer] == AttentionKind.LinearGatedDeltaNet ? LinearDefs(layer) : FullDefs(layer));
        }

        // The global output-side trio: one norm plus a final down/up pair, no inject row at this scope.
        int flattened = StreamCount * Hidden;
        tensors.Add(new TensorDef("output_hc_down.weight", [LowRank, flattened]));
        tensors.Add(new TensorDef("output_hc_norm.weight", [flattened]));
        tensors.Add(new TensorDef("output_hc_up.weight", [flattened, LowRank]));

        return tensors;
    }

    // The same fixture with every layer linear: what the ratios-only interleave (no interval declared) resolves to, so a
    // table can match it — used by the array-sole-authority case.
    private static List<TensorDef> FixtureTensorsAllLinear()
    {
        List<TensorDef> tensors = GlobalDefs();

        for (int layer = 0; layer < Layers; layer++)
        {
            tensors.AddRange(LayerCommonDefs(layer));
            tensors.AddRange(LinearDefs(layer));
        }

        int flattened = StreamCount * Hidden;
        tensors.Add(new TensorDef("output_hc_down.weight", [LowRank, flattened]));
        tensors.Add(new TensorDef("output_hc_norm.weight", [flattened]));
        tensors.Add(new TensorDef("output_hc_up.weight", [flattened, LowRank]));

        return tensors;
    }

    private static List<TensorDef> GlobalDefs() =>
        [new("token_embd.weight", [5, Hidden]), new("output.weight", [5, Hidden])];

    // What every block carries whichever attention it runs: the hyper-connection pair on both sublayer sides and the routed FFN side.
    private static List<TensorDef> LayerCommonDefs(int layer)
    {
        int flattened = StreamCount * Hidden; // 16: the whole residual width every stream tensor crosses
        List<TensorDef> defs = [];

        foreach (string side in new[] { "hc_attn", "hc_ffn" })
        {
            defs.Add(new TensorDef($"blk.{layer}.{side}_inject.weight", [StreamCount, flattened]));
            defs.Add(new TensorDef($"blk.{layer}.{side}_down.weight", [LowRank, flattened]));
            defs.Add(new TensorDef($"blk.{layer}.{side}_up.weight", [flattened, LowRank]));
            defs.Add(new TensorDef($"blk.{layer}.{side}_norm.weight", [flattened]));
        }

        // Router plus stacked expert triplets (shared-expert names optional and absent here).
        defs.Add(new TensorDef($"blk.{layer}.ffn_gate_inp.weight", [ExpertCount, Hidden]));
        defs.Add(new TensorDef($"blk.{layer}.ffn_gate_exps.weight", [ExpertCount, IntermediateSize, Hidden]));
        defs.Add(new TensorDef($"blk.{layer}.ffn_up_exps.weight", [ExpertCount, IntermediateSize, Hidden]));
        defs.Add(new TensorDef($"blk.{layer}.ffn_down_exps.weight", [ExpertCount, Hidden, IntermediateSize]));

        return defs;
    }

    private static List<TensorDef> LinearDefs(int layer) =>
        [
            // Fused qkv buffer at exactly the conv channels, its z gate across inner size, and all seven recurrent-state tensors.
            new($"blk.{layer}.attn_qkv.weight", [2 * GroupCount * StateDim + InnerSize, Hidden]),
            new($"blk.{layer}.attn_gate.weight", [InnerSize, Hidden]),
            new($"blk.{layer}.ssm_a", [TimeStepRank]), // decay weight: no .weight suffix in this family's spelling
            new($"blk.{layer}.ssm_alpha.weight", [TimeStepRank, Hidden]),
            new($"blk.{layer}.ssm_beta.weight", [TimeStepRank, Hidden]),
            new($"blk.{layer}.ssm_conv1d.weight", [2 * GroupCount * StateDim + InnerSize, ConvKernel]),
            new($"blk.{layer}.ssm_dt.bias", [TimeStepRank]),
            new($"blk.{layer}.ssm_norm.weight", [StateDim]),
            new($"blk.{layer}.ssm_out.weight", [Hidden, InnerSize]),
        ];

    private static List<TensorDef> FullDefs(int layer) =>
        [
            // Split projections plus the indexer quartet — no ssm_* or attn_gate material on this layer. The q projection
            // carries the reference's interleaved query-and-gate rows (2 * head_count * head_dim), which is what the engine
            // seam reads when it composes this layer.
            new($"blk.{layer}.attn_q.weight", [2 * 2 * (Hidden / 2), Hidden]),
            new($"blk.{layer}.attn_k.weight", [Hidden / 2, Hidden]),
            new($"blk.{layer}.attn_v.weight", [Hidden / 2, Hidden]),
            new($"blk.{layer}.attn_output.weight", [Hidden, Hidden]),
            new($"blk.{layer}.indexer.q_proj.weight", [IndexerHeads * IndexerKeyLength, Hidden]),
            new($"blk.{layer}.indexer.k_proj.weight", [IndexerKeyLength, Hidden]),
            new($"blk.{layer}.indexer.q_norm.weight", [IndexerKeyLength]),
            new($"blk.{layer}.indexer.k_norm.weight", [IndexerKeyLength]),
        ];

    private string OpenFixtureWriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-qwen4exp-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    private GgufFile OpenFixture(byte[] image) => GgufFile.Open(OpenFixtureWriteTemp(image));

    // GGUF v3 builder with the metadata value types these fixtures need: strings, int32s, floats — and int32/string arrays.
    private static byte[] BuildGguf(IReadOnlyList<(string Key, object Value)> kvPairs, IReadOnlyList<TensorDef> tensors)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        const uint GgufMagic = 0x46554747; // "GGUF" — the same constant the loader validates against

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
                case int[] ints:
                    writer.Write(9u); // ARRAY of INT32
                    writer.Write(5u);
                    writer.Write((ulong)ints.Length);
                    foreach (int i in ints)
                    {
                        writer.Write(i);
                    }

                    break;
                case string[] strings:
                    writer.Write(9u); // ARRAY of STRING
                    writer.Write(8u);
                    writer.Write((ulong)strings.Length);
                    foreach (string s in strings)
                    {
                        WriteGgufString(writer, s);
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

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private sealed record TensorDef(string Name, int[] Dims); // dims in logical (row-major) order; vectors are one-dimensional
}
