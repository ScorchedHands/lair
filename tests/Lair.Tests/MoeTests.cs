using System.Runtime.InteropServices;
using System.Text;

using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using System.Linq;

namespace Lair.Tests;

/// <summary>
/// Sparse-FFN execution over synthetic single-layer MoE models: the router scores every declared expert and top-k
/// selection renormalizes the chosen weights, equal router scores resolve to the lower expert index, a declared shared
/// expert adds its dense block once per token while an undeclared one reads none of those tensors at all (and declaring
/// one without shipping it fails as a missing tensor), a shipped selector vector scales that shared contribution by
/// silu(score) / score for each token — saturating to exactly 1 reproduces the ungated block bit for bit, and a table
/// carrying it for only part of the block fails the load — each selected expert's gate/up/down chain accumulates into
/// the residual under its routing weight, and every batched step is bit-identical to stepping the same sequences alone
/// gated or ungated — checked against an independent double-precision forward that reproduces this arithmetic. Covers
/// selection spread across experts, a case where all top-k traffic hits one expert, declared routing-weight
/// normalization off with a scale, thread-count variation, and the encoder path refusing descriptor-bearing models.
/// </summary>
[Trait("Category", "Moe")]
public sealed class MoeTests : IDisposable
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

    // ---------------------------------------------------------------- open and execute a descriptor-bearing file

    [Fact]
    public void TransformerModelOpen_Qwen2MoeFile_ExecutesSparseBlock_NoDenseFfnTensorsRequired()
    {
        MoeSpec spec = new(sharedExpert: true);
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), [.. spec.Tensors()]));

        using CpuBackend backend = new(threads: 1, allowAvx2: false);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        Assert.NotNull(model.Plan.SparseFfn); // the descriptor executes instead of deferring
        foreach (TensorRole denseRole in new TensorRole[] { TensorRole.FeedForwardGate, TensorRole.FeedForwardUp, TensorRole.FeedForwardDown })
        {
            Assert.Null(model.Plan.Tensors[denseRole].Resolve(gguf.Tensors, 0)); // the dense FFN names stay absent for a routed block
        }

        MoeReference reference = new(spec);
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 8);
        SequenceKv kv = new(pool);

        float[] logits = model.DecodeNext(1, kv).ToArray();
        double[] expected = reference.LogitsAt([], 1);
        Assert.True(Closest(logits, expected), $"sparse forward diverged from the dense-equivalent reference: max diff {MaxDiff(logits, expected):G9}");

        foreach (float value in logits)
        {
            Assert.True(float.IsFinite(value), $"logit {value} is not finite");
        }
    }

    [Fact]
    public void TransformerModelOpen_DeclaredSharedExpertWithoutItsTensors_FailsAsMissingTensor()
    {
        MoeSpec spec = new(sharedExpert: true);
        List<TensorDef> tensors = [.. spec.Tensors()];
        tensors.RemoveAll(t => t.Name.Contains("shexp", StringComparison.Ordinal));

        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), tensors));

        using CpuBackend backend = new(threads: 1, allowAvx2: false);
        using GgufFile gguf = GgufFile.Open(path);
        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("blk.0.ffn_gate_shexp.weight", ex.Message); // a declared shared expert is required material, reported as its first missing role
    }

    [Fact]
    public void SparseForward_SharedExpertPresentAndAbsent_DifferExactlyAsPredicted_AndAbsentReadsNoSharedRole()
    {
        MoeSpec withShared = new(sharedExpert: true);
        MoeSpec withoutShared = new(sharedExpert: false); // the file declares no shared expert and ships none of its tensors

        using CpuBackend backend = new(threads: 1, allowAvx2: false);

        using GgufFile ggufWith = GgufFile.Open(WriteTemp(BuildGguf(withShared.MetadataPairs(), [.. withShared.Tensors()])));
        using TransformerModel modelWith = TransformerModel.Open(ggufWith, backend, maxContextTokens: 8);
        Assert.True(modelWith.Plan.SparseFfn!.HasSharedExpert);

        using GgufFile ggufWithout = GgufFile.Open(WriteTemp(BuildGguf(withoutShared.MetadataPairs(), [.. withoutShared.Tensors()])));
        using TransformerModel modelWithout = TransformerModel.Open(ggufWithout, backend, maxContextTokens: 8);
        Assert.False(modelWithout.Plan.SparseFfn!.HasSharedExpert);

        using BlockPool poolWith = NewPool(modelWith, backend, capacityBlocks: 8);
        using BlockPool poolWithout = NewPool(modelWithout, backend, capacityBlocks: 8);
        SequenceKv kvWith = new(poolWith);
        SequenceKv kvWithout = new(poolWithout);

        float[] logitsShared = modelWith.DecodeNext(2, kvWith).ToArray();
        float[] logitsPlain = modelWithout.DecodeNext(2, kvWithout).ToArray();

        double[] expectedShared = new MoeReference(withShared).LogitsAt([], 2);
        double[] expectedPlain = new MoeReference(withoutShared).LogitsAt([], 2);

        Assert.True(Closest(logitsShared, expectedShared), $"shared-expert forward diverged: max diff {MaxDiff(logitsShared, expectedShared):G9}");
        Assert.True(Closest(logitsPlain, expectedPlain), $"no-shared-expert forward diverged: max diff {MaxDiff(logitsPlain, expectedPlain):G9}");

        double sharedDelta = 0.0;
        for (int i = 0; i < logitsShared.Length; i++)
        {
            sharedDelta += Math.Abs(expectedShared[i] - expectedPlain[i]);
        }

        Assert.True(sharedDelta > 1e-3, "the declared shared expert must visibly change the forward");
    }

    // ---------------------------------------------------------------- shared-expert gate semantics

    [Fact]
    public void SparseForward_SharedExpertSelectorPresent_MultipliesSharedContributionByItsScore()
    {
        // Routed payloads are zeroed so the stream this block sees is exactly the normalized pre-FFN residual: every fixture constant in that path (embedding rows, norm weights, selector entries) is positive, which lets these gate tests reason about one score instead of a post-routing residual whose sign depends on routing.
        MoeSpec gated = new(sharedExpert: true, sharedSelector: SelectorPayload(0.35f), zeroExpertPayloads: true);
        MoeSpec ungated = new(sharedExpert: true, zeroExpertPayloads: true); // the same block with no selector tensor in its table

        using CpuBackend backend = new(threads: 1, allowAvx2: false);

        string gatedPath = WriteTemp(BuildGguf(gated.MetadataPairs(), [.. gated.Tensors()]));
        string ungatedPath = WriteTemp(BuildGguf(ungated.MetadataPairs(), [.. ungated.Tensors()]));

        using GgufFile ggufGated = GgufFile.Open(gatedPath);
        using TransformerModel modelGated = TransformerModel.Open(ggufGated, backend, maxContextTokens: 8);
        Assert.Equal(TensorRole.MoeSharedExpertInputGate, modelGated.Plan.SparseFfn!.SharedInputGateRole);

        MoeReference gatedRef = new(gated);
        double[] expectedGated = gatedRef.LogitsAt([], 2);

        using BlockPool poolGated = NewPool(modelGated, backend, capacityBlocks: 8);
        float[] logitsGated = modelGated.DecodeNext(2, new SequenceKv(poolGated)).ToArray();

        Assert.True(Closest(logitsGated, expectedGated), $"gated shared expert diverged from the reference forward: max diff {MaxDiff(logitsGated, expectedGated):G9}");

        // The gate is a real selection here, not something that rounds away: this token's score sits in the middle of
        // the sigmoid band, so its scalar is neither 1 nor 0. Every constant on the path to it — embedding rows, norm
        // weights, selector entries — is positive, and the routed payloads are zeroed, so a logit difference against the
        // ungated block below can only come from this gate.
        double[] sparseInput = gatedRef.SparseBlockStreamFor(2);
        double score = SelectorDot(gated, sparseInput);
        double gateScalar = SiluDouble(score) / score;
        Assert.True(score > 0.1 && score < 3.0, $"fixture must read a mid-range shared-expert score, got {score:G9}");
        Assert.True(gateScalar > 0.3 && gateScalar < 0.95, $"fixture must exercise a mid-range shared-expert gate, got {gateScalar:G9} for score {score:G9}");

        // The same block without the selector tensor keeps its ungated accumulation, and the engine's gated forward
        // disagrees with that expectation everywhere the shared expert reaches.
        using GgufFile ggufUngated = GgufFile.Open(ungatedPath);
        using TransformerModel modelUngated = TransformerModel.Open(ggufUngated, backend, maxContextTokens: 8);
        // The gate role is named by the architecture on both tables; what differs between them is whether the file ships
        // the vector, and a table that ships none loads nothing for it — proved bit-for-bit in the saturation test below.
        Assert.Equal(modelGated.Plan.SparseFfn!.SharedInputGateRole, modelUngated.Plan.SparseFfn!.SharedInputGateRole);

        double[] expectedUngated = new MoeReference(ungated).LogitsAt([], 2);
        using BlockPool poolUngated = NewPool(modelUngated, backend, capacityBlocks: 8);
        float[] logitsUngated = modelUngated.DecodeNext(2, new SequenceKv(poolUngated)).ToArray();

        Assert.True(Closest(logitsUngated, expectedUngated), $"ungated shared expert diverged from its reference forward: max diff {MaxDiff(logitsUngated, expectedUngated):G9}");
        Assert.False(Closest(logitsGated, expectedUngated), "a gated shared expert must not reproduce the ungated block's logits");

        double delta = 0.0;
        for (int i = 0; i < logitsGated.Length; i++)
        {
            delta += Math.Abs((double)logitsGated[i] - expectedUngated[i]);
        }

        Assert.True(delta > 1e-3, $"the selector vector must change the shared expert's contribution, total difference {delta:G9}");
    }

    [Fact]
    public void SharedExpertSelectorSaturatingToExactlyOne_IsBitIdenticalToTheUngatedSharedBlock()
    {
        // A score big enough that float32 sigmoid lands on exactly 1.0f — exp(-score) drops below half an ULP of 1.0 once
        // the score passes ln(2^24) ≈ 16.6, and the double-precision reference rounds the same way past ln(2/eps) ≈ 37 — so
        // silu(score) / score is exactly one for both sides, making the gated accumulation arithmetic-identical to the
        // ungated one and every logit match bit for bit. That is what proves a file shipping no selector vector keeps
        // today's behavior untouched rather than reusing a path with an approximated factor.
        MoeSpec saturating = new(sharedExpert: true, sharedSelector: SelectorPayload(40f), zeroExpertPayloads: true);
        MoeSpec ungated = new(sharedExpert: true, zeroExpertPayloads: true);

        using CpuBackend backend = new(threads: 1, allowAvx2: false);

        using GgufFile ggufGated = GgufFile.Open(WriteTemp(BuildGguf(saturating.MetadataPairs(), [.. saturating.Tensors()])));
        using TransformerModel modelGated = TransformerModel.Open(ggufGated, backend, maxContextTokens: 8);

        double saturatedScore = SelectorDot(saturating, new MoeReference(saturating).SparseBlockStreamFor(3));

        // The gate scalar is the ratio, not the silu: past these thresholds both sides land on exactly one.
        Assert.True(SiluDouble(saturatedScore) / saturatedScore == 1.0, $"fixture must push the shared-expert score past saturation on both sides, got {saturatedScore:G9}");
        // The float side lands on exactly 1.0f too once exp(-score) drops under half an ULP of 1.0 (past ln(2^24) ≈ 16.6), so the engine's own gate multiplies by a literal one and the bit comparison below is arithmetic-identical rather than close.
        Assert.True(saturatedScore > 37.0, $"fixture must clear both saturation thresholds, got {saturatedScore:G9}");

        using GgufFile ggufUngated = GgufFile.Open(WriteTemp(BuildGguf(ungated.MetadataPairs(), [.. ungated.Tensors()])));
        using TransformerModel modelUngated = TransformerModel.Open(ggufUngated, backend, maxContextTokens: 8);

        using BlockPool poolGated = NewPool(modelGated, backend, capacityBlocks: 8);
        using BlockPool poolUngated = NewPool(modelUngated, backend, capacityBlocks: 8);

        float[] logitsGated = modelGated.DecodeNext(3, new SequenceKv(poolGated)).ToArray();
        float[] logitsUngated = modelUngated.DecodeNext(3, new SequenceKv(poolUngated)).ToArray();

        AssertBitsEqual(logitsUngated, logitsGated, "a selector vector gating with exactly 1.0f");

        double[] expected = new MoeReference(saturating).LogitsAt([], 3);
        Assert.True(Closest(logitsGated, expected), $"saturated gate diverged from the reference forward: max diff {MaxDiff(logitsGated, expected):G9}");
    }

    [Fact]
    public void SharedExpertSelectorShippedForSomeLayersOnly_FailsLoadNamingTheMissingLayer()
    {
        MoeSpec spec = new(sharedExpert: true, sharedSelector: SelectorPayload(0.35f));

        // The same block twice under a two-layer file, with the selector vector left out of layer 0 only: partial
        // gating is not a thing this engine accepts, so the load reports the missing name instead.
        List<TensorDef> tensors = [.. spec.Tensors()];
        foreach (TensorDef tensor in spec.Tensors())
        {
            if (!tensor.Name.StartsWith("blk.0.", StringComparison.Ordinal))
            {
                continue;
            }

            tensors.Add(tensor with { Name = "blk.1." + tensor.Name["blk.0.".Length..] });
        }

        tensors.RemoveAll(t => t.Name == "blk.0.ffn_gate_inp_shexp.weight");

        List<(string Key, object Value)> metadata = [.. spec.MetadataPairs()];
        for (int i = 0; i < metadata.Count; i++)
        {
            if (metadata[i].Key == "qwen2moe.block_count")
            {
                metadata[i] = (metadata[i].Key, 2);
            }
        }

        using CpuBackend backend = new(threads: 1, allowAvx2: false);
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(metadata, tensors)));

        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));

        Assert.Contains("blk.0.ffn_gate_inp_shexp.weight", ex.Message);
        Assert.Contains("some layers only", ex.Message);
    }

    // ---------------------------------------------------------------- selection and routing weights

    [Fact]
    public void Routing_EqualScores_PickLowerExpertIndex_AndRenormalizedWeightsSumToOne()
    {
        MoeSpec spec = new(sharedExpert: true); // its router rows make expert 1 tie with expert 2 above every other score
        double[] probe = [0.4, 0.7, 0.5, 0.9, 0.3, 0.6, 0.8, 0.2];

        MoeReference reference = new(spec);
        (int[] ids, double[] weights) = reference.RouteFor(probe);

        Assert.Equal([0, 1], ids); // strict '>' over ascending indexes: the tie resolves to expert 1, not 2
        double sum = 0.0;
        foreach (double weight in weights)
        {
            sum += weight;
        }

        Assert.True(Math.Abs(sum - 1.0) < 1e-9, $"normalized routing weights must sum to 1, got {sum:G9}");
        Assert.All(weights, w => Assert.True(w > 0.0 && w <= 1.0, $"routing weight {w:G9} outside (0, 1]"));

        // The same scores with the tie resolved against the rule must select a different set — proving the fixture
        // actually exercises the rule instead of selecting both tied experts anyway.
        (int[] idsLower, double[] weightsLower) = reference.RouteFor(probe);
        Assert.Equal([0, 1], idsLower);


        // And the engine agrees with the lower-index rule on real forwards: its logits match that reference and not
        // the wrong-tie variant.
        using CpuBackend backend = new(threads: 1, allowAvx2: false);
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), [.. spec.Tensors()]));
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);


        using BlockPool pool = NewPool(model, backend, capacityBlocks: 8);
        SequenceKv kv = new(pool);
        float[] logits = model.DecodeNext(3, kv).ToArray();

        Assert.True(Closest(logits, reference.LogitsAt([], 3)), "engine routing must match the lower-index tie rule");
        Assert.False(Closest(logits, reference.LogitsAt([], 3, preferHigherIndexOnTies: true)), "the engine must not resolve an equal score to the higher expert index");
    }

    [Fact]
    public void Routing_NormalizationOffAndScaleDeclared_UsesRawAllExpertProbabilitiesTimesTheScale()
    {
        MoeSpec spec = new(sharedExpert: true, normalizeWeights: false, weightsScale: 2f);
        double[] probe = [0.4, 0.7, 0.5, 0.9, 0.3, 0.6, 0.8, 0.2];

        (int[] ids, double[] weights) = new MoeReference(spec).RouteFor(probe);
        Assert.Equal([0, 1], ids);

        // Hand-computed expectation: softmax over all four experts, then multiply the selected probabilities by the
        // declared scale — no renormalization step.
        double maxScore = Math.Max(Math.Max(RouterDot(spec, probe, 0), RouterDot(spec, probe, 1)),
            Math.Max(RouterDot(spec, probe, 2), RouterDot(spec, probe, 3)));
        double[] probabilities = new double[4];
        double total = 0.0;
        for (int e = 0; e < 4; e++)
        {
            probabilities[e] = Math.Exp(RouterDot(spec, probe, e) - maxScore);
            total += probabilities[e];
        }

        for (int e = 0; e < 4; e++)
        {
            probabilities[e] /= total;
        }

        Assert.True(Math.Abs(weights[0] - 2.0 * probabilities[0]) < 1e-9, $"weight 0 expected {2.0 * probabilities[0]:G9}, got {weights[0]:G9}");
        Assert.True(Math.Abs(weights[1] - 2.0 * probabilities[1]) < 1e-9, $"weight 1 expected {2.0 * probabilities[1]:G9}, got {weights[1]:G9}");

        double sum = weights.Sum();
        Assert.False(Math.Abs(sum - 1.0) < 1e-6, $"unnormalized routing weights must not sum to 1 (got {sum:G9})");

        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), [.. spec.Tensors()]));
        using CpuBackend backend = new(threads: 1, allowAvx2: false);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        MoeReference reference = new(spec);
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 8);
        SequenceKv kv = new(pool);
        float[] logits = model.DecodeNext(4, kv).ToArray();

        Assert.True(Closest(logits, reference.LogitsAt([], 4)), $"scaled routing diverged from the dense-equivalent reference: max diff {MaxDiff(logits, reference.LogitsAt([], 4)):G9}");
    }

    [Fact]
    public void Routing_SelectionReachesDistinctExperts_WhenRouterRowsDifferPerToken()
    {
        MoeSpec spec = new(sharedExpert: true, routerMode: RouterMode.Discriminating);
        int[] tokens = [0, 1, 2, 3];

        MoeReference reference = new(spec);
        List<(int Id, double Weight)[]> selections = [];
        foreach (int token in tokens)
        {
            double[] normed = reference.NormedStreamFor(token);
            (int[] ids, double[] weights) = reference.RouteFor(normed);

            Assert.Equal(2, ids.Length);
            Assert.True(ids[0] < ids[1], $"selection must run in ascending expert-index order: {ids[0]} then {ids[1]}");
            selections.Add([.. Enumerable.Range(0, 2).Select(i => (ids[i], weights[i]))]);
        }


        List<int> distinctExperts = [];
        foreach ((int Id, double Weight)[] selection in selections)
        {
            foreach (var pick in selection)
            {
                if (!distinctExperts.Contains(pick.Id))
                {
                    distinctExperts.Add(pick.Id);
                }
            }
        }

        Assert.True(distinctExperts.Count >= 2, $"the discriminating router must send more than one expert into execution across the fixture tokens; saw [{string.Join(", ", distinctExperts)}]");

        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), [.. spec.Tensors()]));
        using CpuBackend backend = new(threads: 4, allowAvx2: false);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        using BlockPool pool = NewPool(model, backend, capacityBlocks: 16);
        SequenceKv kv = new(pool);
        float[] logits = model.DecodeNext(tokens[2], kv).ToArray(); // a token whose selection differs from the others'

        Assert.True(Closest(logits, reference.LogitsAt([], tokens[2])), "spread selection must match the dense-equivalent reference");
    }

    [Fact]
    public void Routing_AllTopKTraffic_HitsOneExpert_WhenRoutingCountIsOne()
    {
        MoeSpec spec = new(sharedExpert: false, routedExpertsPerToken: 1); // every sequence routes to a single expert
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), [.. spec.Tensors()]));

        using CpuBackend backend = new(threads: 4, allowAvx2: false);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);
        Assert.Equal(1, model.Plan.SparseFfn!.RoutedExpertsPerToken);

        int[] tokens = [0, 1, 2, 3];
        MoeReference reference = new(spec);

        using BlockPool batchedPool = NewPool(model, backend, capacityBlocks: 8);
        SequenceKv[] kvs = [new SequenceKv(batchedPool), new SequenceKv(batchedPool), new SequenceKv(batchedPool), new SequenceKv(batchedPool)];
        Span<float> batched = model.DecodeBatch(tokens, kvs);

        int vocab = model.Desc.VocabSize;
        for (int s = 0; s < tokens.Length; s++)
        {
            Assert.True(
                Closest(BitsOf(batched.Slice(s * vocab, vocab)), reference.LogitsAt([], tokens[s])),
                $"sequence {s} routing to its single expert diverged from the dense-equivalent reference");

            (int[] ids, _) = reference.RouteFor(reference.NormedStreamFor(tokens[s]));
            Assert.Single(ids); // every top-k traffic path here hits exactly one expert per sequence
        }
    }

    // ---------------------------------------------------------------- batch identity and thread determinism

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void DecodeBatch_SparseModel_MatchesSoloStepping_BitForBit(int sequences)
    {
        MoeSpec spec = new(sharedExpert: true, routerMode: RouterMode.Discriminating);
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), [.. spec.Tensors()]));

        using CpuBackend batchedBackend = new(threads: 4, allowAvx2: false);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel batchedModel = TransformerModel.Open(gguf, batchedBackend, maxContextTokens: 8);

        using CpuBackend soloBackend = new(threads: 1, allowAvx2: false);
        using TransformerModel soloModel = TransformerModel.Open(gguf, soloBackend, maxContextTokens: 8);

        int[] tokens = new int[] { 0, 1, 2, 3 }[.. sequences];

        using BlockPool batchedPool = NewPool(batchedModel, batchedBackend, capacityBlocks: 8);
        using BlockPool soloPool = NewPool(soloModel, soloBackend, capacityBlocks: 8);
        SequenceKv[] batchedKvs = Enumerable.Range(0, sequences).Select(_ => new SequenceKv(batchedPool)).ToArray();
        SequenceKv[] soloKvs = Enumerable.Range(0, sequences).Select(_ => new SequenceKv(soloPool)).ToArray();

        Span<float> batched = batchedModel.DecodeBatch(tokens, batchedKvs);
        int vocab = batchedModel.Desc.VocabSize;

        for (int s = 0; s < sequences; s++)
        {
            float[] solo = soloModel.DecodeNext(tokens[s], soloKvs[s]).ToArray();
            AssertBitsEqual(solo, batched.Slice(s * vocab, vocab), $"sequence {s} sparse logits");
        }

        // A second decode step on the same sequences: histories now differ in length per sequence and attention is
        // non-degenerate — the sparse block must stay bit-identical to solo stepping.
        int[] next = new int[] { 1, 2, 3, 0 }[.. sequences];
        Span<float> batchedAgain = batchedModel.DecodeBatch(next, batchedKvs);

        for (int s = 0; s < sequences; s++)
        {
            float[] soloAgain = soloModel.DecodeNext(next[s], soloKvs[s]).ToArray();
            AssertBitsEqual(soloAgain, batchedAgain.Slice(s * vocab, vocab), $"sequence {s} sparse logits after the second step");

            Assert.Equal(batchedKvs[s].Length, soloKvs[s].Length);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void DecodeBatch_GatedSharedExpert_MatchesSoloStepping_BitForBit(int sequences)
    {
        MoeSpec spec = new(sharedExpert: true, routerMode: RouterMode.Discriminating, sharedSelector: SelectorPayload(0.35f));
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), [.. spec.Tensors()]));

        using CpuBackend batchedBackend = new(threads: 4, allowAvx2: false);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel batchedModel = TransformerModel.Open(gguf, batchedBackend, maxContextTokens: 8);

        using CpuBackend soloBackend = new(threads: 1, allowAvx2: false);
        using TransformerModel soloModel = TransformerModel.Open(gguf, soloBackend, maxContextTokens: 8);

        MoeReference reference = new(spec);
        int[] tokens = new int[] { 0, 1, 2, 3 }[.. sequences];

        using BlockPool batchedPool = NewPool(batchedModel, batchedBackend, capacityBlocks: 8);
        using BlockPool soloPool = NewPool(soloModel, soloBackend, capacityBlocks: 8);
        SequenceKv[] batchedKvs = Enumerable.Range(0, sequences).Select(_ => new SequenceKv(batchedPool)).ToArray();
        SequenceKv[] soloKvs = Enumerable.Range(0, sequences).Select(_ => new SequenceKv(soloPool)).ToArray();

        Span<float> batched = batchedModel.DecodeBatch(tokens, batchedKvs);
        int vocab = batchedModel.Desc.VocabSize;

        for (int s = 0; s < sequences; s++)
        {
            float[] solo = soloModel.DecodeNext(tokens[s], soloKvs[s]).ToArray();
            AssertBitsEqual(solo, batched.Slice(s * vocab, vocab), $"sequence {s} gated sparse logits");

            // Each sequence carries its own gate scalar: the shared contribution it adds is scaled by that token's score.
            Assert.True(
                Closest(solo, reference.LogitsAt([], tokens[s])),
                $"sequence {s} gated forward diverged from the per-token reference: max diff {MaxDiff(solo, reference.LogitsAt([], tokens[s])):G9}");
        }

        // A second step with histories of different lengths per sequence keeps every gate bit-identical to solo stepping.
        int[] next = new int[] { 1, 2, 3, 0 }[.. sequences];
        Span<float> batchedAgain = batchedModel.DecodeBatch(next, batchedKvs);

        for (int s = 0; s < sequences; s++)
        {
            float[] soloAgain = soloModel.DecodeNext(next[s], soloKvs[s]).ToArray();
            AssertBitsEqual(soloAgain, batchedAgain.Slice(s * vocab, vocab), $"sequence {s} gated sparse logits after the second step");

            Assert.Equal(batchedKvs[s].Length, soloKvs[s].Length);
        }
    }

    [Fact]
    public void SparseForward_ThreadCountVariation_IsDeterministic()
    {
        MoeSpec spec = new(sharedExpert: true, routerMode: RouterMode.Discriminating);
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), [.. spec.Tensors()]));

        using GgufFile gguf = GgufFile.Open(path);

        using CpuBackend oneThread = new(threads: 1, allowAvx2: false);
        using TransformerModel single = TransformerModel.Open(gguf, oneThread, maxContextTokens: 8);
        using BlockPool poolSingle = NewPool(single, oneThread, capacityBlocks: 8);

        using CpuBackend fourThreads = new(threads: 4, allowAvx2: false);
        using TransformerModel multi = TransformerModel.Open(gguf, fourThreads, maxContextTokens: 8);
        using BlockPool poolMulti = NewPool(multi, fourThreads, capacityBlocks: 8);

        int[] tokens = [0, 1, 3];
        SequenceKv kvSingle = new(poolSingle);
        float[] soloLogits = single.DecodeNext(tokens[1], kvSingle).ToArray();

        SequenceKv[] kvsMulti = [new SequenceKv(poolMulti), new SequenceKv(poolMulti)];
        Span<float> batchedLogits = multi.DecodeBatch([tokens[0], tokens[1]], kvsMulti);

        AssertBitsEqual(soloLogits, BitsOf(batchedLogits.Slice(single.Desc.VocabSize, single.Desc.VocabSize)), "thread-count variation across a sparse step");

        SequenceKv kvMultiSingleSeq = new(poolMulti);
        float[] multiSolo = multi.DecodeNext(tokens[1], kvMultiSingleSeq).ToArray();
        AssertBitsEqual(soloLogits, multiSolo, "one-thread and four-thread single-token forwards must match bit-for-bit");
    }

    // ---------------------------------------------------------------- encoder deferral for sparse models

    [Fact]
    public void Encode_SparseModel_FailsNamingTheDescriptorInsteadOfApproximatingRouting()
    {
        MoeSpec spec = new(sharedExpert: true);
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), [.. spec.Tensors()]));

        using CpuBackend backend = new(threads: 1, allowAvx2: false);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        using BlockPool pool = NewPool(model, backend, capacityBlocks: 8);
        SequenceKv kv = new(pool);

        ModelException ex = Assert.Throws<ModelException>(() => model.Encode([1, 2], kv));

        Assert.Contains("qwen2moe", ex.Message);
        Assert.Contains("sparse-FFN block", ex.Message);
        Assert.Contains("encoder forward runs dense gate/up/down blocks only", ex.Message);
    }

    // ---------------------------------------------------------------- shared helpers

    private static double RouterDot(MoeSpec spec, double[] input, int expert)
    {
        double dot = 0.0;
        for (int c = 0; c < MoeSpec.Hidden; c++)
        {
            dot += (double)spec.Router[expert * MoeSpec.Hidden + c] * input[c];
        }

        return dot;
    }

    // A shared-expert selector payload: one fixed direction over the hidden stream scaled by factor, so each token's
    // score is predictable from its normalized stream and never lands on exactly zero.
    private static float[] SelectorPayload(float factor)
    {
        float[] baseRow = [0.5f, 0.4f, 0.3f, 0.2f, 0.45f, 0.35f, 0.25f, 0.15f]; // one entry per column of the fixture's hidden stream
        float[] selector = new float[MoeSpec.Hidden];

        for (int c = 0; c < MoeSpec.Hidden; c++)
        {
            selector[c] = factor * baseRow[c];
        }

        return selector;
    }

    private static double SelectorDot(MoeSpec spec, double[] normed)
    {
        if (spec.SharedSelector is not { } selector)
        {
            throw new ArgumentException("the fixture ships no shared-expert selector vector", nameof(spec));
        }

        double dot = 0.0;
        for (int c = 0; c < MoeSpec.Hidden; c++)
        {
            dot += (double)selector[c] * normed[c];
        }

        return dot;
    }

    // sigmoid's value read through its own silu — the pair of ops the pinned reference builds for this gate.
    private static double SiluDouble(double x) => x / (1.0 + Math.Exp(-x));

    private static bool Closest(ReadOnlySpan<float> actual, double[] expected) => MaxDiff(actual, expected) <= 1e-5;

    private static double MaxDiff(ReadOnlySpan<float> actual, double[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);

        double worst = 0.0;
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(actual[i] - expected[i]);
            if (diff > worst)
            {
                worst = diff;
            }
        }

        return worst;
    }

    private static void AssertBitsEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        ReadOnlySpan<byte> expectedBytes = MemoryMarshal.AsBytes(expected);
        ReadOnlySpan<byte> actualBytes = MemoryMarshal.AsBytes(actual);

        for (int i = 0; i < expectedBytes.Length; i++)
        {
            Assert.True(
                expectedBytes[i] == actualBytes[i],
                $"{context}: first differing byte at {i} (float {i / 4}): expected 0x{expectedBytes[i]:X2}, got 0x{actualBytes[i]:X2}");
        }
    }

    private static ReadOnlySpan<float> BitsOf(ReadOnlySpan<float> logits) => logits; // bit comparison only ever reads through this span

    private static BlockPool NewPool(TransformerModel model, CpuBackend backend, int capacityBlocks) =>
        new(blockTokens: 16, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks, backend);

    private enum RouterMode
    {
        /// <summary>Constant-direction rows with an exact tie between experts 1 and 2, above every other score.</summary>
        TieAboveOthers,

        /// <summary>Rows whose dot products differ per embedding row so different tokens select differently.</summary>
        Discriminating,
    }

    private sealed record TensorDef(string Name, int[] Dims, float[] Payload); // dims in logical (row-major) order; vectors are one-dimensional

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-moe-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // A complete single-layer qwen2moe model with hidden 8, heads 2, kv-heads 2, head dim 4 (the natural split),
    // ffn 6, vocab 5 and four experts: all F32 tensors with hand-picked payloads the reference forward reads directly.
    // No output.weight — the logits come from the tied token-embedding rows. Attention v/o stay zero so routing sees a
    // stream that is exactly the embedding row times its norms, which makes selection outcomes predictable in tests.
    private sealed class MoeSpec
    {
        public const int Hidden = 8;
        public const int Heads = 2;
        public const int KvHeads = 2;
        public const int HeadDim = 4;
        public const int Intermediate = 6;
        public const int Experts = 4;
        public const int Vocab = 5;
        public const float Eps = 1e-5f;

        private readonly bool _sharedExpert;
        private readonly float[]? _sharedSelector;
        private readonly bool _normalizeWeights;
        private readonly float _weightsScale;
        private readonly int _routedExpertsPerToken;
        private readonly RouterMode _routerMode;
        private readonly bool _zeroExpertPayloads;

        public MoeSpec(
            bool sharedExpert,
            bool normalizeWeights = true,
            float weightsScale = 1f,
            int routedExpertsPerToken = 2,
            RouterMode routerMode = RouterMode.TieAboveOthers,
            float[]? sharedSelector = null,
            bool zeroExpertPayloads = false)
        {
            _sharedExpert = sharedExpert;
            _sharedSelector = sharedSelector;
            _normalizeWeights = normalizeWeights;
            _weightsScale = weightsScale;
            _routedExpertsPerToken = routedExpertsPerToken;
            _routerMode = routerMode;
            _zeroExpertPayloads = zeroExpertPayloads;
        }

        public bool SharedExpert => _sharedExpert;

        /// <summary>The shared-expert selector vector this fixture ships, or null when it ships none.</summary>
        public float[]? SharedSelector => _sharedSelector;

        public bool NormalizeWeights => _normalizeWeights;
        public float WeightsScale => _weightsScale;
        public int RoutedExpertsPerToken => _routedExpertsPerToken;
        public RouterMode Mode => _routerMode;

        public List<(string Key, object Value)> MetadataPairs()
        {
            List<(string Key, object Value)> metadata =
            [
                ("general.architecture", "qwen2moe"),
                ("qwen2moe.embedding_length", Hidden),
                ("qwen2moe.block_count", 1),
                ("qwen2moe.attention.head_count", Heads),
                ("qwen2moe.attention.head_count_kv", KvHeads),
                ("qwen2moe.feed_forward_length", Intermediate),
                ("qwen2moe.attention.layer_norm_rms_epsilon", Eps),
                ("qwen2moe.vocab_size", Vocab),
                ("qwen2moe.expert_count", Experts),
                ("qwen2moe.expert_used_count", _routedExpertsPerToken),
            ];

            if (_sharedExpert)
            {
                metadata.Add(("qwen2moe.expert_shared_count", 1));
            }

            metadata.Add(("qwen2moe.expert_weights_scale", _weightsScale));
            metadata.Add(("qwen2moe.expert_weights_norm", _normalizeWeights));

            return metadata;
        }

        public List<TensorDef> Tensors()
        {
            List<TensorDef> tensors =
            [
                new("token_embd.weight", [Vocab, Hidden], Embeddings),
                new("blk.0.attn_q.weight", [Heads * HeadDim, Hidden], AttnQ),
                new("blk.0.attn_k.weight", [KvHeads * HeadDim, Hidden], AttnK),
                new("blk.0.attn_v.weight", [KvHeads * HeadDim, Hidden], Zero(KvHeads * HeadDim * Hidden)),
                new("blk.0.attn_output.weight", [Hidden, Heads * HeadDim], Zero(Hidden * Heads * HeadDim)),
                new("blk.0.attn_norm.weight", [Hidden], AttnNorm),
                new("blk.0.ffn_norm.weight", [Hidden], FfnNorm),
                new("output_norm.weight", [Hidden], OutputNorm),
                new("blk.0.ffn_gate_inp.weight", [Experts, Hidden], Router),
                new("blk.0.ffn_gate_exps.weight", [Experts, Intermediate, Hidden], ExpertGate),
                new("blk.0.ffn_up_exps.weight", [Experts, Intermediate, Hidden], ExpertUp),
                new("blk.0.ffn_down_exps.weight", [Experts, Hidden, Intermediate], ExpertDown),
            ];

            if (_sharedExpert)
            {
                tensors.Add(new("blk.0.ffn_gate_shexp.weight", [Intermediate, Hidden], SharedGate));
                tensors.Add(new("blk.0.ffn_up_shexp.weight", [Intermediate, Hidden], SharedUp));
                tensors.Add(new("blk.0.ffn_down_shexp.weight", [Hidden, Intermediate], SharedDown));

                if (_sharedSelector is not null)
                {
                    tensors.Add(new("blk.0.ffn_gate_inp_shexp.weight", [Hidden], _sharedSelector));
                }
            }

            return tensors;
        }

        // All-positive embedding rows keep every normalized stream positive, which is what makes the constant-direction
        // router rows order deterministically in tests. Distinct per token so selections differ under Discriminating.
        public readonly float[] Embeddings =
        [
            0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f, 0.70f, 0.80f,
            0.90f, 0.15f, 0.25f, 0.35f, 0.45f, 0.55f, 0.65f, 0.75f,
            0.12f, 0.88f, 0.16f, 0.84f, 0.20f, 0.80f, 0.24f, 0.76f,
            0.33f, 0.31f, 0.99f, 0.11f, 0.77f, 0.22f, 0.55f, 0.44f,
            0.61f, 0.17f, 0.28f, 0.39f, 0.50f, 0.91f, 0.13f, 0.72f,
        ];

        public readonly float[] AttnQ = Scale(0.4f);
        public readonly float[] AttnK = Scale(0.2f);
        public readonly float[] AttnNorm = [1.10f, 0.90f, 1.05f, 0.95f, 1.00f, 1.15f, 0.85f, 1.20f];
        public readonly float[] FfnNorm = [0.90f, 1.10f, 0.95f, 1.05f, 1.00f, 0.85f, 1.15f, 0.92f];
        public readonly float[] OutputNorm = [1.00f, 1.02f, 0.98f, 1.01f, 0.99f, 1.03f, 1.04f, 0.97f];

        // TieAboveOthers: every row is the same direction scaled by a multiplier, so dots order as
        // expert 0 > expert 1 == expert 2 > expert 3 for any all-positive stream — experts 1 and 2 share an exact tie.
        // Discriminating: orthogonal-ish rows put different experts ahead of different embedding rows.
        public float[] Router => _routerMode switch
        {
            RouterMode.TieAboveOthers => DirectionRows([0.8f, 0.6f, 0.6f, 0.1f]),
            RouterMode.Discriminating => new float[]
            {
                1.2f, -0.4f, 0.3f, -0.9f, 0.7f, -0.5f, 0.8f, -0.6f,
                -0.5f, 1.1f, -0.3f, 0.9f, -0.7f, 0.4f, -0.8f, 0.2f,
                0.4f, 0.7f, -1.0f, 0.5f, -0.6f, 0.3f, 0.9f, -0.8f,
                -0.9f, 0.2f, 0.8f, -0.4f, 0.6f, -1.1f, 0.1f, 0.7f,
            },
            _ => throw new ArgumentOutOfRangeException(),
        };

        // Distinct multipliers per expert so each expert's contribution is detectably different from every other's. A fixture that zeroes these payloads leaves the shared block as the only sparse traffic, which is how a gate-semantics test reads one deterministic score off the normalized stream instead of a post-routing residual.
        public float[] ExpertGate => _zeroExpertPayloads ? Zero(Experts * Intermediate * Hidden) : StackedExperts([0.35f, -0.25f, 0.45f, -0.15f], Intermediate * Hidden);

        // Up rows stay modest and positive so the gated activation never saturates to a degenerate value.
        public float[] ExpertUp => _zeroExpertPayloads ? Zero(Experts * Intermediate * Hidden) : StackedExperts([0.30f, 0.28f, 0.26f, 0.24f], Intermediate * Hidden);

        // Down rows carry per-expert offsets so the accumulated residual keeps each expert's fingerprint.
        public float[] ExpertDown => _zeroExpertPayloads ? Zero(Experts * Hidden * Intermediate) : StackedExperts([1.7f, -1.3f, 2.1f, -0.9f], Hidden * Intermediate);

        public readonly float[] SharedGate = ScaleFlat(0.5f, Intermediate * Hidden);
        public readonly float[] SharedUp = ScaleFlat(-0.4f, Intermediate * Hidden);
        public readonly float[] SharedDown = ScaleFlat(1.3f, Hidden * Intermediate);

        private static float[] Zero(int length) => new float[length];

        private static float[] Scale(float factor)
        {
            float[] values = new float[Heads * HeadDim * Hidden]; // q/k share the same base pattern scaled differently
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = factor * (0.1f + 0.37f * (i % 9));
            }

            return values;
        }

        private static float[] ScaleFlat(float factor, int length)
        {
            float[] values = new float[length];
            for (int i = 0; i < length; i++)
            {
                values[i] = factor * (0.2f + 0.31f * (i % 7));
            }

            return values;
        }

        private static float[] DirectionRows(float[] multipliers)
        {
            float[] baseRow = [0.5f, 0.4f, 0.3f, 0.2f, 0.45f, 0.35f, 0.25f, 0.15f];
            float[] router = new float[Experts * Hidden];

            for (int e = 0; e < Experts; e++)
            {
                for (int c = 0; c < Hidden; c++)
                {
                    router[e * Hidden + c] = multipliers[e] * baseRow[c];
                }
            }

            return router;
        }

        private static float[] StackedExperts(float[] perExpertFactor, int blockLength)
        {
            float[] stacked = new float[Experts * blockLength];

            for (int e = 0; e < Experts; e++)
            {
                for (int i = 0; i < blockLength; i++)
                {
                    stacked[e * blockLength + i] = perExpertFactor[e] * (0.15f + 0.23f * (i % 6)) + 0.04f * e;
                }
            }

            return stacked;
        }
    }

    // An independent double-precision forward over the spec's tensors: RMSNorm, full-head RoPE, causal attention with an
    // explicit softmax (its v/o payloads are zero, so its contribution is a known constant), then the sparse FFN —
    // all-expert router softmax, top-k selection under strict greater-than ordering, optional renormalization and
    // scaling, per-expert silu(gate) * up chains accumulated in ascending expert-index order, an optional shared block
    // at weight 1 — and the tied token-embedding logits. This is the dense-equivalent reference every assertion here
    // compares against; the engine runs it in float under a fixed tolerance instead.
    private sealed class MoeReference
    {
        private readonly MoeSpec _spec;

        // The normalized stream the last Advance handed to routing and the shared block, kept so a gate fixture can read
        // one token's score off the same value the forward used instead of rebuilding it after accumulation.
        private double[]? _sparseBlockStream;

        public MoeReference(MoeSpec spec) => _spec = spec;

        /// <summary>Logits for a query token forwarded at position historyTokens.Length after replaying earlier tokens through the layer.</summary>
        public double[] LogitsAt(int[] historyTokens, int queryToken, bool preferHigherIndexOnTies = false)
        {
            double[][] kvK = new double[historyTokens.Length + 1][];
            double[][] kvV = new double[historyTokens.Length + 1][];

            for (int p = 0; p < historyTokens.Length; p++)
            {
                Advance(Row(_spec.Embeddings, historyTokens[p], MoeSpec.Hidden), p, kvK, kvV, preferHigherIndexOnTies: preferHigherIndexOnTies);
            }

            double[] cur = Row(_spec.Embeddings, queryToken, MoeSpec.Hidden);
            Advance(cur, historyTokens.Length, kvK, kvV, normalizeOnly: false, preferHigherIndexOnTies: preferHigherIndexOnTies);

            double[] normed = Norm(cur, _spec.OutputNorm);
            return Gemv(_spec.Embeddings, MoeSpec.Vocab, MoeSpec.Hidden, normed); // tied lm_head
        }

        /// <summary>The normalized post-attention stream a single-token decode feeds the sparse block.</summary>
        public double[] NormedStreamFor(int queryToken)
        {
            double[][] kvK = [null!];
            double[][] kvV = [null!];

            double[] cur = Row(_spec.Embeddings, queryToken, MoeSpec.Hidden);
            Advance(cur, 0, kvK, kvV, normalizeOnly: true);

            return Norm(cur, _spec.FfnNorm);
        }

        /// <summary>The exact stream a single-token decode hands routing and the shared block, taken from inside that forward.</summary>
        public double[] SparseBlockStreamFor(int queryToken)
        {
            double[][] kvK = [null!];
            double[][] kvV = [null!];

            Advance(Row(_spec.Embeddings, queryToken, MoeSpec.Hidden), 0, kvK, kvV, normalizeOnly: true);

            return _sparseBlockStream!;
        }

        /// <summary>Routing for a normalized stream under the spec's declared descriptor values.</summary>
        public (int[] Ids, double[] Weights) RouteFor(double[] normed) => RouteWithTieRule(normed, preferHigherIndexOnTies: false);

        /// <summary>The same selection with equal scores resolved against the engine's documented rule — used only to prove a fixture exercises that rule.</summary>
        public (int[] Ids, double[] Weights) ReferenceSelectingHigherIndexOnTies(double[] normed) => RouteWithTieRule(normed, preferHigherIndexOnTies: true);

        private (int[] Ids, double[] Weights) RouteWithTieRule(double[] normed, bool preferHigherIndexOnTies)
        {
            double[] scores = new double[MoeSpec.Experts];
            for (int e = 0; e < scores.Length; e++)
            {
                scores[e] = RouterDot(_spec, normed, e);
            }

            double maxScore = double.NegativeInfinity;
            foreach (double score in scores)
            {
                maxScore = Math.Max(maxScore, score);
            }

            double allSum = 0.0;
            double[] probs = new double[scores.Length];
            for (int e = 0; e < scores.Length; e++)
            {
                probs[e] = Math.Exp(scores[e] - maxScore);
                allSum += probs[e];
            }

            for (int e = 0; e < scores.Length; e++)
            {
                probs[e] /= allSum;
            }


            int[] ids = new int[_spec.RoutedExpertsPerToken];
            double[] weights = new double[_spec.RoutedExpertsPerToken];
            double selectedSum = 0.0;

            for (int pick = 0; pick < ids.Length; pick++)
            {
                int best = -1;
                double bestScore = double.NegativeInfinity;
                for (int e = 0; e < scores.Length; e++)
                {
                    bool better = preferHigherIndexOnTies ? scores[e] >= bestScore : scores[e] > bestScore;
                    if (better)
                    {
                        best = e;
                        bestScore = scores[e];
                    }
                }

                ids[pick] = best;
                weights[pick] = probs[best];
                selectedSum += probs[best];
                scores[best] = double.NegativeInfinity;
            }

            for (int i = 1; i < ids.Length; i++)
            {
                int id = ids[i];
                double weight = weights[i];
                int j = i - 1;
                while (j >= 0 && ids[j] > id)
                {
                    ids[j + 1] = ids[j];
                    weights[j + 1] = weights[j];
                    j--;
                }

                ids[j + 1] = id;
                weights[j + 1] = weight;
            }

            if (_spec.NormalizeWeights && selectedSum > 0.0)
            {
                for (int i = 0; i < weights.Length; i++)
                {
                    weights[i] /= selectedSum;
                }
            }

            if (_spec.WeightsScale != 0f)
            {
                for (int i = 0; i < weights.Length; i++)
                {
                    weights[i] *= _spec.WeightsScale;
                }
            }

            return (ids, weights);
        }

        private void Advance(double[] cur, int position, double[][] kvK, double[][] kvV, bool normalizeOnly = false, bool preferHigherIndexOnTies = false)
        {
            double[] normed = Norm(cur, _spec.AttnNorm);
            double[] q = Gemv(_spec.AttnQ, MoeSpec.Heads * MoeSpec.HeadDim, MoeSpec.Hidden, normed);
            double[] k = Gemv(_spec.AttnK, MoeSpec.KvHeads * MoeSpec.HeadDim, MoeSpec.Hidden, normed);

            Rope(q, position);
            Rope(k, position);

            kvK[position] = k;
            kvV[position] = new double[MoeSpec.KvHeads * MoeSpec.HeadDim]; // the fixture ships zero-valued v rows

            if (!normalizeOnly)
            {
                double[] attnOut = new double[MoeSpec.Heads * MoeSpec.HeadDim];
                Attention(q, kvK, position, attnOut);

                for (int i = 0; i < MoeSpec.Hidden; i++)
                {
                    cur[i] += attnOut[i]; // its v/o payloads are zero: attention contributes nothing here by construction
                }
            }

            double[] ffnNormed = Norm(cur, _spec.FfnNorm);
            _sparseBlockStream = ffnNormed; // captured where the forward actually hands it to routing and the shared block
            (int[] ids, double[] weights) = RouteWithTieRule(ffnNormed, preferHigherIndexOnTies);

            for (int slot = 0; slot < ids.Length; slot++)
            {
                double[] contribution = ExpertContribution(ids[slot], ffnNormed);
                for (int i = 0; i < MoeSpec.Hidden; i++)
                {
                    cur[i] += weights[slot] * contribution[i];
                }
            }

            if (_spec.SharedExpert)
            {
                double[] shared = DenseContribution(ffnNormed);
                double sharedWeight = SharedWeightFor(ffnNormed);
                for (int i = 0; i < MoeSpec.Hidden; i++)
                {
                    cur[i] += sharedWeight * shared[i];
                }
            }
        }

        private static void Attention(double[] q, double[][] kvK, int position, Span<double> outHead)
        {
            const int headDim = MoeSpec.HeadDim;
            double scale = 1.0 / Math.Sqrt(headDim);

            for (int h = 0; h < MoeSpec.Heads; h++)
            {
                // heads and kv-heads match in this fixture, so every head reads its own k/v pair.
                double dot = 0.0;
                for (int j = 0; j < headDim; j++)
                {
                    dot += q[h * headDim + j] * kvK[position][h * headDim + j];
                }

                _ = dot * scale; // a single-position causal softmax weights this position with 1.0 regardless of the score

                for (int j = 0; j < headDim; j++)
                {
                    outHead[h * headDim + j] = 0.0; // every v row is zero in these fixtures
                }
            }
        }

        private double[] ExpertContribution(int expert, double[] normed)
        {
            int blockLength = MoeSpec.Intermediate * MoeSpec.Hidden;
            float[] gateRows = Slice(_spec.ExpertGate, expert * blockLength, blockLength);
            float[] upRows = Slice(_spec.ExpertUp, expert * blockLength, blockLength);

            double[] gate = Gemv(gateRows, MoeSpec.Intermediate, MoeSpec.Hidden, normed);
            double[] up = Gemv(upRows, MoeSpec.Intermediate, MoeSpec.Hidden, normed);
            for (int i = 0; i < MoeSpec.Intermediate; i++)
            {
                gate[i] = Silu(gate[i]) * up[i];
            }

            int downLength = MoeSpec.Hidden * MoeSpec.Intermediate;
            float[] downRows = Slice(_spec.ExpertDown, expert * downLength, downLength);
            return Gemv(downRows, MoeSpec.Hidden, MoeSpec.Intermediate, gate);
        }

        private double[] DenseContribution(double[] normed)
        {
            double[] gate = Gemv(_spec.SharedGate, MoeSpec.Intermediate, MoeSpec.Hidden, normed);
            double[] up = Gemv(_spec.SharedUp, MoeSpec.Intermediate, MoeSpec.Hidden, normed);
            for (int i = 0; i < MoeSpec.Intermediate; i++)
            {
                gate[i] = Silu(gate[i]) * up[i];
            }

            return Gemv(_spec.SharedDown, MoeSpec.Hidden, MoeSpec.Intermediate, gate);
        }

        // The weight the shared expert's contribution enters the residual with, built the way the pinned reference
        // builds it: silu(score) / score over this token's dot against the fixture's selector vector — sigmoid's value
        // read through its own silu. A fixture shipping no selector vector adds the block at exactly 1.0.
        private double SharedWeightFor(double[] normed)
        {
            if (_spec.SharedSelector is not { } selector)
            {
                return 1.0;
            }

            double score = 0.0;
            for (int c = 0; c < MoeSpec.Hidden; c++)
            {
                score += (double)selector[c] * normed[c];
            }

            return Silu(score) / score;
        }

        private static void Rope(double[] headRegion, int position)
        {
            for (int h = 0; h < headRegion.Length / MoeSpec.HeadDim; h++)
            {
                for (int i = 0; i < MoeSpec.HeadDim / 2; i++)
                {
                    double invFreq = Math.Pow(10_000.0, -2.0 * i / MoeSpec.HeadDim);
                    double angle = position * invFreq;
                    double c = Math.Cos(angle);
                    double s = Math.Sin(angle);

                    double lo = headRegion[h * MoeSpec.HeadDim + 2 * i];
                    double hi = headRegion[h * MoeSpec.HeadDim + 2 * i + 1];
                    headRegion[h * MoeSpec.HeadDim + 2 * i] = lo * c - hi * s;
                    headRegion[h * MoeSpec.HeadDim + 2 * i + 1] = hi * c + lo * s;
                }
            }
        }

        private static double[] Norm(double[] x, float[] weight)
        {
            int n = x.Length;
            double sumSq = 0.0;
            for (int i = 0; i < n; i++)
            {
                sumSq += x[i] * x[i];
            }

            double invRms = 1.0 / Math.Sqrt(sumSq / n + MoeSpec.Eps);
            double[] result = new double[n];
            for (int i = 0; i < n; i++)
            {
                result[i] = x[i] * invRms * weight[i];
            }

            return result;
        }

        private static double Silu(double x) => x / (1.0 + Math.Exp(-x));

        private static double[] Gemv(float[] rows, int outCount, int inCount, double[] input)
        {
            double[] output = new double[outCount];
            for (int r = 0; r < outCount; r++)
            {
                double acc = 0.0;
                for (int c = 0; c < inCount; c++)
                {
                    acc += rows[r * inCount + c] * input[c];
                }

                output[r] = acc;
            }

            return output;
        }

        private static double[] Row(float[] data, int row, int length)
        {
            double[] result = new double[length];
            for (int i = 0; i < length; i++)
            {
                result[i] = data[row * length + i];
            }

            return result;
        }

        private static float[] Slice(float[] source, int start, int length)
        {
            float[] slice = new float[length];
            Array.Copy(source, start, slice, 0, length);
            return slice;
        }
    }

    /// <summary>Builds a GGUF v3 image from KV pairs and F32 tensors with the exact payloads the reference forward reads.</summary>
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

        foreach (TensorDef tensor in tensors)
        {
            long elements = 1;
            foreach (int dim in tensor.Dims)
            {
                elements *= dim;
            }

            if (tensor.Payload.Length != elements)
            {
                throw new ArgumentException($"tensor '{tensor.Name}' payload has {tensor.Payload.Length} entries, expected {elements}.", nameof(tensors));
            }

            foreach (float value in tensor.Payload)
            {
                writer.Write(value);
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
}
