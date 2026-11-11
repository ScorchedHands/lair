using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Windowed-KV correctness on the already-pinned Gemma-2 fixture, driven with token ids the way MVP3.4 drove its
/// forward graph: the real file's plan resolves a 4096-wide window across 26 layers that alternates local and full —
/// even-indexed layers read inside the window, odd-indexed ones attend the whole range; at exactly that declared width
/// a causal selection excludes an entire physical block while an unbounded layer selects every one of them, proven
/// against loudly-valued KV bytes rather than inferred from loop bounds; and prompt-cache reuse on this model —
/// publish, adopt, continue — lands bit-for-bit on the same prompt prefilled cold. Cases skip when the fixture file is
/// absent, like the other real-file suites. The pinned window (4096) cannot exclude a block at the short sequence
/// lengths its forwards run cheaply, so exclusion is proven over a pool carrying this plan's own KV geometry and
/// declared width, while the fixture's own forwards gate cache-reuse identity.
/// </summary>
[Collection("model-loaded")] // shares one serialized lane with the other real-model suites: two engines mapping 1.6 GB at once is not what this box should be doing
[Trait("Category", "Layers")]
public sealed class Gemma2WindowedKvTests
{
    private const string FixtureRelativePath = "models/gemma-2-2b-it.Q4_K_M.gguf";

    // Token ids supplied the way MVP3.4 drove this file — chosen against the engine's own vocabulary bounds rather than
    // read out of a recorded oracle, because the pinned fixture's SentencePiece vocabulary (and with it its parity
    // fixture) is what MVP3.11 still has to build. Every assertion below compares LAIR forwards over identical ids;
    // none of them depends on what those ids mean textually.
    private static readonly int[] SharedPrefix = [2, 108, 1304, 144];
    private static readonly int[] WarmTail = [573, 6921];

    // ---------------------------------------------------------------- the pinned plan's per-layer windows

    [Fact]
    public void ModelPlan_RealGemma2Fixture_ResolvesDeclaredWindowAndPerLayerInterleave()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(FixtureRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal("gemma2", plan.Architecture);
        Assert.Equal(4096, plan.AttentionWindow); // what the file declares at gemma2.attention.sliding_window
        Assert.Equal(WindowInterleave.AlternatingLocalFirst, plan.WindowInterleave);
        Assert.Equal(26, plan.LayerCount);

        for (int layer = 0; layer < plan.LayerCount; layer++)
        {
            int window = plan.AttentionWindowForLayer(layer);
            if (layer % 2 == 0)
            {
                Assert.True(window == 4096, $"layer {layer} must read inside the declared 4096-position window, resolved {window}");
            }
            else
            {
                Assert.True(window == 0, $"layer {layer} is gemma2's full-attention half and must stay unbounded, resolved {window}");
            }
        }

        // Nothing per-layer is declared on this file: the single window key plus the profile's pattern produced every
        // value above, which is what the plan contract has to work from until a real per-layer declaration exists.
        Assert.Single(gguf.Metadata.Keys, key => key.Contains("sliding_window", StringComparison.Ordinal));
    }

    // ---------------------------------------------------------------- bounded selection at the fixture's geometry and width

    [Fact]
    public void SelectionAtRealGemma2GeometryAndWindow_ExcludesWholeBlocks_WhileAnUnboundedLayerReadsThem()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(FixtureRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        ModelPlan plan = ModelPlans.Resolve(gguf);

        int localWindow = plan.AttentionWindowForLayer(0); // a local-attention layer of this very file: 4096
        Assert.Equal(4096, localWindow);
        int fullWindow = plan.AttentionWindowForLayer(1); // its odd neighbour: unbounded
        Assert.Equal(0, fullWindow);

        const int BlockTokens = 8;
        const int Positions = 4104; // eight positions past the declared window — exactly one block of KV beyond it
        int totalBlocks = (Positions + BlockTokens - 1) / BlockTokens;

        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        // One layer carrying this file's own KV geometry (4 kv-heads x 144 dimensions): what the window bounds here is
        // exactly one block of this model's per-layer read footprint.
        using BlockPool pool = new(
            blockTokens: BlockTokens,
            layerCount: 1,
            kvHeads: plan.KvHeadCount,
            headDim: plan.HeadDim,
            capacityBlocks: totalBlocks + 4,
            backend);

        int kvRegion = plan.KvHeadCount * plan.HeadDim;
        using SequenceKv kv = new(pool);
        for (int position = 0; position < Positions; position++)
        {
            kv.Append();
            kv.WriteK(position, 0, Row(position, magnitude: 1, kvRegion));
            kv.WriteV(position, 0, Row(position, magnitude: -1, kvRegion));
        }

        int queryPosition = Positions - 1;
        BlockSelection bounded = kv.SelectCausalRead(queryPosition, localWindow);
        Assert.Equal(Positions - localWindow, bounded.FirstPosition); // [8..4103]: the first block's eight positions fall outside it
        Assert.Equal(1, bounded.FirstBlockIndex);
        Assert.Equal(totalBlocks - 1, bounded.SelectedBlocks);
        Assert.Equal(1, bounded.SkippedBlocks); // one whole physical block of live KV is not addressed at all
        Assert.False(bounded.CoversWholeTable);

        BlockSelection unbounded = kv.SelectCausalRead(queryPosition, fullWindow);
        Assert.True(unbounded.CoversWholeTable);
        Assert.Equal(totalBlocks, unbounded.SelectedBlocks);
        Assert.Equal(0, unbounded.SkippedBlocks);

        int region = plan.HeadCount * plan.HeadDim; // 16 heads x 144: this file's query shape
        float[] q = new float[region];
        for (int i = 0; i < region; i++)
        {
            q[i] = 0.125f * (1 + (i % 7));
        }

        float scale = 1f / MathF.Sqrt(plan.HeadDim);

        float[] boundedClean = new float[region];
        LayerOps.CausalAttention(backend, q, plan.HeadCount, plan.KvHeadCount, plan.HeadDim, kv, layer: 0, queryPosition, scale, softcap: 0f, window: localWindow, boundedClean);

        float[] unboundedClean = new float[region];
        LayerOps.CausalAttention(backend, q, plan.HeadCount, plan.KvHeadCount, plan.HeadDim, kv, layer: 0, queryPosition, scale, softcap: 0f, window: fullWindow, unboundedClean);

        // Loudly rewrite exactly the positions a local-attention read excludes — block 0 of this sequence's KV.
        for (int position = 0; position < BlockTokens; position++)
        {
            kv.WriteK(position, 0, Row(position, magnitude: 4_000, kvRegion));
            kv.WriteV(position, 0, Row(position, magnitude: -4_000, kvRegion));
        }

        float[] boundedPoisoned = new float[region];
        LayerOps.CausalAttention(backend, q, plan.HeadCount, plan.KvHeadCount, plan.HeadDim, kv, layer: 0, queryPosition, scale, softcap: 0f, window: localWindow, boundedPoisoned);
        Assert.True(
            boundedClean.AsSpan().SequenceEqual(boundedPoisoned.AsSpan()),
            "a gemma2 local-attention read must not address the block its declared window excludes");

        float[] unboundedPoisoned = new float[region];
        LayerOps.CausalAttention(backend, q, plan.HeadCount, plan.KvHeadCount, plan.HeadDim, kv, layer: 0, queryPosition, scale, softcap: 0f, window: fullWindow, unboundedPoisoned);
        Assert.True(
            !unboundedClean.AsSpan().SequenceEqual(unboundedPoisoned.AsSpan()),
            "gemma2's full-attention half must address that same block: its reads are provably unbounded");

        // The window's left edge is what a masked read refuses to cross.
        Assert.Throws<ArgumentOutOfRangeException>(() => kv.ReadK(bounded, Positions - localWindow - 1, 0));
    }

    // ---------------------------------------------------------------- prompt-cache reuse on the fixture itself

    [Fact]
    public void PrefixReuseOnRealGemma2Fixture_ContinuesBitIdenticallyToAColdRun()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(FixtureRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        const int BlockTokens = 4;

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(modelPath);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 32);

        Assert.Equal("gemma2", model.Plan.Architecture);
        Assert.Equal(WindowInterleave.AlternatingLocalFirst, model.Plan.WindowInterleave);

        int[] promptB = [.. SharedPrefix, .. WarmTail];

        using BlockPool pool = new(
            blockTokens: BlockTokens,
            layerCount: model.Plan.LayerCount,
            kvHeads: model.Plan.KvHeadCount,
            headDim: model.Plan.HeadDim,
            capacityBlocks: 8,
            backend);

        PrefixCache cache = new(pool);

        using SequenceKv publisher = new(pool);
        model.Prefill(SharedPrefix, publisher); // one complete block of real gemma2 KV per layer
        Assert.Equal(1, cache.Publish(SharedPrefix, publisher));
        int publishedBlock = publisher.BlockFor(0);

        using SequenceKv warm = new(pool);
        Assert.Equal(1, cache.TryAdopt(promptB, warm)); // the shared block adopts; only the tail prefills fresh
        Assert.True(warm.IsAdopted(0) && warm.IsAdopted(BlockTokens - 1), "the adopted positions must be marked shared");
        Assert.Equal(publishedBlock, warm.BlockFor(0));

        model.Prefill(WarmTail, warm);
        float[] warmLogits = model.DecodeNext(promptB[^1], warm).ToArray(); // continue from the reused prefix

        using SequenceKv cold = new(pool);
        model.Prefill(promptB, cold);
        float[] coldLogits = model.DecodeNext(promptB[^1], cold).ToArray();

        Assert.True(
            warmLogits.AsSpan().SequenceEqual(coldLogits.AsSpan()),
            "a gemma2 continuation built on an adopted prefix must be bit-identical to the same prompt prefilled from nothing");

        // Below its declared window, both halves of this model's interleave select everything they attend — so reuse is
        // proven against a read footprint that really includes the shared block for every one of the 26 layers.
        int queryPosition = warm.Length - 1;
        Assert.Equal(promptB.Length, queryPosition);
        for (int layer = 0; layer < model.Plan.LayerCount; layer++)
        {
            BlockSelection read = warm.SelectCausalRead(queryPosition, model.Plan.AttentionWindowForLayer(layer));
            Assert.True(read.CoversWholeTable, $"layer {layer} attends the whole short table at position {queryPosition}: its window is wider than the sequence");
            Assert.Equal(0, read.SkippedBlocks);

            // Every layer — local or full — reads through the adopted physical block here.
            Assert.Equal(publishedBlock, warm.BlockFor(read.FirstPosition));
        }

        publisher.Dispose();
        warm.Dispose();
        cold.Dispose();
    }

    // ---------------------------------------------------------------- fixture helpers

    // Distinct small rows per position: two positions' KV can never collapse to equal values, so loud rewrites of the
    // excluded positions move an unbounded read unmistakably rather than by rounding noise.
    private static float[] Row(int position, double magnitude, int length)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = (float)(magnitude * ((position % 9) + 1) / 8.0 * (((i % 5) + 1) / 3.0));
        }

        return values;
    }
}
