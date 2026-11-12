using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Http;
using Lair.Http.Monitoring;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Tokenizer.Templating;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit.Abstractions;

namespace Lair.Tests;

/// <summary>
/// Serve-side proof that this model's hybrid window really bounds KV reads on the pinned
/// <c>models/gemma-3-1b-it-Q3_K_S.gguf</c>: a two-turn chat exchange whose first prompt is longer than the declared
/// 512-position window runs end to end on the production load path, and the engine's own monitor snapshot reports 22
/// windowed layers of 26 with a read footprint strictly below the whole-sequence equivalent. The bounded reads are then
/// proven against the file's own KV geometry rather than inferred from loop bounds: past the window a windowed layer
/// selects exactly the blocks its backward span covers and excludes whole physical blocks, and loudly rewriting exactly
/// those excluded blocks leaves a windowed layer's attention output bit-identical while the full-attention layers this
/// family puts on the whole range read them. The selected-block count for a windowed layer is the same at a short and a
/// long query position, so what a windowed read costs does not grow with the sequence. Cases skip when the fixture is
/// absent, like the other real-model suites, and one engine serves the class on the serialized <c>model-loaded</c> lane.
/// </summary>
[Trait("Category", "Serve")]
[Collection("model-loaded")]
public sealed class Gemma3WindowedServeTests
{
    private const string ModelRelativePath = "models/gemma-3-1b-it-Q3_K_S.gguf";
    private const string ModelName = "gemma3-windowed-serve";
    private const string ModelSha256 = "0ffb15257ccee33bee17e732fb3db55a332801d187e87164488888022fe1634c";

    // What the pinned file declares at gemma3.attention.sliding_window, and what its profile's stride leaves unbounded.
    private const int Window = 512;
    private const int Layers = 26;
    private const int WindowedLayers = 22;
    private const int BlockTokens = 16;

    private static readonly object Gate = new();
    private static GenerationService? _engine;

    private readonly ITestOutputHelper _output;

    public Gemma3WindowedServeTests(ITestOutputHelper output) => _output = output;

    // ---------------------------------------------------------------- serving a prompt past the window

    [Fact]
    public async Task Serve_TwoTurnExchangePastWindow_ContinuesPastTheLocalAttentionWindow()
    {
        GenerationService? engine = OpenEngine();
        if (engine is null)
        {
            return; // pinned model absent: nothing to serve
        }

        Assert.Equal("gemma3", engine.Model.Architecture);

        GenerationRequest FirstTurnRequest() => new()
        {
            Model = ModelName,
            IsChat = true,
            RequestId = "req-gemma3-windowed-first",
            Messages = [new ChatMessage("user", LongContextQuestion())],
            MaxTokens = 48,
            Temperature = 0f,
            Seed = 7,
        };

        GenerationResult first = await engine.GenerateAsync(FirstTurnRequest(), CancellationToken.None);

        _output.WriteLine($"windowed turn 1: finish={first.FinishReason} prompt={first.PromptTokens} completion={first.CompletionTokens} text={Describe(first.Text)}");
        Assert.True(first.CompletionTokens > 0, "the first turn generated no tokens");
        Assert.Contains(first.FinishReason, new[] { "eos", "length" });
        AssertVisibleText(first.Text, "first turn");

        // The exchange really crosses the window: this prompt is longer than the 512 positions a local layer reads.
        Assert.True(
            first.PromptTokens > Window,
            $"the first turn's prompt must exceed the {Window}-position window to prove a past-window read, got {first.PromptTokens} tokens");

        GenerationRequest SecondTurnRequest() => new()
        {
            Model = ModelName,
            IsChat = true,
            RequestId = "req-gemma3-windowed-second",
            Messages =
            [
                new ChatMessage("user", LongContextQuestion()),
                new ChatMessage("assistant", first.Text),
                new ChatMessage("user", "Of the details you just gave, name the one that comes last. Reply with just that detail."),
            ],
            MaxTokens = 48,
            Temperature = 0f,
            Seed = 11,
        };

        GenerationResult second = await engine.GenerateAsync(SecondTurnRequest(), CancellationToken.None);

        _output.WriteLine($"windowed turn 2: finish={second.FinishReason} prompt={second.PromptTokens} completion={second.CompletionTokens} text={Describe(second.Text)}");
        Assert.True(second.CompletionTokens > 0, "the second turn generated no tokens");
        Assert.Contains(second.FinishReason, new[] { "eos", "length" });
        AssertVisibleText(second.Text, "second turn");
        Assert.True(second.PromptTokens > first.PromptTokens, "the second prompt did not carry the first reply");

        string carried = first.Text.Trim();
        Assert.True(carried.Length > 0, "turn one's reply is only whitespace, so there is nothing to carry");
        Assert.Contains(carried, engine.BuildChatPrompt(SecondTurnRequest(), CancellationToken.None), StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- the engine's own footprint numbers

    [Fact]
    public async Task EngineSnapshot_PastWindow_ReportsWindowedReadFootprint_WithoutWholeSequenceFallback()
    {
        GenerationService? engine = OpenEngine();
        if (engine is null)
        {
            return; // pinned model absent: nothing to measure
        }

        EngineMetricsBus bus = engine.AttachMetrics(NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(50));

        GenerationRequest request = new()
        {
            Model = ModelName,
            IsChat = true,
            RequestId = "req-gemma3-windowed-footprint",
            Messages = [new ChatMessage("user", LongContextQuestion())],
            MaxTokens = 32,
            Temperature = 0f,
            Seed = 5,
        };

        GenerationResult result = await engine.GenerateAsync(request, CancellationToken.None);
        Assert.True(result.PromptTokens > Window, $"the footprint case needs a prompt past the {Window}-position window");

        System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();
        while (bus.RingCount == 0 && clock.Elapsed < TimeSpan.FromSeconds(20))
        {
            await Task.Delay(25);
        }

        Assert.True(bus.RingCount > 0, "the metrics bus published no snapshot after a past-window gemma3 exchange");

        MetricsSnapshot snapshot = bus.SnapshotRing()[0];

        // Engine-visible: 22 of the 26 layers carry the window, and the blocks they read are the window-bounded count,
        // not the whole-sequence count every one of them would read if the window fell back to full KV.
        Assert.Equal(WindowedLayers, snapshot.WindowedLayers);

        int budgetBlocks = engine.KvBudgetBlocks;
        int windowedBlocksPerLayer = (Math.Min(Window, budgetBlocks * BlockTokens) + BlockTokens - 1) / BlockTokens;
        long expectedFootprint = (Layers - WindowedLayers) * (long)budgetBlocks + WindowedLayers * (long)windowedBlocksPerLayer;
        long expectedSkipped = WindowedLayers * (long)(budgetBlocks - windowedBlocksPerLayer);
        long wholeSequenceEquivalent = Layers * (long)budgetBlocks;

        _output.WriteLine(
            $"footprint: budget_blocks={budgetBlocks} windowed_blocks_per_layer={windowedBlocksPerLayer} " +
            $"read={snapshot.KvReadFootprintBlocks} skipped={snapshot.SkippedKvReadBlocks} whole_sequence={wholeSequenceEquivalent}");

        Assert.Equal(expectedFootprint, snapshot.KvReadFootprintBlocks);
        Assert.Equal(expectedSkipped, snapshot.SkippedKvReadBlocks);
        Assert.True(expectedSkipped > 0, "a windowed layer must skip whole blocks of live KV at this budget");
        Assert.True(
            snapshot.KvReadFootprintBlocks < wholeSequenceEquivalent,
            $"the windowed read footprint ({snapshot.KvReadFootprintBlocks}) must stay below the whole-sequence equivalent ({wholeSequenceEquivalent})");

        await bus.DisposeAsync();
    }

    // ---------------------------------------------------------------- bounded selection at the file's own geometry

    [Fact]
    public void SelectionPastWindowAtRealGemma3Geometry_ExcludesWholeBlocks_WhileFullAttentionLayersReadThem()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(ModelRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        AssertFixtureHash(modelPath);

        using GgufFile gguf = GgufFile.Open(modelPath);
        ModelPlan plan = ModelPlans.Resolve(gguf);

        Assert.Equal(Window, plan.AttentionWindowForLayer(0)); // a local-attention layer of this very file
        Assert.Equal(0, plan.AttentionWindowForLayer(5));      // its family's full-attention layer in the same group of six

        const int CrossedPositions = Window + 32; // two whole blocks of KV past the window
        int crossedBlocks = (CrossedPositions + BlockTokens - 1) / BlockTokens;

        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        // One layer carrying this file's own KV geometry (one kv-head of 256 dimensions): what the window bounds here is
        // whole physical blocks of this model's per-layer read footprint.
        using BlockPool pool = new(
            blockTokens: BlockTokens,
            layerCount: 1,
            kvHeads: plan.KvHeadCount,
            headDim: plan.HeadDim,
            capacityBlocks: crossedBlocks + 4,
            backend);

        int kvRegion = plan.KvHeadCount * plan.HeadDim;
        using SequenceKv kv = new(pool);

        for (int position = 0; position < CrossedPositions; position++)
        {
            kv.Append();
            kv.WriteK(position, 0, Row(position, magnitude: 1, kvRegion));
            kv.WriteV(position, 0, Row(position, magnitude: -1, kvRegion));
        }

        int queryPosition = CrossedPositions - 1;

        BlockSelection bounded = kv.SelectCausalRead(queryPosition, Window);
        Assert.Equal(CrossedPositions - Window, bounded.FirstPosition); // the first two blocks fall entirely outside the window
        Assert.Equal(2, bounded.FirstBlockIndex);
        Assert.Equal(crossedBlocks - 2, bounded.SelectedBlocks);
        Assert.Equal(2, bounded.SkippedBlocks);
        Assert.False(bounded.CoversWholeTable);

        BlockSelection unbounded = kv.SelectCausalRead(queryPosition, plan.AttentionWindowForLayer(5));
        Assert.True(unbounded.CoversWholeTable);
        Assert.Equal(crossedBlocks, unbounded.SelectedBlocks);
        Assert.Equal(0, unbounded.SkippedBlocks);

        int region = plan.HeadCount * plan.HeadDim; // four heads of 256: this file's query region against a 1152-wide hidden size
        float[] q = new float[region];
        for (int i = 0; i < region; i++)
        {
            q[i] = 0.125f * (1 + (i % 7));
        }

        float scale = 1f / MathF.Sqrt(plan.HeadDim);

        float[] boundedClean = new float[region];
        LayerOps.CausalAttention(backend, q, plan.HeadCount, plan.KvHeadCount, plan.HeadDim, kv, layer: 0, queryPosition, scale, softcap: 0f, window: Window, boundedClean);

        float[] unboundedClean = new float[region];
        LayerOps.CausalAttention(backend, q, plan.HeadCount, plan.KvHeadCount, plan.HeadDim, kv, layer: 0, queryPosition, scale, softcap: 0f, window: 0, unboundedClean);

        // Loudly rewrite exactly the positions a windowed read excludes — the first two blocks of this sequence's KV.
        for (int position = 0; position < 2 * BlockTokens; position++)
        {
            kv.WriteK(position, 0, Row(position, magnitude: 4_000, kvRegion));
            kv.WriteV(position, 0, Row(position, magnitude: -4_000, kvRegion));
        }

        float[] boundedPoisoned = new float[region];
        LayerOps.CausalAttention(backend, q, plan.HeadCount, plan.KvHeadCount, plan.HeadDim, kv, layer: 0, queryPosition, scale, softcap: 0f, window: Window, boundedPoisoned);
        Assert.True(
            boundedClean.AsSpan().SequenceEqual(boundedPoisoned.AsSpan()),
            "a gemma3 windowed read must not address the blocks its declared window excludes");

        float[] unboundedPoisoned = new float[region];
        LayerOps.CausalAttention(backend, q, plan.HeadCount, plan.KvHeadCount, plan.HeadDim, kv, layer: 0, queryPosition, scale, softcap: 0f, window: 0, unboundedPoisoned);
        Assert.True(
            !unboundedClean.AsSpan().SequenceEqual(unboundedPoisoned.AsSpan()),
            "gemma3's full-attention layers must address those same blocks: their reads are provably unbounded");

        // The window's left edge is what a bounded read refuses to cross.
        Assert.Throws<ArgumentOutOfRangeException>(() => kv.ReadK(bounded, CrossedPositions - Window - 1, 0));
    }

    [Fact]
    public void SelectionPastWindowAtRealGemma3Geometry_StaysBounded_AsTheSequenceGrows()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(ModelRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        AssertFixtureHash(modelPath);

        using GgufFile gguf = GgufFile.Open(modelPath);
        ModelPlan plan = ModelPlans.Resolve(gguf);

        const int Positions = 1024; // twice the window: a sequence a windowed layer reads the same way at either end
        int totalBlocks = (Positions + BlockTokens - 1) / BlockTokens;

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using BlockPool pool = new(
            blockTokens: BlockTokens,
            layerCount: 1,
            kvHeads: plan.KvHeadCount,
            headDim: plan.HeadDim,
            capacityBlocks: totalBlocks,
            backend);

        using SequenceKv kv = new(pool);
        for (int position = 0; position < Positions; position++)
        {
            kv.Append();
        }

        int crossedQuery = Window + 32 - 1;
        int grownQuery = Positions - 1;

        BlockSelection crossed = kv.SelectCausalRead(crossedQuery, Window);
        BlockSelection grown = kv.SelectCausalRead(grownQuery, Window);

        int windowedBlocks = Window / BlockTokens; // the window covers exactly this many blocks whatever the sequence length is

        Assert.Equal(windowedBlocks, crossed.SelectedBlocks);
        Assert.Equal(windowedBlocks, grown.SelectedBlocks); // the read footprint did not grow with the sequence
        Assert.Equal(grownQuery - Window + 1, grown.FirstPosition);
        Assert.False(grown.CoversWholeTable);

        BlockSelection grownUnbounded = kv.SelectCausalRead(grownQuery, 0);
        Assert.True(grownUnbounded.CoversWholeTable);
        Assert.Equal(totalBlocks, grownUnbounded.SelectedBlocks);
        Assert.True(
            grownUnbounded.SelectedBlocks > grown.SelectedBlocks,
            $"the whole-sequence read ({grownUnbounded.SelectedBlocks} blocks) must exceed the windowed read ({grown.SelectedBlocks} blocks) at the same position");
    }

    // ---------------------------------------------------------------- the whole model's footprint accounting

    [Fact]
    public void BlockPoolAtRealGemma3Geometry_AccountsWindowedReadFootprint_ForEveryLayer()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(ModelRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        AssertFixtureHash(modelPath);

        using GgufFile gguf = GgufFile.Open(modelPath);
        ModelPlan plan = ModelPlans.Resolve(gguf);

        const int CapacityBlocks = 64;

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using BlockPool pool = new(
            blockTokens: BlockTokens,
            layerCount: plan.LayerCount,
            kvHeads: plan.KvHeadCount,
            headDim: plan.HeadDim,
            capacityBlocks: CapacityBlocks,
            backend);

        int[] attentionWindows = new int[plan.LayerCount];
        for (int layer = 0; layer < plan.LayerCount; layer++)
        {
            attentionWindows[layer] = plan.AttentionWindowForLayer(layer);
        }

        pool.DeclareAttentionWindows(attentionWindows);

        Assert.Equal(WindowedLayers, pool.DeclaredWindowedLayerCount);

        int budgetBlocks = pool.EffectiveBudgetBlocks;
        int windowedBlocksPerLayer = (Math.Min(Window, budgetBlocks * BlockTokens) + BlockTokens - 1) / BlockTokens;

        Assert.Equal(
            (Layers - WindowedLayers) * (long)budgetBlocks + WindowedLayers * (long)windowedBlocksPerLayer,
            pool.KvReadFootprintBlocksAtBudget);
        Assert.Equal(WindowedLayers * (long)(budgetBlocks - windowedBlocksPerLayer), pool.SkippedKvReadFootprintBlocksAtBudget);
        Assert.True(pool.SkippedKvReadFootprintBlocksAtBudget > 0);
        Assert.True(
            pool.KvReadFootprintBlocksAtBudget < Layers * (long)budgetBlocks,
            "the declared window must keep this model's read footprint below the whole-sequence equivalent");
    }

    // ---------------------------------------------------------------- helpers

    // A question whose rendered prompt is far longer than the window: the same detail-list prompt every case uses, so
    // the crossing is a property of the prompt rather than of whichever case asserts it.
    private static string LongContextQuestion() =>
        "Read these instructions and answer the last one only. " +
        string.Join(" ", Enumerable.Range(0, 40).Select(i =>
            $"Item {i}: the quick brown fox jumps over the lazy dog while the quiet farmer watches the river bend near the old stone bridge.")) +
        " Question: which item mentions a bridge? Reply with just the item number.";

    private static void AssertVisibleText(string text, string turn)
    {
        Assert.False(string.IsNullOrEmpty(text), $"{turn} produced empty text");
        Assert.True(text.Trim().Length > 0, $"{turn} produced only whitespace");
        Assert.False(text.Any(c => char.IsControl(c) && c != '\n'), $"{turn} contains control characters");
    }

    private static string Describe(string text)
    {
        string preview = text.Length <= 160 ? text : text[..160] + "...";
        return preview.Replace("\n", "\\n");
    }

    private static void AssertFixtureHash(string modelPath)
    {
        using System.Security.Cryptography.SHA256 hasher = System.Security.Cryptography.SHA256.Create();
        using FileStream stream = File.OpenRead(modelPath);
        string actual = Convert.ToHexString(hasher.ComputeHash(stream)).ToLowerInvariant();

        Assert.True(
            string.Equals(actual, ModelSha256, StringComparison.Ordinal),
            $"gemma3 serve guard: {ModelRelativePath} hashes to {actual}, not the pinned {ModelSha256}: " +
            "these assertions were written against that exact file.");
    }

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

    // One engine for the whole class: opened lazily on first use, never disposed mid-run so a later case in this
    // serialized lane can reuse it. The process exit releases the weights.
    private GenerationService? OpenEngine()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(ModelRelativePath);
        if (modelPath is null)
        {
            _output.WriteLine($"SKIP Serve: {ModelRelativePath} not found walking up from the test assembly directory; the gemma3 windowed serve fixture needs the pinned model file.");
            return null;
        }

        AssertFixtureHash(modelPath);

        lock (Gate)
        {
            return _engine ??= GenerationService.Open(new LairConfig
            {
                Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
                Engine = new EngineConfig { BlockSize = BlockTokens },
                Models =
                [
                    new ModelConfig
                    {
                        Name = ModelName,
                        Path = modelPath,
                        MaxContextTokens = 2048,
                        KvBlocks = 160,
                    },
                ],
            });
        }
    }
}
