using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Http;
using Lair.Kv;
using Lair.Models;
using Lair.Models.Engine;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// What a declared quantized KV pair does at model open on the pinned SmolLM fixture: the named combos open, serve, and
/// report the attention route they closed at, over the typed pool their pair's group layout sizes — and the postures a
/// quantized route does not carry (a dedicated lane, a non-gemv prefill selection) plus pairs with no attention route
/// each refuse naming what was declared and what to change, before any pool is constructed. The forwards run the closed
/// route end-to-end: prefill, single-token decode, and the batched decode window all address the typed sequences, and a
/// greedy stream stays on the float posture's argmax path. Runs inside the serialized real-model lane with the other
/// serve suites.
/// </summary>
[Trait("Category", "Serve")]
[Collection("model-loaded")] // shares one serialized lane with the other real-model suites: two engines mapping 270 MB at once is not what this box should be doing
public sealed class KvQuantizedServeTests
{
    private const string ModelName = "smollm-kv-quantized";
    private const string Prompt = "hello world";

    // ---------------------------------------------------------------- named combos open, serve, and report their closed route

    [Theory]
    [InlineData("q8_0", "q8_0")]
    [InlineData("q8_0", "q4_0")]
    public void GenerationServiceOpen_NamedQuantizedCombo_OpensServesAndReportsItsClosedRoute(string k, string v)
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        KvFormatPair pair = new(Side(k), Side(v));

        using GenerationService engine = GenerationService.Open(Config(k, v), NullLogger.Instance);

        Assert.Equal(pair, engine.ServedModel.ResolvedKvFormats);
        Assert.Equal($"{k}/{v}", engine.ServedModel.ResolvedKvAttentionRoute); // the route the engine closed at is what it reports back

        IKvPool pool = engine.ServedBlockPool;
        Assert.True(pool is BlockPool<Q8Block, Q8Block> or BlockPool<Q8Block, Q4Block>); // the scheduler admits against the typed pool the pair's groups need
        Assert.Equal(pair, pool.Formats);
        Assert.Equal(BlockResidencyMode.Dram, pool.Residency);
        Assert.Equal(0, pool.DeviceBackedBlockCount); // quantized blocks carry no device mirror in this build

        ModelDesc desc = engine.ServedModel.Desc;
        Assert.Equal(
            pair.BytesPerBlock(desc.LayerCount, pool.BlockTokens, desc.KvHeadCount * desc.HeadDim),
            pool.BytesPerBlock); // the reservation states the pair's group footprint, not the float layout

        GenerationResult served = Generate(engine);
        Assert.True(served.CompletionTokens > 0, $"finish='{served.FinishReason}' text='{served.Text}' promptTokens={served.PromptTokens}");
        Assert.False(string.IsNullOrWhiteSpace(served.Text));

        using GenerationService again = GenerationService.Open(Config(k, v), NullLogger.Instance);
        Assert.Equal(served.Text, Generate(again).Text); // the closed route is deterministic across opens of the same posture
    }

    [Theory]
    [InlineData("q8_0", "q8_0")]
    [InlineData("q8_0", "q4_0")]
    public void ServedModel_QuantizedForwardBodies_RunOverTypedSequences_AcrossTheFloatPosturesArgmaxPath(string k, string v)
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GenerationService quantized = GenerationService.Open(Config(k, v), NullLogger.Instance);
        using GenerationService reference = GenerationService.Open(Config("f32", "f32"), NullLogger.Instance);

        TransformerModel model = quantized.ServedModel;
        TransformerModel floatModel = reference.ServedModel;

        int[] prompt = [.. Enumerable.Repeat(7, 24)];

        using IKvSequence quantKv = quantized.ServedBlockPool.NewSequence();
        using IKvSequence floatKv = reference.ServedBlockPool.NewSequence();

        int quantToken = ArgMax([.. model.Prefill(prompt, quantKv)]);
        int floatToken = ArgMax([.. floatModel.Prefill(prompt, floatKv)]);
        Assert.Equal(floatToken, quantToken); // the quantized attention route lands the same next-token argmax the FP32 route produces

        // Single-token decode continues on the typed sequence through the closed route.
        for (int step = 0; step < 3; step++)
        {
            quantToken = ArgMax([.. model.DecodeNext(quantToken, quantKv)]);
            floatToken = ArgMax([.. floatModel.DecodeNext(floatToken, floatKv)]);
            Assert.Equal(floatToken, quantToken);
        }

        // The batched decode window runs the same closed route per sequence: two quantized sequences step together and
        // each row's logits row lands the argmax the float posture produces for the same sequences.
        using IKvSequence quantKvB = quantized.ServedBlockPool.NewSequence();
        using IKvSequence quantKvC = quantized.ServedBlockPool.NewSequence();
        using IKvSequence floatKvB = reference.ServedBlockPool.NewSequence();
        using IKvSequence floatKvC = reference.ServedBlockPool.NewSequence();

        _ = model.Prefill(prompt, quantKvB);
        _ = model.Prefill(prompt, quantKvC);
        _ = floatModel.Prefill(prompt, floatKvB);
        _ = floatModel.Prefill(prompt, floatKvC);

        int[] stepTokens = [quantToken, floatToken];
        float[] quantBatch = [.. model.DecodeBatch(stepTokens, [quantKvB, quantKvC])];
        float[] floatBatch = [.. floatModel.DecodeBatch(stepTokens, [floatKvB, floatKvC])];

        int vocab = model.Desc.VocabSize;
        Assert.Equal(ArgMax(floatBatch[..vocab]), ArgMax(quantBatch[..vocab])); // sequence B's row
        Assert.Equal(ArgMax(floatBatch[vocab..(2 * vocab)]), ArgMax(quantBatch[vocab..(2 * vocab)])); // sequence C's row
    }

    [Theory]
    [InlineData("q8_0", "q8_0")]
    [InlineData("q8_0", "q4_0")]
    public void ServedModel_EncoderPass_RefusesAQuantizedEngineNamingThePair(string k, string v)
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GenerationService engine = GenerationService.Open(Config(k, v), NullLogger.Instance);
        ModelDesc desc = engine.ServedModel.Desc;

        using CpuBackend backend = new();
        using BlockPool floatPool = new(16, desc.LayerCount, desc.KvHeadCount, desc.HeadDim, 4, backend);
        using SequenceKv kv = new(floatPool);

        ModelException error = Assert.Throws<ModelException>(() => engine.ServedModel.Encode([.. Enumerable.Repeat(7, 8)], kv));

        Assert.Contains($"kv k='{k}' v='{v}'", error.Message);
        Assert.Contains("encoder pass runs the FP32 bidirectional attention kernels", error.Message);
    }

    // ---------------------------------------------------------------- postures a quantized route does not carry

    [Theory]
    [InlineData("q8_0", "q8_0")]
    [InlineData("q8_0", "q4_0")]
    public void GenerationServiceOpen_QuantizedComboBesideANonRowWindowSelection_RefusesNamingTheSelectionAndWindow(string k, string v)
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        ModelException error = Assert.Throws<ModelException>(
            () => GenerationService.Open(Config(k, v, prefillKernel: "tiled"), NullLogger.Instance));

        Assert.Contains($"model '{ModelName}' declares kv.quant", error.Message);
        Assert.Contains("'engine.prefill_kernel' 'tiled'", error.Message); // names the key that declared the posture
        Assert.Contains("batched-row window only", error.Message); // names what the quantized route carries instead
    }

    [Theory]
    [InlineData("q8_0", "q8_0")]
    [InlineData("q8_0", "q4_0")]
    public void GenerationServiceOpen_QuantizedComboBesideDedicatedLane_RefusesNamingTheLaneKey(string k, string v)
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        ModelException error = Assert.Throws<ModelException>(
            () => GenerationService.Open(Config(k, v, placement: new PlacementConfig { PrefillLane = PlacementPrefillLaneValues.Dedicated }, threads: 4), NullLogger.Instance));

        Assert.Contains($"model '{ModelName}' declares kv.quant", error.Message);
        Assert.Contains("'placement.prefill_lane'", error.Message);
        Assert.Contains("no dedicated-lane or offload posture", error.Message);
    }

    // ---------------------------------------------------------------- pairs with no attention route

    [Theory]
    [InlineData("f16", "f16")]
    [InlineData("f16", "q8_0")]
    [InlineData("q4_0", "q8_0")]
    [InlineData("f32", "q4_0")]
    [InlineData("q4_0", "q4_0")]
    public void GenerationServiceOpen_PairWithoutARoute_RefusesBeforeAnyPoolIsConstructed(string k, string v)
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        ModelException error = Assert.Throws<ModelException>(
            () => GenerationService.Open(Config(k, v), NullLogger.Instance));

        Assert.Contains($"model '{ModelName}' declares kv.quant k='{k}' v='{v}'", error.Message);
        Assert.Contains("no storage/attention execution support", error.Message);
        Assert.Contains("refusing before the KV pool is constructed", error.Message);
    }

    // ---------------------------------------------------------------- helpers

    private static KvSideFormat Side(string name) => name switch
    {
        "f32" => KvSideFormat.F32,
        "f16" => KvSideFormat.F16,
        "q8_0" => KvSideFormat.Q8_0,
        _ => KvSideFormat.Q4_0,
    };

    private static LairConfig Config(string k, string v, string prefillKernel = "gemv", PlacementConfig? placement = null, int threads = 1) => new()
    {
        Server = new ServerConfig(),
        Engine = new EngineConfig { Threads = threads, PrefillKernel = prefillKernel },
        Models =
        [
            new ModelConfig
            {
                Name = ModelName,
                Path = ModelFixture.Resolve()!,
                MaxContextTokens = 512,
                KvBlocks = 8,
                KvQuant = new KvFormatPair(Side(k), Side(v)),
                Placement = placement ?? new PlacementConfig(),
            },
        ],
    };

    private static GenerationResult Generate(GenerationService engine) =>
        engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = Prompt,
            MaxTokens = 12,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult();

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[best])
            {
                best = i;
            }
        }

        return best;
    }
}
