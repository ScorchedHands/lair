using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Batched decode-step tests over the real SmolLM F16 fixture: a batched step must be bit-identical to stepping
/// each sequence alone, sequence by sequence — logits compared on the raw bits of the floats at every step, and the
/// KV slots each sequence holds compared against its solo-stepped twin. Covers batches of different prompt lengths,
/// a single-sequence batch, sequences sharing identical prompt content, and argument validation.
/// </summary>
[Trait("Category", "Layers")]
public sealed class BatchStepTests
{
    private const int DecodeSteps = 8;

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void DecodeBatch_DifferentPromptLengths_MatchesSoloStepping_BitForBit(int batchSequences)
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        int[] promptA = [.. tokenizer.Encode("Hello there")];
        int[] promptB = [.. BuildLongPrompt(tokenizer)];
        Assert.True(promptA.Length < promptB.Length, "the two sequences must carry different KV histories");

        int[] prompts = batchSequences == 1 ? [promptA.Length] : [promptA.Length, promptB.Length];

        // Batched engine runs threaded: the batched kernels' thread determinism is exercised alongside parity.
        using CpuBackend batchedBackend = new(threads: 4, allowAvx2: true);
        using TransformerModel batchedModel = TransformerModel.Open(gguf, batchedBackend, maxContextTokens: 96);
        using BlockPool batchedPool = NewPool(batchedModel, batchedBackend, capacityBlocks: 16);

        // Solo engine steps each sequence alone; both engines are fed the same token ids, so they stay on one path.
        using CpuBackend soloBackend = new(threads: 1, allowAvx2: true);
        using TransformerModel soloModel = TransformerModel.Open(gguf, soloBackend, maxContextTokens: 96);
        using BlockPool soloPool = NewPool(soloModel, soloBackend, capacityBlocks: 16);

        SequenceKv[] batchedKvs = new SequenceKv[batchSequences];
        SequenceKv[] soloKvs = new SequenceKv[batchSequences];
        for (int s = 0; s < batchSequences; s++)
        {
            batchedKvs[s] = new SequenceKv(batchedPool);
            soloKvs[s] = new SequenceKv(soloPool);
        }

        try
        {
            int vocab = batchedModel.Desc.VocabSize;
            Assert.Equal(soloModel.Desc.VocabSize, vocab);

            int[] nextId = new int[batchSequences];
            batchedModel.Prefill(promptA, batchedKvs[0]);
            soloModel.Prefill(promptA, soloKvs[0]);
            nextId[0] = ArgMax(soloModel.LastLogits()); // prefill leaves the final token's logits: the natural continuation

            if (batchSequences == 2)
            {
                batchedModel.Prefill(promptB, batchedKvs[1]);
                soloModel.Prefill(promptB, soloKvs[1]);
                nextId[1] = ArgMax(soloModel.LastLogits());
            }

            for (int step = 0; step < DecodeSteps; step++)
            {
                Span<float> batched = batchedModel.DecodeBatch(nextId, batchedKvs);
                Assert.Equal(batchSequences * vocab, batched.Length);

                for (int s = 0; s < batchSequences; s++)
                {
                    float[] solo = soloModel.DecodeNext(nextId[s], soloKvs[s]).ToArray();
                    foreach (float v in solo)
                    {
                        Assert.True(float.IsFinite(v), $"sequence {s} step {step}: logit {v} is not finite");
                    }

                    AssertBitsEqual(solo, batched.Slice(s * vocab, vocab), $"sequence {s} step {step} logits");
                    nextId[s] = ArgMax(solo); // greedy continuation keeps both engines on the identical path
                }
            }

            for (int s = 0; s < batchSequences; s++)
            {
                Assert.Equal(prompts[s] + DecodeSteps, batchedKvs[s].Length);
                Assert.Equal(batchedKvs[s].Length, soloKvs[s].Length);
                AssertKvSlotsEqual(batchedKvs[s], soloKvs[s], batchedModel.Desc.LayerCount, prompts[s]);
            }
        }
        finally
        {
            foreach (SequenceKv kv in batchedKvs)
            {
                kv.Dispose();
            }

            foreach (SequenceKv kv in soloKvs)
            {
                kv.Dispose();
            }
        }
    }

    [Fact]
    public void DecodeBatch_SingleSequence_EqualsDecodeNextExactly()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 64);
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 4);
        using SequenceKv soloKv = new(pool);
        using SequenceKv batchKv = new(pool);

        int[] prompt = [.. tokenizer.Encode("Hello there")];
        model.Prefill(prompt, soloKv);
        model.Prefill(prompt, batchKv);

        int vocab = model.Desc.VocabSize;
        int nextId = ArgMax(model.LastLogits());

        for (int step = 0; step < 4; step++)
        {
            float[] solo = model.DecodeNext(nextId, soloKv).ToArray();
            Span<float> batched = model.DecodeBatch([nextId], [batchKv]);

            Assert.Equal(vocab, batched.Length); // a one-sequence batch is a vocab-sized window
            AssertBitsEqual(solo, batched, $"single-sequence batch step {step} logits");

            nextId = ArgMax(solo);
        }

        Assert.Equal(soloKv.Length, batchKv.Length);
        AssertKvSlotsEqual(batchKv, soloKv, model.Desc.LayerCount, prompt.Length);
    }

    [Fact]
    public void DecodeBatch_IdenticalPrompts_StayIdenticalPerSequence()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        using CpuBackend batchedBackend = new(threads: 2, allowAvx2: true);
        using TransformerModel batchedModel = TransformerModel.Open(gguf, batchedBackend, maxContextTokens: 64);
        using BlockPool batchedPool = NewPool(batchedModel, batchedBackend, capacityBlocks: 4);

        using CpuBackend soloBackend = new(threads: 1, allowAvx2: true);
        using TransformerModel soloModel = TransformerModel.Open(gguf, soloBackend, maxContextTokens: 64);
        using BlockPool soloPool = NewPool(soloModel, soloBackend, capacityBlocks: 4);

        SequenceKv[] batchedKvs = [new SequenceKv(batchedPool), new SequenceKv(batchedPool)];
        SequenceKv[] soloKvs = [new SequenceKv(soloPool), new SequenceKv(soloPool)];

        try
        {
            int vocab = batchedModel.Desc.VocabSize;
            int[] prompt = [.. tokenizer.Encode("Hello there")];

            int[] nextId = new int[2];
            for (int s = 0; s < 2; s++)
            {
                batchedModel.Prefill(prompt, batchedKvs[s]);
                soloModel.Prefill(prompt, soloKvs[s]);
                nextId[s] = ArgMax(soloModel.LastLogits());
            }

            Assert.Equal(nextId[0], nextId[1]); // identical prompts start both sequences on the same token

            for (int step = 0; step < 6; step++)
            {
                Span<float> batched = batchedModel.DecodeBatch(nextId, batchedKvs);

                for (int s = 0; s < 2; s++)
                {
                    float[] solo = soloModel.DecodeNext(nextId[s], soloKvs[s]).ToArray();
                    AssertBitsEqual(solo, batched.Slice(s * vocab, vocab), $"shared-prefix sequence {s} step {step} logits");
                    nextId[s] = ArgMax(solo);
                }

                // Two sequences in identical states must produce identical logits from the batched step too.
                AssertBitsEqual(batched.Slice(0, vocab), batched.Slice(vocab, vocab), $"shared-prefix step {step} cross-sequence logits");
            }

            for (int s = 0; s < 2; s++)
            {
                AssertKvSlotsEqual(batchedKvs[s], soloKvs[s], batchedModel.Desc.LayerCount, prompt.Length);
            }
        }
        finally
        {
            foreach (SequenceKv kv in batchedKvs)
            {
                kv.Dispose();
            }

            foreach (SequenceKv kv in soloKvs)
            {
                kv.Dispose();
            }
        }
    }

    [Fact]
    public void DecodeBatch_Validation_RejectsEmptyOversizedMismatchedAndNullEntryBatches()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        using CpuBackend backend = new();
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 32);
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 4);
        using SequenceKv kv = new(pool);

        Assert.Throws<ArgumentException>(() => model.DecodeBatch(new int[0], new SequenceKv[0]));

        Assert.Throws<ArgumentException>(() => model.DecodeBatch(new int[17], new SequenceKv[17]));

        Assert.Throws<ArgumentException>(() => model.DecodeBatch(new int[2], new SequenceKv[1]));

        Assert.Throws<ArgumentException>(() => model.DecodeBatch(new int[1], new SequenceKv[1]));

        Assert.Throws<ArgumentException>(() => model.DecodeBatch([1], new SequenceKv[] { null! }));

        Assert.Throws<ArgumentNullException>(() => model.DecodeBatch(null!, new SequenceKv[1]));
        Assert.Throws<ArgumentNullException>(() => model.DecodeBatch(new int[1], null!));
    }

    // ---------------------------------------------------------------- helpers

    private static BlockPool NewPool(TransformerModel model, CpuBackend backend, int capacityBlocks) =>
        new(blockTokens: 16, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks, backend);

    private static IReadOnlyList<int> BuildLongPrompt(GgufTokenizer tokenizer)
    {
        const string line = "The quick brown fox jumps over the lazy dog while the patient cat watches the rain. ";
        List<int> ids = [];
        for (int round = 0; round < 8 && ids.Count < 40; round++)
        {
            ids.AddRange(tokenizer.Encode(line));
        }

        Assert.True(ids.Count >= 40, $"long prompt built only {ids.Count} tokens; a multi-block sequence is required");
        return ids;
    }

    /// <summary>Compares two float windows on the raw bits of the floats: NaN payloads and signed zeros must match too.</summary>
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

    // Spot-checks K/V slots across the prompt, the prefill/batch boundary, and the batched positions: every slot a
    // batched step wrote must hold the same bits the solo step wrote for the same sequence.
    private static void AssertKvSlotsEqual(SequenceKv batched, SequenceKv solo, int layerCount, int promptLength)
    {
        int[] positions = [0, promptLength - 1, promptLength + 1, batched.Length - 1];
        int[] layers = [0, layerCount / 2, layerCount - 1];

        foreach (int pos in positions)
        {
            foreach (int layer in layers)
            {
                AssertBitsEqual(solo.ReadK(pos, layer), batched.ReadK(pos, layer), $"K at position {pos} layer {layer}");
                AssertBitsEqual(solo.ReadV(pos, layer), batched.ReadV(pos, layer), $"V at position {pos} layer {layer}");
            }
        }
    }

    /// <summary>Picks the top id by descending logit with a strict "greater than" scan: equal logits keep the lower id.</summary>
    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        float best = float.NegativeInfinity;
        int bestIdx = -1;

        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > best)
            {
                best = logits[i];
                bestIdx = i;
            }
        }

        Assert.True(bestIdx >= 0, "logit vector had no selectable entry");
        return bestIdx;
    }
}
