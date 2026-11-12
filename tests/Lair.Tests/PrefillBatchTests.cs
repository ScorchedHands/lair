using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Prefill-identity tests over the pinned real-model fixtures: a prompt forwarded as a batched row window must
/// leave the same logits, the same KV bytes at every position of every layer, and the same first sampled token as
/// the same tokens forwarded one at a time. The engine exposes no per-token prefill entry point, so the reference
/// inside this harness forwards the window through the single-token forward — <see cref="TransformerModel.DecodeNext"/>
/// at each next free position — which is exactly the route the prefill path used to take. Window lengths straddle
/// the engine's row-batch width so a sequence spanning several sub-batches is compared, not only one that fits in
/// a single one, and the causal read set is proven against the encoder's bidirectional shape: the same tokens
/// through <see cref="TransformerModel.Encode"/> leave different KV bytes at the interior positions of the deeper
/// layers, while the batched prefill lands bit-for-bit on the causal reference.
/// </summary>
[Trait("Category", "Prefill")]
[Collection("model-loaded")] // shares the serialized real-model lane with the parity suites so these forwards do not load alongside another engine
public sealed class PrefillBatchIdentityTests
{
    private const int SmolLmDecodeSteps = 6;

    /// <summary>Window lengths that straddle the engine's row-batch width: one row, a partial batch, exact batches, and a partial tail.</summary>
    public static TheoryData<int> WindowLengths() => [1, 2, 15, 16, 17, 31, 32, 33, 40];

    [Theory]
    [MemberData(nameof(WindowLengths))]
    public void Prefill_LogitsAndKvBytes_MatchPerTokenForward_BitForBit(int windowTokens)
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        int[] prompt = BuildPrompt(tokenizer, windowTokens);

        using CpuBackend backend = new(threads: 4, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128);
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 16);
        using SequenceKv batchedKv = new(pool);
        using SequenceKv referenceKv = new(pool);

        // Batched route: one window through the row-batched forward.
        float[] batchedLogits = [.. model.Prefill(prompt, batchedKv)];

        // Reference route: the same tokens one at a time through the single-token forward. Its final
        // forward's logits are the window's final row.
        float[] referenceLogits = [];
        foreach (int tokenId in prompt)
        {
            referenceLogits = [.. model.DecodeNext(tokenId, referenceKv)];
        }

        AssertBitsEqual(referenceLogits, batchedLogits, $"prefill of a {windowTokens}-token window");
        Assert.Equal(ArgMax(referenceLogits), ArgMax(batchedLogits));

        Assert.Equal(referenceKv.Length, batchedKv.Length);
        AssertKvBytesEqual(referenceKv, batchedKv, model.Desc.LayerCount, windowTokens);
    }

    [Fact]
    public void Prefill_ConsecutiveWindows_MatchOneWindowAndPerTokenForward_BitForBit()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        int[] prompt = BuildPrompt(tokenizer, 40);

        using CpuBackend backend = new(threads: 4, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128);
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 16);
        using SequenceKv wholeWindow = new(pool);
        using SequenceKv splitWindows = new(pool);
        using SequenceKv perToken = new(pool);

        float[] wholeLogits = [.. model.Prefill(prompt, wholeWindow)];

        // The chunked route the scheduler issues: consecutive windows of 16, 16, and 8 tokens.
        float[] splitLogits = [];
        for (int offset = 0; offset < prompt.Length; offset += 16)
        {
            int rows = Math.Min(16, prompt.Length - offset);
            splitLogits = [.. model.Prefill(prompt.AsSpan(offset, rows).ToArray(), splitWindows)];
        }

        float[] perTokenLogits = [];
        foreach (int tokenId in prompt)
        {
            perTokenLogits = [.. model.DecodeNext(tokenId, perToken)];
        }

        AssertBitsEqual(perTokenLogits, wholeLogits, "one 40-token window against the per-token forward");
        AssertBitsEqual(perTokenLogits, splitLogits, "consecutive 16/16/8-token windows against the per-token forward");

        Assert.Equal(perToken.Length, wholeWindow.Length);
        Assert.Equal(perToken.Length, splitWindows.Length);
        AssertKvBytesEqual(perToken, wholeWindow, model.Desc.LayerCount, prompt.Length);
        AssertKvBytesEqual(perToken, splitWindows, model.Desc.LayerCount, prompt.Length);
    }

    [Fact]
    public void Prefill_FirstSampledTokenAndContinuation_MatchPerTokenForward()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        int[] prompt = BuildPrompt(tokenizer, 24);

        using CpuBackend backend = new(threads: 4, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128);
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 16);
        using SequenceKv batchedKv = new(pool);
        using SequenceKv referenceKv = new(pool);

        float[] batchedLogits = [.. model.Prefill(prompt, batchedKv)];

        float[] referenceLogits = [];
        foreach (int tokenId in prompt)
        {
            referenceLogits = [.. model.DecodeNext(tokenId, referenceKv)];
        }

        int batchedFirst = ArgMax(batchedLogits);
        int referenceFirst = ArgMax(referenceLogits);
        Assert.Equal(referenceFirst, batchedFirst);

        // The sampled token rides into the next forward on both routes: the continuations must stay bit-identical.
        int batchedNext = batchedFirst;
        int referenceNext = referenceFirst;
        for (int step = 0; step < SmolLmDecodeSteps; step++)
        {
            float[] batchedStep = [.. model.DecodeNext(batchedNext, batchedKv)];
            float[] referenceStep = [.. model.DecodeNext(referenceNext, referenceKv)];
            AssertBitsEqual(referenceStep, batchedStep, $"continuation step {step} after the prefill");
            batchedNext = ArgMax(batchedStep);
            referenceNext = ArgMax(referenceStep);
        }

        AssertKvBytesEqual(referenceKv, batchedKv, model.Desc.LayerCount, prompt.Length + SmolLmDecodeSteps);
    }

    [Fact]
    public void Prefill_KvBytesDifferFromBidirectionalEncoderForward_CausalWithinWindowProven()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        int[] prompt = BuildPrompt(tokenizer, 20);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 64);
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 16);
        using SequenceKv prefillKv = new(pool);
        using SequenceKv encodeKv = new(pool);
        using SequenceKv referenceKv = new(pool);

        model.Prefill(prompt, prefillKv);
        model.Encode(prompt, encodeKv);

        float[] referenceLogits = [];
        foreach (int tokenId in prompt)
        {
            referenceLogits = [.. model.DecodeNext(tokenId, referenceKv)];
        }

        // The prefill's KV is the causal one: identical to the per-token forward at every position and layer.
        AssertKvBytesEqual(referenceKv, prefillKv, model.Desc.LayerCount, prompt.Length);

        // The encoder's KV is a different computation where the two attention shapes can disagree: position 0 and the
        // window's last position have a causal range that equals the encoder's whole-span range, and the first
        // layer's K/V come from the embedding stream before any attention ran, so those slots agree by
        // construction. Everywhere else the encoder scored a query against positions past itself while the causal
        // prefill did not, and a prefill that had run the encoder's shape would not have landed on the causal
        // reference above.
        int candidates = 0;
        int differing = 0;
        for (int position = 1; position < prompt.Length - 1; position++)
        {
            for (int layer = 1; layer < model.Desc.LayerCount; layer++)
            {
                candidates += 2;
                if (!BytesEqual(encodeKv.ReadK(position, layer), prefillKv.ReadK(position, layer)))
                {
                    differing++;
                }

                if (!BytesEqual(encodeKv.ReadV(position, layer), prefillKv.ReadV(position, layer)))
                {
                    differing++;
                }
            }
        }

        Assert.True(
            candidates > 0 && differing >= candidates - 1,
            $"a causal prefill and a bidirectional encode moved {differing} of {candidates} interior K/V slots; the " +
            "prefill's attention shape is not proven distinct from the encoder's");
    }

    [Fact]
    public void Prefill_Llama32Iq4Xs_AcrossSubBatchesMatchesPerTokenForward()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative("models/Llama-3.2-1B-Instruct-IQ4_XS.gguf");
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        int[] prompt = BuildPrompt(tokenizer, 33);

        using CpuBackend backend = new(threads: 4, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128);
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 16);
        using SequenceKv batchedKv = new(pool);
        using SequenceKv referenceKv = new(pool);

        float[] batchedLogits = [.. model.Prefill(prompt, batchedKv)];

        float[] referenceLogits = [];
        foreach (int tokenId in prompt)
        {
            referenceLogits = [.. model.DecodeNext(tokenId, referenceKv)];
        }

        AssertBitsEqual(referenceLogits, batchedLogits, "a 33-token Llama-3.2 window spanning three row sub-batches");
        Assert.Equal(ArgMax(referenceLogits), ArgMax(batchedLogits));
        AssertKvBytesEqual(referenceKv, batchedKv, model.Desc.LayerCount, prompt.Length);
    }

    [Fact]
    public void Prefill_WindowedGemma3_CausalSelectionInsideWindowMatchesPerTokenForward()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative("models/gemma-3-1b-it-Q3_K_S.gguf");
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        int[] prompt = BuildPrompt(tokenizer, 20);

        using CpuBackend backend = new(threads: 4, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 64);
        Assert.True(model.Plan.AttentionWindowForLayer(0) > 0, "the gemma3 plan must resolve a windowed layer for this case to mean anything");

        using BlockPool pool = NewPool(model, backend, capacityBlocks: 16);
        using SequenceKv batchedKv = new(pool);
        using SequenceKv referenceKv = new(pool);

        float[] batchedLogits = [.. model.Prefill(prompt, batchedKv)];

        float[] referenceLogits = [];
        foreach (int tokenId in prompt)
        {
            referenceLogits = [.. model.DecodeNext(tokenId, referenceKv)];
        }

        AssertBitsEqual(referenceLogits, batchedLogits, "a windowed model's prefill window");
        Assert.Equal(ArgMax(referenceLogits), ArgMax(batchedLogits));
        AssertKvBytesEqual(referenceKv, batchedKv, model.Desc.LayerCount, prompt.Length);
    }

    [Fact]
    public void Prefill_Validation_RejectsEmptyWindowBadTokenAndOverflowBeforeAnyAppend()
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

        Assert.Throws<ArgumentNullException>(() => model.Prefill(null!, kv));
        Assert.Throws<ArgumentNullException>(() => model.Prefill([1], null!));

        Assert.Throws<ArgumentException>(() => model.Prefill(new int[0], kv));
        Assert.Equal(0, kv.Length); // the rejected window appended nothing

        Assert.Throws<ArgumentOutOfRangeException>(() => model.Prefill([1, model.Desc.VocabSize], kv));
        Assert.Equal(0, kv.Length); // the rejected token id reserved nothing

        Assert.Throws<InvalidOperationException>(() => model.Prefill(Enumerable.Repeat(1, 33).ToArray(), kv));
        Assert.Equal(0, kv.Length); // a window past the max context failed before any slot was reserved
    }

    // ---------------------------------------------------------------- helpers

    private static BlockPool NewPool(TransformerModel model, CpuBackend backend, int capacityBlocks) =>
        new(blockTokens: 16, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks, backend);

    // A prompt of exactly the requested token count, built from the repeated sentence this suite already uses and
    // trimmed to length: the assertion pins that the window really spans the sub-batch widths the case names.
    private static int[] BuildPrompt(GgufTokenizer tokenizer, int tokens)
    {
        const string line = "The quick brown fox jumps over the lazy dog while the patient cat watches the rain. ";

        List<int> ids = [];
        while (ids.Count < tokens)
        {
            ids.AddRange(tokenizer.Encode(line));
        }

        Assert.True(ids.Count >= tokens, $"prompt build produced {ids.Count} tokens; expected at least {tokens}");
        return [.. ids.Take(tokens)];
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

    private static bool BytesEqual(ReadOnlySpan<float> left, ReadOnlySpan<float> right) =>
        MemoryMarshal.AsBytes(left).SequenceEqual(MemoryMarshal.AsBytes(right));

    // Every position of the prompt and every layer: the whole KV a window wrote, compared slot by slot. A row whose
    // attention had reached a position past its own would move its own K/V bytes at the next layer, and a window
    // boundary that dropped or duplicated a position would show up as a length or slot mismatch here.
    private static void AssertKvBytesEqual(SequenceKv reference, SequenceKv batched, int layerCount, int positions)
    {
        Assert.Equal(positions, batched.Length);

        for (int position = 0; position < positions; position++)
        {
            for (int layer = 0; layer < layerCount; layer++)
            {
                AssertBitsEqual(reference.ReadK(position, layer), batched.ReadK(position, layer), $"K at position {position} layer {layer}");
                AssertBitsEqual(reference.ReadV(position, layer), batched.ReadV(position, layer), $"V at position {position} layer {layer}");
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
