using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
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
/// layers, while the batched prefill lands bit-for-bit on the causal reference. Every identity case runs once per
/// prefill kernel this build executes — the batched-row window and its row-tiled attention variant — so a tiled
/// window must land byte-for-byte on the per-token forward and on what the gemv route left, not merely close to it;
/// dedicated cases compare the two routes against each other directly. A synthetic case pins the tiled walk's causal
/// and selection guards at the kernel boundary itself: a row scores neither past its own position nor outside its
/// windowed selection, exactly as the per-row kernel never does.
/// </summary>
[Trait("Category", "Prefill")]
[Collection("model-loaded")] // shares the serialized real-model lane with the parity suites so these forwards do not load alongside another engine
public sealed class PrefillBatchIdentityTests
{
    private const int SmolLmDecodeSteps = 6;

    // The straddling window lengths both the per-kernel theories and the route-vs-route theory enumerate.
    private static readonly int[] StraddlingWindows = [1, 2, 15, 16, 17, 31, 32, 33, 40];

    /// <summary>Window lengths that straddle the engine's row-batch width: one row, a partial batch, exact batches, and a partial tail.</summary>
    public static TheoryData<int> WindowLengths() => [.. StraddlingWindows];

    /// <summary>The prefill kernels this build executes: the batched row window and its row-tiled attention variant.</summary>
    public static TheoryData<PrefillKernel> PrefillKernels() => [PrefillKernel.GemV, PrefillKernel.Tiled];

    /// <summary>Every straddling window length on each kernel this build executes.</summary>
    public static TheoryData<int, PrefillKernel> WindowLengthsAndKernels()
    {
        TheoryData<int, PrefillKernel> data = [];
        foreach (int tokens in StraddlingWindows)
        {
            data.Add(tokens, PrefillKernel.GemV);
            data.Add(tokens, PrefillKernel.Tiled);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(WindowLengthsAndKernels))]
    public void Prefill_LogitsAndKvBytes_MatchPerTokenForward_BitForBit(int windowTokens, PrefillKernel kernel)
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
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128, logLine: null, prefillKernel: kernel);
        Assert.Equal(kernel, model.ResolvedPrefillKernel); // the engine opened on the selection it was handed

        using BlockPool pool = NewPool(model, backend, capacityBlocks: 16);
        using SequenceKv batchedKv = new(pool);
        using SequenceKv referenceKv = new(pool);

        // Batched route: one window through the row-batched forward on this kernel.
        float[] batchedLogits = [.. model.Prefill(prompt, batchedKv)];

        // Reference route: the same tokens one at a time through the single-token forward. Its final
        // forward's logits are the window's final row.
        float[] referenceLogits = [];
        foreach (int tokenId in prompt)
        {
            referenceLogits = [.. model.DecodeNext(tokenId, referenceKv)];
        }

        AssertBitsEqual(referenceLogits, batchedLogits, $"prefill of a {windowTokens}-token window on the '{PrefillKernelNames.Of(kernel)}' kernel");
        Assert.Equal(ArgMax(referenceLogits), ArgMax(batchedLogits));

        Assert.Equal(referenceKv.Length, batchedKv.Length);
        AssertKvBytesEqual(referenceKv, batchedKv, model.Desc.LayerCount, windowTokens);
    }

    [Theory]
    [MemberData(nameof(PrefillKernels))]
    public void Prefill_ConsecutiveWindows_MatchOneWindowAndPerTokenForward_BitForBit(PrefillKernel kernel)
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
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128, logLine: null, prefillKernel: kernel);
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

        AssertBitsEqual(perTokenLogits, wholeLogits, $"one 40-token window on the '{PrefillKernelNames.Of(kernel)}' kernel against the per-token forward");
        AssertBitsEqual(perTokenLogits, splitLogits, $"consecutive 16/16/8-token windows on the '{PrefillKernelNames.Of(kernel)}' kernel against the per-token forward");

        Assert.Equal(perToken.Length, wholeWindow.Length);
        Assert.Equal(perToken.Length, splitWindows.Length);
        AssertKvBytesEqual(perToken, wholeWindow, model.Desc.LayerCount, prompt.Length);
        AssertKvBytesEqual(perToken, splitWindows, model.Desc.LayerCount, prompt.Length);
    }

    [Theory]
    [MemberData(nameof(PrefillKernels))]
    public void Prefill_FirstSampledTokenAndContinuation_MatchPerTokenForward(PrefillKernel kernel)
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
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128, logLine: null, prefillKernel: kernel);
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
            AssertBitsEqual(referenceStep, batchedStep, $"continuation step {step} after the prefill on the '{PrefillKernelNames.Of(kernel)}' kernel");
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

    [Theory]
    [MemberData(nameof(PrefillKernels))]
    public void Prefill_Llama32Iq4Xs_AcrossSubBatchesMatchesPerTokenForward(PrefillKernel kernel)
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
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128, logLine: null, prefillKernel: kernel);
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 16);
        using SequenceKv batchedKv = new(pool);
        using SequenceKv referenceKv = new(pool);

        float[] batchedLogits = [.. model.Prefill(prompt, batchedKv)];

        float[] referenceLogits = [];
        foreach (int tokenId in prompt)
        {
            referenceLogits = [.. model.DecodeNext(tokenId, referenceKv)];
        }

        AssertBitsEqual(referenceLogits, batchedLogits, $"a 33-token Llama-3.2 window spanning three row sub-batches on the '{PrefillKernelNames.Of(kernel)}' kernel");
        Assert.Equal(ArgMax(referenceLogits), ArgMax(batchedLogits));
        AssertKvBytesEqual(referenceKv, batchedKv, model.Desc.LayerCount, prompt.Length);
    }

    [Theory]
    [MemberData(nameof(PrefillKernels))]
    public void Prefill_WindowedGemma3_CausalSelectionInsideWindowMatchesPerTokenForward(PrefillKernel kernel)
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
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 64, logLine: null, prefillKernel: kernel);
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

        AssertBitsEqual(referenceLogits, batchedLogits, $"a windowed model's prefill window on the '{PrefillKernelNames.Of(kernel)}' kernel");
        Assert.Equal(ArgMax(referenceLogits), ArgMax(batchedLogits));
        AssertKvBytesEqual(referenceKv, batchedKv, model.Desc.LayerCount, prompt.Length);
    }

    // ---------------------------------------------------------------- route-vs-route identity: the two kernels against each other, not just against the per-token reference

    [Theory]
    [MemberData(nameof(WindowLengths))]
    public void Prefill_TiledRouteMatchesGemvRoute_BitForBit(int windowTokens)
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
        Assert.Equal(PrefillKernel.GemV, model.ResolvedPrefillKernel); // the engine opened on the default route

        using BlockPool pool = NewPool(model, backend, capacityBlocks: 32);
        using SequenceKv gemvKv = new(pool);
        float[] gemvLogits = [.. model.Prefill(prompt, gemvKv)];

        // Re-pointed onto the row-tiled route before a fresh window of the same tokens on its own sequence.
        model.ApplyPrefillKernel(PrefillKernel.Tiled);
        Assert.Equal(PrefillKernel.Tiled, model.ResolvedPrefillKernel);

        using SequenceKv tiledKv = new(pool);
        float[] tiledLogits = [.. model.Prefill(prompt, tiledKv)];

        // Byte-for-byte against the gemv route's output — not a tolerance: same logits, same first sampled token.
        AssertBitsEqual(gemvLogits, tiledLogits, $"a {windowTokens}-token window on the row-tiled route against the gemv route");
        int gemvFirst = ArgMax(gemvLogits);
        int tiledFirst = ArgMax(tiledLogits);
        Assert.Equal(gemvFirst, tiledFirst);

        // Every position of every layer's K/V slot compared on raw bytes too.
        AssertKvBytesEqual(gemvKv, tiledKv, model.Desc.LayerCount, windowTokens);

        // And the sampled token riding into a decode continuation keeps both routes byte-identical for six steps.
        int gemvNext = gemvFirst;
        int tiledNext = tiledFirst;
        for (int step = 0; step < SmolLmDecodeSteps; step++)
        {
            float[] gemvStep = [.. model.DecodeNext(gemvNext, gemvKv)];
            float[] tiledStep = [.. model.DecodeNext(tiledNext, tiledKv)];
            AssertBitsEqual(gemvStep, tiledStep, $"decode step {step} after the two routes' prefills");
            gemvNext = ArgMax(gemvStep);
            tiledNext = ArgMax(tiledStep);
        }

        Assert.Equal(gemvNext, tiledNext); // the sampled continuations stay identical all the way through
    }

    [Fact]
    public void Prefill_TiledRouteMatchesGemvRoute_ConsecutiveWindows_BitForBit()
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
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 32);

        float[] gemvLogits = [];
        using SequenceKv gemvKv = new(pool);
        for (int offset = 0; offset < prompt.Length; offset += 16)
        {
            int rows = Math.Min(16, prompt.Length - offset);
            gemvLogits = [.. model.Prefill(prompt.AsSpan(offset, rows).ToArray(), gemvKv)];
        }

        model.ApplyPrefillKernel(PrefillKernel.Tiled);

        float[] tiledLogits = [];
        using SequenceKv tiledKv = new(pool);
        for (int offset = 0; offset < prompt.Length; offset += 16)
        {
            int rows = Math.Min(16, prompt.Length - offset);
            tiledLogits = [.. model.Prefill(prompt.AsSpan(offset, rows).ToArray(), tiledKv)];
        }

        AssertBitsEqual(gemvLogits, tiledLogits, "consecutive 16/16/8-token windows on the row-tiled route against the gemv route");
        Assert.Equal(ArgMax(gemvLogits), ArgMax(tiledLogits));
        AssertKvBytesEqual(gemvKv, tiledKv, model.Desc.LayerCount, prompt.Length);
    }

    [Fact]
    public void Prefill_TiledRouteMatchesGemvRoute_WindowedGemma3_BitForBit()
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
        using SequenceKv gemvKv = new(pool);
        float[] gemvLogits = [.. model.Prefill(prompt, gemvKv)];

        // The windowed layers are where the tiled walk's per-row selection guard earns its keep: re-point and forward again.
        model.ApplyPrefillKernel(PrefillKernel.Tiled);

        using SequenceKv tiledKv = new(pool);
        float[] tiledLogits = [.. model.Prefill(prompt, tiledKv)];

        AssertBitsEqual(gemvLogits, tiledLogits, "a windowed model's prefill on the row-tiled route against the gemv route");
        Assert.Equal(ArgMax(gemvLogits), ArgMax(tiledLogits));
        AssertKvBytesEqual(gemvKv, tiledKv, model.Desc.LayerCount, prompt.Length);
    }

    [Fact]
    public void Prefill_TiledRouteMatchesGemvRoute_Llama32Iq4Xs_BitForBit()
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
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 32);

        float[] gemvLogits = [];
        using SequenceKv gemvKv = new(pool);
        for (int offset = 0; offset < prompt.Length; offset += 16)
        {
            int rows = Math.Min(16, prompt.Length - offset);
            gemvLogits = [.. model.Prefill(prompt.AsSpan(offset, rows).ToArray(), gemvKv)];
        }

        model.ApplyPrefillKernel(PrefillKernel.Tiled);

        float[] tiledLogits = [];
        using SequenceKv tiledKv = new(pool);
        for (int offset = 0; offset < prompt.Length; offset += 16)
        {
            int rows = Math.Min(16, prompt.Length - offset);
            tiledLogits = [.. model.Prefill(prompt.AsSpan(offset, rows).ToArray(), tiledKv)];
        }

        AssertBitsEqual(gemvLogits, tiledLogits, "a 33-token Llama-3.2 window spanning three row sub-batches on the row-tiled route against the gemv route");
        Assert.Equal(ArgMax(gemvLogits), ArgMax(tiledLogits));
        AssertKvBytesEqual(gemvKv, tiledKv, model.Desc.LayerCount, prompt.Length);
    }

    // ---------------------------------------------------------------- kernel-boundary guards: what a row may and may not touch in the tiled walk

    [Fact]
    public void CausalAttentionTiledRows_NeverScoresPastRowPositionAndNeverReadsOutsideItsSelection()
    {
        using CpuBackend backend = new();
        using BlockPool pool = new(blockTokens: 3, layerCount: 1, kvHeads: 2, headDim: 4, capacityBlocks: 8, backend);
        using SequenceKv kv = new(pool);

        // Eight positions filled with a recognizable pattern — distinct per position and kv-head, signed so no value can masquerade as an unread zero. Four consecutive rows start at absolute position 4; under a window of three each row selects [p-2..p], the union is [2..7], and that walk spans all three pool blocks with partial edges on both sides. Row 0 (position 4) must never touch positions 5, 6 or 7 — its future; rows past position 4 must never read position 2, which lies behind their own selection's start.
        for (int position = 0; position < 8; position++)
        {
            float[] kVec = new float[8]; // kvHeads * headDim
            float[] vVec = new float[8];
            for (int h = 0; h < 2; h++)
            {
                for (int j = 0; j < 4; j++)
                {
                    kVec[h * 4 + j] = 10f * (h + 1) + position - 3.5f;
                    vVec[h * 4 + j] = -(10f * (h + 1)) - position;
                }
            }

            kv.Append();
            kv.WriteK(position, 0, kVec);
            kv.WriteV(position, 0, vVec);
        }

        const int rows = 4;
        const int headCount = 4; // grouped-query mapping: two heads per kv-head — the walk's grouping must pair each query with exactly its assigned kv-head slice
        const int headDim = 4;
        const int region = headCount * headDim;

        float[] qRows = new float[rows * region];
        for (int r = 0; r < rows; r++)
        {
            for (int h = 0; h < headCount; h++)
            {
                for (int j = 0; j < headDim; j++)
                {
                    qRows[r * region + h * headDim + j] = (r - 1) * 0.5f + (h - 2) * 0.25f + j * 0.75f; // signed and varying: every dot product is non-trivial, so a stray read moves the row's max or accumulator
                }
            }
        }

        foreach (int window in new int[] { 3, 0 }) // a narrowed walk once and an unbounded one: the causal guard stands on its own too
        {
            float[] tiledOut = new float[rows * region];
            LayerOps.CausalAttentionTiledRows(
                backend, qRows, fromPosition: 4, rows, headCount, kvHeadCount: 2, headDim, kv, layer: 0, attnScale: 1f, softcap: 0f, window,
                runningMaxRowState: new float[rows * headCount], sumExpRowState: new float[rows * headCount], tiledOut);

            // The per-row kernel resolves each row's own selection through SequenceKv — a read outside it fails inside that pass. Landing on those bytes therefore proves the tiled walk scored exactly [max(0, p - window + 1)..p] for every row: no position past its own, none behind its window start, and nothing outside any row's selection is ever addressed by the unioned block walk either — a stray read would move that row's running max or weight sum before normalization.
            for (int r = 0; r < rows; r++)
            {
                float[] perRowOut = new float[region];
                LayerOps.CausalAttention(
                    backend, qRows.AsSpan(r * region, region), headCount, kvHeadCount: 2, headDim, kv, layer: 0, position: 4 + r, attnScale: 1f, softcap: 0f, window, perRowOut);

                AssertBitsEqual(
                    perRowOut,
                    tiledOut.AsSpan(r * region, region),
                    $"row {r} of the tiled walk against its own causal pass at position {4 + r}, window {window}");
            }

            // Direct evidence of both guards on row 0 and row 3: their online state must reflect exactly their own range. Row 0's head-0 output is a convex combination (weights sum to one after normalization) over its selection's V values only — every component lies between the smallest and largest V value of the positions it may read, so reaching past position 4 (its future) or behind its window start would pull in a magnitude outside that envelope. An unbounded row reads from position 0; under a positive window only from max(0, p - window + 1).
            for (int j = 0; j < headDim; j++)
            {
                float firstVisible = window > 0 ? Math.Max(0, 4 - window + 1) : 0;

                float vMin = float.PositiveInfinity;
                float vMax = float.NegativeInfinity;
                for (int position = (int)firstVisible; position <= 4; position++)
                {
                    float value = kv.ReadV(position, 0)[j]; // head 0 maps to kv-head 0 at offset j
                    vMin = Math.Min(vMin, value);
                    vMax = Math.Max(vMax, value);
                }

                Assert.True(
                    tiledOut[j] >= vMin && tiledOut[j] <= vMax,
                    $"row 0 head-0 output component {j} is {tiledOut[j]}, outside the [{vMin}, {vMax}] envelope its selection at window {window} allows — a read past position 4 or behind {firstVisible} would land here");

                // Row 3 (position 7) under a positive window reads only [5..7]: components of an out-of-selection read sit outside that row's own V envelope.
                if (window > 0)
                {
                    float vMinLast = Math.Min(kv.ReadV(5, 0)[j], Math.Min(kv.ReadV(6, 0)[j], kv.ReadV(7, 0)[j]));
                    float vMaxLast = Math.Max(kv.ReadV(5, 0)[j], Math.Max(kv.ReadV(6, 0)[j], kv.ReadV(7, 0)[j]));

                    Assert.True(
                        tiledOut[(rows - 1) * region + j] >= vMinLast && tiledOut[(rows - 1) * region + j] <= vMaxLast,
                        $"row 3 head-0 output component {j} is {tiledOut[(rows - 1) * region + j]}, outside the [{vMinLast}, {vMaxLast}] envelope its selection at window {window} allows — a read behind position 5 would land here");
                }
            }
        }
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
