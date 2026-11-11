using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Forward-level goldens for the declared attention window: a synthetic model with a small window whose logits
/// at a query position beyond the window match a double-precision scalar reference forward that applies the same
/// score mask, and demonstrably differ from the same model with the mask removed; a window at or beyond the
/// sequence length producing logits bit-identical to the same model with no window declared; batched decode and
/// single-token decode bit-identical under a window; the <see cref="TransformerModel.Encode"/> path narrowing each
/// query backward-only — matching the backward-only masked reference across a chunk boundary, differing from the
/// unmasked bidirectional forward at every position whose span it narrows, leaving position 0 with no forward
/// visibility, and keeping an undeclared window bit-identical to the full-sequence path; the mask applied together
/// with an attention-score softcap against a reference doing both; and kernel-level goldens showing the causal range
/// narrowing, the backward-only narrowing around each query — identical to the causal kernel's own output for the
/// same position and window — and that every masked-out position stays readable in the KV cache. Engine-to-engine
/// comparisons are asserted bit-for-bit; reference comparisons use the double-precision tolerance shared by the
/// strategy goldens. Encoder cases carry both the Layers and Embeddings categories.
/// </summary>
[Trait("Category", "Layers")]
public sealed class SlidingWindowLayerTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

    // The window the synthetic model declares: three positions, so a query at position 5 already sees [3..5]
    // instead of [0..5] and the mask moves the output by a detectable amount. The cap is paired with the window
    // in the combined test: the q/k rows put scores at the unit scale and the cap at a quarter of it.
    private const int Window = 3;
    private const float AttentionCap = 0.25f;

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

    // ---------------------------------------------------------------- decode + encode goldens

    [Fact]
    [Trait("Category", "Embeddings")]
    public void Forward_SlidingWindow_MatchesMaskedReference_DiffersFromUnmasked_BatchMatchesSingle()
    {
        WindowModelSpec spec = new();
        string path = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.attention.sliding_window", Window)],
            spec.Tensors()));

        WindowReference reference = new(spec);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        Assert.Equal(Window, model.Plan.AttentionWindow);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 8, backend);
        using SequenceKv kv = new(pool);

        // Inside the window the mask is inert: at positions 0..2 the causal range already fits, so the masked
        // and unmasked references agree with the engine bit-for-bit-equal targets.
        Assert.True(
            Closest(model.DecodeNext(1, kv), reference.LogitsAt([], 1, window: Window)),
            "windowed decode logits at position 0 must match the masked reference forward");

        Assert.True(
            Closest(model.DecodeNext(4, kv), reference.LogitsAt([1], 4, window: Window)),
            "windowed decode logits at position 1 must match the masked reference forward");

        Assert.True(
            Closest(model.DecodeNext(2, kv), reference.LogitsAt([1, 4], 2, window: Window)),
            "windowed decode logits at position 2 must match the masked reference forward");

        // Inside the window the masked reference and the unmasked reference are the same forward.
        Assert.True(
            Closest(model.LastLogits(), reference.LogitsAt([1, 4], 2, window: 0)),
            "a window wider than the causal prefix must leave the position-2 logits equal to the unmasked reference");

        // Beyond the window: position 5 attends [3..5] only, and the unmasked forward is demonstrably different.
        model.Prefill([5, 3], kv);
        float[] maskedLogits = model.DecodeNext(0, kv).ToArray();

        Assert.True(
            Closest(maskedLogits, reference.LogitsAt([1, 4, 2, 5, 3], 0, window: Window)),
            "windowed decode logits at position 5 must match the masked reference forward");

        Assert.False(
            Closest(maskedLogits, reference.LogitsAt([1, 4, 2, 5, 3], 0, window: 0)),
            "a window narrower than the causal prefix must not leave the logits equal to the unmasked reference");

        // Encode: a positive window narrows each query backward-only to [max(0, p - window + 1), p], so every position
        // short of the sequence end loses both its deep history and everything after it.
        using SequenceKv encodeKv = new(pool);
        float[] encoded = model.Encode([1, 4, 2, 5, 3], encodeKv);

        Assert.Equal(5 * model.Plan.HiddenSize, encoded.Length);
        double[][] expectedEncoded = reference.Encoded([1, 4, 2, 5, 3], window: Window);
        for (int p = 0; p < 5; p++)
        {
            for (int i = 0; i < model.Plan.HiddenSize; i++)
            {
                float actual = encoded[p * model.Plan.HiddenSize + i];
                double expected = expectedEncoded[p][i];
                Assert.True(
                    Math.Abs(actual - expected) <= 1e-5 * Math.Max(1.0, Math.Abs(expected)),
                    $"encode hidden[{p}][{i}] under the window: expected {expected:G9}, got {actual:G9}");
            }
        }

        Assert.True(
            Differs(encoded, reference.Encoded([1, 4, 2, 5, 3], window: 0)),
            "the windowed encode rows must differ from the unmasked bidirectional forward");

        // Batched decode must be bit-identical to single-token decode under the window: the same six tokens
        // stepped one per batched call end on the same token at the same position as the prefilled sequence.
        using SequenceKv single = new(pool);
        model.Prefill([1, 4, 2, 5, 3], single);
        float[] singleLogits = model.LastLogits().ToArray();

        using SequenceKv batched = new(pool);
        model.DecodeBatch([1], [batched]);
        model.DecodeBatch([4], [batched]);
        model.DecodeBatch([2], [batched]);
        model.DecodeBatch([5], [batched]);
        Span<float> batchLogits = model.DecodeBatch([3], [batched]);
        Assert.True(
            batchLogits.SequenceEqual(singleLogits.AsSpan()),
            "a token batched through DecodeBatch under the window must be bit-identical to stepping the same sequence alone");
    }

    [Fact]
    [Trait("Category", "Embeddings")]
    public void Forward_WindowAtOrBeyondSequenceLength_DecodesBitIdenticallyToWindowless_EncodeStaysBackwardOnly()
    {
        WindowModelSpec spec = new();
        byte[] image = BuildGguf(spec.MetadataPairs(), spec.Tensors());

        // Window exactly the sequence length (six positions once the query token is appended), a window beyond
        // it, and no window at all: on the causal decode path the mask never excludes a position, so every forward there
        // is bit-for-bit the same computation. On the encoder path a positive window still ends at its own query, so only
        // the final row — whose backward span already covers the whole sequence — is left unchanged by declaring one.
        string exactPath = WriteTemp(BuildGguf([.. spec.MetadataPairs(), ("llama.attention.sliding_window", 6)], spec.Tensors()));
        string widePath = WriteTemp(BuildGguf([.. spec.MetadataPairs(), ("llama.attention.sliding_window", 9)], spec.Tensors()));
        string nonePath = WriteTemp(image);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile exactGguf = GgufFile.Open(exactPath);
        using GgufFile wideGguf = GgufFile.Open(widePath);
        using GgufFile noneGguf = GgufFile.Open(nonePath);
        using TransformerModel exactModel = TransformerModel.Open(exactGguf, backend, maxContextTokens: 8);
        using TransformerModel wideModel = TransformerModel.Open(wideGguf, backend, maxContextTokens: 8);
        using TransformerModel noneModel = TransformerModel.Open(noneGguf, backend, maxContextTokens: 8);

        Assert.Equal(6, exactModel.Plan.AttentionWindow);
        Assert.Equal(9, wideModel.Plan.AttentionWindow);
        Assert.Equal(0, noneModel.Plan.AttentionWindow);

        using BlockPool pool = new(blockTokens: 4, layerCount: exactModel.Plan.LayerCount, kvHeads: exactModel.Plan.KvHeadCount, headDim: exactModel.Plan.HeadDim, capacityBlocks: 16, backend);
        using SequenceKv exactKv = new(pool);
        using SequenceKv wideKv = new(pool);
        using SequenceKv noneKv = new(pool);

        exactModel.Prefill([1, 4, 2, 5, 3], exactKv);
        wideModel.Prefill([1, 4, 2, 5, 3], wideKv);
        noneModel.Prefill([1, 4, 2, 5, 3], noneKv);

        float[] exactLogits = exactModel.DecodeNext(0, exactKv).ToArray();
        float[] wideLogits = wideModel.DecodeNext(0, wideKv).ToArray();
        float[] noneLogits = noneModel.DecodeNext(0, noneKv).ToArray();

        Assert.True(
            exactLogits.AsSpan().SequenceEqual(noneLogits.AsSpan()),
            "a window equal to the sequence length must decode bit-identically to the same model with no window declared");
        Assert.True(
            wideLogits.AsSpan().SequenceEqual(noneLogits.AsSpan()),
            "a window beyond the sequence length must decode bit-identically to the same model with no window declared");

        using SequenceKv exactEncodeKv = new(pool);
        using SequenceKv wideEncodeKv = new(pool);
        using SequenceKv noneEncodeKv = new(pool);
        float[] exactEncoded = exactModel.Encode([1, 4, 2, 5, 3], exactEncodeKv);
        float[] wideEncoded = wideModel.Encode([1, 4, 2, 5, 3], wideEncodeKv);
        float[] noneEncoded = noneModel.Encode([1, 4, 2, 5, 3], noneEncodeKv);

        int hidden = exactModel.Plan.HiddenSize;

        // An exact window and an over-wide one resolve the same backward spans, so their encodes agree bit-for-bit.
        Assert.True(
            exactEncoded.AsSpan().SequenceEqual(wideEncoded.AsSpan()),
            "a window equal to the sequence length and a window beyond it must encode identically: both narrow every query to its own backward span");

        // Every position short of the end loses forward visibility under a positive window, even one that covers all of
        // its history, so those rows move away from the same model with no window declared. The final row's backward
        // span already reaches back over the whole sequence and has nothing after it: there both paths coincide.
        for (int p = 0; p < 5; p++)
        {
            var windowedRow = exactEncoded.AsSpan(p * hidden, hidden);
            var unwindowedRow = noneEncoded.AsSpan(p * hidden, hidden);

            if (p == 4)
            {
                Assert.True(
                    BitsEqual(windowedRow, unwindowedRow),
                    $"encode row {p}: the final position's backward span covers the whole sequence, so its windowed row must match the windowless model bit-for-bit");
            }
            else
            {
                Assert.True(
                    AnyEntryDiffers(windowedRow, unwindowedRow),
                    $"encode row {p}: a positive encoder window must remove forward visibility even when it covers every earlier position");
            }
        }

        // The same five rows against the backward-only scalar reference: exact match at every entry.
        double[][] expectedBackwardOnly = new WindowReference(spec).Encoded([1, 4, 2, 5, 3], window: 6);
        for (int p = 0; p < 5; p++)
        {
            for (int i = 0; i < hidden; i++)
            {
                double expectedValue = expectedBackwardOnly[p][i];
                Assert.True(
                    Math.Abs(exactEncoded[p * hidden + i] - expectedValue) <= 1e-5 * Math.Max(1.0, Math.Abs(expectedValue)),
                    $"encode hidden[{p}][{i}] under a window that covers the whole sequence: expected {expectedValue:G9}, got {exactEncoded[p * hidden + i]:G9}");
            }
        }
    }

    [Fact]
    [Trait("Category", "Embeddings")]
    public void Forward_WindowAcrossEncodeChunks_MatchesMaskedReference_RegardlessOfChunking()
    {
        // Ten positions span two encoder chunks: the mask is applied as a backward-only range ending at each query's own
        // absolute position, so the result must match the unchunked scalar reference across the chunk boundary.
        WindowModelSpec spec = new();
        string path = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.attention.sliding_window", Window)],
            spec.Tensors()));

        WindowReference reference = new(spec);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 16);

        int[] tokens = [1, 4, 2, 5, 3, 0, 1, 5, 2, 4];

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 8, backend);
        using SequenceKv kv = new(pool);

        float[] encoded = model.Encode(tokens, kv);
        double[][] expected = reference.Encoded(tokens, window: Window);

        for (int p = 0; p < tokens.Length; p++)
        {
            for (int i = 0; i < model.Plan.HiddenSize; i++)
            {
                float actual = encoded[p * model.Plan.HiddenSize + i];
                double expectedValue = expected[p][i];
                Assert.True(
                    Math.Abs(actual - expectedValue) <= 1e-5 * Math.Max(1.0, Math.Abs(expectedValue)),
                    $"encode hidden[{p}][{i}] across the chunk boundary under the window: expected {expectedValue:G9}, got {actual:G9}");
            }
        }

        Assert.True(Differs(encoded, reference.Encoded(tokens, window: 0)), "the windowed encode must differ from the unmasked bidirectional forward");
    }

    [Fact]
    [Trait("Category", "Embeddings")]
    public void Encode_PositiveWindow_LeavesPositionZeroNoForwardVisibility_MatchingItsOwnSingleTokenRun()
    {
        WindowModelSpec spec = new();
        string windowedPath = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.attention.sliding_window", Window)],
            spec.Tensors()));
        string unwindowedPath = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors()));

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile windowedGguf = GgufFile.Open(windowedPath);
        using GgufFile unwindowedGguf = GgufFile.Open(unwindowedPath);
        using TransformerModel windowed = TransformerModel.Open(windowedGguf, backend, maxContextTokens: 8);
        using TransformerModel unwindowed = TransformerModel.Open(unwindowedGguf, backend, maxContextTokens: 8);

        int hidden = windowed.Plan.HiddenSize;

        using BlockPool pool = new(blockTokens: 4, layerCount: windowed.Plan.LayerCount, kvHeads: windowed.Plan.KvHeadCount, headDim: windowed.Plan.HeadDim, capacityBlocks: 16, backend);

        float[] windowedRows;
        float[] unwindowedRows;
        float[] windowedSoloRow;
        {
            using SequenceKv kv = new(pool);
            windowedRows = windowed.Encode([1, 4, 2, 5, 3], kv);

            // The same weights with no declared window: position 0 attends forward over the four tokens after it.
            using (SequenceKv unwindowedKv = new(pool))
            {
                unwindowedRows = unwindowed.Encode([1, 4, 2, 5, 3], unwindowedKv);
            }

            // Each model's own single-token run at position 0: the causal-prefix reference for that position. A window
            // has nothing to mask there, so both models produce it identically.
            using (SequenceKv soloKv = new(pool))
            {
                windowedSoloRow = windowed.Encode([1], soloKv);
            }
        }

        float[] unwindowedSoloRow;

        using (SequenceKv unwindowedSoloKv = new(pool))
        {
            unwindowedSoloRow = unwindowed.Encode([1], unwindowedSoloKv);
        }

        Assert.True(
            BitsEqual(windowedRows.AsSpan(0, hidden), windowedSoloRow),
            "position 0 under a positive window sees only itself: its row must match the same model encoding that token alone, bit-for-bit");
        Assert.True(
            BitsEqual(unwindowedSoloRow, windowedSoloRow),
            "a single-token encode has no later position to attend, so declaring a window cannot change it");

        Assert.True(
            AnyEntryDiffers(windowedRows.AsSpan(0, hidden), unwindowedRows.AsSpan(0, hidden)),
            "position 0's row must differ from the same run on the model with no declared window: that one attends forward over its sequence");

        // The undeclared path keeps today's full-sequence bidirectional behavior in both directions: position 0 of an
        // unwindowed run is not that token encoded alone, because it still sees every later position.
        Assert.True(
            AnyEntryDiffers(unwindowedRows.AsSpan(0, hidden), unwindowedSoloRow),
            "an unbounded layer must leave position 0 attending forward: its full-sequence row differs from the same model encoding that token alone");

        // Against the reference oracle: position 0 lands on its backward-only range and not on the full bidirectional one.
        WindowReference reference = new(spec);
        Assert.True(
            Closest(windowedRows.AsSpan(0, hidden), reference.Encoded([1, 4, 2, 5, 3], window: Window)[0]),
            "position 0's encoded row must match the backward-only scalar reference");
        Assert.False(
            Closest(windowedRows.AsSpan(0, hidden), reference.Encoded([1, 4, 2, 5, 3], window: 0)[0]),
            "position 0's encoded row must not match the full bidirectional reference — a declared window removes its forward visibility");

        Assert.True(
            AnyEntryDiffers(windowedRows, unwindowedRows),
            "a windowed model's encoder output must differ from the same model with no declared window");
    }

    [Fact]
    [Trait("Category", "Embeddings")]
    public void Encode_BackwardOnlyWindow_MatchesEachPositionsOwnPrefixRun_AcrossChunks()
    {
        WindowModelSpec spec = new();
        string windowedPath = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.attention.sliding_window", Window)],
            spec.Tensors()));

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(windowedPath);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 16);

        int hidden = model.Plan.HiddenSize;
        int[] tokens = [1, 4, 2, 5, 3, 0, 1, 5, 2, 4]; // ten positions across two encoder chunks and three KV blocks

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 8, backend);

        float[] fullRows;
        {
            using SequenceKv kv = new(pool);
            fullRows = model.Encode(tokens, kv);
        }

        // Under a positive window every query's span is its own backward range and nothing after it, so each row of the
        // ten-position run equals the final row of a run over just that prefix — whatever chunking either run used.
        for (int p = 0; p < tokens.Length; p++)
        {
            float[] prefixRows;
            using (SequenceKv prefixKv = new(pool))
            {
                prefixRows = model.Encode(tokens[..(p + 1)], prefixKv);
            }

            Assert.True(
                BitsEqual(fullRows.AsSpan(p * hidden, hidden), prefixRows.AsSpan(prefixRows.Length - hidden, hidden)),
                $"row {p} of the ten-position encode must match position {p}'s own prefix run bit-for-bit under a backward-only window");
        }

        // With no declared window only the final row has that property: every earlier one attends forward over what follows.
        using GgufFile unwindowedGguf = GgufFile.Open(WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors())));
        using TransformerModel unwindowed = TransformerModel.Open(unwindowedGguf, backend, maxContextTokens: 16);

        float[] unwindowedFullRows;
        {
            using SequenceKv kv = new(pool);
            unwindowedFullRows = unwindowed.Encode(tokens, kv);
        }

        for (int p = 0; p < tokens.Length; p++)
        {
            float[] prefixRows;
            using (SequenceKv prefixKv = new(pool))
            {
                prefixRows = unwindowed.Encode(tokens[..(p + 1)], prefixKv);
            }

            var fullRow = unwindowedFullRows.AsSpan(p * hidden, hidden);
            var lastPrefixRow = prefixRows.AsSpan(prefixRows.Length - hidden, hidden);

            if (p == tokens.Length - 1)
            {
                Assert.True(
                    BitsEqual(fullRow, lastPrefixRow),
                    $"row {p}: the final position of an unwindowed encode already attends exactly its prefix, so its prefix run must agree bit-for-bit");
            }
            else
            {
                Assert.True(
                    AnyEntryDiffers(fullRow, lastPrefixRow),
                    $"row {p} of an unwindowed encode must differ from that position's own prefix run: it attends the positions after it");
            }
        }

        // The windowed rows are also the backward-only scalar reference across both chunk boundaries.
        WindowReference reference = new(spec);
        double[][] expected = reference.Encoded(tokens, window: Window);
        for (int p = 0; p < tokens.Length; p++)
        {
            Assert.True(
                Closest(fullRows.AsSpan(p * hidden, hidden), expected[p]),
                $"row {p} of the ten-position encode must match the backward-only scalar reference");
        }

        // The same weights with no declared window still run today's full-sequence bidirectional path unchanged: every
        // row matches the whole-range reference, which is what a resolved zero window must keep bit-for-bit.
        double[][] expectedUnwindowed = reference.Encoded(tokens, window: 0);
        for (int p = 0; p < tokens.Length; p++)
        {
            Assert.True(
                Closest(unwindowedFullRows.AsSpan(p * hidden, hidden), expectedUnwindowed[p]),
                $"row {p} of the unwindowed ten-position encode must match the full-sequence scalar reference");

            Assert.False(
                BitsEqual(fullRows.AsSpan(p * hidden, hidden), unwindowedFullRows.AsSpan(p * hidden, hidden)),
                $"row {p}: a windowed model's encoder output must differ from the same weights with no declared window");
        }
    }

    [Fact]
    public void Forward_WindowWithSoftcap_MatchesReferenceApplyingBoth()
    {
        WindowModelSpec spec = new();
        string path = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.attention.sliding_window", Window), ("llama.attn_logit_softcapping", AttentionCap)],
            spec.Tensors()));

        WindowReference reference = new(spec);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        Assert.Equal(Window, model.Plan.AttentionWindow);
        Assert.Equal(AttentionCap, model.Plan.AttentionSoftcap);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        model.Prefill([1, 4, 2, 5, 3], kv);
        float[] logits = model.DecodeNext(0, kv).ToArray();

        Assert.True(
            Closest(logits, reference.LogitsAt([1, 4, 2, 5, 3], 0, window: Window, cap: AttentionCap)),
            "windowed and capped logits at position 5 must match the reference forward applying both");

        Assert.False(
            Closest(logits, reference.LogitsAt([1, 4, 2, 5, 3], 0, window: Window, cap: 0f)),
            "the cap must move the windowed logits away from the mask-only reference");

        Assert.False(
            Closest(logits, reference.LogitsAt([1, 4, 2, 5, 3], 0, window: 0, cap: AttentionCap)),
            "the mask must move the capped logits away from the cap-only reference");
    }

    // ---------------------------------------------------------------- attention kernel goldens

    [Fact]
    public void CausalAttention_Window_ExcludesPositionsBeyondRange_AndWideWindowMatchesUnbounded()
    {
        const int Positions = 5;
        const int KernelWindow = 2;
        const int Region = Heads * HeadDim;

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using BlockPool pool = new(blockTokens: 2, layerCount: 1, kvHeads: KvHeads, headDim: HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        float[][] kRows = [[0.5f, -0.25f], [1f, 0f], [-0.75f, 0.6f], [0.2f, 0.8f], [-0.4f, -0.9f]];
        float[][] vRows = [[0.25f, 0.9f], [-1f, 0.125f], [0.4f, -0.3f], [0.7f, -0.55f], [-0.2f, 0.35f]];

        for (int t = 0; t < Positions; t++)
        {
            kv.Append();
            kv.WriteK(t, 0, kRows[t]);
            kv.WriteV(t, 0, vRows[t]);
        }

        float[] q = [0.3f, -0.6f, 0.45f, 0.15f]; // two heads x headDim
        float scale = 1f / MathF.Sqrt(HeadDim);

        float[] windowedOut = new float[Region];
        LayerOps.CausalAttention(backend, q, Heads, KvHeads, HeadDim, kv, layer: 0, position: Positions - 1, scale, softcap: 0f, window: KernelWindow, windowedOut);

        // Hand oracle over exactly the narrowed range [3..4].
        AssertHandComputed(windowedOut, q, kRows, vRows, from: Positions - KernelWindow, to: Positions - 1, scale, "windowed causal attention at position 4");

        float[] unboundedOut = new float[Region];
        LayerOps.CausalAttention(backend, q, Heads, KvHeads, HeadDim, kv, layer: 0, position: Positions - 1, scale, softcap: 0f, window: 0, unboundedOut);

        AssertHandComputed(unboundedOut, q, kRows, vRows, from: 0, to: Positions - 1, scale, "unbounded causal attention at position 4");

        Assert.True(
            !BitsEqual(windowedOut, unboundedOut),
            "the masked range must produce a different attention output than the unbounded range");

        // A window covering the whole causal prefix is the unbounded path bit-for-bit.
        float[] wideOut = new float[Region];
        LayerOps.CausalAttention(backend, q, Heads, KvHeads, HeadDim, kv, layer: 0, position: Positions - 1, scale, softcap: 0f, window: Positions, wideOut);
        Assert.True(
            BitsEqual(wideOut, unboundedOut),
            "a window at least as wide as the causal prefix must match the unbounded kernel bit-for-bit");

        // The mask bounds scores, not memory: every masked-out position is still stored and readable.
        Assert.True(
            kv.Length == Positions && BitsEqual(kv.ReadK(0, 0), kRows[0]) && BitsEqual(kv.ReadV(0, 0), vRows[0]),
            "a windowed read must leave every earlier position's K and V intact in the cache");
    }

    [Fact]
    public void BidirectionalAttention_Window_NarrowsBackwardOnlyAroundEachQuery()
    {
        const int Positions = 7; // KV positions 0..6; the queries themselves live at 2..6
        const int QueryCount = 5;
        const int QueryOffset = 2;
        const int KernelWindow = 2;
        const int Region = Heads * HeadDim;

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using BlockPool pool = new(blockTokens: 2, layerCount: 1, kvHeads: KvHeads, headDim: HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        for (int t = 0; t < Positions; t++)
        {
            kv.Append();
            kv.WriteK(t, 0, [MathF.Sin(0.7f * t + 0.3f), MathF.Cos(1.1f * t)]);
            kv.WriteV(t, 0, [MathF.Sin(2.3f * t + 1f), -MathF.Cos(0.9f * t)]);
        }

        float[] qPositions = new float[QueryCount * Region];
        for (int i = 0; i < qPositions.Length; i++)
        {
            qPositions[i] = MathF.Sin(0.31f * i) + 0.5f * MathF.Cos(1.7f * i);
        }

        float scale = 1f / MathF.Sqrt(HeadDim);

        float[] windowedOut = new float[QueryCount * Region];
        LayerOps.BidirectionalAttention(
            backend, qPositions, QueryCount, queryPositionOffset: QueryOffset, fromPosition: 0, toPosition: Positions - 1,
            Heads, KvHeads, HeadDim, kv, layer: 0, scale, softcap: 0f, window: KernelWindow, windowedOut);

        for (int p = 0; p < QueryCount; p++)
        {
            int queryPosition = QueryOffset + p;
            int from = Math.Max(0, queryPosition - KernelWindow + 1);
            int to = Math.Min(Positions - 1, queryPosition); // the window stops at the query: nothing after it is scored

            float[] expected = new float[Region];
            HandComputeAttention(
                expected, qPositions.AsSpan(p * Region, Region).ToArray(),
                Enumerable.Range(0, Positions).Select(t => kv.ReadK(t, 0).ToArray()).ToList(),
                Enumerable.Range(0, Positions).Select(t => kv.ReadV(t, 0).ToArray()).ToList(),
                from, to, scale);

            for (int i = 0; i < Region; i++)
            {
                Assert.True(
                    Math.Abs(windowedOut[p * Region + i] - expected[i]) <= 1e-5 * Math.Max(1.0, Math.Abs(expected[i])),
                    $"query at position {queryPosition} range [{from}..{to}] entry {i}: expected {expected[i]:G9}, got {windowedOut[p * Region + i]:G9}");
            }

            // Same positions, same window, causal kernel: the encoder narrowing is exactly the decode kernel's range.
            float[] causalOut = new float[Region];
            LayerOps.CausalAttention(backend, qPositions.AsSpan(p * Region, Region), Heads, KvHeads, HeadDim, kv, layer: 0, position: queryPosition, scale, softcap: 0f, window: KernelWindow, causalOut);

            Assert.True(
                BitsEqual(windowedOut.AsSpan(p * Region, Region), causalOut),
                $"query at position {queryPosition}: the windowed bidirectional row must match CausalAttention under the same window bit-for-bit");
        }

        // The same call with the window disabled sees the full given range in both directions, so the narrowing is a real mask.
        float[] fullOut = new float[QueryCount * Region];
        LayerOps.BidirectionalAttention(
            backend, qPositions, QueryCount, queryPositionOffset: QueryOffset, fromPosition: 0, toPosition: Positions - 1,
            Heads, KvHeads, HeadDim, kv, layer: 0, scale, softcap: 0f, window: 0, fullOut);

        Assert.True(
            !BitsEqual(windowedOut.AsSpan(0, Region), fullOut.AsSpan(0, Region)),
            "the query at position 2 under a backward-only window of 2 must differ from the full-range bidirectional run");

        // A positive window that reaches back past every earlier position is still not the unbounded path: it cuts off
        // everything after each query. Only the last query, whose backward span already covers the table, coincides.
        float[] wideOut = new float[QueryCount * Region];
        LayerOps.BidirectionalAttention(
            backend, qPositions, QueryCount, queryPositionOffset: QueryOffset, fromPosition: 0, toPosition: Positions - 1,
            Heads, KvHeads, HeadDim, kv, layer: 0, scale, softcap: 0f, window: Positions, wideOut);

        for (int p = 0; p < QueryCount; p++)
        {
            var wideRow = wideOut.AsSpan(p * Region, Region);
            var fullRow = fullOut.AsSpan(p * Region, Region);

            if (p == QueryCount - 1)
            {
                Assert.True(
                    BitsEqual(wideRow, fullRow),
                    $"query at position {QueryOffset + p}: its backward span covers the whole table, so a window that wide matches the unmasked kernel bit-for-bit");
            }
            else
            {
                Assert.True(
                    AnyEntryDiffers(wideRow, fullRow),
                    $"query at position {QueryOffset + p}: a positive window must still remove its forward visibility against the unmasked run");
            }

            float[] causalOut = new float[Region];
            LayerOps.CausalAttention(backend, qPositions.AsSpan(p * Region, Region), Heads, KvHeads, HeadDim, kv, layer: 0, position: QueryOffset + p, scale, softcap: 0f, window: Positions, causalOut);

            Assert.True(
                BitsEqual(wideRow, causalOut),
                $"query at position {QueryOffset + p}: the wide-windowed bidirectional row must match CausalAttention under the same window bit-for-bit");
        }

        // A query whose windowed range misses the given KV range entirely is an argument error, not a silent zero.
        Assert.Throws<ArgumentException>(() => LayerOps.BidirectionalAttention(
            backend, qPositions, QueryCount, queryPositionOffset: QueryOffset + 4, fromPosition: 0, toPosition: 2,
            Heads, KvHeads, HeadDim, kv, layer: 0, scale, softcap: 0f, window: 1, windowedOut));
    }

    // ---------------------------------------------------------------- reference oracle

    private static bool Closest(ReadOnlySpan<float> actual, double[] expected)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            if (Math.Abs(actual[i] - expected[i]) > 1e-5 * Math.Max(1.0, Math.Abs(expected[i])))
            {
                return false;
            }
        }

        return true;
    }

    private static bool Differs(ReadOnlySpan<float> actual, double[][] expectedRows)
    {
        for (int p = 0; p < expectedRows.Length; p++)
        {
            for (int i = 0; i < expectedRows[p].Length; i++)
            {
                double expected = expectedRows[p][i];
                if (Math.Abs(actual[p * expectedRows[p].Length + i] - expected) > 1e-4 * Math.Max(1.0, Math.Abs(expected)))
                {
                    return true;
                }
            }
        }

        return false;
    }

    // Engine-to-engine divergence: at least one entry of the two rows moves past the shared tolerance, which is what a
    // mask that changes the scored positions must do to an attention output.
    private static bool AnyEntryDiffers(ReadOnlySpan<float> actual, ReadOnlySpan<float> other)
    {
        if (actual.Length != other.Length)
        {
            throw new ArgumentException("Divergence checks compare equal-length rows.", nameof(other));
        }

        for (int i = 0; i < other.Length; i++)
        {
            if (Math.Abs(actual[i] - other[i]) > 1e-4 * Math.Max(1.0, Math.Abs(other[i])))
            {
                return true;
            }
        }

        return false;
    }

    private static bool BitsEqual(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        for (int i = 0; i < a.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
            {
                return false;
            }
        }

        return true;
    }

    private static void AssertHandComputed(float[] actual, float[] q, float[][] kRows, float[][] vRows, int from, int to, float scale, string label)
    {
        float[] expected = new float[actual.Length];
        HandComputeAttention(expected, q, kRows, vRows, from, to, scale);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(
                Math.Abs(actual[i] - expected[i]) <= 1e-5 * Math.Max(1.0, Math.Abs(expected[i])),
                $"{label} entry {i}: expected {expected[i]:G9}, got {actual[i]:G9}");
        }
    }

    // Explicit-softmax oracle in double precision over the visible range only: heads 2, kv-heads 1, so every head
    // reads kv-head (h * 1 / 2) == 0.
    private static void HandComputeAttention(float[] actual, float[] q, IReadOnlyList<float[]> kRows, IReadOnlyList<float[]> vRows, int from, int to, float scale)
    {
        for (int h = 0; h < Heads; h++)
        {
            int count = to - from + 1;
            double[] scores = new double[count];
            double max = double.NegativeInfinity;

            for (int t = from; t <= to; t++)
            {
                double dot = 0.0;
                for (int j = 0; j < HeadDim; j++)
                {
                    dot += q[h * HeadDim + j] * kRows[t][j];
                }

                scores[t - from] = dot * scale;
                max = Math.Max(max, scores[t - from]);
            }

            double sum = 0.0;
            for (int t = from; t <= to; t++)
            {
                scores[t - from] = Math.Exp(scores[t - from] - max);
                sum += scores[t - from];
            }

            for (int j = 0; j < HeadDim; j++)
            {
                double acc = 0.0;
                for (int t = from; t <= to; t++)
                {
                    acc += scores[t - from] / sum * vRows[t][j];
                }

                actual[h * HeadDim + j] = (float)acc;
            }
        }
    }

    // A complete single-layer model with hidden 6, heads 2, kv-heads 1, a declared head dim of 4, ffn 8, vocab 6
    // — all F32 tensors with hand-picked payloads the reference forward reads directly. No output.weight: the
    // logits come from the tied token-embedding rows. The q/k rows sit at the unit scale so scores stay modest
    // at the positions these goldens reach, and the V rows are distinct per position so dropping one from the
    // softmax moves the output by far more than the comparison tolerance.
    private sealed class WindowModelSpec
    {
        public const int Hidden = 6;
        public const int Heads = 2;
        public const int KvHeads = 1;
        public const int HeadDim = 4;
        public const int Intermediate = 8;
        public const int Vocab = 6;
        public const float Eps = 1e-5f;
        public const double RopeTheta = 10_000.0;

        public List<(string Key, object Value)> MetadataPairs() =>
        [
            ("general.architecture", "llama"),
            ("llama.embedding_length", Hidden),
            ("llama.block_count", 1),
            ("llama.attention.head_count", Heads),
            ("llama.attention.head_count_kv", KvHeads),
            ("llama.feed_forward_length", Intermediate),
            ("llama.attention.head_length", HeadDim),
            ("llama.attention.layer_norm_rms_epsilon", Eps),
            ("llama.vocab_size", Vocab),
        ];

        public List<TensorDef> Tensors() =>
        [
            new("token_embd.weight", [Vocab, Hidden], Embeddings),
            new("blk.0.attn_q.weight", [Heads * HeadDim, Hidden], AttnQ),
            new("blk.0.attn_k.weight", [KvHeads * HeadDim, Hidden], AttnK),
            new("blk.0.attn_v.weight", [KvHeads * HeadDim, Hidden], AttnV),
            new("blk.0.attn_output.weight", [Hidden, Heads * HeadDim], AttnO),
            new("blk.0.ffn_gate.weight", [Intermediate, Hidden], FfnGate),
            new("blk.0.ffn_up.weight", [Intermediate, Hidden], FfnUp),
            new("blk.0.ffn_down.weight", [Hidden, Intermediate], FfnDown),
            new("blk.0.attn_norm.weight", [Hidden], AttnNorm),
            new("blk.0.ffn_norm.weight", [Hidden], FfnNorm),
            new("output_norm.weight", [Hidden], OutputNorm),
        ];

        public readonly float[] Embeddings =
        [
            0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f,
            0.11f, -0.21f, 0.31f, -0.41f, 0.51f, -0.61f,
            0.32f, 0.22f, -0.32f, 0.42f, -0.52f, 0.62f,
            -0.13f, 0.23f, 0.33f, -0.43f, 0.53f, 0.63f,
            0.94f, -0.84f, 0.74f, -0.64f, 0.54f, -0.44f,
            0.15f, 0.25f, 0.35f, 0.45f, 0.55f, -0.65f,
        ];

        public readonly float[] AttnQ =
        [
            0.20f, 0.40f, 0.30f, 0.10f, 0.50f, 0.25f,
            0.15f, -0.35f, 0.45f, 0.20f, -0.10f, 0.30f,
            0.35f, 0.10f, -0.25f, 0.40f, 0.15f, -0.30f,
            -0.20f, 0.30f, 0.10f, -0.45f, 0.25f, 0.40f,
            0.45f, -0.15f, 0.20f, 0.35f, -0.40f, 0.10f,
            -0.30f, 0.25f, -0.40f, 0.15f, 0.35f, -0.20f,
            0.10f, 0.50f, -0.15f, -0.30f, 0.20f, 0.45f,
            -0.25f, 0.15f, 0.40f, -0.20f, -0.35f, 0.30f,
        ];

        public readonly float[] AttnK =
        [
            0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f,
            0.20f, -0.10f, 0.40f, 0.30f, -0.20f, 0.50f,
            -0.30f, 0.40f, 0.10f, -0.50f, 0.20f, 0.30f,
            0.40f, 0.15f, -0.35f, 0.25f, 0.45f, -0.25f,
        ];

        // Distinct V rows per position: dropping a position from the softmax moves the output by a detectable amount.
        public readonly float[] AttnV =
        [
            0.09f, 0.08f, 0.07f, 0.06f, 0.05f, 0.04f,
            -0.45f, 0.38f, -0.31f, 0.24f, -0.17f, 0.10f,
            0.30f, -0.22f, 0.14f, -0.06f, 0.42f, -0.34f,
            0.06f, 0.05f, 0.04f, 0.03f, 0.02f, 0.01f,
        ];

        public readonly float[] AttnO =
        [
            0.10f, 0.00f, 0.20f, 0.00f, 0.30f, 0.00f, 0.15f, 0.00f,
            0.00f, 0.30f, 0.00f, 0.40f, 0.00f, 0.50f, 0.00f, 0.25f,
            0.15f, 0.00f, 0.25f, 0.00f, 0.35f, 0.00f, 0.20f, 0.00f,
            0.00f, 0.35f, 0.00f, 0.45f, 0.00f, 0.55f, 0.00f, 0.30f,
            0.10f, 0.00f, 0.15f, 0.00f, 0.25f, 0.00f, 0.35f, 0.00f,
            0.00f, 0.20f, 0.00f, 0.30f, 0.00f, 0.40f, 0.00f, 0.45f,
        ];

        public readonly float[] FfnGate =
        [
            0.10f, 0.11f, 0.12f, 0.13f, 0.14f, 0.15f,
            0.16f, 0.17f, 0.18f, 0.19f, 0.20f, 0.21f,
            0.22f, 0.23f, 0.24f, 0.25f, 0.26f, 0.27f,
            0.28f, 0.29f, 0.30f, 0.31f, 0.32f, 0.33f,
            0.34f, 0.35f, 0.36f, 0.37f, 0.38f, 0.39f,
            0.40f, 0.41f, 0.42f, 0.43f, 0.44f, 0.45f,
            0.46f, 0.47f, 0.48f, 0.49f, 0.50f, 0.51f,
            0.52f, 0.53f, 0.54f, 0.55f, 0.56f, 0.57f,
        ];

        public readonly float[] FfnUp =
        [
            0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f,
            0.07f, 0.08f, 0.09f, 0.10f, 0.11f, 0.12f,
            0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f,
            0.19f, 0.20f, 0.21f, 0.22f, 0.23f, 0.24f,
            0.25f, 0.26f, 0.27f, 0.28f, 0.29f, 0.30f,
            0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f,
            0.37f, 0.38f, 0.39f, 0.40f, 0.41f, 0.42f,
            0.43f, 0.44f, 0.45f, 0.46f, 0.47f, 0.48f,
        ];

        public readonly float[] FfnDown =
        [
            0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.10f, 0.11f, 0.12f,
            0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.20f,
            0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f, 0.27f, 0.28f,
            0.29f, 0.30f, 0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f,
            0.37f, 0.38f, 0.39f, 0.40f, 0.41f, 0.42f, 0.43f, 0.44f,
            0.45f, 0.46f, 0.47f, 0.48f, 0.49f, 0.50f, 0.51f, 0.52f,
        ];

        public readonly float[] AttnNorm = [1.10f, 0.90f, 1.05f, 0.95f, 1.00f, 1.15f];
        public readonly float[] FfnNorm = [0.90f, 1.10f, 0.95f, 1.05f, 1.00f, 0.85f];
        public readonly float[] OutputNorm = [1.00f, 1.02f, 0.98f, 1.01f, 0.99f, 1.03f];
    }

    // A scalar reference forward over the spec's hand-picked tensors: RMSNorm, standard full-head RoPE, causal or
    // bidirectional attention with an explicit softmax and the score mask applied to the visible range, silu(gate)
    // * up, and the tied token-embedding logits. With a positive window both ranges are backward-only —
    // [max(0, i - window + 1), i] on the causal path and the same span ending at its own query on the bidirectional one;
    // an unbounded range is the whole sequence in either direction. Computed in double and compared against the engine's
    // float values with the shared tolerance.
    private sealed class WindowReference
    {
        private readonly WindowModelSpec _spec;

        public WindowReference(WindowModelSpec spec) => _spec = spec;

        /// <summary>Logits for a query token forwarded at position historyTokens.Length after replaying the sequence's earlier tokens through the same layer.</summary>
        public double[] LogitsAt(int[] historyTokens, int queryToken, int window, float cap = 0f)
        {
            double[][] kvK = new double[historyTokens.Length + 1][];
            double[][] kvV = new double[historyTokens.Length + 1][];

            for (int p = 0; p < historyTokens.Length; p++)
            {
                Advance(Row(_spec.Embeddings, historyTokens[p], WindowModelSpec.Hidden), p, kvK, kvV, window, cap);
            }

            double[] cur = Row(_spec.Embeddings, queryToken, WindowModelSpec.Hidden);
            Advance(cur, historyTokens.Length, kvK, kvV, window, cap);

            double[] normed = Norm(cur, _spec.OutputNorm);
            return Gemv(_spec.Embeddings, WindowModelSpec.Vocab, WindowModelSpec.Hidden, normed); // tied lm_head
        }

        public double[][] Encoded(int[] tokens, int window, float cap = 0f)
        {
            int n = tokens.Length;
            double[][] cur = new double[n][];
            for (int i = 0; i < n; i++)
            {
                cur[i] = Row(_spec.Embeddings, tokens[i], WindowModelSpec.Hidden);
            }

            double[][] kvK = new double[n][];
            double[][] kvV = new double[n][];

            double[][] qAll = new double[n][];
            for (int p = 0; p < n; p++)
            {
                double[] normed = Norm(cur[p], _spec.AttnNorm);
                double[] q = Gemv(_spec.AttnQ, WindowModelSpec.Heads * WindowModelSpec.HeadDim, WindowModelSpec.Hidden, normed);
                double[] k = Gemv(_spec.AttnK, WindowModelSpec.KvHeads * WindowModelSpec.HeadDim, WindowModelSpec.Hidden, normed);
                double[] v = Gemv(_spec.AttnV, WindowModelSpec.KvHeads * WindowModelSpec.HeadDim, WindowModelSpec.Hidden, normed);

                Rope(q, p);
                Rope(k, p);

                qAll[p] = q;
                kvK[p] = k;
                kvV[p] = v;
            }

            double[] attnOut = new double[n * WindowModelSpec.Heads * WindowModelSpec.HeadDim];
            for (int p = 0; p < n; p++)
            {
                Attention(qAll[p], kvK, kvV, BidirectionalRange(p, n, window), cap, attnOut.AsSpan(p * WindowModelSpec.Heads * WindowModelSpec.HeadDim));
            }

            for (int p = 0; p < n; p++)
            {
                double[] proj = Gemv(_spec.AttnO, WindowModelSpec.Hidden, WindowModelSpec.Heads * WindowModelSpec.HeadDim, attnOut.AsSpan(p * WindowModelSpec.Heads * WindowModelSpec.HeadDim).ToArray());
                for (int i = 0; i < WindowModelSpec.Hidden; i++)
                {
                    cur[p][i] += proj[i];
                }
            }

            for (int p = 0; p < n; p++)
            {
                double[] normed = Norm(cur[p], _spec.FfnNorm);
                double[] gate = Gemv(_spec.FfnGate, WindowModelSpec.Intermediate, WindowModelSpec.Hidden, normed);
                double[] up = Gemv(_spec.FfnUp, WindowModelSpec.Intermediate, WindowModelSpec.Hidden, normed);
                for (int i = 0; i < WindowModelSpec.Intermediate; i++)
                {
                    gate[i] = Silu(gate[i]) * up[i];
                }

                double[] proj = Gemv(_spec.FfnDown, WindowModelSpec.Hidden, WindowModelSpec.Intermediate, gate);
                for (int i = 0; i < WindowModelSpec.Hidden; i++)
                {
                    cur[p][i] += proj[i];
                }
            }

            double[][] encoded = new double[n][];
            for (int i = 0; i < n; i++)
            {
                encoded[i] = Norm(cur[i], _spec.OutputNorm);
            }

            return encoded;
        }

        private void Advance(double[] cur, int position, double[][] kvK, double[][] kvV, int window, float cap)
        {
            double[] normed = Norm(cur, _spec.AttnNorm);
            double[] q = Gemv(_spec.AttnQ, WindowModelSpec.Heads * WindowModelSpec.HeadDim, WindowModelSpec.Hidden, normed);
            double[] k = Gemv(_spec.AttnK, WindowModelSpec.KvHeads * WindowModelSpec.HeadDim, WindowModelSpec.Hidden, normed);
            double[] v = Gemv(_spec.AttnV, WindowModelSpec.KvHeads * WindowModelSpec.HeadDim, WindowModelSpec.Hidden, normed);

            Rope(q, position);
            Rope(k, position);

            kvK[position] = k;
            kvV[position] = v;

            double[] attnOut = new double[WindowModelSpec.Heads * WindowModelSpec.HeadDim];
            Attention(q, kvK, kvV, CausalRange(position, window), cap, attnOut);

            double[] proj = Gemv(_spec.AttnO, WindowModelSpec.Hidden, WindowModelSpec.Heads * WindowModelSpec.HeadDim, attnOut);
            for (int i = 0; i < WindowModelSpec.Hidden; i++)
            {
                cur[i] += proj[i];
            }

            double[] ffnNormed = Norm(cur, _spec.FfnNorm);
            double[] gate = Gemv(_spec.FfnGate, WindowModelSpec.Intermediate, WindowModelSpec.Hidden, ffnNormed);
            double[] up = Gemv(_spec.FfnUp, WindowModelSpec.Intermediate, WindowModelSpec.Hidden, ffnNormed);
            for (int i = 0; i < WindowModelSpec.Intermediate; i++)
            {
                gate[i] = Silu(gate[i]) * up[i];
            }

            double[] down = Gemv(_spec.FfnDown, WindowModelSpec.Hidden, WindowModelSpec.Intermediate, gate);
            for (int i = 0; i < WindowModelSpec.Hidden; i++)
            {
                cur[i] += down[i];
            }
        }

        // The causal visible range: narrowed to the window when one is given, the full prefix otherwise.
        private static (int From, int To) CausalRange(int position, int window) =>
            (window > 0 ? Math.Max(0, position - window + 1) : 0, position);

        // The bidirectional visible range: backward-only from the query position, clamped to the sequence. A positive
        // window never reaches past the query itself; only an unbounded layer attends what follows it.
        private static (int From, int To) BidirectionalRange(int position, int sequenceLength, int window) =>
            window > 0
                ? (Math.Max(0, position - window + 1), position)
                : (0, sequenceLength - 1);

        private static void Attention(double[] q, double[][] kvK, double[][] kvV, (int From, int To) range, float cap, Span<double> outHead)
        {
            const int headDim = WindowModelSpec.HeadDim;
            double scale = 1.0 / Math.Sqrt(headDim);
            (int from, int to) = range;

            // heads 2, kv-heads 1: every head reads kv-head (h * 1 / 2) == 0.
            for (int h = 0; h < WindowModelSpec.Heads; h++)
            {
                int kvHead = h * WindowModelSpec.KvHeads / WindowModelSpec.Heads;

                double[] scores = new double[to - from + 1];
                double max = double.NegativeInfinity;
                for (int t = from; t <= to; t++)
                {
                    double dot = 0.0;
                    for (int j = 0; j < headDim; j++)
                    {
                        dot += q[h * headDim + j] * kvK[t][kvHead * headDim + j];
                    }

                    double score = dot * scale;
                    if (cap > 0f)
                    {
                        score = (double)cap * Math.Tanh(score / (double)cap); // after scaling, before the softmax
                    }

                    scores[t - from] = score;
                    max = Math.Max(max, score);
                }

                double sum = 0.0;
                for (int t = from; t <= to; t++)
                {
                    scores[t - from] = Math.Exp(scores[t - from] - max);
                    sum += scores[t - from];
                }

                for (int j = 0; j < headDim; j++)
                {
                    double acc = 0.0;
                    for (int t = from; t <= to; t++)
                    {
                        acc += scores[t - from] / sum * kvV[t][kvHead * headDim + j];
                    }

                    outHead[h * headDim + j] = acc;
                }
            }
        }

        private static void Rope(double[] headRegion, int position)
        {
            for (int h = 0; h < headRegion.Length / WindowModelSpec.HeadDim; h++)
            {
                for (int i = 0; i < WindowModelSpec.HeadDim / 2; i++)
                {
                    double invFreq = Math.Pow(WindowModelSpec.RopeTheta, -2.0 * i / WindowModelSpec.HeadDim);
                    double angle = position * invFreq;
                    double c = Math.Cos(angle);
                    double s = Math.Sin(angle);

                    double lo = headRegion[h * WindowModelSpec.HeadDim + 2 * i];
                    double hi = headRegion[h * WindowModelSpec.HeadDim + 2 * i + 1];
                    headRegion[h * WindowModelSpec.HeadDim + 2 * i] = lo * c - hi * s;
                    headRegion[h * WindowModelSpec.HeadDim + 2 * i + 1] = hi * c + lo * s;
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

            double invRms = 1.0 / Math.Sqrt(sumSq / n + WindowModelSpec.Eps);
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
    }

    // ---------------------------------------------------------------- kernel fixture geometry

    private const int Heads = 2;
    private const int KvHeads = 1;
    private const int HeadDim = 2;

    private sealed record TensorDef(string Name, int[] Dims, float[] Payload); // dims in logical (row-major) order; vectors are one-dimensional

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
                case double d:
                    writer.Write(12u); // FLOAT64
                    writer.Write(d);
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

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-window-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }
}
