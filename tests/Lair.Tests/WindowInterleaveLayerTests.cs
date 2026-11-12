using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Forward-level goldens for the resolved window interleave. A synthetic four-layer model whose plan resolves the
/// alternating pattern matches a double-precision scalar reference that masks only the even-indexed layers at a query
/// position beyond the window, and demonstrably differs from the same weights masked on every layer; a plan resolving
/// the all-layers pattern keeps those same weights and the same window bit-identical to the behavior of masking
/// every layer. Batched decode is bit-identical to single-token decode under the alternating pattern, the
/// <see cref="TransformerModel.Encode"/> path narrows each masked layer's queries backward-only and is golden against
/// the even-layers-only reference across a chunk boundary, and a resolved window of zero under the alternating pattern
/// is bit-identical to no window at all. Engine-to-engine comparisons are asserted bit-for-bit; reference comparisons use
/// the double-precision tolerance shared by the strategy goldens. Encoder cases carry both the Layers and Embeddings
/// categories.
/// </summary>
[Trait("Category", "Layers")]
public sealed class WindowInterleaveLayerTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

    // Three positions, so a query at position 5 sees [3..5] on the masked layers and [0..5] on the unmasked ones,
    // and the alternation moves the output by a detectable amount.
    private const int Window = 3;

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

    // ---------------------------------------------------------------- alternating decode golden

    [Fact]
    public void Forward_AlternatingWindow_MatchesEvenLayersOnlyReference_AndDiffersFromEveryLayerMasked()
    {
        InterleaveModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(Window), spec.Tensors()));

        ModelProfile alternating = ModelProfiles.Resolve("llama") with { WindowInterleave = WindowInterleave.AlternatingLocalFirst };

        InterleaveReference reference = new(spec);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, alternating);

        Assert.Equal(WindowInterleave.AlternatingLocalFirst, model.Plan.WindowInterleave);
        Assert.Equal(4, model.Plan.LayerCount);
        Assert.Equal(Window, model.Plan.AttentionWindow);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 12, backend);
        using SequenceKv kv = new(pool);

        // Inside the window the alternation is inert: at positions 0 and 1 every layer's causal range already fits,
        // so both references agree with each other and with the engine.
        Assert.True(
            Closest(model.DecodeNext(1, kv), reference.LogitsAt([], 1, Window, alternating: true)),
            "decode logits at position 0 must match the alternating reference forward");

        float[] positionOne = model.DecodeNext(4, kv).ToArray();

        Assert.True(
            Closest(positionOne, reference.LogitsAt([1], 4, Window, alternating: true)),
            "decode logits at position 1 must match the alternating reference forward");
        Assert.True(
            Closest(positionOne, reference.LogitsAt([1], 4, Window, alternating: false)),
            "a window wider than the causal prefix must leave the position-1 logits equal to the every-layer reference");

        model.Prefill([2, 5, 3], kv);
        float[] logits = model.DecodeNext(0, kv).ToArray();

        // Beyond the window at position 5: the even layers see [3..5] and the odd layers see the whole prefix.
        Assert.True(
            Closest(logits, reference.LogitsAt([1, 4, 2, 5, 3], 0, Window, alternating: true)),
            "decode logits at position 5 must match the reference masking only the even-indexed layers");

        Assert.False(
            Closest(logits, reference.LogitsAt([1, 4, 2, 5, 3], 0, Window, alternating: false)),
            "the same weights masked on every layer must not leave the position-5 logits equal to the alternating forward");
    }

    // ---------------------------------------------------------------- the all-layers pattern is unchanged

    [Fact]
    [Trait("Category", "Embeddings")]
    public void Forward_AllLayersWindow_MatchesEveryLayerReference_AndIsBitIdenticalToAnExplicitAllLayersPlan()
    {
        InterleaveModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(Window), spec.Tensors()));

        InterleaveReference reference = new(spec);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        Assert.Equal(WindowInterleave.AllLayers, model.Plan.WindowInterleave);
        Assert.Equal(Window, model.Plan.AttentionWindow);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 12, backend);

        float[] decodeLogits;
        float[] encoded;
        {
            using SequenceKv kv = new(pool);
            model.Prefill([1, 4, 2, 5, 3], kv);
            decodeLogits = model.DecodeNext(0, kv).ToArray();

            using SequenceKv encodeKv = new(pool);
            encoded = model.Encode([1, 4, 2, 5, 3], encodeKv);
        }

        Assert.True(
            Closest(decodeLogits, reference.LogitsAt([1, 4, 2, 5, 3], 0, Window, alternating: false)),
            "the all-layers pattern must match the reference masking every layer at position 5");

        double[][] expectedEncoded = reference.Encoded([1, 4, 2, 5, 3], Window, alternating: false);
        for (int p = 0; p < 5; p++)
        {
            for (int i = 0; i < model.Plan.HiddenSize; i++)
            {
                float actual = encoded[p * model.Plan.HiddenSize + i];
                double expected = expectedEncoded[p][i];
                Assert.True(
                    Math.Abs(actual - expected) <= 1e-5 * Math.Max(1.0, Math.Abs(expected)),
                    $"encode hidden[{p}][{i}] under the all-layers pattern: expected {expected:G9}, got {actual:G9}");
            }
        }

        Assert.True(
            Differs(encoded, reference.Encoded([1, 4, 2, 5, 3], Window, alternating: true)),
            "the all-layers encode must differ from the reference masking only the even-indexed layers");

        // The same weights and window under a plan that names the all-layers pattern explicitly: bit-identical, so
        // the resolved value adds nothing to a forward that already masked every layer.
        float[] explicitDecodeLogits;
        float[] explicitEncoded;
        {
            using TransformerModel explicitModel = TransformerModel.Open(
                gguf, backend, maxContextTokens: 8, ModelProfiles.Resolve("llama") with { WindowInterleave = WindowInterleave.AllLayers });

            Assert.Equal(WindowInterleave.AllLayers, explicitModel.Plan.WindowInterleave);

            using SequenceKv kv = new(pool);
            explicitModel.Prefill([1, 4, 2, 5, 3], kv);
            explicitDecodeLogits = explicitModel.DecodeNext(0, kv).ToArray();

            using SequenceKv encodeKv = new(pool);
            explicitEncoded = explicitModel.Encode([1, 4, 2, 5, 3], encodeKv);
        }

        Assert.True(
            explicitDecodeLogits.AsSpan().SequenceEqual(decodeLogits.AsSpan()),
            "an explicit all-layers plan must decode bit-identically to the profile-resolved one");
        Assert.True(
            explicitEncoded.AsSpan().SequenceEqual(encoded.AsSpan()),
            "an explicit all-layers plan must encode bit-identically to the profile-resolved one");
    }

    // ---------------------------------------------------------------- batched decode under the alternation

    [Fact]
    public void Forward_AlternatingWindow_DecodeBatchIsBitIdenticalToSingleTokenDecode()
    {
        InterleaveModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(Window), spec.Tensors()));

        ModelProfile alternating = ModelProfiles.Resolve("llama") with { WindowInterleave = WindowInterleave.AlternatingLocalFirst };

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, alternating);

        int[] tokens = [1, 4, 2, 5, 3];

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 12, backend);

        float[] singleLogits;
        {
            using SequenceKv single = new(pool);
            model.Prefill(tokens, single);
            singleLogits = model.LastLogits().ToArray();
        }

        float[] batchLogits;
        {
            using SequenceKv batched = new(pool);
            model.DecodeBatch([tokens[0]], [batched]);
            model.DecodeBatch([tokens[1]], [batched]);
            model.DecodeBatch([tokens[2]], [batched]);
            model.DecodeBatch([tokens[3]], [batched]);
            batchLogits = model.DecodeBatch([tokens[4]], [batched]).ToArray();
        }

        Assert.True(
            batchLogits.AsSpan().SequenceEqual(singleLogits.AsSpan()),
            "a token batched through DecodeBatch under the alternating pattern must be bit-identical to stepping the same sequence alone");

        // Two sequences in one batched call: each matches its own single-token forward at the same position.
        {
            using SequenceKv seqA = new(pool);
            using SequenceKv seqB = new(pool);
            model.Prefill([1, 4, 2, 5], seqA);
            model.Prefill([2, 5, 3, 1], seqB);
            Span<float> pairLogits = model.DecodeBatch([3, 4], [seqA, seqB]);
            float[] firstOfPair = pairLogits.Slice(0, model.Plan.VocabSize).ToArray();
            float[] secondOfPair = pairLogits.Slice(model.Plan.VocabSize, model.Plan.VocabSize).ToArray();

            using SequenceKv soloA = new(pool);
            model.Prefill([1, 4, 2, 5], soloA);
            float[] soloALogits = model.DecodeNext(3, soloA).ToArray();
            Assert.True(
                firstOfPair.AsSpan().SequenceEqual(soloALogits.AsSpan()),
                "the batched first sequence must match its own single-token forward under the alternating pattern");

            using SequenceKv soloB = new(pool);
            model.Prefill([2, 5, 3, 1], soloB);
            float[] soloBLogits = model.DecodeNext(4, soloB).ToArray();
            Assert.True(
                secondOfPair.AsSpan().SequenceEqual(soloBLogits.AsSpan()),
                "the batched second sequence must match its own single-token forward under the alternating pattern");
        }
    }

    // ---------------------------------------------------------------- encode under the alternation

    [Fact]
    [Trait("Category", "Embeddings")]
    public void Forward_AlternatingWindow_EncodeMatchesEvenLayersOnlyReference_AcrossChunks()
    {
        // Under the alternating pattern only the even layers narrow, and each of those narrows backward-only from its own query.
        InterleaveModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(Window), spec.Tensors()));

        ModelProfile alternating = ModelProfiles.Resolve("llama") with { WindowInterleave = WindowInterleave.AlternatingLocalFirst };

        InterleaveReference reference = new(spec);

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 16, alternating);

        int[] tokens = [1, 4, 2, 5, 3, 0, 1, 5, 2, 4]; // ten positions span two encoder chunks

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 12, backend);
        using SequenceKv kv = new(pool);

        float[] encoded = model.Encode(tokens, kv);
        double[][] expected = reference.Encoded(tokens, Window, alternating: true);

        for (int p = 0; p < tokens.Length; p++)
        {
            for (int i = 0; i < model.Plan.HiddenSize; i++)
            {
                float actual = encoded[p * model.Plan.HiddenSize + i];
                double expectedValue = expected[p][i];
                Assert.True(
                    Math.Abs(actual - expectedValue) <= 1e-5 * Math.Max(1.0, Math.Abs(expectedValue)),
                    $"encode hidden[{p}][{i}] under the alternating pattern: expected {expectedValue:G9}, got {actual:G9}");
            }
        }

        Assert.True(
            Differs(encoded, reference.Encoded(tokens, Window, alternating: false)),
            "the alternating encode must differ from the reference masking every layer");
    }

    // ---------------------------------------------------------------- a zero window ignores the pattern

    [Fact]
    public void Forward_AlternatingPatternWithNoDeclaredWindow_IsBitIdenticalToNoPatternAtAll()
    {
        InterleaveModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(null), spec.Tensors()));

        ModelProfile alternating = ModelProfiles.Resolve("llama") with { WindowInterleave = WindowInterleave.AlternatingLocalFirst };

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);

        float[] unpatternedLogits;
        float[] unpatternedEncoded;
        {
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

            Assert.Equal(0, model.Plan.AttentionWindow);
            Assert.Equal(WindowInterleave.AllLayers, model.Plan.WindowInterleave);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 12, backend);
            using SequenceKv kv = new(pool);
            model.Prefill([1, 4, 2, 5, 3], kv);
            unpatternedLogits = model.DecodeNext(0, kv).ToArray();

            using SequenceKv encodeKv = new(pool);
            unpatternedEncoded = model.Encode([1, 4, 2, 5, 3], encodeKv);
        }

        float[] alternatingLogits;
        float[] alternatingEncoded;
        {
            // The same file read under the alternating pattern: an undeclared window is zero, and zero is unbounded
            // on every layer whatever the pattern says, so both forwards are the same computation bit for bit.
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, alternating);

            Assert.Equal(0, model.Plan.AttentionWindow);
            Assert.Equal(WindowInterleave.AlternatingLocalFirst, model.Plan.WindowInterleave);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 12, backend);
            using SequenceKv kv = new(pool);
            model.Prefill([1, 4, 2, 5, 3], kv);
            alternatingLogits = model.DecodeNext(0, kv).ToArray();

            using SequenceKv encodeKv = new(pool);
            alternatingEncoded = model.Encode([1, 4, 2, 5, 3], encodeKv);
        }

        Assert.True(
            alternatingLogits.AsSpan().SequenceEqual(unpatternedLogits.AsSpan()),
            "the alternating pattern with no declared window must decode bit-identically to the same model with no pattern");
        Assert.True(
            alternatingEncoded.AsSpan().SequenceEqual(unpatternedEncoded.AsSpan()),
            "the alternating pattern with no declared window must encode bit-identically to the same model with no pattern");
    }

    // ---------------------------------------------------------------- windowed KV reads and block selection

    [Fact]
    public void Forward_AlternatingWindow_PerLayerSelectionsReadOnlyInWindowBlocks_AndFullLayersSelectWholeTable()
    {
        InterleaveModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(Window), spec.Tensors()));

        ModelProfile alternating = ModelProfiles.Resolve("llama") with { WindowInterleave = WindowInterleave.AlternatingLocalFirst };

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8, alternating);

        Assert.Equal(WindowInterleave.AlternatingLocalFirst, model.Plan.WindowInterleave);
        Assert.Equal(Window, model.Plan.AttentionWindow);

        // Two positions per block, so five positions live in three blocks and a window that excludes the first two
        // positions excludes an entire physical block instead of merely trimming a partial one.
        using BlockPool pool = new(blockTokens: 2, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 8, backend);
        using SequenceKv kv = new(pool);

        model.Prefill([1, 4, 2], kv);
        model.DecodeNext(5, kv); // position 3
        float[] logitsAtFour = model.DecodeNext(3, kv).ToArray(); // position 4: windowed layers see [2..4] only

        Assert.True(
            Closest(logitsAtFour, new InterleaveReference(spec).LogitsAt([1, 4, 2, 5], 3, Window, alternating: true)),
            "the bounded-read forward must still match the full-KV reference masking only the even-indexed layers");

        Assert.Equal(5, kv.Length);
        Assert.Equal(3, kv.Blocks.Count); // positions 0..1 in block 0, 2..3 in block 1, 4 in block 2

        for (int layer = 0; layer < model.Plan.LayerCount; layer++)
        {
            int layerWindow = model.Plan.AttentionWindowForLayer(layer);
            BlockSelection read = kv.SelectCausalRead(queryPosition: 4, layerWindow: layerWindow);

            Assert.Equal(4, read.LastPosition);

            if (layer % 2 == 0)
            {
                // The plan resolves the declared window for this layer, and its selection covers blocks 1..2 only:
                // block 0 — positions 0..1 — is not addressed by any head of a masked layer at this query.
                Assert.Equal(Window, layerWindow);
                Assert.Equal(2, read.FirstPosition);
                Assert.Equal(1, read.FirstBlockIndex);
                Assert.Equal(2, read.SelectedBlocks);
                Assert.Equal(3, read.TotalBlocks);
                Assert.Equal(1, read.SkippedBlocks);
                Assert.False(read.CoversWholeTable);

                // A position the selection does not cover is refused rather than silently reached.
                Assert.Throws<ArgumentOutOfRangeException>(() => kv.ReadK(read, 0, layer));
                Assert.Throws<ArgumentOutOfRangeException>(() => kv.ReadV(read, 1, layer));
            }
            else
            {
                // The full-attention half of the interleave selects every block below its query position: boundedness
                // is provably absent for these layers, on the same model and sequence as the masked ones above.
                Assert.Equal(0, layerWindow);
                Assert.Equal(0, read.FirstPosition);
                Assert.Equal(0, read.FirstBlockIndex);
                Assert.Equal(3, read.SelectedBlocks);
                Assert.Equal(0, read.SkippedBlocks);
                Assert.True(read.CoversWholeTable);

                // The earliest block is inside the selection and reads back its real bytes.
                Assert.Equal(kv.BlockFor(0), kv.BlockFor(read.FirstPosition));
            }
        }
    }

    [Fact]
    public void Forward_BoundedReadsNeverTouchOutOfWindowKv_LiveBytesProveItAgainstTheFullKvReference()
    {
        const int narrow = 2; // a window that excludes whole two-position blocks from position 4 onward

        InterleaveModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(narrow), spec.Tensors()));

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8); // the llama profile masks every layer

        Assert.Equal(WindowInterleave.AllLayers, model.Plan.WindowInterleave);
        Assert.Equal(narrow, model.Plan.AttentionWindow);

        InterleaveReference reference = new(spec);

        using BlockPool pool = new(blockTokens: 2, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 8, backend);

        // Baseline run: the same five tokens on clean KV bytes.
        float[] baseline;
        using (SequenceKv clean = new(pool))
        {
            model.Prefill([1, 4, 2], clean);
            model.DecodeNext(5, clean);
            baseline = model.DecodeNext(3, clean).ToArray(); // position 4 reads [3..4] on every layer
        }

        Assert.True(
            Closest(baseline, reference.LogitsAt([1, 4, 2, 5], 3, narrow, alternating: false)),
            "the all-layers windowed forward must match the full-KV scalar reference masking every layer");

        // Same sequence over KV bytes that carry unmistakable garbage in the positions this query's windows exclude —
        // block 0 (positions 0..1), poisoned after those positions were written and before the bounded read runs. A
        // forward that still lands bit-for-bit on the clean baseline never addressed them; a forward that did would be
        // thrown far off by values this large.
        float[] overPoisoned;
        using SequenceKv poisoned = new(pool); // declared outside the block so the kernel probes below read this same KV
        {
            model.Prefill([1, 4, 2], poisoned);
            model.DecodeNext(5, poisoned); // position 3 lands clean; positions 0..1 are now outside every later window

            for (int position = 0; position <= 1; position++)
            {
                for (int layer = 0; layer < model.Plan.LayerCount; layer++)
                {
                    poisoned.WriteK(position, layer, GarbageRow(model.Plan.KvHeadCount * model.Plan.HeadDim, position, layer, sign: 1));
                    poisoned.WriteV(position, layer, GarbageRow(model.Plan.KvHeadCount * model.Plan.HeadDim, position, layer, sign: -1));
                }
            }

            overPoisoned = model.DecodeNext(3, poisoned).ToArray(); // the bounded forward at position 4

            BlockSelection read = poisoned.SelectCausalRead(queryPosition: 4, layerWindow: narrow);
            Assert.Equal(3, read.FirstPosition); // positions 0..1 are outside every masked layer's selection here
            Assert.Equal(1, read.SkippedBlocks); // the whole poisoned block sits in what this read skips
        }

        Assert.True(
            baseline.AsSpan().SequenceEqual(overPoisoned.AsSpan()),
            "a windowed forward must be bit-identical over KV bytes whose out-of-window positions are poisoned");

        // The same poisoned block is live data: an unbounded selection at the very same position addresses it, and the
        // attention kernel's output moves. So the equality above means "not addressed", not "values coincide".
        int region = model.Plan.HeadCount * model.Plan.HeadDim;
        for (int layer = 0; layer < model.Plan.LayerCount; layer++)
        {
            float[] q = new float[region];
            for (int i = 0; i < region; i++)
            {
                q[i] = 0.25f * (1 + (i % 4));
            }

            float scale = 1f / MathF.Sqrt(model.Plan.HeadDim);
            float[] boundedOut = new float[region];
            LayerOps.CausalAttention(backend, q, model.Plan.HeadCount, model.Plan.KvHeadCount, model.Plan.HeadDim, poisoned, layer, position: 4, scale, softcap: 0f, window: narrow, boundedOut);

            float[] unboundedOut = new float[region];
            LayerOps.CausalAttention(backend, q, model.Plan.HeadCount, model.Plan.KvHeadCount, model.Plan.HeadDim, poisoned, layer, position: 4, scale, softcap: 0f, window: 0, unboundedOut);

            Assert.True(
                !boundedOut.AsSpan().SequenceEqual(unboundedOut.AsSpan()),
                $"layer {layer}: the bounded selection must not address the poisoned block an unbounded read at the same position does");
        }

    }

    [Fact]
    public void SequenceKv_Selection_ExcludesWholeBlocksUnderAWindow_AndAnUnboundedSelectionCoversTheTable()
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using BlockPool pool = new(blockTokens: 2, layerCount: 1, kvHeads: InterleaveModelSpec.KvHeads, headDim: InterleaveModelSpec.HeadDim, capacityBlocks: 8, backend);
        using SequenceKv kv = new(pool);

        for (int position = 0; position < 7; position++) // positions 0..6 across blocks 0..3
        {
            kv.Append();
            kv.WriteK(position, 0, GarbageRow(pool.KvDim, position, layer: 0, sign: 1));
            kv.WriteV(position, 0, GarbageRow(pool.KvDim, position, layer: 0, sign: -1));
        }

        BlockSelection windowed = kv.SelectCausalRead(queryPosition: 6, layerWindow: Window); // [4..6] over a four-block table
        Assert.Equal(4, windowed.FirstPosition);
        Assert.Equal(6, windowed.LastPosition);
        Assert.Equal(2, windowed.FirstBlockIndex);
        Assert.Equal(2, windowed.SelectedBlocks);
        Assert.Equal(3, windowed.LastBlockIndex); // the run is blocks 2..3 of four: whole blocks 0 and 1 are excluded
        Assert.Equal(4, windowed.TotalBlocks);
        Assert.Equal(2, windowed.SkippedBlocks); // blocks 0 and 1 are excluded: whole blocks of live bytes this read never addresses
        Assert.False(windowed.CoversWholeTable);

        BlockSelection unbounded = kv.SelectCausalRead(queryPosition: 6, layerWindow: 0);
        Assert.True(unbounded.CoversWholeTable);
        Assert.Equal(kv.Blocks.Count, unbounded.SelectedBlocks);
        Assert.Equal(0, unbounded.SkippedBlocks);

        // The selection's edges address the same physical blocks the table maps them to.
        Assert.Equal(kv.BlockFor(windowed.FirstPosition), kv.Blocks[windowed.FirstBlockIndex]);
        Assert.Equal(kv.BlockFor(unbounded.LastPosition), kv.Blocks[unbounded.LastBlockIndex]);

        // Reads inside the selection work; reads outside it fail in both directions.
        Assert.True(kv.ReadK(windowed, 4, 0).SequenceEqual(kv.ReadK(4, 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => kv.ReadV(windowed, 3, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => kv.ReadK(windowed, 7, 0));

        // A range read selects exactly the blocks its edges fall in — partial at either edge, whole between.
        BlockSelection ranged = kv.SelectRangeRead(1, 5);
        Assert.Equal(0, ranged.FirstBlockIndex); // block 0 is selected for position 1 alone
        Assert.Equal(2, ranged.LastBlockIndex); // block 2 is selected for positions 4..5
        Assert.Equal(3, ranged.SelectedBlocks);
        Assert.Equal(1, ranged.SkippedBlocks);

        // A selection is a read contract over positions the sequence holds: an edge beyond its length fails.
        Assert.Throws<ArgumentOutOfRangeException>(() => kv.SelectRangeRead(0, 7));
        Assert.Throws<ArgumentOutOfRangeException>(() => kv.SelectCausalRead(queryPosition: 9, layerWindow: Window));
    }

    // ---------------------------------------------------------------- reference oracle

    // Loudly-valued KV rows: magnitudes far past anything this fixture's forwards produce, so a read that addressed
    // them would move logits and attention outputs unmistakably rather than by rounding noise.
    private static float[] GarbageRow(int length, int position, int layer, int sign)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = sign * (50_000f + 1_000f * position + 100f * layer + i);
        }

        return values;
    }

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

    // A complete four-layer model with hidden 6, heads 2, kv-heads 1, a declared head dim of 4, ffn 8, vocab 6 — all
    // F32 tensors with deterministic payloads the reference forward reads directly. Every layer carries its own
    // weights, so a mask that moves one layer moves the logits. No output.weight: the logits come from the tied
    // token-embedding rows.
    private sealed class InterleaveModelSpec
    {
        public const int Layers = 4;
        public const int Hidden = 6;
        public const int Heads = 2;
        public const int KvHeads = 1;
        public const int HeadDim = 4;
        public const int Intermediate = 8;
        public const int Vocab = 6;
        public const float Eps = 1e-5f;
        public const double RopeTheta = 10_000.0;

        public readonly float[] Embeddings;
        public readonly float[][] AttnQ;
        public readonly float[][] AttnK;
        public readonly float[][] AttnV;
        public readonly float[][] AttnO;
        public readonly float[][] FfnGate;
        public readonly float[][] FfnUp;
        public readonly float[][] FfnDown;
        public readonly float[][] AttnNorm;
        public readonly float[][] FfnNorm;
        public readonly float[] OutputNorm;

        public InterleaveModelSpec()
        {
            Embeddings = Matrix(Vocab, Hidden, salt: 411);

            AttnQ = new float[Layers][];
            AttnK = new float[Layers][];
            AttnV = new float[Layers][];
            AttnO = new float[Layers][];
            FfnGate = new float[Layers][];
            FfnUp = new float[Layers][];
            FfnDown = new float[Layers][];
            AttnNorm = new float[Layers][];
            FfnNorm = new float[Layers][];

            for (int layer = 0; layer < Layers; layer++)
            {
                int salt = 11 + 7 * layer;
                AttnQ[layer] = Matrix(Heads * HeadDim, Hidden, salt);
                AttnK[layer] = Matrix(KvHeads * HeadDim, Hidden, salt + 1);
                AttnV[layer] = Matrix(KvHeads * HeadDim, Hidden, salt + 2);
                AttnO[layer] = Matrix(Hidden, Heads * HeadDim, salt + 3);
                FfnGate[layer] = Matrix(Intermediate, Hidden, salt + 4);
                FfnUp[layer] = Matrix(Intermediate, Hidden, salt + 5);
                FfnDown[layer] = Matrix(Hidden, Intermediate, salt + 6);
                AttnNorm[layer] = NormVector(salt + 7);
                FfnNorm[layer] = NormVector(salt + 8);
            }

            OutputNorm = NormVector(97);
        }

        public List<(string Key, object Value)> MetadataPairs(int? window)
        {
            List<(string Key, object Value)> pairs =
            [
                ("general.architecture", "llama"),
                ("llama.embedding_length", Hidden),
                ("llama.block_count", Layers),
                ("llama.attention.head_count", Heads),
                ("llama.attention.head_count_kv", KvHeads),
                ("llama.feed_forward_length", Intermediate),
                ("llama.attention.head_length", HeadDim),
                ("llama.attention.layer_norm_rms_epsilon", Eps),
                ("llama.vocab_size", Vocab),
            ];

            if (window is int declared)
            {
                pairs.Add(("llama.attention.sliding_window", declared));
            }

            return pairs;
        }

        public List<TensorDef> Tensors()
        {
            List<TensorDef> tensors = [new("token_embd.weight", [Vocab, Hidden], Embeddings)];

            for (int layer = 0; layer < Layers; layer++)
            {
                tensors.Add(new($"blk.{layer}.attn_q.weight", [Heads * HeadDim, Hidden], AttnQ[layer]));
                tensors.Add(new($"blk.{layer}.attn_k.weight", [KvHeads * HeadDim, Hidden], AttnK[layer]));
                tensors.Add(new($"blk.{layer}.attn_v.weight", [KvHeads * HeadDim, Hidden], AttnV[layer]));
                tensors.Add(new($"blk.{layer}.attn_output.weight", [Hidden, Heads * HeadDim], AttnO[layer]));
                tensors.Add(new($"blk.{layer}.ffn_gate.weight", [Intermediate, Hidden], FfnGate[layer]));
                tensors.Add(new($"blk.{layer}.ffn_up.weight", [Intermediate, Hidden], FfnUp[layer]));
                tensors.Add(new($"blk.{layer}.ffn_down.weight", [Hidden, Intermediate], FfnDown[layer]));
                tensors.Add(new($"blk.{layer}.attn_norm.weight", [Hidden], AttnNorm[layer]));
                tensors.Add(new($"blk.{layer}.ffn_norm.weight", [Hidden], FfnNorm[layer]));
            }

            tensors.Add(new("output_norm.weight", [Hidden], OutputNorm));

            return tensors;
        }

        // Deterministic payloads on a 0.125 grid: the same values reach the reference forward, every layer's weights
        // differ from every other layer's, and the modest magnitudes keep each layer's softmax broad enough that the
        // engine's float32 path stays inside the shared 1e-5 tolerance against the double-precision reference.
        private static float[] Matrix(int rows, int cols, int salt)
        {
            float[] values = new float[rows * cols];
            for (int i = 0; i < values.Length; i++)
            {
                double raw = Math.Sin((salt * 131 + i * 17) * 0.21);
                values[i] = (float)Math.Round(raw * 8.0) / 8.0f;
            }

            return values;
        }

        // Norm weights clearly off zero and varied across the vector, so consuming a layer's norms is visible.
        private static float[] NormVector(int salt)
        {
            float[] values = new float[Hidden];
            for (int i = 0; i < values.Length; i++)
            {
                double raw = Math.Sin((salt * 43 + i * 29) * 0.17);
                values[i] = 1.0f + ((float)Math.Round(raw * 4.0) / 4.0f) * 0.2f;
            }

            return values;
        }
    }

    // A scalar reference forward over the spec's tensors for either interleave: RMSNorm, standard full-head RoPE,
    // causal or bidirectional attention with an explicit softmax and the score mask applied per layer — every layer
    // when the pattern masks all of them, the even-indexed layers only when it alternates — where a masked layer's
    // range is backward-only from its own query in both paths, silu(gate) * up, and the tied token-embedding logits.
    // A resolved window of zero leaves every layer unbounded whatever the pattern, which is the whole sequence in both
    // directions. Computed in double and compared against the engine's float values with the shared tolerance.
    private sealed class InterleaveReference
    {
        private readonly InterleaveModelSpec _spec;

        public InterleaveReference(InterleaveModelSpec spec) => _spec = spec;

        /// <summary>Logits for a query token forwarded at position historyTokens.Length after replaying the sequence's earlier tokens through every layer.</summary>
        public double[] LogitsAt(int[] historyTokens, int queryToken, int window, bool alternating)
        {
            int positions = historyTokens.Length + 1;
            double[][][] kvK = LayeredKv(positions);
            double[][][] kvV = LayeredKv(positions);

            for (int p = 0; p < historyTokens.Length; p++)
            {
                Advance(Row(_spec.Embeddings, historyTokens[p], InterleaveModelSpec.Hidden), p, kvK, kvV, window, alternating);
            }

            double[] cur = Row(_spec.Embeddings, queryToken, InterleaveModelSpec.Hidden);
            Advance(cur, historyTokens.Length, kvK, kvV, window, alternating);

            double[] normed = Norm(cur, _spec.OutputNorm);
            return Gemv(_spec.Embeddings, InterleaveModelSpec.Vocab, InterleaveModelSpec.Hidden, normed); // tied lm_head
        }

        public double[][] Encoded(int[] tokens, int window, bool alternating)
        {
            int n = tokens.Length;
            double[][] cur = new double[n][];
            for (int i = 0; i < n; i++)
            {
                cur[i] = Row(_spec.Embeddings, tokens[i], InterleaveModelSpec.Hidden);
            }

            double[][][] kvK = LayeredKv(n);
            double[][][] kvV = LayeredKv(n);

            for (int layer = 0; layer < InterleaveModelSpec.Layers; layer++)
            {
                int layerWindow = LayerWindow(layer, window, alternating);

                double[][] qAll = new double[n][];
                for (int p = 0; p < n; p++)
                {
                    double[] normed = Norm(cur[p], _spec.AttnNorm[layer]);
                    double[] q = Gemv(_spec.AttnQ[layer], InterleaveModelSpec.Heads * InterleaveModelSpec.HeadDim, InterleaveModelSpec.Hidden, normed);
                    double[] k = Gemv(_spec.AttnK[layer], InterleaveModelSpec.KvHeads * InterleaveModelSpec.HeadDim, InterleaveModelSpec.Hidden, normed);
                    double[] v = Gemv(_spec.AttnV[layer], InterleaveModelSpec.KvHeads * InterleaveModelSpec.HeadDim, InterleaveModelSpec.Hidden, normed);

                    Rope(q, p);
                    Rope(k, p);

                    qAll[p] = q;
                    kvK[layer][p] = k;
                    kvV[layer][p] = v;
                }

                double[] attnOut = new double[n * InterleaveModelSpec.Heads * InterleaveModelSpec.HeadDim];
                for (int p = 0; p < n; p++)
                {
                    (int from, int to) = layerWindow > 0
                        ? (Math.Max(0, p - layerWindow + 1), p) // backward-only: a masked layer never sees past its own query
                        : (0, n - 1);

                    Attention(qAll[p], kvK[layer], kvV[layer], from, to, attnOut.AsSpan(p * InterleaveModelSpec.Heads * InterleaveModelSpec.HeadDim));
                }

                for (int p = 0; p < n; p++)
                {
                    double[] proj = Gemv(_spec.AttnO[layer], InterleaveModelSpec.Hidden, InterleaveModelSpec.Heads * InterleaveModelSpec.HeadDim, attnOut.AsSpan(p * InterleaveModelSpec.Heads * InterleaveModelSpec.HeadDim).ToArray());
                    for (int i = 0; i < InterleaveModelSpec.Hidden; i++)
                    {
                        cur[p][i] += proj[i];
                    }
                }

                for (int p = 0; p < n; p++)
                {
                    double[] normed = Norm(cur[p], _spec.FfnNorm[layer]);
                    double[] gate = Gemv(_spec.FfnGate[layer], InterleaveModelSpec.Intermediate, InterleaveModelSpec.Hidden, normed);
                    double[] up = Gemv(_spec.FfnUp[layer], InterleaveModelSpec.Intermediate, InterleaveModelSpec.Hidden, normed);
                    for (int i = 0; i < InterleaveModelSpec.Intermediate; i++)
                    {
                        gate[i] = Silu(gate[i]) * up[i];
                    }

                    double[] down = Gemv(_spec.FfnDown[layer], InterleaveModelSpec.Hidden, InterleaveModelSpec.Intermediate, gate);
                    for (int i = 0; i < InterleaveModelSpec.Hidden; i++)
                    {
                        cur[p][i] += down[i];
                    }
                }
            }

            double[][] encoded = new double[n][];
            for (int i = 0; i < n; i++)
            {
                encoded[i] = Norm(cur[i], _spec.OutputNorm);
            }

            return encoded;
        }

        // One causal decode step: every layer takes the stream from the layer below, exactly as the engine's
        // single-token forward does.
        private void Advance(double[] cur, int position, double[][][] kvK, double[][][] kvV, int window, bool alternating)
        {
            for (int layer = 0; layer < InterleaveModelSpec.Layers; layer++)
            {
                int layerWindow = LayerWindow(layer, window, alternating);

                double[] normed = Norm(cur, _spec.AttnNorm[layer]);
                double[] q = Gemv(_spec.AttnQ[layer], InterleaveModelSpec.Heads * InterleaveModelSpec.HeadDim, InterleaveModelSpec.Hidden, normed);
                double[] k = Gemv(_spec.AttnK[layer], InterleaveModelSpec.KvHeads * InterleaveModelSpec.HeadDim, InterleaveModelSpec.Hidden, normed);
                double[] v = Gemv(_spec.AttnV[layer], InterleaveModelSpec.KvHeads * InterleaveModelSpec.HeadDim, InterleaveModelSpec.Hidden, normed);

                Rope(q, position);
                Rope(k, position);

                kvK[layer][position] = k;
                kvV[layer][position] = v;

                double[] attnOut = new double[InterleaveModelSpec.Heads * InterleaveModelSpec.HeadDim];
                Attention(q, kvK[layer], kvV[layer], layerWindow > 0 ? Math.Max(0, position - layerWindow + 1) : 0, position, attnOut);

                double[] proj = Gemv(_spec.AttnO[layer], InterleaveModelSpec.Hidden, InterleaveModelSpec.Heads * InterleaveModelSpec.HeadDim, attnOut);
                for (int i = 0; i < InterleaveModelSpec.Hidden; i++)
                {
                    cur[i] += proj[i];
                }

                double[] ffnNormed = Norm(cur, _spec.FfnNorm[layer]);
                double[] gate = Gemv(_spec.FfnGate[layer], InterleaveModelSpec.Intermediate, InterleaveModelSpec.Hidden, ffnNormed);
                double[] up = Gemv(_spec.FfnUp[layer], InterleaveModelSpec.Intermediate, InterleaveModelSpec.Hidden, ffnNormed);
                for (int i = 0; i < InterleaveModelSpec.Intermediate; i++)
                {
                    gate[i] = Silu(gate[i]) * up[i];
                }

                double[] down = Gemv(_spec.FfnDown[layer], InterleaveModelSpec.Hidden, InterleaveModelSpec.Intermediate, gate);
                for (int i = 0; i < InterleaveModelSpec.Hidden; i++)
                {
                    cur[i] += down[i];
                }
            }
        }

        // The pattern as the forward resolves it: the declared window on the even-indexed layers and nothing on the
        // odd ones when the interleave alternates, the declared window everywhere otherwise.
        private static int LayerWindow(int layer, int window, bool alternating) =>
            window == 0 || !alternating ? window : layer % 2 == 0 ? window : 0;

        private static double[][][] LayeredKv(int positions)
        {
            double[][][] layered = new double[InterleaveModelSpec.Layers][][];
            for (int layer = 0; layer < InterleaveModelSpec.Layers; layer++)
            {
                layered[layer] = new double[positions][];
            }

            return layered;
        }

        private static void Attention(double[] q, double[][] kvK, double[][] kvV, int from, int to, Span<double> outHead)
        {
            const int headDim = InterleaveModelSpec.HeadDim;
            double scale = 1.0 / Math.Sqrt(headDim);

            // heads 2, kv-heads 1: every head reads kv-head (h * 1 / 2) == 0.
            for (int h = 0; h < InterleaveModelSpec.Heads; h++)
            {
                double[] scores = new double[to - from + 1];
                double max = double.NegativeInfinity;
                for (int t = from; t <= to; t++)
                {
                    double dot = 0.0;
                    for (int j = 0; j < headDim; j++)
                    {
                        dot += q[h * headDim + j] * kvK[t][j];
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

                for (int j = 0; j < headDim; j++)
                {
                    double acc = 0.0;
                    for (int t = from; t <= to; t++)
                    {
                        acc += scores[t - from] / sum * kvV[t][j];
                    }

                    outHead[h * headDim + j] = acc;
                }
            }
        }

        private static void Rope(double[] headRegion, int position)
        {
            for (int h = 0; h < headRegion.Length / InterleaveModelSpec.HeadDim; h++)
            {
                for (int i = 0; i < InterleaveModelSpec.HeadDim / 2; i++)
                {
                    double invFreq = Math.Pow(InterleaveModelSpec.RopeTheta, -2.0 * i / InterleaveModelSpec.HeadDim);
                    double angle = position * invFreq;
                    double c = Math.Cos(angle);
                    double s = Math.Sin(angle);

                    double lo = headRegion[h * InterleaveModelSpec.HeadDim + 2 * i];
                    double hi = headRegion[h * InterleaveModelSpec.HeadDim + 2 * i + 1];
                    headRegion[h * InterleaveModelSpec.HeadDim + 2 * i] = lo * c - hi * s;
                    headRegion[h * InterleaveModelSpec.HeadDim + 2 * i + 1] = hi * c + lo * s;
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

            double invRms = 1.0 / Math.Sqrt(sumSq / n + InterleaveModelSpec.Eps);
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
        string path = Path.Combine(Path.GetTempPath(), $"lair-window-interleave-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }
}
