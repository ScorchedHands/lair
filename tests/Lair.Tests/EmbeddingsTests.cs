using System.Runtime.InteropServices;
using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Encoder-only forward groundwork: a hand-computed oracle for the full-window attention op, bit-identity between its
/// restricted prefix window and the causal decode kernel (and inequality with it wherever that window is strictly
/// narrower), encoder shape / finiteness / argument validation on a synthetic two-layer tied-weight GGUF, run-to-run and
/// thread-count determinism of encode plus pooling across chunk boundaries, mean-vs-CLS divergence, L2 normalization,
/// sequential one-token encoder positions matching incremental decode bit-for-bit through the shared tied lm_head,
/// full-window divergence from causal-prefix runs at every position past the first, merged-batch encoder forwards
/// reproducing per-sequence encodes in input order with their own argument validation, pure pooling math against
/// hand-computed values, and a skip-if-absent SmolLM F16 pass proving finite unit-norm repeatable distinct pooled vectors.
/// </summary>
[Trait("Category", "Embeddings")]
public sealed class EmbeddingsTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

    // Synthetic model: two layers, hidden 4, heads 2, one kv-head (GQA mapping h*1/2 sends both heads to kv-head 0),
    // head dim 2 (one RoPE pair per head), ffn 8, vocab 6, no output.weight — the tied lm_head path is exercised.
    private const int Layers = 2;
    private const int Hidden = 4;
    private const int Heads = 2;
    private const int KvHeads = 1;
    private const int HeadDim = 2;
    private const int Ffn = 8;
    private const int Vocab = 6;

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

    // ---------------------------------------------------------------- attention op units

    [Fact]
    public void BidirectionalAttention_FullWindow_MatchesHandComputedSoftmaxOverEveryPosition()
    {
        const int Positions = 3;
        const int Region = Heads * HeadDim; // 4 floats per position, kv-heads share via the GQA mapping

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using BlockPool pool = new(blockTokens: 2, layerCount: 1, kvHeads: KvHeads, headDim: HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        // Distinct K/V rows per position so every query sees three different score terms. Both heads read the single
        // kv-head (h * 1 / 2 == 0), so these rows serve all queries directly.
        float[][] kRows = [[0.5f, -0.25f], [1f, 0f], [-0.75f, 0.6f]];
        float[][] vRows = [[0.25f, 0.9f], [-1f, 0.125f], [0.4f, -0.3f]];

        for (int t = 0; t < Positions; t++)
        {
            kv.Append();
            kv.WriteK(t, 0, kRows[t]); // kvRegion == HeadDim here: one kv-head
            kv.WriteV(t, 0, vRows[t]);
        }

        float[] qPositions = new float[Positions * Region];
        for (int p = 0; p < Positions; p++)
        {
            for (int h = 0; h < Heads; h++)
            {
                for (int j = 0; j < HeadDim; j++)
                {
                    qPositions[p * Region + h * HeadDim + j] = ((p * 7 + h * 3 + j) % 5 - 2) * 0.5f; // finite, signed, distinct
                }
            }
        }

        float scale = 1f / MathF.Sqrt(HeadDim);
        float[] outConcat = new float[Positions * Region];
        LayerOps.BidirectionalAttention(backend, qPositions, Positions, queryPositionOffset: 0, fromPosition: 0, toPosition: Positions - 1, Heads, KvHeads, HeadDim, kv, layer: 0, scale, softcap: 0f, window: 0, outConcat);

        for (int p = 0; p < Positions; p++)
        {
            for (int h = 0; h < Heads; h++)
            {
                // Hand oracle in double precision over the exact stored float rows: full window 0..Positions-1.
                double[] scores = new double[Positions];
                double maxScore = double.NegativeInfinity;
                for (int t = 0; t < Positions; t++)
                {
                    double dot = 0;
                    for (int j = 0; j < HeadDim; j++)
                    {
                        dot += qPositions[p * Region + h * HeadDim + j] * kRows[t][j]; // both heads share kv-head 0
                    }

                    scores[t] = dot * scale;
                    maxScore = Math.Max(maxScore, scores[t]);
                }

                double sumExp = 0;
                for (int t = 0; t < Positions; t++)
                {
                    sumExp += Math.Exp(scores[t] - maxScore);
                }

                for (int j = 0; j < HeadDim; j++)
                {
                    double expected = 0;
                    for (int t = 0; t < Positions; t++)
                    {
                        expected += Math.Exp(scores[t] - maxScore) / sumExp * vRows[t][j];
                    }

                    float actual = outConcat[p * Region + h * HeadDim + j];
                    Assert.True(float.IsFinite(actual), $"position {p} head {h} dim {j}: output {actual} is not finite");
                    Assert.True(
                        Math.Abs(actual - expected) <= 1e-5,
                        $"position {p} head {h} dim {j}: expected {expected:G9}, got {actual:G9}");
                }
            }
        }

        // A query window must be ordered and must stay inside the KV the sequence actually holds.
        Assert.Throws<ArgumentException>(() => LayerOps.BidirectionalAttention(backend, qPositions, Positions, queryPositionOffset: 0, fromPosition: 2, toPosition: 1, Heads, KvHeads, HeadDim, kv, 0, scale, softcap: 0f, window: 0, outConcat));
        Assert.Throws<ArgumentException>(() => LayerOps.BidirectionalAttention(backend, qPositions, Positions, queryPositionOffset: 0, fromPosition: 0, toPosition: Positions, Heads, KvHeads, HeadDim, kv, 0, scale, softcap: 0f, window: 0, outConcat));

        // Operands must cover every query's full set of heads.
        Assert.Throws<ArgumentException>(() => LayerOps.BidirectionalAttention(backend, qPositions, Positions, queryPositionOffset: 0, fromPosition: 0, toPosition: Positions - 1, Heads, KvHeads, HeadDim, kv, 0, scale, softcap: 0f, window: 0, new float[Positions * Region - HeadDim])); // short output buffer
    }

    [Fact]
    public void BidirectionalAttention_WindowToCausalPrefix_IsBitIdenticalAndWidenedWindowDiffers()
    {
        const int Positions = 4; // spans two physical KV blocks: reads across the boundary must be exact
        const int Region = Heads * HeadDim;

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using BlockPool pool = new(blockTokens: 2, layerCount: 1, kvHeads: KvHeads, headDim: HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        float scale = 1f / MathF.Sqrt(HeadDim);

        for (int t = 0; t < Positions; t++)
        {
            kv.Append();
            kv.WriteK(t, 0, [MathF.Sin(0.7f * t + 0.3f), MathF.Cos(1.1f * t)]); // deterministic distinct rows
            kv.WriteV(t, 0, [MathF.Sin(2.3f * t + 1f), -MathF.Cos(0.9f * t)]);
        }

        float[] qPositions = new float[Positions * Region];
        for (int i = 0; i < qPositions.Length; i++)
        {
            qPositions[i] = MathF.Sin(0.31f * i) + 0.5f * MathF.Cos(1.7f * i);
        }

        float[] fullOut = new float[Positions * Region];
        LayerOps.BidirectionalAttention(backend, qPositions, Positions, queryPositionOffset: 0, fromPosition: 0, toPosition: Positions - 1, Heads, KvHeads, HeadDim, kv, layer: 0, scale, softcap: 0f, window: 0, fullOut);

        for (int p = 0; p < Positions; p++)
        {
            float[] causalOut = new float[Region];
            LayerOps.CausalAttention(backend, qPositions.AsSpan(p * Region, Region), Heads, KvHeads, HeadDim, kv, layer: 0, position: p, scale, softcap: 0f, window: 0, causalOut);

            // Restricting the bidirectional window to exactly the causal prefix must reproduce CausalAttention bit-for-bit
            // — this proves the widened loop is a genuine superset of the decode kernel's numeric path.
            float[] windowedOut = new float[Positions * Region];
            LayerOps.BidirectionalAttention(backend, qPositions, Positions, queryPositionOffset: 0, fromPosition: 0, toPosition: p, Heads, KvHeads, HeadDim, kv, layer: 0, scale, softcap: 0f, window: 0, windowedOut);

            AssertBitsEqual(causalOut, windowedOut.AsSpan(p * Region, Region), $"window [0..{p}] vs causal at position {p}");

            if (p < Positions - 1) // only positions with a strictly narrower causal window must diverge under the full one
            {
                Assert.True(
                    !BitsEqual(fullOut.AsSpan(p * Region, Region), causalOut),
                    $"position {p}: full-window attention output must differ from the causal prefix result — the widened window is not an alias of CausalAttention");
            }
        }

        // The final position's causal window already covers every row: there its full-window run must coincide bit-for-bit.
        float[] lastCausalOut = new float[Region];
        LayerOps.CausalAttention(backend, qPositions.AsSpan((Positions - 1) * Region, Region), Heads, KvHeads, HeadDim, kv, layer: 0, position: Positions - 1, scale, softcap: 0f, window: 0, lastCausalOut);

        AssertBitsEqual(fullOut.AsSpan((Positions - 1) * Region, Region), lastCausalOut, "full window vs causal at the final position (identical windows)");

        // A mid-sequence block-local window (chunking shape) sees exactly its own range: distinct from both the full
        // window and the causal prefix, confirming windows are honored literally.
        float[] blockedOut = new float[Positions * Region];
        LayerOps.BidirectionalAttention(backend, qPositions, Positions, queryPositionOffset: 0, fromPosition: 1, toPosition: 2, Heads, KvHeads, HeadDim, kv, layer: 0, scale, softcap: 0f, window: 0, blockedOut);

        for (int p = 0; p < Positions; p++)
        {
            Assert.True(float.IsFinite(blockedOut[p * Region]), "block-window output must stay finite");
        }

        Assert.True(!BitsEqual(blockedOut.AsSpan(3 * Region, Region), fullOut.AsSpan(3 * Region, Region)), "position 3 under window [1..2] must differ from the full window");
    }

    // ---------------------------------------------------------------- synthetic model encoder forward

    [Fact]
    public void Encode_SyntheticModel_ReturnsFiniteRowsAndRejectsBadArguments()
    {
        string path = WriteTemp(BuildTinyGguf());

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 64);
        Assert.False(model.Desc.HasOutputWeights); // tied lm_head path

        int[] tokens = [1, 3, 5, 2];

        using BlockPool pool = NewTinyPool(backend, capacityBlocks: 8);
        using SequenceKv kv = new(pool);

        float[] encoded = model.Encode(tokens, kv);

        Assert.Equal(tokens.Length * Hidden, encoded.Length); // position-major rows of hidden floats
        foreach (float v in encoded)
        {
            Assert.True(float.IsFinite(v), $"encoded value {v} is not finite");
        }

        Assert.True(MaxAbs(encoded) > 1e-6f, "encoder forward produced a degenerate hidden buffer");
        Assert.Equal(tokens.Length, kv.Length); // every position's K/V appended as usual

        // Encoding onto an occupied sequence exercises the absolute-position offset path: one fresh token at position 2.
        using SequenceKv continued = new(pool);
        model.Prefill([4, 0], continued);
        float[] tail = model.Encode([3], continued);
        Assert.Equal(Hidden, tail.Length);
        foreach (float v in tail)
        {
            Assert.True(float.IsFinite(v), $"continued encoder value {v} is not finite");
        }

        // Argument validation: empty / oversized token lists are rejected up front.
        using SequenceKv guard = new(pool);
        Assert.Throws<ArgumentException>(() => model.Encode([], guard));
        Assert.Throws<ArgumentException>(() => model.Encode(new int[TransformerModel.MaxEncodeTokens + 1], guard));

        // Out-of-vocabulary ids and context overflow name the problem; kv stays untouched by a rejected call.
        Assert.Throws<ArgumentException>(() => model.Encode([Vocab, 0], guard));
        Assert.Equal(0, guard.Length);

        using TransformerModel shortContext = TransformerModel.Open(gguf, backend, maxContextTokens: 8);
        using SequenceKv overflow = new(pool);
        ArgumentException contextEx = Assert.Throws<ArgumentException>(() => shortContext.Encode(new int[TransformerModel.MaxEncodeTokens], overflow));
        Assert.Contains("max context", contextEx.Message);
    }

    [Fact]
    public void Encode_AndPooling_RepeatRunsAreBitIdentical_MeanDiffersFromCls()
    {
        string path = WriteTemp(BuildTinyGguf());

        int[] shortRun = [1, 2, 3, 4, 5]; // one encoder chunk
        int[] longRun = [0, 5, 1, 4, 2, 3, 0, 5, 1]; // crosses the chunk boundary

        using GgufFile ggufA = GgufFile.Open(path);
        using CpuBackend backendA = new(threads: 1, allowAvx2: true);
        using TransformerModel modelA = TransformerModel.Open(ggufA, backendA, maxContextTokens: 32);

        using GgufFile ggufB = GgufFile.Open(path);
        using CpuBackend backendB = new(threads: 4, allowAvx2: true); // a different thread count must not shift any bit
        using TransformerModel modelB = TransformerModel.Open(ggufB, backendB, maxContextTokens: 32);

        using BlockPool poolA = NewTinyPool(backendA, capacityBlocks: 8);
        using BlockPool poolB = NewTinyPool(backendB, capacityBlocks: 8);
        using SequenceKv kvA1 = new(poolA);
        using SequenceKv kvA2 = new(poolA); // same instance, second call: buffer reuse must not leak state
        using SequenceKv kvB = new(poolB);

        foreach (int[] tokens in new int[][] { shortRun, longRun })
        {
            float[] run1 = modelA.Encode(tokens, kvA1);
            float[] run2 = modelA.Encode(tokens, kvA2); // repeated call on one engine instance
            float[] run3 = modelB.Encode(tokens, kvB); // independent engine and thread count

            AssertBitsEqual(run1, run2, $"repeat encode of {tokens.Length} tokens on the same engine");
            AssertBitsEqual(run1, run3, $"encode of {tokens.Length} tokens across engines/thread counts");

            float[] mean1 = new float[Hidden];
            float[] mean2 = new float[Hidden];
            Pooling.MeanPool(run1, Hidden, mean1);
            Pooling.MeanPool(run2, Hidden, mean2); // stable pooling over repeated runs: bit-identical results

            AssertBitsEqual(mean1, mean2, $"mean pool stability for {tokens.Length} tokens");

            float[] cls = new float[Hidden];
            Pooling.FirstToken(run1, Hidden, cls); // the first position's row verbatim

            Assert.True(!BitsEqual(mean1, cls), $"mean and CLS vectors must differ for {tokens.Length} distinct positions");

            foreach (float v in mean1)
            {
                Assert.True(float.IsFinite(v), "pooled value is not finite");
            }

            Pooling.NormalizeL2(cls); // L2-normalized variant of the same row differs from both raw vectors and has unit norm
            double norm = 0;
            for (int i = 0; i < Hidden; i++)
            {
                norm += (double)cls[i] * cls[i];
            }

            Assert.True(Math.Abs(norm - 1.0) <= 1e-5, $"normalized CLS has L2 norm {norm:G9}, expected ~1");
        }
    }

    [Fact]
    public void Encode_SequentialPositions_MatchIncrementalDecode_BitForBit()
    {
        string path = WriteTemp(BuildTinyGguf()); // tied weights: the test can apply the same lm_head GEMV to encoder rows

        int[] tokens = [1, 3, 5, 2, 4, 0, 5]; // seven positions across both layers

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 32);

        LogicalTensor embd = gguf.Tensors.Find("token_embd.weight")!;
        Assert.Equal(Vocab * Hidden * 4L, embd.SizeBytes); // F32 fixture rows feed the comparison GEMV directly

        using GgufMmapWeightSource embSource = new(gguf);
        ReadOnlySpan<byte> embBytes = embSource.Resolve(embd);
        WeightRef embRef = new(embBytes, embd.Shape, embd.Format); // identical bytes and shape to the engine's tied lm_head

        using BlockPool pool = NewTinyPool(backend, capacityBlocks: 16);
        using SequenceKv encoderKv = new(pool);
        using SequenceKv decodeKv = new(pool);

        for (int p = 0; p < tokens.Length; p++)
        {
            // One-token Encode at the next position: its window [0..p] is exactly that position's causal prefix, and it
            // reuses the same norms, batched-GEMV kernels, RoPE tables, KV addressing, residuals, and SwiGLU as decode.
            float[] row = model.Encode([tokens[p]], encoderKv);

            Span<float> logits = model.DecodeNext(tokens[p], decodeKv);
            float[] expectedLogits = logits.ToArray(); // incremental causal path for the identical token stream

            float[] headLogits = new float[Vocab];
            backend.Gemv(row, embRef, headLogits);

            AssertBitsEqual(expectedLogits, headLogits, $"sequential encoder position {p} vs incremental decode logits");
        }

        // Sequential one-token encodes are incremental by construction: each position's window is exactly its causal
        // prefix, so the encoder machinery and the decode kernel must agree at every step of a growing sequence.
        Assert.Equal(tokens.Length, encoderKv.Length);
    }

    [Fact]
    public void Encode_FullWindow_DiffersFromCausalPrefixOutputs_AtEveryPositionPastTheFirst()
    {
        string path = WriteTemp(BuildTinyGguf()); // tied weights: compare through the shared lm_head, hidden-level too
        int[] tokens = [1, 3, 5, 2, 4, 0]; // six positions; every prefix below full length has a strictly narrower window

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 32);

        LogicalTensor embd = gguf.Tensors.Find("token_embd.weight")!;

        using GgufMmapWeightSource embSource = new(gguf);
        ReadOnlySpan<byte> embBytes = embSource.Resolve(embd);
        WeightRef embRef = new(embBytes, embd.Shape, embd.Format); // identical bytes and shape to the engine's tied lm_head

        using BlockPool pool = NewTinyPool(backend, capacityBlocks: 16);

        float[] fullEncoded;
        using (SequenceKv fullKv = new(pool))
        {
            fullEncoded = model.Encode(tokens, fullKv);
        }

        for (int p = 1; p < tokens.Length - 1; p++) // positions past the first: their causal window is narrower than the full one
        {
            float[] prefixEncoded;
            using (SequenceKv prefixKv = new(pool))
            {
                prefixEncoded = model.Encode(tokens[..(p + 1)], prefixKv); // its final position IS p under exactly the causal window
            }

            float[] fullLogits = new float[Vocab];
            backend.Gemv(fullEncoded.AsSpan(p * Hidden, Hidden), embRef, fullLogits);

            float[] prefixLogits = new float[Vocab];
            backend.Gemv(prefixEncoded.AsSpan(p * Hidden, Hidden), embRef, prefixLogits);

            Assert.True(!BitsEqual(fullLogits, prefixLogits), $"position {p}: encoder-mode (full-window) logits match the causal-prefix result — bidirectional wiring is not active");
            Assert.True(MaxAbs(fullLogits) > 1e-6f && MaxAbs(prefixLogits) > 1e-6f, "compared logit vectors look degenerate");

            // The hidden directions themselves diverge: compare unit-normalized rows so the distance measures direction.
            float[] fullRow = fullEncoded.AsSpan(p * Hidden, Hidden).ToArray();
            Pooling.NormalizeL2(fullRow);
            float[] prefixRow = prefixEncoded.AsSpan(p * Hidden, Hidden).ToArray();
            Pooling.NormalizeL2(prefixRow);

            double squaredDistance = 0;
            for (int i = 0; i < Hidden; i++)
            {
                double d = fullRow[i] - prefixRow[i];
                squaredDistance += d * d;
            }

            Assert.True(squaredDistance >= 1e-6, $"position {p}: normalized encoder rows differ by only sqrt({squaredDistance:G6}) — too close to call");
        }
    }

    // ---------------------------------------------------------------- merged encoder batches

    [Fact]
    public void EncodeBatch_MergedCall_ReproducesPerSequenceEncode_WithinFloatSlack()
    {
        string path = WriteTemp(BuildTinyGguf());

        int[][] tokenLists =
        [
            [1, 3, 5, 2],                             // four positions: the ordinary member
            [4],                                      // one position: the degenerate member beside a wide one
            [0, 5, 1, 4, 2, 3, 0, 5, 1],             // nine positions: crosses the chunk boundary on its own
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],    // twelve positions: repeated ids still carry distinct positions
        ];

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 64);

        using BlockPool pool = NewTinyPool(backend, capacityBlocks: 32);

        List<SequenceKv> batchKv = [];
        try
        {
            foreach (int[] _ in tokenLists)
            {
                batchKv.Add(new SequenceKv(pool));
            }

            IReadOnlyList<float[]> batched = model.EncodeBatch(tokenLists, batchKv);

            Assert.Equal(tokenLists.Length, batched.Count); // one result array per input sequence, in input order

            for (int s = 0; s < tokenLists.Length; s++)
            {
                Assert.Equal(tokenLists[s].Length * Hidden, batched[s].Length); // laid out exactly like that sequence's single-call result
                Assert.Equal(tokenLists[s].Length, batchKv[s].Length);          // every member advanced its own sequence, not a shared one

                using SequenceKv solo = new(pool);
                float[] single = model.Encode(tokenLists[s], solo); // the independent per-sequence forward this batch must reproduce

                AssertClose(batched[s], single, $"merged batch member {s} vs its independent encode");
            }

            // Members stay independent: a member sharing content with another lands on the same vector, and a member with different content does not.
            Assert.True(!BitsEqual(batched[0], batched[1]), "members with different tokens must not share a hidden buffer");
        }
        finally
        {
            foreach (SequenceKv kv in batchKv)
            {
                kv.Dispose();
            }
        }
    }

    [Fact]
    public void EncodeBatch_PreservesMemberOrder_RepeatInputsMatch_AndRejectsBadArguments()
    {
        string path = WriteTemp(BuildTinyGguf());

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 64);

        using BlockPool pool = NewTinyPool(backend, capacityBlocks: 32);

        int[] a = [1, 2, 3, 4];
        int[] b = [5, 0, 1];

        using (SequenceKv kvA1 = new(pool))
        using (SequenceKv kvB = new(pool))
        using (SequenceKv kvA2 = new(pool))
        {
            IReadOnlyList<float[]> batched = model.EncodeBatch([a, b, a], [kvA1, kvB, kvA2]);

            // Order is the order the inputs arrived in: slot 0 and slot 2 are the same content, and they answer with the same bits.
            Assert.Equal(3, batched.Count);
            AssertBitsEqual(batched[0], batched[2], "the same token list twice in one merged call");
            Assert.True(!BitsEqual(batched[0], batched[1]), "different members must not answer with the same vector");

            using SequenceKv solo = new(pool);
            AssertClose(batched[1], model.Encode(b, solo), "batch member 1 vs its independent encode");
        }

        using SequenceKv guard = new(pool);

        // Shape refusals name the batch's own ceilings; a rejected call never touches a sequence.
        Assert.Throws<ArgumentException>(() => model.EncodeBatch([], []));
        Assert.Throws<ArgumentException>(() => model.EncodeBatch([a], [guard, guard]));
        Assert.Throws<ArgumentException>(() => model.EncodeBatch([a, a], [guard]));

        IReadOnlyList<IReadOnlyList<int>> tooMany = [.. Enumerable.Repeat((IReadOnlyList<int>)new[] { 1 }, TransformerModel.MaxEncodeSequences + 1)];
        IReadOnlyList<SequenceKv> tooManyKv = [.. Enumerable.Repeat(guard, TransformerModel.MaxEncodeSequences + 1)];
        ArgumentException countError = Assert.Throws<ArgumentException>(() => model.EncodeBatch(tooMany, tooManyKv));
        Assert.Contains("sequences", countError.Message);

        // An empty member and a null entry are named by index; oversized members name the per-sequence ceiling.
        IReadOnlyList<IReadOnlyList<int>> withEmpty = [a, [], b];
        Assert.Throws<ArgumentException>(() => model.EncodeBatch(withEmpty, [guard, guard, guard]));
        IReadOnlyList<IReadOnlyList<int>> withNull = new IReadOnlyList<int>[] { a, null!, b };
        ArgumentException nullError = Assert.Throws<ArgumentException>(() => model.EncodeBatch(withNull, [guard, guard, guard]));
        Assert.Contains("entry 1", nullError.Message);

        // Rows past what one merged call rents compose as several calls instead: the refusal names the row ceiling.
        string path2 = WriteTemp(BuildTinyGguf());
        using GgufFile wideGguf = GgufFile.Open(path2);
        using CpuBackend wideBackend = new(threads: 1, allowAvx2: true);
        using TransformerModel wideModel = TransformerModel.Open(wideGguf, wideBackend, maxContextTokens: TransformerModel.MaxEncodeTokens);

        int[] wide = new int[TransformerModel.MaxEncodeTokens];
        for (int i = 0; i < wide.Length; i++)
        {
            wide[i] = i % Vocab;
        }

        IReadOnlyList<IReadOnlyList<int>> tooManyRows = [.. Enumerable.Repeat((IReadOnlyList<int>)wide, TransformerModel.MaxEncodeBatchRows / TransformerModel.MaxEncodeTokens + 1)];
        IReadOnlyList<SequenceKv> rowKv = [.. Enumerable.Repeat(guard, tooManyRows.Count)];
        ArgumentException rowError = Assert.Throws<ArgumentException>(() => wideModel.EncodeBatch(tooManyRows, rowKv));
        Assert.Contains("token rows", rowError.Message);

        // A bad token id anywhere in the batch is refused before any sequence advances.
        Assert.Throws<ArgumentException>(() => model.EncodeBatch([a, [Vocab]], [guard, guard]));
        Assert.Equal(0, guard.Length);
    }

    // ---------------------------------------------------------------- pooling helpers

    [Fact]
    public void Pooling_MeanFirstTokenAndNormalize_MatchHandComputedMath()
    {
        float[] vectors =
        [
            1f, -2f, 0.5f,
            3f, 4f, -1f,
            -0.25f, 0f, 2f, // three rows of length 3
        ];

        float[] mean = new float[3];
        Pooling.MeanPool(vectors, vectorLength: 3, pooled: mean);
        Assert.Equal(1.25f, mean[0]); // (1 + 3 - 0.25) / 3 exactly representable
        Assert.True(Math.Abs(mean[1] - (2f / 3f)) <= 1e-7f, $"mean[1] {mean[1]:G9}");
        Assert.Equal(0.5f, mean[2]); // (0.5 - 1 + 2) / 3 exactly representable

        float[] twoRows = new float[3];
        Pooling.MeanPool(vectors, vectorLength: 3, positionCount: 2, pooled: twoRows); // partial-count overload
        Assert.Equal(2f, twoRows[0]); // (1 + 3) / 2
        Assert.True(Math.Abs(twoRows[1] - 1f) <= 1e-7f, $"partial mean[1] {twoRows[1]:G9}");
        Assert.Equal(-0.25f, twoRows[2]);

        float[] cls = new float[3];
        Pooling.FirstToken(vectors, vectorLength: 3, pooled: cls); // verbatim bits of the first row
        Assert.True(cls.AsSpan().SequenceEqual(vectors[..3].ToArray()));

        float[] unit = [3f, -4f, 0f];
        Pooling.NormalizeL2(unit); // norm 5: exact halves in binary floats
        Assert.Equal(0.6f, unit[0]);
        Assert.Equal(-0.8f, unit[1]);
        Assert.Equal(0f, unit[2]);

        float[] zero = new float[4];
        Pooling.NormalizeL2(zero); // the zero vector stays untouched instead of dividing by zero
        foreach (float v in zero)
        {
            Assert.Equal(0f, v);
        }

        Assert.Throws<ArgumentException>(() => Pooling.MeanPool(vectors, vectorLength: 4, pooled: new float[4])); // rows do not tile the buffer
        Assert.Throws<ArgumentException>(() => Pooling.MeanPool(vectors, vectorLength: 3, pooled: new float[2])); // output too short
        Assert.Throws<ArgumentException>(() => Pooling.FirstToken(new float[2], vectorLength: 3, pooled: new float[3])); // missing row
    }

    // ---------------------------------------------------------------- real SmolLM fixture (skip-if-absent)

    [Fact]
    public void SmolLM_F16_EncodeAndPooling_AreFiniteUnitNormRepeatableAndPromptDistinct()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        int[] promptA = [.. tokenizer.Encode("The capital of France is Paris")];
        int[] promptB = [.. tokenizer.Encode("Distributed systems replicate state across nodes")];
        Assert.True(promptA.Length >= 4 && promptB.Length >= 4, "prompts must span several encoder chunks");

        using CpuBackend backend = new(threads: 1, allowAvx2: true); // deterministic single-thread encode; pooling is pure math
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 64);

        int hidden = model.Desc.HiddenSize;

        float[] EncodeOnce(int[] tokens)
        {
            using BlockPool pool = new(blockTokens: 16, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks: 8, backend);
            using SequenceKv kv = new(pool);

            float[] encoded = model.Encode(tokens, kv); // fresh sequence per call: every token attends to its own sequence only
            Assert.Equal(tokens.Length * hidden, encoded.Length);
            foreach (float v in encoded)
            {
                Assert.True(float.IsFinite(v), $"encoded value {v} is not finite");
            }

            return encoded;
        }

        float[] run1 = EncodeOnce(promptA);
        float[] run2 = EncodeOnce(promptA); // same input twice on one engine: bit-identical hidden states
        AssertBitsEqual(run1, run2, "SmolLM repeat encode");

        float[] mean1 = new float[hidden];
        float[] mean2 = new float[hidden];
        Pooling.MeanPool(run1, hidden, pooled: mean1);
        Pooling.MeanPool(run2, hidden, pooled: mean2);
        AssertBitsEqual(mean1, mean2, "SmolLM repeat pooling");

        foreach (float v in mean1)
        {
            Assert.True(float.IsFinite(v), $"pooled value {v} is not finite");
        }

        Pooling.NormalizeL2(mean1); // L2-normalized pooled vector: unit norm within float slack
        double norm = 0;
        for (int i = 0; i < hidden; i++)
        {
            norm += (double)mean1[i] * mean1[i];
        }

        Assert.True(Math.Abs(norm - 1.0) <= 1e-5, $"normalized pooled vector has L2 norm {norm:G9}, expected ~1");

        float[] normalizedRun2 = [.. mean2]; // normalize the second run's raw mean too and compare bits after scaling
        Pooling.NormalizeL2(normalizedRun2);
        AssertBitsEqual(mean1, normalizedRun2, "normalized repeat pooling");

        float[] otherMean = new float[hidden];
        Pooling.MeanPool(EncodeOnce(promptB), hidden, pooled: otherMean); // a different prompt must land elsewhere in the space
        double squaredDistance = 0;
        for (int i = 0; i < hidden; i++)
        {
            double d = mean1[i] - otherMean[i];
            squaredDistance += d * d;
        }

        Assert.True(squaredDistance >= 1e-6, $"pooled vectors from different prompts differ by only sqrt({squaredDistance:G6})");

        float[] clsRow = new float[hidden];
        Pooling.FirstToken(run1, hidden, pooled: clsRow); // CLS-style extraction over the real fixture differs from the mean as well
        Assert.True(!BitsEqual(mean2, clsRow), "mean and CLS pooling must not coincide on a multi-token prompt");
    }

    // ---------------------------------------------------------------- shared helpers

    private static BlockPool NewTinyPool(CpuBackend backend, int capacityBlocks) =>
        new(blockTokens: 4, layerCount: Layers, kvHeads: KvHeads, headDim: HeadDim, capacityBlocks, backend);

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-embeddings-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    private static float MaxAbs(ReadOnlySpan<float> values)
    {
        float max = 0f;
        foreach (float v in values)
        {
            max = Math.Max(max, Math.Abs(v));
        }

        return max;
    }

    /// <summary>Compares two encoder outputs value-by-value against the same forward run a different way: every slot finite and within 1e-5 absolute of its independent result — the slack float accumulation orderings leave, nowhere near enough to call two directions equal.</summary>
    private static void AssertClose(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{context}: slot {i} is not finite");
            Assert.True(
                Math.Abs(actual[i] - expected[i]) <= 1e-5,
                $"{context}: slot {i} expected {expected[i]:G9}, got {actual[i]:G9}");
        }

        Assert.True(MaxAbs(expected) > 1e-6f, $"{context}: the reference forward looks degenerate");
    }

    /// <summary>Compares two float windows on the raw bits of the floats: NaN payloads and signed zeros must match too.</summary>
    private static bool BitsEqual(ReadOnlySpan<float> left, ReadOnlySpan<float> right) =>
        MemoryMarshal.AsBytes(left).SequenceEqual(MemoryMarshal.AsBytes(right));

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

    // ---------------------------------------------------------------- synthetic GGUF builder

    private sealed record TensorDef(string Name, int[] Dims); // dims in logical (row-major) order; vectors are one-dimensional

    /// <summary>Two-layer llama metadata: hidden 4, heads 2, kv-heads 1, ffn 8, vocab 6 — head dim derives as 4/2 = 2.</summary>
    private static List<(string Key, object Value)> TinyMetadata() =>
    [
        ("general.architecture", "llama"),
        ("llama.embedding_length", Hidden),
        ("llama.block_count", Layers),
        ("llama.attention.head_count", Heads),
        ("llama.attention.head_count_kv", KvHeads),
        ("llama.feed_forward_length", Ffn),
        ("llama.rope.freq_base", 10_000f),
        ("llama.attention.layer_norm_rms_epsilon", 1e-5f),
        ("llama.vocab_size", Vocab),
    ];

    private static List<TensorDef> TinyTensors()
    {
        List<TensorDef> tensors = [new("token_embd.weight", [Vocab, Hidden])]; // tied lm_head: no output.weight tensor

        for (int l = 0; l < Layers; l++)
        {
            tensors.Add(new($"blk.{l}.attn_q.weight", [Heads * HeadDim, Hidden]));
            tensors.Add(new($"blk.{l}.attn_k.weight", [KvHeads * HeadDim, Hidden]));
            tensors.Add(new($"blk.{l}.attn_v.weight", [KvHeads * HeadDim, Hidden]));
            tensors.Add(new($"blk.{l}.attn_output.weight", [Hidden, Heads * HeadDim]));
            tensors.Add(new($"blk.{l}.ffn_gate.weight", [Ffn, Hidden]));
            tensors.Add(new($"blk.{l}.ffn_up.weight", [Ffn, Hidden]));
            tensors.Add(new($"blk.{l}.ffn_down.weight", [Hidden, Ffn]));
            tensors.Add(new($"blk.{l}.attn_norm.weight", [Hidden]));
            tensors.Add(new($"blk.{l}.ffn_norm.weight", [Hidden]));
        }

        tensors.Add(new("output_norm.weight", [Hidden])); // norm weights land clearly above zero in the payload pattern
        return tensors;
    }

    /// <summary>Builds a GGUF v3 image from KV pairs and F32 tensors with deterministic, non-degenerate payloads.</summary>
    private static byte[] BuildTinyGguf() => BuildGguf(TinyMetadata(), TinyTensors());

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

            bool isNorm = tensor.Name.Contains("norm", StringComparison.Ordinal);
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
}
