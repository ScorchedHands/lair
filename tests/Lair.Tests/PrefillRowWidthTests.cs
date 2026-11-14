using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Prefill row-width identity tests: the resolved count a model plan carries for prompt sub-batches is an engine value
/// distinct from the decode batch ceiling, and every measured width must leave what it moves at nothing. The same tokens
/// forwarded through engines opened at 16, 32, and 64 rows — each with its own buffers sized for that count — land
/// raw-byte identical on the per-token forward's logits and KV bytes across both kernels this build executes; a wider
/// window amortizes weight traversals without touching any row's online-softmax order. A case pins that an ordinary open
/// resolves the plan's default row count, distinct from the decode ceiling, and widths outside the engine's allocation
/// ceiling fail at open naming them instead of silently sizing some other width. The synthetic sweep keeps this identity
/// gate cheap enough to run inside the full suite; pinned-fixture cost curves live in bench measurements rather than here.
/// </summary>
[Trait("Category", "Prefill")]
[Collection("model-loaded")] // shares the serialized real-model lane with the parity suites so these forwards do not load alongside another engine
public sealed class PrefillRowWidthIdentityTests
{
    // The measured candidate widths: the default, one doubling, and the engine's allocation ceiling.
    private static readonly int[] MeasuredRowWidths = [16, 32, 64];

    // Window lengths that straddle every measured width: inside a single sub-batch at each count, past 64 so no candidate
    // forwards the window whole, and one exact ceiling batch.
    private static readonly int[] WidthStraddlingWindows = [63, 64, 65, 70];

    /// <summary>Every straddling window length on each kernel this build executes.</summary>
    public static TheoryData<int, PrefillKernel> WindowLengthsAndKernels()
    {
        TheoryData<int, PrefillKernel> data = [];
        foreach (int tokens in WidthStraddlingWindows)
        {
            data.Add(tokens, PrefillKernel.GemV);
            data.Add(tokens, PrefillKernel.Tiled);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(WindowLengthsAndKernels))]
    public void Prefill_LogitsAndKvBytes_IdenticalAcrossMeasuredRowWidths_BitForBit(int windowTokens, PrefillKernel kernel)
    {
        string modelPath = SyntheticLlamaGguf.Write("row-width-synthetic");

        try
        {
            using GgufFile gguf = GgufFile.Open(modelPath);
            int[] prompt = BuildPrompt(windowTokens);

            using CpuBackend backend = new(threads: 2, allowAvx2: true);

            // The baseline every candidate is compared against: the same tokens forwarded one at a time.
            using TransformerModel referenceModel = TransformerModel.Open(gguf, backend, maxContextTokens: 96);
            Assert.Equal(TransformerModel.DefaultPrefillRowWidth, referenceModel.ResolvedPrefillRowWidth); // an ordinary open resolves the plan's default row count

            using BlockPool pool = NewPool(referenceModel, backend, capacityBlocks: 32);
            using SequenceKv perTokenKv = new(pool);
            float[] perTokenLogits = [];
            foreach (int tokenId in prompt)
            {
                perTokenLogits = [.. referenceModel.DecodeNext(tokenId, perTokenKv)];
            }

            foreach (int width in MeasuredRowWidths)
            {
                using TransformerModel model = TransformerModel.Open(
                    gguf, backend, maxContextTokens: 96, logLine: null, prefillKernel: kernel, prefillRowWidth: width);
                Assert.Equal(width, model.ResolvedPrefillRowWidth); // the engine opened on the row count it was handed
                Assert.Equal(width, model.Plan.PrefillRowWidth);    // and the plan carries the resolved value

                using SequenceKv batchedKv = new(pool);
                float[] batchedLogits = [.. model.Prefill(prompt, batchedKv)];

                string context = $"a {windowTokens}-token window at row width {width} on the '{PrefillKernelNames.Of(kernel)}' kernel";
                AssertBitsEqual(perTokenLogits, batchedLogits, $"{context} against the per-token forward");
                Assert.Equal(ArgMax(perTokenLogits), ArgMax(batchedLogits));

                // The same window split into consecutive sub-batches at exactly this engine's resolved width — the route
                // a scheduler-shaped caller issues — lands on the identical bytes too.
                using SequenceKv splitKv = new(pool);
                float[] splitLogits = [];
                for (int offset = 0; offset < prompt.Length; offset += width)
                {
                    int rows = Math.Min(width, prompt.Length - offset);
                    splitLogits = [.. model.Prefill(prompt.AsSpan(offset, rows).ToArray(), splitKv)];
                }

                AssertBitsEqual(perTokenLogits, splitLogits, $"{context} issued as consecutive windows of {width}");

                // Every position of every layer's K/V slot compared on raw bytes against the per-token route. A row whose
                // attention had reached a position past its own would move its own K/V bytes at the next layer, and a
                // sub-batch boundary that dropped or duplicated a position would show up as a length or slot mismatch here.
                AssertKvBytesEqual(perTokenKv, batchedKv, model.Desc.LayerCount, windowTokens);
            }

            using TransformerModel ceilingModel = TransformerModel.Open(
                gguf, backend, maxContextTokens: 96, logLine: null, prefillKernel: kernel, prefillRowWidth: TransformerModel.MaxPrefillRowWidth);
            Assert.Equal(TransformerModel.MaxPrefillRowWidth, ceilingModel.ResolvedPrefillRowWidth); // the engine's own ceiling resolves like any other candidate width

            using SequenceKv decodeWindow = new(pool);
            float[] decodeLogits = [.. ceilingModel.DecodeNext(1, decodeWindow)]; // a wider prefill resolution leaves the single-token forward serving logits as ever
            Assert.Equal(ceilingModel.Plan.VocabSize, decodeLogits.Length);

            Assert.Throws<ModelException>(() => TransformerModel.Open(
                gguf, backend, maxContextTokens: 96, logLine: null, prefillKernel: kernel, prefillRowWidth: TransformerModel.MaxPrefillRowWidth + 1));
            Assert.Throws<ModelException>(() => TransformerModel.Open(
                gguf, backend, maxContextTokens: 96, logLine: null, prefillKernel: kernel, prefillRowWidth: -3));
        }
        finally
        {
            File.Delete(modelPath);
        }
    }

    [Fact]
    public void Plan_ResolvesMeasuredDefaultPrefillRowWidthDistinctFromDecodeCeiling()
    {
        string modelPath = SyntheticLlamaGguf.Write("row-width-default-plan");

        try
        {
            using GgufFile gguf = GgufFile.Open(modelPath);
            ModelPlan plan = ModelPlans.Resolve(gguf);

            // Every file resolved through the registry today carries the measured default row count, and it is distinct from — not derived from — the decode batch ceiling: raising one family's width moves only its own buffers.
            Assert.Equal(TransformerModel.DefaultPrefillRowWidth, plan.PrefillRowWidth);
            Assert.NotEqual(TransformerModel.MaxBatchSequences, plan.PrefillRowWidth);

            using CpuBackend backend = new();
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 32);
            Assert.Equal(plan.PrefillRowWidth, model.ResolvedPrefillRowWidth);
        }
        finally
        {
            File.Delete(modelPath);
        }
    }

    private static BlockPool NewPool(TransformerModel model, CpuBackend backend, int capacityBlocks) =>
        new(blockTokens: 16, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks, backend);

    // A prompt of exactly the requested token count cycling the synthetic vocabulary's ordinary entries.
    private static int[] BuildPrompt(int tokens)
    {
        int[] cycle = [1, 2, 3]; // hello / world / ! — ordinary entries of the synthetic vocabulary

        List<int> ids = [];
        while (ids.Count < tokens)
        {
            ids.AddRange(cycle);
        }

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

    // Every position of the prompt and every layer: the whole KV a window wrote, compared slot by slot.
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
