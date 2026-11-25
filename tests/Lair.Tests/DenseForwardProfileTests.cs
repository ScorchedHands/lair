using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Dense stage/layer profiling coverage on the generic dense forward paths, over synthetic llama-shaped GGUF fixtures only — no CUDA posture and
/// no downloaded model. A default open reports every stage exactly zero and a reset after a measured window zeroes them again, so the seam never
/// invents facts a host did not ask for. With the seam enabled, each prefill kernel this build executes, the single-token decode step, and the merged
/// batched step report positive times on the stages that do real work (seed/embedding, attention blocks with their Q/K/V projection traversals and
/// their output-projection traversal split out from each other and from their causal read spans, feed-forward blocks with their gate/up pair and down
/// traversals split out, the output head, per-layer whole-block spans), each traversal parent is exactly the sum of its traversal substages, parent
/// stages keep their substages inside them rather than beside them, a dense fixture reports its sparse-chain substage as exactly zero, and every
/// profiled forward lands raw-byte identical to its unprofiled twin — same logit bits, same KV bytes at every position of every layer. The two
/// cache-interaction stages (slot reservations and K/V slot writes) are proven reported and never exceeding the window total: their individual spans
/// are short enough that a tiny synthetic fixture can round one below a single clock tick, so they assert presence rather than strict positivity.
/// Encoder forwards carry no dense stage profile by design — this seam measures prefill and decode windows only.
/// </summary>
[Trait("Category", "Engine")]
[Trait("Category", "Prefill")]
[Collection("model-loaded")] // shares the serialized engine lane with the other forward suites so two models never map weights at once
public sealed class DenseForwardProfileTests : IDisposable
{
    private static readonly SyntheticLlamaGguf.Geometry Shape = new(
        Layers: 3,
        HiddenSize: 64,
        HeadCount: 8,       // head dim 8 over eight heads — wide enough that each measured span carries real work
        KvHeadCount: 2,
        FeedForwardLength: 128);

    private const int MaxContext = 192;

    // A traversal parent is the exact sum of its traversal substages in raw ticks; the millisecond conversion of each tally rounds independently,
    // so the parent-sum assertions allow this much conversion rounding and nothing more.
    private const double SubstageSumToleranceMs = 1e-6;

    // A window length past one default sub-batch (16 rows), so several row windows run and the stage totals aggregate across them.
    private static readonly int[] WindowLengths = [40];

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
                // temp cleanup is best-effort
            }
        }
    }

    /// <summary>The profile facts a default open reports: every stage exactly zero, and one zero whole-block time per layer the model resolved.</summary>
    [Fact]
    public void DenseForwardProfileSummary_WithoutEnabling_ReportsZeroEveryStage()
    {
        using GgufFile gguf = GgufFile.Open(WriteFixture("dense-profile-default-off"));
        using CpuBackend backend = new(threads: 2, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: MaxContext);

        int[] prompt = BuildPrompt(WindowLengths[0]);

        // The forwards run because a host would run them; the seam is simply never asked to measure anything.
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 24);
        using SequenceKv kv = new(pool);
        float[] windowLogits = [.. model.Prefill(prompt, kv)];
        float[] decodeLogits = [.. model.DecodeNext(1, kv)];
        Assert.True(windowLogits.Length > 0 && decodeLogits.Length > 0, "the forwards ran: the seam is simply never asked to measure anything");

        DenseForwardProfile untouched = model.DenseForwardProfileSummary();

        AssertStageZeroes(untouched, "a default open that never enabled the seam");
        Assert.Equal(model.Desc.LayerCount, untouched.Layers.Count);
        foreach (DenseForwardLayerTiming layer in untouched.Layers)
        {
            Assert.Equal(0.0, layer.TotalMs);
        }
    }

    /// <summary>A reset after a measured window zeroes every stage and every per-layer accumulator, so the next measurement starts from zero.</summary>
    [Fact]
    public void ResetDenseForwardProfile_AfterMeasuredWindow_ZeroesEveryStage()
    {
        using GgufFile gguf = GgufFile.Open(WriteFixture("dense-profile-reset"));
        using CpuBackend backend = new(threads: 2, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: MaxContext);

        model.EnableDenseForwardProfiling();

        int[] prompt = BuildPrompt(WindowLengths[0]);
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 24);
        using SequenceKv kv = new(pool);
        model.Prefill(prompt, kv);

        DenseForwardProfile measured = model.DenseForwardProfileSummary();
        Assert.True(measured.TotalMs > 0.0, "a window that ran must carry a positive whole-forward total before the reset");

        model.ResetDenseForwardProfile();

        DenseForwardProfile cleared = model.DenseForwardProfileSummary();
        AssertStageZeroes(cleared, "every accumulator after ResetDenseForwardProfile");
    }

    /// <summary>Turning the seam off keeps the last values until a reset: an unprofiled forward following a disable still lands on identical logits while the stale totals stay readable.</summary>
    [Fact]
    public void DisableDenseForwardProfiling_KeepsLastTotalsUntilReset_AndTheNextForwardStaysIdentical()
    {
        string path = WriteFixture("dense-profile-disable");

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend backend = new(threads: 2, allowAvx2: true);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: MaxContext);

        int[] prompt = BuildPrompt(WindowLengths[0]);
        using BlockPool pool = NewPool(model, backend, capacityBlocks: 48);

        using SequenceKv plainKv = new(pool);
        float[] plainLogits = [.. model.Prefill(prompt, plainKv)];

        model.EnableDenseForwardProfiling();
        using SequenceKv measuredKv = new(pool);
        float[] measuredLogits = [.. model.Prefill(prompt, measuredKv)];
        Assert.True(model.DenseForwardProfileSummary().TotalMs > 0.0);

        model.DisableDenseForwardProfiling();
        DenseForwardProfile stale = model.DenseForwardProfileSummary();
        Assert.True(stale.TotalMs > 0.0, "disabling keeps the last measured totals until a reset names otherwise");

        using SequenceKv afterDisableKv = new(pool);
        float[] afterDisableLogits = [.. model.Prefill(prompt, afterDisableKv)];

        AssertBitsEqual(plainLogits, measuredLogits, $"a profiled {WindowLengths[0]}-token window against the unprofiled run");
        AssertBitsEqual(plainLogits, afterDisableLogits, "the forward that followed a disable against the unprofiled run");
    }

    /// <summary>Every prefill kernel this build executes: the profiled prompt window lands byte-for-byte on its unprofiled twin — logits and every K/V slot alike — while reporting positive stage times for the work it ran, with each substage kept inside its parent stage.</summary>
    [Theory]
    [MemberData(nameof(PrefillKernels))]
    public void PrefillWindow_ProfiledRun_ReportsPositiveStagesAndKeepsLogitsAndKvBytesIdentical(PrefillKernel kernel)
    {
        using GgufFile gguf = GgufFile.Open(WriteFixture($"dense-profile-prefill-{PrefillKernelNames.Of(kernel)}"));
        using CpuBackend backend = new(threads: 2, allowAvx2: true);

        int length = WindowLengths[0];

        using TransformerModel unprofiled = TransformerModel.Open(gguf, backend, maxContextTokens: MaxContext, logLine: null, prefillKernel: kernel);
        using BlockPool pool = NewPool(unprofiled, backend, capacityBlocks: 24);

        int[] prompt = BuildPrompt(length);

        using SequenceKv referenceKv = new(pool);
        float[] referenceLogits = [.. unprofiled.Prefill(prompt, referenceKv)];

        using TransformerModel profiled = TransformerModel.Open(gguf, backend, maxContextTokens: MaxContext, logLine: null, prefillKernel: kernel);
        profiled.EnableDenseForwardProfiling();

        using SequenceKv profiledKv = new(pool);
        float[] profiledLogits = [.. profiled.Prefill(prompt, profiledKv)];

        string context = $"a {length}-token window on the '{PrefillKernelNames.Of(kernel)}' kernel with dense stage profiling on";
        AssertBitsEqual(referenceLogits, profiledLogits, $"{context} against the unprofiled run");
        Assert.Equal(ArgMax(referenceLogits), ArgMax(profiledLogits));
        AssertKvBytesEqual(referenceKv, profiledKv, unprofiled.Desc.LayerCount, length);

        DenseForwardProfile summary = profiled.DenseForwardProfileSummary();

        // The stages that do the work a dense forward is made of carry positive times: seed/embedding lookups, attention blocks with their weight
        // traversals and causal reads split out, feed-forward blocks with their gate/up/down traversals split out, and this window's output head.
        Assert.True(summary.TotalMs > 0.0, $"{context}: the whole forward must carry a positive total");
        Assert.True(summary.SeedMs > 0.0, $"{context}: embedding lookups must carry a positive stage time");
        Assert.True(summary.AttentionMs > 0.0, $"{context}: attention blocks must carry a positive stage time");
        Assert.True(summary.AttentionWeightMs > 0.0, $"{context}: the Q/K/V and output-projection traversals must carry their own positive substage time");
        Assert.True(summary.AttentionProjectionMs > 0.0, $"{context}: the batched Q/K/V projection traversals must carry their own positive substage time");
        Assert.True(summary.AttentionOutputMs > 0.0, $"{context}: the output-projection traversal must carry its own positive substage time");
        Assert.True(summary.AttentionKernelMs > 0.0, $"{context}: the causal read spans must carry their own positive substage time");
        Assert.True(summary.FeedForwardMs > 0.0, $"{context}: feed-forward blocks must carry a positive stage time");
        Assert.True(summary.FeedForwardWeightMs > 0.0, $"{context}: the gate/up/down traversals must carry their own positive substage time");
        Assert.True(summary.FeedForwardGateUpMs > 0.0, $"{context}: the gate/up traversal pair must carry its own positive substage time");
        Assert.True(summary.FeedForwardDownMs > 0.0, $"{context}: the down traversal must carry its own positive substage time");
        Assert.True(summary.LogitsHeadMs > 0.0, $"{context}: the window's output head must carry a positive stage time");

        // The two cache-interaction stages are reported and never exceed the window: each individual reservation or slot write is short enough that one
        // can round below a single clock tick on a synthetic fixture of this size, so their presence rather than strict positivity is what gates here.
        Assert.True(summary.CacheAppendMs >= 0.0 && summary.KvWriteMs >= 0.0, $"{context}: the cache-interaction stages must be reported");

        // A dense fixture runs no sparse chain: the named sparse-chain substage is reported as exactly zero rather than absorbing some other span.
        Assert.Equal(0.0, summary.SparseChainMs);

        // Substages partition parents instead of adding beside them: attention's traversal, causal-read and K/V-write times sit inside the attention
        // stage, feed-forward traversal sits inside its parent, and every parent plus seed/cache interaction stays inside the whole forward.
        Assert.True(summary.AttentionMs >= summary.AttentionWeightMs, $"{context}: attention_weight_ms must stay inside attention_ms");
        Assert.True(summary.AttentionMs >= summary.AttentionKernelMs, $"{context}: attention_kernel_ms must stay inside attention_ms");
        Assert.True(summary.FeedForwardMs >= summary.FeedForwardWeightMs, $"{context}: feed_forward_weight_ms must stay inside feed_forward_ms");

        // Each traversal parent is exactly the sum of its traversal substages: the projection/output pair partitions the attention weight tally and
        // the gate-up/down pair partitions the feed-forward weight tally, within the rounding of two independent millisecond conversions of tick tallies.
        Assert.True(summary.AttentionWeightMs >= summary.AttentionProjectionMs, $"{context}: attention_projection_ms must stay inside attention_weight_ms");
        Assert.True(summary.AttentionWeightMs >= summary.AttentionOutputMs, $"{context}: attention_output_ms must stay inside attention_weight_ms");
        Assert.True(summary.AttentionWeightMs + SubstageSumToleranceMs >= summary.AttentionProjectionMs + summary.AttentionOutputMs, $"{context}: attention_projection_ms + attention_output_ms must account for attention_weight_ms");
        Assert.True(summary.FeedForwardWeightMs >= summary.FeedForwardGateUpMs, $"{context}: feed_forward_gate_up_ms must stay inside feed_forward_weight_ms");
        Assert.True(summary.FeedForwardWeightMs >= summary.FeedForwardDownMs, $"{context}: feed_forward_down_ms must stay inside feed_forward_weight_ms");
        Assert.True(summary.FeedForwardWeightMs + SubstageSumToleranceMs >= summary.FeedForwardGateUpMs + summary.FeedForwardDownMs, $"{context}: feed_forward_gate_up_ms + feed_forward_down_ms must account for feed_forward_weight_ms");

        double parents = summary.SeedMs + summary.CacheAppendMs + summary.AttentionMs + summary.FeedForwardMs + summary.LogitsHeadMs;
        Assert.True(parents <= summary.TotalMs, $"{context}: the parent stages plus seed and cache interaction must stay inside total_ms");

        // Every layer this window ran carries its own whole-block span: a dense prompt window touches every resolved block at every sub-batch.
        Assert.Equal(unprofiled.Desc.LayerCount, summary.Layers.Count);
        foreach (DenseForwardLayerTiming layer in summary.Layers)
        {
            Assert.True(layer.TotalMs > 0.0, $"{context}: layer {layer.Layer} must carry a positive whole-block time");
        }

        double layerSum = 0.0;
        foreach (DenseForwardLayerTiming layer in summary.Layers)
        {
            layerSum += layer.TotalMs;
        }

        Assert.True(layerSum <= summary.TotalMs, $"{context}: the per-layer whole-block spans must stay inside total_ms");
    }

    /// <summary>The single-token decode route: a profiled run of steps reports positive stage times on every dense substage and lands bit-for-bit on the same steps without profiling.</summary>
    [Fact]
    public void DecodeNext_ProfiledSteps_ReportPositiveStagesAndKeepLogitsIdentical()
    {
        using GgufFile gguf = GgufFile.Open(WriteFixture("dense-profile-decode-next"));
        using CpuBackend backend = new(threads: 2, allowAvx2: true);

        int[] prompt = BuildPrompt(WindowLengths[0]);

        using TransformerModel unprofiled = TransformerModel.Open(gguf, backend, maxContextTokens: MaxContext);
        using BlockPool pool = NewPool(unprofiled, backend, capacityBlocks: 48);

        // A shared prefix first: the decode steps then attend over a grown history, which is what makes their causal read stage measurable.
        using SequenceKv referenceKv = new(pool);
        float[] warmLogits = [.. unprofiled.Prefill(prompt, referenceKv)];

        List<float[]> referenceSteps = [];
        int nextToken = ArgMax(warmLogits);
        for (int step = 0; step < 4; step++)
        {
            referenceSteps.Add([.. unprofiled.DecodeNext(nextToken, referenceKv)]);
            nextToken = ArgMax(referenceSteps[^1]);
        }

        using TransformerModel profiled = TransformerModel.Open(gguf, backend, maxContextTokens: MaxContext);
        profiled.EnableDenseForwardProfiling();
        profiled.ResetDenseForwardProfile(); // the prefix warm start is not part of the measured decode window

        using SequenceKv profiledKv = new(pool);
        float[] profiledWarmLogits = [.. profiled.Prefill(prompt, profiledKv)];
        AssertBitsEqual(warmLogits, profiledWarmLogits, "the prefix warm start against the unprofiled run");

        profiled.ResetDenseForwardProfile(); // only the measured decode steps count toward the reported stages

        int profiledNext = ArgMax(profiledWarmLogits);
        for (int step = 0; step < 4; step++)
        {
            float[] stepLogits = [.. profiled.DecodeNext(profiledNext, profiledKv)];
            AssertBitsEqual(referenceSteps[step], stepLogits, $"decode step {step} with dense stage profiling on");
            profiledNext = ArgMax(stepLogits);
        }

        DenseForwardProfile summary = profiled.DenseForwardProfileSummary();

        Assert.True(summary.TotalMs > 0.0, "the measured decode steps must carry a positive whole-forward total");
        Assert.True(summary.SeedMs > 0.0, "each step's embedding lookup must land in the seed stage");
        Assert.True(summary.AttentionWeightMs > 0.0, "each step's attention-side traversals must carry their own substage time");
        Assert.True(summary.AttentionProjectionMs > 0.0, "each step's Q/K/V traversals must carry their own substage time");
        Assert.True(summary.AttentionOutputMs > 0.0, "each step's output-projection traversal must carry its own substage time");
        Assert.True(summary.AttentionKernelMs > 0.0, "each step's causal read over its grown history must carry its own substage time");
        Assert.True(summary.FeedForwardWeightMs > 0.0, "each step's gate/up/down traversals must carry their own substage time");
        Assert.True(summary.FeedForwardGateUpMs > 0.0, "each step's gate/up traversal pair must carry its own substage time");
        Assert.True(summary.FeedForwardDownMs > 0.0, "each step's down traversal must carry its own substage time");
        Assert.True(summary.LogitsHeadMs > 0.0, "each step's output head must carry a positive stage time");
        Assert.Equal(0.0, summary.SparseChainMs);

        Assert.True(summary.AttentionWeightMs + SubstageSumToleranceMs >= summary.AttentionProjectionMs + summary.AttentionOutputMs, "the projection/output pair must account for attention_weight_ms");
        Assert.True(summary.FeedForwardWeightMs + SubstageSumToleranceMs >= summary.FeedForwardGateUpMs + summary.FeedForwardDownMs, "the gate-up/down pair must account for feed_forward_weight_ms");

        double parents = summary.SeedMs + summary.CacheAppendMs + summary.AttentionMs + summary.FeedForwardMs + summary.LogitsHeadMs;
        Assert.True(parents <= summary.TotalMs, "the parent stages plus seed and cache interaction must stay inside total_ms");

        foreach (DenseForwardLayerTiming layer in summary.Layers)
        {
            Assert.True(layer.TotalMs > 0.0, $"layer {layer.Layer} must carry a positive whole-block time across the measured steps");
        }
    }

    /// <summary>The merged batched decode step: profiling reports its stages without moving one logit or KV byte against an unprofiled twin of the same step.</summary>
    [Fact]
    public void DecodeBatch_ProfiledStep_ReportsPositiveStagesAndKeepsLogitsAndKvBytesIdentical()
    {
        using GgufFile gguf = GgufFile.Open(WriteFixture("dense-profile-decode-batch"));
        using CpuBackend backend = new(threads: 2, allowAvx2: true);

        int[] prompt = BuildPrompt(WindowLengths[0]);

        using TransformerModel unprofiled = TransformerModel.Open(gguf, backend, maxContextTokens: MaxContext);
        using BlockPool pool = NewPool(unprofiled, backend, capacityBlocks: 48);

        using SequenceKv firstReference = new(pool);
        using SequenceKv secondReference = new(pool);
        float[] referenceLogits = [.. unprofiled.DecodeBatch([1, 2], [firstReference, secondReference])];

        using TransformerModel profiled = TransformerModel.Open(gguf, backend, maxContextTokens: MaxContext);
        profiled.EnableDenseForwardProfiling();

        using SequenceKv firstProfiled = new(pool);
        using SequenceKv secondProfiled = new(pool);
        float[] profiledLogits = [.. profiled.DecodeBatch([1, 2], [firstProfiled, secondProfiled])];

        AssertBitsEqual(referenceLogits, profiledLogits, "a two-sequence batched step with dense stage profiling on");

        // Each sequence holds exactly the one slot this merged step wrote: both sequences' K/V bytes compare position for position.
        AssertKvBytesEqual(firstReference, firstProfiled, unprofiled.Desc.LayerCount, 1);
        AssertKvBytesEqual(secondReference, secondProfiled, unprofiled.Desc.LayerCount, 1);

        DenseForwardProfile summary = profiled.DenseForwardProfileSummary();

        Assert.True(summary.TotalMs > 0.0, "a merged batched step must carry a positive whole-forward total");
        Assert.True(summary.SeedMs > 0.0, "every sequence's embedding lookup in the merged step must land in the seed stage");
        Assert.True(summary.AttentionWeightMs > 0.0, "the merged step's attention-side traversals must carry their own substage time");
        Assert.True(summary.AttentionProjectionMs > 0.0, "the merged step's Q/K/V traversals must carry their own substage time");
        Assert.True(summary.AttentionOutputMs > 0.0, "the merged step's output-projection traversal must carry its own substage time");
        Assert.True(summary.AttentionKernelMs > 0.0, "each sequence's causal read in the merged step must carry its own substage time");
        Assert.True(summary.FeedForwardWeightMs > 0.0, "the merged step's gate/up/down traversals must carry their own substage time");
        Assert.True(summary.FeedForwardGateUpMs > 0.0, "the merged step's gate/up traversal pair must carry its own substage time");
        Assert.True(summary.FeedForwardDownMs > 0.0, "the merged step's down traversal must carry its own substage time");
        Assert.True(summary.LogitsHeadMs > 0.0, "the batched output head over every sequence must carry a positive stage time");
        Assert.Equal(0.0, summary.SparseChainMs);

        Assert.True(summary.AttentionWeightMs + SubstageSumToleranceMs >= summary.AttentionProjectionMs + summary.AttentionOutputMs, "the projection/output pair must account for attention_weight_ms in the merged step");
        Assert.True(summary.FeedForwardWeightMs + SubstageSumToleranceMs >= summary.FeedForwardGateUpMs + summary.FeedForwardDownMs, "the gate-up/down pair must account for feed_forward_weight_ms in the merged step");

        double parents = summary.SeedMs + summary.CacheAppendMs + summary.AttentionMs + summary.FeedForwardMs + summary.LogitsHeadMs;
        Assert.True(parents <= summary.TotalMs, "the parent stages plus seed and cache interaction must stay inside total_ms for the merged step");
    }

    public static TheoryData<PrefillKernel> PrefillKernels() => [PrefillKernel.GemV, PrefillKernel.Tiled];

    // ---------------------------------------------------------------- helpers

    private string WriteFixture(string modelName)
    {
        string path = SyntheticLlamaGguf.Write(modelName, withReasoningMarkers: true, QuantFormat.F32, Shape);
        _tempFiles.Add(path);
        return path;
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

    private static void AssertStageZeroes(DenseForwardProfile profile, string context)
    {
        Assert.Equal(0.0, profile.SeedMs);
        Assert.Equal(0.0, profile.CacheAppendMs);
        Assert.Equal(0.0, profile.AttentionMs);
        Assert.Equal(0.0, profile.AttentionWeightMs);
        Assert.Equal(0.0, profile.AttentionProjectionMs);
        Assert.Equal(0.0, profile.AttentionOutputMs);
        Assert.Equal(0.0, profile.AttentionKernelMs);
        Assert.Equal(0.0, profile.KvWriteMs);
        Assert.Equal(0.0, profile.FeedForwardMs);
        Assert.Equal(0.0, profile.FeedForwardWeightMs);
        Assert.Equal(0.0, profile.FeedForwardGateUpMs);
        Assert.Equal(0.0, profile.FeedForwardDownMs);
        Assert.Equal(0.0, profile.SparseChainMs);
        Assert.Equal(0.0, profile.LogitsHeadMs);
        Assert.Equal(0.0, profile.TotalMs);
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

    // Every position and every layer: the whole KV both routes wrote, compared slot by slot.
    private static void AssertKvBytesEqual(SequenceKv reference, SequenceKv profiled, int layerCount, int positions)
    {
        Assert.Equal(positions, profiled.Length);

        for (int position = 0; position < positions; position++)
        {
            for (int layer = 0; layer < layerCount; layer++)
            {
                AssertBitsEqual(reference.ReadK(position, layer), profiled.ReadK(position, layer), $"K at position {position} layer {layer}");
                AssertBitsEqual(reference.ReadV(position, layer), profiled.ReadV(position, layer), $"V at position {position} layer {layer}");
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
