using System.Diagnostics;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// End-to-end coverage of the real SmolLM-135M-Instruct Q4_K_M fixture (skip-if-absent): the loader accepts the
/// mixed quant family the file actually ships (Q4_K/Q5_0/Q6_K/Q8_0/F32, asserted as an exact inventory), the
/// engine opens it on the tied lm_head path over the Q8_0 token_embd, and a greedy 8-token decode keeps every
/// logit finite and yields a bit-identical argmax sequence between 1-thread and 4-thread engine runs.
/// </summary>
[Trait("Category", "Kernel")]
public sealed class QuantFamilyFixtureTests
{
    private const string FixtureRelativePath = "models/smollm-135m-instruct-q4_k_m.gguf";

    private const int DecodeTokens = 8;

    // ---------------------------------------------------------------- loader acceptance of the mixed quant family

    [Fact]
    public void GgufModelLoader_RealQ4KMFixture_AcceptsMixedQuantFamily()
    {
        string? modelPath = ResolveModelPath();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        LogicalTensorTable table = new GgufModelLoader().Load(modelPath);

        // Exact inventory of the file as shipped (llama.cpp Q4_K_M rules for this 135M model): attention and FFN
        // projections are Q5_0, but attn_v and ffn_down split between a base format and an upgraded one per layer,
        // token_embd is Q8_0, and every norm vector stays F32. The file ships tied weights: there is no output.weight.
        Dictionary<string, Dictionary<QuantFormat, int>> expected = new(StringComparer.Ordinal)
        {
            ["attn_q.weight"] = new() { [QuantFormat.Q5_0] = 30 },
            ["attn_k.weight"] = new() { [QuantFormat.Q5_0] = 30 },
            ["attn_v.weight"] = new() { [QuantFormat.Q5_0] = 16, [QuantFormat.Q8_0] = 14 },
            ["attn_output.weight"] = new() { [QuantFormat.Q5_0] = 30 },
            ["attn_norm.weight"] = new() { [QuantFormat.F32] = 30 },
            ["ffn_gate.weight"] = new() { [QuantFormat.Q5_0] = 30 },
            ["ffn_up.weight"] = new() { [QuantFormat.Q5_0] = 30 },
            ["ffn_down.weight"] = new() { [QuantFormat.Q4_K] = 16, [QuantFormat.Q6_K] = 14 },
            ["ffn_norm.weight"] = new() { [QuantFormat.F32] = 30 },
            ["token_embd.weight"] = new() { [QuantFormat.Q8_0] = 1 },
            ["output_norm.weight"] = new() { [QuantFormat.F32] = 1 },
        };

        Dictionary<string, Dictionary<QuantFormat, int>> observed = new(StringComparer.Ordinal);
        foreach (LogicalTensor tensor in table.Tensors)
        {
            // "blk.N.<name>.weight" groups under "<name>.weight"; global tensors keep their own name.
            string suffix = tensor.Name.StartsWith("blk.", StringComparison.Ordinal)
                ? tensor.Name[(tensor.Name.IndexOf('.', 4) + 1)..]
                : tensor.Name;

            if (!observed.TryGetValue(suffix, out Dictionary<QuantFormat, int>? counts))
            {
                counts = new Dictionary<QuantFormat, int>();
                observed[suffix] = counts;
            }

            counts[tensor.Format] = counts.GetValueOrDefault(tensor.Format) + 1;
        }

        // Canonical sorted (suffix, format, count) triples: dictionary insertion order varies with file order.
        List<(string Suffix, QuantFormat Format, int Count)> observedTriples = observed
            .SelectMany(g => g.Value.Select(f => (g.Key, f.Key, f.Value)))
            .OrderBy(t => t.Item1, StringComparer.Ordinal)
            .ThenBy(t => t.Item2)
            .ToList();
        List<(string Suffix, QuantFormat Format, int Count)> expectedTriples = expected
            .SelectMany(g => g.Value.Select(f => (g.Key, f.Key, f.Value)))
            .OrderBy(t => t.Item1, StringComparer.Ordinal)
            .ThenBy(t => t.Item2)
            .ToList();

        Assert.Equal(expectedTriples, observedTriples);
        Assert.Equal(272, table.Tensors.Count);

        // Spot-checks the quant-family claim rests on: attn_k is Q5_0, attn_v appears in both Q5_0 (base layers,
        // e.g. blk.3) and Q8_0 (upgraded layers, e.g. blk.0), ffn_down carries the Q6_K upgrades, and the tied
        // table has no output.weight tensor.
        Assert.Equal(QuantFormat.Q5_0, table.Find("blk.0.attn_k.weight")!.Format);
        Assert.Equal(QuantFormat.Q8_0, table.Find("blk.0.attn_v.weight")!.Format);
        Assert.Equal(QuantFormat.Q5_0, table.Find("blk.3.attn_v.weight")!.Format);
        Assert.Equal(QuantFormat.Q6_K, table.Find("blk.0.ffn_down.weight")!.Format);
        Assert.Equal(QuantFormat.Q4_K, table.Find("blk.3.ffn_down.weight")!.Format);
        Assert.Equal(QuantFormat.Q8_0, table.Find("token_embd.weight")!.Format);
        Assert.Null(table.Find("output.weight"));

        using GgufFile gguf = GgufFile.Open(modelPath);
        ModelDesc desc = ModelDesc.FromGguf(gguf);
        Assert.Equal("llama", desc.Architecture);
        Assert.False(desc.HasOutputWeights); // tied lm_head over token_embd

        Console.WriteLine(
            $"Q4_K_M fixture inventory ({table.Tensors.Count} tensors): " +
            string.Join(", ", expected.SelectMany(g => g.Value.Select(f => $"{f.Key} x{f.Value} ({g.Key})"))));
    }

    // ---------------------------------------------------------------- greedy decode, thread-invariant argmax

    [Fact]
    public void Q4KMFixture_GreedyDecode_ArgmaxSequenceBitIdenticalAcrossThreadCounts()
    {
        string? modelPath = ResolveModelPath();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        IReadOnlyList<int> promptIds = tokenizer.Encode("The capital of France is");
        Assert.NotEmpty(promptIds);
        Assert.True(promptIds.Count <= 20, $"prompt must stay short to bound runtime, got {promptIds.Count} tokens");

        (int[] Ids, float[] FirstStepLogits, double ElapsedMs) RunOnce(int threads)
        {
            Stopwatch sw = Stopwatch.StartNew();

            using CpuBackend backend = new(threads: threads);
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 64);

            Assert.Equal("llama", model.Desc.Architecture);
            Assert.False(model.Desc.HasOutputWeights); // tied lm_head path over the Q8_0 token_embd rows

            using BlockPool pool = new(blockTokens: 16, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks: 4, backend);
            using SequenceKv kv = new(pool);

            model.Prefill(promptIds, kv);

            // Prefill discards per-token logits by contract, but LastLogits is already the next-token
            // distribution for the final prompt token: greedy sampling of it seeds the decode without re-forwarding.
            int[] ids = new int[DecodeTokens];
            float[] firstStepLogits = [];

            for (int step = 0; step < DecodeTokens; step++)
            {
                Span<float> logits = step == 0 ? model.LastLogits() : model.DecodeNext(ids[step - 1], kv);

                foreach (float v in logits)
                {
                    Assert.True(float.IsFinite(v), $"threads={threads} step {step}: logit {v} is not finite");
                }

                ids[step] = ArgMax(logits);
                if (step == 0)
                {
                    firstStepLogits = logits.ToArray(); // model-owned buffer: snapshot before the next forward
                }
            }

            Assert.Equal(promptIds.Count + DecodeTokens - 1, kv.Length); // the final generated id is sampled, not forwarded

            sw.Stop();
            return (ids, firstStepLogits, sw.Elapsed.TotalMilliseconds);
        }

        var run1 = RunOnce(threads: 1);
        var run4 = RunOnce(threads: 4);

        Assert.True(
            run1.FirstStepLogits.AsSpan().SequenceEqual(run4.FirstStepLogits.AsSpan()),
            "first-step logits must be bit-identical across thread counts");
        Assert.True(
            run1.Ids.AsSpan().SequenceEqual(run4.Ids.AsSpan()),
            $"greedy argmax sequence must be bit-identical across thread counts: 1T [{string.Join(", ", run1.Ids)}] vs 4T [{string.Join(", ", run4.Ids)}]");

        Console.WriteLine(
            $"Q4_K_M greedy decode: prompt {promptIds.Count} tokens, {DecodeTokens} tokens decoded in {run1.ElapsedMs:F0} ms (1T) / {run4.ElapsedMs:F0} ms (4T); " +
            $"argmax ids [{string.Join(", ", run1.Ids)}]; text: {tokenizer.Decode(run1.Ids)}");
    }

    // ---------------------------------------------------------------- shared helpers

    /// <summary>Greedy pick: strict "greater than" scan so equal logits keep the lower id (Sampler's greedy tie rule).</summary>
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

    // The test host's working directory varies per runner: walk up from the assembly location to find the repo fixture.
    private static string? ResolveModelPath() => ModelFixture.ResolveRepoRelative(FixtureRelativePath);
}
