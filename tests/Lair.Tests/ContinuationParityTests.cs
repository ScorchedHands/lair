using System.Text.Json;
using System.Text.Json.Serialization;

using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Cross-tool continuation parity: LAIR's greedy multi-step continuation of every pinned F16 prompt must reproduce
/// the recorded llama.cpp b12032 continuation, not merely its first next-token distribution. First-step gates can
/// pass while per-position logits drift, and that drift snowballs into degenerate output as context grows; this
/// gate is what catches forward-math regressions (RoPE pairing, attention accumulation) at the length where they
/// actually bite. The quantized fixtures are excluded on purpose: their documented near-tie flips move the argmax
/// within a few tokens of the start, so exact continuations there are not attainable — ParityTests' cross-tool
/// logit tolerance is what covers those files instead.
/// </summary>
[Trait("Category", "Parity")]
public sealed class ContinuationParityTests
{
    private static readonly string[] Fixtures = ["tests/fixtures/parity-smollm-f16.json"];

    public static TheoryData<string, int> PromptCases()
    {
        TheoryData<string, int> cases = [];
        foreach (string fixture in Fixtures)
        {
            string? path = ModelFixture.ResolveRepoRelative(fixture);
            if (path is null)
            {
                continue; // fixtures travel with the repo: absent means nothing to enumerate
            }

            foreach (int index in Enumerable.Range(0, Load(path).Prompts.Length))
            {
                cases.Add(fixture, index);
            }
        }

        return cases;
    }

    [Theory]
    [MemberData(nameof(PromptCases))]
    public void GreedyContinuation_MatchesPinnedReference(string fixturePath, int promptIndex)
    {
        string? fixtureFile = ModelFixture.ResolveRepoRelative(fixturePath);
        if (fixtureFile is null)
        {
            return; // no recorded continuation to compare against
        }

        ParityFixture fixture = Load(fixtureFile);
        ParityPrompt entry = fixture.Prompts[promptIndex];

        string? modelPath = ModelFixture.ResolveRepoRelative(fixture.ModelFile);
        if (modelPath is null)
        {
            return; // real-model cases skip when the GGUF file is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile file = GgufFile.Open(modelPath);
        using CpuBackend backend = new();
        using TransformerModel model = TransformerModel.Open(file, backend, 1024);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(file);

        IReadOnlyList<int> promptTokens = tokenizer.AddBosToken
            ? tokenizer.EncodeWithBos(entry.Prompt)
            : tokenizer.Encode(entry.Prompt);

        using BlockPool pool = new(16, model.Desc.LayerCount, model.Desc.KvHeadCount, model.Desc.HeadDim, 64, backend);
        using SequenceKv kv = new(pool);
        model.Prefill(promptTokens, kv);

        List<int> generated = [];
        Span<float> logits = model.LastLogits();
        for (int step = 0; step < entry.MaxTokens; step++)
        {
            int tokenId = Argmax(logits);
            if (tokenId == tokenizer.EosId)
            {
                break;
            }

            generated.Add(tokenId);
            logits = model.DecodeNext(tokenId, kv);
        }

        string lair = tokenizer.Decode(generated, includeSpecial: false).TrimEnd('\r', '\n');
        string reference = entry.Output.TrimEnd('\r', '\n');

        Assert.True(
            lair == reference,
            $"prompt '{entry.Prompt.Trim()}' continuation diverged from the pinned {fixture.LlamaCppTag} " +
            $"reference after {generated.Count} greedy token(s).\n" +
            $"lair:      {Describe(lair)}\n" +
            $"reference: {Describe(reference)}\n" +
            $"first difference at character {FirstDifference(lair, reference)}.");

        Assert.True(
            generated.Count >= 8,
            $"prompt '{entry.Prompt.Trim()}' stopped after only {generated.Count} token(s); the pinned reference " +
            $"ran further, so this case no longer exercises multi-block context.");
    }

    private static int Argmax(Span<float> logits)
    {
        int best = 0;
        for (int id = 1; id < logits.Length; id++)
        {
            if (logits[id] > logits[best])
            {
                best = id;
            }
        }

        return best;
    }

    private static string Describe(string text) =>
        "\"" + text.Replace("\n", "\\n") + "\"";

    private static int FirstDifference(string a, string b)
    {
        int limit = Math.Min(a.Length, b.Length);
        for (int i = 0; i < limit; i++)
        {
            if (a[i] != b[i])
            {
                return i;
            }
        }

        return limit;
    }

    private static ParityFixture Load(string fixturePath) =>
        JsonSerializer.Deserialize<ParityFixture>(File.ReadAllText(fixturePath), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
        })!;

    private sealed record ParityFixture(
        [property: JsonPropertyName("model_file")] string ModelFile,
        [property: JsonPropertyName("llama_cpp_tag")] string LlamaCppTag,
        [property: JsonPropertyName("prompts")] ParityPrompt[] Prompts);

    private sealed record ParityPrompt(
        [property: JsonPropertyName("prompt")] string Prompt,
        [property: JsonPropertyName("max_tokens")] int MaxTokens,
        [property: JsonPropertyName("output")] string Output);
}
