using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Logit-agreement parity against the pinned llama.cpp server reference recorded in tests/fixtures. Per prompt:
/// the model file must match its recorded hash; LAIR's argmax token must appear among the reference top-5 tokens
/// (pinned llama.cpp CLI and server modes disagree on identical greedy runs, so cross-tool exact continuations are
/// unattainable — membership of our pick in the reference leaders is the honest gate); every reference-top-5
/// token's calibrated LAIR logprob from fixture generation must reproduce within tolerance, including documented
/// absences; and each shared candidate's live LAIR logprob must stay inside a cross-tool band around the pinned
/// llama.cpp value — tight on F16, where the two pipelines agree to a few thousandths of a nat, looser on the
/// quantized files where weight noise legitimately moves rankings. A forward-math regression that shifts the whole
/// distribution by ~1 nat therefore fails here even when no single candidate breaks a cross-tool tie; continuations
/// themselves are pinned in ContinuationParityTests for F16. The calibration data is produced by the CalibrationProbe
/// test during scripts/generate-parity.ps1 regeneration.
/// </summary>
[Trait("Category", "Parity")]
[Collection("model-loaded")] // shares one serialized lane with the latency-budget suite so a large fixture loading alongside it cannot starve that measurement
public sealed class ParityTests
{
    private const string F16Fixture = "tests/fixtures/parity-smollm-f16.json";

    // The Q8_0 fixture is both a reference set and a real quantized load: the model under test IS that file.
    private const string Q8Fixture = "tests/fixtures/parity-smollm-q8.json";

    // Same for the Q4_K_M fixture. On this file q4-style weight noise flips a near tie in the first-step ranking,
    // so LAIR's argmax and the pinned reference's top pick differ while remaining mutual top-5 members; both sides'
    // calibrated logprobs reproduce exactly, which is what these gates actually verify.
    private const string Q4KMFixture = "tests/fixtures/parity-smollm-q4km.json";

    // The Qwen3 Q8_0 fixture: a real qwen3 load (per-head QK-norm by profile default, 16 x 128 attention over a
    // 1024-hidden model, tied output head) against the same pinned reference and the same quantized-file band.
    private const string Qwen3Fixture = "tests/fixtures/parity-qwen3.json";

    // The Llama-3.2 Q3_K_S fixture: a real llama load (32 x 8 GQA at head_dim 64 over a 2048-hidden, 16-layer model)
    // against the same pinned reference and the same quantized-file band.
    private const string Llama32Q3KFixture = "tests/fixtures/parity-llama32-q3ks.json";

    // The Llama-3.2 Q2_K fixture: the same architecture at 2-bit weights, and the suite's closest case to the
    // cross-tool band — on the chat-header prompt the reference's '<' candidate sits 0.86 nats from LAIR's value.
    private const string Llama32Q2KFixture = "tests/fixtures/parity-llama32-q2k.json";

    // The YaRN fixture: the pinned Qwen3 Q8_0 file with qwen3.rope.scaling.type = yarn, .factor = 2 and
    // .original_context_length = 512 declared on a copy produced by scripts/patch-gguf-rope-scaling.ps1, so both
    // engines read the same declared keys and the scaled rotary tables and attention factor run inside the band.
    private const string Qwen3YaRNFixture = "tests/fixtures/parity-qwen3-yarn.json";

    // The Llama-3.2 IQ4_XS fixture: the same 16-layer llama architecture with every attention-value and MLP weight
    // at IQ4_XS (96 IQ4_XS tensors beside 16 Q5_K attention-value tensors and a Q6_K token-embedding tensor) against
    // the same pinned reference and the same quantized-file band.
    private const string Llama32Iq4XsFixture = "tests/fixtures/parity-llama32-iq4xs.json";

    // The Llama-3.2 UD IQ3_XXS fixture: a dynamic quant whose per-layer tiers vary inside one file — IQ3_XXS,
    // IQ4_XS, IQ3_S, IQ2_XS and IQ2_S tensors side by side over the same architecture — against the same pinned
    // reference and the same quantized-file band.
    private const string Llama32UdIq3XxsFixture = "tests/fixtures/parity-llama32-udiq3xxs.json";

    // The Gemma-2 Q4_K_M fixture: a real gemma2 load (SentencePiece vocabulary resolved from the file's own
    // tokenizer metadata, and its alternating local/full attention blocks applied by the profile) against the same
    // pinned reference and the same quantized-file band. The chat-header prompt uses ChatML markers this model does
    // not use in its own turn template; both engines read them as ordinary text, which is what the recorded outputs
    // show — each side still lands on a one-sentence answer about Mars.
    private const string Gemma2Q4KFixture = "tests/fixtures/parity-gemma2.json";

    // The Phi-3-mini Q2_K fixture: a real phi3 load (one fused qkv buffer per layer read as three head-region slices,
    // the gate half of the gated pair living inside blk.N.ffn_up.weight with no separate gate tensor at all, 96-wide
    // heads rotated in rotate-half order, and an untied Q6_K output head) against the same pinned reference and the
    // same quantized-file band.
    private const string Phi3Q2KFixture = "tests/fixtures/parity-phi3.json";

    /// <summary>Ranking window whose reproduction the fixtures calibrate.</summary>
    private const int CalibrationWindow = 16;

    /// <summary>Absolute logprob tolerance for reproducing a calibrated LAIR distribution entry.</summary>
    private static readonly double ReproductionTolerance = 0.2;

    // Cross-tool band against the pinned llama.cpp value itself. F16 agrees with the reference to a few thousandths
    // of a nat, so anything looser than this hides forward-math drift; the quantized files carry real weight noise
    // and are calibrated near their tie-flips, so they get a wider band that still catches a ~1 nat shift.
    private static double CrossToolTolerance(string fixturePath) =>
        fixturePath == F16Fixture ? 0.05 : 1.0;

    // The band applies to candidates carrying sampling weight: within this many nats of the reference's own top-1,
    // a candidate's softmax weight is at least ~0.25% of the leader's, so its absolute logprob is a decision-relevant
    // quantity and both pipelines must agree on it. Deeper candidates are compared only on membership in LAIR's
    // calibrated window and on reproduction of LAIR's own calibrated value, because their absolute cross-entropy is
    // dominated by tail shape, where sub-nat logit differences reorder the tail wholesale.
    private const double CrossToolWindowNats = 6.0;

    public static TheoryData<string, int> PromptCases()
    {
        TheoryData<string, int> cases = [];
        foreach (string fixture in new[] { F16Fixture, Q8Fixture, Q4KMFixture, Qwen3Fixture, Llama32Q3KFixture, Llama32Q2KFixture, Qwen3YaRNFixture, Llama32Iq4XsFixture, Llama32UdIq3XxsFixture, Gemma2Q4KFixture, Phi3Q2KFixture })
        {
            string? path = ModelFixture.ResolveRepoRelative(fixture);
            if (path is null)
            {
                continue; // fixtures travel with the repo: absent means nothing to enumerate
            }

            foreach (int index in Enumerable.Range(0, LoadFixture(path).Prompts.Length))
            {
                cases.Add(fixture, index);
            }
        }

        return cases;
    }

    [Theory]
    [MemberData(nameof(PromptCases))]
    public void NextTokenDistribution_MatchesPinnedReferenceAndCalibration(string fixturePath, int promptIndex)
    {
        string? fixtureFile = ModelFixture.ResolveRepoRelative(fixturePath);
        if (fixtureFile is null)
        {
            return; // no recorded outputs to compare against
        }

        ParityFixture fixture = LoadFixture(fixtureFile);
        ParityPrompt entry = fixture.Prompts[promptIndex];

        string? modelPath = ModelFixture.ResolveRepoRelative(fixture.ModelFile);
        if (modelPath is null)
        {
            return; // real-model cases skip when the GGUF file is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        AssertRecordedHash(fixture, modelPath);

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

        List<(int Id, double Logprob)> top = TopWithLogprobs(model.LastLogits(), CalibrationWindow);

        Assert.True(
            entry.TopLogprobs.Any(reference => reference.Id == top[0].Id),
            $"prompt '{entry.Prompt.Trim()}' argmax {top[0].Id}={top[0].Logprob:F4} is not among the pinned " +
            $"reference top-5 [{string.Join(", ", entry.TopLogprobs.Select(t => $"{t.Id}={t.Logprob:F4}"))}] — " +
            $"the engine's next-token decision moved away from the reference.\nlair:      [{Format(top)}]");

        foreach (ParityToken reference in entry.TopLogprobs)
        {
            int slot = top.FindIndex(t => t.Id == reference.Id);
            if (reference.LairLogprob is null)
            {
                Assert.True(
                    slot < 0,
                    $"prompt '{entry.Prompt.Trim()}' token {reference.Id}('{reference.Token}') was calibrated as " +
                    $"absent from LAIR's top-{CalibrationWindow} but now ranks at position {slot + 1} with logprob " +
                    $"{(slot >= 0 ? top[slot].Logprob : double.NaN):F4}.\nlair: [{Format(top)}]");
                continue;
            }

            Assert.True(
                slot >= 0,
                $"prompt '{entry.Prompt.Trim()}' token {reference.Id}('{reference.Token}') was calibrated into " +
                $"LAIR's top-{CalibrationWindow} at logprob {reference.LairLogprob:F4} but is now absent.\n" +
                $"lair: [{Format(top)}]");

            double delta = Math.Abs(top[slot].Logprob - reference.LairLogprob.Value);
            Assert.True(
                delta <= ReproductionTolerance,
                $"prompt '{entry.Prompt.Trim()}' token {reference.Id}('{reference.Token}') logprob moved " +
                $"{delta:F4} from its calibrated value {reference.LairLogprob:F4} to {top[slot].Logprob:F4} — " +
                $"beyond the {ReproductionTolerance:F2} reproduction tolerance.\nlair: [{Format(top)}]");

            double crossToolDelta = Math.Abs(top[slot].Logprob - reference.Logprob);
            double belowLeader = reference.Logprob - entry.TopLogprobs[0].Logprob;
            Assert.True(
                belowLeader < -CrossToolWindowNats || crossToolDelta <= CrossToolTolerance(fixturePath),
                $"prompt '{entry.Prompt.Trim()}' token {reference.Id}('{reference.Token}') sits {crossToolDelta:F4} nats " +
                $"from the pinned llama.cpp logprob {reference.Logprob:F4} (LAIR now {top[slot].Logprob:F4}), beyond the " +
                $"{CrossToolTolerance(fixturePath):F2} cross-tool band, and it is only {-belowLeader:F4} nats below the " +
                $"reference's top-1 — inside the {-CrossToolWindowNats:F1}-nat window where a candidate carries " +
                $"sampling weight.\nlair: [{Format(top)}]");
        }
    }

    private static List<(int Id, double Logprob)> TopWithLogprobs(ReadOnlySpan<float> logits, int k)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > max)
            {
                max = logits[i];
            }
        }

        double sumExp = 0.0;
        for (int i = 0; i < logits.Length; i++)
        {
            sumExp += Math.Exp(logits[i] - max);
        }

        double logSumExp = max + Math.Log(sumExp);

        List<(int Id, double Logprob)> top = [];
        for (int id = 0; id < logits.Length; id++)
        {
            double logprob = logits[id] - logSumExp;
            if (top.Count < k || logprob > top[^1].Logprob)
            {
                int at = top.BinarySearch((id, logprob), Comparer<(int Id, double Logprob)>.Create(
                    (a, b) => a.Logprob != b.Logprob ? b.Logprob.CompareTo(a.Logprob) : a.Id.CompareTo(b.Id)));
                if (at < 0)
                {
                    at = ~at;
                }

                top.Insert(at, (id, logprob));
                if (top.Count > k)
                {
                    top.RemoveAt(k);
                }
            }
        }

        return top;
    }

    private static string Format(List<(int Id, double Logprob)> top) =>
        string.Join(", ", top.Select(t => $"{t.Id}={t.Logprob:F4}"));

    private static void AssertRecordedHash(ParityFixture fixture, string modelPath)
    {
        using FileStream stream = File.OpenRead(modelPath);
        byte[] digest = SHA256.HashData(stream);
        string actual = Convert.ToHexString(digest).ToLowerInvariant();

        Assert.True(
            string.Equals(actual, fixture.ModelSha256, StringComparison.Ordinal),
            $"model file '{fixture.ModelFile}' hashes to {actual}, not the recorded {fixture.ModelSha256}: " +
            "the reference outputs were generated against a different model binary.");
    }

    private static ParityFixture LoadFixture(string path) =>
        JsonSerializer.Deserialize<ParityFixture>(File.ReadAllText(path), JsonOptions)
            ?? throw new InvalidOperationException($"fixture '{path}' deserialized to null");

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    private sealed record ParityFixture(
        [property: JsonPropertyName("model_file")] string ModelFile,
        [property: JsonPropertyName("model_sha256")] string ModelSha256,
        [property: JsonPropertyName("prompts")] ParityPrompt[] Prompts);

    private sealed record ParityPrompt(
        [property: JsonPropertyName("prompt")] string Prompt,
        [property: JsonPropertyName("max_tokens")] int MaxTokens,
        [property: JsonPropertyName("output")] string Output,
        [property: JsonPropertyName("top_logprobs")] ParityToken[] TopLogprobs);

    private sealed record ParityToken(
        [property: JsonPropertyName("id")] int Id,
        [property: JsonPropertyName("token")] string Token,
        [property: JsonPropertyName("logprob")] double Logprob,
        [property: JsonPropertyName("lair_logprob")] double? LairLogprob);
}
