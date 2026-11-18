using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Logit-agreement parity for LAIR's quantized KV serving route against the pinned llama.cpp b12032 reference recorded
/// with matching cache types (-ctk/-ctv) in tests/fixtures. Each fixture names its kv_quant pair, so this suite opens the
/// exact production seam serving uses — TransformerModel's quantized-pair overload over the file's resolved single-span
/// route and the integer-dot encoding this backend resolved, with the typed block pool that pair's group layout sizes —
/// rather than the FP32 posture. The gates mirror ParityTests: LAIR's argmax must appear among the reference top-5; every
/// calibrated lair_logprob reproduces within the reproduction tolerance, including documented absences; and each candidate
/// inside the sampling window (within 6 nats of the reference top-1) stays inside a quantized-KV cross-tool band around
/// the pinned llama.cpp value. That band is per mode: a fixture may state it as 'quantized_kv_band' after measurement, and
/// the source constants below are initial placeholders pending main's generation run — they are deliberately separate from
/// the dense-family bands in ParityTests because here the perturbation comes from KV storage quantization over an F16 file,
/// not weight noise. Fixture calibration data is produced by the CalibrationProbe test under scripts/generate-kv-cache-type-parity.ps1.
/// </summary>
[Trait("Category", "Parity")]
[Collection("model-loaded")] // shares one serialized lane with the other real-model suites so a quantized engine opening alongside them cannot starve that measurement
public sealed class KvQuantizedParityTests
{
    private const string Q8Q8Fixture = "tests/fixtures/parity-kv-q8q8-smollm-f16.json";
    private const string K8V4Fixture = "tests/fixtures/parity-kv-k8v4-smollm-f16.json";

    /// <summary>Ranking window whose reproduction the fixtures calibrate.</summary>
    private const int CalibrationWindow = 16;

    /// <summary>Absolute logprob tolerance for reproducing a calibrated LAIR distribution entry, same as the FP32 parity suite.</summary>
    private static readonly double ReproductionTolerance = 0.2;

    // Initial quantized-KV cross-tool bands per mode, pending main's measured values: q8_0 attention storage is faithful on
    // both sides and needs only a modest widening over the F16 agreement, while a q4_0 V side moves every attended value by
    // its group scale. A fixture that states 'quantized_kv_band' overrides these after generation; main reviews the measured
    // drift then tightens what remains here. These bands gate nothing until their fixtures exist — absent fixtures skip.
    private static double SourceQuantizedKvBand(KvSideFormat k, KvSideFormat v) => (k, v) switch
    {
        (KvSideFormat.Q8_0, KvSideFormat.Q8_0) => 0.25, // measured q8_0/q8_0 band on SmolLM-135M-F16 with margin over observed max relevant delta 0.1347
        (KvSideFormat.Q8_0, KvSideFormat.Q4_0) => 0.50, // measured k8v4 band on SmolLM-135M-F16 with margin over observed max relevant delta 0.3856
        _ => throw new InvalidOperationException($"no quantized-KV band constant for k='{Of(k)}' v='{Of(v)}'"),
    };

    // The band applies to candidates carrying sampling weight: within this many nats of the reference's own top-1, a
    // candidate's softmax weight is decision-relevant and both pipelines must agree on it — same window as ParityTests.
    private const double CrossToolWindowNats = 6.0;

    public static TheoryData<string, int> PromptCases()
    {
        TheoryData<string, int> cases = [];
        foreach (string fixture in new[] { Q8Q8Fixture, K8V4Fixture })
        {
            string? path = ModelFixture.ResolveRepoRelative(fixture);
            ParityPrompt[] prompts = path is null ? [] : LoadFixture(path).Prompts ?? [];
            if (prompts.Length == 0)
            {
                // An absent or empty fixture still needs one data row — xunit 2.9 fails an empty TheoryData outright rather
                // than skipping it — and the test body returns as soon as there is nothing recorded, so nothing is asserted.
                cases.Add(fixture, 0);
                continue;
            }

            foreach (int index in Enumerable.Range(0, prompts.Length))
            {
                cases.Add(fixture, index);
            }
        }

        return cases;
    }

    [Theory]
    [MemberData(nameof(PromptCases))]
    public void NextTokenDistribution_MatchesPinnedQuantizedReferenceAndCalibration(string fixturePath, int promptIndex)
    {
        string? fixtureFile = ModelFixture.ResolveRepoRelative(fixturePath);
        if (fixtureFile is null)
        {
            return; // no recorded outputs to compare against
        }

        ParityFixture fixture = LoadFixture(fixtureFile);

        // A present-but-empty prompts array carries no recorded outputs to compare; the sentinel data row exists only so
        // xunit has something to enumerate, and this guard lets it pass rather than index out of range.
        if (fixture.Prompts is null || fixture.Prompts.Length == 0)
        {
            return;
        }

        ParityPrompt entry = fixture.Prompts[promptIndex];

        KvFormatPair pair = ResolveDeclaredPair(fixture, fixturePath);

        string? modelPath = ModelFixture.ResolveRepoRelative(fixture.ModelFile);
        if (modelPath is null)
        {
            return; // real-model cases skip when the GGUF file is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        AssertRecordedHash(fixture, modelPath);

        using GgufFile file = GgufFile.Open(modelPath);
        using CpuBackend backend = new();
        using TransformerModel model = TransformerModel.Open(
            file,
            backend,
            1024, // the decode settings recorded in these fixtures run at this context.
            logLine: null,
            PrefillKernel.GemV, // the batched-row window posture the quantized attention route carries.
            ModelPlans.Resolve(file).Route,
            pair,
            backend.ResolvedIntegerDotPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(file);

        IReadOnlyList<int> promptTokens = tokenizer.AddBosToken
            ? tokenizer.EncodeWithBos(entry.Prompt)
            : tokenizer.Encode(entry.Prompt);

        List<(int Id, double Logprob)> top = pair.V == KvSideFormat.Q8_0
            ? QuantizedTop16<Q8Block, Q8Block>(model, backend, pair, promptTokens)
            : QuantizedTop16<Q8Block, Q4Block>(model, backend, pair, promptTokens);

        double band = fixture.QuantizedKvBand ?? SourceQuantizedKvBand(pair.K, pair.V);
        string modeLabel = $"{Of(pair.K)}/{Of(pair.V)}";

        Assert.True(
            entry.TopLogprobs.Any(reference => reference.Id == top[0].Id),
            $"prompt '{entry.Prompt.Trim()}' argmax {top[0].Id}={top[0].Logprob:F4} is not among the pinned " +
            $"reference top-5 [{string.Join(", ", entry.TopLogprobs.Select(t => $"{t.Id}={t.Logprob:F4}"))}] — " +
            $"the {modeLabel} quantized route's next-token decision moved away from the reference.\nlair:      [{Format(top)}]");

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
                belowLeader < -CrossToolWindowNats || crossToolDelta <= band,
                $"prompt '{entry.Prompt.Trim()}' token {reference.Id}('{reference.Token}') sits {crossToolDelta:F4} nats " +
                $"from the pinned llama.cpp logprob {reference.Logprob:F4} (LAIR now {top[slot].Logprob:F4}), beyond the " +
                $"{band:F2}-nat quantized-KV band for {modeLabel}, and it is only {-belowLeader:F4} nats below the " +
                $"reference's top-1 — inside the {-CrossToolWindowNats:F1}-nat window where a candidate carries " +
                $"sampling weight.\nlair: [{Format(top)}]");
        }
    }

    /// <summary>
    /// The kv_quant pair this fixture declares, validated against the pairs this harness serves. A present fixture naming
    /// an unsupported pair is a failure, not a skip: the recorded reference outputs are meaningless for any posture LAIR
    /// cannot open.
    /// </summary>
    private static KvFormatPair ResolveDeclaredPair(ParityFixture fixture, string fixturePath)
    {
        const string SupportedPairs = "'q8_0/q8_0' and 'q8_0/q4_0'";

        if (fixture.KvQuant is null || fixture.KvQuant.K is null || fixture.KvQuant.V is null)
        {
            throw new InvalidOperationException(
                $"fixture '{fixturePath}' carries no kv_quant pair: quantized KV parity fixtures must name the k and v " +
                $"cache types their reference outputs were captured with (this harness records {SupportedPairs}).");
        }

        KvSideFormat? k = Side(fixture.KvQuant.K);
        KvSideFormat? v = Side(fixture.KvQuant.V);
        if (k is null || v is null)
        {
            throw new InvalidOperationException(
                $"fixture '{fixturePath}' declares kv_quant k='{fixture.KvQuant.K}' v='{fixture.KvQuant.V}': " +
                $"LAIR's quantized attention route instantiates over 'q8_0' on the k side and 'q8_0' or 'q4_0' on the v " +
                $"side, so this suite reads only {SupportedPairs}.");
        }

        if (k != KvSideFormat.Q8_0 || (v != KvSideFormat.Q8_0 && v != KvSideFormat.Q4_0))
        {
            throw new InvalidOperationException(
                $"fixture '{fixturePath}' declares kv_quant k='{Of(k.Value)}' v='{Of(v.Value)}': this suite serves only " +
                $"{SupportedPairs}; regenerate it against a supported pair or move it out of the quantized KV fixture set.");
        }

        return new KvFormatPair(k.Value, v.Value);
    }

    private static List<(int Id, double Logprob)> QuantizedTop16<TK, TV>(TransformerModel model, IBackend backend, KvFormatPair formats, IReadOnlyList<int> tokens)
        where TK : struct, IKvFormat<TK>
        where TV : struct, IKvFormat<TV>
    {
        using BlockPool<TK, TV> pool = new(16, model.Desc.LayerCount, model.Desc.KvHeadCount, model.Desc.HeadDim, 64, backend, formats); // blockTokens 16 x capacityBlocks 64 covers the fixture's 1024-token decode window.
        using IKvSequence kv = pool.NewSequence();
        model.Prefill(tokens, kv);

        return TopWithLogprobs(model.LastLogits());
    }

    private static List<(int Id, double Logprob)> TopWithLogprobs(ReadOnlySpan<float> logits)
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
            if (top.Count < CalibrationWindow || logprob > top[^1].Logprob)
            {
                int at = top.BinarySearch((id, logprob), Comparer<(int Id, double Logprob)>.Create(
                    (a, b) => a.Logprob != b.Logprob ? b.Logprob.CompareTo(a.Logprob) : a.Id.CompareTo(b.Id)));
                if (at < 0)
                {
                    at = ~at;
                }

                top.Insert(at, (id, logprob));
                if (top.Count > CalibrationWindow)
                {
                    top.RemoveAt(CalibrationWindow);
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

    private static KvSideFormat? Side(string name) => name.Trim().ToLowerInvariant() switch
    {
        "q8_0" => KvSideFormat.Q8_0,
        "q4_0" => KvSideFormat.Q4_0,
        _ => null,
    };

    private static string Of(KvSideFormat side) => Lair.Kv.KvSideFormatNames.Of(side);

    private sealed record ParityFixture(
        [property: JsonPropertyName("model_file")] string ModelFile,
        [property: JsonPropertyName("model_sha256")] string ModelSha256,
        [property: JsonPropertyName("kv_quant")] KvQuant? KvQuant,
        [property: JsonPropertyName("quantized_kv_band")] double? QuantizedKvBand,
        [property: JsonPropertyName("prompts")] ParityPrompt[] Prompts);

    private sealed record KvQuant(
        [property: JsonPropertyName("k")] string? K,
        [property: JsonPropertyName("v")] string? V);

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
