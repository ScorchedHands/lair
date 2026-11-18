using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Calibration probe used by scripts/generate-parity.ps1 and scripts/generate-kv-cache-type-parity.ps1: when
/// LAIR_CALIB_MODEL and LAIR_CALIB_OUT are set, writes LAIR's top-16 next-token logprobs for each parity prompt (keyed
/// p1..p4 in the generator's order) to the given file. Without those variables it does nothing, so normal test runs pay
/// no cost. The calibration posture is FP32 unless both new opt-in variables LAIR_CALIB_KV_K and LAIR_CALIB_V name a
/// quantized KV pair; when they are present but name anything outside this harness's supported pairs ('q8_0/q8_0' and
/// 'q8_0/q4_0'), the probe fails naming what was declared, which variables own them, and the supported pairs — it never
/// falls back to FP32 for a calibration that asked to be quantized. A quantized calibration opens the same CPU route
/// serving uses: TransformerModel's quantized-pair seam over the file's resolved single-span route with the integer-dot
/// encoding this backend resolved, and the typed block pool its pair's group layout sizes.
/// </summary>
[Trait("Category", "Calib")]
public sealed class CalibrationProbeTests
{
    private static readonly (string Key, string Prompt)[] Cases =
    [
        ("p1", "The capital of France is"),
        ("p2", "Once upon a time"),
        ("p3", "def fibonacci(n):\n"),
        ("p4", "<|im_start|>user\nReply with a single sentence about the planet Mars.\n<|im_end|>\n<|im_start|>assistant\n"),
    ];

    [Fact]
    public void WriteTop16PerPrompt()
    {
        string? modelPath = Environment.GetEnvironmentVariable("LAIR_CALIB_MODEL");
        string? outPath = Environment.GetEnvironmentVariable("LAIR_CALIB_OUT");
        if (modelPath is null || outPath is null)
        {
            return;
        }

        KvFormatPair? quantized = ResolveCalibrationKvQuant(); // absent means today's FP32 posture, unchanged.

        using GgufFile file = GgufFile.Open(modelPath);
        using CpuBackend backend = new(8);
        using TransformerModel model = quantized is null
            ? TransformerModel.Open(file, backend, 1024)
            : TransformerModel.Open(
                file,
                backend,
                1024,
                logLine: null,
                PrefillKernel.GemV, // the batched-row window posture the quantized attention route carries.
                ModelPlans.Resolve(file).Route, // this file's resolved single DRAM-resident span; a split is refused by the engine for a quantized pair.
                quantized.Value,
                backend.ResolvedIntegerDotPath);
        GgufTokenizer tok = GgufTokenizer.FromGguf(file);

        List<string> lines = [];
        foreach ((string key, string prompt) in Cases)
        {
            IReadOnlyList<int> ids = tok.AddBosToken ? tok.EncodeWithBos(prompt) : tok.Encode(prompt);

            List<(int Id, double Lp)> top;
            if (quantized is null)
            {
                using BlockPool pool = new(16, model.Desc.LayerCount, model.Desc.KvHeadCount, model.Desc.HeadDim, 64, backend);
                using SequenceKv kv = new(pool);
                model.Prefill(ids, kv);
                top = TopWithLogprobs(model.LastLogits());
            }
            else if (quantized.Value.V == KvSideFormat.Q8_0)
            {
                top = QuantizedTop16<Q8Block, Q8Block>(model, backend, quantized.Value, ids);
            }
            else
            {
                top = QuantizedTop16<Q8Block, Q4Block>(model, backend, quantized.Value, ids);
            }

            lines.Add($"CALIB {key} " + string.Join("|", top.Select(t => $"{t.Id}={t.Lp:F4}")));
        }

        File.WriteAllLines(outPath, lines);
    }

    /// <summary>
    /// The quantized KV posture this calibration run declares: null when neither opt-in variable is set (today's FP32
    /// behavior), otherwise the validated pair. A declaration naming one side only, an unrecognized format name, or a
    /// combination outside 'q8_0/q8_0' and 'q8_0/q4_0' fails with an actionable message instead of calibrating FP32.
    /// </summary>
    private static KvFormatPair? ResolveCalibrationKvQuant()
    {
        string? rawK = Environment.GetEnvironmentVariable("LAIR_CALIB_KV_K");
        string? rawV = Environment.GetEnvironmentVariable("LAIR_CALIB_V");
        if (rawK is null && rawV is null)
        {
            return null; // no opt-in declared: the probe calibrates the default FP32 route exactly as before.
        }

        const string SupportedPairs = "'q8_0/q8_0' and 'q8_0/q4_0'";

        if (string.IsNullOrWhiteSpace(rawK) || string.IsNullOrWhiteSpace(rawV))
        {
            throw new InvalidOperationException(
                "LAIR calibration declares a quantized KV posture with only one side set: LAIR_CALIB_KV_K=" +
                $"'{rawK}' and LAIR_CALIB_V='{rawV}'. Name both variables, or neither to calibrate the FP32 route; " +
                $"the pairs this harness calibrates are {SupportedPairs}.");
        }

        KvSideFormat? k = SideForCalibration(rawK);
        KvSideFormat? v = SideForCalibration(rawV);
        if (k is null || v is null)
        {
            throw new InvalidOperationException(
                $"LAIR calibration declares kv.quant k='{rawK}' v='{rawV}': " +
                $"{(k is null ? $"'{rawK}' is not a KV format name LAIR calibrates (use 'q8_0' or 'q4_0')" : $"'{rawV}' is not a KV format name LAIR calibrates (use 'q8_0' or 'q4_0')")}. " +
                $"Set LAIR_CALIB_KV_K and LAIR_CALIB_V to one of the pairs this harness calibrates: {SupportedPairs}; refusing rather than falling back to FP32 calibration.");
        }

        if (k != KvSideFormat.Q8_0 || (v != KvSideFormat.Q8_0 && v != KvSideFormat.Q4_0))
        {
            throw new InvalidOperationException(
                $"LAIR calibration declares kv.quant k='{rawK}' v='{rawV}': LAIR's quantized attention route instantiates " +
                $"over 'q8_0' on the k side and 'q8_0' or 'q4_0' on the v side, so this harness calibrates only {SupportedPairs}. " +
                "Set both variables to a supported pair, or unset them to calibrate the FP32 route; refusing rather than falling back.");
        }

        return new KvFormatPair(k.Value, v.Value);
    }

    private static KvSideFormat? SideForCalibration(string name) => name.Trim().ToLowerInvariant() switch
    {
        "q8_0" => KvSideFormat.Q8_0,
        "q4_0" => KvSideFormat.Q4_0,
        _ => null,
    };

    /// <summary>One prompt's top-16 next-token logprobs over the typed pool this quantized pair addresses.</summary>
    private static List<(int Id, double Lp)> QuantizedTop16<TK, TV>(TransformerModel model, IBackend backend, KvFormatPair formats, IReadOnlyList<int> ids)
        where TK : struct, IKvFormat<TK>
        where TV : struct, IKvFormat<TV>
    {
        using BlockPool<TK, TV> pool = new(16, model.Desc.LayerCount, model.Desc.KvHeadCount, model.Desc.HeadDim, 64, backend, formats); // blockTokens 16 x capacityBlocks 64 covers the probe's 1024-token window.
        using IKvSequence kv = pool.NewSequence();
        model.Prefill(ids, kv);

        return TopWithLogprobs(model.LastLogits());
    }

    private static List<(int Id, double Lp)> TopWithLogprobs(ReadOnlySpan<float> logits)
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

        double lse = max + Math.Log(sumExp);
        List<(int Id, double Lp)> top = [];
        for (int id = 0; id < logits.Length; id++)
        {
            double lp = logits[id] - lse;
            if (top.Count < 16 || lp > top[^1].Lp)
            {
                int at = top.BinarySearch((id, lp), Comparer<(int Id, double Lp)>.Create(
                    (a, b) => a.Lp != b.Lp ? b.Lp.CompareTo(a.Lp) : a.Id.CompareTo(b.Id)));
                if (at < 0)
                {
                    at = ~at;
                }

                top.Insert(at, (id, lp));
                if (top.Count > 16)
                {
                    top.RemoveAt(16);
                }
            }
        }

        return top;
    }
}
