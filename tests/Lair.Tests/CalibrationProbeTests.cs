using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Calibration probe used by scripts/generate-parity.ps1: when LAIR_CALIB_MODEL and LAIR_CALIB_OUT are set, writes
/// LAIR's top-16 next-token logprobs for each parity prompt (keyed p1..p4 in the generator's order) to the given
/// file. Without those variables it does nothing, so normal test runs pay no cost.
/// </summary>
[Trait("Category", "Calib")]
public sealed class CalibrationProbeTests
{
    private static readonly (string Key, string Prompt)[] Cases =
    [
        ("p1", "The capital of France is"),
        ("p2", "Once upon a time"),
        ("p3", "def fibonacci(n):\n"),
        ("p4", "<|im_start|>user\nHello\n<|im_end|>\n<|im_start|>assistant\n"),
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

        using GgufFile file = GgufFile.Open(modelPath);
        using CpuBackend backend = new(8);
        using TransformerModel model = TransformerModel.Open(file, backend, 1024);
        GgufTokenizer tok = GgufTokenizer.FromGguf(file);

        List<string> lines = [];
        foreach ((string key, string prompt) in Cases)
        {
            IReadOnlyList<int> ids = tok.AddBosToken ? tok.EncodeWithBos(prompt) : tok.Encode(prompt);
            using BlockPool pool = new(16, model.Desc.LayerCount, model.Desc.KvHeadCount, model.Desc.HeadDim, 64, backend);
            using SequenceKv kv = new(pool);
            model.Prefill(ids, kv);

            Span<float> logits = model.LastLogits();
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

            lines.Add($"CALIB {key} " + string.Join("|", top.Select(t => $"{t.Id}={t.Lp:F4}")));
        }

        File.WriteAllLines(outPath, lines);
    }
}
