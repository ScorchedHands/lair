using System.Text;

using Lair.Loader;
using Lair.Models;

namespace Lair.Tests;

/// <summary>
/// Load-time resolution of the attention-window interleave: the gemma2 profile resolves the alternating pattern —
/// the declared window masking the even-indexed layers and leaving the odd-indexed ones unbounded — while every
/// other registered profile and the generic llama-shaped fallback resolve the all-layers pattern. The pattern comes
/// from the resolved profile alone: it is the same whether or not the file declares
/// <c>&lt;prefix&gt;attention.sliding_window</c>, and that key stays the only source of the window length.
/// </summary>
[Trait("Category", "Loader")]
public sealed class WindowInterleavePlanTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

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

    [Fact]
    public void ModelPlan_Gemma2Profile_ResolvesAlternatingLocalFirst_WithOrWithoutADeclaredWindow()
    {
        using (GgufFile undeclared = GgufFile.Open(WriteTemp(BuildGguf(Metadata("gemma2"), []))))
        {
            ModelPlan plan = ModelPlans.Resolve(undeclared);

            Assert.Equal(WindowInterleave.AlternatingLocalFirst, plan.WindowInterleave);
            Assert.Equal(0, plan.AttentionWindow); // no declaration: unbounded whatever the pattern
        }

        using GgufFile declared = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("gemma2"), ("gemma2.attention.sliding_window", 4096)], [])));
        ModelPlan declaredPlan = ModelPlans.Resolve(declared);

        Assert.Equal(WindowInterleave.AlternatingLocalFirst, declaredPlan.WindowInterleave);
        Assert.Equal(4096, declaredPlan.AttentionWindow); // the declared key stays the only source of the length
    }

    [Fact]
    public void ModelPlan_EveryOtherProfileAndTheFallback_ResolvesAllLayers_WithOrWithoutADeclaredWindow()
    {
        foreach (string prefix in new string[] { "llama", "qwen2", "qwen3" })
        {
            using (GgufFile profileUndeclared = GgufFile.Open(WriteTemp(BuildGguf(Metadata(prefix), []))))
            {
                ModelPlan profilePlan = ModelPlans.Resolve(profileUndeclared);

                Assert.Equal(WindowInterleave.AllLayers, profilePlan.WindowInterleave);
                Assert.Equal(0, profilePlan.AttentionWindow);
            }

            using GgufFile profileDeclared = GgufFile.Open(WriteTemp(BuildGguf(
                [.. Metadata(prefix), ($"{prefix}.attention.sliding_window", 2048)], [])));
            ModelPlan profileDeclaredPlan = ModelPlans.Resolve(profileDeclared);

            Assert.Equal(WindowInterleave.AllLayers, profileDeclaredPlan.WindowInterleave);
            Assert.Equal(2048, profileDeclaredPlan.AttentionWindow);
        }

        // The generic llama-shaped fallback keeps the llama-shaped pattern whatever architecture it was reached under.
        using (GgufFile fallbackUndeclared = GgufFile.Open(WriteTemp(BuildGguf(Metadata("myarch", tiny: true), TinyTensors()))))
        {
            ModelPlan fallbackPlan = ModelPlans.Resolve(fallbackUndeclared);

            Assert.Equal(ModelProfiles.GenericArchitecture, fallbackPlan.Profile);
            Assert.Equal(WindowInterleave.AllLayers, fallbackPlan.WindowInterleave);
        }

        using GgufFile fallbackDeclared = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("myarch", tiny: true), ("myarch.attention.sliding_window", 1024)], TinyTensors())));
        ModelPlan fallbackDeclaredPlan = ModelPlans.Resolve(fallbackDeclared);

        Assert.Equal(ModelProfiles.GenericArchitecture, fallbackDeclaredPlan.Profile);
        Assert.Equal(WindowInterleave.AllLayers, fallbackDeclaredPlan.WindowInterleave);
        Assert.Equal(1024, fallbackDeclaredPlan.AttentionWindow);
    }

    // ---------------------------------------------------------------- shared helpers

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-window-interleave-plan-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // Canonical metadata pairs per architecture (parse-only fixtures need no tensors); the tiny variant keeps the
    // geometry small for the generic-fallback fixtures, whose tensor table the fallback gate must match.
    private static List<(string Key, object Value)> Metadata(string prefix, bool tiny = false)
    {
        if (tiny)
        {
            return
            [
                ("general.architecture", prefix),
                ($"{prefix}.embedding_length", 4),
                ($"{prefix}.block_count", 1),
                ($"{prefix}.attention.head_count", 2),
                ($"{prefix}.attention.head_count_kv", 2),
                ($"{prefix}.feed_forward_length", 8),
                ($"{prefix}.attention.layer_norm_rms_epsilon", 1e-5f),
                ($"{prefix}.vocab_size", 6),
            ];
        }

        return
        [
            ("general.architecture", prefix),
            ($"{prefix}.embedding_length", 576),
            ($"{prefix}.block_count", 30),
            ($"{prefix}.attention.head_count", 9),
            ($"{prefix}.attention.head_count_kv", 3),
            ($"{prefix}.feed_forward_length", 1536),
            ($"{prefix}.attention.layer_norm_rms_epsilon", 1e-5f),
            ($"{prefix}.vocab_size", 49_152),
        ];
    }

    // A complete single-layer llama-shaped tensor table the generic fallback gate accepts; payloads are
    // irrelevant to plan resolution.
    private static List<TensorDef> TinyTensors() =>
        [
            new("token_embd.weight", [6, 4]),
            new("blk.0.attn_q.weight", [4, 4]),
            new("blk.0.attn_k.weight", [4, 4]),
            new("blk.0.attn_v.weight", [4, 4]),
            new("blk.0.attn_output.weight", [4, 4]),
            new("blk.0.ffn_gate.weight", [8, 4]),
            new("blk.0.ffn_up.weight", [8, 4]),
            new("blk.0.ffn_down.weight", [4, 8]),
            new("blk.0.attn_norm.weight", [4]),
            new("blk.0.ffn_norm.weight", [4]),
            new("output_norm.weight", [4]),
        ];

    /// <summary>Builds a GGUF v3 image from KV pairs; no tensors are needed for parse-only resolution.</summary>
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

            for (long e = 0; e < elements; e++)
            {
                writer.Write(0f);
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

    private sealed record TensorDef(string Name, int[] Dims); // dims in logical (row-major) order; vectors are one-dimensional
}
