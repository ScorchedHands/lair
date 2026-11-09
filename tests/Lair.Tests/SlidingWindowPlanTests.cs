using System.Text;

using Lair.Loader;
using Lair.Models;

namespace Lair.Tests;

/// <summary>
/// Load-time resolution of the declared attention window: an absent <c>attention.sliding_window</c> resolves to
/// the disabled zero for every profile including the generic llama-shaped fallback, a declared value resolves
/// exactly through the shared integer read (int32, uint32, and int64 declarations), a value declared under
/// another profile's prefix is not the plan's key, and a non-positive, zero, fractional, text, or
/// out-of-int-range declaration fails the load with a message naming the key and the declared value.
/// </summary>
[Trait("Category", "Loader")]
public sealed class SlidingWindowPlanTests : IDisposable
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
    public void ModelPlan_AbsentWindowKey_ResolvesDisabled_ForEveryProfile()
    {
        foreach (string architecture in new string[] { "llama", "qwen2", "qwen3" })
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata(architecture), [])));
            ModelPlan plan = ModelPlans.Resolve(gguf);

            Assert.Equal(0, plan.AttentionWindow);
        }

        // The generic llama-shaped fallback resolves the same key under the declared architecture's prefix.
        using GgufFile fallback = GgufFile.Open(WriteTemp(BuildGguf(Metadata("myarch", tiny: true), TinyTensors())));
        ModelPlan fallbackPlan = ModelPlans.Resolve(fallback);
        Assert.Equal(ModelProfiles.GenericArchitecture, fallbackPlan.Profile);
        Assert.Equal(0, fallbackPlan.AttentionWindow);
    }

    [Fact]
    public void ModelPlan_DeclaredWindowKey_ResolvesExactly_UnderTheProfilePrefix()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.attention.sliding_window", 4096)], [])));
        Assert.Equal(4096, ModelPlans.Resolve(gguf).AttentionWindow);

        // An int64 and a uint32 declaration resolve through the same integer read.
        using GgufFile int64 = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("qwen3"), ("qwen3.attention.sliding_window", 8192L)], [])));
        Assert.Equal(8192, ModelPlans.Resolve(int64).AttentionWindow);

        using GgufFile uint32 = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("qwen2"), ("qwen2.attention.sliding_window", 2048u)], [])));
        Assert.Equal(2048, ModelPlans.Resolve(uint32).AttentionWindow);

        // The key belongs to the prefix: the same declaration under the other profile's prefix is not seen.
        using GgufFile wrongPrefix = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("qwen2.attention.sliding_window", 4096)], [])));
        Assert.Equal(0, ModelPlans.Resolve(wrongPrefix).AttentionWindow);

        // The fallback profile resolves the declaration under the declared architecture's own prefix.
        using GgufFile fallback = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("myarch", tiny: true), ("myarch.attention.sliding_window", 1024)], TinyTensors())));
        ModelPlan fallbackPlan = ModelPlans.Resolve(fallback);
        Assert.Equal(ModelProfiles.GenericArchitecture, fallbackPlan.Profile);
        Assert.Equal(1024, fallbackPlan.AttentionWindow);
    }

    [Theory]
    [InlineData(0, "0")]
    [InlineData(-1, "-1")]
    [InlineData(-4096, "-4096")]
    [InlineData(3.5f, "3.5")]
    [InlineData(0.5f, "0.5")]
    [InlineData(3000000000L, "3000000000")] // an int64 declaration past the int32 range cannot become a window
    public void ModelPlan_InvalidWindowDeclaration_FailsLoadNamingKeyAndValue(object declared, string valueText)
    {
        foreach (string prefix in new string[] { "llama", "qwen2", "qwen3" })
        {
            string key = prefix + ".attention.sliding_window";
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf([.. Metadata(prefix), (key, declared)], [])));
            ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

            Assert.Contains(key, ex.Message);
            Assert.Contains(valueText, ex.Message);
        }
    }

    [Fact]
    public void ModelPlan_TextWindowDeclaration_FailsLoadNamingKeyAndValue()
    {
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(
            [.. Metadata("llama"), ("llama.attention.sliding_window", "wide")], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelPlans.Resolve(gguf));

        Assert.Contains("llama.attention.sliding_window", ex.Message);
        Assert.Contains("wide", ex.Message);
    }

    // ---------------------------------------------------------------- shared helpers

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-window-plan-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // Canonical metadata pairs per architecture (parse-only fixtures need no tensors); the tiny variant keeps
    // the geometry small for the fallback fixtures.
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
                case uint u:
                    writer.Write(4u); // UINT32
                    writer.Write(u);
                    break;
                case long l:
                    writer.Write(11u); // INT64
                    writer.Write(l);
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
