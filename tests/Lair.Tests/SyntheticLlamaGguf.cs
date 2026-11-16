using System.Text;

using Lair.Contracts;

namespace Lair.Tests;

internal static class SyntheticLlamaGguf
{
    private const string ChatMlTemplate =
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

    /// <summary>The tensor geometry one synthetic file declares: transformer layer count, hidden size, attention head and kv-head counts, and feed-forward width.</summary>
    internal sealed record Geometry(int Layers, int HiddenSize, int HeadCount, int KvHeadCount, int FeedForwardLength);

    // The original tiny shape every consumer has always read — hidden 4 over two heads of dimension 2, one kv head, a feed-forward width of 8.
    private static readonly Geometry TinyShape = new(Layers: 2, HiddenSize: 4, HeadCount: 2, KvHeadCount: 1, FeedForwardLength: 8);

    /// <summary>
    /// Writes one tiny synthetic llama-shaped GGUF. With <paramref name="withReasoningMarkers"/> the vocabulary carries
    /// <c>&lt;think&gt;</c> and <c>&lt;/think&gt;</c> as ordinary entries, so a reasoning budget declared on this model has
    /// whole-marker tokens to bind; without them the vocabulary carries none, which is the shape a budget gate must refuse.
    /// Every tensor ships in <paramref name="format"/>: F32 keeps the original byte pattern exactly, while F16 and BF16
    /// payloads carry quarter-scale values that survive both stored formats unchanged. This overload declares the
    /// original tiny geometry every consumer has always read.
    /// </summary>
    public static string Write(string modelName, bool withReasoningMarkers = true, QuantFormat format = QuantFormat.F32) =>
        Write(modelName, withReasoningMarkers, format, TinyShape);

    /// <summary>
    /// The same file shape at a declared geometry: the matrix tensors scale and every other byte stays as the tiny model writes it.
    /// A wider entry exists so an envelope gate can watch weight bytes actually cross to device memory; F16/BF16 payloads keep
    /// their quarter-scale values, which survive both stored formats unchanged on either side of a split route. The head dimension
    /// derives from hidden size divided by head count exactly as plan resolution computes it, so the declared tensor shapes match
    /// what the forward engine checks against the metadata.
    /// </summary>
    public static string Write(string modelName, bool withReasoningMarkers, QuantFormat format, Geometry geometry)
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

        string[] tokens = withReasoningMarkers
            ? ["<unk>", "hello", "world", "!", "<s>", "</s>", "<think>", "</think>"]
            : ["<unk>", "hello", "world", "!", "<s>", "</s>"];
        int[] tokenTypes = withReasoningMarkers
            ? [2, 1, 1, 1, 4, 5, 1, 1]
            : [2, 1, 1, 1, 4, 5];
        int vocabSize = tokens.Length;

        // The per-head dimension is derived the way plan resolution derives it when a file declares no explicit head length.
        int headDim = geometry.HiddenSize / geometry.HeadCount;
        int qRows = geometry.HeadCount * headDim;
        int kvRows = geometry.KvHeadCount * headDim;

        List<(string Name, int[] Dims)> tensors = [new("token_embd.weight", [vocabSize, geometry.HiddenSize])];
        for (int layer = 0; layer < geometry.Layers; layer++)
        {
            tensors.Add(($"blk.{layer}.attn_q.weight", [qRows, geometry.HiddenSize]));
            tensors.Add(($"blk.{layer}.attn_k.weight", [kvRows, geometry.HiddenSize]));
            tensors.Add(($"blk.{layer}.attn_v.weight", [kvRows, geometry.HiddenSize]));
            tensors.Add(($"blk.{layer}.attn_output.weight", [geometry.HiddenSize, qRows]));
            tensors.Add(($"blk.{layer}.ffn_gate.weight", [geometry.FeedForwardLength, geometry.HiddenSize]));
            tensors.Add(($"blk.{layer}.ffn_up.weight", [geometry.FeedForwardLength, geometry.HiddenSize]));
            tensors.Add(($"blk.{layer}.ffn_down.weight", [geometry.HiddenSize, geometry.FeedForwardLength]));
            tensors.Add(($"blk.{layer}.attn_norm.weight", [geometry.HiddenSize]));
            tensors.Add(($"blk.{layer}.ffn_norm.weight", [geometry.HiddenSize]));
        }

        tensors.Add(new("output_norm.weight", [geometry.HiddenSize]));

        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u);          // version
        writer.Write((ulong)tensors.Count);
        writer.Write(19ul);        // 18 metadata keys plus the chat template

        WriteStringKv(writer, "general.architecture", "llama");
        WriteStringKv(writer, "general.name", modelName);
        WriteInt32Kv(writer, "llama.embedding_length", geometry.HiddenSize);
        WriteInt32Kv(writer, "llama.block_count", geometry.Layers);
        WriteInt32Kv(writer, "llama.attention.head_count", geometry.HeadCount);
        WriteInt32Kv(writer, "llama.attention.head_count_kv", geometry.KvHeadCount);
        WriteInt32Kv(writer, "llama.feed_forward_length", geometry.FeedForwardLength);
        WriteFloat32Kv(writer, "llama.rope.freq_base", 10_000f);
        WriteFloat32Kv(writer, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        WriteInt32Kv(writer, "llama.vocab_size", vocabSize);
        WriteStringKv(writer, "tokenizer.ggml.model", "gpt2");
        WriteStringArrayKv(writer, "tokenizer.ggml.tokens", tokens);
        WriteInt32ArrayKv(writer, "tokenizer.ggml.token_type", tokenTypes);
        WriteStringArrayKv(writer, "tokenizer.ggml.merges", []);
        WriteUint32Kv(writer, "tokenizer.ggml.bos_token_id", 4);
        WriteUint32Kv(writer, "tokenizer.ggml.eos_token_id", 5);
        WriteUint32Kv(writer, "tokenizer.ggml.unknown_token_id", 0);
        WriteBoolKv(writer, "tokenizer.ggml.add_bos_token", false);
        WriteStringKv(writer, "tokenizer.chat_template", ChatMlTemplate);

        long offset = 0;
        foreach ((string name, int[] dims) in tensors)
        {
            WriteString(writer, name);
            writer.Write((uint)dims.Length);
            for (int d = dims.Length - 1; d >= 0; d--) // GGUF stores dims fastest-varying first
            {
                writer.Write((ulong)dims[d]);
            }

            writer.Write(GgmlTypeId(format));
            writer.Write((ulong)offset);

            long elements = 1;
            foreach (int dim in dims)
            {
                elements *= dim;
            }

            offset += elements * ElementBytes(format);
        }

        long headerLength = stream.Position;
        long dataStart = (headerLength + 31) / 32 * 32; // GGUF default alignment is 32
        for (long i = headerLength; i < dataStart; i++)
        {
            writer.Write((byte)0);
        }

        int ordinal = 0;
        foreach ((string _, int[] dims) in tensors)
        {
            long elements = 1;
            foreach (int dim in dims)
            {
                elements *= dim;
            }

            for (long e = 0; e < elements; e++)
            {
                float value = SampleValue(ordinal, e, format);
                switch (format)
                {
                    case QuantFormat.F32:
                        writer.Write(value);
                        break;
                    case QuantFormat.F16:
                        writer.Write(BitConverter.HalfToInt16Bits((Half)value));
                        break;
                    default: // BF16: the f32 high half written little-endian.
                        writer.Write((ushort)(BitConverter.SingleToUInt32Bits(value) >> 16));
                        break;
                }
            }

            ordinal++;
        }

        writer.Flush();

        string path = Path.Combine(Path.GetTempPath(), $"lair-budget-scheduling-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, stream.ToArray());
        return path;
    }

    private static uint GgmlTypeId(QuantFormat format) => format switch
    {
        QuantFormat.F32 => 0u,
        QuantFormat.F16 => 1u,
        _ => 30u, // BF16
    };

    private static int ElementBytes(QuantFormat format) => format == QuantFormat.F32 ? 4 : 2;

    private static float SampleValue(int ordinal, long e, QuantFormat format) => format switch
    {
        QuantFormat.F32 => ordinal % 2 == 1 ? 1.25f : MathF.Sin((ordinal * 17 + e) * 0.13f),
        _ => ((ordinal * 7 + (int)(e % 9)) % 9 - 4) / 4f, // quarter-scale values exact in both stored formats
    };

    private static void WriteString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private static void WriteStringKv(BinaryWriter writer, string key, string value)
    {
        WriteString(writer, key);
        writer.Write(8u); // STRING
        WriteString(writer, value);
    }

    private static void WriteInt32Kv(BinaryWriter writer, string key, int value)
    {
        WriteString(writer, key);
        writer.Write(5u); // INT32
        writer.Write(value);
    }

    private static void WriteUint32Kv(BinaryWriter writer, string key, uint value)
    {
        WriteString(writer, key);
        writer.Write(4u); // UINT32
        writer.Write(value);
    }

    private static void WriteFloat32Kv(BinaryWriter writer, string key, float value)
    {
        WriteString(writer, key);
        writer.Write(6u); // FLOAT32
        writer.Write(value);
    }

    private static void WriteBoolKv(BinaryWriter writer, string key, bool value)
    {
        WriteString(writer, key);
        writer.Write(7u); // BOOL
        writer.Write((byte)(value ? 1 : 0));
    }

    private static void WriteStringArrayKv(BinaryWriter writer, string key, string[] values)
    {
        WriteString(writer, key);
        writer.Write(9u); // ARRAY
        writer.Write(8u); // element type STRING
        writer.Write((ulong)values.Length);
        foreach (string value in values)
        {
            WriteString(writer, value);
        }
    }

    private static void WriteInt32ArrayKv(BinaryWriter writer, string key, int[] values)
    {
        WriteString(writer, key);
        writer.Write(9u); // ARRAY
        writer.Write(5u); // element type INT32
        writer.Write((ulong)values.Length);
        foreach (int value in values)
        {
            writer.Write(value);
        }
    }
}
