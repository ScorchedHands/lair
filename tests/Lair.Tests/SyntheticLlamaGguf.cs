using System.Text;

namespace Lair.Tests;

internal static class SyntheticLlamaGguf
{
    private const string ChatMlTemplate =
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

    /// <summary>
    /// Writes one tiny synthetic llama-shaped GGUF. With <paramref name="withReasoningMarkers"/> the vocabulary carries
    /// <c>&lt;think&gt;</c> and <c>&lt;/think&gt;</c> as ordinary entries, so a reasoning budget declared on this model has
    /// whole-marker tokens to bind; without them the vocabulary carries none, which is the shape a budget gate must refuse.
    /// </summary>
    public static string Write(string modelName, bool withReasoningMarkers = true)
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

        List<(string Name, int[] Dims)> tensors = [new("token_embd.weight", [vocabSize, 4])];
        for (int layer = 0; layer < 2; layer++)
        {
            tensors.Add(($"blk.{layer}.attn_q.weight", [4, 4]));
            tensors.Add(($"blk.{layer}.attn_k.weight", [2, 4]));
            tensors.Add(($"blk.{layer}.attn_v.weight", [2, 4]));
            tensors.Add(($"blk.{layer}.attn_output.weight", [4, 4]));
            tensors.Add(($"blk.{layer}.ffn_gate.weight", [8, 4]));
            tensors.Add(($"blk.{layer}.ffn_up.weight", [8, 4]));
            tensors.Add(($"blk.{layer}.ffn_down.weight", [4, 8]));
            tensors.Add(($"blk.{layer}.attn_norm.weight", [4]));
            tensors.Add(($"blk.{layer}.ffn_norm.weight", [4]));
        }

        tensors.Add(new("output_norm.weight", [4]));

        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u);          // version
        writer.Write((ulong)tensors.Count);
        writer.Write(19ul);        // 18 metadata keys plus the chat template

        WriteStringKv(writer, "general.architecture", "llama");
        WriteStringKv(writer, "general.name", modelName);
        WriteInt32Kv(writer, "llama.embedding_length", 4);
        WriteInt32Kv(writer, "llama.block_count", 2);
        WriteInt32Kv(writer, "llama.attention.head_count", 2);
        WriteInt32Kv(writer, "llama.attention.head_count_kv", 1);
        WriteInt32Kv(writer, "llama.feed_forward_length", 8);
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

            writer.Write(0u); // ggml type F32
            writer.Write((ulong)offset);

            long elements = 1;
            foreach (int dim in dims)
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
                writer.Write(ordinal % 2 == 1 ? 1.25f : MathF.Sin((ordinal * 17 + e) * 0.13f));
            }

            ordinal++;
        }

        writer.Flush();

        string path = Path.Combine(Path.GetTempPath(), $"lair-budget-scheduling-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, stream.ToArray());
        return path;
    }

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
