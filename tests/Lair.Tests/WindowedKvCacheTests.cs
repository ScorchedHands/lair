using System.Text;

using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Prompt-cache reuse under windowed KV reads, on a synthetic four-layer model read through the alternating
/// interleave: a published prefix is adopted instead of prefilled and its continuation logits stay bit-identical to
/// a cold run even though half the layers never address most of the shared blocks; per-layer block selections over an
/// adopted table exclude whole shared blocks on the windowed layers while the full-attention layers still select the
/// entire table; and a write into an in-window position of an adopted block copies it private without disturbing the
/// publisher or the selection that reads it. Engine-to-engine comparisons are asserted bit-for-bit.
/// </summary>
[Trait("Category", "Cache")]
public sealed class WindowedKvCacheTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

    // Three positions per window, two tokens per block: a query beyond the window excludes whole blocks, which is what
    // makes shared-block reuse observable rather than incidental.
    private const int Window = 3;
    private const int BlockTokens = 2;

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
    public void AdoptedPrefix_UnderAlternatingWindows_ContinuesBitIdenticallyToAColdRun()
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata(Window), Tensors())));
        using TransformerModel model = TransformerModel.Open(
            gguf, backend, maxContextTokens: 32, ModelProfiles.Resolve("llama") with { WindowInterleave = WindowInterleave.AlternatingLocalFirst });

        Assert.Equal(WindowInterleave.AlternatingLocalFirst, model.Plan.WindowInterleave);
        Assert.Equal(Window, model.Plan.AttentionWindow);
        Assert.Equal(4, model.Plan.LayerCount);

        using BlockPool pool = new(blockTokens: BlockTokens, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 20, backend);
        PrefixCache cache = new(pool);

        int[] sharedPrefix = [1, 4, 2, 5, 3, 0]; // three full two-token blocks of identical leading text
        int[] tailB = [5, 1]; // fresh ids inside the fixture's six-token vocabulary: only the tail differs from the published prompt
        int[] promptB = [.. sharedPrefix, .. tailB];

        // Publisher: the prefix prefills real model KV and every complete block is published.
        using SequenceKv publisher = new(pool);
        model.Prefill(sharedPrefix, publisher);
        Assert.Equal(3, cache.Publish(sharedPrefix, publisher));
        int[] publishedBlocks = [.. publisher.Blocks];

        // Warm: the same leading blocks adopt instead of prefilling, and only the differing tail forwards fresh.
        using SequenceKv warm = new(pool);
        Assert.Equal(3, cache.TryAdopt(promptB, warm));
        Assert.Equal(sharedPrefix.Length, warm.Length); // prefill resumes after the shared prefix
        Assert.True(warm.IsAdopted(0) && warm.IsAdopted(3), "the adopted positions must be marked shared");
        for (int index = 0; index < publishedBlocks.Length; index++)
        {
            Assert.Equal(publishedBlocks[index], warm.Blocks[index]); // the same physical blocks back both tables
            Assert.True(pool.RefCount(publishedBlocks[index]) >= 2, "publisher and cache references must still hold the shared block");
        }

        model.Prefill(tailB, warm);
        float[] warmLogits = model.DecodeNext(2, warm).ToArray(); // position 8 continues from the reused prefix

        Assert.Equal(sharedPrefix.Length + tailB.Length, warm.Length - 1);

        // Cold baseline: the identical prompt prefilled whole on a fresh sequence and cache-free comparison.
        using SequenceKv cold = new(pool);
        model.Prefill(promptB, cold);
        float[] coldLogits = model.DecodeNext(2, cold).ToArray();

        Assert.True(
            warmLogits.AsSpan().SequenceEqual(coldLogits.AsSpan()),
            "a cache-hit continuation must be bit-identical to the same prompt prefilled from nothing");

        // And the reuse really is what a windowed read sees: at this query position the masked layers select only the
        // tail-side blocks, skipping whole adopted blocks, while every unmasked layer selects all of them.
        int queryPosition = warm.Length - 1;
        for (int layer = 0; layer < model.Plan.LayerCount; layer++)
        {
            BlockSelection read = warm.SelectCausalRead(queryPosition, model.Plan.AttentionWindowForLayer(layer));

            if (layer % 2 == 0)
            {
                Assert.Equal(6, read.FirstPosition); // [6..8]: the shared blocks covering positions 0..5 are not addressed
                Assert.True(read.SkippedBlocks >= 3, $"a masked layer selected {read.SelectedBlocks} of {read.TotalBlocks} blocks and skipped only {read.SkippedBlocks}");
            }
            else
            {
                Assert.Equal(0, read.FirstPosition);
                Assert.True(read.CoversWholeTable, "an unmasked layer must select every block the sequence holds, adopted ones included");

                // Its reads do reach the shared bytes: position 1 lives in an adopted block.
                Assert.Equal(publishedBlocks[0], warm.BlockFor(1));
            }
        }

        publisher.Dispose();
        warm.Dispose();
        cold.Dispose();
    }

    [Fact]
    public void WriteIntoAnAdoptedInWindowPosition_CopiesPrivate_AndTheSelectionKeepsReadingIntactBytes()
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(Metadata(Window), Tensors())));
        using TransformerModel model = TransformerModel.Open(
            gguf, backend, maxContextTokens: 32, ModelProfiles.Resolve("llama") with { WindowInterleave = WindowInterleave.AlternatingLocalFirst });

        using BlockPool pool = new(blockTokens: BlockTokens, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 12, backend);
        PrefixCache cache = new(pool);

        int[] prompt = [1, 4, 2, 5]; // two published blocks
        using SequenceKv publisher = new(pool);
        model.Prefill(prompt, publisher);
        Assert.Equal(2, cache.Publish(prompt, publisher));

        using SequenceKv sharer = new(pool);
        Assert.Equal(2, cache.TryAdopt([.. prompt, 4], sharer));

        int sharedBlock = sharer.BlockFor(3); // positions 2..3 sit in the second adopted block
        BlockSelection read = sharer.SelectCausalRead(queryPosition: 3, layerWindow: Window); // [1..3] covers both adopted blocks
        Assert.Equal(0, read.FirstBlockIndex);
        Assert.Equal(2, read.SelectedBlocks);

        float[] sharedKBefore = publisher.ReadK(3, 0).ToArray();
        Assert.True(read.FirstPosition <= 3 && 3 <= read.LastPosition);
        Assert.True(sharer.ReadK(read, 3, 0).SequenceEqual(sharedKBefore), "a windowed read must see the publisher's bytes through the shared block");

        // A rewrite inside the selection copies that block private first: the selection keeps addressing the same
        // logical blocks, the writer sees its own value, and the holder it shares with is untouched.
        float[] rewritten = LoudRow(pool.KvDim);
        sharer.WriteK(3, 0, rewritten);

        Assert.False(sharer.IsAdopted(3));
        Assert.NotEqual(sharedBlock, sharer.BlockFor(3));
        Assert.True(sharer.ReadK(read, 3, 0).SequenceEqual(rewritten), "the selection must read the private copy after copy-on-write");
        Assert.True(publisher.ReadK(3, 0).ToArray().AsSpan().SequenceEqual(sharedKBefore), "the publisher's shared block must be undisturbed");

        BlockSelection after = sharer.SelectCausalRead(queryPosition: 3, layerWindow: Window);
        Assert.Equal(read.FirstBlockIndex, after.FirstBlockIndex); // the same logical run selects; only its physical mapping moved
        Assert.NotEqual(publisher.Blocks[1], sharer.Blocks[after.LastBlockIndex]);

        publisher.Dispose();
        sharer.Dispose();
    }

    // ---------------------------------------------------------------- fixture model

    private static List<(string Key, object Value)> Metadata(int window) =>
        [
            ("general.architecture", "llama"),
            ("llama.embedding_length", 6),
            ("llama.block_count", 4),
            ("llama.attention.head_count", 2),
            ("llama.attention.head_count_kv", 1),
            ("llama.feed_forward_length", 8),
            ("llama.attention.head_length", 4),
            ("llama.attention.layer_norm_rms_epsilon", 1e-5f),
            ("llama.vocab_size", 6),
            ("llama.attention.sliding_window", window),
        ];

    private static List<TensorDef> Tensors()
    {
        List<TensorDef> tensors = [new("token_embd.weight", [6, 6], Weights(7, 6, 6))];

        for (int layer = 0; layer < 4; layer++)
        {
            int salt = 11 + 7 * layer;
            tensors.Add(new($"blk.{layer}.attn_q.weight", [8, 6], Weights(salt, 8, 6)));
            tensors.Add(new($"blk.{layer}.attn_k.weight", [4, 6], Weights(salt + 1, 4, 6)));
            tensors.Add(new($"blk.{layer}.attn_v.weight", [4, 6], Weights(salt + 2, 4, 6)));
            tensors.Add(new($"blk.{layer}.attn_output.weight", [6, 8], Weights(salt + 3, 6, 8)));
            tensors.Add(new($"blk.{layer}.ffn_gate.weight", [8, 6], Weights(salt + 4, 8, 6)));
            tensors.Add(new($"blk.{layer}.ffn_up.weight", [8, 6], Weights(salt + 5, 8, 6)));
            tensors.Add(new($"blk.{layer}.ffn_down.weight", [6, 8], Weights(salt + 6, 6, 8)));
            tensors.Add(new($"blk.{layer}.attn_norm.weight", [6], Norms(salt + 7)));
            tensors.Add(new($"blk.{layer}.ffn_norm.weight", [6], Norms(salt + 8)));
        }

        tensors.Add(new("output_norm.weight", [6], Norms(97)));

        return tensors;
    }

    // Small deterministic payloads on a 1/8 grid: magnitudes stay modest, so every layer's softmax is broad and the
    // engine's forwards are stable enough to compare bit-for-bit between two runs of the same weights.
    private static float[] Weights(int salt, int rows, int cols)
    {
        float[] values = new float[rows * cols];
        for (int i = 0; i < values.Length; i++)
        {
            double raw = Math.Sin((salt * 131 + i * 17) * 0.21);
            values[i] = (float)Math.Round(raw * 8.0) / 8.0f;
        }

        return values;
    }

    private static float[] Norms(int salt)
    {
        float[] values = new float[6];
        for (int i = 0; i < 6; i++)
        {
            double raw = Math.Sin((salt * 43 + i * 29) * 0.17);
            values[i] = 1.0f + ((float)Math.Round(raw * 4.0) / 4.0f) * 0.2f;
        }

        return values;
    }

    private static float[] LoudRow(int length)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = 50_000f + i;
        }

        return values;
    }

    private sealed record TensorDef(string Name, int[] Dims, float[] Payload); // dims in logical (row-major) order; vectors are one-dimensional

    /// <summary>Builds a GGUF v3 image from KV pairs and F32 tensors — the same fixture shape the Layers suites use.</summary>
    private static byte[] BuildGguf(IReadOnlyList<(string Key, object Value)> kvPairs, IReadOnlyList<TensorDef> tensors)
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

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
            foreach (float value in tensor.Payload)
            {
                writer.Write(value);
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

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-windowed-cache-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }
}
