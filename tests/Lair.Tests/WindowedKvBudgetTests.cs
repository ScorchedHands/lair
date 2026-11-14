using System.Diagnostics;
using System.Text;
using System.Text.Json;

using Lair.Backends.Cpu;
using Lair.Http.Monitoring;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// Windowed KV budget accounting: the same model geometry with a declared window reports a smaller read footprint than
/// the same geometry without one, and an engine snapshot carries those numbers in its monitor JSON. Storage stays whole-KV;
/// these fields describe what attention selects under the current soft budget, not reclaimed physical blocks.
/// </summary>
[Trait("Category", "Cache")]
[Trait("Category", "Metrics")]
public sealed class WindowedKvBudgetTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

    private const int Window = 3;
    private const int BlockTokens = 2;
    private const int CapacityBlocks = 4;
    private const int Layers = 4;

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
    public void BlockPoolReadFootprint_DiffersForWindowedAndUnwindowedPlans_AtSameGeometry()
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);

        long windowedReadBlocks = 0L;
        long windowedSkippedBlocks = 0L;
        int windowedLayers = 0;
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildImage(Window)));
            using TransformerModel model = OpenFixtureModel(gguf, backend);

            Assert.Equal(WindowInterleave.AlternatingLocalFirst, model.Plan.WindowInterleave);
            Assert.Equal(Window, model.Plan.AttentionWindow);
            Assert.Equal(Layers, model.Plan.LayerCount);

            using BlockPool pool = NewDeclaredPool(model, backend);
            windowedReadBlocks = pool.KvReadFootprintBlocksAtBudget;
            windowedSkippedBlocks = pool.SkippedKvReadFootprintBlocksAtBudget;
            windowedLayers = pool.DeclaredWindowedLayerCount;
        }

        long unboundedReadBlocks = 0L;
        long unboundedSkippedBlocks = 0L;
        int unboundedLayers = 0;
        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildImage(null)));
            using TransformerModel model = OpenFixtureModel(gguf, backend);

            Assert.Equal(WindowInterleave.AlternatingLocalFirst, model.Plan.WindowInterleave);
            Assert.Equal(0, model.Plan.AttentionWindow);
            Assert.Equal(Layers, model.Plan.LayerCount);

            using BlockPool pool = NewDeclaredPool(model, backend);
            unboundedReadBlocks = pool.KvReadFootprintBlocksAtBudget;
            unboundedSkippedBlocks = pool.SkippedKvReadFootprintBlocksAtBudget;
            unboundedLayers = pool.DeclaredWindowedLayerCount;
        }

        Assert.Equal(2, windowedLayers); // even-indexed layers only under the alternating pattern
        Assert.Equal(0, unboundedLayers);

        Assert.Equal(12L, windowedReadBlocks); // two masked layers select ceil(3 / 2) = 2 blocks; full-attention layers keep all four
        Assert.Equal(4L, windowedSkippedBlocks);

        Assert.Equal(16L, unboundedReadBlocks);
        Assert.Equal(0L, unboundedSkippedBlocks);

        Assert.NotEqual(unboundedReadBlocks, windowedReadBlocks);
        Assert.True(windowedReadBlocks < unboundedReadBlocks);
        Assert.True(windowedSkippedBlocks > unboundedSkippedBlocks);
    }

    [Fact]
    public async Task EngineSnapshot_CarriesWindowedKvFootprintNumbers_InMonitorJson()
    {
        SnapshotFacts windowed = await RunEngineHarnessAsync(Window);
        SnapshotFacts unbounded = await RunEngineHarnessAsync(null);

        Assert.NotEqual(unbounded.KvReadFootprintBlocks, windowed.KvReadFootprintBlocks);
        Assert.True(windowed.KvReadFootprintBlocks < unbounded.KvReadFootprintBlocks);
        Assert.Equal(4L, windowed.SkippedKvReadBlocks);
        Assert.Equal(0L, unbounded.SkippedKvReadBlocks);

        using JsonDocument document = JsonDocument.Parse(windowed.Json);
        JsonElement root = document.RootElement;

        foreach (JsonProperty property in root.EnumerateObject())
        {
            Assert.StartsWith("metrics_", property.Name);
        }

        Assert.True(root.TryGetProperty("metrics_kv_windowed_layers", out JsonElement windowedLayerProperty));
        Assert.Equal(2, windowedLayerProperty.GetInt32());

        Assert.True(root.TryGetProperty("metrics_kv_read_footprint_blocks", out JsonElement readFootprintProperty));
        Assert.Equal(windowed.KvReadFootprintBlocks, readFootprintProperty.GetInt64());

        Assert.True(root.TryGetProperty("metrics_kv_skipped_read_blocks", out JsonElement skippedFootprintProperty));
        Assert.Equal(windowed.SkippedKvReadBlocks, skippedFootprintProperty.GetInt64());
    }

    private static TransformerModel OpenFixtureModel(GgufFile gguf, CpuBackend backend) =>
        TransformerModel.Open(
            gguf,
            backend,
            maxContextTokens: 8,
            ModelProfiles.Resolve("llama") with { WindowInterleave = WindowInterleave.AlternatingLocalFirst });

    private static BlockPool NewDeclaredPool(TransformerModel model, CpuBackend backend)
    {
        BlockPool pool = new(
            blockTokens: BlockTokens,
            layerCount: model.Plan.LayerCount,
            kvHeads: model.Plan.KvHeadCount,
            headDim: model.Plan.HeadDim,
            capacityBlocks: CapacityBlocks,
            backend);

        int[] attentionWindows = new int[model.Plan.LayerCount];
        for (int layer = 0; layer < model.Plan.LayerCount; layer++)
        {
            attentionWindows[layer] = model.Plan.AttentionWindowForLayer(layer);
        }

        pool.DeclareAttentionWindows(attentionWindows);
        return pool;
    }

    private async Task<SnapshotFacts> RunEngineHarnessAsync(int? window)
    {
        CpuBackend? backend = null;
        GgufFile? gguf = null;
        TransformerModel? model = null;
        BlockPool? pool = null;
        GenerationScheduler? scheduler = null;
        EngineMetricsBus? bus = null;

        try
        {
            backend = new CpuBackend(threads: 1, allowAvx2: true);
            string path = WriteTemp(BuildImage(window));
            gguf = GgufFile.Open(path);
            model = OpenFixtureModel(gguf, backend);
            pool = NewDeclaredPool(model, backend);
            scheduler = new GenerationScheduler(model, pool, maxBatchTokens: 8);
            bus = new EngineMetricsBus("windowed-kv-budget", scheduler, pool, cache: null, NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(30));

            PipelineHandle handle = scheduler.Submit(new ScheduledRequest
            {
                PromptTokens = [1, 4],
                MaxTokens = 2,
                Sampling = new SamplingParams { Temperature = 0f },
                EosId = 0,
            }, CancellationToken.None);

            _ = await handle.Completion;

            Stopwatch clock = Stopwatch.StartNew();
            while (bus.RingCount == 0 && clock.Elapsed < TimeSpan.FromSeconds(15))
            {
                await Task.Delay(25);
            }

            Assert.True(bus.RingCount > 0, "the metrics bus published no snapshot after a windowed-KV load");

            MetricsSnapshot snapshot = bus.SnapshotRing()[0];
            return new SnapshotFacts(
                snapshot.ToJson(),
                snapshot.KvReadFootprintBlocks,
                snapshot.SkippedKvReadBlocks);
        }
        finally
        {
            if (bus is not null)
            {
                await bus.DisposeAsync();
            }

            scheduler?.Dispose();
            pool?.Dispose();
            model?.Dispose();
            gguf?.Dispose();
            backend?.Dispose();
        }
    }

    private static byte[] BuildImage(int? window) => BuildGguf(Metadata(window), Tensors());

    private static List<(string Key, object Value)> Metadata(int? window)
    {
        List<(string Key, object Value)> pairs =
        [
            ("general.architecture", "llama"),
            ("llama.embedding_length", 6),
            ("llama.block_count", Layers),
            ("llama.attention.head_count", 2),
            ("llama.attention.head_count_kv", 1),
            ("llama.feed_forward_length", 8),
            ("llama.attention.head_length", 4),
            ("llama.attention.layer_norm_rms_epsilon", 1e-5f),
            ("llama.vocab_size", 6),
        ];

        if (window.HasValue)
        {
            pairs.Add(("llama.attention.sliding_window", window.Value));
        }

        return pairs;
    }

    private static List<TensorDef> Tensors()
    {
        List<TensorDef> tensors = [new("token_embd.weight", [6, 6], Weights(7, 6, 6))];

        for (int layer = 0; layer < Layers; layer++)
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
        string path = Path.Combine(Path.GetTempPath(), $"lair-windowed-budget-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    private sealed record SnapshotFacts(string Json, long KvReadFootprintBlocks, long SkippedKvReadBlocks);
}
