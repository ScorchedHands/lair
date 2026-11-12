using System.Net;
using System.Text;
using System.Text.Json;

using Lair.Host.Config;
using Lair.Http;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;

namespace Lair.Tests;

/// <summary>
/// Advertised sparse-FFN facts on the OpenAI-compatible model surface: a served expert-routed model reports its resolved
/// strategy as one wire field, while a dense block omits that key entirely. Both cases run through the real app builder and
/// its configured snake-case/ignore-null serializer rather than an internal DTO directly.
/// </summary>
[Trait("Category", "Api")]
public sealed class SparseFfnApiTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    [Fact]
    public async Task ModelsEndpoint_ReportsResolvedSparseStrategy_ForRoutedModel_AndOmitsItForDenseModel()
    {
        string routedPath = WriteSyntheticGguf("qwen2moe", sparse: true);
        (bool routedHasField, string? routedValue) = await InspectModelsEntryAsync(routedPath, "sparse-api-routed");

        Assert.True(routedHasField); // the served model routes experts, so its resolved variant has a place on the wire
        Assert.Equal("dense_resident", routedValue);           // absent a file-declared selection, every profile's default answers here

        string densePath = WriteSyntheticGguf("qwen2", sparse: false);
        (bool denseHasField, string? denseValue) = await InspectModelsEntryAsync(densePath, "sparse-api-dense");

        Assert.False(denseHasField); // a model with no expert-routed block carries no strategy value for the endpoint to report
        Assert.Null(denseValue);
    }

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
                // A model still holding its mmap view keeps the file on Windows; the temp directory absorbs it.
            }
        }

        _tempFiles.Clear();
    }

    private static async Task<(bool HasField, string? Value)> InspectModelsEntryAsync(string path, string modelName)
    {
        LairConfig config = new()
        {
            Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
            Engine = new EngineConfig(),
            Models =
            [
                new ModelConfig
                {
                    Name = modelName,
                    Path = path,
                    MaxContextTokens = 512,
                    KvBlocks = 32,
                },
            ],
        };

        GenerationService engine = GenerationService.Open(config);
        WebApplication app = ApiApp.Build(config, engine);
        await ((IHost)app).StartAsync(CancellationToken.None);

        try
        {
            using HttpClient http = new();
            using HttpResponseMessage response = await http.GetAsync(app.Urls.Single() + "/v1/models", CancellationToken.None);
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);

            string bodyText = await response.Content.ReadAsStringAsync(CancellationToken.None);
            using JsonDocument document = JsonDocument.Parse(bodyText);
            JsonElement entry = document.RootElement.GetProperty("data")[0];

            bool hasField = entry.TryGetProperty("sparse_ffn_strategy", out JsonElement field);
            return (hasField, hasField ? field.GetString() : null);
        }
        finally
        {
            await ((IHost)app).StopAsync(TimeSpan.FromSeconds(5));
            await app.DisposeAsync();
            engine.Dispose();
        }
    }

    // A synthetic model the API can serve end to end: same tiny BPE vocabulary and geometry the existing HTTP fixtures use, but
    // with either qwen2's dense feed-forward triplet or qwen2moe's routed expert block. The GGUF metadata is complete enough for
    // GenerationService.Open (tokenizer, forward engine, KV pool, scheduler) without a chat template; tool parsing stays irrelevant here.
    private string WriteSyntheticGguf(string architecture, bool sparse)
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

        List<(string Key, object Value)> metadata =
        [
            ("general.architecture", architecture),
            ("general.name", $"lair-sparse-api-{architecture}"),
            ($"{architecture}.embedding_length", 4),
            ($"{architecture}.block_count", 1),
            ($"{architecture}.attention.head_count", 2),
            ($"{architecture}.attention.head_count_kv", 2),
            ($"{architecture}.feed_forward_length", 8),
            ($"{architecture}.rope.freq_base", 10_000f),
            ($"{architecture}.attention.layer_norm_rms_epsilon", 1e-5f),
            ($"{architecture}.vocab_size", 6),
        ];

        if (sparse)
        {
            metadata.Add(($"{architecture}.expert_count", 8));
            metadata.Add(($"{architecture}.expert_used_count", 2));
            metadata.Add(($"{architecture}.expert_shared_count", 1));
            metadata.Add(($"{architecture}.expert_weights_scale", 1f));
            metadata.Add(($"{architecture}.expert_weights_norm", true));
        }

        string[] tokens = ["<unk>", "hello", "world", "!", "<s>", "</s>"];
        int[] tokenTypes = [2, 1, 1, 1, 4, 5];

        metadata.Add(("tokenizer.ggml.model", "gpt2"));
        metadata.Add(("tokenizer.ggml.tokens", tokens));
        metadata.Add(("tokenizer.ggml.token_type", tokenTypes));
        metadata.Add(("tokenizer.ggml.merges", Array.Empty<string>()));
        metadata.Add(("tokenizer.ggml.bos_token_id", 4));
        metadata.Add(("tokenizer.ggml.eos_token_id", 5));
        metadata.Add(("tokenizer.ggml.unknown_token_id", 0));
        metadata.Add(("tokenizer.ggml.add_bos_token", false));

        List<(string Name, int[] Dims)> tensors =
        [
            new("token_embd.weight", [6, 4]),
            new("blk.0.attn_q.weight", [4, 4]),
            new("blk.0.attn_k.weight", [4, 4]), // kv-heads * head-dim == hidden here: heads 2 over a head dim of 2
            new("blk.0.attn_v.weight", [4, 4]),
            new("blk.0.attn_output.weight", [4, 4]),
        ];

        if (sparse)
        {
            tensors.Add(new("blk.0.ffn_gate_inp.weight", [8, 4]));       // the router matrix over hidden and expert count
            tensors.Add(new("blk.0.ffn_gate_exps.weight", [8, 8, 4]));    // stacked per-expert gate/up/down triplets: intermediate, hidden, experts
            tensors.Add(new("blk.0.ffn_up_exps.weight", [8, 8, 4]));      // all eight declared experts ship resident here, exactly as the forward engine loads them today
            tensors.Add(new("blk.0.ffn_down_exps.weight", [8, 4, 8]));    // hidden, intermediate, experts for down projection

            tensors.Add(new("blk.0.ffn_gate_shexp.weight", [8, 4]));      // qwen2moe's shared expert dense triplet plus its per-token selector vector
            tensors.Add(new("blk.0.ffn_up_shexp.weight", [8, 4]));        // the descriptor names these roles only because metadata declares a shared count above
            tensors.Add(new("blk.0.ffn_down_shexp.weight", [4, 8]));      // no dense gate/up/down block exists for an expert-routed model to fall back through
            tensors.Add(new("blk.0.ffn_gate_inp_shexp.weight", [4]));     // the shared-expert selector vector is hidden-sized and one-dimensional
        }
        else
        {
            tensors.Add(new("blk.0.ffn_gate.weight", [8, 4]));
            tensors.Add(new("blk.0.ffn_up.weight", [8, 4]));
            tensors.Add(new("blk.0.ffn_down.weight", [4, 8]));
        }

        tensors.Add(new("blk.0.attn_norm.weight", [4]));
        tensors.Add(new("blk.0.ffn_norm.weight", [4]));
        tensors.Add(new("output_norm.weight", [4])); // no output.weight: the tied token-embedding path is exercised instead

        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u);          // version
        writer.Write((ulong)tensors.Count);
        writer.Write((ulong)metadata.Count);

        foreach ((string key, object value) in metadata)
        {
            WriteKv(writer, key, value);
        }

        long offset = 0;
        foreach ((string name, int[] dims) in tensors)
        {
            WriteString(writer, name);
            writer.Write((uint)dims.Length);
            for (int d = dims.Length - 1; d >= 0; d--) // GGUF stores dimensions fastest-varying first
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
        foreach ((string name, int[] dims) in tensors)
        {
            long elements = 1;
            foreach (int dim in dims)
            {
                elements *= dim;
            }

            bool isNorm = name.Contains("norm", StringComparison.Ordinal);
            for (long e = 0; e < elements; e++)
            {
                float value = MathF.Sin((ordinal * 17 + e) * 0.13f); // deterministic, finite values everywhere
                writer.Write(isNorm ? 1.25f + value : value);       // norm weights stay clearly nonzero for RMS normalization
            }

            ordinal++;
        }

        writer.Flush();

        string path = Path.Combine(Path.GetTempPath(), $"lair-sparse-api-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, stream.ToArray());
        _tempFiles.Add(path);
        return path;
    }

    private static void WriteKv(BinaryWriter writer, string key, object value)
    {
        WriteString(writer, key);

        switch (value)
        {
            case string s:
                writer.Write(8u); // STRING
                WriteString(writer, s);
                break;
            case int i:
                writer.Write(5u); // INT32
                writer.Write(i);
                break;
            case float f:
                writer.Write(6u); // FLOAT32
                writer.Write(f);
                break;
            case bool b:
                writer.Write(7u); // BOOL
                writer.Write((byte)(b ? 1 : 0));
                break;
            case IReadOnlyList<string> strings:
                writer.Write(9u); // ARRAY
                writer.Write(8u); // element type STRING
                writer.Write((ulong)strings.Count);
                foreach (string item in strings)
                {
                    WriteString(writer, item);
                }

                break;
            case IReadOnlyList<int> ints:
                writer.Write(9u); // ARRAY
                writer.Write(5u); // element type INT32
                writer.Write((ulong)ints.Count);
                foreach (int item in ints)
                {
                    writer.Write(item);
                }

                break;
            default:
                throw new ArgumentException($"Unsupported fixture value type {value.GetType().Name}.", nameof(value));
        }
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }
}
