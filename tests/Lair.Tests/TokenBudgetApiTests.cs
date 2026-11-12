using System.Net;
using System.Text;
using System.Text.Json;
using Lair.Host.Config;
using Lair.Http;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;

namespace Lair.Tests;

/// <summary>
/// The request-side token-budget surface on synthetic tiny models: <c>max_reasoning_tokens</c> and
/// <c>max_response_tokens</c> arrive as documented extension fields on both chat and completion bodies, an override above the
/// ceiling the loaded model declares clamps there with the clamp reported on the same body, a reasoning override on a model whose
/// plan resolves no reasoning region is refused naming that model, a value below 1 fails naming the field like every other sized
/// knob, and a request carrying neither field keeps today's response shape — no extension key at all. Every case runs through the
/// real app builder and its snake-case/ignore-null serializer rather than an internal DTO.
/// </summary>
[Trait("Category", "Api")]
public sealed class TokenBudgetApiTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    [Fact]
    public async Task ChatOverride_AboveDeclaredCeilings_ClampsEachSideAndReportsBothClamps()
    {
        await using ServedModel served = await OpenAsync("budget-chat-model", maxReasoningTokens: 64, maxResponseTokens: 32);

        JsonElement body = await PostJsonAsync(served, "/v1/chat/completions", ChatJson(
            "budget-chat-model",
            ("max_tokens", "4"),
            ("temperature", "0"),
            ("max_reasoning_tokens", "128"),
            ("max_response_tokens", "64")));

        Assert.False(string.IsNullOrEmpty(body.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()));

        JsonElement reports = body.GetProperty("token_budgets");
        Assert.Equal(2, reports.GetArrayLength());

        JsonElement reasoning = reports[0];
        Assert.Equal("max_reasoning_tokens", reasoning.GetProperty("field").GetString());
        Assert.Equal(128, reasoning.GetProperty("requested").GetInt32());
        Assert.Equal(64, reasoning.GetProperty("applied").GetInt32());     // the model's declared ceiling is what stands
        Assert.True(reasoning.GetProperty("clamped").GetBoolean());

        JsonElement response = reports[1];
        Assert.Equal("max_response_tokens", response.GetProperty("field").GetString());
        Assert.Equal(64, response.GetProperty("requested").GetInt32());
        Assert.Equal(32, response.GetProperty("applied").GetInt32());
        Assert.True(response.GetProperty("clamped").GetBoolean());
    }

    [Fact]
    public async Task CompletionOverride_AboveDeclaredCeiling_ClampsAndReportsTheClamp()
    {
        await using ServedModel served = await OpenAsync("budget-completion-model", maxReasoningTokens: 64, maxResponseTokens: null);

        JsonElement body = await PostJsonAsync(served, "/v1/completions", CompletionJson(
            "budget-completion-model",
            ("prompt", "\"hello world\""),
            ("max_tokens", "4"),
            ("temperature", "0"),
            ("max_reasoning_tokens", "256")));

        Assert.False(string.IsNullOrEmpty(body.GetProperty("choices")[0].GetProperty("text").GetString()));

        JsonElement reasoning = Assert.Single(body.GetProperty("token_budgets").EnumerateArray());
        Assert.Equal("max_reasoning_tokens", reasoning.GetProperty("field").GetString());
        Assert.Equal(256, reasoning.GetProperty("requested").GetInt32());
        Assert.Equal(64, reasoning.GetProperty("applied").GetInt32());
        Assert.True(reasoning.GetProperty("clamped").GetBoolean());
    }

    [Fact]
    public async Task OverrideAtOrBelowCeiling_IsReportedUnclamped_AndAnOverrideOnASideWithNoCeilingIsAcceptedAsGiven()
    {
        await using ServedModel served = await OpenAsync("budget-inrange-model", maxReasoningTokens: 64, maxResponseTokens: null);

        JsonElement inRange = await PostJsonAsync(served, "/v1/chat/completions", ChatJson(
            "budget-inrange-model",
            ("max_tokens", "2"),
            ("temperature", "0"),
            ("max_reasoning_tokens", "64")));

        JsonElement reasoning = Assert.Single(inRange.GetProperty("token_budgets").EnumerateArray());
        Assert.Equal(64, reasoning.GetProperty("requested").GetInt32());
        Assert.Equal(64, reasoning.GetProperty("applied").GetInt32());
        Assert.False(reasoning.GetProperty("clamped").GetBoolean()); // an override the ceiling never lowered still reports itself

        JsonElement noCeiling = await PostJsonAsync(served, "/v1/completions", CompletionJson(
            "budget-inrange-model",
            ("prompt", "\"hello world\""),
            ("max_tokens", "2"),
            ("temperature", "0"),
            ("max_response_tokens", "96")));

        JsonElement response = Assert.Single(noCeiling.GetProperty("token_budgets").EnumerateArray());
        Assert.Equal("max_response_tokens", response.GetProperty("field").GetString());
        Assert.Equal(96, response.GetProperty("applied").GetInt32()); // the model declared no ceiling for this side: opt-in budgeting, not a silent default
        Assert.False(response.GetProperty("clamped").GetBoolean());
    }

    [Fact]
    public async Task AbsentOverrides_KeepTodayResponseShape_NoBudgetKeyOnEitherSurface()
    {
        await using ServedModel served = await OpenAsync("budget-absent-model", maxReasoningTokens: 64, maxResponseTokens: 32);

        JsonElement chat = await PostJsonAsync(served, "/v1/chat/completions", ChatJson(
            "budget-absent-model",
            ("max_tokens", "2"),
            ("temperature", "0")));
        Assert.False(chat.TryGetProperty("token_budgets", out _)); // today's body exactly: the extension is absent when nothing was asked for

        JsonElement completion = await PostJsonAsync(served, "/v1/completions", CompletionJson(
            "budget-absent-model",
            ("prompt", "\"hello world\""),
            ("max_tokens", "2"),
            ("temperature", "0")));
        Assert.False(completion.TryGetProperty("token_budgets", out _));

        // The token accounting a plain request already reported stays what the model produced, unchanged by the new fields existing.
        int chatTokens = chat.GetProperty("usage").GetProperty("completion_tokens").GetInt32();
        Assert.True(chatTokens is > 0 and <= 2);
    }

    [Fact]
    public async Task ReasoningOverride_ModelWithoutResolvedRegion_Returns400NamingTheModel()
    {
        await using ServedModel served = await OpenAsync("budget-unresolved-model", maxReasoningTokens: null, maxResponseTokens: 32);

        (HttpResponseMessage response, JsonElement body) = await PostExpectingAsync(served, "/v1/chat/completions", ChatJson(
            "budget-unresolved-model",
            ("max_tokens", "4"),
            ("temperature", "0"),
            ("max_reasoning_tokens", "64")));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.True(response.Headers.TryGetValues(ApiApp.RequestIdHeader, out _));
        Assert.Equal("invalid_request", body.GetProperty("error").GetProperty("code").GetString());
        Assert.Equal("invalid_request_error", body.GetProperty("error").GetProperty("type").GetString());

        string message = body.GetProperty("error").GetProperty("message").GetString()!;
        Assert.Contains("budget-unresolved-model", message); // the model whose load resolved no region is named, not a guessed one
        Assert.Contains("max_reasoning_tokens", message);
    }

    [Fact]
    public async Task ResponseOverride_ModelWithoutResolvedRegion_StillAppliesAsPlainTokenAccounting()
    {
        await using ServedModel served = await OpenAsync("budget-response-only-model", maxReasoningTokens: null, maxResponseTokens: 48);

        JsonElement body = await PostJsonAsync(served, "/v1/completions", CompletionJson(
            "budget-response-only-model",
            ("prompt", "\"hello world\""),
            ("max_tokens", "2"),
            ("temperature", "0"),
            ("max_response_tokens", "64")));

        JsonElement report = Assert.Single(body.GetProperty("token_budgets").EnumerateArray());
        Assert.Equal("max_response_tokens", report.GetProperty("field").GetString());
        Assert.Equal(48, report.GetProperty("applied").GetInt32());
        Assert.True(report.GetProperty("clamped").GetBoolean()); // a response ceiling needs no reasoning region to bound against
    }

    [Theory]
    [InlineData("max_reasoning_tokens", "0")]
    [InlineData("max_reasoning_tokens", "-1")]
    [InlineData("max_response_tokens", "0")]
    [InlineData("max_response_tokens", "-7")]
    public async Task BudgetOverrideBelowOne_Returns400NamingFieldAndValue(string field, string value)
    {
        await using ServedModel served = await OpenAsync("budget-invalid-model", maxReasoningTokens: 64, maxResponseTokens: 32);

        (HttpResponseMessage response, JsonElement body) = await PostExpectingAsync(served, "/v1/completions", CompletionJson(
            "budget-invalid-model",
            ("prompt", "\"hello world\""),
            ("max_tokens", "2"),
            ("temperature", "0"),
            (field, value)));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal("invalid_request", body.GetProperty("error").GetProperty("code").GetString());
        string message = body.GetProperty("error").GetProperty("message").GetString()!;
        Assert.Contains(field, message);
        Assert.Contains($"'{field}' must be 1 or greater", message); // the same wording the config loader uses for these keys
    }

    [Fact]
    public async Task StreamedOverrideAboveCeiling_ReportsTheClampOnTheClosingChunkOnly()
    {
        await using ServedModel served = await OpenAsync("budget-stream-model", maxReasoningTokens: 64, maxResponseTokens: null);

        string requestBody = ChatJson(
            "budget-stream-model",
            ("max_tokens", "4"),
            ("temperature", "0"),
            ("stream", "true"),
            ("max_reasoning_tokens", "128"));

        using HttpRequestMessage request = new(HttpMethod.Post, served.Url + "/v1/chat/completions")
        {
            Content = new StringContent(requestBody, Encoding.UTF8, "application/json"),
        };

        using HttpResponseMessage response = await served.Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);

        List<JsonElement> chunks = [];
        using (Stream stream = await response.Content.ReadAsStreamAsync())
        using (StreamReader reader = new(stream))
        {
            while (true)
            {
                string? line = await reader.ReadLineAsync();
                if (line is null || line.Length == 0)
                {
                    continue;
                }

                Assert.StartsWith("data: ", line);
                if (line == "data: [DONE]")
                {
                    break;
                }

                chunks.Add(JsonDocument.Parse(line["data: ".Length..]).RootElement);
            }
        }

        JsonElement closing = chunks[^1];
        Assert.True(closing.GetProperty("choices")[0].TryGetProperty("finish_reason", out _));

        JsonElement reasoning = Assert.Single(closing.GetProperty("token_budgets").EnumerateArray());
        Assert.Equal(128, reasoning.GetProperty("requested").GetInt32());
        Assert.Equal(64, reasoning.GetProperty("applied").GetInt32());
        Assert.True(reasoning.GetProperty("clamped").GetBoolean());

        // The delta frames stay exactly as they were: the report rides once, on the frame that closes the stream.
        foreach (JsonElement chunk in chunks[..^1])
        {
            Assert.False(chunk.TryGetProperty("token_budgets", out _));
        }
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

    // ---------------------------------------------------------------- request plumbing

    private static string ChatJson(string model, params (string Key, string RawJson)[] fields)
    {
        List<(string Key, string RawJson)> all = [("model", Str(model)), ("messages", """[{"role":"user","content":"hello"}]""")];
        all.AddRange(fields);
        return JsonBody(all);
    }

    private static string CompletionJson(string model, params (string Key, string RawJson)[] fields)
    {
        List<(string Key, string RawJson)> all = [("model", Str(model))];
        all.AddRange(fields);
        return JsonBody(all);
    }

    // Builds a request body from (property name, raw JSON value) pairs so the cases stay readable.
    private static string JsonBody(List<(string Key, string RawJson)> fields)
    {
        using MemoryStream buffer = new();
        using (Utf8JsonWriter writer = new(buffer))
        {
            writer.WriteStartObject();
            foreach ((string key, string raw) in fields)
            {
                writer.WritePropertyName(key);
                writer.WriteRawValue(raw);
            }

            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.ToArray());
    }

    private static string Str(string value) => "\"" + value + "\"";

    private async Task<JsonElement> PostJsonAsync(ServedModel served, string path, string jsonBody)
    {
        (HttpResponseMessage response, JsonElement body) = await PostExpectingAsync(served, path, jsonBody);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        return body;
    }

    private static async Task<(HttpResponseMessage, JsonElement)> PostExpectingAsync(ServedModel served, string path, string jsonBody)
    {
        using StringContent content = new(jsonBody, Encoding.UTF8, "application/json");
        HttpResponseMessage response = await served.Http.PostAsync(served.Url + path, content);
        return (response, JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement);
    }

    // ---------------------------------------------------------------- served synthetic model

    private sealed class ServedModel : IAsyncDisposable
    {
        private readonly WebApplication _app;
        private readonly GenerationService _engine;

        public ServedModel(WebApplication app, GenerationService engine)
        {
            _app = app;
            _engine = engine;
            Http = new HttpClient();
            Url = app.Urls.Single();
        }

        public HttpClient Http { get; }

        public string Url { get; }

        public async ValueTask DisposeAsync()
        {
            await ((IHost)_app).StopAsync(TimeSpan.FromSeconds(5));
            await _app.DisposeAsync();
            _engine.Dispose();
            Http.Dispose();
        }
    }

    private async Task<ServedModel> OpenAsync(string modelName, int? maxReasoningTokens, int? maxResponseTokens)
    {
        string path = WriteSyntheticGguf();
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
                    MaxReasoningTokens = maxReasoningTokens,
                    MaxResponseTokens = maxResponseTokens,
                },
            ],
        };

        GenerationService engine = GenerationService.Open(config);
        WebApplication app = ApiApp.Build(config, engine);
        await ((IHost)app).StartAsync(CancellationToken.None);
        return new ServedModel(app, engine);
    }

    // The same tiny llama-shaped model the parser tests serve — a BPE vocabulary carrying <think> and </think> as ordinary entries, two layers, a ChatML template — so both
    // chat and completion requests reach the real request pipeline without loading a fixture model, and a reasoning ceiling declared on it has whole-marker tokens to bind.
    private const string ChatMlTemplate =
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

    private string WriteSyntheticGguf()
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

        string[] tokens = ["<unk>", "hello", "world", "!", "<s>", "</s>", "<think>", "</think>"];
        int[] tokenTypes = [2, 1, 1, 1, 4, 5, 1, 1];

        List<(string Name, int[] Dims)> tensors = [new("token_embd.weight", [8, 4])];
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
        WriteStringKv(writer, "general.name", "lair-budget-fixture");
        WriteInt32Kv(writer, "llama.embedding_length", 4);
        WriteInt32Kv(writer, "llama.block_count", 2);
        WriteInt32Kv(writer, "llama.attention.head_count", 2);
        WriteInt32Kv(writer, "llama.attention.head_count_kv", 1);
        WriteInt32Kv(writer, "llama.feed_forward_length", 8);
        WriteFloat32Kv(writer, "llama.rope.freq_base", 10_000f);
        WriteFloat32Kv(writer, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        WriteInt32Kv(writer, "llama.vocab_size", 8);
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

        string path = Path.Combine(Path.GetTempPath(), $"lair-budget-tests-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, stream.ToArray());
        _tempFiles.Add(path);
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
