using System.Net;
using System.Text;
using System.Text.Json;
using Lair.Http;
using Lair.Host.Config;
using Lair.Templating.Minja;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Lair.Tests;

/// <summary>
/// The resolved output-parser plan end to end on synthetic tiny models: the load-time choice for every
/// <c>tool_parser</c> value, <c>auto</c> reproducing today's capability precedence byte-for-byte for a template that
/// renders tools itself versus one that does not, <c>fenced</c> forcing the injected block even for a tool-capable
/// template, <c>none</c> closing the pipeline so content stays byte-identical with no <c>tool_calls</c>, the load log
/// naming the resolved parser, <c>/v1/models</c> reporting it, and the key applying as a new-requests-only hot edit.
/// The reasoning-region slot every plan carries stays inert.
/// </summary>
[Trait("Category", "Api")]
public sealed class ToolParserTests : IDisposable
{
    private static readonly ToolDefinition WeatherTool = new(
        "get_weather",
        "Fetch current weather for a city.",
        """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""");

    private readonly List<string> _tempFiles = [];

    // ---------------------------------------------------------------- auto precedence (unchanged behavior)

    [Fact]
    public void AutoToolCapableTemplate_ResolvesTemplate_AndRendersToolsItself_ByteIdenticalToToday()
    {
        using GenerationService engine = OpenSynthetic(ChatMlToolsTemplate, "auto-capable-model", new CapturingLogger());

        Assert.Equal("template", engine.Model.ToolParser);
        Assert.Equal(ToolParser.Template, engine.ResolvedOutputParser.ToolParser);
        Assert.NotNull(engine.ResolvedOutputParser.Parser); // the fenced detector stays wired for a template-rendered model
        Assert.Null(engine.ResolvedOutputParser.Reasoning); // the reasoning slot is inert for every plan resolved today

        GenerationRequest request = ToolRequest("auto-capable-model");
        string prompt = engine.BuildChatPrompt(request);

        // Today's path exactly: the template's own tools rendering is the whole tool surface, byte-for-byte.
        MinjaChatTemplate template = new(ChatMlToolsTemplate);
        string rendered = template.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "hello")],
            AddGenerationPrompt = true,
            Tools = [TemplateToolValues.Of(WeatherTool)],
        }).Prompt;
        Assert.Equal(rendered, prompt);

        Assert.Contains("[tool]", prompt);
        Assert.DoesNotContain("You can call these tools:", prompt);
        Assert.DoesNotContain("```json", prompt);
        Assert.DoesNotContain(ToolPromptComposer.ContractLine, prompt);
    }

    [Fact]
    public void AutoToollessTemplate_ResolvesFenced_AndInjectsByteIdenticalToToday()
    {
        using GenerationService engine = OpenSynthetic(ChatMlTemplate, "auto-plain-model", new CapturingLogger());

        Assert.Equal("fenced", engine.Model.ToolParser);
        Assert.Equal(ToolParser.Fenced, engine.ResolvedOutputParser.ToolParser);

        GenerationRequest request = ToolRequest("auto-plain-model");
        string prompt = engine.BuildChatPrompt(request);

        MinjaChatTemplate template = new(ChatMlTemplate);
        string rendered = template.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "hello")],
            AddGenerationPrompt = true,
        }).Prompt;

        // The same composed block today's precedence appends — nothing moved.
        Assert.Equal(ToolPromptComposer.Compose(rendered, [WeatherTool]), prompt);
    }

    [Fact]
    public void AutoModelWithoutChatTemplate_ResolvesFenced()
    {
        string path = WriteSyntheticGguf(chatTemplate: null);
        using GenerationService engine = GenerationService.Open(ConfigFor(path, "no-template-model", toolParser: null), new CapturingLogger());

        Assert.Null(engine.ChatTemplateCapabilities);
        Assert.Equal("fenced", engine.Model.ToolParser);
    }

    // ---------------------------------------------------------------- explicit choices

    [Fact]
    public void FencedChoice_ForcesInjection_EvenForToolCapableTemplate()
    {
        using GenerationService engine = OpenSynthetic(ChatMlToolsTemplate, "fenced-choice-model", new CapturingLogger(), toolParser: "fenced");

        Assert.True(engine.ChatTemplateCapabilities!.SupportsTools);
        Assert.Equal("fenced", engine.Model.ToolParser);

        string prompt = engine.BuildChatPrompt(ToolRequest("fenced-choice-model"));

        Assert.Contains("\nYou can call these tools:\n", prompt);
        Assert.Contains("\"name\":\"get_weather\"", prompt);
        Assert.True(prompt.EndsWith(ToolPromptComposer.ContractLine), "the reply contract line must close the injected block");
        Assert.DoesNotContain("[tool]", prompt); // the template's own tools loop did not run
    }

    [Fact]
    public void TemplateChoice_SelectsTemplateRendering_EvenForToollessTemplate()
    {
        using GenerationService engine = OpenSynthetic(ChatMlTemplate, "template-choice-model", new CapturingLogger(), toolParser: "template");

        Assert.False(engine.ChatTemplateCapabilities!.SupportsTools);
        Assert.Equal("template", engine.Model.ToolParser);

        string prompt = engine.BuildChatPrompt(ToolRequest("template-choice-model"));

        // The declared tools enter the template context; LAIR injects nothing, fenced or otherwise.
        MinjaChatTemplate template = new(ChatMlTemplate);
        string rendered = template.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "hello")],
            AddGenerationPrompt = true,
            Tools = [TemplateToolValues.Of(WeatherTool)],
        }).Prompt;
        Assert.Equal(rendered, prompt);

        Assert.DoesNotContain("You can call these tools:", prompt);
        Assert.DoesNotContain("```json", prompt);
    }

    // ---------------------------------------------------------------- inert choice

    [Fact]
    public async Task NoneChoice_ClosesPipeline_ContentByteIdenticalWithNoToolCalls()
    {
        using GenerationService engine = OpenSynthetic(ChatMlToolsTemplate, "none-choice-model", new CapturingLogger(), toolParser: "none");

        Assert.Equal("none", engine.Model.ToolParser);
        Assert.Equal(ToolParser.None, engine.ResolvedOutputParser.ToolParser);
        Assert.Null(engine.ResolvedOutputParser.Parser); // no parser instance: the decoded text is never read

        GenerationRequest withTools = ToolRequest("none-choice-model");
        GenerationRequest withoutTools = withTools with { Tools = null };

        // The prompt is byte-identical to the same request with no tools at all: no template tools, no fenced block.
        Assert.Equal(engine.BuildChatPrompt(withoutTools), engine.BuildChatPrompt(withTools));

        // Greedy runs on the identical prompt produce identical bytes, and neither carries tool_calls.
        GenerationResult toolsRun = await engine.GenerateAsync(withTools, CancellationToken.None);
        GenerationResult plainRun = await engine.GenerateAsync(withoutTools, CancellationToken.None);
        Assert.Equal(plainRun.Text, toolsRun.Text);
        Assert.Equal(plainRun.PromptTokens, toolsRun.PromptTokens);
        Assert.Null(toolsRun.ToolCalls);
        Assert.NotEqual("tool_calls", toolsRun.FinishReason);
    }

    [Fact]
    public async Task NoneChoice_WireShape_HasNoToolCalls_AndMatchesToollessRequestBytes()
    {
        const string messages = """[{"role":"user","content":"hello"}]""";
        const string tools = """[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]""";

        string path = WriteSyntheticGguf(ChatMlToolsTemplate);
        LairConfig config = ConfigFor(path, "none-wire-model", toolParser: "none");
        using GenerationService engine = GenerationService.Open(config, new CapturingLogger());
        using WebApplication app = ApiApp.Build(config, engine);
        await ((IHost)app).StartAsync(CancellationToken.None);

        try
        {
            using HttpClient http = new();

            using StringContent toolsBody = new(
                JsonBody(
                    ("model", Str("none-wire-model")),
                    ("messages", messages),
                    ("max_tokens", "4"),
                    ("temperature", "0"),
                    ("tools", tools)),
                Encoding.UTF8,
                "application/json");
            using HttpResponseMessage toolsResponse = await http.PostAsync(app.Urls.Single() + "/v1/chat/completions", toolsBody, CancellationToken.None);
            Assert.Equal(HttpStatusCode.OK, toolsResponse.StatusCode);
            JsonElement toolsJson = JsonDocument.Parse(await toolsResponse.Content.ReadAsStringAsync(CancellationToken.None)).RootElement;

            using StringContent plainBody = new(
                JsonBody(
                    ("model", Str("none-wire-model")),
                    ("messages", messages),
                    ("max_tokens", "4"),
                    ("temperature", "0")),
                Encoding.UTF8,
                "application/json");
            using HttpResponseMessage plainResponse = await http.PostAsync(app.Urls.Single() + "/v1/chat/completions", plainBody, CancellationToken.None);
            Assert.Equal(HttpStatusCode.OK, plainResponse.StatusCode);
            JsonElement plainJson = JsonDocument.Parse(await plainResponse.Content.ReadAsStringAsync(CancellationToken.None)).RootElement;

            // Declaring tools changes nothing on the wire: same prompt accounting, same content bytes, no tool_calls.
            Assert.False(toolsJson.GetProperty("choices")[0].GetProperty("message").TryGetProperty("tool_calls", out _));
            Assert.NotEqual("tool_calls", toolsJson.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
            Assert.Equal(
                plainJson.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString(),
                toolsJson.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString());
            Assert.Equal(
                plainJson.GetProperty("usage").GetProperty("prompt_tokens").GetInt32(),
                toolsJson.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());
        }
        finally
        {
            await ((IHost)app).StopAsync(TimeSpan.FromSeconds(5));
            await app.DisposeAsync();
        }
    }

    // ---------------------------------------------------------------- observability

    [Fact]
    public void LoadPath_LogsResolvedParser_NamingModelAndChoice()
    {
        (string Template, string Name, string? Configured, string Resolved)[] cases =
        [
            (ChatMlToolsTemplate, "log-auto-capable", null, "template"),
            (ChatMlTemplate, "log-auto-plain", null, "fenced"),
            (ChatMlToolsTemplate, "log-fenced", "fenced", "fenced"),
            (ChatMlToolsTemplate, "log-none", "none", "none"),
        ];

        foreach ((string template, string name, string? configured, string resolved) in cases)
        {
            CapturingLogger logger = new();
            using GenerationService engine = OpenSynthetic(template, name, logger, toolParser: configured);

            string line = Assert.Single(logger.Messages(LogLevel.Information), m => m.Contains("tool_parser"));
            Assert.Contains($"model='{name}'", line);
            Assert.Contains($"choice={resolved}", line);
            Assert.Equal(resolved, engine.Model.ToolParser);
        }
    }

    [Fact]
    public async Task ModelsEndpoint_ReportsResolvedParser_LikeQuantAndArchitecture()
    {
        string path = WriteSyntheticGguf(ChatMlToolsTemplate);
        LairConfig config = ConfigFor(path, "advertised-model", toolParser: "none");
        using GenerationService engine = GenerationService.Open(config, new CapturingLogger());
        using WebApplication app = ApiApp.Build(config, engine);
        await ((IHost)app).StartAsync(CancellationToken.None);

        try
        {
            using HttpClient http = new();
            using HttpResponseMessage response = await http.GetAsync(app.Urls.Single() + "/v1/models", CancellationToken.None);
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);

            JsonElement body = JsonDocument.Parse(await response.Content.ReadAsStringAsync(CancellationToken.None)).RootElement;
            JsonElement entry = body.GetProperty("data")[0];
            Assert.Equal("advertised-model", entry.GetProperty("id").GetString());
            Assert.False(string.IsNullOrEmpty(entry.GetProperty("quant").GetString()));
            Assert.False(string.IsNullOrEmpty(entry.GetProperty("architecture").GetString()));
            Assert.Equal("none", entry.GetProperty("tool_parser").GetString());
        }
        finally
        {
            await ((IHost)app).StopAsync(TimeSpan.FromSeconds(5));
            await app.DisposeAsync();
        }
    }

    // ---------------------------------------------------------------- hot-reload application (new-requests-only)

    [Fact]
    public void ToolParserReload_ApliesAsNewRequestsOnly_AndLaterRequestsFollowTheNewChoice()
    {
        using GenerationService engine = OpenSynthetic(ChatMlToolsTemplate, "reload-parser-model", new CapturingLogger());

        // Opened auto on a tool-capable template: the template surface is in force.
        Assert.Equal("template", engine.Model.ToolParser);

        LairConfig candidate = engine.EffectiveConfiguration with
        {
            Models = [engine.EffectiveConfiguration.Models[0] with { ToolParser = "none" }],
        };

        FieldResult applied = Assert.Single(engine.ApplyConfiguration(candidate));
        Assert.Equal("models[0].tool_parser", applied.Change.Field);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, applied.Change.MutationClass);
        Assert.Equal(ConfigApplyOutcome.Applied, applied.Outcome);

        Assert.Equal("none", engine.Model.ToolParser); // the advertised fact follows the applied choice

        GenerationRequest request = ToolRequest("reload-parser-model");
        Assert.Equal(engine.BuildChatPrompt(request with { Tools = null }), engine.BuildChatPrompt(request));
    }

    [Fact]
    public void ForceFencedToolsReload_ReResolvesAutoPlan_NewRequestsFollow()
    {
        using GenerationService engine = OpenSynthetic(ChatMlToolsTemplate, "reload-fence-model", new CapturingLogger());

        Assert.Equal("template", engine.Model.ToolParser);

        LairConfig candidate = engine.EffectiveConfiguration with
        {
            Models = [engine.EffectiveConfiguration.Models[0] with { ForceFencedTools = true }],
        };

        FieldResult applied = Assert.Single(engine.ApplyConfiguration(candidate));
        Assert.Equal("models[0].force_fenced_tools", applied.Change.Field);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, applied.Change.MutationClass);
        Assert.Equal(ConfigApplyOutcome.Applied, applied.Outcome);

        // The auto plan re-resolved with the override: later requests take the fenced block.
        Assert.Equal("fenced", engine.Model.ToolParser);
        Assert.Contains("\nYou can call these tools:\n", engine.BuildChatPrompt(ToolRequest("reload-fence-model")));
    }

    // ---------------------------------------------------------------- request plumbing

    private static string JsonBody(params (string Key, string RawJson)[] fields)
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

    private static GenerationRequest ToolRequest(string model) => new()
    {
        Model = model,
        IsChat = true,
        RequestId = "req-parser",
        Messages = [new Lair.Tokenizer.Templating.ChatMessage("user", "hello")],
        MaxTokens = 4,
        Temperature = 0f,
        Tools = [WeatherTool],
    };

    // ---------------------------------------------------------------- synthetic model fixture

    private const string ChatMlToolsTemplate =
        "{% if tools is defined %}[tool]{% for t in tools %}{{ t['function']['name'] }}|{{ t['function']['description'] }}|{{ t['function']['parameters'] | tojson }}{% endfor %}{% endif %}"
        + "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
        + "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

    private const string ChatMlTemplate =
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

    private GenerationService OpenSynthetic(string chatTemplate, string modelName, ILogger logger, string? toolParser = null)
    {
        string path = WriteSyntheticGguf(chatTemplate);
        return GenerationService.Open(ConfigFor(path, modelName, toolParser), logger);
    }

    private static LairConfig ConfigFor(string modelPath, string modelName, string? toolParser) => new()
    {
        Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
        Engine = new EngineConfig(),
        Models =
        [
            new ModelConfig
            {
                Name = modelName,
                Path = modelPath,
                MaxContextTokens = 512,
                KvBlocks = 32,
                ToolParser = toolParser ?? "auto",
            },
        ],
    };

    // Two-layer llama weights (hidden 4, heads 2, kv-heads 1, ffn 8, vocab 6) with a six-token BPE vocabulary and
    // the requested chat_template (or none), so GenerationService.Open runs the full load path — tokenizer, forward
    // engine, KV pool, scheduler, chat-template parse, parser-plan resolve — against a model the test controls.
    private string WriteSyntheticGguf(string? chatTemplate)
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

        string[] tokens = ["<unk>", "hello", "world", "!", "<s>", "</s>"];
        int[] tokenTypes = [2, 1, 1, 1, 4, 5];

        List<(string Name, int[] Dims)> tensors = [new("token_embd.weight", [6, 4])];
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

        int kvCount = 18 + (chatTemplate is not null ? 1 : 0);

        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u); // version
        writer.Write((ulong)tensors.Count);
        writer.Write((ulong)kvCount);

        WriteStringKv(writer, "general.architecture", "llama");
        WriteStringKv(writer, "general.name", "lair-parser-fixture");
        WriteInt32Kv(writer, "llama.embedding_length", 4);
        WriteInt32Kv(writer, "llama.block_count", 2);
        WriteInt32Kv(writer, "llama.attention.head_count", 2);
        WriteInt32Kv(writer, "llama.attention.head_count_kv", 1);
        WriteInt32Kv(writer, "llama.feed_forward_length", 8);
        WriteFloat32Kv(writer, "llama.rope.freq_base", 10_000f);
        WriteFloat32Kv(writer, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        WriteInt32Kv(writer, "llama.vocab_size", 6);
        WriteStringKv(writer, "tokenizer.ggml.model", "gpt2");
        WriteStringArrayKv(writer, "tokenizer.ggml.tokens", tokens);
        WriteInt32ArrayKv(writer, "tokenizer.ggml.token_type", tokenTypes);
        WriteStringArrayKv(writer, "tokenizer.ggml.merges", []);
        WriteUint32Kv(writer, "tokenizer.ggml.bos_token_id", 4);
        WriteUint32Kv(writer, "tokenizer.ggml.eos_token_id", 5);
        WriteUint32Kv(writer, "tokenizer.ggml.unknown_token_id", 0);
        WriteBoolKv(writer, "tokenizer.ggml.add_bos_token", false);
        if (chatTemplate is not null)
        {
            WriteStringKv(writer, "tokenizer.chat_template", chatTemplate);
        }

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

        string path = Path.Combine(Path.GetTempPath(), $"lair-parser-tests-{Guid.NewGuid():N}.gguf");
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

    // ---------------------------------------------------------------- capturing logger

    private sealed class CapturingLogger : ILogger
    {
        private readonly List<(LogLevel Level, string Message)> _entries = [];

        public IReadOnlyList<string> Messages(LogLevel level) =>
            [.. _entries.Where(entry => entry.Level == level).Select(entry => entry.Message)];

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            ArgumentNullException.ThrowIfNull(formatter);
            _entries.Add((logLevel, formatter(state, exception)));
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
}
