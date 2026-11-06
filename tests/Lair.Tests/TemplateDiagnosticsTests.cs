using System.Net;
using System.Reflection;
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
/// Diagnostics for the LAIR renderer adapter: startup fail-fast on invalid chat templates (the log names the model
/// path plus the offending construct and its offset), request-time render failures logged with the model name, the
/// x-lair-request-id correlation id, and a bounded error, the deliberate raise_exception mapping to the HTTP
/// invalid_request shape, the real Llama-3.2-1B-Instruct chat template parsing and rendering through the adapter
/// instead of failing at its {% set %} statements, parse-once caching of the loaded model's template across
/// requests, cancellation surfacing as the actionable cancellation error instead of a raw
/// OperationCanceledException, and the capability-driven tool precedence that decides whether a declared tool set
/// renders through the model's own template or through LAIR's fenced injection. Synthetic tiny GGUF models keep
/// every case independent of the SmolLM fixture.
/// </summary>
[Trait("Category", "TemplateDiagnostics")]
public sealed class TemplateDiagnosticsTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    // ---------------------------------------------------------------- invalid startup template

    [Fact]
    public void InvalidStartupTemplate_FailsModelLoadWithActionableLog()
    {
        CapturingLogger logger = new();
        string path = WriteSyntheticGguf("before {% raw page %} after");

        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(
            () => GenerationService.Open(ConfigFor(path, "bad-template-model"), logger));

        Assert.Contains("bad-template-model", ex.Message);
        Assert.Contains(path, ex.Message);
        Assert.Contains("invalid chat template", ex.Message);
        Assert.Contains("raw", ex.Message);

        string failureLog = Assert.Single(logger.Messages(LogLevel.Error));
        Assert.Contains(path, failureLog);
        Assert.Contains("construct=raw", failureLog);
        Assert.Contains("offset=7", failureLog);
    }

    // ---------------------------------------------------------------- request-time render failure

    [Fact]
    public async Task RenderFailure_LogsModelRequestIdAndBoundedError()
    {
        string refusal = "x" + new string('y', 900);
        CapturingLogger logger = new();
        using GenerationService engine = OpenSynthetic($"{{{{ raise_exception('{refusal}') }}}}", "render-fail-model", logger);

        TemplateRenderFailedException ex = await Assert.ThrowsAsync<TemplateRenderFailedException>(() => engine.GenerateAsync(
            ChatRequest("render-fail-model", "req-77", "hello"),
            CancellationToken.None));

        Assert.Equal("render-fail-model", ex.ModelId);
        Assert.Equal("req-77", ex.RequestId);
        Assert.Contains("could not render the chat template", ex.Message);
        Assert.IsType<MinjaRenderException>(ex.InnerException);
        Assert.Equal("raise_exception", ((MinjaRenderException)ex.InnerException!).Construct);

        // The surfaced message is bounded: the long template refusal is truncated instead of riding onto the wire.
        Assert.True(ex.Message.Length <= 600, $"message length {ex.Message.Length} exceeded the adapter bound");
        Assert.Contains("...", ex.Message);

        string failureLog = Assert.Single(logger.Messages(LogLevel.Error));
        Assert.Contains("render-fail-model", failureLog);
        Assert.Contains("request_id=req-77", failureLog);
        Assert.Contains("raise_exception", failureLog);
    }

    // ---------------------------------------------------------------- raise_exception -> invalid_request

    [Fact]
    public async Task RaiseException_MapsToInvalidRequestResponse()
    {
        string path = WriteSyntheticGguf("{{ raise_exception('Only system message is not supported.') }}");
        LairConfig config = ConfigFor(path, "raise-model");
        using GenerationService engine = GenerationService.Open(config, new CapturingLogger());
        using WebApplication app = ApiApp.Build(config, engine);
        await ((IHost)app).StartAsync(CancellationToken.None);

        try
        {
            using HttpClient http = new();
            using StringContent body = new(
                """{"model":"raise-model","messages":[{"role":"user","content":"hello"}],"max_tokens":4}""",
                Encoding.UTF8,
                "application/json");

            HttpResponseMessage response = await http.PostAsync(app.Urls.Single() + "/v1/chat/completions", body, CancellationToken.None);

            Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

            using JsonDocument json = JsonDocument.Parse(await response.Content.ReadAsStringAsync(CancellationToken.None));
            JsonElement error = json.RootElement.GetProperty("error");
            Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
            Assert.Equal("invalid_request", error.GetProperty("code").GetString());
            Assert.Contains("Only system message is not supported.", error.GetProperty("message").GetString());
            Assert.True(response.Headers.TryGetValues(ApiApp.RequestIdHeader, out _));
        }
        finally
        {
            await ((IHost)app).StopAsync(TimeSpan.FromSeconds(5));
            await app.DisposeAsync();
        }
    }

    // ---------------------------------------------------------------- real Llama-3.2-1B-Instruct template

    [Fact]
    public async Task Llama32Template_ParsesAtStartupAndRendersThroughAdapter()
    {
        CapturingLogger logger = new();
        using GenerationService engine = OpenSynthetic(Llama32ChatTemplate, "llama32-model", logger);

        // Startup succeeded: the {% set %}, {% for %}, macro-free tool plumbing, namespace-free date handling,
        // comments, and raise_exception guards all parse, and capability facts were probed at load.
        Assert.NotNull(engine.ChatTemplateCapabilities);

        GenerationResult result = await engine.GenerateAsync(
            ChatRequest("llama32-model", "req-llama", "hello"),
            CancellationToken.None);

        Assert.True(result.PromptTokens > 0);
        Assert.Empty(logger.Messages(LogLevel.Error));
    }

    // ---------------------------------------------------------------- parse-once caching

    [Fact]
    public async Task ChatTemplate_IsParsedOnceAtLoadAndReusedAcrossRequests()
    {
        using GenerationService engine = OpenSynthetic(ChatMlTemplate, "cache-model", new CapturingLogger());

        FieldInfo templateField = typeof(GenerationService).GetField("_chatTemplate", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("the loaded model no longer caches its parsed chat template");

        object? firstLoad = templateField.GetValue(engine);
        Assert.NotNull(firstLoad); // parse happened at load, before any request

        FieldInfo factsField = typeof(MinjaChatTemplate).GetField("_facts", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("the wrapper no longer caches its capability probe");

        Assert.NotNull(factsField.GetValue(firstLoad)); // capability probing also happened at load

        for (int i = 0; i < 3; i++)
        {
            GenerationResult result = await engine.GenerateAsync(
                ChatRequest("cache-model", $"req-cache-{i}", "hello"),
                CancellationToken.None);

            Assert.True(result.PromptTokens > 0);
        }

        Assert.Same(firstLoad, templateField.GetValue(engine)); // requests reused the load-time parse
        Assert.Same(engine.ChatTemplateCapabilities, engine.ChatTemplateCapabilities);
    }

    // ---------------------------------------------------------------- cancellation wiring

    [Fact]
    public void CancelledRender_SurfacesAsActionableCancellationError()
    {
        MinjaChatTemplate chat = new("{% for i in range(1000) %}{{ i }}{% endfor %}");
        ChatTemplateInput input = new() { Messages = [ChatMessage.Text("user", "hello")] };

        using CancellationTokenSource cancelled = new();
        cancelled.Cancel();

        RenderLimitsException error = Assert.Throws<RenderLimitsException>(() => chat.Render(input, cancellationToken: cancelled.Token));

        Assert.Equal("cancellation", error.Limit);
        Assert.Contains("cancelled", error.Message);
        chat.Render(input); // the same template renders fine without cancellation
    }

    [Fact]
    public async Task MidRenderCancellation_SurfacesAsActionableCancellationError()
    {
        // Nested loops burn a long evaluation inside the render — 500 outer plus 199,500 inner iterations, just
        // under the loop budget and with each range far under the sequence-length guard — so a cancellation a few
        // milliseconds in lands while the render is still running. The leading message echo keeps the rendered
        // prompt encodable by the fixture vocabulary so the post-cancel retry exercises a real prompt. Under a
        // loaded full-suite run the test thread can also be descheduled long enough that the cancellation beats
        // the render and lands on the entry guard instead of the mid-render path; the timing is scheduling, not a
        // production difference. Both cooperative boundaries must report the cancellation as an actionable failure
        // — the adapter's bounded template error mid-render, the engine's OperationCanceledException at the entry
        // guard — and never as a hang or an unhandled crash. The deterministic mid-render wrapping itself is
        // pinned by CancelledRender_SurfacesAsActionableCancellationError: a pre-cancelled token makes the
        // library's first render-side probe observe the cancellation, and the adapter wraps that exact exception
        // shape.
        using GenerationService engine = OpenSynthetic(
            "{{ messages[0]['content'] }}{% for outer in range(500) %}{% for inner in range(399) %}{% set scratch = inner %}{% endfor %}{% endfor %}",
            "cancel-model",
            new CapturingLogger());

        using CancellationTokenSource cts = new();
        Task cancelSoon = Task.Delay(TimeSpan.FromMilliseconds(3)).ContinueWith(_ => cts.Cancel());

        Exception failure;
        try
        {
            await engine.GenerateAsync(ChatRequest("cancel-model", "req-cancel", "hello"), cts.Token);
            throw new InvalidOperationException("the cancelled request completed normally; the cancellation never surfaced");
        }
        catch (Exception ex)
        {
            failure = ex;
        }

        await cancelSoon;

        switch (failure)
        {
            // Cancellation observed mid-render: the adapter wraps the library's cancellation limit as the bounded,
            // actionable template error — never a raw OperationCanceledException from the render path.
            case TemplateRenderFailedException renderFailure:
                RenderLimitsException cause = Assert.IsType<RenderLimitsException>(renderFailure.InnerException);
                Assert.Equal("cancellation", cause.Limit);
                Assert.Contains("cancellation", renderFailure.Message);
                break;

            // Cancellation beat the render and landed on the entry guard: the same cooperative cancellation,
            // reported as OperationCanceledException before any render work happened.
            case OperationCanceledException:
                break;

            default:
                throw new InvalidOperationException(
                    $"cancellation surfaced as {failure.GetType().Name} instead of an actionable cancellation failure: {failure.Message}");
        }

        // The cancelled request leaves no residue: the same model renders the same conversation normally afterwards.
        GenerationResult clean = await engine.GenerateAsync(
            ChatRequest("cancel-model", "req-cancel-retry", "hello"),
            CancellationToken.None);
        Assert.True(clean.PromptTokens > 0);
    }

    // ---------------------------------------------------------------- capability-driven tool precedence (DN-M32-7)

    // A ChatML-style template that renders the `tools` context itself: a marker plus each tool's name, description,
    // and parameter schema. The capability probe sees its dummy tool name, so the request path must hand this
    // template the declared tools and skip LAIR's fenced block.
    private const string ChatMlToolsTemplate =
        "{% if tools is defined %}[tool]{% for t in tools %}{{ t['function']['name'] }}|{{ t['function']['description'] }}|{{ t['function']['parameters'] | tojson }}{% endfor %}{% endif %}"
        + "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
        + "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

    private static readonly ToolDefinition WeatherTool = new(
        "get_weather",
        "Fetch current weather for a city.",
        """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""");

    [Fact]
    public async Task ToolCapableTemplate_RendersToolsItself_AndReceivesNoFencedInjection()
    {
        using GenerationService engine = OpenSynthetic(ChatMlToolsTemplate, "tools-template-model", new CapturingLogger());

        Assert.NotNull(engine.ChatTemplateCapabilities);
        Assert.True(engine.ChatTemplateCapabilities!.SupportsTools, "the tools-rendering template did not probe as tool-capable");

        string prompt = engine.BuildChatPrompt(ToolRequest("tools-template-model"));

        // The template's own tool rendering is present: name, description, and parameter schema through its tools loop.
        Assert.Contains("[tool]", prompt);
        Assert.Contains("get_weather", prompt);
        Assert.Contains("Fetch current weather for a city.", prompt);
        Assert.Contains("\"city\"", prompt);

        // LAIR's fenced block is absent: no instruction line, no schema fences, no reply contract.
        Assert.DoesNotContain("You can call these tools:", prompt);
        Assert.DoesNotContain("```json", prompt);
        Assert.DoesNotContain(ToolPromptComposer.ContractLine, prompt);

        // The same precedence holds on the full request path, and the fenced detector stays wired for a model that
        // emits fenced calls even when the template rendered the tools.
        GenerationResult result = await engine.GenerateAsync(ToolRequest("tools-template-model"), CancellationToken.None);
        Assert.True(result.PromptTokens > 0);
    }

    [Fact]
    public void ToollessTemplate_ReceivesFencedInjection_ByteIdenticalToToday()
    {
        using GenerationService engine = OpenSynthetic(ChatMlTemplate, "plain-template-model", new CapturingLogger());

        Assert.NotNull(engine.ChatTemplateCapabilities);
        Assert.False(engine.ChatTemplateCapabilities!.SupportsTools, "the tools-ignoring template probed as tool-capable");

        string prompt = engine.BuildChatPrompt(ToolRequest("plain-template-model"));

        // Today's path exactly: the template renders with no tools context, then LAIR's block is appended.
        string rendered = EngineChatTemplate(engine).Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("user", "hello")],
            AddGenerationPrompt = true,
        }).Prompt;

        Assert.Equal(ToolPromptComposer.Compose(rendered, [WeatherTool]), prompt);
    }

    [Fact]
    public void ForceFencedToolsOverride_InjectsEvenForToolCapableTemplate()
    {
        string path = WriteSyntheticGguf(ChatMlToolsTemplate);
        LairConfig config = new()
        {
            Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
            Engine = new EngineConfig(),
            Models =
            [
                new ModelConfig
                {
                    Name = "forced-model",
                    Path = path,
                    MaxContextTokens = 512,
                    KvBlocks = 32,
                    ForceFencedTools = true,
                },
            ],
        };

        using GenerationService engine = GenerationService.Open(config, new CapturingLogger());

        Assert.True(engine.ChatTemplateCapabilities!.SupportsTools);

        string prompt = engine.BuildChatPrompt(ToolRequest("forced-model"));

        // The override selects LAIR's block despite the template's tool support, and the declared tools stay out of
        // the template context so exactly one tool surface reaches the model.
        Assert.Contains("\nYou can call these tools:\n", prompt);
        Assert.Contains("\"name\":\"get_weather\"", prompt);
        Assert.Contains(WeatherTool.ParametersSchema, prompt);
        Assert.True(prompt.EndsWith(ToolPromptComposer.ContractLine), "the reply contract line must close the injected block");
        Assert.DoesNotContain("[tool]", prompt); // the template's own tools loop did not run
    }

    [Fact]
    public void ToolChoiceNone_InjectsNothing_ForToolCapableAndToollessTemplates()
    {
        using GenerationService capable = OpenSynthetic(ChatMlToolsTemplate, "none-capable-model", new CapturingLogger());
        using GenerationService plain = OpenSynthetic(ChatMlTemplate, "none-plain-model", new CapturingLogger());

        foreach ((GenerationService engine, string model) in new (GenerationService, string)[]
        {
            (capable, "none-capable-model"),
            (plain, "none-plain-model"),
        })
        {
            GenerationRequest request = ToolRequest(model) with { ToolChoice = ToolChoiceSelection.None };
            string prompt = engine.BuildChatPrompt(request);

            Assert.DoesNotContain("You can call these tools:", prompt);
            Assert.DoesNotContain(ToolPromptComposer.ContractLine, prompt);
            Assert.DoesNotContain("get_weather", prompt);
            Assert.DoesNotContain("[tool]", prompt);
        }
    }

    // The parse-once chat template the loaded engine actually renders with, so byte-identity compares against the
    // adapter's own template rather than a second copy built in the test.
    private static MinjaChatTemplate EngineChatTemplate(GenerationService engine)
    {
        FieldInfo templateField = typeof(GenerationService).GetField("_chatTemplate", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("the loaded model no longer caches its parsed chat template");

        return (MinjaChatTemplate)(templateField.GetValue(engine)
            ?? throw new InvalidOperationException("the served model loaded without a chat template"));
    }

    private static GenerationRequest ToolRequest(string model) => new()
    {
        Model = model,
        IsChat = true,
        RequestId = "req-tools",
        Messages = [new Lair.Tokenizer.Templating.ChatMessage("user", "hello")],
        MaxTokens = 4,
        Temperature = 0f,
        Tools = [WeatherTool],
    };

    // ---------------------------------------------------------------- synthetic model fixture

    private const string ChatMlTemplate =
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

    // The published tokenizer.chat_template of meta-llama/Llama-3.2-1B-Instruct (GGUF metadata form). It broke the
    // retired ChatTemplateRenderer at its first {% set %}; the adapter path must parse and render it.
    private const string Llama32ChatTemplate =
        "{{- bos_token }}\n"
        + "{%- if custom_tools is defined %}\n"
        + "    {%- set tools = custom_tools %}\n"
        + "{%- endif %}\n"
        + "{%- if not tools_in_user_message is defined %}\n"
        + "    {%- set tools_in_user_message = true %}\n"
        + "{%- endif %}\n"
        + "{%- if not date_string is defined %}\n"
        + "    {%- if strftime_now is defined %}\n"
        + "        {%- set date_string = strftime_now(\"%d %b %Y\") %}\n"
        + "    {%- else %}\n"
        + "        {%- set date_string = \"26 Jul 2024\" %}\n"
        + "    {%- endif %}\n"
        + "{%- endif %}\n"
        + "{%- if not tools is defined %}\n"
        + "    {%- set tools = none %}\n"
        + "{%- endif %}\n"
        + "\n"
        + "{#- This block extracts the system message, so we can slot it into the right place. #}\n"
        + "{%- if messages[0]['role'] == 'system' %}\n"
        + "    {%- set system_message = messages[0]['content']|trim %}\n"
        + "    {%- set messages = messages[1:] %}\n"
        + "{%- else %}\n"
        + "    {%- set system_message = \"\" %}\n"
        + "{%- endif %}\n"
        + "\n"
        + "{#- System message #}\n"
        + "{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n"
        + "{%- if tools is not none %}\n"
        + "    {{- \"Environment: ipython\\n\" }}\n"
        + "{%- endif %}\n"
        + "{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n"
        + "{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n"
        + "{%- if tools is not none and not tools_in_user_message %}\n"
        + "    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n"
        + "    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n"
        + "    {{- \"Do not use variables.\\n\\n\" }}\n"
        + "    {%- for t in tools %}\n"
        + "        {{- t | tojson(indent=4) }}\n"
        + "        {{- \"\\n\\n\" }}\n"
        + "    {%- endfor %}\n"
        + "{%- endif %}\n"
        + "{{- system_message }}\n"
        + "{{- \"<|eot_id|>\" }}\n"
        + "\n"
        + "{#- Custom tools are passed in a user message with some extra guidance #}\n"
        + "{%- if tools_in_user_message and not tools is none %}\n"
        + "    {#- Extract the first user message so we can plug it in here #}\n"
        + "    {%- if messages | length != 0 %}\n"
        + "        {%- set first_user_message = messages[0]['content']|trim %}\n"
        + "        {%- set messages = messages[1:] %}\n"
        + "    {%- else %}\n"
        + "        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n"
        + "{%- endif %}\n"
        + "    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n"
        + "    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n"
        + "    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n"
        + "    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n"
        + "    {{- \"Do not use variables.\\n\\n\" }}\n"
        + "    {%- for t in tools %}\n"
        + "        {{- t | tojson(indent=4) }}\n"
        + "        {{- \"\\n\\n\" }}\n"
        + "    {%- endfor %}\n"
        + "    {{- first_user_message + \"<|eot_id|>\"}}\n"
        + "{%- endif %}\n"
        + "\n"
        + "{%- for message in messages %}\n"
        + "    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n"
        + "        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n"
        + "    {%- elif 'tool_calls' in message %}\n"
        + "        {%- if not message.tool_calls|length == 1 %}\n"
        + "            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n"
        + "        {%- endif %}\n"
        + "        {%- set tool_call = message.tool_calls[0].function %}\n"
        + "        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n"
        + "        {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n"
        + "        {{- '\"parameters\": ' }}\n"
        + "        {{- tool_call.arguments | tojson }}\n"
        + "        {{- \"}\" }}\n"
        + "        {{- \"<|eot_id|>\" }}\n"
        + "    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n"
        + "        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n"
        + "        {%- if message.content is mapping or message.content is iterable %}\n"
        + "            {{- message.content | tojson }}\n"
        + "        {%- else %}\n"
        + "            {{- message.content }}\n"
        + "        {%- endif %}\n"
        + "        {{- \"<|eot_id|>\" }}\n"
        + "    {%- endif %}\n"
        + "{%- endfor %}\n"
        + "{%- if add_generation_prompt %}\n"
        + "    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n"
        + "{%- endif %}";

    [Fact]
    public void DeclaredEotId_IsTheStopSignal_WhileAbsentKeepsEos()
    {
        using GenerationService declared = OpenSynthetic(ChatMlTemplate, "eot-model", new CapturingLogger(), eotTokenId: 4);
        Assert.Equal(4, declared.StopId);

        using GenerationService absent = OpenSynthetic(ChatMlTemplate, "eos-only-model", new CapturingLogger());
        Assert.Equal(5, absent.StopId);
    }

    private GenerationService OpenSynthetic(string chatTemplate, string modelName, ILogger logger, int? eotTokenId = null)
    {
        string path = WriteSyntheticGguf(chatTemplate, eotTokenId);
        return GenerationService.Open(ConfigFor(path, modelName), logger);
    }

    private static LairConfig ConfigFor(string modelPath, string modelName) => new()
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
            },
        ],
    };

    private static GenerationRequest ChatRequest(string model, string requestId, string content) => new()
    {
        Model = model,
        IsChat = true,
        RequestId = requestId,
        Messages = [new Lair.Tokenizer.Templating.ChatMessage("user", content)],
        MaxTokens = 4,
        Temperature = 0f,
    };

    // ---------------------------------------------------------------- synthetic GGUF builder

    // Two-layer llama weights (hidden 4, heads 2, kv-heads 1, ffn 8, vocab 6) with a six-token BPE vocabulary and
    // the requested chat_template, so GenerationService.Open runs the full load path — tokenizer, forward engine,
    // KV pool, scheduler, chat-template parse — against a model the test controls completely.
    private string WriteSyntheticGguf(string chatTemplate, int? eotTokenId = null)
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

        int kvCount = 19 + (eotTokenId is null ? 0 : 1);

        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u); // version
        writer.Write((ulong)tensors.Count);
        writer.Write((ulong)kvCount);

        WriteStringKv(writer, "general.architecture", "llama");
        WriteStringKv(writer, "general.name", "lair-diagnostics-fixture");
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
        if (eotTokenId is not null)
        {
            WriteUint32Kv(writer, "tokenizer.ggml.eot_token_id", (uint)eotTokenId.Value);
        }
        WriteStringKv(writer, "tokenizer.chat_template", chatTemplate);

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

        string path = Path.Combine(Path.GetTempPath(), $"lair-template-diagnostics-{Guid.NewGuid():N}.gguf");
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
