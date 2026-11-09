using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Lair.Http;
using Lair.Host.Config;
using Lair.Loader;
using Lair.Templating.Minja;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;

namespace Lair.Tests;

/// <summary>
/// Parser-level goldens for the template-native tool-call reader over synthetic model output in the rendered shape of
/// the pinned Llama-3.2 chat template: that template instructs "Respond in the format {"name": function name,
/// "parameters": dictionary of argument name and its value}" and renders an accepted call back as exactly one such
/// object (see tests/fixtures/parity-templates-llama32.json). The reader accepts brace-balanced objects holding exactly
/// <c>name</c> (a declared tool) and <c>parameters</c>; the fenced contract's <c>{"name":...,"arguments":...}</c> shape,
/// unknown names, malformed or unterminated objects, and extra properties all stay ordinary text byte-for-byte.
/// </summary>
[Trait("Category", "ToolCalls")]
public sealed class TemplateNativeToolCallParserTests
{
    private static readonly TemplateNativeToolOutputParser Parser = new();

    // The exact assistant tool-call block the pinned template renders for a calculator call (fixture case
    // tool_call_with_response_gen), which is the byte shape a model following the instruction line must emit.
    private const string FixtureCall = """{"name": "calculator", "parameters": {"expression": "2+2"}}""";

    [Fact]
    public void Golden_FixtureRenderedCallObject_IsRecognizedWithCompactedArguments()
    {
        ToolDetection detection = Parser.ReadCalls(FixtureCall, ["calculator"]);

        Assert.Single(detection.Calls);
        Assert.Equal("calculator", detection.Calls[0].Name);
        // The fixture renders spaced arguments; the call carries them compactly for the wire, in the JSON encoder's escaped form exactly as the fenced path emits them.
        Assert.Equal("""{"expression":"2\u002B2"}""", detection.Calls[0].Arguments);
        Assert.Equal(string.Empty, detection.VisibleText);
    }

    [Fact]
    public void Golden_ProseAroundCallObject_KeepsProseAndStripsOnlyTheCall()
    {
        string text = "Let me calculate that.\n" + FixtureCall;

        ToolDetection detection = Parser.ReadCalls(text, ["calculator"]);

        Assert.Single(detection.Calls);
        Assert.Equal("calculator", detection.Calls[0].Name);
        Assert.Equal("Let me calculate that.\n", detection.VisibleText); // only the object is cut; the newline ahead of it stays
    }

    [Fact]
    public void Golden_TwoCallsInDocumentOrder_ReportBothAndKeepSurroundingProse()
    {
        string text = "First.\n" + FixtureCall + "\nThen weather.\n"
            + """{"name": "get_weather", "parameters": {"city": "Oslo"}}""" + "\nDone.";

        ToolDetection detection = Parser.ReadCalls(text, ["calculator", "get_weather"]);

        Assert.Equal(2, detection.Calls.Count);
        Assert.Equal("calculator", detection.Calls[0].Name);
        Assert.Equal("""{"expression":"2\u002B2"}""", detection.Calls[0].Arguments);
        Assert.Equal("get_weather", detection.Calls[1].Name);
        Assert.Equal("{\"city\":\"Oslo\"}", detection.Calls[1].Arguments);

        string visible = detection.VisibleText;
        Assert.StartsWith("First.", visible);
        Assert.Contains("Then weather.", visible);
        Assert.EndsWith("Done.", visible);
        Assert.DoesNotContain("\"name\"", visible);
    }

    [Fact]
    public void Golden_FencedContractShape_IsNotATemplateCall_StaysOrdinaryTextByteForByte()
    {
        // The fenced path's reply contract shape — name/arguments — is not the template's name/parameters shape.
        string text = "Answer.\n```json\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"2+2\"}}\n```\nThe end.";

        ToolDetection detection = Parser.ReadCalls(text, ["calculator"]);

        Assert.Empty(detection.Calls);
        Assert.Equal(text, detection.VisibleText);
    }

    [Fact]
    public void Golden_UnknownToolName_StaysOrdinaryTextByteForByte()
    {
        string text = "Answer.\n" + """{"name": "lookup_address", "parameters": {"street": "Elm"}}""" + "\nThe end.";

        ToolDetection detection = Parser.ReadCalls(text, ["calculator"]);

        Assert.Empty(detection.Calls);
        Assert.Equal(text, detection.VisibleText);
    }

    [Theory]
    [InlineData("""{"name": "calculator", "parameters": """)]          // unterminated object
    [InlineData("""{"name": "calculator", "parameters": {""")]         // unterminated parameters object and string
    [InlineData("""{"name": "calculator"}""")]                         // no parameters property
    [InlineData("""{"parameters": {"expression": "2+2"}}""")]         // no name property
    [InlineData("""{"name": 42, "parameters": {"expression": "2+2"}}""")] // name is not a string
    [InlineData("""{"name": "calculator", "parameters": {"expression": "2+2"}, "id": 7}""")] // extra property
    [InlineData("""{"name": "calculator", "parameters": {"expression": "2+2"}, "arguments": {}}""")] // fenced-shape member mixed in
    [InlineData("""{"name": "calculator", "name": "calculator", "parameters": {}}""")] // duplicate name
    [InlineData("not json at all")]                                    // no object at all
    [InlineData("""{"name": "calculator", "parameters": }""")]         // malformed parameters
    public void Golden_MalformedCallShape_StaysOrdinaryTextByteForByte(string body)
    {
        string text = "Answer.\n" + body + "\nThe end.";

        ToolDetection detection = Parser.ReadCalls(text, ["calculator"]);

        Assert.Empty(detection.Calls);
        Assert.Equal(text, detection.VisibleText);
    }

    [Fact]
    public void Golden_BracesInsideQuotedValues_DoNotEndTheCandidateAndDoNotShiftVisibleBytes()
    {
        string text = "Say {keep this} and {\"name\": \"calculator\", \"parameters\": {\"expression\": \"{2+2}\"}} please.";

        ToolDetection detection = Parser.ReadCalls(text, ["calculator"]);

        Assert.Single(detection.Calls); // the ordinary brace run before the call is never a candidate: only the parseable object is
        Assert.Equal("calculator", detection.Calls[0].Name);
        Assert.Equal("""{"expression":"{2\u002B2}"}""", detection.Calls[0].Arguments); // braces inside the quoted value survive the call, in the encoder's escaped form like every fenced-path argument does
        Assert.Equal("Say {keep this} and please.", detection.VisibleText);
    }

    [Fact]
    public void Golden_PrettyPrintedArguments_ArriveCompactlyOnTheCall()
    {
        string body = """{ "name" : "calculator" , "parameters" : { "expression" : "2 + 2" , "unit" : 2 } }""";

        ToolDetection detection = Parser.ReadCalls(body, ["calculator"]);

        Assert.Single(detection.Calls);
        Assert.Equal("""{"expression":"2 \u002B 2","unit":2}""", detection.Calls[0].Arguments);
    }

    [Fact]
    public void Golden_ForcedChoice_AcceptsOnlyTheForcedNameAmongDeclaredTools()
    {
        string text = FixtureCall + "\n" + """{"name": "get_weather", "parameters": {"city": "Oslo"}}""";

        // A forced choice narrows the accepted-name set to that one member of the declared tools.
        ToolDetection detection = Parser.ReadCalls(text, ["get_weather"]);

        Assert.Single(detection.Calls);
        Assert.Equal("get_weather", detection.Calls[0].Name);
        Assert.Contains(FixtureCall, detection.VisibleText); // the non-forced call stays ordinary text, object included
    }

    [Fact]
    public void Golden_CompactWireArgumentsMatchTheFencedPathsEncodingRules()
    {
        // Same ToolCall record the fenced detector produces: name plus compactly re-encoded arguments.
        ToolDetection native = Parser.ReadCalls(FixtureCall, ["calculator"]);
        ToolCall call = Assert.Single(native.Calls);

        ToolDetection fenced = ToolCallDetector.Detect(
            "```json\n" + """{"name":"calculator","arguments":{"expression": "2+2"}}""" + "\n```", ["calculator"]);
        ToolCall fencedCall = Assert.Single(fenced.Calls);

        Assert.Equal(fencedCall.Name, call.Name);
        Assert.Equal(fencedCall.Arguments, call.Arguments);
    }
}

/// <summary>
/// The resolved plan for a model whose chat template renders tools itself: the pinned Llama-3.2 template (carried byte
/// for byte from tests/fixtures/parity-templates-llama32.json into a synthetic GGUF so the resolve path runs against the
/// real template) resolves to <c>template</c> with the template-native reader wired, the fenced block never reaches the
/// prompt, and the fenced-JSON detector's input shape is not read as a call by the plan's parser — the fenced path is
/// inactive for this model. Runs inside the serialized API collection so its engine never loads alongside another.
/// </summary>
[Trait("Category", "ToolCalls")]
[Collection("lair-api-serialized")]
public sealed class TemplateNativeToolCallPlanTests : IDisposable
{
    private static readonly ToolDefinition CalculatorTool = new(
        "calculator",
        "Perform arithmetic on a numeric expression",
        """{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}""");

    // The fenced contract's exact call shape — what the fenced detector accepts and the template-native reader must not.
    private const string FencedCallText = "```json\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"2+2\"}}\n```";

    private readonly List<string> _tempFiles = [];

    [Fact]
    public void PinnedLlama32Template_ResolvesTemplateWithTemplateNativeParser()
    {
        using GenerationService engine = OpenPinnedTemplateModel("pinned-resolve-model");

        Assert.True(engine.ChatTemplateCapabilities!.SupportsTools);
        Assert.Equal("template", engine.Model.ToolParser);
        Assert.Equal(ToolParser.Template, engine.ResolvedOutputParser.ToolParser);
        Assert.IsType<TemplateNativeToolOutputParser>(engine.ResolvedOutputParser.Parser);
        Assert.Equal(ReasoningPolicy.Keep, engine.ResolvedOutputParser.Reasoning);
    }

    [Fact]
    public void PinnedLlama32Prompt_RendersToolsThroughTheTemplate_WithNoFencedContract()
    {
        using GenerationService engine = OpenPinnedTemplateModel("pinned-prompt-model");

        string prompt = engine.BuildChatPrompt(ToolRequest("pinned-prompt-model"));

        // The template's own instruction lines are the whole tool surface (fixture case tools_declared_gen).
        Assert.Contains("Given the following functions, please respond with a JSON for a function call", prompt);
        Assert.Contains("""Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.""", prompt);
        Assert.Contains("\"name\": \"calculator\"", prompt); // the declared tool reached the template context

        // The fenced path is inactive: LAIR injected nothing.
        Assert.DoesNotContain("You can call these tools:", prompt);
        Assert.DoesNotContain(ToolPromptComposer.ContractLine, prompt);
    }

    [Fact]
    public void PinnedLlama32Plan_FencedDetectorShapeIsNotReadAsACall_WhileTheDetectorItselfWouldAcceptIt()
    {
        using GenerationService engine = OpenPinnedTemplateModel("pinned-inactive-model");

        IToolOutputParser planParser = engine.ResolvedOutputParser.Parser!;

        // The same text the fenced path would read as a call: under this model's resolved parser it is ordinary text.
        ToolDetection planReading = planParser.ReadCalls(FencedCallText, ["calculator"]);
        Assert.Empty(planReading.Calls);
        Assert.Equal(FencedCallText, planReading.VisibleText);

        // Proof that the fenced reader is a different, inactive reader: given the same text it reports a call.
        ToolDetection fencedReading = ToolCallDetector.Detect(FencedCallText, ["calculator"]);
        Assert.Single(fencedReading.Calls);
    }

    [Fact]
    public void PinnedLlama32Plan_ReadsTheTemplatesOwnCallShape_AsACall()
    {
        using GenerationService engine = OpenPinnedTemplateModel("pinned-native-model");

        // The exact object the pinned template renders for an accepted calculator call (fixture case
        // tool_call_with_response_gen): the plan's parser recognizes it with the wire's compact arguments.
        ToolDetection detection = engine.ResolvedOutputParser.Parser!.ReadCalls(
            """{"name": "calculator", "parameters": {"expression": "2+2"}}""",
            ["calculator"]);

        ToolCall call = Assert.Single(detection.Calls);
        Assert.Equal("calculator", call.Name);
        Assert.Equal("""{"expression":"2\u002B2"}""", call.Arguments);
    }

    [Fact]
    public void ExplicitFencedChoice_OnTheSameTemplate_KeepsTheFencedParser_Active()
    {
        using GenerationService engine = OpenPinnedTemplateModel("pinned-fenced-choice-model", toolParser: "fenced");

        Assert.Equal("fenced", engine.Model.ToolParser);
        Assert.IsType<FencedJsonToolOutputParser>(engine.ResolvedOutputParser.Parser);

        ToolDetection fencedReading = engine.ResolvedOutputParser.Parser!.ReadCalls(FencedCallText, ["calculator"]);
        Assert.Single(fencedReading.Calls); // the override selects the fenced reader back
    }

    // ---------------------------------------------------------------- fixture plumbing

    private GenerationRequest ToolRequest(string model) => new()
    {
        Model = model,
        IsChat = true,
        RequestId = "req-native-parser",
        Messages = [new Lair.Tokenizer.Templating.ChatMessage("user", "What is 2+2?")],
        MaxTokens = 4,
        Temperature = 0f,
        Tools = [CalculatorTool],
    };

    // The pinned Llama-3.2 template plus a synthetic tiny model body: GenerationService.Open runs the full load path —
    // tokenizer, forward engine, capability probe, plan resolve — against the template the parity fixture records.
    private GenerationService OpenPinnedTemplateModel(string modelName, string? toolParser = null)
    {
        string path = WriteSyntheticGguf(PinnedTemplate());
        return GenerationService.Open(new LairConfig
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
                    ToolParser = toolParser ?? "auto",
                },
            ],
        });
    }

    // The pinned template source, read from the committed parity fixture with its recorded sha256 guard.
    private static string PinnedTemplate()
    {
        string fixtureFile = ModelFixture.ResolveRepoRelative("tests/fixtures/parity-templates-llama32.json")
            ?? throw new InvalidOperationException("the pinned template fixture is absent");

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(fixtureFile));
        string template = document.RootElement.GetProperty("chat_template").GetString()!;
        string recordedSha = document.RootElement.GetProperty("chat_template_sha256").GetString()!;

        if (Sha256Text(template) != recordedSha)
        {
            throw new InvalidOperationException(
                $"template fixture guard: the fixture's chat_template hashes to {Sha256Text(template)} but the fixture records {recordedSha}");
        }

        return template;
    }

    private static string Sha256Text(string text)
    {
        using SHA256 hasher = SHA256.Create();
        return Convert.ToHexString(hasher.ComputeHash(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();
    }

    // Two-layer llama weights (hidden 4, heads 2, kv-heads 1, ffn 8, vocab 6) with a six-token BPE vocabulary and the
    // requested chat_template, shaped like the synthetic fixture the tool-parser suites already use.
    private string WriteSyntheticGguf(string chatTemplate)
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

        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u); // version
        writer.Write((ulong)tensors.Count);
        writer.Write((ulong)19);

        WriteStringKv(writer, "general.architecture", "llama");
        WriteStringKv(writer, "general.name", "lair-native-parser-fixture");
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

        string path = Path.Combine(Path.GetTempPath(), $"lair-native-parser-tests-{Guid.NewGuid():N}.gguf");
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

/// <summary>
/// Fixture: opens the real pinned Llama-3.2-1B-Instruct-Q4_K_M (the model the template fixture was captured against)
/// with the default tool configuration so the plan resolves from the model's own template. When the GGUF is absent the
/// app never starts and every test in this class skips, matching how the other real-model suites skip. Runs inside the
/// serialized API collection so its engine never loads alongside another.
/// </summary>
[Collection("lair-api-serialized")]
public sealed class Llama32ToolsApiFixture : IAsyncLifetime
{
    public const string ModelName = "llama32-test";

    private GenerationService? _engine;
    private WebApplication? _app;

    /// <summary>The server's base URL, or empty when the pinned model is absent and nothing was started.</summary>
    public string BaseUrl { get; private set; } = string.Empty;

    /// <summary>The opened engine, for the plan-shape assertions.</summary>
    public GenerationService Engine => _engine ?? throw new InvalidOperationException("the fixture model is absent");

    public async Task InitializeAsync()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative("models/Llama-3.2-1B-Instruct-Q4_K_M.gguf");
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        LairConfig config = new()
        {
            Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
            Engine = new EngineConfig(),
            Models =
            [
                new ModelConfig
                {
                    Name = ModelName,
                    Path = modelPath,
                    MaxContextTokens = 512,
                    KvBlocks = 32,
                },
            ],
        };

        _engine = GenerationService.Open(config);
        _app = ApiApp.Build(config, _engine);
        await ((IHost)_app).StartAsync(CancellationToken.None);
        BaseUrl = _app.Urls.Single();
    }

    public async Task DisposeAsync()
    {
        if (_app is not null)
        {
            await ((IHost)_app).StopAsync(TimeSpan.FromSeconds(5));
            await _app.DisposeAsync();
        }

        _engine?.Dispose();
    }
}

/// <summary>
/// Live tool-call behavior for the pinned Llama-3.2 model on its own template: the resolved parser is the template-
/// native reader, the declared tools reach the model through the template's rendering with no fenced contract injected,
/// and a chat request declaring tools returns an OpenAI-shaped result whose recognized calls carry the template's own
/// name/parameters markup on the fenced wire shape. Emission itself is model-dependent per the detection contract; only
/// resolution, prompt surface, and wire shape are asserted here.
/// </summary>
[Trait("Category", "ToolCalls")]
[Collection("lair-api-serialized")]
public class Llama32ToolsApiTests : IClassFixture<Llama32ToolsApiFixture>
{
    private static readonly HttpClient Http = new();

    private static readonly string[] FinishReasons = ["stop", "length", "eos", "tool_calls"];

    private const string ToolsJson = """[{"type":"function","function":{"name":"calculator","description":"Perform arithmetic on a numeric expression","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}]""";
    private static readonly Regex CallIdPattern = new("^call_[0-9a-f]{12}$", RegexOptions.Compiled);

    private readonly Llama32ToolsApiFixture _fixture;

    public Llama32ToolsApiTests(Llama32ToolsApiFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public void RealModel_ResolvesTemplateWithTemplateNativeParser_FencedPathInactive()
    {
        if (_fixture.BaseUrl.Length == 0)
        {
            return; // skip when the pinned model is absent
        }

        GenerationService engine = _fixture.Engine;
        Assert.Equal("template", engine.Model.ToolParser);
        Assert.Equal(ToolParser.Template, engine.ResolvedOutputParser.ToolParser);
        Assert.IsType<TemplateNativeToolOutputParser>(engine.ResolvedOutputParser.Parser);

        // The fenced reader is not what this model's plan runs: its call shape reads as no call under the resolved parser.
        ToolDetection fencedShape = engine.ResolvedOutputParser.Parser!.ReadCalls(
            "```json\n{\"name\":\"calculator\",\"arguments\":{\"expression\":\"2+2\"}}\n```", ["calculator"]);
        Assert.Empty(fencedShape.Calls);
    }

    [Fact]
    public async Task ChatWithTools_NonStream_ReturnsWellFormedToolShape()
    {
        if (_fixture.BaseUrl.Length == 0)
        {
            return; // skip when the pinned model is absent
        }

        JsonElement body = await PostJsonAsync("/v1/chat/completions", ChatBody(
            """[{"role":"user","content":"What is 2+2? Use the calculator tool."}]""", ToolsJson));

        Assert.Equal("chat.completion", body.GetProperty("object").GetString());

        JsonElement choice = body.GetProperty("choices")[0];
        Assert.Contains(choice.GetProperty("finish_reason").GetString(), FinishReasons);

        JsonElement message = choice.GetProperty("message");
        Assert.Equal("assistant", message.GetProperty("role").GetString());
        Assert.True(message.TryGetProperty("content", out _)); // content is always reported, possibly empty

        if (message.TryGetProperty("tool_calls", out JsonElement calls))
        {
            AssertToolCallArray(calls);
            Assert.Equal("tool_calls", choice.GetProperty("finish_reason").GetString());
        }

        int completionTokens = body.GetProperty("usage").GetProperty("completion_tokens").GetInt32();
        Assert.True(completionTokens > 0);
    }

    [Fact]
    public async Task ChatWithTools_Stream_CarriesToolCallDeltasBeforeTheFinalChunk()
    {
        if (_fixture.BaseUrl.Length == 0)
        {
            return; // skip when the pinned model is absent
        }

        List<JsonElement> chunks = await StreamAsync("/v1/chat/completions", ChatBody(
            """[{"role":"user","content":"What is 2+2? Use the calculator tool."}]""", ToolsJson, stream: true));

        Assert.Equal("assistant", chunks[0].GetProperty("choices")[0].GetProperty("delta").GetProperty("role").GetString());

        string finishReason = chunks[^1].GetProperty("choices")[0].GetProperty("finish_reason").GetString()!;
        Assert.Contains(finishReason, FinishReasons);

        List<JsonElement> callDeltas = chunks
            .Select(chunk => chunk.GetProperty("choices")[0].GetProperty("delta"))
            .Where(delta => delta.TryGetProperty("tool_calls", out _))
            .ToList();

        foreach (JsonElement delta in callDeltas)
        {
            AssertToolCallArray(delta.GetProperty("tool_calls"));
        }

        // A recognized call must surface on the wire and flip the finish reason; content deltas stay well-formed either way.
        if (finishReason == "tool_calls")
        {
            Assert.NotEmpty(callDeltas);
        }
    }

    private static void AssertToolCallArray(JsonElement calls)
    {
        Assert.True(calls.GetArrayLength() >= 1);

        foreach (JsonElement call in calls.EnumerateArray())
        {
            Assert.Matches(CallIdPattern, call.GetProperty("id").GetString());
            Assert.Equal("function", call.GetProperty("type").GetString());

            JsonElement function = call.GetProperty("function");
            Assert.Equal("calculator", function.GetProperty("name").GetString()); // the only declared tool in these tests

            string arguments = function.GetProperty("arguments").GetString()!;
            Assert.NotEmpty(arguments);

            // The arguments must be valid, parseable JSON that already sits compactly in the wire string.
            using JsonDocument parsed = JsonDocument.Parse(arguments);
            Assert.Equal(arguments, parsed.RootElement.GetRawText());
        }
    }

    // ---------------------------------------------------------------- request plumbing

    private static string ChatBody(string messagesJson, string toolsJson, bool stream = false)
    {
        List<(string Key, string RawJson)> fields =
        [
            ("model", "\"" + Llama32ToolsApiFixture.ModelName + "\""),
            ("messages", messagesJson),
            ("max_tokens", "48"),
            ("temperature", "0"),
            ("tools", toolsJson),
        ];

        if (stream)
        {
            fields.Add(("stream", "true"));
        }

        return JsonBody([.. fields]);
    }

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

    private async Task<List<JsonElement>> StreamAsync(string path, string jsonBody)
    {
        using HttpRequestMessage request = new(HttpMethod.Post, Url(path))
        {
            Content = new StringContent(jsonBody, Encoding.UTF8, "application/json"),
        };

        using HttpResponseMessage response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);

        List<JsonElement> chunks = [];
        string lastDataLine = string.Empty;
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
                lastDataLine = line;
                if (line == "data: [DONE]")
                {
                    break;
                }

                chunks.Add(JsonDocument.Parse(line["data: ".Length..]).RootElement);
            }
        }

        Assert.Equal("data: [DONE]", lastDataLine); // the done sentinel is unchanged by tool deltas
        return chunks;
    }

    private string Url(string path) => _fixture.BaseUrl + path;

    private async Task<JsonElement> PostJsonAsync(string path, string jsonBody)
    {
        using HttpRequestMessage request = new(HttpMethod.Post, Url(path))
        {
            Content = new StringContent(jsonBody, Encoding.UTF8, "application/json"),
        };

        HttpResponseMessage response = await Http.SendAsync(request);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        return JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement;
    }
}
