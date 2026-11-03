using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using Lair.Api.Monitoring;
using Lair.Host.Config;
using Lair.Models.Scheduling;
using Lair.Tokenizer.Templating;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.StaticFiles;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging;

namespace Lair.Api;

/// <summary>
/// Builds the OpenAI-compatible HTTP surface over one opened <see cref="GenerationService"/>:
/// <c>/v1/chat/completions</c> (JSON and SSE), <c>/v1/completions</c> (JSON and SSE), <c>/v1/models</c>,
/// <c>/healthz</c>, and the operator stream <c>/monitor/stream</c>. Every response carries an
/// <c>x-lair-request-id</c> header and one structured log line; engine exceptions are converted to the OpenAI
/// error shape <c>{error:{message,type,code}}</c>.
/// </summary>
public static class ApiApp
{
    /// <summary>Correlation header carried by every response, including errors.</summary>
    public const string RequestIdHeader = "x-lair-request-id";

    public static WebApplication Build(LairConfig config, GenerationService gen)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(gen);

        WebApplicationBuilder builder = WebApplication.CreateBuilder([]);

        // Kestrel binds exactly the configured address; port 0 asks for an ephemeral port.
        builder.WebHost.ConfigureKestrel(options =>
        {
            IPAddress address = IPAddress.TryParse(config.Server.Bind, out IPAddress? parsed)
                ? parsed
                : Dns.GetHostAddresses(config.Server.Bind).First(a => a.AddressFamily == AddressFamily.InterNetwork);
            options.Listen(address, config.Server.Port);
        });

        builder.Logging.AddFilter("Microsoft.AspNetCore.Hosting.Diagnostics", LogLevel.None);

        builder.Services.Configure<Microsoft.AspNetCore.Http.Json.JsonOptions>(options =>
        {
            JsonSerializerOptions json = options.SerializerOptions;
            json.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
            json.DictionaryKeyPolicy = JsonNamingPolicy.SnakeCaseLower;
            json.WriteIndented = false;
            json.DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull;
        });

        WebApplication app = builder.Build();
        ILogger logger = app.Services.GetRequiredService<ILogger<GenerationService>>();

        app.Use(async (context, next) =>
        {
            string requestId = Guid.NewGuid().ToString("N");
            context.Response.Headers[RequestIdHeader] = requestId;

            Stopwatch clock = Stopwatch.StartNew();
            RequestAudit audit = new();
            context.Items[RequestAuditKey] = audit;
            context.Items[RequestIdItemKey] = requestId; // handlers hand the same id to the engine so monitor lifecycle rows correlate with this response

            try
            {
                await next(context);
            }
            catch (Exception ex)
            {
                if (context.Response.HasStarted)
                {
                    // Status and headers are already on the wire; the only honest completion is to abort.
                    throw;
                }

                await WriteErrorAsync(context.Response, ex, logger, requestId);
            }
            finally
            {
                clock.Stop();
                LogRequest(logger, requestId, context, audit, clock.ElapsedMilliseconds);
            }
        });

        // A HttpContext-only lambda binds to the RequestDelegate overload, which discards a returned
        // IResult, so the result is executed explicitly here.
        app.MapPost("/v1/chat/completions", async (HttpContext http) =>
            await HandleChat(await ReadChatBodyAsync(http.Request), gen, http).ExecuteAsync(http));

        app.MapPost("/v1/completions", async (HttpContext http) =>
            await HandleCompletion(await ReadBodyAsync<CompletionBody>(http.Request), gen, http).ExecuteAsync(http));

        app.MapGet("/v1/models", () =>
        {
            ModelsResponse models = new(
                "list",
                [new ModelEntry(
                    gen.Model.Id,
                    "model",
                    DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    "lair",
                    gen.Model.ContextLength,
                    gen.Model.Quant,
                    gen.Model.Architecture,
                    gen.ModelType)]);

            return Results.Json(models, statusCode: 200);
        });

        app.MapGet("/healthz", () => Results.Json(new HealthResponse("ok", true)));

        // Operator surface, deliberately outside /v1/*: one SSE event per engine snapshot. The bus attaches here so
        // only serving hosts sample; chat-style hosts never pay for it.
        EngineMetricsBus metrics = gen.AttachMetrics(app.Services.GetRequiredService<ILoggerFactory>(), MetricsSnapshotInterval);

        app.MapGet("/monitor/stream", async (HttpContext http) => await MonitorStreamHandler.HandleAsync(http, metrics));

        MapWebUi(app);

        return app;
    }

    // Conventional .NET SPA hosting: the ClientApp bundle is staged into the web root (wwwroot/ui) by the
    // Host build, served at /ui, and any unmatched /ui/* client route falls back to the shell so client-side
    // routes resolve. The bare root redirects permanently to /ui/ so `lair serve` opens in a browser. All of
    // this is registered only when the build embedded the UI (a SkipWebUi build keeps the previous 404
    // behavior everywhere); existing API endpoints are untouched.
    // Microsoft.AspNetCore.SpaServices.Extensions still ships for net10, but its UseSpa middleware rewrites
    // every request without a matched endpoint to the default page from the web root — incompatible with a
    // /ui mount, a root redirect, and preserving 404s outside /ui — and its SourcePath only powers dev-server
    // proxies this app does not use. The current ASP.NET Core recommendation (SPA templates since 6.0) is
    // exactly this shape: staged web-root assets plus a MapFallback shell route.
    private static void MapWebUi(WebApplication app)
    {
        string uiRoot = Path.Combine(AppContext.BaseDirectory, "wwwroot", "ui");
        string shellPath = Path.Combine(uiRoot, "index.html");
        if (!Directory.Exists(uiRoot) || !File.Exists(shellPath))
        {
            return;
        }

        // An exact-match endpoint wins over the fallback below, so the redirect only claims `/` itself.
        app.MapGet("/", () => Results.Redirect("/ui/", permanent: true));

        IFileProvider uiFiles = new PhysicalFileProvider(uiRoot);

        // The default mapping table lacks .map; sourcemaps would otherwise not be served at all.
        FileExtensionContentTypeProvider contentTypes = new();
        contentTypes.Mappings[".map"] = "application/json";

        string shellHtml = File.ReadAllText(shellPath);

        // A terminal middleware instead of UseStaticFiles: the fallback endpoint below matches every /ui/* path in
        // routing, and static-file middleware declines requests that already carry a matched endpoint. This branch
        // serves real bundle files (directory default included) and ends the pipeline; anything it passes through
        // lands on the shell-fallback endpoint.
        app.Use(async (http, next) =>
        {
            const string requestPath = "/ui";
            if (!http.Request.Path.StartsWithSegments(requestPath, StringComparison.OrdinalIgnoreCase))
            {
                await next(http);
                return;
            }

            if (!(HttpMethods.IsGet(http.Request.Method) || HttpMethods.IsHead(http.Request.Method)))
            {
                await next(http);
                return;
            }

            if (http.Request.Path.Value is not string decodedPath)
            {
                await next(http);
                return;
            }

            string relative = decodedPath[requestPath.Length..].TrimStart('/');
            if (relative.Length == 0)
            {
                // GET /ui and /ui/ serve the shell as the directory default.
                http.Response.ContentType = "text/html; charset=utf-8";
                await http.Response.WriteAsync(shellHtml);
                return;
            }

            IFileInfo file = uiFiles.GetFileInfo(relative);
            if (!file.Exists || file.IsDirectory)
            {
                await next(http);
                return;
            }

            if (!contentTypes.TryGetContentType(relative, out string? contentType))
            {
                contentType = "application/octet-stream";
            }

            http.Response.ContentType = contentType;
            http.Response.ContentLength = file.Length;
            using Stream stream = file.CreateReadStream();
            await stream.CopyToAsync(http.Response.Body);
        });

        app.MapFallback("/ui/{*path}", () => Results.Text(shellHtml, "text/html; charset=utf-8"));
    }

    /// <summary>Cadence of engine snapshot lines and stream events while the engine is active.</summary>
    internal static readonly TimeSpan MetricsSnapshotInterval = TimeSpan.FromSeconds(2);

    private const string RequestAuditKey = "lair.request.audit";
    private const string RequestIdItemKey = "lair.request.id";

    // The middleware-minted correlation id, read back by handlers to key monitor lifecycle frames on the same value
    // this response carries in x-lair-request-id.
    private static string? RequestIdOf(HttpContext http) =>
        http.Items.TryGetValue(RequestIdItemKey, out object? value) ? value as string : null;

    private static IResult HandleChat(ChatRequestInput input, GenerationService gen, HttpContext http)
    {
        ChatCompletionBody body = input.Body;

        if (body.Messages is null || body.Messages.Count == 0)
        {
            return InvalidRequest("a chat request requires a non-empty 'messages' list");
        }

        if (SamplingRangeError(body.PresencePenalty, body.FrequencyPenalty, body.TopLogprobs) is string rangeError)
        {
            return InvalidRequest(rangeError);
        }

        GenerationRequest request = new()
        {
            Model = body.Model ?? string.Empty,
            IsChat = true,
            Messages = [.. body.Messages.Select(m => new ChatMessage(m.Role ?? "user", m.Content ?? string.Empty))],
            MaxTokens = body.MaxTokens ?? 128,
            Temperature = body.Temperature ?? 1.0f,
            TopP = body.TopP ?? 0.9f,
            TopK = body.TopK ?? 40,
            Seed = body.Seed,
            Stop = body.Stop,
            PresencePenalty = body.PresencePenalty ?? 0f,
            FrequencyPenalty = body.FrequencyPenalty ?? 0f,
            LogitBias = body.LogitBias,
            LogProbs = body.LogProbs ?? false,
            TopLogprobs = body.TopLogprobs ?? 0,
            Tools = input.Tools.Count > 0 ? input.Tools : null,
            ToolChoice = input.ToolChoice.Mode == ToolChoiceMode.Auto ? null : input.ToolChoice,
            Stream = body.Stream ?? false,
            RequestId = RequestIdOf(http),
        };

        return RunGeneration(request, gen, http);
    }

    private static IResult HandleCompletion(CompletionBody body, GenerationService gen, HttpContext http)
    {
        if (body.Prompt is null)
        {
            return InvalidRequest("a completion request requires a 'prompt'");
        }

        if (SamplingRangeError(body.PresencePenalty, body.FrequencyPenalty, body.TopLogprobs) is string rangeError)
        {
            return InvalidRequest(rangeError);
        }

        GenerationRequest request = new()
        {
            Model = body.Model ?? string.Empty,
            IsChat = false,
            Prompt = body.Prompt,
            MaxTokens = body.MaxTokens ?? 128,
            Temperature = body.Temperature ?? 1.0f,
            TopP = body.TopP ?? 0.9f,
            TopK = body.TopK ?? 40,
            Seed = body.Seed,
            Stop = body.Stop,
            PresencePenalty = body.PresencePenalty ?? 0f,
            FrequencyPenalty = body.FrequencyPenalty ?? 0f,
            LogitBias = body.LogitBias,
            LogProbs = body.LogProbs ?? false,
            TopLogprobs = body.TopLogprobs ?? 0,
            Stream = body.Stream ?? false,
            RequestId = RequestIdOf(http),
        };

        return RunGeneration(request, gen, http);
    }

    // Sampling-knob ranges checked before the request reaches the engine; null means every value is in range.
    private static string? SamplingRangeError(float? presencePenalty, float? frequencyPenalty, int? topLogprobs) =>
        presencePenalty is < -2f or > 2f ? $"'presence_penalty' must be in [-2, 2], got {presencePenalty}." :
        frequencyPenalty is < -2f or > 2f ? $"'frequency_penalty' must be in [-2, 2], got {frequencyPenalty}." :
        topLogprobs is < 0 or > 20 ? $"'top_logprobs' must be in [0, 20], got {topLogprobs}." :
        null;

    private static IResult InvalidRequest(string message) =>
        Results.Json(
            new ErrorResponse(new ErrorDetail(message, "invalid_request_error", "invalid_request")),
            statusCode: 400);

    private static IResult RunGeneration(GenerationRequest request, GenerationService gen, HttpContext http)
    {
        if (AuditOf(http) is RequestAudit audit)
        {
            audit.Model = request.Model;
        }

        if (request.Stream)
        {
            return Results.Stream(
                stream => StreamCompletionsAsync(request, gen, stream, http),
                contentType: "text/event-stream");
        }

        GenerationResult result = gen.GenerateAsync(request, http.RequestAborted).GetAwaiter().GetResult();
        AuditOf(http)?.Completion(result);

        return request.IsChat
            ? Results.Json(BuildChatCompletion(request, gen, result))
            : Results.Json(BuildTextCompletion(request, gen, result));
    }

    private static async Task StreamCompletionsAsync(GenerationRequest request, GenerationService gen, Stream stream, HttpContext http)
    {
        http.Response.ContentType = "text/event-stream";
        http.Response.Headers.CacheControl = "no-cache";

        string completionId = "chatcmpl-" + Guid.NewGuid().ToString("N");
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        string model = request.Model;

        await WriteChunkAsync(stream, new ChunkPayload(
            completionId, "chat.completion.chunk", created, model,
            [new ChunkChoice(0, new ChunkDelta(Role: "assistant"), null)]), http);

        List<string> pieces = [];
        GenerationRequest pump = request with { OnToken = piece => pieces.Add(piece) };

        GenerationResult result = await gen.GenerateAsync(pump, http.RequestAborted);
        AuditOf(http)?.Completion(result);

        if (request.LogProbs && result.LogProbs is { Count: > 0 } entries)
        {
            // With logprobs on, every generated token gets its own content chunk carrying its logprob entry.
            foreach (TokenLogprob entry in entries)
            {
                await WriteChunkAsync(stream, new ChunkPayload(
                    completionId, "chat.completion.chunk", created, model,
                    [new ChunkChoice(0, new ChunkDelta(Content: gen.DecodeToken(entry.TokenId), Logprobs: WireEntry(gen, entry)), null)]), http);
            }
        }
        else
        {
            foreach (string piece in pieces)
            {
                await WriteChunkAsync(stream, new ChunkPayload(
                    completionId, "chat.completion.chunk", created, model,
                    [new ChunkChoice(0, new ChunkDelta(Content: piece), null)]), http);
            }
        }

        if (result.ToolCalls is { Count: > 0 } recognized)
        {
            // Recognized calls arrive after every content delta as complete-call deltas, one chunk per call in document order; the fenced text that produced them was already delivered in those deltas. Each carries its own generated id and compact arguments string.
            foreach (ToolCallWire wire in WireCalls(recognized) ?? [])
            {
                await WriteChunkAsync(stream, new ChunkPayload(
                    completionId, "chat.completion.chunk", created, model,
                    [new ChunkChoice(0, new ChunkDelta(ToolCalls: [wire]), null)]), http);
            }
        }

        await WriteChunkAsync(stream, new ChunkPayload(
            completionId, "chat.completion.chunk", created, model,
            [new ChunkChoice(0, new ChunkDelta(), result.FinishReason)]), http);

        await stream.WriteAsync(DoneEvent, http.RequestAborted);
        await stream.FlushAsync(http.RequestAborted);
    }

    private static async Task WriteChunkAsync(Stream stream, ChunkPayload payload, HttpContext http)
    {
        byte[] line = JsonSerializer.SerializeToUtf8Bytes(payload, WireJson);
        await stream.WriteAsync(SsePrefix, http.RequestAborted);
        await stream.WriteAsync(line, http.RequestAborted);
        await stream.WriteAsync(Newline, http.RequestAborted);
        await stream.FlushAsync(http.RequestAborted);
    }

    private static readonly byte[] SsePrefix = "data: "u8.ToArray();
    private static readonly byte[] Newline = "\n\n"u8.ToArray();
    private static readonly byte[] DoneEvent = "\ndata: [DONE]\n\n"u8.ToArray();

    private static readonly JsonSerializerOptions WireJson = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
    };

    // Bodies are bound explicitly so malformed JSON surfaces as JsonException to the error middleware
    // instead of the framework's silent empty 400.
    private static readonly JsonSerializerOptions RequestJson = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    private static async Task<T> ReadBodyAsync<T>(HttpRequest request) where T : class
    {
        using JsonDocument document = await JsonDocument.ParseAsync(request.Body, cancellationToken: request.HttpContext.RequestAborted);
        return document.RootElement.Deserialize<T>(RequestJson)
            ?? throw new ArgumentException("the request body deserialized to null; expected a JSON object for this endpoint.");
    }

    // The chat body plus its tool fields resolved while the parsed document is still alive, so no JsonElement outlives it.
    private sealed record ChatRequestInput(ChatCompletionBody Body, IReadOnlyList<ToolDefinition> Tools, ToolChoiceSelection ToolChoice);

    private static async Task<ChatRequestInput> ReadChatBodyAsync(HttpRequest request)
    {
        using JsonDocument document = await JsonDocument.ParseAsync(request.Body, cancellationToken: request.HttpContext.RequestAborted);
        ChatCompletionBody body = document.RootElement.Deserialize<ChatCompletionBody>(RequestJson)
            ?? throw new ArgumentException("the request body deserialized to null; expected a JSON object for this endpoint.");

        IReadOnlyList<ToolDefinition> tools = ToolWire.ParseTools(document.RootElement);
        ToolChoiceSelection toolChoice = ToolWire.ParseChoice(document.RootElement, tools);
        return new ChatRequestInput(body, tools, toolChoice);
    }

    private static ChatCompletionResponse BuildChatCompletion(GenerationRequest request, GenerationService gen, GenerationResult result) =>
        new(
            "chatcmpl-" + Guid.NewGuid().ToString("N"),
            "chat.completion",
            DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            request.Model,
            [new ChatChoice(0, new AssistantMessage("assistant", result.Text, WireCalls(result.ToolCalls)), result.FinishReason, LogProbsReportOf(request, gen, result))],
            UsageOf(result));

    private static TextCompletionResponse BuildTextCompletion(GenerationRequest request, GenerationService gen, GenerationResult result) =>
        new(
            "cmpl-" + Guid.NewGuid().ToString("N"),
            "text_completion",
            DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            request.Model,
            [new TextChoice(0, result.Text, result.FinishReason, LogProbsReportOf(request, gen, result))],
            UsageOf(result));

    // The logprobs report rides on the choice only when the request asked for it; its entries mirror the
    // generated tokens in order, each carrying the sampled distribution's top candidates.
    private static LogProbsReport? LogProbsReportOf(GenerationRequest request, GenerationService gen, GenerationResult result) =>
        request.LogProbs
            ? new LogProbsReport([.. (result.LogProbs ?? []).Select(entry => WireEntry(gen, entry))])
            : null;

    private static LogProbEntry WireEntry(GenerationService gen, TokenLogprob entry) =>
        new(
            entry.TokenId,
            gen.DecodeToken(entry.TokenId),
            entry.LogProb,
            entry.TopLogProbs.Count == 0
                ? null
                : [.. entry.TopLogProbs.Select(top => new LogProbEntry(top.TokenId, gen.DecodeToken(top.TokenId), top.LogProb, null))]);

    private static Usage UsageOf(GenerationResult result) =>
        new(result.PromptTokens, result.CompletionTokens, result.PromptTokens + result.CompletionTokens);

    // Recognized calls take the OpenAI wire shape: a generated id per call, the function type, and arguments as a compact JSON string.
    private static IReadOnlyList<ToolCallWire>? WireCalls(IReadOnlyList<ToolCall>? calls) =>
        calls is { Count: > 0 }
            ? [.. calls.Select(call => new ToolCallWire(ToolCallId(), "function", new FunctionCallWire(call.Name, call.Arguments)))]
            : null;

    private static string ToolCallId() => "call_" + Guid.NewGuid().ToString("N")[..12];

    private static RequestAudit? AuditOf(HttpContext http) =>
        http.Items.TryGetValue(RequestAuditKey, out object? value) ? value as RequestAudit : null;

    private static async Task WriteErrorAsync(HttpResponse response, Exception ex, ILogger logger, string requestId)
    {
        (int status, string type, string code) = ex switch
        {
            ModelNotFoundException => (404, "invalid_request_error", "model_not_found"),
            ContextLengthExceededException => (400, "invalid_request_error", "context_length_exceeded"),
            EngineBusyException => (503, "engine_overloaded_error", "engine_overloaded"),
            ArgumentException or JsonException => (400, "invalid_request_error", "invalid_request"),
            BadHttpRequestException => (400, "invalid_request_error", "invalid_request"),
            OperationCanceledException => (499, "client_disconnected_error", "client_disconnected"),
            _ => (500, "internal_error", "internal"),
        };

        if (code == "internal")
        {
            logger.LogError(ex, "unhandled engine fault request_id={RequestId}", requestId);
        }

        response.StatusCode = status;
        response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(response.Body, new ErrorResponse(new ErrorDetail(ex.Message, type, code)), WireJson);
    }

    private static void LogRequest(ILogger logger, string requestId, HttpContext http, RequestAudit audit, long durationMs)
    {
        logger.LogInformation(
            "request_id={RequestId} route={Route} model={Model} status={Status} prompt_tokens={PromptTokens} completion_tokens={CompletionTokens} duration_ms={DurationMs}",
            requestId,
            http.Request.Path.ToString(),
            audit.Model ?? string.Empty,
            http.Response.StatusCode,
            audit.PromptTokens,
            audit.CompletionTokens,
            durationMs);
    }

    /// <summary>Mutable per-request facts the middleware logs when the response completes.</summary>
    private sealed class RequestAudit
    {
        public string? Model { get; set; }
        public int PromptTokens { get; set; } = -1;
        public int CompletionTokens { get; set; } = -1;

        public void Completion(GenerationResult result)
        {
            PromptTokens = result.PromptTokens;
            CompletionTokens = result.CompletionTokens;
        }
    }
}
