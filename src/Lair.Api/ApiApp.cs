using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using Lair.Host.Config;
using Lair.Tokenizer.Templating;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace Lair.Api;

/// <summary>
/// Builds the OpenAI-compatible HTTP surface over one opened <see cref="GenerationService"/>:
/// <c>/v1/chat/completions</c> (JSON and SSE), <c>/v1/completions</c> (JSON and SSE), <c>/v1/models</c>,
/// and <c>/healthz</c>. Every response carries an <c>x-lair-request-id</c> header and one structured log line;
/// engine exceptions are converted to the OpenAI error shape <c>{error:{message,type,code}}</c>.
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
            await HandleChat(await ReadBodyAsync<ChatCompletionBody>(http.Request), gen, http).ExecuteAsync(http));

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
                    gen.Model.Architecture)]);

            return Results.Json(models, statusCode: 200);
        });

        app.MapGet("/healthz", () => Results.Json(new HealthResponse("ok", true)));

        return app;
    }

    private const string RequestAuditKey = "lair.request.audit";

    private static IResult HandleChat(ChatCompletionBody body, GenerationService gen, HttpContext http)
    {
        if (body.Messages is null || body.Messages.Count == 0)
        {
            return InvalidRequest("a chat request requires a non-empty 'messages' list");
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
            Stream = body.Stream ?? false,
        };

        return RunGeneration(request, gen, http);
    }

    private static IResult HandleCompletion(CompletionBody body, GenerationService gen, HttpContext http)
    {
        if (body.Prompt is null)
        {
            return InvalidRequest("a completion request requires a 'prompt'");
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
            Stream = body.Stream ?? false,
        };

        return RunGeneration(request, gen, http);
    }

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
            ? Results.Json(BuildChatCompletion(request.Model, result))
            : Results.Json(BuildTextCompletion(request.Model, result));
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

        foreach (string piece in pieces)
        {
            await WriteChunkAsync(stream, new ChunkPayload(
                completionId, "chat.completion.chunk", created, model,
                [new ChunkChoice(0, new ChunkDelta(Content: piece), null)]), http);
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

    private static ChatCompletionResponse BuildChatCompletion(string model, GenerationResult result) =>
        new(
            "chatcmpl-" + Guid.NewGuid().ToString("N"),
            "chat.completion",
            DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            model,
            [new ChatChoice(0, new ChatMessage("assistant", result.Text), result.FinishReason)],
            UsageOf(result));

    private static TextCompletionResponse BuildTextCompletion(string model, GenerationResult result) =>
        new(
            "cmpl-" + Guid.NewGuid().ToString("N"),
            "text_completion",
            DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            model,
            [new TextChoice(0, result.Text, result.FinishReason)],
            UsageOf(result));

    private static Usage UsageOf(GenerationResult result) =>
        new(result.PromptTokens, result.CompletionTokens, result.PromptTokens + result.CompletionTokens);

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
