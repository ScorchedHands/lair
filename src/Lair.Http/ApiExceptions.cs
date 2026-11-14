namespace Lair.Http;

/// <summary>
/// The request named a model this service does not serve.
/// </summary>
public sealed class ModelNotFoundException : Exception
{
    public ModelNotFoundException(string requested)
        : base($"model '{requested}' is not served by this engine. Serve it by adding it to the models list in the " +
               "config, or request a model id that is listed there.")
    {
        Requested = requested;
    }

    /// <summary>The model id from the request.</summary>
    public string Requested { get; }
}

/// <summary>
/// The prompt alone does not fit the model's context window, so generation cannot start. This is the request layer's token-budget refusal:
/// it names the model and the limit kind that blocked — the resolved context window — alongside the counts a caller can act on. A null model id
/// keeps the generic wording for callers that have no named model to report.
/// </summary>
public sealed class ContextLengthExceededException : Exception
{
    public ContextLengthExceededException(long promptTokens, long limit, string? modelId = null)
        : base(BuildMessage(promptTokens, limit, modelId))
    {
        PromptTokens = promptTokens;
        Limit = limit;
        ModelId = modelId;
    }

    /// <summary>Token count of the rejected prompt.</summary>
    public long PromptTokens { get; }

    /// <summary>The context limit the prompt was measured against.</summary>
    public long Limit { get; }

    /// <summary>The served model id, when its caller named one: every refusal then names it alongside the token-budget limit that blocked.</summary>
    public string? ModelId { get; }

    private static string BuildMessage(long promptTokens, long limit, string? modelId) =>
        (modelId is null ? "" : $"model '{modelId}' cannot start this generation: ") +
        $"prompt is {promptTokens} tokens but the context window is {limit} tokens, and at least one token " +
        "must stay free for generation. Shorten the prompt or raise max_context_tokens and kv_blocks in the config.";
}

/// <summary>
/// The engine cannot admit this request right now: either its prompt plus token budget exceeds what the KV pool can back (kv_capacity), or every
/// admission slot of the model's concurrency cap is held by sequences already admitted (max_concurrent_requests). Both refusals surface through
/// this one overload-family type so callers keep a single bounded error shape; each factory names the limit kind and the model it belongs to. The
/// default form covers an engine-side failure observed mid-run, where no fresh admission decision exists to name.
/// </summary>
public sealed class EngineBusyException : Exception
{
    public EngineBusyException()
        : base("the engine cannot admit this request: its prompt plus token budget exceeds the token slots the " +
               "KV pool can back. Retry once the running generations finish, or raise kv_blocks in the config.")
    {
    }

    private EngineBusyException(string message)
        : base(message)
    {
    }

    /// <summary>The refusal for a submission whose prompt plus token budget outruns the KV pool's free slots (kv_capacity).</summary>
    public static EngineBusyException ForKvCapacity(string modelId, long freeSlots) =>
        new($"model '{modelId}' cannot admit this request against its kv_capacity: its prompt plus token budget exceeds " +
            $"the {freeSlots} token slots the KV pool can back right now. Retry once the running generations finish, or raise kv_blocks in the config.");

    /// <summary>The refusal for a submission arriving while every admission slot of the model's concurrency cap is held (max_concurrent_requests).</summary>
    public static EngineBusyException ForConcurrencyQuota(string modelId, int maxConcurrentRequests) =>
        new($"model '{modelId}' cannot admit this request against its concurrency quota: all {maxConcurrentRequests} admission slots of " +
            "'max_concurrent_requests' are held by sequences already admitted. Retry once a running generation finishes, or raise " +
            "max_concurrent_requests in the config.");
}

/// <summary>
/// The loaded model's chat template parsed but failed to render a request's conversation: a deliberate
/// <c>raise_exception</c>, a template limit, or a cancelled render. The message is bounded and names the model, so the
/// HTTP layer surfaces it as an actionable <c>invalid_request</c> rather than an unhandled engine fault. A cancelled
/// render reaches this type through the library's <see cref="Lair.Templating.Minja.RenderLimitsException"/>, never a
/// raw <see cref="OperationCanceledException"/>.
/// </summary>
public sealed class TemplateRenderFailedException : Exception
{
    private const int MaxDetailLength = 512;

    public TemplateRenderFailedException(string modelId, string requestId, string detail, Exception cause)
        : base(BuildMessage(modelId, detail), cause)
    {
        ModelId = modelId;
        RequestId = requestId;
    }

    /// <summary>The model whose template failed.</summary>
    public string ModelId { get; }

    /// <summary>The request correlation id (<c>x-lair-request-id</c>) this failure belongs to, empty when the caller supplied none.</summary>
    public string RequestId { get; }

    private static string BuildMessage(string modelId, string detail)
    {
        string bounded = detail.Length <= MaxDetailLength ? detail : detail[..MaxDetailLength] + "...";
        return $"model '{modelId}' could not render the chat template for this request: {bounded}";
    }
}
