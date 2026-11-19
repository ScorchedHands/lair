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
/// admission slot of the model's concurrency cap is held by sequences already admitted (max_concurrent_requests), or a submission waiting for such a
/// slot ran out of room in the model's queue (max_queued_requests) or out of its bounded wait (queue_timeout_ms), or the host device ledger's shared
/// copy-holding chain exhausted before this submission's reservation could be seated anywhere (budget exhaustion), or the named model has been removed
/// from the served set and is draining what already runs instead of accepting new work (retirement). All refusals surface through
/// this one overload-family type so callers keep a single bounded error shape; each factory names the limit kind, the model it belongs to, and the
/// wire code that distinguishes it — the capacity, quota, and queue refusals keep the code they have always answered with, and a ledger exhaustion answers
/// with its own. The default form covers an engine-side failure observed mid-run, where no fresh admission decision exists to name.
/// </summary>
public sealed class EngineBusyException : Exception
{
    public EngineBusyException()
        : base("the engine cannot admit this request: its prompt plus token budget exceeds the token slots the " +
               "KV pool can back. Retry once the running generations finish, or raise kv_blocks in the config.")
    {
        Code = "engine_overloaded";
    }

    private EngineBusyException(string message, string code)
        : base(message)
    {
        Code = code;
    }

    /// <summary>The wire error code this refusal answers with: <c>engine_overloaded</c> for the capacity, quota, and request-queue refusals, <c>budget_exhausted</c> for a host-ledger chain exhaustion, <c>model_retiring</c> for a submission naming a model being drained out of the served set.</summary>
    public string Code { get; }

    /// <summary>The refusal for a submission whose prompt plus token budget outruns the KV pool's free slots (kv_capacity).</summary>
    public static EngineBusyException ForKvCapacity(string modelId, long freeSlots) =>
        new($"model '{modelId}' cannot admit this request against its kv_capacity: its prompt plus token budget exceeds " +
            $"the {freeSlots} token slots the KV pool can back right now. Retry once the running generations finish, or raise kv_blocks in the config.",
            "engine_overloaded");

    /// <summary>The refusal for a submission arriving while every admission slot of the model's concurrency cap is held (max_concurrent_requests).</summary>
    public static EngineBusyException ForConcurrencyQuota(string modelId, int maxConcurrentRequests) =>
        new($"model '{modelId}' cannot admit this request against its concurrency quota: all {maxConcurrentRequests} admission slots of " +
            "'max_concurrent_requests' are held by sequences already admitted. Retry once a running generation finishes, or raise " +
            "max_concurrent_requests in the config.",
            "engine_overloaded");

    /// <summary>The refusal for a submission the host device ledger's shared copy-holding chain could not seat: the chain is exhausted at the named tier, and the breakdown says what every consulted position holds. The ledger's exhaustion is host-scoped accounting across opened pools — never a re-decision of this engine's own KV cap — so it rides the same bounded overload family with its own code.</summary>
    public static EngineBusyException ForBudgetExhaustion(string modelId, Lair.Models.BudgetResource resource, Lair.Models.BudgetTier tier, string breakdown) =>
        new($"model '{modelId}' cannot admit this request against its budgets: the host ledger exhausted the '{Lair.Models.BudgetResourceNames.Of(resource)}' copy-holding chain at tier " +
            $"'{Lair.Models.BudgetTierNames.Of(tier)}'. {breakdown}. Retry once the running generations finish, or raise the ceiling this chain declares in the config.",
            "budget_exhausted");

    /// <summary>The refusal for a submission naming a model the host is retiring out of its served set: the model stopped accepting new requests while its already-admitted work drains toward release. It rides the same bounded overload family — the posture that blocks is retirement, not capacity or quota — with its own code so callers can tell a transient overload from a withdrawal.</summary>
    public static EngineBusyException ForModelRetiring(string modelId) =>
        new($"model '{modelId}' no longer accepts requests: it has been removed from the served models list and is draining what already runs before its resources are released. Re-serve it by adding it back to the config's models list, or request a model id that stays listed.",
            "model_retiring");

    /// <summary>The refusal for a submission arriving while every admission slot of the model's concurrency cap is held and the model's queue of submissions waiting for a slot is already full. The limit posture named is <c>max_queued_requests</c>: raising it admits more waiters, freeing a running generation frees a slot for whoever is already waiting.</summary>
    public static EngineBusyException ForQueueFull(string modelId, int maxQueuedRequests) =>
        new($"model '{modelId}' cannot admit this request against its request queue: all admission slots of 'max_concurrent_requests' are held and {maxQueuedRequests} submissions are already waiting for one under 'max_queued_requests'. Retry once a running generation finishes, or raise max_queued_requests in the config.",
            "engine_overloaded");

    /// <summary>The refusal for a submission that waited for an admission slot longer than the model's bounded wait allows. The limit posture named is <c>queue_timeout_ms</c>: every concurrency slot stayed held for the whole wait, so raising either this ceiling or 'max_concurrent_requests' changes what a later submission can expect.</summary>
    public static EngineBusyException ForQueueTimeout(string modelId, int queueTimeoutMs) =>
        new($"model '{modelId}' cannot admit this request against its request queue: no admission slot freed within {queueTimeoutMs} ms of 'queue_timeout_ms', so the wait ended with every concurrency slot still held. Retry once a running generation finishes, or raise max_concurrent_requests in the config.",
            "engine_overloaded");
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
