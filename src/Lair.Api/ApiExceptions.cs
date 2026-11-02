namespace Lair.Api;

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
/// The prompt alone does not fit the model's context window, so generation cannot start.
/// </summary>
public sealed class ContextLengthExceededException : Exception
{
    public ContextLengthExceededException(long promptTokens, long limit)
        : base($"prompt is {promptTokens} tokens but the model context is {limit} tokens, and at least one token " +
               "must stay free for generation. Shorten the prompt or raise max_context_tokens and kv_blocks in the config.")
    {
        PromptTokens = promptTokens;
        Limit = limit;
    }

    /// <summary>Token count of the rejected prompt.</summary>
    public long PromptTokens { get; }

    /// <summary>The context limit the prompt was measured against.</summary>
    public long Limit { get; }
}

/// <summary>
/// The KV pool cannot back this request's prompt plus token budget alongside what is already running.
/// </summary>
public sealed class EngineBusyException : Exception
{
    public EngineBusyException()
        : base("the engine cannot admit this request: its prompt plus token budget exceeds the token slots the " +
               "KV pool can back. Retry once the running generations finish, or raise kv_blocks in the config.")
    {
    }
}
