using Lair.Tokenizer.Templating;

namespace Lair.Http;

/// <summary>
/// One generation call. Chat requests render <see cref="Messages"/> through the model's chat template;
/// completion requests send <see cref="Prompt"/> verbatim. Sampling knobs map directly onto the sampler's
/// parameters; <see cref="OnToken"/> receives each decoded piece while the sequence is still decoding when
/// <see cref="Stream"/> is set.
/// </summary>
public sealed record GenerationRequest
{
    /// <summary>Id of the model to run, matched against <see cref="ModelInfo.Id"/>.</summary>
    public required string Model { get; init; }

    /// <summary>True renders <see cref="Messages"/> through the chat template; false sends <see cref="Prompt"/> as-is.</summary>
    public bool IsChat { get; init; }

    /// <summary>Raw prompt text for completion requests.</summary>
    public string? Prompt { get; init; }

    /// <summary>Conversation turns for chat requests.</summary>
    public IReadOnlyList<ChatMessage>? Messages { get; init; }

    /// <summary>Maximum number of tokens to generate.</summary>
    public int MaxTokens { get; init; } = 128;

    /// <summary>True when the caller supplied <see cref="MaxTokens"/> explicitly. False lets a resolved model budget pair replace today's default ceiling without overloading the OpenAI field.</summary>
    public bool MaxTokensWasSupplied { get; init; } = true;

    /// <summary>Per-request override of the model's reasoning-token ceiling, carried as the value admitted for this request — already clamped to what the loaded model declares. Null carries no override: today's unbudgeted behavior.</summary>
    public int? MaxReasoningTokens { get; init; }

    /// <summary>Per-request override of the model's response-token ceiling, carried as the value admitted for this request — already clamped to what the loaded model declares. Null carries no override: today's unbudgeted behavior.</summary>
    public int? MaxResponseTokens { get; init; }

    /// <summary>Softmax temperature; 0 selects greedy decoding.</summary>
    public float Temperature { get; init; } = 1.0f;

    /// <summary>Nucleus mass kept after the top-k cut, in (0, 1].</summary>
    public float TopP { get; init; } = 0.9f;

    /// <summary>Number of highest-probability candidates kept before the top-p cut; 0 disables the cut.</summary>
    public int TopK { get; init; } = 40;

    /// <summary>RNG seed for reproducible sampling; null uses a fresh, unseeded generator.</summary>
    public long? Seed { get; init; }

    /// <summary>Literal substrings that end generation when they appear in the decoded text.</summary>
    public IReadOnlyList<string>? Stop { get; init; }

    /// <summary>Additive penalty on each token already present in prompt plus generated output. In [-2, 2]; 0 disables it.</summary>
    public float PresencePenalty { get; init; } = 0f;

    /// <summary>Additive penalty scaled by how often each token appears in prompt plus generated output. In [-2, 2]; 0 disables it.</summary>
    public float FrequencyPenalty { get; init; } = 0f;

    /// <summary>Per-token logit offsets added at every sampling step, keyed by token string or numeric token id; values in [-100, 100]. Null disables them.</summary>
    public IReadOnlyDictionary<string, float>? LogitBias { get; init; }

    /// <summary>True reports per-generated-token log probabilities on the result and inside streamed deltas.</summary>
    public bool LogProbs { get; init; } = false;

    /// <summary>Highest-probability candidates reported alongside each sampled token. In [0, 20]; only used when <see cref="LogProbs"/> is set.</summary>
    public int TopLogprobs { get; init; } = 0;

    /// <summary>Declared tools whose schemas are injected into the rendered chat prompt and whose names the fenced-JSON detector accepts. Null declares none. Chat requests only.</summary>
    public IReadOnlyList<ToolDefinition>? Tools { get; init; }

    /// <summary>How tool use is constrained: auto (default), none (no injection, no detection), or a forced declared function. Null means auto.</summary>
    public ToolChoiceSelection? ToolChoice { get; init; }

    /// <summary>True invokes <see cref="OnToken"/> with each decoded piece as it is produced.</summary>
    public bool Stream { get; init; }

    /// <summary>Streaming sink for <see cref="Stream"/>; ignored otherwise.</summary>
    public Action<string>? OnToken { get; init; }

    /// <summary>Correlation id the monitor stream keys this generation's lifecycle events on — the API host supplies its own <c>x-lair-request-id</c>; null lets the engine mint one. Not part of the OpenAI wire body.</summary>
    public string? RequestId { get; init; }
}
