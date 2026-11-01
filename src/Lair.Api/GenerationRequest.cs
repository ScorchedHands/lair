using Lair.Tokenizer.Templating;

namespace Lair.Api;

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

    /// <summary>True invokes <see cref="OnToken"/> with each decoded piece as it is produced.</summary>
    public bool Stream { get; init; }

    /// <summary>Streaming sink for <see cref="Stream"/>; ignored otherwise.</summary>
    public Action<string>? OnToken { get; init; }
}
