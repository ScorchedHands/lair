namespace Lair.Http;

/// <summary>
/// One encoder call against a model opened in the encode-only serving posture: every input text is tokenized like an ordinary
/// prompt, forwarded through the bidirectional encoder path, and answered as one vector per input. There are no sampling knobs —
/// nothing here samples — and the pooling choice that shapes each vector belongs to the model's configuration rather than to the request.
/// </summary>
public sealed record EmbeddingRequest
{
    /// <summary>Id of the model to run, matched against <see cref="ModelInfo.Id"/>.</summary>
    public required string Model { get; init; }

    /// <summary>The input texts, in the order they must come back as. One text answers one vector; a list answers one vector per entry at its own index.</summary>
    public required IReadOnlyList<string> Inputs { get; init; }

    /// <summary>Caller-supplied correlation id carried onto monitor lifecycle rows when this engine publishes them; null lets the facade mint one.</summary>
    public string? RequestId { get; init; }
}

/// <summary>One answered embedding: the vector for one input text, and how many tokens that input's encoder forward actually consumed.</summary>
public sealed record EmbeddingVector(IReadOnlyList<float> Values, int InputTokens);

/// <summary>The vectors an encoder call answered, in request order — every entry at its own index regardless of how many composed encoder batches the inputs ran through — plus the posture those vectors were shaped with: total input tokens across the call, the pooling choice resolved from the configuration in force when the request was admitted, and whether normalization followed pooling.</summary>
public sealed record EmbeddingOutcome(
    IReadOnlyList<EmbeddingVector> Vectors,
    int TotalInputTokens,
    string PoolingMode,
    bool Normalized);
