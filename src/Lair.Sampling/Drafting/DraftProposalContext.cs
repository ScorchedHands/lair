namespace Lair.Sampling.Drafting;

/// <summary>
/// What a draft head sees at one decode boundary: the sequence's vocabulary size, its full token history (prompt tokens
/// followed by already-generated tokens), and the logits row that produced the most recent accepted sample — which is what
/// heads built on top of an LM readout consume. The arrays are borrowed for the duration of one <see cref="IDraftHead.Propose"/>
/// call: a head must not retain or mutate them. Exposed as spans so heads read them without copying, while the context itself
/// stays usable in ordinary delegate and interface positions.
/// </summary>
public readonly struct DraftProposalContext
{
    private readonly int[] _history;
    private readonly float[]? _lastSampledLogits;

    public DraftProposalContext(int vocabSize, int[] history, float[]? lastSampledLogits)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);
        ArgumentNullException.ThrowIfNull(history);
        VocabSize = vocabSize;
        _history = history;
        _lastSampledLogits = lastSampledLogits;
    }

    /// <summary>Vocabulary size of the target model whose sequence is being drafted.</summary>
    public int VocabSize { get; }

    /// <summary>Prompt tokens followed by already-generated tokens — everything accepted so far, in order.</summary>
    public ReadOnlySpan<int> History => _history.AsSpan();

    /// <summary>The logits row the most recent accepted token was sampled from (empty before the first generated token).</summary>
    public ReadOnlySpan<float> LastSampledLogits => _lastSampledLogits is null ? [] : _lastSampledLogits.AsSpan();
}
