using Lair.Sampling;

namespace Lair.Models.Scheduling;

/// <summary>
/// One sequence handed to the <see cref="GenerationScheduler"/>: the prompt to prefill, the token budget,
/// the sampling knobs, and the end-of-sequence id. All fields are token-level; text-level stop handling
/// stays in the caller that decodes the streamed token ids.
/// </summary>
public sealed record ScheduledRequest
{
    /// <summary>Prompt token ids, prefilled in chunks of at most the scheduler's max-batch-tokens size.</summary>
    public required int[] PromptTokens { get; init; }

    /// <summary>Maximum number of tokens to generate (the token budget; eos does not count against it).</summary>
    public required int MaxTokens { get; init; }

    /// <summary>Per-sequence sampling knobs; the scheduler builds one <see cref="Sampler"/> from them.</summary>
    public required SamplingParams Sampling { get; init; }

    /// <summary>Token id that ends the sequence when sampled; never forwarded into the sequence.</summary>
    public required int EosId { get; init; }

    /// <summary>Streaming sink invoked on the scheduler thread with each sampled token id, before it is forwarded.</summary>
    public Action<int>? OnTokenId { get; init; }

    /// <summary>
    /// When set, the scheduler captures a <see cref="TokenLogprob"/> entry for every generated token from its own
    /// sampling step, reporting this many top candidates; null captures nothing.
    /// </summary>
    public int? LogProbTopN { get; init; } = null;
}
