using Lair.Kv;
using Lair.Sampling;

namespace Lair.Models.Scheduling;

/// <summary>
/// One sequence in flight inside the <see cref="GenerationScheduler"/>: its request, its own sampler (so its
/// RNG stream never interleaves with another sequence's), its paged KV cache, and its position in the fixed
/// prefill-then-decode state machine. The scheduler thread is the only writer of the mutable fields; the
/// cancellation flag is the one field any thread may set.
/// </summary>
internal sealed class Sequence
{
    private int _cancelRequested;

    public Sequence(ScheduledRequest request, CancellationToken ct, BlockPool pool, long id)
    {
        Request = request;
        Cancellation = ct;
        Id = id;
        Sampler = new Sampler(request.Sampling);
        Kv = new SequenceKv(pool);
        Outcome = new TaskCompletionSource<SequenceOutcome>(TaskCreationOptions.RunContinuationsAsynchronously);
        Handle = new SequenceHandle(this);
    }

    public long Id { get; }

    public ScheduledRequest Request { get; }

    public CancellationToken Cancellation { get; }

    public Sampler Sampler { get; }

    public SequenceKv Kv { get; }

    public TaskCompletionSource<SequenceOutcome> Outcome { get; }

    public SequenceHandle Handle { get; }

    /// <summary>Prompt tokens forwarded so far; the sequence prefills while this is below the prompt length.</summary>
    public int PromptOffset;

    /// <summary>The sampled token awaiting its next decode step; valid once decoding starts.</summary>
    public int PendingToken;

    public readonly List<int> Generated = [];

    public bool IsPrefilling => PromptOffset < Request.PromptTokens.Length;

    public bool IsDecoding => PromptOffset == Request.PromptTokens.Length;

    /// <summary>True once the outcome is published; the scheduler removes finished sequences from its working set.</summary>
    public bool IsFinished => Outcome.Task.IsCompleted;

    /// <summary>True while the sequence still has token budget left to advance.</summary>
    public bool HasBudget => Generated.Count < Request.MaxTokens;

    public void RequestCancel() => Volatile.Write(ref _cancelRequested, 1);

    public bool CancelRequested => Volatile.Read(ref _cancelRequested) != 0;
}
