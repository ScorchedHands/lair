namespace Lair.Models.Scheduling;

/// <summary>
/// A sequence admitted to the <see cref="GenerationScheduler"/>. The handle carries the completion task and
/// lets the submitter request cancellation; tokens stream through the request's <see cref="ScheduledRequest.OnTokenId"/>
/// hook on the scheduler thread.
/// </summary>
public sealed class SequenceHandle
{
    private readonly Sequence _sequence;

    internal SequenceHandle(Sequence sequence) => _sequence = sequence;

    /// <summary>Completes with the sequence's outcome once it finishes.</summary>
    public Task<SequenceOutcome> Completion => _sequence.Outcome.Task;

    /// <summary>
    /// Requests cancellation: the sequence finishes with <see cref="FinishReason.Canceled"/> at the next step
    /// boundary and its KV blocks are released. Cancelling a finished sequence is a no-op.
    /// </summary>
    public void Cancel() => _sequence.RequestCancel();
}
