namespace Lair.Models.Scheduling;

/// <summary>
/// The caller-facing handle onto one pipeline admitted to the <see cref="GenerationScheduler"/>. It carries the completion task
/// and lets the submitter request cancellation; tokens stream through the request's <see cref="ScheduledRequest.OnTokenId"/> hook
/// on the scheduler thread. A pipeline that is still waiting for an advancement slot reports nothing until it starts advancing —
/// its handle behaves exactly like one already stepping, including cancelling before its first step.
/// </summary>
public sealed class PipelineHandle
{
    private readonly RequestPipeline _pipeline;

    internal PipelineHandle(RequestPipeline pipeline) => _pipeline = pipeline;

    /// <summary>Completes with the pipeline's outcome once it finishes.</summary>
    public Task<SequenceOutcome> Completion => _pipeline.Outcome.Task;

    /// <summary>
    /// Requests cancellation: the pipeline finishes with <see cref="FinishReason.Canceled"/> at the next step boundary and its KV
    /// blocks are released. Cancelling a finished pipeline is a no-op, as is cancelling one that never got an advancement slot.
    /// </summary>
    public void Cancel() => _pipeline.RequestCancel();
}
