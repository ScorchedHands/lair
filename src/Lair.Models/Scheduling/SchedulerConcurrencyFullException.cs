namespace Lair.Models.Scheduling;

/// <summary>
/// Raised when a sequence is submitted while every concurrency slot the scheduler's cap allows is already held by admitted sequences.
/// This is the admission refusal that names its limit kind as the model's concurrency quota: the submission is turned away at once
/// rather than queued behind what already runs, and nothing about it disturbs an in-flight pipeline. The scheduler keeps serving every
/// admitted sequence normally; only this rejected submission fails.
/// </summary>
public sealed class SchedulerConcurrencyFullException : Exception
{
    public SchedulerConcurrencyFullException(int maxConcurrentRequests, string? modelName = null)
        : base(BuildMessage(maxConcurrentRequests, modelName))
    {
        MaxConcurrentRequests = maxConcurrentRequests;
        ModelName = modelName;
    }

    /// <summary>The cap whose slots were all held when the submission was rejected.</summary>
    public int MaxConcurrentRequests { get; }

    /// <summary>The model this scheduler serves, when its construction named one: every refusal then names it alongside the limit kind that blocked.</summary>
    public string? ModelName { get; }

    private static string BuildMessage(int maxConcurrentRequests, string? modelName) =>
        modelName is null
            ? $"the engine cannot admit this sequence against its concurrency quota: all {maxConcurrentRequests} " +
              "admission slots of the configured 'max_concurrent_requests' cap are held by sequences already admitted. " +
              "Wait for a running sequence to finish, or raise max_concurrent_requests in the config."
            : $"model '{modelName}' cannot admit this sequence against its concurrency quota: all {maxConcurrentRequests} " +
              "admission slots of its 'max_concurrent_requests' cap are held by sequences already admitted. " +
              "Wait for a running sequence to finish, or raise max_concurrent_requests in the config.";
}
