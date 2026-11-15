namespace Lair.Models.Scheduling;

/// <summary>
/// Raised when a sequence is submitted while the scheduler's KV pool cannot back its prompt plus token budget. This is one of two
/// admission refusals this control plane reports, each naming the limit kind that blocked it: here kv_capacity (token slots still free
/// in the pool); <see cref="SchedulerConcurrencyFullException"/> names the concurrency quota slot instead. The scheduler keeps running;
/// only the rejected submission fails.
/// </summary>
public sealed class SchedulerFullException : Exception
{
    public SchedulerFullException(long freeSlots, string? modelName = null)
        : base(BuildMessage(freeSlots, modelName))
    {
        FreeSlots = freeSlots;
        ModelName = modelName;
    }

    /// <summary>Token slots the pool had free when the submission was rejected.</summary>
    public long FreeSlots { get; }

    /// <summary>The model this scheduler serves, when its construction named one: every refusal then names it alongside the limit kind that blocked.</summary>
    public string? ModelName { get; }

    private static string BuildMessage(long freeSlots, string? modelName) =>
        modelName is null
            ? $"the KV pool cannot admit this sequence: it needs more than the {freeSlots} token slots still " +
              "free in the pool. Wait for running sequences to finish, or raise kv_blocks in the config."
            : $"model '{modelName}' cannot admit this sequence against its kv_capacity: it needs more than the " +
              $"{freeSlots} token slots still free in the KV pool. Wait for running sequences to finish, or raise kv_blocks in the config.";
}
