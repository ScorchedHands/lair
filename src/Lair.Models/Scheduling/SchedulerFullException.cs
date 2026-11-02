namespace Lair.Models.Scheduling;

/// <summary>
/// Raised when a sequence is submitted while the scheduler's KV pool cannot back its prompt plus token
/// budget. The scheduler keeps running; only the rejected submission fails.
/// </summary>
public sealed class SchedulerFullException : Exception
{
    public SchedulerFullException(long freeSlots)
        : base($"the KV pool cannot admit this sequence: it needs more than the {freeSlots} token slots still " +
               "free in the pool. Wait for running sequences to finish, or raise kv_blocks in the config.")
    {
        FreeSlots = freeSlots;
    }

    /// <summary>Token slots the pool had free when the submission was rejected.</summary>
    public long FreeSlots { get; }
}
