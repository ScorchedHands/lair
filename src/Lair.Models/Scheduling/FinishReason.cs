namespace Lair.Models.Scheduling;

/// <summary>Why a scheduled sequence stopped advancing.</summary>
public enum FinishReason
{
    /// <summary>The sequence sampled its end-of-sequence token; eos was never forwarded into the sequence.</summary>
    Eos,

    /// <summary>The sequence exhausted its token budget.</summary>
    Length,

    /// <summary>Cancellation was observed between steps; the sequence's blocks were released.</summary>
    Canceled,

    /// <summary>Forwarding the sequence failed (for example the shared KV pool ran out of blocks); only this sequence failed.</summary>
    Failed,
}
