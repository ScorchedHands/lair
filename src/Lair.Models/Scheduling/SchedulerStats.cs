namespace Lair.Models.Scheduling;

/// <summary>
/// Live engine counters a <see cref="GenerationScheduler"/> maintains while it runs. Every value is a plain
/// cumulative count updated at step boundaries and sequence completion on the scheduler thread only — no
/// allocation, no lock beyond the gate the step already holds — so observing them costs nothing measurable to
/// generation itself. Readers see each field as an atomically published long; pairs of fields are read-your-writes
/// consistent per field but not snapped transactionally across fields.
/// </summary>
public sealed class SchedulerStats
{
    // Written on the scheduler thread only (every step and every finish runs under the gate); volatile reads make
    // each value visible to observer threads without locking them out of the gate.

    /// <summary>Decode steps the loop has executed since the scheduler opened.</summary>
    public long DecodeSteps;

    /// <summary>Total microseconds spent inside decode windows across all executed steps.</summary>
    public long DecodeWindowMicrosTotal;

    /// <summary>Sum over decode steps of how many sequences advanced in that step — divide by <see cref="DecodeSteps"/> for average batch occupancy.</summary>
    public long WindowSequenceSum;

    /// <summary>Prefill chunks forwarded since the scheduler opened.</summary>
    public long PrefillChunksIssued;

    /// <summary>Prompt tokens prefilled (forwarded through the engine) across all those chunks.</summary>
    public long PrefillTokensForwarded;

    /// <summary>Tokens advanced by decode steps — every generated token except a final eos, so it matches outcomes' completion counts.</summary>
    public long DecodeTokensAdvanced;

    /// <summary>Prompt tokens sampled out of prefill logits but never forwarded: the first generated token of each sequence that prefilled to completion and did not end on eos there.</summary>
    public long PrefillSampledTokens;

    /// <summary>Tokens produced by sequences that ended in a failed step (batch or context faults).</summary>
    public long FailedSequenceTokens;

    /// <summary>Sequences admitted into the active set since the scheduler opened.</summary>
    public long SequencesAdmitted;

    /// <summary>Sequences whose leading prompt blocks were adopted from the prefix cache at admission.</summary>
    public long CacheHitSequences;

    /// <summary>Prompt tokens skipped by those adoptions — recomputed nowhere, prefilled as if present.</summary>
    public long CacheSkippedTokens;

    /// <summary>Cumulative count over sequences of every full prompt block published into the prefix cache after prefill (renewals included).</summary>
    public long CachePublishedBlocksTotal;

    /// <summary>Prompt tokens covered by those publications.</summary>
    public long CachePublishedTokensTotal;

    /// <summary>Sequences that finished on eos, length budget, cancellation, or a fault since the scheduler opened.</summary>
    public long FinishedEos;
    public long FinishedLength;
    public long FinishedCanceled;
    public long FinishedFailed;

    /// <summary>Decode sequences currently advancing in steps — sampled at each step boundary by observers.</summary>
    public int ActiveDecodingSequences;

    /// <summary>All live sequences (prefilling and decoding) as of the last completed step boundary.</summary>
    public int ActiveSequenceCount;

    internal SchedulerStats()
    {
    }
}
