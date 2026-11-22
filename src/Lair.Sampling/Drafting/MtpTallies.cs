namespace Lair.Sampling.Drafting;

/// <summary>
/// What one drafted decode boundary produced: the authoritative target tokens emitted at this step (always exactly what a
/// non-drafted step would have sampled, in order), how many draft proposals were consulted, and where acceptance stopped.
/// A mismatch is recorded by its 0-based proposal index — the position whose draft token differed from the target's own
/// sample; that position's target token was still emitted as authoritative output. Rejected drafts at and after the
/// mismatch were never fed to the target model.
/// </summary>
public sealed record DraftStepOutcome(
    IReadOnlyList<int> EmittedTokens,
    int ProposedCount,
    int AcceptedCount)
{
    /// <summary>0-based index of the first proposal that did not match the target's own sample; null when every proposal matched or none was rejected.</summary>
    public int? FirstMismatchIndex { get; init; }

    /// <summary>True when this boundary ran with no proposals at all — an ordinary decode step through the drafted path.</summary>
    public bool NoProposals => ProposedCount == 0;

    /// <summary>True when every proposal matched and at least one was proposed.</summary>
    public bool FullyAccepted => AcceptedCount > 0 && AcceptedCount == ProposedCount;
}

/// <summary>
/// Running accept-rate accounting for one drafted decode stream: per-boundary draft/verify tallies a monitor or bench can
/// read, plus the ratio they imply. Single-writer by design — the same control-plane thread that advances the sequence owns
/// these counters, mirroring how scheduler stats are maintained.
/// </summary>
public sealed class MtpTallies
{
    /// <summary>Decode boundaries advanced through this drafted stream.</summary>
    public int StepsAdvanced { get; internal set; }

    /// <summary>Total draft tokens proposed across every boundary (the denominator of the accept rate).</summary>
    public long DraftsProposed { get; internal set; }

    /// <summary>Total draft tokens accepted — proposals that matched the target's own sample at their position.</summary>
    public long DraftTokensAccepted { get; internal set; }

    /// <summary>Boundaries where a proposal mismatched the target's sample before all proposals were consumed.</summary>
    public int MismatchedSteps { get; internal set; }

    /// <summary>Boundaries where every proposed draft was accepted (at least one proposal).</summary>
    public int FullyAcceptedSteps { get; internal set; }

    /// <summary>Boundaries that ran with an empty proposal list.</summary>
    public int NoProposalSteps { get; internal set; }

    /// <summary>Proposals consulted but never accepted — rejected at the first mismatch, including every later position of that step's window.</summary>
    public long DraftsRejected => DraftsProposed - DraftTokensAccepted;

    /// <summary>Accepted proposals over proposed drafts; 0 when nothing was ever proposed.</summary>
    public double AcceptRate => DraftsProposed == 0 ? 0.0 : (double)DraftTokensAccepted / DraftsProposed;

    /// <summary>Folds one boundary's draft/verify outcome into the running counters.</summary>
    public void Record(DraftStepOutcome outcome)
    {
        StepsAdvanced++;
        DraftsProposed += outcome.ProposedCount;
        DraftTokensAccepted += outcome.AcceptedCount;
        if (outcome.NoProposals)
        {
            NoProposalSteps++;
        }

        if (outcome.FirstMismatchIndex is not null)
        {
            MismatchedSteps++;
        }

        if (outcome.FullyAccepted)
        {
            FullyAcceptedSteps++;
        }
    }

    /// <summary>An immutable copy of the current tallies, for reporting without exposing the running counters.</summary>
    public MtpTallySnapshot Snapshot() => new(
        StepsAdvanced,
        DraftsProposed,
        DraftTokensAccepted,
        DraftsRejected,
        MismatchedSteps,
        FullyAcceptedSteps,
        NoProposalSteps,
        AcceptRate);

    /// <summary>Adds another stream's tallies into these counters (multi-sequence aggregation for host-level reporting).</summary>
    public void Absorb(MtpTallies other)
    {
        ArgumentNullException.ThrowIfNull(other);
        StepsAdvanced += other.StepsAdvanced;
        DraftsProposed += other.DraftsProposed;
        DraftTokensAccepted += other.DraftTokensAccepted;
        MismatchedSteps += other.MismatchedSteps;
        FullyAcceptedSteps += other.FullyAcceptedSteps;
        NoProposalSteps += other.NoProposalSteps;
    }
}

/// <summary>Immutable view of an <see cref="MtpTallies"/> state at one moment.</summary>
public sealed record MtpTallySnapshot(
    int StepsAdvanced,
    long DraftsProposed,
    long DraftTokensAccepted,
    long DraftsRejected,
    int MismatchedSteps,
    int FullyAcceptedSteps,
    int NoProposalSteps,
    double AcceptRate);
