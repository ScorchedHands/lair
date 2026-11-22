namespace Lair.Sampling.Drafting;

/// <summary>The outcome of resolving whether a model may draft, and why.</summary>
public sealed record DraftHeadDecision(bool Enabled, string Reason)
{
    /// <summary>True when drafting is refused — the reason names what is missing or wrong.</summary>
    public bool Refused => !Enabled;
}

/// <summary>
/// The eligibility gate for speculative decoding on one model. A draft head must exist before any sequence drafts: a host
/// supplies it as an <see cref="IDraftHead"/> (a side artifact, a stand-in model, or a scripted plan) when the target's own
/// file carries no multi-token-prediction tensors. With neither available, drafting is refused with a reason naming the
/// model and what would make it eligible — never silently replaced by unverified free-running drafts.
/// </summary>
public static class MtpDraftPolicy
{
    /// <summary>
    /// Resolves one model's drafting posture from what the host configured: an explicit head wins; without a head, native
    /// draft tensors (as reported by the loader for this artifact) are required; with neither, refusal.
    /// </summary>
    public static DraftHeadDecision Resolve(string modelName, bool hasNativeDraftTensors, IDraftHead? configuredHead)
    {
        ArgumentNullException.ThrowIfNull(modelName);

        if (configuredHead is not null)
        {
            return new DraftHeadDecision(
                true,
                $"model '{modelName}': drafting enabled through a host-supplied draft head of cap {configuredHead.MaxDraftTokens}.");
        }

        if (hasNativeDraftTensors)
        {
            // Native heads are consumed by an engine-side provider that binds them to this seam; until such a provider is
            // wired for the family, declaring tensors alone does not enable drafting here.
            return new DraftHeadDecision(
                false,
                $"model '{modelName}': the artifact declares draft-head tensors but no head was supplied through the drafting seam.");
        }

        return new DraftHeadDecision(
            false,
            $"model '{modelName}': MTP drafting is refused — this model carries no draft-head tensors and none was configured; " +
            "supply an IDraftHead (a side artifact or a stand-in head) to enable speculative decoding.");
    }
}
