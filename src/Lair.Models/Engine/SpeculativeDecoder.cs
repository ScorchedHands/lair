using Lair.Kv;
using Lair.Models.Scheduling;
using Lair.Sampling;
using Lair.Sampling.Drafting;

namespace Lair.Models.Engine;

/// <summary>
/// What one drafted decode boundary produced: the authoritative tokens this boundary emitted (the target's own samples, in
/// order — a draft token appears here only because it matched what the target itself would have produced), the draft/verify
/// tallies for that boundary, and the finish reason when the boundary ended the sequence. Null <see cref="Finished"/> means
/// the run continues at the next boundary with <see cref="SpeculativeDecoder.PendingToken"/> advanced.
/// </summary>
public sealed record DraftedAdvance(
    IReadOnlyList<int> EmittedTokens,
    DraftStepOutcome Draft,
    FinishReason? Finished);

/// <summary>The terminal outcome of a whole drafted run.</summary>
public sealed record DraftedRunResult(FinishReason Reason, IReadOnlyList<int> Tokens, MtpTallySnapshot Tallies);

/// <summary>
/// CPU-only speculative decode orchestration over an injected target-verify forward. One instance owns one sequence stream's
/// sampling authority — its own seeded <see cref="Sampler"/> and penalty/bias transforms applied over this run's history at
/// every sample, exactly as a served pipeline samples — so drafted runs reproduce the non-drafted sampler posture token for
/// token, RNG-stream continuity included.
///
/// Verification is accept-prefix: each boundary consults up to the draft head's window and forwards only authoritative
/// target tokens through <paramref name="verifyForward"/>. A proposal counts as accepted when it equals the target's own
/// sample at that position; the first mismatch ends acceptance, its target token is still emitted as authoritative output,
/// and every proposal from that position on was never fed to the model — so a rejected draft cannot write KV bytes, advance
/// positions, or disturb the sampler. Each verify forward is exactly one non-drafted decode step of this sequence: same
/// feeds (the pending token, then each accepted sample), same append order, same logits rows. The result stream is therefore
/// identical to running the same model and sampling posture without drafting, boundary grouping included — a drafted run's
/// per-token decisions are one-for-one the baseline step sequence.
/// </summary>
public sealed class SpeculativeDecoder
{
    private readonly SamplingParams _params;
    private readonly Sampler _sampler;
    private readonly IDraftHead _head;
    private readonly int _vocabSize;
    private readonly List<int> _generated = [];

    // The logits row the most recent accepted sample came from, snapshotted after transforms so a draft head sees exactly
    // what was sampled from. Heads that ignore it pay only the copy.
    private float[]? _lastSampledLogits;
    private int _pendingToken;
    private bool _started;

    public SpeculativeDecoder(SamplingParams sampling, IDraftHead head, int vocabSize)
    {
        ArgumentNullException.ThrowIfNull(sampling);
        ArgumentNullException.ThrowIfNull(head);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);

        _params = sampling;
        _sampler = new Sampler(sampling); // one RNG stream for this run's whole sequence, like a pipeline owns its sampler
        _head = head;
        _vocabSize = vocabSize;
    }

    /// <summary>Accept-rate accounting accumulated across every boundary of this run.</summary>
    public MtpTallies Tallies { get; } = new();

    /// <summary>The authoritative generated stream so far (the stop token is never part of it).</summary>
    public IReadOnlyList<int> Generated => _generated;

    /// <summary>The last accepted sample awaiting its next decode forward.</summary>
    public int PendingToken => _pendingToken;

    /// <summary>True once the first generated token has been sampled from prefill logits.</summary>
    public bool Started => _started;

    /// <summary>
    /// Samples this run's first generated token from already-computed prefill logits, mirroring how a served pipeline turns
    /// its prompt window into a decode boundary: transforms over the history so far (the prompt), one sample, and no forward.
    /// Returns null when that sample is <paramref name="eosId"/> — the sequence finished before any drafting happened, exactly
    /// as an undrafted run would finish at eos on its first logits row.
    /// </summary>
    public int? StartFromLogits(Span<float> logits, IReadOnlyList<int> promptTokens, int eosId)
    {
        ArgumentNullException.ThrowIfNull(promptTokens);

        if (_started)
        {
            throw new InvalidOperationException("this drafted run already sampled its first token; advance it with Advance instead.");
        }

        int first = SampleAuthority(logits, promptTokens);
        _lastSampledLogits = [.. logits]; // snapshot after transforms: what the sampler actually saw
        if (first == eosId)
        {
            return null;
        }

        _generated.Add(first);
        _pendingToken = first;
        _started = true;
        return first;
    }

    /// <summary>
    /// Advances one drafted decode boundary. The draft head proposes from this run's history and last sampled logits; the
    /// verify loop then runs target forwards over authoritative tokens only, accepting proposals while they match the
    /// target's own samples and stopping at the first mismatch — whose target token is emitted as ordinary output. Eos ends
    /// the boundary with <see cref="FinishReason.Eos"/> (the stop token is not emitted); exhausting the generation ceiling
    /// ends it with <see cref="FinishReason.Length"/> after that final token's normal emit-and-check, mirroring the served
    /// decode loop step for step. Requires a started run and remaining budget.
    /// </summary>
    public DraftedAdvance Advance(
        Func<int, IKvSequence, Span<float>> verifyForward,
        IKvSequence kv,
        IReadOnlyList<int> promptTokens,
        int generationTokenCeiling,
        int eosId,
        Action<int>? onTokenId = null)
    {
        ArgumentNullException.ThrowIfNull(verifyForward);
        ArgumentNullException.ThrowIfNull(kv);
        ArgumentNullException.ThrowIfNull(promptTokens);

        if (!_started)
        {
            throw new InvalidOperationException("a drafted run must sample its first token from prefill logits before advancing decode.");
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(generationTokenCeiling);
        if (_generated.Count >= generationTokenCeiling)
        {
            throw new InvalidOperationException($"this sequence already holds {_generated.Count} generated tokens at a ceiling of {generationTokenCeiling}.");
        }

        // The window never outruns the remaining budget: proposals beyond what this run may still generate are not consulted.
        int budgetLeft = generationTokenCeiling - _generated.Count;
        IReadOnlyList<int> proposalsRaw = _head.Propose(new DraftProposalContext(_vocabSize, BuildHistory(promptTokens), _lastSampledLogits));
        int window = Math.Min(proposalsRaw.Count, budgetLeft);
        List<int> proposals = new(window);
        for (int i = 0; i < window; i++)
        {
            proposals.Add(proposalsRaw[i]);
        }

        int accepted = 0;
        int comparisons = 0;
        int? mismatchIndex = null;
        FinishReason? finished = null;
        List<int> emitted = [];
        int pending = _pendingToken;

        // The first iteration always runs: a boundary with an empty window is exactly one ordinary decode step. Later
        // iterations run only to verify the next proposal position, so a fully accepted window consumes no sampling beyond
        // what its drafts covered — and every forward here feeds an authoritative token.
        while (true)
        {
            // One verify forward == one non-drafted decode step of this sequence: the fed token is authoritative (the pending
            // sample, then each accepted draft), so whatever it appends to KV is what an undrafted run would have appended.
            Span<float> logits = verifyForward(pending, kv);
            int next = SampleAuthority(logits, promptTokens);

            if (next == eosId)
            {
                finished = FinishReason.Eos; // the stop token ends the sequence and is never emitted or forwarded on
                break;
            }

            _lastSampledLogits = [.. logits]; // snapshot after transforms: what this boundary's sampler actually saw

            bool hasProposal = accepted < proposals.Count;
            if (hasProposal)
            {
                comparisons++;
                if (proposals[accepted] != next)
                {
                    mismatchIndex = accepted; // this position's draft was rejected, but the target's own sample still belongs to the output stream
                }
            }

            bool matched = hasProposal && proposals[accepted] == next;

            emitted.Add(next);
            _generated.Add(next);

            if (matched)
            {
                accepted++; // the proposal outcome is recorded even when this very token finishes the ceiling — it was consulted and verified
            }

            bool lengthFinished = _generated.Count >= generationTokenCeiling;
            if (!lengthFinished)
            {
                pending = next;
                onTokenId?.Invoke(next); // the served decode loop reports a continuing step's token, not the one that hit the ceiling
            }

            if (lengthFinished)
            {
                finished = FinishReason.Length; // same emit-then-check order as the served decode loop: this token counts, then the ceiling finishes
                break;
            }

            if (!matched)
            {
                break; // an empty window ends the boundary at one ordinary step; a mismatch ends acceptance and its later proposals were never consulted or fed
            }

            // A fully consumed window ends here: every remaining position of this run continues at the next boundary, so no
            // sampling happens beyond what the drafts covered.
            if (accepted == proposals.Count)
            {
                break;
            }
        }

        _pendingToken = pending;

        DraftStepOutcome outcome = new(emitted, comparisons, accepted) { FirstMismatchIndex = mismatchIndex };
        Tallies.Record(outcome);

        return new DraftedAdvance(emitted, outcome, finished);
    }

    /// <summary>
    /// Whole-run convenience: starts from a caller-supplied read of the prompt window's logits (the same row an undrafted
    /// pipeline samples its first token from), then advances drafted boundaries until eos or the generation ceiling. Returns
    /// the finish reason, the authoritative token stream, and the accumulated accept tallies.
    /// </summary>
    public DraftedRunResult RunDrafted(
        Func<int, IKvSequence, Span<float>> verifyForward,
        Func<Span<float>> prefillLogitsReader,
        IKvSequence kv,
        IReadOnlyList<int> promptTokens,
        int generationTokenCeiling,
        int eosId,
        Action<int>? onTokenId = null)
    {
        ArgumentNullException.ThrowIfNull(verifyForward);
        ArgumentNullException.ThrowIfNull(prefillLogitsReader);

        if (StartFromLogits(prefillLogitsReader(), promptTokens, eosId) is null)
        {
            return new DraftedRunResult(FinishReason.Eos, [], Tallies.Snapshot());
        }

        while (true)
        {
            DraftedAdvance step = Advance(verifyForward, kv, promptTokens, generationTokenCeiling, eosId, onTokenId);
            if (step.Finished is FinishReason reason)
            {
                return new DraftedRunResult(reason, [.. _generated], Tallies.Snapshot());
            }
        }
    }

    // Transforms this run's own sampling authority applies over its history before every draw — penalties scaled by this
    // sequence's prompt plus already-generated tokens and per-token logit biases added in place — identical to what a served
    // pipeline does on the logits row it samples from. The span is the model-owned forward buffer, mutated exactly as the
    // scheduler mutates its batched rows; nothing downstream re-reads it after sampling.
    private int SampleAuthority(Span<float> row, IReadOnlyList<int> promptTokens)
    {
        if (_params.HasLogitTransform)
        {
            SamplingTransforms.Apply(row, _params, promptTokens, _generated);
        }

        return _sampler.Sample(row);
    }

    // Fresh history array per proposal: prompt plus every accepted sample so far, the same view a served pipeline's
    // transform authority sees at its own sampling boundary. Built fresh rather than shared because head implementations may
    // hand it around freely within one Propose call without aliasing this run's state.
    private int[] BuildHistory(IReadOnlyList<int> promptTokens)
    {
        int[] history = new int[promptTokens.Count + _generated.Count];
        for (int i = 0; i < promptTokens.Count; i++)
        {
            history[i] = promptTokens[i];
        }

        for (int i = 0; i < _generated.Count; i++)
        {
            history[promptTokens.Count + i] = _generated[i];
        }

        return history;
    }
}
