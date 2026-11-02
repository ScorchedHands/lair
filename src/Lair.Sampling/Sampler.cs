namespace Lair.Sampling;

/// <summary>
/// Vocabulary sampler over the logits span produced by a forward pass. Candidate selection
/// follows the llama.cpp order: drop non-finite and banned tokens, softmax at the configured
/// temperature, cut to the top-k survivors, then shrink to the top-p nucleus of those
/// survivors, renormalize, and draw with the seeded generator. Greedy mode (temperature &lt;= 0)
/// takes the argmax, breaking ties toward the lowest token id so decoding stays deterministic.
/// </summary>
public sealed class Sampler
{
    private readonly SamplingParams _params;
    private readonly Random _random;

    public Sampler(SamplingParams p)
    {
        ArgumentNullException.ThrowIfNull(p);
        if (p.Temperature < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(p), $"Temperature must be >= 0, got {p.Temperature}.");
        }

        if (p.TopK < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(p), $"TopK must be >= 0 (0 disables the top-k cut), got {p.TopK}.");
        }

        if (p.TopP <= 0f || p.TopP > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(p), $"TopP must be in (0, 1], got {p.TopP}.");
        }

        if (p.PresencePenalty is < -2f or > 2f)
        {
            throw new ArgumentOutOfRangeException(nameof(p), $"PresencePenalty must be in [-2, 2], got {p.PresencePenalty}.");
        }

        if (p.FrequencyPenalty is < -2f or > 2f)
        {
            throw new ArgumentOutOfRangeException(nameof(p), $"FrequencyPenalty must be in [-2, 2], got {p.FrequencyPenalty}.");
        }

        if (p.LogitBias is not null)
        {
            foreach ((int id, float bias) in p.LogitBias)
            {
                if (bias is < -100f or > 100f)
                {
                    throw new ArgumentOutOfRangeException(nameof(p), $"LogitBias for token {id} must be in [-100, 100], got {bias}.");
                }
            }
        }

        _params = p;
        _random = p.Seed.HasValue ? new Random(MixSeed(p.Seed.Value)) : new Random();
    }

    // Random only accepts an int seed, so fold the long seed's halves together to keep the
    // full 64-bit input distinguishable down to a 2^-32 collision rate.
    private static int MixSeed(long seed) => unchecked((int)(seed ^ (seed >> 32)));

    /// <summary>Draws the next token from the full vocabulary.</summary>
    /// <exception cref="InvalidOperationException">Every logit was non-finite, so no candidate survived.</exception>
    public int Sample(ReadOnlySpan<float> logits) => Sample(logits, bannedTokens: null);

    /// <summary>
    /// Draws the next token, excluding <paramref name="bannedTokens"/> before candidate selection.
    /// </summary>
    /// <exception cref="InvalidOperationException">No candidate survived the non-finite and banned filters.</exception>
    public int Sample(ReadOnlySpan<float> logits, IReadOnlySet<int>? bannedTokens)
    {
        List<Candidate> candidates = CollectCandidates(logits, bannedTokens);
        if (candidates.Count == 0)
        {
            throw new InvalidOperationException(NoSurvivorsMessage(logits, bannedTokens));
        }

        if (_params.Greedy)
        {
            return GreedyWinner(candidates);
        }

        ApplySoftmax(candidates, _params.Temperature);
        SortByProb(candidates);
        ApplyTopK(candidates);
        ApplyTopP(candidates);
        return Draw(candidates);
    }

    /// <summary>
    /// The <paramref name="n"/> highest-probability tokens with their softmax probabilities at
    /// temperature 1 over the finite logits, for logprobs reporting. Non-finite logits are
    /// excluded; fewer than <paramref name="n"/> valid tokens yields the full list.
    /// </summary>
    public IReadOnlyList<(int TokenId, double Prob)> TopLogits(ReadOnlySpan<float> logits, int n)
    {
        if (n < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(n), $"n must be >= 0, got {n}.");
        }

        List<Candidate> candidates = CollectCandidates(logits, banned: null);
        if (candidates.Count == 0 || n == 0)
        {
            return [];
        }

        ApplySoftmax(candidates, 1.0f);
        SortByProb(candidates);
        if (candidates.Count > n)
        {
            candidates.RemoveRange(n, candidates.Count - n);
        }

        return [.. candidates.Select(c => (c.Id, c.Prob))];
    }

    /// <summary>
    /// The log-probability entry for <paramref name="chosenId"/> under the distribution actually sampled
    /// from at this step, reported over the full vocabulary. The reported distribution is the softmax of
    /// the given logits at the configured temperature — after any temperature scaling but before the
    /// top-k and top-p cuts — so the numbers describe the sampling distribution, not the truncated one;
    /// greedy mode (temperature &lt;= 0) reports at temperature 1, the argmax it performs being equivalent
    /// to that distribution's argmax. <paramref name="topN"/> highest-probability tokens are returned
    /// sorted descending (probability ties toward the lower token id); when the chosen token is not among
    /// them it is appended, so it is always present. Non-finite logits are excluded.
    /// </summary>
    /// <exception cref="InvalidOperationException">No finite logit existed, so the distribution is empty.</exception>
    public (double LogProb, IReadOnlyList<(int TokenId, double LogProb)> Top) LogProbsFor(
        ReadOnlySpan<float> logits,
        int chosenId,
        int topN)
    {
        if (topN < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(topN), $"topN must be >= 0, got {topN}.");
        }

        List<Candidate> candidates = CollectCandidates(logits, banned: null);
        if (candidates.Count == 0)
        {
            throw new InvalidOperationException("no candidate tokens: every logit was non-finite");
        }

        ApplySoftmax(candidates, _params.Greedy ? 1.0f : _params.Temperature);
        SortByProb(candidates);

        Candidate? chosen = null;
        int chosenRank = -1;
        int limit = Math.Min(topN, candidates.Count);
        for (int i = 0; i < candidates.Count; i++)
        {
            if (candidates[i].Id == chosenId)
            {
                chosen = candidates[i];
                chosenRank = i;
                break;
            }
        }

        if (chosen is null)
        {
            throw new ArgumentException(
                $"token {chosenId} is not in the reported distribution (its logit is non-finite).",
                nameof(chosenId));
        }

        List<(int TokenId, double LogProb)> top = [.. candidates.Take(limit).Select(c => (c.Id, Math.Log(c.Prob)))];
        if (chosenRank >= limit)
        {
            top.Add((chosenId, Math.Log(chosen.Prob)));
        }

        return (Math.Log(chosen.Prob), top);
    }

    private static List<Candidate> CollectCandidates(ReadOnlySpan<float> logits, IReadOnlySet<int>? banned)
    {
        List<Candidate> candidates = [];
        for (int id = 0; id < logits.Length; id++)
        {
            if (float.IsFinite(logits[id]) && (banned is null || !banned.Contains(id)))
            {
                candidates.Add(new Candidate(id, logits[id]));
            }
        }

        return candidates;
    }

    private static string NoSurvivorsMessage(ReadOnlySpan<float> logits, IReadOnlySet<int>? banned)
    {
        bool anyFinite = false;
        foreach (float v in logits)
        {
            if (float.IsFinite(v))
            {
                anyFinite = true;
                break;
            }
        }

        return anyFinite
            ? "no candidate tokens: every finite logit was banned"
            : "all logits invalid";
    }

    private static int GreedyWinner(List<Candidate> candidates)
    {
        // Ties go to the lowest token id: candidates are collected in ascending id order and
        // the comparison is strict >, so the first (lowest-id) holder of the maximum wins.
        Candidate best = candidates[0];
        for (int i = 1; i < candidates.Count; i++)
        {
            if (candidates[i].Logit > best.Logit)
            {
                best = candidates[i];
            }
        }

        return best.Id;
    }

    private static void ApplySoftmax(List<Candidate> candidates, float temperature)
    {
        float maxLogit = float.NegativeInfinity;
        foreach (Candidate c in candidates)
        {
            if (c.Logit > maxLogit)
            {
                maxLogit = c.Logit;
            }
        }

        double sum = 0.0;
        foreach (Candidate c in candidates)
        {
            c.Prob = Math.Exp((c.Logit - maxLogit) / temperature);
            sum += c.Prob;
        }

        foreach (Candidate c in candidates)
        {
            c.Prob /= sum;
        }
    }

    private static void SortByProb(List<Candidate> candidates)
    {
        candidates.Sort(static (a, b) =>
        {
            int byProb = b.Prob.CompareTo(a.Prob);
            return byProb != 0 ? byProb : a.Id.CompareTo(b.Id);
        });
    }

    private void ApplyTopK(List<Candidate> candidates)
    {
        if (_params.TopK > 0 && candidates.Count > _params.TopK)
        {
            candidates.RemoveRange(_params.TopK, candidates.Count - _params.TopK);
        }
    }

    private void ApplyTopP(List<Candidate> candidates)
    {
        if (_params.TopP >= 1f)
        {
            return;
        }

        double cumulative = 0.0;
        int nucleus = candidates.Count;
        for (int i = 0; i < candidates.Count; i++)
        {
            cumulative += candidates[i].Prob;
            if (cumulative >= _params.TopP)
            {
                nucleus = i + 1;
                break;
            }
        }

        if (candidates.Count > nucleus)
        {
            candidates.RemoveRange(nucleus, candidates.Count - nucleus);
        }
    }

    private int Draw(List<Candidate> candidates)
    {
        double total = 0.0;
        foreach (Candidate c in candidates)
        {
            total += c.Prob;
        }

        double target = _random.NextDouble() * total;
        double cumulative = 0.0;
        foreach (Candidate c in candidates)
        {
            cumulative += c.Prob;
            if (target < cumulative)
            {
                return c.Id;
            }
        }

        return candidates[^1].Id;
    }

    private sealed class Candidate
    {
        public Candidate(int id, float logit)
        {
            Id = id;
            Logit = logit;
        }

        public int Id { get; }
        public float Logit { get; }
        public double Prob { get; set; }
    }
}
