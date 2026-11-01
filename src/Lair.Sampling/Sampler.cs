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
