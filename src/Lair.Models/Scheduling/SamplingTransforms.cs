using Lair.Sampling;

namespace Lair.Models.Scheduling;

/// <summary>
/// Logits transforms applied at the sampling sites, immediately before each draw: presence and frequency
/// penalties over the sequence's own prompt-plus-generated history, and constant per-token logit biases.
/// Each call mutates exactly the logits row it is handed — a per-sequence row of a batched forward or the
/// single sequence's prefill logits — so one sequence's transform never reaches another sequence's row, and
/// the row is fully overwritten by the next forward of that sequence.
/// </summary>
internal static class SamplingTransforms
{
    /// <summary>
    /// Subtracts <c>presencePenalty</c> from the logit of every token that appears at least once in
    /// <paramref name="history"/> and <c>frequencyPenalty × occurrences</c> from every token that appears,
    /// then adds each <see cref="SamplingParams.LogitBias"/> entry to its token's logit. Ids outside the
    /// row's range are ignored.
    /// </summary>
    public static void Apply(Span<float> row, SamplingParams p, IReadOnlyList<int> prompt, IReadOnlyList<int> generated)
    {
        if (p.PresencePenalty != 0f || p.FrequencyPenalty != 0f)
        {
            Dictionary<int, int> counts = [];
            foreach (int id in prompt)
            {
                counts[id] = counts.TryGetValue(id, out int n) ? n + 1 : 1;
            }

            foreach (int id in generated)
            {
                counts[id] = counts.TryGetValue(id, out int n) ? n + 1 : 1;
            }

            foreach ((int id, int occurrences) in counts)
            {
                if (id >= 0 && id < row.Length)
                {
                    row[id] -= p.PresencePenalty + p.FrequencyPenalty * occurrences;
                }
            }
        }

        if (p.LogitBias is not null)
        {
            foreach ((int id, float bias) in p.LogitBias)
            {
                if (id >= 0 && id < row.Length)
                {
                    row[id] += bias;
                }
            }
        }
    }
}
