using Lair.Sampling;

namespace Lair.Tests;

/// <summary>
/// Sampler tests: greedy argmax and tie-breaking, seeded determinism across instances and runs,
/// temperature extremes, the top-k then top-p candidate cascade, banned-token exclusion,
/// non-finite logit handling, frequency agreement with a known distribution, and TopLogits.
/// </summary>
[Trait("Category", "Sampling")]
public class SamplingTests
{
    private static readonly double[] Expected4 = [0.643_914_46, 0.236_811_69, 0.087_144_35, 0.032_129_50];

    [Fact]
    public void Greedy_PicksArgmax_AndTieGoesToLowestId()
    {
        Sampler sampler = new(new SamplingParams { Temperature = 0f });
        Assert.Equal(2, sampler.Sample([1.0f, 3.0f, 5.0f, 2.0f]));

        // Tie on the maximum: lowest id wins.
        Assert.Equal(1, sampler.Sample([4.0f, 7.0f, 7.0f, 4.0f]));
    }

    [Fact]
    public void SeededSampler_IsReproducibleWithinRunAndAcrossInstances()
    {
        float[] logits = [2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f];
        SamplingParams p = new() { Temperature = 0.8f, TopK = 0, TopP = 1.0f, Seed = 12345 };

        Sampler a = new(p);
        int[] first = [.. Enumerable.Range(0, 100).Select(_ => a.Sample(logits))];

        Sampler b = new(p);
        int[] second = [.. Enumerable.Range(0, 100).Select(_ => b.Sample(logits))];

        Assert.Equal(first, second);

        // The same seed must reproduce the same sequence on a fresh instance later.
        Sampler c = new(p);
        int[] third = [.. Enumerable.Range(0, 100).Select(_ => c.Sample(logits))];
        Assert.Equal(first, third);

        // A different seed must diverge (otherwise the generator is a no-op).
        Sampler d = new(new SamplingParams { Temperature = 0.8f, TopK = 0, TopP = 1.0f, Seed = 999 });
        int[] other = [.. Enumerable.Range(0, 100).Select(_ => d.Sample(logits))];
        Assert.NotEqual(first, other);
    }

    [Fact]
    public void Temperature_NearZeroBehavesGreedy_AndUnitSoftmaxSumsToOne()
    {
        Sampler nearZero = new(new SamplingParams { Temperature = 0.001f, TopK = 0, TopP = 1f, Seed = 7 });
        float[] logits = [1.0f, 2.0f, 3.0f, 4.0f];
        for (int i = 0; i < 100; i++)
        {
            Assert.Equal(3, nearZero.Sample(logits));
        }

        // T=1 over the full vocabulary: TopLogits reports the softmax, which must sum to 1.
        Sampler unit = new(new SamplingParams { Temperature = 1f });
        double sum = unit.TopLogits(logits, logits.Length).Sum(t => t.Prob);
        Assert.Equal(1.0, sum, 12);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => new Sampler(new SamplingParams { Temperature = -0.5f }));
    }

    [Fact]
    public void TopK_OneIsArgmax_AndTopK_TwoOnlyEverPicksTheTopTwo()
    {
        Sampler argmax = new(new SamplingParams { Temperature = 1f, TopK = 1, TopP = 1f, Seed = 3 });
        float[] logits = [1.0f, 5.0f, 2.0f, 0.5f, 3.0f];
        for (int i = 0; i < 100; i++)
        {
            Assert.Equal(1, argmax.Sample(logits));
        }

        Sampler top2 = new(new SamplingParams { Temperature = 1f, TopK = 2, TopP = 1f, Seed = 42 });
        HashSet<int> seen = [];
        for (int i = 0; i < 1_000; i++)
        {
            seen.Add(top2.Sample(logits));
        }

        Assert.Equal([1, 4], seen.OrderBy(x => x));
    }

    [Fact]
    public void TopP_NucleusPrefixIsRespected()
    {
        // Softmax over exp = {1, 0.5, 0.25, 0.0625, 0.0625} (sum 1.875) gives probs
        // {0.5333, 0.2667, 0.1333, 0.0333, 0.0333}; the nucleus at p = 0.9 is exactly
        // {0, 1, 2} because the cumulative sum crosses 0.9 at token 2 (0.9333).
        Sampler nucleus = new(new SamplingParams { Temperature = 1f, TopK = 0, TopP = 0.9f, Seed = 11 });
        float[] logits = [0.0f, -0.693_147_2f, -1.386_294_4f, -2.772_588_7f, -2.772_588_7f];

        HashSet<int> seen = [];
        for (int i = 0; i < 1_000; i++)
        {
            seen.Add(nucleus.Sample(logits));
        }

        Assert.True(seen.IsSubsetOf([0, 1, 2]), $"nucleus leaked outside the prefix: {string.Join(',', seen)}");
        Assert.Equal(3, seen.Count);
    }

    [Fact]
    public void BannedTokens_AreNeverReturned_AndBanningEverythingThrows()
    {
        Sampler sampler = new(new SamplingParams { Temperature = 1f, TopK = 0, TopP = 1f, Seed = 5 });
        float[] logits = [3.0f, 2.0f, 1.0f, 0.0f];
        HashSet<int> banned = [0, 1];

        HashSet<int> seen = [];
        for (int i = 0; i < 500; i++)
        {
            seen.Add(sampler.Sample(logits, banned));
        }

        Assert.Equal([2, 3], seen.OrderBy(x => x));

        // Greedy respects bans too: with the argmax banned, the next-highest wins.
        Sampler greedy = new(new SamplingParams { Temperature = 0f });
        Assert.Equal(1, greedy.Sample(logits, new HashSet<int>([0])));

        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(
            () => sampler.Sample(logits, new HashSet<int>([0, 1, 2, 3])));
        Assert.Contains("banned", ex.Message);
    }

    [Fact]
    public void NonFiniteLogits_AreExcluded_AndAllInvalidThrows()
    {
        Sampler greedy = new(new SamplingParams { Temperature = 0f });
        float[] mixed = [float.NaN, 1.0f, float.NegativeInfinity, 2.0f, float.PositiveInfinity];
        Assert.Equal(3, greedy.Sample(mixed));

        Sampler sampler = new(new SamplingParams { Temperature = 1f, TopK = 0, TopP = 1f, Seed = 9 });
        HashSet<int> seen = [];
        for (int i = 0; i < 200; i++)
        {
            seen.Add(sampler.Sample(mixed));
        }

        Assert.Equal([1, 3], seen.OrderBy(x => x));

        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(
            () => sampler.Sample([float.NaN, float.NegativeInfinity, float.PositiveInfinity]));
        Assert.Contains("all logits invalid", ex.Message);
    }

    [Fact]
    public void SeededFrequencies_MatchKnownDistributionWithinOnePercent()
    {
        Sampler sampler = new(new SamplingParams { Temperature = 1f, TopK = 0, TopP = 1f, Seed = 20_261_101 });
        float[] logits = [2.0f, 1.0f, 0.0f, -1.0f];

        const int samples = 20_000;
        int[] counts = new int[4];
        for (int i = 0; i < samples; i++)
        {
            counts[sampler.Sample(logits)]++;
        }

        for (int id = 0; id < 4; id++)
        {
            double observed = counts[id] / (double)samples;
            Assert.True(
                Math.Abs(observed - Expected4[id]) < 0.01,
                $"token {id}: observed {observed:P3} vs expected {Expected4[id]:P3}");
        }
    }

    [Fact]
    public void TopLogits_IsOrderedAndProbsSumToOne()
    {
        Sampler sampler = new(new SamplingParams { Temperature = 0.5f });
        float[] logits = [1.0f, 3.0f, 2.0f, 0.0f];

        IReadOnlyList<(int TokenId, double Prob)> top = sampler.TopLogits(logits, 3);

        Assert.Equal([1, 2, 0], top.Select(t => t.TokenId));
        Assert.True(top[0].Prob >= top[1].Prob && top[1].Prob >= top[2].Prob);

        double full = sampler.TopLogits(logits, logits.Length).Sum(t => t.Prob);
        Assert.Equal(1.0, full, 12);

        // n larger than the vocabulary returns every valid token, still summing to 1.
        IReadOnlyList<(int TokenId, double Prob)> clamped = sampler.TopLogits([float.NaN, 1.0f, 2.0f], 10);
        Assert.Equal([2, 1], clamped.Select(t => t.TokenId));
        Assert.Equal(1.0, clamped.Sum(t => t.Prob), 12);
    }
}
