using System.Diagnostics;
using Lair.Http;
using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Continuous-batching tests over the real SmolLM F16 fixture: eight concurrent requests through one service
/// must produce outputs identical to running them one after another (greedy and seeded), inter-token latency
/// stays in budget while sequences share decode steps, and KV exhaustion is reported as capacity rather than
/// as a concurrency refusal.
/// </summary>
[Trait("Category", "Concurrency")]
public sealed class ConcurrencyTests
{
    private const string ModelName = "smollm-concurrency";
    private const int MaxTokens = 32;

    // Inter-token budget for a shared decode step on the dev box: the acceptance line for batching latency.
    private const long ItlBudgetMs = 400;

    private static readonly string[] Prompts =
    [
        "Once upon a time in a land far away,",
        "It was a dark and stormy night, and",
        "My favorite food is",
        "In the morning I like to drink",
        "Three tips for staying healthy are:",
        "The best way to learn programming is",
        "Dear diary, today I went to",
        "Once upon a time there was a little cat who",
    ];

    [Fact]
    public async Task EightConcurrentRequests_MatchSequentialOutputs_Greedy()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GenerationService service = OpenService(kvBlocks: 64, maxContextTokens: 512);

        GenerationResult[] sequential = await RunSequentialAsync(service, GreedyRequest);
        GenerationResult[] concurrent = await RunConcurrentAsync(service, GreedyRequest);

        AssertOutputsMatch(sequential, concurrent, "greedy");
    }

    [Fact]
    public async Task EightConcurrentRequests_MatchSequentialOutputs_Seeded()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GenerationService service = OpenService(kvBlocks: 64, maxContextTokens: 512);

        GenerationResult[] sequential = await RunSequentialAsync(service, SeededRequest);
        GenerationResult[] concurrent = await RunConcurrentAsync(service, SeededRequest);

        AssertOutputsMatch(sequential, concurrent, "seeded");
    }

    [Fact]
    public async Task EightConcurrentRequests_InterTokenLatencyStaysInBudget()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GenerationService service = OpenService(kvBlocks: 64, maxContextTokens: 512);

        Stopwatch clock = Stopwatch.StartNew();
        List<List<long>> gaps = [];
        for (int i = 0; i < Prompts.Length; i++)
        {
            gaps.Add([]);
        }

        GenerationRequest Build(int index) => new()
        {
            Model = ModelName,
            IsChat = false,
            Prompt = Prompts[index],
            MaxTokens = MaxTokens,
            Temperature = 0f,
            Stream = true,
            OnToken = _ =>
            {
                // Callbacks run on the scheduler thread, one sequence at a time, so each sequence's own gap
                // list has a single writer.
                List<long> sequenceGaps = gaps[index];
                long stamp = clock.ElapsedMilliseconds;
                if (sequenceGaps.Count > 0)
                {
                    sequenceGaps[^1] = stamp - sequenceGaps[^1];
                }

                sequenceGaps.Add(stamp); // the placeholder becomes the next gap once the next token lands
            },
        };

        GenerationResult[] concurrent = await RunConcurrentAsync(service, Build);

        List<long> allGaps = [];
        for (int i = 0; i < Prompts.Length; i++)
        {
            Assert.False(string.IsNullOrEmpty(concurrent[i].Text), $"sequence {i} produced no text");

            List<long> sequenceGaps = gaps[i];
            Assert.True(sequenceGaps.Count >= 2, $"sequence {i} streamed only {sequenceGaps.Count} token(s)");
            allGaps.AddRange(sequenceGaps[..^1]); // the trailing placeholder is not a gap
        }

        allGaps.Sort();
        long p95 = allGaps[(allGaps.Count * 95) / 100];
        Assert.True(
            p95 <= ItlBudgetMs,
            $"inter-token p95 was {p95} ms across {allGaps.Count} gaps; budget is {ItlBudgetMs} ms");
    }

    [Fact]
    public async Task KvPoolExhausted_SecondSubmissionIsEngineBusy()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        // One 16-slot block: the first request crosses every block its sequence will ever need, so the
        // second submission has nothing left to admit.
        using GenerationService service = OpenService(kvBlocks: 1, maxContextTokens: 128);

        Task<GenerationResult> admitted = service.GenerateAsync(
            CompletionRequest("Hello there", maxTokens: 12),
            CancellationToken.None);

        EngineBusyException busy = await Assert.ThrowsAsync<EngineBusyException>(() => service.GenerateAsync(
            CompletionRequest("hi", maxTokens: 4),
            CancellationToken.None));

        Assert.Contains("KV pool", busy.Message);

        GenerationResult first = await admitted;
        Assert.Equal(12, first.CompletionTokens);
        Assert.Equal("length", first.FinishReason);
    }

    [Fact]
    public async Task SchedulerRejectsSubmissionThatOutgrowsThePool()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        using CpuBackend backend = new(threads: 1);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128);
        using BlockPool pool = new(
            blockTokens: 16,
            layerCount: model.Desc.LayerCount,
            kvHeads: model.Desc.KvHeadCount,
            headDim: model.Desc.HeadDim,
            capacityBlocks: 1,
            backend);
        using GenerationScheduler scheduler = new(model, pool, maxBatchTokens: 512);

        int[] prompt = [.. tokenizer.Encode("Hello there")];

        // Prompt plus budget crosses more blocks than the single-block pool can ever hand out.
        SchedulerFullException full = Assert.Throws<SchedulerFullException>(() => scheduler.Submit(
            new ScheduledRequest
            {
                PromptTokens = prompt,
                MaxTokens = 32,
                Sampling = new SamplingParams { Temperature = 0f },
                EosId = tokenizer.EosId,
            },
            CancellationToken.None));
        Assert.Equal(16, full.FreeSlots);

        // A submission that fits the pool's blocks is admitted and runs to its token budget.
        SequenceHandle handle = scheduler.Submit(
            new ScheduledRequest
            {
                PromptTokens = prompt,
                MaxTokens = 6,
                Sampling = new SamplingParams { Temperature = 0f },
                EosId = tokenizer.EosId,
            },
            CancellationToken.None);

        SequenceOutcome outcome = await handle.Completion;
        Assert.Equal(FinishReason.Length, outcome.FinishReason);
        Assert.Equal(6, outcome.CompletionTokens);
        Assert.Equal(prompt.Length, outcome.PromptTokens);
    }

    // ---------------------------------------------------------------- helpers

    internal static GenerationService OpenService(int kvBlocks, int maxContextTokens, int maxBatchTokens = 512)
    {
        string modelPath = ModelFixture.Resolve() ?? throw new InvalidOperationException(
            $"GGUF fixture '{ModelFixture.SmolLmRelativePath}' was not found walking up from the test assembly directory.");

        return GenerationService.Open(new LairConfig
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig { MaxBatchTokens = maxBatchTokens },
            Models =
            [
                new ModelConfig
                {
                    Name = ModelName,
                    Path = modelPath,
                    MaxContextTokens = maxContextTokens,
                    KvBlocks = kvBlocks,
                },
            ],
        });
    }

    internal static GenerationRequest CompletionRequest(string prompt, int maxTokens) => new()
    {
        Model = ModelName,
        IsChat = false,
        Prompt = prompt,
        MaxTokens = maxTokens,
        Temperature = 0f,
    };

    private static GenerationRequest GreedyRequest(int index) => CompletionRequest(Prompts[index], MaxTokens);

    private static GenerationRequest SeededRequest(int index) => new()
    {
        Model = ModelName,
        IsChat = false,
        Prompt = Prompts[index],
        MaxTokens = MaxTokens,
        Temperature = 0.8f,
        Seed = 1234,
    };

    private static async Task<GenerationResult[]> RunSequentialAsync(GenerationService service, Func<int, GenerationRequest> build)
    {
        GenerationResult[] results = new GenerationResult[Prompts.Length];
        for (int i = 0; i < Prompts.Length; i++)
        {
            results[i] = await service.GenerateAsync(build(i), CancellationToken.None).ConfigureAwait(false);
        }

        return results;
    }

    private static async Task<GenerationResult[]> RunConcurrentAsync(GenerationService service, Func<int, GenerationRequest> build)
    {
        Task<GenerationResult>[] running = [.. Enumerable.Range(0, Prompts.Length).Select(i => service.GenerateAsync(build(i), CancellationToken.None))];
        return await Task.WhenAll(running).ConfigureAwait(false);
    }

    private static void AssertOutputsMatch(GenerationResult[] sequential, GenerationResult[] concurrent, string label)
    {
        Assert.Equal(sequential.Length, concurrent.Length);
        for (int i = 0; i < sequential.Length; i++)
        {
            Assert.Equal(sequential[i].Text, concurrent[i].Text);
            Assert.Equal(sequential[i].FinishReason, concurrent[i].FinishReason);
            Assert.Equal(sequential[i].CompletionTokens, concurrent[i].CompletionTokens);
            Assert.Equal(sequential[i].PromptTokens, concurrent[i].PromptTokens);
        }

        Assert.True(
            concurrent.All(result => !string.IsNullOrEmpty(result.Text)),
            $"{label}: concurrent run produced an empty completion");
    }
}

/// <summary>
/// Prefill-protection test over the real SmolLM F16 fixture: a long prompt prefills in bounded chunks while
/// short requests run alongside it, and their inter-token latency stays in budget. The deployment target
/// serves 8K-token prompts; against this engine's measured per-token forward cost on the dev box (the plan
/// baseline is ~70–105 ms/token single-stream), a chunk that keeps short-request gaps under the 400 ms p95
/// budget is only about ten tokens wide. A ~400-token prompt split into chunks of at most 8 — prefilled side
/// by side with shared decode steps for the short requests across every step of its prefill — reproduces that
/// protection mechanism, scaled down from an 8K prompt to this fixture's context and speed.
/// </summary>
[Trait("Category", "Prefill")]
public sealed class PrefillProtectionTests
{
    private const int MaxBatchTokens = 8;
    private const int LongMaxTokens = 32;
    private const int ShortMaxTokens = 32;
    private const long ItlBudgetMs = 400;

    private static readonly string[] ShortPrompts =
    [
        "Once upon a time in a land far away,",
        "My favorite food is",
        "Dear diary, today I went to",
    ];

    [Fact]
    public async Task ChunkedLongPrefill_KeepsShortRequestsCorrectAndInLatencyBudget()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        string longPrompt = BuildLongPrompt(tokenizer);

        using GenerationService service = ConcurrencyTests.OpenService(
            kvBlocks: 256,
            maxContextTokens: 4096,
            maxBatchTokens: MaxBatchTokens);

        // The long request is admitted first, so its prompt prefills chunk by chunk while the short
        // requests queue behind it and then share decode steps with it.
        Task<GenerationResult> longRun = service.GenerateAsync(
            ConcurrencyTests.CompletionRequest(longPrompt, LongMaxTokens),
            CancellationToken.None);

        Stopwatch clock = Stopwatch.StartNew();
        List<List<long>> gaps = [[], [], []];
        Task<GenerationResult>[] shortRuns =
        [
            .. Enumerable.Range(0, ShortPrompts.Length).Select(index => service.GenerateAsync(
                new GenerationRequest
                {
                    Model = "smollm-concurrency",
                    IsChat = false,
                    Prompt = ShortPrompts[index],
                    MaxTokens = ShortMaxTokens,
                    Temperature = 0f,
                    Stream = true,
                    OnToken = _ =>
                    {
                        List<long> sequenceGaps = gaps[index];
                        long stamp = clock.ElapsedMilliseconds;
                        if (sequenceGaps.Count > 0)
                        {
                            sequenceGaps[^1] = stamp - sequenceGaps[^1];
                        }

                        sequenceGaps.Add(stamp);
                    },
                },
                CancellationToken.None)),
        ];

        GenerationResult[] shortResults = await Task.WhenAll(shortRuns);

        List<long> allGaps = [];
        for (int i = 0; i < ShortPrompts.Length; i++)
        {
            Assert.False(string.IsNullOrEmpty(shortResults[i].Text), $"short request {i} produced no text");
            Assert.True(gaps[i].Count >= 2, $"short request {i} streamed only {gaps[i].Count} token(s)");
            allGaps.AddRange(gaps[i][..^1]);
        }

        allGaps.Sort();
        long p95 = allGaps[(allGaps.Count * 95) / 100];
        Assert.True(
            p95 <= ItlBudgetMs,
            $"inter-token p95 was {p95} ms across {allGaps.Count} gaps while a {tokenizer.Encode(longPrompt).Count}-token prompt was prefilled; budget is {ItlBudgetMs} ms");

        // Correct outputs: the same short requests run one after another on a fresh engine.
        using GenerationService baseline = ConcurrencyTests.OpenService(kvBlocks: 256, maxContextTokens: 4096);
        for (int i = 0; i < ShortPrompts.Length; i++)
        {
            GenerationResult expected = await baseline.GenerateAsync(
                ConcurrencyTests.CompletionRequest(ShortPrompts[i], ShortMaxTokens),
                CancellationToken.None);

            Assert.Equal(expected.Text, shortResults[i].Text);
            Assert.Equal(expected.FinishReason, shortResults[i].FinishReason);
        }

        GenerationResult longResult = await longRun;
        Assert.False(string.IsNullOrEmpty(longResult.Text), "the long-prompt request produced no text");
        Assert.Equal(LongMaxTokens, longResult.CompletionTokens);
    }

    private static string BuildLongPrompt(GgufTokenizer tokenizer)
    {
        const string line = "The quick brown fox jumps over the lazy dog while the patient cat watches the rain. ";
        string prompt = string.Concat(Enumerable.Repeat(line, 24));

        int tokens = tokenizer.Encode(prompt).Count;
        Assert.True(
            tokens >= 380 && tokens <= 460,
            $"long prompt fixture built {tokens} tokens; expected around 400");

        return prompt;
    }
}
