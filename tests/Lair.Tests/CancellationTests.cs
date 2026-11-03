using Lair.Http;
using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Cancellation-isolation tests over the real SmolLM F16 fixture: aborting one sequence mid-flight releases every KV block it held,
/// leaves each other in-flight sequence's output identical to a run where nothing was cancelled, and keeps the batch loop admitting
/// and completing later work. Block reclamation is asserted against <see cref="BlockPool.FreeBlocks"/> on schedulers built directly
/// over a known pool with no prefix cache attached: published prefixes stay referenced by their cache entries after their publisher
/// finishes, so only a cache-free scheduler can measure what one sequence's release actually returns to the free list.
/// </summary>
[Trait("Category", "Cancel")]
public sealed class CancellationTests
{
    // The aborted request must stream this many tokens before its cancellation lands: deep enough that every sequence sharing the
    // decode steps is demonstrably mid-decode, shallow enough that no 48-token run can finish between observing it and cancelling.
    private const int AbortAfterTokens = 6;

    // Wall-clock guard for every wait in these tests: a wait that outlives it fails as an unfinished sequence rather than hanging.
    private const int WaitBudgetMs = 30_000;

    // Token budget every request in this suite runs with: long enough that an abort always lands mid-flight.
    private const int MaxTokens = 48;

    private static readonly string[] ConcurrentPrompts =
    [
        "Once upon a time in a land far away,",
        "It was a dark and stormy night, and",
        "My favorite food is",
    ];

    // The same long prompt the chunked-prefill suite uses: ~407 tokens prefilled eight at a step spans dozens of steps, so a
    // cancellation issued right after admission lands while its prompt is still mid-chunk.
    private const string LongPrompt = "The quick brown fox jumps over the lazy dog while the patient cat watches the rain. ";

    [Fact]
    public async Task MidBatchAbort_AbortedSequenceIsCanceled_AndOthersMatchANoCancelRun()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GenerationService service = ConcurrencyTests.OpenService(kvBlocks: 64, maxContextTokens: 512);

        CancellationTokenSource abort = new();
        int[] streamed = [0, 0, 0];

        Task<GenerationResult>[] running =
        [
            service.GenerateAsync(StreamRequest(0, streamed), CancellationToken.None),
            service.GenerateAsync(StreamRequest(1, streamed), abort.Token),
            service.GenerateAsync(StreamRequest(2, streamed), CancellationToken.None),
        ];

        // All three sequences share every decode step, so once the doomed one has streamed a few tokens all three are mid-decode.
        Assert.True(
            await WaitUntilAsync(() => Volatile.Read(ref streamed[1]) >= AbortAfterTokens),
            $"the aborted request streamed only {streamed[1]} token(s) before finishing; it was never cancelled mid-flight");

        abort.Cancel();

        GenerationResult first = await running[0];
        GenerationResult third = await running[2];
        await Assert.ThrowsAsync<OperationCanceledException>(() => running[1]);

        // The survivors are bit-identical to the same prompts running on a fresh service where nothing was cancelled.
        using GenerationService baseline = ConcurrencyTests.OpenService(kvBlocks: 64, maxContextTokens: 512);
        foreach ((GenerationResult actual, int promptIndex) in new (GenerationResult, int)[] { (first, 0), (third, 2) })
        {
            GenerationResult expected = await baseline.GenerateAsync(
                ConcurrencyTests.CompletionRequest(ConcurrentPrompts[promptIndex], MaxTokens),
                CancellationToken.None);

            Assert.Equal(expected.Text, actual.Text);
            Assert.Equal(expected.FinishReason, actual.FinishReason);
            Assert.Equal(expected.CompletionTokens, actual.CompletionTokens);
            Assert.Equal(expected.PromptTokens, actual.PromptTokens);
        }
    }

    [Fact]
    public async Task StopLiteral_EndsTheSequenceEarly_AndReportsStop()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GenerationService service = ConcurrencyTests.OpenService(kvBlocks: 64, maxContextTokens: 512);

        GenerationResult plain = await service.GenerateAsync(
            ConcurrencyTests.CompletionRequest(ConcurrentPrompts[0], MaxTokens),
            CancellationToken.None);

        Assert.False(string.IsNullOrEmpty(plain.Text));
        Assert.True(plain.CompletionTokens >= 4, "the fixture run was too short to show an early stop");

        // A literal the run itself proves occurs: the sequence must end the moment it streams it, not run out its budget.
        string literal = plain.Text[..Math.Min(2, plain.Text.Length)];

        GenerationResult stopped = await service.GenerateAsync(
            new GenerationRequest
            {
                Model = "smollm-concurrency",
                IsChat = false,
                Prompt = ConcurrentPrompts[0],
                MaxTokens = MaxTokens,
                Temperature = 0f,
                Stop = [literal],
            },
            CancellationToken.None);

        Assert.Equal("stop", stopped.FinishReason);
        Assert.True(
            stopped.CompletionTokens < plain.CompletionTokens,
            $"the stop literal ended the sequence after {stopped.CompletionTokens} of {plain.CompletionTokens} tokens");
        Assert.True(plain.Text.StartsWith(stopped.Text, StringComparison.Ordinal), "the stopped text is not a prefix of the full run");
        Assert.False(stopped.Text.Contains(literal, StringComparison.Ordinal), "the stop literal survived in the reported text");
    }

    [Fact]
    public async Task MidDecodeAbort_ReleasesEveryBlockOfTheCanceledSequence()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using SchedulerHarness harness = new(capacityBlocks: 32, maxBatchTokens: 512, maxContextTokens: 512);
        int freeBefore = harness.Pool.FreeBlocks;

        CancellationTokenSource abort = new();
        int streamed = 0;

        SequenceHandle handle = harness.Scheduler.Submit(
            new ScheduledRequest
            {
                PromptTokens = [.. harness.Tokenizer.Encode("Once upon a time in a land far away,")],
                MaxTokens = 48,
                Sampling = new SamplingParams { Temperature = 0f },
                EosId = harness.Tokenizer.EosId,
                OnTokenId = _ => Volatile.Write(ref streamed, streamed + 1),
            },
            abort.Token);

        Assert.True(
            await WaitUntilAsync(() => Volatile.Read(ref streamed) >= AbortAfterTokens),
            $"the cancelled sequence streamed only {streamed} token(s) before finishing; it was never cancelled mid-decode");

        abort.Cancel();
        SequenceOutcome outcome = await handle.Completion;

        Assert.Equal(FinishReason.Canceled, outcome.FinishReason);
        Assert.True(outcome.CompletionTokens >= AbortAfterTokens, "the outcome reports fewer tokens than were streamed before the abort");

        // The blocks the sequence held at its abort — its prompt block plus every block its decode crossed — are all back.
        Assert.Equal(freeBefore, harness.Pool.FreeBlocks);
        Assert.Equal(0, harness.Pool.UsedBlocks);
    }

    [Fact]
    public async Task AbortDuringPrefill_ReleasesItsBlocks_AndTheConcurrentSequenceIsUnaffected()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using SchedulerHarness harness = new(capacityBlocks: 32, maxBatchTokens: 8, maxContextTokens: 512);
        int freeBefore = harness.Pool.FreeBlocks;

        int[] prompt = [.. harness.Tokenizer.Encode(string.Concat(Enumerable.Repeat(LongPrompt, 24)))];
        Assert.True(prompt.Length >= 380 && prompt.Length <= 460, $"long prompt fixture built {prompt.Length} tokens; expected around 400");

        CancellationTokenSource abort = new();

        // Eight prompt tokens per step: by the time the pool has handed this sequence three blocks its prefill is demonstrably
        // mid-chunk, while the short sequence alongside it has been admitted and is prefilled or decoding in the same steps.
        SequenceHandle longHandle = harness.Scheduler.Submit(
            new ScheduledRequest
            {
                PromptTokens = prompt,
                MaxTokens = 48,
                Sampling = new SamplingParams { Temperature = 0f },
                EosId = harness.Tokenizer.EosId,
            },
            abort.Token);

        int[] shortPrompt = [.. harness.Tokenizer.Encode("My favorite food is")];
        SequenceHandle shortHandle = harness.Scheduler.Submit(
            new ScheduledRequest
            {
                PromptTokens = shortPrompt,
                MaxTokens = 48,
                Sampling = new SamplingParams { Temperature = 0f },
                EosId = harness.Tokenizer.EosId,
            },
            CancellationToken.None);

        Assert.True(
            await WaitUntilAsync(() => harness.Pool.UsedBlocks >= 3),
            "the long-prompt sequence never crossed into its third prefill block; the abort did not land mid-prefill");

        abort.Cancel();

        SequenceOutcome longOutcome = await longHandle.Completion;
        SequenceOutcome shortOutcome = await shortHandle.Completion;

        Assert.Equal(FinishReason.Canceled, longOutcome.FinishReason);
        Assert.Equal(0, longOutcome.CompletionTokens); // its prompt never finished prefilling, so it sampled nothing
        Assert.True(shortOutcome.CompletionTokens > 0, "the concurrent short sequence produced no tokens");

        // Both sequences are done: every block either sequence touched is back in the free list.
        Assert.Equal(freeBefore, harness.Pool.FreeBlocks);
        Assert.Equal(0, harness.Pool.UsedBlocks);

        // The short sequence's output is identical to running it alone on a fresh engine.
        using SchedulerHarness baseline = new(capacityBlocks: 32, maxBatchTokens: 512, maxContextTokens: 512);
        SequenceOutcome expected = await baseline.Scheduler.Submit(
            new ScheduledRequest
            {
                PromptTokens = shortPrompt,
                MaxTokens = 48,
                Sampling = new SamplingParams { Temperature = 0f },
                EosId = baseline.Tokenizer.EosId,
            },
            CancellationToken.None).Completion;

        Assert.Equal(expected.GeneratedTokens, shortOutcome.GeneratedTokens);
    }

    [Fact]
    public async Task CancelAtAdmission_FinishesCanceled_AndTheLoopKeepsServing()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using SchedulerHarness harness = new(capacityBlocks: 32, maxBatchTokens: 512, maxContextTokens: 512);
        int freeBefore = harness.Pool.FreeBlocks;

        CancellationTokenSource alreadyCancelled = new();
        alreadyCancelled.Cancel();

        int[] prompt = [.. harness.Tokenizer.Encode("Once upon a time in a land far away,")];

        SequenceHandle canceled = harness.Scheduler.Submit(
            new ScheduledRequest
            {
                PromptTokens = prompt,
                MaxTokens = 48,
                Sampling = new SamplingParams { Temperature = 0f },
                EosId = harness.Tokenizer.EosId,
            },
            alreadyCancelled.Token);

        // A sequence whose token was cancelled before it ever ran finishes at the first step boundary it reaches, having
        // allocated nothing: the loop survives it and the next submission runs normally.
        SequenceOutcome outcome = await canceled.Completion;
        Assert.Equal(FinishReason.Canceled, outcome.FinishReason);
        Assert.Equal(0, outcome.CompletionTokens);
        Assert.Equal(freeBefore, harness.Pool.FreeBlocks);
        Assert.Equal(0, harness.Pool.UsedBlocks);

        SequenceOutcome next = await harness.Scheduler.Submit(
            new ScheduledRequest
            {
                PromptTokens = prompt,
                MaxTokens = 6,
                Sampling = new SamplingParams { Temperature = 0f },
                EosId = harness.Tokenizer.EosId,
            },
            CancellationToken.None).Completion;

        Assert.Equal(FinishReason.Length, next.FinishReason);
        Assert.Equal(6, next.CompletionTokens);
        Assert.Equal(freeBefore, harness.Pool.FreeBlocks);
    }

    // ---------------------------------------------------------------- helpers

    private static GenerationRequest StreamRequest(int index, int[] streamed) => new()
    {
        Model = "smollm-concurrency",
        IsChat = false,
        Prompt = ConcurrentPrompts[index],
        MaxTokens = MaxTokens,
        Temperature = 0f,
        Stream = true,
        // One counter slot per request, written only by the scheduler thread as that sequence samples.
        OnToken = _ => Interlocked.Increment(ref streamed[index]),
    };

    private static async Task<bool> WaitUntilAsync(Func<bool> condition)
    {
        System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();
        while (!condition())
        {
            if (clock.ElapsedMilliseconds > WaitBudgetMs)
            {
                return false;
            }

            await Task.Delay(50);
        }

        return true;
    }

    /// <summary>
    /// One real-model engine with a <see cref="GenerationScheduler"/> and <see cref="BlockPool"/> built directly on it, so pool
    /// free-block counts can be read while sequences run. No prefix cache is attached: reclamation is measured against the raw
    /// pool, where a finished sequence's blocks are its only holders.
    /// </summary>
    private sealed class SchedulerHarness : IDisposable
    {
        public GgufFile Gguf { get; }
        public GgufTokenizer Tokenizer { get; }
        public CpuBackend Backend { get; }
        public TransformerModel Model { get; }
        public BlockPool Pool { get; }
        public GenerationScheduler Scheduler { get; }

        public SchedulerHarness(int capacityBlocks, int maxBatchTokens, int maxContextTokens)
        {
            string modelPath = ModelFixture.Resolve() ?? throw new InvalidOperationException(
                $"GGUF fixture '{ModelFixture.SmolLmRelativePath}' was not found walking up from the test assembly directory.");

            Gguf = GgufFile.Open(modelPath);
            Tokenizer = GgufTokenizer.FromGguf(Gguf);
            Backend = new CpuBackend(threads: 1);
            Model = TransformerModel.Open(Gguf, Backend, maxContextTokens);
            Pool = new BlockPool(
                blockTokens: 16,
                layerCount: Model.Desc.LayerCount,
                kvHeads: Model.Desc.KvHeadCount,
                headDim: Model.Desc.HeadDim,
                capacityBlocks: capacityBlocks,
                Backend);
            Scheduler = new GenerationScheduler(Model, Pool, maxBatchTokens);
        }

        public void Dispose()
        {
            // Same release order the service uses: the scheduler returns every sequence's blocks before the pool they live in.
            Scheduler.Dispose();
            Pool.Dispose();
            Model.Dispose();
            Gguf.Dispose();
            Backend.Dispose();
        }
    }
}
