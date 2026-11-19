using System.Diagnostics;
using Lair.Http;

namespace Lair.Tests;

/// <summary>
/// Request-queue admission behavior over the real SmolLM F16 fixture: a submission arriving while every concurrency slot is held waits for its
/// model's queue seat and completes normally when one frees, a submission past the configured depth answers the bounded overload refusal naming the
/// model and <c>max_queued_requests</c>, and a submission still waiting when its engine begins retiring answers the retirement refusal promptly
/// rather than outlasting its ceiling for a slot that will never free. The sequences holding slots park on their own streamed tokens so every wait is
/// observed against an admission decision that was actually in force, not against a run that happened to finish first.
/// </summary>
[Trait("Category", "Scheduling")]
[Trait("Category", "Api")]
[Collection("model-loaded")] // shares one serialized lane with the real-model suites: two engines loading at once starves the timing these gates measure
public sealed class RequestQueueTests
{
    private const int HoldTokens = 32;

    [Fact]
    public async Task BoundedQueue_QueuesOneSubmission_RefusesBeyondDepth_AndAdmitsQueuedWorkWhenTheSlotFrees()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using GenerationService service = ConcurrencyTests.OpenService(
            kvBlocks: 64,
            maxContextTokens: 512,
            maxConcurrentRequests: 1,
            maxQueuedRequests: 1); // one seat may wait; nothing beyond it queues

        Task<GenerationResult> holding = await StartHoldingSequenceAsync(service);

        Task<GenerationResult> queued = service.GenerateAsync(ConcurrencyTests.CompletionRequest("queued behind the held slot", 8), CancellationToken.None);
        Assert.True(
            await WaitForQueueDepthAsync(service, atLeast: 1, TimeSpan.FromSeconds(30)),
            "the second submission never entered this engine's request queue while its only concurrency slot was held");

        // A third arrival finds the depth already full: it answers the bounded overload refusal naming this model and its configured depth, without
        // disturbing the running sequence or the queued one that holds the seat.
        EngineBusyException busy = await Assert.ThrowsAsync<EngineBusyException>(
            () => service.GenerateAsync(ConcurrencyTests.CompletionRequest("beyond the queue depth", 4), CancellationToken.None));
        Assert.Equal("engine_overloaded", busy.Code);
        Assert.Contains(service.Model.Id, busy.Message);
        Assert.Contains("max_queued_requests", busy.Message);

        GenerationResult held = await holding;
        GenerationResult admittedAfterWaiting = await queued;

        Assert.False(string.IsNullOrEmpty(held.Text), "the sequence holding the slot produced no text");
        Assert.False(string.IsNullOrEmpty(admittedAfterWaiting.Text), "the queued submission produced no text after its slot freed");
        Assert.Equal(0, service.QueuedRequestCount); // every seat released as its submission was admitted

        // The posture governs admission, not a permanent wall: with the slots free again, an ordinary submission is admitted normally.
        GenerationResult refilled = await service.GenerateAsync(ConcurrencyTests.CompletionRequest("hi", 4), CancellationToken.None);
        Assert.False(string.IsNullOrEmpty(refilled.Text));
    }

    [Fact]
    public async Task RetiringEngine_RefusesQueuedSubmission_Promptly_WithoutWaitingOutItsCeiling()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        // No queue_timeout_ms key here: a submission left waiting would sit out the documented default ceiling, so an answer that arrives well inside it
        // can only have come from the engine telling its queued submissions it has stopped accepting new work.
        using GenerationService service = ConcurrencyTests.OpenService(
            kvBlocks: 64,
            maxContextTokens: 512,
            maxConcurrentRequests: 1,
            maxQueuedRequests: 2);

        Task<GenerationResult> holding = await StartHoldingSequenceAsync(service);

        Task<GenerationResult> queued = service.GenerateAsync(ConcurrencyTests.CompletionRequest("queued behind the held slot", 8), CancellationToken.None);
        Assert.True(
            await WaitForQueueDepthAsync(service, atLeast: 1, TimeSpan.FromSeconds(30)),
            "the submission never entered this engine's request queue while its only concurrency slot was held");

        service.MarkRetiring(); // what a served-model removal does when its drain begins: new submissions stop being accepted here

        Stopwatch clock = Stopwatch.StartNew();
        EngineBusyException retiring = await Assert.ThrowsAsync<EngineBusyException>(() => queued);
        clock.Stop();

        Assert.Equal("model_retiring", retiring.Code);
        Assert.Contains(service.Model.Id, retiring.Message);
        Assert.True(
            clock.ElapsedMilliseconds < 5_000,
            $"the queued submission waited {clock.ElapsedMilliseconds} ms for a retirement answer instead of stopping when its engine began draining");
        Assert.Equal(0, service.QueuedRequestCount); // the seat this submission held released with its refusal

        GenerationResult admitted = await holding;
        Assert.False(string.IsNullOrEmpty(admitted.Text), "already-admitted work was disturbed by retiring the model");
    }

    // ---------------------------------------------------------------- helpers

    // Starts a sequence that proves it was admitted (its first streamed token) and then parks on its own steps, so gates below observe admission state
    // against a slot that is demonstrably held rather than racing whatever order two submissions happened to take.
    private static async Task<Task<GenerationResult>> StartHoldingSequenceAsync(GenerationService service)
    {
        TaskCompletionSource<bool> firstToken = new(TaskCreationOptions.RunContinuationsAsynchronously);

        bool parkedOnce = false;
        GenerationRequest holding = new()
        {
            Model = service.Model.Id,
            IsChat = false,
            Prompt = "Once upon a time in a land far away,",
            MaxTokens = HoldTokens,
            Temperature = 0f,
            Stream = true,
            OnToken = _ =>
            {
                firstToken.TrySetResult(true);

                if (!parkedOnce)
                {
                    parkedOnce = true;
                    Thread.Sleep(700); // hold this slot long enough for a queued submission and its refusal to be observed against it
                }
            },
        };

        Task<GenerationResult> task = service.GenerateAsync(holding, CancellationToken.None);

        if (await Task.WhenAny(firstToken.Task, task) != firstToken.Task || task.IsCompleted)
        {
            throw new InvalidOperationException("the holding sequence finished before it proved admission — its slot was never observably held");
        }

        return task;
    }

    private static async Task<bool> WaitForQueueDepthAsync(GenerationService service, int atLeast, TimeSpan budget)
    {
        Stopwatch clock = Stopwatch.StartNew();
        while (clock.Elapsed < budget)
        {
            if (service.QueuedRequestCount >= atLeast)
            {
                return true;
            }

            await Task.Delay(25);
        }

        return false;
    }
}
