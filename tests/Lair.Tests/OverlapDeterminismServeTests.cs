using System.Globalization;
using System.Runtime.InteropServices;

using Lair.Host.Config;
using Lair.Http;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Scheduling;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// MVP6.1's determinism-under-overlap gate at the serve seam on a real accelerator. One model's token stream — text, streamed pieces, per-token ids,
/// and each sampling step's logprob distribution reported at full round-trip precision — is identical whether its steps ran alone or while another
/// served model advanced device work concurrently through the same shared co-scheduling window, and the overlap itself is proven rather than assumed:
/// the window is sampled continuously and must report both models holding device step slots at once. The solo baseline runs the measured model under
/// exactly the launch posture the overlapped phase gives it (its own fair-share cap bounds one sequence per boundary on both sides), so what differs
/// between the phases is only whose kernels contend for the device — not batch width, cache state, engine instance, or format. A second fact reads the
/// same prompts through a merged-step solo run and asserts their ids and text survive window-bound overlapped serving: fair-share deferral retargets
/// when math launches, never which token a pipeline samples. A third reads both engines' live composition reports continuously and a fourth the arbiter
/// seam device-free: two models' pipelines compose as separate steps and no merged batch ever carries another plan's candidate, which is what keeps
/// cross-model shared batch steps non-default. A fifth submits distinct prompts concurrently against one engine and asserts they answer exactly as their
/// sequential run did — the posture a model serving several requests at once actually runs. Every overlapped request also completes whole while its
/// model's excess candidates defer naming that share: a launch decision taken before a step starts, never a preemption of one in flight. Synthetic F32
/// fixtures keep every assertion exact — no tolerance band appears anywhere in this file, because nothing here compares a quantized format against an
/// oracle or one backend against another.
/// </summary>
[Trait("Category", "Parity")]
[Trait("Category", "Serve")]
[Trait("Category", "Cuda")]
[Collection("model-loaded")] // shares the serialized engine lane with the suites that measure working-set envelopes here: these gates hold several accelerator contexts open at once
public sealed class OverlapDeterminismServeTests : IDisposable
{
    private const int MeasuredMaxTokens = 24;

    // A share whose window cap resolves to a single step slot, so every boundary of the measured model launches one sequence and defers its excess —
    // the same launch posture in the solo baseline and under contention, which is what makes an exact per-step comparison honest rather than lucky.
    private const int MeasuredShare = 12;

    private const string MeasuredModelName = "overlap-determinism-measured";

    private static readonly string[] Prompts = ["hello world", "world !", "! hello", "hello world !"];

    // The contending model's own traffic: concurrent, longer-running requests whose device work is still in flight while the measured model advances.
    private static readonly string[] ContenderPrompts = ["world world!", "hello ! hello", "! world world", "world hello !"];

    private const int ContenderMaxTokens = 48;

    // Samples where both models held window slots at one instant. The phase gates on the contending model launching through the window before the
    // measured model starts advancing, so this floor says the overlap was real and sustained, not merely scheduled.
    private const int OverlapObservationFloor = 5;

    // Boundaries sampled while an overlapped phase ran, each checked against the single-model composition invariant. Serving a synthetic model
    // across dozens of decode steps leaves the sampler thousands of chances to read a decided boundary, so this floor proves sampling ran without
    // resting on one lucky window.
    private const int CompositionSampleFloor = 10;

    // How long an overlapped phase waits for the contending model's first device launch before measuring against it.
    private const int OccupancyWaitMs = 20_000;

    private readonly List<string> _tempFiles = [];

    [Fact]
    public async Task OverlappedModel_TokenStreamAndLogprobsIdenticalToItsSoloRun()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        string measuredPath = SyntheticModel(MeasuredModelName);
        string contenderPath = SyntheticModel("overlap-determinism-contender");

        try
        {
            Assert.Equal(1, DeviceCoScheduler.GrantedCap(MeasuredShare)); // this model's cap is one window slot in every phase below

            // Solo baseline: the same requests issued the same way against the measured model alone — nothing else resident on the device.
            ServedFacts[] solo;
            using (GenerationService soloEngine = GenerationService.Open(Config(measuredPath, MeasuredModelName, MeasuredShare), NullLogger.Instance))
            {
                solo = await Task.WhenAll(Prompts.Select(prompt => Task.Run(() => Run(soloEngine, prompt, MeasuredMaxTokens))));

                Assert.True(
                    soloEngine.ServedScheduler.DeferredCompositionCandidates > 0,
                    "the solo baseline composed every concurrent request into one step: its declared share bounded nothing here");
            }

            // Overlap: the same model on a fresh engine, advancing while a second device-resident model runs its own requests through the very same window.
            using GenerationService measured = GenerationService.Open(Config(measuredPath, MeasuredModelName, MeasuredShare), NullLogger.Instance);
            using GenerationService contender = GenerationService.Open(
                Config(contenderPath, "overlap-determinism-contender", 60),
                NullLogger.Instance,
                ledger: null,
                compute: null,
                coScheduler: measured.HostCoScheduler);

            Assert.Same(measured.HostCoScheduler, contender.HostCoScheduler); // one window bounds both models' device launches
            DeviceCoScheduler window = measured.HostCoScheduler ?? throw new InvalidOperationException("a model resolved onto an accelerator-backed segment advances under a co-scheduling window");
            Assert.Equal("cuda", window.DeviceId);

            Task<ServedFacts>[] contenderRuns = [.. ContenderPrompts.Select(prompt => Task.Run(() => Run(contender, prompt, ContenderMaxTokens)))];

            // The contending model must already hold window slots before the measured model starts advancing: without this barrier the two
            // phases could drift apart on a busy box and every later sample would read one resident model instead of an overlap.
            Assert.True(
                WaitUntil(() => window.OutstandingStepSlots("overlap-determinism-contender") > 0, OccupancyWaitMs),
                "the contending model launched no device work through the shared window: it never advanced beside this engine");

            OverlapProbe probe = new(measured, contender);
            probe.Start(); // samples the window's occupancy and both schedulers' composition reports while every request is in flight

            Task<ServedFacts>[] measuredRuns = [.. Prompts.Select(prompt => Task.Run(() => Run(measured, prompt, MeasuredMaxTokens)))];

            ServedFacts[] overlapped = await Task.WhenAll(measuredRuns);
            probe.Stop(); // sampling ends with the phase both models were resident in; the contending requests keep running to completion after it

            await Task.WhenAll(contenderRuns); // the contending model's own requests all ran to completion beside these tokens

            Assert.Empty(probe.CompositionViolations); // checked continuously against the other engine's plan; reported here so a failing boundary names itself

            for (int i = 0; i < solo.Length; i++)
            {
                Assert.Equal(solo[i].Text, overlapped[i].Text); // the served text moved nowhere under contention
                Assert.Equal(solo[i].StreamedText, overlapped[i].StreamedText); // and neither did the streamed pieces that built it

                Assert.Equal(solo[i].PromptTokens, overlapped[i].PromptTokens);
                Assert.Equal(solo[i].CompletionTokens, overlapped[i].CompletionTokens);
                Assert.Equal(solo[i].FinishReason, overlapped[i].FinishReason);

                Assert.Equal(solo[i].TokenIds, overlapped[i].TokenIds); // the token ids this model streamed are its solo run's, in order

                // Every sampling step's distribution — each candidate id and its logprob written at full round-trip precision. Exact doubles here
                // compare one engine posture against itself with a contended device: batch width, cache state, format, and launch posture all stand
                // identical on both sides of this assertion, so nothing but overlap separates them.
                Assert.Equal(solo[i].LogprobDigest, overlapped[i].LogprobDigest);
            }

            Assert.True(
                probe.MeasuredSlotSamples > 0 && probe.ContenderSlotSamples > 0,
                $"the window never reported both models' seats occupied at any sampled instant across {probe.Observations} samples " +
                $"({probe.MeasuredSlotSamples} measured-side, {probe.ContenderSlotSamples} contender-side): one engine advanced while the other was nowhere near the device");

            Assert.True(
                probe.OverlapObservations >= OverlapObservationFloor,
                $"only {probe.OverlapObservations} of {probe.Observations} window samples observed both models holding device step slots at once " +
                $"({probe.MeasuredSlotSamples} measured-side, {probe.ContenderSlotSamples} contender-side): the two engines never actually overlapped on the shared window");

            Assert.True(
                probe.CompositionSamples >= CompositionSampleFloor,
                $"the composition seam was sampled only {probe.CompositionSamples} times during the overlapped phase");

            GenerationScheduler measuredScheduler = measured.ServedScheduler;
            Assert.True(
                measuredScheduler.DeferredCompositionCandidates > 0,
                "a one-slot share against four concurrent requests deferred nothing on the shared window: no request here was ever bounded by its fair-share");
            Assert.Contains(
                measuredScheduler.LastCompositionRefusals,
                refusal =>
                    refusal.Block == CompositionBlock.DeviceShare
                    && refusal.Reason.Contains($"compute_share {MeasuredShare}%", StringComparison.Ordinal)
                    && refusal.Reason.Contains(MeasuredModelName, StringComparison.Ordinal)
                    && refusal.Reason.Contains("cuda", StringComparison.Ordinal)); // the deferral names the share it deferred against

            Assert.True(contender.ServedScheduler.Stats.DecodeSteps > 0); // the contending model really advanced while these tokens were sampled

            Assert.Equal(0, window.OutstandingStepSlots(MeasuredModelName));
            Assert.Equal(0, window.OutstandingStepSlots("overlap-determinism-contender")); // every launched slot returned as its forward completed — nothing was cancelled out of the window
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }
    }

    [Fact]
    public async Task MergedStepSoloRun_SameTokenIdsAndTextUnderWindowBoundOverlap()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        string measuredPath = SyntheticModel(MeasuredModelName);
        string contenderPath = SyntheticModel("overlap-determinism-merged-contender");

        try
        {
            // The same prompts through a full-window share: every concurrent candidate merges into one step instead of deferring against a cap.
            ServedFacts[] mergedSolo;
            using (GenerationService merged = GenerationService.Open(Config(measuredPath, MeasuredModelName, 100), NullLogger.Instance))
            {
                Assert.Equal(DeviceCoScheduler.WindowStepSlots, DeviceCoScheduler.GrantedCap(100));

                mergedSolo = await Task.WhenAll(Prompts.Select(prompt => Task.Run(() => Run(merged, prompt, MeasuredMaxTokens))));

                Assert.Equal(0, merged.ServedScheduler.DeferredCompositionCandidates); // a full-share model was never bounded by the window here
            }

            using GenerationService measured = GenerationService.Open(Config(measuredPath, MeasuredModelName, MeasuredShare), NullLogger.Instance);
            using GenerationService contender = GenerationService.Open(
                Config(contenderPath, "overlap-determinism-merged-contender", 60),
                NullLogger.Instance,
                ledger: null,
                compute: null,
                coScheduler: measured.HostCoScheduler);

            Task<ServedFacts>[] runs = [.. Prompts.Select(prompt => Task.Run(() => Run(measured, prompt, MeasuredMaxTokens)))];

            ServedFacts[] overlapped = await Task.WhenAll(runs);

            for (int i = 0; i < mergedSolo.Length; i++)
            {
                // A window-bound run answers the same tokens its merged-step solo run answered. Ids and text carry this fact rather than raw logprob
                // doubles: batch width is what differs between these two postures, and that axis already belongs to the per-format parity gates.
                Assert.Equal(mergedSolo[i].Text, overlapped[i].Text);
                Assert.Equal(mergedSolo[i].StreamedText, overlapped[i].StreamedText);
                Assert.Equal(mergedSolo[i].TokenIds, overlapped[i].TokenIds);
            }

            Assert.True(
                measured.ServedScheduler.DeferredCompositionCandidates > 0,
                "the window-bound run composed every candidate into one step: fair-share deferred nothing here");
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }
    }

    [Fact]
    public async Task TwoEnginesOnOneWindow_NeverReportACompositionCarryingAnotherModelsPlan()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        string measuredPath = SyntheticModel(MeasuredModelName);
        string contenderPath = SyntheticModel("overlap-determinism-composition-contender");

        try
        {
            using GenerationService measured = GenerationService.Open(Config(measuredPath, MeasuredModelName, 40), NullLogger.Instance);
            using GenerationService contender = GenerationService.Open(
                Config(contenderPath, "overlap-determinism-composition-contender", 30),
                NullLogger.Instance,
                ledger: null,
                compute: null,
                coScheduler: measured.HostCoScheduler);

            Task<ServedFacts>[] runs = [.. Prompts.Select(prompt => Task.Run(() => Run(measured, prompt, MeasuredMaxTokens)))];

            // Both engines carry their own in-flight requests here: the invariant is read while two models actually advance on one window.
            Task<ServedFacts>[] contenderRuns = [.. ContenderPrompts.Select(prompt => Task.Run(() => Run(contender, prompt, ContenderMaxTokens)))];

            DeviceCoScheduler window = measured.HostCoScheduler ?? throw new InvalidOperationException("a model resolved onto an accelerator-backed segment advances under a co-scheduling window");
            Assert.True(
                WaitUntil(() => window.OutstandingStepSlots(contender.Model.Id) > 0 || window.OutstandingStepSlots(MeasuredModelName) > 0, OccupancyWaitMs),
                "no engine launched device work through the shared window: this phase would have sampled one model's steps only");

            OverlapProbe probe = new(measured, contender); // samples both schedulers' reports against the other engine's plan
            probe.Start();

            await Task.WhenAll(runs);
            await Task.WhenAll(contenderRuns);

            probe.Stop();

            Assert.Empty(probe.CompositionViolations); // every sampled boundary stepped one model over its own plan, on both engines
            Assert.True(
                probe.CompositionSamples >= CompositionSampleFloor,
                $"the composition seam was sampled only {probe.CompositionSamples} times during this overlapped phase");

            GenerationScheduler measuredScheduler = measured.ServedScheduler;
            CompositionReport lastMeasured = measuredScheduler.LastStepComposition;
            Assert.NotEmpty(lastMeasured.Compositions);

            foreach (StepComposition composition in lastMeasured.Compositions)
            {
                Assert.True(ReferenceEquals(composition.Key.Plan, measured.ServedModel.Plan)); // the step this scheduler published advances its own model's plan
                Assert.Null(lastMeasured.CompositionFor(new StepRouteKey(contender.ServedModel.Plan, contender.ResolvedRoute))); // and no group was keyed on the other engine

                Assert.All(composition.Composed, candidate => Assert.True(ReferenceEquals(candidate.Plan, composition.Key.Plan)));
            }

            CompositionReport lastContender = contender.ServedScheduler.LastStepComposition;
            if (lastContender.Compositions.Count > 0) // this model's own steps may already have drained by the time its requests finished
            {
                Assert.All(lastContender.Compositions, composition => Assert.True(ReferenceEquals(composition.Key.Plan, contender.ServedModel.Plan)));
            }

            Assert.True(measured.ServedScheduler.Stats.DecodeSteps > 0); // the measured model stepped its own sequences...
            Assert.True(contender.ServedScheduler.Stats.DecodeSteps > 0); // ...and so did the contending model, on the same window
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }
    }

    [Fact]
    public void Arbiter_ComposesSeparateStepsPerModel_NeverOneSharedBatchStep()
    {
        ModelPlan left = ResolveHybridPlan("arbiter-left");
        ModelPlan right = ResolveHybridPlan("arbiter-right");

        CompositionReport report = new StepCompositionArbiter().Compose(
            [Candidate(left, 1), Candidate(left, 2), Candidate(right, 3)],
            steppingSegmentIndex: 2);

        Assert.Equal(2, report.Compositions.Count); // two plans advanced here, so two logical steps were composed — and nothing was deferred to make it so
        Assert.Empty(report.Deferrals);

        foreach (StepComposition composition in report.Compositions)
        {
            ModelPlan owner = composition.Key.Plan;

            Assert.All(composition.Composed, candidate => Assert.True(
                ReferenceEquals(candidate.Plan, owner),
                "a merged step keyed on one plan carried a candidate advancing another model's plan — cross-model batch steps would have to be default for that"));

            Assert.All(composition.Deferred, deferred => Assert.True(
                ReferenceEquals(deferred.Candidate.Plan, owner),
                "a deferral was recorded against a composition it does not belong to"));
        }

        StepComposition leftStep = report.CompositionFor(new StepRouteKey(left, left.Route))!;
        StepComposition rightStep = report.CompositionFor(new StepRouteKey(right, right.Route))!;

        Assert.Equal(2, leftStep.MergedSequenceCount); // each model's own pipelines merged with one another...
        Assert.Single(rightStep.Composed);            // ...and never with the other model's.

        // Two loads of one file shape are still two steps: the merge key compares plans by reference, so a field-for-field match shares nothing.
        ModelPlan twin = ResolveHybridPlan("arbiter-left");
        CompositionReport twins = new StepCompositionArbiter().Compose([Candidate(left, 1), Candidate(twin, 2)], steppingSegmentIndex: 2);

        Assert.Equal(2, twins.Compositions.Count);
        Assert.All(twins.Compositions, composition => Assert.Single(composition.Composed));

        // A fair-share grant belongs to one model's occupancy and bounds only that model's step beside an untouched one.
        DeviceShareLimit granted = new(0, MeasuredShare, "arbiter-left", left.Route.Segments[1].Describe(1), left);
        CompositionReport bounded = new StepCompositionArbiter().Compose(
            [Candidate(left, 1), Candidate(right, 2)],
            steppingSegmentIndex: 2,
            deviceShare: granted);

        Assert.Equal(2, bounded.Compositions.Count);
        Assert.All(bounded.Deferrals, deferral => Assert.True(ReferenceEquals(deferral.Candidate.Plan, left))); // the starved model deferred its own candidate; nothing crossed into the other step
    }

    // ---------------------------------------------------------------- harness

    /// <summary>One served request reduced to the facts this gate compares: text, streamed pieces, token accounting, and every sampling step's distribution.</summary>
    private sealed record ServedFacts(
        string Text,
        string StreamedText,
        int PromptTokens,
        int CompletionTokens,
        string FinishReason,
        string TokenIds,
        string LogprobDigest);

    /// <summary>Samples the shared window's occupancy and both engines' composition reports while their requests are in flight. Every read here is an observer-only diagnostic.</summary>
    private sealed class OverlapProbe
    {
        private readonly GenerationService _measured;
        private readonly GenerationService _contender;
        private readonly string _contenderName;
        private readonly Thread _thread;

        public long OverlapObservations;
        public long Observations;
        public long CompositionSamples;
        public long MeasuredSlotSamples; // samples where only the measured model held window slots
        public long ContenderSlotSamples; // samples where only the contending model held them
        public readonly List<string> CompositionViolations = [];

        private volatile bool _stopped;

        public OverlapProbe(GenerationService measured, GenerationService contender)
        {
            _measured = measured;
            _contender = contender;
            _contenderName = contender.Model.Id; // the id its engine seats on the window under and keys its requests on
            _thread = new Thread(Sample) { IsBackground = true };
        }

        public void Start() => _thread.Start();

        /// <summary>Ends sampling and joins the sampler, so every observation this probe made is settled before a test reads its counters.</summary>
        public void Stop()
        {
            _stopped = true;
            _thread.Join(TimeSpan.FromMinutes(1));
        }

        private void Sample()
        {
            DeviceCoScheduler window = _measured.HostCoScheduler!;
            string measuredName = _measured.Model.Id; // the id its engine seats on the window under and keys its requests on

            while (!_stopped)
            {
                int measuredSlots = window.OutstandingStepSlots(measuredName);
                int contenderSlots = window.OutstandingStepSlots(_contenderName);

                Observations++;

                if (measuredSlots > 0)
                {
                    MeasuredSlotSamples++;
                }

                if (contenderSlots > 0)
                {
                    ContenderSlotSamples++;
                }

                if (measuredSlots > 0 && contenderSlots > 0)
                {
                    OverlapObservations++; // one instant where both models' device work stood in flight on the shared window
                }

                CheckComposition(
                    _measured,
                    new StepRouteKey(_contender.ServedModel.Plan, _contender.ResolvedRoute),
                    "measured");
                CheckComposition(
                    _contender,
                    new StepRouteKey(_measured.ServedModel.Plan, _measured.ResolvedRoute),
                    "contender");

                Thread.SpinWait(400); // a short pause so the loop samples across many step boundaries rather than one queue depth
            }
        }

        private void CheckComposition(GenerationService engine, StepRouteKey foreignKey, string side)
        {
            CompositionReport report = engine.ServedScheduler.LastStepComposition;
            if (report.Compositions.Count == 0 && report.Deferrals.Count == 0)
            {
                return; // no boundary had decided anything at this sample
            }

            CompositionSamples++;

            foreach (StepComposition composition in report.Compositions)
            {
                if (!ReferenceEquals(composition.Key.Plan, engine.ServedModel.Plan))
                {
                    Violation($"{side}: a composed step was keyed on a plan other than its own model's ({composition.MergedSequenceCount} sequence(s) at segment {composition.SegmentIndex})");
                    continue;
                }

                if (report.CompositionFor(foreignKey) is not null)
                {
                    Violation($"{side}: a boundary published a composition keyed on the other engine's plan and route");
                }

                foreach (StepCandidate candidate in composition.Composed)
                {
                    if (!ReferenceEquals(candidate.Plan, composition.Key.Plan))
                    {
                        Violation($"{side}: pipeline {candidate.PipelineId} merged into a step advancing another model's plan");
                    }
                }

                foreach (DeferredStepCandidate deferred in composition.Deferred)
                {
                    if (!ReferenceEquals(deferred.Candidate.Plan, composition.Key.Plan))
                    {
                        Violation($"{side}: pipeline {deferred.Candidate.PipelineId} was deferred by a step it does not belong to");
                    }
                }
            }

            foreach (DeferredStepCandidate deferral in report.Deferrals)
            {
                if (!ReferenceEquals(deferral.Candidate.Plan, engine.ServedModel.Plan))
                {
                    Violation($"{side}: a boundary's deferral list named a candidate advancing another model's plan");
                }
            }
        }

        private void Violation(string message)
        {
            lock (CompositionViolations)
            {
                CompositionViolations.Add(message);
            }
        }
    }

    [Fact]
    public async Task ConcurrentDistinctPromptSubmissionsOnOneEngine_AnswerIdenticallyToTheirSequentialRun()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        string measuredPath = SyntheticModel(MeasuredModelName);

        // Distinct prompts per burst: every submission inserts its own memo entry, which is the posture a shared tokenizer must survive.
        string[] distinctPrompts = ["hello world", "world hello", "! world", "world !", "hello ! world", "! world hello"];

        try
        {
            // Sequential baseline on one engine: what each prompt answers when nothing else submits against this model's tokenizer beside it.
            ServedFacts[] sequential;
            using (GenerationService engine = GenerationService.Open(Config(measuredPath, MeasuredModelName, 100), NullLogger.Instance))
            {
                sequential = [.. distinctPrompts.Select(prompt => Run(engine, prompt, MeasuredMaxTokens))];
            }

            // The same prompts through several concurrent submissions on one engine, repeated so a shared-memo race would surface rather than hide.
            for (int round = 0; round < 3; round++)
            {
                using GenerationService contended = GenerationService.Open(Config(measuredPath, MeasuredModelName, 100), NullLogger.Instance);

                ServedFacts[] concurrent = await Task.WhenAll(distinctPrompts.Select(prompt => Task.Run(() => Run(contended, prompt, MeasuredMaxTokens))));

                for (int i = 0; i < sequential.Length; i++)
                {
                    // Ids and text carry this fact: submissions that merged into shared steps still answer the tokens their solo run answered.
                    Assert.Equal(sequential[i].Text, concurrent[i].Text);
                    Assert.Equal(sequential[i].StreamedText, concurrent[i].StreamedText);
                    Assert.Equal(sequential[i].TokenIds, concurrent[i].TokenIds);
                }

                Assert.True(contended.ServedScheduler.PeakAdvancingPipelines > 1, $"round {round} never held more than one pipeline admitted at once");
            }
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }
    }

    // Polls a window diagnostic until it reports what the phase needs, or its budget runs out: this gates on residency facts the seam itself
    // publishes rather than on a sleep that might land before either model launched anything.
    private static bool WaitUntil(Func<bool> condition, int budgetMs)
    {
        System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();
        while (!condition())
        {
            if (clock.ElapsedMilliseconds > budgetMs)
            {
                return false;
            }

            Thread.Sleep(2); // a short pause so the poll reads many scheduler boundaries rather than one instant
        }

        return true;
    }

    private static ServedFacts Run(GenerationService engine, string prompt, int maxTokens)
    {
        List<string> pieces = []; // one list per request: streaming callbacks run on that sequence's scheduler thread, in order.

        GenerationResult result = engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = prompt,
            MaxTokens = maxTokens,
            Temperature = 0f, // greedy: the token stream is a function of the logits alone
            Stream = true,
            OnToken = piece => pieces.Add(piece),
            LogProbs = true,
            TopLogprobs = 5,
        }, CancellationToken.None).GetAwaiter().GetResult();

        Assert.True(
            result.LogProbs is not null,
            $"the engine reported no per-token logprob distribution for a request that asked for one ({result.CompletionTokens} generated tokens)");

        IReadOnlyList<TokenLogprob> steps = result.LogProbs!;
        Assert.Equal(result.CompletionTokens, steps.Count); // one distribution per generated token, in generation order

        return new ServedFacts(
            result.Text,
            string.Concat(pieces),
            result.PromptTokens,
            result.CompletionTokens,
            result.FinishReason,
            string.Join(",", steps.Select(entry => entry.TokenId)),
            string.Join("\n", steps.Select(LogprobFact)));
    }

    // One sampling step as one comparable string: the chosen id and logprob at full round-trip precision, then each top candidate beside it.
    private static string LogprobFact(TokenLogprob entry) =>
        $"{entry.TokenId}={Exact(entry.LogProb)}|{string.Join(",", entry.TopLogProbs.Select(top => $"{top.TokenId}:{Exact(top.LogProb)}"))}";

    private static string Exact(double value) => value.ToString("R", CultureInfo.InvariantCulture);

    private static LairConfig Config(string modelPath, string modelName, int computeShare) => new()
    {
        Server = new ServerConfig(),
        Engine = new EngineConfig { Threads = 1 },
        Models =
        [
            new ModelConfig
            {
                Name = modelName,
                Path = modelPath,
                MaxContextTokens = 512,
                KvBlocks = 32,
                MaxConcurrentRequests = Prompts.Length + ContenderPrompts.Length, // room for every request this gate issues against one engine concurrently
                Placement = new PlacementConfig { GpuLayers = 1 },
                ComputeShare = computeShare,
            },
        ],
    };

    private static StepCandidate Candidate(ModelPlan plan, long pipelineId) => new() { PipelineId = pipelineId, Plan = plan };

    // The route shape every composition fact here reads: one hybrid-shaped group over [cpu prefill, cuda block span, cpu decode tail].
    private ModelPlan ResolveHybridPlan(string modelName)
    {
        string path = SyntheticLlamaGguf.Write(modelName);
        _tempFiles.Add(path);

        using GgufFile gguf = GgufFile.Open(path);
        ModelPlan plan = ModelPlans.Resolve(gguf);

        return plan with
        {
            Route = ModelRoutes.Resolve(new RouteRequest
            {
                ModelName = modelName,
                Stages =
                [
                    new(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId),
                    new("split_block", "cuda"),
                    new(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId),
                ],
                AvailableDevices = [ModelRoute.CpuDeviceId, "cuda"],
            }),
        };
    }

    private string SyntheticModel(string modelName)
    {
        string path = SyntheticLlamaGguf.Write(modelName);
        _tempFiles.Add(path);
        return path;
    }

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
                // A model still holding its mmap view keeps the file on Windows; the temp directory absorbs it.
            }
        }

        _tempFiles.Clear();
    }
}
