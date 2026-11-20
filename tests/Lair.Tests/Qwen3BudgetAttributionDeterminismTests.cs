using System.Globalization;
using System.Runtime.InteropServices;

using Lair.Host.Config;
using Lair.Http;
using Lair.Models.Scheduling;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// Narrow MVP6.1 regression gate for the budget-attribution failure class: on the pinned thinking fixture, a greedy (temperature 0) capped exchange —
/// whose trajectory sits close to its reasoning ceiling — must attribute exactly the same ceiling whether its model advanced alone or while a second
/// device-resident model ran through the very same co-scheduling window. Sampling is pinned so what this gate measures is only contention: identical finish
/// reason, budget attribution, completion count, streamed pieces, per-token ids, and every sampling step's logprob distribution are demanded across the two
/// phases. The overlap itself is proven by continuous occupancy sampling — measured-side work stands in flight first, then the contending model launches
/// beside it — not assumed. No tolerance band appears anywhere in this file.
/// </summary>
[Trait("Category", "Parity")]
[Trait("Category", "Serve")]
[Trait("Category", "Cuda")]
[Collection("model-loaded")] // shares the serialized engine lane with the suites that hold several accelerator contexts open on this box
public sealed class Qwen3BudgetAttributionDeterminismTests : IDisposable
{
    private const string ModelRelativePath = "models/Qwen3-0.6B-Q8_0.gguf";

    // The same ceilings the wire-side attribution facts declare: a long reasoning budget beside a deliberately short response ceiling.
    private const int ReasoningCeiling = 256;
    private const int ResponseCeiling = 4;

    // A share whose window cap resolves to one step slot, so both phases give this model the same launch posture and only contention differs.
    private const int MeasuredShare = 12;

    private const string Prompt = "What is the capital of France? Explain your reasoning, then answer.";

    // The contending model's own traffic: concurrent, long-running requests whose device work stays in flight while this engine advances.
    private static readonly string[] ContenderPrompts = ["world world!", "hello ! hello", "! world world", "world hello !"];
    private const int ContenderMaxTokens = 512;

    // Samples where both models held window slots at one instant, after the measured model's own device work was already in flight. The contending model
    // launches only once this engine holds a slot, so this floor says overlap was real and sustained across many decode steps, not merely scheduled.
    private const int OverlapObservationFloor = 5;

    private const int OccupancyWaitMs = 30_000;

    private readonly List<string> _tempFiles = [];

    [Fact]
    public async Task GreedyCappedExchange_AttributesTheSameCeilingSoloAndUnderCoScheduledOverlap()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        string? measuredPath = ModelFixture.ResolveRepoRelative(ModelRelativePath);
        if (measuredPath is null)
        {
            return; // pinned thinking fixture absent: this gate needs the real reasoning vocabulary
        }

        try
        {
            Assert.Equal(1, DeviceCoScheduler.GrantedCap(MeasuredShare)); // one window slot per boundary in every phase below

            string contenderPath = SyntheticModel("budget-attribution-contender");

            // Solo baseline: the capped exchange issued against this model alone — nothing else resident on the device.
            AttributionFacts solo;
            using (GenerationService soloEngine = GenerationService.Open(Config(measuredPath, "qwen3-budget-solo", MeasuredShare), NullLogger.Instance))
            {
                solo = Run(soloEngine);

                Assert.Equal("length", solo.FinishReason); // the exchange ended on a ceiling, not on end-of-sequence
                Assert.Equal("response", solo.BudgetExhausted); // and that ceiling is the short response budget this config declares
            }

            using GenerationService measured = GenerationService.Open(Config(measuredPath, "qwen3-budget-measured", MeasuredShare), NullLogger.Instance);
            using GenerationService contender = GenerationService.Open(
                Config(contenderPath, "budget-attribution-contender", 60),
                NullLogger.Instance,
                ledger: null,
                compute: null,
                coScheduler: measured.HostCoScheduler);

            Assert.Same(measured.HostCoScheduler, contender.HostCoScheduler); // one window bounds both models' device launches
            DeviceCoScheduler window = measured.HostCoScheduler ?? throw new InvalidOperationException("a model resolved onto an accelerator-backed segment advances under a co-scheduling window");
            Assert.Equal("cuda", window.DeviceId);

            // The measured exchange starts first and must already hold a window slot before the contending model is let in: with a heavy real model beside a
            // synthetic one, launching the contender first would drain it long before this engine's decode began, which is scheduling order, not overlap.
            AttributionFacts overlapped;
            using CancellationTokenSource contenderCancel = new();

            Task<AttributionFacts> measuredRun = Task.Run(() => Run(measured));

            Assert.True(
                WaitUntil(() => window.OutstandingStepSlots(measured.Model.Id) > 0, OccupancyWaitMs),
                "this engine launched no device work through the shared window: nothing contended beside it");

            OverlapProbe probe = new(measured, contender);
            probe.Start(); // samples occupancy while both models' requests are in flight

            Task<AttributionFacts>[] contenderRuns = [.. ContenderPrompts.Select(prompt => Task.Run(() => RunContender(contender, prompt, contenderCancel.Token)))];

            try
            {
                overlapped = await measuredRun;
            }
            finally
            {
                probe.Stop(); // sampling ends with the phase both models were resident in
                contenderCancel.Cancel(); // the contending model's own work drains out once this gate has what it needs
            }

            try
            {
                await Task.WhenAll(contenderRuns);
            }
            catch (OperationCanceledException)
            {
                // A cancelled contender request is expected teardown here; overlap was already proven by the sampled window occupancy.
            }

            Assert.True(
                probe.MeasuredSlotSamples > 0 && probe.ContenderSlotSamples > 0,
                $"the window never reported both models' seats occupied at any sampled instant across {probe.Observations} samples " +
                $"({probe.MeasuredSlotSamples} measured-side, {probe.ContenderSlotSamples} contender-side): one engine advanced while the other was nowhere near the device");

            Assert.True(
                probe.OverlapObservations >= OverlapObservationFloor,
                $"only {probe.OverlapObservations} of {probe.Observations} window samples observed both models holding device step slots at once " +
                $"({probe.MeasuredSlotSamples} measured-side, {probe.ContenderSlotSamples} contender-side): the two engines never actually overlapped on the shared window");

            Assert.Equal(solo.FinishReason, overlapped.FinishReason); // contention moved which ceiling bounded nothing
            Assert.Equal("response", overlapped.BudgetExhausted); // and it never flipped attribution between reasoning and response
            Assert.Equal(solo.BudgetExhausted, overlapped.BudgetExhausted);

            Assert.Equal(solo.CompletionTokens, overlapped.CompletionTokens);
            Assert.Equal(solo.Text, overlapped.Text); // the served text moved nowhere under contention
            Assert.Equal(solo.StreamedText, overlapped.StreamedText); // and neither did the streamed pieces that built it

            Assert.Equal(solo.TokenIds, overlapped.TokenIds); // the token ids this model streamed are its solo run's, in order

            // Every sampling step's distribution — each candidate id and its logprob at full round-trip precision. Exact doubles here compare one engine
            // posture against itself with a contended device: batch width, cache state, format, and launch posture all stand identical on both sides of this
            // assertion, so nothing but overlap separates them. A shared-lane hazard that perturbs logits under contention would surface as an id or
            // distribution difference long before any ceiling attribution could flip.
            Assert.Equal(solo.LogprobDigest, overlapped.LogprobDigest);

            Assert.True(
                window.OutstandingStepSlots(contender.Model.Id) >= 0 && probe.ContenderSlotSamples > 0,
                "the contending model never advanced device work through the shared window while this engine decoded");
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

    // ---------------------------------------------------------------- harness

    /// <summary>One capped exchange reduced to exactly what this gate compares across the two phases.</summary>
    private sealed record AttributionFacts(
        string FinishReason,
        string BudgetExhausted,
        int CompletionTokens,
        string Text,
        string StreamedText,
        string TokenIds,
        string LogprobDigest);

    /// <summary>Samples the shared window's occupancy while both engines' requests are in flight. Observer-only diagnostics.</summary>
    private sealed class OverlapProbe
    {
        private readonly GenerationService _measured;
        private readonly GenerationService _contender;
        private readonly Thread _thread;

        public long Observations;
        public long OverlapObservations;
        public long MeasuredSlotSamples;
        public long ContenderSlotSamples;

        private volatile bool _stopped;

        public OverlapProbe(GenerationService measured, GenerationService contender)
        {
            _measured = measured;
            _contender = contender;
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
            string measuredName = _measured.Model.Id;
            string contenderName = _contender.Model.Id;

            while (!_stopped)
            {
                int measuredSlots = window.OutstandingStepSlots(measuredName);
                int contenderSlots = window.OutstandingStepSlots(contenderName);

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

                Thread.SpinWait(400); // a short pause so the loop samples across many step boundaries rather than one queue depth
            }
        }
    }

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

    private static AttributionFacts Run(GenerationService engine) => RequestAndCapture(engine, Prompt, CancellationToken.None);

    private static AttributionFacts RunContender(GenerationService engine, string prompt, CancellationToken cancellationToken)
    {
        // The contender only needs to hold device work in flight beside the measured exchange; its own answers are not compared here.
        GenerationResult result = engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = prompt,
            MaxTokens = ContenderMaxTokens,
            Temperature = 0f,
        }, cancellationToken).GetAwaiter().GetResult();

        return Facts(result, streamedText: ""); // placeholder facts this gate never compares for the contender side
    }

    private static AttributionFacts RequestAndCapture(GenerationService engine, string prompt, CancellationToken cancellationToken)
    {
        List<string> pieces = []; // one list per request: streaming callbacks run on that sequence's scheduler thread, in order.

        GenerationResult result = engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = true,
            Messages = [new Lair.Tokenizer.Templating.ChatMessage("user", prompt)],
            MaxTokens = ReasoningCeiling + ResponseCeiling, // carried as the combined ceiling exactly as an unsupplied max_tokens would resolve it
            MaxTokensWasSupplied = false,
            Temperature = 0f, // greedy: attribution is a function of the logits alone, never of a per-request RNG draw
            Stream = true,
            OnToken = piece => pieces.Add(piece),
            LogProbs = true,
            TopLogprobs = 5,
        }, cancellationToken).GetAwaiter().GetResult();

        Assert.True(
            result.LogProbs is not null,
            $"the engine reported no per-token logprob distribution for a request that asked for one ({result.CompletionTokens} generated tokens)");

        IReadOnlyList<TokenLogprob> steps = result.LogProbs!;
        Assert.Equal(result.CompletionTokens, steps.Count); // one distribution per generated token, in generation order

        return Facts(
            result,
            string.Concat(pieces),
            string.Join(",", steps.Select(entry => entry.TokenId)),
            string.Join("\n", steps.Select(LogprobFact)));
    }

    private static AttributionFacts Facts(
        GenerationResult result,
        string streamedText,
        string tokenIds = "",
        string logprobDigest = "") => new(
            FinishReason: result.FinishReason,
            BudgetExhausted: result.BudgetExhausted ?? "<none>",
            CompletionTokens: result.CompletionTokens,
            Text: result.Text,
            StreamedText: streamedText,
            TokenIds: tokenIds,
            LogprobDigest: logprobDigest);

    // One sampling step as one comparable string: the chosen id and logprob at full round-trip precision, then each top candidate beside it.
    private static string LogprobFact(TokenLogprob entry) =>
        $"{entry.TokenId}={Exact(entry.LogProb)}|{string.Join(",", entry.TopLogProbs.Select(top => $"{top.TokenId}:{Exact(top.LogProb)}"))}";

    private static string Exact(double value) => value.ToString("R", CultureInfo.InvariantCulture);

    private LairConfig Config(string modelPath, string modelName, int computeShare)
    {
        return new LairConfig
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig { Threads = 4 }, // a bounded pool per engine here so the contended phase cannot starve this serialized lane's neighbours
            Models =
            [
                new ModelConfig
                {
                    Name = modelName,
                    Path = modelPath,
                    MaxContextTokens = 768, // room for this gate's prompts plus both ceilings and the contender's own long run
                    KvBlocks = 64,
                    MaxReasoningTokens = ReasoningCeiling, // the ceilings this gate attributes are declared on the model, exactly as the wire facts declare them
                    MaxResponseTokens = ResponseCeiling,
                    MaxConcurrentRequests = ContenderPrompts.Length + 1, // room for every request this gate issues against one engine concurrently
                    Placement = new PlacementConfig { GpuLayers = 1 },
                    ComputeShare = computeShare,
                },
            ],
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
