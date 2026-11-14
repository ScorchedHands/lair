using System.Text;
using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Http;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Scheduler-side token-budget math on a synthetic tiny model: a resolved reasoning-plus-response pair is what admission reserves
/// and what decode stops at, an explicit <c>max_tokens</c> clamps that resolved ceiling without replacing either ceiling, and
/// a batched run under both ceilings is token-identical to the same sequences stepping alone.
/// </summary>
[Trait("Category", "Scheduling")]
[Collection("model-loaded")]
public sealed class TokenBudgetSchedulingTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    [Fact]
    public void BudgetPair_ReservationUsesResolvedCeiling_AndExplicitMaxTokensClampsIt()
    {
        string path = SyntheticLlamaGguf.Write("budget-scheduler-reservation");
        _tempFiles.Add(path);
        using SchedulerHarness harness = new(path, capacityBlocks: 16, maxContextTokens: 64);

        int[] prompt = [.. harness.Tokenizer.Encode("hello world")];
        ScheduledRequest budgeted = Request(prompt, maxTokens: 10, reasoning: 2, response: 3);

        Assert.Equal(5, budgeted.GenerationTokenCeiling);
        Assert.Equal((prompt.Length + 5 - 2) / harness.Pool.BlockTokens + 1, harness.Scheduler.BlocksNeededFor(budgeted));

        ScheduledRequest capped = budgeted with { MaxTokens = 3 };
        Assert.Equal(3, capped.GenerationTokenCeiling);
        Assert.Equal((prompt.Length + 3 - 2) / harness.Pool.BlockTokens + 1, harness.Scheduler.BlocksNeededFor(capped));
    }

    [Fact]
    public async Task BudgetPair_AdmissionReservesBothBudgets_AndRejectsWhenPoolCannotCoverThem()
    {
        string path = SyntheticLlamaGguf.Write("budget-scheduler-admission");
        _tempFiles.Add(path);
        using SchedulerHarness probe = new(path, capacityBlocks: 16, maxContextTokens: 64);

        int[] prompt = [.. probe.Tokenizer.Encode("hello world")];
        ScheduledRequest budgeted = Request(prompt, maxTokens: 10, reasoning: 2, response: 3);
        int needed = probe.Scheduler.BlocksNeededFor(budgeted);

        using SchedulerHarness tight = new(path, capacityBlocks: needed, maxContextTokens: 64);
        SequenceOutcome admitted = await tight.Scheduler.Submit(budgeted, CancellationToken.None).Completion;
        Assert.Equal(FinishReason.Length, admitted.FinishReason);
        Assert.Equal(5, admitted.CompletionTokens);

        using SchedulerHarness tooSmall = new(path, capacityBlocks: needed - 1, maxContextTokens: 64);
        Assert.Throws<SchedulerFullException>(() => tooSmall.Scheduler.Submit(budgeted, CancellationToken.None));
    }

    [Fact]
    public async Task BudgetPair_BatchedDecodeMatchesSingleTokenRun_UnderBothCeilings()
    {
        string path = SyntheticLlamaGguf.Write("budget-scheduler-batched");
        _tempFiles.Add(path);
        using SchedulerHarness harness = new(path, capacityBlocks: 32, maxContextTokens: 64);

        int[] promptA = [.. harness.Tokenizer.Encode("hello")];
        int[] promptB = [.. harness.Tokenizer.Encode("world")];
        ScheduledRequest requestA = Request(promptA, maxTokens: 10, reasoning: 2, response: 2);
        ScheduledRequest requestB = Request(promptB, maxTokens: 10, reasoning: 2, response: 2);

        PipelineHandle handleA = harness.Scheduler.Submit(requestA, CancellationToken.None);
        PipelineHandle handleB = harness.Scheduler.Submit(requestB, CancellationToken.None);
        SequenceOutcome batchedA = await handleA.Completion;
        SequenceOutcome batchedB = await handleB.Completion;

        Assert.Equal(FinishReason.Length, batchedA.FinishReason);
        Assert.Equal(4, batchedA.CompletionTokens);
        Assert.Equal(FinishReason.Length, batchedB.FinishReason);
        Assert.Equal(4, batchedB.CompletionTokens);

        using SchedulerHarness baseline = new(path, capacityBlocks: 32, maxContextTokens: 64);
        SequenceOutcome singleA = await baseline.Scheduler.Submit(requestA, CancellationToken.None).Completion;
        SequenceOutcome singleB = await baseline.Scheduler.Submit(requestB, CancellationToken.None).Completion;

        Assert.Equal(singleA.GeneratedTokens, batchedA.GeneratedTokens);
        Assert.Equal(singleB.GeneratedTokens, batchedB.GeneratedTokens);
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

    private static ScheduledRequest Request(int[] promptTokens, int maxTokens, int reasoning, int response) => new()
    {
        PromptTokens = promptTokens,
        MaxTokens = maxTokens,
        ReasoningTokenBudget = reasoning,
        ResponseTokenBudget = response,
        Sampling = new SamplingParams { Temperature = 0f },
        // A id outside this synthetic vocabulary keeps the run on the length path instead of ending on the model's real EOS token.
        EosId = 999,
    };

    private sealed class SchedulerHarness : IDisposable
    {
        public GgufFile Gguf { get; }
        public GgufTokenizer Tokenizer { get; }
        public CpuBackend Backend { get; }
        public TransformerModel Model { get; }
        public BlockPool Pool { get; }
        public GenerationScheduler Scheduler { get; }

        public SchedulerHarness(string modelPath, int capacityBlocks, int maxContextTokens)
        {
            Gguf = GgufFile.Open(modelPath);
            Tokenizer = GgufTokenizer.FromGguf(Gguf);
            Backend = new CpuBackend(threads: 1);
            Model = TransformerModel.Open(Gguf, Backend, maxContextTokens);
            Pool = new BlockPool(
                blockTokens: 4,
                layerCount: Model.Desc.LayerCount,
                kvHeads: Model.Desc.KvHeadCount,
                headDim: Model.Desc.HeadDim,
                capacityBlocks: capacityBlocks,
                Backend);
            Scheduler = new GenerationScheduler(Model, Pool, maxBatchTokens: 16);
        }

        public void Dispose()
        {
            Scheduler.Dispose();
            Pool.Dispose();
            Model.Dispose();
            Gguf.Dispose();
            Backend.Dispose();
        }
    }
}

/// <summary>
/// Reasoning-budget behavior on the pinned Qwen3 fixture: a reasoning ceiling closes an open thinking region at the ceiling, the
/// response ceiling stops visible tokens after that close, and the run still finishes <c>length</c> rather than leaking an
/// unterminated reasoning block. The class skips cleanly when the pinned model file is absent.
/// </summary>
[Trait("Category", "Scheduling")]
[Collection("model-loaded")]
public sealed class TokenBudgetReasoningSchedulingTests
{
    private const string ModelRelativePath = "models/Qwen3-0.6B-Q8_0.gguf";
    private const string ModelName = "qwen3-token-budget-scheduling";

    [Fact]
    public async Task ReasoningCeiling_ClosesOpenRegion_AndResponseCeilingStopsVisibleTokens()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(ModelRelativePath);
        if (modelPath is null)
        {
            return; // pinned model absent: this reasoning-ceiling proof needs the real thinking fixture
        }

        using GenerationService engine = GenerationService.Open(ConfigFor(modelPath, "strip"));

        GenerationResult capped = await engine.GenerateAsync(new GenerationRequest
        {
            Model = ModelName,
            IsChat = true,
            RequestId = "req-token-budget-scheduling",
            Messages = [new Lair.Tokenizer.Templating.ChatMessage("user", "What is the capital of France? Explain your reasoning, then answer.")],
            MaxTokens = 128,
            MaxTokensWasSupplied = false, // the model's resolved budget pair is what bounds this run, not the OpenAI field's default
            Temperature = 0f,
            Seed = 11,
        },
            CancellationToken.None);

        Assert.Equal("length", capped.FinishReason);
        Assert.True(capped.CompletionTokens > 0, "the capped run generated no tokens");
        Assert.True(capped.CompletionTokens <= 12, $"the capped run overshot reasoning=8 + response=4 by reporting {capped.CompletionTokens} generated tokens");

        Assert.NotNull(capped.ReasoningRegions);
        Assert.Single(capped.ReasoningRegions);
        Assert.False(string.IsNullOrEmpty(capped.Text), "the response ceiling left no visible text after closing the reasoning region");
        Assert.DoesNotContain("<think>", capped.Text, StringComparison.Ordinal);
        Assert.DoesNotContain("</think>", capped.Text, StringComparison.Ordinal);
    }

    private static LairConfig ConfigFor(string modelPath, string reasoning) => new()
    {
        Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
        Engine = new EngineConfig(),
        Models =
        [
            new ModelConfig
            {
                Name = ModelName,
                Path = modelPath,
                MaxContextTokens = 1024,
                KvBlocks = 64,
                MaxReasoningTokens = 8,
                MaxResponseTokens = 4,
                Reasoning = reasoning,
            },
        ],
    };
}
