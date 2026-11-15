using Lair.Host.Config;
using Lair.Http;
using Lair.Models;

namespace Lair.Tests;

/// <summary>
/// Placement token identity against a shared host ledger: two engines opened over the same synthetic model, one whose
/// weight claim seats at its chain's first copy-holding tier and one whose claim demotes to the next position because
/// the earlier engine already holds that tier. The placement difference is pure ledger posture — every forward still
/// reads the file's own tensors through its opened view — so both engines must produce byte-identical generations for
/// the same prompt, exactly as an unspiled and a spilled placement always have done against synthetic capacity tables.
/// </summary>
[Trait("Category", "Budget")]
[Trait("Category", "Parity")]
public sealed class DeviceLedgerPlacementParityTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

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

    [Fact]
    public void DemotedWeightPlacement_AndFirstTierPlacement_GenerateIdenticalTokens_OnTheSameModel()
    {
        string path = SyntheticLlamaGguf.Write("ledger-parity");
        _tempFiles.Add(path);

        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Weights, Capacity = 1 }]);

        using GenerationService unspiled = GenerationService.Open(ConfigFor(path, "parity-unspiled"), null, ledger);
        using GenerationService spilled = GenerationService.Open(ConfigFor(path, "parity-spilled"), null, ledger);

        IReadOnlyList<DeviceLedgerClaim> claims = ledger.Claims; // the placement this test is about: one tier held, its neighbour demoted to
        Assert.Equal(BudgetTier.Dram, claims[0].Tier);
        Assert.False(claims[0].Demoted);
        Assert.Equal(BudgetTier.Nvme, claims[1].Tier);
        Assert.True(claims[1].Demoted);

        const string prompt = "hello world"; // the same tokens through both placements, greedy so every run is comparable

        GenerationResult firstPass = Generate(unspiled, prompt);
        GenerationResult secondPass = Generate(spilled, prompt);

        Assert.Equal(firstPass.Text, secondPass.Text);                       // identical visible text across an unseated and a demoted placement
        Assert.Equal(firstPass.CompletionTokens, secondPass.CompletionTokens);
        Assert.Equal(firstPass.FinishReason, secondPass.FinishReason);
    }

    private static GenerationResult Generate(GenerationService engine, string prompt) =>
        engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = prompt,
            MaxTokens = 8,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult();

    private static LairConfig ConfigFor(string modelPath, string modelName) => new()
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
                KvBlocks = 8,
                Budgets = [new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Dram, BudgetTier.Nvme], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1024 } }],
            },
        ],
    };
}
