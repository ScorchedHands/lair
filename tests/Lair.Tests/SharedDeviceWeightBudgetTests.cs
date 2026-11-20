using System.Runtime.InteropServices;

using Lair.Host.Config;
using Lair.Http;
using Lair.Models;

namespace Lair.Tests;

/// <summary>
/// One device weight ceiling shared across a multi-model serve: the residency an engine opens reports itself into host accounting, so a
/// model opened beside it claims against the same accelerator posture instead of landing somewhere its chain never meant to live. The first
/// claim takes what its opened residency backs; the over-commit spills down its own chain's next position rather than displacing it. These
/// gates open real accelerator backends, so they run inside the serialized resource envelope every other hybrid gate uses and skip where no
/// usable driver or device sits behind the seam.
/// </summary>
[Trait("Category", "Budget")]
[Collection("resource-envelope")]
public sealed class SharedDeviceWeightBudgetTests : IDisposable
{
    private readonly List<string> _tempPaths = [];

    public void Dispose()
    {
        foreach (string path in _tempPaths)
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

        _tempPaths.Clear();
    }

    [Fact]
    public void Open_TwoAcceleratorResidentModels_ShareOneDeviceWeightCeiling_AndSpillTheSecondDownItsOwnChain()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return; // the CUDA driver seam is exercised on Windows against nvcuda.dll.
        }

        string alphaPath = SyntheticModel();
        string betaPath = SyntheticModel();

        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Weights, Capacity = 0 }]); // no system-memory weight backing here: every claim must land where an opened residency is accounted.

        try
        {
            using ModelRegistry registry = ModelRegistry.Open(
                ConfigWith([AcceleratorModel("alpha", alphaPath), AcceleratorModel("beta", betaPath)], vramMb: 1),
                logger: null,
                transport: null,
                ledger);

            CapacitySeam deviceWeights = Assert.Single(ledger.Seams, seam => seam.Tier == BudgetTier.Vram && seam.Resource == BudgetResource.Weights); // the ceiling this host declared for that residency is what accounts it.
            Assert.Equal(1L, deviceWeights.Capacity);

            List<DeviceLedgerClaim> claims = [.. ledger.Claims.Where(claim => claim.Resource == BudgetResource.Weights)];
            Assert.Equal(["alpha", "beta"], [.. claims.Select(claim => claim.Model)]); // registration order is open order.
            Assert.Equal(BudgetTier.Vram, claims[0].Tier);  // the first model takes what its opened residency backs — at a position below where its chain starts, because this posture declared none there.
            Assert.True(claims[0].Demoted);
            Assert.Equal(BudgetTier.Disk, claims[1].Tier);  // the over-commit spills to the next position of its own chain instead of displacing it.
            Assert.True(claims[1].AtOriginStore);

            using ModelRegistry.Lease first = registry.BeginRequest("alpha");
            Assert.True(Generate(first.Engine, maxTokens: 4).CompletionTokens > 0);

            using ModelRegistry.Lease second = registry.BeginRequest("beta"); // a claim that spilled to its origin store still serves what it reads from there.
            Assert.True(Generate(second.Engine, maxTokens: 4).CompletionTokens > 0);
        }
        catch (Exception ex) when (IsUnavailableCudaSeam(ex))
        {
            return; // no usable driver or device behind this host's accelerator seam: the shared device posture cannot be opened here.
        }
    }

    private string SyntheticModel()
    {
        string path = SyntheticLlamaGguf.Write("shared-device-budget-model");
        _tempPaths.Add(path);
        return path;
    }

    private static ModelConfig AcceleratorModel(string name, string path) => new()
    {
        Name = name,
        Path = path,
        MaxContextTokens = 512,
        KvBlocks = 4,
        Placement = new PlacementConfig { GpuLayers = 2 }, // every transformer layer of the tiny fixture runs on device: this open establishes accelerator-resident weight copies.
        Budgets = [new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Dram, BudgetTier.Vram, BudgetTier.Disk] }],
    };

    private static LairConfig ConfigWith(IReadOnlyList<ModelConfig> models, long? vramMb) => new()
    {
        Server = new ServerConfig(),
        Engine = new EngineConfig { Threads = 1 },
        Models = [.. models],
        Host = vramMb is long declaredVramMb ? new HostCapacityConfig { VramMb = declaredVramMb } : null,
    };

    private static GenerationResult Generate(GenerationService engine, int maxTokens) =>
        engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = "hello world",
            MaxTokens = maxTokens,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult();

    private static bool IsUnavailableCudaSeam(Exception error) =>
        error is DllNotFoundException
            or EntryPointNotFoundException
            or TypeInitializationException
            || (error is InvalidOperationException invalid && invalid.Message.Contains(ModelRoute.CudaBackendSeam, StringComparison.Ordinal));
}
