using Lair.Models;

namespace Lair.Tests;

[Trait("Category", "Loader")]
public sealed class ModelBudgetResolverTests
{
    [Fact]
    public void Resolve_WeightsChainToOriginStore_LandsWithoutAnyPoolClaim()
    {
        ResolvedBudgetSet resolved = ModelBudgets.Resolve(new BudgetRequest
        {
            ModelName = "qwen-test",
            Declared =
            [
                new BudgetChain
                {
                    Resource = BudgetResource.Weights,
                    Order = [BudgetTier.Dram, BudgetTier.Nvme],
                    Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 2048 },
                },
            ],
            OpenedPoolTiers = [BudgetTier.Dram],
        });

        ResolvedBudgetChain weights = resolved.For(BudgetResource.Weights)!;
        Assert.Equal([BudgetTier.Dram, BudgetTier.Nvme], weights.Tiers.Select(position => position.Tier));

        ResolvedBudgetTier resident = weights.FirstCopyHoldingTier()!;
        Assert.Equal(BudgetTier.Dram, resident.Tier);
        Assert.False(resident.IsOriginStore);
        Assert.Equal(2048L, resident.Cap);

        // The file already holds the complete weight set on its origin store: that position needs no ceiling of ours and claims nothing.
        ResolvedBudgetTier origin = weights.At(BudgetTier.Nvme)!;
        Assert.True(origin.IsOriginStore);
        Assert.Null(origin.Cap);
    }

    [Fact]
    public void Resolve_AcceleratorResidentClaim_ReportsTheMissingBackendSeam()
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelBudgets.Resolve(new BudgetRequest
        {
            ModelName = "qwen-test",
            Declared =
            [
                new BudgetChain
                {
                    Resource = BudgetResource.Kv,
                    Order = [BudgetTier.Dram, BudgetTier.Vram],
                    Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 512 },
                },
            ],
            OpenedPoolTiers = [BudgetTier.Dram],
        }));

        Assert.Contains("model 'qwen-test'", error.Message);
        Assert.Contains("budgets for resource 'kv'", error.Message);
        Assert.Contains("no opened pool backs working copies at tier 'vram'", error.Message);
        Assert.Contains(ModelRoute.CudaBackendSeam, error.Message); // declared vocabulary reports what is missing instead of dropping the position.
    }

    [Fact]
    public void Resolve_WriteSideDiskClaim_ReportsTheMissingWritingMachinery()
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelBudgets.Resolve(new BudgetRequest
        {
            ModelName = "qwen-test",
            Declared =
            [
                new BudgetChain
                {
                    Resource = BudgetResource.PrefillBuffers,
                    Order = [BudgetTier.Dram, BudgetTier.Nvme],
                    Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 64 },
                },
            ],
            OpenedPoolTiers = [BudgetTier.Dram],
        }));

        Assert.Contains("budgets for resource 'prefill_buffers'", error.Message);
        Assert.Contains("no opened pool backs working copies at tier 'nvme'", error.Message);
        Assert.Contains("write-side bytes reach storage only where something writes them", error.Message);
    }

    [Fact]
    public void Resolve_KvChainStaysSystemMemoryBackedOnThisHost()
    {
        ResolvedBudgetSet resolved = ModelBudgets.Resolve(new BudgetRequest
        {
            ModelName = "qwen-test",
            Declared =
            [
                new BudgetChain
                {
                    Resource = BudgetResource.Kv,
                    Order = [BudgetTier.Dram],
                    Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 512 },
                },
            ],
            OpenedPoolTiers = [BudgetTier.Dram],
        });

        ResolvedBudgetChain kv = resolved.For(BudgetResource.Kv)!;

        // The declared ceiling lands as declared: resolution records it and claims nothing against capacity.
        ResolvedBudgetTier position = Assert.Single(kv.Tiers);
        Assert.False(position.IsOriginStore);
        Assert.Equal(512L, position.Cap);
    }

    [Fact]
    public void Resolve_NothingDeclared_ResolvesToTheEmptySet()
    {
        ResolvedBudgetSet resolved = ModelBudgets.Resolve(new BudgetRequest
        {
            ModelName = "qwen-test",
            Declared = [],
            OpenedPoolTiers = [BudgetTier.Dram],
        });

        Assert.True(resolved.IsEmpty);
        Assert.Null(resolved.For(BudgetResource.Weights));
    }

    [Fact]
    public void Resolve_DuplicateResourceDeclaration_Refused()
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelBudgets.Resolve(new BudgetRequest
        {
            ModelName = "qwen-test",
            Declared =
            [
                new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram] },
                new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram, BudgetTier.Vram] },
            ],
            OpenedPoolTiers = [BudgetTier.Dram],
        }));

        Assert.Contains("declares budgets for resource 'kv' more than once", error.Message);
    }

    [Fact]
    public void Resolve_CeilingOutsideItsChain_RefusedBeforeAnyPostureQuestion()
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelBudgets.Resolve(new BudgetRequest
        {
            ModelName = "qwen-test",
            Declared =
            [
                new BudgetChain
                {
                    Resource = BudgetResource.Kv,
                    Order = [BudgetTier.Dram],
                    Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Vram] = 64 },
                },
            ],
            OpenedPoolTiers = [BudgetTier.Dram],
        }));

        Assert.Contains("a ceiling binds tier 'vram', which this chain's order ['dram'] does not name", error.Message);
    }

    [Fact]
    public void Resolve_NonPositiveCeiling_RefusedNamingUnit()
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelBudgets.Resolve(new BudgetRequest
        {
            ModelName = "qwen-test",
            Declared =
            [
                new BudgetChain
                {
                    Resource = BudgetResource.Kv,
                    Order = [BudgetTier.Dram],
                    Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 0 },
                },
            ],
            OpenedPoolTiers = [BudgetTier.Dram],
        }));

        Assert.Contains("KV block ceilings must be 1 or greater", error.Message);
    }

    [Fact]
    public void ValidateDeclaredChains_AcceptsFutureShapedDeclarationsWithoutConsultingCapacity()
    {
        // A configuration layer that knows nothing about opened pools runs the same structural rules resolution applies: an accelerator tier is legal material to declare, and what a host can carry it is decided elsewhere.
        ModelBudgets.ValidateDeclaredChains(
            "qwen-test",
            [
                new BudgetChain
                {
                    Resource = BudgetResource.Kv,
                    Order = [BudgetTier.Dram, BudgetTier.Vram],
                    Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 512 },
                },
                new BudgetChain
                {
                    Resource = BudgetResource.Weights,
                    Order = [BudgetTier.Dram, BudgetTier.Nvme],
                },
            ]);
    }

    [Fact]
    public void ValidateDeclaredChains_OutOfOrderChain_Refused()
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelBudgets.ValidateDeclaredChains(
            "qwen-test",
            [new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Nvme, BudgetTier.Dram] }]));

        Assert.Contains("order ['nvme', 'dram'] is out of order", error.Message);
    }
}
