using Lair.Host.Config;
using Lair.Models;

namespace Lair.Tests;

/// <summary>
/// Hot-reload classification for the KV residency axis and the block ceiling that bounds an accelerator-resident
/// reservation: both are established when the engine opens, because reseating an opened pool's blocks across host and
/// device storage is not something a live engine does. The host KV ceiling beside them keeps its live class, and a
/// candidate the loader rejected still names each of these fields with its class in the reload summary.
/// </summary>
[Trait("Category", "HotReload")]
public sealed class KvResidencyHotReloadTests : IDisposable
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
                // Best-effort cleanup of temp fixtures.
            }
        }
    }

    [Fact]
    public void Classify_KvResidencyAndVramKvCeiling_AreRestartOnly()
    {
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("models[0].placement.kv_residency"));
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("models[0].budgets.kv.caps.vram_blocks"));

        // The chain that positions those blocks is structure resolved at open too, while the host ceiling beside it is the
        // live soft cap this engine has always been able to retarget — the residency axis changes neither.
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("models[0].budgets.kv.order"));
        Assert.Equal(ConfigMutationClass.Live, ConfigMutationTable.Classify("models[0].budgets.kv.caps.dram_blocks"));
    }

    [Theory]
    [InlineData("vram")]
    [InlineData("split")]
    public void Diff_KvResidencyChange_ReportsRestartOnly(string residency)
    {
        LairConfig current = ConfigWith(new ModelConfig
        {
            Name = "kv-residency-test",
            Path = "a.gguf",
            Placement = new PlacementConfig { KvResidency = "dram" },
        });

        LairConfig candidate = ConfigWith(new ModelConfig
        {
            Name = "kv-residency-test",
            Path = "a.gguf",
            Placement = new PlacementConfig { KvResidency = residency },
        });

        FieldChange change = Assert.Single(ConfigMutationTable.Diff(current, candidate));
        Assert.Equal("models[0].placement.kv_residency", change.Field);
        Assert.Equal("dram", change.OldValue);
        Assert.Equal(residency, change.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, change.MutationClass);
    }

    [Fact]
    public void Diff_VramKvCeilingChange_ReportsRestartOnly_WhileTheHostCeilingStaysLive()
    {
        LairConfig current = ConfigWith(new ModelConfig
        {
            Name = "kv-residency-test",
            Path = "a.gguf",
            KvBlocks = 256,
            Budgets = [new BudgetChain
            {
                Resource = BudgetResource.Kv,
                Order = [BudgetTier.Dram, BudgetTier.Vram],
                Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 256, [BudgetTier.Vram] = 64 },
            }],
        });

        LairConfig candidate = ConfigWith(new ModelConfig
        {
            Name = "kv-residency-test",
            Path = "a.gguf",
            KvBlocks = 320,
            Budgets = [new BudgetChain
            {
                Resource = BudgetResource.Kv,
                Order = [BudgetTier.Dram, BudgetTier.Vram],
                Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 320, [BudgetTier.Vram] = 128 },
            }],
        });

        IReadOnlyList<FieldChange> changes = ConfigMutationTable.Diff(current, candidate);

        FieldChange hostCeiling = Assert.Single(changes, change => change.Field == "models[0].budgets.kv.caps.dram_blocks");
        Assert.Equal(ConfigMutationClass.Live, hostCeiling.MutationClass);

        FieldChange deviceCeiling = Assert.Single(changes, change => change.Field == "models[0].budgets.kv.caps.vram_blocks");
        Assert.Equal("64", deviceCeiling.OldValue);
        Assert.Equal("128", deviceCeiling.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, deviceCeiling.MutationClass);
    }

    [Fact]
    public void ExtractEdits_RejectedResidencyCandidate_StillNamesFieldAndClass()
    {
        LairConfig current = ConfigWith(new ModelConfig
        {
            Name = "kv-residency-test",
            Path = "a.gguf",
            Placement = new PlacementConfig { KvResidency = "dram" },
        });

        // The loader rejects this candidate outright — a residency outside the declared set — yet the edit still reports
        // the field it changed with the class that says why it cannot apply to a running engine.
        string path = WriteFile("""
            models:
              - name: kv-residency-test
                path: 'a.gguf'
                placement:
                  kv_residency: gpu

            """);

        FieldChange edit = Assert.Single(ConfigMutationTable.ExtractEdits(current, path));
        Assert.Equal("models[0].placement.kv_residency", edit.Field);
        Assert.Equal("dram", edit.OldValue);
        Assert.Equal("gpu", edit.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, edit.MutationClass);
    }

    [Fact]
    public void ExtractEdits_RejectedVramCeilingCandidate_StillNamesFieldAndClass()
    {
        LairConfig current = ConfigWith(new ModelConfig
        {
            Name = "kv-residency-test",
            Path = "a.gguf",
            KvBlocks = 256,
            Budgets = [new BudgetChain
            {
                Resource = BudgetResource.Kv,
                Order = [BudgetTier.Dram, BudgetTier.Vram],
                Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 256, [BudgetTier.Vram] = 64 },
            }],
        });

        string path = WriteFile("""
            models:
              - name: kv-residency-test
                path: 'a.gguf'
                budgets:
                  kv:
                    order: [dram, vram]
                    caps: { dram_blocks: 256, vram_blocks: 0 }

            """);

        IReadOnlyList<FieldChange> edits = ConfigMutationTable.ExtractEdits(current, path);

        FieldChange edit = Assert.Single(edits);
        Assert.Equal("models[0].budgets.kv.caps.vram_blocks", edit.Field);
        Assert.Equal("64", edit.OldValue);
        Assert.Equal("0", edit.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, edit.MutationClass);
    }

    private static LairConfig ConfigWith(ModelConfig model) => new()
    {
        Server = new ServerConfig(),
        Engine = new EngineConfig(),
        Models = [model],
        Ui = new UiConfig(),
    };

    private string WriteFile(string contents)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-kv-residency-reload-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, contents);
        _tempPaths.Add(path);
        return path;
    }
}
