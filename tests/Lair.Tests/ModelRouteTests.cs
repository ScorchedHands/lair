using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Http;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Load-time route resolution: the ordered device segments a model's execution path resolves to, from placement input
/// alone. A CPU-only placement collapses its stages into one executable CPU segment carrying this model's thread cap and
/// pinned core share; split and hybrid shapes keep their segments in stage order and resolve as value sets without any
/// accelerator present — an accelerator-backed segment carries a declared stream priority or SM share unchanged, refused only when out of
/// range, with nothing enforced against a device ledger yet; every unresolvable declaration fails naming the model and the offending segment,
/// including a segment whose device no opened backend reports — which is what an engine open refuses too, never falling back to CPU.
/// </summary>
[Trait("Category", "Loader")]
[Collection("model-loaded")] // shares one serialized lane with the other suites that open engines on this box
public sealed class ModelRouteTests : IDisposable
{
    // A device identity standing in for an opened accelerator backend: resolution accepts a segment naming it when the
    // caller declares such a device available, and every execution path refuses it because no such backend exists here.
    private const string AcceleratorDeviceId = "cuda";

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

    // ---------------------------------------------------------------- CPU-only resolution

    [Fact]
    public void ModelRoutes_CpuPlacementStages_CollapseIntoOneSegmentCarryingPlacementQuota()
    {
        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-cpu-placement",
            Stages = [new(ModelRoutes.PrefillStage, "cpu"), new(ModelRoutes.DecodeStage, "cpu")],
            AvailableDevices = ["cpu"],
            ThreadQuota = 4,
            CoreSet = new HashSet<int> { 0, 1, 2, 3 },
        });

        Assert.Single(route.Segments);

        DeviceSegment segment = route.Segments[0];
        Assert.Equal("cpu", segment.DeviceId);
        Assert.Equal(WeightTier.Dram, segment.WeightTier);
        Assert.Equal(TransformerModel.MaxBatchSequences, segment.BatchCap);
        Assert.Equal(4, segment.Quota.Threads);
        Assert.True(segment.Quota.Cores.IsSupersetOf([0, 1, 2, 3]));
    }

    [Fact]
    public void ModelRoutes_UnpinnedCpuPlacement_LeavesThreadShareAtTheEnginesOwnChoice()
    {
        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-cpu-unpinned",
            Stages = [new(ModelRoutes.DecodeStage, "cpu")],
            AvailableDevices = ["cpu"],
        });

        DeviceSegment segment = Assert.Single(route.Segments);
        Assert.Null(segment.Quota.Threads);
        Assert.Empty(segment.Quota.Cores);
    }

    [Fact]
    public void ModelPlan_AFileAlone_ResolvesOneExecutableCpuSegment()
    {
        string path = SyntheticLlamaGguf.Write("route-plan-default");
        _tempFiles.Add(path);

        using GgufFile gguf = GgufFile.Open(path);
        ModelPlan plan = ModelPlans.Resolve(gguf);

        DeviceSegment segment = Assert.Single(plan.Route.Segments);
        Assert.Equal(ModelRoute.CpuDeviceId, segment.DeviceId);
        Assert.Equal(WeightTier.Dram, segment.WeightTier);
        Assert.Equal(TransformerModel.MaxBatchSequences, segment.BatchCap);
    }

    // ---------------------------------------------------------------- split and hybrid shapes as value sets

    [Fact]
    public void ModelRoutes_SplitStages_KeepTheirOrder_AndResolveWithoutTheDevicePresent()
    {
        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-split",
            Stages =
            [
                new(ModelRoutes.PrefillStage, "cpu"),
                new("split_block", AcceleratorDeviceId),
                new(ModelRoutes.DecodeStage, "cpu"),
            ],
            AvailableDevices = ["cpu", AcceleratorDeviceId],
            ThreadQuota = 2,
            CoreSet = new HashSet<int> { 5 },
        });

        // Stage order survives: the CPU span before the accelerator block and the one after it stay separate positions of
        // the route instead of collapsing around the middle segment.
        Assert.Equal(3, route.Segments.Count);
        Assert.Equal("cpu", route.Segments[0].DeviceId);
        Assert.Equal(AcceleratorDeviceId, route.Segments[1].DeviceId);
        Assert.Equal("cpu", route.Segments[2].DeviceId);

        // The CPU segments claim this model's core-set share and thread cap; the accelerator segment carries its shape with
        // nothing claimed against a device this build cannot address.
        Assert.Equal(2, route.Segments[0].Quota.Threads);
        Assert.Contains(5, route.Segments[0].Quota.Cores);
        Assert.Equal(2, route.Segments[2].Quota.Threads);
        Assert.Contains(5, route.Segments[2].Quota.Cores);
        Assert.Null(route.Segments[1].Quota.Threads);
        Assert.Empty(route.Segments[1].Quota.Cores);

        foreach (DeviceSegment segment in route.Segments)
        {
            Assert.Equal(TransformerModel.MaxBatchSequences, segment.BatchCap);
        }
    }

    [Fact]
    public void ModelRoutes_BackendedAcceleratorSegment_CarriesDeclaredStreamPriorityAndSmShare_Unchanged()
    {
        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-accelerator-shares",
            Stages =
            [
                new(ModelRoutes.PrefillStage, "cpu"),
                new("split_block", AcceleratorDeviceId),
                new(ModelRoutes.DecodeStage, "cpu"),
            ],
            AvailableDevices = ["cpu", AcceleratorDeviceId],
            ThreadQuota = 2,
            CoreSet = new HashSet<int> { 5 },
            AcceleratorStreamPriority = 3,
            AcceleratorSmSharePercent = 25.0,
        });

        // The declared accelerator shares land on the device-backed segment unchanged — vocabulary a later execution reads, claimed against no ledger now.
        Assert.Equal(3, route.Segments[1].Quota.StreamPriority);
        Assert.Equal(25.0, route.Segments[1].Quota.SmSharePercent);

        // CPU segments keep their placement quota and carry neither accelerator field: the vocabulary names shares for a device they do not run on.
        foreach (int index in new[] { 0, 2 })
        {
            Assert.Null(route.Segments[index].Quota.StreamPriority);
            Assert.Null(route.Segments[index].Quota.SmSharePercent);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-4)]
    public void ModelRoutes_InvalidAcceleratorStreamPriority_Refuses_NamingModelAndSegment(int streamPriority)
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-bad-stream-priority",
            Stages = [new(ModelRoutes.DecodeStage, AcceleratorDeviceId)],
            AvailableDevices = ["cpu", AcceleratorDeviceId],
            AcceleratorStreamPriority = streamPriority,
        }));

        Assert.Contains("model 'route-bad-stream-priority'", error.Message);
        Assert.Contains($"segment 0 ('{AcceleratorDeviceId}'", error.Message, StringComparison.Ordinal);
        Assert.Contains("stream priority", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.5)]
    [InlineData(120.0)]
    public void ModelRoutes_InvalidAcceleratorSmShare_Refuses_NamingModelAndSegment(double smShare)
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-bad-sm-share",
            Stages = [new(ModelRoutes.DecodeStage, AcceleratorDeviceId)],
            AvailableDevices = ["cpu", AcceleratorDeviceId],
            AcceleratorSmSharePercent = smShare,
        }));

        Assert.Contains("model 'route-bad-sm-share'", error.Message);
        Assert.Contains($"segment 0 ('{AcceleratorDeviceId}'", error.Message, StringComparison.Ordinal);
        Assert.Contains("streaming-multiprocessor share must be greater than 0 and at most 100", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ModelRoutes_AcceleratorOnlyShape_ResolvesAsRouteValues_UnderAcceleratorResidency()
    {
        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-accelerator-only",
            Stages = [new(ModelRoutes.DecodeStage, AcceleratorDeviceId)],
            AvailableDevices = ["cpu", AcceleratorDeviceId],
            KvResidency = WeightTier.Vram,
        });

        DeviceSegment segment = Assert.Single(route.Segments);
        Assert.Equal(AcceleratorDeviceId, segment.DeviceId);
        Assert.Equal(WeightTier.Vram, segment.WeightTier);
    }

    [Fact]
    public void ModelRoutes_AcceleratorResidencyOnACpuStage_FailsNamingModelAndSegment()
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-vram-on-cpu",
            Stages = [new(ModelRoutes.DecodeStage, "cpu")],
            AvailableDevices = ["cpu"],
            KvResidency = WeightTier.Vram,
        }));

        Assert.Contains("model 'route-vram-on-cpu'", error.Message);
        Assert.Contains("stage 'decode'", error.Message);
        Assert.Contains("'vram' cannot be held by a segment on 'cpu'", error.Message);
    }

    // ---------------------------------------------------------------- unresolvable declarations

    [Fact]
    public void ModelRoutes_NoStage_FailsNamingTheModel()
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-no-stage",
            Stages = [],
            AvailableDevices = ["cpu"],
        }));

        Assert.Contains("model 'route-no-stage' declares no execution stage", error.Message);
    }

    [Fact]
    public void ModelRoutes_RepeatedStageName_FailsNamingModelAndStage()
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-duplicate-stage",
            Stages = [new(ModelRoutes.DecodeStage, "cpu"), new("decode", "cpu")],
            AvailableDevices = ["cpu"],
        }));

        Assert.Contains("model 'route-duplicate-stage'", error.Message);
        Assert.Contains("execution stage 'decode' more than once", error.Message);
    }

    [Fact]
    public void ModelRoutes_StageNamingNoDevice_FailsNamingModelAndStage()
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-blank-device",
            Stages = [new(ModelRoutes.DecodeStage, " ")],
            AvailableDevices = ["cpu"],
        }));

        Assert.Contains("model 'route-blank-device', stage 'decode'", error.Message);
        Assert.Contains("must name the compute device it runs on", error.Message);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    [InlineData(64)] // wider than the ceiling a step may advance sequences on: prefill rows are not merged-batch sequences
    public void ModelRoutes_BatchCapOutsideTheEngineCeiling_FailsNamingModelAndSegment(int batchCap)
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-bad-cap",
            Stages = [new(ModelRoutes.DecodeStage, "cpu")],
            AvailableDevices = ["cpu"],
            BatchCap = batchCap,
        }));

        Assert.Contains("model 'route-bad-cap'", error.Message);
        Assert.Contains($"segment 0 ('cpu' at '{WeightTier.Dram}')", error.Message);
        Assert.Contains(
            $"outside the range 1..{TransformerModel.MaxBatchSequences}",
            error.Message);
    }

    [Fact]
    public void ModelRoutes_NegativeCoreId_FailsNamingModelAndSegment()
    {
        ModelException error = Assert.Throws<ModelException>(() => ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-bad-core",
            Stages = [new(ModelRoutes.DecodeStage, "cpu")],
            AvailableDevices = ["cpu"],
            CoreSet = new HashSet<int> { -1 },
        }));

        Assert.Contains("model 'route-bad-core'", error.Message);
        Assert.Contains("segment 0 ('cpu' at 'Dram')", error.Message);
        Assert.Contains("logical core '-1'", error.Message);
    }

    [Fact]
    public void ModelRoutes_AcceleratorSegmentWithNoBackend_FailsNamingModelSegmentAndMissingBackend()
    {
        // The host-shaped call: only the CPU backend this service opens is available, so a stage naming another device
        // refuses here instead of reaching an engine that would run the model somewhere else.
        ModelException error = Assert.Throws<ModelException>(() => ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-missing-backend",
            Stages = [new(ModelRoutes.DecodeStage, AcceleratorDeviceId)],
            AvailableDevices = ["cpu"],
        }));

        Assert.Contains("model 'route-missing-backend'", error.Message);
        Assert.Contains("segment 0 ('cuda' at 'Dram')", error.Message);
        Assert.Contains("no compute backend reports device 'cuda'", error.Message);
    }

    // ---------------------------------------------------------------- engine open against the resolved route

    [Fact]
    public void EngineOpen_CpuRouteFromPlacement_LandsTheResolvedValuesOnThePlan()
    {
        string path = SyntheticLlamaGguf.Write("route-engine-cpu");
        _tempFiles.Add(path);

        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-engine-cpu",
            Stages = [new(ModelRoutes.PrefillStage, "cpu"), new(ModelRoutes.DecodeStage, "cpu")],
            AvailableDevices = ["cpu"],
            ThreadQuota = 3,
            CoreSet = new HashSet<int> { 0, 1 },
        });

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend backend = new(threads: 1);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 64, logLine: null, PrefillKernel.GemV, route);

        DeviceSegment segment = Assert.Single(model.Plan.Route.Segments);
        Assert.Equal("cpu", segment.DeviceId);
        Assert.Equal(WeightTier.Dram, segment.WeightTier);
        Assert.Equal(route.Segments[0].BatchCap, segment.BatchCap);
        Assert.Equal(3, segment.Quota.Threads);
        Assert.True(segment.Quota.Cores.IsSupersetOf([0, 1]));
    }

    [Fact]
    public void EngineOpen_RouteCrossingToAnUnavailableDevice_FailsNamingTheSegment_AndNeverRunsOnCpu()
    {
        string path = SyntheticLlamaGguf.Write("route-engine-crossing");
        _tempFiles.Add(path);

        // Resolved against a synthetic available device: the route value set is real material, and opening this engine over it is what refuses.
        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "route-engine-crossing",
            Stages = [new(ModelRoutes.PrefillStage, "cpu"), new("split_block", AcceleratorDeviceId), new(ModelRoutes.DecodeStage, "cpu")],
            AvailableDevices = ["cpu", AcceleratorDeviceId],
        });

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend backend = new(threads: 1);

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, backend, maxContextTokens: 64, logLine: null, PrefillKernel.GemV, route));

        Assert.Contains("segment 1 ('cuda' at 'Dram') names a device this engine has no backend for", error.Message);
        Assert.Contains("the opened compute backend reports 'cpu'", error.Message);
        Assert.Contains("refused rather than run on CPU", error.Message);
    }

    // ---------------------------------------------------------------- host placement to resolved route values

    [Fact]
    [Trait("Category", "Config")]
    public void HostPlacement_ResolvesOntoTheOpenedModelsPlan()
    {
        string path = SyntheticLlamaGguf.Write("route-host-placement");
        _tempFiles.Add(path);

        LairConfig config = YamlConfigLoader.Load(WriteYaml($"""
            engine:
              threads: 2
            models:
              - name: route-host
                path: '{path}'
                max_context_tokens: 128
                kv_blocks: 8
                placement:
                  device: cpu
                  prefill: cpu
                  core_set: "0-1"
                  thread_quota: 3
            """));

        using GenerationService service = GenerationService.Open(config);

        DeviceSegment segment = Assert.Single(service.ResolvedRoute.Segments);
        Assert.Equal("cpu", segment.DeviceId);
        Assert.Equal(WeightTier.Dram, segment.WeightTier);
        Assert.Equal(TransformerModel.MaxBatchSequences, segment.BatchCap);
        Assert.Equal(3, segment.Quota.Threads);
        Assert.True(segment.Quota.Cores.IsSupersetOf([0, 1]));
    }

    private string WriteYaml(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-route-config-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, yaml);
        _tempFiles.Add(path);
        return path;
    }
}
