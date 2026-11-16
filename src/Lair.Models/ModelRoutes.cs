namespace Lair.Models;

using Lair.Contracts;
using Lair.Kv;

/// <summary>One execution stage a model's placement declares: the part of the path it names (<c>"prefill"</c>,
/// <c>"decode"</c>) and the compute device that part runs on. Stages reach a route in declaration order, and each may
/// name the boundary of the ordered path its span starts at — what a host's placement resolves onto when it splits a model across devices.</summary>
public sealed record RouteStage(string Name, string DeviceId)
{
    internal int SplitPosition { get; init; } = ModelRoute.WholeModelSplitPosition;
}

/// <summary>What one model's placement resolution supplies to route resolution: its name for every message this owns,
/// the ordered execution stages with the device each declares, the device ids the backends already opened report, and
/// the CPU quota numbers placement carries today — a thread cap and a pinned core set — plus any accelerator stream priority and SM
/// share declared for an accelerator-backed segment. Nothing here reads a config key; callers map what their configuration states onto these values.</summary>
public sealed record RouteRequest
{
    public required string ModelName { get; init; }

    /// <summary>The model's execution stages in declaration order: consecutive stages on one device collapse into one segment unless they name different boundaries of the ordered path, and a stage after another device stays its own position of the route.</summary>
    public required IReadOnlyList<RouteStage> Stages { get; init; }

    /// <summary>The device ids the opened compute backends report. A stage naming any other device resolves structurally but refuses here naming that missing backend — nothing falls back to a device that happens to work.</summary>
    public required IReadOnlyList<string> AvailableDevices { get; init; }

    /// <summary>The weight tier each opened backend reports for the device id it answers. A device named in <see cref="AvailableDevices"/> with no entry here states no device-resident posture, so its segments read host-resident weights — the posture every backend without persistent device copies reports. This is where a resolved segment's weight tier comes from; the model's KV residency names no segment tier.</summary>
    public IReadOnlyDictionary<string, WeightTier> DeviceWeightTiers { get; init; } = new Dictionary<string, WeightTier>();

    /// <summary>How placement declares this model's KV block residency — <c>Dram</c>, <c>Vram</c>, or <c>Split</c> — landed on every resolved segment as the block residencies that segment can step over. An accelerator-resident declaration no opened pool can back refuses naming the model and what is missing. This axis is independent of a segment's weight tier: a whole-model accelerator span over device-resident weights still steps sequences whose blocks live wherever this residency says they do.</summary>
    public BlockResidencyMode KvResidency { get; init; } = BlockResidencyMode.Dram;

    /// <summary>True when the opened compute backends can hold a device-backed KV mirror — the seam a device-backed block pool opens its persistent device copy through. An accelerator-resident KV declaration with this false refuses naming that seam rather than falling back to host residency; a host-only residency never consults it.</summary>
    public bool DeviceBackedKvMirrorSupport { get; init; }

    /// <summary>The model's hard cap on worker threads (1 or greater); null leaves each CPU segment at the engine's own thread choice.</summary>
    public int? ThreadQuota { get; init; }

    /// <summary>The logical cores pinned to this model: its core-set share, claimed by every resolved CPU segment. Empty means unpinned.</summary>
    public IReadOnlySet<int> CoreSet { get; init; } = new HashSet<int>();

    /// <summary>A compute stream priority declared for this model's accelerator-backed segments (1 or greater); null claims none. Nothing a placement key sets today — these two fields are the internal share vocabulary a resolved route carries so later device execution consumes it unchanged, and resolution validates them without promising any enforcement.</summary>
    public int? AcceleratorStreamPriority { get; init; }

    /// <summary>The streaming-multiprocessor share declared for this model's accelerator-backed segments (greater than 0 and at most 100); null claims none. Declared the same internal way as the stream priority.</summary>
    public double? AcceleratorSmSharePercent { get; init; }

    /// <summary>An explicit batch cap for the resolved segments; null takes the engine's own ceiling on sequences one step advances. A value outside that range refuses naming each offending segment.</summary>
    public int? BatchCap { get; init; }
}

/// <summary>
/// Resolves a model's declared placement stages into the route its execution path takes, and owns every rule that
/// resolution enforces: consecutive stages landing on one device at one boundary position collapse into one CPU span so a
/// CPU-only model names exactly one segment over DRAM-resident weights; stages landing on different devices, or naming
/// distinct boundaries of the ordered path, stay separate segments in declaration order — which is how a host's layer split
/// reaches execution. Each
/// resolved segment carries its compute device identity, weight tier, batch cap against the engine's own ceiling, and
/// quota taken from what placement declares today — plus any accelerator stream priority and SM share a caller declared on an
/// accelerator-backed segment, landed unchanged for later execution to consume and refused here only when out of range. No new config key is
/// read here. A stage naming a device no opened backend reports fails naming the model, the offending segment, and that missing backend rather
/// than falling back to CPU silently, exactly as every other unimplemented capability reports. The resolved value set is what later step
/// composition consumes unchanged; nothing in this resolution executes it.
/// </summary>
public static class ModelRoutes
{
    /// <summary>The stage name a decode-side placement declaration carries.</summary>
    public const string DecodeStage = "decode";

    /// <summary>The stage name a prefill-side placement declaration carries.</summary>
    public const string PrefillStage = "prefill";

    /// <summary>
    /// Resolves one model's placement into its route value set. Callers supply what their own configuration already
    /// states — the ordered execution stages with the device each declares, the model's thread cap and pinned core set,
    /// and the device ids the backends they opened report — and this owns how those become segments: collapse by consecutive
    /// stages agreeing on device, residency, and boundary position, engine ceilings as batch caps, CPU quota from placement
    /// alone, and the model's KV residency landed as the block residencies every resolved segment can step over. An
    /// unresolvable declaration (no stage, an unnamed or repeated stage, a blank device, a cap outside the engine's range,
    /// a KV residency no opened pool can back, a device no backend reports) throws naming the model and what it belongs to.
    /// </summary>
    public static ModelRoute Resolve(RouteRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.Stages);
        ArgumentNullException.ThrowIfNull(request.AvailableDevices);

        if (string.IsNullOrWhiteSpace(request.ModelName))
        {
            throw new ModelException(
                "route resolution declares no model name: every refusal this reports names the model its route belongs to");
        }

        string model = request.ModelName;

        if (request.Stages.Count == 0)
        {
            throw new ModelException(
                $"model '{model}' declares no execution stage: a route must name at least the device its work runs on");
        }

        HashSet<string> seenStages = new(StringComparer.Ordinal);
        List<DeviceSegment> segments = [];

        // The KV residency axis is one declaration for the whole model, so it resolves once here rather than per stage: which
        // device a span runs on says nothing about where this model's blocks live, and an accelerator-resident declaration is
        // answered by whether the caller opened the mirror seam those pools copy through — never by falling back to host
        // residency that happens to run.
        IReadOnlyList<WeightTier> blockResidencies = ResolveBlockResidencies(request);

        foreach (RouteStage stage in request.Stages)
        {
            if (string.IsNullOrWhiteSpace(stage.Name))
            {
                throw new ModelException(
                    $"model '{model}' declares an execution stage with no name: every named stage decides which position of the route its device lands on");
            }

            if (!seenStages.Add(stage.Name))
            {
                throw new ModelException(
                    $"model '{model}' declares execution stage '{stage.Name}' more than once: one stage names one position of the route");
            }

            if (string.IsNullOrWhiteSpace(stage.DeviceId))
            {
                throw new ModelException(
                    $"model '{model}', stage '{stage.Name}': an execution stage must name the compute device it runs on");
            }

            WeightTier tier = TierForStage(request, stage);

            // Consecutive stages landing on one device at one residency at the same boundary position collapse into a single
            // segment: today's prefill and decode both naming 'cpu' is one CPU span of the model, not two segments that would
            // make every step look like a crossing. A repeat after another device stays its own position — that ordering is
            // exactly what a split route means, and nothing here reorders it. Two stages that name distinct boundaries stay
            // distinct positions even when their device and residency agree: the boundary is part of what identifies a span.
            if (segments.Count > 0
                && segments[^1].DeviceId == stage.DeviceId
                && segments[^1].WeightTier == tier
                && segments[^1].SplitPosition == stage.SplitPosition)
            {
                continue;
            }

            DeviceSegment segment = new()
            {
                DeviceId = stage.DeviceId,
                WeightTier = tier,
                BlockResidencyTiers = blockResidencies,
                SplitPosition = stage.SplitPosition,
                BatchCap = request.BatchCap ?? Engine.TransformerModel.MaxBatchSequences,
                Quota = IsCpuDevice(stage.DeviceId)
                    ? new SegmentQuota { Threads = request.ThreadQuota, Cores = request.CoreSet }
                    : new SegmentQuota { StreamPriority = request.AcceleratorStreamPriority, SmSharePercent = request.AcceleratorSmSharePercent },
            };

            ValidateSegment(request, segments.Count, segment);
            segments.Add(segment);
        }

        return new ModelRoute { Segments = [.. segments] };
    }

    // The tier a stage's weights land at: the weight tier the opened compute backend on that stage's device reports. A
    // device whose opened backend states no device-resident posture reads host-resident weights, which is what every
    // backend without persistent device copies reports today. The model's KV residency never enters here: it decides which
    // pools hold a sequence's blocks and what those segments can step over, not which tier a span's weights sit at.
    private static WeightTier TierForStage(RouteRequest request, RouteStage stage) =>
        request.DeviceWeightTiers.TryGetValue(stage.DeviceId, out WeightTier tier) ? tier : WeightTier.Dram;

    // The block residencies the declared KV residency lands on: system memory alone for a host-only declaration, accelerator
    // memory for an accelerator-resident one, and both tiers for a divided reservation — because a split pool holds some of its
    // ids in each range, so a step over any segment of this model can address either. A declaration naming accelerator-backed
    // blocks with no opened mirror seam refuses here naming the seam those pools run in, exactly as every other capability
    // this build declares before it carries; nothing falls back to residency the caller did not ask for.
    private static IReadOnlyList<WeightTier> ResolveBlockResidencies(RouteRequest request) => request.KvResidency switch
    {
        BlockResidencyMode.Dram => [WeightTier.Dram],

        BlockResidencyMode.Vram or BlockResidencyMode.Split when !request.DeviceBackedKvMirrorSupport => throw new ModelException(
            $"model '{request.ModelName}': kv residency '{BlockResidencyModeNames.Of(request.KvResidency)}' declares its KV blocks device-backed, and no opened backend can hold a " +
            $"device-backed KV mirror for them — those pools keep their persistent device copy in '{ModelRoute.CudaBackendSeam}', which this open has none, so the residency is refused " +
            $"rather than run against host-resident pools"),

        BlockResidencyMode.Vram => [WeightTier.Vram],

        // A divided reservation carries both residencies: its leading ids sit in a device-backed range and its tail stays host-only,
        // so every segment of this route can step over either tier.
        BlockResidencyMode.Split => [WeightTier.Dram, WeightTier.Vram],

        _ => throw new ModelException(
            $"model '{request.ModelName}': kv residency '{request.KvResidency}' is not a residency a placement declares " +
            $"({string.Join(", ", BlockResidencyModeNames.DeclaredNames)})"),
    };

    // Every rule one resolved segment must satisfy before it can carry work: its batch cap inside the ceiling the engine's
    // own step math honors, a quota whose numbers mean something for the device class they name, and a compute backend
    // actually opened on the device it runs on. Each refusal names the model plus this segment by position and identity so
    // a load failure says which part of the route to change.
    private static void ValidateSegment(RouteRequest request, int index, DeviceSegment segment)
    {
        string model = request.ModelName;

        if (segment.BatchCap is < 1 or > Engine.TransformerModel.MaxBatchSequences)
        {
            throw new ModelException(
                $"model '{model}', {segment.Describe(index)}: batch cap '{segment.BatchCap}' is outside the range " +
                $"1..{Engine.TransformerModel.MaxBatchSequences} a step may advance sequences on — omit it to take that ceiling");
        }

        // A declared boundary names the layer this span starts at, so anything else says nothing about where a split lands:
        // the whole-model position is what an unsplit route carries; every other value must be a real layer index or — for a final segment only, standing on no transformer layers of its own while the output head still runs there — exactly this model's layer count. The engine bounds it against the file it opens over as it always has.
        if (segment.SplitPosition != ModelRoute.WholeModelSplitPosition && segment.SplitPosition < 0)
        {
            throw new ModelException(
                $"model '{model}', {segment.Describe(index)}: split position '{segment.SplitPosition}' is invalid — a declared boundary names the layer its span starts at, or nothing to keep this model one span");
        }

        if (IsCpuDevice(segment.DeviceId))
        {
            if (segment.Quota.Threads is < 1)
            {
                throw new ModelException(
                    $"model '{model}', {segment.Describe(index)}: thread quota '{segment.Quota.Threads}' is invalid — a CPU segment's worker cap must be 1 or greater, or absent to leave it at the engine's own choice");
            }

            foreach (int core in segment.Quota.Cores)
            {
                if (core < 0)
                {
                    throw new ModelException(
                        $"model '{model}', {segment.Describe(index)}: core set declares logical core '{core}'; a CPU segment's pinned cores must be non-negative ids");
                }
            }

            return;
        }

        // A segment on any other device class resolves — the route vocabulary names a split or hybrid shape before that
        // backend exists — but only when the caller's opened backends report its device, and it carries an unclaimed quota
        // because nothing has been allocated against a device this build cannot address. With no such backend opened the
        // refusal names the missing compute backend rather than running the model on some other device: exactly what every
        // other unimplemented capability reports, and never a silent CPU fallback.
        bool backed = false;
        foreach (string opened in request.AvailableDevices)
        {
            if (string.Equals(opened, segment.DeviceId, StringComparison.Ordinal))
            {
                backed = true;
                break;
            }
        }

        if (!backed)
        {
            throw new ModelException(
                $"model '{model}', {segment.Describe(index)}: no compute backend reports device '{segment.DeviceId}' — " +
                $"accelerator-backed segments run in '{ModelRoute.CudaBackendSeam}', which this build does not carry, and " +
                $"this model has [{DescribeDevices(request.AvailableDevices)}] opened for it; a route naming another device is refused rather than run on CPU");
        }

        // A non-CPU segment's worker-thread and core claims stay empty: this build cannot address that device's threads or cores, so
        // resolution records the shape without inventing numbers against it. The accelerator share fields a caller declared land on the
        // backed segment unchanged for later execution to consume — validated here for range only, never enforced against any device ledger yet.
        if (segment.Quota.StreamPriority is < 1)
        {
            throw new ModelException(
                $"model '{model}', {segment.Describe(index)}: stream priority '{segment.Quota.StreamPriority}' is invalid — an accelerator segment's compute stream priority must be 1 or greater, or absent to claim none");
        }

        if (segment.Quota.SmSharePercent is not null and (<= 0.0 or > 100.0))
        {
            throw new ModelException(
                $"model '{model}', {segment.Describe(index)}: SM share '{segment.Quota.SmSharePercent}' is invalid — an accelerator segment's streaming-multiprocessor share must be greater than 0 and at most 100, or absent to claim none");
        }

        return;
    }

    private static string DescribeDevices(IReadOnlyList<string> devices) =>
        string.Join(", ", devices.Select(device => $"'{device}'"));

    private static bool IsCpuDevice(string deviceId) => string.Equals(deviceId, ModelRoute.CpuDeviceId, StringComparison.Ordinal);
}
