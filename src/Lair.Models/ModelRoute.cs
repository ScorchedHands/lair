namespace Lair.Models;

/// <summary>
/// The resolved execution path of one model: an ordered set of device segments, each naming the compute device it
/// runs on, where its weights are held while it computes, the batch cap a step may advance on it, and the quota a
/// step holds against it. A route is a value set resolved once at load — consecutive stages that land on one device
/// collapse into a single segment, so every CPU-only model today carries exactly one CPU segment over DRAM-resident
/// weights — and a hybrid shape keeps its segments in stage order rather than collapsing to a single device label.
/// Nothing splits a sequence across segments: the whole batch of a step rides the same route, and the ordering says
/// which segment's work runs first.
/// </summary>
public sealed record ModelRoute
{
    /// <summary>The device id every segment of a CPU-only route names, and the id the CPU compute backend reports.</summary>
    public const string CpuDeviceId = "cpu";

    /// <summary>
    /// The segments in execution order: what a step advances first is <c>Segments[0]</c>, and each later segment's work
    /// follows it over the same batch. Never empty — a route naming no segment cannot be run, and only the resolution
    /// paths build one.
    /// </summary>
    public required IReadOnlyList<DeviceSegment> Segments { get; init; }

    /// <summary>The single CPU segment a plan carries when nothing says where else its stages could run.</summary>
    internal static ModelRoute CpuDefault() => new()
    {
        Segments = [new DeviceSegment
        {
            DeviceId = CpuDeviceId,
            WeightTier = WeightTier.Dram,
            BatchCap = Engine.TransformerModel.MaxBatchSequences,
            Quota = new SegmentQuota(),
        }],
    };
}

/// <summary>
/// One device span of a model's execution path. A segment is whole-model material today and becomes one half of a
/// split when a route names more than it: the identity says which compute backend runs this part, the tier says where
/// its weights are resident while it does, the batch cap is how many sequences one step may advance on it at once,
/// and the quota is what that step holds against it for as long as it runs.
/// </summary>
public sealed record DeviceSegment
{
    /// <summary>The compute device this segment runs on: a backend-reported id (<see cref="ModelRoute.CpuDeviceId"/>, or an accelerator id once one exists), not a placement label.</summary>
    public required string DeviceId { get; init; }

    /// <summary>Where this segment's weights are held while it computes. DRAM-resident is the only residency any engine reads today.</summary>
    public WeightTier WeightTier { get; init; } = WeightTier.Dram;

    /// <summary>The largest number of sequences one step may advance on this segment at once: resolved from the engine's own batch ceiling, and a value outside it refuses rather than silently widening some other buffer.</summary>
    public required int BatchCap { get; init; }

    /// <summary>The quota a step holds against this segment for its whole duration. CPU segments take their core-set share here; a segment on a device no backend reports never reaches execution, so it carries the shape without claiming any resource.</summary>
    public SegmentQuota Quota { get; init; } = new();

    /// <summary>The identity an error message names this segment by: its position in the route and the device it runs on.</summary>
    public string Describe(int index) => $"segment {index} ('{DeviceId}' at '{WeightTier}')";
}

/// <summary>Where a resource is resident while compute reads it. DRAM is what every engine path holds today; an accelerator tier stays vocabulary until the device exists to back it.</summary>
public enum WeightTier
{
    /// <summary>Resident in system memory: where weights, KV blocks, and activation buffers live for every model served today.</summary>
    Dram,

    /// <summary>Resident on an accelerator's own memory. Named by the vocabulary only until a backend reports such a device; a segment at this tier refuses before any engine allocates against it.</summary>
    Vram,
}

/// <summary>
/// The share of one segment's compute a step holds: for CPU segments that is its worker-thread cap and its pinned
/// core set — the same two numbers placement declares today. A null thread cap leaves the segment at the engine's own
/// thread choice; an empty core set claims no pinning, which is what every unpinned model resolves to now. The accelerator
/// share fields are the matching vocabulary for a device-backed segment: a stream priority and a streaming-multiprocessor
/// share carried on the route so later execution consumes them unchanged — null until something declares them, because quota
/// for a device class with no backend stays unclaimed rather than invented.
/// </summary>
public sealed record SegmentQuota
{
    /// <summary>Hard cap on the worker threads this segment participates with (1 or greater); null leaves it at the engine's own thread choice.</summary>
    public int? Threads { get; init; }

    /// <summary>The logical cores pinned to this segment's workers: its core-set share. Empty means unpinned, and a non-empty set is what admission counts against another model's claim.</summary>
    public IReadOnlySet<int> Cores { get; init; } = new HashSet<int>();

    /// <summary>A compute stream priority for an accelerator-backed segment (1 or greater); null claims no stream ordering. A CPU segment never carries one — the field names vocabulary a later device execution reads, not something this build enforces.</summary>
    public int? StreamPriority { get; init; }

    /// <summary>The share of an accelerator's streaming multiprocessors this segment holds (greater than 0 and at most 100); null claims none. As with the stream priority, a CPU segment leaves it unclaimed.</summary>
    public double? SmSharePercent { get; init; }
}
