using System.Collections.Concurrent;
using System.Globalization;
using Lair.Http.Monitoring;
using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;
using Lair.Host.Config;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;
using Lair.Templating.Minja;
using Lair.Tokenizer;
using Microsoft.Extensions.Logging;

namespace Lair.Http;

/// <summary>
/// Single-model generation facade: opens one configured GGUF model (weights, forward engine, tokenizer,
/// chat template, paged KV pool) and runs <see cref="GenerationRequest"/>s against it. Requests are admitted
/// to a continuous-batching scheduler, so admitted sequences decode together in shared steps and a long
/// prompt never blocks shorter ones. Admission refusals keep one coherent taxonomy naming the limit kind that blocked:
/// <see cref="EngineBusyException"/> for kv_capacity (the pool cannot back this request's prompt plus token budget) or for the model's
/// concurrency quota (<c>max_concurrent_requests</c> fully held by admitted sequences), and <see cref="ContextLengthExceededException"/> for the
/// resolved token-budget window; each names the served model.
/// </summary>
public sealed class GenerationService : IDisposable
{
    private readonly GgufFile _file;
    private readonly CpuBackend _backend;

    // The dedicated prefill lane this model opened with, when placement declared one: the second execution backend prompt windows run on and the load-resolved split that capped it. Worker pools, masks, activation buffers, and collision posture are facts of this pair for the engine's whole lifetime — a live thread-quota edit re-derives only the decode side from them, and an unmappable edit is refused against them rather than half-applied. Null for every shared-posture model, where nothing here changes behavior.
    private readonly CpuBackend? _prefillsBackend;
    private readonly PrefillLaneLayout? _prefillsLane;

    // The accelerator compute backend a declared layer split opened beside this service's decode-side CPU backend: the execution backend resolved for the route segment that names device 'cuda'. It holds no worker pool and answers no scheduler or monitor fact — the decode-side CPU backend keeps every one of those. Null whenever no split was declared, which is every model served today.
    private readonly CudaBackend? _splitExecution;

    // The accelerator backend this open owns solely as the storage owner for a device-backed KV pool's persistent block copies: opened when placement declares an accelerator-resident or divided reservation and no layer split opened one already. It runs no forward, holds no worker pool, and copies no weight tensor — its only seam is the mirror the pool opens through it, while this service's CPU backend keeps owning the arena every attention read addresses. Null for every host-only reservation.
    private readonly CudaBackend? _kvMirrorExecution;

    private readonly TransformerModel _model;
    private readonly GgufTokenizer _tokenizer;
    private readonly MinjaChatTemplate? _chatTemplate;
    private readonly TemplateCapabilities? _chatTemplateCapabilities;

    // The output-parser plan resolved once at model load (reference-swapped under the reload gate when a
    // new-requests-only tool key changes): request behavior consumes this resolved value instead of re-deciding.
    private OutputParserPlan _outputParser;

    // The reasoning-marker pair this model's vocabulary was proven to carry whole at open (null when it cannot), plus
    // the reason it could not — a declared reasoning ceiling is only ever applied against that proof, at load and on reload.
    private readonly ReasoningRegionMarkers? _reasoningMarkers;
    private readonly string _markerRefusalDetail;

    private readonly ILogger? _logger;
    private readonly BlockPool _pool;
    private readonly GenerationScheduler _scheduler;
    private EngineMetricsBus? _metrics;

    // The host device ledger this engine's claims register against. Supplied by hosts that keep host-scoped accounting
    // across several opened engines; otherwise the ledger a host opens over the tiers its pools back, with each opened
    // tier stating the explicit unbounded capacity posture — a public total no one declared is never invented. Weight,
    // KV, and prefill claims land here in registration order; the ledger records what the chains decided and bounds
    // nothing that the existing pool and scheduler seams do not already bound.
    private DeviceLedger? _ledger;

    // The host compute accounting this engine's worker claim registered against, and the share its own backend runs with. Supplied by hosts
    // that keep capacity accounting across several opened engines — then earlier claims cap this model's participation rather than refusing
    // it — and otherwise derived from this process's reported topology over an empty ledger. The grant is what monitor facts report as
    // requested against granted, so a partial seat reads the same way on every surface that reports worker counts.
    private ComputeCapacityLedger? _compute;
    private ComputeGrant? _computeGrant;

    // The placement this engine's worker claim seated on: the node (or nodes) placement chose for this model beside the posture it
    // declared and the posture placement applied. Established when the engine opens — a live edit of the spread key records the new
    // preference for later placement decisions while this value, the opened worker pool, and the posture its weight pages first-touched
    // to stay exactly as the open resolved them.
    private NumaPlacement? _placement;

    // The report-only update candidate this model's pinned pack has against the hub's current HEAD, resolved once at
    // open through the transport seam. Null for a model with no published source, no candidate, or a check that could
    // not be consulted. It feeds only the monitor's additive candidate fact — never the load path, a fetch, or a pin.
    private string? _updateCandidate;

    // Generations currently publishing monitor lifecycle frames, keyed by their stream request id — the in-flight
    // half of a resync state dump. Trackers are added at admission and removed when their completion frame publishes.
    private readonly ConcurrentDictionary<string, RequestTracker> _monitoredRequests = new(StringComparer.Ordinal);

    // Change-application state: the configuration currently in force (restart-only refusals keep their old values),
    // the advertised model facts recomputed whenever a live or new-requests field changes, and the last reload attempt.
    // The applies serialize on one gate so concurrent apply calls never interleave half-applied states.
    private readonly object _reloadGate = new();
    private LairConfig _current;
    private ModelInfo _modelInfo; // reference-swapped under the reload gate so observers read a complete record
    private ReloadSummary? _lastReload;

    /// <summary>The metrics bus attached by the API host, if any — null for hosts that never attach one.</summary>
    internal EngineMetricsBus? Metrics => _metrics;

    /// <summary>The resolved dedicated prefill-lane layout this model opened with — null for every shared-posture model. The load-resolved split live quota edits map against. Internal diagnostics for hosts and tests; monitor reporting is a later seam.</summary>
    internal PrefillLaneLayout? DedicatedPrefillLanes => _prefillsLane;

    /// <summary>The worker cap the lane's execution backend holds from open — null when the model shares its single backend. Internal diagnostics for hosts and tests.</summary>
    internal int? DedicatedPrefillLaneWorkerCap => _prefillsBackend?.EffectiveWorkerCount;

    /// <summary>The placement this engine's worker claim seated on, resolved when the engine opened — null until then. Internal diagnostics for hosts and tests.</summary>
    internal NumaPlacement? ResolvedPlacement => _placement;

    // The KV block pool this engine serves from: its residency mode, device-backed range, and mirror counters are what a loaded model actually holds, read here so an observer never has to infer them from a declared placement value. Internal diagnostics for hosts and tests.
    internal BlockPool ServedBlockPool => _pool;

    // The scheduler advancing this engine's pipelines: its last composition report says which candidates merged on which segment and what kept any of them out, read here so an observer can tell where sequences actually stepped instead of inferring it from placement. Internal diagnostics for hosts and tests.
    internal GenerationScheduler ServedScheduler => _scheduler;

    // The persistent weight residency the split execution backend holds open for this model's accelerator span: the byte total across exactly the allocations its kernels read and dispose releases, so an observer sees what an opened whole-model hybrid posture actually keeps in device memory instead of inferring it from placement. Internal diagnostics for hosts and tests; null whenever no layer split was declared, and zero on a host-mapped posture whose weights never enter persistent device memory.
    internal long? SplitExecutionPersistentDeviceWeightBytes => _splitExecution?.PersistentDeviceWeightBytes;

    /// <summary>The allocation count behind that residency — one per weight tensor the copy posture has cached. Internal diagnostics for hosts and tests.</summary>
    internal int? SplitExecutionPersistentDeviceWeightAllocations => _splitExecution?.PersistentDeviceWeightAllocations;

    // Engine-side accelerator accounting across every compute backend this open holds: device-measured kernel busy time,
    // runtime copy tallies, and the default-copy weight residency currently held, summed across instances. A host-mapped
    // instance contributes its own zero weight residency rather than a fabricated reading, and an open with no accelerator
    // backend answers nothing at all — which is how CPU-only snapshots keep these facts absent instead of zero-filled.
    internal AcceleratorAccounting? OpenedAcceleratorAccounting => BuildAcceleratorAccounting();

    // The tier the ledger seated this model's whole-reservation KV claim at — what a live soft-cap edit retargets there, leaving any device-backed range claimed beside it on its own standing seat. Null when the model declared no KV chain to register.
    private BudgetTier? _reservationKvClaimTier;

    private bool _disposed;

    private GenerationService(
        GgufFile file,
        CpuBackend backend,
        TransformerModel model,
        GgufTokenizer tokenizer,
        MinjaChatTemplate? chatTemplate,
        TemplateCapabilities? chatTemplateCapabilities,
        OutputParserPlan outputParser,
        ReasoningRegionMarkers? reasoningMarkers,
        string markerRefusalDetail,
        ILogger? logger,
        BlockPool pool,
        GenerationScheduler scheduler,
        ModelInfo info,
        LairConfig config,
        CpuBackend? prefillsBackend = null,
        PrefillLaneLayout? prefillsLane = null,
        CudaBackend? splitExecution = null,
        CudaBackend? kvMirrorExecution = null)
    {
        _file = file;
        _backend = backend;
        _prefillsBackend = prefillsBackend;
        _prefillsLane = prefillsLane;

        // The accelerator side a declared layer split opened arrives as a fact of this open, held for the engine's whole lifetime. So does an accelerator backend opened only to own a device-backed block pool's persistent copies.
        _splitExecution = splitExecution;
        _kvMirrorExecution = kvMirrorExecution;
        _model = model;
        _tokenizer = tokenizer;
        _chatTemplate = chatTemplate;
        _chatTemplateCapabilities = chatTemplateCapabilities;
        _outputParser = outputParser;
        _reasoningMarkers = reasoningMarkers;
        _markerRefusalDetail = markerRefusalDetail;
        _logger = logger;
        _pool = pool;
        _scheduler = scheduler;
        _current = config;
        _modelInfo = info;
    }

    /// <summary>The served model's advertised facts; requests must name <see cref="ModelInfo.Id"/>. Recomputed when a hot-reloaded field changes what the engine advertises.</summary>
    public ModelInfo Model => Volatile.Read(ref _modelInfo);

    /// <summary>
    /// Capability facts detected once for the loaded model's chat template (null when the model has no chat template).
    /// These are the facts a request-time tool-precedence decision consumes to choose between the template's own tool
    /// rendering and LAIR's fenced-tool injection; they are published here so that decision has a single source of
    /// truth on the loaded-model path.
    /// </summary>
    public TemplateCapabilities? ChatTemplateCapabilities => _chatTemplateCapabilities;

    /// <summary>
    /// The output-parser plan resolved for the loaded model at load (re-resolved only when a new-requests-only tool
    /// key is reloaded). Its <c>ToolParser</c> choice decides which tool surface a chat request gets and whether its
    /// decoded text is read for calls; its reasoning-region slot stays inert until the separate recognizer fills it.
    /// </summary>
    internal OutputParserPlan ResolvedOutputParser => Volatile.Read(ref _outputParser);

    /// <summary>The id that ends a generation: the model's declared EOT id (<c>tokenizer.ggml.eot_token_id</c>) when it has one, otherwise the EOS id.</summary>
    internal int StopId => _tokenizer.EotId >= 0 ? _tokenizer.EotId : _tokenizer.EosId;

    /// <summary>Vocabulary size of the served model; token ids at or above it have no vocabulary text.</summary>
    public int VocabSize => Model.VocabSize;

    /// <summary>Worker count the served model's CPU backend actually runs with, after engine threads and the model's thread quota are combined.</summary>
    public int EffectiveWorkerCount => _backend.EffectiveWorkerCount;

    /// <summary>The configured kind of the served model as advertised on <c>/v1/models</c>. The config loader accepts only "chat" in this build; surfaces that must exclude embed models filter on this fact instead of an assumption.</summary>
    public string ModelType => _current.Models[0].Type;

    /// <summary>
    /// Attaches the metrics bus that samples this engine's counters and feeds <c>/monitor/stream</c>. The API host
    /// calls it once while building the app; hosts that never attach one simply produce no snapshots. Idempotent:
    /// a second call returns the already-attached bus.
    /// </summary>
    internal EngineMetricsBus AttachMetrics(ILoggerFactory loggerFactory, TimeSpan interval)
    {
        ArgumentNullException.ThrowIfNull(loggerFactory);
        if (_metrics is not null)
        {
            return _metrics;
        }

        PrefixCache? cache = _cacheForMetrics;
        EngineMetricsBus bus = new(Model.Id, _scheduler, _pool, cache, loggerFactory, interval);

        // The monitor's request-side state starts from the configuration in force: the recent-request ring capacity,
        // the live in-flight views a resync can dump, and the model facts snapshots carry additively.
        bus.SetRecentRequestCapacity(_current.Ui.RecentRequests);
        bus.AttachInFlightProvider(InFlightRequestViews);
        bus.AttachModelFacts(BuildMonitorFacts(_current));

        // The monitor's accelerator facts ride the compute backends this open holds, read from their own accounting as
        // each tick samples. An engine that opened no accelerator backend attaches a provider that answers null, so its
        // snapshots carry nothing in those slots rather than an idle-hardware reading.
        bus.AttachAcceleratorProvider(BuildAcceleratorAccounting);

        _metrics = bus;
        return bus;
    }

    // The route load resolved for the served model, assembly-internal so loader-side validation can read what a host's
    // placement became instead of re-deriving it: step composition reads this same value through the opened plan.
    internal ModelRoute ResolvedRoute => _model.Plan.Route;

    // Snapshot identity/placement facts straight from the configuration in force, the backend's live worker count, and
    // the formats the opened model's dense weight matrices actually carry. Quota posture rides too: the concurrency cap
    // as configured now (a hot edit re-pushes it with everything else), the resolved route's segments in execution order,
    // and — for a dedicated lane posture — the load-resolved split, each side's pinned mask, and what the engine sized
    // its lane activation family from. A shared posture reports only the posture itself; nothing here re-derives what
    // open already resolved.
    private ModelFacts BuildMonitorFacts(LairConfig config) => new(
        config.Models[0].Type,
        _backend.EffectiveWorkerCount,
        FormatCoreRanges(config.Models[0].Placement.CoreSet),
        config.Models[0].Placement.ThreadQuota,
        config.Models[0].Placement.Device,
        config.Models[0].Placement.Prefill,
        // What the opened pool actually holds blocks at, not what placement declared: a posture-neutral declaration and an explicit host-resident one both report the residency this engine's pools run, and no snapshot can name a mode its pool does not hold.
        BlockResidencyModeNames.Of(_pool.Residency),
        config.Engine.PrefillKernel,
        string.Join(", ", _model.MatmulWeightFormats),
        config.Models[0].MaxConcurrentRequests,
        BuildRouteSegmentFacts(ResolvedRoute),
        config.Models[0].Placement.PrefillLane == PlacementPrefillLaneValues.Dedicated
            ? PlacementPrefillLaneValues.Dedicated
            : PlacementPrefillLaneValues.Shared,
        PrefillClaimThreads: _prefillsLane?.EffectiveClaimThreads ?? 0,
        PrefillDecodeShareThreads: _prefillsLane?.DecodeShareThreads ?? 0,
        PrefillLaneShareThreads: _prefillsBackend?.EffectiveWorkerCount ?? 0,
        PrefillDecodeCoreSet: FormatCoreRanges(_prefillsLane?.DecodeCoreIds),
        PrefillLaneCoreSet: FormatCoreRanges(_prefillsLane?.PrefillCoreIds),
        PrefillActivationBytes: _model.PrefillLaneActivationBytes,
        PrefillRowWidth: _model.ResolvedPrefillRowWidth,
        EffectivePrefillRowWidth: _model.EffectivePrefillRowWidth,
        PrefillWorkingSetBytes: _model.PrefillWorkingSetBytes,
        PrefillRowClamp: _model.PrefillRowClamp?.Describe(),
        LedgerOccupancy: BuildLedgerOccupancy(_ledger),
        ComputeRequestedThreads: _computeGrant?.RequestedCores,
        ComputeGrantedThreads: _computeGrant?.GrantedCores,
        CoreOvercommit: config.Models[0].Placement.CoreOvercommit ? true : null,
        SharedCoreModels: BuildSharedCoreFacts(_compute, _computeGrant),
        PlacementNode: _placement is { NodeIndices.Count: 1 } openedPlacement ? openedPlacement.NodeIndices[0] : null,
        PlacementNodes: _placement is null ? null : string.Join(",", _placement.NodeIndices),
        PlacementSpread: _placement is null ? null : NumaSpreadNames.Of(_placement.Applied),
        PlacementSpreadRequested: _placement is null ? null : NumaSpreadNames.Of(_placement.Requested),
        PlacementNodeAllocations: _computeGrant is { NodeAllocations.Count: > 0 } seatedGrant ? seatedGrant.NodeAllocationText() : null,
        PlacementAdvisory: _placement?.Advisory,
        UpdateCandidate: _updateCandidate,
        // The CPU integer-dot posture this engine's opened backend runs, read from the fact open resolved — not re-derived here.
        CpuIsa: _backend.ResolvedSelection.ResolvedIsa,
        CpuGather: _backend.ResolvedSelection.GatherMode);

    // Sums what each compute backend this open owns answers about its own device work: cumulative kernel busy time and
    // runtime copy tallies since that instance opened, plus the weight residency it holds in persistent device memory now.
    // Both instances are fixed at construction, so a sampling tick reads two fields and one small record — no re-derivation.
    private AcceleratorAccounting? BuildAcceleratorAccounting()
    {
        CudaBackend? split = _splitExecution;
        CudaBackend? mirror = _kvMirrorExecution;

        if (split is null && mirror is null)
        {
            return null; // this serve runs no accelerator backend: there is no device-side fact to report.
        }

        long kernelBusyMicroseconds = 0;
        long runtimeTransferBytes = 0;
        long runtimeTransferCalls = 0;
        long persistentDeviceWeightBytes = 0;

        if (split is CudaBackend splitInstance)
        {
            kernelBusyMicroseconds += splitInstance.KernelBusyMicrosecondsTotal;
            runtimeTransferBytes += splitInstance.RuntimeTransferBytesTotal;
            runtimeTransferCalls += splitInstance.RuntimeTransferCallsTotal;
            persistentDeviceWeightBytes += splitInstance.PersistentDeviceWeightBytes;
        }

        if (mirror is CudaBackend mirrorInstance)
        {
            kernelBusyMicroseconds += mirrorInstance.KernelBusyMicrosecondsTotal;
            runtimeTransferBytes += mirrorInstance.RuntimeTransferBytesTotal;
            runtimeTransferCalls += mirrorInstance.RuntimeTransferCallsTotal;
            persistentDeviceWeightBytes += mirrorInstance.PersistentDeviceWeightBytes; // a mapped posture caches no device weight copy, so it adds nothing here
        }

        return new AcceleratorAccounting(kernelBusyMicroseconds, runtimeTransferBytes, runtimeTransferCalls, persistentDeviceWeightBytes);
    }

    // The report-only candidate a pinned model pack has against hub HEAD, resolved once at open: load the pack's pin
    // for this model, consult HEAD and its tree through the seam, and report the candidate summary only when HEAD
    // separates from the pin. A model with no manifest entry to report against, a candidate-free check, or a check the
    // hub could not answer (offline, gated, unreachable) all report nothing — and a failed check never blocks the
    // open, re-pins, downloads, or rewrites the manifest. This is monitor reporting only; the load path already ran.
    private static string? ResolveUpdateCandidate(ModelConfig model, IModelHubTransport transport, ILogger? logger)
    {
        try
        {
            string packDirectory = PackDirectoryOf(model);
            ModelManifest? manifest = ModelManifestStore.Load(packDirectory);
            ModelManifestEntry? entry = manifest?.Models.FirstOrDefault(candidate => string.Equals(candidate.Model, model.Name, StringComparison.Ordinal));
            if (entry is null)
            {
                return null; // no pinned entry to report a candidate against.
            }

            ModelUpdateCandidate candidate = new ModelUpdateChecker(transport).CheckAsync(entry, CancellationToken.None).GetAwaiter().GetResult();
            return candidate.HasCandidate ? candidate.Summary : null;
        }
        catch (Exception ex) when (ex is HttpRequestException or IOException or UnauthorizedAccessException or ModelFetchException or ConfigException)
        {
            logger?.LogWarning(
                "update_candidate model='{Model}' check skipped: {Reason}",
                model.Name,
                ex.Message);
            return null;
        }
    }

    // Mirrors the resolver's pack layout: the pack directory is the directory beside the model's declared file, which
    // is where its manifest pin — and so its candidate report — lives.
    private static string PackDirectoryOf(ModelConfig model)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model.Path);

        string full = Path.GetFullPath(model.Path);
        string? directory = Path.GetDirectoryName(full);
        if (string.IsNullOrWhiteSpace(directory))
        {
            throw new ModelFetchException($"Model '{model.Name}' path '{model.Path}' names no pack directory to read its manifest from.");
        }

        return directory;
    }

    // Which seated compute grants ride logical processors another grant already holds — the models an operator needs to see together when a
    // share was clamped or opted into overlap. Nothing shared reports null, so snapshots of a host where every grant is its own read exactly as they did before this fact existed.
    private static IReadOnlyList<string>? BuildSharedCoreFacts(ComputeCapacityLedger? compute, ComputeGrant? grant)
    {
        if (compute is null || grant is null || !grant.SharesHeldCores)
        {
            return null;
        }

        List<string> models = [.. compute.ModelsSharingHeldCores];
        if (!models.Contains(grant.Model))
        {
            models.Add(grant.Model);
        }

        return models;
    }

    // The resolved route as the monitor and the model listing report it: segments in execution order, each with its
    // device identity, residency tier, batch cap, and quota exactly as load resolved them. Live edits move worker
    // participation against a segment; they never rewrite which segments exist or what those carry.
    private static IReadOnlyList<ModelRouteSegment> BuildRouteSegmentFacts(ModelRoute route) =>
        [.. route.Segments.Select(segment => new ModelRouteSegment(
            segment.DeviceId,
            WeightTierNames.Of(segment.WeightTier),
            segment.BatchCap,
            segment.Quota.Threads,
            FormatCoreRanges(segment.Quota.Cores)))];

    // Additive ledger occupancy for operator views: one entry per tier position a seated claim holds, carrying the
    // opened seam's capacity posture, what the claims hold against it, and the contested holders only where several
    // claimants share a tier. A ledger that seated no claims reports nothing, so snapshots of an engine that claims
    // nothing — every model declaring no budgets today — read exactly as they did before the ledger seam existed.
    private static IReadOnlyList<LedgerOccupancyEntry>? BuildLedgerOccupancy(DeviceLedger? ledger)
    {
        if (ledger is null)
        {
            return null;
        }

        IReadOnlyList<DeviceLedgerClaim> claims = ledger.Claims;
        if (claims.Count == 0)
        {
            return null;
        }

        List<LedgerOccupancyEntry> entries = [];
        foreach (DeviceLedgerClaim claim in claims)
        {
            if (entries.Exists(entry => entry.Resource == BudgetResourceNames.Of(claim.Resource) && entry.Tier == BudgetTierNames.Of(claim.Tier)))
            {
                continue;
            }

            CapacitySeam? seam = ledger.SeamFor(claim.Tier, claim.Resource);
            IReadOnlyList<DeviceLedgerClaim> holders = ledger.ClaimsAt(claim.Tier, claim.Resource);
            bool contested = holders.Count > 1;

            entries.Add(new LedgerOccupancyEntry(
                BudgetResourceNames.Of(claim.Resource),
                BudgetTierNames.Of(claim.Tier),
                DeviceLedger.UnitName(claim.Resource),
                seam?.Capacity,
                holders.Sum(holder => holder.Amount),
                contested,
                contested ? [.. holders.Select(holder => new LedgerHolderEntry(holder.Model, holder.Amount))] : null));
        }

        return [.. entries];
    }

    // Pinned core ids as compressed ranges ("0-23", "1, 4-6"); null when nothing is pinned. Accepts any id set —
    // a placement's own pins or one lane side of a dedicated split reads through the same formatter so every core fact
    // on either surface renders alike.
    internal static string? FormatCoreRanges(IEnumerable<int>? cores)
    {
        if (cores is null)
        {
            return null;
        }

        int[] sorted = [.. cores];
        if (sorted.Length == 0)
        {
            return null;
        }
        Array.Sort(sorted);

        List<string> ranges = [];
        int runStart = sorted[0];
        int previous = sorted[0];
        for (int i = 1; i <= sorted.Length; i++)
        {
            if (i == sorted.Length || sorted[i] != previous + 1)
            {
                ranges.Add(runStart == previous ? runStart.ToString() : $"{runStart}-{previous}");
                if (i < sorted.Length)
                {
                    runStart = sorted[i];
                    previous = sorted[i];
                }
            }
            else
            {
                previous = sorted[i];
            }
        }

        return string.Join(", ", ranges);
    }

    // Current in-flight request views for a resync: volatile reads off each tracker, ordered by id so repeated
    // dumps of the same moment read alike.
    private IReadOnlyList<RequestPayload> InFlightRequestViews()
    {
        List<RequestPayload> views = [];
        foreach (KeyValuePair<string, RequestTracker> entry in _monitoredRequests)
        {
            views.Add(entry.Value.View());
        }

        views.Sort(static (a, b) => string.CompareOrdinal(a.RequestId, b.RequestId));
        return views;
    }

    // The scheduler owns its prefix cache privately; the service keeps what it passed so metrics can report cached blocks.
    private PrefixCache? _cacheForMetrics;

    /// <summary>Decodes one token id through the served model's tokenizer, special tokens included so control pieces round-trip.</summary>
    public string DecodeToken(int tokenId) => _tokenizer.Decode([tokenId], includeSpecial: true);

    /// <summary>The KV block budget admission currently spends from — the soft cap within <see cref="ReservedKvBlocks"/>.</summary>
    public int KvBudgetBlocks => _pool.EffectiveBudgetBlocks;

    /// <summary>Physical KV blocks reserved for this engine at open; never changes while it runs.</summary>
    public int ReservedKvBlocks => _pool.CapacityBlocks;

    /// <summary>Physical blocks currently handed out by the pool, including shared cache references and active sequences.</summary>
    public int KvUsedBlocks => _pool.UsedBlocks;

    /// <summary>Prefix-cache entries holding a block reference right now — what a shrinking budget drains through LRU eviction.</summary>
    public int CachedPrefixBlocks => _cacheForMetrics?.CachedBlocks ?? 0;

    /// <summary>The scheduler's live engine counters (prefill chunks and tokens, decode steps, occupancy) as the step loop maintains them.</summary>
    public SchedulerStats Stats => _scheduler.Stats;

    /// <summary>The configuration currently in force: candidates applied so far merged over the opened one, with refused restart-only fields still at their old values. Read-only snapshot for hosts that diff reloads against it.</summary>
    public LairConfig EffectiveConfiguration => Volatile.Read(ref _current);

    /// <summary>The most recent completed reload attempt as one structured summary, or null if none has finished. Hosts assert against this instead of parsing log lines.</summary>
    public ReloadSummary? LastReload => Volatile.Read(ref _lastReload);

    /// <summary>Records the outcome of one reload attempt (applied or rejected) for programmatic observation by hosts and tests.</summary>
    public void ReportReload(ReloadSummary summary)
    {
        ArgumentNullException.ThrowIfNull(summary);
        Volatile.Write(ref _lastReload, summary);
    }

    /// <summary>
    /// Entry point of the change-application pipeline: diffs a fully validated candidate configuration against what is
    /// currently in force and dispatches every difference by its mutation class. Live changes apply to the running
    /// engine (prefill width at the next step boundary, worker participation from the next kernel call, KV budget as one
    /// gated cap-flip-plus-drain); new-requests-only changes update admission for later requests while sequences already in
    /// flight run on unchanged limits; restart-only fields are refused naming field and class, their old values retained.
    /// The effective configuration merges exactly the applied fields — a candidate carrying both an applicable live change
    /// and a refused restart-only change ends up with the first in force and the second still at its old value. Returns one
    /// per-field result per difference; nothing here re-validates (candidates arrive validated) and no failure leaves a
    /// half-applied state: each field's apply is independent, and refusals are inert by construction.
    /// </summary>
    public IReadOnlyList<FieldResult> ApplyConfiguration(LairConfig candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        ObjectDisposedException.ThrowIf(_disposed, typeof(GenerationService));

        lock (_reloadGate)
        {
            List<FieldResult> results = [];
            LairConfig effective = _current;

            // The plan re-resolved when a new-requests-only tool key lands, so later requests follow the candidate's
            // value while sequences already in flight stay on their resolved limits.
            OutputParserPlan replanned;

            foreach (FieldChange change in ConfigMutationTable.Diff(effective, candidate))
            {
                switch (change.Field)
                {
                    case "engine.max_batch_tokens":
                        _scheduler.ApplyMaxBatchTokens(candidate.Engine.MaxBatchTokens);
                        effective = effective with { Engine = effective.Engine with { MaxBatchTokens = candidate.Engine.MaxBatchTokens } };
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            "prefill chunk width takes effect at the next step boundary; sequences mid-chunk finish that chunk at the old width"));
                        break;

                    case "engine.prefill_step_budget_ms":
                        _scheduler.ApplyPrefillStepBudget(candidate.Engine.PrefillStepBudgetMs);
                        effective = effective with { Engine = effective.Engine with { PrefillStepBudgetMs = candidate.Engine.PrefillStepBudgetMs } };
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            change.NewValue == "(unset)"
                                ? "prefill chunk width returns to the configured ceiling from the next chunk issue; a forward already running finishes at whatever width was chosen before this edit"
                                : $"future prefill chunks are sized from measured step cost toward {change.NewValue} ms within the configured ceiling, from the next chunk issue on; a forward already running finishes at whatever width was chosen before this edit"));
                        break;

                    case "models[0].placement.thread_quota":
                        if (_prefillsLane is { } openLane)
                        {
                            // A dedicated posture maps this edit onto the decode side alone: the load-resolved prefill share keeps its pool, mask, activation buffers, and collision posture for the model's whole lifetime, while the decode cap re-derives from the new quota within the engine's own thread choice. An edit that cannot leave at least one worker on each side is a lane-layout change this seam cannot rewrite live — refused naming field, model, and what to do instead, with the effective quota untouched.
                            int engineThreadsChoice = effective.Engine.Threads == 0 ? CpuBackend.AutoWorkerCount : effective.Engine.Threads;

                            if (!ModelPrefillLanes.TryMapLiveThreadQuota(openLane, engineThreadsChoice, candidate.Models[0].Placement.ThreadQuota, out int mappedDecodeShare))
                            {
                                results.Add(new FieldResult(change, ConfigApplyOutcome.Refused,
                                    $"{change.Field} refused for model '{effective.Models[0].Name}' while a dedicated prefill lane is open: the new quota leaves no decode-side worker once the load-resolved {openLane.PrefillShareThreads}-worker prefill share holds its side. Raise the quota to cover both lanes, set 'placement.prefill_lane' back to shared for a restart, or restart lair with a split this claim can carry."));
                                break;
                            }

                            int dedicatedBefore = _backend.EffectiveWorkerCount;
                            int dedicatedAfter = _backend.ApplyThreadQuota(mappedDecodeShare); // the mapped decode share caps this side within the engine's own thread choice — never re-splitting the lane itself
                            _scheduler.ApplySteppingThreadQuota(candidate.Models[0].Placement.ThreadQuota);

                            effective = effective with { Models = [effective.Models[0] with { Placement = effective.Models[0].Placement with { ThreadQuota = candidate.Models[0].Placement.ThreadQuota } }] };
                            results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                                $"dedicated lane posture: decode-side workers {dedicatedBefore} -> {dedicatedAfter}, mapped from the new quota against the fixed {openLane.PrefillShareThreads}-worker prefill share; participation adjusts from the next kernel call (the scheduler's next step), lowered workers idle; merged-step quota headroom follows the new value from the next composition on"));
                            break;
                        }

                        int workersBefore = _backend.EffectiveWorkerCount;
                        int workersAfter = _backend.ApplyThreadQuota(candidate.Models[0].Placement.ThreadQuota);

                        // The same live value re-resolves the stepping segment's quota headroom, so future merged steps bound against it
                        // instead of stale load-time numbers: lowered participation retargets which candidates may join a later step rather than
                        // preempting one already composed. Route identity and shape stay as load resolved them.
                        _scheduler.ApplySteppingThreadQuota(candidate.Models[0].Placement.ThreadQuota);

                        effective = effective with { Models = [effective.Models[0] with { Placement = effective.Models[0].Placement with { ThreadQuota = candidate.Models[0].Placement.ThreadQuota } }] };
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"effective workers {workersBefore} -> {workersAfter}; participation adjusts from the next kernel call (the scheduler's next step), lowered workers idle; merged-step quota headroom follows the new value from the next composition on"));
                        break;

                    case "models[0].kv_blocks":
                        GenerationScheduler.KvBudgetApply budget = _scheduler.ApplyKvBudget(candidate.Models[0].KvBlocks);
                        RecordAppliedKvClaim(budget);
                        effective = effective with { Models = [effective.Models[0] with { KvBlocks = candidate.Models[0].KvBlocks } ] };
                        RefreshAdvertisedModel(effective);
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"KV budget {change.OldValue} -> {budget.AppliedBlocks} blocks within the {_pool.CapacityBlocks}-block physical reservation; drained {budget.EvictedCacheEntries} cached prefix blocks above the new cap (active sequences untouched)"));
                        break;

                    case "models[0].max_context_tokens":
                        effective = effective with { Models = [effective.Models[0] with { MaxContextTokens = candidate.Models[0].MaxContextTokens } ] };
                        RefreshAdvertisedModel(effective);
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"admission now validates against a {_modelInfo.ContextLength}-token window; sequences already in flight are unaffected"));
                        break;

                    case "models[0].force_fenced_tools":
                        effective = effective with { Models = [effective.Models[0] with { ForceFencedTools = candidate.Models[0].ForceFencedTools }] };
                        replanned = ResolvePlanFor(effective);
                        Volatile.Write(ref _outputParser, replanned);
                        RefreshAdvertisedModel(effective);
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"chat requests admitted from now on {(candidate.Models[0].ForceFencedTools ? "inject LAIR's fenced tool block even for a tool-capable template" : "follow the loaded template's tool capabilities")}; sequences already in flight are unaffected"));
                        break;

                    case "models[0].tool_parser":
                        effective = effective with { Models = [effective.Models[0] with { ToolParser = candidate.Models[0].ToolParser }] };
                        replanned = ResolvePlanFor(effective);
                        Volatile.Write(ref _outputParser, replanned);
                        RefreshAdvertisedModel(effective);
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"chat requests admitted from now on resolve to the {ToolParserNames.Of(replanned.ToolParser)} tool parser; sequences already in flight are unaffected"));
                        break;

                    case "models[0].reasoning":
                        effective = effective with { Models = [effective.Models[0] with { Reasoning = candidate.Models[0].Reasoning }] };
                        replanned = ResolvePlanFor(effective);
                        Volatile.Write(ref _outputParser, replanned);
                        RefreshAdvertisedModel(effective);
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"chat requests admitted from now on carry recognized reasoning regions as {ReasoningPolicyNames.Of(replanned.Reasoning)}; sequences already in flight are unaffected"));
                        break;

                    case "models[0].max_reasoning_tokens":
                        if (candidate.Models[0].MaxReasoningTokens is >= 1 && _reasoningMarkers is null)
                        {
                            results.Add(new FieldResult(change, ConfigApplyOutcome.Refused, MarkerRefusalForReload(change)));
                            break;
                        }

                        effective = effective with { Models = [effective.Models[0] with { MaxReasoningTokens = candidate.Models[0].MaxReasoningTokens } ] };
                        replanned = ResolvePlanFor(effective);
                        Volatile.Write(ref _outputParser, replanned);
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"reasoning budget {change.OldValue} -> {change.NewValue}: the configuration in force carries it for requests admitted from now on; sequences already in flight are unaffected"));
                        break;

                    case "models[0].max_response_tokens":
                        effective = effective with { Models = [effective.Models[0] with { MaxResponseTokens = candidate.Models[0].MaxResponseTokens } ] };
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"response budget {change.OldValue} -> {change.NewValue}: the configuration in force carries it for requests admitted from now on; sequences already in flight are unaffected"));
                        break;

                    case "models[0].max_concurrent_requests":
                        _scheduler.ApplyMaxConcurrentRequests(candidate.Models[0].MaxConcurrentRequests);

                        // The advertised cap rides the same swap as every other live-advertised fact, so /v1/models always
                        // reports what submissions admitted now are bounded by.
                        effective = effective with { Models = [effective.Models[0] with { MaxConcurrentRequests = candidate.Models[0].MaxConcurrentRequests } ] };
                        Volatile.Write(ref _modelInfo, _modelInfo with { MaxConcurrentRequests = candidate.Models[0].MaxConcurrentRequests });
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"admission cap {change.OldValue} -> {change.NewValue}: submissions from now on are refused once every slot of the new cap is held; sequences already admitted keep their slots and finish normally"));
                        break;

                    case "models[0].placement.numa_spread":
                        // The preference is recorded for later placement decisions; the node this engine opened on, the posture its
                        // worker pool masks to, and the posture its weight pages first-touched to stay as the open resolved them.
                        // Re-binding an opened engine's placement is what a restart is for, and nothing here claims otherwise.
                        effective = effective with { Models = [effective.Models[0] with { Placement = effective.Models[0].Placement with { NumaSpread = candidate.Models[0].Placement.NumaSpread } }] };
                        _logger?.LogInformation(
                            "numa_placement_reload model='{Model}' opened_spread={OpenedSpread} preference={Preference} opened_posture=established",
                            Model.Id,
                            _placement is null ? "(unresolved)" : NumaSpreadNames.Of(_placement.Applied),
                            candidate.Models[0].Placement.NumaSpread);
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"spread posture {change.OldValue} -> {change.NewValue}: the preference is recorded for later placement decisions; the node this engine opened on and the posture its worker pool and weight pages established remain as the open resolved them — re-binding an opened engine's placement takes a restart"));
                        break;

                    case "ui.recent_requests":
                        _metrics?.SetRecentRequestCapacity(candidate.Ui.RecentRequests);
                        effective = effective with { Ui = candidate.Ui };
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"monitor recent-request ring now holds at most {candidate.Ui.RecentRequests} completed requests; entries beyond the new cap dropped oldest-first"));
                        break;

                    case "engine.prefill_kernel":
                        PrefillKernel kernel = ResolvePrefillKernel(effective.Models[0].Name, candidate.Engine.PrefillKernel);
                        try
                        {
                            _model.ApplyPrefillKernel(kernel);
                        }
                        catch (ModelException ex)
                        {
                            // A selection this build cannot run for the served model is refused naming it, what is
                            // missing, and which of that model's weight formats lacks a dense-matrix path — the engine
                            // keeps the route it opened with, and no request runs on a fallback selection.
                            results.Add(new FieldResult(change, ConfigApplyOutcome.Refused, $"{change.Field} refused for model '{effective.Models[0].Name}': {ex.Message}"));
                            break;
                        }

                        effective = effective with { Engine = effective.Engine with { PrefillKernel = candidate.Engine.PrefillKernel } };
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"prefill route selection {change.OldValue} -> {change.NewValue}: the engine re-reads it at every prefill window boundary; sequences already mid-window finish that window on the route they started with"));
                        break;

                    // A budgeted system-memory KV ceiling is the same live cap 'kv_blocks' has always been: it routes through this engine's one
                    // enforcement seam rather than adding a second, and what reaches the scheduler is the canonical block count every other KV edit uses.
                    case "models[0].budgets.kv.caps.dram_blocks":
                        if (!CarriesSystemMemoryKvTier(effective.Models[0]))
                        {
                            results.Add(new FieldResult(change, ConfigApplyOutcome.Refused,
                                $"{change.Field} refused for model '{effective.Models[0].Name}': the chain this ceiling binds was established when the engine opened, and nothing live holds working copies at 'dram' to retarget. Restart with 'dram' named in 'models[].budgets.kv.order', or edit 'kv_blocks' on a model that declares no budgets."));
                            break;
                        }

                        long ceiling = CandidateSystemMemoryKvCeiling(candidate.Models[0]) ?? candidate.Models[0].KvBlocks;
                        GenerationScheduler.KvBudgetApply budgeted = _scheduler.ApplyKvBudget(checked((int)ceiling));
                        RecordAppliedKvClaim(budgeted);
                        effective = effective with { Models = [MergeAppliedKvCeiling(effective.Models[0], ceiling)] };
                        RefreshAdvertisedModel(effective);
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"KV budget {change.OldValue} -> {budgeted.AppliedBlocks} blocks within the {_pool.CapacityBlocks}-block physical reservation; drained {budgeted.EvictedCacheEntries} cached prefix blocks above the new cap (active sequences untouched)"));
                        break;

                    case "models[0].budgets.weights.caps.dram_mb" or "models[0].budgets.weights.caps.vram_mb":
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Refused,
                            $"{change.Field} refused for model '{effective.Models[0].Name}' (restart-only): a weight ceiling binds what every forward reads, and retargeting it means re-reading slabs from the model file — that migration is not implemented in this build. Change it by restarting lair."));
                        break;

                    case "models[0].budgets.prefill_buffers.caps.dram_mb" or "models[0].budgets.prefill_buffers.caps.vram_mb":
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Refused,
                            $"{change.Field} refused for model '{effective.Models[0].Name}' (new-requests): a ceiling on prefill working buffers classifies for later requests only — the physical clamp is where this value lands, when the engine opens its row window, and a live edit cannot re-size the window or activation buffers already established. Nothing in force changed."));
                        break;

                    default:
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Refused, RefusalDetail(change)));
                        break;
                }
            }

            _current = effective;

            // Snapshots must speak the configuration in force: re-push model facts after every apply so additive
            // snapshot fields (worker threads, quota) track live-applied changes without waiting for a new tick's config.
            EngineMetricsBus? metricsForFacts = _metrics;
            if (metricsForFacts is not null)
            {
                metricsForFacts.AttachModelFacts(BuildMonitorFacts(effective));
            }

            return results;
        }
    }

    /// <summary>Records a live KV soft-cap application on the host ledger when this engine has a standing claim for that resource.</summary>
    private void RecordAppliedKvClaim(GenerationScheduler.KvBudgetApply budget)
    {
        if (_ledger?.UpdateStandingClaimAmount(Model.Id, BudgetResource.Kv, budget.AppliedBlocks, _reservationKvClaimTier) == true)
        {
            _logger?.LogInformation(
                "ledger_claim_update model='{Model}' resource='kv' amount_blocks={AmountBlocks}",
                Model.Id,
                budget.AppliedBlocks);
        }
    }

    // Advertised model facts recomputed from the effective configuration and live pool state: the window admission
    // validates against is the smallest of configured max context, what the forward engine can actually address
    // (its precomputed RoPE extent), and the token slots the current KV budget can back. A raise beyond the engine's
    // addressed extent clamps there — re-cutting those tables belongs to a model reload, not to a hot edit. The tool
    // parser rides on the same swap so /v1/models always reports what requests admitted now follow.
    private void RefreshAdvertisedModel(LairConfig effective)
    {
        long backed = Math.Min((long)_pool.EffectiveBudgetBlocks * _pool.BlockTokens, _model.MaxContextTokens);
        int context = (int)Math.Min(effective.Models[0].MaxContextTokens, backed);

        Volatile.Write(ref _modelInfo, _modelInfo with { ContextLength = context, KvBlocks = _pool.EffectiveBudgetBlocks, ToolParser = ToolParserNames.Of(_outputParser.ToolParser), Reasoning = ReasoningPolicyNames.Of(_outputParser.Reasoning) });
    }

    // The plan the configuration in force resolves to: same registry decision as model load, against the capabilities
    // probed once for this template. Only a new-requests-only reload calls it; generation never re-decides. A declared
    // reasoning ceiling travels through the same resolve carrying the marker pair proven from this vocabulary at open —
    // the region later requests may bound is that pair, never a markup guessed per request.
    private OutputParserPlan ResolvePlanFor(LairConfig effective) =>
        OutputParserRegistry.Resolve(effective.Models[0].ToolParser, effective.Models[0].ForceFencedTools, _chatTemplateCapabilities, effective.Models[0].Reasoning, effective.Models[0].MaxReasoningTokens is >= 1 ? _reasoningMarkers : null);

    // The kernel selection one configuration resolves to for the forward engine: only the three loader-accepted names
    // map, and anything else — reachable only on a programmatically built configuration — fails naming model and value
    // instead of defaulting onto some other route. Whether a mapped selection actually has a kernel is the engine's
    // answer at open and on apply, not this resolver's.
    private static PrefillKernel ResolvePrefillKernel(string modelName, string value) => value switch
    {
        PrefillKernelNames.GemV => PrefillKernel.GemV,
        PrefillKernelNames.Tiled => PrefillKernel.Tiled,
        PrefillKernelNames.Gemm => PrefillKernel.Gemm,
        _ => throw new InvalidOperationException(
            $"model '{modelName}' declares engine.prefill_kernel '{value}': 'prefill_kernel' accepts only '{PrefillKernelNames.GemV}', '{PrefillKernelNames.Tiled}', or '{PrefillKernelNames.Gemm}'."),
    };

    // The spread posture one configuration resolves to for placement: only the three accepted names map, and anything
    // else — reachable only on a programmatically built configuration — fails naming model and value instead of defaulting
    // onto some other posture. Which node the mapped posture became on this host's reported topology is placement's answer
    // at open, not this resolver's.
    private static NumaSpreadPreference ResolveNumaSpread(string modelName, string value) =>
        NumaSpreadNames.TryParse(value, out NumaSpreadPreference preference)
            ? preference
            : throw new InvalidOperationException(
                $"model '{modelName}' declares 'placement.numa_spread' '{value}': 'numa_spread' accepts {NumaSpreadNames.AcceptedVocabulary}.");

    // The core ids the backends this open builds pin to. A declared core set is authoritative and passes through untouched.
    // An unpinned posture bounds its worker pool to the single node placement chose when that node reports its core ids and
    // this host has a real affinity seam: the mask is the node's own reported ids, never a guess. On a host whose affinity
    // seam is the documented no-op, an unpinned model stays unpinned rather than carrying a mask that pins nothing; a posture
    // that spread across nodes names nodes rather than cores, and ids outside what a processor mask can express are not
    // invented into a partial pin.
    private static IReadOnlySet<int> EffectiveBackendCoreSet(HostTopology topology, NumaPlacement placement, IReadOnlySet<int> declaredCoreSet)
    {
        if (declaredCoreSet.Count > 0 || placement.NodeIndices.Count != 1 || !CpuBackend.SupportsCorePinning)
        {
            return declaredCoreSet;
        }

        HostNodeCapacity? node = topology.Nodes.FirstOrDefault(candidate => candidate.Index == placement.NodeIndices[0]);
        if (node is null || node.CoreIds.Count == 0 || node.CoreIds.Any(core => core < 0 || core > CpuBackend.MaxMaskableCoreId))
        {
            return declaredCoreSet; // a node this posture does not report, one that states no ids, or one whose ids a mask cannot express: nothing to derive
        }

        return new HashSet<int>(node.CoreIds);
    }

    // A byte count stated in the megabyte unit weight and prefill-buffer claims and ceilings count in, rounded up so
    // a claim never under-reports what it holds against a ceiling declared in that unit.
    private static long WeightClaimMegabytes(long bytes) => (bytes + 1024 * 1024 - 1) / (1024 * 1024);

    // The residency tiers this open's pools back working copies at: system memory always, plus accelerator memory when the
    // accelerator backend it opened reads weights from device copies it owns. A backend launched against registered host regions
    // opens no such tier — its weights never enter persistent device memory — and no placement of any other resource gains a
    // residency here.
    private static IReadOnlyList<BudgetTier> OpenedPoolTiersFor(CudaBackend? splitExecution) =>
        splitExecution is not null && splitExecution.WeightTier == Lair.Contracts.WeightTier.Vram
            ? [BudgetTier.Dram, BudgetTier.Vram]
            : [BudgetTier.Dram];

    // The ledger an engine opens when it carries no shared accounting: every tier its pools back for every budgetable resource, each
    // stating the explicit unbounded capacity posture — a public total no configuration declared is never invented. Accelerator memory
    // holding weight copies is the exception on host totals: opening that tier states the ceiling 'host.vram_mb' declares for it when
    // the configuration names one, and keeps the unbounded posture where it does not, because nothing here consults a driver-detected
    // device total. An accelerator route this engine opened also carries its prompt windows' activation bytes across to the device per
    // sub-batch, so that tier gets an unbounded seam for prefill buffers too — what bounds a claim there is the model's own declared
    // ceiling at that chain position, never an invented host total. No seam opens at accelerator memory for any resource whose backing
    // this open did not establish: block pools live in system memory until a device-backed residency is opened beside them.
    private static DeviceLedger OpenHostLedger(string modelName, HostCapacityConfig? host, CudaBackend? splitExecution, CudaBackend? kvStorageOwner, ILogger? logger)
    {
        BudgetResource[] resources = [BudgetResource.Weights, BudgetResource.Kv, BudgetResource.PrefillBuffers];
        List<CapacitySeam> seams = [];

        foreach (BudgetTier tier in OpenedPoolTiersFor(splitExecution))
        {
            foreach (BudgetResource resource in resources)
            {
                if (tier == BudgetTier.Vram && resource is not (BudgetResource.Weights or BudgetResource.PrefillBuffers))
                {
                    continue; // no device-backed pool for this resource is opened here, so nothing accounts against that position
                }

                seams.Add(new CapacitySeam { Tier = tier, Resource = resource, Capacity = null });
            }
        }

        if (kvStorageOwner is not null)
        {
            // A resolved device-backed residency opens an accelerator-tier seam for block pools specifically: what bounds those claims is each model's own declared finite KV ceiling at that position, because a public device total no configuration states is never invented into a block count — the byte size of one block belongs to a model, not to a host.
            seams.Add(new CapacitySeam { Tier = BudgetTier.Vram, Resource = BudgetResource.Kv, Capacity = null });
        }
        if (splitExecution is not null && splitExecution.WeightTier == Lair.Contracts.WeightTier.Vram)
        {
            long? declaredMb = host?.VramMb;
            seams.Add(new CapacitySeam { Tier = BudgetTier.Vram, Resource = BudgetResource.Weights, Capacity = declaredMb });

            logger?.LogInformation(
                "vram_capacity model='{Model}' device={Device} vram_mb={VramMb}",
                modelName,
                splitExecution.DeviceId,
                declaredMb is long mb ? mb.ToString(CultureInfo.InvariantCulture) : "(undeclared: unbounded capacity posture)");
        }

        return new DeviceLedger(seams);
    }

    // Declared capacity totals read against the posture this host reported. A disagreement is logged and accepted rather than refused: an
    // operator may state a number detection could not see, which is exactly what a per-node override exists for, while byte totals no node stated
    // read as incomparable rather than as a mismatch with what was declared. The posture itself is always reported so the comparison has a source.
    private static void ReportDeclaredCapacity(string modelName, HostCapacityConfig? host, ComputeCapacityLedger compute, ILogger? logger)
    {
        HostTopology topology = compute.Topology;

        logger?.LogInformation(
            "compute_topology model='{Model}' nodes={NodeCount} logical_processors={TotalLogicalProcessors}{Bytes}",
            modelName,
            topology.Nodes.Count,
            topology.TotalLogicalProcessors,
            topology.ReportedMemoryMb is long reportedMb ? $" node_mb_total={reportedMb}" : " node_bytes=unreported");

        if (host is null)
        {
            return; // nothing declared: today's posture states no public totals and compares against none.
        }

        DeclaredTotalsReport totals = compute.CheckDeclaredTotals(host.CpuCores, host.DramMb);
        if (totals.ReportsCoreMismatch || totals.ReportsByteMismatch)
        {
            logger?.LogWarning(
                "host_capacity model='{Model}' {Detail} — accepted against reported topology",
                modelName,
                totals.Describe());
        }

        foreach (NodeCapacityOverride nodeOverride in host.Nodes)
        {
            HostNodeCapacity? reported = topology.Nodes.FirstOrDefault(node => node.Index == nodeOverride.Index);
            if (reported is null)
            {
                logger?.LogWarning(
                    "host_capacity model='{Model}' declares node index {DeclaredIndex} this posture does not report ({Posture}) — accepted as an override of detection",
                    modelName,
                    nodeOverride.Index,
                    topology.Describe());
                continue;
            }

            if (nodeOverride.Cores is int declaredCores && declaredCores != reported.LogicalProcessors)
            {
                logger?.LogWarning(
                    "host_capacity model='{Model}' declares cores={DeclaredCores} for node index {NodeIndex} against reported {ReportedCores} — accepted as an override of detection",
                    modelName,
                    declaredCores,
                    nodeOverride.Index,
                    reported.LogicalProcessors);
            }

            // A posture that stated no per-node bytes has nothing to compare a declared override against; accepting silently is the honest reading there.
            if (nodeOverride.Mb is long declaredMb && reported.MemoryMb is long reportedNodeMb && declaredMb != reportedNodeMb)
            {
                logger?.LogWarning(
                    "host_capacity model='{Model}' declares megabytes={DeclaredMegabytes} for node index {NodeIndex} against reported {ReportedMegabytes} — accepted as an override of detection",
                    modelName,
                    declaredMb,
                    nodeOverride.Index,
                    reportedNodeMb);
            }
        }
    }

    // Whether this model's budgets declare a ceiling at the system-memory tier of their KV chain — the one budgeted ceiling that is
    // canonical for today's reservation and live soft cap, so its agreement with the legacy key is what opening reports once.
    private static bool DeclaresSystemMemoryKvCeiling(ModelConfig model)
    {
        if (model.Budgets is null)
        {
            return false;
        }

        foreach (BudgetChain chain in model.Budgets)
        {
            if (chain.Resource == BudgetResource.Kv && chain.Caps.ContainsKey(BudgetTier.Dram))
            {
                return true;
            }
        }

        return false;
    }

    // The finite ceiling this model's budgets declare for KV blocks held on a device: the block count declared at its chain's accelerator position, read straight from what was configured. A resolved device-backed residency is admitted only against one — no host total stands in for it, and nothing here guesses a bound a declaration never stated.
    private static long? DeclaredDeviceKvCeiling(ModelConfig model)
    {
        if (model.Budgets is null)
        {
            return null;
        }

        foreach (BudgetChain chain in model.Budgets)
        {
            if (chain.Resource == BudgetResource.Kv && chain.Caps.TryGetValue(BudgetTier.Vram, out long ceiling))
            {
                return ceiling;
            }
        }

        return null;
    }

    // Whether this configuration's KV chain names the system-memory tier at all: only such a chain has live working copies for a budgeted
    // ceiling to retarget, and one that does not was established as something else when the engine opened.
    private static bool CarriesSystemMemoryKvTier(ModelConfig model) => DeclaresChainAtTier(model, BudgetResource.Kv, BudgetTier.Dram);

    private static long? CandidateSystemMemoryKvCeiling(ModelConfig model)
    {
        if (model.Budgets is null)
        {
            return null;
        }

        foreach (BudgetChain chain in model.Budgets)
        {
            if (chain.Resource == BudgetResource.Kv && chain.Caps.TryGetValue(BudgetTier.Dram, out long ceiling))
            {
                return ceiling;
            }
        }

        return null;
    }

    // Merges one applied system-memory KV ceiling into the configuration in force: this model's other declared chains and every other tier of
    // its own chain stay exactly as load resolved them, so a refused sibling edit in the same candidate cannot ride along on this merge.
    private static ModelConfig MergeAppliedKvCeiling(ModelConfig current, long newDramBlocks)
    {
        List<BudgetChain> merged = [];

        foreach (BudgetChain chain in current.Budgets ?? [])
        {
            if (chain.Resource == BudgetResource.Kv && chain.Order.Contains(BudgetTier.Dram))
            {
                Dictionary<BudgetTier, long> caps = [];
                foreach (KeyValuePair<BudgetTier, long> cap in chain.Caps)
                {
                    caps[cap.Key] = cap.Value;
                }

                caps[BudgetTier.Dram] = newDramBlocks;
                merged.Add(chain with { Caps = caps });
                continue;
            }

            merged.Add(chain);
        }

        return current with { KvBlocks = checked((int)newDramBlocks), Budgets = [.. merged] };
    }

    private static bool DeclaresChainAtTier(ModelConfig model, BudgetResource resource, BudgetTier tier)
    {
        if (model.Budgets is null)
        {
            return false;
        }

        foreach (BudgetChain chain in model.Budgets)
        {
            if (chain.Resource == resource && chain.Order.Contains(tier))
            {
                return true;
            }
        }

        return false;
    }

    private string MarkerRefusalForReload(FieldChange change) =>
        $"{change.Field} refused: {_markerRefusalDetail}. The loaded model's vocabulary cannot carry the reasoning markers a 'max_reasoning_tokens' budget would bound; remove the key from this model entry or serve a model whose vocabulary carries them.";

    private static string RefusalDetail(FieldChange change) => change.Field switch
    {
        "models[0].path" => $"{change.Field} requires restart (restart-only): '{change.OldValue}' -> '{change.NewValue}'. Server-side model reload is not implemented in this build — stop and restart lair to serve a different model file.",
        "engine.block_size" => $"{change.Field} requires restart (restart-only): {change.OldValue} -> {change.NewValue}. KV block geometry is fixed for the lifetime of a running engine; change it by restarting lair.",
        "server.bind" or "server.port" => $"{change.Field} requires restart (restart-only): {change.OldValue} -> {change.NewValue}. The HTTP listener binds once at startup.",
        _ => $"{change.Field} requires restart (restart-only): {change.OldValue} -> {change.NewValue}. This field is established when the engine opens; change it by restarting lair.",
    };

    /// <summary>
    /// Opens the first model declared in the configuration: GGUF container, mmap-backed weights through the
    /// forward engine, tokenizer, chat template, a KV pool sized by the model entry, a prefix cache over that pool,
    /// and the batch scheduler over them. The prefix cache lives exactly as long as the pool, so each opened
    /// service starts with an empty cache. The effective context is the smaller of the configured max context and
    /// the token slots the KV pool can hold. Weight, KV, and prefill claims this model declares register on the
    /// ledger a host opens over the tiers its pools back — each opened tier stating the explicit unbounded capacity
    /// posture, since a public total no configuration declared is never invented.
    /// </summary>
    public static GenerationService Open(LairConfig config, ILogger? logger = null) =>
        Open(config, logger, ledger: null);

    /// <summary>
    /// The shared-ledger form of the same open: <paramref name="ledger"/> is the host-scoped accounting several
    /// programmatically opened engines share, so their claims register against one set of capacity seams in the order
    /// the opens happen — first-come-first-served across engines, earlier claims never revoked or displaced. A null
    /// ledger is today's posture: this engine gets its own ledger over the tiers its pools back, bounded nowhere a
    /// public total was not stated. No host configuration path reaches a supplied ledger — it is a programmatic seam.
    /// </summary>
    public static GenerationService Open(LairConfig config, ILogger? logger, DeviceLedger? ledger) =>
        Open(config, logger, ledger, compute: null);

    /// <summary>
    /// The shared-accounting form of the same open. <paramref name="ledger"/> is host-scoped byte accounting several programmatically opened
    /// engines share, so their claims register against one set of capacity seams in the order the opens happen — first-come-first-served across
    /// engines, earlier claims never revoked or displaced; a null ledger is today's posture, this engine's own ledger over the tiers its pools
    /// back with each stating the explicit unbounded capacity posture. <paramref name="compute"/> is host-scoped core accounting shared the same
    /// way: an injected one carries both what the host reported about nodes and every share seated ahead of this model, so a later claim is
    /// clamped to what its predecessors left instead of being refused or preempting them; a null one opens fresh accounting over this process's
    /// reported topology. No host configuration path reaches either seam — they are programmatic seams. <paramref name="cpuFeatures"/> is a third programmatic CPU-posture seam: production opens leave it null and detect the process posture, while tests may state one explicitly so an unavailable forced ISA can be proven to refuse before any backend exists.
    /// </summary>
    public static GenerationService Open(LairConfig config, ILogger? logger, DeviceLedger? ledger, ComputeCapacityLedger? compute, IModelHubTransport? transport = null, CpuFeaturePosture? cpuFeatures = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Models.Count == 0)
        {
            throw new InvalidOperationException(
                "the configuration declares no models: add at least one entry under 'models:' before opening the engine.");
        }

        ModelConfig modelConfig = config.Models[0];
        int blockTokens = config.Engine.BlockSize;

        // The KV pool holds KvBlocks * blockTokens token slots, reserved physically at open. Admission validates against the
        // smaller of the configured max context and those slots; the forward engine's addressable extent covers the whole
        // reservation so a later hot raise of max_context_tokens stays applicable up to what the pool can back — raising it
        // beyond the reservation would need new RoPE tables, i.e. a model reload.
        int reservedSlots = checked(modelConfig.KvBlocks * blockTokens);
        int maxContext = Math.Min(modelConfig.MaxContextTokens, reservedSlots);

        // The model's placement decides the backend's shape: the engine thread choice (0 = auto-sized from physical
        // cores) capped by the model's thread quota, and the model's core set pinned onto every pool worker. A declared
        // dedicated prefill lane resolves its split first — a refusal naming the model and key lands here before any
        // backend exists, so no half-lane engine ever opens — and then each side gets its own cap: the decode-side
        // backend runs on the full engine choice with this step's derived share as its starting quota (a live edit may
        // retarget that cap within the same ceiling), while the lane backend is sized to its load-resolved share alone.
        // A pinned posture hands each side a disjoint mask of the declared cores; an unpinned one bounds participation by worker count,
        // bounded to the node placement chose where this host can pin.
        PrefillLaneLayout? prefillsLane = null;
        CpuBackend? prefillsBackend = null;
        CpuBackend backend;

        // The host posture this engine's worker claim accounts against: an injected compute ledger carries both what the host reported about
        // its nodes and every share seated ahead of this model, so a later oversubscribed claim is clamped to what its predecessors left rather
        // than refused or allowed to preempt them; without one, this open derives the process posture once — nodes and logical processors as the
        // OS reports them, byte totals reported where stated — and seats claims against that host total. Which node a share draws from belongs to placement, not accounting.
        compute ??= new ComputeCapacityLedger(HostTopologyProbe.Detect());

        ReportDeclaredCapacity(modelConfig.Name, config.Host, compute, logger); // warn-and-accept: an operator's declared totals read against what this posture reported.

        int engineThreadsChoice = config.Engine.Threads == 0 ? CpuBackend.AutoWorkerCount : config.Engine.Threads;
        int requestedShare = modelConfig.Placement.ThreadQuota is { } declaredQuota
            ? Math.Min(engineThreadsChoice, declaredQuota)
            : engineThreadsChoice;

        // Placement derives before any backend opens: this model's declared spread posture resolves against the posture the host
        // reported and the per-node load earlier claims seated, so the node its workers run on is decided while no worker pool exists.
        // A posture the reported topology cannot honour — a pinned set beside a spread posture, or a pinned set spanning nodes — refuses
        // the open naming model, key, and conflict, exactly like a half-lane posture does: no half-placed engine ever opens.
        NumaPlacement placement = NumaPlacementResolver.Resolve(
            compute.Topology,
            [new NumaClaim
            {
                Model = modelConfig.Name,
                RequestedCores = requestedShare,
                Spread = ResolveNumaSpread(modelConfig.Name, modelConfig.Placement.NumaSpread),
                CoreSet = modelConfig.Placement.CoreSet,
            }],
            compute.NodeAllocations)[0];

        logger?.LogInformation(
            "numa_placement model='{Model}' {NodeFact} spread={Requested}→{Applied}{CoreIds}{Advisory}",
            modelConfig.Name,
            placement.NodeIndices.Count == 1 ? $"node={placement.NodeIndices[0]}" : $"nodes={string.Join(",", placement.NodeIndices)}",
            NumaSpreadNames.Of(placement.Requested),
            NumaSpreadNames.Of(placement.Applied),
            placement.CoreIds.Count > 0 ? $" core_set='{FormatCoreRanges(placement.CoreIds)}'" : string.Empty,
            placement.Advisory is null ? string.Empty : $" advisory='{placement.Advisory}'");

        // The claim registers before any backend opens, in this open's order, on the nodes placement chose: earlier seats cap this
        // participation instead of refusing it, the grant logs requested against granted beside the per-node share it seated, and a
        // partial seat is visible at load rather than inferred from a worker count.
        ComputeGrant computeGrant = compute.RegisterClaim(modelConfig.Name, requestedShare, placement, modelConfig.Placement.CoreOvercommit);

        logger?.LogInformation(
            "compute_grant model='{Model}' requested={Requested} granted={Granted} exclusive_cores={ExclusiveCores} shared_cores={SharedCores} node_allocations={NodeAllocations} nodes={NodeCount} logical_processors={TotalLogicalProcessors}",
            modelConfig.Name,
            computeGrant.RequestedCores,
            computeGrant.GrantedCores,
            computeGrant.ExclusiveCores,
            computeGrant.SharesHeldCores,
            computeGrant.NodeAllocationText(),
            compute.Topology.Nodes.Count,
            compute.TotalCores);

        // A full grant leaves today's construction untouched: the engine thread choice and this model's quota stand exactly as configured. Only a partial grant adds a cap, it caps participation inside what this model itself asked to run with, and how auto-sizing chose its count stays as it was — live worker edits keep their existing semantics against the engine's own choice.
        int? grantedQuota = computeGrant.GrantedCores >= requestedShare
            ? modelConfig.Placement.ThreadQuota
            : Math.Min(modelConfig.Placement.ThreadQuota ?? computeGrant.GrantedCores, computeGrant.GrantedCores);

        // The core ids every backend this open builds pins to: a declared core set is authoritative and stands exactly as configured.
        // An unpinned posture bounds its worker pool to the node placement chose when that node reports core ids and this host's
        // affinity seam is real — a node-derived mask, never a guess. Where the seam is a documented no-op or a posture spread across
        // nodes, nothing is masked and the posture bounds participation by worker count alone, as it always did.
        IReadOnlySet<int> backendCoreSet = EffectiveBackendCoreSet(compute.Topology, placement, modelConfig.Placement.CoreSet);

        // The CPU integer-dot ISA axis resolves exactly once for this open, before any backend exists: every kernel path this
        // model runs — decode side, a dedicated prefill lane's side, threaded or not — uses one encoding for the engine's whole
        // lifetime. A forced selection this host cannot run refuses here naming the model, the key as declared, and what the CPU
        // does not report; nothing answers it with some other path that happens to be available.
        if (!CpuIsaNames.TryParse(config.Engine.Cpu.Isa, out CpuIsa requestedIsa))
        {
            throw new ModelException(
                $"model '{modelConfig.Name}' declares 'engine.cpu.isa' '{config.Engine.Cpu.Isa}': 'cpu.isa' accepts {CpuIsaNames.AcceptedVocabulary}.");
        }

        CpuFeaturePosture resolvedFeatures = cpuFeatures ?? CpuIsaProbe.Detect();

        if (!CpuIsaProbe.TryResolve(requestedIsa, resolvedFeatures, out _, out string? missingFeature))
        {
            throw new ModelException(
                $"model '{modelConfig.Name}' declares 'engine.cpu.isa' '{config.Engine.Cpu.Isa}': this host's CPU does not report " +
                $"{missingFeature}, and a forced ISA is never answered with some other integer-dot path that happens to be available — declare '{CpuIsaNames.Auto}' to select the highest path this host runs, or run on hardware reporting {missingFeature}.");
        }

        if (modelConfig.Placement.PrefillLane == PlacementPrefillLaneValues.Dedicated)
        {
            // A dedicated posture sub-allocates both lane sides from this model's granted share, so a clamped claim narrows the split instead of letting one side reach past what the host seated for it. A node-derived core set splits inside the chosen node: each lane claims its side of that node's reported ids.
            prefillsLane = ModelPrefillLanes.ResolveDedicated(modelConfig.Name, engineThreadsChoice, grantedQuota, backendCoreSet);

            backend = new(engineThreadsChoice, new HashSet<int>(prefillsLane.DecodeCoreIds), prefillsLane.DecodeShareThreads, requestedIsa: requestedIsa, cpuFeatures: resolvedFeatures);
            prefillsBackend = new(prefillsLane.PrefillShareThreads, new HashSet<int>(prefillsLane.PrefillCoreIds), requestedIsa: requestedIsa, cpuFeatures: resolvedFeatures);
        }
        else
        {
            backend = new(config.Engine.Threads, backendCoreSet, grantedQuota, requestedIsa: requestedIsa, cpuFeatures: resolvedFeatures);
        }

        // The placement's layer split opens the accelerator compute backend beside this service's decode-side CPU backend — and only when a model declares one: an absent or zero key never touches the CUDA seam at all. A declared split with no usable driver or device behind it fails here naming the model, the key as declared, and where accelerator-backed segments run; nothing falls back to some other posture that happens to work.
        CudaBackend? splitExecution = null;

        if (modelConfig.Placement.GpuLayers > 0 && modelConfig.Placement.PrefillLane == PlacementPrefillLaneValues.Dedicated)
        {
            throw new ModelException(
                $"model '{modelConfig.Name}' declares 'placement.gpu_layers' {modelConfig.Placement.GpuLayers} beside " +
                $"'placement.prefill_lane' '{PlacementPrefillLaneValues.Dedicated}': a dedicated prefill lane carries one stage-side execution backend, and this build has no crossing across a split route — resolve placement onto one segment or open no lane.");
        }

        if (modelConfig.Placement.GpuLayers > 0)
        {
            try
            {
                splitExecution = new();
            }
            catch (Exception ex) when (ex is CudaException or DllNotFoundException or EntryPointNotFoundException or TypeInitializationException)
            {
                throw new InvalidOperationException(
                    $"model '{modelConfig.Name}' declares 'placement.gpu_layers' {modelConfig.Placement.GpuLayers}: the split opens a compute backend on device '{CudaBackend.DeviceIdValue}', accelerator-backed segments run in '{ModelRoute.CudaBackendSeam}', and this host opened no usable CUDA driver or device behind that seam — refusing rather than running this model's layers somewhere else.",
                    ex);
            }

            logger?.LogInformation(
                "gpu_layers model='{Model}' cuda_layers={GpuLayers} device={Device}",
                modelConfig.Name,
                modelConfig.Placement.GpuLayers,
                CudaBackend.DeviceIdValue);
        }

        // KV residency is its own placement axis beside the layer split, resolved here onto the posture this build's pools can actually run: an absent or posture-neutral declaration lands on host-resident blocks exactly as every model served today does, and an accelerator-resident or divided one needs both a finite ceiling declared for those blocks and a backend that owns persistent device storage to keep them in. The ceiling is checked before any seam opens — a device-backed reservation with no finite ceiling refuses naming the model, its resource, the tier, and what it failed to declare rather than opening a pool bounded by nothing.
        if (!BlockResidencyModeNames.TryResolvePlacementValue(modelConfig.Placement.KvResidency, out BlockResidencyMode kvResidency))
        {
            throw new ModelException(
                $"model '{modelConfig.Name}' declares 'placement.kv_residency' '{modelConfig.Placement.KvResidency}': " +
                $"'kv_residency' accepts {BlockResidencyModeNames.AcceptedVocabulary}.");
        }

        long? declaredKvDeviceCeiling = DeclaredDeviceKvCeiling(modelConfig);

        if (kvResidency != BlockResidencyMode.Dram && declaredKvDeviceCeiling is null)
        {
            throw new ModelException(
                $"model '{modelConfig.Name}' declares 'placement.kv_residency' '{BlockResidencyModeNames.Of(kvResidency)}': device-backed KV blocks are admitted only against a finite ceiling, and this model's budgets declare none for resource '{BudgetResourceNames.Kv}' at tier '{BudgetTierNames.Vram}'. " +
                $"Declare 'models[].budgets.kv.caps.vram_blocks' with the block count this host may keep resident on the device — or resolve the residency to '{BlockResidencyModeNames.Dram}', which needs no ceiling because it opens no device pool.");
        }

        // The persistent range a resolved reservation declares for itself: an accelerator-resident one covers every id its pool can hand out, and a divided one defaults to half of them in allocation order. A declared finite ceiling narrows that default; nothing widens past the midpoint or past what the pool reserves. Whether what remains fits is decided against the ledger below, so an over-capacity load refuses with the same breakdown every other exhausted chain reports — before any device-resident byte exists.
        int deviceBackedBlocks = kvResidency switch
        {
            BlockResidencyMode.Vram => modelConfig.KvBlocks,
            BlockResidencyMode.Split => (int)Math.Min(Math.Max(1L, modelConfig.KvBlocks / 2), Math.Max(1L, declaredKvDeviceCeiling!.Value)),
            _ => 0,
        };

        // A device-backed reservation needs a backend that owns persistent block copies. When the placement already opened an accelerator compute backend for a layer split, that instance owns them beside its own weight residency; when every forward stays on this service's CPU backend, this second seam exists only to hold those copies — it maps no weight tensor into device memory, runs no kernel, and answers no scheduler or monitor fact, while the CPU backend keeps owning the arena attention reads. No usable driver or device behind that seam fails the open naming what is missing rather than seating blocks somewhere the declaration never asked for.
        CudaBackend? kvMirrorExecution = null;

        if (deviceBackedBlocks > 0 && splitExecution is null)
        {
            try
            {
                kvMirrorExecution = new(CudaWeightResidency.HostMapped); // no weight copy ever reaches this instance: its only resident material is the block range a mirror declares.
            }
            catch (Exception ex) when (ex is CudaException or DllNotFoundException or EntryPointNotFoundException or TypeInitializationException)
            {
                throw new InvalidOperationException(
                    $"model '{modelConfig.Name}' declares 'placement.kv_residency' '{BlockResidencyModeNames.Of(kvResidency)}': device-backed blocks keep their persistent copy in '{ModelRoute.CudaBackendSeam}', and this host opened no usable CUDA driver or device behind that seam — refusing rather than serving this model against host-resident pools.",
                    ex);
            }

            logger?.LogInformation(
                "kv_residency model='{Model}' residency={Residency} declared_ceiling_blocks={DeclaredCeilingBlocks} device_backed_blocks={DeviceBackedBlocks} device={Device}",
                modelConfig.Name,
                BlockResidencyModeNames.Of(kvResidency),
                declaredKvDeviceCeiling!.Value,
                deviceBackedBlocks,
                CudaBackend.DeviceIdValue);
        }

        // Whichever accelerator backend this open owns becomes the storage owner of a device-backed pool's persistent copies: the layer split's compute backend when placement opened one, and otherwise the seam opened only to hold those blocks. Null whenever every block this engine hands out stays host-resident.
        CudaBackend? kvStorageOwner = kvMirrorExecution ?? (deviceBackedBlocks > 0 ? splitExecution : null);

        // The model's placement resolves to its route value set here, against the compute backends this service opens for it:
        // consecutive stages naming 'cpu' collapse into one CPU segment carrying this model's thread cap and pinned core share,
        // a declared split lands its first span on device 'cuda' from layer 0 with every later layer — through the last one, where
        // the output head runs — back on cpu at boundary gpu_layers, and a stage naming any other backend fails now naming that
        // segment and what is missing rather than opening an engine that would run the model somewhere else. The resolved segments
        // land on the plan at open unchanged — nothing re-derives them while the model stays open. Each segment's weight tier is
        // what the backend opened for its position reports: a default-opened accelerator backend reads the device copies it owns,
        // so its span resolves at the accelerator tier, while every host-resident posture — CPU kernels and an accelerator backend
        // launched against registered host regions — resolves at system memory.
        ModelRoute route;
        ResolvedBudgetSet budgets;

        try
        {
            List<RouteStage> stages = splitExecution is null
                ? [new(ModelRoutes.PrefillStage, modelConfig.Placement.Prefill), new(ModelRoutes.DecodeStage, modelConfig.Placement.Device)]
                :
                [
                    new RouteStage(ModelRoutes.PrefillStage, CudaBackend.DeviceIdValue) { SplitPosition = 0 },
                    new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId) { SplitPosition = modelConfig.Placement.GpuLayers },
                ];

            Dictionary<string, Lair.Contracts.WeightTier> deviceWeightTiers = new() { [backend.DeviceId] = backend.WeightTier };
            if (splitExecution is not null)
            {
                deviceWeightTiers[splitExecution.DeviceId] = splitExecution.WeightTier;
            }

            route = ModelRoutes.Resolve(new RouteRequest
            {
                ModelName = modelConfig.Name,
                Stages = stages,
                AvailableDevices = splitExecution is null ? [backend.DeviceId] : [backend.DeviceId, splitExecution.DeviceId],
                DeviceWeightTiers = deviceWeightTiers,

                // The KV axis rides the route exactly as resolution expects it: what placement declared became a mode above, and whether an opened backend can hold persistent block copies is this open's answer — never the compute segment's weight tier. A storage-only accelerator seam is deliberately absent from AvailableDevices because no stage runs on it; only where its blocks live follows it.
                KvResidency = kvResidency,
                DeviceBackedKvMirrorSupport = kvStorageOwner is not null,

                ThreadQuota = modelConfig.Placement.ThreadQuota,
                CoreSet = modelConfig.Placement.CoreSet,
            });

            // The model's declared budgets resolve against the pools this service opens for it, here and once: each chain lands as what an
            // opened host can actually carry, and a claim nothing backs fails now naming the model, the resource, the tier, and the missing
            // capability — before any weight view or reservation is spent on posture that cannot hold it. What every backend this build opens
            // holds working copies in is system memory; an accelerator backend opened with device-resident weight copies opens that tier for
            // weight claims too, a resolved device-backed KV residency opens the accelerator tier for block pools specifically while compute and attention stay host-side, and disk-backed write-side pools remain vocabulary those chains may name but nothing here carries yet.
            budgets = ModelBudgets.Resolve(new BudgetRequest
            {
                ModelName = modelConfig.Name,
                Declared = modelConfig.Budgets ?? [],
                OpenedPoolTiers = OpenedPoolTiersFor(splitExecution),
                OpenedKvPoolTiers = deviceBackedBlocks > 0 ? [BudgetTier.Vram] : [],
            });
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
        {
            splitExecution?.Dispose(); // a route or budget this configuration cannot resolve must not leave an opened accelerator backend standing.
            kvMirrorExecution?.Dispose(); // nor one opened only to hold device-backed blocks.
            throw;
        }

        // The ledger this engine's claims account against: a supplied one is a host's shared accounting across several
        // programmatically opened engines, so their claims register against one set of capacity seams in the order the
        // opens happen; an omitted one is this engine's own ledger over the tiers its pools back, each stating the
        // explicit unbounded capacity posture — the answer a host gives when it backs a tier but states no public total.
        // Claims register in this open's order: weights once the tensor table is read, prefill from the activation
        // posture the engine families hold, a device-backed block range as its own standing claim before any pool opens, and the KV reservation as the pool opens.
        ledger ??= OpenHostLedger(modelConfig.Name, config.Host, splitExecution, kvStorageOwner, logger);

        GgufFile? file = null;
        TransformerModel? model = null;
        BlockPool? pool = null;
        GenerationScheduler? scheduler = null;
        string? updateCandidate = null;
        BudgetTier? reservationKvClaimTier = null; // where the ledger seated this engine's whole-reservation KV claim, so a live soft-cap edit retargets that seat and only that seat.
        try
        {
            // A model that declares a published source is verified against its pack manifest before any byte of the
            // GGUF is opened: a present artifact verifies with no transport call, an absent pinned file fetches exactly
            // the pinned revision, and a disagreement or a failed fetch refuses the open here — before a model starts
            // serving — rather than after. A model with no source, and every host that supplies no transport, keeps
            // today's local-path posture untouched.
            if (transport is not null && modelConfig.Source is not null)
            {
                new ModelArtifactResolver(transport).ResolveAsync(modelConfig, CancellationToken.None).GetAwaiter().GetResult();

                // Report-only candidate reporting rides the same seam the load gate used: a model with a published
                // source gets one HEAD/tree consult against its pinned pack. A candidate-free or failed check omits
                // the monitor fact — a report never blocks the open, re-pins, or touches the load path.
                updateCandidate = ResolveUpdateCandidate(modelConfig, transport, logger);
            }

            file = GgufFile.Open(modelConfig.Path);
            // Profile-scoped tokenizer defaults applied at load (absent BOS/EOS keys) surface as structured
            // log lines on the host logger.
            GgufTokenizer tokenizer = GgufTokenizer.FromGguf(file, line => logger?.LogInformation("{TokenizerLog}", line));

            // The reasoning markers a budget would bound are proven against this vocabulary once here: each must arrive
            // whole as one token. A model declaring max_reasoning_tokens on a vocabulary that cannot carry them is refused
            // now — before weights are mapped — instead of serving with a ceiling that could bind nothing.
            bool markersProven = OutputParserRegistry.TryResolveMarkerPair(tokenizer, out ReasoningRegionMarkers? markerPair, out string markerRefusal);
            if (!markersProven && modelConfig.MaxReasoningTokens is >= 1)
            {
                logger?.LogError(
                    "max_reasoning_tokens model='{Model}' cannot bind a reasoning region: {Detail}",
                    modelConfig.Name,
                    markerRefusal);

                throw new InvalidOperationException(
                    $"model '{modelConfig.Name}' declares 'max_reasoning_tokens' but its tokenizer cannot carry the reasoning markers that budget would bound: {markerRefusal}. Remove 'max_reasoning_tokens' from this model entry or serve a model whose vocabulary carries them.");
            }

            // Profile resolution events (the generic llama-shaped fallback applied to an unknown architecture)
            // surface as structured log lines the same way. The configuration's prefill kernel selection reaches the
            // forward engine here: a selection this build cannot run for this file is refused naming the served model,
            // the value, and which of that file's own weight formats has no dense-matrix path — the posture every
            // unimplemented capability takes — and nothing lands on a route that was not selected.
            try
            {
                model = splitExecution is null
                    ? TransformerModel.Open(
                        file,
                        backend,
                        reservedSlots,
                        line => logger?.LogInformation("{ProfileLog}", line),
                        ResolvePrefillKernel(modelConfig.Name, config.Engine.PrefillKernel),
                        route,
                        prefillsBackend,
                        prefillsLane,
                        budgets)
                    : TransformerModel.Open(
                        file,
                        backend,
                        reservedSlots,
                        line => logger?.LogInformation("{ProfileLog}", line),
                        ResolvePrefillKernel(modelConfig.Name, config.Engine.PrefillKernel),
                        route,
                        [splitExecution, backend], // the resolved execution set answers every segment position in order: 'cuda' first, this service's decode-side CPU backend after it.
                        budgets);
            }
            catch (ModelException ex) when (ex.Message.StartsWith("prefill kernel '", StringComparison.Ordinal))
            {
                // The engine names the selection and the weight formats it has no dense-matrix path for; this service
                // adds which served model the refusal is about, so a load failure says what to change in the config.
                throw new ModelException($"model '{modelConfig.Name}': {ex.Message}");
            }

            // Weight claims register at model open, once the tensor table has been read: the amount is the summed
            // logical tensor bytes the forward reads, stated in this resource's megabyte unit, against the first
            // copy-holding position the resolved weights chain names that an opened capacity seam backs. A claim that
            // lands below the chain's first copy-holding position records the chain's own demotion — an earlier claim
            // is never revoked, preempted, or displaced to seat it — and a chain whose every position is held reports
            // its exhaustion here, as a load failure naming the model, the resource, the tier, and the ledger breakdown.
            if (budgets.For(BudgetResource.Weights) is { } weightsChain)
            {
                long logicalWeightBytes = 0;
                foreach (LogicalTensor tensor in file.Tensors.Tensors)
                {
                    logicalWeightBytes += tensor.SizeBytes;
                }

                ClaimOutcome weightClaim = ledger.RegisterClaim(modelConfig.Name, weightsChain, WeightClaimMegabytes(logicalWeightBytes));
                if (!weightClaim.Seated)
                {
                    throw new BudgetExhaustedException(modelConfig.Name, BudgetResource.Weights, weightClaim.ExhaustedAtTier!.Value, weightClaim.Breakdown);
                }

                logger?.LogInformation(
                    "ledger_claim model='{Model}' resource='weights' tier='{Tier}' amount_mb={AmountMb} chain_position={ChainPosition} demoted={Demoted} logical_bytes={LogicalBytes}",
                    modelConfig.Name,
                    BudgetTierNames.Of(weightClaim.Claim!.Tier),
                    weightClaim.Claim.Amount,
                    weightClaim.Claim.ChainPosition,
                    weightClaim.Claim.Demoted,
                    logicalWeightBytes);
            }

            // The prefill claim records the activation posture the opened engine actually holds for its prompt windows — the dedicated lane
            // family's bytes where a lane opened, the effective row window's per-row cost beside its fixed tail where one backend serves both
            // paths — beside the row width the plan resolved and the width the engine runs under a declared ceiling. The engine's physical clamp
            // already narrowed that window to what the chain's own ceilings allow; the claim states what the narrowed posture holds, and a chain
            // whose every position is held — including one too small for even a single row's bytes — reports its exhaustion here as a load failure.
            if (budgets.For(BudgetResource.PrefillBuffers) is { } prefillChain)
            {
                long activationBytes = model.PrefillWorkingSetBytes;
                ClaimOutcome prefillClaim = ledger.RegisterClaim(modelConfig.Name, prefillChain, model.PrefillWorkingSetMegabytes);
                if (!prefillClaim.Seated)
                {
                    throw new BudgetExhaustedException(modelConfig.Name, BudgetResource.PrefillBuffers, prefillClaim.ExhaustedAtTier!.Value, prefillClaim.Breakdown);
                }

                string clampDetail = model.PrefillRowClamp is { } clamp
                    ? $" clamp_tier='{BudgetTierNames.Of(clamp.Tier)}' clamp_cap_mb={clamp.CapMegabytes} row_bytes={clamp.RowActivationBytes} clamp={clamp.AllowsPositiveWidth}"
                    : string.Empty;
                logger?.LogInformation(
                    "ledger_claim model='{Model}' resource='prefill_buffers' tier='{Tier}' amount_mb={AmountMb} chain_position={ChainPosition} demoted={Demoted} row_width={RowWidth} effective_row_width={EffectiveRowWidth} activation_bytes={ActivationBytes}{ClampDetail}",
                    modelConfig.Name,
                    BudgetTierNames.Of(prefillClaim.Claim!.Tier),
                    prefillClaim.Claim.Amount,
                    prefillClaim.Claim.ChainPosition,
                    prefillClaim.Claim.Demoted,
                    model.ResolvedPrefillRowWidth,
                    model.EffectivePrefillRowWidth,
                    activationBytes,
                    clampDetail);
            }

            // A resolved device-backed range registers as a standing claim of its own before any block crosses to the device, walked against exactly the positions that can hold persistent copies. An exhausted chain refuses here — with the breakdown every other exhaustion reports, naming model, resource, tier, and what each consulted position holds against its ceiling — rather than opening a pool bounded by nothing or serving work its declared ceiling cannot bound.
            if (deviceBackedBlocks > 0)
            {
                ResolvedBudgetChain? resolvedKvChain = budgets.For(BudgetResource.Kv);
                List<ResolvedBudgetTier> devicePositions = [];

                if (resolvedKvChain is not null)
                {
                    foreach (ResolvedBudgetTier position in resolvedKvChain.Tiers)
                    {
                        if (!position.IsOriginStore && position.Tier == BudgetTier.Vram)
                        {
                            devicePositions.Add(position);
                        }
                    }
                }

                if (devicePositions.Count == 0)
                {
                    throw new ModelException(
                        $"model '{modelConfig.Name}' declares 'placement.kv_residency' '{BlockResidencyModeNames.Of(kvResidency)}': device-backed blocks seat against a copy-holding '{BudgetTierNames.Vram}' position of their '{BudgetResourceNames.Kv}' chain, and this model resolves none. " +
                        $"Name '{BudgetTierNames.Vram}' in 'models[].budgets.kv.order' with a finite ceiling to bound {deviceBackedBlocks} blocks.");
                }

                ClaimOutcome deviceClaim = ledger.RegisterClaim(
                    modelConfig.Name,
                    new ResolvedBudgetChain { Resource = BudgetResource.Kv, Tiers = [.. devicePositions] },
                    deviceBackedBlocks);

                if (!deviceClaim.Seated)
                {
                    throw new BudgetExhaustedException(modelConfig.Name, BudgetResource.Kv, deviceClaim.ExhaustedAtTier!.Value, deviceClaim.Breakdown);
                }

                logger?.LogInformation(
                    "ledger_claim model='{Model}' resource='kv' tier='{Tier}' amount_blocks={AmountBlocks} chain_position={ChainPosition} demoted={Demoted}",
                    modelConfig.Name,
                    BudgetTierNames.Of(deviceClaim.Claim!.Tier),
                    deviceClaim.Claim.Amount,
                    deviceClaim.Claim.ChainPosition,
                    deviceClaim.Claim.Demoted);
            }

            // The pool opens with the residency this open resolved: its arena stays owned by this service's CPU backend — the allocation and free list every attention read addresses through — while only the declared leading block range carries a persistent copy in the accelerator backend this open owns. A host-only reservation passes no device-backed range and opens no mirror at all, exactly as it always has.
            pool = new BlockPool(
                blockTokens,
                model.Plan.LayerCount,
                model.Plan.KvHeadCount,
                model.Plan.HeadDim,
                modelConfig.KvBlocks,
                backend,
                kvResidency,
                deviceBackedBlocks,
                mirrorOwner: (IBackend?)kvStorageOwner ?? backend);

            if (deviceBackedBlocks > 0)
            {
                logger?.LogInformation(
                    "kv_residency model='{Model}' residency={Residency} device_backed_blocks={DeviceBackedBlocks} persistent_device_bytes={PersistentDeviceBytes} bytes_per_block={BytesPerBlock}",
                    modelConfig.Name,
                    BlockResidencyModeNames.Of(pool.Residency),
                    pool.DeviceBackedBlockCount,
                    pool.DeviceMirrorBytes,
                    pool.BytesPerBlock);
            }

            // The KV claim observes the pool opening that already happened: the reservation the pool reserved at open,
            // stated in this resource's block unit, against the first copy-holding position the resolved KV chain names
            // that an opened capacity seam backs. The pool's own soft cap stays the enforcement seam — this claim
            // records where the chain seated the reservation, and a chain held end-to-end by claims registered ahead
            // refuses the open as a load failure rather than shrinking the reservation.
            if (budgets.For(BudgetResource.Kv) is { } kvChain)
            {
                ClaimOutcome kvClaim = ledger.RegisterClaim(modelConfig.Name, kvChain, modelConfig.KvBlocks);
                if (!kvClaim.Seated)
                {
                    throw new BudgetExhaustedException(modelConfig.Name, BudgetResource.Kv, kvClaim.ExhaustedAtTier!.Value, kvClaim.Breakdown);
                }

                reservationKvClaimTier = kvClaim.Claim!.Tier; // the standing seat this engine's live block budget governs.

                logger?.LogInformation(
                    "ledger_claim model='{Model}' resource='kv' tier='{Tier}' amount_blocks={AmountBlocks} chain_position={ChainPosition} demoted={Demoted}",
                    modelConfig.Name,
                    BudgetTierNames.Of(kvClaim.Claim!.Tier),
                    kvClaim.Claim.Amount,
                    kvClaim.Claim.ChainPosition,
                    kvClaim.Claim.Demoted);
            }

            int[] attentionWindows = new int[model.Plan.LayerCount];
            for (int layer = 0; layer < model.Plan.LayerCount; layer++)
            {
                attentionWindows[layer] = model.Plan.AttentionWindowForLayer(layer);
            }

            pool.DeclareAttentionWindows(attentionWindows);

            PrefixCache cache = new(pool);

            // The scheduler carries this model's admission cap and name: the configured concurrency limit bounds how many pipelines live at
            // once (absent bounds nothing), refusals name the served model, and the route resolution above already landed placement quotas on
            // the plan that composition steps against. A configured step-time budget is carried along the same way: chunk width then comes from
            // measured step cost within the configured ceiling instead of being the ceiling itself, and absent leaves it as the ceiling.
            scheduler = new GenerationScheduler(
                model,
                pool,
                config.Engine.MaxBatchTokens,
                cache,
                maxConcurrentRequests: modelConfig.MaxConcurrentRequests ?? GenerationScheduler.UncappedConcurrencyCap,
                modelName: modelConfig.Name,
                prefillStepBudgetMs: config.Engine.PrefillStepBudgetMs,
                ledger: ledger);

            // The chat template is parsed once here, at model load, and cached for the loaded instance's lifetime: no
            // per-request parse and no lazy first-request parse. An unparseable template fails model startup fast,
            // naming the model path plus the offending construct and its offset. The model's literal BOS/EOS pieces
            // are bound to the template's bos_token/eos_token context variables so the wrapper can report whether a
            // rendered prompt already carries the model BOS.
            MinjaChatTemplate? chatTemplate = null;
            TemplateCapabilities? chatTemplateCapabilities = null;
            if (file.Metadata.ChatTemplate is string source)
            {
                string bosToken = LiteralTokenPiece(tokenizer, tokenizer.BosId);
                string eosToken = LiteralTokenPiece(tokenizer, tokenizer.EosId);

                try
                {
                    chatTemplate = new MinjaChatTemplate(source, bosToken, eosToken);
                }
                catch (MinjaSyntaxException ex)
                {
                    logger?.LogError(
                        "model path={ModelPath} has an invalid chat template: construct={Construct} at offset={Position}",
                        modelConfig.Path,
                        ex.Construct,
                        ex.Position);

                    throw new InvalidOperationException(
                        $"model '{modelConfig.Name}' at '{modelConfig.Path}' has an invalid chat template: {ex.Message}",
                        ex);
                }

                // Capability facts are probed once at load so request-time tool-precedence decisions have them without
                // re-probing per request. Probing never throws: a probe that cannot render reports that capability absent.
                chatTemplateCapabilities = chatTemplate.DetectCapabilities();
            }

            // The output-parser plan is resolved once here, from the load-probed capabilities plus the model's tool
            // configuration: every later chat request consumes this plan instead of re-deciding which tool surface it
            // gets, and the resolved choice is announced on the host's load log in the style of the tokenizer-default
            // and profile-resolution lines.
            OutputParserPlan outputParser = OutputParserRegistry.Resolve(modelConfig.ToolParser, modelConfig.ForceFencedTools, chatTemplateCapabilities, modelConfig.Reasoning, modelConfig.MaxReasoningTokens is >= 1 ? markerPair : null);
            logger?.LogInformation(
                "tool_parser model='{Model}' choice={Choice}",
                modelConfig.Name,
                ToolParserNames.Of(outputParser.ToolParser));
            logger?.LogInformation(
                "reasoning model='{Model}' policy={Policy}",
                modelConfig.Name,
                ReasoningPolicyNames.Of(outputParser.Reasoning));
            if (outputParser.ReasoningRegion is { } reasoningRegion)
            {
                logger?.LogInformation(
                    "reasoning_region model='{Model}' open_token_id={OpenTokenId} close_token_id={CloseTokenId}",
                    modelConfig.Name,
                    reasoningRegion.OpenTokenId,
                    reasoningRegion.CloseTokenId);
            }

            // The integer-dot posture the opened backends actually run is announced in the same style: it states the encoding
            // the ISA axis resolved to against this host's posture and the grid staging that encoding feeds, so an auto run
            // and a forced-ISA run of one model diff on those two fields. A dedicated lane resolves both sides against the
            // same axis and posture, so one line states the pair every side runs.
            CpuSelectionReport selection = backend.ResolvedSelection;

            logger?.LogInformation(
                "cpu_selection model='{Model}' isa={Isa} gather={Gather}",
                modelConfig.Name,
                selection.ResolvedIsa,
                selection.GatherMode);

            // The prefill route the forward engine resolved to is announced in the same style: it reports what this
            // engine actually runs, and a selection with no kernel never reaches here because opening refused it.
            logger?.LogInformation(
                "prefill_kernel model='{Model}' kernel={Kernel}",
                modelConfig.Name,
                PrefillKernelNames.Of(model.ResolvedPrefillKernel));

            // Both spellings of this model's system-memory KV ceiling arrived and agreed — the loader refuses any other combination — so startup sizing is unchanged and one line says which spelling carries it forward. A legacy key alone or budgets alone log nothing here, exactly as they behaved before.
            if (modelConfig.LegacyKvBlocksDeclared && DeclaresSystemMemoryKvCeiling(modelConfig))
            {
                logger?.LogInformation(
                    "kv_blocks model='{Model}' value={Value} deprecated: 'budgets.kv.caps.dram_blocks' states the same ceiling, and budgets are canonical for what LAIR holds at system memory",
                    modelConfig.Name,
                    modelConfig.KvBlocks);
            }

            // The sparse-FFN facts a routed block resolves to are reported the same way: they come straight from the
            // model's plan, and only a file that declares experts carries them. A dense model advertises nothing here.
            if (model.Plan.SparseFfn is { } sparsePlan)
            {
                logger?.LogInformation(
                    "sparse_ffn model='{Model}' layout={Layout} rank={Rank} strategy={Strategy}",
                    modelConfig.Name,
                    sparsePlan.Layout,
                    sparsePlan.Rank ?? (object)"none",
                    SparseFfnStrategyNames.Of(model.Plan.SparseFfnStrategy!.Value));
            }

            ModelInfo info = new(
                modelConfig.Name,
                maxContext,
                modelConfig.KvBlocks,
                DominantQuant(file.Tensors),
                model.Plan.Architecture,
                model.Plan.VocabSize,
                ToolParserNames.Of(outputParser.ToolParser),
                ReasoningPolicyNames.Of(outputParser.Reasoning),
                model.Plan.SparseFfnStrategy is { } strategy ? SparseFfnStrategyNames.Of(strategy) : null,
                modelConfig.MaxConcurrentRequests);

            GenerationService service = new(file, backend, model, tokenizer, chatTemplate, chatTemplateCapabilities, outputParser, markerPair, markersProven ? string.Empty : markerRefusal, logger, pool, scheduler, info, config, prefillsBackend, prefillsLane, splitExecution, kvMirrorExecution);
            service._cacheForMetrics = cache; // the metrics bus reports cached blocks from the same cache the scheduler uses
            service._ledger = ledger; // the claims this open registered ride the same ledger the scheduler reports admissions to
            service._reservationKvClaimTier = reservationKvClaimTier; // a live block-budget edit retargets exactly the seat this claim took
            service._compute = compute; // the core accounting this model's grant seated against, shared with any engine opened beside it
            service._computeGrant = computeGrant; // requested against granted, reported on monitor facts beside the live worker count
            service._placement = placement; // the node this engine seated on, reported on monitor facts beside its grant
            service._updateCandidate = updateCandidate; // the report-only candidate this pinned pack has against hub HEAD, reported additively on the monitor
            file = null;
            model = null;
            pool = null;
            scheduler = null;
            return service;
        }
        catch
        {
            scheduler?.Dispose();
            pool?.Dispose();
            model?.Dispose();
            file?.Dispose();
            backend.Dispose();
            prefillsBackend?.Dispose();
            splitExecution?.Dispose(); // the accelerator side of a declared layer split belongs to this open exactly as the CPU backends do.
            kvMirrorExecution?.Dispose(); // so does an accelerator backend opened only to hold device-backed blocks, whose pool never got built.
            throw;
        }
    }

    /// <summary>
    /// Runs one generation to completion through the batch scheduler. The prompt is tokenized, prefilled in chunks,
    /// then decoded in shared steps until eos, the token budget, or a stop string. Prompt prefixes already computed by
    /// earlier requests are adopted from the prefix cache instead of recomputed, and the adopted token count is
    /// reported on the result as <see cref="GenerationResult.CachedPromptTokens"/>. Cancellation surfaces as
    /// <see cref="OperationCanceledException"/>.
    /// </summary>
    public async Task<GenerationResult> GenerateAsync(GenerationRequest request, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, typeof(GenerationService));

        if (!string.Equals(request.Model, Model.Id, StringComparison.Ordinal))
        {
            throw new ModelNotFoundException(request.Model);
        }

        cancellationToken.ThrowIfCancellationRequested();

        if (request.PresencePenalty is < -2f or > 2f)
        {
            throw new ArgumentException($"'presence_penalty' must be in [-2, 2], got {request.PresencePenalty}.", nameof(request));
        }

        if (request.FrequencyPenalty is < -2f or > 2f)
        {
            throw new ArgumentException($"'frequency_penalty' must be in [-2, 2], got {request.FrequencyPenalty}.", nameof(request));
        }

        if (request.TopLogprobs is < 0 or > 20)
        {
            throw new ArgumentException($"'top_logprobs' must be in [0, 20], got {request.TopLogprobs}.", nameof(request));
        }

        // Budget overrides are sized request knobs too: a value below 1 is refused here for programmatic callers exactly as the
        // API host refuses it on the wire. The ceiling itself belongs to the loaded model and is applied where the override is
        // admitted, so what reaches this facade is already resolved.
        if (request.MaxReasoningTokens is < 1)
        {
            throw new ArgumentException($"'max_reasoning_tokens' must be 1 or greater, got {request.MaxReasoningTokens}.", nameof(request));
        }

        if (request.MaxResponseTokens is < 1)
        {
            throw new ArgumentException($"'max_response_tokens' must be 1 or greater, got {request.MaxResponseTokens}.", nameof(request));
        }

        IReadOnlyDictionary<int, float>? logitBias = ResolveLogitBias(request.LogitBias);

        // Tool declarations and the choice are validated here too so programmatic callers meet the same rules as HTTP ones.
        // The pipeline is active only for chat requests that declare tools under a mode other than none: which tool surface
        // the request gets — the template's own rendering or LAIR's fenced block — is resolved once by capability
        // precedence, and detection reads recognized calls out of the decoded text against the names this choice allows.
        // The plan this request runs on is the one resolved when it was admitted, captured once here: a new-requests-only
        // reload re-resolves the shared plan for later requests and leaves an already-admitted sequence on what it started
        // with — its tool surface, its reasoning policy, everything else about its output follows the same one value.
        OutputParserPlan plan = Volatile.Read(ref _outputParser);

        ToolPipeline tools = ResolveToolPipeline(request, plan);

        IReadOnlyCollection<string> callableNames = tools.Active
            ? tools.Choice.Mode == ToolChoiceMode.Function
                ? [tools.Choice.FunctionName!]
                : [.. tools.DeclaredTools!.Select(tool => tool.Name)]
            : [];

        IReadOnlyList<int> prompt = request.IsChat
            ? EncodeChat(request, tools, cancellationToken)
            : EncodeCompletion(request);

        if (prompt.Count >= Model.ContextLength)
        {
            // The token-budget refusal is recorded where it decides: the monitor's diagnostic ring sees exactly what
            // admission turned away, while this request still answers through its own error shape.
            ContextLengthExceededException contextRefusal = new(prompt.Count, Model.ContextLength, Model.Id);
            _metrics?.RecordAdmissionRefusal(AdmissionRefusalKinds.ContextLength, Model.Id, contextRefusal.Message);
            throw contextRefusal;
        }

        ModelConfig modelConfig = EffectiveConfiguration.Models[0];
        ResolvedTokenBudgets budgets = ResolveTokenBudgets(request, modelConfig, plan);
        int generationCeiling = ResolveGenerationCeiling(request, budgets);
        int maxTokens = Math.Min(generationCeiling, Model.ContextLength - prompt.Count);
        if (maxTokens <= 0)
        {
            // The clamp left no budget: the sequence generates nothing and reports the budget as spent.
            return new GenerationResult(string.Empty, prompt.Count, 0, "length");
        }

        // Monitor lifecycle keys on the caller's correlation id when one was given (the API host passes the same value it
        // answers with in x-lair-request-id); programmatic callers get a generated id so their stream rows still correlate.
        EngineMetricsBus? bus = _metrics;
        string monitorRequestId = !string.IsNullOrWhiteSpace(request.RequestId) ? request.RequestId! : Guid.NewGuid().ToString("N");
        RequestTracker? tracker = bus is null ? null : new(bus, monitorRequestId, Model.Id, request.IsChat ? "chat" : "completion", prompt.Count);

        List<string> pieces = [];
        string streamed = string.Empty;
        int decodedChars = 0;
        bool stopped = false;
        PipelineHandle? handle = null;
        BudgetStreamState? budgetState = budgets.ReasoningTokens is null && budgets.ResponseTokens is null ? null : new BudgetStreamState(plan.ReasoningRegion);
        bool budgetCeilingReached = false;

        // The stop signal is a cancellation the sequence observes at its next step boundary rather than a call through the
        // handle: the scheduler can invoke the streaming hook for the first sampled token before this call has its handle
        // back, and a stop literal seen in that token must still end the sequence.
        using CancellationTokenSource stopSignal = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

        // The stop signal is the model's declared EOT id when it has one (tokenizer.ggml.eot_token_id);
        // models without an EOT id keep today's EOS behavior.
        int stopId = StopId;

        ScheduledRequest scheduled = new()
        {
            PromptTokens = [.. prompt],
            MaxTokens = maxTokens,
            ReasoningTokenBudget = budgets.ReasoningTokens,
            ResponseTokenBudget = budgets.ResponseTokens,
            Sampling = new SamplingParams
            {
                Temperature = request.Temperature,
                TopK = request.TopK,
                TopP = request.TopP,
                Seed = request.Seed,
                PresencePenalty = request.PresencePenalty,
                FrequencyPenalty = request.FrequencyPenalty,
                LogitBias = logitBias,
            },
            EosId = stopId,
            LogProbTopN = request.LogProbs ? Math.Max(request.TopLogprobs, 0) : null,
            OnTokenId = tokenId =>
            {
                // Step-boundary tok/s sampling rides the hook that already runs for every generated token: one tick,
                // two array writes, and at most a throttled lifecycle publish.
                tracker?.ObserveToken();

                string piece = _tokenizer.Decode([tokenId]);
                if (budgetState is not null)
                {
                    if (budgetState.Feed(piece, tokenId, budgets.ReasoningTokens, budgets.ResponseTokens))
                    {
                        budgetCeilingReached = true;
                        stopSignal.Cancel();
                    }
                }

                if (piece.Length == 0)
                {
                    return;
                }

                pieces.Add(piece);
                if (request.Stream && request.OnToken is not null)
                {
                    request.OnToken(piece);
                }

                streamed += piece;
                int stopIndex = StopIndex(streamed, request.Stop);
                if (stopIndex >= 0)
                {
                    // The stop literal is already streamed; trim it out of the final text below.
                    decodedChars = stopIndex;
                    stopped = true;
                    stopSignal.Cancel();
                }
            },
        };

        try
        {
            handle = _scheduler.Submit(scheduled, stopSignal.Token);
        }
        catch (SchedulerFullException full)
        {
            // The scheduler's capacity vocabulary is an engine-busy condition at this facade: surface it as the API-level EngineBusyException
            // naming kv_capacity and the served model rather than leaking a scheduling type to callers. No lifecycle frames were published — a
            // refused admission never entered the stream, but its refusal lands in the monitor's diagnostic ring so an operator sees what blocked.
            EngineBusyException busy = EngineBusyException.ForKvCapacity(Model.Id, full.FreeSlots);
            _metrics?.RecordAdmissionRefusal(AdmissionRefusalKinds.KvCapacity, Model.Id, busy.Message);
            throw busy;
        }
        catch (SchedulerConcurrencyFullException quota)
        {
            // A submission past this model's concurrency cap is the other engine-busy refusal: same bounded overload family at the wire, with a
            // message naming max_concurrent_requests and the served model. Nothing already admitted was disturbed by it; the monitor records the
            // refusal for operators exactly as its kv_capacity sibling does.
            EngineBusyException busy = EngineBusyException.ForConcurrencyQuota(Model.Id, quota.MaxConcurrentRequests);
            _metrics?.RecordAdmissionRefusal(AdmissionRefusalKinds.ConcurrencyQuota, Model.Id, busy.Message);
            throw busy;
        }
        catch (BudgetExhaustedException exhausted)
        {
            // A submission the host ledger's shared chain could not seat anywhere is the third bounded overload refusal: the exhausted chain is
            // host accounting across opened pools, not this engine's own KV cap re-deciding anything, so it surfaces through the same overload
            // family at the wire with the budget-exhaustion code. Nothing already admitted was disturbed; the monitor records the refusal exactly
            // as its kv_capacity and quota siblings do.
            EngineBusyException busy = EngineBusyException.ForBudgetExhaustion(Model.Id, exhausted.Resource, exhausted.Tier, exhausted.Breakdown);
            _metrics?.RecordAdmissionRefusal(AdmissionRefusalKinds.BudgetExhaustion, Model.Id, busy.Message);
            throw busy;
        }

        if (tracker is RequestTracker admitted)
        {
            _monitoredRequests[admitted.RequestId] = admitted;
            admitted.PublishStart();
        }

        SequenceOutcome outcome;
        try
        {
            outcome = await handle.Completion.ConfigureAwait(false);
        }
        finally
        {
            // Safety net: a sequence that is still running when this call is interrupted must not hold its blocks.
            handle.Cancel();
        }

        string text = _tokenizer.Decode(outcome.GeneratedTokens, includeSpecial: false);
        string finishReason;
        if (stopped)
        {
            text = text[..Math.Min(decodedChars, text.Length)];
            finishReason = "stop";

            // The scheduler reports a stop-ended sequence as cancelled — only the facade sees it was really the stop
            // literal, so its tally feeds the metrics bus's finish-reason counts.
            _metrics?.RecordStopCompletion();
        }
        else if (outcome.FinishReason == FinishReason.Failed)
        {
            finishReason = "failed";
        }
        else if (outcome.FinishReason == FinishReason.Canceled)
        {
            finishReason = budgetCeilingReached && !stopped && !cancellationToken.IsCancellationRequested ? "length" : "cancelled";
        }
        else if (outcome.FinishReason == FinishReason.Eos)
        {
            finishReason = "eos";
        }
        else
        {
            finishReason = "length";
        }

        string? budgetExhausted = null;
        if (budgetState is not null)
        {
            if (budgetState.ResponseCeilingReached)
            {
                budgetExhausted = "response";
            }
            else if (budgetState.ReasoningCeilingReached)
            {
                budgetExhausted = "reasoning";
            }
            else if (finishReason == "length")
            {
                // The token that exhausts the scheduler's combined ceiling never reaches the token hook, so a length finish
                // still has to be attributed from the resolved pair itself: no response ceiling means the reasoning ceiling
                // bounded the run, and a response ceiling present but not yet marked by the hook is what the combined ceiling stopped.
                budgetExhausted = budgets.ResponseTokens is null
                    ? budgets.ReasoningTokens is null ? null : "reasoning"
                    : budgets.ReasoningTokens is null || budgetState.ReasoningTokens < budgets.ReasoningTokens.Value ? "response" : "reasoning";
            }
        }

        // The completion frame publishes before failure semantics surface: a failed or cancelled generation still gets
        // its final lifecycle row with the status it ended on, so stream state matches engine state.
        if (tracker is RequestTracker finished)
        {
            _monitoredRequests.TryRemove(finished.RequestId, out _);
            finished.Complete(finishReason, outcome.PrefillCachedTokens, finishReason, budgetExhausted);
        }

        if (finishReason == "failed")
        {
            throw new EngineBusyException();
        }

        if (finishReason == "cancelled")
        {
            // A sequence the scheduler cancelled without a stop literal is either the caller's token or the safety net.
            cancellationToken.ThrowIfCancellationRequested();
            throw new OperationCanceledException(cancellationToken);
        }

        // A recognized call takes over the result: its markup vanishes from the visible text, the calls ride along as data,
        // and the finish reason becomes tool_calls no matter how decoding ended. What counts as a call follows the plan's
        // parser — fenced blocks for an injected contract, the template's own object shape when the template rendered the
        // tools — so each surface is read only by its matching reader; an inert plan has no parser, so its decoded text is
        // never read and its content stays byte-for-byte what the model generated.
        // Reasoning regions are recognized from the markup this generation actually contains, whichever model produced it:
        // boundaries are recorded either way, and only a strip policy withholds them from visible content. A keep policy
        // leaves the text byte-for-byte as generated, which is what every model serves today. Recognition runs before call
        // reading so a call markup inside reasoning that gets stripped never reaches the parser as a call. The plan in force
        // is the one this sequence was admitted on, not whatever a reload has since resolved.
        if (budgetState is not null)
        {
            text = ApplyBudgetStreamState(text, budgetState);
        }

        ReasoningDetection reasoning = plan.Reasoning == ReasoningPolicy.Strip
            ? ReasoningRecognition.Strip(text)
            : new ReasoningDetection(ReasoningRecognition.FindRegions(text), text);
        if (plan.Reasoning == ReasoningPolicy.Strip)
        {
            text = reasoning.VisibleText;
        }

        IReadOnlyList<ToolCall>? toolCalls = null;
        IToolOutputParser? outputParser = plan.Parser;
        if (tools.Active && outputParser is not null)
        {
            ToolDetection detection = outputParser.ReadCalls(text, callableNames);
            if (detection.Calls.Count > 0)
            {
                text = detection.VisibleText;
                finishReason = "tool_calls";
                toolCalls = detection.Calls;
            }
        }

        return new GenerationResult(
            text,
            outcome.PromptTokens,
            outcome.CompletionTokens,
            finishReason,
            outcome.PrefillCachedTokens,
            request.LogProbs ? outcome.LogProbs : null,
            toolCalls,
            reasoning.Regions.Count > 0 ? reasoning.Regions : null,
            plan.Reasoning,
            budgetExhausted);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        // The metrics sampler notices disposal at its next tick and completes every open stream subscription on the
        // way out; it is not worth blocking engine teardown on one sampling delay, so this is deliberately unawaited.
        if (_metrics is not null)
        {
            ValueTask completion = _metrics.DisposeAsync();
            if (!completion.IsCompletedSuccessfully)
            {
                // The sampler swallows its own tick faults; this just keeps any late one from surfacing as an
                // unobserved task exception during teardown.
                completion.AsTask().ContinueWith(static faulted => _ = faulted.Exception, TaskContinuationOptions.OnlyOnFaulted);
            }

            _metrics = null;
        }

        // The scheduler releases every sequence's blocks before the pool they belong to goes away; the forward
        // engine's mmap view must be released before the file handle it maps.
        _scheduler.Dispose();
        _pool.Dispose();
        _model.Dispose(); // the engine releases every staging arena to the backend that allocated it before any backend goes away.
        _file.Dispose();
        _backend.Dispose();
        _prefillsBackend?.Dispose(); // the lane's pool workers belong to this service exactly as the decode side's do

        // Both accelerator backends this open owns go last: the pool that pinned a host arena flushed across its mirror while that arena was still alive, so what remains here is releasing device storage and context — for the layer split's compute backend, and for one opened only to hold device-backed blocks.
        _splitExecution?.Dispose();
        _kvMirrorExecution?.Dispose();
    }

    private IReadOnlyList<int> EncodeChat(GenerationRequest request, ToolPipeline tools, CancellationToken cancellationToken)
    {
        ChatPrompt prompt = BuildChatPrompt(request, tools, cancellationToken);

        // BOS policy: the template is rendered exactly, then the tokenizer's own BOS is suppressed only when the
        // rendered prompt already starts with the model BOS token. No blind stripping — when the template did not emit
        // the BOS, the model's add_bos_token policy still prefixes it.
        bool prefixBos = _tokenizer.AddBosToken && !prompt.RenderStartsWithBos;
        return prefixBos
            ? _tokenizer.EncodeWithBos(prompt.Text)
            : _tokenizer.Encode(prompt.Text);
    }

    // The full chat-prompt build the request path performs: validate the conversation, render the model's template,
    // and apply the resolved tool precedence — the template's own tool rendering or LAIR's fenced block, never both.
    private ChatPrompt BuildChatPrompt(GenerationRequest request, ToolPipeline tools, CancellationToken cancellationToken)
    {
        IReadOnlyList<Lair.Tokenizer.Templating.ChatMessage>? messages = request.Messages;
        if (messages is null || messages.Count == 0)
        {
            throw new ArgumentException(
                "a chat request requires a non-empty 'messages' list; send a completion request with a raw 'prompt' instead.",
                nameof(request));
        }

        if (_chatTemplate is null)
        {
            throw new InvalidOperationException(
                $"model '{Model.Id}' has no 'tokenizer.chat_template' in its GGUF metadata, so chat requests cannot be " +
                "rendered; serve a model with a chat template or send a completion request with a raw 'prompt'.");
        }

        ChatTemplateInput input = new()
        {
            Messages = [.. messages.Select(MapChatMessage)],
            AddGenerationPrompt = true,

            // DN-M32-7 precedence: when the template owns the tool surface, the declared tools enter the template
            // context and the template formats them; the fenced-injection path keeps them out of the context so
            // exactly one tool surface reaches the model.
            Tools = tools.TemplateTools is null ? null : [.. tools.TemplateTools.Select(TemplateToolValues.Of)],
        };

        ChatTemplateRenderResult render;
        try
        {
            render = _chatTemplate.Render(input, cancellationToken: cancellationToken);
        }
        catch (MinjaRenderException ex)
        {
            // A template that parsed but fails to render this request — a deliberate raise_exception, a limit, or a
            // cancelled render (the library reports cancellation as RenderLimitsException, never a raw
            // OperationCanceledException) — surfaces as a bounded, actionable template error naming the model and the
            // correlation id. The HTTP layer maps it to an invalid_request response.
            string requestId = string.IsNullOrWhiteSpace(request.RequestId) ? string.Empty : request.RequestId!;
            string detail = ex is RenderLimitsException limits ? $"{limits.Limit} limit ({ex.Construct})" : ex.Construct;
            _logger?.LogError(
                "model '{Model}' request_id={RequestId} chat template render failed ({Detail}): {Message}",
                Model.Id,
                requestId,
                detail,
                ex.Message);

            throw new TemplateRenderFailedException(Model.Id, requestId, ex.Message, ex);
        }

        // The tool block rides on the rendered text before encoding: the injection point is the prompt string, so every
        // template family receives the same instruction line, schema fences, and reply contract.
        string rendered = render.Prompt;
        if (tools.InjectedTools is not null)
        {
            rendered = ToolPromptComposer.Compose(rendered, tools.InjectedTools);
        }

        return new ChatPrompt(rendered, render.StartsWithBos);
    }

    // One built chat prompt: the exact string the tokenizer receives, and whether the template render already
    // carried the model BOS (the tokenizer's own BOS is suppressed only in that case).
    private sealed record ChatPrompt(string Text, bool RenderStartsWithBos);

    /// <summary>
    /// The exact chat-prompt build the request path performs for one chat request — tool validation, capability-driven
    /// tool precedence, template render, and fenced injection — returned as the prompt string exactly as it is
    /// tokenized. Hosts and tests observe the adapter's real prompt decision through this seam.
    /// </summary>
    internal string BuildChatPrompt(GenerationRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        return BuildChatPrompt(request, ResolveToolPipeline(request, Volatile.Read(ref _outputParser)), cancellationToken).Text;
    }

    /// <summary>
    /// The request's resolved tool pipeline: the declared tools, the resolved choice, whether the pipeline is active
    /// for this request, and which tool surface the active pipeline uses — the tools the template renders itself or
    /// the tools LAIR injects as its fenced block. Exactly one of the two tool lists is non-null when active.
    /// </summary>
    private sealed record ToolPipeline(
        IReadOnlyList<ToolDefinition>? DeclaredTools,
        ToolChoiceSelection Choice,
        bool Active,
        IReadOnlyList<ToolDefinition>? TemplateTools,
        IReadOnlyList<ToolDefinition>? InjectedTools);

    private ToolPipeline ResolveToolPipeline(GenerationRequest request, OutputParserPlan outputParser)
    {
        IReadOnlyList<ToolDefinition>? declaredTools = ValidateToolDeclarations(request.Tools);
        ToolChoiceSelection choice = ResolveToolChoice(request.ToolChoice, declaredTools);

        // ToolChoiceMode.None closes the pipeline before any precedence decision: no template tools, no injection.
        // An inert plan closes it the same way for every request: the model gets no tool surface at all.
        if (!request.IsChat || declaredTools is not { Count: > 0 } || choice.Mode == ToolChoiceMode.None || outputParser.ToolParser == ToolParser.None)
        {
            return new ToolPipeline(declaredTools, choice, false, null, null);
        }

        return outputParser.ToolParser == ToolParser.Fenced
            ? new ToolPipeline(declaredTools, choice, true, null, declaredTools)
            : new ToolPipeline(declaredTools, choice, true, declaredTools, null);
    }

    private static ChatMessage MapChatMessage(Lair.Tokenizer.Templating.ChatMessage message) =>
        new() { Role = message.Role, Content = MinjaValue.Of(message.Content) };

    // The literal vocabulary text of a special token id, or empty when the id is unset. This is the string bound to the
    // template's bos_token/eos_token context variables and compared against a rendered prompt for BOS suppression.
    private static string LiteralTokenPiece(GgufTokenizer tokenizer, int tokenId) =>
        tokenId >= 0 ? tokenizer.Decode([tokenId], includeSpecial: true) : string.Empty;

    private IReadOnlyList<int> EncodeCompletion(GenerationRequest request)
    {
        if (request.Prompt is null)
        {
            throw new ArgumentException(
                "a completion request requires a 'prompt'; set 'is_chat' with 'messages' to use the model's chat template.",
                nameof(request));
        }

        return _tokenizer.AddBosToken ? _tokenizer.EncodeWithBos(request.Prompt) : _tokenizer.Encode(request.Prompt);
    }

    // logit_bias keys are token strings or numeric token ids. A numeric key names its id directly; a string
    // key resolves through the served tokenizer and must encode to exactly one vocabulary token. Anything
    // unresolvable, out of vocabulary, or out of the [-100, 100] value range is rejected naming the key.
    private IReadOnlyDictionary<int, float>? ResolveLogitBias(IReadOnlyDictionary<string, float>? bias)
    {
        if (bias is null || bias.Count == 0)
        {
            return null;
        }

        Dictionary<int, float> resolved = [];
        foreach ((string key, float value) in bias)
        {
            int tokenId;
            if (int.TryParse(key, NumberStyles.Integer, CultureInfo.InvariantCulture, out int numericId))
            {
                tokenId = numericId;
                if (tokenId < 0 || tokenId >= Model.VocabSize)
                {
                    throw new ArgumentException(
                        $"'logit_bias' key '{key}' is not a token id of this model's vocabulary.",
                        nameof(bias));
                }
            }
            else
            {
                IReadOnlyList<int> encoded = _tokenizer.Encode(key);
                if (encoded.Count != 1)
                {
                    throw new ArgumentException(
                        $"'logit_bias' key '{key}' does not resolve to a single token of this model's vocabulary.",
                        nameof(bias));
                }

                tokenId = encoded[0];
            }

            if (value is < -100f or > 100f)
            {
                throw new ArgumentException(
                    $"'logit_bias' value for '{key}' must be in [-100, 100], got {value}.",
                    nameof(bias));
            }

            resolved[tokenId] = value;
        }

        return resolved;
    }

    private ResolvedTokenBudgets ResolveTokenBudgets(GenerationRequest request, ModelConfig modelConfig, OutputParserPlan plan)
    {
        int? reasoning = request.MaxReasoningTokens;
        if (reasoning is not null)
        {
            if (!plan.HasResolvedReasoningRegion)
            {
                throw new ArgumentException(
                    $"model '{Model.Id}' resolves no reasoning region, so 'max_reasoning_tokens' cannot be applied to it: declare max_reasoning_tokens for the model or omit the field.",
                    nameof(request));
            }

            if (modelConfig.MaxReasoningTokens is int reasoningCeiling && reasoning > reasoningCeiling)
            {
                reasoning = reasoningCeiling;
            }
        }
        else
        {
            reasoning = modelConfig.MaxReasoningTokens;
        }

        int? response = request.MaxResponseTokens;
        if (response is null)
        {
            response = modelConfig.MaxResponseTokens;
        }
        else if (modelConfig.MaxResponseTokens is int responseCeiling && response > responseCeiling)
        {
            response = responseCeiling;
        }

        if (reasoning is < 1)
        {
            throw new ArgumentException($"'max_reasoning_tokens' must be 1 or greater, got {reasoning}.", nameof(request));
        }

        if (response is < 1)
        {
            throw new ArgumentException($"'max_response_tokens' must be 1 or greater, got {response}.", nameof(request));
        }

        return new ResolvedTokenBudgets(reasoning, response);
    }

    private static int ResolveGenerationCeiling(GenerationRequest request, ResolvedTokenBudgets budgets)
    {
        if (budgets.ReasoningTokens is null && budgets.ResponseTokens is null)
        {
            return request.MaxTokens;
        }

        int sum = (budgets.ReasoningTokens ?? 0) + (budgets.ResponseTokens ?? 0);
        if (sum <= 0)
        {
            return request.MaxTokens;
        }

        return request.MaxTokensWasSupplied ? Math.Min(request.MaxTokens, sum) : sum;
    }

    private static string ApplyBudgetStreamState(string text, BudgetStreamState state)
    {
        if (state.SyntheticCloseChar >= 0)
        {
            int closeAt = Math.Min(state.SyntheticCloseChar, text.Length);
            text = text[..closeAt] + ReasoningRecognition.CloseMarkup + text[closeAt..];
        }

        if (state.ResponseStopChar >= 0)
        {
            int stopAt = state.ResponseStopChar;
            if (state.SyntheticCloseChar >= 0 && stopAt >= state.SyntheticCloseChar)
            {
                stopAt += ReasoningRecognition.CloseMarkup.Length;
            }

            stopAt = Math.Min(stopAt, text.Length);
            text = text[..stopAt];
        }

        return text;
    }

    private sealed record ResolvedTokenBudgets(int? ReasoningTokens, int? ResponseTokens);

    private sealed class BudgetStreamState
    {
        private const string Open = ReasoningRecognition.OpenMarkup;
        private const string Close = ReasoningRecognition.CloseMarkup;

        // The ids model open proved deliver each marker whole for this model (-1 when the vocabulary carries no such entry).
        // A budgeted model always has them; a marker-less vocabulary can only ever hold a response budget, where region
        // boundaries are read from the text the pieces carry.
        private readonly int _openTokenId;
        private readonly int _closeTokenId;
        private readonly List<string> _pieces = [];
        private int _charPos;
        private bool _inReasoning;

        public BudgetStreamState(ReasoningRegionMarkers? markers)
        {
            _openTokenId = markers?.OpenTokenId ?? -1;
            _closeTokenId = markers?.CloseTokenId ?? -1;
        }

        public int ReasoningTokens;
        public int VisibleTokens;
        public int SyntheticCloseChar = -1;
        public int ResponseStopChar = -1;
        public bool ReasoningCeilingReached;
        public bool ResponseCeilingReached;

        public bool Feed(string piece, int tokenId, int? reasoningCeiling, int? responseCeiling)
        {
            ArgumentNullException.ThrowIfNull(piece);

            _pieces.Add(piece);

            bool openArrivesWhole = _openTokenId >= 0 && tokenId == _openTokenId;
            bool closeArrivesWhole = _closeTokenId >= 0 && tokenId == _closeTokenId;

            bool startsInReasoning = _inReasoning;
            bool tokenIsReasoning = startsInReasoning || openArrivesWhole || piece.Contains(Open, StringComparison.Ordinal);

            int scan = 0;
            while (scan < piece.Length)
            {
                if (!_inReasoning)
                {
                    if (openArrivesWhole)
                    {
                        _inReasoning = true;
                        scan = piece.Length;
                        continue;
                    }

                    int open = piece.IndexOf(Open, scan, StringComparison.Ordinal);
                    if (open < 0)
                    {
                        break;
                    }

                    _inReasoning = true;
                    scan = open + Open.Length;
                }
                else
                {
                    if (closeArrivesWhole)
                    {
                        _inReasoning = false;
                        scan = piece.Length;
                        continue;
                    }

                    int close = piece.IndexOf(Close, scan, StringComparison.Ordinal);
                    if (close < 0)
                    {
                        break;
                    }

                    _inReasoning = false;
                    scan = close + Close.Length;
                }
            }

            _charPos += piece.Length;

            if (tokenIsReasoning)
            {
                ReasoningTokens++;
            }
            else
            {
                VisibleTokens++;
            }

            if (reasoningCeiling is int reasoningLimit && ReasoningTokens == reasoningLimit && _inReasoning)
            {
                ReasoningCeilingReached = true;
                SyntheticCloseChar = _charPos;
                _inReasoning = false;
            }

            if (responseCeiling is int responseLimit && !tokenIsReasoning && VisibleTokens == responseLimit)
            {
                ResponseCeilingReached = true;
                ResponseStopChar = _charPos;
                return true;
            }

            return false;
        }
    }

    private static int StopIndex(string text, IReadOnlyList<string>? stopStrings)
    {
        if (stopStrings is null)
        {
            return -1;
        }

        int best = -1;
        foreach (string stop in stopStrings)
        {
            if (stop.Length == 0)
            {
                continue;
            }

            int index = text.IndexOf(stop, StringComparison.Ordinal);
            if (index >= 0 && (best < 0 || index < best))
            {
                best = index;
            }
        }

        return best;
    }

    // A tool declaration must carry a non-empty unique name; an absent or empty list declares nothing and the pipeline stays closed.
    private static IReadOnlyList<ToolDefinition>? ValidateToolDeclarations(IReadOnlyList<ToolDefinition>? tools)
    {
        if (tools is null || tools.Count == 0)
        {
            return null;
        }

        HashSet<string> names = new(StringComparer.Ordinal);
        foreach (ToolDefinition tool in tools)
        {
            if (tool.Name.Length == 0)
            {
                throw new ArgumentException("'tools' contains a declaration with an empty 'name'.", nameof(tools));
            }

            if (!names.Add(tool.Name))
            {
                throw new ArgumentException($"'tools' declares the tool '{tool.Name}' more than once.", nameof(tools));
            }
        }

        return tools;
    }

    // A null choice means auto. A forced function must name one of the declared tools: only calls naming it are then accepted.
    private static ToolChoiceSelection ResolveToolChoice(ToolChoiceSelection? choice, IReadOnlyList<ToolDefinition>? declaredTools)
    {
        if (choice is null || choice.Mode == ToolChoiceMode.Auto)
        {
            return ToolChoiceSelection.Auto;
        }

        if (choice.Mode == ToolChoiceMode.None)
        {
            return ToolChoiceSelection.None;
        }

        string forced = choice.FunctionName ?? throw new ArgumentException(
            "'tool_choice' of mode function must name a tool.", nameof(choice));

        bool declared = false;
        foreach (ToolDefinition tool in declaredTools ?? [])
        {
            if (string.Equals(tool.Name, forced, StringComparison.Ordinal))
            {
                declared = true;
                break;
            }
        }

        if (!declared)
        {
            throw new ArgumentException(
                $"'tool_choice' names '{forced}', which is not one of the declared tools.", nameof(choice));
        }

        return ToolChoiceSelection.Function(forced);
    }

    // The file's quantization is the format its stored bytes bulk lives in, not the format with the most tensors: a model
    // whose weights are quantized and whose norm vectors sit at full precision reports the weight format even when those
    // norms outnumber every other group, and a huge-vocabulary model whose embedding carries more stored bytes than all of
    // its quantized weight groups together reports that format instead. Ties break on tensor count and then on ordinal
    // format name, so the reported value does not depend on table order. Internal for direct coverage of that rule.
    internal static string DominantQuant(LogicalTensorTable table)
    {
        List<IGrouping<QuantFormat, LogicalTensor>> groups = [.. table.Tensors.GroupBy(tensor => tensor.Format)];
        if (groups.Count == 0)
        {
            throw new InvalidOperationException("cannot report the quantization of a file with no tensors");
        }

        return groups
            .OrderByDescending(group => group.Sum(tensor => tensor.SizeBytes))
            .ThenByDescending(group => group.Count())
            .ThenBy(group => group.Key.ToString(), StringComparer.Ordinal)
            .First()
            .Key
            .ToString();
    }
}
