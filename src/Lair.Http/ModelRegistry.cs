using Lair.Host.Config;
using Lair.Http.Monitoring;
using Lair.Models;
using Microsoft.Extensions.Logging;

namespace Lair.Http;

/// <summary>
/// The served-model registry: every model the configuration declares opens as its own <see cref="GenerationService"/> instance,
/// each with its own scheduler, KV pool, and backends, while all of them share one host device ledger and one compute accounting —
/// so claims register in open order against what earlier models already hold. Requests name their model and route to that instance;
/// a request naming nothing this registry serves answers exactly as today's single-model engine answers an unknown id. A reload
/// diffs the declared model list by name: a newly added model loads in the background behind every claim already held and is advertised
/// only once ready, a removed model stops accepting requests, drains what it admitted, releases its ledger claims before its pools are
/// freed, and then leaves the advertised set — while every other instance keeps serving untouched. Per-instance field edits ride each
/// instance's own change-application pipeline against a view of the candidate carrying just that model, so today's mutation classes,
/// refusals, and applied-merge semantics apply unchanged to every served model at once.
/// </summary>
public sealed class ModelRegistry : IDisposable, IConfigApplyTarget
{
    /// <summary>Where one instance stands in its serve lifecycle.</summary>
    internal enum ServedPosture
    {
        /// <summary>The engine is loading; the id is not advertised and requests name it as unserved until it opens.</summary>
        Loading,

        /// <summary>The engine serves: submissions route to it.</summary>
        Ready,

        /// <summary>The model left the declared set: new submissions are refused while already-admitted work drains toward release.</summary>
        Draining,
    }

    internal sealed class ServedInstance
    {
        public required string Name { get; init; }

        // Null until a loading instance's open completes; every routing decision that reaches a non-null engine re-checks posture first.
        public GenerationService? Engine;

        // The declared entry this instance follows: the declaration it started from while loading, kept current for pending loads so
        // reload bookkeeping can name what is being loaded even before an engine exists behind it.
        public required ModelConfig Declaration { get; set; }

        public volatile ServedPosture Posture = ServedPosture.Loading;

        private int _inFlight;
        private TaskCompletionSource? _drainedSignal; // created the moment draining begins and completed when the last admitted request finishes

        public int InFlight => Volatile.Read(ref _inFlight);

        public void BeginAdmission() => Interlocked.Increment(ref _inFlight);

        /// <summary>Releases one admission slot; returns true when this was the last in-flight request of an instance already draining.</summary>
        public bool EndAdmission()
        {
            int remaining = Interlocked.Decrement(ref _inFlight);
            return Posture == ServedPosture.Draining && remaining == 0;
        }

        /// <summary>Marks the instance as retiring and hands back the signal its drain completion awaits — created under the reload gate, before any request can release one.</summary>
        public TaskCompletionSource BeginDrain()
        {
            Posture = ServedPosture.Draining;
            TaskCompletionSource drained = new(TaskCreationOptions.RunContinuationsAsynchronously);
            _drainedSignal = drained;

            if (Volatile.Read(ref _inFlight) == 0)
            {
                drained.TrySetResult(); // nothing was admitted while it still accepted work: the drain is already complete
            }

            return drained;
        }

        /// <summary>Completes the drain signal when a request finished after draining had already begun.</summary>
        public void SignalDrained() => _drainedSignal?.TrySetResult();
    }

    private readonly object _reloadGate = new();
    private readonly ILogger? _logger;
    private readonly IModelHubTransport? _transport;

    // Host-scoped accounting every instance of this registry shares: adopted from the first opened engine (or supplied whole by a host that
    // pre-builds capacity seams). Claims register in open order against what earlier instances hold, and removal releases each model's claims
    // before its pools go away — so an over-budget addition refuses with the ledger breakdown instead of displacing a seated claim.
    private DeviceLedger? _ledger;
    private ComputeCapacityLedger? _compute;

    // The served posture as one immutable snapshot: instances in declared order, replaced whole under the reload gate while readers take a volatile read.
    private ServedInstance[] _posture = [];

    // Background opens for added models, chained so each runs after the addition started before it — the order claims register against host accounting follows the declared order of the pass that queued them. Extended only under the reload gate.
    private Task _pendingLoads = Task.CompletedTask;

    // The configuration whose model list produced the current posture — the baseline a candidate's raw-scalar extraction reads against and the shape per-instance views carry their engine/server/ui values from.
    private LairConfig _candidateBase;

    private ReloadSummary? _lastReload;

    // Host shutdown flips this under the reload gate; late drain completions read it lock-free and leave already-disposed engines alone.
    private volatile bool _disposed;

    private ModelRegistry(LairConfig config, ILogger? logger, IModelHubTransport? transport)
    {
        ArgumentNullException.ThrowIfNull(config);

        _logger = logger;
        _transport = transport;
        _candidateBase = config;
    }

    /// <summary>
    /// Opens every model the configuration declares, in declared order: each opens with its own engine against the host ledger and compute
    /// accounting the first open established, so later models seat their claims after everything already held. A model that fails to open
    /// disposes what opened before it and refuses the whole startup — a multi-model serve begins only when every declared model can run.
    /// </summary>
    public static ModelRegistry Open(LairConfig config, ILogger? logger = null, IModelHubTransport? transport = null) =>
        Open(config, logger, transport, ledger: null, compute: null);

    internal static ModelRegistry Open(
        LairConfig config,
        ILogger? logger,
        IModelHubTransport? transport,
        DeviceLedger? ledger,
        ComputeCapacityLedger? compute = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Models.Count == 0)
        {
            throw new InvalidOperationException(
                "the configuration declares no models: add at least one entry under 'models:' before opening the registry.");
        }

        ModelRegistry registry = new(config, logger, transport);
        List<ServedInstance> opened = [];
        try
        {
            foreach (ModelConfig model in config.Models)
            {
                ServedInstance instance = new() { Name = model.Name, Declaration = model };

                // The first open seeds the host accounting when a caller supplied none — exactly today's single-engine posture, where the
                // ledger over its opened tiers and the compute grant live on the engine itself. Every later model registers against them.
                GenerationService engine = GenerationService.Open(
                    config with { Models = [model] },
                    logger,
                    registry._ledger ?? ledger,
                    registry._compute ?? compute,
                    transport);

                if (registry._ledger is null)
                {
                    registry._ledger = engine.HostDeviceLedger;
                    registry._compute = engine.HostCompute;
                }

                instance.Engine = engine;
                instance.Posture = ServedPosture.Ready;
                opened.Add(instance);
            }
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
        {
            foreach (ServedInstance instance in opened)
            {
                // Claims release before pools free, exactly as a removal releases them: a declared posture that cannot open every model it
                // declares serves nothing, so the claims its earlier opens seated must leave the host accounting rather than stand there
                // under names no engine holds — otherwise a refused startup would keep occupying tiers and cores.
                _ = registry._ledger?.ReleaseClaims(instance.Name);
                _ = registry._compute?.ReleaseGrant(instance.Name);

                instance.Engine?.Dispose();
            }

            _ = ex;
            throw; // startup stays atomic: a declared model that cannot open leaves no partially served registry behind.
        }

        Volatile.Write(ref registry._posture, [.. opened]); // published whole: readers take a volatile read of the snapshot array.
        return registry;
    }

    /// <summary>The advertised model facts, in declared order: every ready instance plus any still draining — loading instances stay unadvertised until their engine opens.</summary>
    public IReadOnlyList<ModelInfo> AdvertisedModels() =>
        [.. Volatile.Read(ref _posture)
            .Where(instance => instance.Posture is ServedPosture.Ready or ServedPosture.Draining && instance.Engine is not null)
            .Select(instance => instance.Engine!.Model)];

    /// <summary>The ids a request can name right now — the advertised set in declared order.</summary>
    public IReadOnlyList<string> AdvertisedIds() =>
        [.. Volatile.Read(ref _posture).Where(instance => instance.Posture is ServedPosture.Ready or ServedPosture.Draining).Select(instance => instance.Name)];

    /// <summary>The opened engines behind the advertised set, in declared order — what a host builds its HTTP surface over.</summary>
    internal IReadOnlyList<GenerationService> AdvertisedEngines() =>
        [.. Volatile.Read(ref _posture)
            .Where(instance => instance.Posture is ServedPosture.Ready or ServedPosture.Draining && instance.Engine is not null)
            .Select(instance => instance.Engine!)];

    /// <summary>The first advertised engine — the one whose snapshot bus a monitor stream samples while multi-instance monitoring stays out of scope.</summary>
    internal GenerationService MetricsEngine
    {
        get
        {
            ServedInstance? primary = Volatile.Read(ref _posture)
                .FirstOrDefault(instance => instance.Posture == ServedPosture.Ready && instance.Engine is not null);

            return primary?.Engine ?? throw new InvalidOperationException("the registry advertises no opened engine to sample monitor snapshots from.");
        }
    }

    /// <summary>The configuration the current posture follows: every live engine's own effective model entry merged back onto the candidate shape it loaded from. Hosts diff reloads against this.</summary>
    public LairConfig EffectiveConfiguration
    {
        get
        {
            ServedInstance[] snapshot = Volatile.Read(ref _posture);
            List<ModelConfig> models = [];

            foreach (ServedInstance instance in snapshot)
            {
                GenerationService? engine = instance.Engine;
                models.Add(engine is not null ? engine.EffectiveConfiguration.Models[0] : instance.Declaration);
            }

            return _candidateBase with { Models = [.. models] };
        }
    }

    /// <summary>The host device ledger every instance of this registry claims against. Internal diagnostics for hosts and tests.</summary>
    internal DeviceLedger? Ledger => Volatile.Read(ref _ledger);

    /// <summary>The host compute accounting every instance's worker grant seated against. Internal diagnostics for hosts and tests.</summary>
    internal ComputeCapacityLedger? Compute => Volatile.Read(ref _compute);

    /// <summary>The most recent completed reload attempt as one structured summary, or null if none has finished. Hosts assert against this instead of parsing log lines.</summary>
    public ReloadSummary? LastReload => Volatile.Read(ref _lastReload);

    /// <summary>Records the outcome of one reload attempt (applied or rejected) for programmatic observation by hosts and tests.</summary>
    public void ReportReload(ReloadSummary summary)
    {
        ArgumentNullException.ThrowIfNull(summary);
        Volatile.Write(ref _lastReload, summary);
    }

    /// <summary>
    /// The request-routing seam the HTTP surface resolves every named model through: a ready instance admits this submission and hands back
    /// its lease (dispose it when the generation finishes), which is also what drain completion waits on. An id that names no advertised
    /// instance — unknown, still loading, or already released — answers with <see cref="ModelNotFoundException"/> exactly as today's engine
    /// answers an unknown model; a retiring instance refuses new submissions with <see cref="EngineBusyException"/> carrying the retirement code.
    /// </summary>
    internal Lease BeginRequest(string modelId)
    {
        ObjectDisposedException.ThrowIf(_disposed, typeof(ModelRegistry));

        ServedInstance? instance = Find(modelId);
        if (instance is null || instance.Posture == ServedPosture.Loading)
        {
            throw new ModelNotFoundException(modelId); // a loading model reads as unserved until it advertises — the same refusal an unknown id gets.
        }

        GenerationService? engine = instance.Engine;
        if (engine is null || instance.Posture == ServedPosture.Draining)
        {
            string reason = $"model '{instance.Name}' no longer accepts requests: it has been removed from the served models list and is draining what already runs before its resources are released.";

            // The refusal lands on this model's own monitor bus when one exists, exactly like every other bounded admission refusal.
            engine?.Metrics?.RecordAdmissionRefusal(AdmissionRefusalKinds.ModelRetiring, instance.Name, reason);
            throw EngineBusyException.ForModelRetiring(instance.Name);
        }

        instance.BeginAdmission();
        return new Lease(instance, engine);
    }

    /// <summary>One admitted request: it names the instance generation runs on and releases its admission slot when disposed.</summary>
    internal sealed class Lease : IDisposable
    {
        private ServedInstance? _instance;

        internal Lease(ServedInstance instance, GenerationService engine)
        {
            _instance = instance;
            Engine = engine;
        }

        /// <summary>The opened engine this submission routes to.</summary>
        public GenerationService Engine { get; }

        public void Dispose()
        {
            ServedInstance? instance = Interlocked.Exchange(ref _instance, null);
            if (instance is not null && instance.EndAdmission())
            {
                instance.SignalDrained(); // the last in-flight request of a retiring model finished: its release may proceed.
            }
        }
    }

    /// <summary>
    /// Entry point of the change-application pipeline for a reloaded configuration: it diffs the declared model list by name against what is
    /// served, dispatches every per-field difference through the owning instance's own change-application pipeline (against a view carrying just
    /// that model, so today's mutation classes and refusal wording apply unchanged), starts background loads for newly added models behind every
    /// claim already held, and drains removed models before releasing their ledger claims. Removals report first, then each matched model's field
    /// results in declared order, then additions — one flat per-field result list the reload host logs and summarizes exactly like a single-model apply.
    /// </summary>
    public IReadOnlyList<FieldResult> ApplyConfiguration(LairConfig candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);

        lock (_reloadGate)
        {
            ObjectDisposedException.ThrowIf(_disposed, typeof(ModelRegistry));

            if (candidate.Models.Count == 0)
            {
                throw new InvalidOperationException(
                    "the reloaded configuration declares no models: a serving host must keep at least one entry under 'models:' — remove the serve or restart it instead.");
            }

            List<FieldResult> results = [];
            ServedInstance[] current = Volatile.Read(ref _posture);

            // Removals first: they free nothing synchronously (claims stay until their drain completes), but declaring them early keeps every
            // later bookkeeping decision reading a posture that already knows what is leaving.
            foreach (ServedInstance instance in current)
            {
                if (!candidate.Models.Any(declared => string.Equals(declared.Name, instance.Name, StringComparison.Ordinal)))
                {
                    int liveIndex = Array.FindIndex(current, held => ReferenceEquals(held, instance));
                    results.Add(BeginRemoval(instance, liveIndex));
                }
            }

            // Matched models next: each applies its own field diffs through its own pipeline against a candidate view carrying only that entry.
            HashSet<string> hostWideApplied = []; // engine/server/ui/host keys are one decision across every instance — an applied outcome reports once, while refusals stay per-model because their details name which model refused them.

            for (int index = 0; index < candidate.Models.Count; index++)
            {
                ModelConfig declared = candidate.Models[index];
                ServedInstance? matched = current.FirstOrDefault(instance => string.Equals(instance.Name, declared.Name, StringComparison.Ordinal));

                if (matched is null)
                {
                    continue; // additions are decided after every live instance has had its edits applied.
                }

                GenerationService? engine = matched.Engine;
                if (engine is not null && engine.Model.Id == matched.Name)
                {
                    foreach (FieldResult result in engine.ApplyConfiguration(candidate with { Models = [declared] }))
                    {
                        bool modelScoped = result.Change.Field.StartsWith("models[0]", StringComparison.Ordinal);

                        if (!modelScoped && result.Outcome == ConfigApplyOutcome.Applied && !hostWideApplied.Add(result.Change.Field))
                        {
                            continue; // already reported for this pass: the same host-wide value reached every instance's pipeline.
                        }

                        results.Add(ReportedAt(index, result));
                    }

                    continue;
                }

                // A loading instance has no opened pipeline yet: edits to its pending declaration are refused naming that state rather than half-applied against an engine that does not exist.
                LairConfig pendingView = _candidateBase with { Models = [matched.Declaration] };
                foreach (FieldChange change in ConfigMutationTable.Diff(pendingView, candidate with { Models = [declared] }))
                {
                    results.Add(new FieldResult(
                        Reported(index, change),
                        ConfigApplyOutcome.Refused,
                        $"{change.Field} refused for model '{matched.Name}' while it is still loading: the open in progress follows the declaration it started from — further edits to this entry apply once it advertises."));
                }

                matched.Declaration = declared; // later decisions (and a re-open after a failed load) read the newest declaration.
            }

            // Additions last: each new model is validated against what live instances still hold, then loads in the background behind every claim already seated.
            for (int index = 0; index < candidate.Models.Count; index++)
            {
                ModelConfig declared = candidate.Models[index];
                if (current.Any(instance => string.Equals(instance.Name, declared.Name, StringComparison.Ordinal)))
                {
                    continue; // matched above: this is an edit pass on a live instance, not an addition.
                }

                results.Add(BeginAddition(declared, index));
            }

            // The posture snapshot itself was maintained incrementally above (additions appended at kick-off, removals leave when their drain finishes); the candidate is what later bookkeeping diffs and views build from.
            _candidateBase = candidate;
            return [.. results];
        }
    }

    private FieldResult BeginRemoval(ServedInstance instance, int liveIndex)
    {
        ModelConfig declaration = DeclarationOf(instance);
        TaskCompletionSource drained = instance.BeginDrain();
        string canonical = CanonicalModel(declaration);

        // The drain completes asynchronously; claims release before the engine's pools and backends are freed, then the id leaves the advertised set.
        _ = Task.Run(() => FinishRemovalAsync(instance, drained));

        FieldChange change = new($"models[{liveIndex}]", canonical, "(removed)", ConfigMutationClass.Live);
        return new FieldResult(
            change,
            ConfigApplyOutcome.Applied,
            $"model '{instance.Name}' left the declared set: it stops accepting requests, {instance.InFlight} already-admitted request(s) drain to completion, its ledger claims release before its pools and backends are freed, and then it leaves the advertised models");
    }

    private async Task FinishRemovalAsync(ServedInstance instance, TaskCompletionSource drained)
    {
        try
        {
            await drained.Task.ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
        {
            _logger?.LogError("registry_removal model='{Model}' outcome=failed error={Reason}", instance.Name, ex.Message);
        }

        if (_disposed)
        {
            return; // host shutdown already disposed every engine this drain could release.
        }

        IReadOnlyList<DeviceLedgerClaim> claims = _ledger?.ReleaseClaims(instance.Name) ?? [];
        bool grantReleased = _compute?.ReleaseGrant(instance.Name) == true;

        instance.Engine?.Dispose();

        lock (_reloadGate)
        {
            RebuildPosture(Volatile.Read(ref _posture).Where(held => !ReferenceEquals(held, instance)).ToArray());
        }

        _logger?.LogInformation(
            "registry_removal model='{Model}' outcome=released claims_released={ClaimsReleased} compute_grant_released={GrantReleased}",
            instance.Name,
            claims.Count,
            grantReleased);
    }

    private FieldResult BeginAddition(ModelConfig declared, int candidateIndex)
    {
        string? conflict = CoreConflictAgainstLive(declared);
        if (conflict is not null)
        {
            FieldChange refused = new($"models[{candidateIndex}]", "(absent)", CanonicalModel(declared), ConfigMutationClass.Live);
            return new FieldResult(refused, ConfigApplyOutcome.Refused, conflict); // nothing opened: the live posture stays exactly as it was.
        }

        ServedInstance instance = new() { Name = declared.Name, Declaration = declared };

        // The posture gains the pending instance and its open joins the load queue under one gate: additions seat in declared order, each
        // walking the host accounting after every claim already held — including what an earlier addition of this same pass seated.
        lock (_reloadGate)
        {
            RebuildPosture([.. Volatile.Read(ref _posture), instance]);
            _pendingLoads = _pendingLoads.ContinueWith(_ => LoadAddedModel(instance), CancellationToken.None, TaskContinuationOptions.None, TaskScheduler.Default);
        }

        FieldChange change = new($"models[{candidateIndex}]", "(absent)", CanonicalModel(declared), ConfigMutationClass.Live);
        return new FieldResult(
            change,
            ConfigApplyOutcome.Applied,
            $"model '{declared.Name}' added to the declared set: it loads in the background behind every claim already held and advertises only once its engine opens");
    }

    private void LoadAddedModel(ServedInstance instance)
    {
        if (_disposed)
        {
            lock (_reloadGate)
            {
                RebuildPosture(Volatile.Read(ref _posture).Where(held => !ReferenceEquals(held, instance)).ToArray()); // host shutdown already took the whole posture down.
            }

            return;
        }

        // The load reads the newest declaration and candidate shape at the moment its open actually runs, so an edit landing while it queued is followed rather than raced past.
        LairConfig view = _candidateBase with { Models = [instance.Declaration] };
        try
        {
            GenerationService engine = GenerationService.Open(view, _logger, _ledger, _compute, _transport);
            instance.Engine = engine;
            instance.Posture = ServedPosture.Ready;

            lock (_reloadGate)
            {
                RebuildPosture(Volatile.Read(ref _posture)); // the snapshot identity stays put; only this instance's posture changed.
            }

            _logger?.LogInformation("registry_add model='{Model}' outcome=served", instance.Name);
        }
        catch (BudgetExhaustedException exhausted)
        {
            // A chain the live posture cannot seat against claims already held: the refusal names model, resource, tier, and ledger
            // breakdown in its own words. Nothing already serving was disturbed — no claim of this addition stands behind it either, since
            // an open that produced no engine releases what it had registered before failing.
            _logger?.LogError(
                "model_load_failed model='{Model}' outcome=refused resource='{Resource}' tier='{Tier}' error={Reason}",
                instance.Name,
                BudgetResourceNames.Of(exhausted.Resource),
                BudgetTierNames.Of(exhausted.Tier),
                exhausted.Message);

            RemoveUnseatedAddition(instance);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
        {
            // Any other posture this host cannot open — a chain naming an unopened residency, or a backend it has no device for — refuses the
            // same way: added to nothing, advertised by nobody, every live instance untouched.
            _logger?.LogError("model_load_failed model='{Model}' outcome=refused error={Reason}", instance.Name, ex.Message);

            RemoveUnseatedAddition(instance);
        }
    }

    private void RemoveUnseatedAddition(ServedInstance instance)
    {
        lock (_reloadGate)
        {
            RebuildPosture(Volatile.Read(ref _posture).Where(held => !ReferenceEquals(held, instance)).ToArray());
        }
    }

    // A hot addition's pinned cores against every core a live instance still pins — ready models and retiring ones alike, since a draining model's
    // workers contend until its release completes. The rule mirrors the loader's disjoint-pins contract: an overlap refuses naming both models and the
    // core, unless this addition opts into core overcommit for its own claim. Two additions in one pass are already covered by the loader's own validation.
    private string? CoreConflictAgainstLive(ModelConfig candidate)
    {
        if (candidate.Placement.CoreSet.Count == 0 || candidate.Placement.CoreOvercommit)
        {
            return null; // an unpinned posture pins nothing, and an opted-in claim shares held cores under the grant rules instead of being refused.
        }

        foreach (ServedInstance live in Volatile.Read(ref _posture))
        {
            ModelConfig declaration = DeclarationOf(live);
            if (!candidate.Placement.CoreSet.Overlaps(declaration.Placement.CoreSet))
            {
                continue;
            }

            int core = candidate.Placement.CoreSet.OrderBy(id => id).First(coreId => declaration.Placement.CoreSet.Contains(coreId));
            return $"models '{declaration.Name}' and '{candidate.Name}' both pin core {core}: a model added to the live posture cannot overlap cores an already-served model pins — disjoint the 'placement.core_set' of one entry or declare 'placement.core_overcommit: true' on '{candidate.Name}'.";
        }

        return null;
    }

    private static ModelConfig DeclarationOf(ServedInstance instance) =>
        instance.Engine is not null ? instance.Engine.EffectiveConfiguration.Models[0] : instance.Declaration;

    // One per-field result re-keyed to the model's position in the reloaded list: a sole live model keeps today's `models[0].` field names verbatim,
    // while every additional served model reports its own ordinal so one flat summary never attributes an edit to the wrong entry. Host-wide keys
    // (engine/server/ui/host) keep their shared name — they describe one decision across the whole serve, not a per-model position.
    private static FieldResult ReportedAt(int index, FieldResult result) => new(Reported(index, result.Change), result.Outcome, result.Detail);

    private static FieldChange Reported(int index, FieldChange change) =>
        index == 0 || !change.Field.StartsWith("models[0]", StringComparison.Ordinal)
            ? change
            : change with { Field = $"models[{index}]{change.Field["models[0]".Length..]}" };

    private void RebuildPosture(ServedInstance[] snapshot) => Volatile.Write(ref _posture, [.. snapshot]);

    private ServedInstance? Find(string modelId)
    {
        if (string.IsNullOrEmpty(modelId))
        {
            return null; // an unnamed request names no served model: the registry answers it exactly as today's engine answers a mismatched id.
        }

        foreach (ServedInstance instance in Volatile.Read(ref _posture))
        {
            if (string.Equals(instance.Name, modelId, StringComparison.Ordinal))
            {
                return instance;
            }
        }

        return null;
    }

    private static string CanonicalModel(ModelConfig model) => $"name={model.Name};path={model.Path}";

    public void Dispose()
    {
        lock (_reloadGate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true; // late drain completions see this and leave already-disposed engines alone.
        }

        foreach (ServedInstance instance in Volatile.Read(ref _posture))
        {
            instance.Engine?.Dispose();
        }

        Volatile.Write(ref _posture, []);
    }
}
