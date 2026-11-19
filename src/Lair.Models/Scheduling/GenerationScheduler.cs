using System.Diagnostics;
using System.Threading.Channels;
using Lair.Contracts;
using Lair.Kv;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Sampling;

namespace Lair.Models.Scheduling;

/// <summary>
/// Continuous-batching scheduler over one forward engine and one paged KV pool. Each admitted request is a
/// <see cref="RequestPipeline"/> — its own state machine (admit → transform → prefill → decode → finish), sampler, KV cache, and
/// admission reservation — advanced on this one scheduler thread by the control plane here, while the step executor decides how its
/// math runs: every admitted pipeline takes part in shared decode steps (a single batched forward per step for all decoding
/// pipelines), with at most one prefill chunk interleaved into that boundary — so a long prompt never blocks short pipelines. A
/// submission's admission is decided against <see cref="PipelineConcurrencyGate"/> slots and KV capacity, independently of which batch its
/// math rides in: a submission arriving while every cap slot is held is refused naming the model and its quota rather than queued behind
/// what already runs, and a merged step hands each pipeline the logits row an unmerged step would have produced for it, so a pipeline's
/// token stream is identical whether it advanced alone or inside a merged window. Step composition is route-aware: admitted pipelines
/// share one logical forward pass when they advance the same resolved model plan over its whole ordered route, hold KV the stepped segment
/// can address, fit that segment's batch cap, and leave quota headroom against it for what already joined — so a group whose route spans
/// several segments merges as one step rather than splitting per device or segment. A candidate kept out of one merged window by those
/// limits is deferred with the blocking limit named, holds its admission slot, and rides a later boundary: composition deferral never turns
/// into an admission refusal. Pipelines run in a fixed order on this thread — ascending pipeline id, and ids are handed out at admission in submission
/// order, so the oldest admitted sequence of this model advances first at every boundary — each with its own sampler. Admission counts the
/// blocks a pipeline will cross over its whole lifetime, so an admitted pipeline never trips <see cref="KvFullException"/> mid-run; when
/// the free list runs short and a prefix cache is attached, admission first reclaims cached prompt-prefix blocks (least-recently-used
/// first, never disturbing active pipelines) before rejecting. Admitted pipelines adopt leading cached prompt-prefix blocks instead of
/// prefilling them, and completed full prompt blocks are published back into the cache for later requests; an adopted block produces
/// bit-identical KV bytes to recomputing that prefix, so shared prefixes change speed, not results.
/// </summary>
public sealed class GenerationScheduler : IDisposable
{
    /// <summary>The cap value a scheduler runs with when nothing bounds how many pipelines it admits at once.</summary>
    public const int UncappedConcurrencyCap = PipelineConcurrencyGate.Uncapped;

    /// <summary>Prompt tokens a budgeted scheduler forwards in its first chunks, before this engine's prefill cost has been measured or inferred from what already stepped. A deliberate probe: with no cost to size against, the first boundary starts narrow rather than guessing at a wide step.</summary>
    internal const int InitialPrefillProbeTokens = 8;

    private readonly TransformerModel _model;
    private readonly IKvPool _pool;
    private readonly PrefixCache? _cache;

    // The model this scheduler serves, named in every admission refusal so each one says which limit kind blocked it and for
    // which model. Null keeps the generic wording schedulers without a name use.
    private readonly string? _modelName;

    // Prefill chunk width, re-read at every chunk issue so a hot edit lands at the next step boundary: a pipeline already
    // mid-chunk finishes that chunk at the old width. Volatile because the apply path writes it from a caller thread while only the
    // scheduler thread consumes it.
    private volatile int _maxBatchTokens;

    // Wall-clock budget a prefill step aims at, in whole microseconds, or zero when no budget is configured and chunk width stays as set.
    // Re-read at every chunk issue so a hot edit lands from the next chunk: a forward already running finishes at whatever width was chosen
    // before the edit, never split mid-step. An apply path writes it from a caller thread and only this scheduler's own thread consumes it, so
    // both sides go through Volatile — same publishing the step counters use.
    private long _prefillStepBudgetMicros;

    // Prefill cost reporting: measured microseconds per prompt token as an exponential moving average over issued chunks (zero until something has
    // been timed), and the width of the last chunk that produced a measurement. Selection scales future boundaries from that last chunk's width and
    // duration rather than from this per-token figure, whose job is to say what a measured step cost; both are written at boundaries on this
    // scheduler's own thread only, and observers read them as volatile below.

    private double _prefillMicrosPerToken;

    private int _measuredPrefillWidthTokens;

    // The most recent chunk selection and what it measured: the last boundary's width is read back by ChoosePrefillChunk when a budget scales the
    // next one, and these values are reported apart from each other because they bound different things — one bounds this step's time, the other is
    // the row window the engine splits it into. No admission decision or forward path consults them beyond that scaling.

    private volatile int _lastPrefillChosenTokens;

    private volatile int _lastPrefillSubBatchRows;


    private long _lastPrefillStepMicros;

    // Admission authority over how many pipelines this scheduler holds admitted at once, decided per submission and independent of step
    // composition: pipeline slots here bound concurrency, KV capacity bounds admission against real blocks, and a composition deferral
    // consults neither — it only decides which merged window an already-admitted pipeline rides.
    private readonly PipelineConcurrencyGate _progress;

    // The host ledger this scheduler reports its landed KV reservations to, when its caller supplied one. Purely an observer of
    // decisions the existing seams already made: a submission is admitted exactly as the concurrency gate and the pool's own
    // capacity accounting admit it, and only then does the claim record where the model's KV chain seated the reservation. A
    // claim the shared chain cannot seat refuses this submission naming the exhausted chain — host-scoped accounting across
    // opened pools, not a re-decision of the pool's soft cap or this gate — and never disturbs an admitted pipeline.
    private readonly DeviceLedger? _ledger;

    // Serializes every mutation of the pool and cache that is not issued from the scheduler thread: admission-time
    // eviction runs on caller threads, while all forwards, adoptions, publications, releases, slot handovers, and shutdown finishes
    // run under this gate. Holding it across a whole step keeps those paths disjoint without finer-grained locking;
    // submissions only ever wait for one step boundary.
    private readonly object _gate = new();

    // Submissions arrive here from caller threads; only the scheduler thread reads it. Unbounded because admission is bounded at
    // submission time — a refused submission never enters it, and an admitted one waits here only until its first boundary runs.
    private readonly Channel<RequestPipeline> _inbox;

    // Scheduler-thread-only working set of live pipelines, kept in ascending pipeline id: newly admitted ones first, then
    // prefilling ones, then decoding ones. A pipeline sits here from admission until its outcome is published.
    private readonly List<RequestPipeline> _active = [];

    // Scratch lists for the current decode window, cleared (never reallocated) per window.
    private readonly List<RequestPipeline> _window = [];
    private readonly List<int> _windowTokens = [];
    private readonly List<IKvSequence> _windowKvs = [];

    // Route-aware step composition over this model's admitted pipelines: which candidates share one logical forward pass at a
    // boundary, decided on shared plan-and-route identity plus KV residency against the segment being stepped, that segment's batch
    // cap, and its quota headroom for what already joined. It is separate from admission — the gate above decides only whether a
    // submission may exist as a pipeline, this decides which admitted pipelines ride one merged window — so an over-quota candidate
    // rides a later boundary without disturbing anyone already stepping. Scheduler-thread only: composition runs under the gate.
    private readonly StepCompositionArbiter _composition = new();

    // The route position a merged decode step advances on for this model: the last segment of its resolved route, which is the single
    // CPU span for every route this build can execute. A group whose route crosses devices stays one composition across that crossing
    // rather than splitting per segment; what steps each side of such a crossing is the work that opens backends per device.
    private readonly int _steppingSegment;

    // The stepping segment's quota as currently in force for future compositions: seeded from what load resolved and replaced wholesale
    // when a live knob re-resolves it (see ApplySteppingThreadQuota), so headroom checks read the current value rather than stale
    // load-time numbers while route identity — segments, batch caps, residency claims — stays exactly as load resolved it. Volatile
    // because an apply path writes it from a caller thread and only the scheduler thread consumes it at boundaries; pipelines already
    // admitted keep their slots either way.
    private volatile SegmentQuota _steppingQuota = new();

    // Scratch for the current boundary's step candidates and their pipelines, cleared (never reallocated) per boundary, with an id map
    // turning each composed candidate back into the pipeline whose math rides its window.
    private readonly List<StepCandidate> _candidates = [];
    private readonly List<RequestPipeline> _candidatePipelines = [];
    private readonly Dictionary<long, int> _candidateIndex = new();

    // Step-boundary engine counters, written only on the scheduler thread while the gate is held and read as
    // volatile longs by observers — collection costs a handful of increments per step, never per token.
    private readonly Stopwatch _stepClock = new();

    // One started/stopped shared Stopwatch per prefill chunk measures that chunk's forward work, which is what a step-time budget sizes the
    // next chunk from: no allocation and nothing inside the token loop.
    private readonly Stopwatch _prefillClock = new();

    /// <summary>Cumulative engine counters maintained live by the step loop; see <see cref="SchedulerStats"/>.</summary>
    public SchedulerStats Stats { get; } = new();

    private readonly Task _thread;
    private long _nextId;
    private bool _disposed;

    /// <param name="model">The forward engine shared by every pipeline.</param>
    /// <param name="pool">The paged KV pool every pipeline appends into.</param>
    /// <param name="maxBatchTokens">Largest prefill chunk forwarded in one step; must be at least 1.</param>
    /// <param name="cache">Optional prefix cache over the same pool: enables prompt-prefix adoption and publication. Without it, scheduling behaves exactly as without caching.</param>
    /// <param name="maxConcurrentRequests">How many pipelines this scheduler may hold admitted at once; must be at least 1, or <see cref="UncappedConcurrencyCap"/> to bound nothing. It caps admission: a submission arriving while every slot is held is refused with <see cref="SchedulerConcurrencyFullException"/> naming the model and its quota rather than queued behind what already runs.</param>
    /// <param name="modelName">The served model's id, named in every admission refusal so each one says which limit kind blocked it. Null keeps the generic wording.</param>
    /// <param name="prefillStepBudgetMs">Wall-clock budget, in milliseconds (1 or greater), that future prefill steps aim at; null forwards chunks at <paramref name="maxBatchTokens"/> exactly as before this knob existed. A present value sizes each chunk from measured step cost instead — see <see cref="ApplyPrefillStepBudget"/>.</param>
    /// <param name="ledger">The host device ledger this scheduler reports its landed KV reservations to, when its caller keeps host-scoped accounting. Supplying one observes chain decisions after the existing admission seams decide — a landed sequence rides inside what its model already seats, so no step is bounded and nothing admitted is disturbed; only a chain every position of which other models hold refuses that submission with <see cref="BudgetExhaustedException"/>. Null — what every scheduler built without host accounting uses — behaves exactly as before the ledger seam existed.</param>
    public GenerationScheduler(TransformerModel model, IKvPool pool, int maxBatchTokens, PrefixCache? cache = null, int maxConcurrentRequests = UncappedConcurrencyCap, string? modelName = null, int? prefillStepBudgetMs = null, DeviceLedger? ledger = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(pool);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxBatchTokens, 1);
        if (prefillStepBudgetMs is < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(prefillStepBudgetMs), "a prefill step budget must be 1 millisecond or greater, or null to forward chunks at the configured width.");
        }

        _model = model;
        _pool = pool;
        _cache = cache;
        _maxBatchTokens = maxBatchTokens;
        _prefillStepBudgetMicros = prefillStepBudgetMs is int budget ? (long)budget * 1_000 : 0;
        _modelName = modelName;
        _progress = new PipelineConcurrencyGate(maxConcurrentRequests);
        _ledger = ledger;

        // A merged decode step advances on this model's decode-side segment: the last position of its resolved route. For every route
        // openable in this build that is its single CPU span, so composition bounds merged sequences by exactly the cap load resolved for
        // it; a multi-segment route keeps its segments here as material to compose over even though no engine runs across them yet. Its
        // quota starts at what load resolved and follows live re-resolution from there (see ApplySteppingThreadQuota).
        _steppingSegment = model.Plan.Route.Segments.Count - 1;
        _steppingQuota = model.Plan.Route.Segments[_steppingSegment].Quota;

        _inbox = Channel.CreateUnbounded<RequestPipeline>();

        _thread = Task.Factory.StartNew(
            RunLoop,
            CancellationToken.None,
            TaskCreationOptions.LongRunning,
            TaskScheduler.Default);
    }

    /// <summary>How many pipelines this scheduler may hold admitted at once — the cap its progress gate enforces on submissions.</summary>
    public int MaxConcurrentRequests => _progress.MaxConcurrentRequests;

    /// <summary>Pipelines holding an admission slot right now, i.e. every live pipeline from submission until it finishes.</summary>
    public int AdvancingPipelineCount => _progress.HeldSlots;

    /// <summary>Highest concurrent admitted-pipeline count this scheduler has ever held in one window — zero only when nothing ran.</summary>
    public int PeakAdvancingPipelines => _progress.PeakHeldSlots;

    /// <summary>The compositions decided at the most recent step boundary: which candidates merged into one logical step over a shared route, and which were kept out with the cap or quota that blocked them named. Composition diagnostics for tests and internal accounting only — no admission decision, response shape, or forward path consults it.</summary>
    internal CompositionReport LastStepComposition => _composition.LastReport;

    /// <summary>Candidates this scheduler has kept out of a merged step since it opened because the stepping segment's batch cap or quota headroom was exhausted. Each one still advanced its pipeline at a later boundary, so this counts deferred compositions rather than refused submissions.</summary>
    internal long DeferredCompositionCandidates => _composition.DeferredCandidateCount;

    /// <summary>The refusals from the most recent boundary that kept any candidate out of a merged step, each naming the cap or quota that blocked it. Internal composition diagnostics: they report what bound a step and never turn into an admission refusal.</summary>
    internal IReadOnlyList<DeferredStepCandidate> LastCompositionRefusals => _composition.MostRecentRefusals;

    /// <summary>The route position merged decode steps compose on for this model: the last segment of its resolved execution path.</summary>
    internal int SteppingSegmentIndex => _steppingSegment;

    /// <summary>Prompt tokens the most recent prefill chunk forwarded — chosen from the step-time budget when one is configured and from the configured width otherwise. Internal selection diagnostics: they report what a boundary chose and never move a token.</summary>
    internal int LastPrefillChosenTokens => _lastPrefillChosenTokens;

    /// <summary>The widest sub-batch rows that chunk ran under this engine's resolved prefill row count, reported apart from the chunk's own token count because the two bound different things — width bounds step time, rows are what the engine splits it into and caps itself at. Internal diagnostics only.</summary>
    internal int LastPrefillSubBatchRows => _lastPrefillSubBatchRows;

    /// <summary>The row width every prompt window this scheduler hands its engine is capped at: the resolved count when nothing declared bounds it, narrowed to what a prefill-buffer ceiling allows when one binds. Internal selection diagnostics read apart from <see cref="LastPrefillChosenTokens"/> and <see cref="LastPrefillSubBatchRows"/> because they describe what caps rows across boundaries rather than what one boundary chose or ran.</summary>
    internal int EffectivePrefillRowWidth => _model.EffectivePrefillRowWidth;

    /// <summary>What this scheduler's engine resolved its row window against a declared prefill-buffer ceiling — the binding tier and cap, one sub-batch row's activation cost, requested beside allowed width, or null when no ceiling binds. Internal diagnostics: they report why rows narrowed and never move a token.</summary>
    internal PrefillRowWidthClamp? ResolvedPrefillRowClamp => _model.PrefillRowClamp;


    /// <summary>Wall-clock microseconds measured for the most recent prefill chunk's forward work, zero until a chunk has been timed. Internal diagnostics: this is the feedback a configured budget sizes future chunks from.</summary>
    internal long LastPrefillStepMicros => Volatile.Read(ref _lastPrefillStepMicros);

    /// <summary>The step-time prefill budget currently in force, in whole microseconds; zero when no budget is configured and chunk width stays as set. A hot edit lands here from the next chunk issue on.</summary>
    internal long PrefillStepBudgetMicros => Volatile.Read(ref _prefillStepBudgetMicros);

    /// <summary>The measured cost per prompt token across this scheduler's timed prefill chunks, zero until this engine has been timed for prefill. Internal diagnostics: reported so a host can see what a step cost, while budgeted choices scale from the last chunk's width and duration.</summary>
    internal double EstimatedPrefillMicrosecondsPerToken => Volatile.Read(ref _prefillMicrosPerToken);

    /// <summary>The chunk width measurement was fed back from — zero until a chunk has been timed, and the width the next budgeted boundary scales against. Internal diagnostics.</summary>
    internal int MeasuredPrefillWidthTokens => _measuredPrefillWidthTokens;

    /// <summary>
    /// Admits one request as a pipeline and returns its handle: the submission first takes an admission slot from this scheduler's
    /// concurrency cap — refused immediately with <see cref="SchedulerConcurrencyFullException"/> naming the model and its quota when every
    /// slot is already held by admitted sequences, rather than queued behind them — and only then pays for what the pool must hold for its
    /// whole run. The pipeline starts advancing at the next step boundary, never runs on more than one thread, and carries the sequence number this admission
    /// handed out — so boundaries take admitted pipelines of this model in submission order, whatever order the caller threads reached the inbox in. Admission spends from the
    /// pool's effective free side: with a prefix cache attached, admission pressure first evicts least-recently-used cached blocks (only
    /// those no active pipeline still references) and rechecks capacity before failing, so an admitted pipeline never trips
    /// <see cref="KvFullException"/> mid-run — its whole-lifetime reservation opens with its admission. A refusal releases everything it
    /// took: a rejected submission leaves no slot, no window, and no pool state behind.
    /// </summary>
    /// <exception cref="SchedulerConcurrencyFullException">Every slot of this scheduler's concurrency cap is held by already-admitted sequences; the refusal names the model and its quota.</exception>
    /// <exception cref="SchedulerFullException">The prompt plus the token budget needs more blocks than the pool currently has free, even after cache eviction.</exception>
    public PipelineHandle Submit(ScheduledRequest request, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, typeof(GenerationScheduler));
        if (request.PromptTokens.Length == 0)
        {
            throw new ArgumentException("a scheduled request needs at least one prompt token.", nameof(request));
        }

        if (request.MaxTokens <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), $"MaxTokens must be greater than zero, got {request.MaxTokens}.");
        }

        if (request.ReasoningTokenBudget is < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), $"ReasoningTokenBudget must be zero or greater, got {request.ReasoningTokenBudget}.");
        }

        if (request.ResponseTokenBudget is < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), $"ResponseTokenBudget must be zero or greater, got {request.ResponseTokenBudget}.");
        }

        int generationCeiling = request.GenerationTokenCeiling;
        if (generationCeiling <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), $"the resolved token budget must produce a generation ceiling greater than zero, got {generationCeiling}.");
        }

        int needed = BlocksNeeded(request.PromptTokens.Length, generationCeiling);

        long sequence; // assigned under the gate below, before this submission's pipeline is built with it.

        lock (_gate)
        {
            // Admission takes its concurrency slot first: an over-cap submission is refused here rather than queued behind what already
            // runs, so a capped model bounds how many pipelines exist live — never a queue that hides the cap. An uncapped gate hands every
            // submission its slot exactly as it always has, and nothing about this refusal disturbs an admitted pipeline: lowering the cap
            // later stops new admissions without preempting what already holds slots.
            if (!_progress.TryAcquireSlot())
            {
                throw new SchedulerConcurrencyFullException(_progress.MaxConcurrentRequests, _modelName);
            }

            try
            {
                // Admission spends from the pool's effective free side: physically free blocks that sit beyond a shrunk
                // soft budget are unreachable until cached-block eviction brings usage under the cap. Evicting one cache
                // entry moves at most one block out of the used column (a block shared with an active pipeline stays in
                // use), so both checks re-read EffectiveFreeBlocks instead of assuming each evicted block counts.
                if (needed > _pool.EffectiveFreeBlocks && _cache is not null)
                {
                    _cache.EvictBlocks(needed - _pool.EffectiveFreeBlocks);
                }

                if (needed > _pool.EffectiveFreeBlocks)
                {
                    throw new SchedulerFullException((long)_pool.EffectiveFreeBlocks * _pool.BlockTokens, _modelName);
                }
            }
            catch
            {
                // A refused capacity check gives its slot straight back: only the rejected submission is affected.
                _progress.ReleaseSlot();
                throw;
            }

            // The host ledger reports this admission's landed reservation only after the concurrency slot and the pool's own
            // capacity accounting already admitted the submission — the observation follows the existing seams, it never
            // re-decides them. A landed sequence rides inside what its model's standing claim already seats; only other
            // models' seated claims compete for a tier, so nothing here double-counts one engine's own memory or displaces
            // an earlier seating. A chain whose every position is held by others refuses this submission naming the
            // exhausted chain, and gives its slot straight back exactly as a refused capacity check did; every
            // already-admitted pipeline stays whole either way.
            if (_ledger is not null && _model.Plan.Budgets.For(BudgetResource.Kv) is { } kvChain)
            {
                ClaimOutcome observation = _ledger.ObserveAdmission(_modelName ?? "engine", kvChain, needed);
                if (!observation.Seated)
                {
                    _progress.ReleaseSlot();
                    throw new BudgetExhaustedException(_modelName ?? "engine", BudgetResource.Kv, observation.ExhaustedAtTier!.Value, observation.Breakdown);
                }
            }

            // The same observation reports this admission's prompt-window working set against the model's resolved prefill-buffer chain: the
            // engine already narrowed its row window to what its own declared ceilings allow, and the standing seat for that narrowed working
            // set was registered when the model opened. A landing rides inside that seat — nothing admitted is displaced and no live block moves —
            // while a chain whose every host-backed position other claims now fill refuses this submission naming the exhausted chain, exactly
            // as the KV landing above does. A model declaring no prefill-buffer chain never consults one.
            if (_ledger is not null && _model.Plan.Budgets.For(BudgetResource.PrefillBuffers) is { } prefillChain)
            {
                ClaimOutcome prefillObservation = _ledger.ObserveAdmission(_modelName ?? "engine", prefillChain, _model.PrefillWorkingSetMegabytes);
                if (!prefillObservation.Seated)
                {
                    _progress.ReleaseSlot();
                    throw new BudgetExhaustedException(_modelName ?? "engine", BudgetResource.PrefillBuffers, prefillObservation.ExhaustedAtTier!.Value, prefillObservation.Breakdown);
                }
            }

            Stats.SequencesAdmitted++;

            // The sequence number is handed out here, under the same gate that took this submission's admission slot: ascending id is what every boundary advances in, so an admitted pipeline's place ahead of later submissions is fixed by its arrival at admission rather than by whichever caller thread happens to reach the inbox first.
            sequence = Interlocked.Increment(ref _nextId);
        }

        RequestPipeline pipeline = new(request, ct, _pool, sequence);
        pipeline.HoldsSlot = true; // taken under the gate above and returned exactly once, when this pipeline finishes

        // The admission window carries this submission's reserved headroom across its whole run: if a budget shrink
        // lands while it is decoding (usage pushed above the cap by cache eviction that could not reclaim pinned
        // blocks), its own appends still reach what Submit already accounted for. Closed when the pipeline finishes, and held just as
        // firmly from admission on — otherwise waiting would cost capacity a later submission could steal.
        _pool.BeginAdmissionWindow(needed);
        pipeline.AdmissionWindowOpened = true;

        if (!_inbox.Writer.TryWrite(pipeline))
        {
            pipeline.HoldsSlot = false;
            _progress.ReleaseSlot();
            pipeline.AdmissionWindowOpened = false;
            _pool.EndAdmissionWindow();
            throw new ObjectDisposedException(nameof(GenerationScheduler));
        }

        return pipeline.Handle;
    }

    /// <summary>
    /// Blocks a pipeline of <paramref name="promptTokens"/> plus <paramref name="maxTokens"/> generated tokens crosses over its whole
    /// lifetime: the block containing the first prompt token through the block containing the last token the pipeline will ever forward
    /// (the final sampled token is never forwarded). Adopted prefix blocks are already in this count — they occupy exactly the positions
    /// they cover.
    /// </summary>
    private int BlocksNeeded(int promptTokens, int maxTokens)
    {
        int blockTokens = _pool.BlockTokens;
        int firstBlock = 0;
        int lastBlock = (promptTokens + maxTokens - 2) / blockTokens;
        return lastBlock - firstBlock + 1;
    }

    /// <summary>The block count this pool reserves for one scheduled request across its lifetime, using the request's resolved generation ceiling.</summary>
    public int BlocksNeededFor(ScheduledRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        return BlocksNeeded(request.PromptTokens.Length, request.GenerationTokenCeiling);
    }

    /// <summary>
    /// Applies a new prefill chunk width. The value is read fresh at every chunk issue, so the change takes effect at
    /// the next step boundary: pipelines already mid-chunk finish that chunk at the old width, and decode steps are
    /// untouched. Any in-flight pipeline's token stream stays deterministic either way — chunk boundaries never move
    /// tokens, only where one forward stops and the next begins. With a step-time budget in force this value is the ceiling
    /// a chosen width may reach rather than the width itself: the budget decides how wide future chunks are aimed to be
    /// (see <see cref="ApplyPrefillStepBudget"/>), and lowering this cap narrows them from its next boundary on.
    /// </summary>
    public void ApplyMaxBatchTokens(int maxBatchTokens)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxBatchTokens, 1);
        ObjectDisposedException.ThrowIf(_disposed, typeof(GenerationScheduler));

        _maxBatchTokens = maxBatchTokens;
    }

    /// <summary>
    /// Applies a new wall-clock budget that future prefill steps aim at, or null to drop it and forward chunks at the configured width. The
    /// value is read fresh at every chunk issue, so a change takes effect from the next prefill chunk this scheduler issues: a forward already
    /// running finishes at whatever width was chosen before the edit — never split mid-step — decode steps are untouched, and any in-flight
    /// pipeline's token stream stays identical either way because boundaries move forwards, not results. A present budget sizes each future chunk
    /// from what this engine measured: floor one prompt token, ceiling <see cref="ApplyMaxBatchTokens"/> as configured here, with the width scaled
    /// by how the last boundary's duration sat against the target, and the engine's own row ceilings keeping every sub-batch inside them whatever
    /// width is chosen.
    /// </summary>
    public void ApplyPrefillStepBudget(int? prefillStepBudgetMs)
    {
        if (prefillStepBudgetMs is < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(prefillStepBudgetMs), "a prefill step budget must be 1 millisecond or greater, or null to forward chunks at the configured width.");
        }

        ObjectDisposedException.ThrowIf(_disposed, typeof(GenerationScheduler));

        Volatile.Write(ref _prefillStepBudgetMicros, prefillStepBudgetMs is int budget ? (long)budget * 1_000 : 0);
    }


    /// <summary>
    /// Applies a new cap on how many pipelines this scheduler may hold admitted at once. It takes effect from the next submission on:
    /// pipelines already admitted keep their slots and finish normally, so lowering the cap stops further admissions while held slots remain
    /// at or above the new value rather than preempting work in flight, and raising it admits more submissions from that point on. A null
    /// cap bounds nothing again — every submission reaches admission as far as capacity allows. Nothing about an admitted pipeline's tokens
    /// changes either way — which pipelines advance together is step composition, and merging never moves a token.
    /// </summary>
    public void ApplyMaxConcurrentRequests(int? maxConcurrentRequests)
    {
        int cap = maxConcurrentRequests ?? UncappedConcurrencyCap;
        ArgumentOutOfRangeException.ThrowIfLessThan(cap, 1);
        ObjectDisposedException.ThrowIf(_disposed, typeof(GenerationScheduler));

        _progress.ApplyMaxConcurrentRequests(cap);
    }

    /// <summary>
    /// Waits until this scheduler's admission gate has room for another pipeline, up to <paramref name="budget"/>, and answers whether that
    /// room appeared before the wait ran out or <paramref name="ct"/> cancelled. It reserves nothing: a resolved-true answer says only that a slot
    /// came free (or the cap widened), so the caller still submits through <see cref="Submit"/> and is decided there — capacity, budget chains,
    /// and every other admission limit apply exactly as they do to a fresh submission. A scheduler already disposing answers false rather than
    /// leaving submissions waiting on room that can no longer be freed. Hosts that bound how long their submissions wait for a slot use this seam;
    /// <see cref="Submit"/> itself stays immediate-refusal, so an uncapped or fail-fast posture never blocks here at all.
    /// </summary>
    public Task<bool> WaitForAdmissionSlotAsync(TimeSpan budget, CancellationToken ct) =>
        _disposed ? Task.FromResult(false) : _progress.WaitForFreeSlotAsync(budget, ct);

    /// <summary>Submissions this scheduler's admission gate currently has waiting for room — the depth its wait list holds right now. Observers read it as a snapshot; nothing here bounds how many submissions their host lets wait.</summary>
    public int AdmissionWaitDepth => _progress.WaitingForSlotCount;

    /// <summary>
    /// Re-resolves the stepping segment's worker-thread quota to a new live value (or null to leave it at the engine's own thread choice).
    /// It takes effect from the next composition on: headroom checks for future merged steps bound against this value instead of what load
    /// resolved, while pipelines already admitted keep their slots and finish normally — lowering a quota retargets which candidates may join
    /// a later step, never preempts one already stepping. Route identity stays untouched here: segments, batch caps, residency claims, and the
    /// pinned core set are load-resolved material this seam cannot rewrite (a core-set change is a restart-only edit at configuration level).
    /// </summary>
    public void ApplySteppingThreadQuota(int? threadQuota)
    {
        if (threadQuota is < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threadQuota), "a stepping segment's worker-thread quota must be 1 or greater, or null to leave it at the engine's own choice.");
        }

        ObjectDisposedException.ThrowIf(_disposed, typeof(GenerationScheduler));

        _steppingQuota = _steppingQuota with { Threads = threadQuota };
    }

    /// <summary>The outcome of applying a new KV block budget to the pool this scheduler admits against.</summary>
    public readonly record struct KvBudgetApply(int AppliedBlocks, int EvictedCacheEntries);

    /// <summary>
    /// Applies a new soft cap on KV usage within the pool's fixed physical reservation. The gate is held across the whole
    /// operation so the budget flip and the cached-block drain land as one step-boundary change: raising the cap makes
    /// already-free blocks allocatable again immediately, up to the physical reservation (the applied value reports any
    /// clamp); lowering it withholds new allocations and drains least-recently-used cache entries until in-use count sits
    /// at or below the cap — active pipelines' blocks are never touched, so a mid-run pipeline completes normally. The
    /// freed usage flows straight into admission: <see cref="Submit"/> budgets against effective free blocks from its next call on.
    /// </summary>
    public KvBudgetApply ApplyKvBudget(int kvBlocks)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvBlocks);
        ObjectDisposedException.ThrowIf(_disposed, typeof(GenerationScheduler));

        lock (_gate)
        {
            int applied = _pool.SetEffectiveBudget(kvBlocks);
            int evicted = _cache?.TrimToBudget(applied) ?? 0;
            return new KvBudgetApply(applied, evicted);
        }
    }

    /// <summary>
    /// Stops the scheduler: the step loop exits at the next boundary, every unfinished pipeline finishes with
    /// <see cref="FinishReason.Canceled"/>, and every pipeline's KV blocks are released. Safe to call twice.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        // Submissions waiting for a slot this scheduler can no longer free are answered now rather than left to their own deadlines: the step
        // loop that returns slots is stopping, so nothing they wait on still exists.
        _progress.CloseWaiters();

        _inbox.Writer.TryComplete();
        _thread.GetAwaiter().GetResult();
    }

    private void RunLoop()
    {
        try
        {
            while (true)
            {
                lock (_gate)
                {
                    // Drain and adoption run as one boundary: every newly admitted pipeline adopts its cached prefix
                    // before any prefill or decode work can touch the pool for it. Adoption is not advancement — a pipeline drained at this
                    // boundary holds the admission slot Submit took for it, but issues no forward until this same boundary (or the next)
                    // starts it advancing from PipelineStage.Admitted.
                    while (_inbox.Reader.TryRead(out RequestPipeline? admitted))
                    {
                        AdoptPrefix(admitted);
                        _active.Add(admitted);
                    }

                    if (_active.Count > 0)
                    {
                        Step();
                    }
                }

                if (_active.Count == 0)
                {
                    // Idle: park on the inbox signal — no busy-spin — until the next submission or Dispose. The gate is
                    // not held while parked, so admission-time eviction from caller threads proceeds freely. A pipeline waiting for a
                    // slot keeps this loop stepping instead of parking, because slots free only at boundaries.
                    if (!_inbox.Reader.WaitToReadAsync().AsTask().GetAwaiter().GetResult())
                    {
                        return; // channel completed: the scheduler is disposed
                    }

                    continue;
                }
            }
        }
        finally
        {
            lock (_gate)
            {
                // Drain whatever is queued and release every pipeline's blocks, including submissions that never made
                // it into a step. A pipeline that never got a slot holds none to release.
                while (_inbox.Reader.TryRead(out RequestPipeline? orphaned))
                {
                    Finish(orphaned, FinishReason.Canceled);
                }

                foreach (RequestPipeline pipeline in _active)
                {
                    Finish(pipeline, FinishReason.Canceled);
                }
            }
        }
    }

    private void Step()
    {
        // Ascending pipeline id is the deterministic step order: queued and prefilling pipelines first, then decoding ones.
        _active.Sort(static (a, b) => a.Id.CompareTo(b.Id));

        // Cancellation is observed at the step boundary, before any forward work for this step — including for a pipeline that never
        // advanced past admission.
        for (int i = _active.Count - 1; i >= 0; i--)
        {
            RequestPipeline pipeline = _active[i];
            if (pipeline.CancelRequested || pipeline.Cancellation.IsCancellationRequested)
            {
                Finish(pipeline, FinishReason.Canceled);
            }
        }

        // Progress starts here: every live pipeline already holds the admission slot Submit took for it — a submission past the cap was
        // refused before this boundary rather than queued to it, and the cap never preempts work in flight — so idle admitted ones simply
        // start advancing at this boundary, oldest first. Starting decides who advances; nothing about it says which batch a pipeline's
        // math joins once the executor runs it.
        foreach (RequestPipeline pipeline in _active)
        {
            if (pipeline.IsFinished || !pipeline.HoldsSlot || pipeline.Stage != PipelineStage.Admitted)
            {
                continue; // already advancing, or finished at this boundary: nothing to start here
            }

            if (pipeline.PromptOffset < pipeline.Request.PromptTokens.Length)
            {
                pipeline.StartPrefilling();
            }
            else
            {
                // Admission adopted every prompt block from the cache, so no chunk remains to forward: this pipeline goes straight to
                // decoding at the same boundary it was chosen at, exactly as an entirely cached prefix did before advancement was gated.
                pipeline.StartDecoding();
            }
        }

        // At most one prefill chunk per step, oldest first: decode pipelines always run. A pipeline finished at this boundary
        // (cancellation) is skipped so no forward ever touches its released blocks, and a pipeline still waiting to start advancing issues
        // none until it does. Under a dedicated lane posture the chunk's KV blocks are reserved here on the scheduler thread and its
        // prompt window runs concurrently with this step's decode windows; shared posture keeps forwarding inline exactly as before.
        LanePrefillAttempt? lanePrefill = null;
        foreach (RequestPipeline pipeline in _active)
        {
            if (pipeline.IsPrefilling && !pipeline.IsFinished)
            {
                if (_model.DedicatedPrefillLane is not null)
                {
                    lanePrefill = IssueLanePrefill(pipeline);
                }
                else
                {
                    PrefillChunk(pipeline);
                }

                break;
            }
        }


        // Decode candidates for this boundary, in ascending pipeline id: every advancing decoding pipeline offers itself to step
        // composition with the plan it advances under and where its KV lives. Pipelines whose next append would cross the engine's
        // context limit are failed before they can be offered, so no forward ever runs one past the tables it has angles for. The window
        // budget uses the pool's current allocation ceiling — under an open admission window that reaches past a hot-shrunk cap to what
        // those admitted pipelines already reserved — sampled after this step's prefill (and any copy-on-write it performed), since those
        // are the last pool changes visible to window accounting: one snapshot per boundary governs every composition stepped at it.
        int blocksLeft = Math.Max(Math.Min(_pool.FreeBlocks, _pool.AllocationCeiling - _pool.UsedBlocks), 0);

        // Where one advancing pipeline's blocks actually live: the residency its own block table holds, read from the pool that handed those ids out. A host-only pool answers system memory for every sequence; a device-backed range answers accelerator memory for the ids inside it; and an unstarted sequence is about to take the first id, which sits in the device-backed range whenever a pool declares one. This is what composition consults rather than any declared label.
        WeightTier KvTierFor(IKvSequence kv) => _pool.IsDeviceBacked(kv.Blocks.Count > 0 ? kv.Blocks[^1] : 0) ? WeightTier.Vram : WeightTier.Dram;

        _candidates.Clear();
        _candidatePipelines.Clear();
        _candidateIndex.Clear();

        foreach (RequestPipeline pipeline in _active)
        {
            if (!pipeline.IsDecoding || pipeline.IsFinished)
            {
                continue;
            }

            if (pipeline.Kv.Length >= _model.MaxContextTokens)
            {
                Finish(pipeline, FinishReason.Failed, new InvalidOperationException(
                    $"sequence position {pipeline.Kv.Length} exceeds this model's max context of {_model.MaxContextTokens} tokens."));
                continue;
            }

            int index = _candidatePipelines.Count;
            _candidates.Add(new StepCandidate { PipelineId = pipeline.Id, Plan = _model.Plan, KvResidency = KvTierFor(pipeline.Kv) });
            _candidatePipelines.Add(pipeline);
            _candidateIndex[pipeline.Id] = index;
        }

        // Composition decides which of these candidates share one logical step: they merge when they advance this same plan over its
        // whole route — a hybrid-shaped group stays whole rather than splitting per device or segment — and only while the stepping
        // segment's batch cap has room for them, their KV residency is addressable by it, and they leave quota headroom against it for
        // what already joined. The headroom check reads this scheduler's current effective quota for that segment (seeded from load
        // resolution, re-resolved when a live knob changes), so future steps are bounded by the value in force now rather than stale
        // load-time numbers, while route identity and shape stay exactly as load resolved them. A candidate kept out holds its slot and
        // reservation, rides no forward at this boundary, and composes again at the next one — a deferral is never an admission refusal;
        // nothing about a merged window moves another pipeline's tokens, because each samples from its own logits row.
        CompositionReport report = _composition.Compose(_candidates, _steppingSegment, _steppingQuota);

        int advancedThisStep = 0;
        foreach (StepComposition composition in report.Compositions)
        {
            // One forward pass per composed step. Every append this window performs is accounted for here before the batch is issued, so
            // a pipeline whose next token needs a block the pool cannot hand out fails rather than tripping halfway through a batch — and
            // only that pipeline fails, exactly as it did when every advancing decoding pipeline shared one window unconditionally.
            _window.Clear();

            foreach (StepCandidate candidate in composition.Composed)
            {
                RequestPipeline pipeline = _candidatePipelines[_candidateIndex[candidate.PipelineId]];

                if (pipeline.Kv.Length % _pool.BlockTokens == 0)
                {
                    if (blocksLeft == 0)
                    {
                        Finish(pipeline, FinishReason.Failed, new KvFullException(
                            $"KV block pool cannot extend this sequence: the sequence at position {pipeline.Kv.Length} " +
                            $"needs a new block and only {_pool.EffectiveFreeBlocks} blocks are allocatable under the " +
                            $"{_pool.EffectiveBudgetBlocks}-block budget on a pool of {_pool.CapacityBlocks}."));
                        continue;
                    }

                    blocksLeft--;
                }

                _window.Add(pipeline);
            }

            if (_window.Count == 0)
            {
                continue; // this composition's candidates all failed their append check: no forward is issued for them at this boundary
            }

            // Step-boundary clock: one started/stopped shared Stopwatch per step measures the forward work of this
            // step's decode windows; no allocation and nothing inside the token loop.
            _stepClock.Restart();
            DecodeWindow(_window, 0, _window.Count);
            _stepClock.Stop();
            Stats.DecodeSteps++;
            Stats.DecodeWindowMicrosTotal += _stepClock.ElapsedTicks * 1_000_000 / Stopwatch.Frequency;
            Stats.WindowSequenceSum += _window.Count;

            advancedThisStep += _window.Count;
        }

        // A dedicated lane's chunk joined at this boundary: the prompt window ran beside every decode window above, and its logits are read here — after the join — so nothing samples from a forward still writing. The pipeline that finished prefilling starts decoding now and rides the next boundary's window; a shared-posture step already advanced it inline before composition began.
        JoinLanePrefill(lanePrefill);

        Stats.ActiveDecodingSequences = advancedThisStep;


        for (int i = _active.Count - 1; i >= 0; i--)
        {
            if (_active[i].IsFinished)
            {
                _active.RemoveAt(i);
            }
        }

        Stats.ActiveSequenceCount = _active.Count;
    }

    /// <summary>
    /// Adopts a freshly admitted pipeline's cached prompt prefix, when a cache is attached: the scheduler walks full
    /// leading blocks while their rolling token-prefix keys match published entries and stored-token verification
    /// passes, shares those physical blocks into the pipeline (read-only), and leaves its prefill starting at the
    /// first unmatched block. A partial tail block is never adopted — it must be forwarded fresh so the request gets
    /// its own logits path. The skipped-prefill token count lands on the pipeline for its outcome report.
    /// </summary>
    private void AdoptPrefix(RequestPipeline pipeline)
    {
        if (_cache is null || pipeline.Request.PromptTokens.Length < _pool.BlockTokens)
        {
            return; // a prompt shorter than one block can never share a cached prefix block
        }

        int adoptedBlocks = _cache.TryAdopt(pipeline.Request.PromptTokens, pipeline.Kv);
        if (adoptedBlocks > 0)
        {
            // The adopted blocks cover the prompt's leading positions: prefill resumes after them, and the skipped
            // token count rides along on the outcome as this pipeline's cache-hit report.
            int skipped = adoptedBlocks * _pool.BlockTokens;
            pipeline.PromptOffset = skipped;
            pipeline.PrefillCachedTokens = skipped;
            Stats.CacheHitSequences++;
            Stats.CacheSkippedTokens += skipped;
        }
    }

    // How many prompt tokens this boundary forwards for one pipeline. With no step-time budget in force that is the configured width exactly as
    // it always was; with one, the width is aimed at the budget from measured cost — never below a single token, never past the configured
    // ceiling or whatever remains of this prompt. Selection lives here rather than in the forward path: which window a pipeline's math rides is
    // scheduling, and the engine keeps enforcing its own row ceilings on top of whatever width lands here. Read fresh at every issue, so a hot
    // edit changes future boundaries only — never the width a step already running was chosen at.
    private int ChoosePrefillChunk(RequestPipeline pipeline)
    {
        int remaining = pipeline.Request.PromptTokens.Length - pipeline.PromptOffset;

        long budgetMicros = Volatile.Read(ref _prefillStepBudgetMicros);

        if (budgetMicros == 0)
        {
            return Math.Min(remaining, _maxBatchTokens);
        }

        int ceiling = Math.Min(remaining, _maxBatchTokens);

        int measuredWidth = _measuredPrefillWidthTokens;
        long measuredMicros = Volatile.Read(ref _lastPrefillStepMicros);

        if (measuredWidth <= 0 || measuredMicros <= 0)
        {
            // Nothing has been timed on this engine for a chunk yet: forward the probe width rather than guess at a wide first step, so
            // a budgeted scheduler's opening chunks cannot overshoot its own target badly before measurement arrives.
            return Math.Min(InitialPrefillProbeTokens, ceiling);
        }

        // Scale the width that was just measured by how its duration sat against this target: the next boundary aims at the same wall clock time.
        // Extrapolating a per-token cost instead would misread either direction of an engine's cost curve — a wide forward amortizes fixed work, so
        // its cost per token predicts narrow steps too cheaply and pins selection at the ceiling no matter how tight the budget is. Proportional
        // scaling from a width this engine actually ran converges on that curve wherever it sits.
        double scaled = measuredWidth * (double)budgetMicros / measuredMicros;

        if (scaled >= ceiling)
        {
            // Measured cost says this step can already forward everything left of the prompt inside its time target.
            return ceiling;
        }

        int chosen = Math.Max(1, (int)(scaled + 0.5));

        return chosen;
    }

    // Cost feedback recorded at this boundary: microseconds per prompt token of the chunk that just ran, smoothed over recent chunks so one noisy step
    // does not retune everything at once. A forward too fast to register a microsecond counts as no evidence rather than as free work. This is what
    // <see cref="EstimatedPrefillMicrosecondsPerToken"/> reports for observation; selection itself scales from the width and duration of the chunk
    // that last ran (see ChoosePrefillChunk), because per-token cost moves with width on any engine whose forwards amortize fixed work. Width is fed
    // back per token — never per sub-batch row — so the estimate carries the amortization a wide window enjoyed.
    private void FeedBackMeasuredCost(int tokens, long stepMicros)
    {
        if (tokens <= 0 || stepMicros <= 0)
        {
            return;
        }

        double measured = stepMicros / (double)tokens;

        Volatile.Write(
            ref _prefillMicrosPerToken,
            Volatile.Read(ref _prefillMicrosPerToken) <= 0.0
                ? measured
                : (Volatile.Read(ref _prefillMicrosPerToken) * 3.0 + measured) / 4.0);

        _measuredPrefillWidthTokens = tokens;
    }

    private void PrefillChunk(RequestPipeline pipeline)
    {
        int[] prompt = pipeline.Request.PromptTokens;
        int chunk = ChoosePrefillChunk(pipeline);

        try
        {
            // One chosen step, timed as one step: this is the measurement a step-time budget sizes future chunks from. No allocation and
            // nothing inside the token loop — the clock wraps the whole forward, sub-batches included.
            _prefillClock.Restart();
            _model.Prefill(prompt[pipeline.PromptOffset..(pipeline.PromptOffset + chunk)], pipeline.Kv);
            _prefillClock.Stop();

            AdvanceAfterPrefillChunk(pipeline, chunk, _prefillClock.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);
        }
        catch (Exception ex) when (ex is KvFullException || ex is InvalidOperationException)
        {
            Finish(pipeline, FinishReason.Failed, ex);
        }
    }

    // What a completed chunk leaves behind, shared by both postures: the boundary's chosen width and sub-batch rows land in the counters before anything else moves, cost feeds back from this chunk's own forward duration, cache publication runs for every full prompt block, and — when no chunks remain — sampling starts decoding exactly as it always did. The logits read here are the prefill route's: an alias of the model-owned buffer under a shared posture, lane-owned storage that no decode-side window touched while the chunk ran beside this step's decode windows under a dedicated one.
    private void AdvanceAfterPrefillChunk(RequestPipeline pipeline, int chunk, long forwardMicros)
    {
        // The engine's own row window splits this step into sub-batches underneath it — its resolved count, narrowed by a declared prefill-buffer
        // ceiling when one binds. Reported apart from the step's token width — which a budget bounds by time and a buffer cap bounds by rows,
        // while these are what the forward actually ran under — so neither number stands in for the other here.
        int subBatchRows = Math.Min(chunk, _model.EffectivePrefillRowWidth);

        // Recorded before the counters move, so an observer that sees a chunk counted also sees what that boundary chose and measured.
        _lastPrefillChosenTokens = chunk;
        _lastPrefillSubBatchRows = subBatchRows;
        Volatile.Write(ref _lastPrefillStepMicros, forwardMicros);

        pipeline.PromptOffset += chunk;
        Stats.PrefillChunksIssued++;
        Stats.PrefillTokensForwarded += chunk;
        Stats.PrefillStepMicrosTotal += forwardMicros;
        Stats.PrefillSubBatchSum += (chunk + subBatchRows - 1) / subBatchRows;
        if (_model.PrefillRowClamp is not null)
        {
            // A ceiling-bound window counts its chunks apart from the total so a clamp's effect on step count reads beside what it did to rows.
            Stats.ClampedPrefillChunksIssued++;
        }
        if (subBatchRows > Stats.PeakPrefillSubBatchRows)
        {
            Stats.PeakPrefillSubBatchRows = subBatchRows;
        }

        FeedBackMeasuredCost(chunk, forwardMicros);


        if (pipeline.IsPrefilling)
        {
            return; // more chunks remain; the next step forwards the next slice
        }

        if (_cache is not null)
        {
            // Prefill done: every full prompt-prefix block now has its KV fully written, so publish them in chain
            // order. Blocks already under their key (adopted by this same pipeline, or published earlier by an
            // identical prefix) are only renewed; the partial final block stays out of the cache.
            int publishedBlocks = _cache.Publish(pipeline.Request.PromptTokens, pipeline.Kv);
            Stats.CachePublishedBlocksTotal += publishedBlocks;
            Stats.CachePublishedTokensTotal += (long)publishedBlocks * _pool.BlockTokens;
        }

        // Prefill left the final prompt token's logits: this pipeline transforms and samples them on its own row, exactly as a
        // solo run does. The token is reported now but forwarded on the next step.
        pipeline.StartTransforming();
        int first = pipeline.SampleFrom(_model.PrefillRouteLogits());

        if (first == pipeline.Request.EosId)
        {
            Finish(pipeline, FinishReason.Eos);
            return;
        }

        pipeline.Generated.Add(first);
        Stats.PrefillSampledTokens++;
        pipeline.PendingToken = first;
        pipeline.StartDecoding();
        pipeline.Request.OnTokenId?.Invoke(first);
    }

    // A dedicated lane's chunk issue, run on the scheduler thread before any concurrent work: the width is chosen exactly as an inline step would choose it, and every block those appends cross into is reserved here — so the prompt window running off-thread advances its sequence without a single pool mutation mid-flight. The forward itself runs on a worker thread against the model's lane backend; its own stopwatch times just that forward, which is what budget sizing feeds from below. A reservation the pool cannot satisfy fails this pipeline at this boundary exactly as an inline append trip did.
    private LanePrefillAttempt? IssueLanePrefill(RequestPipeline pipeline)
    {
        int[] prompt = pipeline.Request.PromptTokens;
        int chunk = ChoosePrefillChunk(pipeline);

        try
        {
            // A dedicated lane is a float-posture shape — a quantized pair refuses one at model open — so this sequence is always the pool's float table here.
            ((SequenceKv)pipeline.Kv).ReserveAppends(chunk);
        }
        catch (Exception ex) when (ex is KvFullException || ex is InvalidOperationException)
        {
            Finish(pipeline, FinishReason.Failed, ex);
            return null;
        }

        int offset = pipeline.PromptOffset;
        IKvSequence kv = pipeline.Kv;
        System.Threading.Tasks.Task<long> forward = System.Threading.Tasks.Task.Run(() =>
        {
            Stopwatch clock = Stopwatch.StartNew();

            // The model picks this route's own backend and activation family internally — the lane side of both was resolved when it opened.
            _model.Prefill(prompt[offset..(offset + chunk)], kv);

            clock.Stop();
            return clock.ElapsedTicks * 1_000_000 / Stopwatch.Frequency;
        });

        return new LanePrefillAttempt(pipeline, chunk, forward);
    }

    // Joins the boundary's lane prefill once this step's decode windows have run. The logits sampled here are the lane family's — read only after the join, so no pipeline ever samples from a prompt window still writing. A fault in the off-thread forward surfaces as this pipeline's failure exactly like an inline one did; every other exception rides out of Step unchanged, as it already does for shared-posture forwards that trip outside the caught pair.
    private void JoinLanePrefill(LanePrefillAttempt? attempt)
    {
        if (attempt is null)
        {
            return;
        }

        try
        {
            long forwardMicros = attempt.Forward.GetAwaiter().GetResult();
            AdvanceAfterPrefillChunk(attempt.Pipeline, attempt.Chunk, forwardMicros);
        }
        catch (Exception ex) when (ex is KvFullException || ex is InvalidOperationException)
        {
            Finish(attempt.Pipeline, FinishReason.Failed, ex);
        }
    }

    private sealed record LanePrefillAttempt(RequestPipeline Pipeline, int Chunk, System.Threading.Tasks.Task<long> Forward);



    /// <summary>
    /// Advances <paramref name="count"/> pipelines from <paramref name="window"/> by one token each in one batched
    /// forward, splitting the window into engine-sized batches, then has every pipeline transform and sample its own logits row.
    /// The caller has already accounted for every append this step performs, so a forward that still trips the pool or context limit is
    /// a fault: every pipeline in the affected batch fails and the scheduler loop keeps running. Each pipeline's slice of the batched
    /// logits holds exactly what its solo forward would have produced, which is why a window of one and a window merged from many give
    /// that pipeline identical tokens.
    /// </summary>
    private void DecodeWindow(List<RequestPipeline> window, int start, int count)
    {
        for (int offset = 0; offset < count; offset += TransformerModel.MaxBatchSequences)
        {
            int batch = Math.Min(TransformerModel.MaxBatchSequences, count - offset);
            DecodeBatchOf(window, start + offset, batch);
        }
    }

    private void DecodeBatchOf(List<RequestPipeline> window, int start, int count)
    {
        _windowTokens.Clear();
        _windowKvs.Clear();
        for (int i = start; i < start + count; i++)
        {
            _windowTokens.Add(window[i].PendingToken);
            _windowKvs.Add(window[i].Kv);
        }

        try
        {
            // DecodeBatch reserves each pipeline's next slot before forwarding, so a pipeline whose append crosses the
            // pool's limit surfaces here — as the only failure point of the step.
            Span<float> logits = _model.DecodeBatch(_windowTokens, _windowKvs);

            int vocab = _model.Plan.VocabSize;
            for (int i = 0; i < count; i++)
            {
                RequestPipeline pipeline = window[start + i];
                int next = pipeline.SampleFrom(logits.Slice(i * vocab, vocab));

                if (next == pipeline.Request.EosId)
                {
                    Finish(pipeline, FinishReason.Eos); // eos ends the sequence and is never forwarded
                    continue;
                }

                pipeline.Generated.Add(next);
                Stats.DecodeTokensAdvanced++;
                if (!pipeline.HasBudget)
                {
                    Finish(pipeline, FinishReason.Length);
                    continue;
                }

                pipeline.PendingToken = next;
                pipeline.Request.OnTokenId?.Invoke(next);
            }
        }
        catch (Exception ex) when (ex is KvFullException || ex is InvalidOperationException)
        {
            // The batched forward reserves every sequence's slot before writing any layer, so which appends landed is
            // unknowable: fail the whole batch rather than re-forward sequences onto a second slot. Failed-token
            // accounting rides on each Finish.
            for (int i = start; i < start + count; i++)
            {
                Finish(window[i], FinishReason.Failed, ex);
            }
        }
    }

    /// <summary>
    /// Publishes the pipeline's outcome and releases its KV blocks and advancement slot. The generated ids are copied out of the
    /// pipeline first, so nothing downstream touches the cache after this point; disposing is idempotent, which makes a second finish
    /// (for example an eos in the sampling loop followed by a batch fault) harmless, and only the finish that held a slot returns one.
    /// Releasing covers adopted shared blocks too: they return to their remaining owners (the prefix cache entry or another pipeline),
    /// and only reach the free list when the last reference goes. Blocks go back before the completion is published: an observer woken by
    /// the task may immediately submit a new pipeline that needs them, and a freed slot lets this boundary's waiting pipelines start at
    /// the next one. The admission window closes with the same handover — from here on, only the current soft cap governs pool
    /// allocations for later admissions.
    /// </summary>
    private void Finish(RequestPipeline pipeline, FinishReason reason, Exception? error = null)
    {
        if (reason == FinishReason.Failed)
        {
            Stats.FailedSequenceTokens += pipeline.Generated.Count;
        }

        SequenceOutcome outcome = new([.. pipeline.Generated], pipeline.Request.PromptTokens.Length, pipeline.Generated.Count, reason, error, pipeline.PrefillCachedTokens, [.. pipeline.LogProbEntries]);

        pipeline.Kv.Dispose();

        if (pipeline.HoldsSlot)
        {
            pipeline.HoldsSlot = false;
            _progress.ReleaseSlot();
        }

        if (pipeline.AdmissionWindowOpened)
        {
            pipeline.AdmissionWindowOpened = false;
            _pool.EndAdmissionWindow();
        }

        // The terminal stage lands before the completion is published, so a finished pipeline is never selectable for step work.
        pipeline.MarkFinished();

        pipeline.Outcome.TrySetResult(outcome);

        switch (reason)
        {
            case FinishReason.Eos:
                Stats.FinishedEos++;
                break;
            case FinishReason.Length:
                Stats.FinishedLength++;
                break;
            case FinishReason.Canceled:
                Stats.FinishedCanceled++;
                break;
            default:
                Stats.FinishedFailed++;
                break;
        }
    }
}
