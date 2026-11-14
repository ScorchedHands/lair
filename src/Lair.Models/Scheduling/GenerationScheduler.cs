using System.Diagnostics;
using System.Threading.Channels;
using Lair.Kv;
using Lair.Models.Engine;
using Lair.Sampling;

namespace Lair.Models.Scheduling;

/// <summary>
/// Continuous-batching scheduler over one forward engine and one paged KV pool. Each admitted request is a
/// <see cref="RequestPipeline"/> — its own state machine (admit → transform → prefill → decode → finish), sampler, KV cache, and
/// admission reservation — advanced on this one scheduler thread by the control plane here, while the step executor decides how its
/// math runs: every advancing pipeline takes part in shared decode steps (a single batched forward per step for all decoding
/// pipelines), with at most one prefill chunk interleaved into that boundary — so a long prompt never blocks short pipelines. A
/// pipeline's progress is decided against <see cref="PipelineConcurrencyGate"/> slots, independently of which batch its math rides in:
/// an over-cap pipeline waits admitted and idle until a slot frees, and a merged step hands each pipeline the logits row an unmerged
/// step would have produced for it, so a pipeline's token stream is identical whether it advanced alone or inside a merged window.
/// Pipelines run in a fixed order (ascending id) on this thread, each with its own sampler. Admission counts the blocks a pipeline will
/// cross over its whole lifetime, so an admitted pipeline never trips <see cref="KvFullException"/> mid-run; when the free list runs
/// short and a prefix cache is attached, admission first reclaims cached prompt-prefix blocks (least-recently-used first, never
/// disturbing active pipelines) before rejecting. Admitted pipelines adopt leading cached prompt-prefix blocks instead of prefilling
/// them, and completed full prompt blocks are published back into the cache for later requests; an adopted block produces bit-identical
/// KV bytes to recomputing that prefix, so shared prefixes change speed, not results.
/// </summary>
public sealed class GenerationScheduler : IDisposable
{
    private readonly TransformerModel _model;
    private readonly BlockPool _pool;
    private readonly PrefixCache? _cache;

    // Prefill chunk width, re-read at every chunk issue so a hot edit lands at the next step boundary: a pipeline already
    // mid-chunk finishes that chunk at the old width. Volatile because the apply path writes it from a caller thread while only the
    // scheduler thread consumes it.
    private volatile int _maxBatchTokens;

    // Progress authority over which admitted pipelines may advance engine work, decided per boundary and independent of step
    // composition: pipeline slots here bound concurrency, KV capacity bounds admission, and neither consults the other's bookkeeping.
    private readonly PipelineConcurrencyGate _progress;

    // Serializes every mutation of the pool and cache that is not issued from the scheduler thread: admission-time
    // eviction runs on caller threads, while all forwards, adoptions, publications, releases, slot handovers, and shutdown finishes
    // run under this gate. Holding it across a whole step keeps those paths disjoint without finer-grained locking;
    // submissions only ever wait for one step boundary.
    private readonly object _gate = new();

    // Submissions arrive here from caller threads; only the scheduler thread reads it. Unbounded because
    // admission is gated on KV capacity, not on queue size.
    private readonly Channel<RequestPipeline> _inbox;

    // Scheduler-thread-only working set of live pipelines, kept in ascending pipeline id: queued and prefilling ones first, then
    // decoding ones. A pipeline sits here from admission until its outcome is published, including while it waits for a slot.
    private readonly List<RequestPipeline> _active = [];

    // Scratch lists for the current decode window, cleared (never reallocated) per window.
    private readonly List<RequestPipeline> _window = [];
    private readonly List<int> _windowTokens = [];
    private readonly List<SequenceKv> _windowKvs = [];

    // Step-boundary engine counters, written only on the scheduler thread while the gate is held and read as
    // volatile longs by observers — collection costs a handful of increments per step, never per token.
    private readonly Stopwatch _stepClock = new();

    /// <summary>Cumulative engine counters maintained live by the step loop; see <see cref="SchedulerStats"/>.</summary>
    public SchedulerStats Stats { get; } = new();

    private readonly Task _thread;
    private long _nextId;
    private bool _disposed;

    /// <param name="model">The forward engine shared by every pipeline.</param>
    /// <param name="pool">The paged KV pool every pipeline appends into.</param>
    /// <param name="maxBatchTokens">Largest prefill chunk forwarded in one step; must be at least 1.</param>
    /// <param name="cache">Optional prefix cache over the same pool: enables prompt-prefix adoption and publication. Without it, scheduling behaves exactly as without caching.</param>
    /// <param name="maxConcurrentRequests">How many pipelines may advance engine work at once; must be at least 1, or <see cref="PipelineConcurrencyGate.Uncapped"/> to bound nothing. It caps advancement only: an over-cap submission is admitted and waits for a slot rather than being rejected.</param>
    public GenerationScheduler(TransformerModel model, BlockPool pool, int maxBatchTokens, PrefixCache? cache = null, int maxConcurrentRequests = PipelineConcurrencyGate.Uncapped)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(pool);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxBatchTokens, 1);

        _model = model;
        _pool = pool;
        _cache = cache;
        _maxBatchTokens = maxBatchTokens;
        _progress = new PipelineConcurrencyGate(maxConcurrentRequests);
        _inbox = Channel.CreateUnbounded<RequestPipeline>();

        _thread = Task.Factory.StartNew(
            RunLoop,
            CancellationToken.None,
            TaskCreationOptions.LongRunning,
            TaskScheduler.Default);
    }

    /// <summary>How many pipelines this scheduler lets advance engine work at once — the cap its progress gate enforces.</summary>
    public int MaxConcurrentRequests => _progress.MaxConcurrentRequests;

    /// <summary>Pipelines holding an advancement slot right now, i.e. the ones eligible for step work at the next boundary.</summary>
    public int AdvancingPipelineCount => _progress.HeldSlots;

    /// <summary>Highest concurrent pipeline count this scheduler has ever advanced in one window of boundaries — zero only when nothing ran.</summary>
    public int PeakAdvancingPipelines => _progress.PeakHeldSlots;

    /// <summary>
    /// Admits one request as a pipeline and returns its handle: the pipeline is admitted to the working set at the next step boundary,
    /// starts prefilling once an advancement slot is free (immediately under an uncapped cap), and never runs on more than one thread.
    /// The submission is rejected while the pool's free blocks cannot back every block the pipeline will cross — with a prefix cache
    /// attached, admission pressure first evicts least-recently-used cached blocks (only those no active pipeline still references) and
    /// rechecks capacity before failing. So an admitted pipeline never trips <see cref="KvFullException"/> mid-run, including while it
    /// waits for a slot: its whole-lifetime reservation opens with its admission.
    /// </summary>
    /// <exception cref="SchedulerFullException">
    /// The prompt plus the token budget needs more blocks than the pool currently has free, even after cache eviction.
    /// </exception>
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

        lock (_gate)
        {
            // Admission spends from the pool's effective free side: physically free blocks that sit beyond a shrunk
            // soft budget are unreachable until cached-block eviction brings usage under the cap. Evicting one cache
            // entry moves at most one block out of the used column (a block shared with an active pipeline stays in
            // use), so both checks re-read EffectiveFreeBlocks instead of assuming each evicted block counts. Concurrency is not
            // checked here: a submission over the cap is admitted and waits for a slot, because its capacity reservation is what this
            // gate can promise it either way.
            if (needed > _pool.EffectiveFreeBlocks && _cache is not null)
            {
                _cache.EvictBlocks(needed - _pool.EffectiveFreeBlocks);
            }

            if (needed > _pool.EffectiveFreeBlocks)
            {
                throw new SchedulerFullException((long)_pool.EffectiveFreeBlocks * _pool.BlockTokens);
            }
        }

        RequestPipeline pipeline = new(request, ct, _pool, Interlocked.Increment(ref _nextId));

        // The admission window carries this submission's reserved headroom across its whole run: if a budget shrink
        // lands while it is decoding (usage pushed above the cap by cache eviction that could not reclaim pinned
        // blocks), its own appends still reach what Submit already accounted for. Closed when the pipeline finishes, and held just as
        // firmly while an over-cap pipeline waits to advance — otherwise waiting would cost capacity a later submission could steal.
        _pool.BeginAdmissionWindow(needed);
        pipeline.AdmissionWindowOpened = true;

        Stats.SequencesAdmitted++;
        if (!_inbox.Writer.TryWrite(pipeline))
        {
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
    /// tokens, only where one forward stops and the next begins.
    /// </summary>
    public void ApplyMaxBatchTokens(int maxBatchTokens)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxBatchTokens, 1);
        ObjectDisposedException.ThrowIf(_disposed, typeof(GenerationScheduler));

        _maxBatchTokens = maxBatchTokens;
    }

    /// <summary>
    /// Applies a new cap on how many pipelines may advance engine work at once. It takes effect at the next step boundary: pipelines
    /// already advancing keep their slots and finish normally, so lowering the cap stops new advancement rather than preempting work in
    /// flight, and raising it releases pipelines that were waiting for a slot from that boundary on. Nothing about an in-flight
    /// pipeline's tokens changes either way — which pipelines advance together is step composition, and merging never moves a token.
    /// </summary>
    public void ApplyMaxConcurrentRequests(int maxConcurrentRequests)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxConcurrentRequests, 1);
        ObjectDisposedException.ThrowIf(_disposed, typeof(GenerationScheduler));

        _progress.ApplyMaxConcurrentRequests(maxConcurrentRequests);
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
                    // before any prefill or decode work can touch the pool for it. Adoption is not advancement — a pipeline that cannot
                    // get an advancement slot at this boundary stays here in <see cref="PipelineStage.Admitted"/>, holding its admission
                    // reservation but issuing no forward, until a slot frees.
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

        // Progress selection: pipelines already advancing keep their slots untouched here (the cap never preempts work in flight), and
        // idle admitted ones take whatever slots this boundary has free, oldest first. Selection decides who advances; nothing about it
        // says which batch a pipeline's math joins once the executor runs it.
        foreach (RequestPipeline pipeline in _active)
        {
            if (pipeline.IsFinished || pipeline.HoldsSlot)
            {
                continue;
            }

            if (!_progress.TryAcquireSlot())
            {
                continue; // cap fully held by pipelines already advancing: this one waits at admission, still holding its reservation
            }

            pipeline.HoldsSlot = true;
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
        // (cancellation) is skipped so no forward ever touches its released blocks, and an over-cap admitted pipeline is never chosen.
        foreach (RequestPipeline pipeline in _active)
        {
            if (pipeline.IsPrefilling && !pipeline.IsFinished)
            {
                PrefillChunk(pipeline);
                break;
            }
        }

        // Decode step: every advancing decoding pipeline takes part in one shared batched forward. Pipelines whose next append
        // would cross the pool's block limit or the engine's context limit are failed before the batch is issued, so
        // the batched forward never fails halfway through and only the doomed pipelines fail. The window budget uses
        // the pool's current allocation ceiling — under an open admission window that reaches past a hot-shrunk cap to what those
        // admitted pipelines already reserved — sampled after this step's prefill (and any copy-on-write it performed), since those are
        // the last pool changes visible to window accounting.
        _window.Clear();
        int blocksLeft = Math.Max(Math.Min(_pool.FreeBlocks, _pool.AllocationCeiling - _pool.UsedBlocks), 0);
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

        if (_window.Count > 0)
        {
            // Step-boundary clock: one started/stopped shared Stopwatch per step measures the forward work of this
            // step's decode windows; no allocation and nothing inside the token loop.
            _stepClock.Restart();
            DecodeWindow(_window, 0, _window.Count);
            _stepClock.Stop();
            Stats.DecodeSteps++;
            Stats.DecodeWindowMicrosTotal += _stepClock.ElapsedTicks * 1_000_000 / Stopwatch.Frequency;
            Stats.WindowSequenceSum += _window.Count;
        }

        Stats.ActiveDecodingSequences = _window.Count;

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

    private void PrefillChunk(RequestPipeline pipeline)
    {
        int[] prompt = pipeline.Request.PromptTokens;
        int chunk = Math.Min(prompt.Length - pipeline.PromptOffset, _maxBatchTokens);

        try
        {
            _model.Prefill(prompt[pipeline.PromptOffset..(pipeline.PromptOffset + chunk)], pipeline.Kv);
            pipeline.PromptOffset += chunk;
            Stats.PrefillChunksIssued++;
            Stats.PrefillTokensForwarded += chunk;

            if (pipeline.IsPrefilling)
            {
                return; // more chunks remain; the next step forwards the next slice
            }

            if (_cache is not null)
            {
                // Prefill done: every full prompt-prefix block now has its KV fully written, so publish them in chain
                // order. Blocks already under their key (adopted by this same pipeline, or published earlier by an
                // identical prefix) are only renewed; the partial final block stays out of the cache.
                int publishedBlocks = _cache.Publish(prompt, pipeline.Kv);
                Stats.CachePublishedBlocksTotal += publishedBlocks;
                Stats.CachePublishedTokensTotal += (long)publishedBlocks * _pool.BlockTokens;
            }

            // Prefill left the final prompt token's logits: this pipeline transforms and samples them on its own row, exactly as a
            // solo run does. The token is reported now but forwarded on the next step.
            pipeline.StartTransforming();
            int first = pipeline.SampleFrom(_model.LastLogits());

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
        catch (Exception ex) when (ex is KvFullException || ex is InvalidOperationException)
        {
            Finish(pipeline, FinishReason.Failed, ex);
        }
    }

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
