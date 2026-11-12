using System.Diagnostics;
using System.Threading.Channels;
using Lair.Kv;
using Lair.Models.Engine;
using Lair.Sampling;

namespace Lair.Models.Scheduling;

/// <summary>
/// Continuous-batching scheduler over one forward engine and one paged KV pool. Every admitted sequence advances in
/// shared decode steps (a single batched forward per step for all decoding sequences), while at most one prefill chunk
/// per step interleaves with them — so a long prompt never blocks short sequences. Sequences run in a fixed order
/// (ascending handle id) on a dedicated scheduler thread, each with its own sampler, so a sequence's token stream is
/// identical to running it alone against the same engine. Admission counts the blocks a sequence will cross over its
/// whole lifetime, so an admitted sequence never trips <see cref="KvFullException"/> mid-run; when the free list runs
/// short and a prefix cache is attached, admission first reclaims cached prefix blocks (least-recently-used first,
/// never disturbing active sequences) before rejecting. Admitted sequences adopt leading cached prompt-prefix blocks
/// instead of prefilling them, and completed full prompt blocks are published back into the cache for later requests;
/// an adopted block produces bit-identical KV bytes to recomputing that prefix, so shared prefixes change speed, not
/// results.
/// </summary>
public sealed class GenerationScheduler : IDisposable
{
    private readonly TransformerModel _model;
    private readonly BlockPool _pool;
    private readonly PrefixCache? _cache;

    // Prefill chunk width, re-read at every chunk issue so a hot edit lands at the next step boundary: a sequence
    // already mid-chunk finishes that chunk at the old width. Volatile because the apply path writes it from a
    // caller thread while only the scheduler thread consumes it.
    private volatile int _maxBatchTokens;

    // Serializes every mutation of the pool and cache that is not issued from the scheduler thread: admission-time
    // eviction runs on caller threads, while all forwards, adoptions, publications, releases, and shutdown finishes
    // run under this gate. Holding it across a whole step keeps those paths disjoint without finer-grained locking;
    // submissions only ever wait for one step boundary.
    private readonly object _gate = new();

    // Submissions arrive here from caller threads; only the scheduler thread reads it. Unbounded because
    // admission is gated on KV capacity, not on queue size.
    private readonly Channel<Sequence> _inbox;

    // Scheduler-thread-only working set, kept in ascending sequence id: prefillers first, then decoding sequences.
    private readonly List<Sequence> _active = [];

    // Scratch lists for the current decode window, cleared (never reallocated) per window.
    private readonly List<Sequence> _window = [];
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

    /// <param name="model">The forward engine shared by every sequence.</param>
    /// <param name="pool">The paged KV pool every sequence appends into.</param>
    /// <param name="maxBatchTokens">Largest prefill chunk forwarded in one step; must be at least 1.</param>
    /// <param name="cache">Optional prefix cache over the same pool: enables prompt-prefix adoption and publication. Without it, scheduling behaves exactly as without caching.</param>
    public GenerationScheduler(TransformerModel model, BlockPool pool, int maxBatchTokens, PrefixCache? cache = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(pool);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxBatchTokens, 1);

        _model = model;
        _pool = pool;
        _cache = cache;
        _maxBatchTokens = maxBatchTokens;
        _inbox = Channel.CreateUnbounded<Sequence>();

        _thread = Task.Factory.StartNew(
            RunLoop,
            CancellationToken.None,
            TaskCreationOptions.LongRunning,
            TaskScheduler.Default);
    }

    /// <summary>
    /// Admits one sequence and returns its handle; the sequence starts prefilling at the next step boundary. The
    /// submission is rejected while the pool's free blocks cannot back every block the sequence will cross — with a
    /// prefix cache attached, admission pressure first evicts least-recently-used cached blocks (only those no active
    /// sequence still references) and rechecks capacity before failing. So an admitted sequence never trips
    /// <see cref="KvFullException"/> mid-run.
    /// </summary>
    /// <exception cref="SchedulerFullException">
    /// The prompt plus the token budget needs more blocks than the pool currently has free, even after cache eviction.
    /// </exception>
    public SequenceHandle Submit(ScheduledRequest request, CancellationToken ct)
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
            // entry moves at most one block out of the used column (a block shared with an active sequence stays in
            // use), so both checks re-read EffectiveFreeBlocks instead of assuming each evicted block counts.
            if (needed > _pool.EffectiveFreeBlocks && _cache is not null)
            {
                _cache.EvictBlocks(needed - _pool.EffectiveFreeBlocks);
            }

            if (needed > _pool.EffectiveFreeBlocks)
            {
                throw new SchedulerFullException((long)_pool.EffectiveFreeBlocks * _pool.BlockTokens);
            }
        }

        Sequence sequence = new(request, ct, _pool, Interlocked.Increment(ref _nextId));

        // The admission window carries this submission's reserved headroom across its whole run: if a budget shrink
        // lands while it is decoding (usage pushed above the cap by cache eviction that could not reclaim pinned
        // blocks), its own appends still reach what Submit already accounted for. Closed when the sequence finishes.
        _pool.BeginAdmissionWindow(needed);
        sequence.AdmissionWindowOpened = true;

        Stats.SequencesAdmitted++;
        if (!_inbox.Writer.TryWrite(sequence))
        {
            sequence.AdmissionWindowOpened = false;
            _pool.EndAdmissionWindow();
            throw new ObjectDisposedException(nameof(GenerationScheduler));
        }

        return sequence.Handle;
    }

    /// <summary>
    /// Blocks a sequence of <paramref name="promptTokens"/> plus <paramref name="maxTokens"/> generated tokens crosses
    /// over its lifetime: the block containing the first prompt token through the block containing the last token the
    /// sequence will ever forward (the final sampled token is never forwarded). Adopted prefix blocks are already in
    /// this count — they occupy exactly the positions they cover.
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
    /// the next step boundary: sequences already mid-chunk finish that chunk at the old width, and decode steps are
    /// untouched. Any in-flight sequence's token stream stays deterministic either way — chunk boundaries never move
    /// tokens, only where one forward stops and the next begins.
    /// </summary>
    public void ApplyMaxBatchTokens(int maxBatchTokens)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxBatchTokens, 1);
        ObjectDisposedException.ThrowIf(_disposed, typeof(GenerationScheduler));

        _maxBatchTokens = maxBatchTokens;
    }

    /// <summary>The outcome of applying a new KV block budget to the pool this scheduler admits against.</summary>
    public readonly record struct KvBudgetApply(int AppliedBlocks, int EvictedCacheEntries);

    /// <summary>
    /// Applies a new soft cap on KV usage within the pool's fixed physical reservation. The gate is held across the whole
    /// operation so the budget flip and the cached-block drain land as one step-boundary change: raising the cap makes
    /// already-free blocks allocatable again immediately, up to the physical reservation (the applied value reports any
    /// clamp); lowering it withholds new allocations and drains least-recently-used cache entries until in-use count sits
    /// at or below the cap — active sequences' blocks are never touched, so a mid-run sequence completes normally. The
    /// freed usage flows straight into admission: <see cref="Submit"/> budgets against effective free blocks from its
    /// next call on.
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
    /// Stops the scheduler: the step loop exits at the next boundary, every unfinished sequence finishes with
    /// <see cref="FinishReason.Canceled"/>, and every sequence's KV blocks are released. Safe to call twice.
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
                    // Drain and adoption run as one boundary: every newly admitted sequence adopts its cached prefix
                    // before any prefill or decode work can touch the pool for it.
                    while (_inbox.Reader.TryRead(out Sequence? admitted))
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
                    // not held while parked, so admission-time eviction from caller threads proceeds freely.
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
                // Drain whatever is queued and release every sequence's blocks, including submissions that never made
                // it into a step.
                while (_inbox.Reader.TryRead(out Sequence? orphaned))
                {
                    Finish(orphaned, FinishReason.Canceled);
                }

                foreach (Sequence sequence in _active)
                {
                    Finish(sequence, FinishReason.Canceled);
                }
            }
        }
    }

    private void Step()
    {
        // Ascending handle id is the deterministic step order: prefillers (oldest first) then decoding sequences.
        _active.Sort(static (a, b) => a.Id.CompareTo(b.Id));

        // Cancellation is observed at the step boundary, before any forward work for this step.
        for (int i = _active.Count - 1; i >= 0; i--)
        {
            Sequence sequence = _active[i];
            if (sequence.CancelRequested || sequence.Cancellation.IsCancellationRequested)
            {
                Finish(sequence, FinishReason.Canceled);
            }
        }

        // At most one prefill chunk per step, oldest first: decode sequences always run. A sequence finished at this
        // boundary (cancellation) is skipped so no forward ever touches its released blocks.
        foreach (Sequence sequence in _active)
        {
            if (sequence.IsPrefilling && !sequence.IsFinished)
            {
                PrefillChunk(sequence);
                break;
            }
        }

        // Decode step: every decoding sequence advances in one shared batched forward. Sequences whose next append
        // would cross the pool's block limit or the engine's context limit are failed before the batch is issued, so
        // the batched forward never fails halfway through and only the doomed sequences fail. The window budget uses
        // the pool's current allocation ceiling — under an open admission window that reaches past a hot-shrunk cap to
        // what those admitted sequences already reserved — sampled after this step's prefill (and any copy-on-write it
        // performed), since those are the last pool changes visible to window accounting.
        _window.Clear();
        int blocksLeft = Math.Max(Math.Min(_pool.FreeBlocks, _pool.AllocationCeiling - _pool.UsedBlocks), 0);
        foreach (Sequence sequence in _active)
        {
            if (!sequence.IsDecoding || sequence.IsFinished)
            {
                continue;
            }

            if (sequence.Kv.Length >= _model.MaxContextTokens)
            {
                Finish(sequence, FinishReason.Failed, new InvalidOperationException(
                    $"sequence position {sequence.Kv.Length} exceeds this model's max context of {_model.MaxContextTokens} tokens."));
                continue;
            }

            if (sequence.Kv.Length % _pool.BlockTokens == 0)
            {
                if (blocksLeft == 0)
                {
                    Finish(sequence, FinishReason.Failed, new KvFullException(
                        $"KV block pool cannot extend this sequence: the sequence at position {sequence.Kv.Length} " +
                        $"needs a new block and only {_pool.EffectiveFreeBlocks} blocks are allocatable under the " +
                        $"{_pool.EffectiveBudgetBlocks}-block budget on a pool of {_pool.CapacityBlocks}."));
                    continue;
                }

                blocksLeft--;
            }

            _window.Add(sequence);
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
    /// Adopts a freshly admitted sequence's cached prompt prefix, when a cache is attached: the scheduler walks full
    /// leading blocks while their rolling token-prefix keys match published entries and stored-token verification
    /// passes, shares those physical blocks into the sequence (read-only), and leaves its prefill starting at the
    /// first unmatched block. A partial tail block is never adopted — it must be forwarded fresh so the request gets
    /// its own logits path. The skipped-prefill token count lands on the sequence for its outcome report.
    /// </summary>
    private void AdoptPrefix(Sequence sequence)
    {
        if (_cache is null || sequence.Request.PromptTokens.Length < _pool.BlockTokens)
        {
            return; // a prompt shorter than one block can never share a cached prefix block
        }

        int adoptedBlocks = _cache.TryAdopt(sequence.Request.PromptTokens, sequence.Kv);
        if (adoptedBlocks > 0)
        {
            // The adopted blocks cover the prompt's leading positions: prefill resumes after them, and the skipped
            // token count rides along on the outcome as this sequence's cache-hit report.
            int skipped = adoptedBlocks * _pool.BlockTokens;
            sequence.PromptOffset = skipped;
            sequence.PrefillCachedTokens = skipped;
            Stats.CacheHitSequences++;
            Stats.CacheSkippedTokens += skipped;
        }
    }

    private void PrefillChunk(Sequence sequence)
    {
        int[] prompt = sequence.Request.PromptTokens;
        int chunk = Math.Min(prompt.Length - sequence.PromptOffset, _maxBatchTokens);

        try
        {
            _model.Prefill(prompt[sequence.PromptOffset..(sequence.PromptOffset + chunk)], sequence.Kv);
            sequence.PromptOffset += chunk;
            Stats.PrefillChunksIssued++;
            Stats.PrefillTokensForwarded += chunk;

            if (sequence.IsPrefilling)
            {
                return; // more chunks remain; the next step forwards the next slice
            }

            if (_cache is not null)
            {
                // Prefill done: every full prompt-prefix block now has its KV fully written, so publish them in chain
                // order. Blocks already under their key (adopted by this same sequence, or published earlier by an
                // identical prefix) are only renewed; the partial final block stays out of the cache.
                int publishedBlocks = _cache.Publish(prompt, sequence.Kv);
                Stats.CachePublishedBlocksTotal += publishedBlocks;
                Stats.CachePublishedTokensTotal += (long)publishedBlocks * _pool.BlockTokens;
            }

            // Prefill left the final prompt token's logits: sample the first generated token from them, exactly as
            // the single-flight loop did. It is reported now but forwarded on the next step.
            int first = SampleForSequence(sequence, _model.LastLogits());
            if (first == sequence.Request.EosId)
            {
                Finish(sequence, FinishReason.Eos);
                return;
            }

            sequence.Generated.Add(first);
            Stats.PrefillSampledTokens++;
            sequence.PendingToken = first;
            sequence.Request.OnTokenId?.Invoke(first);
        }
        catch (Exception ex) when (ex is KvFullException || ex is InvalidOperationException)
        {
            Finish(sequence, FinishReason.Failed, ex);
        }
    }

    /// <summary>
    /// Advances <paramref name="count"/> sequences from <paramref name="window"/> by one token each in one batched
    /// forward, splitting the window into engine-sized batches, then samples every sequence from its own logits row.
    /// The caller has already accounted for every append this step performs, so a forward that still trips the pool or
    /// context limit is a fault: every sequence in the affected batch fails and the scheduler loop keeps running.
    /// </summary>
    private void DecodeWindow(List<Sequence> window, int start, int count)
    {
        for (int offset = 0; offset < count; offset += TransformerModel.MaxBatchSequences)
        {
            int batch = Math.Min(TransformerModel.MaxBatchSequences, count - offset);
            DecodeBatchOf(window, start + offset, batch);
        }
    }

    private void DecodeBatchOf(List<Sequence> window, int start, int count)
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
            // DecodeBatch reserves each sequence's next slot before forwarding, so a sequence whose append crosses the
            // pool's limit surfaces here — as the only failure point of the step.
            Span<float> logits = _model.DecodeBatch(_windowTokens, _windowKvs);

            int vocab = _model.Plan.VocabSize;
            for (int i = 0; i < count; i++)
            {
                Sequence sequence = window[start + i];
                int next = SampleForSequence(sequence, logits.Slice(i * vocab, vocab));

                if (next == sequence.Request.EosId)
                {
                    Finish(sequence, FinishReason.Eos); // eos ends the sequence and is never forwarded
                    continue;
                }

                sequence.Generated.Add(next);
                Stats.DecodeTokensAdvanced++;
                if (!sequence.HasBudget)
                {
                    Finish(sequence, FinishReason.Length);
                    continue;
                }

                sequence.PendingToken = next;
                sequence.Request.OnTokenId?.Invoke(next);
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
    /// Samples one sequence's next token from its own logits row. When the sequence's sampling params carry
    /// penalties or a logit bias, the row is transformed in place first — penalties over that sequence's prompt
    /// plus already-generated history, biases added to their tokens' logits — so each sequence's transform stays
    /// on its own row and the sampler pipeline itself sees only the adjusted logits. When the request asked for
    /// logprobs, the chosen token's entry is captured from the transformed row; the eos token ends a sequence
    /// without being generated, so it gets no entry.
    /// </summary>
    private static int SampleForSequence(Sequence sequence, Span<float> row)
    {
        SamplingParams p = sequence.Request.Sampling;
        if (p.HasLogitTransform)
        {
            SamplingTransforms.Apply(row, p, sequence.Request.PromptTokens, sequence.Generated);
        }

        int token = sequence.Sampler.Sample(row);
        if (sequence.Request.LogProbTopN is int topN && token != sequence.Request.EosId)
        {
            (double logProb, IReadOnlyList<(int TokenId, double LogProb)> top) = sequence.Sampler.LogProbsFor(row, token, topN);
            sequence.LogProbEntries.Add(new TokenLogprob(token, logProb, top));
        }

        return token;
    }

    /// <summary>
    /// Publishes the sequence's outcome and releases its KV blocks. The generated ids are copied out of the sequence
    /// first, so nothing downstream touches the cache after this point; disposing is idempotent, which makes a second
    /// finish (for example an eos in the sampling loop followed by a batch fault) harmless. Releasing covers adopted
    /// shared blocks too: they return to their remaining owners (the prefix cache entry or another sequence), and only
    /// reach the free list when the last reference goes.
    /// </summary>
    private void Finish(Sequence sequence, FinishReason reason, Exception? error = null)
    {
        if (reason == FinishReason.Failed)
        {
            Stats.FailedSequenceTokens += sequence.Generated.Count;
        }

        SequenceOutcome outcome = new([.. sequence.Generated], sequence.Request.PromptTokens.Length, sequence.Generated.Count, reason, error, sequence.PrefillCachedTokens, [.. sequence.LogProbEntries]);

        // Blocks go back before the completion is published: an observer woken by the task may immediately submit a
        // new sequence that needs them. The admission window closes with the same handover — from here on, only the
        // current soft cap governs pool allocations for later admissions.
        sequence.Kv.Dispose();

        if (sequence.AdmissionWindowOpened)
        {
            sequence.AdmissionWindowOpened = false;
            _pool.EndAdmissionWindow();
        }

        sequence.Outcome.TrySetResult(outcome);

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
