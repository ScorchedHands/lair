using System.Threading.Channels;
using Lair.Kv;
using Lair.Models.Engine;

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
    private readonly int _maxBatchTokens;

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

        lock (_gate)
        {
            int needed = BlocksNeeded(request.PromptTokens.Length, request.MaxTokens);
            if (needed > _pool.FreeBlocks && _cache is not null)
            {
                // Admission pressure: the pool cannot back this sequence as things stand — reclaim cached prefix
                // blocks for it, oldest first. Active sequences' allocations are never touched by eviction.
                _cache.EvictBlocks(needed - _pool.FreeBlocks);
            }

            if (needed > _pool.FreeBlocks)
            {
                throw new SchedulerFullException(_pool.FreeSlots);
            }
        }

        Sequence sequence = new(request, ct, _pool, Interlocked.Increment(ref _nextId));
        if (!_inbox.Writer.TryWrite(sequence))
        {
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
        // the batched forward never fails halfway through and only the doomed sequences fail. The free-block count is
        // sampled after this step's prefill (and any copy-on-write it performed), since those are the last pool
        // changes visible to window accounting.
        _window.Clear();
        int blocksLeft = _pool.FreeBlocks;
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
                        $"KV block pool is full: the sequence at position {sequence.Kv.Length} needs a new block " +
                        $"and all {_pool.CapacityBlocks} blocks are in use."));
                    continue;
                }

                blocksLeft--;
            }

            _window.Add(sequence);
        }

        if (_window.Count > 0)
        {
            DecodeWindow(_window, 0, _window.Count);
        }

        for (int i = _active.Count - 1; i >= 0; i--)
        {
            if (_active[i].IsFinished)
            {
                _active.RemoveAt(i);
            }
        }
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

            if (sequence.IsPrefilling)
            {
                return; // more chunks remain; the next step forwards the next slice
            }

            if (_cache is not null)
            {
                // Prefill done: every full prompt-prefix block now has its KV fully written, so publish them in chain
                // order. Blocks already under their key (adopted by this same sequence, or published earlier by an
                // identical prefix) are only renewed; the partial final block stays out of the cache.
                _cache.Publish(prompt, sequence.Kv);
            }

            // Prefill left the final prompt token's logits: sample the first generated token from them, exactly as
            // the single-flight loop did. It is reported now but forwarded on the next step.
            int first = sequence.Sampler.Sample(_model.LastLogits());
            if (first == sequence.Request.EosId)
            {
                Finish(sequence, FinishReason.Eos);
                return;
            }

            sequence.Generated.Add(first);
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

            int vocab = _model.Desc.VocabSize;
            for (int i = 0; i < count; i++)
            {
                Sequence sequence = window[start + i];
                int next = sequence.Sampler.Sample(logits.Slice(i * vocab, vocab));

                if (next == sequence.Request.EosId)
                {
                    Finish(sequence, FinishReason.Eos); // eos ends the sequence and is never forwarded
                    continue;
                }

                sequence.Generated.Add(next);
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
            // unknowable: fail the whole batch rather than re-forward sequences onto a second slot.
            for (int i = start; i < start + count; i++)
            {
                Finish(window[i], FinishReason.Failed, ex);
            }
        }
    }

    /// <summary>
    /// Publishes the sequence's outcome and releases its KV blocks. The generated ids are copied out of the sequence
    /// first, so nothing downstream touches the cache after this point; disposing is idempotent, which makes a second
    /// finish (for example an eos in the sampling loop followed by a batch fault) harmless. Releasing covers adopted
    /// shared blocks too: they return to their remaining owners (the prefix cache entry or another sequence), and only
    /// reach the free list when the last reference goes.
    /// </summary>
    private static void Finish(Sequence sequence, FinishReason reason, Exception? error = null)
    {
        SequenceOutcome outcome = new([.. sequence.Generated], sequence.Request.PromptTokens.Length, sequence.Generated.Count, reason, error, sequence.PrefillCachedTokens);

        // Blocks go back before the completion is published: an observer woken by the task may immediately submit a
        // new sequence that needs them.
        sequence.Kv.Dispose();
        sequence.Outcome.TrySetResult(outcome);
    }
}
