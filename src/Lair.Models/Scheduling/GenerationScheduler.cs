using System.Threading.Channels;
using Lair.Kv;
using Lair.Models.Engine;

namespace Lair.Models.Scheduling;

/// <summary>
/// Continuous-batching scheduler over one forward engine and one paged KV pool. Every admitted sequence
/// advances in shared decode steps (a single batched forward per step for all decoding sequences), while at
/// most one prefill chunk per step interleaves with them — so a long prompt never blocks short sequences.
/// Sequences run in a fixed order (ascending handle id) on a dedicated scheduler thread, each with its own
/// sampler, so a sequence's token stream is identical to running it alone against the same engine.
/// Admission counts the blocks a sequence will cross over its whole lifetime, so an admitted sequence never
/// trips <see cref="KvFullException"/> mid-run; a submission that does not fit fails immediately instead.
/// </summary>
public sealed class GenerationScheduler : IDisposable
{
    private readonly TransformerModel _model;
    private readonly BlockPool _pool;
    private readonly int _maxBatchTokens;

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
    public GenerationScheduler(TransformerModel model, BlockPool pool, int maxBatchTokens)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(pool);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxBatchTokens, 1);

        _model = model;
        _pool = pool;
        _maxBatchTokens = maxBatchTokens;
        _inbox = Channel.CreateUnbounded<Sequence>();

        _thread = Task.Factory.StartNew(
            RunLoop,
            CancellationToken.None,
            TaskCreationOptions.LongRunning,
            TaskScheduler.Default);
    }

    /// <summary>
    /// Admits one sequence and returns its handle; the sequence starts prefilling at the next step boundary.
    /// The submission is rejected while the pool's free blocks cannot back every block the sequence will
    /// cross, so an admitted sequence never trips <see cref="KvFullException"/> mid-run.
    /// </summary>
    /// <exception cref="SchedulerFullException">
    /// The prompt plus the token budget needs more blocks than the pool currently has free.
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

        if (BlocksNeeded(request.PromptTokens.Length, request.MaxTokens) > _pool.FreeBlocks)
        {
            throw new SchedulerFullException(_pool.FreeSlots);
        }

        Sequence sequence = new(request, ct, _pool, Interlocked.Increment(ref _nextId));
        if (!_inbox.Writer.TryWrite(sequence))
        {
            throw new ObjectDisposedException(nameof(GenerationScheduler));
        }

        return sequence.Handle;
    }

    /// <summary>
    /// Blocks a sequence of <paramref name="promptTokens"/> plus <paramref name="maxTokens"/> generated tokens
    /// crosses over its lifetime: the block containing the first prompt token through the block containing the
    /// last token the sequence will ever forward (the final sampled token is never forwarded).
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
                while (_inbox.Reader.TryRead(out Sequence? admitted))
                {
                    _active.Add(admitted);
                }

                if (_active.Count == 0)
                {
                    // Idle: park on the inbox signal — no busy-spin — until the next submission or Dispose.
                    if (!_inbox.Reader.WaitToReadAsync().AsTask().GetAwaiter().GetResult())
                    {
                        return; // channel completed: the scheduler is disposed
                    }

                    continue;
                }

                Step();
            }
        }
        finally
        {
            // Drain whatever is queued and release every sequence's blocks, including submissions that
            // never made it into a step.
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

        // At most one prefill chunk per step, oldest first: decode sequences always run. A sequence finished
        // at this boundary (cancellation) is skipped so no forward ever touches its released blocks.
        foreach (Sequence sequence in _active)
        {
            if (sequence.IsPrefilling && !sequence.IsFinished)
            {
                PrefillChunk(sequence);
                break;
            }
        }

        // Decode step: every decoding sequence advances in one shared batched forward. Sequences whose next
        // append would cross the pool's block limit or the engine's context limit are failed before the batch
        // is issued, so the batched forward never fails halfway through and only the doomed sequences fail.
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

            // Prefill left the final prompt token's logits: sample the first generated token from them,
            // exactly as the single-flight loop did. It is reported now but forwarded on the next step.
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
    /// Advances <paramref name="count"/> sequences from <paramref name="window"/> by one token each in one
    /// batched forward, splitting the window into engine-sized batches, then samples every sequence from its
    /// own logits row. The caller has already accounted for every append this step performs, so a forward that
    /// still trips the pool or context limit is a fault: every sequence in the affected batch fails and the
    /// scheduler loop keeps running.
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
            // DecodeBatch reserves each sequence's next slot before forwarding, so a sequence whose append
            // crosses the pool's limit surfaces here — as the only failure point of the step.
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
            // The batched forward reserves every sequence's slot before writing any layer, so which appends
            // landed is unknowable: fail the whole batch rather than re-forward sequences onto a second slot.
            for (int i = start; i < start + count; i++)
            {
                Finish(window[i], FinishReason.Failed, ex);
            }
        }
    }

    /// <summary>
    /// Publishes the sequence's outcome and releases its KV blocks. The generated ids are copied out of the
    /// sequence first, so nothing downstream touches the cache after this point; disposing is idempotent, which
    /// makes a second finish (for example an eos in the sampling loop followed by a batch fault) harmless.
    /// </summary>
    private static void Finish(Sequence sequence, FinishReason reason, Exception? error = null)
    {
        SequenceOutcome outcome = new([.. sequence.Generated], sequence.Request.PromptTokens.Length, sequence.Generated.Count, reason, error);

        // Blocks go back before the completion is published: an observer woken by the task may immediately
        // submit a new sequence that needs them.
        sequence.Kv.Dispose();
        sequence.Outcome.TrySetResult(outcome);
    }
}
