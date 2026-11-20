namespace Lair.Models.Scheduling;

/// <summary>
/// Control-plane accounting over how many pipelines a scheduler holds admitted at once — a per-engine concurrency cap, taken by one
/// slot per submitted pipeline and returned when that pipeline finishes. It is the seam between admission (which pipelines exist) and
/// step composition (which batch a pipeline's math rides in): <c>Submit</c> takes this gate's slot for every pipeline it admits, so a
/// submission arriving while every slot is held is refused naming its model's quota rather than queued behind what already runs. The
/// cap never preempts or disturbs a pipeline already admitted: lowering the value stops new admissions from the next submission on
/// and lets work in flight keep its slot until it finishes, and raising it admits more pipelines from that point on. Composition
/// deferrals — candidates kept out of one merged step by a segment's batch cap or quota headroom — are decided elsewhere and never
/// consult this accounting: a deferred pipeline still holds its admission slot and rides a later boundary.
/// The gate also answers the question a waiting submission asks instead of refusing it: <see cref="WaitForFreeSlotAsync"/> resolves when
/// room for another admission appears, so a caller that bounds its own wait can block on capacity instead of being refused at once — while
/// this gate still reserves nothing for anyone who has not taken a slot. A waiter woken here re-enters admission under the scheduler's own
/// lock and is decided there exactly as a fresh submission would be, including refusals for KV capacity or budget chains.
/// </summary>
internal sealed class PipelineConcurrencyGate
{
    /// <summary>The cap value that bounds nothing: every submission is admitted as far as capacity reaches.</summary>
    public const int Uncapped = int.MaxValue;

    // Cap is volatile because a hot edit writes it from a caller thread while only the scheduler's gate consumes it, at submissions and boundaries.
    private volatile int _maxConcurrentRequests;

    // Slot bookkeeping belongs to the lock that admits pipelines (every acquire and release runs under the scheduler's gate); HeldSlots and
    // PeakHeldSlots are read as volatile snapshots by observers that never take that gate.
    private volatile int _heldSlots;
    private volatile int _peakHeldSlots;

    // Waiting-for-room bookkeeping lives on its own short lock, never held across a step or an admission: a submission whose model lets it wait
    // registers here and is handed each slot the moment a pipeline returns one, earliest registered first. The list only decides who learns that
    // room exists; admitting still happens under the scheduler's gate, so nothing here reserves capacity for a waiter.
    private readonly object _waitGate = new();
    private readonly List<TaskCompletionSource<bool>> _slotWaiters = [];

    public PipelineConcurrencyGate(int maxConcurrentRequests)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxConcurrentRequests, 1);
        _maxConcurrentRequests = maxConcurrentRequests;
    }

    /// <summary>The cap currently in force: the largest number of pipelines this gate will hand slots to at one time.</summary>
    public int MaxConcurrentRequests => _maxConcurrentRequests;

    /// <summary>Slots held right now — one per pipeline admitted under this gate and not yet finished.</summary>
    public int HeldSlots => _heldSlots;

    /// <summary>Highest concurrent slot count this gate has ever handed out, so a scheduler that never advanced two pipelines at once is distinguishable from one that could not.</summary>
    public int PeakHeldSlots => _peakHeldSlots;

    /// <summary>Submissions this gate currently has waiting for room — the depth of its wait list, read as a volatile-consistent snapshot by observers that never take either lock.</summary>
    public int WaitingForSlotCount
    {
        get
        {
            lock (_waitGate)
            {
                return _slotWaiters.Count;
            }
        }
    }

    /// <summary>
    /// Takes the admission slot for the pipeline about to be submitted. Returns false when every slot is already held — the caller
    /// refuses that submission naming its model's quota rather than queueing it, so a submission never blocks on capacity it cannot free.
    /// </summary>
    public bool TryAcquireSlot()
    {
        if (_heldSlots >= MaxConcurrentRequests)
        {
            return false;
        }

        _heldSlots++;
        if (_heldSlots > _peakHeldSlots)
        {
            _peakHeldSlots = _heldSlots;
        }

        return true;
    }

    /// <summary>Returns the slot a finishing pipeline held. Releasing a slot that was never taken is a fault, not a no-op: it would let one pipeline hand out two.</summary>
    /// <exception cref="InvalidOperationException">No slot was held to release.</exception>
    public void ReleaseSlot()
    {
        if (_heldSlots == 0)
        {
            throw new InvalidOperationException("no advancement slot is held, so there is nothing to release.");
        }

        _heldSlots--;

        // The room this release made goes to the earliest submission still waiting for it. Waking that waiter reserves nothing: it re-enters
        // admission under the scheduler's gate and can still be refused there by capacity or budget, exactly as any fresh submission would.
        HandOffFreedSlot();
    }

    /// <summary>
    /// Applies a new cap. It takes effect from the next submission on: pipelines already admitted keep their slots and finish normally,
    /// so lowering never preempts work in flight — it stops further admissions while held slots remain at or above the new value, and an
    /// over-cap pipeline waiting for nothing is refused instead of queued. Raising admits more submissions from that point on, and every
    /// submission already waiting learns about the widened cap immediately rather than after its own wait runs out.
    /// </summary>
    public void ApplyMaxConcurrentRequests(int maxConcurrentRequests)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxConcurrentRequests, 1);

        // Publish first: a waiter woken by this reads the new cap when it re-enters admission, so a widened cap cannot be seen as still full.
        _maxConcurrentRequests = maxConcurrentRequests;

        while (TryDequeueLiveWaiter(requireRoom: true) is TaskCompletionSource<bool> waiting)
        {
            waiting.TrySetResult(true); // one wake-up per slot the new cap actually opened.
        }
    }

    /// <summary>
    /// Resolves true when this gate has just made room for another admission — a pipeline returned its slot or a cap raise widened the cap —
    /// and false when the wait ran out, was cancelled, or the gate closed. A resolved-true waiter holds nothing: it still competes for the
    /// slot under the scheduler's own admission lock, where capacity and budget checks decide as they do for any fresh submission. With no
    /// waiter registered ahead of this one and a slot already free, the answer comes back immediately instead of waiting to be handed room
    /// that is being offered nowhere. The wait keeps its deadline live until the wait itself finishes, so an unheld slot never leaves a caller
    /// blocked past its own budget.
    /// </summary>
    public async Task<bool> WaitForFreeSlotAsync(TimeSpan budget, CancellationToken ct)
    {
        if (ct.IsCancellationRequested || (budget <= TimeSpan.Zero && budget != System.Threading.Timeout.InfiniteTimeSpan))
        {
            return false; // nothing left to wait on: the caller answers with whatever its own deadline or cancellation means.
        }

        lock (_waitGate)
        {
            if (_slotWaiters.Count == 0 && _heldSlots < _maxConcurrentRequests)
            {
                return true; // room already stands here and nobody is ahead of this submission in line for it.
            }
        }

        TaskCompletionSource<bool> waiter = new(TaskCreationOptions.RunContinuationsAsynchronously);
        lock (_waitGate)
        {
            _slotWaiters.Add(waiter);
        }

        // The deadline and its registration outlive the await below — disposing them earlier would disarm the very timeout this wait is bounded by.
        using CancellationTokenSource deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        if (budget != System.Threading.Timeout.InfiniteTimeSpan)
        {
            deadline.CancelAfter(budget);
        }

        using CancellationTokenRegistration registration = deadline.Token.Register(static state => ((TaskCompletionSource<bool>)state!).TrySetResult(false), waiter);

        bool handedRoom;
        try
        {
            handedRoom = await waiter.Task.ConfigureAwait(false);
        }
        finally
        {
            // An expired or cancelled wait leaves the list here rather than sitting in it until some later slot hands its wake-up to a caller that already answered.
            lock (_waitGate)
            {
                int index = _slotWaiters.IndexOf(waiter);
                if (index >= 0)
                {
                    _slotWaiters.RemoveAt(index);
                }
            }

            waiter.TrySetResult(false); // a wait that left the list without an answer is closed as no-room.
        }

        return handedRoom;
    }

    /// <summary>Closes every outstanding wait with a false answer: the scheduler is going away, so nothing it admitted can still free a slot for a submission that was waiting. Safe to call twice.</summary>
    public void CloseWaiters()
    {
        while (TryDequeueLiveWaiter(requireRoom: false) is TaskCompletionSource<bool> waiting)
        {
            waiting.TrySetResult(false);
        }
    }

    private void HandOffFreedSlot()
    {
        if (TryDequeueLiveWaiter(requireRoom: true) is TaskCompletionSource<bool> waiting)
        {
            waiting.TrySetResult(true);
        }
    }

    // Takes the earliest wait still owed an answer, retiring any entry that already has one (an expired or cancelled caller whose cleanup lost
    // this race) so a freed slot is never spent waking nobody. requireRoom says whether the current cap actually leaves room to offer.
    private TaskCompletionSource<bool>? TryDequeueLiveWaiter(bool requireRoom)
    {
        lock (_waitGate)
        {
            while (_slotWaiters.Count > 0 && _slotWaiters[0].Task.IsCompleted)
            {
                _slotWaiters.RemoveAt(0); // answered already: its own cleanup will find nothing here.
            }

            if (_slotWaiters.Count == 0 || (requireRoom && _heldSlots >= _maxConcurrentRequests))
            {
                return null;
            }

            TaskCompletionSource<bool> waiting = _slotWaiters[0];
            _slotWaiters.RemoveAt(0);
            return waiting;
        }
    }
}
