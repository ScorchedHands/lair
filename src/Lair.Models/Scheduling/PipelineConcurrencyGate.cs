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
    }

    /// <summary>
    /// Applies a new cap. It takes effect from the next submission on: pipelines already admitted keep their slots and finish normally,
    /// so lowering never preempts work in flight — it stops further admissions while held slots remain at or above the new value, and an
    /// over-cap pipeline waiting for nothing is refused instead of queued. Raising admits more submissions from that point on.
    /// </summary>
    public void ApplyMaxConcurrentRequests(int maxConcurrentRequests)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxConcurrentRequests, 1);
        _maxConcurrentRequests = maxConcurrentRequests;
    }
}
