namespace Lair.Models.Scheduling;

/// <summary>
/// Control-plane accounting over how many pipelines may advance their math at once — a per-engine concurrency cap, held one slot
/// per advancing pipeline and released when that pipeline finishes. It is the seam between admission (which pipelines exist) and
/// step composition (which batch a pipeline's math rides in): a pipeline that cannot get a slot here simply stays admitted without
/// engine work until one frees, so reaching the cap never preempts or disturbs a pipeline already stepping. Held slots are read by
/// no admission decision: capacity limits stay where they were measured against real resources, and this gate only decides progress.
/// </summary>
internal sealed class PipelineConcurrencyGate
{
    /// <summary>The cap value that bounds nothing: every admitted pipeline advances as soon as it can.</summary>
    public const int Uncapped = int.MaxValue;

    // Cap is volatile because a hot edit writes it from a caller thread while only the scheduler thread consumes it, at boundaries.
    private volatile int _maxConcurrentRequests;

    // Slot bookkeeping belongs to the scheduler thread (every acquire and release runs under the scheduler's gate); HeldSlots and
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

    /// <summary>Slots held right now — one per pipeline advancing engine work under this gate.</summary>
    public int HeldSlots => _heldSlots;

    /// <summary>Highest concurrent slot count this gate has ever handed out, so a scheduler that never advanced two pipelines at once is distinguishable from one that could not.</summary>
    public int PeakHeldSlots => _peakHeldSlots;

    /// <summary>
    /// Takes one advancement slot for the pipeline about to start engine work. Returns false when every slot is already held — the
    /// caller leaves its pipeline admitted and idle rather than waiting here, so a boundary never blocks on capacity it cannot free.
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
    /// Applies a new cap. It takes effect at the next boundary: pipelines already holding slots keep them and finish normally, while
    /// acquisitions stop once held slots reach the new value — so lowering the cap never preempts work in flight, it only stops more
    /// from starting. Raising it releases idle admitted pipelines from the next boundary on.
    /// </summary>
    public void ApplyMaxConcurrentRequests(int maxConcurrentRequests)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxConcurrentRequests, 1);
        _maxConcurrentRequests = maxConcurrentRequests;
    }
}
