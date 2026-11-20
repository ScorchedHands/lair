namespace Lair.Models.Scheduling;

/// <summary>
/// The host seam that co-schedules every served model resident on one accelerator: each model advances its own scheduler, and this seam
/// bounds how much device work each one may launch inside a shared co-scheduling window. The window is the set of concurrent device step
/// slots the seam admits for a model at once — a model declaring a compute share may hold at most its share of the window's slots, floor
/// one slot so a share can never round a resident model out of the device entirely, and a model resident without a declared share is
/// unbounded by fair-share (only its own segment batch caps and quotas bound it, exactly as before this seam existed). A grant decides
/// which candidates a scheduler may launch at its next boundary; nothing here cancels, aborts, or displaces a step already running — a
/// model that already holds slots keeps them until its forwards complete and release them, so deferral is a launch decision taken before
/// a step starts, never a preemption of one in flight. Grants and releases are exact: one slot per launched sequence, returned when that
/// sequence's forward has finished, so a window's occupancy is what the device is actually doing rather than a guess.
///
/// One seam per shared device, shared by every model resident on it — the same programmatic-seam posture the host device ledger and the
/// compute accounting use: a host opens one and hands it to each engine opened against that device. It claims nothing against any capacity
/// ledger, opens no backend, allocates no buffer, and changes no token a pipeline produces: overlap changes when kernels run, not whose
/// logits a step answers.
/// </summary>
public sealed class DeviceCoScheduler
{
    /// <summary>The concurrent device step slots one co-scheduling window carries; a model's fair-share cap is its share of this many slots.</summary>
    public const int WindowStepSlots = 8;

    private readonly string _deviceId;
    private readonly object _gate = new();
    private readonly Dictionary<string, ShareEntry> _resident = [];

    private sealed class ShareEntry
    {
        public int? SharePercent; // null: resident on this device with no declared share — bounded by its own caps alone, today's posture.

        public int OutstandingSlots; // sequences this model has launched whose forwards have not returned yet.
    }

    public DeviceCoScheduler(string deviceId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(deviceId);
        _deviceId = deviceId;
    }

    /// <summary>The device this seam co-schedules: the id every model registered against it advances device work on.</summary>
    public string DeviceId => _deviceId;

    /// <summary>
    /// Registers a model as resident on this device with its declared fair-share percent (1 through 100). A model registered with a share
    /// may hold at most that share of the window's step slots with launched-but-unfinished device work; a model registered without one
    /// participates in the window unbounded by fair-share, which is exactly how every model advanced before this seam existed. Registering
    /// the same model twice updates its share instead of opening a second seat.
    /// </summary>
    public void RegisterModel(string modelName, int? computeSharePercent)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelName);
        if (computeSharePercent is < 1 or > 100)
        {
            throw new ArgumentOutOfRangeException(
                nameof(computeSharePercent),
                $"a model's compute share must be an integer from 1 to 100 — the percent of one co-scheduling window ({WindowStepSlots} step slot(s)) " +
                $"this model may hold against device '{_deviceId}' — or null to declare no share.");
        }

        lock (_gate)
        {
            if (!_resident.TryGetValue(modelName, out ShareEntry? entry))
            {
                entry = new ShareEntry { SharePercent = computeSharePercent };
                _resident.Add(modelName, entry);
                return;
            }

            entry.SharePercent = computeSharePercent;
        }
    }

    /// <summary>
    /// Re-resolves a resident model's fair-share to a new value (or null to declare no share again). It takes effect from the next grant
    /// on: a raised cap lets later boundaries launch more, a lowered cap retargets which candidates a later composition may launch, and
    /// sequences already launched keep their slots until their forwards return — a live re-share never preempts device work in flight.
    /// Returns false when the model is not resident on this device.
    /// </summary>
    public bool UpdateComputeShare(string modelName, int? computeSharePercent)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelName);
        if (computeSharePercent is < 1 or > 100)
        {
            throw new ArgumentOutOfRangeException(
                nameof(computeSharePercent),
                $"a model's compute share must be an integer from 1 to 100 — the percent of one co-scheduling window ({WindowStepSlots} step slot(s)) " +
                $"this model may hold against device '{_deviceId}' — or null to declare no share.");
        }

        lock (_gate)
        {
            if (!_resident.TryGetValue(modelName, out ShareEntry? entry))
            {
                return false;
            }

            entry.SharePercent = computeSharePercent;
            return true;
        }
    }

    /// <summary>
    /// Releases a model's seat on this device: it launches no further work through this seam. Slots its in-flight sequences already hold
    /// are released normally by those completions; a model leaving while its forwards still run releases them as they return rather than
    /// having them cancelled. Returns true when a seat was removed.
    /// </summary>
    public bool UnregisterModel(string modelName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelName);

        lock (_gate)
        {
            return _resident.Remove(modelName);
        }
    }

    /// <summary>
    /// The grant this seam hands a model for one step boundary: how many of the <paramref name="requested"/> candidate sequences it may
    /// launch device work for right now. A model with a declared share is granted at most its fair-share cap minus the slots its own
    /// launched-but-unfinished sequences already hold, so the cap bounds this model's device occupancy and nothing else's; a model
    /// resident without a declared share is granted everything it asked for. A grant below what was requested is what a scheduler defers
    /// over the excess with — the deferral names the share, the model, and the device segment the window bounds (see <see cref="CapText"/>).
    /// </summary>
    public CoScheduleGrant RequestStepSlots(string modelName, int requested)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelName);
        ArgumentOutOfRangeException.ThrowIfNegative(requested);

        lock (_gate)
        {
            if (!_resident.TryGetValue(modelName, out ShareEntry? entry) || entry.SharePercent is null)
            {
                // Not seat-bound by fair-share: either this seam never registered the model, or the model declares no share. Either way the
                // seam launches everything the scheduler offered it, exactly as scheduling behaved before any fair-share existed.
                return new CoScheduleGrant(requested, ShareBound: false, SharePercent: null, CapSlots: null);
            }

            int cap = GrantedCap(entry.SharePercent.Value);
            int room = Math.Max(cap - entry.OutstandingSlots, 0);
            int granted = Math.Min(requested, room);
            return new CoScheduleGrant(granted, ShareBound: true, entry.SharePercent, cap);
        }
    }

    /// <summary>
    /// Records that a model launched device work for <paramref name="launched"/> sequences at a boundary, called after composition decided
    /// what the grant admitted and before those forwards run, so a concurrent observer sees this model's occupancy while its work is live.
    /// A model this seam seats without a declared share tracks occupancy for observation alone — nothing bounds it. A model the seam does
    /// not seat tracks nothing.
    /// </summary>
    public void ReserveStepSlots(string modelName, int launched)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelName);
        ArgumentOutOfRangeException.ThrowIfNegative(launched);
        if (launched == 0)
        {
            return;
        }

        lock (_gate)
        {
            if (!_resident.TryGetValue(modelName, out ShareEntry? entry))
            {
                return;
            }

            entry.OutstandingSlots += launched;
        }
    }

    /// <summary>
    /// Returns the slots the sequences launched for one boundary occupy, once their forwards have completed. Called after a step's forwards
    /// finish, never before, so a slot is free only when the device work that held it is genuinely done. A completion for a model this seam
    /// no longer seats is a no-op: its in-flight work already left when its seat did.
    /// </summary>
    public void CompleteStepSlots(string modelName, int launched)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelName);
        ArgumentOutOfRangeException.ThrowIfNegative(launched);
        if (launched == 0)
        {
            return;
        }

        lock (_gate)
        {
            if (!_resident.TryGetValue(modelName, out ShareEntry? entry))
            {
                return;
            }

            // A release can never drive a model's occupancy below zero: only launches this seam granted are counted here, and a re-share
            // mid-flight leaves already-launched sequences whole (their slots return exactly once, when their forwards return).
            entry.OutstandingSlots = Math.Max(entry.OutstandingSlots - launched, 0);
        }
    }

    /// <summary>The fair-share cap in step slots a declared share resolves to: its percent of the window, floored at one slot so no resident model with a declared share can ever be rounded off the device.</summary>
    public static int GrantedCap(int computeSharePercent)
    {
        if (computeSharePercent is < 1 or > 100)
        {
            throw new ArgumentOutOfRangeException(
                nameof(computeSharePercent),
                $"a model's compute share must be an integer from 1 to 100 — the percent of one co-scheduling window ({WindowStepSlots} step slot(s)) — got {computeSharePercent}.");
        }

        return Math.Max(1, WindowStepSlots * computeSharePercent / 100);
    }

    /// <summary>The cap text a deferral names when a grant came up short: the model, its declared share, and how many window slots that share bounds it to.</summary>
    public static string CapText(string modelName, int sharePercent, string segmentDescription) =>
        $"model '{modelName}' compute_share {sharePercent}% bounds device '{segmentDescription}' to {GrantedCap(sharePercent)} " +
        $"co-scheduling step slot(s) of this window's {WindowStepSlots}";

    /// <summary>What a grant answered: the slot count this model may launch now, whether a declared share bound it, the share, and the cap that share resolves to. Internal composition material: it decides which candidates a boundary launches, and nothing a grant touches moves a token.</summary>
    public readonly record struct CoScheduleGrant(int GrantedSlots, bool ShareBound, int? SharePercent, int? CapSlots);

    // Diagnostics for hosts and tests: what a model's seat currently holds against the window. Observers only — no admission or composition
    // decision reads them.
    /// <summary>The slots a model's launched-but-unfinished device work currently holds in this window; zero for an unbudgeted or unseated model.</summary>
    public int OutstandingStepSlots(string modelName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelName);
        lock (_gate)
        {
            return _resident.TryGetValue(modelName, out ShareEntry? entry) ? entry.OutstandingSlots : 0;
        }
    }

    /// <summary>The fair-share percent a resident model currently carries, or null when it declares none.</summary>
    public int? ShareFor(string modelName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelName);
        lock (_gate)
        {
            return _resident.TryGetValue(modelName, out ShareEntry? entry) ? entry.SharePercent : null;
        }
    }
}
