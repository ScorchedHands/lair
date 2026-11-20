namespace Lair.Http.Monitoring;

/// <summary>
/// The host-scoped event line several engine buses publish through: one monotonic id counter and one fan-out lock
/// shared by every bus a monitor stream samples, so ids name one point in one total stream no matter which engine
/// produced the frame, and publish order matches id order host-wide because id assignment and delivery happen under
/// the same lock. A bus with no lineage keeps its own counter and lock and behaves exactly as it did before this
/// seam existed; a bus with one joins the line it carries.
/// </summary>
public sealed class MonitorEventLineage
{
    private long _eventSeq;

    /// <summary>The lock every bus sharing this line publishes under, so assignment and delivery stay ordered together.</summary>
    internal object Fanout { get; } = new();

    /// <summary>The next event id on this line. Call only while holding <see cref="Fanout"/>.</summary>
    internal long NextEventId() => ++_eventSeq;

    /// <summary>The highest id this line has handed out (0 before the first). Read outside the lock by reconnect replay.</summary>
    internal long LatestEventId => Volatile.Read(ref _eventSeq);
}
