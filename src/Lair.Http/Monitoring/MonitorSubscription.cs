using System.Threading.Channels;

namespace Lair.Http.Monitoring;

/// <summary>One delivered monitor stream frame: its server-scoped event id, the versioned event name, and the compact JSON payload.</summary>
public sealed record MonitorFrame(long Id, string EventName, string Json);

/// <summary>
/// One live view of the bus's event flow. Snapshots arrive coalesced: a capacity-one drop-oldest queue always holds
/// exactly the newest published snapshot this reader has not consumed yet, so a slow consumer never blocks the
/// publisher and only ever sees the latest one (intermediate frames are intentionally dropped). Request lifecycle
/// frames ride an independent bounded queue under the same drop-oldest policy — the oldest unread frame yields to
/// newer ones. Both queues complete when the subscription is closed or the bus disposes; writes to a completed
/// writer are inert no-ops, so publishing never blocks on a dead stream either. Fullness probes report each drop
/// for the bus's dropped-events tally: every publish runs under the bus fan-out lock, so probe and write pair up
/// consistently per subscriber even though publishers arrive from different threads.
/// </summary>
public sealed class MonitorSubscription
{
    /// <summary>Stable identifier for this subscriber's attach/detach log lines.</summary>
    public string Id { get; }

    internal Channel<MonitorFrame> Snapshots { get; } = Channel.CreateBounded<MonitorFrame>(
        new BoundedChannelOptions(1)
        {
            FullMode = BoundedChannelFullMode.DropOldest, // latest-pending coalescing: a newer snapshot replaces the unread one
            SingleReader = true,
        });

    internal Channel<MonitorFrame> Requests { get; } = Channel.CreateBounded<MonitorFrame>(
        new BoundedChannelOptions(EngineMetricsBus.RequestQueueCapacity)
        {
            FullMode = BoundedChannelFullMode.DropOldest, // bounded backlog for a slow reader; oldest frames yield to newer ones
            SingleReader = true,
        });

    internal MonitorSubscription(long sequence) => Id = $"sub-{sequence}";

    /// <summary>False when this write replaced an unread pending snapshot — i.e. this subscriber missed one to coalescing.</summary>
    internal bool PublishSnapshotCoalesced(MonitorFrame frame)
    {
        if (Snapshots.Reader.Completion.IsCompleted)
        {
            return true; // dead stream: the write is inert and nobody lost an event worth counting
        }

        bool coalescedAway = Snapshots.Reader.CanCount && Snapshots.Reader.Count >= 1; // full: drop-oldest discards the unread frame for this one
        Snapshots.Writer.TryWrite(frame);
        return !coalescedAway;
    }

    /// <summary>False when this write evicted an older unread request frame from this subscriber's bounded queue.</summary>
    internal bool PublishRequest(MonitorFrame frame)
    {
        if (Requests.Reader.Completion.IsCompleted)
        {
            return true; // dead stream: the write is inert and nobody lost an event worth counting
        }

        bool droppedOldest = Requests.Reader.CanCount && Requests.Reader.Count >= EngineMetricsBus.RequestQueueCapacity;
        Requests.Writer.TryWrite(frame);
        return !droppedOldest;
    }

    internal void Close()
    {
        Snapshots.Writer.TryComplete();
        Requests.Writer.TryComplete();
    }
}
