using System.Globalization;
using System.Text;
using System.Threading.Channels;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace Lair.Http.Monitoring;

/// <summary>
/// Serves the live engine feed as versioned Server-Sent Events on <c>/monitor/stream</c>. Every frame carries a
/// server-scoped incrementing <c>id:</c>, an event name from <see cref="MonitorEventNames"/>, and one JSON payload
/// stamped with the schema version: per-model snapshots (the established <c>metrics_</c> vocabulary, now including
/// <c>metrics_events_version</c>, <c metrics_event_seq</c>, and drop tallies), request lifecycle frames keyed by the
/// existing <c>x-lair-request-id</c>, and resync state dumps. Comment frames (<c>: ping</c>) keep proxies from
/// dropping the connection while the engine is quiet; a client hang-up ends the stream cleanly through
/// <see cref="HttpRequest.RequestAborted"/>. The route lives outside <c>/v1/*</c> deliberately — it is operator
/// surface, not part of the OpenAI-compatible API contract.
/// </summary>
/// <remarks>
/// Reconnect semantics (Last-Event-ID): a frame id names one point in the host's single total event sequence — every
/// bus a stream samples publishes through one <see cref="MonitorEventLineage"/>, so ids never repeat across engines —
/// but request frames and coalesced snapshots are transient, and only snapshots persist for replay, newest-last
/// inside each bus's 300-snapshot ring. On connect with a <c>Last-Event-ID</c> header the header is judged against
/// each bus separately: ids naming retained snapshots re-deliver every snapshot newer than them, oldest-first across
/// all buses, then one fresh <c>resync</c> per bus closes whatever transient state replay cannot carry for that engine
/// — in-flight requests and completed-request entries lost to coalescing. An id older than a bus's whole ring, or
/// naming nothing that bus retained (a request/resync frame's id, or a snapshot already evicted), gets that bus's
/// <c>resync</c> instead of partial replay: current state plus whatever snapshots are still retained, never an
/// error. An id at or above every bus's newest known event means the client is caught up and receives only the live
/// tail. A connect without the header (a fresh stream) starts straight on the live tail like before. Snapshots
/// published while a replay is being written may also arrive through the live subscription — clients dedupe by frame
/// id, so an at-most-once duplicate costs nothing and no missed event can slip past either channel.
/// </remarks>
internal static class MonitorStreamHandler
{
    private const int CommentIntervalMs = 15_000;

    public static async Task HandleAsync(HttpContext http, EngineMetricsBus bus) =>
        await HandleAsync(http, [bus]);

    public static async Task HandleAsync(HttpContext http, IReadOnlyList<EngineMetricsBus> buses)
    {
        ArgumentNullException.ThrowIfNull(buses);

        ILogger logger = http.RequestServices.GetRequiredService<ILoggerFactory>().CreateLogger(typeof(MonitorStreamHandler));

        CancellationToken ct = http.RequestAborted;
        http.Response.ContentType = "text/event-stream";
        http.Response.Headers.CacheControl = "no-cache";

        if (buses.Count == 0)
        {
            // Nothing to sample: answer an empty live stream rather than a fault — the same clean close a shutdown
            // mid-stream produces. Serving hosts always advertise at least one bus, so this is a defensive posture.
            await WriteRawAsync(http, ": connected\n\n"u8.ToArray(), ct);
            return;
        }

        // One subscription per sampled bus: every engine's frames land on this one stream, and a bus whose engine
        // is disposed mid-stream drops out — its queued frames drain, then it stops voting — while the rest keep
        // flowing. The pairing stays positional so detach unsubscribes exactly what attach opened.
        List<EngineMetricsBus> sampled = [.. buses];
        List<MonitorSubscription> subscriptions = [.. sampled.Select(bus => bus.Subscribe())];
        try
        {
            string? lastEventIdHeader = http.Request.Headers["Last-Event-ID"];
            logger.LogInformation(
                "monitor_stream event=attach buses={BusCount} last_event_id={LastEventId}",
                sampled.Count,
                lastEventIdHeader ?? "(none)");

            // A comment frame first: the client learns the stream is live before anything else lands.
            await WriteRawAsync(http, ": connected\n\n"u8.ToArray(), ct);

            if (TryParseLastEventId(lastEventIdHeader, out long headerId))
            {
                await ReplayAsync(http, sampled, headerId, ct);
            }

            while (true)
            {
                MonitorFrame frame;
                using (CancellationTokenSource frameCts = CancellationTokenSource.CreateLinkedTokenSource(ct))
                {
                    frameCts.CancelAfter(CommentIntervalMs);
                    try
                    {
                        frame = await ReadNextAsync(subscriptions, frameCts.Token);
                    }
                    catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                    {
                        // Nothing within the interval: keep-alive comment, then wait again.
                        await WriteRawAsync(http, ": ping\n\n"u8.ToArray(), ct);
                        continue;
                    }
                }

                await WriteFrameAsync(http, frame, ct);
            }
        }
        catch (ChannelClosedException)
        {
            // Every sampled bus completed its subscription (host shutdown): end the stream quietly.
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            // Client hang-up: nothing left to do.
        }
        finally
        {
            logger.LogInformation("monitor_stream event=detach buses={BusCount}", sampled.Count);
            for (int index = 0; index < subscriptions.Count; index++)
            {
                sampled[index].Unsubscribe(subscriptions[index]);
            }
        }
    }

    // Reconnect replay for a Last-Event-ID header, judged per sampled bus: retained snapshots newer than the header
    // re-deliver oldest-first across every bus — ids are host-unique and monotonic, so the merged walk needs no
    // dedupe — and each bus whose transient state replay cannot carry closes with its own resync. A bus whose header
    // predates its whole ring resyncs instead of pretending a full replay; a bus that already sits at or above the
    // header contributes nothing and lives on through the live tail.
    private static async Task ReplayAsync(HttpContext http, IReadOnlyList<EngineMetricsBus> buses, long headerId, CancellationToken ct)
    {
        List<MonitorFrame> missed = [];
        List<EngineMetricsBus> resyncBuses = [];

        foreach (EngineMetricsBus bus in buses)
        {
            bool staleBeyondRing = bus.PublishedTotal > EngineMetricsBus.RingCapacity
                && headerId < bus.OldestRetainedSnapshotId;

            if (staleBeyondRing)
            {
                // The id predates everything this bus's ring kept — replaying all its retained frames would pretend
                // at completeness a state dump answers better.
                resyncBuses.Add(bus);
                continue;
            }

            IReadOnlyList<MonitorFrame> frames = bus.SnapshotReplayAfter(headerId);

            if (frames.Count == 0)
            {
                if (headerId < bus.LatestEventId)
                {
                    // Nothing snapshot-newer exists for this id but this bus has moved past it: the client's last
                    // seen event was transient. Converge through resync instead of guessing.
                    resyncBuses.Add(bus);
                }

                continue; // caught up on this bus, or naming nothing it retained while it never advanced
            }

            missed.AddRange(frames);
            resyncBuses.Add(bus); // replay carried snapshots only; this bus's in-flight and completed requests ride its resync
        }

        missed.Sort(static (a, b) => a.Id.CompareTo(b.Id)); // oldest-first across buses: host ids order the whole replay
        foreach (MonitorFrame frame in missed)
        {
            await WriteFrameAsync(http, frame, ct);
        }

        foreach (EngineMetricsBus bus in resyncBuses)
        {
            await WriteResyncAsync(http, bus, ct);
        }
    }

    private static readonly byte[] SsePrefix = "data: "u8.ToArray();
    private static readonly byte[] Newline = "\n\n"u8.ToArray();

    // Waits for the next frame from any sampled bus — snapshots coalesced to their newest per bus, request frames in
    // order. A bus whose subscription closed loses its vote once its queued frames have drained; the read surfaces
    // ChannelClosedException only when every sampled bus has closed, which is how host shutdown ends the stream.
    private static async Task<MonitorFrame> ReadNextAsync(List<MonitorSubscription> subscriptions, CancellationToken ct)
    {
        while (true)
        {
            // Hand back anything already queued, newest-pending snapshot before that bus's request frames.
            foreach (MonitorSubscription subscription in subscriptions)
            {
                if (subscription.Snapshots.Reader.TryRead(out MonitorFrame? snapshot))
                {
                    return snapshot!;
                }

                if (subscription.Requests.Reader.TryRead(out MonitorFrame? request))
                {
                    return request!;
                }
            }

            List<Task<bool>> waits = [];
            foreach (MonitorSubscription subscription in subscriptions)
            {
                if (subscription.Snapshots.Reader.Completion.IsCompleted && subscription.Requests.Reader.Completion.IsCompleted)
                {
                    continue; // this bus is drained and closed: no vote left, the rest keep the stream alive
                }

                waits.Add(subscription.Snapshots.Reader.WaitToReadAsync(ct).AsTask());
                waits.Add(subscription.Requests.Reader.WaitToReadAsync(ct).AsTask());
            }

            if (waits.Count == 0)
            {
                // Every queue completed: the next read throws ChannelClosedException, which HandleAsync treats as host shutdown.
                ct.ThrowIfCancellationRequested();
                return await subscriptions[0].Snapshots.Reader.ReadAsync(ct);
            }

            // Observe faults on whichever wait loses the race — a cancelled loser must not surface as an
            // unobserved task exception when it finalizes.
            foreach (Task<bool> wait in waits)
            {
                _ = wait.ContinueWith(static faulted => _ = faulted.Exception, TaskContinuationOptions.OnlyOnFaulted);
            }

            Task winner = await Task.WhenAny(waits);

            if (!await (Task<bool>)winner)
            {
                continue; // this queue closed; whatever the other queues still hold drains on the next pass
            }
        }
    }

    private static bool TryParseLastEventId(string? header, out long id) =>
        long.TryParse(header, NumberStyles.Integer, CultureInfo.InvariantCulture, out id);

    private static async Task WriteResyncAsync(HttpContext http, EngineMetricsBus bus, CancellationToken ct) =>
        await WriteFrameAsync(http, new MonitorFrame(0, MonitorEventNames.Resync, bus.BuildResync().ToJson()), ct);

    private static async Task WriteFrameAsync(HttpContext http, MonitorFrame frame, CancellationToken ct)
    {
        if (frame.Id > 0)
        {
            await http.Response.Body.WriteAsync(Encoding.UTF8.GetBytes($"id: {frame.Id}\nevent: {frame.EventName}\n"), ct);
        }
        else
        {
            // A resync is transient state, not a replayable point in the sequence: it carries no frame id, so a
            // client's Last-Event-ID continues to name its last durable snapshot and reconnects stay precise.
            await http.Response.Body.WriteAsync(Encoding.UTF8.GetBytes($"event: {frame.EventName}\n"), ct);
        }

        await http.Response.Body.WriteAsync(SsePrefix, ct);
        await http.Response.Body.WriteAsync(Encoding.UTF8.GetBytes(frame.Json), ct);
        await http.Response.Body.WriteAsync(Newline, ct);
        await http.Response.Body.FlushAsync(ct);
    }

    private static async Task WriteRawAsync(HttpContext http, byte[] bytes, CancellationToken ct)
    {
        await http.Response.Body.WriteAsync(bytes, ct);
        await http.Response.Body.FlushAsync(ct);
    }
}
