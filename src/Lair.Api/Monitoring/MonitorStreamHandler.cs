using System.Globalization;
using System.Text;
using System.Threading.Channels;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace Lair.Api.Monitoring;

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
/// Reconnect semantics (Last-Event-ID): a frame id names one point in the bus's single total event sequence, but
/// request frames and coalesced snapshots are transient — only snapshots persist for replay, newest-last inside the
/// 300-snapshot ring. On connect with a <c>Last-Event-ID</c> header: ids naming retained snapshots re-deliver every
/// snapshot newer than them (oldest first), then one fresh <c>resync</c> closes whatever transient state replay
/// cannot carry — in-flight requests and completed-request entries lost to coalescing. An id older than the whole
/// ring, or an id that matches nothing retained (a request/resync frame's id, or a snapshot already evicted), gets
/// a <c>resync</c> instead of partial replay: current state plus whatever snapshots are still retained, never an
/// error. An id at or above the newest known event means the client is caught up and receives only the live tail.
/// A connect without the header (a fresh stream) starts straight on the live tail like before. Snapshots published
/// while a replay is being written may also arrive through the live subscription — clients dedupe by frame id, so
/// an at-most-once duplicate costs nothing and no missed event can slip past either channel.
/// </remarks>
internal static class MonitorStreamHandler
{
    private const int CommentIntervalMs = 15_000;

    public static async Task HandleAsync(HttpContext http, EngineMetricsBus bus)
    {
        ArgumentNullException.ThrowIfNull(bus);

        ILogger logger = http.RequestServices.GetRequiredService<ILoggerFactory>().CreateLogger(typeof(MonitorStreamHandler));

        CancellationToken ct = http.RequestAborted;
        http.Response.ContentType = "text/event-stream";
        http.Response.Headers.CacheControl = "no-cache";

        MonitorSubscription subscription = bus.Subscribe();
        try
        {
            string? lastEventIdHeader = http.Request.Headers["Last-Event-ID"];
            logger.LogInformation(
                "monitor_stream event=attach id={SubscriberId} last_event_id={LastEventId}",
                subscription.Id,
                lastEventIdHeader ?? "(none)");

            // A comment frame first: the client learns the stream is live before anything else lands.
            await WriteRawAsync(http, ": connected\n\n"u8.ToArray(), ct);

            if (TryParseLastEventId(lastEventIdHeader, out long headerId))
            {
                bool staleBeyondRing = bus.PublishedTotal > EngineMetricsBus.RingCapacity
                    && headerId < bus.OldestRetainedSnapshotId;

                if (staleBeyondRing)
                {
                    // The id predates everything the ring kept — replaying all 300 retained frames would pretend
                    // at completeness a state dump answers better.
                    await WriteResyncAsync(http, bus, ct);
                }
                else
                {
                    IReadOnlyList<MonitorFrame> missed = bus.SnapshotReplayAfter(headerId);

                    foreach (MonitorFrame frame in missed)
                    {
                        await WriteFrameAsync(http, frame, ct);
                    }

                    if (missed.Count == 0 && headerId < bus.LatestEventId)
                    {
                        // Nothing snapshot-newer exists for this id but the bus has moved past it: the client's
                        // last seen event was transient. Converge through resync instead of guessing.
                        await WriteResyncAsync(http, bus, ct);
                    }

                    if (missed.Count > 0)
                    {
                        // Replay carried snapshots only; one state dump restores in-flight and completed requests too.
                        await WriteResyncAsync(http, bus, ct);
                    }
                }
            }

            while (true)
            {
                MonitorFrame frame;
                using (CancellationTokenSource frameCts = CancellationTokenSource.CreateLinkedTokenSource(ct))
                {
                    frameCts.CancelAfter(CommentIntervalMs);
                    try
                    {
                        frame = await ReadNextAsync(subscription, frameCts.Token);
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
            // The bus completed the subscription (host shutdown): end the stream quietly.
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            // Client hang-up: nothing left to do.
        }
        finally
        {
            logger.LogInformation("monitor_stream event=detach id={SubscriberId}", subscription.Id);
            bus.Unsubscribe(subscription);
        }
    }

    private static readonly byte[] SsePrefix = "data: "u8.ToArray();
    private static readonly byte[] Newline = "\n\n"u8.ToArray();

    // Waits for the next frame from either queue — snapshots coalesced to their newest, request frames in order.
    // A closed queue loses its vote; both closed surfaces as ChannelClosedException through a final read attempt.
    private static async Task<MonitorFrame> ReadNextAsync(MonitorSubscription subscription, CancellationToken ct)
    {
        while (true)
        {
            if (subscription.Snapshots.Reader.TryRead(out MonitorFrame? snapshot))
            {
                return snapshot!;
            }

            if (subscription.Requests.Reader.TryRead(out MonitorFrame? request))
            {
                return request!;
            }

            Task<bool> snapshots = subscription.Snapshots.Reader.WaitToReadAsync(ct).AsTask();
            Task<bool> requests = subscription.Requests.Reader.WaitToReadAsync(ct).AsTask();

            // Observe faults on whichever wait loses the race — a cancelled loser must not surface as an
            // unobserved task exception when it finalizes.
            _ = snapshots.ContinueWith(static faulted => _ = faulted.Exception, TaskContinuationOptions.OnlyOnFaulted);
            _ = requests.ContinueWith(static faulted => _ = faulted.Exception, TaskContinuationOptions.OnlyOnFaulted);

            Task winner = await Task.WhenAny(snapshots, requests);

            if (winner == snapshots)
            {
                bool open = await snapshots;
                if (!open)
                {
                    return await ReadAfterClosedAsync(subscription, ct); // Close() completes both queues together: whatever the other still holds drains next
                }

                continue;
            }

            bool requestsOpen = await requests;
            if (!requestsOpen)
            {
                return await ReadAfterClosedAsync(subscription, ct);
            }
        }
    }

    // One queue closed: hand back whatever the other still holds, ending with ChannelClosedException once both are done.
    private static async Task<MonitorFrame> ReadAfterClosedAsync(MonitorSubscription subscription, CancellationToken ct)
    {
        if (subscription.Snapshots.Reader.TryRead(out MonitorFrame? snapshot))
        {
            return snapshot!;
        }

        if (subscription.Requests.Reader.TryRead(out MonitorFrame? request))
        {
            return request!;
        }

        // Both queues completed: the next read throws ChannelClosedException, which HandleAsync treats as host shutdown.
        ct.ThrowIfCancellationRequested();
        return await subscription.Snapshots.Reader.ReadAsync(ct);
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
