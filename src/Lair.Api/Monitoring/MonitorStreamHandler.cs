using System.Text;
using System.Threading.Channels;
using Microsoft.AspNetCore.Http;

namespace Lair.Api.Monitoring;

/// <summary>
/// Serves the live metrics stream as Server-Sent Events on <c>/monitor/stream</c>: one JSON snapshot per event,
/// newest snapshots flowing from a fresh subscription opened when the request arrives. Comment frames (<c>: ping</c>)
/// keep proxies from dropping the connection while the engine is quiet and no snapshots are being published; a
/// client hang-up ends the stream cleanly through <see cref="HttpRequest.RequestAborted"/>. The route lives outside
/// <c>/v1/*</c> deliberately — it is operator surface, not part of the OpenAI-compatible API contract.
/// </summary>
internal static class MonitorStreamHandler
{
    private const int CommentIntervalMs = 15_000;

    public static async Task HandleAsync(HttpContext http, EngineMetricsBus bus)
    {
        ArgumentNullException.ThrowIfNull(bus);

        CancellationToken ct = http.RequestAborted;
        http.Response.ContentType = "text/event-stream";
        http.Response.Headers.CacheControl = "no-cache";

        ChannelReader<MetricsSnapshot> reader = bus.Subscribe();
        try
        {
            // A comment frame first: the client learns the stream is live before the engine's next snapshot lands.
            await WriteRawAsync(http, ": connected\n\n"u8.ToArray(), ct);

            while (true)
            {
                MetricsSnapshot snapshot;
                using (CancellationTokenSource frameCts = CancellationTokenSource.CreateLinkedTokenSource(ct))
                {
                    frameCts.CancelAfter(CommentIntervalMs);
                    try
                    {
                        snapshot = await reader.ReadAsync(frameCts.Token);
                    }
                    catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                    {
                        // No snapshot within the interval: keep-alive comment, then wait again.
                        await WriteRawAsync(http, ": ping\n\n"u8.ToArray(), ct);
                        continue;
                    }
                }

                byte[] json = Encoding.UTF8.GetBytes(snapshot.ToJson());
                await http.Response.Body.WriteAsync(SsePrefix, ct);
                await http.Response.Body.WriteAsync(json, ct);
                await http.Response.Body.WriteAsync(Newline, ct);
                await http.Response.Body.FlushAsync(ct);
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
            bus.Unsubscribe(reader);
        }
    }

    private static readonly byte[] SsePrefix = "data: "u8.ToArray();
    private static readonly byte[] Newline = "\n\n"u8.ToArray();

    private static async Task WriteRawAsync(HttpContext http, byte[] bytes, CancellationToken ct)
    {
        await http.Response.Body.WriteAsync(bytes, ct);
        await http.Response.Body.FlushAsync(ct);
    }
}
