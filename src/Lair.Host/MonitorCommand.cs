using System.Net.Http;

namespace Lair.Host;

/// <summary>
/// Live engine monitor: follows the server's <c>/monitor/stream</c> SSE snapshots and re-renders one compact table
/// per snapshot — model, decode tok/s, average step time, batch occupancy, prefix-cache hit share, KV block usage,
/// working set, CPU share, active sequences — with an accelerator tail (kernel busy share, runtime PCIe transfer tallies,
/// persistent device weight residency) on rows only where the served engine reports those facts. A CPU-only serve renders
/// its columns as they always have and shows nothing about a device it does not hold open. Ctrl+C (or any cancellation)
/// exits cleanly with 0; a server that cannot be reached produces exactly one actionable error line and a non-zero exit.
/// </summary>
public static class MonitorCommand
{
    /// <summary>Renders the monitor table for one engine until cancelled, writing rows to <paramref name="output"/>. Rows closer together than <paramref name="minRenderIntervalMs"/> are folded so a chatty server cannot flood the console.</summary>
    public static async Task<int> RunAsync(Uri baseUri, TextWriter output, CancellationToken cancellationToken, int minRenderIntervalMs = 0)
    {
        ArgumentNullException.ThrowIfNull(baseUri);
        ArgumentNullException.ThrowIfNull(output);

        using HttpClient http = new() { Timeout = Timeout.InfiniteTimeSpan };

        HttpResponseMessage response;
        try
        {
            HttpRequestMessage request = new(HttpMethod.Get, StreamUrl(baseUri));
            response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        }
        catch (Exception ex)
        {
            // One line the operator can act on: what was tried and why it failed.
            await output.WriteLineAsync($"error: cannot reach monitor stream at {StreamUrl(baseUri)} ({ex.Message})");
            return 1;
        }

        using (response)
        {
            if (!response.IsSuccessStatusCode)
            {
                await output.WriteLineAsync($"error: server answered {(int)response.StatusCode} for {StreamUrl(baseUri)} — is this a lair serve host?");
                return 1;
            }

            try
            {
                using Stream body = await response.Content.ReadAsStreamAsync(cancellationToken);
                StreamReader reader = new(body);

                await output.WriteLineAsync("model                 tok/s   step ms  seqs/step  cache hit    KV used/cap  active");

                long lastRenderMs = 0;
                bool renderedAny = false;

                while (!cancellationToken.IsCancellationRequested)
                {
                    string? line = await reader.ReadLineAsync(cancellationToken);
                    if (line is null)
                    {
                        break; // server closed the stream: nothing left to watch
                    }

                    if (!line.StartsWith("data: ", StringComparison.Ordinal))
                    {
                        continue; // SSE comment frames and separators carry no snapshot
                    }

                    MetricsRow? row = ParseSnapshot(line["data: ".Length..]);
                    if (row is null)
                    {
                        continue;
                    }

                    long nowMs = Environment.TickCount64;
                    if (renderedAny && nowMs - lastRenderMs < minRenderIntervalMs)
                    {
                        continue; // fold rows that arrive faster than the requested render cadence
                    }

                    renderedAny = true;
                    lastRenderMs = nowMs;
                    await output.WriteLineAsync(Render(row));
                    await output.FlushAsync(cancellationToken);
                }
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                // Ctrl+C: clean exit.
            }

            return 0;
        }
    }

    private static string StreamUrl(Uri baseUri) => new Uri(baseUri, "/monitor/stream").ToString();

    /// <summary>Extracts the table fields from one snapshot JSON document; null when the line is not a readable snapshot.</summary>
    internal static MetricsRow? ParseSnapshot(string json)
    {
        try
        {
            using System.Text.Json.JsonDocument document = System.Text.Json.JsonDocument.Parse(json);
            System.Text.Json.JsonElement root = document.RootElement;

            if (!root.TryGetProperty("metrics_model", out System.Text.Json.JsonElement model))
            {
                return null;
            }

            // Accelerator facts ride the same snapshot additively: a CPU-only serve carries none of them, so its rows keep
            // rendering exactly as they did and never show an idle-hardware zero that no engine fact supports.
            double? gpuUtilization = ReadOptionalDouble(root, "metrics_gpu_utilization_percent");
            long? transferBytes = ReadOptionalLong(root, "metrics_pcie_transfer_bytes_total");
            long? transferCalls = ReadOptionalLong(root, "metrics_pcie_transfers_total");
            long? persistentVramBytes = ReadOptionalLong(root, "metrics_persistent_vram_bytes");

            AcceleratorRow? accelerator = gpuUtilization is null && transferBytes is null && transferCalls is null && persistentVramBytes is null
                ? null
                : new AcceleratorRow(gpuUtilization, transferBytes ?? 0L, transferCalls ?? 0L, persistentVramBytes ?? 0L);

            return new MetricsRow(
                model.GetString() ?? string.Empty,
                ReadDouble(root, "metrics_tok_s"),
                ReadDouble(root, "metrics_step_time_avg_ms"),
                ReadDouble(root, "metrics_batch_occupancy_seqs_per_step"),
                ReadLong(root, "metrics_cache_hit_sequences_total"),
                ReadLong(root, "metrics_requests_total"),
                ReadInt(root, "metrics_kv_used_blocks"),
                ReadInt(root, "metrics_kv_capacity_blocks"),
                ReadInt(root, "metrics_active_sequences"),
                accelerator);
        }
        catch (System.Text.Json.JsonException)
        {
            return null; // keep-alive comments or a malformed frame must not end the table
        }
    }

    private static string Render(MetricsRow row)
    {
        double hitPercent = row.RequestsTotal > 0 ? 100.0 * row.CacheHitSequences / row.RequestsTotal : 0.0;
        string line = $"{row.Model,-20} {row.TokensPerSec,7:F2} {row.StepTimeMs,8:F1} {row.Occupancy,9:F2} {hitPercent,8:F1}% {row.KvUsedBlocks + "/" + row.KvCapacityBlocks,13}  {row.ActiveSequences,6}";

        if (row.Accelerator is AcceleratorRow accelerator)
        {
            string utilization = accelerator.GpuUtilizationPercent is double percent ? $"{percent:F2}" : "n/a";
            line += $"   gpu {utilization}%  pcie {Megabytes(accelerator.PcieTransferBytesTotal)} over {accelerator.PcieTransfersTotal} copies  persistent-vram {Megabytes(accelerator.PersistentVramBytes)}";
        }

        return line;
    }

    private static string Megabytes(long bytes) => $"{bytes / (1024.0 * 1024.0):F1} MiB";

    private static double ReadDouble(System.Text.Json.JsonElement root, string name) =>
        root.TryGetProperty(name, out System.Text.Json.JsonElement value) && value.TryGetDouble(out double parsed) ? parsed : 0.0;

    // An additive optional field: absent or null reads as nothing rather than zero, so a CPU-only row never gains an accelerator column it cannot support.
    private static double? ReadOptionalDouble(System.Text.Json.JsonElement root, string name) =>
        root.TryGetProperty(name, out System.Text.Json.JsonElement value) && value.ValueKind == System.Text.Json.JsonValueKind.Number && value.TryGetDouble(out double parsed) ? parsed : null;

    private static long? ReadOptionalLong(System.Text.Json.JsonElement root, string name) =>
        root.TryGetProperty(name, out System.Text.Json.JsonElement value) && value.ValueKind == System.Text.Json.JsonValueKind.Number && value.TryGetInt64(out long parsed) ? parsed : null;

    private static long ReadLong(System.Text.Json.JsonElement root, string name) =>
        root.TryGetProperty(name, out System.Text.Json.JsonElement value) && value.TryGetInt64(out long parsed) ? parsed : 0L;

    private static int ReadInt(System.Text.Json.JsonElement root, string name) =>
        root.TryGetProperty(name, out System.Text.Json.JsonElement value) && value.TryGetInt32(out int parsed) ? parsed : 0;

    /// <summary>The table fields lifted from one snapshot. Accelerator facts ride along only when the served engine reports them, so a CPU-only row renders its columns unchanged.</summary>
    internal sealed record MetricsRow(
        string Model,
        double TokensPerSec,
        double StepTimeMs,
        double Occupancy,
        long CacheHitSequences,
        long RequestsTotal,
        int KvUsedBlocks,
        int KvCapacityBlocks,
        int ActiveSequences,
        AcceleratorRow? Accelerator = null);

    /// <summary>Engine-side accelerator facts one snapshot carries: windowed kernel busy share beside the runtime copy tallies and the persistent device weight residency currently held. A utilization reading no snapshot filled stays null rather than claiming an idle device.</summary>
    internal sealed record AcceleratorRow(
        double? GpuUtilizationPercent,
        long PcieTransferBytesTotal,
        long PcieTransfersTotal,
        long PersistentVramBytes);
}
