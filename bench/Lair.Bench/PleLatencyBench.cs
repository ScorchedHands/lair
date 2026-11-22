using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

using Lair.Contracts;
using Lair.Loader;
using Lair.Models;

namespace Lair.Bench;

/// <summary>
/// PLE n-gram gather measurement against a real pinned split model: the reference hash and history normalization turn each token position into
/// one row per head, and those rows are addressed out of the physical shard that holds the table. It measures what gathering costs — index
/// computation alone, resolved-row reads through the loader's mapping (cold first-touch separated from warm repeats), and an unbuffered
/// FILE_FLAG_NO_BUFFERING read arm as the expensive-storage baseline — publishes byte-for-byte parity between the mapped rows and independent
/// sector-aligned file reads of the same offsets, tallies how many distinct table rows a token stream actually touches against what the model's
/// PLE residency ceiling allows, and changes no posture or config. Storage and governance evidence only: it benchmarks no forward pass and runs
/// no qwen4exp model execution.
/// </summary>
internal static class PleLatencyBench
{
    private const int SectorWindow = 4096; // windows align to 4 KiB: valid on 512e and 4Kn volumes alike, the same posture the gather bench's streaming substrate takes.

    public static int Run(string[] args)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("usage: ple-latency <gguf path> [--tokens N] [--reps R] [--substrate mmap|nocache|compare] [--resident-rows N] [--ple-cap-mb MB] [--sample K] [--out DIR]");
            return 1;
        }

        string modelPath = args[1];
        int tokens = 2048;
        int reps = 3;
        string substrate = "mmap";
        long residentRows = -1; // negative means: use the distinct rows this run's own token stream touched.
        long pleCapMb = 256;
        int sampleRows = 96;
        string outRoot = "reports";

        for (int i = 2; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--tokens":
                    tokens = ReadInt(args, ref i, "--tokens");
                    break;
                case "--reps":
                    reps = ReadInt(args, ref i, "--reps");
                    break;
                case "--substrate":
                    substrate = args[++i];
                    break;
                case "--resident-rows":
                    residentRows = ReadLong(args, ref i, "--resident-rows");
                    break;
                case "--ple-cap-mb":
                    pleCapMb = ReadLong(args, ref i, "--ple-cap-mb");
                    break;
                case "--sample":
                    sampleRows = ReadInt(args, ref i, "--sample");
                    break;
                case "--out":
                    outRoot = args[++i];
                    break;
                default:
                    Console.WriteLine($"ple-latency: unknown option '{args[i]}'");
                    return 1;
            }
        }

        if (tokens is < 1 or > 65_536)
        {
            Console.WriteLine("ple-latency: --tokens must be between 1 and 65536 — this bench stays bounded to one token stream");
            return 1;
        }

        if (reps is < 1 or > 20)
        {
            Console.WriteLine("ple-latency: --reps must be between 1 and 20");
            return 1;
        }

        if (substrate is not ("mmap" or "nocache" or "compare"))
        {
            Console.WriteLine("ple-latency: --substrate must be mmap, nocache, or compare");
            return 1;
        }

        bool windowsHost = OperatingSystem.IsWindows();
        if (substrate is ("nocache" or "compare") && !windowsHost)
        {
            Console.WriteLine("ple-latency: --substrate nocache/compare is a Windows measurement seam; this host runs mmap");
            return 1;
        }

        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"ple-latency: model file '{modelPath}' was not found");
            return 1;
        }

        using GgufFile model = GgufFile.Open(modelPath); // an assembled split opens whole; the PLE tensor's own shard is what the gather maps.
        LogicalTensor? declared = model.Tensors.Find("per_layer_token_embd.weight");

        if (declared is null)
        {
            Console.WriteLine($"ple-latency: '{model.Path}' declares no 'per_layer_token_embd.weight' tensor, so there is no PLE table to gather from");
            return 1;
        }

        PleNgramTable table = PleNgramTable.FromGguf(model, declared); // every refusal this seam raises names the model file, tensor, and metadata field at fault.

        Console.WriteLine($"ple-latency: {table.NgramSize}-gram x {table.HeadsPerNgram} heads = {table.HeadCount} rows/token over a {table.TableRows}-row '{table.Tensor.Name}' table " +
            $"({table.RowElements} elements, {table.RowStrideBytes} packed bytes/row, format {table.Tensor.Format})");

        // A deterministic token stream with real EOS ids and deliberately short windows mixed in: the normalization this seam mirrors has to be exercised against the model's own declared geometry.
        int[] historyLengths = new int[tokens];
        int[][] histories = BuildHistories(tokens, table, out long eosCutWindows, out long missingPredecessorWindows);

        List<Dictionary<string, object>> notes = [];

        // Arm A: index computation only — what the hash and normalization cost per token position.
        long[] headRows = new long[table.HeadCount];
        List<double> indexNanoseconds = [];
        HashSet<long> touchedRows = [];
        long rowsComputed = 0;

        for (int rep = 0; rep < reps; rep++)
        {
            for (int tokenIndex = 0; tokenIndex < tokens; tokenIndex++)
            {
                long startTicks = Stopwatch.GetTimestamp();
                table.ComputeRowIndices(histories[tokenIndex], headRows);
                indexNanoseconds.Add(Nanoseconds(startTicks, Stopwatch.GetTimestamp()));

                foreach (long row in headRows)
                {
                    touchedRows.Add(row);
                }

                rowsComputed += table.HeadCount;
            }
        }

        LatencyStats indexStats = LatencyStats.Of(indexNanoseconds);
        Console.WriteLine($"  index: p50 {indexStats.P50:F0} ns/token, p90 {indexStats.P90:F0}, p99 {indexStats.P99:F0}, max {indexStats.Max:F0} — {rowsComputed:N0} rows computed across {reps} reps");

        long distinctRows = touchedRows.Count;
        double footprintMb = (double)distinctRows * table.RowStrideBytes / PleResidencyGate.Megabytes;
        Console.WriteLine($"  reach: {distinctRows:N0} distinct table rows ({footprintMb:F1} MiB of packed rows at a {table.RowStrideBytes}-byte stride) out of {table.TableRows:N0} declared");

        List<long> sampleRowIndices = PickSampleRows(touchedRows, sampleRows);
        Dictionary<string, object?> parityFacts = [];

        // Arm B: resolved-row reads through the loader's row gather — first-touch pages reported apart from warm repeats.
        LatencyStats? mappedCold = null;
        LatencyStats? mappedWarm = null;

        if (substrate is "mmap" or "compare")
        {
            using PleRowSource rows = new(model, table);
            Console.WriteLine($"  substrate mmap: mapping '{rows.ShardPath}'");

            for (int rep = 0; rep < reps; rep++)
            {
                List<double> samples = [];

                foreach (long rowIndex in touchedRows)
                {
                    long startTicks = Stopwatch.GetTimestamp();
                    ReadOnlySpan<byte> row = rows.ResolveRow(rowIndex);
                    _ = Touch(row); // the span's own bytes are read so the measurement counts real page traffic, not address arithmetic.
                    samples.Add(Nanoseconds(startTicks, Stopwatch.GetTimestamp()));
                }

                if (rep == 0)
                {
                    mappedCold = LatencyStats.Of(samples);
                }
            }

            if (reps > 1)
            {
                mappedWarm = LatencyStats.Of(WarmSamples(model, table, touchedRows, reps)); // warm repeats run against a fresh gather so their pages are already resident.
            }

            Console.WriteLine($"  mapped rows: first-touch p50 {mappedCold!.P50:F0} ns/row (p99 {mappedCold.P99:F0}), " +
                $"{(mappedWarm is null ? "warm pass not measured below two reps" : $"warm p50 {mappedWarm.P50:F0} ns/row (p99 {mappedWarm.P99:F0})")}");

            foreach (long rowIndex in sampleRowIndices)
            {
                ReadOnlySpan<byte> mapped = rows.ResolveRow(rowIndex);
                byte[] independent = ReadRowFromFile(model, table, rowIndex); // a plain file read of the same addressed bytes is the parity basis for this mapping.

                if (!mapped.SequenceEqual(independent))
                {
                    Console.WriteLine($"  parity: row {rowIndex} differs between the mapped gather and an independent file read");
                    return 1;
                }
            }

            parityFacts["substrate"] = "mmap";
            parityFacts["rows_compared"] = sampleRowIndices.Count;
            parityFacts["bytes_compared_per_row"] = table.RowStrideBytes;
            parityFacts["byte_for_byte_equal"] = true;
        }

        // Arm C: the expensive-storage baseline — unbuffered sector-aligned reads of exactly those rows' containing windows.
        LatencyStats? unbufferedCold = null;
        long streamedBytes = 0;
        long pageFaultDelta = 0;

        if (substrate is "nocache" or "compare")
        {
            string ownerPath = OwningShardPath(model, table);
            IntPtr handle = WinColdReadInterop.CreateFileW(ownerPath, WinColdReadInterop.GenericRead, WinColdReadInterop.FileShareAll, IntPtr.Zero, WinColdReadInterop.OpenExisting, 0x80 | WinColdReadInterop.NoBuffering, IntPtr.Zero);

            if (handle == WinColdReadInterop.InvalidHandle)
            {
                Console.WriteLine($"ple-latency: CreateFileW('{ownerPath}', FILE_FLAG_NO_BUFFERING) failed (win32={Marshal.GetLastWin32Error()})");
                return 1;
            }

            try
            {
                long faultsBefore = WinColdReadInterop.PageFaultCountOfCurrentProcess(); // the baseline tallies every fault this arm must fetch a page for.
                List<double> samples = [];

                foreach (long rowIndex in touchedRows)
                {
                    long startTicks = Stopwatch.GetTimestamp();
                    streamedBytes += ReadRowWindowUnbuffered(handle, model, table, ownerPath, rowIndex);
                    samples.Add(Nanoseconds(startTicks, Stopwatch.GetTimestamp())); // each row costs its containing 4 KiB windows' worth of unbuffered transfer.
                }

                unbufferedCold = LatencyStats.Of(samples);
                pageFaultDelta = WinColdReadInterop.PageFaultCountOfCurrentProcess() - faultsBefore;
            }
            finally
            {
                WinColdReadInterop.CloseHandle(handle);
            }

            Console.WriteLine($"  substrate nocache: '{ownerPath}' read with FILE_FLAG_NO_BUFFERING — {streamedBytes:N0} streamed bytes, +{pageFaultDelta:N0} page faults");
        }

        // The residency ceiling answer: what the model's own PLE budget allows against how many rows this stream actually reached.
        long requestedRows = residentRows >= 0 ? residentRows : distinctRows;
        ResolvedBudgetChain pleChain = ModelBudgets.Resolve(new BudgetRequest
        {
            ModelName = Path.GetFileNameWithoutExtension(model.Path),
            Declared = [new BudgetChain { Resource = BudgetResource.Ple, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = pleCapMb } }],
            OpenedPoolTiers = [BudgetTier.Dram],
        }).For(BudgetResource.Ple)!;

        PleRowResidencyClamp? clamp = PleResidencyGate.Compute(pleChain, table.RowStrideBytes, requestedRows);
        string gateLine = clamp is null
            ? $"'{BudgetResourceNames.Ple}' residency: {requestedRows:N0} rows fit beside the {pleCapMb}-megabyte ceiling ({requestedRows * table.RowStrideBytes / PleResidencyGate.Megabytes:F1} MiB of packed rows)"
            : clamp.Describe();

        Console.WriteLine($"  budget gate: {gateLine}");

        notes.Add(new Dictionary<string, object> { ["note"] = "Storage and governance measurement only: hash/normalization cost, row addressing through the shard that holds the table, and residency accounting against a declared PLE ceiling. No forward pass runs and no model execution is attempted." });
        if (substrate == "compare")
        {
            notes.Add(new Dictionary<string, object> { ["note"] = "Substrate compare: mapped rows are byte-for-byte checked against independent sector-aligned file reads of the same offsets, and the unbuffered arm streams those rows' containing 4 KiB windows with no cached mapping held over them." });
        }

        string gatheredFrom = OwningShardPath(model, table); // recorded as the seam resolves it: which physical shard backs these rows.

        string reportDir = WriteReport(
            outRoot,
            args,
            modelPath,
            gatheredFrom,
            model.Shards?.Count ?? 1,
            table,
            tokens,
            reps,
            substrate,
            indexStats,
            mappedCold,
            mappedWarm,
            unbufferedCold,
            distinctRows,
            requestedRows,
            streamedBytes,
            pageFaultDelta,
            eosCutWindows,
            missingPredecessorWindows,
            clamp,
            parityFacts,
            notes);

        Console.WriteLine($"ple-latency report: {Path.GetFullPath(reportDir)}");
        return 0;
    }

    private static int[][] BuildHistories(int tokens, PleNgramTable table, out long eosCutWindows, out long missingPredecessorWindows)
    {
        ulong state = 0x2545F4914F6CDD1DUL; // a fixed LCG: the same token stream every run, so passes compare against each other.
        int[][] histories = new int[tokens][];
        eosCutWindows = 0;
        missingPredecessorWindows = 0;

        for (int i = 0; i < tokens; i++)
        {
            state = unchecked(state * 6364136223846793005UL + 1442695040888963407UL);
            int tokenId = (int)(state >> 33) % 1_000_000;

            // Every 97th position carries the model's own EOS id, and window lengths vary from one to the full n-gram: real cut and missing-history traffic against declared metadata.
            if (i % 97 == 0)
            {
                tokenId = table.EosTokenId;
            }

            int length = (int)((state >> 16) % (ulong)table.NgramSize) + 1;
            int[] window = new int[length];
            window[0] = tokenId;

            bool missing = false;

            for (int s = 1; s < length; s++)
            {
                state = unchecked(state * 6364136223846793005UL + 1442695040888963407UL);

                if (s % 5 == 0)
                {
                    window[s] = table.EosTokenId; // an EOS predecessor cuts everything at and behind it.
                }
                else
                {
                    window[s] = (int)(state >> 34) % 1_000_000;

                    if ((state & 0x3FUL) == 0)
                    {
                        window[s] = -1; // a predecessor with no cache cell reads as the reference's null token.
                        missing = true;
                    }
                }
            }

            histories[i] = window;

            if (window.Contains(table.EosTokenId))
            {
                eosCutWindows++;
            }

            if (missing)
            {
                missingPredecessorWindows++;
            }
        }

        return histories;
    }

    private static List<double> WarmSamples(GgufFile model, PleNgramTable table, HashSet<long> touchedRows, int reps)
    {
        using PleRowSource rows = new(model, table);
        List<double> samples = [];

        for (int rep = 1; rep < reps; rep++)
        {
            foreach (long rowIndex in touchedRows)
            {
                long startTicks = Stopwatch.GetTimestamp();
                ReadOnlySpan<byte> row = rows.ResolveRow(rowIndex);
                _ = Touch(row);
                samples.Add(Nanoseconds(startTicks, Stopwatch.GetTimestamp()));
            }
        }

        return samples;
    }

    private static int Touch(ReadOnlySpan<byte> row)
    {
        int checksum = 0;

        for (int i = 0; i < row.Length; i++)
        {
            checksum ^= row[i]; // every byte of the resolved row is read: this measures page traffic on the bytes a gather actually consumes.
        }

        return checksum;
    }

    private static List<long> PickSampleRows(HashSet<long> touchedRows, int count) =>
        touchedRows.Take(Math.Min(count, touchedRows.Count)).ToList();

    private static string OwningShardPath(GgufFile model, PleNgramTable table)
    {
        if (model.Shards is null)
        {
            return Path.GetFullPath(model.Path);
        }

        foreach (GgufFile shard in model.Shards)
        {
            if (shard.Tensors.Find(table.Tensor.Name) is not null)
            {
                return Path.GetFullPath(shard.Path);
            }
        }

        throw new InvalidOperationException($"no shard of '{model.Path}' declares tensor '{table.Tensor.Name}'");
    }

    private static byte[] ReadRowFromFile(GgufFile model, PleNgramTable table, long rowIndex)
    {
        string ownerPath = OwningShardPath(model, table);
        GgufFile owner = model.Shards is null ? model : FirstOwnerShard(model, table);

        using FileStream file = new(ownerPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1 << 16);
        long absolute = owner.DataStart + table.Tensor.Offset + rowIndex * table.RowStrideBytes;
        file.Seek(absolute, SeekOrigin.Begin);

        byte[] buffer = new byte[(int)table.RowStrideBytes];
        int filled = 0;

        while (filled < buffer.Length)
        {
            int read = file.Read(buffer, filled, buffer.Length - filled);

            if (read == 0)
            {
                throw new IOException($"independent read of row {rowIndex} stopped after {filled} bytes");
            }

            filled += read;
        }

        return buffer;
    }

    private static GgufFile FirstOwnerShard(GgufFile model, PleNgramTable table) =>
        model.Shards!.First(shard => shard.Tensors.Find(table.Tensor.Name) is not null);

    private static long ReadRowWindowUnbuffered(IntPtr handle, GgufFile model, PleNgramTable table, string ownerPath, long rowIndex)
    {
        GgufFile owner = model.Shards is null ? model : FirstOwnerShard(model, table);
        long absolute = owner.DataStart + table.Tensor.Offset + rowIndex * table.RowStrideBytes;

        // NO_BUFFERING transfers must start and end on sector boundaries: read the whole 4 KiB windows this row's packed bytes fall inside.
        long windowStart = absolute / SectorWindow * SectorWindow;
        long windowEnd = (absolute + table.RowStrideBytes + SectorWindow - 1) / SectorWindow * SectorWindow;
        int length = (int)(windowEnd - windowStart);

        if (!WinColdReadInterop.SetFilePointerEx(handle, windowStart, out long newOffset, 0) || newOffset != windowStart)
        {
            throw new IOException($"unbuffered read could not seek to {windowStart} in '{ownerPath}' (win32={Marshal.GetLastWin32Error()})");
        }

        byte[] buffer = new byte[length];

        GCHandle pinned = GCHandle.Alloc(buffer, GCHandleType.Pinned);

        try
        {
            int filled = 0;

            while (filled < length)
            {
                if (!WinColdReadInterop.ReadFile(handle, pinned.AddrOfPinnedObject() + filled, (uint)(length - filled), out uint read, IntPtr.Zero) || read == 0)
                {
                    throw new IOException($"unbuffered read of window [{windowStart}..{windowEnd}) in '{ownerPath}' stopped after {filled} bytes (win32={Marshal.GetLastWin32Error()})");
                }

                filled += (int)read; // NO_BUFFERING transfers can return short, sector-sized pieces.
            }
        }
        finally
        {
            pinned.Free();
        }

        return length;
    }

    private static double Nanoseconds(long startTicks, long endTicks) =>
        (endTicks - startTicks) * 1_000_000_000.0 / Stopwatch.Frequency;

    private static string WriteReport(
        string outRoot,
        string[] args,
        string modelPath,
        string gatheredFrom,
        int shardCount,
        PleNgramTable table,
        int tokens,
        int reps,
        string substrate,
        LatencyStats indexStats,
        LatencyStats? mappedCold,
        LatencyStats? mappedWarm,
        LatencyStats? unbufferedCold,
        long distinctRows,
        long requestedRows,
        long streamedBytes,
        long pageFaultDelta,
        long eosCutWindows,
        long missingPredecessorWindows,
        PleRowResidencyClamp? clamp,
        Dictionary<string, object?> parityFacts,
        List<Dictionary<string, object>> notes)
    {
        string stamp = DateTime.UtcNow.ToString("yyyy-MM-ddTHH-mm-ssZ");
        string directory = Path.Combine(outRoot, stamp);
        Directory.CreateDirectory(directory);

        JsonSerializerOptions json = new() { WriteIndented = true };

        Dictionary<string, object?> manifest = new()
        {
            ["command"] = "ple-latency",
            ["started_utc"] = stamp,
            ["argv"] = args,
            ["model_path"] = Path.GetFullPath(modelPath),
            ["split_shard_count"] = shardCount,
            ["gathered_from"] = gatheredFrom, // the seam's own answer to which physical file holds this tensor's payload.
        };

        Dictionary<string, object?> host = new()
        {
            ["machine"] = Environment.MachineName,
            ["os_description"] = RuntimeInformation.OSDescription,
            ["os_architecture"] = RuntimeInformation.OSArchitecture.ToString(),
            ["processor_count"] = Environment.ProcessorCount,
            ["total_memory_bytes_available_to_gc"] = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes,
        };

        Dictionary<string, object?> bench = new()
        {
            ["table_facts"] = new Dictionary<string, object?>
            {
                ["tensor_name"] = table.Tensor.Name,
                ["format"] = table.Tensor.Format.ToString(),
                ["logical_dims"] = table.Tensor.Shape.Dims,
                ["payload_bytes"] = table.Tensor.SizeBytes,
                ["ngram_size"] = table.NgramSize,
                ["heads_per_ngram"] = table.HeadsPerNgram,
                ["head_count"] = table.HeadCount,
                ["table_rows"] = table.TableRows,
                ["row_elements"] = table.RowElements,
                ["row_stride_bytes"] = table.RowStrideBytes,
                ["eos_token_id"] = table.EosTokenId,
            },
            ["run_shape"] = new Dictionary<string, object?> { ["tokens"] = tokens, ["reps"] = reps, ["substrate"] = substrate },
            ["history_traffic"] = new Dictionary<string, object?>
            {
                ["windows_containing_eos"] = eosCutWindows,
                ["windows_with_missing_predecessors"] = missingPredecessorWindows,
            },
            ["index_latency_ns_per_token"] = indexStats.ToDictionary(),
            ["mapped_row_latency_ns_first_touch"] = mappedCold?.ToDictionary(),
            ["mapped_row_latency_ns_warm"] = mappedWarm?.ToDictionary(),
            ["unbuffered_row_read_ns_cold"] = unbufferedCold?.ToDictionary(),
            ["reach"] = new Dictionary<string, object?>
            {
                ["distinct_rows_touched"] = distinctRows,
                ["requested_resident_rows"] = requestedRows,
                ["packed_bytes_of_requested_rows"] = requestedRows * table.RowStrideBytes,
                ["unbuffered_streamed_bytes"] = streamedBytes,
                ["page_fault_delta_during_unbuffered_arm"] = pageFaultDelta,
            },
            ["mapped_vs_file_parity"] = parityFacts.Count == 0 ? null : new Dictionary<string, object?>(parityFacts),
            ["residency_gate"] = clamp is null ? "no clamp: declared ceiling already fits the requested rows" : new Dictionary<string, object?>
            {
                ["binding_tier"] = BudgetTierNames.Of(clamp.Tier),
                ["cap_megabytes"] = clamp.CapMegabytes,
                ["row_bytes"] = clamp.RowBytes,
                ["requested_rows"] = clamp.RequestedRows,
                ["allowed_rows"] = clamp.AllowedRows,
            },
            ["notes"] = notes,
        };

        using (FileStream stream = File.Create(Path.Combine(directory, "manifest.json")))
        {
            JsonSerializer.Serialize(stream, manifest, json);
        }

        using (FileStream stream = File.Create(Path.Combine(directory, "host.json")))
        {
            JsonSerializer.Serialize(stream, host, json);
        }

        using (FileStream stream = File.Create(Path.Combine(directory, "bench.json")))
        {
            JsonSerializer.Serialize(stream, bench, json);
        }

        Dictionary<string, object?> cli = new() { ["argv"] = args };

        using (FileStream stream = File.Create(Path.Combine(directory, "cli.json")))
        {
            JsonSerializer.Serialize(stream, cli, json);
        }

        List<object> datasets =
        [
            new { label = "index ns/token", data = new[] { indexStats.P50, indexStats.P90, indexStats.P99 } },
        ];

        if (mappedCold is not null)
        {
            datasets.Add(new { label = "mapped first-touch ns/row", data = new[] { mappedCold.P50, mappedCold.P90, mappedCold.P99 } });
        }

        if (mappedWarm is not null)
        {
            datasets.Add(new { label = "mapped warm ns/row", data = new[] { mappedWarm.P50, mappedWarm.P90, mappedWarm.P99 } });
        }

        if (unbufferedCold is not null)
        {
            datasets.Add(new { label = "unbuffered cold ns/row", data = new[] { unbufferedCold.P50, unbufferedCold.P90, unbufferedCold.P99 } });
        }

        var chart = new
        {
            type = "bar",
            data = new { labels = new[] { "p50", "p90", "p99" }, datasets },
            options = new
            {
                plugins = new { legend = new { display = true } },
                scales = new { y = new { beginAtZero = true, title = new { display = true, text = "nanoseconds" } } },
            },
        };

        string chartJson = JsonSerializer.Serialize(chart);
        string parityLine = parityFacts.Count == 0
            ? "not run"
            : $"{parityFacts["rows_compared"]} rows x {parityFacts["bytes_compared_per_row"]} bytes/row — byte-for-byte equal: {parityFacts["byte_for_byte_equal"]}";
        string notesHtml = string.Join("", notes.Select(note => $"<li>{Esc(note["note"].ToString() ?? "")}</li>"));

        File.WriteAllText(
            Path.Combine(directory, "report.html"),
            $$"""
             <!DOCTYPE html>
             <html lang="en">
             <head>
             <meta charset="utf-8" />
             <title>ple-latency {{Esc(stamp)}}</title>
             <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
             <style>
             body { font-family: system-ui, sans-serif; margin: 2rem; color: #1c1c1c; max-width: 980px; }
             h1 { font-size: 1.35rem; } h2 { font-size: 1.1rem; margin-top: 1.6rem; }
             table { border-collapse: collapse; margin-top: .5rem; width: 100%; }
             th, td { border: 1px solid #bbb; padding: .4rem .7rem; text-align: left; font-size: .92rem; vertical-align: top; }
             th { background: #f0f0f0; } .ok { color: #0a7d32; font-weight: 600; } .bad { color: #b00020; font-weight: 600; }
             canvas { max-width: 860px; margin-top: 1rem; }
             .note { color: #555; font-size: .9rem; }
             code { background: #f4f4f4; padding: 0 .3rem; }
             </style>
             </head>
             <body>
             <h1>PLE n-gram gather latency</h1>
             <p class="note">{{table.NgramSize}}-gram x {{table.HeadsPerNgram}} heads = {{table.HeadCount}} rows/token over a {{table.TableRows:N0}}-row <code>{{Esc(table.Tensor.Name)}}</code> table ({{table.RowStrideBytes}} packed bytes/row, {{table.Tensor.Format}}).</p>

             <h2>Table facts</h2>
             <table>
             <tr><th>Tensor</th><td>{{Esc(table.Tensor.Name)}}</td></tr>
             <tr><th>Format / row width</th><td>{{table.Tensor.Format}} · {{table.RowElements}} elements · {{table.RowStrideBytes}} packed bytes/row</td></tr>
             <tr><th>Logical dims</th><td>{{string.Join(" x ", table.Tensor.Shape.Dims)}}</td></tr>
             <tr><th>Payload bytes</th><td>{{table.Tensor.SizeBytes:N0}}</td></tr>
             <tr><th>N-gram geometry</th><td>{{table.NgramSize}}-gram x {{table.HeadsPerNgram}} heads = {{table.HeadCount}} rows/token · EOS id {{table.EosTokenId}}</td></tr>
             </table>

             <h2>Run shape and history traffic</h2>
             <table>
             <tr><th>Tokens / reps / substrate</th><td>{{tokens:N0}} · {{reps:N0}} · {{Esc(substrate)}}</td></tr>
             <tr><th>Windows containing EOS</th><td>{{eosCutWindows:N0}}</td></tr>
             <tr><th>Windows with missing predecessors</th><td>{{missingPredecessorWindows:N0}}</td></tr>
             </table>

             <h2>Latency percentiles</h2>
             <table>
             <tr><th>Arm</th><th>p50</th><th>p90</th><th>p99</th><th>min</th><th>max</th></tr>
             <tr><td>index ns/token</td><td>{{indexStats.P50:F0}}</td><td>{{indexStats.P90:F0}}</td><td>{{indexStats.P99:F0}}</td><td>{{indexStats.Min:F0}}</td><td>{{indexStats.Max:F0}}</td></tr>
             <tr><td>mapped first-touch ns/row</td><td>{{(mappedCold is null ? "not run" : mappedCold.P50.ToString("F0"))}}</td><td>{{(mappedCold is null ? "not run" : mappedCold.P90.ToString("F0"))}}</td><td>{{(mappedCold is null ? "not run" : mappedCold.P99.ToString("F0"))}}</td><td>{{(mappedCold is null ? "not run" : mappedCold.Min.ToString("F0"))}}</td><td>{{(mappedCold is null ? "not run" : mappedCold.Max.ToString("F0"))}}</td></tr>
             <tr><td>mapped warm ns/row</td><td>{{(mappedWarm is null ? "not run" : mappedWarm.P50.ToString("F0"))}}</td><td>{{(mappedWarm is null ? "not run" : mappedWarm.P90.ToString("F0"))}}</td><td>{{(mappedWarm is null ? "not run" : mappedWarm.P99.ToString("F0"))}}</td><td>{{(mappedWarm is null ? "not run" : mappedWarm.Min.ToString("F0"))}}</td><td>{{(mappedWarm is null ? "not run" : mappedWarm.Max.ToString("F0"))}}</td></tr>
             <tr><td>unbuffered cold ns/row</td><td>{{(unbufferedCold is null ? "not run" : unbufferedCold.P50.ToString("F0"))}}</td><td>{{(unbufferedCold is null ? "not run" : unbufferedCold.P90.ToString("F0"))}}</td><td>{{(unbufferedCold is null ? "not run" : unbufferedCold.P99.ToString("F0"))}}</td><td>{{(unbufferedCold is null ? "not run" : unbufferedCold.Min.ToString("F0"))}}</td><td>{{(unbufferedCold is null ? "not run" : unbufferedCold.Max.ToString("F0"))}}</td></tr>
             </table>

             <canvas id="pleLatency" width="860" height="380"></canvas>
             <script>
             new Chart(document.getElementById('pleLatency'), {{chartJson}});
             </script>

             <h2>Reach and residency</h2>
             <table>
             <tr><th>Distinct rows touched</th><td>{{distinctRows:N0}}</td></tr>
             <tr><th>Requested resident rows</th><td>{{requestedRows:N0}}</td></tr>
             <tr><th>Packed bytes requested</th><td>{{requestedRows * table.RowStrideBytes:N0}}</td></tr>
             <tr><th>Unbuffered streamed bytes</th><td>{{streamedBytes:N0}}</td></tr>
             <tr><th>Page-fault delta during unbuffered arm</th><td>{{pageFaultDelta:N0}}</td></tr>
             <tr><th>Residency gate</th><td>{{Esc(clamp?.Describe() ?? "declared ceiling already fits the requested rows")}}</td></tr>
             </table>

             <h2>Parity</h2>
             <p class="note">{{Esc(parityLine)}}</p>

             <h2>Host</h2>
             <table>
             <tr><th>Machine / OS / arch</th><td>{{Esc(Environment.MachineName)}} · {{Esc(RuntimeInformation.OSDescription)}} · {{RuntimeInformation.OSArchitecture}}</td></tr>
             <tr><th>Processors / GC memory bytes</th><td>{{Environment.ProcessorCount}} · {{GC.GetGCMemoryInfo().TotalAvailableMemoryBytes:N0}}</td></tr>
             </table>

             <h2>Notes</h2>
             <ul class="note">{{notesHtml}}</ul>
             </body></html>
             """);

        return directory;
    }

    private static string Esc(string value) => System.Net.WebUtility.HtmlEncode(value);

    private sealed record LatencyStats(double P50, double P90, double P99, double Min, double Max)
    {
        /// <summary>Percentiles over one arm's samples: every reported value is a sample this run actually took.</summary>
        public static LatencyStats Of(List<double> samples)
        {
            if (samples.Count == 0)
            {
                return new LatencyStats(0, 0, 0, 0, 0);
            }

            double[] sorted = [.. samples];
            Array.Sort(sorted);

            return new LatencyStats(Percentile(sorted, 0.50), Percentile(sorted, 0.90), Percentile(sorted, 0.99), sorted[0], sorted[^1]);
        }

        private static double Percentile(double[] sorted, double fraction) =>
            sorted[Math.Min((int)Math.Ceiling(fraction * sorted.Length) - 1, sorted.Length - 1)];

        public Dictionary<string, object?> ToDictionary() => new()
        {
            ["p50"] = Math.Round(P50),
            ["p90"] = Math.Round(P90),
            ["p99"] = Math.Round(P99),
            ["min"] = Math.Round(Min),
            ["max"] = Math.Round(Max),
        };

        public string Describe() => $"p50 {P50:F0} p90 {P90:F0} p99 {P99:F0} max {Max:F0}";
    }

    private static int ReadInt(string[] args, ref int index, string option)
    {
        if (!int.TryParse(args[++index], out int value))
        {
            throw new FormatException($"ple-latency: {option} needs an integer");
        }

        return value;
    }

    private static long ReadLong(string[] args, ref int index, string option)
    {
        if (!long.TryParse(args[++index], out long value))
        {
            throw new FormatException($"ple-latency: {option} needs an integer");
        }

        return value;
    }
}
