using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

using Lair.Contracts;
using Lair.Loader;
using Lair.Models;

namespace Lair.Bench;

/// <summary>
/// PLE access-scheduling and placement measurement against a real pinned split model. The exact production n-gram hash and history normalization turn each token position into one row per head, then four policy arms run an identical stream over those rows: demand-only direct gather, computed-address prefetch with bounded in-flight depth for future rows, frequency-sketch hot-slot residency under a declared cap beside prefetch overlap, and the same
/// residency without any prefetch to separate what staying resident buys from what scheduling overlaps buy. On Linux the resident arena is transparent-huge-page backed anonymous memory admitted through libc mmap/madvise; placement arms bind gather threads to node cpulists, first-touch arenas on chosen nodes, and read results back off smaps/numa_maps. Hardware counters are reported as a limitation where the host has none — this lane never claims LLC or
/// stall-cycle numbers it could not count. Storage/governance evidence only: no forward pass runs and no qwen4exp model execution is attempted.
/// </summary>
internal static class PlePolicyBench
{
    private const int Megabytes = 1024 * 1024;

    public static int Run(string[] args)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("usage: ple-policy <gguf path> [--tokens N] [--reps R] [--caps MiB,MiB,...] [--inflight-rows D] [--overlap-depth K] [--token-dist uniform|zipf] [--vocab-window W] [--ple-cap-mb MB] [--numa] [--substrate-label TEXT] [--out DIR]");
            return 1;
        }

        string modelPath = args[1];
        int tokens = 65_536;
        int reps = 2;
        List<long> capsMb = [2, 8, 32];
        long inflightRows = 256; // DN-M7-11's own posture: keep ~256 PLE reads in flight rather than serializing them per token.
        int overlapDepth = 2;
        string tokenDist = "zipf";
        int vocabWindow = 4096;
        long pleCapMb = 256; // the declared models[].budgets.ple ceiling this measurement resolves its chains against.
        bool numaAxis = false;
        string substrateLabel = "unspecified";
        string outRoot = "reports";

        for (int i = 2; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--tokens": tokens = ReadInt(args, ref i, "--tokens"); break;
                case "--reps": reps = ReadInt(args, ref i, "--reps"); break;
                case "--caps":
                    capsMb.Clear();

                    foreach (string part in args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                    {
                        if (!long.TryParse(part, out long cap) || cap < 1 || cap > 4096)
                        {
                            Console.WriteLine("ple-policy: --caps takes whole megabytes between 1 and 4096");
                            return 1;
                        }

                        capsMb.Add(cap);
                    }

                    if (capsMb.Count == 0)
                    {
                        Console.WriteLine("ple-policy: --caps listed no ceilings");
                        return 1;
                    }

                    break;
                case "--inflight-rows": inflightRows = ReadLong(args, ref i, "--inflight-rows"); break;
                case "--overlap-depth": overlapDepth = ReadInt(args, ref i, "--overlap-depth"); break;
                case "--token-dist": tokenDist = args[++i]; break;
                case "--vocab-window": vocabWindow = ReadInt(args, ref i, "--vocab-window"); break;
                case "--ple-cap-mb": pleCapMb = ReadLong(args, ref i, "--ple-cap-mb"); break;
                case "--numa": numaAxis = true; break;
                case "--substrate-label": substrateLabel = args[++i]; break;
                case "--out": outRoot = args[++i]; break;
                default:
                    Console.WriteLine($"ple-policy: unknown option '{args[i]}'");
                    return 1;
            }
        }

        if (tokens is < 64 or > 2_097_152)
        {
            Console.WriteLine("ple-policy: --tokens must be between 64 and 2097152 — this lane stays bounded to one token stream");
            return 1;
        }

        if (reps is < 2 or > 8)
        {
            Console.WriteLine("ple-policy: --reps must be between 2 and 8 so a first-touch pass and at least one warm pass both get reported");
            return 1;
        }

        if (inflightRows is < 16 or > 4096)
        {
            Console.WriteLine("ple-policy: --inflight-rows must be between 16 and 4096 — the bounded in-flight depth this arm measures");
            return 1;
        }

        if (overlapDepth is < 1 or > 8)
        {
            Console.WriteLine("ple-policy: --overlap-depth must be between 1 and 8 outstanding prefetch windows");
            return 1;
        }

        if (tokenDist is not ("uniform" or "zipf"))
        {
            Console.WriteLine("ple-policy: --token-dist must be uniform or zipf");
            return 1;
        }

        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"ple-policy: model file '{modelPath}' was not found");
            return 1;
        }

        using GgufFile model = GgufFile.Open(modelPath); // an assembled split opens whole; the PLE tensor's own shard is what every arm maps.
        LogicalTensor? declared = model.Tensors.Find("per_layer_token_embd.weight");

        if (declared is null)
        {
            Console.WriteLine($"ple-policy: '{model.Path}' declares no 'per_layer_token_embd.weight' tensor, so there is no PLE table to gather from");
            return 1;
        }

        PleNgramTable table = PleNgramTable.FromGguf(model, declared); // every refusal this seam raises names the model file, tensor, and metadata field at fault.
        string gatheredFromShard = OwningShardPath(model, table);

        Console.WriteLine($"ple-policy: {table.NgramSize}-gram x {table.HeadsPerNgram} heads = {table.HeadCount} rows/token over a {table.TableRows:N0}-row '{table.Tensor.Name}' table " +
            $"({table.RowElements} elements, {table.RowStrideBytes} packed bytes/row, format {table.Tensor.Format}); gathered from shard '{gatheredFromShard}'");

        int[][] histories = BuildHistories(tokens, table, vocabWindow, tokenDist, out long eosCutWindows, out long missingPredecessorWindows);
        Console.WriteLine($"  stream: {tokens:N0} tokens · distribution={tokenDist}{(tokenDist == "zipf" ? $" vocab_window={vocabWindow}" : string.Empty)} · reps={reps} · caps={(string.Join(",", capsMb.Select(c => $"{c} MiB")))}");

        // One untimed sketch pass over the identical stream every arm replays: it fixes both the distinct-row reach and the priority order hot-slot selection admits under each cap.
        PleFrequencySketch sketch = new();
        long[] scratchRows = new long[table.HeadCount];

        foreach (int[] history in histories)
        {
            table.ComputeRowIndices(history, scratchRows); // the exact production normalization: prefetch and residency never compute addresses differently from demand.

            foreach (long row in scratchRows)
            {
                sketch.Observe(row);
            }
        }

        List<long> ranked = sketch.RankedByFrequency();
        Console.WriteLine($"  sketch: {sketch.DistinctRows:N0} distinct rows over {sketch.TotalObservations:N0} gathered positions");

        ResolvedBudgetChain pleChain = ModelBudgets.Resolve(new BudgetRequest
        {
            ModelName = Path.GetFileNameWithoutExtension(model.Path),
            Declared = [new BudgetChain { Resource = BudgetResource.Ple, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = pleCapMb } }],
            OpenedPoolTiers = [BudgetTier.Dram],
        }).For(BudgetResource.Ple)!;

        PleRowResidencyClamp? declaredGate = PleResidencyGate.Compute(pleChain, table.RowStrideBytes, sketch.DistinctRows); // the model's own chain answer for what this stream reached.
        Console.WriteLine($"  budget gate: {declaredGate?.Describe() ?? $"'{BudgetResourceNames.Ple}' residency: {sketch.DistinctRows:N0} rows fit beside the {pleCapMb}-megabyte ceiling"}");

        List<Dictionary<string, object?>> armReports = [];
        Dictionary<string, object?>? numaFacts = null;

        foreach (long cap in capsMb)
        {
            long allowedByClamp = PleResidencyGate.FitsRows(cap, table.RowStrideBytes);
            Console.WriteLine($"  cap {cap} MiB allows {allowedByClamp:N0} resident rows at a {table.RowStrideBytes}-byte stride (selection ranked {ranked.Count:N0})");

            armReports.Add(StampArm(MeasureDemand("demand-only", model, table, histories, reps), cap));
            armReports.Add(StampArm(MeasureDemand("prefetch-overlap", model, table, histories, reps, inflightRows: inflightRows, overlapDepth: overlapDepth), cap));
            armReports.Add(StampArm(MeasureHot("hot-resident-prefetch", model, table, histories, reps, cap * Megabytes, ranked, usePrefetch: true, inflightRows: inflightRows, overlapDepth: overlapDepth), cap));
            armReports.Add(StampArm(MeasureHot("hot-resident-no-prefetch", model, table, histories, reps, cap * Megabytes, ranked, usePrefetch: false), cap));

            if (numaAxis && cap == capsMb[^1]) // the placement axis runs once against the widest cap so residency holds its full admitted set while threads move.
            {
                numaFacts = RunNumaPlacement(model, table, histories, reps, cap * Megabytes, ranked);
            }

            Console.WriteLine();
        }

        List<Dictionary<string, object?>> notes =
        [
            Note("Arms simulate layer-2 access scheduling over the real table's row addressing with a synthetic token stream; production qwen4exp forward integration is not implemented and no claim here covers it."),
            Note($"The stream uses {tokenDist} token ids ({(tokenDist == "zipf" ? $"a cubed-uniform skew over a {vocabWindow}-id vocabulary window concentrates row frequencies so hot-slot selection has something to select" : "uniform ids across 1e6..2e6, the no-skew control")}) with EOS cut and missing-predecessor windows mixed in; policy comparisons hold this stream identical across arms."),
            Note("hot-resident-prefetch versus hot-resident-no-prefetch isolates what staying resident buys from what prefetch overlap buys; demand-only is the floor of both."),
        ];

        string reportDir = WriteReport(outRoot, args, modelPath, gatheredFromShard, table, tokens, reps, capsMb, inflightRows, overlapDepth, tokenDist, vocabWindow, pleCapMb, substrateLabel, sketch.DistinctRows, eosCutWindows, missingPredecessorWindows, declaredGate, armReports, numaFacts, notes);

        Console.WriteLine($"ple-policy report: {Path.GetFullPath(reportDir)}");
        return 0;
    }

    private static Dictionary<string, object?> Note(string text) => new() { ["note"] = text };

    /// <summary>A demand arm through the cold shard mapping, with or without a bounded computed-address prefetch pipeline beside it.</summary>
    private static Dictionary<string, object?> MeasureDemand(
        string armName, GgufFile model, PleNgramTable table, int[][] histories, int reps, long inflightRows = 0, int overlapDepth = 1)
    {
        using PleRowSource cold = new(model, table);

        Console.WriteLine($"  arm {armName}: mapping '{cold.ShardPath}'{(inflightRows > 0 ? $" prefetch=inflight_rows<={inflightRows} windows<={overlapDepth}" : " demand-only")}");

        return MeasureCore(armName, cold, provider: null, table, histories, reps, inflightRows, overlapDepth);
    }

    /// <summary>A hot-slot arm: the sketch's priority order is admitted under the cap first (admission cost reported apart from gather timing), then the identical stream runs through residency with or without prefetch overlap.</summary>
    private static Dictionary<string, object?> MeasureHot(
        string armName, GgufFile model, PleNgramTable table, int[][] histories, int reps, long capBytes, List<long> ranked, bool usePrefetch, long inflightRows = 0, int overlapDepth = 1)
    {
        using PleRowSource cold = new(model, table);

        Console.WriteLine($"  arm {armName}: arena capacity {capBytes / Megabytes} MiB on {(OperatingSystem.IsLinux() ? "anonymous mmap + MADV_HUGEPAGE ask" : "plain committed (non-Linux host: the THP ask is not emulated)")}");

        using PleHotRowProvider provider = new(cold, capBytes, OperatingSystem.IsLinux() ? PleHotRowBackMode.TransparentHugePage : PleHotRowBackMode.PlainCommitted);
        PleHotRowAdmission admission = provider.AdmitHotSet(ranked); // the sketch's priority order decides who gets clamped out.

        Console.WriteLine($"    admission — {admission.Describe()}");
        Console.WriteLine($"    arena copies: {provider.CopiedBytesTotal:N0} bytes first-touched across {provider.ParityCheckedRows:N0} parity-checked rows (madvise applied: {provider.BackingFacts["madvise_hugepage_succeeded"]})");

        Dictionary<string, object?> backingFacts = new(provider.BackingFacts);
        Dictionary<string, object?>? coverage = provider.ReadBackingMeasurement(); // smaps/numa_maps facts about what actually landed in residency.

        if (coverage is not null)
        {
            foreach ((string key, object? value) in coverage)
            {
                backingFacts[key] = value;
            }
        }

        Dictionary<string, object?> result = MeasureCore(armName, cold, provider, table, histories, reps, usePrefetch ? inflightRows : 0, overlapDepth);
        result["admission"] = new Dictionary<string, object?>
        {
            ["requested_rows"] = admission.RequestedRows,
            ["allowed_rows"] = admission.AllowedRows,
            ["admitted_rows"] = admission.AdmittedRows,
            ["refused_by_clamp_rows"] = admission.RefusedByClampRows,
            ["resident_bytes"] = admission.ResidentBytes,
        };
        result["arena_backing_facts"] = backingFacts;
        return result;
    }

    /// <summary>The measurement every arm shares: first rep reports cold first-touch percentiles with fault deltas, later reps warm against the same mapping. Prefetch windows are issued ahead of demand and drained inside the timed wall so overlap is measured as it actually pays — never after a pass that already waited for everything.</summary>
    private static Dictionary<string, object?> MeasureCore(
        string armName, PleRowSource cold, PleHotRowProvider? provider, PleNgramTable table, int[][] histories, int reps, long inflightRows, int overlapDepth)
    {
        List<double> firstTouchSamples = [];
        List<double> warmSamples = [];
        List<Dictionary<string, object?>> passFacts = [];

        PrefetchPipeline? pipeline = inflightRows > 0 ? new PrefetchPipeline(cold, table, histories, provider, inflightRows, overlapDepth) : null; // a residency-aware prefetcher skips rows that already live in the arena.

        long perPassResolutions = (long)histories.Length * table.HeadCount;
        double warmWallSeconds = 0;

        for (int rep = 0; rep < reps; rep++)
        {
            FaultSnapshot before = ReadFaults();
            Stopwatch wall = Stopwatch.StartNew();

            List<double> samples = rep == 0 ? firstTouchSamples : warmSamples;
            long[] headRows = new long[table.HeadCount]; // one scratch per pass: the gather writes every head row of a position into it exactly as production does.

            for (int tokenIndex = 0; tokenIndex < histories.Length; tokenIndex++)
            {
                pipeline?.ScheduleBefore(tokenIndex);

                long startTicks = Stopwatch.GetTimestamp();

                table.ComputeRowIndices(histories[tokenIndex], headRows); // index cost rides in the gather step: a layer-2 stall measurement must charge it like any other consumer would.

                foreach (long row in headRows)
                {
                    ReadOnlySpan<byte> resolved = provider is not null ? provider.ResolveRow(row) : cold.ResolveRow(row);
                    _ = Touch(resolved); // the span's own bytes are read so this counts real page traffic, not address arithmetic.
                }

                samples.Add(Nanoseconds(startTicks, Stopwatch.GetTimestamp()));
            }

            pipeline?.Drain(); // outstanding windows finish inside the timed wall: overlap that never had to wait is exactly what a policy arm pays for.

            wall.Stop();

            FaultSnapshot after = ReadFaults();

            if (rep > 0)
            {
                warmWallSeconds += wall.Elapsed.TotalSeconds;
            }

            passFacts.Add(new Dictionary<string, object?>
            {
                ["pass"] = rep == 0 ? "first_touch" : $"warm_{rep}",
                ["wall_seconds"] = Math.Round(wall.Elapsed.TotalSeconds, 6),
                ["minor_fault_delta"] = after.Minor - before.Minor,
                ["major_fault_delta"] = after.Major - before.Major,
            });

            Console.WriteLine($"    pass {passFacts[^1]["pass"]}: wall {wall.Elapsed.TotalSeconds:F3}s · faults +{(after.Minor - before.Minor):N0} minor / +{(after.Major - before.Major):N0} major");
        }

        long coldResolutions = provider is null ? perPassResolutions * reps : provider.MissesTotal; // a demand arm read shard bytes for every resolution; a hot arm paid them only on its misses.
        long streamedBytesEstimate = (coldResolutions + pipeline?.RowsTouched ?? 0) * table.RowStrideBytes;

        Dictionary<string, object?> facts = new()
        {
            ["arm"] = armName,
            ["gather_latency_ns_per_token_first_touch"] = LatencyStats.Of(firstTouchSamples).ToDictionary(),
            ["gather_latency_ns_per_token_warm"] = LatencyStats.Of(warmSamples).ToDictionary(),
            ["warm_tokens_per_second_aggregate"] = warmWallSeconds > 0 ? Math.Round(warmSamples.Count / warmWallSeconds, 2) : null,
            ["row_resolutions_total_all_passes"] = perPassResolutions * reps,
            ["cold_row_bytes_estimated_streamed"] = streamedBytesEstimate,
            ["passes"] = passFacts,
        };

        if (pipeline is not null)
        {
            facts["prefetch_facts"] = new Dictionary<string, object?>
            {
                ["inflight_rows_cap_per_window"] = inflightRows,
                ["overlap_depth_windows"] = overlapDepth,
                ["lookahead_tokens_per_window"] = pipeline.LookaheadTokens,
                ["windows_issued"] = pipeline.WindowsIssued,
                ["rows_touched_by_prefetchers"] = pipeline.RowsTouched,
                ["madvise_willneed_attempts"] = pipeline.MadviseAttempts,
                ["madvise_willneed_accepted_ranges"] = pipeline.MadviseApplied,
            };
        }

        if (provider is not null)
        {
            long hits = provider.HitsTotal;
            long misses = provider.MissesTotal;

            facts["residency_tallies"] = new Dictionary<string, object?>
            {
                ["hits_total_all_passes"] = hits,
                ["misses_total_all_passes"] = misses,
                ["hit_rate"] = Math.Round((double)hits / (hits + misses), 4),
                ["resident_bytes_held"] = provider.ResidentBytes,
            };
        }

        Console.WriteLine($"    arm {armName}: warm p50 {LatencyStats.Of(warmSamples).P50:F0} ns/token · first-touch p50 {LatencyStats.Of(firstTouchSamples).P50:F0} ns/token" +
            (provider is not null ? $" · hits {provider.HitsTotal:N0} / misses {provider.MissesTotal:N0}" : string.Empty));

        return facts;
    }

    /// <summary>Placement variants over the hot-slot arm: unbound control, arena and gather both bound to one node, and a remote variant whose arena first-touch landed on another node while gathers run local — read back through affinity verification plus smaps/numa_maps coverage.</summary>
    private static Dictionary<string, object?> RunNumaPlacement(
        GgufFile model, PleNgramTable table, int[][] histories, int reps, long capBytes, List<long> ranked)
    {
        if (!OperatingSystem.IsLinux())
        {
            Console.WriteLine("  numa axis: skipped — placement primitives exist on Linux hosts only");

            return new Dictionary<string, object?> { ["skipped_reason"] = "placement measurement is a Linux seam; this host reports none" };
        }

        List<(int Node, string CpuList)> nodes = LinuxPlacementInterop.DetectNumaNodes();

        if (nodes.Count == 0)
        {
            Console.WriteLine("  numa axis: skipped — sysfs reported no NUMA node directories to bind against");

            return new Dictionary<string, object?> { ["skipped_reason"] = "no /sys/devices/system/node entries readable on this host" };
        }

        nodes.Sort((left, right) => left.Node.CompareTo(right.Node));

        Console.WriteLine($"  numa axis: nodes detected — {(string.Join(", ", nodes.Select(n => $"node{n.Node} cpulist {n.CpuList}")))}");

        Dictionary<string, object?> nodesFacts = new();

        foreach ((int node, string cpuList) in nodes)
        {
            nodesFacts[$"node{node}"] = cpuList;
        }

        List<Dictionary<string, object?>> variants = [];
        int localNode = nodes[0].Node;
        int remoteNode = nodes.Count > 1 ? nodes[^1].Node : -1; // a single-node host cannot express "remote": that variant is absent rather than faked.

        foreach ((string label, int arenaBindNode, int gatherBindNode) in PlacementVariants(localNode, remoteNode))
        {
            Console.WriteLine($"    variant {label}: arena first-touch node={arenaBindNode} gather-bound node={gatherBindNode}");

            variants.Add(MeasurePlacementVariant(label, model, table, histories, reps, capBytes, ranked, arenaBindNode, gatherBindNode));
        }

        return new Dictionary<string, object?>
        {
            ["nodes_detected"] = nodesFacts,
            ["variants"] = variants,
        };
    }

    private static List<(string Label, int ArenaBindNode, int GatherBindNode)> PlacementVariants(int localNode, int remoteNode)
    {
        List<(string, int, int)> variants = [("unbound", -1, -1), ($"local-node{localNode}", localNode, localNode)];

        if (remoteNode >= 0 && remoteNode != localNode)
        {
            variants.Add(($"arena-on-node{remoteNode}-gather-on-node{localNode}", remoteNode, localNode)); // the cross-node posture DN-M314-17's local preference is meant to avoid.
        }

        return [.. variants];
    }

    private static Dictionary<string, object?> MeasurePlacementVariant(
        string label, GgufFile model, PleNgramTable table, int[][] histories, int reps, long capBytes, List<long> ranked, int arenaBindNode, int gatherBindNode)
    {
        using PleRowSource cold = new(model, table);

        Dictionary<string, object?> placementFacts = [];

        // Admission runs on a thread whose affinity and memory policy name the arena node: first-touch during copy is what decides where residency physically lives.
        Task<PleHotRowProvider> admissionTask = Task.Run(() =>
        {
            if (arenaBindNode >= 0)
            {
                PlacementAttempt bind = LinuxPlacementInterop.BindCurrentThreadToNode(arenaBindNode);
                placementFacts["arena_thread_bind"] = bind.Description;

                PlacementAttempt policy = LinuxPlacementInterop.TryBindMemoryPolicyToNode(arenaBindNode); // an attempted MPOL_BIND beside first-touch discipline: refused attempts are reported, not hidden.
                placementFacts["arena_memory_policy"] = policy.Description;
            }
            else
            {
                placementFacts["arena_thread_bind"] = "unbound control (" + LinuxPlacementInterop.ClearMemoryPolicy().Description + ")";
            }

            PleHotRowProvider provider = new(cold, capBytes);
            provider.AdmitHotSet(ranked); // first-touch lands here, on the bound thread.
            return provider;
        });

        admissionTask.Wait();
        using PleHotRowProvider provider = admissionTask.Result;

        if (gatherBindNode >= 0)
        {
            PlacementAttempt bind = LinuxPlacementInterop.BindCurrentThreadToNode(gatherBindNode); // binding the measuring thread itself: prefetch workers re-bind inside their own delegates rather than inheriting this.
            placementFacts["gather_thread_bind"] = bind.Description;
        }

        placementFacts["gather_thread_cpu_list_after_binding"] = LinuxPlacementInterop.ReadCurrentThreadCpuList();

        Dictionary<string, object?>? coverage = provider.ReadBackingMeasurement(); // where the admitted pages actually landed, per node.
        placementFacts["arena_coverage_after_admission"] = coverage;

        Console.WriteLine($"      admission: {provider.Admission?.Describe()}");

        if (coverage is not null)
        {
            Console.WriteLine($"      arena coverage smaps AnonHugePages={coverage["smaps_anon_hugepages_bytes_over_arena"]} rss={coverage["smaps_rss_bytes_over_arena"]} numa_maps entries={coverage["numa_maps_entries_over_arena"]}");

            if (coverage["numa_pages_by_node"] is Dictionary<string, object?> pagesByNode)
            {
                Console.WriteLine($"      arena pages by node: {string.Join(", ", pagesByNode.Select(pair => $"{pair.Key}={pair.Value}"))}");
            }
        }

        Dictionary<string, object?> facts = MeasureCore($"placement:{label}", cold, provider, table, histories, reps, inflightRows: 256, overlapDepth: 2);
        facts["numa_placement_facts"] = placementFacts;

        if (gatherBindNode >= 0)
        {
            LinuxPlacementInterop.ClearMemoryPolicy(); // reset posture so the next variant starts from the kernel default rather than inheriting this binding.
        }

        return facts;
    }

    private static int[][] BuildHistories(int tokens, PleNgramTable table, int vocabWindow, string distribution, out long eosCutWindows, out long missingPredecessorWindows)
    {
        ulong state = 0x2545F4914F6CDD1DUL; // a fixed LCG: the same token stream every run and every arm, so policies compare against each other.
        int[][] histories = new int[tokens][];
        eosCutWindows = 0;
        missingPredecessorWindows = 0;

        for (int i = 0; i < tokens; i++)
        {
            state = unchecked(state * 6364136223846793005UL + 1442695040888963407UL);

            int tokenId = distribution == "zipf" ? ZipfTokenId(ref state, vocabWindow) : UniformTokenId(ref state);

            // Every 97th position carries the model's own EOS id: real cut traffic against declared metadata, same posture ple-latency took.
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
                    window[s] = distribution == "zipf" ? ZipfTokenId(ref state, vocabWindow) : UniformTokenId(ref state);

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

    private static int UniformTokenId(ref ulong state) => 1_000_000 + (int)((state >> 33) % 1_000_000); // a no-skew control: nothing concentrates, so hot-slot selection earns its residency only from breadth.

    private static int ZipfTokenId(ref ulong state, int vocabWindow)
    {
        double unit = (double)(state >> 11) / (double)(ulong.MaxValue >> 11); // cubed-uniform skew over the vocabulary window: low ranks dominate traffic the way real n-gram reuse does.
        int rank = (int)(Math.Pow(unit, 3) * vocabWindow);

        return Math.Clamp(rank, 0, vocabWindow - 1);
    }

    /// <summary>One bounded prefetch pipeline: future windows' exact computed rows are touched on worker threads while demand consumes the current window, with at most overlapDepth windows outstanding and inflightRows distinct row touches per window.</summary>
    private sealed class PrefetchPipeline
    {
        private readonly PleRowSource _cold;
        private readonly PleNgramTable _table;
        private readonly int[][] _histories;
        private readonly PleHotRowProvider? _provider; // residency-aware prefetch skips rows that already live in the arena.

        private readonly long _inflightRows;
        private readonly int _overlapDepth;

        private readonly List<Task> _pending = [];

        public PrefetchPipeline(PleRowSource cold, PleNgramTable table, int[][] histories, PleHotRowProvider? provider, long inflightRows, int overlapDepth)
        {
            _cold = cold;
            _table = table;
            _histories = histories;
            _provider = provider;
            _inflightRows = inflightRows;
            _overlapDepth = overlapDepth;

            LookaheadTokens = Math.Max(1, (int)Math.Min(int.MaxValue - 2, inflightRows / Math.Max(1, table.HeadCount))); // the window a row cap translates into: rows in flight divided by heads per token.
        }

        public int LookaheadTokens { get; }

        public long WindowsIssued;

        public long RowsTouched;

        public long MadviseAttempts;

        public long MadviseApplied;

        /// <summary>Issue the next future window at every lookahead boundary, holding at most overlapDepth windows outstanding — the bounded in-flight discipline itself.</summary>
        public void ScheduleBefore(int tokenIndex)
        {
            if (tokenIndex % LookaheadTokens != 0)
            {
                return;
            }

            int from = tokenIndex + LookaheadTokens; // prefetch always runs ahead of demand: the window being touched is never the one being measured.

            if (from >= _histories.Length)
            {
                return;
            }

            while (_pending.Count >= _overlapDepth)
            {
                _pending[0].Wait(); // a full pipeline waits for its oldest window instead of letting unbounded speculative work pile up.
                _pending.RemoveAt(0);
            }

            int to = Math.Min(_histories.Length, from + LookaheadTokens);
            _pending.Add(Task.Run(() => TouchWindow(from, to)));
            WindowsIssued++;
        }

        public void Drain()
        {
            foreach (Task task in _pending)
            {
                task.Wait(); // drain inside the timed wall: overlap is measured as it pays, not after a pass that already waited for everything.
            }

            _pending.Clear();
        }

        private void TouchWindow(int from, int to)
        {
            long[] scratch = new long[_table.HeadCount];
            HashSet<long> distinctRows = []; // one window's own dedup: a row touched twice by lookahead buys nothing.
            long lowestAddress = long.MaxValue;
            long highestAddress = 0L;

            for (int tokenIndex = from; tokenIndex < to; tokenIndex++)
            {
                _table.ComputeRowIndices(_histories[tokenIndex], scratch); // the same exact hash/history normalization demand uses — computed addresses, never guessed ones.

                foreach (long row in scratch)
                {
                    if (_provider is not null && _provider.IsResident(row))
                    {
                        continue; // a resident future row needs no speculative shard read: prefetching it would measure the arena twice.
                    }

                    if (distinctRows.Contains(row) || distinctRows.Count >= _inflightRows)
                    {
                        continue;
                    }

                    distinctRows.Add(row);

                    if (_cold.TryMappedAddressOf(row, out nint mappedAddress))
                    {
                        long address = (long)mappedAddress;
                        lowestAddress = Math.Min(lowestAddress, address);
                        highestAddress = Math.Max(highestAddress, address);
                    }
                }

                if (distinctRows.Count >= _inflightRows)
                {
                    break; // the window's own in-flight cap: rows past it wait for the next window rather than crowding this one.
                }
            }

            if (distinctRows.Count == 0)
            {
                return;
            }

            if (OperatingSystem.IsLinux() && lowestAddress <= highestAddress)
            {
                long pageFloorStart = (long)lowestAddress & ~(long)(4096 - 1); // one OS prefetch call over the window's whole covered span: DN-M7-11's posture, not a per-row hint storm.
                long ceilEnd = ((long)highestAddress + 4095) & ~(long)(4096 - 1);

                Interlocked.Increment(ref MadviseAttempts);

                if (LinuxPlacementInterop.MadviseWillNeed((nint)pageFloorStart, checked((nuint)(ceilEnd - pageFloorStart))))
                {
                    Interlocked.Increment(ref MadviseApplied);
                }
            }

            foreach (long row in distinctRows)
            {
                _ = _cold.ResolveRow(row)[0]; // the speculative demand fault itself: overlap is what races against the measured gather step.
                Interlocked.Increment(ref RowsTouched);
            }
        }
    }

    private static Dictionary<string, object?> StampArm(Dictionary<string, object?> facts, long capMb)
    {
        facts["cap_megabytes"] = capMb; // arms run per declared ceiling so clamp effects and policy effects stay attributable to the same number.
        return facts;
    }

    private static double Nanoseconds(long startTicks, long endTicks) =>
        (endTicks - startTicks) * 1_000_000_000.0 / Stopwatch.Frequency;

    private static int Touch(ReadOnlySpan<byte> row)
    {
        int checksum = 0;

        for (int i = 0; i < row.Length; i++)
        {
            checksum ^= row[i]; // every byte of the resolved row is read: this measures page traffic on the bytes a gather actually consumes.
        }

        return checksum;
    }

    private readonly record struct FaultSnapshot(long Minor, long Major);

    private static FaultSnapshot ReadFaults() => OperatingSystem.IsLinux()
        ? ToSnapshot(LinuxPlacementInterop.ReadMinorMajorFaultsOfCurrentProcess())
        : new FaultSnapshot(WindowsPageFaultCount(), 0L); // Windows exposes one combined counter today: the minor/major split stays a Linux-arm fact.

    private static FaultSnapshot ToSnapshot((long MinorFaults, long MajorFaults) faults) => new(faults.Item1, faults.Item2);

    private static long WindowsPageFaultCount()
    {
        try
        {
            return WinColdReadInterop.PageFaultCountOfCurrentProcess(); // the same psapi posture ple-latency's nocache arm used.
        }
        catch (Exception)
        {
            return 0L; // a host where psapi refuses falls back to streamed-byte tallies alone, labeled as such in the report.
        }
    }

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

    private static string WriteReport(
        string outRoot, string[] args, string modelPath, string gatheredFromShard, PleNgramTable table, int tokens, int reps, List<long> capsMb,
        long inflightRows, int overlapDepth, string tokenDist, int vocabWindow, long pleCapMb, string substrateLabel, long distinctRowsReached,
        long eosCutWindows, long missingPredecessorWindows, PleRowResidencyClamp? declaredGate, List<Dictionary<string, object?>> armReports,
        Dictionary<string, object?>? numaFacts, List<Dictionary<string, object?>> notes)
    {
        string stamp = DateTime.UtcNow.ToString("yyyy-MM-ddTHH-mm-ssZ");
        string directory = Path.Combine(outRoot, stamp);
        Directory.CreateDirectory(directory);

        Dictionary<string, object?> manifest = new()
        {
            ["schemaVersion"] = 1,
            ["reportId"] = $"ple-policy-{stamp}",
            ["generatedAtUtc"] = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"),
            ["phase"] = "MVP7",
            ["lane"] = OperatingSystem.IsLinux() ? "CPU-only PLE access-scheduling/placement measurement (huge-page hot-row provider + NUMA arms)" : "PLE access-scheduling measurement on a non-Linux host: plain-committed residency, no placement claims",
            ["goal"] = "Compare demand-only gather against computed-address prefetch overlap and frequency-sketch hot-slot residency with/without prefetch under declared PLE caps; measure huge-page backing and NUMA-local versus remote placement where the platform offers them.",
            ["command_line"] = string.Join(' ', args),
            ["working_directory"] = Path.GetFullPath(Directory.GetCurrentDirectory()),
            ["model_path"] = Path.GetFullPath(modelPath),
            ["gathered_from_shard"] = gatheredFromShard, // recorded the way the seam itself resolves which physical file holds this tensor's payload.
            ["substrate_label"] = substrateLabel,
            ["files"] = new[] { "manifest.json", "host.json", "bench.json", "cli.json", "report.html" },
        };

        Dictionary<string, object?> host = new()
        {
            ["machine"] = Environment.MachineName,
            ["os_description"] = RuntimeInformation.OSDescription,
            ["os_architecture"] = RuntimeInformation.OSArchitecture.ToString(),
            ["processor_count"] = Environment.ProcessorCount,
            ["total_memory_bytes_available_to_gc"] = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes,
        };

        if (OperatingSystem.IsLinux())
        {
            host["hugepage_facts"] = LinuxPlacementInterop.ReadHugePageFacts();
            host["hardware_counter_facts"] = LinuxPlacementInterop.ReadHardwareCounterFacts();
            host["numa_nodes_from_sysfs"] = ToNodeDictionary(LinuxPlacementInterop.DetectNumaNodes());
        }

        Dictionary<string, object?> tableFacts = new()
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
        };

        Dictionary<string, object?> gateFacts = declaredGate is null
            ? new Dictionary<string, object?> { ["clamp"] = "none", ["text"] = "the declared PLE ceiling already fits every row this stream reached" }
            : new Dictionary<string, object?>
            {
                ["clamp"] = "binding",
                ["binding_tier"] = BudgetTierNames.Of(declaredGate.Tier),
                ["cap_megabytes"] = declaredGate.CapMegabytes,
                ["row_bytes"] = declaredGate.RowBytes,
                ["requested_rows"] = declaredGate.RequestedRows,
                ["allowed_rows"] = declaredGate.AllowedRows,
            };

        Dictionary<string, object?> bench = new()
        {
            ["table_facts"] = tableFacts,
            ["run_shape"] = new Dictionary<string, object?> { ["tokens"] = tokens, ["reps"] = reps, ["caps_megabytes"] = capsMb },
            ["stream_facts"] = new Dictionary<string, object?>
            {
                ["token_distribution"] = tokenDist,
                ["vocab_window"] = vocabWindow,
                ["distinct_rows_touched_by_stream"] = distinctRowsReached,
                ["windows_containing_eos"] = eosCutWindows,
                ["windows_with_missing_predecessors"] = missingPredecessorWindows,
            },
            ["declared_budget_gate"] = gateFacts,
            ["policy_knobs"] = new Dictionary<string, object?>
            {
                ["inflight_rows_cap_per_window"] = inflightRows,
                ["overlap_depth_windows"] = overlapDepth,
                ["ple_declared_cap_megabytes"] = pleCapMb,
            },
            ["arms"] = armReports,
            ["numa_placement"] = numaFacts,
            ["notes"] = notes,
        };

        WriteJson(Path.Combine(directory, "manifest.json"), manifest);
        WriteJson(Path.Combine(directory, "host.json"), host);
        WriteJson(Path.Combine(directory, "bench.json"), bench);
        WriteJson(Path.Combine(directory, "cli.json"), new Dictionary<string, object?> { ["argv"] = args });

        List<object> datasets = [];

        foreach (Dictionary<string, object?> arm in armReports)
        {
            if (arm["gather_latency_ns_per_token_warm"] is Dictionary<string, object?> warm && warm["p50"] is double p50 && warm["p90"] is double p90 && warm["p99"] is double p99)
            {
                datasets.Add(new { label = $"{arm["arm"]} @ {arm["cap_megabytes"]} MiB", data = new[] { p50, p90, p99 } });
            }
        }

        string chartJson = JsonSerializer.Serialize(new
        {
            type = "bar",
            data = new { labels = new[] { "p50", "p90", "p99" }, datasets },
            options = new
            {
                plugins = new { legend = new { display = true } },
                scales = new { y = new { beginAtZero = true, title = new { display = true, text = "nanoseconds per token gather step (warm passes)" } } },
            },
        });

        List<string> armRows = [];

        foreach (Dictionary<string, object?> arm in armReports)
        {
            Dictionary<string, object?> warmStats = arm["gather_latency_ns_per_token_warm"] as Dictionary<string, object?> ?? new();
            Dictionary<string, object?> coldStats = arm["gather_latency_ns_per_token_first_touch"] as Dictionary<string, object?> ?? new();

            string residencyLine = arm.TryGetValue("residency_tallies", out object? talliesValue) && talliesValue is Dictionary<string, object?> tallies
                ? $"{tallies["hits_total_all_passes"]} hits / {tallies["misses_total_all_passes"]} misses · resident {Convert.ToInt64(tallies["resident_bytes_held"]):N0} B"
                : "no residency";

            string admissionLine = arm.TryGetValue("admission", out object? admissionValue) && admissionValue is Dictionary<string, object?> admission
                ? $"allowed {Convert.ToInt64(admission["allowed_rows"]):N0} / admitted {Convert.ToInt64(admission["admitted_rows"]):N0} (refused {Convert.ToInt64(admission["refused_by_clamp_rows"]):N0})"
                : "n/a";

            string prefetchLine = arm.TryGetValue("prefetch_facts", out object? prefetchValue) && prefetchValue is Dictionary<string, object?> prefetch
                ? $"{prefetch["windows_issued"]} windows · {prefetch["rows_touched_by_prefetchers"]} rows touched · madvise {prefetch["madvise_willneed_accepted_ranges"]}/{prefetch["madvise_willneed_attempts"]} accepted"
                : "no prefetch";

            armRows.Add(
                $"<tr><td>{Esc((string)arm["arm"]!)}</td><td>{arm["cap_megabytes"]} MiB</td>" +
                $"<td>{FormatNs(warmStats, "p50")}</td><td>{FormatNs(coldStats, "p50")}</td>" +
                $"<td>{residencyLine}</td><td>{admissionLine}</td><td>{prefetchLine}</td></tr>");
        }

        List<string> numaRows = [];

        if (numaFacts?["variants"] is List<Dictionary<string, object?>> variants)
        {
            foreach (Dictionary<string, object?> variant in variants)
            {
                Dictionary<string, object?> placement = variant["numa_placement_facts"] as Dictionary<string, object?> ?? new();

                numaRows.Add(
                    $"<tr><td>{Esc((string)variant["arm"]!)}</td>" +
                    $"<td>{FormatNs(variant["gather_latency_ns_per_token_warm"] as Dictionary<string, object?> ?? new(), "p50")}</td>" +
                    $"<td>{Esc(placement.TryGetValue("arena_thread_bind", out object? bind) ? bind?.ToString() ?? "n/a" : "n/a")}</td>" +
                    $"<td>{Esc((string?)placement["gather_thread_cpu_list_after_binding"] ?? "n/a")}</td></tr>");
            }
        }

        string numaHtml = numaFacts is null
            ? string.Empty
            : $"<h2>NUMA placement variants</h2><table><tr><th>Variant</th><th>warm p50 ns/token</th><th>arena binding attempt</th><th>gather cpu list after bind</th></tr>{string.Join("", numaRows)}</table>";

        string notesHtml = string.Join("", notes.Select(note => $"<li>{Esc(note["note"]?.ToString() ?? "")}</li>"));

        File.WriteAllText(
            Path.Combine(directory, "report.html"),
            $$"""
             <!DOCTYPE html>
             <html lang="en">
             <head>
             <meta charset="utf-8" />
             <title>ple-policy {{Esc(stamp)}}</title>
             <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
             <style>
             body { font-family: system-ui, sans-serif; margin: 2rem; color: #1c1c1c; max-width: 980px; }
             h1 { font-size: 1.35rem; } h2 { font-size: 1.1rem; margin-top: 1.6rem; }
             table { border-collapse: collapse; margin-top: .5rem; width: 100%; }
             th, td { border: 1px solid #bbb; padding: .4rem .7rem; text-align: left; font-size: .92rem; vertical-align: top; }
             th { background: #f0f0f0; } canvas { max-width: 860px; margin-top: 1rem; }
             .note { color: #555; font-size: .9rem; } code { background: #f4f4f4; padding: 0 .3rem; }
             </style>
             </head>
             <body>
             <h1>PLE access-scheduling and placement measurement</h1>
             <p class="note">{{table.NgramSize}}-gram x {{table.HeadsPerNgram}} heads = {{table.HeadCount}} rows/token over a {{table.TableRows:N0}}-row <code>{{Esc(table.Tensor.Name)}}</code> table ({{table.RowStrideBytes}} packed bytes/row, {{table.Tensor.Format}}); substrate label: {{Esc(substrateLabel)}}.</p>

             <h2>Arms</h2>
             <table>
             <tr><th>Arm</th><th>Cap</th><th>warm p50 ns/token</th><th>first-touch p50 ns/token</th><th>residency tallies</th><th>clamp/admission</th><th>prefetch</th></tr>
             {{string.Join("", armRows)}}
             </table>

             <canvas id="plePolicy" width="860" height="380"></canvas>
             <script>
             new Chart(document.getElementById('plePolicy'), {{chartJson}});
             </script>

             {{numaHtml}}

             <h2>Notes</h2>
             <ul class="note">{{notesHtml}}</ul>
             </body></html>
             """);

        return directory;
    }

    private static Dictionary<string, object?> ToNodeDictionary(List<(int Node, string CpuList)> nodes)
    {
        Dictionary<string, object?> facts = new();

        foreach ((int node, string cpuList) in nodes)
        {
            facts[$"node{node}"] = cpuList;
        }

        return facts;
    }

    private static void WriteJson(string path, Dictionary<string, object?> value)
    {
        using FileStream stream = File.Create(path);
        JsonSerializer.Serialize(stream, value, new JsonSerializerOptions { WriteIndented = true });
    }

    private static string FormatNs(Dictionary<string, object?> stats, string key) =>
        stats.TryGetValue(key, out object? value) && value is not null ? $"{Convert.ToDouble(value):F0}" : "n/a";

    private static string Esc(string value) => System.Net.WebUtility.HtmlEncode(value);

    private sealed record LatencyStats(double P50, double P90, double P99, double Min, double Max)
    {
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
    }

    private static int ReadInt(string[] args, ref int index, string option)
    {
        if (!int.TryParse(args[++index], out int value))
        {
            throw new FormatException($"ple-policy: {option} needs an integer");
        }

        return value;
    }

    private static long ReadLong(string[] args, ref int index, string option)
    {
        if (!long.TryParse(args[++index], out long value))
        {
            throw new FormatException($"ple-policy: {option} needs an integer");
        }

        return value;
    }
}
