using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Text.Json.Serialization;
using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Bench;

/// <summary>
/// Gather-vs-dense-equivalent decode measurement over a pinned synthetic qwen2moe fixture: the same sequences and token
/// streams run through the merged-decode gather path (one union walk of contiguous expert slabs per layer step, with the
/// batch's shared experts gathered once) and through dense-equivalent solo stepping (each sequence stepped alone), timed
/// against each other, checked bit-for-bit for output equivalence, and reported with the step's union active-expert
/// amortization. Emits a timestamped report directory (manifest.json, host.json, bench.json, cli.json, report.html).
/// Bench-only: no posture or config change, and the fixture is synthetic F32 material measuring the gather mechanism,
/// not a quantized production model.
/// </summary>
public static class MoeGatherBench
{
    public static int Run(string[] args)
    {
        int seqs = 8;
        int tokens = 64;
        int reps = 5;
        int layers = 2;
        int hidden = 512;
        int experts = 32;
        int routed = 8;
        int inter = 512;
        int vocab = 256;
        string outRoot = "reports";
        string fixtureDir = Path.GetTempPath(); // where the synthetic fixture is written: point it at disk-backed scratch on hosts whose temp dir is RAM (tmpfs) so streaming arms really read from storage
        string substrate = "mmap"; // mmap keeps the historical OS-cached mapping; stream answers each cold stacked-expert resolution from an unbuffered whole-tensor read while dense weights stay cached — a real expensive-read baseline for the residency arm to save against (Windows: FILE_FLAG_NO_BUFFERING, Linux: O_DIRECT with a fadvise-DONTNEED fallback)
        double hotBudgetMb = 0.0; // zero disables the residency arm entirely: default runs stay exactly the prefetch-only gather-vs-solo comparison they were before this seam existed
        double hotReserveMb = -1.0; // negative means reserve the KV working-set estimate from pool geometry, honoring budget = quota − reserve as the tier contract states
        int hotSwapPeriod = 16; // layer-step notes per aging/rebalance round: small enough for a short synthetic stream to ramp its hot set during warmup

        for (int i = 1; i < args.Length; i++)
        {
            if (i + 1 >= args.Length)
            {
                Console.WriteLine($"gather: option '{args[i]}' needs a value");
                return 1;
            }

            switch (args[i])
            {
                case "--seqs": seqs = ParseInt(args[++i], seqs); break;
                case "--tokens": tokens = ParseInt(args[++i], tokens); break;
                case "--reps": reps = Math.Max(1, ParseInt(args[++i], reps)); break;
                case "--layers": layers = ParseInt(args[++i], layers); break;
                case "--hidden": hidden = ParseInt(args[++i], hidden); break;
                case "--experts": experts = ParseInt(args[++i], experts); break;
                case "--routed": routed = ParseInt(args[++i], routed); break;
                case "--inter": inter = ParseInt(args[++i], inter); break;
                case "--vocab": vocab = ParseInt(args[++i], vocab); break;
                case "--out": outRoot = args[++i]; break;
                case "--fixture-dir": fixtureDir = args[++i]; break;
                case "--substrate": substrate = args[++i].ToLowerInvariant(); break;
                case "--hot-budget-mb": hotBudgetMb = ParseDouble(args[++i], 0.0); break;
                case "--hot-reserve-mb": hotReserveMb = ParseDouble(args[++i], -1.0); break;
                case "--hot-swap-period": hotSwapPeriod = Math.Max(1, ParseInt(args[++i], 16)); break;
                default:
                    Console.WriteLine($"gather: unknown option '{args[i]}'");
                    return 1;
            }
        }

        if (seqs < 2 || seqs > TransformerModel.MaxBatchSequences)
        {
            Console.WriteLine($"gather: --seqs must be between 2 and the engine's batch ceiling ({TransformerModel.MaxBatchSequences})");
            return 1;
        }

        if (routed < 1 || routed > experts)
        {
            Console.WriteLine("gather: --routed must be between 1 and --experts");
            return 1;
        }

        // The streaming substrate is a cold-read measurement seam on both supported hosts (Windows FILE_FLAG_NO_BUFFERING, Linux O_DIRECT with a fadvise-DONTNEED fallback); other platforms keep the historical mmap posture.
        if (substrate == "stream" && !OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
        {
            Console.WriteLine("gather: --substrate stream is implemented for Windows and Linux hosts; this host runs --substrate mmap");
            return 1;
        }

        if (substrate != "mmap" && substrate != "stream")
        {
            Console.WriteLine("gather: --substrate must be mmap or stream");
            return 1;
        }

        long hotBudgetBytes = (long)Math.Round(hotBudgetMb * 1024.0 * 1024.0); // zero when the arm is disabled
        if (hotBudgetMb < 0.0)
        {
            Console.WriteLine("gather: --hot-budget-mb must be at least 0 (0 disables the residency arm)");
            return 1;
        }

        int heads = 4;
        int headDim = hidden / heads;
        int kvHeads = heads;
        if (hidden % heads != 0)
        {
            Console.WriteLine("gather: --hidden must be divisible by the fixture's four attention heads");
            return 1;
        }

        DateTime startedUtc = DateTime.UtcNow;
        string stamp = startedUtc.ToString("yyyy'-'MM'-'dd'T'HH'-'mm'-'ss'Z'", CultureInfo.InvariantCulture);
        string reportId = $"moe-gather-{stamp}";

        Console.WriteLine($"gather fixture: qwen2moe synthetic layers={layers} hidden={hidden} heads={heads} headDim={headDim} kvHeads={kvHeads} inter={inter} experts={experts} routed={routed} vocab={vocab} sharedExpert=yes weights=F32");
        Console.WriteLine($"gather posture: seqs={seqs} tokensPerSequence={tokens} reps={reps} outRoot={outRoot} substrate={substrate} engineThreads=auto avx2={Avx2.IsSupported} physicalCores={CpuBackend.DetectedPhysicalCoreCount} logicalProcessors={Environment.ProcessorCount}");
        Console.WriteLine(hotBudgetBytes > 0
            ? $"gather residency arm: enabled budgetMb={hotBudgetMb:F1} reserveMb={(hotReserveMb >= 0.0 ? hotReserveMb.ToString("F1") : "from pool geometry")} swapPeriodLayerSteps={hotSwapPeriod}"
            : "gather residency arm: disabled (pass --hot-budget-mb to add the RAM-hot-tier gather arm)");

        string fixturePath = Path.Combine(fixtureDir, $"lair-moe-gather-{Guid.NewGuid():N}.gguf");
        byte[] image = BuildFixtureImage(layers, hidden, heads, headDim, kvHeads, inter, experts, routed, vocab);
        File.WriteAllBytes(fixturePath, image);
        Console.WriteLine($"gather fixture file={fixturePath} bytes={image.Length} ({image.Length / (1024.0 * 1024.0):F1} MiB)");

        try
        {
            return Measure(
                reportId, stamp, startedUtc, outRoot, fixturePath, args, seqs, tokens, reps, layers, hidden, heads, kvHeads, headDim, inter, experts, routed, vocab, substrate, hotBudgetBytes, hotReserveMb, hotSwapPeriod);
        }
        finally
        {
            TryDelete(fixturePath);
        }
    }

    private static int Measure(
        string reportId,
        string stamp,
        DateTime startedUtc,
        string outRoot,
        string fixturePath,
        string[] invokedArgs,
        int seqs,
        int tokens,
        int reps,
        int layers,
        int hidden,
        int heads,
        int kvHeads,
        int headDim,
        int inter,
        int experts,
        int routed,
        int vocab,
        string substrate,
        long hotBudgetBytes,
        double hotReserveMbOverride,
        int hotSwapPeriod)
    {
        long expertBytes = (long)inter * hidden * 4; // F32 rows: one expert slab inside each stacked role
        using CpuBackend backend = new(allowAvx2: true);

        using GgufFile gatherGguf = GgufFile.Open(fixturePath);
        using IBenchColdBacking gatherBacking = OpenSubstrateBacking(substrate, fixturePath, gatherGguf);
        CountingSlabSource counting = new(gatherBacking);
        using TransformerModel gatherModel = TransformerModel.Open(gatherGguf, counting, backend, maxContextTokens: tokens + 4);

        int vocabFromFile = gatherModel.Desc.VocabSize;
        if (vocabFromFile != vocab)
        {
            Console.WriteLine($"gather: fixture vocab mismatch ({vocabFromFile} vs {vocab})");
            return 1;
        }

        using GgufFile soloGguf = GgufFile.Open(fixturePath);
        using IBenchColdBacking soloBacking = OpenSubstrateBacking(substrate, fixturePath, soloGguf);
        using TransformerModel soloModel = TransformerModel.Open(soloGguf, soloBacking, backend, maxContextTokens: tokens + 4);

        if (substrate != "mmap")
        {
            Console.WriteLine($"gather cold-read backing: {gatherBacking.BackingMode}"); // names the host-specific route these arms actually read through
        }

        int capacityBlocks = ((tokens + 16) / 16 + 1) * seqs + 2; // every sequence holds its own blocks in the shared pool simultaneously
        using BlockPool gatherPool = NewPool(gatherModel, backend, capacityBlocks);
        using BlockPool soloPool = NewPool(soloModel, backend, capacityBlocks);

        // Warm both engines once outside the timed reps so thread pools and page residency settle before measurement.
        RunGatherStream(gatherModel, gatherPool, seqs, tokens, vocab);
        RunSoloStream(soloModel, soloPool, seqs, tokens, vocab);

        long faultsBeforeGather = CurrentPageFaultCount();
        long stackReadsBeforeGather = gatherBacking.StackResolutionsTotal;
        long streamedBytesBeforeGather = gatherBacking.BytesStreamedTotal;

        double[] gatherSeconds = new double[reps];
        for (int rep = 0; rep < reps; rep++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            RunGatherStream(gatherModel, gatherPool, seqs, tokens, vocab);
            sw.Stop();
            gatherSeconds[rep] = sw.Elapsed.TotalSeconds;
        }

        long faultsAfterGather = CurrentPageFaultCount();
        long gatherStackReadsTimed = gatherBacking.StackResolutionsTotal - stackReadsBeforeGather;
        long gatherStreamedBytesTimed = gatherBacking.BytesStreamedTotal - streamedBytesBeforeGather;

        long faultsBeforeSolo = CurrentPageFaultCount();
        long stackReadsBeforeSolo = soloBacking.StackResolutionsTotal;
        long streamedBytesBeforeSolo = soloBacking.BytesStreamedTotal;

        double[] soloSeconds = new double[reps];
        for (int rep = 0; rep < reps; rep++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            RunSoloStream(soloModel, soloPool, seqs, tokens, vocab);
            sw.Stop();
            soloSeconds[rep] = sw.Elapsed.TotalSeconds;
        }

        long faultsAfterSolo = CurrentPageFaultCount();
        long soloStackReadsTimed = soloBacking.StackResolutionsTotal - stackReadsBeforeSolo;
        long soloStreamedBytesTimed = soloBacking.BytesStreamedTotal - streamedBytesBeforeSolo;

        double gatherMedian = Median(gatherSeconds);
        double soloMedian = Median(soloSeconds);
        long sequenceTokens = (long)seqs * tokens;
        double gatherTps = sequenceTokens / gatherMedian;
        double soloTps = sequenceTokens / soloMedian;

        Console.WriteLine($"gather arm: merged-decode median={gatherMedian:F4}s min={Min(gatherSeconds):F4} max={Max(gatherSeconds):F4} tok/s={gatherTps:F2}");
        Console.WriteLine($"dense-equivalent arm: solo-stepping median={soloMedian:F4}s min={Min(soloSeconds):F4} max={Max(soloSeconds):F4} tok/s={soloTps:F2}");

        if (substrate != "mmap")
        {
            Console.WriteLine(
                $"cold-read evidence substrate={substrate}: gather arm stackRoleReads={gatherStackReadsTimed} streamedMiB={gatherStreamedBytesTimed / (1024.0 * 1024.0):F1} pageFaults={faultsAfterGather - faultsBeforeGather} " +
                $"| solo arm stackRoleReads={soloStackReadsTimed} streamedMiB={soloStreamedBytesTimed / (1024.0 * 1024.0):F1} pageFaults={faultsAfterSolo - faultsBeforeSolo}");
        }

        // Parity pass: same token streams through both arms, every step's logits captured and compared bit-for-bit.
        float[] gatherLogits = new float[seqs * tokens * vocab];
        RunGatherStream(gatherModel, gatherPool, seqs, tokens, vocab, gatherLogits);
        float[] soloLogits = new float[seqs * tokens * vocab];
        RunSoloStream(soloModel, soloPool, seqs, tokens, vocab, soloLogits);

        (int mismatchedFloats, double maxAbsDiff) = CompareLogitsBitForBit(gatherLogits, soloLogits); // every step's logits compared bit-for-bit between the merged-decode arm and solo stepping

        bool bitExact = mismatchedFloats == 0;
        Console.WriteLine($"parity: gathered-vs-dense-equivalent logits floats={gatherLogits.Length} exact={(bitExact ? "yes" : "no")} mismatches={mismatchedFloats}{(bitExact ? string.Empty : $" maxAbsDiff={maxAbsDiff:G9}")}");

        // Instrumentation pass: one untimed full stream through the gather arm, counting slab resolutions per step.
        int[] unionPerStep = new int[tokens];
        long totalGateResolutions = 0;
        HashSet<int> distinctExpertsGlobal = [];
        for (int step = 0; step < tokens; step++)
        {
            counting.SlabResolutions.Clear();

            SequenceKv[] kvs = Enumerable.Range(0, seqs).Select(_ => new SequenceKv(gatherPool)).ToArray();
            int[] batchTokens = BatchTokensAt(seqs, vocab, step);
            gatherModel.DecodeBatch(batchTokens, kvs); // returns borrowed span; the parity pass above already captured values

            List<int> gateExperts = [.. counting.SlabResolutions.Where(r => r.Tensor.Contains("ffn_gate_exps", StringComparison.Ordinal)).Select(r => r.Expert)];
            unionPerStep[step] = gateExperts.Count;
            totalGateResolutions += gateExperts.Count;
            foreach (int expert in gateExperts)
            {
                distinctExpertsGlobal.Add(expert);
            }

            int upCount = counting.SlabResolutions.Count(r => r.Tensor.Contains("ffn_up_exps", StringComparison.Ordinal));
            int downCount = counting.SlabResolutions.Count(r => r.Tensor.Contains("ffn_down_exps", StringComparison.Ordinal));
            if (upCount != gateExperts.Count || downCount != gateExperts.Count)
                {
                    Console.WriteLine($"gather instrumentation: role resolution counts diverged at step {step}: gate={gateExperts.Count} up={upCount} down={downCount}");
                    return 1;
                }

            foreach (SequenceKv kv in kvs)
            {
                kv.Dispose();
            }
        }

        int selectionSlotsPerStep = seqs * routed; // per layer step: every sequence contributes its top-k selections
        double averageUnion = totalGateResolutions / (double)(tokens * layers);
        double amortizationPercent = 100.0 * (1.0 - averageUnion / selectionSlotsPerStep);

        Console.WriteLine(
            $"union: avgDistinctExpertsGatheredPerLayerStep={averageUnion:F2} of selectionSlots={selectionSlotsPerStep} " +
            $"amortized={(amortizationPercent >= 0 ? amortizationPercent : 0):F1}% gatheredBytes/layerStep={averageUnion * expertBytes * 3 / (1024.0 * 1024.0):F1} MiB " +
            $"denseEquivalentBytes/layerStep={(selectionSlotsPerStep * expertBytes * 3) / (1024.0 * 1024.0):F1} MiB distinctExpertsAcrossRun={distinctExpertsGlobal.Count}/{experts}");

        // Optional third arm: the same merged-decode gather answering through a RAM residency tier — hot experts held as
        // private copies chosen by capacity/cost, cold ones falling back to contiguous slab views. Identical streams and
        // file bytes; parity compares its logits bit-for-bit against the prefetch-only gather arm, and one untimed tally
        // pass after ResetAnswerTallies measures hit rate and saved-byte estimates for a clean replay of the stream.
        ArmResult? hotArm = null;
        ResidencyResult? residency = null;
        ParityResult? tierParity = null;
        long hotFaultDelta = 0, hotStackReadsTimed = 0, hotStreamedBytesTimed = 0; // cold-read evidence for the hot arm's timed reps, when that arm ran

        if (hotBudgetBytes > 0)
        {
            long kvWorkingSetEstimateBytes = capacityBlocks * 16L * kvHeads * headDim * 2 * 4 * layers; // what the pool's blocks hold for every sequence's K/V at full occupancy — the reserve the tier contract subtracts from quota unless --hot-reserve-mb states one
            long hotReserveBytes = hotReserveMbOverride >= 0.0 ? (long)Math.Round(hotReserveMbOverride * 1024.0 * 1024.0) : kvWorkingSetEstimateBytes;

            using GgufFile hotGguf = GgufFile.Open(fixturePath);
            using IBenchColdBacking hotBacking = OpenSubstrateBacking(substrate, fixturePath, hotGguf);
            using RamHotExpertTier tier = new(hotBacking, hotBudgetBytes, hotReserveBytes, hotSwapPeriod);

            using TransformerModel hotModel = TransformerModel.Open(hotGguf, tier, backend, maxContextTokens: tokens + 4);
            using BlockPool hotPool = NewPool(hotModel, backend, capacityBlocks);

            RunGatherStream(hotModel, hotPool, seqs, tokens, vocab); // warm pass: thread pools settle and the tier's aging rounds ramp its hot set before measurement reps run

            long faultsBeforeHot = CurrentPageFaultCount();
            long stackReadsBeforeHot = hotBacking.StackResolutionsTotal;
            long streamedBytesBeforeHot = hotBacking.BytesStreamedTotal;

            double[] hotSeconds = new double[reps];
            for (int rep = 0; rep < reps; rep++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                RunGatherStream(hotModel, hotPool, seqs, tokens, vocab);
                sw.Stop();
                hotSeconds[rep] = sw.Elapsed.TotalSeconds;
            }

            long faultsAfterHot = CurrentPageFaultCount();
            hotStackReadsTimed = hotBacking.StackResolutionsTotal - stackReadsBeforeHot;
            hotStreamedBytesTimed = hotBacking.BytesStreamedTotal - streamedBytesBeforeHot;
            hotFaultDelta = faultsAfterHot - faultsBeforeHot;

            double hotMedian = Median(hotSeconds);
            double hotTps = sequenceTokens / hotMedian;
            Console.WriteLine($"ram-hot-tier gather arm: merged-decode median={hotMedian:F4}s min={Min(hotSeconds):F4} max={Max(hotSeconds):F4} tok/s={hotTps:F2}");

            float[] tierLogits = new float[seqs * tokens * vocab];
            RunGatherStream(hotModel, hotPool, seqs, tokens, vocab, tierLogits);
            (int tierMismatches, double tierMaxDiff) = CompareLogitsBitForBit(gatherLogits, tierLogits);

            bool tierExact = tierMismatches == 0;
            Console.WriteLine($"parity: gathered-vs-ram-hot-tier logits floats={tierLogits.Length} exact={(tierExact ? "yes" : "no")} mismatches={tierMismatches}{(tierExact ? string.Empty : $" maxAbsDiff={tierMaxDiff:G9}")}");

            tier.ResetAnswerTallies(); // harness-facing reset, then one untimed replay pass so the reported tallies describe a clean measurement stream rather than warmup+reps cumulative counters
            RunGatherStream(hotModel, hotPool, seqs, tokens, vocab);

            long hits = tier.HitsTotal;
            long misses = tier.MissesTotal;
            double hitRatePercent = hits + misses > 0 ? 100.0 * hits / (hits + misses) : 0.0;
            double hotDeltaPercent = gatherTps > 0 ? (hotTps - gatherTps) / gatherTps * 100.0 : 0.0;

            Console.WriteLine(
                $"residency: tier={tier.Kind} budgetBytes={tier.BudgetBytes} residentBytes={tier.ResidentBytes} hotUnits={tier.HotUnits}/{experts * layers} " +
                $"swapPeriodLayerSteps={hotSwapPeriod} hits={hits} misses={misses} hitRate={hitRatePercent:F1}% savedEstimateMiB={tier.SavedBytesEstimateTotal / (1024.0 * 1024.0):F1} " +
                $"promotions={tier.PromotionsTotal} demotions={tier.DemotionsTotal} tok/s-vs-prefetch-only={(hotDeltaPercent >= 0 ? "+" : string.Empty)}{hotDeltaPercent:F2}%");

            if (substrate != "mmap")
            {
                Console.WriteLine(
                    $"cold-read evidence substrate={substrate}: hot arm stackRoleReads={hotStackReadsTimed} streamedMiB={hotStreamedBytesTimed / (1024.0 * 1024.0):F1} pageFaults={hotFaultDelta} " +
                    $"(resident hits avoided {tier.SavedBytesEstimateTotal / (1024.0 * 1024.0):F1} MiB of role reads in the untimed tally pass)");
            }

            hotArm = new ArmResult
            {
                Id = "gather-ram-hot-tier",
                Harness = "LAIR engine DecodeBatch union gather over RamHotExpertTier (capacity/cost hot expert copies + cold slab views)",
                Metrics = [new Metric { Test = "decode tok/s", TokensPerSecond = Round(hotTps) }],
                MedianSeconds = Round(hotMedian),
                MinSeconds = Round(Min(hotSeconds)),
                MaxSeconds = Round(Max(hotSeconds)),
            };

            tierParity = new ParityResult
            {
                BitExact = tierExact,
                ComparedFloats = tierLogits.Length,
                MismatchedFloats = tierMismatches,
                MaxAbsDiff = tierExact ? 0.0 : Round(tierMaxDiff),
            };

            residency = new ResidencyResult
            {
                TierKind = tier.Kind,
                BudgetBytes = tier.BudgetBytes,
                ResidentBytes = tier.ResidentBytes,
                HotUnits = tier.HotUnits,
                SwapPeriodLayerSteps = hotSwapPeriod,
                HitsTotal = hits,
                MissesTotal = misses,
                HitRatePercent = Round(hitRatePercent),
                SavedBytesEstimateTotal = tier.SavedBytesEstimateTotal,
                PromotionsTotal = tier.PromotionsTotal,
                DemotionsTotal = tier.DemotionsTotal,
                TokensPerSecondDeltaVsPrefetchOnlyPercent = Round(hotDeltaPercent),
            };
        }

        string reportDir = Path.Combine(outRoot, stamp);
        Directory.CreateDirectory(reportDir);

        HostReport host = CaptureHost(backend, seqs, tokens, reps);

        List<ArmResult> arms =
            [
                new ArmResult
                {
                    Id = "gather-merged-decode",
                    Harness = "LAIR engine DecodeBatch union gather (CpuGatherKernel over ExpertSlabWeightSource views)",
                    Metrics = [new Metric { Test = "decode tok/s", TokensPerSecond = Round(gatherTps) }],
                    MedianSeconds = Round(gatherMedian),
                    MinSeconds = Round(Min(gatherSeconds)),
                    MaxSeconds = Round(Max(gatherSeconds)),
                },
                new ArmResult
                {
                    Id = "dense-equivalent-solo",
                    Harness = "LAIR engine DecodeNext per-sequence stepping (same token streams, fresh KV)",
                    Metrics = [new Metric { Test = "decode tok/s", TokensPerSecond = Round(soloTps) }],
                    MedianSeconds = Round(soloMedian),
                    MinSeconds = Round(Min(soloSeconds)),
                    MaxSeconds = Round(Max(soloSeconds)),
                },
            ];

        if (hotArm is not null)
        {
            arms.Add(hotArm); // the RAM-hot-tier gather arm joins the comparison only when --hot-budget-mb enables it; default runs publish exactly the two-arm report they always have
        }

        List<string> comparisonNotes =
            [
                "The fixture is synthetic deterministic F32 qwen2moe material: it measures the gather mechanism (union walks, slab views, prefetch staging), not a quantized production model.",
                "Both arms run identical sequences and token streams over identical file bytes; parity compares every step's logits bit-for-bit between the merged-decode arm and solo stepping.",
                "Union stats come from an instrumentation pass that counts slab resolutions through the gather kernel: unionGateResolutionsPerDecodeStep lists gate-role resolutions per decode step summed over the fixture's layers, while averages and byte tallies are reported per layer step — each distinct expert of a step's union resolves exactly one slab per role.",
            ];

        if (residency is not null)
        {
            comparisonNotes.Add("The ram-hot-tier arm runs the same merged-decode gather with its weight source wrapped in the RAM residency tier: experts resident by capacity/cost answer slab reads from private byte-identical copies, everything else resolves through cold contiguous views. Its tallies come from one untimed replay pass after ResetAnswerTallies — hit rate and saved-byte estimate describe that clean stream; parity proves its logits bit-identical to the prefetch-only gather arm's.");
        }

        if (substrate == "stream")
        {
            comparisonNotes.Add($"Substrate stream ({gatherBacking.BackingMode}): expert role tensors resolve through cold whole-role reads into fresh buffers — unbuffered FILE_FLAG_NO_BUFFERING transfers on Windows, sector-aligned O_DIRECT preads whose read range is released with posix_fadvise(POSIX_FADV_DONTNEED) on Linux (a buffered+discard descriptor when the filesystem refuses direct I/O) — while dense weights answer from ordinary buffered (OS-cached) reads so only expert traffic pays storage cost. Both arms pay this for their cold resolutions — the prefetch-only gather arm streams every union expert's roles, and a tier miss streams that expert chain's role tensors through the same slab-view route (the loader has no per-slab staging today); resident experts answer from private copies with zero streamed bytes.");
        }


        ColdReadResult? coldReads = substrate == "mmap" ? null : new ColdReadResult
        {
            SubstrateId = substrate,
            GatherStackRoleReadsTimedReps = gatherStackReadsTimed,
            GatherStreamedBytesTimedReps = gatherStreamedBytesTimed,
            GatherPageFaultDeltaTimedReps = faultsAfterGather - faultsBeforeGather,
            SoloStackRoleReadsTimedReps = soloStackReadsTimed,
            SoloStreamedBytesTimedReps = soloStreamedBytesTimed,
            SoloPageFaultDeltaTimedReps = faultsAfterSolo - faultsBeforeSolo,
            HotArmRan = residency is not null,
            HotStackRoleReadsTimedReps = hotStackReadsTimed,
            HotStreamedBytesTimedReps = hotStreamedBytesTimed,
            HotPageFaultDeltaTimedReps = hotFaultDelta,
        };

        BenchReport bench = new()
        {
            SchemaVersion = 1,
            ReportId = reportId,
            Substrate = substrate,
            ColdReads = coldReads,
            Arms = arms,
            Parity = new ParityResult
            {
                BitExact = bitExact,
                ComparedFloats = gatherLogits.Length,
                MismatchedFloats = mismatchedFloats,
                MaxAbsDiff = bitExact ? 0.0 : Round(maxAbsDiff),
            },
            TierVsGatherParity = tierParity, // null when the residency arm did not run — default reports keep their exact earlier shape
            Residency = residency,
            UnionStats = new UnionResult
            {
                SelectionSlotsPerLayerStep = selectionSlotsPerStep,
                AverageDistinctExpertsGatheredPerLayerStep = Round(averageUnion),
                AmortizationPercent = Round(amortizationPercent),
                GatheredBytesPerLayerStep = (long)Math.Round(averageUnion * expertBytes * 3),
                DenseEquivalentBytesPerLayerStep = selectionSlotsPerStep * expertBytes * 3,
                DistinctExpertsAcrossRun = distinctExpertsGlobal.Count,
                GateResolutionsPerDecodeStep = [.. unionPerStep],
            },
        };

        ManifestReport manifest = new()
        {
            SchemaVersion = 1,
            ReportId = reportId,
            GeneratedAtUtc = startedUtc.ToString("yyyy'-'MM'-'dd'T'HH':'mm':'ss'Z'", CultureInfo.InvariantCulture),
            Phase = "MVP7",
            Branch = DetectBranch(),
            Lane = "Local CPU gather/dense-equivalent comparison (synthetic qwen2moe fixture)",
            Goal = "Measure the merged-decode union gather against dense-equivalent solo stepping on identical sequences and token streams, prove bit-identical parity, and record the step's union active-expert amortization.",
            Files = ["manifest.json", "host.json", "bench.json", "cli.json", "report.html"],
            ComparisonNotes = comparisonNotes,
        };

        WriteJson(Path.Combine(reportDir, "manifest.json"), manifest);
        WriteJson(Path.Combine(reportDir, "host.json"), host);
        WriteJson(Path.Combine(reportDir, "bench.json"), bench);
        WriteJson(
            Path.Combine(reportDir, "cli.json"),
            new { schemaVersion = 1, reportId, commandLine = Environment.CommandLine + " " + string.Join(' ', invokedArgs), workingDirectory = Directory.GetCurrentDirectory() });

        File.WriteAllText(Path.Combine(reportDir, "report.html"), BuildHtml(manifest, host, bench));

        Console.WriteLine($"gather report dir={Path.GetFullPath(reportDir)}");

        bool tierParityOk = tierParity is null || tierParity.BitExact; // the residency arm must also prove bit-identical output whenever it runs: residency changes where slab bytes come from, never the arithmetic over them
        return bitExact && tierParityOk ? 0 : 1;
    }

    private static void RunGatherStream(TransformerModel model, BlockPool pool, int seqs, int tokens, int vocab, float[]? captureLogits = null)
    {
        SequenceKv[] kvs = Enumerable.Range(0, seqs).Select(_ => new SequenceKv(pool)).ToArray();
        try
        {
            for (int step = 0; step < tokens; step++)
            {
                Span<float> logits = model.DecodeBatch(BatchTokensAt(seqs, vocab, step), kvs);

                if (captureLogits is not null)
                {
                    int offset = step * seqs * vocab;
                    for (int s = 0; s < seqs; s++)
                    {
                        logits.Slice(s * vocab, vocab).ToArray().CopyTo(captureLogits.AsSpan(offset + s * vocab, vocab));
                    }
                }

                if (float.IsNaN(logits[0]))
                {
                    throw new InvalidOperationException($"gather: NaN in batched logits at step {step}");
                }
            }
        }
        finally
        {
            foreach (SequenceKv kv in kvs)
            {
                kv.Dispose();
            }
        }
    }

    private static void RunSoloStream(TransformerModel model, BlockPool pool, int seqs, int tokens, int vocab, float[]? captureLogits = null)
    {
        for (int s = 0; s < seqs; s++)
        {
            SequenceKv kv = new(pool);
            try
            {
                for (int step = 0; step < tokens; step++)
                {
                    Span<float> logits = model.DecodeNext(TokenId(s, step, vocab), kv);

                    if (captureLogits is not null)
                    {
                        int offset = step * seqs * vocab;
                        logits.ToArray().CopyTo(captureLogits.AsSpan(offset + s * vocab, vocab));
                    }

                    if (float.IsNaN(logits[0]))
                    {
                        throw new InvalidOperationException($"gather: NaN in solo logits at sequence {s} step {step}");
                    }
                }
            }
            finally
            {
                kv.Dispose();
            }
        }
    }

    // Token streams differ per sequence (offset 613) and advance deterministically through the vocab, so sequences share
    // some hot experts while disagreeing on others — exactly the overlap the union gather amortizes.
    private static int[] BatchTokensAt(int seqs, int vocab, int step) => Enumerable.Range(0, seqs).Select(s => TokenId(s, step, vocab)).ToArray();

    private static int TokenId(int sequence, int step, int vocab) => (int)(((long)step * 977 + (long)sequence * 613 + 1) % vocab);

    private static BlockPool NewPool(TransformerModel model, CpuBackend backend, int capacityBlocks) =>
        new(blockTokens: 16, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks, backend);

    private static HostReport CaptureHost(CpuBackend backend, int seqs, int tokens, int reps)
    {
        List<string> matched = [];
        if (System.Runtime.Intrinsics.X86.Avx.IsSupported) matched.Add("avx");
        if (Avx2.IsSupported) matched.Add("avx2");
        if (Fma.IsSupported) matched.Add("fma");
        if (Ssse3.IsSupported) matched.Add("ssse3");

        List<string> absent = ["vnni", "bf16"]; // CpuBackend's ISA vocabulary reports no VNNI or BF16 posture on this seam today

        return new HostReport
        {
            SchemaVersion = 1,
            Host = new HostFacts
            {
                Hostname = Environment.MachineName,
                Os = RuntimeInformation.OSDescription,
                Framework = RuntimeInformation.FrameworkDescription,
                LogicalProcessors = Environment.ProcessorCount,
                PhysicalCores = CpuBackend.DetectedPhysicalCoreCount,
                BackendWorkers = backend.EffectiveWorkerCount,
                MemoryTotalBytes = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes,
                CpuFlagsMatched = matched,
                CpuFlagsAbsent = absent,
                Lane = "CPU-only",
            },
        };
    }

    private static string DetectBranch()
    {
        try
        {
            ProcessStartInfo start = new("git", "rev-parse --abbrev-ref HEAD")
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
            };

            using Process? process = Process.Start(start);
            if (process is null)
            {
                return "unknown";
            }

            string branch = process.StandardOutput.ReadToEnd().Trim();
            process.WaitForExit();
            return string.IsNullOrEmpty(branch) ? "unknown" : branch;
        }
        catch
        {
            return "unknown";
        }
    }

    // ---------------------------------------------------------------- fixture generation (deterministic, seed-pinned)

    private static byte[] BuildFixtureImage(int layers, int hidden, int heads, int headDim, int kvHeads, int inter, int experts, int routed, int vocab)
    {
        const float Eps = 1e-5f;
        const int SectorPad = 4096; // fixture files pad to this boundary so streaming substrates can read every payload window with sector-aligned unbuffered reads
        Random rng = new(20261121);

        List<(string Key, object Value)> metadata =
        [
            ("general.architecture", "qwen2moe"),
            ("qwen2moe.embedding_length", hidden),
            ("qwen2moe.block_count", layers),
            ("qwen2moe.attention.head_count", heads),
            ("qwen2moe.attention.head_count_kv", kvHeads),
            ("qwen2moe.feed_forward_length", inter),
            ("qwen2moe.attention.layer_norm_rms_epsilon", Eps),
            ("qwen2moe.vocab_size", vocab),
            ("qwen2moe.expert_count", experts),
            ("qwen2moe.expert_used_count", routed),
            ("qwen2moe.expert_shared_count", 1),
            ("qwen2moe.expert_weights_scale", 1f),
            ("qwen2moe.expert_weights_norm", true),
        ];

        List<TensorImage> tensors = [];

        float[] embeddings = new float[vocab * hidden];
        for (int i = 0; i < embeddings.Length; i++)
        {
            embeddings[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        tensors.Add(new("token_embd.weight", [vocab, hidden], F32Bytes(embeddings)));
        tensors.Add(new("output_norm.weight", [hidden], F32Bytes(FlatNorm(hidden))));

        float[] baseDirection = new float[hidden];
        for (int c = 0; c < hidden; c++)
        {
            baseDirection[c] = 0.45f + 0.1f * (c % 7);
        }

        // Expert popularity bands: a hot band of eight, a warm band of eight, and the rest cold. Router rows keep each
        // expert's score near its band so tokens disagree within the hot/warm region while hot experts dominate — the
        // overlap pattern a union gather amortizes across merged decode sequences.
        int[] popularityOrder = Enumerable.Range(0, experts).ToArray();
        for (int i = popularityOrder.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (popularityOrder[i], popularityOrder[j]) = (popularityOrder[j], popularityOrder[i]);
        }

        float[] popByExpert = new float[experts];
        for (int rank = 0; rank < experts; rank++)
        {
            int expert = popularityOrder[rank];
            popByExpert[expert] = rank switch
            {
                < 8 => 1.35f - 0.04f * rank + 0.02f * (expert % 3),
                < 16 => 1.05f - 0.02f * (rank - 8) + 0.02f * (expert % 3),
                _ => 0.55f + 0.02f * (expert % 4),
            };
        }

        for (int layer = 0; layer < layers; layer++)
        {
            tensors.Add(new($"blk.{layer}.attn_q.weight", [heads * headDim, hidden], F32Bytes(AttentionPayload(hidden, heads * headDim, 0.4f))));
            tensors.Add(new($"blk.{layer}.attn_k.weight", [kvHeads * headDim, hidden], F32Bytes(AttentionPayload(hidden, kvHeads * headDim, 0.2f))));
            tensors.Add(new($"blk.{layer}.attn_v.weight", [kvHeads * headDim, hidden], new byte[kvHeads * headDim * hidden * 4])); // zero v/o payloads: attention contributes nothing and the measurement stays on the FFN gather path
            tensors.Add(new($"blk.{layer}.attn_output.weight", [hidden, heads * headDim], new byte[hidden * heads * headDim * 4]));

            tensors.Add(new($"blk.{layer}.attn_norm.weight", [hidden], F32Bytes(FlatNorm(hidden))));
            tensors.Add(new($"blk.{layer}.ffn_norm.weight", [hidden], F32Bytes(FlatNorm(hidden))));

            float[] router = new float[experts * hidden];
            for (int e = 0; e < experts; e++)
            {
                float noiseScale = popByExpert[e] >= 1.0f ? 0.5f : 0.34f; // band-local noise: tokens reorder inside a band, bands stay separated
                for (int c = 0; c < hidden; c++)
                {
                    router[e * hidden + c] = popByExpert[e] * baseDirection[c] + noiseScale * (float)(rng.NextDouble() * 2.0 - 1.0);
                }
            }

            tensors.Add(new($"blk.{layer}.ffn_gate_inp.weight", [experts, hidden], F32Bytes(router)));

            float[] expertFactor = new float[experts];
            for (int e = 0; e < experts; e++)
            {
                expertFactor[e] = 0.03f + 0.045f * popByExpert[e]; // modest so gate/up activations and the down projection stay finite
            }

            tensors.Add(new($"blk.{layer}.ffn_gate_exps.weight", [experts, inter, hidden], F32Bytes(StackedExperts(expertFactor, experts, inter * hidden))));
            tensors.Add(new($"blk.{layer}.ffn_up_exps.weight", [experts, inter, hidden], F32Bytes(StackedUp(expertFactor, experts, inter * hidden))));
            tensors.Add(new($"blk.{layer}.ffn_down_exps.weight", [experts, hidden, inter], F32Bytes(StackedDown(expertFactor, experts, hidden * inter))));

            float[] shared = new float[inter * hidden];
            for (int i = 0; i < shared.Length; i++)
            {
                shared[i] = 0.05f * (0.2f + 0.31f * (i % 7));
            }

            tensors.Add(new($"blk.{layer}.ffn_gate_shexp.weight", [inter, hidden], F32Bytes(shared)));
            tensors.Add(new($"blk.{layer}.ffn_up_shexp.weight", [inter, hidden], F32Bytes(SignedVariant(shared))));
            tensors.Add(new($"blk.{layer}.ffn_down_shexp.weight", [hidden, inter], F32Bytes(DownFrom(shared, hidden * inter))));
        }

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        const uint GgufMagic = 0x46554747; // "GGUF"
        writer.Write(GgufMagic);
        writer.Write(3u); // version
        writer.Write((ulong)tensors.Count);
        writer.Write((ulong)metadata.Count);

        foreach ((string key, object value) in metadata)
        {
            WriteGgufString(writer, key);
            switch (value)
            {
                case string s:
                    writer.Write(8u); // STRING
                    WriteGgufString(writer, s);
                    break;
                case int i:
                    writer.Write(5u); // INT32
                    writer.Write(i);
                    break;
                case float f:
                    writer.Write(6u); // FLOAT32
                    writer.Write(f);
                    break;
                case bool b:
                    writer.Write(7u); // BOOL
                    writer.Write((byte)(b ? 1 : 0));
                    break;
                default:
                    throw new ArgumentException($"unsupported fixture value type {value.GetType().Name}");
            }
        }

        long offset = 0;
        foreach (TensorImage tensor in tensors)
        {
            // GGUF stores dims fastest-varying first: the logical row-major order is written reversed.
            int[] stored = new int[tensor.Dims.Length];
            for (int d = 0; d < tensor.Dims.Length; d++)
            {
                stored[d] = tensor.Dims[tensor.Dims.Length - 1 - d];
            }

            WriteGgufString(writer, tensor.Name);
            writer.Write((uint)stored.Length);
            foreach (int dim in stored)
            {
                writer.Write((ulong)dim);
            }

            writer.Write(0u); // ggml type F32
            writer.Write((ulong)offset);
            offset += tensor.Bytes.Length;
        }

        long headerLength = stream.Position;
        long dataStart = (headerLength + 31) / 32 * 32; // GGUF default alignment is 32
        for (long i = headerLength; i < dataStart; i++)
        {
            writer.Write((byte)0);
        }

        foreach (TensorImage tensor in tensors)
        {
            writer.Write(tensor.Bytes);
        }

        // Pad the file tail to a 4 KiB multiple: unbuffered sector-aligned reads then cover every byte of every payload window without straddling EOF. The loader reads declared tensors only, so trailing zero bytes are inert for every substrate.
        long paddedLength = (stream.Position + SectorPad - 1) / SectorPad * SectorPad;
        for (long padOffset = stream.Position; padOffset < paddedLength; padOffset++)
        {
            writer.Write((byte)0);
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static float[] AttentionPayload(int hidden, int rows, float factor)
    {
        float[] values = new float[rows * (long)hidden]; // modest positive attention payload so decode stays finite; v/o stay zero and contribute nothing
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = factor * (0.1f + 0.37f * (i % 9));
        }

        return values;
    }

    private static float[] FlatNorm(int hidden)
    {
        float[] values = new float[hidden];
        for (int c = 0; c < hidden; c++)
        {
            values[c] = 1.0f + 0.15f * ((c % 9) - 4) / 4f;
        }

        return values;
    }

    private static float[] StackedExperts(float[] factor, int experts, int blockLength)
    {
        float[] stacked = new float[experts * (long)blockLength]; // gate rows carry each expert's fingerprint so contributions are detectably distinct
        for (int e = 0; e < experts; e++)
        {
            for (int i = 0; i < blockLength; i++)
            {
                stacked[e * (long)blockLength + i] = factor[e] * (0.3f + 0.23f * (i % 6));
            }
        }

        return stacked;
    }

    private static float[] StackedUp(float[] factor, int experts, int blockLength)
    {
        float[] stacked = new float[experts * (long)blockLength]; // modest positive up rows keep the gated activation from saturating
        for (int e = 0; e < experts; e++)
        {
            for (int i = 0; i < blockLength; i++)
            {
                stacked[e * (long)blockLength + i] = factor[e] * (0.18f + 0.21f * (i % 6));
            }
        }

        return stacked;
    }

    private static float[] StackedDown(float[] factor, int experts, int blockLength)
    {
        float[] stacked = new float[experts * (long)blockLength]; // down rows carry per-expert offsets so each expert keeps a distinct fingerprint
        for (int e = 0; e < experts; e++)
        {
            float sign = (e % 2 == 0) ? 1.4f : -1.2f;
            for (int i = 0; i < blockLength; i++)
            {
                stacked[e * (long)blockLength + i] = factor[e] * sign * (0.3f + 0.17f * (i % 5));
            }
        }

        return stacked;
    }

    private static float[] SignedVariant(float[] shared)
    {
        float[] values = new float[shared.Length];
        for (int i = 0; i < shared.Length; i++)
        {
            values[i] = -0.7f * shared[i];
        }

        return values;
    }

    private static float[] DownFrom(float[] shared, int length)
    {
        float[] values = new float[length]; // the down triplet reuses the shared pattern scaled so its contribution stays moderate
        for (int i = 0; i < length; i++)
        {
            values[i] = 1.3f * (shared.Length > 0 ? shared[i % shared.Length] : 0.2f);
        }

        return values;
    }

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private record TensorImage(string Name, int[] Dims, byte[] Bytes);

    private static byte[] F32Bytes(float[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

    private static double Median(double[] values)
    {
        double[] sorted = [.. values];
        Array.Sort(sorted);
        int mid = sorted.Length / 2;
        return sorted.Length % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0;
    }

    private static double Min(double[] values) => values.Min();

    // Bit-for-bit logit comparison shared by every parity pairing in this lane: differing float bits are counted, and the widest absolute gap is kept for reporting when a pairing diverges.
    private static (int MismatchedFloats, double MaxAbsDiff) CompareLogitsBitForBit(float[] expected, float[] actual)
    {
        if (expected.Length != actual.Length)
        {
            throw new ArgumentException($"logit capture length mismatch ({expected.Length} vs {actual.Length})");
        }

        int mismatches = 0;
        double maxDiff = 0.0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
            {
                mismatches++;
                maxDiff = Math.Max(maxDiff, Math.Abs((double)expected[i] - actual[i]));
            }
        }

        return (mismatches, maxDiff);
    }
    private static double Max(double[] values) => values.Max();
    private static double Round(double value) => Math.Round(value, 4);

    private static int ParseInt(string raw, int fallback) => int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) ? v : fallback;

    private static double ParseDouble(string raw, double fallback) => double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) ? v : fallback;

    private static void TryDelete(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (IOException)
        {
        }
    }

    // ---------------------------------------------------------------- report shape

    private sealed class ManifestReport
    {
        [JsonPropertyName("schemaVersion")] public int SchemaVersion { get; init; }
        [JsonPropertyName("reportId")] public string ReportId { get; init; } = "";
        [JsonPropertyName("generatedAtUtc")] public string GeneratedAtUtc { get; init; } = "";
        [JsonPropertyName("phase")] public string Phase { get; init; } = "";
        [JsonPropertyName("branch")] public string Branch { get; init; } = "";
        [JsonPropertyName("lane")] public string Lane { get; init; } = "";
        [JsonPropertyName("goal")] public string Goal { get; init; } = "";
        [JsonPropertyName("files")] public List<string> Files { get; init; } = [];
        [JsonPropertyName("comparisonNotes")] public List<string> ComparisonNotes { get; init; } = [];
    }

    private sealed class HostReport
    {
        [JsonPropertyName("schemaVersion")] public int SchemaVersion { get; init; }
        [JsonPropertyName("host")] public HostFacts Host { get; init; } = new();
    }

    private sealed class HostFacts
    {
        [JsonPropertyName("hostname")] public string Hostname { get; init; } = "";
        [JsonPropertyName("os")] public string Os { get; init; } = "";
        [JsonPropertyName("framework")] public string Framework { get; init; } = "";
        [JsonPropertyName("logicalProcessors")] public int LogicalProcessors { get; init; }
        [JsonPropertyName("physicalCores")] public int PhysicalCores { get; init; }
        [JsonPropertyName("backendWorkers")] public int BackendWorkers { get; init; }
        [JsonPropertyName("memoryTotalBytes")] public long MemoryTotalBytes { get; init; }
        [JsonPropertyName("cpuFlagsMatched")] public List<string> CpuFlagsMatched { get; init; } = [];
        [JsonPropertyName("cpuFlagsAbsent")] public List<string> CpuFlagsAbsent { get; init; } = [];
        [JsonPropertyName("lane")] public string Lane { get; init; } = "";
    }

    private sealed class BenchReport
    {
        [JsonPropertyName("schemaVersion")] public int SchemaVersion { get; init; }
        [JsonPropertyName("reportId")] public string ReportId { get; init; } = "";

        [JsonPropertyName("substrate")] public string Substrate { get; init; } = "mmap";

        /// <summary>Per-arm cold-read tallies for the timed reps, present only when a streaming substrate ran.</summary>
        [JsonPropertyName("coldReads")] public ColdReadResult? ColdReads { get; init; }
        [JsonPropertyName("arms")] public List<ArmResult> Arms { get; init; } = [];
        [JsonPropertyName("parity")] public ParityResult Parity { get; init; } = new();

        /// <summary>Gather arm vs RAM-hot-tier parity, present only when the residency arm ran.</summary>
        [JsonPropertyName("tierVsGatherParity")] public ParityResult? TierVsGatherParity { get; init; }

        /// <summary>The RAM tier's own accounting after its clean tally pass, present only when the residency arm ran.</summary>
        [JsonPropertyName("residency")] public ResidencyResult? Residency { get; init; }
        [JsonPropertyName("unionStats")] public UnionResult UnionStats { get; init; } = new();
    }

    private sealed class ArmResult
    {
        [JsonPropertyName("id")] public string Id { get; init; } = "";
        [JsonPropertyName("harness")] public string Harness { get; init; } = "";
        [JsonPropertyName("metrics")] public List<Metric> Metrics { get; init; } = [];
        [JsonPropertyName("medianSeconds")] public double MedianSeconds { get; init; }
        [JsonPropertyName("minSeconds")] public double MinSeconds { get; init; }
        [JsonPropertyName("maxSeconds")] public double MaxSeconds { get; init; }
    }

    private sealed class Metric
    {
        [JsonPropertyName("test")] public string Test { get; init; } = "";
        [JsonPropertyName("tokensPerSecond")] public double TokensPerSecond { get; init; }
    }

    private sealed class ParityResult
    {
        [JsonPropertyName("bitExact")] public bool BitExact { get; init; }
        [JsonPropertyName("comparedFloats")] public int ComparedFloats { get; init; }
        [JsonPropertyName("mismatchedFloats")] public int MismatchedFloats { get; init; }
        [JsonPropertyName("maxAbsDiff")] public double MaxAbsDiff { get; init; }
    }

    private sealed class UnionResult
    {
        [JsonPropertyName("selectionSlotsPerLayerStep")] public int SelectionSlotsPerLayerStep { get; init; }
        [JsonPropertyName("averageDistinctExpertsGatheredPerLayerStep")] public double AverageDistinctExpertsGatheredPerLayerStep { get; init; }
        [JsonPropertyName("amortizationPercent")] public double AmortizationPercent { get; init; }
        [JsonPropertyName("gatheredBytesPerLayerStep")] public long GatheredBytesPerLayerStep { get; init; }
        [JsonPropertyName("denseEquivalentBytesPerLayerStep")] public long DenseEquivalentBytesPerLayerStep { get; init; }
        [JsonPropertyName("distinctExpertsAcrossRun")] public int DistinctExpertsAcrossRun { get; init; }
        [JsonPropertyName("unionGateResolutionsPerDecodeStep")] public List<int> GateResolutionsPerDecodeStep { get; init; } = []; // gate-role slab resolutions per decode step summed over the fixture's layers: every sequence contributes its top-k selections, and each distinct expert of the step's union resolves exactly one slab per role
    }

    /// <summary>Per-arm cold-read tallies captured around each arm's timed reps on a streaming substrate: whole-role expert resolutions, bytes attributed to storage reads, and the process page-fault delta.</summary>
    private sealed class ColdReadResult
    {
        [JsonPropertyName("substrateId")] public string SubstrateId { get; init; } = "";
        [JsonPropertyName("gatherStackRoleReadsTimedReps")] public long GatherStackRoleReadsTimedReps { get; init; }
        [JsonPropertyName("gatherStreamedBytesTimedReps")] public long GatherStreamedBytesTimedReps { get; init; }
        [JsonPropertyName("gatherPageFaultDeltaTimedReps")] public long GatherPageFaultDeltaTimedReps { get; init; }
        [JsonPropertyName("soloStackRoleReadsTimedReps")] public long SoloStackRoleReadsTimedReps { get; init; }
        [JsonPropertyName("soloStreamedBytesTimedReps")] public long SoloStreamedBytesTimedReps { get; init; }
        [JsonPropertyName("soloPageFaultDeltaTimedReps")] public long SoloPageFaultDeltaTimedReps { get; init; }
        [JsonPropertyName("hotArmRan")] public bool HotArmRan { get; init; }
        [JsonPropertyName("hotStackRoleReadsTimedReps")] public long HotStackRoleReadsTimedReps { get; init; }
        [JsonPropertyName("hotStreamedBytesTimedReps")] public long HotStreamedBytesTimedReps { get; init; }
        [JsonPropertyName("hotPageFaultDeltaTimedReps")] public long HotPageFaultDeltaTimedReps { get; init; }
    }

    private sealed class ResidencyResult
    {
        [JsonPropertyName("tierKind")] public string TierKind { get; init; } = "";

        /// <summary>The tier's admission ceiling — RAM quota less the stated working-set reserve.</summary>
        [JsonPropertyName("budgetBytes")] public long BudgetBytes { get; init; }

        /// <summary>Expert chains resident at the end of the tally pass, in bytes.</summary>
        [JsonPropertyName("residentBytes")] public long ResidentBytes { get; init; }

        /// <summary>Hot expert-chain units held across the fixture's layers when the tallies were read.</summary>
        [JsonPropertyName("hotUnits")] public int HotUnits { get; init; }

        /// <summary>Layer-step notes per aging/rebalance round for this run.</summary>
        [JsonPropertyName("swapPeriodLayerSteps")] public int SwapPeriodLayerSteps { get; init; }

        /// <summary>Slab resolutions answered from residency during the untimed tally pass.</summary>
        [JsonPropertyName("hitsTotal")] public long HitsTotal { get; init; }

        /// <summary>Known-expert slab resolutions that fell through to cold views in the same pass.</summary>
        [JsonPropertyName("missesTotal")] public long MissesTotal { get; init; }

        /// <summary>Hits over all tallied slab resolutions, as a percent of the clean replay pass.</summary>
        [JsonPropertyName("hitRatePercent")] public double HitRatePercent { get; init; }

        /// <summary>Cumulative bytes those hit resolutions avoided reading cold.</summary>
        [JsonPropertyName("savedBytesEstimateTotal")] public long SavedBytesEstimateTotal { get; init; }

        /// <summary>Promotions the capacity/cost model made across this arm's streams (warmup included — cumulative since open).</summary>
        [JsonPropertyName("promotionsTotal")] public long PromotionsTotal { get; init; }

        /// <summary>Margin-gated demotions out of residency in the same window.</summary>
        [JsonPropertyName("demotionsTotal")] public long DemotionsTotal { get; init; }

        /// <summary>The hot-tier arm's decode tok/s delta against the prefetch-only gather arm, as a percent.</summary>
        [JsonPropertyName("tokensPerSecondDeltaVsPrefetchOnlyPercent")] public double TokensPerSecondDeltaVsPrefetchOnlyPercent { get; init; }
    }

    private static void WriteJson(string path, object value) =>
        File.WriteAllText(path, System.Text.Json.JsonSerializer.Serialize(value, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));

    private static string BuildHtml(ManifestReport manifest, HostReport host, BenchReport bench)
    {
        ArmResult gather = bench.Arms[0];
        ArmResult solo = bench.Arms[1];
        double deltaPercent = solo.Metrics[0].TokensPerSecond > 0
            ? (gather.Metrics[0].TokensPerSecond - solo.Metrics[0].TokensPerSecond) / solo.Metrics[0].TokensPerSecond * 100.0
            : 0.0;

        string armsRows = string.Join(
            "\n",
            bench.Arms.Select(a => $"<tr><td>{Esc(a.Id)}</td><td>{Esc(a.Harness)}</td><td>{a.Metrics[0].TokensPerSecond:F2}</td><td>{a.MedianSeconds:F4}</td><td>{a.MinSeconds:F4}</td><td>{a.MaxSeconds:F4}</td></tr>"));

        List<string> parityRows = [
            $"<tr><td>gather vs dense-equivalent</td><td>{bench.Parity.ComparedFloats}</td><td>{bench.Parity.MismatchedFloats}</td><td class=\"{(bench.Parity.BitExact ? "ok" : "bad")}\">{(bench.Parity.BitExact ? "bit-exact" : "diverged")}</td></tr>",
        ];

        if (bench.TierVsGatherParity is ParityResult tierPairing) // the residency pairing only shows up when that arm ran — default reports render exactly as they did before this seam existed
        {
            parityRows.Add(
                $"<tr><td>gather vs ram-hot-tier</td><td>{tierPairing.ComparedFloats}</td><td>{tierPairing.MismatchedFloats}</td><td class=\"{(tierPairing.BitExact ? "ok" : "bad")}\">{(tierPairing.BitExact ? "bit-exact" : "diverged")}</td></tr>");
        }

        string parityRowsHtml = string.Join("\n", parityRows);
        string unionRow = $"<tr><td>{bench.UnionStats.SelectionSlotsPerLayerStep}</td><td>{bench.UnionStats.AverageDistinctExpertsGatheredPerLayerStep:F2}</td><td>{bench.UnionStats.AmortizationPercent:F1}%</td><td>{bench.UnionStats.GatheredBytesPerLayerStep / (1024.0 * 1024.0):F1}</td><td>{bench.UnionStats.DenseEquivalentBytesPerLayerStep / (1024.0 * 1024.0):F1}</td></tr>";

        string residencySection = bench.Residency is ResidencyResult res
            ? $$"""

<h2>RAM hot tier — capacity/cost residency behind the gather</h2>
<table>
<tr><th>Tier</th><td>{{Esc(res.TierKind)}}</td></tr>
<tr><th>Budget bytes (quota − reserve)</th><td>{{res.BudgetBytes}}</td></tr>
<tr><th>Resident bytes / hot units</th><td>{{res.ResidentBytes}} · {{res.HotUnits}}</td></tr>
<tr><th>Swap period (layer steps per aging round)</th><td>{{res.SwapPeriodLayerSteps}}</td></tr>
<tr><th>Hits · misses · hit rate</th><td>{{res.HitsTotal}} · {{res.MissesTotal}} · <span class="{{(res.HitRatePercent > 0.0 ? "ok" : "bad")}}">{{res.HitRatePercent:F1}}%</span></td></tr>
<tr><th>Saved cold-read estimate</th><td>{{res.SavedBytesEstimateTotal / (1024.0 * 1024.0):F1}} MiB</td></tr>
<tr><th>Promotions · demotions</th><td>{{res.PromotionsTotal}} · {{res.DemotionsTotal}}</td></tr>
<tr><th>Tok/s delta vs prefetch-only gather</th><td>{{(res.TokensPerSecondDeltaVsPrefetchOnlyPercent >= 0.0 ? "+" : string.Empty)}}{{res.TokensPerSecondDeltaVsPrefetchOnlyPercent:F2}}%</td></tr>
</table>
"""
            : string.Empty; // no residency arm ran: the HTML renders exactly as it did before this seam existed
        string notesLi = string.Join("\n", manifest.ComparisonNotes.Select(n => $"<li>{Esc(n)}</li>"));
        string labelsJson = string.Join(", ", bench.Arms.Select(a => "'" + Esc(a.Id) + "'"));
        string dataJson = string.Join(", ", bench.Arms.Select(a => a.Metrics[0].TokensPerSecond.ToString("F2", CultureInfo.InvariantCulture)));

        return $$"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{{Esc(manifest.ReportId)}}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
body { font-family: system-ui, sans-serif; margin: 2rem; color: #1c1c1c; }
h1 { font-size: 1.35rem; } h2 { font-size: 1.1rem; margin-top: 1.6rem; }
table { border-collapse: collapse; margin-top: .5rem; }
th, td { border: 1px solid #bbb; padding: .4rem .7rem; text-align: left; font-size: .92rem; }
th { background: #f0f0f0; } .ok { color: #0a7d32; font-weight: 600; } .bad { color: #b00020; font-weight: 600; }
canvas { max-width: 720px; margin-top: 1rem; }
.note { color: #555; font-size: .88rem; }
</style>
</head>
<body>
<h1>{{Esc(manifest.ReportId)}}</h1>
<p class="note">{{Esc(manifest.Lane)}} · {{Esc(manifest.GeneratedAtUtc)}} · branch {{Esc(manifest.Branch)}}</p>

<h2>Arms — merged-decode gather vs dense-equivalent solo stepping</h2>
<table>
<tr><th>Arm</th><th>Harness</th><th>Tok/s (median)</th><th>Median s</th><th>Min s</th><th>Max s</th></tr>
{{armsRows}}
</table>
<p class="note">gather vs dense-equivalent delta = {{deltaPercent:F2}}% (positive means the gather arm is faster)</p>

<h2>Parity — gathered logits vs dense-equivalent, bit-for-bit</h2>
<table>
<tr><th>Pairing</th><th>Compared floats</th><th>Mismatched</th><th>Exact</th></tr>
{{parityRowsHtml}}
</table>

<h2>Union active-expert amortization (instrumentation pass)</h2>
<table>
<tr><th>Selection slots / layer step</th><th>Avg distinct experts gathered</th><th>Amortized</th><th>Gathered MiB / layer step</th><th>Dense-equivalent MiB / layer step</th></tr>
{{unionRow}}
</table>

{{residencySection}}
<h2>Decode throughput comparison</h2>
<canvas id="arms" width="720" height="360"></canvas>
<script>
new Chart(document.getElementById('arms'), {
  type: 'bar',
  data: {
    labels: [{{labelsJson}}],
    datasets: [{ label: 'decode tok/s (median)', data: [{{dataJson}}] }]
  },
  options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, title: { display: true, text: 'tokens/s' } } } }
});
</script>

<h2>Host</h2>
<table>
<tr><th>Hostname</th><td>{{Esc(host.Host.Hostname)}}</td></tr>
<tr><th>OS</th><td>{{Esc(host.Host.Os)}}</td></tr>
<tr><th>Logical processors / physical cores / backend workers</th><td>{{host.Host.LogicalProcessors}} / {{host.Host.PhysicalCores}} / {{host.Host.BackendWorkers}}</td></tr>
<tr><th>CPU flags matched</th><td>{{Esc(string.Join(", ", host.Host.CpuFlagsMatched))}}</td></tr>
</table>

<h2>Notes</h2>
<ul>
{{notesLi}}
</ul>
</body>
</html>
""";
    }

    private static IBenchColdBacking OpenSubstrateBacking(string substrate, string fixturePath, GgufFile gguf) => substrate switch
    {
        "stream" when OperatingSystem.IsWindows() => new UnbufferedStackReadWeightSource(fixturePath, gguf),
        "stream" => new LinuxColdReadWeightSource(fixturePath, gguf),
        _ => new MappedArenaWeightSource(gguf),
    };

    // Page-fault tallies per arm: Windows counts through psapi, Linux sums /proc/self/stat minor+major faults. Per-arm deltas beside the streamed-byte tallies are the cold-read evidence this lane publishes without perf counters; a host where neither is readable reports 0 and leans on the byte tallies alone.
    private static long CurrentPageFaultCount() => OperatingSystem.IsWindows() ? WinColdReadInterop.PageFaultCountOfCurrentProcess() : LinuxColdReadInterop.PageFaultsOfCurrentProcess();

    private static string Esc(string value) => System.Net.WebUtility.HtmlEncode(value);
}

/// <summary>A whole-tensor weight source that also answers slab resolutions, counting every (tensor, expert) slab the gather kernel resolves.</summary>
internal sealed class CountingSlabSource : IWeightSource, IExpertSlabSource
{
    private readonly IWeightSource _backing;
    private readonly ExpertSlabWeightSource _slabs;

    public CountingSlabSource(IWeightSource backing)
    {
        _backing = backing;
        _slabs = new ExpertSlabWeightSource(backing);
    }

    public List<(string Tensor, int Expert)> SlabResolutions { get; } = [];

    public ReadOnlySpan<byte> Resolve(LogicalTensor tensor) => _backing.Resolve(tensor);

    public ReadOnlySpan<byte> ResolveSlab(LogicalTensor stack, int expertIndex, long expertBytes)
    {
        SlabResolutions.Add((stack.Name, expertIndex));
        return _slabs.ResolveSlab(stack, expertIndex, expertBytes);
    }

    public void Dispose()
    {
    }
}

/// <summary>The cold-read accounting seam every gather arm on this lane reads through: whole-tensor resolution plus per-arm tallies of expert-role resolutions and the bytes attributed to storage for them.</summary>
internal interface IBenchColdBacking : IWeightSource
{
    /// <summary>The substrate id these tallies describe ("mmap", "nocache", or "stream").</summary>
    string SubstrateId { get; }

    /// <summary>How this backing actually serves cold reads on the current host (e.g. windows no-buffering, linux o_direct+discard, linux discard).</summary>
    string BackingMode { get; }

    /// <summary>Cumulative resolutions of stacked expert-role tensors (whole role tensor per resolution).</summary>
    long StackResolutionsTotal { get; }

    /// <summary>Cumulative payload bytes attributed to storage reads by those resolutions.</summary>
    long BytesStreamedTotal { get; }
}

/// <summary>The historical OS-cached mapping posture, wrapped so every arm reports the same tally interface (its counters stay zero: cached pages are not attributable per resolution).</summary>
internal sealed class MappedArenaWeightSource : IBenchColdBacking
{
    private readonly GgufMmapWeightSource _backing;

    public MappedArenaWeightSource(GgufFile gguf) => _backing = new GgufMmapWeightSource(gguf);

    public string SubstrateId => "mmap";
    public string BackingMode => "mapped-os-cache";
    public long StackResolutionsTotal => 0L;
    public long BytesStreamedTotal => 0L;

    public ReadOnlySpan<byte> Resolve(LogicalTensor tensor) => _backing.Resolve(tensor);

    public void Dispose() => _backing.Dispose();
}

/// <summary>Bench-side streaming substrate: stacked expert-role tensors answer whole-tensor resolutions from unbuffered FILE_FLAG_NO_BUFFERING reads into fresh buffers, while dense weights answer from ordinary buffered (OS-cached) reads so only expert traffic pays storage cost. No cached mapping of the fixture is held while unbuffered reads run — a cached section over the same bytes is what makes no-buffering reads refuse service. The loader has no per-slab staging today, so a slab view over these sources materializes its role tensor — exactly what both gather arms pay for cold experts here.</summary>
internal unsafe sealed class UnbufferedStackReadWeightSource : IBenchColdBacking
{
    private const int SectorWindow = 4096; // windows align to 4 KiB: valid on 512e and 4Kn volumes alike
    private const int ReadChunkBytes = 1 << 17; // unbuffered reads prefer 128 KiB sector-aligned chunks, shrinking when a storage filter refuses the request size

    private readonly FileStream _dense; // buffered, OS-cached reads for the non-expert tensors — the resident-dense posture a serving engine keeps
    private readonly IntPtr _noBufferHandle;
    private readonly long _fileSize;
    private readonly long _dataStart;
    private byte* _scratch; // aligned read window, grown as needed and freed on dispose
    private int _scratchCapacity;
    private long _stackResolutions;
    private long _bytesStreamed;

    public UnbufferedStackReadWeightSource(string path, GgufFile gguf)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        if (!OperatingSystem.IsWindows())
        {
            throw new PlatformNotSupportedException("FILE_FLAG_NO_BUFFERING streaming reads are the Windows cold-read backing; Linux hosts use the O_DIRECT/discard backing");
        }

        _dense = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1 << 16, FileOptions.SequentialScan);
        _fileSize = gguf.FileSize;
        _dataStart = gguf.DataStart;

        _noBufferHandle = WinColdReadInterop.CreateFileW(path, WinColdReadInterop.GenericRead, WinColdReadInterop.FileShareAll, IntPtr.Zero, WinColdReadInterop.OpenExisting, 0x80 | WinColdReadInterop.NoBuffering, IntPtr.Zero);
        if (_noBufferHandle == WinColdReadInterop.InvalidHandle)
        {
            _dense.Dispose();
            throw new IOException($"stream substrate: CreateFileW('{path}', FILE_FLAG_NO_BUFFERING) failed (win32={Marshal.GetLastWin32Error()})");
        }
    }

    public string SubstrateId => "stream";
    public string BackingMode => "windows no-buffering";
    public long StackResolutionsTotal => Volatile.Read(ref _stackResolutions);
    public long BytesStreamedTotal => Volatile.Read(ref _bytesStreamed);

    public ReadOnlySpan<byte> Resolve(LogicalTensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);

        if (!ExpertRoleNames.IsStackedExpertTensor(tensor.Name))
        {
            byte[] dense = new byte[tensor.SizeBytes]; // dense weights stay cached through buffered reads: the measurement isolates expert traffic, like a serving posture that keeps them hot
            _dense.Seek(_dataStart + tensor.Offset, SeekOrigin.Begin);
            _dense.ReadExactly(dense);
            return new ReadOnlySpan<byte>(dense);
        }

        byte[] payload = ReadUnbuffered(_dataStart + tensor.Offset, tensor.SizeBytes);
        Interlocked.Add(ref _stackResolutions, 1);
        Interlocked.Add(ref _bytesStreamed, tensor.SizeBytes);
        return new ReadOnlySpan<byte>(payload); // one fresh buffer per resolution: spans stay readable for their consumer's lifetime, and the streamed cost is exactly what a staging-less disk provider pays
    }

    private byte[] ReadUnbuffered(long offset, long length)
    {
        if (offset < 0 || length < 0 || offset > _fileSize || length > _fileSize - offset)
        {
            throw new InvalidOperationException($"unbuffered read [{offset}..{offset + length}) does not fit the {_fileSize}-byte fixture");
        }

        byte[] payload = new byte[length];

        long alignedStart = offset & ~(long)(SectorWindow - 1);
        long windowEnd = Math.Min(_fileSize, (offset + length + SectorWindow - 1) & ~(long)(SectorWindow - 1));
        int written = 0; // payload bytes copied out of successive sector-aligned chunks

        for (long chunkOffset = alignedStart; chunkOffset < windowEnd;)
        {
            int chunkLength = (int)Math.Min(ReadChunkBytes, windowEnd - chunkOffset); // every chunk is a whole number of sectors: both ends stay 4 KiB-aligned inside the padded fixture
            EnsureScratch(chunkLength);

            // NO_BUFFERING transfers can return short reads in sector-sized pieces, and some storage filters refuse requests above their own transfer cap outright: fill the chunk with successive aligned sub-reads, halving the request size when a filter rejects one.
            int filled = 0;
            int attemptLength = Math.Min(chunkLength, ReadChunkBytes);
            int shrinkAttemptsAtFloor = 0;

            while (filled < chunkLength)
            {
                int requestLength = Math.Min(attemptLength, chunkLength - filled);
                if (!WinColdReadInterop.SetFilePointerEx(_noBufferHandle, chunkOffset + filled, out _, 0))
                {
                    throw new IOException($"stream substrate: SetFilePointerEx to 0x{chunkOffset + filled:X16} failed (win32={Marshal.GetLastWin32Error()})");
                }

                if (!WinColdReadInterop.ReadFile(_noBufferHandle, (nint)(_scratch + filled), (uint)requestLength, out uint read, IntPtr.Zero) || read == 0)
                {
                    if (attemptLength > SectorWindow)
                    {
                        attemptLength = Math.Max(SectorWindow, attemptLength / 2);
                        continue; // retry the same offset with a smaller aligned request before giving up on this chunk
                    }

                    if (++shrinkAttemptsAtFloor > 4)
                    {
                        throw new IOException($"stream substrate: unbuffered ReadFile of {requestLength} bytes at 0x{chunkOffset + filled:X16} returned {read} (win32={Marshal.GetLastWin32Error()})");
                    }

                    continue; // one more try at the sector floor before failing this chunk
                }

                if ((filled + read) % SectorWindow != 0 && chunkLength - filled > read) // partial transfers must keep the remaining region sector-aligned, or a later aligned sub-read would fault the no-buffering contract
                {
                    throw new IOException($"stream substrate: unbuffered ReadFile returned an unaligned partial transfer of {read} bytes at 0x{chunkOffset + filled:X16}");
                }

                filled += (int)read;
                attemptLength = Math.Min(chunkLength - filled, ReadChunkBytes); // a successful read re-arms the preferred request size
            }

            long copyStart = Math.Max(chunkOffset, offset); // head/tail outside the chunk belong to neighbouring sectors: only this tensor's own bytes are kept — same content a slab view would slice from mmap bytes either way
            long copyEnd = Math.Min(chunkOffset + chunkLength, offset + length);
            if (copyEnd > copyStart)
            {
                fixed (byte* destination = payload)
                {
                    Buffer.MemoryCopy((byte*)_scratch + (copyStart - chunkOffset), destination + written, copyEnd - copyStart, copyEnd - copyStart);
                }

                written += (int)(copyEnd - copyStart);
            }

            chunkOffset += chunkLength;
        }

        if (written != length)
        {
            throw new IOException($"stream substrate: assembled {written} of {length} requested payload bytes");
        }

        return payload;
    }

    private void EnsureScratch(int bytesNeeded)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(_scratchCapacity, 1 << 27); // a growable aligned window stays bounded well under any fixture this lane sizes

        if (_scratchCapacity >= bytesNeeded)
        {
            return;
        }

        int capacity = Math.Max(SectorWindow, _scratchCapacity == 0 ? SectorWindow : _scratchCapacity * 2);
        while (capacity < bytesNeeded && capacity < (1 << 30))
        {
            capacity *= 2;
        }

        if (_scratch != null)
        {
            NativeMemory.AlignedFree(_scratch);
        }

        _scratch = (byte*)NativeMemory.AlignedAlloc((nuint)capacity, SectorWindow);
        _scratchCapacity = capacity;
    }

    public void Dispose()
    {
        if (_scratch != null)
        {
            NativeMemory.AlignedFree(_scratch);
            _scratch = null;
        }

        if (_noBufferHandle != WinColdReadInterop.InvalidHandle && _noBufferHandle != IntPtr.Zero)
        {
            WinColdReadInterop.CloseHandle(_noBufferHandle);
        }

        _dense.Dispose();
    }
}

/// <summary>Linux cold-read backing for the streaming substrate: stacked expert-role tensors answer whole-tensor resolutions through sector-aligned pread calls on an O_DIRECT descriptor — or, when the filesystem refuses direct I/O, on a buffered descriptor whose read range is released afterwards with posix_fadvise(POSIX_FADV_DONTNEED) so every later resolution pays storage again instead of reusing pages this process already fetched. Dense weights answer from ordinary buffered reads, mirroring the Windows backing's resident-dense posture.</summary>
internal unsafe sealed class LinuxColdReadWeightSource : IBenchColdBacking
{
    private const int SectorWindow = 4096; // windows align to 4 KiB: valid on 512e and 4Kn volumes alike
    private const int ReadChunkBytes = 1 << 17; // sector-aligned chunks keep pread requests bounded

    private readonly int _denseFd; // buffered, OS-cached reads for the non-expert tensors — the resident-dense posture a serving engine keeps
    private readonly int _expertFd;
    private readonly bool _direct; // true when the filesystem accepted O_DIRECT on this descriptor
    private readonly long _fileSize;
    private readonly long _dataStart;
    private byte* _scratch; // aligned read window for direct transfers, grown as needed and freed on dispose
    private int _scratchCapacity;
    private long _stackResolutions;
    private long _bytesStreamed;

    public LinuxColdReadWeightSource(string path, GgufFile gguf)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        if (!OperatingSystem.IsLinux())
        {
            throw new PlatformNotSupportedException("the O_DIRECT/discard cold-read backing is the Linux streaming substrate");
        }

        _denseFd = LinuxColdReadInterop.Open(path, 0 /*O_RDONLY*/, 0u);
        if (_denseFd < 0)
        {
            throw new IOException($"stream substrate: open('{path}') failed (errno={Marshal.GetLastPInvokeError()})");
        }

        _expertFd = LinuxColdReadInterop.Open(path, LinuxColdReadInterop.DirectReadFlags, 0u);
        if (_expertFd >= 0)
        {
            _direct = true; // O_DIRECT honored: reads bypass the page cache outright
        }
        else
        {
            _expertFd = LinuxColdReadInterop.Open(path, 0 /*O_RDONLY*/, 0u); // buffered fallback: coldness comes from discarding each read range afterwards
            if (_expertFd < 0)
            {
                int errno = Marshal.GetLastPInvokeError();
                LinuxColdReadInterop.Close(_denseFd);
                throw new IOException($"stream substrate: open('{path}') for expert reads failed (errno={errno})");
            }

            _direct = false;
        }

        _fileSize = gguf.FileSize;
        _dataStart = gguf.DataStart;
    }

    public string SubstrateId => "stream";
    public string BackingMode => _direct ? "linux o_direct+discard" : "linux buffered+discard";
    public long StackResolutionsTotal => Volatile.Read(ref _stackResolutions);
    public long BytesStreamedTotal => Volatile.Read(ref _bytesStreamed);

    public ReadOnlySpan<byte> Resolve(LogicalTensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);

        if (!ExpertRoleNames.IsStackedExpertTensor(tensor.Name))
        {
            byte[] dense = new byte[tensor.SizeBytes]; // dense weights stay cached through buffered reads: the measurement isolates expert traffic, like a serving posture that keeps them hot
            fixed (byte* buffer = dense)
            {
                ReadExactly(_denseFd, _dataStart + tensor.Offset, buffer, checked((int)tensor.SizeBytes));
            }

            return new ReadOnlySpan<byte>(dense);
        }

        byte[] payload = ReadCold(_dataStart + tensor.Offset, tensor.SizeBytes);
        Interlocked.Add(ref _stackResolutions, 1);
        Interlocked.Add(ref _bytesStreamed, tensor.SizeBytes);
        return new ReadOnlySpan<byte>(payload); // one fresh buffer per resolution: spans stay readable for their consumer's lifetime, and the streamed cost is exactly what a staging-less disk provider pays
    }

    private byte[] ReadCold(long offset, long length)
    {
        if (offset < 0 || length < 0 || offset > _fileSize || length > _fileSize - offset)
        {
            throw new InvalidOperationException($"cold read [{offset}..{offset + length}) does not fit the {_fileSize}-byte fixture");
        }

        byte[] payload = new byte[length];

        if (!_direct)
        {
            fixed (byte* buffer = payload)
            {
                ReadExactly(_expertFd, offset, buffer, checked((int)length));
            }

            LinuxColdReadInterop.DiscardRange(_expertFd, offset, length); // release the pages this resolution just fetched: the next resolution of the same bytes pays storage again
            return payload;
        }

        long alignedStart = offset & ~(long)(SectorWindow - 1);
        long windowEnd = Math.Min(_fileSize, (offset + length + SectorWindow - 1) & ~(long)(SectorWindow - 1));
        int written = 0; // payload bytes copied out of successive sector-aligned chunks

        for (long chunkOffset = alignedStart; chunkOffset < windowEnd;)
        {
            int chunkLength = (int)Math.Min(ReadChunkBytes, windowEnd - chunkOffset); // every chunk is a whole number of sectors: both ends stay 4 KiB-aligned inside the padded fixture
            EnsureScratch(chunkLength);

            ReadExactly(_expertFd, chunkOffset, _scratch, chunkLength); // whole-sector aligned transfer: pread's own short-read loop fills the chunk

            long copyStart = Math.Max(chunkOffset, offset); // head/tail outside the chunk belong to neighbouring sectors: only this tensor's own bytes are kept — same content a slab view would slice from mmap bytes either way
            long copyEnd = Math.Min(chunkOffset + chunkLength, offset + length);
            if (copyEnd > copyStart)
            {
                fixed (byte* destination = payload)
                {
                    Buffer.MemoryCopy(_scratch + (copyStart - chunkOffset), destination + written, copyEnd - copyStart, copyEnd - copyStart);
                }

                written += (int)(copyEnd - copyStart);
            }

            chunkOffset += chunkLength;
        }

        if (written != length)
        {
            throw new IOException($"stream substrate: assembled {written} of {length} requested payload bytes");
        }

        LinuxColdReadInterop.DiscardRange(_expertFd, alignedStart, windowEnd - alignedStart); // harmless when O_DIRECT bypassed the cache, and it keeps filesystems that silently buffer direct reads honest too

        return payload;
    }

    private void ReadExactly(int fd, long offset, byte* destination, int length)
    {
        int filled = 0;
        while (filled < length)
        {
            long read = LinuxColdReadInterop.Pread(fd, (nint)(destination + filled), (nuint)(length - filled), offset + filled);
            if (read <= 0)
            {
                throw new IOException($"stream substrate: pread at 0x{offset + filled:X16} returned {read} (errno={Marshal.GetLastPInvokeError()})");
            }

            filled += (int)read; // short reads are legal on Linux descriptors: keep filling the same window
        }
    }

    private void EnsureScratch(int bytesNeeded)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(_scratchCapacity, 1 << 27); // a growable aligned window stays bounded well under any fixture this lane sizes

        if (_scratchCapacity >= bytesNeeded)
        {
            return;
        }

        int capacity = Math.Max(SectorWindow, _scratchCapacity == 0 ? SectorWindow : _scratchCapacity * 2);
        while (capacity < bytesNeeded && capacity < (1 << 30))
        {
            capacity *= 2;
        }

        if (_scratch != null)
        {
            NativeMemory.AlignedFree(_scratch);
        }

        _scratch = (byte*)NativeMemory.AlignedAlloc((nuint)capacity, SectorWindow);
        _scratchCapacity = capacity;
    }

    public void Dispose()
    {
        if (_scratch != null)
        {
            NativeMemory.AlignedFree(_scratch);
            _scratch = null;
        }

        LinuxColdReadInterop.Close(_expertFd);
        LinuxColdReadInterop.Close(_denseFd);
    }
}

internal static class ExpertRoleNames
{
    /// <summary>The fixture's stacked MoE expert roles — the tensors a gather reads per-expert slabs from, and the only ones this lane streams on the streaming substrates.</summary>
    public static bool IsStackedExpertTensor(string tensorName) =>
        tensorName.Contains("ffn_gate_exps", StringComparison.Ordinal) || tensorName.Contains("ffn_up_exps", StringComparison.Ordinal) || tensorName.Contains("ffn_down_exps", StringComparison.Ordinal);
}

internal static class WinColdReadInterop
{
    public const uint GenericRead = 0x8000_0000;
    public const uint GenericWrite = 0x4000_0000;
    public const uint FileShareAll = 0x1 | 0x2 | 0x4; // READ | WRITE | DELETE
    public const uint OpenExisting = 3;
    public const uint NoBuffering = 0x2000_0000; // FILE_FLAG_NO_BUFFERING: reads bypass the system file cache entirely
    public static readonly IntPtr InvalidHandle = new(-1);

    [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr CreateFileW(string lpFileName, uint dwDesiredAccess, uint dwShareMode, IntPtr lpSecurityAttributes, uint dwCreationDisposition, uint dwFlagsAndAttributes, IntPtr hTemplateFile);

    [DllImport("kernel32", SetLastError = true)]
    public static extern bool CloseHandle(IntPtr hObject);

    [DllImport("kernel32", SetLastError = true)]
    public static extern bool SetFilePointerEx(IntPtr hFile, long distanceToMove, out long newOffset, int moveMethod);

    [DllImport("kernel32", SetLastError = true)]
    public static extern bool ReadFile(IntPtr hFile, nint lpBuffer, uint nNumberOfBytesToRead, out uint lpNumberOfBytesRead, IntPtr lpOverlapped);

    /// <summary>PROCESS_MEMORY_COUNTERS.PageFaultCount for the current process — every fault that must fetch a page counts here (soft and hard alike; streamed-byte tallies separate storage reads).</summary>
    public static long PageFaultCountOfCurrentProcess()
    {
        ProcessMemoryCounter counters = new();
        counters.Cb = (uint)Marshal.SizeOf<ProcessMemoryCounter>();

        if (!GetProcessMemoryInfo(new IntPtr(-1), ref counters, counters.Cb)) // -1: the current-process pseudo-handle psapi accepts
        {
            throw new InvalidOperationException($"psapi GetProcessMemoryInfo failed (win32={Marshal.GetLastWin32Error()})");
        }

        return counters.PageFaultCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessMemoryCounter
    {
        public uint Cb;
        public uint PageFaultCount;
        public nuint PeakWorkingSetSize;
        public nuint WorkingSetSize;
        public nuint QuotaPeakPagedPoolUsage;
        public nuint QuotaPagedPoolUsage;
        public nuint QuotaPeakNonPagedPoolUsage;
        public nuint QuotaNonPagedPoolUsage;
        public nuint PagefileUsage;
        public nuint PeakPagefileUsage;
    }

    [DllImport("psapi", SetLastError = true)]
    private static extern bool GetProcessMemoryInfo(IntPtr processHandle, ref ProcessMemoryCounter counters, uint cb);
}

internal static class LinuxColdReadInterop
{
    public const int DirectReadFlags = 0 | 0x4000; // O_RDONLY | O_DIRECT (asm-generic value on x86-64): reads bypass the page cache outright

    private const int PosixFadvDontNeed = 4;

    [DllImport("libc", EntryPoint = "open", SetLastError = true)]
    public static extern int Open(string path, int flags, uint mode);

    [DllImport("libc", EntryPoint = "pread", SetLastError = true)]
    public static extern long Pread(int fd, nint buffer, nuint count, long offset);

    [DllImport("libc", EntryPoint = "close", SetLastError = true)]
    public static extern int Close(int fd);

    // posix_fadvise reports its failure as a returned errno instead of the -1/errno pairing other syscalls use.
    [DllImport("libc", EntryPoint = "posix_fadvise", SetLastError = true)]
    private static extern int PosixFadvise(int fd, long offset, long length, int advice);

    /// <summary>POSIX_FADV_DONTNEED over a read range: pages this process fetched for that window are released so the next resolution of the same bytes pays storage again.</summary>
    public static void DiscardRange(int fd, long offset, long length) => _ = PosixFadvise(fd, offset, length, PosixFadvDontNeed);

    /// <summary>Minor + major fault counters for this process from /proc/self/stat (fields after the command paren: index 7 minor faults, index 9 major faults).</summary>
    public static long PageFaultsOfCurrentProcess()
    {
        try
        {
            string stat = File.ReadAllText("/proc/self/stat");
            int closeParen = stat.LastIndexOf(')');
            if (closeParen < 0 || closeParen + 1 >= stat.Length)
            {
                return 0L;
            }

            // Fields are single-space separated after the parenthesized command, so a comm containing spaces cannot shift the indices.
            string[] fields = stat.Substring(closeParen + 1).Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (fields.Length < 10 || !long.TryParse(fields[7], NumberStyles.Integer, CultureInfo.InvariantCulture, out long minor) || !long.TryParse(fields[9], NumberStyles.Integer, CultureInfo.InvariantCulture, out long major))
            {
                return 0L;
            }

            return minor + major;
        }
        catch (IOException)
        {
            return 0L; // a host without a readable /proc falls back to the streamed-byte tallies alone
        }
    }
}
