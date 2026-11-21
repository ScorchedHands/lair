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
        Console.WriteLine($"gather posture: seqs={seqs} tokensPerSequence={tokens} reps={reps} outRoot={outRoot} engineThreads=auto avx2={Avx2.IsSupported} physicalCores={CpuBackend.DetectedPhysicalCoreCount} logicalProcessors={Environment.ProcessorCount}");

        string fixturePath = Path.Combine(Path.GetTempPath(), $"lair-moe-gather-{Guid.NewGuid():N}.gguf");
        byte[] image = BuildFixtureImage(layers, hidden, heads, headDim, kvHeads, inter, experts, routed, vocab);
        File.WriteAllBytes(fixturePath, image);
        Console.WriteLine($"gather fixture file={fixturePath} bytes={image.Length} ({image.Length / (1024.0 * 1024.0):F1} MiB)");

        try
        {
            return Measure(
                reportId, stamp, startedUtc, outRoot, fixturePath, args, seqs, tokens, reps, layers, hidden, heads, kvHeads, headDim, inter, experts, routed, vocab);
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
        int vocab)
    {
        long expertBytes = (long)inter * hidden * 4; // F32 rows: one expert slab inside each stacked role
        using CpuBackend backend = new(allowAvx2: true);

        using GgufFile gatherGguf = GgufFile.Open(fixturePath);
        using GgufMmapWeightSource gatherBacking = new(gatherGguf);
        CountingSlabSource counting = new(gatherBacking);
        using TransformerModel gatherModel = TransformerModel.Open(gatherGguf, counting, backend, maxContextTokens: tokens + 4);

        int vocabFromFile = gatherModel.Desc.VocabSize;
        if (vocabFromFile != vocab)
        {
            Console.WriteLine($"gather: fixture vocab mismatch ({vocabFromFile} vs {vocab})");
            return 1;
        }

        using GgufFile soloGguf = GgufFile.Open(fixturePath);
        using TransformerModel soloModel = TransformerModel.Open(soloGguf, backend, maxContextTokens: tokens + 4);

        int capacityBlocks = ((tokens + 16) / 16 + 1) * seqs + 2; // every sequence holds its own blocks in the shared pool simultaneously
        using BlockPool gatherPool = NewPool(gatherModel, backend, capacityBlocks);
        using BlockPool soloPool = NewPool(soloModel, backend, capacityBlocks);

        // Warm both engines once outside the timed reps so thread pools and page residency settle before measurement.
        RunGatherStream(gatherModel, gatherPool, seqs, tokens, vocab);
        RunSoloStream(soloModel, soloPool, seqs, tokens, vocab);

        double[] gatherSeconds = new double[reps];
        for (int rep = 0; rep < reps; rep++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            RunGatherStream(gatherModel, gatherPool, seqs, tokens, vocab);
            sw.Stop();
            gatherSeconds[rep] = sw.Elapsed.TotalSeconds;
        }

        double[] soloSeconds = new double[reps];
        for (int rep = 0; rep < reps; rep++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            RunSoloStream(soloModel, soloPool, seqs, tokens, vocab);
            sw.Stop();
            soloSeconds[rep] = sw.Elapsed.TotalSeconds;
        }

        double gatherMedian = Median(gatherSeconds);
        double soloMedian = Median(soloSeconds);
        long sequenceTokens = (long)seqs * tokens;
        double gatherTps = sequenceTokens / gatherMedian;
        double soloTps = sequenceTokens / soloMedian;

        Console.WriteLine($"gather arm: merged-decode median={gatherMedian:F4}s min={Min(gatherSeconds):F4} max={Max(gatherSeconds):F4} tok/s={gatherTps:F2}");
        Console.WriteLine($"dense-equivalent arm: solo-stepping median={soloMedian:F4}s min={Min(soloSeconds):F4} max={Max(soloSeconds):F4} tok/s={soloTps:F2}");

        // Parity pass: same token streams through both arms, every step's logits captured and compared bit-for-bit.
        float[] gatherLogits = new float[seqs * tokens * vocab];
        RunGatherStream(gatherModel, gatherPool, seqs, tokens, vocab, gatherLogits);
        float[] soloLogits = new float[seqs * tokens * vocab];
        RunSoloStream(soloModel, soloPool, seqs, tokens, vocab, soloLogits);

        int mismatchedFloats = 0;
        double maxAbsDiff = 0.0;
        for (int i = 0; i < gatherLogits.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(gatherLogits[i]) != BitConverter.SingleToInt32Bits(soloLogits[i]))
            {
                mismatchedFloats++;
                maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs((double)gatherLogits[i] - soloLogits[i]));
            }
        }

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

        string reportDir = Path.Combine(outRoot, stamp);
        Directory.CreateDirectory(reportDir);

        HostReport host = CaptureHost(backend, seqs, tokens, reps);
        BenchReport bench = new()
        {
            SchemaVersion = 1,
            ReportId = reportId,
            Arms =
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
            ],
            Parity = new ParityResult
            {
                BitExact = bitExact,
                ComparedFloats = gatherLogits.Length,
                MismatchedFloats = mismatchedFloats,
                MaxAbsDiff = bitExact ? 0.0 : Round(maxAbsDiff),
            },
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
            ComparisonNotes =
            [
                "The fixture is synthetic deterministic F32 qwen2moe material: it measures the gather mechanism (union walks, slab views, prefetch staging), not a quantized production model.",
                "Both arms run identical sequences and token streams over identical file bytes; parity compares every step's logits bit-for-bit between the merged-decode arm and solo stepping.",
                "Union stats come from an instrumentation pass that counts slab resolutions through the gather kernel: unionGateResolutionsPerDecodeStep lists gate-role resolutions per decode step summed over the fixture's layers, while averages and byte tallies are reported per layer step — each distinct expert of a step's union resolves exactly one slab per role.",
            ],
        };

        WriteJson(Path.Combine(reportDir, "manifest.json"), manifest);
        WriteJson(Path.Combine(reportDir, "host.json"), host);
        WriteJson(Path.Combine(reportDir, "bench.json"), bench);
        WriteJson(
            Path.Combine(reportDir, "cli.json"),
            new { schemaVersion = 1, reportId, commandLine = Environment.CommandLine + " " + string.Join(' ', invokedArgs), workingDirectory = Directory.GetCurrentDirectory() });

        File.WriteAllText(Path.Combine(reportDir, "report.html"), BuildHtml(manifest, host, bench));

        Console.WriteLine($"gather report dir={Path.GetFullPath(reportDir)}");
        return bitExact ? 0 : 1;
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
    private static double Max(double[] values) => values.Max();
    private static double Round(double value) => Math.Round(value, 4);

    private static int ParseInt(string raw, int fallback) => int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) ? v : fallback;

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
        [JsonPropertyName("arms")] public List<ArmResult> Arms { get; init; } = [];
        [JsonPropertyName("parity")] public ParityResult Parity { get; init; } = new();
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
        string parityRow = $"<tr><td>{bench.Parity.ComparedFloats}</td><td>{bench.Parity.MismatchedFloats}</td><td class=\"{(bench.Parity.BitExact ? "ok" : "bad")}\">{(bench.Parity.BitExact ? "bit-exact" : "diverged")}</td></tr>";
        string unionRow = $"<tr><td>{bench.UnionStats.SelectionSlotsPerLayerStep}</td><td>{bench.UnionStats.AverageDistinctExpertsGatheredPerLayerStep:F2}</td><td>{bench.UnionStats.AmortizationPercent:F1}%</td><td>{bench.UnionStats.GatheredBytesPerLayerStep / (1024.0 * 1024.0):F1}</td><td>{bench.UnionStats.DenseEquivalentBytesPerLayerStep / (1024.0 * 1024.0):F1}</td></tr>";
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
<tr><th>Compared floats</th><th>Mismatched</th><th>Exact</th></tr>
{{parityRow}}
</table>

<h2>Union active-expert amortization (instrumentation pass)</h2>
<table>
<tr><th>Selection slots / layer step</th><th>Avg distinct experts gathered</th><th>Amortized</th><th>Gathered MiB / layer step</th><th>Dense-equivalent MiB / layer step</th></tr>
{{unionRow}}
</table>

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
