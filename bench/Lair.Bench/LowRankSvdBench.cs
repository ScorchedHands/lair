using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Loader;

namespace Lair.Bench;

/// <summary>
/// Analytical low-rank viability spike for stacked MoE expert tensors: per-expert slabs of the gate/up/down
/// stacks are resolved through the gather seam's slab views (contiguous byte ranges over the mmap mapping, one
/// expert at a time — never a whole-stack copy), every row dequantizes to float32 through the engine's own CPU
/// dequant route, and each matrix's exact singular spectrum comes from numpy.linalg.svd driven by a spike script.
/// Rank-energy fractions, per-role aggregate medians, an active-routed-expert bytes-per-token baseline read off
/// the assembled tensor table, and conservative fp32-factorization storage estimates for candidate ranks publish
/// as a timestamped report directory (manifest.json, host.json, bench.json, cli.json, report.html). This is
/// analytical viability evidence — it measures how much energy a matrix's leading singular directions hold; it
/// benchmarks no low-rank kernel and changes no posture or config.
/// </summary>
public static class LowRankSvdBench
{
    private static readonly int[] EnergyLadderRanks = [8, 16, 32, 64, 96, 128, 192, 256]; // published cumulative-energy fractions per matrix
    private static readonly double[] EnergyThresholdsPercent = [50.0, 75.0, 90.0, 95.0, 99.0]; // rank needed to reach each fraction of total energy
    private static readonly int[] StorageCandidateRanks = [8, 16, 32, 64, 96, 128]; // conservative fp32-factorization element counts

    public static int Run(string[] args)
    {
        if (args.Length < 2 || string.IsNullOrEmpty(args[1]))
        {
            Console.WriteLine("usage: lowrank-svd <gguf path> [--layers 0,16,32] [--experts 0,7] [--roles gate,up,down] [--out reports]");
            return 1;
        }

        List<int> layers = [0, 16, 32];
        List<int> experts = [0, 7];
        List<string> roles = ["gate", "up", "down"];
        string outRoot = "reports";

        for (int i = 2; i < args.Length; i++)
        {
            if (i + 1 >= args.Length)
            {
                Console.WriteLine($"lowrank-svd: option '{args[i]}' needs a value");
                return 1;
            }

            switch (args[i])
            {
                case "--layers":
                    layers = ParseIntList(args[++i]);
                    break;
                case "--experts":
                    experts = ParseIntList(args[++i]);
                    break;
                case "--roles":
                    roles = [.. args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                        .Select(r => r.ToLowerInvariant())];
                    break;
                case "--out":
                    outRoot = args[++i];
                    break;
                default:
                    Console.WriteLine($"lowrank-svd: unknown option '{args[i]}'");
                    return 1;
            }
        }

        layers = [.. layers.Where(l => l >= 0).Distinct().OrderBy(l => l)];
        experts = [.. experts.Where(e => e >= 0).Distinct().OrderBy(e => e)];
        roles = [.. roles.Distinct()];

        if (layers.Count == 0 || experts.Count == 0)
        {
            Console.WriteLine("lowrank-svd: --layers and --experts must list non-negative indices");
            return 1;
        }

        foreach (string role in roles)
        {
            if (role is not ("gate" or "up" or "down"))
            {
                Console.WriteLine($"lowrank-svd: unknown role '{role}' (expected gate, up, or down)");
                return 1;
            }
        }

        if (roles.Count == 0)
        {
            Console.WriteLine("lowrank-svd: --roles listed no roles");
            return 1;
        }

        DateTime startedUtc = DateTime.UtcNow;
        string stamp = startedUtc.ToString("yyyy'-'MM'-'dd'T'HH'-'mm'-'ss'Z'", CultureInfo.InvariantCulture);
        string reportId = $"lowrank-svd-{stamp}";

        using GgufFile gguf = GgufFile.Open(args[1]); // an assembled split opens whole: sibling shards join into one logical table
        string arch = gguf.Metadata.GetString("general.architecture", "general");
        int blockCount = gguf.Metadata.GetInt($"{arch}.block_count", 0);
        int expertCount = gguf.Metadata.GetInt($"{arch}.expert_count", 0);
        int usedCount = gguf.Metadata.GetInt($"{arch}.expert_used_count", 0);
        int hidden = gguf.Metadata.GetInt($"{arch}.embedding_length", 0);
        int inter = gguf.Metadata.GetInt($"{arch}.expert_feed_forward_length", 0);

        Console.WriteLine($"lowrank-svd model={gguf.Path} arch={arch} blocks={blockCount} experts={expertCount} used={usedCount} hidden={hidden} expertInter={inter} shards={(gguf.Shards is null ? "single-file" : gguf.Shards.Count.ToString(CultureInfo.InvariantCulture))}");

        if (blockCount <= 0 || expertCount <= 0 || usedCount <= 0 || usedCount > expertCount || hidden <= 0 || inter <= 0)
        {
            Console.WriteLine("lowrank-svd: model metadata lacks the MoE geometry needed for this spike (block_count/expert_count/expert_used_count/embedding_length/expert_feed_forward_length)");
            return 1;
        }

        if (layers.Exists(l => l >= blockCount))
        {
            Console.WriteLine($"lowrank-svd: a requested layer index reaches past the model's {blockCount} layers");
            return 1;
        }

        if (experts.Exists(e => e >= expertCount))
        {
            Console.WriteLine($"lowrank-svd: a requested expert index reaches past the model's {expertCount} experts");
            return 1;
        }

        Dictionary<string, LogicalTensor> stacksByName = new(StringComparer.Ordinal);
        foreach (LogicalTensor tensor in gguf.Tensors.Tensors)
        {
            stacksByName[tensor.Name] = tensor; // assembled split tables are disjoint by construction
        }

        List<StackGeometry> selected = [];
        List<string> skippedNotes = [];
        foreach (int layer in layers)
        {
            foreach (string role in roles)
            {
                string name = $"blk.{layer}.ffn_{role}_exps.weight";
                if (!stacksByName.TryGetValue(name, out LogicalTensor? stack))
                {
                    skippedNotes.Add($"{name}: not present in the assembled tensor table");
                    continue;
                }

                (int rows, int cols) = role == "down" ? (hidden, inter) : (inter, hidden);
                int[] dims = stack.Shape.Dims;
                if (dims.Length != 3 || dims[0] != expertCount || dims[1] != rows || dims[2] != cols)
                {
                    skippedNotes.Add($"{name}: dims [{string.Join(",", dims)}] do not match the expected stacked shape [{expertCount},{rows},{cols}]");
                    continue;
                }

                if (stack.SizeBytes % expertCount != 0)
                {
                    skippedNotes.Add($"{name}: payload {stack.SizeBytes} bytes does not divide evenly across {expertCount} experts");
                    continue;
                }

                long perExpert = stack.SizeBytes / expertCount;
                TensorShape expertShape = new([rows, cols]);
                long stride = expertShape.RowStrideBytes(stack.Format);
                if (stride <= 0 || stride > int.MaxValue || rows * stride != perExpert)
                {
                    skippedNotes.Add($"{name}: a {expertCount}-way split gives {perExpert} bytes/expert but the row packing wants {rows} x {stride}; contiguous-expert geometry does not hold");
                    continue;
                }

                selected.Add(new StackGeometry(name, stack.Format, rows, cols, perExpert, stride));
            }
        }

        if (selected.Count == 0)
        {
            Console.WriteLine("lowrank-svd: no sampled stacked expert tensor survived its geometry checks");
            foreach (string note in skippedNotes)
            {
                Console.WriteLine($"  skip {note}");
            }

            return 1;
        }

        foreach (string note in skippedNotes)
        {
            Console.WriteLine($"lowrank-svd: skipping {note}");
        }

        using CpuBackend backend = new(allowAvx2: true); // engine dequant route on this host's best supported width, called row-by-row serially
        using GgufMmapWeightSource backing = new(gguf); // one mapping set for the whole run; slab views borrow it and matrices are copied out before the next resolution
        ExpertSlabWeightSource slabs = new(backing);

        string tempDir = Path.Combine(Path.GetTempPath(), $"lair-lowrank-svd-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            Console.WriteLine($"lowrank-svd sampling: layers=[{string.Join(",", layers)}] experts=[{string.Join(",", experts)}] roles=[{string.Join(",", roles)}] matrices={selected.Count * experts.Count} peakMatrixResidency={(long)selected.Max(s => (long)s.Rows * s.Cols) * sizeof(float)} bytes tempDir={tempDir}"); // the largest single dequantized expert matrix any step of this run holds resident

            List<MatrixExport> exports = [];
            long peakMatrixBytes = 0;
            foreach (StackGeometry geometry in selected)
            {
                LogicalTensor stack = stacksByName[geometry.Name];

                foreach (int expert in experts)
                {
                    ReadOnlySpan<byte> slab = slabs.ResolveSlab(stack, expert, geometry.ExpertBytes); // contiguous per-expert view: the same seam a gather walks, zero copy

                    long elements = checked((long)geometry.Rows * geometry.Cols);
                    float[] matrix = new float[checked((int)elements)]; // one dequantized expert resident at a time; never the whole stack
                    WeightRef w = new(slab, new TensorShape([geometry.Rows, geometry.Cols]), geometry.Format);

                    int nonFiniteValues = 0;
                    for (int row = 0; row < geometry.Rows; row++)
                    {
                        backend.DequantRow(w, row, matrix.AsSpan(row * geometry.Cols, geometry.Cols)); // engine CPU dequant route into float32 rows
                    }

                    foreach (float value in matrix)
                    {
                        if (!float.IsFinite(value))
                        {
                            nonFiniteValues++;
                        }
                    }

                    string id = $"{geometry.Name}#e{expert}"; // unique across the run: tensor name plus expert index
                    if (nonFiniteValues > 0 || !float.IsFinite(matrix[0]))
                    {
                        Console.WriteLine($"lowrank-svd: dequantized matrix {id} carries {nonFiniteValues} non-finite value(s)");
                        return 1;
                    }

                    string fileName = $"m_{MatrixSlug(geometry, expert)}.f32.bin";
                    using (FileStream file = File.OpenWrite(Path.Combine(tempDir, fileName)))
                    {
                        file.Write(MemoryMarshal.AsBytes(matrix.AsSpan())); // raw little-endian float32 row-major for the spectrum script
                    }

                    peakMatrixBytes = Math.Max(peakMatrixBytes, elements * sizeof(float));
                    exports.Add(new MatrixExport(id, geometry.Name, LayerOf(geometry.Name), expert, RoleOf(geometry.Name), geometry.Format.ToString(), geometry.Rows, geometry.Cols, geometry.StrideBytes, geometry.ExpertBytes, fileName));

                    Console.WriteLine($"lowrank-svd exported {id} format={geometry.Format} rows={geometry.Rows} cols={geometry.Cols} packedExpertBytes={geometry.ExpertBytes} fp32MatrixBytes={elements * sizeof(float)}");
                }
            }

            string? scriptPath = ResolveSpikeScript();
            if (scriptPath is null)
            {
                Console.WriteLine("lowrank-svd: the spike spectrum script spikes/lowrank-svd.py was not found next to this binary or under bench/Lair.Bench");
                return 1;
            }

            string? probe = RunPythonProbe(); // `python -c "import numpy"` posture check plus version evidence, before any exact work is attempted
            if (probe is null)
            {
                Console.WriteLine("lowrank-svd: python with numpy is unavailable on this host — exact singular spectra are required for this spike, and no managed approximation stands in for them");
                return 1;
            }

            Console.WriteLine($"lowrank-svd probe: {probe} script={scriptPath}");

            string indexPath = Path.Combine(tempDir, "index.json");
            WriteJson(indexPath, new PythonIndex { Matrices = [.. exports.Select(e => new PythonMatrixEntry { Id = e.Id, File = Path.GetFullPath(Path.Combine(tempDir, e.FileName)), Rows = e.Rows, Cols = e.Cols })] });

            string spectraPath = Path.Combine(tempDir, "spectra-out.json");
            int pythonExit = RunPythonScript(scriptPath, indexPath, spectraPath);
            if (pythonExit != 0 || !File.Exists(spectraPath))
            {
                Console.WriteLine($"lowrank-svd: the spectrum script failed with exit code {pythonExit} — exact singular spectra are required; no approximation was substituted");
                return 1;
            }

            PythonSpectra? spectra = JsonSerializer.Deserialize<PythonSpectra>(File.ReadAllText(spectraPath), new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            if (spectra is null)
            {
                Console.WriteLine("lowrank-svd: the spectrum script produced no readable result JSON");
                return 1;
            }

            Dictionary<string, PythonMatrixSpectrum> byId = [];
            foreach (PythonMatrixSpectrum spectrum in spectra.Matrices)
            {
                byId[spectrum.Id] = spectrum;
            }

            List<MatrixResult> results = [];
            bool crossCheckPassed = true;
            foreach (MatrixExport export in exports)
            {
                if (!byId.TryGetValue(export.Id, out PythonMatrixSpectrum? spectrum))
                {
                    Console.WriteLine($"lowrank-svd: the spectrum script returned no result for matrix {export.Id}");
                    return 1;
                }

                MatrixResult result = DeriveMatrixResult(export, spectrum);
                crossCheckPassed &= result.SpectraFrobeniusCrossCheckRelDiff < 1e-4; // sum(sigma^2) must match the independently measured ||A||_F^2 well inside any float32 rounding slack
                results.Add(result);

                Console.WriteLine(
                    $"lowrank-svd spectrum {export.Id} sigmaTop=[{string.Join(" ", result.TopSingularValues.Take(4).Select(v => v.ToString("G6", CultureInfo.InvariantCulture)))}] " +
                    $"energy@8={Energy(result, 8):F5} energy@128={Energy(result, 128):F5} rank90={result.RankThresholds[90]} rank95={result.RankThresholds[95]}");
            }

            List<RoleAggregate> aggregates = AggregateByRole(results);
            foreach (RoleAggregate aggregate in aggregates)
            {
                Console.WriteLine(
                    $"lowrank-svd role {aggregate.Role}: matrices={aggregate.MatrixCount} medianRank50={aggregate.RankThresholds[50]} " +
                    $"medianRank75={aggregate.RankThresholds[75]} medianRank90={aggregate.RankThresholds[90]} " +
                    $"medianRank95={aggregate.RankThresholds[95]} medianRank99={aggregate.RankThresholds[99]}");
            }

            BaselineAnalysis baseline = ComputeBaseline(stacksByName, blockCount, expertCount, usedCount);
            Console.WriteLine($"lowrank-svd baseline stackedExpertBytes={baseline.StackedExpertTotalBytes} activePackedBytesPerToken={baseline.CurrentActivePackedBytesPerToken:F0} fullFp32ReferenceBytesPerToken={baseline.FullFp32ReferenceBytesPerToken:F0}");

            List<LowRankEstimate> estimates = ComputeLowRankEstimates(baseline, hidden, inter);
            foreach (LowRankEstimate estimate in estimates)
            {
                Console.WriteLine(
                    $"lowrank-svd lowrank r={estimate.Rank}: estimatedActiveFp32BytesPerToken={estimate.EstimatedActiveBytesPerToken:F0} " +
                    $"reductionVsPackedBaselinePercent={estimate.ReductionVsCurrentQuantizedPercent:F2} reductionVsFullFp32Percent={estimate.ReductionVsFullFp32ReferencePercent:F2}");
            }

            string recommendation = JudgeViability(results, aggregates, out string criteriaMet);
            Console.WriteLine($"lowrank-svd viability: {recommendation} ({criteriaMet})");

            DateTime finishedUtc = DateTime.UtcNow;

            BenchReport bench = new()
            {
                SchemaVersion = 1,
                ReportId = reportId,
                SampleConfig = new SampleConfig
                {
                    ModelPath = Path.GetFullPath(args[1]),
                    Architecture = arch,
                    BlockCount = blockCount,
                    ExpertCount = expertCount,
                    UsedExpertCount = usedCount,
                    HiddenSize = hidden,
                    ExpertFeedForwardLength = inter,
                    Layers = [.. layers],
                    Experts = [.. experts],
                    Roles = [.. roles],
                },
                PeakMatrixBytes = peakMatrixBytes,
                DequantRoute = "CpuBackend.DequantRow over contiguous per-expert slab views (ExpertSlabWeightSource on GgufMmapWeightSource)",
                SpectrumTool = probe,
                SpectraCrossCheckPassed = crossCheckPassed,
                Matrices = [.. results],
                RoleAggregates = aggregates,
                Baseline = baseline,
                LowRankEstimates = estimates,
                Viability = new ViabilityResult
                {
                    Recommendation = recommendation,
                    CriteriaMet = criteriaMet,
                    DownMatricesSampled = results.Count(r => r.Role == "down"),
                    GateUpMatricesSampled = results.Count(r => r.Role is "gate" or "up"),
                },
            };

            List<string> notes =
                [
                    "Analytical viability evidence only: this measures how much of each dequantized expert matrix's energy its leading singular directions hold. It is not a low-rank kernel throughput benchmark and changes no posture or config.",
                    "Matrices come from the checkpoint as stored: gate/up/down slabs are already sub-4-bit quantizations, so their spectra include quantization-error energy spread across every mode; a rank-k collapse measured here is an upper bound on what a low-rank sidecar could retain before accuracy costs bite.",
                    "Storage estimates assume fp32 factor coefficients/bases (min(m*n, m*r + n*r) elements per expert). The current baseline they compare against is the checkpoint's own packed quantized bytes, so positive reduction percentages mean the fp32 sidecar beats today's storage and negative ones say it does not yet.",
                    "Sampling is deterministic and bounded: a fixed layer/expert subset of every stacked role tensor; spectra are exact numpy.linalg.svd results over full dequantized matrices, one expert resident at a time (peak matrix residency reported in bench.json).",
                ];

            ManifestReport manifest = new()
            {
                SchemaVersion = 1,
                ReportId = reportId,
                GeneratedAtUtc = startedUtc.ToString("yyyy'-'MM'-'dd'T'HH':'mm':'ss'Z'", CultureInfo.InvariantCulture),
                Phase = "MVP7",
                Branch = DetectBranch(),
                Lane = "Local CPU analytical spike (exact numpy singular spectra over dequantized Flash-Next expert slabs)",
                Goal = "Measure whether the stacked MoE expert matrices of this checkpoint carry a low-rank knee worth a factorization experiment, against an exact active-bytes-per-token baseline from the assembled tensor table.",
                Files = ["manifest.json", "host.json", "bench.json", "cli.json", "report.html"],
                ComparisonNotes = notes,
            };

            HostReport host = CaptureHost(backend);

            string reportDir = Path.Combine(outRoot, stamp);
            Directory.CreateDirectory(reportDir);

            WriteJson(Path.Combine(reportDir, "manifest.json"), manifest);
            WriteJson(Path.Combine(reportDir, "host.json"), host);
            WriteJson(Path.Combine(reportDir, "bench.json"), bench);
            WriteJson(
                Path.Combine(reportDir, "cli.json"),
                new { schemaVersion = 1, reportId, commandLine = Environment.CommandLine + " " + string.Join(' ', args), workingDirectory = Directory.GetCurrentDirectory(), elapsedSeconds = Math.Round((finishedUtc - startedUtc).TotalSeconds, 3) });

            File.WriteAllText(Path.Combine(reportDir, "report.html"), BuildHtml(manifest, host, bench));

            Console.WriteLine($"lowrank-svd report dir={Path.GetFullPath(reportDir)}");
            return 0;
        }
        finally
        {
            try
            {
                Directory.Delete(tempDir, recursive: true); // exported matrices and the script's exchange files are intermediates; every published number lives in the report directory
            }
            catch (IOException)
            {
            }
        }
    }

    // ---------------------------------------------------------------- spectra derivation + aggregates

    private static MatrixResult DeriveMatrixResult(MatrixExport export, PythonMatrixSpectrum spectrum)
    {
        double[] s = spectrum.SingularValues;
        if (s.Length == 0 || s[0] <= 0.0)
        {
            throw new InvalidOperationException($"matrix {export.Id} produced an empty or zero-maximum spectrum");
        }

        double sumSq = 0.0;
        for (int i = 0; i < s.Length; i++)
        {
            sumSq += s[i] * s[i];
        }

        double relDiff = Math.Abs(sumSq - spectrum.FrobeniusNormSquared) / Math.Max(1e-30, spectrum.FrobeniusNormSquared); // the script measures ||A||_F^2 independently of its own svd output: agreement proves spectra and matrix agree

        List<double> cumulative = new(s.Length);
        double running = 0.0;
        for (int i = 0; i < s.Length; i++)
        {
            running += s[i] * s[i];
            cumulative.Add(running / sumSq); // descending-singular-value order: index i is the energy fraction of the leading (i+1) directions
        }

        Dictionary<int, double> energyAtRanks = [];
        foreach (int rank in EnergyLadderRanks)
        {
            if (rank <= s.Length)
            {
                energyAtRanks[rank] = Round(cumulative[rank - 1]); // cumulative fraction at this ladder rank; ranks past min(m,n) are omitted because the spectrum is exhausted there
            }
        }

        Dictionary<int, int> rankThresholds = [];
        foreach (int threshold in RankKeyOrder)
        {
            double fraction = threshold / 100.0;
            int rank = 1;
            while (rank < cumulative.Count && cumulative[rank - 1] < fraction)
            {
                rank++;
            }

            rankThresholds[threshold] = rank; // the leading-rank count whose energy first reaches the threshold; min(m,n) when none does before exhaustion
        }

        return new MatrixResult
        {
            Id = export.Id,
            TensorName = export.TensorName,
            Layer = export.Layer,
            Expert = export.Expert,
            Role = export.Role,
            Format = export.Format,
            Rows = export.Rows,
            Cols = export.Cols,
            PackedExpertBytes = export.PackedExpertBytes,
            Fp32MatrixBytes = (long)export.Rows * export.Cols * sizeof(float),
            FrobeniusNormSquared = Round(spectrum.FrobeniusNormSquared),
            SpectraFrobeniusCrossCheckRelDiff = relDiff < 1e-9 ? 0.0 : Math.Round(relDiff, 8),
            TopSingularValues = [.. s.Select(v => Math.Round(v, 6))], // full descending spectrum: min(rows, cols) values — every published ladder rank (max 256) sits inside it here
            EnergyAtRanks = energyAtRanks,
            RankThresholds = rankThresholds,
        };
    }

    private static List<RoleAggregate> AggregateByRole(List<MatrixResult> results) =>
        [.. results.GroupBy(r => r.Role).OrderBy(g => g.Key, StringComparer.Ordinal).Select(g =>
        {
            MatrixResult[] group = [.. g];
            int spectrumLength = group.Min(m => m.TopSingularValues.Count); // every sampled matrix of the role carries min(rows, cols) values: medians pair index-for-index across them

            double[] medianSpectrum = new double[spectrumLength];
            for (int i = 0; i < spectrumLength; i++)
            {
                // Element-wise medians across the role's sampled matrices: index i is each matrix's (i+1)-th singular value, never a cross-matrix mode of one spectrum.
                double[] column = [.. group.Select(m => m.TopSingularValues[i])];
                medianSpectrum[i] = Round(MedianOf(column));
            }

            Dictionary<int, double> medianEnergyLadder = [];
            foreach (int rank in EnergyLadderRanks)
            {
                double[] perMatrix = [.. group.Where(m => m.EnergyAtRanks.ContainsKey(rank)).Select(m => m.EnergyAtRanks[rank])];
                if (perMatrix.Length > 0)
                {
                    medianEnergyLadder[rank] = Round(MedianOf(perMatrix)); // element-wise median cumulative-energy fraction per ladder rank across the role's sampled matrices
                }
            }

            Dictionary<int, int> rankThresholds = [];
            foreach (int threshold in RankKeyOrder)
            {
                int[] perMatrixRanks = [.. group.Select(m => m.RankThresholds[threshold])];
                rankThresholds[threshold] = MedianOf(perMatrixRanks); // per-role medians of the sampled matrices' rank-to-threshold values — these are what the viability rule reads alongside the raw per-matrix table
            }

            return new RoleAggregate
            {
                Role = g.Key,
                MatrixCount = group.Length,
                MedianSpectrumTop256 = [.. medianSpectrum.Take(256)], // published aggregate spectrum: element-wise medians of the sampled matrices' descending spectra (first 256 singular values)
                MedianEnergyLadder = medianEnergyLadder,
                RankThresholds = rankThresholds,
            };
        })];

    private static string JudgeViability(List<MatrixResult> results, List<RoleAggregate> aggregates, out string criteriaMet)
    {
        int[] downRank90 = [.. results.Where(r => r.Role == "down").Select(r => r.RankThresholds[90])];
        if (downRank90.Length < 3)
        {
            criteriaMet = $"inconclusive: the decision reads the down role's rank requirements and needs at least 3 sampled down matrices, but it carries {downRank90.Length}";
            return "inconclusive";
        }

        int[] downRank95 = [.. results.Where(r => r.Role == "down").Select(r => r.RankThresholds[95])];
        double medianDown90 = MedianOf(downRank90); // medians over the sampled matrices' own thresholds, not over role-summary values: raw per-matrix evidence is what the rule compares
        int medianDown95 = MedianOf(downRank95);

        RoleAggregate[] gateUpContext = [.. aggregates.Where(a => a.Role is "gate" or "up" && a.MatrixCount > 0)];
        string gateUpMedian90 = gateUpContext.Length > 0 ? MedianOf([.. gateUpContext.Select(a => (double)a.RankThresholds[90])]).ToString("F1", CultureInfo.InvariantCulture) : "n/a"; // gate/up inform the read but never decide it: their medians ride in bench.json for context

        criteriaMet = $"down-role median 90%-energy rank {medianDown90:F1} (rule wants <=64) and median 95% rank {medianDown95} (rule wants <=128); gate/up aggregate median 90% rank {gateUpMedian90} for context";
        return medianDown90 <= 64.0 && medianDown95 <= 128 ? "possible_low_rank" : "no_knee"; // strong knee only when the output/down projections collapse this early; near-flat spectra on an already sub-4-bit checkpoint mean no room for a low-rank sidecar to win
    }

    private static readonly int[] RankKeyOrder = [50, 75, 90, 95, 99];

    // ---------------------------------------------------------------- baseline + storage analysis (whole assembled table)

    private static BaselineAnalysis ComputeBaseline(Dictionary<string, LogicalTensor> stacksByName, int blockCount, int expertCount, int usedCount)
    {
        long stackedTotal = 0;
        Dictionary<string, long> roleTotals = new() { ["gate"] = 0, ["up"] = 0, ["down"] = 0 };
        Dictionary<string, int> roleLayerCounts = new() { ["gate"] = 0, ["up"] = 0, ["down"] = 0 };
        long fullFp32ElementsPerExpertSum = 0; // summed per-expert element counts across every stacked tensor: the dequantized-fp32 reference for an active-token weighting

        List<LayerRoleBytes> perLayer = [];
        for (int layer = 0; layer < blockCount; layer++)
        {
            long layerBytes = 0;
            foreach (string role in new[] { "gate", "up", "down" })
            {
                if (!stacksByName.TryGetValue($"blk.{layer}.ffn_{role}_exps.weight", out LogicalTensor? stack) || stack.Shape.Dims.Length != 3 || stack.Shape.Dims[0] != expertCount)
                {
                    continue; // tallies only what the table actually carries: a missing or malformed stacked role keeps its layer count short instead of guessing bytes
                }

                long elementsPerExpert = checked((long)stack.Shape.Dims[1] * stack.Shape.Dims[2]);
                stackedTotal += stack.SizeBytes;
                roleTotals[role] += stack.SizeBytes;
                roleLayerCounts[role]++;
                fullFp32ElementsPerExpertSum += elementsPerExpert; // per expert: every layer's three roles contribute their own m*n, and the active weighting scales by used_count below
                layerBytes += stack.SizeBytes;
            }

            if (layerBytes > 0)
            {
                perLayer.Add(new LayerRoleBytes { Layer = layer, StackedPackedBytes = layerBytes }); // one row per contributing layer: what an un-routed full sweep of this layer's MoE FFN would read across gate/up/down stacks
            }
        }

        double activeRatio = (double)usedCount / expertCount; // the router's active share: each token touches used_count experts out of expert_count in every sparse layer

        return new BaselineAnalysis
        {
            ExpertCount = expertCount,
            UsedExpertCount = usedCount,
            ActiveRatio = Round(activeRatio),
            StackedExpertTotalBytes = stackedTotal, // every layer's gate/up/down stacked packed bytes summed across the assembled table — what a full expert sweep reads per token if no routing happened
            CurrentActivePackedBytesPerToken = (long)Math.Round(stackedTotal * activeRatio), // exact baseline: that total scaled by used/expert_count, the router's active share of every layer step
            FullFp32ReferenceBytesPerToken = checked((long)Math.Round(fullFp32ElementsPerExpertSum * sizeof(float) * expertCount * activeRatio)), // same weighting against dequantized fp32 storage — the factorization comparison's second axis
            PerRoleTotalsPackedBytes = [.. roleTotals.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => new RoleTotalRow { Role = kv.Key, TotalStackedPackedBytes = kv.Value, LayersWithRole = roleLayerCounts[kv.Key], AvgPerLayerPackedBytes = roleLayerCounts[kv.Key] > 0 ? Round((double)kv.Value / roleLayerCounts[kv.Key]) : 0.0 })],
            PerLayerStackedPackedBytes = perLayer, // one row per contributing layer: what an un-routed full sweep of this layer's MoE FFN would read across gate/up/down stacks — before any low-rank idea applies
        };
    }

    private static List<LowRankEstimate> ComputeLowRankEstimates(BaselineAnalysis baseline, int hidden, int inter)
    {
        // Role geometry is fixed by the model metadata (gate/up: expertInter x hidden; down: hidden x expertInter), so these estimates cover every layer the table carries — sampled matrices only proved that geometry exists in practice.
        List<(string Role, int Rows, int Cols, int Layers)> roleGeometry = [.. baseline.PerRoleTotalsPackedBytes.Select(row => (row.Role, row.Role == "down" ? hidden : inter, row.Role == "down" ? inter : hidden, row.LayersWithRole))];

        List<LowRankEstimate> estimates = [];
        foreach (int rank in StorageCandidateRanks)
        {
            long lowRankBytesPerToken = 0; // summed over every layer and role: used_count active experts x fp32 factor elements per expert
            bool anyGeometry = false;
            foreach ((string _, int rows, int cols, int layers) in roleGeometry)
            {
                if (layers == 0 || baseline.CurrentActivePackedBytesPerToken <= 0)
                {
                    continue;
                }

                anyGeometry = true;
                long fullElements = (long)rows * cols;
                long factorElements = Math.Min(fullElements, ((long)rows + cols) * rank); // conservative count of coefficients plus bases at this rank: min(m*n, m*r + n*r) — capped at the matrix's own size
                lowRankBytesPerToken += layers * baseline.UsedExpertCount * factorElements * sizeof(float); // fp32 sidecar assumption: full-precision factors per active expert
            }

            if (!anyGeometry || lowRankBytesPerToken == 0)
            {
                continue;
            }

            estimates.Add(new LowRankEstimate
            {
                Rank = rank,
                EstimatedActiveBytesPerToken = lowRankBytesPerToken, // fp32 sidecar assumption: coefficients and bases stored at full precision per active expert
                ReductionVsCurrentQuantizedPercent = Round(100.0 * (1.0 - lowRankBytesPerToken / (double)baseline.CurrentActivePackedBytesPerToken)), // positive means the fp32 factorization beats today's packed quantized storage on bytes/token; negative says it does not yet
                ReductionVsFullFp32ReferencePercent = baseline.FullFp32ReferenceBytesPerToken > 0 ? Round(100.0 * (1.0 - lowRankBytesPerToken / (double)baseline.FullFp32ReferenceBytesPerToken)) : double.NaN, // same weighting against dequantized fp32 storage — the factorization comparison's second axis
            });
        }

        return estimates;
    }

    private static double Energy(MatrixResult result, int rank) => result.EnergyAtRanks.TryGetValue(rank, out double value) ? value : 1.0; // a ladder rank past the spectrum means every direction is already counted: full energy

    private static string LadderCsv(RoleAggregate aggregate) => string.Join(", ", EnergyLadderRanks.Select(r => (aggregate.MedianEnergyLadder.TryGetValue(r, out double value) ? value : 1.0).ToString("F5", CultureInfo.InvariantCulture))); // chart series values for one role's median cumulative-energy ladder

    private static string RankCsv(RoleAggregate aggregate) => string.Join(", ", RankKeyOrder.Select(t => aggregate.RankThresholds[t].ToString(CultureInfo.InvariantCulture))); // chart series values for one role's median rank requirements

    // ---------------------------------------------------------------- python spike script driving

    private static string? ResolveSpikeScript()
    {
        List<string> candidates = [Path.Combine(AppContext.BaseDirectory, "spikes", "lowrank-svd.py")];

        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        for (int hop = 0; hop < 8 && directory?.Parent is not null; hop++)
        {
            directory = directory.Parent;
            candidates.Add(Path.Combine(directory.FullName, "bench", "Lair.Bench", "spikes", "lowrank-svd.py")); // running from bin output still finds the script beside its source tree
        }

        foreach (string candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                return Path.GetFullPath(candidate);
            }
        }

        return null;
    }

    private static string? RunPythonProbe()
    {
        ProcessStartInfo start = new("python") { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        start.ArgumentList.Add("-c");
        start.ArgumentList.Add("import numpy, sys; print('numpy ' + numpy.__version__ + ' python ' + sys.version.split()[0])");

        using Process? process = StartPython(start);
        if (process is null)
        {
            return null; // no `python` on PATH: reported as a clear failure rather than approximating spectra in managed code
        }

        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
        {
            Console.WriteLine($"lowrank-svd python probe failed: {(string.IsNullOrWhiteSpace(stderr) ? $"exit code {process.ExitCode}" : stderr.Trim())}");
            return null;
        }

        return stdout.Trim();
    }

    private static int RunPythonScript(string scriptPath, string indexPath, string spectraOut)
    {
        ProcessStartInfo start = new("python") { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        start.ArgumentList.Add(scriptPath);
        start.ArgumentList.Add(indexPath);
        start.ArgumentList.Add(spectraOut);

        using Process? process = StartPython(start);
        if (process is null)
        {
            return -1; // python vanished between probe and run: the caller reports it as a failure with no approximation substituted
        }

        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (!string.IsNullOrWhiteSpace(stdout))
        {
            Console.WriteLine($"lowrank-svd script: {stdout.Trim()}");
        }

        if (process.ExitCode != 0 && !string.IsNullOrWhiteSpace(stderr))
        {
            Console.WriteLine($"lowrank-svd script stderr: {stderr.Trim()}");
        }

        return process.ExitCode;
    }

    private static Process? StartPython(ProcessStartInfo start)
    {
        try
        {
            return Process.Start(start);
        }
        catch (Exception)
        {
            return null; // no `python` on PATH: the caller reports it as a clear failure rather than approximating spectra in managed code
        }
    }

    // ---------------------------------------------------------------- report shape + rendering

    private static HostReport CaptureHost(CpuBackend backend)
    {
        List<string> matched = [];
        if (System.Runtime.Intrinsics.X86.Avx.IsSupported) matched.Add("avx");
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported) matched.Add("avx2");
        if (System.Runtime.Intrinsics.X86.Fma.IsSupported) matched.Add("fma");

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
                Lane = "CPU-only analytical spike",
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

    private static string BuildHtml(ManifestReport manifest, HostReport host, BenchReport bench)
    {
        SampleConfig sample = bench.SampleConfig;

        List<string> matrixRows = [.. bench.Matrices.Select(m =>
            $"<tr><td>{Esc(m.Id)}</td><td>{m.Layer}/{m.Expert}/{m.Role}</td><td>{Esc(m.Format)}</td><td>{m.Rows}x{m.Cols}</td>" +
            $"<td>{string.Join(", ", m.TopSingularValues.Take(3).Select(v => v.ToString("G6", CultureInfo.InvariantCulture)))}</td>" +
            $"<td>@8: {Energy(m, 8):F4} · @128: {Energy(m, 128):F4}</td><td>90%→{m.RankThresholds[90]} · 95%→{m.RankThresholds[95]}</td></tr>")];

        List<string> aggregateRows = [.. bench.RoleAggregates.Select(a =>
            $"<tr><td>{Esc(a.Role)}</td><td>{a.MatrixCount}</td>" + string.Join(string.Empty, RankKeyOrder.Select(t => $"<td>{a.RankThresholds[t]}</td>")) + "</tr>")];

        List<string> ladderRows = [.. bench.RoleAggregates.SelectMany(a => EnergyLadderRanks.Where(r => a.MedianEnergyLadder.ContainsKey(r)).Select(r =>
            $"<tr><td>{Esc(a.Role)}</td><td>{r}</td><td>{a.MedianEnergyLadder[r]:F5}</td></tr>"))];

        List<string> baselineRoleRows = [.. bench.Baseline.PerRoleTotalsPackedBytes.Select(kv =>
            $"<tr><td>{Esc(kv.Role)}</td><td>{kv.TotalStackedPackedBytes:N0}</td><td>{kv.LayersWithRole}</td><td>{kv.AvgPerLayerPackedBytes:F0}</td></tr>")];

        List<string> estimateRows = [.. bench.LowRankEstimates.Select(e =>
            $"<tr><td>r={e.Rank}</td><td>{e.EstimatedActiveBytesPerToken:N0}</td>" +
            $"<td class=\"{(e.ReductionVsCurrentQuantizedPercent > 0.0 ? "ok" : "bad")}\">{e.ReductionVsCurrentQuantizedPercent:F2}%</td>" +
            $"<td>{(double.IsNaN(e.ReductionVsFullFp32ReferencePercent) ? "n/a" : $"{e.ReductionVsFullFp32ReferencePercent:F2}%")}</td></tr>")];

        List<string> chartDatasets = [.. bench.RoleAggregates.Where(a => a.MedianEnergyLadder.Count > 0).Select(a =>
            $"{{ label: '{Esc(a.Role)} median cumulative energy', data: [{LadderCsv(a)}] }}")];

        List<string> rankChartDatasets = [.. bench.RoleAggregates.Where(a => a.MatrixCount > 0).Select(a =>
            $"{{ label: '{Esc(a.Role)} median rank needed', data: [{RankCsv(a)}] }}")];


        string notesLi = string.Join("\n", manifest.ComparisonNotes.Select(n => $"<li>{Esc(n)}</li>"));
        bool kneeFound = bench.Viability.Recommendation == "possible_low_rank";

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
canvas { max-width: 720px; margin-top: 1rem; } .note { color: #555; font-size: .88rem; }
</style>
</head>
<body>
<h1>{{Esc(manifest.ReportId)}}</h1>
<p class="note">{{Esc(manifest.Lane)}} · {{Esc(manifest.GeneratedAtUtc)}} · branch {{Esc(manifest.Branch)}}</p>

<h2>Sample config</h2>
<table>
<tr><th>Model</th><td>{{Esc(sample.ModelPath)}}</td></tr>
<tr><th>Arch / blocks / experts (used)</th><td>{{Esc(sample.Architecture)}} · {{sample.BlockCount}} · {{sample.ExpertCount}} ({{sample.UsedExpertCount}} routed/token)</td></tr>
<tr><th>Hidden / expert feed-forward</th><td>{{sample.HiddenSize}} · {{sample.ExpertFeedForwardLength}}</td></tr>
<tr><th>Sampling</th><td>layers [{{string.Join(", ", sample.Layers)}}] experts [{{string.Join(", ", sample.Experts)}}] roles [{{string.Join(", ", sample.Roles)}}]</td></tr>
<tr><th>Dequant route · spectrum tool</th><td>{{Esc(bench.DequantRoute)}} · {{Esc(bench.SpectrumTool)}}</td></tr>
<tr><th>Spectra cross-check (sum σ² vs ‖A‖F²)</th><td class="{{(bench.SpectraCrossCheckPassed ? "ok" : "bad")}}">{{(bench.SpectraCrossCheckPassed ? "passed for every matrix" : "a matrix failed the cross-check")}}</td></tr>
</table>

<h2>Viability recommendation — analytical evidence, not a kernel benchmark</h2>
<p><span class="{{(kneeFound ? "ok" : "bad")}}" style="font-size:1.05rem">{{Esc(bench.Viability.Recommendation)}}</span></p>
<p class="note">{{Esc(bench.Viability.CriteriaMet)}} · down matrices sampled {{bench.Viability.DownMatricesSampled}} · gate/up sampled {{bench.Viability.GateUpMatricesSampled}}</p>

<h2>Per-matrix spectra summary (full descending spectra live in bench.json)</h2>
<table>
<tr><th>Matrix</th><th>L/E/role</th><th>Format</th><th>Dims</th><th>Top σ</th><th>Cumulative energy @8 / @128</th><th>Rank to reach 90% · 95%</th></tr>
{{string.Join("\n", matrixRows)}}
</table>

<h2>Median rank needed per role (sampled matrices)</h2>
<table>
<tr><th>Role</th><th>Matrices</th>{{string.Join(string.Empty, RankKeyOrder.Select(t => $"<th>{t}% energy</th>"))}}</tr>
{{string.Join("\n", aggregateRows)}}
</table>

<h2>Median cumulative-energy ladder per role</h2>
<table>
<tr><th>Role</th><th>Rank</th><th>Cumulative energy fraction</th></tr>
{{string.Join("\n", ladderRows)}}
</table>

<h2>Active routed-expert bytes/token baseline (assembled tensor table)</h2>
<p class="note">Total stacked expert packed bytes {{bench.Baseline.StackedExpertTotalBytes:N0}} · active ratio used/experts = {{bench.Baseline.UsedExpertCount}}/{{bench.Baseline.ExpertCount}} ≈ {{bench.Baseline.ActiveRatio:F6}} · current active packed bytes/token <b>{{bench.Baseline.CurrentActivePackedBytesPerToken:N0}}</b> · dequantized-fp32 reference bytes/token {{bench.Baseline.FullFp32ReferenceBytesPerToken:N0}}</p>
<table>
<tr><th>Role</th><th>Total stacked packed bytes</th><th>Layers carrying role</th><th>Avg per layer</th></tr>
{{string.Join("\n", baselineRoleRows)}}
</table>

<h2>Conservative fp32 low-rank storage estimates (min(m·n, m·r + n·r) elements)</h2>
<table>
<tr><th>Candidate rank</th><th>Estimated active bytes/token (fp32 sidecar)</th><th>Reduction vs current quantized packed baseline</th><th>Reduction vs dequantized-fp32 reference</th></tr>
{{string.Join("\n", estimateRows)}}
</table>

<h2>Cumulative energy — role median spectra (element-wise medians over sampled matrices)</h2>
<canvas id="energy" width="720" height="360"></canvas>
<script>
new Chart(document.getElementById('energy'), {
  type: 'line',
  data: { labels: [{{string.Join(", ", EnergyLadderRanks.Select(r => $"'r{r}'"))}}], datasets: [{{string.Join(", ", chartDatasets)}}] },
  options: { plugins: { title: { display: true, text: 'median cumulative energy fraction vs rank' } }, scales: { y: { min: 0.5, max: 1.0, title: { display: true, text: 'fraction of total matrix energy' } } } }
});
</script>

<h2>Rank needed for each energy threshold — role medians</h2>
<canvas id="ranks" width="720" height="360"></canvas>
<script>
new Chart(document.getElementById('ranks'), {
  type: 'bar',
  data: { labels: [{{string.Join(", ", RankKeyOrder.Select(t => $"'{t}%'"))}}], datasets: [{{string.Join(", ", rankChartDatasets)}}] },
  options: { scales: { y: { beginAtZero: true, title: { display: true, text: 'median leading-rank count' } } } }
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

    // ---------------------------------------------------------------- small helpers

    private static int LayerOf(string name)
    {
        int start = name.IndexOf('.') + 1;
        int end = name.IndexOf('.', start);
        return int.Parse(name[start..end], CultureInfo.InvariantCulture);
    }

    private static string RoleOf(string name) => name.Contains(".ffn_gate_", StringComparison.Ordinal) ? "gate" : name.Contains(".ffn_up_", StringComparison.Ordinal) ? "up" : "down";

    private static string MatrixSlug(StackGeometry geometry, int expert) => $"{LayerOf(geometry.Name)}_{RoleOf(geometry.Name)}_{expert}";

    private static List<int> ParseIntList(string raw) => [.. raw.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(part => int.Parse(part, CultureInfo.InvariantCulture))];

    private static double MedianOf(double[] values)
    {
        if (values.Length == 0)
        {
            return double.NaN;
        }

        Array.Sort(values);
        int mid = values.Length / 2;
        return values.Length % 2 == 1 ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
    }

    private static int MedianOf(int[] values) => (int)Math.Round(MedianOf([.. values.Select(v => (double)v)]));

    private static double Round(double value) => Math.Round(value, 6);

    private static void WriteJson(string path, object value) =>
        File.WriteAllText(path, System.Text.Json.JsonSerializer.Serialize(value, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));

    // ---------------------------------------------------------------- run-local geometry state + JSON DTOs (python exchange and published report)

    private sealed record StackGeometry(string Name, QuantFormat Format, int Rows, int Cols, long ExpertBytes, long StrideBytes);

    private sealed record MatrixExport(
        string Id, string TensorName, int Layer, int Expert, string Role, string Format,
        int Rows, int Cols, long StrideBytes, long PackedExpertBytes, string FileName);

    private sealed class PythonIndex
    {
        [JsonPropertyName("matrices")] public List<PythonMatrixEntry> Matrices { get; init; } = [];
    }

    private sealed class PythonMatrixEntry
    {
        [JsonPropertyName("id")] public string Id { get; init; } = "";
        [JsonPropertyName("file")] public string File { get; init; } = "";
        [JsonPropertyName("rows")] public int Rows { get; init; }
        [JsonPropertyName("cols")] public int Cols { get; init; }
    }

    private sealed class PythonSpectra
    {
        [JsonPropertyName("schemaVersion")] public int SchemaVersion { get; init; }
        [JsonPropertyName("matrices")] public List<PythonMatrixSpectrum> Matrices { get; init; } = [];
    }

    private sealed class PythonMatrixSpectrum
    {
        [JsonPropertyName("id")] public string Id { get; init; } = "";
        [JsonPropertyName("rows")] public int Rows { get; init; }
        [JsonPropertyName("cols")] public int Cols { get; init; }
        [JsonPropertyName("singularValues")] public double[] SingularValues { get; init; } = [];
        [JsonPropertyName("frobeniusNormSquared")] public double FrobeniusNormSquared { get; init; }
    }

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
        [JsonPropertyName("lane")] public string Lane { get; init; } = "";
    }

    private sealed class BenchReport
    {
        [JsonPropertyName("schemaVersion")] public int SchemaVersion { get; init; }
        [JsonPropertyName("reportId")] public string ReportId { get; init; } = "";
        [JsonPropertyName("sampleConfig")] public SampleConfig SampleConfig { get; init; } = new();
        [JsonPropertyName("peakMatrixBytes")] public long PeakMatrixBytes { get; init; } // one dequantized expert matrix resident at a time — the spike never holds more than this above its mmap views
        [JsonPropertyName("dequantRoute")] public string DequantRoute { get; init; } = "";
        [JsonPropertyName("spectrumTool")] public string SpectrumTool { get; init; } = "";
        [JsonPropertyName("spectraCrossCheckPassed")] public bool SpectraCrossCheckPassed { get; init; } // sum(sigma^2) matched the independently measured ||A||_F^2 for every matrix: spectra and exported bytes agree
        [JsonPropertyName("matrices")] public List<MatrixResult> Matrices { get; init; } = [];
        [JsonPropertyName("roleAggregates")] public List<RoleAggregate> RoleAggregates { get; init; } = []; // per-role median spectra/rank requirements over the sampled matrices — medians across matrices, never modes of one spectrum
        [JsonPropertyName("baselineBytesPerToken")] public BaselineAnalysis Baseline { get; init; } = new();
        [JsonPropertyName("lowRankEstimates")] public List<LowRankEstimate> LowRankEstimates { get; init; } = []; // conservative fp32-factorization storage estimates per candidate rank, compared against both the packed quantized baseline and a dequantized-fp32 reference
        [JsonPropertyName("viability")] public ViabilityResult Viability { get; init; } = new(); // analytical evidence only: no kernel throughput is measured or implied here
    }

    private sealed class SampleConfig
    {
        [JsonPropertyName("modelPath")] public string ModelPath { get; init; } = "";
        [JsonPropertyName("architecture")] public string Architecture { get; init; } = "";
        [JsonPropertyName("blockCount")] public int BlockCount { get; init; }
        [JsonPropertyName("expertCount")] public int ExpertCount { get; init; }
        [JsonPropertyName("usedExpertCount")] public int UsedExpertCount { get; init; }
        [JsonPropertyName("hiddenSize")] public int HiddenSize { get; init; }
        [JsonPropertyName("expertFeedForwardLength")] public int ExpertFeedForwardLength { get; init; }
        [JsonPropertyName("layers")] public List<int> Layers { get; init; } = [];
        [JsonPropertyName("experts")] public List<int> Experts { get; init; } = [];
        [JsonPropertyName("roles")] public List<string> Roles { get; init; } = [];
    }

    private sealed class MatrixResult
    {
        [JsonPropertyName("id")] public string Id { get; init; } = "";
        [JsonPropertyName("tensorName")] public string TensorName { get; init; } = "";
        [JsonPropertyName("layer")] public int Layer { get; init; }
        [JsonPropertyName("expert")] public int Expert { get; init; }
        [JsonPropertyName("role")] public string Role { get; init; } = "";
        [JsonPropertyName("format")] public string Format { get; init; } = "";
        [JsonPropertyName("rows")] public int Rows { get; init; }
        [JsonPropertyName("cols")] public int Cols { get; init; }
        [JsonPropertyName("packedExpertBytes")] public long PackedExpertBytes { get; init; } // this expert's slab as stored: the contiguous byte range a gather reads cold today
        [JsonPropertyName("fp32MatrixBytes")] public long Fp32MatrixBytes { get; init; } // rows x cols in float32 — what one dequantized expert costs resident, and the full-size reference for factorization caps
        [JsonPropertyName("frobeniusNormSquared")] public double FrobeniusNormSquared { get; init; }
        [JsonPropertyName("spectraFrobeniusCrossCheckRelDiff")] public double SpectraFrobeniusCrossCheckRelDiff { get; init; } // relative gap between sum(sigma^2) and the independently measured ||A||_F^2: spectra and matrix agree when it stays tiny
        [JsonPropertyName("topSingularValues")] public List<double> TopSingularValues { get; init; } = []; // full descending spectrum: min(rows, cols) values — every published ladder rank (max 256) sits inside it here
        [JsonPropertyName("energyAtRanks")] public Dictionary<int, double> EnergyAtRanks { get; init; } = []; // cumulative energy fraction at each ladder rank the matrix's spectrum reaches (8/16/32/64/96/128/192/256)
        [JsonPropertyName("rankThresholds")] public Dictionary<int, int> RankThresholds { get; init; } = []; // leading-rank count needed per percent threshold (50/75/90/95/99); min(m,n) when none is reached before exhaustion — near-full thresholds are the no-knee signature
    }

    private sealed class RoleAggregate
    {
        [JsonPropertyName("role")] public string Role { get; init; } = "";
        [JsonPropertyName("matrixCount")] public int MatrixCount { get; init; }
        [JsonPropertyName("medianSpectrumTop256")] public List<double> MedianSpectrumTop256 { get; init; } = []; // element-wise medians across the role's sampled matrices: index i is each matrix's (i+1)-th singular value, never a cross-matrix mode of one spectrum
        [JsonPropertyName("medianEnergyLadder")] public Dictionary<int, double> MedianEnergyLadder { get; init; } = []; // element-wise median cumulative-energy fraction per ladder rank across the role's sampled matrices
        [JsonPropertyName("rankThresholds")] public Dictionary<int, int> RankThresholds { get; init; } = []; // per-role medians of the sampled matrices' rank-to-threshold values — these are what the viability rule reads alongside the raw per-matrix table
    }

    private sealed class BaselineAnalysis
    {
        [JsonPropertyName("expertCount")] public int ExpertCount { get; init; }
        [JsonPropertyName("usedExpertCount")] public int UsedExpertCount { get; init; }
        [JsonPropertyName("activeRatio")] public double ActiveRatio { get; init; } // the router's active share: each token touches used_count experts out of expert_count in every sparse layer
        [JsonPropertyName("stackedExpertTotalBytes")] public long StackedExpertTotalBytes { get; init; } // every layer's gate/up/down stacked packed bytes summed across the assembled table — what a full expert sweep reads per token if no routing happened
        [JsonPropertyName("currentActivePackedBytesPerToken")] public long CurrentActivePackedBytesPerToken { get; init; } // exact baseline: that total scaled by used/expert_count, the router's active share of every layer step
        [JsonPropertyName("fullFp32ReferenceBytesPerToken")] public long FullFp32ReferenceBytesPerToken { get; init; } // same weighting against dequantized fp32 storage — the factorization comparison's second axis
        [JsonPropertyName("perRoleTotalsPackedBytes")] public List<RoleTotalRow> PerRoleTotalsPackedBytes { get; init; } = [];
        [JsonPropertyName("perLayerStackedPackedBytes")] public List<LayerRoleBytes> PerLayerStackedPackedBytes { get; init; } = []; // one row per contributing layer: what an un-routed full sweep of this layer's MoE FFN would read across gate/up/down stacks
    }

    private sealed class RoleTotalRow
    {
        [JsonPropertyName("role")] public string Role { get; init; } = "";
        [JsonPropertyName("totalStackedPackedBytes")] public long TotalStackedPackedBytes { get; init; } // summed stacked payload bytes across every layer carrying this role — the full-sweep cost for one role's expert stacks
        [JsonPropertyName("layersWithRole")] public int LayersWithRole { get; init; } // how many layers' tables actually carried a well-formed stacked tensor of this role: short counts report missing or malformed entries instead of guessed bytes
        [JsonPropertyName("avgPerLayerPackedBytes")] public double AvgPerLayerPackedBytes { get; init; }
    }

    private sealed class LayerRoleBytes
    {
        [JsonPropertyName("layer")] public int Layer { get; init; }
        [JsonPropertyName("stackedPackedBytes")] public long StackedPackedBytes { get; init; } // what an un-routed full sweep of this layer's MoE FFN would read across gate/up/down stacks — before any low-rank idea applies
    }

    private sealed class LowRankEstimate
    {
        [JsonPropertyName("rank")] public int Rank { get; init; }
        [JsonPropertyName("estimatedActiveBytesPerToken")] public long EstimatedActiveBytesPerToken { get; init; } // fp32 sidecar assumption: coefficients and bases stored at full precision per active expert, summed over every layer the table carries
        [JsonPropertyName("reductionVsCurrentQuantizedPercent")] public double ReductionVsCurrentQuantizedPercent { get; init; } // positive means the fp32 factorization beats today's packed quantized storage on bytes/token; negative says it does not yet — an honest viability axis, since this checkpoint already stores experts sub-4-bit
        [JsonPropertyName("reductionVsFullFp32ReferencePercent")] public double ReductionVsFullFp32ReferencePercent { get; init; } // same weighting against dequantized fp32 storage: a factorization only earns its complexity when it beats the quantized baseline, so this axis is context, not the decision
    }

    private sealed class ViabilityResult
    {
        [JsonPropertyName("recommendation")] public string Recommendation { get; init; } = ""; // no_knee | possible_low_rank | inconclusive — analytical evidence only: no kernel throughput is measured or implied here
        [JsonPropertyName("criteriaMet")] public string CriteriaMet { get; init; } = "";
        [JsonPropertyName("downMatricesSampled")] public int DownMatricesSampled { get; init; } // the decision reads the down role's rank requirements and needs at least 3 sampled matrices to be conclusive — a strong knee means median 90%-energy rank <=64 AND median 95% <=128 across those samples
        [JsonPropertyName("gateUpMatricesSampled")] public int GateUpMatricesSampled { get; init; } // gate/up inform the read but never decide it: their medians ride in bench.json for context, since output/down collapse is what a low-rank sidecar would first have to survive
    }

}
