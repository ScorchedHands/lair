using System.Diagnostics;
using System.Globalization;

namespace Lair.Bench;

/// <summary>
/// Kernel-level attribution seam for the Q4_K wide-batch dense dot (internal measurement mode, bench-only):
/// times `CpuBackend.GemvBatch` on synthetic pinned Q4_K tensors at realistic Qwen2.5-3B feed-forward widths
/// (`11008x2048` gate/up and `2048x11008` down) across batch counts 1/8/16/32/64, once per route: the auto posture's
/// scratch batch-dot traversal (chain-group register chains over one materialized row), a forced streaming fallback
/// (`MaxScratchCols` pinned to its floor so every count keeps `BatchQ4KAvx2`'s stack-accumulator chain walk), an
/// optional column-block seam arm at a stated block width, and an optional narrow diagnostic arm that pins the chain
/// group to 1 (one accumulator re-walking the staged hot row per batch vector). Every arm computes bit-identical
/// outputs for the same fixture, verified in-run; only their traversal differs. Arms are measured like the committed
/// Xeon runners measure arms: one untimed parity pass per arm, two untimed settle passes (forward then reversed) so
/// tier-up and page/cache settling land on every arm before timing, then `--reps` timed rounds whose arm order is
/// rotated round by round — an arm's number never benefits from a warmer machine state than its neighbours saw. Each
/// line publishes true median/min/max wall ms plus a per-arm dispersion pair (sample stddev_ms and the max-min band as a
/// fraction of that median) across its own rotated rounds, ns/MAC and effective weight GB/s so amortization collapse across
/// counts is visible as a per-count efficiency curve at fixed thread posture, and so a schedule change can be read against
/// the run-to-run jitter it claims to remove. An optional dispatch-schedule arm states the unpromoted static contiguous row
/// partition crossover: identical kernels over the same fixture with each participant walking one contiguous range of canonical
    /// rows instead of claiming scheduling blocks dynamically, joined to the float bit-parity guard because only who walks a row
    /// changes. An optional budget-gated repacked Q4_K residency arm prepares an AVX2-only backend's own resident copy under a stated cap and reads it back through the explicit seam; its divergence is classified as activation-quantization loss like integer arms, ineligible counts are skipped rather than refused mid-measurement, and over-cap preparations print refusal lines instead of timing anything. No production route or config posture changes.

/// </summary>
internal static class Q4KWideBatchBench
{
    public static int Run(string[] args)
    {
        if (args.Length > 1 && (args[1] == "help" || args[1] == "--help"))
        {
            Usage();
            return 0;
        }

        int threads = 1; // single-threaded by default: this seam attributes per-kernel efficiency, not host scaling
        int reps = 3;
        bool diagScratchG1 = false;
        bool integerDotArm = false;
        List<int> integerChainGroups = []; // each stated group width adds its own integer arm beside the default one, so candidate widths share the same settle and rotation schedule as every other arm here.
        List<int> staticPartitionWidths = []; // each stated crossover adds a dispatch-schedule arm: identical kernels on contiguous per-participant row ranges instead of dynamic block claiming.
        List<int> counts = [1, 8, 16, 32, 64];
        List<(int Rows, int Cols)> shapes = [(11_008, 2_048), (2_048, 11_008)]; // gate/up and down widths of Qwen2.5-3B hidden=2048, intermediate=11008
        List<int> residencyWidths = []; // each stated crossover adds a budget-gated resident-copy arm: an AVX2-only backend prepared under its own cap, reading that copy back through the explicit seam at or above this count.
        long repackedCapMegabytes = 0; // 0 means exact-fit rounded up per shape; a stated value either admits every residency arm or prints their refusal lines and leaves them out of timing.

        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--threads":
                    if (!TryInt(args, ++i, out threads) || threads < 1)
                    {
                        Console.WriteLine("q4k-widebatch: --threads must be at least 1");
                        return 1;
                    }

                    break;
                case "--reps":
                    if (!TryInt(args, ++i, out reps) || reps < 3)
                    {
                        Console.WriteLine("q4k-widebatch: --reps must be at least 3 (one forward and one reversed rotation round plus a middle to take the median from)");
                        return 1;
                    }

                    break;
                case "--diag-scratch-g1":
                    diagScratchG1 = true; // adds the chain-group-1 scratch arm (stack-chain re-walk per batch vector) as a diagnostic third route
                    break;
                case "--integer-dot":
                    integerDotArm = true; // adds the unpromoted integer-domain Q4_K dot arm: stated width 8 on an AVX2-only posture, measured beside the float routes with its divergence classified rather than asserted identical
                    break;
                case "--integer-chain-group":
                    if (i + 1 >= args.Length)
                    {
                        Console.WriteLine("q4k-widebatch: option '--integer-chain-group' needs a value");
                        return 1;
                    }

                    List<int> parsedGroups = [];
                    foreach (string part in args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                    {
                        if (!int.TryParse(part, out int group) || group < 1 || group > Lair.Backends.Cpu.CpuKernelTuning.ChainGroupMax)
                        {
                            Console.WriteLine($"q4k-widebatch: invalid chain group '{part}' (expected 1..{Lair.Backends.Cpu.CpuKernelTuning.ChainGroupMax})");
                            return 1;
                        }

                        parsedGroups.Add(group);
                    }

                    if (parsedGroups.Count == 0)
                    {
                        Console.WriteLine("q4k-widebatch: --integer-chain-group listed no widths");
                        return 1;
                    }

                    integerChainGroups = [.. parsedGroups.Distinct().OrderBy(g => g)]; // each stated width becomes its own arm beside the default one, sharing this run's settle and rotation schedule.
                    break;
                case "--static-partition":
                    if (i + 1 >= args.Length)
                    {
                        Console.WriteLine("q4k-widebatch: option '--static-partition' needs a value");
                        return 1;
                    }

                    List<int> parsedWidths = [];
                    foreach (string part in args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                    {
                        if (!int.TryParse(part, out int width) || width < 1 || width > Lair.Backends.Cpu.CpuKernelTuningOverrides.StaticRowPartitionMax)
                        {
                            Console.WriteLine($"q4k-widebatch: invalid static-partition width '{part}' (expected 1..{Lair.Backends.Cpu.CpuKernelTuningOverrides.StaticRowPartitionMax})");
                            return 1;
                        }

                        parsedWidths.Add(width);
                    }

                    if (parsedWidths.Count == 0)
                    {
                        Console.WriteLine("q4k-widebatch: --static-partition listed no widths");
                        return 1;
                    }

                    staticPartitionWidths = [.. parsedWidths.Distinct().OrderBy(w => w)]; // each stated crossover adds its own schedule arm beside the auto arm, sharing this run's settle and rotation schedule.
                    break;
                case "--shapes":
                    if (i + 1 >= args.Length)
                    {
                        Console.WriteLine("q4k-widebatch: option '--shapes' needs a value");
                        return 1;
                    }

                    List<(int Rows, int Cols)> parsedShapes = [];
                    foreach (string part in args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                    {
                        string[] halves = part.Split('x', 2);
                        if (halves.Length != 2 || !int.TryParse(halves[0], out int r) || !int.TryParse(halves[1], out int c) || r < 1 || c < 1)
                        {
                            Console.WriteLine($"q4k-widebatch: invalid shape '{part}' (expected rowsxcols)");
                            return 1;
                        }

                        parsedShapes.Add((r, c));
                    }

                    if (parsedShapes.Count == 0)
                    {
                        Console.WriteLine("q4k-widebatch: --shapes listed no shapes");
                        return 1;
                    }

                    shapes = parsedShapes;
                    break;
                case "--counts":
                    if (i + 1 >= args.Length)
                    {
                        Console.WriteLine("q4k-widebatch: option '--counts' needs a value");
                        return 1;
                    }

                    List<int> parsedCounts = [];
                    foreach (string part in args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                    {
                        if (!int.TryParse(part, out int count) || count < 1 || count > 64)
                        {
                            Console.WriteLine($"q4k-widebatch: invalid count '{part}' (expected 1..64)");
                            return 1;
                        }

                        parsedCounts.Add(count);
                    }

                    if (parsedCounts.Count == 0)
                    {
                        Console.WriteLine("q4k-widebatch: --counts listed no counts");
                        return 1;
                    }

                    counts = [.. parsedCounts.Distinct().OrderBy(c => c)];
                    break;
                case "--repacked-residency":
                    if (i + 1 >= args.Length)
                    {
                        Console.WriteLine("q4k-widebatch: option '--repacked-residency' needs a value");
                        return 1;
                    }

                    List<int> parsedResidencyWidths = [];
                    foreach (string part in args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                    {
                        if (!int.TryParse(part, out int width) || width < 1 || width > Lair.Backends.Cpu.CpuKernelTuning.MatmulRouteBatchWidthMax)
                        {
                            Console.WriteLine($"q4k-widebatch: invalid repacked-residency width '{part}' (expected 1..{Lair.Backends.Cpu.CpuKernelTuning.MatmulRouteBatchWidthMax})");
                            return 1;
                        }

                        parsedResidencyWidths.Add(width);
                    }

                    if (parsedResidencyWidths.Count == 0)
                    {
                        Console.WriteLine("q4k-widebatch: --repacked-residency listed no widths");
                        return 1;
                    }

                    residencyWidths = [.. parsedResidencyWidths.Distinct().OrderBy(w => w)]; // each stated crossover adds its own resident-copy arm beside the float/integer arms.
                    break;
                case "--repacked-cap-mb":
                    if (i + 1 >= args.Length)
                    {
                        Console.WriteLine("q4k-widebatch: option '--repacked-cap-mb' needs a value");
                        return 1;
                    }

                    string capArg = args[++i];
                    if (!TryMegabytes(capArg, out repackedCapMegabytes))
                    {
                        Console.WriteLine($"q4k-widebatch: invalid --repacked-cap-mb '{capArg}' (expected auto or a non-negative megabyte count)");
                        return 1;
                    }

                    break;
                default:
                    Usage();
                    return 1;
            }
        }

        Console.WriteLine($"q4k-widebatch threads={threads} reps={reps} avx2={(System.Runtime.Intrinsics.X86.Avx.IsSupported ? "yes" : "no")} shapes={string.Join(",", shapes.Select(s => $"{s.Rows}x{s.Cols}"))} counts={string.Join(",", counts)}");

        foreach ((int rows, int cols) in shapes)
        {
            Lair.Contracts.TensorShape shape = new([rows, cols]);
            long strideBytes = shape.RowStrideBytes(Lair.Contracts.QuantFormat.Q4_K);
            byte[] weights = new byte[rows * strideBytes];
            PackSyntheticQ4K(weights, rows, cols);

            const int MaxCount = 64;
            float[] inputs = new float[MaxCount * cols];
            Random rng = new(20_261_127);
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            Lair.Contracts.WeightRef weight = new(weights, shape, Lair.Contracts.QuantFormat.Q4_K);

            using Lair.Backends.Cpu.CpuBackend autoRoute = new(threads: threads);
            using Lair.Backends.Cpu.CpuBackend streamingRoute = new(threads: threads, tuning: new Lair.Backends.Cpu.CpuKernelTuningOverrides { MaxScratchCols = Lair.Backends.Cpu.CpuKernelTuning.MaxScratchColsMin });
            // The unpromoted tile executor arm states the Q4_K crossover (8-row x 256-column tiles whose chains stay live across a column block): included as an attribution reference because it is today's other candidate traversal for this exact pass.
            using Lair.Backends.Cpu.CpuBackend tileRoute = new(threads: threads, tuning: new Lair.Backends.Cpu.CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 8 });
            using Lair.Backends.Cpu.CpuBackend? scratchG1 = diagScratchG1
                ? new Lair.Backends.Cpu.CpuBackend(threads: threads, tuning: new Lair.Backends.Cpu.CpuKernelTuningOverrides { ChainGroup = 1 })
                : null;

            // The unpromoted integer-domain arm states the Q4_K batch width (8) on an AVX2-only posture so its ladder is forced deterministically even on hosts whose auto selection lands on vpdpbusd. Its outputs are expected to diverge from the float arms by activation-quantization loss, which this harness classifies rather than asserting identical. A stated chain group adds another integer arm beside it — same width, narrower shared-input grouping — so candidate widths land in one run's settle and rotation schedule instead of separate runs.
            List<(string Name, Lair.Backends.Cpu.CpuBackend Backend)> integerArms = [];

            if (integerDotArm)
            {
                integerArms.Add(("integer-dot", new Lair.Backends.Cpu.CpuBackend(threads: threads, cpuFeatures: new Lair.Backends.Cpu.CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false), tuning: new Lair.Backends.Cpu.CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8 })));
            }

            foreach (int group in integerChainGroups)
            {
                integerArms.Add(($"integer-g{group}", new Lair.Backends.Cpu.CpuBackend(threads: threads, cpuFeatures: new Lair.Backends.Cpu.CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false), tuning: new Lair.Backends.Cpu.CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8, ChainGroup = group })));
            }

            // The unpromoted static contiguous row partition arms: identical kernels and traversals, only the dispatch schedule differs — each participant walks one contiguous range of canonical rows instead of claiming scheduling blocks dynamically. These are float-classified arms, so this run's bit-parity guard asserts they write exactly what the auto arm writes.
            List<(string Name, Lair.Backends.Cpu.CpuBackend Backend)> staticArms = [];

            foreach (int width in staticPartitionWidths)
            {
                staticArms.Add(($"static-w{width}", new Lair.Backends.Cpu.CpuBackend(threads: threads, tuning: new Lair.Backends.Cpu.CpuKernelTuningOverrides { StaticRowPartitionBatchWidth = width })));
            }

            // The budget-gated repacked Q4_K residency arms: each stated crossover opens an AVX2-only backend that prepared its own resident copy under the run's cap and reads it back through the explicit seam. An over-cap preparation prints a refusal line instead of timing anything — no narrower fallback, no silent narrowing, just what the gate said.
            List<(string Name, Lair.Backends.Cpu.CpuBackend Backend)> residencyArms = [];
            Dictionary<string, Lair.Backends.Cpu.Q4KRepackedMatrix> repackedCopies = new(StringComparer.Ordinal);
            string repackedArmFacts = "";
            List<string> repackedRefusalNotes = [];

            if (residencyWidths.Count > 0)
            {
                long autoCapMegabytes = (rows * Lair.Backends.Cpu.Q4KResidencyLayout.RepackedRowBytes(cols) + 1024L * 1024 - 1) / (1024L * 1024); // exact fit rounded up, so the default arm set measures admitted copies rather than gate refusals.
                long effectiveCapMegabytes = repackedCapMegabytes == 0 ? autoCapMegabytes : repackedCapMegabytes;

                foreach (int width in residencyWidths)
                {
                    string name = $"repacked-w{width}";
                    Lair.Backends.Cpu.CpuBackend backend = new(threads: threads, cpuFeatures: new Lair.Backends.Cpu.CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false), tuning: new Lair.Backends.Cpu.CpuKernelTuningOverrides { Q4KRepackedResidencyBatchWidth = width });

                    try
                    {
                        Lair.Backends.Cpu.Q4KRepackedMatrix matrix = backend.PrepareDenseQ4KResidency(weight, effectiveCapMegabytes);
                        repackedCopies.Add(name, matrix);
                        residencyArms.Add((name, backend)); // an admitted resident-copy arm is just another member of the same measured set.

                        repackedArmFacts += $" {name}_tuning=[q4k_repacked_residency_batch_width={backend.ResolvedTuning.Q4KRepackedResidencyBatchWidth},cap_mb={effectiveCapMegabytes},repacked_bytes={matrix.RepackedBytes},resident_backend_bytes={backend.PreparedQ4KResidencyBytes}]";
                    }
                    catch (Exception error) when (error is Lair.Backends.Cpu.Q4KRepackedResidencyException or OutOfMemoryException || error is OverflowException)
                    {
                        backend.Dispose(); // a refused arm has no copy to read back; its refusal line records the gate's own message instead.

                        repackedRefusalNotes.Add($" shape={rows}x{cols} arm={name}_refusal=[cap_mb={effectiveCapMegabytes},message={error.Message}]");
                    }
                }
            }

            // One arm list drives every phase below (parity, settle, rotated timing): arms are measured as a set, never one after another to completion.
            List<(string Name, Lair.Backends.Cpu.CpuBackend Backend)> arms = [("auto", autoRoute), ("streaming", streamingRoute), ("tile", tileRoute)];

            if (scratchG1 is not null)
            {
                arms.Add(("scratch-g1", scratchG1));
            }

            List<int> integerArmIndices = [];
            foreach ((string name, Lair.Backends.Cpu.CpuBackend backend) in integerArms)
            {
                integerArmIndices.Add(arms.Count);
                arms.Add((name, backend)); // an integer arm is just another member of the same measured set.
            }

            string staticArmFacts = "";
            foreach ((string name, Lair.Backends.Cpu.CpuBackend backend) in staticArms)
            {
                arms.Add((name, backend)); // a schedule arm is another member of the same measured set, with its own posture facts published beside it.
                staticArmFacts += $" {name}_tuning=[static_row_partition_batch_width={backend.ResolvedTuning.StaticRowPartitionBatchWidth},row_block={backend.ResolvedTuning.RowBlock},parallel_row_threshold={backend.ResolvedTuning.ParallelRowThreshold},workers={backend.EffectiveWorkerCount}]";
            }

            List<int> residencyArmIndices = []; // these arms are classified like integer arms: activation staging quantizes FP32 inputs, so divergence from the float routes is published rather than asserted identical.
            foreach ((string name, Lair.Backends.Cpu.CpuBackend backend) in residencyArms)
            {
                int armIndex = arms.Count;

                // The prepared copy lives in repackedCopies beside this backend: only the explicit seam overload reads that pairing back.
                residencyArmIndices.Add(armIndex);
                arms.Add((name, backend)); // a resident-copy arm joins the same settle and rotation schedule as every other measured arm here.
            }

            Console.WriteLine($"q4k-widebatch shape={rows}x{cols} stride_bytes={strideBytes} packed_mib={(weights.Length / (1024.0 * 1024.0)):F2} auto_tuning=[chain_group={autoRoute.ResolvedTuning.ChainGroup},max_scratch_cols={autoRoute.ResolvedTuning.MaxScratchCols}] streaming_tuning=[chain_group={streamingRoute.ResolvedTuning.ChainGroup},max_scratch_cols={streamingRoute.ResolvedTuning.MaxScratchCols}] tile_tuning=[matmul_route_batch_width_q4k={tileRoute.ResolvedTuning.MatmulRouteBatchWidthQ4K}]{staticArmFacts}{repackedArmFacts}");

            foreach (string note in repackedRefusalNotes)
            {
                Console.WriteLine($"q4k-widebatch{note}"); // a stated cap that does not admit the asked-for copy ends this arm set here — no timing line follows for it.
            }

            foreach (int count in counts)
            {
                // A resident-copy read is eligible only at or above its stated crossover; ineligible arms are skipped per count rather than refused mid-measurement, so the same arm set can still be measured across small and wide batch counts.
                List<int> activeArms = [];

                foreach (int armIndex in Enumerable.Range(0, arms.Count))
                {
                    if (residencyArmIndices.Contains(armIndex) && count < arms[armIndex].Backend.ResolvedTuning.Q4KRepackedResidencyBatchWidth)
                    {
                        continue; // below its stated crossover this seam refuses by contract; the run records that as a skip, not an error.
                    }

                    activeArms.Add(armIndex);
                }

                float[][] outs = new float[activeArms.Count][];
                Lair.Backends.Cpu.Q4KRepackedMatrix?[] armMatrices = new Lair.Backends.Cpu.Q4KRepackedMatrix?[activeArms.Count];

                for (int pos = 0; pos < activeArms.Count; pos++)
                {
                    int armIndex = activeArms[pos];

                    outs[pos] = new float[count * rows];
                    repackedCopies.TryGetValue(arms[armIndex].Name, out armMatrices[pos]); // ordinary arms carry no matrix and run their resolved traversal.

                    RunBatch(arms[armIndex].Backend, weight, inputs.AsSpan(0, count * cols), count, armMatrices[pos], outs[pos]); // untimed parity pass: every eligible arm must write its own contract for this fixture before any timing line below is read as a comparison.

                    if (float.IsNaN(outs[pos][0]))
                    {
                        Console.WriteLine($"q4k-widebatch shape={rows}x{cols} count={count} arm={arms[armIndex].Name}: WARNING NaN in first output value");
                    }
                }

                bool bitIdentical = true;
                for (int pos = 1; pos < activeArms.Count; pos++)
                {
                    int armIndex = activeArms[pos];

                    if (integerArmIndices.Contains(armIndex) || residencyArmIndices.Contains(armIndex))
                    {
                        continue; // integer and resident-copy arms quantize activations: divergence from the float arms is expected and classified below instead.
                    }

                    if (!BitsEqual(outs[0], outs[pos]))
                    {
                        bitIdentical = false;
                    }
                }

                List<double>[] samples = new List<double>[activeArms.Count];
                for (int pos = 0; pos < activeArms.Count; pos++)
                {
                    samples[pos] = [];
                }

                void RunArms(bool forward, bool timed, Lair.Contracts.WeightRef w) // WeightRef is a ref struct and cannot be captured here; it arrives as a plain parameter.
                {
                    for (int k = 0; k < activeArms.Count; k++)
                    {
                        int pos = forward ? k : activeArms.Count - 1 - k;

                        if (!timed)
                        {
                            RunBatch(arms[activeArms[pos]].Backend, w, inputs.AsSpan(0, count * cols), count, armMatrices[pos], outs[pos]); // untimed settle pass over the same buffers.
                            continue;
                        }

                        Stopwatch sw = Stopwatch.StartNew();
                        RunBatch(arms[activeArms[pos]].Backend, w, inputs.AsSpan(0, count * cols), count, armMatrices[pos], outs[pos]); // timed pass reuses the parity buffer — outputs are bit-identical across rounds by construction.
                        sw.Stop();
                        samples[pos].Add(sw.Elapsed.TotalMilliseconds);
                    }
                }

                // Two untimed settle passes (forward then reversed): tier-up and page/cache settling reach every arm equally before any timed sample exists.
                RunArms(forward: true, timed: false, weight);
                RunArms(forward: false, timed: false, weight);

                // Rotated timing rounds, the committed Xeon runners' measured-order discipline: round r visits the arms in order if r is even and reversed if odd, so no arm's sample set sees a warmer machine than its neighbours'.
                for (int round = 0; round < reps; round++)
                {
                    RunArms(forward: round % 2 == 0, timed: true, weight);
                }

                string eligibilityNote = activeArms.Count == arms.Count ? "" : $" eligible_arms={activeArms.Count}/{arms.Count} skipped_below_stated_width={arms.Count - activeArms.Count}";

                for (int pos = 0; pos < activeArms.Count; pos++)
                {
                    double[] sorted = [.. samples[pos]];
                    Array.Sort(sorted);
                    Console.WriteLine(StatsLine(rows, cols, count, threads, reps, weight.Bytes.Length, arms[activeArms[pos]].Name, sorted));
                }

                string divergenceNote = "";
                for (int pos = 0; pos < activeArms.Count; pos++)
                {
                    int armIndex = activeArms[pos];

                    if (!integerArmIndices.Contains(armIndex) && !residencyArmIndices.Contains(armIndex))
                    {
                        continue; // float-classified arms already share the bit-parity guard above.
                    }

                    double maxAbsDiff = 0;
                    int differing = 0;
                    for (int i = 0; i < outs[pos].Length; i++)
                    {
                        if (BitConverter.SingleToUInt32Bits(outs[0][i]) != BitConverter.SingleToUInt32Bits(outs[pos][i]))
                        {
                            differing++;
                        }

                        maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs((double)outs[pos][i] - outs[0][i]));
                    }

                    // expected activation-quantization loss against the float arms, published not asserted — each quantized arm carries its own divergence record.
                    string tokenName = arms[armIndex].Name == "integer-dot" ? "integer_divergence" : $"{arms[armIndex].Name}_divergence";
                    divergenceNote += $" {tokenName}=[differing_elements={differing}/{outs[pos].Length},max_abs_diff={maxAbsDiff:E3}]";
                }

                Console.WriteLine($"q4k-widebatch shape={rows}x{cols} count={count}{eligibilityNote} arms_bit_parity={(bitIdentical ? "bit_identical" : "MISMATCH")}{divergenceNote}");
            }

            foreach ((string name, Lair.Backends.Cpu.CpuBackend backend) in integerArms.Concat(staticArms).Concat(residencyArms))
            {
                if (repackedCopies.TryGetValue(name, out Lair.Backends.Cpu.Q4KRepackedMatrix? matrix))
                {
                    matrix.Dispose(); // a resident copy's block is the shape loop's own allocation; it ends with its preparing backend.
                }

                backend.Dispose(); // the shape's stated-width arms end here, on the same per-shape scope the float arms above already use.
            }
        }

        return 0;
    }

    // Ordinary arms run their resolved traversal; a resident-copy arm names its prepared matrix and goes through the explicit seam overload instead.
    private static void RunBatch(Lair.Backends.Cpu.CpuBackend backend, Lair.Contracts.WeightRef weight, ReadOnlySpan<float> inputs, int count, Lair.Backends.Cpu.Q4KRepackedMatrix? repacked, Span<float> outputs)
    {
        if (repacked is null)
        {
            backend.GemvBatch(inputs, count, weight, outputs);

            return;
        }

        backend.GemvBatch(inputs, count, weight, repacked, outputs); // the seam's own eligibility checks run here: width, posture, and matrix identity all refuse rather than substituting another traversal.
    }

    private static bool TryMegabytes(string value, out long megabytes)
    {
        if (string.Equals(value, "auto", StringComparison.OrdinalIgnoreCase))
        {
            megabytes = 0; // zero states the harness's own exact-fit rounded-up cap for each shape.

            return true;
        }

        if (long.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out long parsed) && parsed >= 0)
        {
            megabytes = parsed;

            return true;
        }

        megabytes = 0;

        return false;
    }

    private static string StatsLine(int rows, int cols, int count, int threads, int reps, long weightBytes, string arm, double[] sortedSamples)
    {
        // True median of the rotated rounds (middle sample for an odd round count; midpoint of the two middle samples otherwise), with min/max publishing the band.
        double medMs = sortedSamples.Length % 2 == 1 ? sortedSamples[sortedSamples.Length / 2] : (sortedSamples[sortedSamples.Length / 2 - 1] + sortedSamples[sortedSamples.Length / 2]) / 2.0;
        long macs = (long)rows * cols * count;

        // Run-to-run dispersion for this arm's own posture across the rotated rounds: sample standard deviation, plus the max-min band as a fraction of that median — the same band metric committed reports quote when an auto-arm collapse is attributed to dispatch behavior.
        double mean = 0.0;
        foreach (double sample in sortedSamples)
        {
            mean += sample;
        }

        mean /= sortedSamples.Length;

        double varianceSum = 0.0;
        foreach (double sample in sortedSamples)
        {
            double deviation = sample - mean;
            varianceSum += deviation * deviation;
        }

        double stddevMs = sortedSamples.Length > 1 ? Math.Sqrt(varianceSum / (sortedSamples.Length - 1)) : 0.0;
        double bandPct = medMs > 0.0 ? (sortedSamples[^1] - sortedSamples[0]) / medMs * 100.0 : 0.0;

        return string.Create(
            CultureInfo.InvariantCulture,
            $"q4k-widebatch shape={rows}x{cols} arm={arm} count={count} threads={threads} rounds={reps} med_ms={medMs:F3} min_ms={sortedSamples[0]:F3} max_ms={sortedSamples[^1]:F3} stddev_ms={stddevMs:F4} band_pct={bandPct:F2} ns_per_mac={(medMs * 1_000_000.0 / macs):F4} gflops={(2.0 * macs / (medMs / 1000.0) / 1e9):F3} weight_gbs={(weightBytes / (medMs / 1000.0) / 1e9):F3}");
    }

    private static bool BitsEqual(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        for (int i = 0; i < a.Length; i++)
        {
            if (BitConverter.SingleToUInt32Bits(a[i]) != BitConverter.SingleToUInt32Bits(b[i]))
            {
                return false;
            }
        }

        return true;
    }

    private static bool TryInt(string[] args, int index, out int value)
    {
        if (index < args.Length && int.TryParse(args[index], NumberStyles.Integer, CultureInfo.InvariantCulture, out value))
        {
            return true;
        }

        Console.WriteLine($"q4k-widebatch: expected an integer at argument '{(index < args.Length ? args[index] : "<missing>")}'");
        value = 0;
        return false;
    }

    // Same structurally-valid synthetic Q4_K blocks the existing gemv bench packs: safe half-float scales, plausible group codes, pseudo-random nibbles.
    private static void PackSyntheticQ4K(byte[] weights, int rows, int cols)
    {
        const ushort dBits = 0x2E3B; // ~0.185 half-float scale so decoded magnitudes stay near unity

        Random rng = new(99);
        long stride = (long)((cols + 255) / 256) * 144;
        for (int row = 0; row < rows; row++)
        {
            int blocks = (cols + 255) / 256;
            for (int b = 0; b < blocks; b++)
            {
                long blockBase = row * stride + b * 144L;

                weights[blockBase] = (byte)(dBits & 0xFF);
                weights[(int)blockBase + 1] = (byte)(dBits >> 8);
                weights[(int)blockBase + 2] = (byte)(dBits & 0xFF); // dmin uses the same safe scale
                weights[(int)blockBase + 3] = (byte)(dBits >> 8);

                for (int s = 0; s < 12; s++)
                {
                    byte v = (byte)rng.Next(64); // stays inside the 6-bit fields after masking by the kernel
                    weights[(int)blockBase + 4 + s] = v;
                }

                for (int q = 0; q < 128; q++)
                {
                    weights[(int)blockBase + 16 + q] = (byte)rng.Next(256); // both nibbles are valid quants
                }
            }
        }
    }

    private static void Usage()
    {
        Console.WriteLine("q4k-widebatch [--threads N=1] [--reps R=3] [--shapes rowsxcols,...=11008x2048,2048x11008] [--counts 1,8,16,32,64] [--diag-scratch-g1] [--integer-dot] [--integer-chain-group 2,4,8] [--static-partition 8,16] [--repacked-residency 8,16] [--repacked-cap-mb auto|N] (kernel-level attribution for the Q4_K batched dense dot: auto scratch route vs forced streaming fallback vs the unpromoted tile-executor arm (Q4_K crossover stated at 8; counts below it keep the batched-row walk), a chain-group-1 diagnostic, optionally the unpromoted integer-domain AVX2 Q4_K dot pinned at width 8 on an AVX2-only posture — each listed --integer-chain-group value adds another integer arm beside it at that stated shared-input grouping — optionally one arm per listed --static-partition crossover (the unpromoted dispatch schedule: identical kernels with contiguous per-participant canonical row ranges instead of dynamic block claiming; these arms join the float bit-parity guard because only who walks a row changes), and optionally one budget-gated repacked Q4_K residency arm per listed --repacked-residency width. Each resident-copy arm prepares its own AVX2-only backend copy under --repacked-cap-mb (default auto: exact fit rounded up per shape) or prints that arm's refusal line instead of timing it; counts below an arm's stated crossover are skipped with eligible_arms/skipped_below_stated_width tokens, and residency divergence from the float routes is classified as activation-quantization loss like integer arms. Resident-copy facts publish q4k_repacked_residency_batch_width/cap_mb/repacked_bytes/resident_backend_bytes; divergence from the float arms is classified for quantized arms only, while every arm publishes true med/min/max ms plus stddev_ms and band_pct dispersion across its own rotated rounds, ns/MAC, GFLOP/s and effective weight GB/s; arms share settle passes and rotated timing rounds like the committed Xeon runners; bench-only measurement seam - no posture or config change)");
    }
}
