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
/// line publishes true median/min/max wall ms plus ns/MAC and effective weight GB/s so amortization collapse across
/// counts is visible as a per-count efficiency curve at fixed thread posture. No production route or config posture
/// changes.
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
        List<int> counts = [1, 8, 16, 32, 64];
        List<(int Rows, int Cols)> shapes = [(11_008, 2_048), (2_048, 11_008)]; // gate/up and down widths of Qwen2.5-3B hidden=2048, intermediate=11008

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

            // The unpromoted integer-domain arm states the Q4_K batch width (8) on an AVX2-only posture so its ladder is forced deterministically even on hosts whose auto selection lands on vpdpbusd. Its outputs are expected to diverge from the float arms by activation-quantization loss, which this harness classifies rather than asserting identical.
            using Lair.Backends.Cpu.CpuBackend? integerDot = integerDotArm
                ? new Lair.Backends.Cpu.CpuBackend(threads: threads, cpuFeatures: new Lair.Backends.Cpu.CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false), tuning: new Lair.Backends.Cpu.CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8 })
                : null;

            // One arm list drives every phase below (parity, settle, rotated timing): arms are measured as a set, never one after another to completion.
            List<(string Name, Lair.Backends.Cpu.CpuBackend Backend)> arms = [("auto", autoRoute), ("streaming", streamingRoute), ("tile", tileRoute)];

            if (scratchG1 is not null)
            {
                arms.Add(("scratch-g1", scratchG1));
            }

            int integerArmIndex = -1;
            if (integerDot is not null)
            {
                integerArmIndex = arms.Count;
                arms.Add(("integer-dot", integerDot));
            }

            Console.WriteLine($"q4k-widebatch shape={rows}x{cols} stride_bytes={strideBytes} packed_mib={(weights.Length / (1024.0 * 1024.0)):F2} auto_tuning=[chain_group={autoRoute.ResolvedTuning.ChainGroup},max_scratch_cols={autoRoute.ResolvedTuning.MaxScratchCols}] streaming_tuning=[chain_group={streamingRoute.ResolvedTuning.ChainGroup},max_scratch_cols={streamingRoute.ResolvedTuning.MaxScratchCols}] tile_tuning=[matmul_route_batch_width_q4k={tileRoute.ResolvedTuning.MatmulRouteBatchWidthQ4K}]");

            foreach (int count in counts)
            {
                float[][] outs = new float[arms.Count][];
                for (int i = 0; i < arms.Count; i++)
                {
                    outs[i] = new float[count * rows];
                    arms[i].Backend.GemvBatch(inputs.AsSpan(0, count * cols), count, weight, outs[i]); // untimed parity pass: every arm must write identical bits for this fixture before any timing line below is read as a comparison.

                    if (float.IsNaN(outs[i][0]))
                    {
                        Console.WriteLine($"q4k-widebatch shape={rows}x{cols} count={count} arm={arms[i].Name}: WARNING NaN in first output value");
                    }
                }

                bool bitIdentical = true;
                for (int i = 1; i < arms.Count; i++)
                {
                    if (i == integerArmIndex)
                    {
                        continue; // the integer arm quantizes its activations: divergence from the float arms is expected and classified below instead.
                    }

                    if (!BitsEqual(outs[0], outs[i]))
                    {
                        bitIdentical = false;
                    }
                }

                List<double>[] samples = new List<double>[arms.Count];
                for (int i = 0; i < arms.Count; i++)
                {
                    samples[i] = [];
                }

                void RunArms(bool forward, bool timed, Lair.Contracts.WeightRef w) // WeightRef is a ref struct and cannot be captured here; it arrives as a plain parameter.
                {
                    for (int k = 0; k < arms.Count; k++)
                    {
                        int idx = forward ? k : arms.Count - 1 - k;

                        if (!timed)
                        {
                            arms[idx].Backend.GemvBatch(inputs.AsSpan(0, count * cols), count, w, outs[idx]); // untimed settle pass over the same buffers.
                            continue;
                        }

                        Stopwatch sw = Stopwatch.StartNew();
                        arms[idx].Backend.GemvBatch(inputs.AsSpan(0, count * cols), count, w, outs[idx]); // timed pass reuses the parity buffer — outputs are bit-identical across rounds by construction.
                        sw.Stop();
                        samples[idx].Add(sw.Elapsed.TotalMilliseconds);
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

                for (int i = 0; i < arms.Count; i++)
                {
                    double[] sorted = [.. samples[i]];
                    Array.Sort(sorted);
                    Console.WriteLine(StatsLine(rows, cols, count, threads, reps, weight.Bytes.Length, arms[i].Name, sorted));
                }

                string divergenceNote = "";
                if (integerArmIndex >= 0)
                {
                    double maxAbsDiff = 0;
                    int differing = 0;
                    for (int i = 0; i < outs[integerArmIndex].Length; i++)
                    {
                        if (BitConverter.SingleToUInt32Bits(outs[0][i]) != BitConverter.SingleToUInt32Bits(outs[integerArmIndex][i]))
                        {
                            differing++;
                        }

                        maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs((double)outs[integerArmIndex][i] - outs[0][i]));
                    }

                    divergenceNote = $" integer_divergence=[differing_elements={differing}/{outs[integerArmIndex].Length},max_abs_diff={maxAbsDiff:E3}]"; // expected activation-quantization loss against the float arms, published not asserted.
                }

                Console.WriteLine($"q4k-widebatch shape={rows}x{cols} count={count} arms_bit_parity={(bitIdentical ? "bit_identical" : "MISMATCH")}{divergenceNote}");
            }
        }

        return 0;
    }

    private static string StatsLine(int rows, int cols, int count, int threads, int reps, long weightBytes, string arm, double[] sortedSamples)
    {
        // True median of the rotated rounds (middle sample for an odd round count; midpoint of the two middle samples otherwise), with min/max publishing the band.
        double medMs = sortedSamples.Length % 2 == 1 ? sortedSamples[sortedSamples.Length / 2] : (sortedSamples[sortedSamples.Length / 2 - 1] + sortedSamples[sortedSamples.Length / 2]) / 2.0;
        long macs = (long)rows * cols * count;

        return string.Create(
            CultureInfo.InvariantCulture,
            $"q4k-widebatch shape={rows}x{cols} arm={arm} count={count} threads={threads} rounds={reps} med_ms={medMs:F3} min_ms={sortedSamples[0]:F3} max_ms={sortedSamples[^1]:F3} ns_per_mac={(medMs * 1_000_000.0 / macs):F4} gflops={(2.0 * macs / (medMs / 1000.0) / 1e9):F3} weight_gbs={(weightBytes / (medMs / 1000.0) / 1e9):F3}");
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
        Console.WriteLine("q4k-widebatch [--threads N=1] [--reps R=3] [--shapes rowsxcols,...=11008x2048,2048x11008] [--counts 1,8,16,32,64] [--diag-scratch-g1] [--integer-dot] (kernel-level attribution for the Q4_K batched dense dot: auto scratch route vs forced streaming fallback vs the unpromoted tile-executor arm (Q4_K crossover stated at 8; counts below it keep the batched-row walk), a chain-group-1 diagnostic, and optionally the unpromoted integer-domain AVX2 Q4_K dot pinned at width 8 on an AVX2-only posture — its divergence from the float arms is classified as expected activation-quantization loss instead of asserted identical; over synthetic pinned tensors at realistic feed-forward widths; arms share settle passes and rotated timing rounds like the committed Xeon runners, then publish per-count true med/min/max ms, ns/MAC, GFLOP/s and effective weight GB/s with a bit-parity guard between float arms; bench-only measurement seam - no posture or config change)");
    }
}
