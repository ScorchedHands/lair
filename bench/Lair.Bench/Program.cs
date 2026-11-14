using System.Diagnostics;
using System.Globalization;
using System.Runtime.Intrinsics.X86;
using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Tokenizer;

if (args.Length == 0)
{
    Console.WriteLine("usage: Lair.Bench <gemv|gguf-dump|gemv-file|prefill-profile|help>");
    return 1;
}

return args[0] switch
{
    "gemv" => BenchGemv(),
    "gguf-dump" => DumpGguf(args.Length > 1 ? args[1] : @"models\SmolLM-135M-Instruct-F16.gguf"),
    "gemv-file" => BenchFileGemv(args),
    "prefill-profile" => BenchPrefillProfile(args),
    _ => Usage(),
};

static int Usage()
{
    Console.WriteLine("unknown subcommand; expected gemv, gguf-dump <path>, gemv-file <gguf path> [--threads N] [--reps R] [--max-rows M], or prefill-profile <gguf path> [--threads N] [--lengths L1,L2,...] [--widths W1,W2,...] [--kernel gemv,tiled,gemm] [--reps R] [--per-token] [--out <path>]");
    return 1;
}

// Print every KV pair and tensor header of a GGUF file, for verifying metadata key spellings and tensor names.
static int DumpGguf(string path)
{
    using GgufFile gguf = GgufFile.Open(path);

    Console.WriteLine($"file={gguf.Path} version={gguf.Version} alignment={gguf.Alignment} dataStart={gguf.DataStart} size={gguf.FileSize}");
    Console.WriteLine($"kv_pairs={gguf.Metadata.Count} tensors={gguf.Tensors.Tensors.Count}");

    foreach (string key in gguf.Metadata.Keys.OrderBy(k => k, StringComparer.Ordinal))
    {
        GgufValue value = gguf.Metadata.Get(key);
        string rendered;
        if (value.Type == GgufValueType.Array)
        {
            IReadOnlyList<GgufValue> elements = value.AsArray();
            rendered = $"ARRAY[{elements.Count}] of {elements[0].Type}: " +
                string.Join(", ", elements.Take(8).Select(e => e.ToString()));
        }
        else
        {
            rendered = $"{value.Type}: {value}";
        }

        Console.WriteLine($"KV  {key} = {rendered}");
    }

    foreach (LogicalTensor tensor in gguf.Tensors.Tensors)
    {
        Console.WriteLine(
            $"TENSOR {tensor.Name} dims=[{string.Join(",", tensor.Shape.Dims)}] format={tensor.Format} " +
            $"offset={tensor.Offset} bytes={tensor.SizeBytes}");
    }

    return 0;
}

// Synthetic dense GEMV: a full 4096x4096 Q4_K weight matrix, decoded scalar / AVX2-single-threaded / threaded.
static int BenchGemv()
{
    const int Rows = 4096;
    const int Cols = 4096;

    TensorShape shape = new([Rows, Cols]);
    long strideBytes = shape.RowStrideBytes(QuantFormat.Q4_K); // ceil(4096/256) * 144 = 2304 bytes per row
    byte[] weights = new byte[Rows * strideBytes];
    PackSyntheticQ4K(weights, Rows, Cols);

    float[] x = new float[Cols];
    Random rng = new(20261031);
    for (int i = 0; i < Cols; i++)
    {
        x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
    }

    float[] output = new float[Rows];
    WeightRef w = new(weights, shape, QuantFormat.Q4_K);
    double weightBytes = weights.Length;

    Console.WriteLine($"gemv: Q4_K {Rows}x{Cols}, {weightBytes / (1024.0 * 1024.0):F1} MiB of packed weights");

    RunCase("scalar   (allowAvx2=false, threads=1)", new CpuBackend(threads: 1, allowAvx2: false), x, w, output, weightBytes);
    RunCase("avx2     (allowAvx2=true,  threads=1)", new CpuBackend(threads: 1, allowAvx2: true), x, w, output, weightBytes);
    RunCase("threaded (allowAvx2=true,  threads=auto)", new CpuBackend(), x, w, output, weightBytes);

    return 0;
}

static void RunCase(string label, CpuBackend backend, float[] x, WeightRef w, float[] output, double weightBytes)
{
    // Warm up the kernels and worker pool once, then time repeated runs.
    backend.Gemv(x, w, output);

    const int iterations = 5;
    double bestSeconds = double.PositiveInfinity;
    for (int rep = 0; rep < iterations; rep++)
    {
        Stopwatch sw = Stopwatch.StartNew();
        backend.Gemv(x, w, output);
        sw.Stop();
        bestSeconds = Math.Min(bestSeconds, sw.Elapsed.TotalSeconds);

        // Touch the result so timing cannot be optimized away and errors surface immediately.
        if (float.IsNaN(output[0]))
        {
            Console.WriteLine($"{label}: NaN in output");
        }
    }

    double rowsPerSecond = w.Shape.Rows / bestSeconds;
    double gigabytesPerSecond = weightBytes / bestSeconds / 1e9;
    Console.WriteLine($"{label,-42} time={bestSeconds * 1000.0:F2} ms  rows/s={rowsPerSecond:F0}  GB/s={gigabytesPerSecond:F2}");

    backend.Dispose();
}

// Fill the packed image with structurally valid Q4_K blocks: safe half-float scales, plausible 6-bit group codes,
// and pseudo-random nibbles. Exact weight values do not matter for throughput; block structure does.
static void PackSyntheticQ4K(byte[] weights, int rows, int cols)
{
    const ushort dBits = 0x2E3B; // ~0.185 half-float scale so decoded magnitudes stay near unity

    Random rng = new(99);
    for (int row = 0; row < rows; row++)
    {
        int blocks = (cols + 255) / 256; // cols is a multiple of 256 here: no padded tail needed in the bench
        for (int b = 0; b < blocks; b++)
        {
            int blockBase = (int)(row * ((long)((cols + 255) / 256) * 144)) + b * 144;

            weights[blockBase] = (byte)(dBits & 0xFF);
            weights[blockBase + 1] = (byte)(dBits >> 8);
            weights[blockBase + 2] = (byte)(dBits & 0xFF); // dmin uses the same safe scale
            weights[blockBase + 3] = (byte)(dBits >> 8);

            for (int s = 0; s < 12; s++)
            {
                byte v = (byte)rng.Next(64); // stays inside the 6-bit fields after masking by the kernel
                weights[blockBase + 4 + s] = v;
            }

            for (int q = 0; q < 128; q++)
            {
                weights[blockBase + 16 + q] = (byte)rng.Next(256); // both nibbles are valid quants
            }
        }
    }
}

// Real-file GEMV: mmap one GGUF, take representative weight rows per quant format that file actually carries, and time
// CpuBackend.Gemv single-threaded plus GemvBatch at a thread quota over the packed bytes exactly as they sit in the
// file. No LAIR-side packer runs here: every byte timed is a byte read from the model file through the same mmap
// weight source the engine uses, so per-format cost is measured against real quantized data rather than synthetic blocks.
static int BenchFileGemv(string[] args)
{
    if (args.Length < 2)
    {
        Console.WriteLine("usage: gemv-file <gguf path> [--threads N] [--reps R] [--max-rows M]");
        return 1;
    }

    int quota = 8;
    // reps is deliberately large: on a box running other workloads, the first passes over a tensor region are dominated
    // by page residency rather than kernel cost, and a small rep count reports those as if they were warm. At this many
    // repetitions every selected tensor converges (verified against 30/60/150-rep runs on Q4_K_M: the first-measured
    // tensor in each format group inflated ~2-4x at low reps and settled by 150).
    int reps = 150;
    int maxRows = 1024;
    for (int i = 2; i < args.Length; i += 2)
    {
        if (i + 1 >= args.Length)
        {
            Console.WriteLine($"gemv-file: option '{args[i]}' needs a value");
            return 1;
        }

        switch (args[i])
        {
            case "--threads":
                quota = ParseOption(args[i + 1], quota);
                break;
            case "--reps":
                reps = Math.Max(1, ParseOption(args[i + 1], reps));
                break;
            case "--max-rows":
                maxRows = ParseOption(args[i + 1], maxRows);
                break;
            default:
                Console.WriteLine($"gemv-file: unknown option '{args[i]}'");
                return 1;
        }
    }

    if (quota < 1)
    {
        Console.WriteLine("gemv-file: --threads must be at least 1");
        return 1;
    }

    const int BatchProbe = 16;      // CpuBackend's batch cap, so the probe measures the engine's widest batched shape
    const int ParallelRowProbe = 256; // CpuBackend's row-parallel threshold, used only to label whether a run engaged workers

    using GgufFile gguf = GgufFile.Open(args[1]);
    string arch = gguf.Metadata.GetString("general.architecture", "general");
    int hidden = gguf.Metadata.GetInt($"{arch}.embedding_length", 0);

    Console.WriteLine($"gemv-file file={gguf.Path} version={gguf.Version} alignment={gguf.Alignment} tensors={gguf.Tensors.Tensors.Count} size={gguf.FileSize}");
    Console.WriteLine(
        $"  arch={arch} embedding_length={hidden} avx2={Avx2.IsSupported} physicalCores={CpuBackend.DetectedPhysicalCoreCount} " +
        $"logicalProcessors={Environment.ProcessorCount} quotaThreads={quota} reps={reps} maxRowsPerTensor={maxRows}");

    List<RowView> views = [];
    foreach (LogicalTensor tensor in gguf.Tensors.Tensors)
    {
        int[] dims = tensor.Shape.Dims;
        if (dims.Length < 2 || dims[^1] <= 0)
        {
            continue; // one-dimensional tensors (norms, biases) are not GEMV operands
        }

        int cols = dims[^1];
        long stride = tensor.Shape.RowStrideBytes(tensor.Format);
        if (stride <= 0 || stride > int.MaxValue)
        {
            continue;
        }

        // A GGUF payload is one packed row after another over its innermost dimension, so a rank-3 slab reads as rows x cols too.
        long flatRows = tensor.Shape.ElementCount / cols;
        if (flatRows * stride != tensor.SizeBytes)
        {
            continue; // not a plain row stream for this format: skip rather than guess at its packing
        }

        views.Add(new RowView(tensor, cols, flatRows, dims.Length > 2));
    }

    using GgufMmapWeightSource source = new(gguf);
    using CpuBackend singleThreaded = new(threads: 1, allowAvx2: true);
    using CpuBackend quotaBackend = new(threadQuota: quota, allowAvx2: true); // engine posture: auto thread choice capped by the model quota

    Console.WriteLine($"  backend workers: gemv1T={singleThreaded.EffectiveWorkerCount} quotaRun={quotaBackend.EffectiveWorkerCount}");

    Random activations = new(20261110);
    foreach (QuantFormat format in views.Select(v => v.Tensor.Format).Distinct().OrderBy(f => f.ToString(), StringComparer.Ordinal))
    {
        List<RowView> pool = [.. views.Where(v => v.Tensor.Format == format)];

        // Prefer attention/MLP tensors whose contraction width is the model's embedding length: those are the rows decode streams.
        List<RowView> selected;
        string geometryNote;
        List<RowView> attnMlpHidden = [.. pool
            .Where(v => v.Cols == hidden && IsAttentionOrMlp(v.Tensor.Name))
            .OrderBy(v => v.Tensor.Name, StringComparer.Ordinal)
            .Take(4)];
        if (attnMlpHidden.Count > 0)
        {
            selected = attnMlpHidden;
            geometryNote = "attention/MLP rows at embedding width";
        }
        else
        {
            List<RowView> hiddenWidthAny = [.. pool
                .Where(v => v.Cols == hidden)
                .OrderBy(v => v.Tensor.Name, StringComparer.Ordinal)
                .Take(4)];
            if (hiddenWidthAny.Count > 0)
            {
                selected = hiddenWidthAny;
                geometryNote = "embedding/LM-head rows at embedding width (no attention/MLP tensor of this format carries it)";
            }
            else
            {
                selected = [.. pool.OrderByDescending(v => v.Tensor.SizeBytes).ThenBy(v => v.Tensor.Name, StringComparer.Ordinal).Take(4)];
                geometryNote = $"rows of contraction width {selected[0].Cols} (no tensor of this format carries the embedding width)";
            }
        }

        long rowsTotal = 0;
        long elementsTotal = 0;
        double bytesTotal = 0;
        double secondsSingle = 0;
        double secondsQuotaGemv = 0;
        double secondsQuotaBatch = 0;
        double coldSecondsSingle = 0;
        double coldSecondsQuotaGemv = 0;
        double secondsSingleRes = 0;
        double secondsQuotaGemvRes = 0;
        double secondsQuotaBatchRes = 0;
        bool anySerialAtQuota = false;

        Console.WriteLine($"FORMAT {format} ({geometryNote}) tensors={selected.Count}");
        foreach (RowView view in selected)
        {
            LogicalTensor tensor = view.Tensor;
            long stride = tensor.Shape.RowStrideBytes(format);
            int rows = (int)Math.Min(view.Rows, maxRows);

            ReadOnlySpan<byte> full = source.Resolve(tensor);
            ReadOnlySpan<byte> bytes = full.Slice(0, checked((int)(rows * stride)));
            TensorShape shape = new([rows, view.Cols]);
            WeightRef w = new(bytes, shape, format);

            float[] x = NextActivations(activations, view.Cols);
            float[] output = new float[rows];
            (double coldSingle, double s1) = TimeGemv(singleThreaded, w, x, output, reps);

            float[] batchInputs = new float[checked((long)BatchProbe * view.Cols)]; // GemvBatch's input layout is count activation vectors of cols each
            for (int vector = 0; vector < BatchProbe; vector++)
            {
                NextActivations(activations, view.Cols).CopyTo(batchInputs.AsSpan(vector * view.Cols, view.Cols));
            }

            float[] batchOutputs = new float[checked((long)BatchProbe * rows)]; // the engine's count*rows output layout
            (double coldBatch, double sB) = TimeGemvBatch(quotaBackend, w, batchInputs, BatchProbe, batchOutputs, reps);

            bool threadedEngages = quotaBackend.EffectiveWorkerCount > 1 && rows >= ParallelRowProbe;
            if (!threadedEngages)
            {
                anySerialAtQuota = true;
            }

            (double coldQuotaGemv, double sQ) = TimeGemv(quotaBackend, w, x, output, reps); // threaded single-vector GEMV: the decode shape

            rowsTotal += rows;
            elementsTotal += (long)rows * view.Cols;
            bytesTotal += rows * stride;
            secondsSingle += s1;
            secondsQuotaGemv += sQ;
            secondsQuotaBatch += sB;
            coldSecondsSingle += coldSingle;
            coldSecondsQuotaGemv += coldQuotaGemv;

            // Second pass on a private copy of the identical bytes: this machine runs other workloads alongside the bench,
            // so a mapping-timed run can be dominated by demand faults and page trimming rather than kernel cost. Copying
            // the mapped rows into an array keeps every timed byte real file content but separates streaming cost from
            // kernel cost. Both sets of numbers are reported.
            long regionBytes = checked(rows * stride);
            byte[] residentBytesBuffer = new byte[(int)regionBytes]; // private committed memory holding a copy of the mapped file bytes
            full.Slice(0, (int)regionBytes).CopyTo(residentBytesBuffer);

            {
                WeightRef wResident = new(residentBytesBuffer, shape, format);

                (double coldR1, double r1) = TimeGemv(singleThreaded, wResident, x, output, reps);
                (double coldRB, double rb) = TimeGemvBatch(quotaBackend, wResident, batchInputs, BatchProbe, batchOutputs, reps);
                (double coldRQ, double rq) = TimeGemv(quotaBackend, wResident, x, output, reps);

                secondsSingleRes += r1;
                secondsQuotaGemvRes += rq;
                secondsQuotaBatchRes += rb;

                Console.WriteLine(
                    $"  tensor {tensor.Name} dims=[{string.Join(",", tensor.Shape.Dims)}]" +
                    (view.Flattened ? " viewed as flat rows" : string.Empty) +
                    $" measuredRows={rows} cols={view.Cols} strideBytes={stride} bpw={(stride * 8.0 / view.Cols):F2}");
                Console.WriteLine(
                    $"    mmap-direct gemv1T warm ns/GEMV={s1 * 1e9 / rows:F1} GB/s={rows * (double)stride / s1 / 1e9:F2}" +
                    $" firstPass ns/GEMV={coldSingle * 1e9 / rows:F1}");
                Console.WriteLine(
                    $"    mmap-direct gemv{quotaBackend.EffectiveWorkerCount}T warm ns/GEMV={sQ * 1e9 / rows:F1} GB/s={rows * (double)stride / sQ / 1e9:F2}" +
                    $" firstPass ns/GEMV={coldQuotaGemv * 1e9 / rows:F1}" +
                    $" (threading {(threadedEngages ? "engaged" : "serial: below the 256-row threshold")})");
                Console.WriteLine(
                    $"    mmap-direct batch{BatchProbe} warm ns/dot={sB * 1e9 / ((long)rows * BatchProbe):F2}" +
                    $" GB/s={rows * (double)stride / sB / 1e9:F2} firstPass ns/dot={coldBatch * 1e9 / ((long)rows * BatchProbe):F2}");
                Console.WriteLine(
                    $"    resident    gemv1T ns/GEMV={r1 * 1e9 / rows:F1} GB/s={regionBytes / r1 / 1e9:F2}" +
                    $" firstPass ns/GEMV={coldR1 * 1e9 / rows:F1}");
                Console.WriteLine(
                    $"    resident    gemv{quotaBackend.EffectiveWorkerCount}T ns/GEMV={rq * 1e9 / rows:F1} GB/s={regionBytes / rq / 1e9:F2}" +
                    $" firstPass ns/GEMV={coldRQ * 1e9 / rows:F1}");
                Console.WriteLine(
                    $"    resident    batch{BatchProbe} ns/dot={rb * 1e9 / ((long)rows * BatchProbe):F2}" +
                    $" GB/s={regionBytes / rb / 1e9:F2} firstPass ns/dot={coldRB * 1e9 / ((long)rows * BatchProbe):F2}");
            }
        }

        double bpw = bytesTotal * 8.0 / elementsTotal;
        Console.WriteLine($"SUMMARY {format,-8} rows={rowsTotal} colsAvg={(double)elementsTotal / rowsTotal:F0} bpw={bpw:F2} bytes={bytesTotal:F0}");
        Console.WriteLine(
            $"  mmap-direct gemv   threads=1     warm ns/GEMV={secondsSingle * 1e9 / rowsTotal:F1}  ns/elt={secondsSingle * 1e9 / elementsTotal:F3}  " +
            $"GB/s={bytesTotal / secondsSingle / 1e9:F2}  firstPass GB/s={bytesTotal / coldSecondsSingle / 1e9:F2}");
        Console.WriteLine(
            $"  mmap-direct gemv   threads={quotaBackend.EffectiveWorkerCount}     warm ns/GEMV={secondsQuotaGemv * 1e9 / rowsTotal:F1}  " +
            $"GB/s={bytesTotal / secondsQuotaGemv / 1e9:F2}{(anySerialAtQuota ? "  (some tensors stayed serial: below the 256-row parallel threshold)" : string.Empty)}");
        Console.WriteLine(
            $"  mmap-direct batch{BatchProbe} threads={quotaBackend.EffectiveWorkerCount}     warm ns/dot={secondsQuotaBatch * 1e9 / ((rowsTotal) * BatchProbe):F3}  " +
            $"GB/s={bytesTotal / secondsQuotaBatch / 1e9:F2}");
        Console.WriteLine(
            $"  resident    gemv   threads=1     ns/GEMV={secondsSingleRes * 1e9 / rowsTotal:F1}  ns/elt={secondsSingleRes * 1e9 / elementsTotal:F3}  " +
            $"GB/s={bytesTotal / secondsSingleRes / 1e9:F2}");
        Console.WriteLine(
            $"  resident    gemv   threads={quotaBackend.EffectiveWorkerCount}     ns/GEMV={secondsQuotaGemvRes * 1e9 / rowsTotal:F1}  " +
            $"GB/s={bytesTotal / secondsQuotaGemvRes / 1e9:F2}");
        Console.WriteLine(
            $"  resident    batch{BatchProbe} threads={quotaBackend.EffectiveWorkerCount}     ns/dot={secondsQuotaBatchRes * 1e9 / ((rowsTotal) * BatchProbe):F3}  " +
            $"GB/s={bytesTotal / secondsQuotaBatchRes / 1e9:F2}");
    }

    return 0;
}

// One timed GEMV shape: time every pass, including the first (which pays the mmap demand faults), and report both
// the cold first pass and the best warmed pass. The warmed number is what a resident decode loop sees; the cold one
// records how much of a measurement was page-fault cost rather than kernel cost.
static (double FirstPass, double Best) TimeGemv(CpuBackend backend, WeightRef w, ReadOnlySpan<float> x, Span<float> output, int reps)
{
    Stopwatch first = Stopwatch.StartNew();
    backend.Gemv(x, w, output);
    first.Stop();

    double best = first.Elapsed.TotalSeconds;
    for (int rep = 0; rep < reps; rep++)
    {
        Stopwatch sw = Stopwatch.StartNew();
        backend.Gemv(x, w, output);
        sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalSeconds);
    }

    if (float.IsNaN(output[0]))
    {
        Console.WriteLine("  NaN in GEMV output");
    }

    return (first.Elapsed.TotalSeconds, best);
}

static (double FirstPass, double Best) TimeGemvBatch(CpuBackend backend, WeightRef w, ReadOnlySpan<float> inputs, int count, Span<float> outputs, int reps)
{
    Stopwatch first = Stopwatch.StartNew();
    backend.GemvBatch(inputs, count, w, outputs);
    first.Stop();

    double best = first.Elapsed.TotalSeconds;
    for (int rep = 0; rep < reps; rep++)
    {
        Stopwatch sw = Stopwatch.StartNew();
        backend.GemvBatch(inputs, count, w, outputs);
        sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalSeconds);
    }

    if (float.IsNaN(outputs[0]))
    {
        Console.WriteLine("  NaN in batched GEMV output");
    }

    return (first.Elapsed.TotalSeconds, best);
}

static float[] NextActivations(Random rng, int cols)
{
    float[] x = new float[cols];
    for (int i = 0; i < cols; i++)
    {
        x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
    }

    return x;
}

static bool IsAttentionOrMlp(string name) =>
    name.Contains(".attn_", StringComparison.Ordinal) || name.Contains(".ffn_", StringComparison.Ordinal);

// Prefill cost profile: forward prompt windows of the requested lengths through the engine's batched-prefill path
// (and optionally through the single-token forward, the route prefill took before row batching) over a fresh KV per
// measurement, then split each window's measured cost into a weight-traversal term and an attention term. A window of
// L rows runs in ceil(L / resolvedPrefillRowWidth) sub-batches — 16 unless --widths resolves another count — and every
// dense weight matrix is traversed once per sub-batch regardless of row count, so the weight term is a step function of
// L while attention grows with the summed causal span, L(L+1)/2 per layer and head. Both coefficients come from one
// least-squares fit of measured cost against the two terms over every measured length. The implied GB/s divides the
// packed attention/FFN bytes the engine streams per traversal by the fitted weight cost. The crossover length is where
// the fitted attention term reaches the fitted weight term. With --widths each candidate row width opens its own engine
// through the plan-resolved seam and gets its own fit, table, sub-batch counts at that resolved count, and a byte
// estimate of one full sub-batch's activation streams computed from geometry — allocation bytes at that row count, not
// measured residency. Omitting --widths runs exactly the single default engine as before. --kernel gemv,tiled,gemm adds
// a second axis: every named selection opens its own engine and gets its own fit, and a selection the file cannot run is
// reported as a refusal line naming what the engine lacks rather than falling back to a route that happens to work. The
// report closes with this process's peak working set, which is one number for the whole run: per-kernel peaks come from
// separate processes. Omitting --kernel keeps the default selection and the output shape unchanged.
static int BenchPrefillProfile(string[] args)
{
    if (args.Length < 2)
    {
        Console.WriteLine("usage: prefill-profile <gguf path> [--threads N] [--lengths 16,32,...] [--widths 16,32,...] [--kernel gemv,tiled,gemm] [--reps R] [--per-token] [--out <path>]");
        return 1;
    }

    int threads = 0; // 0 = the backend's auto-sizing, same posture as the serve surface with engine.threads unset
    int reps = 3;
    bool perToken = false;
    string? outPath = null;
    List<int> lengths = [16, 32, 64, 128, 256, 512, 1024, 1536, 2048];
    List<int>? rowWidths = null; // the resolved-row-count axis: one candidate engine per listed width; absent keeps the single default engine
    List<PrefillKernel> kernels = [PrefillKernel.GemV]; // the kernel axis: one candidate engine per named selection; absent keeps the engine's default route
    bool kernelAxis = false; // an explicit --kernel prints a kernel header per candidate section; the omitted default keeps the report shape unchanged
    for (int i = 2; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--threads":
                if (i + 1 >= args.Length)
                {
                    Console.WriteLine("prefill-profile: option '--threads' needs a value");
                    return 1;
                }

                threads = ParseOption(args[++i], threads);
                break;
            case "--reps":
                if (i + 1 >= args.Length)
                {
                    Console.WriteLine("prefill-profile: option '--reps' needs a value");
                    return 1;
                }

                reps = Math.Max(1, ParseOption(args[++i], reps));
                break;
            case "--lengths":
                if (i + 1 >= args.Length)
                {
                    Console.WriteLine("prefill-profile: option '--lengths' needs a value");
                    return 1;
                }

                List<int> parsed = [];
                foreach (string part in args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                {
                    if (!int.TryParse(part, out int length) || length < 1)
                    {
                        Console.WriteLine($"prefill-profile: invalid length '{part}'");
                        return 1;
                    }

                    parsed.Add(length);
                }

                if (parsed.Count == 0)
                {
                    Console.WriteLine("prefill-profile: --lengths listed no positive lengths");
                    return 1;
                }

                lengths = [.. parsed.Distinct().OrderBy(l => l)];
                break;
            case "--widths":
                if (i + 1 >= args.Length)
                {
                    Console.WriteLine("prefill-profile: option '--widths' needs a value");
                    return 1;
                }

                List<int> parsedWidths = [];
                foreach (string part in args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                {
                    if (!int.TryParse(part, out int width) || width < 1)
                    {
                        Console.WriteLine($"prefill-profile: invalid row width '{part}'");
                        return 1;
                    }

                    parsedWidths.Add(width);
                }

                if (parsedWidths.Count == 0)
                {
                    Console.WriteLine("prefill-profile: --widths listed no positive row widths");
                    return 1;
                }

                rowWidths = [.. parsedWidths.Distinct().OrderBy(w => w)];
                break;
            case "--kernel":
                if (i + 1 >= args.Length)
                {
                    Console.WriteLine("prefill-profile: option '--kernel' needs a value");
                    return 1;
                }

                List<PrefillKernel> parsedKernels = [];
                foreach (string part in args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                {
                    switch (part)
                    {
                        case PrefillKernelNames.GemV:
                            parsedKernels.Add(PrefillKernel.GemV);
                            break;
                        case PrefillKernelNames.Tiled:
                            parsedKernels.Add(PrefillKernel.Tiled);
                            break;
                        case PrefillKernelNames.Gemm:
                            parsedKernels.Add(PrefillKernel.Gemm);
                            break;
                        default:
                            Console.WriteLine($"prefill-profile: invalid kernel '{part}' (expected {PrefillKernelNames.GemV}, {PrefillKernelNames.Tiled}, or {PrefillKernelNames.Gemm})");
                            return 1;
                    }
                }

                if (parsedKernels.Count == 0)
                {
                    Console.WriteLine("prefill-profile: --kernel listed no kernels");
                    return 1;
                }

                kernels = [.. parsedKernels.Distinct()];
                kernelAxis = true;
                break;
            case "--per-token":
                perToken = true;
                break;
            case "--out":
                if (i + 1 >= args.Length)
                {
                    Console.WriteLine("prefill-profile: option '--out' needs a value");
                    return 1;
                }

                outPath = args[++i];
                break;
            default:
                Console.WriteLine($"prefill-profile: unknown option '{args[i]}'");
                return 1;
        }
    }

    if (threads < 0)
    {
        Console.WriteLine("prefill-profile: --threads must be 0 (auto) or positive");
        return 1;
    }

    // A candidate width past the engine's allocation ceiling is refused here naming it, before a file opens — the same refusal the engine raises at open for an out-of-range count.
    if (rowWidths is not null)
    {
        foreach (int width in rowWidths)
        {
            if (width > TransformerModel.MaxPrefillRowWidth)
            {
                Console.WriteLine($"prefill-profile: row width {width} exceeds the engine's resolved prefill row ceiling of {TransformerModel.MaxPrefillRowWidth}");
                return 1;
            }
        }
    }

    using GgufFile gguf = GgufFile.Open(args[1]);
    using CpuBackend backend = new(threads: threads, allowAvx2: true);

    // Each candidate engine owns and releases its own mmap view of the file: a shared source would be released by whichever
    // candidate disposes first while later candidates still read tensors through it.
    int maxLength = lengths[^1];

    // Geometry every candidate shares over one file: pool sizing and the header line come from the resolved plan so no measurement depends on whichever engine happens to be open. Candidate engines differ only in their resolved prefill row count.
    ModelPlan geometry = ModelPlans.Resolve(gguf);

    // Packed bytes the engine streams per weight traversal: the per-layer attention and feed-forward matrices. The
    // output head is read once per window, not once per sub-batch, so it is accounted as its own constant term.
    long traversalBytes = 0;
    long outputHeadBytes = 0;
    foreach (LogicalTensor tensor in gguf.Tensors.Tensors)
    {
        if (IsAttentionOrMlp(tensor.Name))
        {
            traversalBytes += tensor.SizeBytes;
        }
        else if (tensor.Name.Contains("output.weight", StringComparison.Ordinal) || tensor.Name.Contains("lm_head", StringComparison.Ordinal))
        {
            outputHeadBytes += tensor.SizeBytes;
        }
    }

    GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
    int[] prompt = BuildPrefillPrompt(tokenizer, maxLength);

    int blockTokens = 16; // the same block granularity the tests and the engine's KV side use
    int capacityBlocks = (maxLength + blockTokens - 1) / blockTokens + 8; // slack past the longest window so no measurement trips the pool
    using BlockPool pool = new(blockTokens, geometry.LayerCount, geometry.KvHeadCount, geometry.HeadDim, capacityBlocks, backend);

    int windowedLayers = 0;
    for (int layer = 0; layer < geometry.LayerCount; layer++)
    {
        if (geometry.AttentionWindowForLayer(layer) > 0)
        {
            windowedLayers++;
        }
    }

    Console.WriteLine($"prefill-profile file={gguf.Path} bytes={gguf.FileSize} threads={backend.EffectiveWorkerCount} maxBatchSequences={TransformerModel.MaxBatchSequences}");
    Console.WriteLine(
        $"  arch={geometry.Architecture} layers={geometry.LayerCount} hidden={geometry.HiddenSize} heads={geometry.HeadCount} " +
        $"kvHeads={geometry.KvHeadCount} headDim={geometry.HeadDim} vocab={geometry.VocabSize} windowedLayers={windowedLayers} avx2={Avx2.IsSupported}");
    Console.WriteLine(
        $"  traversalBytes={traversalBytes} outputHeadBytes={outputHeadBytes} kvBlocks={capacityBlocks} blockTokens={blockTokens} " +
        $"reps={reps} perToken={(perToken ? "on" : "off")} lengths={string.Join(",", lengths)}" +
        (rowWidths is not null ? $" rowWidths={string.Join(",", rowWidths)}" : "") +
        (kernelAxis ? $" kernels={string.Join(",", kernels.Select(PrefillKernelNames.Of))}" : ""));

    // One engine per candidate resolved width per requested kernel; without either axis this is exactly the single default engine the profile has always opened. Each candidate disposes before the next opens, so peak model memory stays one engine wide even as their tables interleave in the report body below. A kernel the engine refuses for this file contributes a refusal line and no table: the selection never falls back to the route that happens to run.
    List<int> candidateWidths = rowWidths ?? [0]; // 0: public open with no explicit resolution — whatever the plan resolves today
    List<(PrefillKernel Kernel, int Width)> candidateEngines = [];
    foreach (PrefillKernel kernel in kernels)
    {
        foreach (int width in candidateWidths)
        {
            candidateEngines.Add((kernel, width));
        }
    }

    List<string> reportBody = []; // summary and table lines of every candidate, for --out; blank separators are console-only
    PrefillKernel? sectionedKernel = null; // a kernel header is printed once per kernel section when the axis was requested
    foreach ((PrefillKernel kernel, int requested) in candidateEngines)
    {
        string kernelName = PrefillKernelNames.Of(kernel);
        if (kernelAxis && kernel != sectionedKernel)
        {
            Console.WriteLine();
            string kernelLine = $"prefill-profile kernel={kernelName}";
            reportBody.Add(kernelLine);
            Console.WriteLine(kernelLine);
            sectionedKernel = kernel;
        }

        TransformerModel? opened = OpenCandidate(gguf, backend, maxLength, kernel, requested, out string? refusalReason);
        if (opened is null)
        {
            string refusalLine = $"  kernel={kernelName} status=refused reason={refusalReason}";
            reportBody.Add(refusalLine);
            Console.WriteLine(refusalLine);
            continue;
        }

        using TransformerModel model = opened;

        int resolvedRows = model.ResolvedPrefillRowWidth; // the engine's own answer to what its prompt windows carry

        if (rowWidths is not null)
        {
            Console.WriteLine();
            string axisLine = $"prefill-profile rowWidth={resolvedRows} subBatchActivationBytes={EstimatedSubBatchActivationBytes(geometry, resolvedRows)}";
            reportBody.Add(axisLine);
            Console.WriteLine(axisLine);
        }

        List<PrefillRow> rows = [];
        foreach (int length in lengths)
        {
            int subBatches = (length + resolvedRows - 1) / resolvedRows;

            // Warm pass then best-of-reps timed passes, each over a fresh SequenceKv: a window that reused a previous
            // window's KV would score against already-written blocks and report the shorter run's residency instead.
            double batchedMs = double.PositiveInfinity;
            for (int rep = 0; rep < reps + 1; rep++)
            {
                using SequenceKv kv = new(pool);
                Stopwatch sw = Stopwatch.StartNew();
                model.Prefill(prompt.AsSpan(0, length).ToArray(), kv);
                sw.Stop();
                if (rep > 0)
                {
                    batchedMs = Math.Min(batchedMs, sw.Elapsed.TotalMilliseconds);
                }
            }

            double perTokenMs = double.NaN;
            if (perToken)
            {
                perTokenMs = double.PositiveInfinity;
                for (int rep = 0; rep < reps + 1; rep++)
                {
                    using SequenceKv kv = new(pool);
                    Stopwatch sw = Stopwatch.StartNew();
                    for (int i = 0; i < length; i++)
                    {
                        model.DecodeNext(prompt[i], kv); // the route prefill used before row batching: one token at each next free position
                    }

                    sw.Stop();
                    if (rep > 0)
                    {
                        perTokenMs = Math.Min(perTokenMs, sw.Elapsed.TotalMilliseconds);
                    }
                }
            }

            rows.Add(new PrefillRow(length, subBatches, batchedMs, perTokenMs));
        }

        // Cost model: measured cost is a weight term proportional to the sub-batch count plus an attention term
        // proportional to the summed causal span, L(L+1)/2. Both coefficients come from one least-squares fit over every
        // measured length, solved as 2x2 normal equations, so neither term is read off a single measurement and one bad
        // measurement on one length cannot flip the split.
        double weightMsPerTraversal = double.NaN;
        double attentionMsPerPair = double.NaN;
        string fitNote;
        double s11 = 0.0;
        double s12 = 0.0;
        double s22 = 0.0;
        double b1 = 0.0;
        double b2 = 0.0;
        foreach (PrefillRow row in rows)
        {
            double traversals = row.SubBatches;
            double pairs = (double)row.Length * (row.Length + 1) / 2.0;
            s11 += traversals * traversals;
            s12 += traversals * pairs;
            s22 += pairs * pairs;
            b1 += traversals * row.BatchedMs;
            b2 += pairs * row.BatchedMs;
        }

        double determinant = s11 * s22 - s12 * s12;
        if (rows.Count >= 2 && determinant > 0.0)
        {
            weightMsPerTraversal = (b1 * s22 - b2 * s12) / determinant;
            attentionMsPerPair = (s11 * b2 - s12 * b1) / determinant;
            fitNote = $"least-squares fit over {rows.Count} lengths";
        }
        else
        {
            PrefillRow only = rows.OrderBy(r => r.BatchedMs / r.SubBatches).First();
            weightMsPerTraversal = only.BatchedMs / only.SubBatches;
            attentionMsPerPair = 0.0;
            fitNote = "from min per-traversal cost (the fit needs at least two distinct sub-batch counts)";
        }

        // A negative attention coefficient means the measured costs do not rise with the causal span, which happens when
        // another workload steals cores mid-run: the cost curve stops being a function of the prompt length at all. The
        // split is reported as unusable rather than printed as a model that no measurement supports.
        bool splitUsable = attentionMsPerPair > 0.0 && weightMsPerTraversal > 0.0;
        if (!splitUsable)
        {
            fitNote = "UNRELIABLE: measured costs are not separable into the two terms on this run";
        }

        double impliedWeightGbPerSecond = traversalBytes / (weightMsPerTraversal / 1000.0) / 1e9;

        // Crossover at this engine's resolved row width: the first length where the fitted attention term reaches the
        // fitted weight term. The quadratic term is only valid up to the model's resolved attention windows, so the
        // crossover is a floor.
        PrefillRow last = rows[^1];
        int crossover = -1;
        if (splitUsable)
        {
            for (int length = 1; length <= 4 * maxLength; length++)
            {
                double attention = attentionMsPerPair * length * (length + 1) / 2.0;
                double weight = ((length + resolvedRows - 1) / resolvedRows) * weightMsPerTraversal;
                if (attention >= weight)
                {
                    crossover = length;
                    break;
                }
            }
        }

        string[] summaryLines =
        [
            $"  weightFit={weightMsPerTraversal:F2} ms/traversal ({fitNote}) impliedWeightGB/s={impliedWeightGbPerSecond:F2}",
            $"  attentionFit={attentionMsPerPair * 1000.0:F3} us per position pair",
            crossover > 0
                ? $"  crossover: fitted attention term >= fitted weight term at length >= {crossover}"
                : $"  crossover: not reached within 4x the longest measured length ({last.Length} tokens)",
        ];

        foreach (string summaryLine in summaryLines)
        {
            reportBody.Add(summaryLine);
            Console.WriteLine(summaryLine);
        }

        Console.WriteLine();
        List<string> headers = ["length", "subBatches", "batchedMs", "batchedTok/s"];
        if (perToken)
        {
            headers.Add("perTokenMs");
            headers.Add("perTokenTok/s");
        }

        headers.Add("weightMs");
        headers.Add("attentionMs");
        headers.Add("fitResidualMs");
        headers.Add("weightGB/s");
        string headerLine = string.Join("  ", headers.Select(h => h.PadLeft(Math.Max(h.Length, 12))));
        reportBody.Add(headerLine);
        Console.WriteLine(headerLine);

        foreach (PrefillRow row in rows)
        {
            double weightEst = row.SubBatches * weightMsPerTraversal;
            double attentionEst = attentionMsPerPair * (double)row.Length * (row.Length + 1) / 2.0;
            List<string> cells =
            [
                row.Length.ToString(CultureInfo.InvariantCulture),
                row.SubBatches.ToString(CultureInfo.InvariantCulture),
                $"{row.BatchedMs:F1}",
                $"{row.Length / (row.BatchedMs / 1000.0):F2}",
            ];

            if (perToken)
            {
                cells.Add($"{row.PerTokenMs:F1}");
                cells.Add($"{row.Length / (row.PerTokenMs / 1000.0):F2}");
            }

            cells.Add($"{weightEst:F1}");
            cells.Add($"{attentionEst:F1}");
            cells.Add($"{row.BatchedMs - weightEst - attentionEst:F1}");
            cells.Add($"{impliedWeightGbPerSecond:F2}");
            string tableLine = string.Join("  ", cells.Select(c => c.PadLeft(12)));
            reportBody.Add(tableLine);
            Console.WriteLine(tableLine);
        }
    }

    // Peak working set of this process at the close of the run — the engine's own mapped weights plus whatever the
    // measured windows allocated. It is one number for the whole process: a run that measured several kernels reports
    // the peak across all of them, so committed per-kernel peaks come from separate single-kernel processes.
    double peakWorkingSetMb = Process.GetCurrentProcess().PeakWorkingSet64 / (1024.0 * 1024.0);
    string peakLine = kernels.Count > 1
        ? $"  peakWorkingSetMb={peakWorkingSetMb:F1} cumulative across this process run: measure per-kernel peaks in separate processes"
        : $"  peakWorkingSetMb={peakWorkingSetMb:F1}";
    reportBody.Add(peakLine);
    Console.WriteLine(peakLine);

    if (outPath is not null)
    {
        List<string> fileLines =
        [
            $"prefill-profile file={gguf.Path} bytes={gguf.FileSize} threads={backend.EffectiveWorkerCount} maxBatchSequences={TransformerModel.MaxBatchSequences}",
            $"  arch={geometry.Architecture} layers={geometry.LayerCount} hidden={geometry.HiddenSize} heads={geometry.HeadCount} kvHeads={geometry.KvHeadCount} headDim={geometry.HeadDim} vocab={geometry.VocabSize} windowedLayers={windowedLayers} avx2={Avx2.IsSupported}",
            $"  traversalBytes={traversalBytes} outputHeadBytes={outputHeadBytes} kvBlocks={capacityBlocks} blockTokens={blockTokens} reps={reps} perToken={(perToken ? "on" : "off")} lengths={string.Join(",", lengths)}" + (rowWidths is not null ? $" rowWidths={string.Join(",", rowWidths)}" : "") + (kernelAxis ? $" kernels={string.Join(",", kernels.Select(PrefillKernelNames.Of))}" : ""),
            .. reportBody,
        ];

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        File.WriteAllLines(outPath, fileLines);
        Console.WriteLine($"wrote table to {outPath}");
    }

    return 0;
}

// A prompt of exactly the requested token count from the repeated sentence the prefill tests use, trimmed to length:
// token values only drive embedding lookups, so any in-vocabulary ids exercise the same weight and attention work.
static int[] BuildPrefillPrompt(GgufTokenizer tokenizer, int tokens)
{
    const string line = "The quick brown fox jumps over the lazy dog while the patient cat watches the rain. ";

    List<int> ids = [];
    while (ids.Count < tokens)
    {
        ids.AddRange(tokenizer.Encode(line));
    }

    return [.. ids.Take(tokens)];
}

// Activation bytes one full sub-batch carries at a given row count, computed from the plan's geometry as FP32 stream
// widths: each row holds its residual, normed, and projection streams (hidden), its query heads plus attention output
// (headCount x headDim), its k/v write sources twice (kvHeadCount x headDim), and the gate/up feed-forward halves
// (intermediateSize). The vocab-sized logits tail stays out: a prefill window reaches the output head only through
// its final row. These are allocation bytes at that row count, not measured process residency.
static long EstimatedSubBatchActivationBytes(ModelPlan plan, int rows) =>
    4L * rows * (3L * plan.HiddenSize + 2L * plan.HeadCount * plan.HeadDim + 2L * plan.KvHeadCount * plan.HeadDim + 2L * plan.IntermediateSize);

// Open one candidate engine on a stated kernel and resolved row width (0 keeps the plan's default row count). A kernel
// this file cannot run comes back as null with the engine's refusal reason, so the caller reports the refusal instead of
// opening some other route.
static TransformerModel? OpenCandidate(GgufFile gguf, CpuBackend backend, int maxContextTokens, PrefillKernel kernel, int rowWidth, out string? refusalReason)
{
    try
    {
        refusalReason = null;
        return rowWidth == 0
            ? TransformerModel.Open(gguf, backend, maxContextTokens: maxContextTokens, logLine: null, kernel)
            : TransformerModel.Open(gguf, backend, maxContextTokens: maxContextTokens, logLine: null, kernel, rowWidth);
    }
    catch (ModelException failure)
    {
        refusalReason = CollapseWhitespace(failure.Message);
        return null;
    }
}

// A refusal arrives as one long sentence; collapsing its whitespace keeps it on a single report line without rewording it.
static string CollapseWhitespace(string text) => string.Join(' ', text.Split(" \t\r\n", StringSplitOptions.RemoveEmptyEntries));

static int ParseOption(string value, int fallback) => int.TryParse(value, out int parsed) ? parsed : fallback;

internal sealed record RowView(LogicalTensor Tensor, int Cols, long Rows, bool Flattened);

internal sealed record PrefillRow(int Length, int SubBatches, double BatchedMs, double PerTokenMs);
