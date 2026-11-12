using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Loader;

if (args.Length == 0)
{
    Console.WriteLine("usage: Lair.Bench <gemv|gguf-dump|gemv-file|help>");
    return 1;
}

return args[0] switch
{
    "gemv" => BenchGemv(),
    "gguf-dump" => DumpGguf(args.Length > 1 ? args[1] : @"models\SmolLM-135M-Instruct-F16.gguf"),
    "gemv-file" => BenchFileGemv(args),
    _ => Usage(),
};

static int Usage()
{
    Console.WriteLine("unknown subcommand; expected gemv, gguf-dump <path>, or gemv-file <gguf path> [--threads N] [--reps R] [--max-rows M]");
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

static int ParseOption(string value, int fallback) => int.TryParse(value, out int parsed) ? parsed : fallback;

internal sealed record RowView(LogicalTensor Tensor, int Cols, long Rows, bool Flattened);
