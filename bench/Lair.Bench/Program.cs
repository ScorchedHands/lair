using System.Diagnostics;
using Lair.Backends.Cpu;
using Lair.Core.Contracts;
using Lair.Loader;

if (args.Length == 0)
{
    Console.WriteLine("usage: Lair.Bench <gemv|gguf-dump|help>");
    return 1;
}

return args[0] switch
{
    "gemv" => BenchGemv(),
    "gguf-dump" => DumpGguf(args.Length > 1 ? args[1] : @"models\SmolLM-135M-Instruct-F16.gguf"),
    _ => Usage(),
};

static int Usage()
{
    Console.WriteLine("unknown subcommand; expected gemv or gguf-dump <path>");
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
