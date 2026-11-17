using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using Lair.Contracts;

namespace Lair.Backends.Cpu;

/// <summary>The sub-4-bit i-quant grid-load encoding axis: how a staged 32-value sub-block reaches its magnitude bytes in the format's fixed grid table. Both encodings stage byte-for-byte identical magnitudes and sign codes - they are alternative load shapes over one table, not alternative numerics.</summary>
internal enum GridLoadEncoding
{
    /// <summary>The baseline: one indexed entry load per eight-value group.</summary>
    Scalar,

    /// <summary>Hardware gathers issue whole entries at once (qword gathers for the 8-byte-entry grids, dword gathers for the 4-byte-entry grids).</summary>
    Gather,
}

/// <summary>The reporting names of an opened backend's resolved grid encoding: which staging shape its i-quant dots actually run. Internal because only a backend resolves one; the host reads it through <see cref="CpuBackend.ResolvedSelection"/>.</summary>
internal static class GridLoadEncodingNames
{
    public const string Scalar = "scalar";
    public const string Gather = "gather";

    public static string Of(GridLoadEncoding grid) => grid == GridLoadEncoding.Gather ? Gather : Scalar;
}

public sealed unsafe partial class CpuBackend
{
    // Roughly two cache lines of weight bytes streamed ahead by the Q8_K row kernels' prefetch hints. Prefetch is a pure hint:
    // it never changes computed bytes and every hinted address lies inside the tensor rows this call already owns.
    private const int GridPrefetchWindowBytes = 2048;

    /// <summary>One sub-4-bit kernel's resolved execution posture: its integer dot encoding, the grid-load staging feeds to that dot, and whether weight prefetch hints run.</summary>
    private readonly record struct Q8KKernelOptions(IntegerDotPath Dot, GridLoadEncoding Grid, bool Prefetch);

    // Resolved at open like engine.cpu.isa resolves its dot path: fixed for this backend's whole lifetime. Auto opens share one
    // process-wide measurement; forced values arrive only through internal seams (test and bench constructors), never host config.
    private readonly GridLoadEncoding _grid;
    private readonly bool _prefetchWeights;

    /// <summary>The sub-4-bit grid-load encoding this backend stages every i-quant GEMV with - scalar indexed staging or hardware gathers.</summary>
    internal GridLoadEncoding ResolvedGridEncoding => _grid;

    /// <summary>Whether weight-traversal prefetch hints run in this backend's Q8_K row kernels (outputs never depend on it).</summary>
    internal bool PrefetchWeightsEnabled => _prefetchWeights;

    // The auto grid encoding is measured exactly once per process, at the first auto engine open that has a vector posture.
    // Later auto opens read the cached answer without re-benchmarking; forced opens never consult or disturb it.
    private static readonly object GridProbeGate = new();
    private static GridLoadEncoding? _autoGridCache;

    /// <summary>Resolves the grid encoding for an auto open: a process-wide synthetic sub-4-bit GEMV benchmark, measured once and cached.</summary>
    internal static GridLoadEncoding ResolveAutoGrid(CpuFeaturePosture features)
    {
        if (!features.Avx2Available || !Avx2.IsSupported)
        {
            return GridLoadEncoding.Scalar; // without a vector posture there is no gather encoding to measure against the baseline.
        }

        lock (GridProbeGate)
        {
            return _autoGridCache ??= BenchmarkGridEncodings(features);
        }
    }

    // The synthetic benchmark behind ResolveAutoGrid: representative Q8_K GEMVs over generated sub-4-bit weights, measured through two forced backends with the row-hint axis off. Best-of interleaved passes absorb tiered-compilation and scheduling noise; gather is accepted only when it wins overall by a clear margin without collapsing badly on any format.
    private static GridLoadEncoding BenchmarkGridEncodings(CpuFeaturePosture features)
    {
        const int Rows = 128;
        const int Cols = 1024;
        const int Reps = 6;

        using CpuBackend scalarBackend = new(threads: 1, coreSet: null, threadQuota: null, allowAvx2: true, requestedIsa: CpuIsa.Auto, cpuFeatures: features, forcedGrid: GridLoadEncoding.Scalar, forcePrefetchWeights: false);
        using CpuBackend gatherBackend = new(threads: 1, coreSet: null, threadQuota: null, allowAvx2: true, requestedIsa: CpuIsa.Auto, cpuFeatures: features, forcedGrid: GridLoadEncoding.Gather, forcePrefetchWeights: false);

        QuantFormat[] formats =
        [
            QuantFormat.IQ2_XXS,
            QuantFormat.IQ2_XS,
            QuantFormat.IQ2_S,
            QuantFormat.IQ3_XXS,
            QuantFormat.IQ3_S,
        ];

        TensorShape shape = new([Rows, Cols]);
        Random rng = new(20261115);
        float[] x = new float[Cols];
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = unchecked(2f * rng.NextSingle() - 1f);
        }

        float[] scalarOutput = new float[Rows];
        float[] gatherOutput = new float[Rows];
        long scalarTotalTicks = 0;
        long gatherTotalTicks = 0;

        foreach (QuantFormat format in formats)
        {
            long strideBytes = shape.RowStrideBytes(format);
            byte[] weights = new byte[checked((int)(Rows * strideBytes))];
            rng.NextBytes(weights);

            WeightRef w = new(weights, shape, format);

            scalarBackend.Gemv(x, w, scalarOutput); // warm this format's kernels before timing them.
            gatherBackend.Gemv(x, w, gatherOutput);

            long scalarBestTicks = long.MaxValue;
            long gatherBestTicks = long.MaxValue;

            for (int rep = 0; rep < Reps; rep++)
            {
                long t0 = Stopwatch.GetTimestamp();
                scalarBackend.Gemv(x, w, scalarOutput);
                long scalarTicks = Stopwatch.GetTimestamp() - t0;

                long t1 = Stopwatch.GetTimestamp();
                gatherBackend.Gemv(x, w, gatherOutput);
                long gatherTicks = Stopwatch.GetTimestamp() - t1;

                if (scalarTicks < scalarBestTicks)
                {
                    scalarBestTicks = scalarTicks;
                }

                if (gatherTicks < gatherBestTicks)
                {
                    gatherBestTicks = gatherTicks;
                }
            }

            scalarTotalTicks += scalarBestTicks;
            gatherTotalTicks += gatherBestTicks;

            if (gatherBestTicks > scalarBestTicks * 6 / 5)
            {
                return GridLoadEncoding.Scalar; // one clear collapse is enough to keep the conservative encoding.
            }
        }

        return gatherTotalTicks * 25 < scalarTotalTicks * 24 ? GridLoadEncoding.Gather : GridLoadEncoding.Scalar;
    }

    /// <summary>Issues a pure prefetch hint at one weight address when this backend runs hints (prefetch instructions perform no read and change nothing).</summary>
    private static void PrefetchAhead(byte* p, bool enabled)
    {
        if (enabled && Sse.IsSupported)
        {
            Sse.Prefetch0(p); // prefetcht0 hint: retain the line across cache levels; no memory access happens here.
        }
    }

    /// <summary>Streams future rows of a Q8_K row loop ahead within the window, clamped to the rows this call owns.</summary>
    private static void PrefetchRowWindow(byte* bytes, long stride, int rows, int row, bool enabled)
    {
        if (!enabled || stride <= 0)
        {
            return;
        }

        long current = (long)row * stride;
        long last = ((long)rows - 1) * stride; // the final valid row start: no hint is issued past this call's own region.

        if (last <= current)
        {
            return;
        }

        long target = Math.Min(current + Math.Max(GridPrefetchWindowBytes, stride), last);
        PrefetchAhead(bytes + target, enabled: true);
    }

}
