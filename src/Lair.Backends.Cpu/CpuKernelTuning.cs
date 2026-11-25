namespace Lair.Backends.Cpu;

/// <summary>
/// Explicit CPU kernel tuning overrides. Every field is nullable: a null field leaves that knob to the backend's posture-resolved
/// default, and only stated fields pin values for measured sweeps (bench/test seams) or host configuration (<c>engine.cpu.tuning</c>).
/// A pinned value outside its accepted range refuses the open naming the qualified key instead of clamping it quietly.
/// </summary>
public sealed record CpuKernelTuningOverrides
{
    /// <summary>Output rows per scheduling unit for threaded dense kernels (<see cref="CpuKernelTuning.RowBlockMin"/>..<see cref="CpuKernelTuning.RowBlockMax"/>) or null to resolve.</summary>
    public int? RowBlock { get; init; }

    /// <summary>Row count at which threaded dispatch starts paying for itself (1 or greater) or null to resolve.</summary>
    public int? ParallelRowThreshold { get; init; }

    /// <summary>Scratch batch-dot chain group width (<see cref="CpuKernelTuning.ChainGroupMin"/>..<see cref="CpuKernelTuning.ChainGroupMax"/>) or null to resolve.</summary>
    public int? ChainGroup { get; init; }

    /// <summary>Row-scratch column ceiling for the scratch batch-dot route (<see cref="CpuKernelTuning.MaxScratchColsMin"/>..<see cref="CpuKernelTuning.MaxScratchColsMax"/>) or null to resolve.</summary>
    public int? MaxScratchCols { get; init; }

    /// <summary>The posture-resolved posture itself: no field pinned, every value resolved from detected topology and worker count.</summary>
    internal static readonly CpuKernelTuningOverrides Auto = new();
}

/// <summary>
/// One backend's resolved CPU kernel tuning. The core-count-dependent knobs are resolved once from the opened backend's posture —
/// its effective worker count (engine thread choice capped by quota) and vector-posture presence — never fixed to one machine class:
/// a row block measured on an i9 is wrong on a Xeon or a laptop, while explicit overrides may pin values for measured sweeps.
/// A value only ever changes how work is partitioned between independent units (which rows each worker claims, when threading pays
/// off, which batch-vector groups share one scratch pass, and where the scratch route stops being cache-safe); no resolved value
/// reorders a row's dot or an accumulator chain, so every posture — auto or overridden — produces byte-identical outputs.
/// </summary>
public sealed record CpuKernelTuning
{
    /// <summary>A row block below 8 is too small against per-row dispatch overhead.</summary>
    public const int RowBlockMin = 1;

    /// <summary>The committed dense-traversal baseline's widest measured row block: blocks above it stop paying off against tail imbalance on wide matrices.</summary>
    public const int RowBlockMax = 64;

    /// <summary>A dispatch threshold below one row is meaningless — nothing would ever thread.</summary>
    public const int ParallelRowThresholdMin = 1;

    /// <summary>One chain per scratch batch pass: the streaming-route floor, where every batch vector keeps its own walk of the hot scratch row.</summary>
    public const int ChainGroupMin = 1;

    // Eight Vector256 accumulator chains plus one weight chunk fit AVX2's sixteen YMM registers without spills — the register budget is a
    // property of the vector posture every kernel here runs under, not of one CPU model. Wider groups would spill accumulators to stack
    // and lose exactly what grouping buys; they refuse rather than substitute.
    public const int ChainGroupMax = 8;

    /// <summary>The narrowest row staging worth materializing: below this the streaming kernels' in-register feeds already cover the matrix.</summary>
    public const int MaxScratchColsMin = 512;

    // A staged row holds one float per column. Past 64K columns (256 KiB) it no longer stays cache-resident beside the packed row bytes
    // on any posture these kernels run on, and the streaming route wins there anyway — so wider ceilings refuse rather than pretend to tune.
    public const int MaxScratchColsMax = 65_536;

    /// <summary>Output rows per scheduling unit for threaded dense kernels: every row-block partition of a GEMV/GemvBatch/Gemm/fused dispatch claims this many contiguous rows (the final block may be shorter).</summary>
    public required int RowBlock { get; init; }

    /// <summary>The row count at which a kernel hands its blocks to the worker pool instead of streaming them on the calling thread.</summary>
    public required int ParallelRowThreshold { get; init; }

    /// <summary>Scratch batch-dot chain group width: consecutive batch vectors whose accumulator chains share one pass over a materialized row, bounded by the vector posture's register budget.</summary>
    public required int ChainGroup { get; init; }

    /// <summary>The scratch batch-dot route's column ceiling: matrices wider than this keep their streaming kernels instead of staging a whole row.</summary>
    public required int MaxScratchCols { get; init; }

    /// <summary>
    /// Resolves one backend's tuning from its effective worker count and vector posture, applying explicit overrides field by field.
    /// Auto defaults:
    /// - Row block = clamp(512 / workers, 8, 64): about half a page of weight rows per scheduling unit at any participation level — the
    ///   measured optimum on this box's 32-worker posture lands exactly on 16; fewer workers get larger contiguous blocks (fewer hand-offs,
    ///   longer prefetch streams), and more workers get smaller ones (finer load balance). Clamped to [8, <see cref="RowBlockMax"/>].
    /// - Parallel threshold = max(row block x workers / 2, 2 x row block): the measured optimum at 32 workers with a 16-row block lands
    ///   exactly on 256; because the row block itself scales inversely with worker count this stays near that point across postures while
    ///   still tracking both factors and never dispatching below two blocks of work.
    /// - Chain group = <see cref="ChainGroupMax"/> when a vector posture runs (AVX2's register budget) and 1 without one, where the scratch route never executes anyway.
    /// - Max scratch columns = 16_384 floats: a staged row plus its packed bytes stay cache-resident at that width on AVX2-class postures; wider matrices keep their streaming kernels.
    /// An override outside its accepted range fails the open naming the qualified key, its value, and the accepted span — never clamped silently.
    /// </summary>
    internal static CpuKernelTuning Resolve(int effectiveWorkerCount, bool vectorPosture, CpuKernelTuningOverrides overrides)
    {
        ArgumentNullException.ThrowIfNull(overrides);

        int workers = Math.Max(effectiveWorkerCount, 1); // a backend always runs at least the calling thread itself.

        int autoRowBlock = Math.Clamp(512 / workers, 8, RowBlockMax);
        int rowBlock = overrides.RowBlock is { } pinnedRowBlock
            ? Validate(pinnedRowBlock, "engine.cpu.tuning.row_block", RowBlockMin, RowBlockMax, "row blocks above the ceiling stop paying off against tail imbalance on wide matrices")
            : autoRowBlock;

        int parallelThreshold = overrides.ParallelRowThreshold is { } pinnedThreshold
            ? Validate(pinnedThreshold, "engine.cpu.tuning.parallel_row_threshold", ParallelRowThresholdMin, int.MaxValue, "a dispatch threshold must be at least 1 row")
            : Math.Max(rowBlock * workers / 2, 2 * rowBlock);

        // The register budget belongs to the vector posture: eight chains plus a weight chunk fit AVX2's sixteen YMM registers; without that posture no scratch route runs at all.
        int autoChainGroup = vectorPosture ? ChainGroupMax : ChainGroupMin;
        int chainGroup = overrides.ChainGroup is { } pinnedGroup
            ? Validate(pinnedGroup, "engine.cpu.tuning.chain_group", ChainGroupMin, ChainGroupMax, "a chain group wider than eight spills accumulators past the vector posture's register budget")
            : autoChainGroup;

        const int AutoMaxScratchCols = 16_384; // cache-safe staging ceiling for AVX2-class postures (see MaxScratchColsMax).
        int maxScratchCols = overrides.MaxScratchCols is { } pinnedCols
            ? Validate(pinnedCols, "engine.cpu.tuning.max_scratch_cols", MaxScratchColsMin, MaxScratchColsMax, "wider staging no longer stays cache-resident beside the packed row it replaces")
            : AutoMaxScratchCols;

        return new CpuKernelTuning { RowBlock = rowBlock, ParallelRowThreshold = parallelThreshold, ChainGroup = chainGroup, MaxScratchCols = maxScratchCols };
    }

    private static int Validate(int value, string key, int min, int max, string rationale) =>
        value >= min && value <= max
            ? value
            : throw new ArgumentOutOfRangeException(
                key, value, $"{key} {value} is outside the tunable range [{min}, {(max == int.MaxValue ? "unbounded" : max.ToString(System.Globalization.CultureInfo.InvariantCulture))}]: {rationale}.");
}
