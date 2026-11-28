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

    /// <summary>Batch count at or above which an eligible single-matrix batched pass takes the blocked-matmul tile route (<see cref="MatmulRouteBatchWidthMin"/>..<see cref="MatmulRouteBatchWidthMax"/>; 0 keeps every batched pass on its batched-row kernel) or null to resolve.</summary>
    public int? MatmulRouteBatchWidth { get; init; }

    /// <summary>The same crossover stated for Q4_K dense matrices specifically, which no resolved posture promotes on its own (<see cref="MatmulRouteBatchWidthMin"/>..<see cref="MatmulRouteBatchWidthMax"/>) or null to resolve.</summary>
    public int? MatmulRouteBatchWidthQ4K { get; init; }

    /// <summary>The crossover at which a shared-input fused pair hands each matrix's rows to the blocked matmul tile executor instead of its own scratch traversal (<see cref="MatmulRouteBatchWidthMin"/>..<see cref="MatmulRouteBatchWidthMax"/>, 0 keeps every fused pair on its measured traversal). No resolved posture states this: it is a measurement seam, not a promotion. Null leaves the pair route exactly as today.</summary>
    public int? PairMatmulRouteBatchWidth { get; init; }

    /// <summary>The batch count at or above which an eligible Q4_K dense pass runs its rows through the integer-domain AVX2 executor (Q8_K-staged activations and pmaddubsw/pmaddwd chains) instead of today's float batched-row kernels (<see cref="MatmulRouteBatchWidthMin"/>..<see cref="MatmulRouteBatchWidthMax"/>) or null to resolve. No resolved posture states it: the variant is a measurement seam whose parity classification against pinned goldens decides promotion, and its integer executor only runs when this backend's ISA resolution landed on the AVX2 rung.</summary>
    public int? Q4KIntegerDotBatchWidth { get; init; }

    /// <summary>The batch count at or above which a threaded single-matrix dense batched dispatch hands each participant one contiguous range of canonical rows instead of letting workers claim scheduling blocks dynamically (<see cref="StaticRowPartitionMin"/>..<see cref="StaticRowPartitionMax"/>; 0 keeps today's dynamic claim loop everywhere) or null to resolve. No resolved posture states it: the static partition is a measurement seam, and stating it changes only which participant walks each independent row — never a kernel, an accumulator order, or a byte written.</summary>
    public int? StaticRowPartitionBatchWidth { get; init; }

    /// <summary>The batch count at or above which a repacked resident Q4_K read is allowed to run (<see cref="MatmulRouteBatchWidthMin"/>..<see cref="MatmulRouteBatchWidthMax"/>) or null to resolve. No resolved posture states it, and no engine open prepares the residency this crossover reads: it gates an explicit internal seam only (see <c>CpuBackend+Q4KRepackedResidency.cs</c>), where a call naming its prepared copy refuses when this width is unstated or below its batch count.</summary>
    public int? Q4KRepackedResidencyBatchWidth { get; init; }

    /// <summary>A stated static-partition width below one batch vector would claim the single-vector decode path too, which this seam deliberately leaves alone. 0 keeps every dispatch on today's dynamic claim loop.</summary>
    public const int StaticRowPartitionMin = 0;

    /// <summary>The batch ceiling every dense kernel's scratch is sized for: a static-partition width above it could never be reached by a call that threads at all.</summary>
    public const int StaticRowPartitionMax = CpuKernelTuning.MatmulRouteBatchWidthMax;

    /// <summary>Accumulator chains sharing one pass over a shared dot operand (<see cref="CpuKernelTuning.SharedInputDotChainGroupMin"/>..<see cref="CpuKernelTuning.SharedInputDotChainGroupMax"/>; 0 keeps every such dot individual) or null to resolve.</summary>
    public int? SharedInputDotChainGroup { get; init; }

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

    // The fused pair route keeps both matrices' accumulator chains live beside the two weight chunks it reads, so its register-budget
    // group is half the single-row group: four chains per matrix plus two weight chunks fit the same sixteen YMM registers. Like the
    // single-row budget it is a property of the vector posture, not of one CPU model.
    public const int PairChainGroupMax = ChainGroupMax / 2;

    // The fused triple route keeps all three matrices' accumulator chains live beside the three weight chunks it reads, so its register-budget
    // group is the widest one whose chains plus three weight chunks stay inside the sixteen YMM registers with headroom for the shared input
    // vectors the triple feeds: three chains per matrix (nine accumulators) plus three weight chunks is twelve live registers. A wider triple
    // group would spill the very accumulators grouping exists to keep live, so it caps at three rather than substitute. Like the other budgets
    // it is a property of the vector posture, not of one CPU model.
    public const int TripleChainGroupMax = 3;

    /// <summary>The narrowest row staging worth materializing: below this the streaming kernels' in-register feeds already cover the matrix.</summary>
    public const int MaxScratchColsMin = 512;

    // A staged row holds one float per column. Past 64K columns (256 KiB) it no longer stays cache-resident beside the packed row bytes
    // on any posture these kernels run on, and the streaming route wins there anyway — so wider ceilings refuse rather than pretend to tune.
    public const int MaxScratchColsMax = 65_536;

    /// <summary>0 disables the hybrid dense-matrix route: every batched pass keeps its batched-row kernel.</summary>
    public const int MatmulRouteBatchWidthMin = 0;

    /// <summary>The batch ceiling every dense kernel's scratch is sized for: a hybrid width above it could never dispatch.</summary>
    public const int MatmulRouteBatchWidthMax = 64;

    /// <summary>0 keeps every shared-operand dot individual: the batched chain route stays off.</summary>
    public const int SharedInputDotChainGroupMin = 0;

    // A shared-operand dot batch is bounded by the same register budget as every other chain-group kernel here: ChainGroupMax chains plus
    // one common-vector chunk fit AVX2's sixteen YMM registers. Wider groups spill accumulators past that budget, so they refuse rather
    // than substitute — like every other chain group in this posture vocabulary it is a property of the vector posture, not of one CPU model.
    public const int SharedInputDotChainGroupMax = ChainGroupMax;

    /// <summary>Output rows per scheduling unit for threaded dense kernels: every row-block partition of a GEMV/GemvBatch/Gemm/fused dispatch claims this many contiguous rows (the final block may be shorter).</summary>
    public required int RowBlock { get; init; }

    /// <summary>The row count at which a kernel hands its blocks to the worker pool instead of streaming them on the calling thread.</summary>
    public required int ParallelRowThreshold { get; init; }

    /// <summary>Scratch batch-dot chain group width: consecutive batch vectors whose accumulator chains share one pass over a materialized row, bounded by the vector posture's register budget.</summary>
    public required int ChainGroup { get; init; }

    /// <summary>Fused pair chain group width: consecutive batch vectors whose accumulator chains for both matrices of a pair share one pass over the pair's two materialized rows, bounded by the register budget for A+B chains plus two weight chunks.</summary>
    public required int PairChainGroup { get; init; }

    /// <summary>Fused triple chain group width: consecutive batch vectors whose accumulator chains for all three matrices of a triple share one pass over the triple's three materialized rows, bounded by the register budget for A+B+C chains plus three weight chunks.</summary>
    public required int TripleChainGroup { get; init; }

    /// <summary>The scratch batch-dot route's column ceiling: matrices wider than this keep their streaming kernels instead of staging a whole row.</summary>
    public required int MaxScratchCols { get; init; }

    /// <summary>Batch count at or above which an eligible single-matrix batched pass routes its rows through the blocked matmul tile executor instead of the batched-row scratch dot (0 disables the hybrid route). Only formats that carry a blocked matmul take it; every other format keeps its batched-row kernel.</summary>
    public required int MatmulRouteBatchWidth { get; init; }

    /// <summary>The same crossover for Q4_K dense matrices, which no auto posture promotes (0 unless stated): stating it through an internal seam makes the choice measurable and refusable rather than promoted from posture. A local AVX2-class prefill stage profile shows single-matrix Q4_K passes taking the tile route cheaper while fused gate/up pairs stay on their own traversal; promotion still waits for same-target-host validation.
    /// At 0 every unpromoted Q4_K batched pass keeps its batched-row kernel.</summary>
    public required int MatmulRouteBatchWidthQ4K { get; init; }

    /// <summary>The crossover at which a shared-input fused pair hands each matrix's rows to the blocked matmul tile executor instead of walking them through the pair scratch traversal (0 unless stated): no auto posture states it, because the pair route shares one activation walk across both matrices' chains today while the tile executor amortizes its activation loads across eight tile rows per chain group — which side pays at a given batch width is a target-host question an internal seam opens for measurement rather than something a posture decides. Whatever width a call runs at, each matrix's outputs keep that format's own ascending chunk walk, FMA association, and scalar-tail cleanup, so both traversals stay byte-identical.</summary>
    public required int PairMatmulRouteBatchWidth { get; init; }

    /// <summary>The batch count at or above which a Q4_K dense pass takes the integer-domain AVX2 executor instead of today's float batched-row kernels (0 unless stated): no posture promotes it, because quantized activations lose precision against FP32 operands and promotion waits for parity classification against pinned goldens plus seam-and-window measurement. The variant keeps its own determinism contract — ascending super-block walks with fixed int32 chain order and a fixed per-block float fold sequence, batched = solo = threaded at any worker count — but it is a different executor over lossy operands rather than another traversal of the same values: parity against unpinned-class expectations stays an open question until measured. A stated width only takes effect when this backend's ISA resolution landed on the AVX2 rung; every other posture and every format outside Q4_K keeps its current kernels at any value here.</summary>
    public required int Q4KIntegerDotBatchWidth { get; init; }

    /// <summary>Accumulator chains sharing one pass over a shared dot operand: how many consecutive inputs of a shared-operand dot batch keep their own accumulator chain beside the common vector's chunk (0 keeps every such dot individual). A batch wider than this is split into consecutive chunks of it.</summary>
    public required int SharedInputDotChainGroup { get; init; }

    /// <summary>The crossover at which a threaded single-matrix dense batched dispatch (the batched-row route or the blocked-matmul tile route) replaces dynamic block claiming with one contiguous canonical row range per participant — llama-style <c>[p*rows/T, (p+1)*rows/T)</c> resolved from the call's own worker count and scheduling-block width (0 unless stated): no posture promotes it, because static ranges trade the claim loop's load-stealing for a schedule with zero hand-off jitter, and which side pays at a given thread quota is exactly what the seam exists to measure. Whatever width a call states, every row keeps its own independent walk, so batched equals solo equals threaded byte-for-byte across any partition this value selects; single-vector GEMV decode, fused pair/triple traversals, shared-operand dot batching, and an unthreaded dispatch never consult it.</summary>
    public required int StaticRowPartitionBatchWidth { get; init; }

    /// <summary>The batch count at or above which an explicit repacked resident Q4_K read may run (0 unless stated): no posture promotes it, because nothing in this build's engine opens a resident copy — the crossover exists so a measurement seam that does state one can also state when its reads engage, and an unstated width refuses the seam rather than letting prepared bytes change what any pass runs.</summary>
    public required int Q4KRepackedResidencyBatchWidth { get; init; }

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
    /// - Pair chain group = half the resolved chain group, floored at 1 and capped at <see cref="PairChainGroupMax"/>: the fused pair route
    ///   keeps both matrices' chains live beside two weight chunks, so its register budget is half the single-row group.
    /// - Triple chain group = the resolved chain group capped at <see cref="TripleChainGroupMax"/>: the fused triple route keeps three
    ///   matrices' chains live beside three weight chunks, so its register budget is three chains per matrix plus three weight chunks.
    /// - Max scratch columns = 16_384 floats: a staged row plus its packed bytes stay cache-resident at that width on AVX2-class postures; wider matrices keep their streaming kernels.
    /// - Matmul route batch width = 8 with a vector posture and 0 without one: at or above it a single-matrix batched pass whose format carries a blocked matmul runs through the tile executor, where one activation load per chunk feeds every tile row's chain instead of the batched-row scratch route re-reading the batch once per chain group.
    /// - Q4_K matmul route batch width = 0 at every posture: this build measures the tile crossover for its Q6_K dense traversal and measured a Q4_K dense-matrix pass as a regression on the AVX2-only target host, so promoting Q4_K to that route is something an internal seam states explicitly (for measurement) rather than something a posture decides.
    /// - Pair matmul route batch width = 0 at every posture: the fused pair traversal already shares one activation walk across both matrices' chains and no measurement separates it from two tile dispatches yet, so handing a fused pair's rows to the tile executor is something an internal seam states explicitly (for measurement) rather than something a posture decides.
    /// - Shared-input dot chain group = <see cref="ChainGroupMax"/> with a vector posture and 0 without one: how many consecutive inputs of a shared-operand dot batch keep their own accumulator chain beside one load of the common operand's chunk; 0 keeps every such dot individual, where no batched chain kernel runs.
    /// - Static row partition batch width = 0 at every posture, vector or not: threaded dense dispatches keep today's dynamic block claiming until an internal seam states a crossover for measurement. A stated width only decides which participant walks each independent contiguous range of canonical rows, so outputs stay byte-identical whatever it resolves to.
    /// - Repacked resident Q4_K read width = 0 at every posture, vector or not: no posture prepares a resident copy and no dense forward asks for one, so the crossover gates an explicit internal seam only (see CpuBackend+Q4KRepackedResidency.cs). A stated value never reaches any float route.
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

        // The fused pair budget follows the resolved chain group, not a machine class: a pair keeps both matrices' chains live beside
        // two weight chunks, so its register-budget group is half the single-row group (eight chains pair at four). A narrower pinned
        // chain group pairs at half that width, and without a vector posture the pair route never runs anyway.
        int pairChainGroup = vectorPosture ? Math.Clamp(chainGroup / 2, ChainGroupMin, PairChainGroupMax) : ChainGroupMin;

        // The fused triple budget follows the resolved chain group the same way: a triple keeps three matrices' chains live beside three
        // weight chunks, so its register-budget group caps at TripleChainGroupMax (three chains per matrix plus three weight chunks fit the
        // vector posture's register budget). A narrower resolved chain group caps the triple group to it, and without a vector posture the
        // triple route never runs anyway.
        int tripleChainGroup = vectorPosture ? Math.Min(TripleChainGroupMax, chainGroup) : ChainGroupMin;

        const int AutoMaxScratchCols = 16_384; // cache-safe staging ceiling for AVX2-class postures (see MaxScratchColsMax).
        int maxScratchCols = overrides.MaxScratchCols is { } pinnedCols
            ? Validate(pinnedCols, "engine.cpu.tuning.max_scratch_cols", MaxScratchColsMin, MaxScratchColsMax, "wider staging no longer stays cache-resident beside the packed row it replaces")
            : AutoMaxScratchCols;

        // The hybrid dense-matrix route's batch width: at or above it, a single-matrix batched pass whose format carries a
        // blocked matmul runs its rows through the tile executor, where one activation-vector load per eight-column chunk
        // feeds every tile row's accumulator chain — the batched-row scratch route re-reads the activation batch once per
        // chain group instead. The measured crossover for tile reuse at prefill batch widths is a full chain group of batch
        // vectors; 0 without a vector posture, where the tile kernels do not run anyway.
        const int AutoMatmulRouteBatchWidth = 8; // the full chain-group width: the measured crossover where tile reuse starts paying at prefill batch widths
        int autoMatmulRouteBatchWidth = vectorPosture ? AutoMatmulRouteBatchWidth : 0; // without a vector posture the tile kernels never run, so the hybrid route stays off
        int matmulRouteBatchWidth = overrides.MatmulRouteBatchWidth is { } pinnedWidth
            ? Validate(pinnedWidth, "engine.cpu.tuning.matmul_route_batch_width", MatmulRouteBatchWidthMin, MatmulRouteBatchWidthMax, "a hybrid width above the batch ceiling never dispatches")
            : autoMatmulRouteBatchWidth;

        // The same crossover stated for Q4_K dense matrices rather than resolved from posture, leaving the measured Q6_K crossover untouched: a local AVX2-class prefill stage profile shows single-matrix Q4_K passes taking the tile route cheaper while fused gate/up pairs remain on their own traversal. That is not yet target-host promotion evidence, so no auto posture promotes this format; an internal seam states a width when same-host measurement wants to open that question. Whatever
        // the value, it only chooses which executor walks one matrix's chunks in the same ascending order with the same operand association, so outputs stay byte-identical at every width.
        const int AutoMatmulRouteBatchWidthQ4K = 0; // no auto promotion: Q4_K dense-matrix traversal stays on its batched-row kernels until a target-host measurement separates the two routes
        int matmulRouteBatchWidthQ4K = overrides.MatmulRouteBatchWidthQ4K is { } pinnedQ4KWidth
            ? Validate(pinnedQ4KWidth, "engine.cpu.tuning.matmul_route_batch_width_q4k", MatmulRouteBatchWidthMin, MatmulRouteBatchWidthMax, "a hybrid width above the batch ceiling never dispatches")
            : AutoMatmulRouteBatchWidthQ4K;

        // The fused pair's own crossover into that same tile executor, stated rather than resolved: today every scratch-eligible fused pair walks its two staged rows through one shared activation pass, and no measurement yet separates that traversal from two per-matrix tile dispatches at a given batch width — the shared-input win (one activation load feeds both matrices' chains) against the tile win (one activation load feeds eight tile rows' chains). Whatever an internal seam states here, each matrix's outputs keep its format's own ascending chunk walk with the same operand association and scalar-tail cleanup, so a pair on either route is byte-identical to its two independent batched passes.
        const int AutoPairMatmulRouteBatchWidth = 0; // no posture promotes it: the fused pair traversal stays in force until a target-host measurement separates the two routes for pairs
        int pairMatmulRouteBatchWidth = overrides.PairMatmulRouteBatchWidth is { } pinnedPairWidth
            ? Validate(pinnedPairWidth, "engine.cpu.tuning.pair_matmul_route_batch_width", MatmulRouteBatchWidthMin, MatmulRouteBatchWidthMax, "a hybrid width above the batch ceiling never dispatches")
            : AutoPairMatmulRouteBatchWidth;

        // The integer-domain Q4_K executor's own crossover, stated rather than resolved: no posture promotes a quantized-activation dot over FP32 operands without parity classification against pinned goldens first (the activation staging loses precision against them), and the kernel only runs when an opened backend's ISA resolution landed on the AVX2 rung — so at every auto posture this stays 0, and stating it through an internal seam makes the variant measurable while its outputs keep their own fixed-order determinism contract inside themselves. Whatever width a call states, chains advance in ascending super-block order with exact int32 lane totals folded per block; grouping only chooses how many chains share one nibble-unpack pass, so batched equals solo equals threaded at any value here.
        const int AutoQ4KIntegerDotBatchWidth = 0; // no posture promotes it: Q4_K dense passes keep their float traversals until a seam plus window measurement and a parity classification separate the integer executor.
        int q4kIntegerDotBatchWidth = overrides.Q4KIntegerDotBatchWidth is { } pinnedIntegerWidth
            ? Validate(pinnedIntegerWidth, "engine.cpu.tuning.q4k_integer_dot_batch_width", MatmulRouteBatchWidthMin, MatmulRouteBatchWidthMax, "an integer-executor width above the batch ceiling never dispatches")
            : AutoQ4KIntegerDotBatchWidth;

        // Shared-operand dot batching: at this chain-group width a shared-input dot call keeps that many inputs' accumulator chains live
        // beside one load of the common vector's chunk, where individual dots re-read the common operand once per input. The measured
        // crossover for reuse is again a full chain group; 0 without a vector posture, where every such dot stays scalar and re-reading
        // the shared operand costs nothing the batch would remove. A pinned width never widens past the register budget — it only decides
        // how many independent chains share one pass over the common operand, so outputs stay byte-identical at every value here.
        const int AutoSharedInputDotChainGroup = ChainGroupMax; // a full chain group: where sharing one common-vector load starts paying
        int autoSharedInputDotChainGroup = vectorPosture ? AutoSharedInputDotChainGroup : SharedInputDotChainGroupMin;
        int sharedInputDotChainGroup = overrides.SharedInputDotChainGroup is { } pinnedDotGroup
            ? Validate(pinnedDotGroup, "engine.cpu.tuning.shared_input_dot_chain_group", SharedInputDotChainGroupMin, SharedInputDotChainGroupMax, "a shared-dot chain group wider than eight spills accumulators past the vector posture's register budget")
            : autoSharedInputDotChainGroup;

        // The static contiguous row partition's own crossover, stated rather than resolved: today every threaded dense dispatch hands its
        // scheduling blocks to whichever participant claims one next, which balances a stalled worker at the cost of a shared counter and a
        // lock per claim. A stated width gives each participant one contiguous range of canonical rows instead — no hand-off jitter, but no
        // stealing either, so a descheduled participant delays its whole range. Which side pays is a thread-quota question this build has
        // measured only at 8/16 workers on the lab lane and once at threads=32 (the recorded auto-arm dispersion), so no posture promotes it:
        // an internal seam states a width when same-host measurement wants to open that question. Whatever value lands here, every row keeps
        // its own independent walk, so batched equals solo equals threaded byte-for-byte and the only thing the choice moves is who walks it.
        const int AutoStaticRowPartitionBatchWidth = 0; // no posture promotes it: dense dispatches keep today's dynamic claim loop until a same-build A/B separates the two schedules at a given quota.
        int staticRowPartitionBatchWidth = overrides.StaticRowPartitionBatchWidth is { } pinnedPartitionWidth
            ? Validate(pinnedPartitionWidth, "engine.cpu.tuning.static_row_partition_batch_width", CpuKernelTuningOverrides.StaticRowPartitionMin, CpuKernelTuningOverrides.StaticRowPartitionMax, "a static-partition width above the batch ceiling never dispatches")
            : AutoStaticRowPartitionBatchWidth;

        // The repacked resident Q4_K read's own crossover, stated rather than resolved: no posture opens a resident copy and nothing in any dense
        // forward asks for one, so this stays 0 at every auto posture; an internal seam that prepared a matrix under a budget gate states the width
        // beside it to say which batch counts may read the copy back. Whatever value lands here only gates explicit seam calls — no float route, no
        // unprepared pass, and no engine open consults it, so every production traversal stays byte-identical whatever this resolves to.
        const int AutoQ4KRepackedResidencyBatchWidth = 0; // no posture states a width: the residency seam engages only where an internal call both prepared a copy and stated its crossover.
        int q4kRepackedResidencyBatchWidth = overrides.Q4KRepackedResidencyBatchWidth is { } pinnedRepinnedWidth
            ? Validate(pinnedRepinnedWidth, "engine.cpu.tuning.q4k_repacked_residency_batch_width", MatmulRouteBatchWidthMin, MatmulRouteBatchWidthMax, "a residency-read width above the batch ceiling never dispatches")
            : AutoQ4KRepackedResidencyBatchWidth;

        return new CpuKernelTuning { RowBlock = rowBlock, ParallelRowThreshold = parallelThreshold, ChainGroup = chainGroup, PairChainGroup = pairChainGroup, TripleChainGroup = tripleChainGroup, MaxScratchCols = maxScratchCols, MatmulRouteBatchWidth = matmulRouteBatchWidth, MatmulRouteBatchWidthQ4K = matmulRouteBatchWidthQ4K, PairMatmulRouteBatchWidth = pairMatmulRouteBatchWidth, Q4KIntegerDotBatchWidth = q4kIntegerDotBatchWidth, SharedInputDotChainGroup = sharedInputDotChainGroup, StaticRowPartitionBatchWidth = staticRowPartitionBatchWidth, Q4KRepackedResidencyBatchWidth = q4kRepackedResidencyBatchWidth };
    }

    private static int Validate(int value, string key, int min, int max, string rationale) =>
        value >= min && value <= max
            ? value
            : throw new ArgumentOutOfRangeException(
                key, value, $"{key} {value} is outside the tunable range [{min}, {(max == int.MaxValue ? "unbounded" : max.ToString(System.Globalization.CultureInfo.InvariantCulture))}]: {rationale}.");
}
