using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Lair.Contracts;

namespace Lair.Backends.Cpu;

/// <summary>
/// The byte layout and per-row cost vocabulary of the unpromoted repacked Q4_K weight residency
/// (<see cref="CpuBackend.PrepareDenseQ4KResidency"/>). A packed Q4_K super-block holds 256 elements in 144 bytes: two half-float
/// block coefficients, twelve packed six-bit scale/min codes, and 128 nibble-packed quant bytes. A repacked resident copy keeps those
/// sixteen header bytes verbatim beside the same super-block's quants unpacked into one byte lane per element in ascending order —
/// 272 bytes per 256 elements instead of 144, so a full dense matrix costs <c>ceil(cols/256) * 272</c> resident bytes per row against
/// the file's own <c>* 144</c>. The unpacking buys the dot its nibble windows: every group is one contiguous 32-byte load, and eight
/// accumulator chains share each weight load with no AND/SRL in their inner loop. These constants are the single source of truth for
/// that cost — <see cref="Lair.Models.Q4KRepackedResidencyGate"/> bounds resident rows against declared budgets through them rather
/// than restating a second layout, and refusal messages name both numbers so an over-cap attempt reads as budget arithmetic, not a guess.
/// </summary>
public static class Q4KResidencyLayout
{
    /// <summary>Bytes one packed Q4_K super-block costs in the model file's own encoding.</summary>
    public const int PackedBlockBytes = 144;

    /// <summary>The block coefficients and eight six-bit scale/min codes a repacked copy keeps verbatim ahead of its unpacked quants.</summary>
    public const int MetadataBytes = 16;

    /// <summary>Elements — and one byte lane each — per Q4_K super-block in the resident copy.</summary>
    public const int QuantsPerBlock = 256;

    /// <summary>Bytes one repacked super-block costs: the verbatim header plus one byte lane per element of its own block.</summary>
    public const int RepackedBlockBytes = MetadataBytes + QuantsPerBlock; // 272

    private static long BlocksFor(int cols) => (cols + QuantsPerBlock - 1L) / QuantsPerBlock;

    /// <summary>Packed row stride a dense Q4_K matrix of this column count costs the model file.</summary>
    public static long PackedRowBytes(int cols)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(cols, 1);
        return BlocksFor(cols) * PackedBlockBytes;
    }

    /// <summary>Resident row stride a repacked copy of that same matrix adds beside the file's packed bytes.</summary>
    public static long RepackedRowBytes(int cols)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(cols, 1);
        return BlocksFor(cols) * RepackedBlockBytes;
    }

    /// <summary>The super-block count a row of this column count holds — identical for the packed rows and their repacked copies.</summary>
    public static long BlocksPerRow(int cols) => BlocksFor(cols);
}

/// <summary>
/// Raised when a requested repacked Q4_K residency cannot be admitted: over its stated weights ceiling, on a posture whose integer-dot
/// resolution never runs the rung this copy is read by, or for a weight that is not a dense matrix in Q4_K. Every message names what it
/// compared — packed bytes against repacked bytes against the stated ceiling, both per-row costs and the ceiling number itself — so an
/// over-cap attempt reads as budget arithmetic naming its numbers rather than a silent narrowing; this seam never allocates less than asked
/// for in place of refusing.
/// </summary>
public sealed class Q4KRepackedResidencyException : InvalidOperationException
{
    public Q4KRepackedResidencyException(string message)
        : base(message)
    {
    }
}

/// <summary>
/// One resident repacked copy of a dense Q4_K matrix, built once by <see cref="CpuBackend.PrepareDenseQ4KResidency"/> and read by that
/// backend's batched passes through <c>GemvBatch(..., repacked, ...)</c>. The bytes are an aligned block the preparing backend owns: rows
/// stay in canonical order with each super-block laid out as its packed header plus ascending unpacked quants, so a row walk addresses one
/// contiguous 32-byte group per dot step and every worker reads identical operands from the same copy. Disposal frees the block; a matrix
/// whose preparing backend was disposed refuses further use rather than reading freed memory.
/// </summary>
public sealed unsafe class Q4KRepackedMatrix : IDisposable
{
    private byte* _buffer; // aligned resident rows: rows * blocksPerRow * 272 bytes

    internal Q4KRepackedMatrix(int rows, int cols, long packedBytes)
    {
        Rows = rows;
        Cols = cols;
        BlocksPerRow = (int)Q4KResidencyLayout.BlocksPerRow(cols);
        PackedBytes = packedBytes;
        RepackedBytes = (long)rows * BlocksPerRow * Q4KResidencyLayout.RepackedBlockBytes;

        _buffer = null!;
    }

    /// <summary>Canonical row count of the matrix this copy holds.</summary>
    public int Rows { get; }

    /// <summary>Contraction width of the matrix this copy holds.</summary>
    public int Cols { get; }

    /// <summary>Q4_K super-blocks per row — identical for the packed source and this copy.</summary>
    public int BlocksPerRow { get; }

    /// <summary>Packed bytes the source matrix already costs (144-byte super-blocks).</summary>
    public long PackedBytes { get; internal set; }

    /// <summary>Resident bytes this copy adds beside them (272-byte super-blocks).</summary>
    public long RepackedBytes { get; }

    /// <summary>Whether this copy still owns its resident block.</summary>
    public bool IsPrepared => _buffer is not null;

    internal byte* Buffer => _buffer;

    internal void Allocate()
    {
        if (_buffer is not null)
        {
            throw new InvalidOperationException("a repacked Q4_K residency allocates exactly once");
        }

        _buffer = (byte*)NativeMemory.AlignedAlloc((nuint)RepackedBytes, 64);
        if (_buffer is null)
        {
            throw new OutOfMemoryException($"Failed to allocate the {RepackedBytes}-byte repacked Q4_K residency block.");
        }
    }

    public void Dispose()
    {
        byte* buffer = _buffer;

        if (buffer is null)
        {
            return; // disposal is idempotent: a copy already released holds no block.
        }

        _buffer = null;
        NativeMemory.AlignedFree(buffer);
    }
}

/// <summary>
/// The budget-gated repacked Q4_K residency seam (internal, unpromoted). Preparation unpacks one dense Q4_K matrix's rows into the
/// resident layout and refuses — naming packed bytes, repacked bytes, both per-row costs, and the stated ceiling number in megabytes — when
/// the copy does not fit inside a caller-stated weights-budget cap; it never allocates a smaller residency instead of refusing, and nothing in
/// any resolved posture prepares one. Selection is separate: an eligible batched Q4_K pass takes this route only through the explicit overload
/// at or above <c>engine.cpu.tuning.q4k_repacked_residency_batch_width</c>, on a backend whose integer-dot resolution landed on the AVX2 rung,
/// against a matrix prepared by that same backend. The kernel keeps its own determinism contract — ascending super-block walks with fixed int32
/// lane accumulation order and the same per-block float fold sequence the landed integer-domain dot applies — so batched equals solo equals
/// threaded at any worker count or chain grouping; like every activation-quantized route it is not expected bit-identical to the float kernels,
/// whose divergence classification against pinned goldens stays an open promotion question. Today no default posture prepares a copy and
/// no resolved engine posture enables the lookup; ordinary dense forwards consult this seam only when an internal measurement caller has
/// prepared matching copies on that backend and explicitly engaged it. What lands here is the measurement surface plus its refusal semantics,
/// not promoted residency (see docs/engine/residency-and-budgets.md).
/// </summary>
public sealed unsafe partial class CpuBackend
{
    // How many repacked bytes every backend open currently holds resident — reporting only, so a run can name what an admitted copy cost.
    private long _repackedResidencyBytes;

    /// <summary>Total resident bytes held by repacked Q4_K matrices prepared through this backend (reporting fact of the seam).</summary>
    internal long PreparedQ4KResidencyBytes => System.Threading.Volatile.Read(ref _repackedResidencyBytes);

    // Measurement engagement state for an explicitly opened engine arm only. Preparation records each copy's source identity so a later
    // ordinary batched pass can find exactly that matrix; nothing in any default posture sets this flag, and a disabled or unmatched backend
    // keeps its resolved traversal unchanged even when a prepared copy exists beside it.
    private readonly Dictionary<(nint, int), Q4KRepackedMatrix> _q4kResidencyBySource = new();
    private int _q4kMeasurementEngaged; // 0 keeps every ordinary dense pass on its resolved traversal.

    /// <summary>Whether this backend's ordinary dense batched passes may consult prepared repacked Q4_K copies (internal measurement seam only).</summary>
    internal bool Q4KMeasurementEngaged => System.Threading.Volatile.Read(ref _q4kMeasurementEngaged) != 0;

    /// <summary>The cap stated when the engine-open measurement arm was engaged, for reporting facts beside admitted copies.</summary>
    internal long MeasurementQ4KCapMegabytes { get; private set; }

    /// <summary>
    /// Engages this backend's ordinary dense batched Q4_K passes on prepared resident copies (internal measurement seam only). This does not
    /// allocate anything and never promotes a posture: it only says that, from now on, an eligible call whose source span was already admitted by
    /// <see cref="PrepareDenseQ4KResidency"/> reads the copy instead of its resolved float traversal. A backend without the AVX2 integer-dot rung
    /// refuses engagement because no prepared copy could be read there anyway.
    /// </summary>
    internal void EnableQ4KRepackedResidencyMeasurements(long capMegabytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(capMegabytes);

        if (!_useAvx2 || _integerDot != IntegerDotPath.Avx2)
        {
            throw new Q4KRepackedResidencyException(
                $"repacked residency measurement engagement requires the AVX2 integer-dot rung: this backend resolved " +
                $"'{ResolvedIntegerDotNames.Of(_integerDot)}', {useAvx2Note()}.");
        }

        MeasurementQ4KCapMegabytes = capMegabytes;
        System.Threading.Interlocked.Exchange(ref _q4kMeasurementEngaged, 1);
    }

    /// <summary>
    /// Prepares a resident repacked copy of one dense Q4_K matrix under a stated weights-budget cap in megabytes — the value a caller resolved
    /// from that model's own budget machinery (the ceiling division <see cref="Lair.Models.Q4KRepackedResidencyGate"/> applies to declared
    /// chains, beside whatever ledger headroom an opened host reports). Refusal paths: a matrix outside dense Q4_K; a backend whose integer-dot
    /// resolution never runs the AVX2 rung this copy is read by (a VNNI or scalar posture keeps its own kernels and says so); and a resident copy
    /// above the stated ceiling, which throws naming both per-row costs, the packed/total byte counts, and the cap in megabytes. Nothing here
    /// clamps to fit: over-cap requests refuse whole, and no resolved posture reaches this call at all.
    /// </summary>
    internal Q4KRepackedMatrix PrepareDenseQ4KResidency(WeightRef w, long weightsBudgetCapMegabytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(weightsBudgetCapMegabytes);

        int[] dims = w.Shape.Dims;
        if (dims.Length < 2 || dims[0] <= 0 || dims[^1] <= 0)
        {
            throw new Q4KRepackedResidencyException(
                "repacked residency prepares dense matrices only: the weight's shape is not a matrix with positive row and column counts.");
        }

        int rows = dims[0];
        int cols = dims[^1];

        if (w.Format != QuantFormat.Q4_K)
        {
            throw new Q4KRepackedResidencyException(
                $"repacked residency is stated for format Q4_K only: the weight carries format {w.Format}, and this seam never repacks another " +
                "format's rows or borrows Q4_K's cost numbers for it.");
        }

        long packedRowBytes = Q4KResidencyLayout.PackedRowBytes(cols);
        long expectedPackedBytes = (long)rows * packedRowBytes;
        if ((long)w.Bytes.Length != expectedPackedBytes)
        {
            throw new Q4KRepackedResidencyException(
                $"WeightRef byte length is {w.Bytes.Length} but dense Q4_K shape {rows}x{cols} requires {expectedPackedBytes} bytes " +
                $"({rows} rows x {packedRowBytes}-byte packed row stride).");
        }

        if (!_useAvx2 || _integerDot != IntegerDotPath.Avx2)
        {
            throw new Q4KRepackedResidencyException(
                $"repacked Q4_K residency is read by the AVX2 integer-dot rung only: this backend resolved '{ResolvedIntegerDotNames.Of(_integerDot)}' " +
                useAvx2Note() + ".");
        }

        long repackedRowBytes = Q4KResidencyLayout.RepackedRowBytes(cols);
        long repackedBytes = (long)rows * repackedRowBytes;
        long capBytes = weightsBudgetCapMegabytes > long.MaxValue / Megabyte ? long.MaxValue : checked(weightsBudgetCapMegabytes * Megabyte);

        if (repackedBytes > capBytes)
        {
            // The refusal names every number the decision compared: this build's own budget key vocabulary, both per-row costs in bytes, the
            // matrix totals against each other, and the ceiling stated in megabytes. Nothing narrows the residency to fit — over-cap requests end here.
            throw new Q4KRepackedResidencyException(
                $"repacked Q4_K residency refused: format Q4_K dense matrix {rows}x{cols} costs {expectedPackedBytes} packed bytes " +
                $"({Q4KResidencyLayout.PackedBlockBytes}-byte super-blocks, {packedRowBytes} per row) and its resident copy adds {repackedBytes} bytes " +
                $"({Q4KResidencyLayout.RepackedBlockBytes} per 256-element super-block, {repackedRowBytes} per row), above the stated 'weights.caps.dram_mb' " +
                $"ceiling of {weightsBudgetCapMegabytes} megabytes ({capBytes} bytes) — no repacked rows are admitted and nothing falls back to a narrower copy.");
        }

        Q4KRepackedMatrix matrix = new(rows, cols, expectedPackedBytes);

        fixed (byte* pSource = w.Bytes)
        {
            matrix.Allocate();

            try
            {
                RepackQ4KRows(pSource, matrix.Buffer, rows, cols);
            }
            catch
            {
                matrix.Dispose();
                throw;
            }
        }

        RegisterPreparedQ4KSource(w.Bytes, matrix);

        Interlocked.Add(ref _repackedResidencyBytes, repackedBytes);
        return matrix;
    }

    /// <summary>
    /// One batched Q4_K dense pass over a prepared resident copy (internal seam overload). Eligibility is stated twice on purpose: the call names
    /// the matrix it wants read back from residency, and <c>engine.cpu.tuning.q4k_repacked_residency_batch_width</c> must be pinned at or above this
    /// call's batch count — an unstated width refuses naming the qualified key instead of letting a prepared copy change what any pass runs. The
    /// weight span handed alongside must describe exactly the matrix that was packed into the copy (shape, format, and byte length), so a wrong
    /// pairing refuses rather than reading mismatched bytes. Staging is the same shared Q8_K activation region and exact group-sum table the landed
    /// integer-domain executor builds — one quantized set per batch vector plus its int16 group sums, built once for the call — and rows partition
    /// through this backend's ordinary dispatch gates (row block, parallel threshold, static-partition schedule), each row an independent walk.
    /// </summary>
    internal void GemvBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Q4KRepackedMatrix repacked, Span<float> outputs)
    {
        ArgumentNullException.ThrowIfNull(repacked);

        int[] dims = w.Shape.Dims;
        if (dims.Length < 2 || dims[0] <= 0 || dims[^1] <= 0)
        {
            throw new ArgumentException("Weight shape must be a matrix with positive row and column counts.", nameof(w));
        }

        int rows = dims[0];
        int cols = dims[^1];

        if (w.Format != QuantFormat.Q4_K || repacked.Rows != rows || repacked.Cols != cols)
        {
            throw new ArgumentException(
                $"the resident copy holds dense Q4_K rows of a {repacked.Rows}x{repacked.Cols} matrix; the WeightRef handed with it is " +
                $"{rows}x{cols} in format {w.Format}. A repacked read names exactly the matrix that was prepared.", nameof(repacked));
        }

        long packedRowBytes = Q4KResidencyLayout.PackedRowBytes(cols);
        if ((long)w.Bytes.Length != (long)rows * packedRowBytes || w.Bytes.Length != repacked.PackedBytes)
        {
            throw new ArgumentException(
                $"WeightRef byte length is {w.Bytes.Length} but the prepared residency was built from {repacked.PackedBytes} packed bytes " +
                $"({rows} rows x {packedRowBytes}-byte packed row stride).", nameof(w));
        }

        if (!repacked.IsPrepared)
        {
            throw new ObjectDisposedException(nameof(Q4KRepackedMatrix), "the resident copy was disposed; its preparing backend no longer holds it.");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, MaxBatch);

        if ((long)inputs.Length != (long)count * cols)
        {
            throw new ArgumentException(
                $"Batched input length {inputs.Length} must equal count {count} times the weight column count {cols}.", nameof(inputs));
        }

        if ((long)outputs.Length < (long)count * rows)
        {
            throw new ArgumentException(
                $"Batched output length {outputs.Length} is shorter than count {count} times the weight row count {rows}.", nameof(outputs));
        }

        int width = _tuning.Q4KRepackedResidencyBatchWidth;
        if (width <= 0 || count < width)
        {
            throw new ArgumentOutOfRangeException(
                "engine.cpu.tuning.q4k_repacked_residency_batch_width", count,
                $"a repacked Q4_K read requires a stated crossover at or above the call's batch count: resolved " +
                $"{_tuning.Q4KRepackedResidencyBatchWidth} against count {count}. No posture states this width; stating it through an internal seam is what opens the route.");
        }

        if (!_useAvx2 || _integerDot != IntegerDotPath.Avx2)
        {
            throw new Q4KRepackedResidencyException(
                $"repacked Q4_K residency reads on the AVX2 integer-dot rung only: this backend resolved '{ResolvedIntegerDotNames.Of(_integerDot)}', " +
                useAvx2Note());
        }

        int blocks = repacked.BlocksPerRow; // activation staging and resident copy share one super-block count per row.
        byte[] qaRent = ArrayPool<byte>.Shared.Rent(count * blocks * QkK);
        float[] daRent = ArrayPool<float>.Shared.Rent(count * blocks);
        short[] groupSumsRent = ArrayPool<short>.Shared.Rent(count * blocks * Q4KGroups);

        try
        {
            fixed (float* pIn = inputs)
            fixed (float* pOut = outputs)
            fixed (byte* qa = qaRent)
            fixed (float* da = daRent)
            fixed (short* groupSums = groupSumsRent)
            {
                QuantizeActivationsToQ8K(pIn, count, cols, qa, da); // identical operands for every row and worker of the call.
                StageQ4KIntegerGroupSums(qa, blocks, count, groupSums); // each chain's exact activation group sums — this route folds them with the same helpers as the landed integer executor.

                int workers = _effectiveThreads;
                CpuKernelTuning tuning = _tuning;
                bool parallel = workers > 1 && rows >= tuning.ParallelRowThreshold;
                int repackedStride = blocks * Q4KResidencyLayout.RepackedBlockBytes;

                if (!parallel)
                {
                    for (int row = 0; row < rows; row++)
                    {
                        RowBatchDotQ4KRepacked(
                            repacked.Buffer + (long)row * repackedStride, qa, da, groupSums, blocks, count, rows, tuning.ChainGroup, pOut + row);
                    }

                    return;
                }

                WorkerPool pool = GetOrCreatePool();
                bool staticRowPartition = tuning.StaticRowPartitionBatchWidth > 0 && count >= tuning.StaticRowPartitionBatchWidth;
                RunThreadedDispatch(pool, (rows + tuning.RowBlock - 1) / tuning.RowBlock, workers, staticRowPartition,
                    new RepackedBatchRowJob(repacked.Buffer, repackedStride, qa, da, groupSums, blocks, pOut, rows, count, tuning));
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(qaRent);
            ArrayPool<float>.Shared.Return(daRent);
            ArrayPool<short>.Shared.Return(groupSumsRent); // the call's shared staging tables: rows and workers read them, never copy them.
        }
    }

    /// <summary>Reads a prepared resident copy through the ordinary dense batch route only when this backend was explicitly engaged, its stated crossover covers the call, and the exact source span has an admitted matrix.</summary>
    private bool TryRunQ4KMeasurementResidency(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs)
    {
        if (System.Threading.Volatile.Read(ref _q4kMeasurementEngaged) == 0)
        {
            return false; // default posture: a prepared copy beside an unengaged backend changes nothing.
        }

        int width = _tuning.Q4KRepackedResidencyBatchWidth;
        if (width <= 0 || count < width)
        {
            return false; // the explicit seam's own crossover governs ordinary reads too; below it every pass keeps its resolved traversal.
        }

        if (w.Format != QuantFormat.Q4_K)
        {
            return false; // only dense Q4_K rows have a repacked layout to read back.
        }

        (nint address, int length) key = SourceKey(w.Bytes);
        if (!_q4kResidencyBySource.TryGetValue(key, out var matrix))
        {
            return false; // this backend has no admitted copy for that exact source span; the resolved route runs unchanged.
        }

        if (!matrix.IsPrepared)
        {
            throw new ObjectDisposedException(nameof(Q4KRepackedMatrix), "the engaged engine arm named a resident copy whose preparing backend already released it.");
        }

        GemvBatch(inputs, count, w, matrix, outputs); // the explicit overload re-checks matrix identity and its own refusal semantics.
        return true;
    }

    private void RegisterPreparedQ4KSource(ReadOnlySpan<byte> bytes, Q4KRepackedMatrix matrix)
    {
        (nint address, int length) key = SourceKey(bytes);

        if (!_q4kResidencyBySource.TryAdd(key, matrix))
        {
            matrix.Dispose(); // the duplicate copy never gets counted or read: one prepared residency belongs to exactly one source span.
            throw new Q4KRepackedResidencyException(
                $"a repacked Q4_K residency already exists for this exact {bytes.Length}-byte source span; preparing it twice would let two resident copies claim the same matrix.");
        }
    }

    private static (nint Address, int Length) SourceKey(ReadOnlySpan<byte> bytes) =>
        bytes.Length == 0 ? ((nint)0, 0) : (((nint)(long)Unsafe.AsPointer(ref MemoryMarshal.GetReference(bytes)), bytes.Length));

    private void DisposeQ4KMeasurementResidency()
    {
        foreach (Q4KRepackedMatrix matrix in _q4kResidencyBySource.Values)
        {
            matrix.Dispose(); // backend disposal releases every copy it prepared, including engine-open measurement arms not disposed by hand.
        }

        _q4kResidencyBySource.Clear();
        System.Threading.Interlocked.Exchange(ref _repackedResidencyBytes, 0L);
    }

    private string useAvx2Note() => _useAvx2
        ? "so its i-quant dots run another encoding and this resident layout is not what they read"
        : "and a backend opened without its vector posture runs no vector kernel that could read it";

    // Unpack one dense Q4_K matrix's rows into the resident layout: sixteen header bytes copied verbatim per super-block, then each group-pair
    // window expanded to 64 byte lanes in element order — low nibbles of its 32 source bytes first (group 2jp), high nibbles second (group 2jp+1).
    // Padded columns past the real width repack exactly as their source bytes hold them: every dot against a padded lane multiplies that chain's
    // staged zero quant, so padding contributes nothing to any group sum whatever route reads it (the same staging semantics the Q8_K dots define).
    private static void RepackQ4KRows(byte* source, byte* destination, int rows, int cols)
    {
        int blocks = (int)Q4KResidencyLayout.BlocksPerRow(cols);

        for (int row = 0; row < rows; row++)
        {
            byte* srcRow = source + (long)row * blocks * Q4KResidencyLayout.PackedBlockBytes;
            byte* dstRow = destination + (long)row * blocks * Q4KResidencyLayout.RepackedBlockBytes;

            for (int b = 0; b < blocks; b++)
            {
                byte* srcBlock = srcRow + b * Q4KResidencyLayout.PackedBlockBytes;
                byte* dstBlock = dstRow + b * Q4KResidencyLayout.RepackedBlockBytes;

                for (int i = 0; i < Q4KResidencyLayout.MetadataBytes; i++)
                {
                    dstBlock[i] = srcBlock[i]; // the two half-float coefficients and twelve packed scale/min codes, byte-for-byte.
                }

                byte* quants = srcBlock + Q4KResidencyLayout.MetadataBytes;
                byte* lanes = dstBlock + Q4KResidencyLayout.MetadataBytes;

                for (int jp = 0; jp < QkK / 64; jp++) // four group-pair windows per super-block: each source window covers groups 2jp and 2jp+1.
                {
                    byte* srcPair = quants + 32 * jp;
                    byte* dstPair = lanes + 64 * jp;

                    for (int l = 0; l < 32; l++)
                    {
                        byte packed = srcPair[l];
                        dstPair[l] = (byte)(packed & 0x0F); // group 2jp's element, in the same order its scalar and integer traversals read it.
                        dstPair[32 + l] = (byte)(packed >> 4); // group 2jp+1's element from that source byte's high nibble.
                    }
                }
            }
        }
    }

    /// <summary>
    /// Row-block executor for one repacked resident read. Raw pointers stay valid only while the pinned staging spans in the call's fixed scope do,
    /// and WorkerPool.Run guarantees every claimed block finishes before it returns; rows are independent walks of identical shared operands, so any
    /// partition — dynamic claiming or a stated static schedule — writes bit-identical bytes.
    /// </summary>
    private sealed unsafe class RepackedBatchRowJob : IGemvJob
    {
        private readonly byte* _copyRows; // the resident copy's row region: rows * blocksPerRow * 272 bytes, canonical order.
        private readonly int _rowStride;
        private readonly byte* _qa;
        private readonly float* _da;
        private readonly short* _groupSums;
        private readonly int _blocks;
        private readonly float* _out;
        private readonly int _rows;
        private readonly int _count;
        private readonly int _chainGroup;
        private readonly int _rowBlock;

        internal RepackedBatchRowJob(byte* copyRows, int rowStride, byte* qa, float* da, short* groupSums, int blocks, float* outBase, int rowCount, int count, CpuKernelTuning tuning)
        {
            _copyRows = copyRows;
            _rowStride = rowStride;
            _qa = qa;
            _da = da;
            _groupSums = groupSums;
            _blocks = blocks;
            _out = outBase;
            _rows = rowCount;
            _count = count;
            _chainGroup = tuning.ChainGroup;
            _rowBlock = tuning.RowBlock;
        }

        public void ProcessBlock(int block)
        {
            int firstRow = block * _rowBlock;
            int lastRow = Math.Min(firstRow + _rowBlock, _rows);

            for (int row = firstRow; row < lastRow; row++)
            {
                RowBatchDotQ4KRepacked(_copyRows + (long)row * _rowStride, _qa, _da, _groupSums, _blocks, _count, _rows, _chainGroup, _out + row);
            }
        }
    }

    // One resident row's batched dot. Chain grouping arrives from the call's resolved tuning snapshot and only decides how many accumulator chains
    // share one contiguous group load: each chain advances its own staged activation region in the same ascending order whatever group it sits in, so
    // every output is bit-identical at any group width or split — and identical between a threaded dispatch and an unthreaded walk of the same copy.
    private static void RowBatchDotQ4KRepacked(byte* repackedRow, byte* qa, float* da, short* groupSums, int blocks, int count, int rows, int chainGroupWidth, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch];

        fixed (float* pAcc = acc)
        {
            for (int b0 = 0; b0 < count; b0++)
            {
                pAcc[b0] = 0f; // every chain starts its row cold: the fold sequence is identical whatever group walked it.
            }

            int width = Math.Min(chainGroupWidth, CpuKernelTuning.ChainGroupMax); // a pinned group never exceeds the register budget here either.

            long activationStride = (long)blocks * QkK; // one vector's staged region: whole super-blocks, as QuantizeActivationsToQ8K laid them out.
            long sumsStride = (long)blocks * Q4KGroups; // the same chain-by-chain layout for this call's group-sum table (eight exact sums per block).

            for (int b0 = 0; b0 < count;)
            {
                int chains = Math.Min(width, count - b0); // a ragged tail below one group runs through the stacked-chain arm.

                byte* qaGroup = qa + b0 * activationStride;
                float* daGroup = da + b0 * blocks;
                short* sumsGroup = groupSums + b0 * sumsStride;

                if (chains == CpuKernelTuning.ChainGroupMax)
                {
                    Q4KRepackedChains8(repackedRow, qaGroup, daGroup, sumsGroup, blocks, activationStride, sumsStride, pAcc + b0);
                }
                else
                {
                    Q4KRepackedStackChains(repackedRow, qaGroup, daGroup, sumsGroup, activationStride, sumsStride, blocks, chains, pAcc + b0);
                }

                b0 += chains;
            }

            for (int b = 0; b < count; b++)
            {
                outBase[(long)b * rows] = pAcc[b]; // each chain's output is its own accumulator: grouping never folds one chain into another.
            }
        }
    }

    // The register-budget arm: eight chains' int32 lane accumulators stay live across the whole row while every group load (two contiguous 32-byte
    // quants windows per nibble window) feeds all of them — a repacked read's whole point is that no chain re-unpacks what the copy already unpacked.
    private static void Q4KRepackedChains8(byte* row, byte* qaBase, float* daBase, short* groupSumsBase, int blocks, long activationStride, long sumsStride, float* acc)
    {
        Span<short> mins = stackalloc short[Q4KGroups]; // one block's unpacked mins: read once per block and shared by every chain's fold.

        for (int i = 0; i < blocks; i++)
        {
            byte* block = row + (long)i * Q4KResidencyLayout.RepackedBlockBytes;

            float dHalf = ReadHalfLE(block); // the same half-float coefficients every Q4_K route reads, folded per chain at this block boundary.
            float minHalf = ReadHalfLE(block + 2);

            LoadGroupMinsQ4K(block, mins); // the copied header carries the identical packed scale/min codes GetScaleMinK4 unpacks elsewhere.

            byte* quants = block + Q4KResidencyLayout.MetadataBytes;
            byte* qaRow = qaBase + (long)i * QkK;
            short* sumsRow = groupSumsBase + i * Q4KGroups;

            Vector256<int> s0 = Vector256<int>.Zero, s1 = Vector256<int>.Zero, s2 = Vector256<int>.Zero, s3 = Vector256<int>.Zero;
            Vector256<int> s4 = Vector256<int>.Zero, s5 = Vector256<int>.Zero, s6 = Vector256<int>.Zero, s7 = Vector256<int>.Zero;

            for (int jp = 0; jp < QkK / 64; jp++)
            {
                byte* groupBase = quants + 64 * jp; // groups 2jp and 2jp+1 sit contiguous in the copy: each is one 32-byte lane window.

                Vector256<byte> weightLo = Avx.LoadVector256(groupBase); // group 2jp's unpacked quants, byte lanes in [0, 15].
                Vector256<byte> weightHi = Avx.LoadVector256(groupBase + 32); // group 2jp+1's.

                int scaleIndexLo = GetGroupScaleQ4K(block, 2 * jp); // the same six-bit codes every Q4_K kernel unpacks, broadcast per group window here instead of shuffled per pair position.
                Vector256<short> scaleLo = Vector256.Create((short)scaleIndexLo);
                int scaleHiValue = GetGroupScaleQ4K(block, 2 * jp + 1);
                Vector256<short> scaleHi = Vector256.Create((short)scaleHiValue);

                byte* q8 = qaRow + 64 * jp; // each chain's own staged activation windows at the same ascending position.

                s0 = Q4KRepackedAccumulate(s0, weightLo, weightHi, scaleLo, scaleHi, q8);
                s1 = Q4KRepackedAccumulate(s1, weightLo, weightHi, scaleLo, scaleHi, q8 + activationStride);
                s2 = Q4KRepackedAccumulate(s2, weightLo, weightHi, scaleLo, scaleHi, q8 + 2 * activationStride);
                s3 = Q4KRepackedAccumulate(s3, weightLo, weightHi, scaleLo, scaleHi, q8 + 3 * activationStride);
                s4 = Q4KRepackedAccumulate(s4, weightLo, weightHi, scaleLo, scaleHi, q8 + 4 * activationStride);
                s5 = Q4KRepackedAccumulate(s5, weightLo, weightHi, scaleLo, scaleHi, q8 + 5 * activationStride);
                s6 = Q4KRepackedAccumulate(s6, weightLo, weightHi, scaleLo, scaleHi, q8 + 6 * activationStride);
                s7 = Q4KRepackedAccumulate(s7, weightLo, weightHi, scaleLo, scaleHi, q8 + 7 * activationStride);
            }

            // Block boundary: every chain folds its exact int32 totals into float in the same fixed order the landed integer executor applies —
            // nibble total under the fused activation x weight scale first, then the min-weighted group sums under the negated min coefficient.
            FoldQ4KIntegerBlock(acc + 0, s0, Q4KMinTermFromSums(sumsRow, mins), daBase[i] * dHalf, -(daBase[i] * minHalf));
            FoldQ4KIntegerBlock(acc + 1, s1, Q4KMinTermFromSums(sumsRow + sumsStride, mins), daBase[blocks + i] * dHalf, -(daBase[blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 2, s2, Q4KMinTermFromSums(sumsRow + 2 * sumsStride, mins), daBase[2 * blocks + i] * dHalf, -(daBase[2 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 3, s3, Q4KMinTermFromSums(sumsRow + 3 * sumsStride, mins), daBase[3 * blocks + i] * dHalf, -(daBase[3 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 4, s4, Q4KMinTermFromSums(sumsRow + 4 * sumsStride, mins), daBase[4 * blocks + i] * dHalf, -(daBase[4 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 5, s5, Q4KMinTermFromSums(sumsRow + 5 * sumsStride, mins), daBase[5 * blocks + i] * dHalf, -(daBase[5 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 6, s6, Q4KMinTermFromSums(sumsRow + 6 * sumsStride, mins), daBase[6 * blocks + i] * dHalf, -(daBase[6 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 7, s7, Q4KMinTermFromSums(sumsRow + 7 * sumsStride, mins), daBase[7 * blocks + i] * dHalf, -(daBase[7 * blocks + i] * minHalf));
        }
    }

    // The grouped arm for narrower pinned groups and ragged tails: the same ascending walk with each chain's eight int32 accumulator lanes staged
    // through a stack slot instead of live YMM registers — more load/store traffic, identical bits.
    private static void Q4KRepackedStackChains(byte* row, byte* qaBase, float* daBase, short* groupSumsBase, long activationStride, long sumsStride, int blocks, int chains, float* acc)
    {
        Span<int> lanes = stackalloc int[CpuKernelTuning.ChainGroupMax * 8]; // eight exact accumulator lanes per chain.

        Span<short> mins = stackalloc short[Q4KGroups];

        for (int i = 0; i < blocks; i++)
        {
            byte* block = row + (long)i * Q4KResidencyLayout.RepackedBlockBytes;

            float dHalf = ReadHalfLE(block);
            float minHalf = ReadHalfLE(block + 2);

            LoadGroupMinsQ4K(block, mins); // the same two shuffles serve every chain in this group.

            byte* quants = block + Q4KResidencyLayout.MetadataBytes;
            byte* qaRow = qaBase + (long)i * QkK;
            short* sumsRow = groupSumsBase + i * Q4KGroups;

            fixed (int* pLanes = lanes)
            {
                for (int c = 0; c < chains * 8; c++)
                {
                    pLanes[c] = 0; // every chain starts each super-block cold: lane totals cover exactly one block.
                }

                for (int jp = 0; jp < QkK / 64; jp++)
                {
                    byte* groupBase = quants + 64 * jp; // the contiguous group windows feed every chain in the group.

                    Vector256<byte> weightLo = Avx.LoadVector256(groupBase);
                    Vector256<byte> weightHi = Avx.LoadVector256(groupBase + 32);

                    Vector256<short> scaleLo = Vector256.Create((short)GetGroupScaleQ4K(block, 2 * jp));
                    Vector256<short> scaleHi = Vector256.Create((short)GetGroupScaleQ4K(block, 2 * jp + 1));

                    byte* q8 = qaRow + 64 * jp; // each group window spans exactly its own two staged activation windows.

                    for (int c = 0; c < chains; c++)
                    {
                        int* chainLanes = pLanes + c * 8; // the accumulator slot: one load and store per group pair, identical lane order to the register arm.

                        Vector256<int> accChain = Avx.LoadVector256(chainLanes);
                        accChain = Q4KRepackedAccumulate(accChain, weightLo, weightHi, scaleLo, scaleHi, q8 + (long)c * activationStride);
                        Avx.Store(chainLanes, accChain);
                    }
                }

                for (int c = 0; c < chains; c++)
                {
                    FoldQ4KIntegerBlock(
                        acc + c,
                        Avx.LoadVector256(pLanes + c * 8), // the chain's eight exact lane totals, folded in fixed order.
                        Q4KMinTermFromSums(sumsRow + c * sumsStride, mins), // this chain's table row for this block: chains sit blocks x eight sums apart.
                        daBase[(long)c * blocks + i] * dHalf,
                        -(daBase[(long)c * blocks + i] * minHalf));
                }
            }
        }
    }

    // One (chain, group pair) of the resident read: each group's contiguous 32-byte quant window dotted against that chain's staged signed quants into
    // adjacent-pair int16 sums (pmaddubsw), then folded through pmaddwd against the group scale broadcast across every pair lane — two adjacent pair
    // products per exact int32 lane, ascending element order inside each product. A pair sum is at most 15 * 127 * 2 = 3810 and a scale-folded dword
    // covers one group's four elements under sc <= 63 (at most 4 * 63 * 1905), so no lane saturates anywhere; per-block totals fold into float at the
    // boundary exactly like every other route here. Only the two activation loads and four integer MACs per chain remain in this shape: the weight side
    // is a plain contiguous load because the residency already unpacked it.
    private static Vector256<int> Q4KRepackedAccumulate(Vector256<int> accChain, Vector256<byte> weightLo, Vector256<byte> weightHi, Vector256<short> scaleLo, Vector256<short> scaleHi, byte* q8)
    {
        Vector256<sbyte> aL = Avx.LoadVector256(q8).AsSByte(); // group 2jp's quants for this chain: signed, in [-127, 127] by the quantizer.
        Vector256<short> pairsL = Avx2.MultiplyAddAdjacent(weightLo, aL);

        Vector256<sbyte> aH = Avx.LoadVector256(q8 + 32).AsSByte(); // group 2jp+1's quants: the contiguous window beside the first in both rows.
        Vector256<short> pairsH = Avx2.MultiplyAddAdjacent(weightHi, aH);

        Vector256<int> sumj = Avx2.Add(Avx2.MultiplyAddAdjacent(pairsL, scaleLo), Avx2.MultiplyAddAdjacent(pairsH, scaleHi));
        return Avx2.Add(accChain, sumj); // ascending group-pair order inside the chain's block accumulator.
    }

    // One group's six-bit scale read straight out of this copy's copied header — GetScaleMinK4's own unpacking for that group index, applied to a whole
    // contiguous window instead of per-pair shuffle positions because repacked groups are no longer interleaved across nibble windows.
    private static int GetGroupScaleQ4K(byte* block, int group)
    {
        GetScaleMinK4(group, block + 4, out byte sc, out _);
        return sc;
    }

    // The megabyte unit every weights ceiling counts in — the same constant budget gates divide against (see Lair.Models residency gates).
    private const long Megabyte = 1024L * 1024;
}
