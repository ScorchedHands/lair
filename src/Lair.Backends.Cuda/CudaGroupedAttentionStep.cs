using System.Numerics;

using Lair.Contracts;

namespace Lair.Backends.Cuda;

/// <summary>
/// The composed window over a grouped-query head layout the committed attention kernels cannot shift-map directly: the
/// walk maps each staged query head to its kv-head by a bit shift, so it carries whole power-of-two groups of query
/// heads per kv-head. A model whose group is not a power of two — nine query heads over three kv-heads is the shape this
/// exists for — runs through padded staging instead: every real head keeps its own kv-head's group slot and one dummy
/// region absorbs each partial group, so the mapping the kernels hold stays exactly as committed. The model layout
/// crosses this seam unchanged in both directions; only the walk's staged rows are padded, and a composed window reads
/// back just the real head regions of the result while every dummy region it wrote is dropped here rather than reaching
/// any caller. Everything else — the KV write into the step's storage, the causal walk over it, the transfer tallies, and
/// refusal wording for anything outside these buffers — belongs to the wrapped <see cref="CudaDeviceStep"/> itself.
/// </summary>
internal sealed class CudaGroupedAttentionStep : IDeviceAttentionStep
{
    private readonly CudaDeviceStep _step;

    // The staging geometry the walk actually runs over: staged heads per row, and each real head's slot inside a padded row — group base times the padded group size plus its position within the group. Dummy slots (the tail of each partial group) hold zeros on every stage so no uninitialized region reaches the walk.
    private readonly int _stagedHeads;
    private readonly int[] _slotOfRealHead;

    // Padded staging and result scratch sized for this step's widest window: one padded row per staged row, real head regions copied in before each composition and out after it. Sized once at open — a composed posture runs these buffers on every layer of every window, so no forward allocates against them.
    private readonly float[] _paddedQueries;
    private readonly float[] _paddedOutput;

    internal CudaGroupedAttentionStep(CudaDeviceStep step, int headCount, int kvHeadCount)
    {
        ArgumentNullException.ThrowIfNull(step);

        _step = step;

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvHeadCount);
        if (headCount % kvHeadCount != 0)
        {
            throw new ArgumentException(
                $"composed step geometry HeadCount {headCount} is not a whole multiple of KvHeadCount {kvHeadCount}: " +
                "grouped-query attention maps consecutive query heads onto one kv-head.",
                nameof(kvHeadCount));
        }

        int groupReal = headCount / kvHeadCount;
        if (BitOperations.IsPow2((uint)groupReal))
        {
            throw new ArgumentException(
                $"composed step geometry HeadCount {headCount} over KvHeadCount {kvHeadCount} is a group of {groupReal} query " +
                "heads per kv-head: the shift mapping already walks this shape, so it needs no padded staging — open the " +
                "step directly.",
                nameof(headCount));
        }

        int groupStaged = (int)BitOperations.RoundUpToPowerOf2((uint)groupReal);
        _stagedHeads = checked(kvHeadCount * groupStaged);

        if (_stagedHeads != step.HeadCount || kvHeadCount != step.KvHeadCount)
        {
            throw new ArgumentException(
                $"composed step geometry {headCount} query heads over {kvHeadCount} kv-heads pads to {_stagedHeads} staged " +
                $"heads, but the wrapped step walks {step.HeadCount}.",
                nameof(step));
        }

        _slotOfRealHead = new int[headCount];
        for (int head = 0; head < headCount; head++)
        {
            _slotOfRealHead[head] = (head / kvHeadCount) * groupStaged + head % kvHeadCount;
        }

        long paddedRowBytes = checked((long)_stagedHeads * step.HeadDim);
        _paddedQueries = new float[checked(step.MaxWindowRows * paddedRowBytes)];
        _paddedOutput = new float[checked(step.MaxWindowRows * paddedRowBytes)];
    }

    /// <summary>Positions the wrapped step's KV storage holds per layer.</summary>
    public int MaxPositions => _step.MaxPositions;

    /// <summary>Layers the wrapped step's KV storage holds a key/value region for.</summary>
    public int LayerCount => _step.LayerCount;

    /// <summary>The model layout's query heads — fewer than the padded staging this walk runs over, which is what this seam exists to hide.</summary>
    public int HeadCount => _slotOfRealHead.Length;

    /// <summary>Key/value heads one row of the model layout carries.</summary>
    public int KvHeadCount => _step.KvHeadCount;

    /// <summary>Dimensions of one head region.</summary>
    public int HeadDim => _step.HeadDim;

    /// <summary>The widest sub-batch this composition stages at once, in the model's row width.</summary>
    public int MaxWindowRows => _step.MaxWindowRows;

    /// <summary>Whether the wrapped step walks a pool's resident device store rather than its own arena.</summary>
    public bool KvStoreBacked => _step.KvStoreBacked;

    /// <summary>Total device bytes the wrapped step holds — this seam adds no allocations of its own beyond host staging scratch.</summary>
    public long DeviceBytes => _step.DeviceBytes;

    public void RunAttentionWindow(
        int layer,
        int positionBase,
        ReadOnlySpan<float> qRows,
        ReadOnlySpan<float> kRows,
        ReadOnlySpan<float> vRows,
        float attnScale,
        float softcap,
        int window,
        Span<float> output)
    {
        int headDim = HeadDim;
        int realRegion = HeadCount * headDim;
        int kvRegion = KvHeadCount * headDim;

        if (qRows.Length == 0 || qRows.Length % realRegion != 0)
        {
            throw new ArgumentException(
                $"composed window query rows must cover whole rows of HeadCount x HeadDim ({realRegion} elements), got {qRows.Length}.",
                nameof(qRows));
        }

        int rows = qRows.Length / realRegion;
        if (kRows.Length != (long)rows * kvRegion || vRows.Length != (long)rows * kvRegion)
        {
            throw new ArgumentException(
                $"composed window key and value rows must each cover {rows} rows of KvHeadCount x HeadDim ({kvRegion} elements), " +
                $"got {kRows.Length} and {vRows.Length}.",
                nameof(kRows));
        }

        if (output.Length < (long)rows * realRegion)
        {
            throw new ArgumentException(
                $"composed window attention output must cover {rows} rows of HeadCount x HeadDim ({realRegion} elements each), got {output.Length}.",
                nameof(output));
        }

        int stagedRegion = _stagedHeads * headDim;
        Span<float> paddedWindow = _paddedQueries.AsSpan(0, checked(rows * stagedRegion));

        // A whole zeroed window per composition: real heads land in their group slots and each partial group's dummy tail stays zero, so the walk never scores an uninitialized region.
        paddedWindow.Clear();
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> row = qRows.Slice(r * realRegion, realRegion);
            Span<float> stagedRow = paddedWindow.Slice(r * stagedRegion, stagedRegion);

            for (int head = 0; head < HeadCount; head++)
            {
                int slot = _slotOfRealHead[head];
                row.Slice(head * headDim, headDim).CopyTo(stagedRow.Slice(slot * headDim, headDim));
            }
        }

        Span<float> stagedOut = _paddedOutput.AsSpan(0, checked(rows * stagedRegion));
        _step.RunAttentionWindow(layer, positionBase, paddedWindow, kRows, vRows, attnScale, softcap, window, stagedOut);

        for (int r = 0; r < rows; r++)
        {
            Span<float> outRow = output.Slice(r * realRegion, realRegion);

            for (int head = 0; head < HeadCount; head++)
            {
                int slot = _slotOfRealHead[head];
                stagedOut.Slice(r * stagedRegion + slot * headDim, headDim).CopyTo(outRow.Slice(head * headDim, headDim));
            }
        }
    }

    public void Dispose() => _step.Dispose();
}
