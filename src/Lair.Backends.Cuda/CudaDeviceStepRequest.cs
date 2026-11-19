using System.Numerics;

namespace Lair.Backends.Cuda;

/// <summary>
/// The rotary pairing a device step's rotation kernel applies to staged q/k rows. AdjacentPairs rotates dimensions
/// <c>(2i, 2i + 1)</c> of each head region, the layout llama-family checkpoints lay their rotary rows in; SplitHalfPairs
/// rotates <c>(i, i + headDim / 2)</c>, the layout the qwen3 and gemma-family checkpoints use. The forward engine's
/// resolved pairing decides which one a step runs, and the angle tables are the same position-major pair tables either
/// pairing reads.
/// </summary>
public enum RopePairing
{
    /// <summary>Each rotary pair occupies two adjacent head dimensions.</summary>
    AdjacentPairs,

    /// <summary>Each rotary pair occupies one head dimension and its partner half a head away.</summary>
    SplitHalfPairs,
}

/// <summary>
/// The shape one on-device transformer step is opened for: the positions and layers its KV arena holds, the head
/// geometry its attention walk runs over, the pairing its rotation applies, and the widest staged window its row
/// buffers hold. Every value is validated before any device work starts, so a geometry the kernel family cannot walk
/// refuses naming the field and the value rather than faulting mid-step.
/// </summary>
public sealed record CudaDeviceStepRequest
{
    /// <summary>Positions the KV arena holds per layer; a step may only write and read positions below this.</summary>
    public required int MaxPositions { get; init; }

    /// <summary>Layers the KV arena holds a K and V region for.</summary>
    public required int LayerCount { get; init; }

    /// <summary>Query heads one row of the step carries.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Key/value heads one row of the step carries; grouped-query attention maps consecutive query heads onto them.</summary>
    public required int KvHeadCount { get; init; }

    /// <summary>Dimensions of one head region, which is also the width of one attention lane.</summary>
    public required int HeadDim { get; init; }

    /// <summary>The rotary pairing the rotation kernel applies to staged query and key rows.</summary>
    public RopePairing Pairing { get; init; } = RopePairing.AdjacentPairs;

    /// <summary>Rows one staged window may hold: the widest prefill sub-batch or decode batch this step composes at once.</summary>
    public int MaxWindowRows { get; init; } = DefaultMaxWindowRows;

    /// <summary>The staged-window width a request that names none gets.</summary>
    public const int DefaultMaxWindowRows = 64;

    // The head-dimension ceiling the attention kernel's shared window is sized for: one product slot per lane.
    internal const int MaxHeadDim = 128;

    /// <summary>
    /// Query heads per kv-head: validation keeps the head count a whole multiple of the kv-head count and the group a
    /// power of two, which is the group size the attention kernel's shift-based mapping divides by.
    /// </summary>
    internal int HeadsPerKvHead => HeadCount / KvHeadCount;

    /// <summary>
    /// Refuses a geometry the step kernels cannot walk, naming each field and the value that broke it: a head dimension
    /// the rotation cannot pair or the attention lanes cannot cover, a head count the grouped-query mapping cannot
    /// divide or a group size its shift cannot express, a non-positive arena, or a staged window wider than the
    /// buffers the step allocates.
    /// </summary>
    public void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxPositions);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(LayerCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(HeadCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(KvHeadCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxWindowRows);

        if (HeadDim <= 0)
        {
            throw new ArgumentException($"device step request HeadDim {HeadDim} must be a positive head dimension.", nameof(HeadDim));
        }

        if (HeadDim % 2 != 0)
        {
            throw new ArgumentException(
                $"device step request HeadDim {HeadDim} must be even: rotary pairs need two dimensions each.",
                nameof(HeadDim));
        }

        if (HeadDim > MaxHeadDim)
        {
            throw new ArgumentException(
                $"device step request HeadDim {HeadDim} exceeds the {MaxHeadDim} dimensions the attention kernel's lanes cover.",
                nameof(HeadDim));
        }

        if (HeadCount % KvHeadCount != 0)
        {
            throw new ArgumentException(
                $"device step request HeadCount {HeadCount} is not a whole multiple of KvHeadCount {KvHeadCount}: " +
                "grouped-query attention maps consecutive query heads onto one kv-head.",
                nameof(KvHeadCount));
        }

        if (!BitOperations.IsPow2(HeadsPerKvHead))
        {
            throw new ArgumentException(
                $"device step request HeadCount {HeadCount} over KvHeadCount {KvHeadCount} is a group of {HeadsPerKvHead} " +
                "query heads per kv-head: the attention kernel maps each query head to its kv-head by a bit shift, so " +
                "the group size must be a power of two.",
                nameof(KvHeadCount));
        }
    }
}
