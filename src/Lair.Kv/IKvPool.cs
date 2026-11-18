namespace Lair.Kv;

/// <summary>
/// The pool surface the scheduler, metrics, and prefix cache address, whichever KV format pair a model opened with: the same
/// free-list/refcounted block pool, the same soft-budget/admission-window bookkeeping, and the same windowed-read footprint
/// accounting whether blocks store FP32 slots or a quantized pair's groups. <see cref="BlockPool"/> and
/// <see cref="BlockPool{TK, TV}"/> implement it, so a host that resolved a quantized pair at open hands the same scheduler the
/// float posture runs while the format-specific views stay on the concrete pool. <see cref="NewSequence"/> hands out the
/// sequence type this pool's storage is addressed by, so a pipeline never guesses which concrete sequence its pool backs.
/// Device-resident members report host-only posture on a quantized pool: quantized blocks carry no device mirror in this build.
/// </summary>
public interface IKvPool : IDisposable
{
    int BlockTokens { get; }

    int LayerCount { get; }

    int KvHeads { get; }

    int HeadDim { get; }

    /// <summary>Values per token for one of K or V: kvHeads * headDim.</summary>
    int KvDim { get; }

    /// <summary>Total bytes stored in one physical block across all layers and both K/V sides, for whichever pair this pool opened with.</summary>
    long BytesPerBlock { get; }

    /// <summary>The resolved KV format pair this pool's block layout was sized for.</summary>
    KvFormatPair Formats { get; }

    int CapacityBlocks { get; }

    /// <summary>Soft usage cap within the physical reservation.</summary>
    int EffectiveBudgetBlocks { get; }

    /// <summary>Free blocks still reachable under the soft cap — what admission and step accounting actually spend from.</summary>
    int EffectiveFreeBlocks { get; }

    /// <summary>The ceiling allocations currently reach: the soft cap, or an open admission window's wider reservation.</summary>
    int AllocationCeiling { get; }

    int FreeBlocks { get; }

    /// <summary>Physical blocks currently handed out, including shared (refcount &gt; 1) owners.</summary>
    int UsedBlocks { get; }

    /// <summary>Total token slots the pool can hold at full capacity.</summary>
    long SlotCapacity { get; }

    /// <summary>Token slots still available in free blocks.</summary>
    long FreeSlots { get; }

    /// <summary>The residency mode this pool opened with. A quantized pool is host-only.</summary>
    BlockResidencyMode Residency { get; }

    /// <summary>How many leading block ids are declared device-backed. Zero for a host-only pool.</summary>
    int DeviceBackedBlockCount { get; }

    /// <summary>True when this physical id falls inside the device-backed range.</summary>
    bool IsDeviceBacked(int physicalBlockId);

    /// <summary>The number of layers declared with a finite attention window.</summary>
    int DeclaredWindowedLayerCount { get; }

    /// <summary>Logical KV blocks the model's attention would select across all layers at the current soft budget.</summary>
    long KvReadFootprintBlocksAtBudget { get; }

    /// <summary>Whole logical blocks excluded by declared windows at the current soft budget.</summary>
    long SkippedKvReadFootprintBlocksAtBudget { get; }

    /// <summary>Takes the next free block and returns its physical id with refcount 1.</summary>
    int Allocate();

    /// <summary>Increments the reference count of a live block so another owner may share it.</summary>
    void AddRef(int physicalBlockId);

    /// <summary>Decrements the reference count; the block returns to the free list only at zero.</summary>
    void Release(int physicalBlockId);

    /// <summary>Current reference count of a block; zero means the block is free.</summary>
    int RefCount(int physicalBlockId);

    /// <summary>Sets the soft usage cap, clamped into [1, <see cref="CapacityBlocks"/>], and returns the value applied.</summary>
    int SetEffectiveBudget(int blocks);

    /// <summary>Opens one admission window for a sequence admitted against the effective budget.</summary>
    void BeginAdmissionWindow(int neededBlocks);

    /// <summary>Closes one admission window; the shared ceiling resets once every open window has closed.</summary>
    void EndAdmissionWindow();

    /// <summary>Declares the per-layer attention windows this pool accounts for. Read-side accounting only.</summary>
    void DeclareAttentionWindows(ReadOnlySpan<int> layerWindows);

    /// <summary>Opens the sequence type this pool's storage is addressed by — the float sequence for an FP32 pool, the typed sequence for a quantized one.</summary>
    IKvSequence NewSequence();
}
