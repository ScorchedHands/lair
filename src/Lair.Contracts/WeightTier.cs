namespace Lair.Contracts;

/// <summary>
/// Where a resource is resident while compute reads it. System memory is what every engine path holds today; an
/// accelerator tier names the residency a compute backend reports once its kernels read weights from device memory it
/// owns rather than from host memory the weight source owns.
/// </summary>
public enum WeightTier
{
    /// <summary>Resident in system memory: where weights, KV blocks, and activation buffers live for every model served today.</summary>
    Dram,

    /// <summary>Resident on an accelerator's own memory. A backend reports this only when its kernels read weights from persistent device memory it owns; a segment at this tier refuses before any engine allocates against a backend that reports the other posture.</summary>
    Vram,
}
