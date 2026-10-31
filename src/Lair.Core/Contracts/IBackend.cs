using System.Buffers;

namespace Lair.Core.Contracts;

/// <summary>
/// Compute backend. Currently covers arena management only: aligned allocation and
/// release of native memory regions. Kernel operations are added by backend kernel
/// implementations.
/// </summary>
public interface IBackend : IDisposable
{
    string DeviceId { get; }

    MemoryHandle AllocateAligned(long bytes, int alignment);

    void FreeAligned(MemoryHandle handle);
}
