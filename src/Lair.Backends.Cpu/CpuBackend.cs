using System.Buffers;
using System.Runtime.InteropServices;
using Lair.Core.Contracts;

namespace Lair.Backends.Cpu;

public sealed unsafe class CpuBackend : IBackend
{
    public string DeviceId => "cpu";

    public MemoryHandle AllocateAligned(long bytes, int alignment)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0)
        {
            throw new ArgumentException("Alignment must be a positive power of two.", nameof(alignment));
        }

        void* pointer = NativeMemory.AlignedAlloc((nuint)bytes, (nuint)alignment);
        if (pointer is null)
        {
            throw new OutOfMemoryException($"Failed to allocate {bytes} aligned bytes.");
        }

        return new MemoryHandle(pointer);
    }

    public void FreeAligned(MemoryHandle handle)
    {
        NativeMemory.AlignedFree(handle.Pointer);
    }

    public void Dispose()
    {
    }
}
