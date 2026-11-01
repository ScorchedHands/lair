using System.Buffers;
using Lair.Backends.Cpu;
using Lair.Core.Contracts;

namespace Lair.Tests;

public class ContractSmokeTests
{
    [Fact]
    public void CpuWeightArena_Resolve_ReturnsSeededBytes()
    {
        byte[] image = new byte[64];
        for (int i = 0; i < image.Length; i++)
        {
            image[i] = (byte)(i * 3 + 1);
        }

        LogicalTensor tensor = new(
            "w.test",
            new TensorShape([4, 8]),
            QuantFormat.Q8_0,
            Offset: 16,
            SizeBytes: 32);
        LogicalTensorTable table = new([tensor]);

        using CpuWeightArena arena = new(table, image);

        ReadOnlySpan<byte> resolved = arena.Resolve(tensor);
        Assert.True(resolved.SequenceEqual(image.AsSpan(16, 32)));
        Assert.Same(tensor, table.Find("w.test"));
        Assert.Null(table.Find("missing"));
    }

    [Fact]
    public unsafe void CpuBackend_AllocateAligned_ReturnsAlignedHandle_AndFreeSucceeds()
    {
        using CpuBackend backend = new();
        MemoryHandle handle = backend.AllocateAligned(4096, 64);
        try
        {
            long address = (nint)handle.Pointer;
            Assert.Equal(0L, address % 64);
        }
        finally
        {
            backend.FreeAligned(handle);
        }
    }
}
