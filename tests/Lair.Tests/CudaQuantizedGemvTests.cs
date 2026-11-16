using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// GPU gate for the fused Q4_K GEMV kernel: hand-built Q4_K blocks whose six-bit scale/min codes exceed 15 for groups
/// 4-7 (the high bits split across two header words), whose pair windows carry distinct low and high nibbles, and whose
/// final block or group is only partially filled, run through the CUDA kernel against the packed bytes exactly as a
/// weight tensor would. Device results must match a host-side scalar reference chain within floating-point tolerance,
/// batched launches must match per-batch, and row dequantization must agree with CpuBackend's scalar expansion.
/// </summary>
[Trait("Category", "Cuda")]
public sealed class CudaQuantizedGemvTests
{
    private static float Activation(int c) => (float)Math.Sin(0.11 * c + 1.7) * 1.3f;

    private static CudaBackend OpenCuda(CudaWeightResidency residency = CudaWeightResidency.DeviceCopy)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        try
        {
            return new CudaBackend(residency);
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }
    }

    // Hand-built Q4_K fixture: fixed block halves, six-bit group codes that exceed 15 for groups 4-7, per-group nibble
    // codes, and the matching expanded weights computed with the same scalar chain the kernels mirror.
    private static (byte[] Bytes, float[] Weights) BuildQ4KFixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 255) / 256;
        byte[] bytes = new byte[rows * blocksPerRow * 144];
        float[] weights = new float[rows * cols];

        Half dHalf = (Half)0.25f;
        Half mHalf = (Half)0.125f;
        ushort dBits = (ushort)BitConverter.HalfToInt16Bits(dHalf);
        ushort mBits = (ushort)BitConverter.HalfToInt16Bits(mHalf);
        float d = (float)dHalf;
        float m = (float)mHalf;

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 144;

                bytes[block] = (byte)dBits;
                bytes[block + 1] = (byte)(dBits >> 8);
                bytes[block + 2] = (byte)mBits;
                bytes[block + 3] = (byte)(mBits >> 8);

                byte[] sc = new byte[8];
                byte[] mn = new byte[8];
                for (int j = 0; j < 8; j++)
                {
                    sc[j] = (byte)(5 + 7 * j); // groups 4-7 land at 33..54: their high bits live in header words one and two
                    mn[j] = (byte)(3 + 5 * j); // groups 4-7 land at 23..38 likewise
                }

                for (int k = 0; k < 4; k++)
                {
                    bytes[block + 4 + k] = (byte)((sc[k] & 0x3F) | ((sc[k + 4] >> 4) << 6));
                    bytes[block + 8 + k] = (byte)((mn[k] & 0x3F) | ((mn[k + 4] >> 4) << 6));
                    bytes[block + 12 + k] = (byte)((sc[k + 4] & 0xF) | ((mn[k + 4] & 0xF) << 4));
                }

                for (int j = 0; j < 8; j++)
                {
                    int subStart = j * 32;
                    if (subStart >= cols - b * 256)
                    {
                        break; // padded tail: no real columns in this group
                    }

                    int limit = Math.Min(32, cols - b * 256 - subStart);
                    int window = block + 16 + 32 * (j / 2);
                    for (int l = 0; l < limit; l++)
                    {
                        byte code = (byte)((j * 3 + l) & 0xF);
                        if ((j & 1) == 0)
                        {
                            bytes[window + l] = (byte)(bytes[window + l] | code);
                        }
                        else
                        {
                            bytes[window + l] = (byte)(bytes[window + l] | (code << 4));
                        }

                        weights[r * cols + b * 256 + subStart + l] = code * (d * sc[j]) - m * mn[j];
                    }
                }
            }
        }

        return (bytes, weights);
    }

    private static float[] ExpectedOutput(int rows, int cols, float[] weights, ReadOnlySpan<float> x)
    {
        float[] output = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            float acc = 0f;
            for (int c = 0; c < cols; c++)
            {
                acc += x[c] * weights[r * cols + c];
            }

            output[r] = acc;
        }

        return output;
    }

    private static void AssertClose(float expected, float actual, float tolerance = 1e-3f)
    {
        Assert.InRange(actual, expected - tolerance, expected + tolerance);
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance = 1e-3f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            AssertClose(expected[i], actual[i], tolerance);
        }
    }

    private static float[] Activations(int cols)
    {
        float[] x = new float[cols];
        for (int c = 0; c < cols; c++)
        {
            x[c] = Activation(c);
        }

        return x;
    }

    [Fact]
    public void CudaQ4K_Gemv_ExactBlockMultiple_MatchesHostScalarReference()
    {
        const int rows = 3;
        const int cols = 256;
        (byte[] bytes, float[] weights) = BuildQ4KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);
        float[] x = Activations(cols);
        float[] expected = ExpectedOutput(rows, cols, weights, x);

        using (CudaBackend cuda = OpenCuda())
        {
            float[] actual = new float[rows];
            cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q4_K), actual);
            AssertClose(expected, actual);
        }

        using (CpuBackend cpu = new(threads: 1, allowAvx2: false))
        {
            float[] actual = new float[rows];
            cpu.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q4_K), actual);
            AssertClose(expected, actual);
        }
    }

    [Fact]
    public void CudaQ4K_Gemv_PartialFinalBlock_StopsAtRealColumns()
    {
        const int rows = 2;
        const int cols = 300; // two blocks: the second covers 44 real columns across groups 0 and 1 only
        (byte[] bytes, float[] weights) = BuildQ4KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);
        float[] x = Activations(cols);
        float[] expected = ExpectedOutput(rows, cols, weights, x);

        using CudaBackend cuda = OpenCuda();
        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q4_K), actual);
        AssertClose(expected, actual);
    }

    [Fact]
    public void CudaQ4K_Gemv_PartialGroupTail_StopsAtRealColumns()
    {
        const int rows = 2;
        const int cols = 270; // two blocks: the second fills only the first 14 columns of group 0
        (byte[] bytes, float[] weights) = BuildQ4KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);
        float[] x = Activations(cols);
        float[] expected = ExpectedOutput(rows, cols, weights, x);

        using CudaBackend cuda = OpenCuda();
        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q4_K), actual);
        AssertClose(expected, actual);
    }

    [Fact]
    public void CudaQ4K_GemvBatch_MatchesHostScalarReferencePerBatch()
    {
        const int rows = 2;
        const int cols = 256;
        const int count = 3;
        (byte[] bytes, float[] weights) = BuildQ4KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        float[] inputs = new float[count * cols];
        for (int bIdx = 0; bIdx < count; bIdx++)
        {
            for (int c = 0; c < cols; c++)
            {
                inputs[bIdx * cols + c] = Activation(c + bIdx * 7);
            }
        }

        float[] expected = new float[count * rows];
        for (int bIdx = 0; bIdx < count; bIdx++)
        {
            float[] one = ExpectedOutput(rows, cols, weights, inputs.AsSpan(bIdx * cols, cols));
            for (int r = 0; r < rows; r++)
            {
                expected[bIdx * rows + r] = one[r];
            }
        }

        using CudaBackend cuda = OpenCuda();
        float[] actual = new float[count * rows];
        cuda.GemvBatch(inputs, count, new WeightRef(bytes, shape, QuantFormat.Q4_K), actual);
        AssertClose(expected, actual);
    }

    [Fact]
    public void CudaQ4K_DequantRow_MatchesCpuBackendScalarExpansion()
    {
        const int rows = 2;
        const int cols = 300;
        (byte[] bytes, float[] weights) = BuildQ4KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);
        WeightRef w = new(bytes, shape, QuantFormat.Q4_K);

        using CudaBackend cuda = OpenCuda();
        using CpuBackend cpu = new(threads: 1, allowAvx2: false);

        float[] cudaOut = new float[cols];
        float[] cpuOut = new float[cols];
        for (int r = 0; r < rows; r++)
        {
            cuda.DequantRow(w, r, cudaOut);
            cpu.DequantRow(w, r, cpuOut);
            AssertClose(weights.AsSpan(r * cols, cols).ToArray(), cudaOut);
            AssertClose(cpuOut, cudaOut);
        }
    }

    [Fact]
    public void CudaQ4K_Gemv_HostMappedPosture_RunsFromRegisteredHostBytes()
    {
        const int rows = 3;
        const int cols = 256;
        (byte[] bytes, float[] weights) = BuildQ4KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);
        float[] x = Activations(cols);
        float[] expected = ExpectedOutput(rows, cols, weights, x);

        using CudaBackend cuda = OpenCuda(CudaWeightResidency.HostMapped);
        Assert.Equal(CudaWeightResidency.HostMapped, cuda.WeightResidency);

        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q4_K), actual);
        AssertClose(expected, actual);
    }

    [Fact]
    public void CudaQ4K_Gemv_ColumnWeights_MatchFixtureExpansion()
    {
        // One-hot activations turn each GEMV into a direct read of one expanded weight: every column of the full
        // fixture must match its host-side scalar expansion, so no group can borrow scale, min, or nibble bytes
        // from another group's window.
        const int rows = 1;
        const int cols = 256;
        (byte[] bytes, float[] weights) = BuildQ4KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        using CudaBackend cuda = OpenCuda();
        for (int c = 0; c < cols; c++)
        {
            float[] x = new float[cols];
            x[c] = 1f;
            float[] actual = new float[rows];
            cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q4_K), actual);
            AssertClose(weights[c], actual[0]);
        }
    }

    [Fact]
    public void CudaQ4K_Gemv_LowNibbleGroup_MatchesHandComputedWeights()
    {
        // Minimal block: d = 1, sc0 = 1, mn0 = 0, so group 0's weights are the low nibbles themselves.
        const int cols = 256;
        byte[] bytes = new byte[144];
        ushort dBits = (ushort)BitConverter.HalfToInt16Bits((Half)1f);
        bytes[0] = (byte)dBits;
        bytes[1] = (byte)(dBits >> 8);
        bytes[2] = 0;
        bytes[3] = 0;
        bytes[4] = 1; // sc0 = 1, mn0 = 0
        for (int l = 0; l < 16; l++)
        {
            bytes[16 + l] = (byte)(l & 0xF);
        }

        TensorShape shape = new([1, cols]);
        using CudaBackend cuda = OpenCuda();
        for (int c = 0; c < 16; c++)
        {
            float[] x = new float[cols];
            x[c] = 1f;
            float[] actual = new float[1];
            cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q4_K), actual);
            AssertClose((float)(c & 0xF), actual[0]);
        }
    }

    [Fact]
    public void CudaQ4K_Gemv_HighNibbleGroup_MatchesHandComputedWeights()
    {
        // Same block shape with sc1 = 1: group 1's weights are the high nibbles of the shared pair window.
        const int cols = 256;
        byte[] bytes = new byte[144];
        ushort dBits = (ushort)BitConverter.HalfToInt16Bits((Half)1f);
        bytes[0] = (byte)dBits;
        bytes[1] = (byte)(dBits >> 8);
        bytes[2] = 0;
        bytes[3] = 0;
        bytes[5] = 1; // sc1 = 1, mn1 = 0
        for (int l = 0; l < 16; l++)
        {
            bytes[16 + l] = (byte)((l & 0xF) << 4);
        }

        TensorShape shape = new([1, cols]);
        using CudaBackend cuda = OpenCuda();
        for (int l = 0; l < 16; l++)
        {
            int c = 32 + l;
            float[] x = new float[cols];
            x[c] = 1f;
            float[] actual = new float[1];
            cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q4_K), actual);
            AssertClose((float)(l & 0xF), actual[0]);
        }
    }

    [Fact]
    public void CudaQ4K_Gemv_Group1ScaleMin_MatchesHandComputedWeights()
    {
        // d = 0.25, min = 0.125, sc1 = 12 (byte5 low six bits), mn1 = 8 (byte9 low six bits): group 1 weights are
        // nibble * 3 - 1 while every other group stays zero-scaled.
        const int cols = 256;
        byte[] bytes = new byte[144];
        ushort dBits = (ushort)BitConverter.HalfToInt16Bits((Half)0.25f);
        ushort mBits = (ushort)BitConverter.HalfToInt16Bits((Half)0.125f);
        bytes[0] = (byte)dBits;
        bytes[1] = (byte)(dBits >> 8);
        bytes[2] = (byte)mBits;
        bytes[3] = (byte)(mBits >> 8);
        bytes[5] = 12; // sc1 low bits
        bytes[9] = 8;  // mn1 low bits
        for (int l = 0; l < 16; l++)
        {
            bytes[16 + l] = (byte)((l & 0xF) << 4);
        }

        TensorShape shape = new([1, cols]);
        using CudaBackend cuda = OpenCuda();
        for (int l = 0; l < 16; l++)
        {
            int c = 32 + l;
            float[] x = new float[cols];
            x[c] = 1f;
            float[] actual = new float[1];
            cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q4_K), actual);
            AssertClose((float)(l & 0xF) * 3f - 1f, actual[0]);
        }
    }

    [Fact]
    public void CudaQ4K_Gemv_Group1IgnoresSplitHighHeaderBits()
    {
        // byte5 and byte9 also carry the two high bits of sc5/mn5 in their top positions: group 1's weights must use
        // only its own six-bit codes (sc1 = 12, mn1 = 8) and ignore the split halves belonging to groups 4-7.
        const int cols = 256;
        byte[] bytes = new byte[144];
        ushort dBits = (ushort)BitConverter.HalfToInt16Bits((Half)0.25f);
        ushort mBits = (ushort)BitConverter.HalfToInt16Bits((Half)0.125f);
        bytes[0] = (byte)dBits;
        bytes[1] = (byte)(dBits >> 8);
        bytes[2] = (byte)mBits;
        bytes[3] = (byte)(mBits >> 8);
        bytes[5] = 140; // sc1 low bits 12 plus high bits from sc5
        bytes[9] = 72;  // mn1 low bits 8 plus high bits from mn5
        for (int l = 0; l < 16; l++)
        {
            bytes[16 + l] = (byte)((l & 0xF) << 4);
        }

        TensorShape shape = new([1, cols]);
        using CudaBackend cuda = OpenCuda();
        for (int l = 0; l < 16; l++)
        {
            int c = 32 + l;
            float[] x = new float[cols];
            x[c] = 1f;
            float[] actual = new float[1];
            cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q4_K), actual);
            AssertClose((float)(l & 0xF) * 3f - 1f, actual[0]);
        }
    }

    [Fact]
    public void CudaQ4K_Gemv_Group5ReassemblesSplitHighHeaderBits()
    {
        // sc5 and mn5 are six-bit codes split across three header words: byte13 carries both low nibbles, byte5 bit 6
        // adds the high bits of sc5, and byte9 bit 6 adds the high bits of mn5. With d = 0.25, min = 0.125, sc5 = 21,
        // and mn5 = 21, group 5's weights are nibble * 5.25 - 2.625; the same bit positions must leave group 1's own
        // six-bit codes (masked to bits 0..5) untouched.
        const int cols = 256;
        byte[] bytes = new byte[144];
        ushort dBits = (ushort)BitConverter.HalfToInt16Bits((Half)0.25f);
        ushort mBits = (ushort)BitConverter.HalfToInt16Bits((Half)0.125f);
        bytes[0] = (byte)dBits;
        bytes[1] = (byte)(dBits >> 8);
        bytes[2] = (byte)mBits;
        bytes[3] = (byte)(mBits >> 8);
        bytes[5] = 1 << 6; // high bits of sc5 in byte5 top two bits
        bytes[9] = 1 << 6; // high bits of mn5 in byte9 top two bits
        bytes[13] = 5 | (5 << 4); // low nibbles: sc5 = 21, mn5 = 21 after reassembly
        for (int l = 0; l < 16; l++)
        {
            bytes[80 + l] = (byte)((l & 0xF) << 4); // group 5 nibbles: high half of the pair-4/5 window
            bytes[16 + l] = (byte)((l & 0xF) << 4); // group 1 nibbles live too, but its masked codes stay zero-scaled
        }

        TensorShape shape = new([1, cols]);
        using CudaBackend cuda = OpenCuda();
        for (int l = 0; l < 16; l++)
        {
            int c = 160 + l;
            float[] x = new float[cols];
            x[c] = 1f;
            float[] actual = new float[1];
            cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q4_K), actual);
            AssertClose((float)(l & 0xF) * 5.25f - 2.625f, actual[0]);
        }

        // Group 1 sees byte5/byte9 through its own six-bit mask only: sc1 = mn1 = 0 keeps columns 32..47 at zero.
        for (int l = 0; l < 16; l++)
        {
            int c = 32 + l;
            float[] x = new float[cols];
            x[c] = 1f;
            float[] actual = new float[1];
            cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q4_K), actual);
            AssertClose(0f, actual[0]);
        }
    }
}
