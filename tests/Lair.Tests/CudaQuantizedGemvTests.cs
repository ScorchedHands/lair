using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// GPU gate for the fused packed GEMV kernels: hand-built Q4_K blocks whose six-bit scale/min codes exceed 15 for groups
/// 4-7 (the high bits split across two header words), whose pair windows carry distinct low and high nibbles, and whose
/// final block or group is only partially filled; and hand-built Q6_K blocks carrying signed int8 group scales including
/// -128, six-bit codes that walk every nibble class across the ql window and its qh 2-bit fields, a partial tail block,
/// and a negative subnormal half-float header whose sign has to survive widening. All run through the CUDA kernel against
/// the packed bytes exactly as a weight tensor would. Device results must match a host-side scalar reference chain within
/// floating-point tolerance, single-column reads must agree byte for byte with that expansion, batched launches must match
/// per-batch, and row dequantization must agree with CpuBackend's scalar expansion.
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

    // Hand-built Q6_K fixture: a half-float block header d, sixteen signed int8 group scales (including -128 and other
    // negative codes), and six-bit quant codes that walk every nibble class — low nibbles for quarters 0..1 of each
    // half, high nibbles for 2..3, with the top two bits spread across the qh window byte at bit shift 2*quarter. The
    // expanded weights are computed with the same scalar chain both backends mirror: scale = d * sc_g, w = (q - 32) * scale.
    private static (byte[] Bytes, float[] Weights) BuildQ6KFixture(int rows, int cols, Half dHalf, sbyte[]? scaleOverride = null)
    {
        int blocksPerRow = (cols + 255) / 256;
        byte[] bytes = new byte[rows * blocksPerRow * 210];
        float[] weights = new float[rows * cols];

        ushort dBits = (ushort)BitConverter.HalfToInt16Bits(dHalf);
        float d = (float)dHalf;

        sbyte[] scalesByGroup = scaleOverride ?? new sbyte[] { 88, 47, 110, -128, 57, 106, -72, -59, 84, 86, -62, 95, 78, -120, -73, 71 };

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 210;

                bytes[block + 208] = (byte)dBits;
                bytes[block + 209] = (byte)(dBits >> 8);

                for (int g = 0; g < 16; g++)
                {
                    int subStart = g * 16;
                    if (subStart >= cols - b * 256)
                    {
                        break; // padded tail: no real columns in this group
                    }

                    bytes[block + 192 + g] = (byte)scalesByGroup[g];

                    int h = g >> 3; // 128-element half of the super-block
                    int k = g & 7; // 16-element group inside the half
                    int quarter = k >> 1; // which 32-element quarter of the half this group starts in
                    int limit = Math.Min(16, cols - b * 256 - subStart);

                    for (int e = 0; e < limit; e++)
                    {
                        int six = ((g * 4 + e) & 0x3F) ^ (((r + b) << 3) & 0x3F); // distinct codes across every nibble class

                        ref byte qbyte = ref bytes[block + 64 * h + 16 * (k & 3) + e];
                        if ((quarter >> 1) == 0)
                        {
                            qbyte = (byte)((qbyte & 0xF0) | (six & 0xF));
                        }
                        else
                        {
                            qbyte = (byte)((qbyte & 0x0F) | (six << 4));
                        }

                        bytes[block + 128 + 32 * h + 16 * (k & 1) + e] |= (byte)((six >> 4) << (2 * quarter));

                        weights[r * cols + b * 256 + subStart + e] = (six - 32) * (d * scalesByGroup[g]);
                    }
                }
            }
        }

        return (bytes, weights);
    }

    // Cross-device dot identity is a band, not raw bytes: the device chain and the host scalar chain agree on every
    // weight value exactly, so their sums stay within a tight relative distance of each other.
    private static void AssertCloseBand(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (int i = 0; i < expected.Length; i++)
        {
            float scale = Math.Max(Math.Abs(expected[i]), Math.Abs(actual[i]));
            Assert.True(
                scale == 0f || Math.Abs(expected[i] - actual[i]) <= 1e-3f * scale,
                $"{context}: output[{i}] differs: expected {expected[i]:G9}, got {actual[i]:G9}");
        }
    }

    // A single expanded weight must match across the seam exactly, with one exception: -0 and +0 are the same weight
    // value, and the host scalar chain and the device chain differ only in the sign of a zero product.
    private static void AssertSameWeight(float expected, float actual, string context)
    {
        if (expected == 0f && actual == 0f)
        {
            return;
        }

        Assert.True(expected == actual, $"{context}: expected {expected:G9}, got {actual:G9}");
    }

    [Fact]
    public void CudaQ6K_Gemv_ExactBlockMultiple_MatchesHostScalarReference()
    {
        const int rows = 3;
        const int cols = 256;
        (byte[] bytes, float[] weights) = BuildQ6KFixture(rows, cols, (Half)0.25f);
        TensorShape shape = new([rows, cols]);
        float[] x = Activations(cols);
        float[] expected = ExpectedOutput(rows, cols, weights, x);

        using CudaBackend cuda = OpenCuda();
        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q6_K), actual);
        AssertCloseBand(expected, actual, "cuda Q6_K exact block multiple");
    }

    [Fact]
    public void CudaQ6K_Gemv_PartialFinalBlock_StopsAtRealColumns()
    {
        const int rows = 2;
        const int cols = 300; // two blocks: the second fills groups 0..2 only, with group 2 partially filled
        (byte[] bytes, float[] weights) = BuildQ6KFixture(rows, cols, (Half)(-0.25f));
        TensorShape shape = new([rows, cols]);
        float[] x = Activations(cols);
        float[] expected = ExpectedOutput(rows, cols, weights, x);

        using CudaBackend cuda = OpenCuda();
        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q6_K), actual);
        AssertCloseBand(expected, actual, "cuda Q6_K partial final block");
    }

    [Fact]
    public void CudaQ6K_GemvBatch_MatchesHostScalarReferencePerBatch()
    {
        const int rows = 2;
        const int cols = 512; // a multi-block row: the batch loop must re-walk every block for each activation vector
        const int count = 3;
        (byte[] bytes, float[] weights) = BuildQ6KFixture(rows, cols, (Half)(-0.25f));
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
        cuda.GemvBatch(inputs, count, new WeightRef(bytes, shape, QuantFormat.Q6_K), actual);
        AssertCloseBand(expected, actual, "cuda Q6_K batched launch");
    }

    [Fact]
    public void CudaQ6K_Gemv_ColumnWeights_MatchFixtureExpansion()
    {
        // One-hot activations read one expanded weight per GEMV: every column of a full sixteen-group row must carry its
        // own group's signed scale and its own nibble window, so no group can borrow bytes from another.
        const int rows = 1;
        const int cols = 256;
        (byte[] bytes, float[] weights) = BuildQ6KFixture(rows, cols, (Half)(-0.25f));
        TensorShape shape = new([rows, cols]);

        using CudaBackend cuda = OpenCuda();
        for (int c = 0; c < cols; c++)
        {
            float[] x = new float[cols];
            x[c] = 1f;
            float[] actual = new float[rows];
            cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q6_K), actual);
            AssertSameWeight(weights[c], actual[0], $"cuda Q6_K column {c}");
        }
    }

    [Fact]
    public void CudaQ6K_Gemv_NegativeSubnormalHeader_ScalesEveryGroup()
    {
        // A block whose half-float d is negative and subnormal (below the half normal range): its sign has to survive
        // the widening chain, or every group in the block reads as zero instead of scaled weights.
        const int cols = 256;
        Half dHalf = (Half)(-6.2e-06f);
        ushort dBits = (ushort)BitConverter.HalfToInt16Bits(dHalf);

        byte[] bytes = new byte[210];
        bytes[208] = (byte)dBits;
        bytes[209] = (byte)(dBits >> 8);

        sbyte sc = -37; // one negative group scale, so the header sign cannot hide inside a second negation
        for (int g = 0; g < 16; g++)
        {
            bytes[192 + g] = (byte)sc;
        }

        int six = 45; // q - 32 = 13: low nibble 13 in the window's low half, high bits 3 shifted into its own quarter field
        for (int g = 0; g < 16; g++)
        {
            int h = g >> 3;
            int k = g & 7;
            int quarter = k >> 1;

            ref byte qbyte = ref bytes[64 * h + 16 * (k & 3)];
            if ((quarter >> 1) == 0)
            {
                qbyte = (byte)((qbyte & 0xF0) | (six & 0xF));
            }
            else
            {
                qbyte = (byte)((qbyte & 0x0F) | (six << 4));
            }

            bytes[128 + 32 * h + 16 * (k & 1)] |= (byte)((six >> 4) << (2 * quarter));
        }

        TensorShape shape = new([1, cols]);
        using CudaBackend cuda = OpenCuda();
        // The crafted code sits at element 0 of every group, so the probes walk each group's first column: a header sign
        // lost in widening would show up as a zero or wrongly signed weight in every group at once.
        foreach (int c in new[] { 0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240 })
        {
            float[] x = new float[cols];
            x[c] = 1f;
            float[] actual = new float[1];
            cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q6_K), actual);

            Assert.True(actual[0] > 0f, $"cuda Q6_K subnormal header column {c} read {actual[0]}: expected a positive scaled weight");
            AssertCloseBand([(float)(six - 32) * ((float)dHalf * sc)], actual, $"cuda Q6_K subnormal header column {c}");
        }
    }

    [Fact]
    public void CudaQ6K_DequantRow_MatchesCpuBackendScalarExpansion()
    {
        const int rows = 2;
        const int cols = 300; // a partial tail block: padded quants stay unread in both expansions
        (byte[] bytes, float[] weights) = BuildQ6KFixture(rows, cols, (Half)(-0.25f));
        TensorShape shape = new([rows, cols]);
        WeightRef w = new(bytes, shape, QuantFormat.Q6_K);

        using CudaBackend cuda = OpenCuda();
        using CpuBackend cpu = new(threads: 1, allowAvx2: false);

        float[] cudaOut = new float[cols];
        float[] cpuOut = new float[cols];
        for (int r = 0; r < rows; r++)
        {
            cuda.DequantRow(w, r, cudaOut);
            cpu.DequantRow(w, r, cpuOut);

            // Row expansion carries no accumulation chain: the device answer is the host scalar math byte for byte.
            BatchKernelTests.AssertBitsEqual(weights.AsSpan(r * cols, cols).ToArray(), cudaOut, $"cuda Q6_K row {r} vs fixture expansion");
            BatchKernelTests.AssertBitsEqual(cpuOut, cudaOut, $"cuda Q6_K row {r} vs CpuBackend scalar expansion");
        }
    }

    [Fact]
    public void CudaQ6K_Gemv_HostMappedPosture_RunsFromRegisteredHostBytes()
    {
        const int rows = 3;
        const int cols = 256;
        (byte[] bytes, float[] weights) = BuildQ6KFixture(rows, cols, (Half)(-0.25f));
        TensorShape shape = new([rows, cols]);
        float[] x = Activations(cols);
        float[] expected = ExpectedOutput(rows, cols, weights, x);

        using CudaBackend cuda = OpenCuda(CudaWeightResidency.HostMapped);
        Assert.Equal(CudaWeightResidency.HostMapped, cuda.WeightResidency);

        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q6_K), actual);
        AssertCloseBand(expected, actual, "cuda Q6_K host-mapped posture");
    }

    [Fact]
    public void CudaQ6K_Gemv_GroupScaleAndNibbleWindowsDoNotCrossBorrow()
    {
        // One block whose only non-zero group scale belongs to group 5: that group's columns carry (q - 32) * sc while
        // every other column stays zero, so a nibble window or qh field cannot leak into a neighbouring group.
        const int cols = 256;
        sbyte[] onlyGroup5 = new sbyte[16];
        onlyGroup5[5] = -4;

        (byte[] bytes, float[] weights) = BuildQ6KFixture(1, cols, (Half)1f, onlyGroup5);
        TensorShape shape = new([1, cols]);
        WeightRef w = new(bytes, shape, QuantFormat.Q6_K);

        using CudaBackend cuda = OpenCuda();
        using CpuBackend cpu = new(threads: 1, allowAvx2: false);

        float[] hostRow = new float[cols];
        cpu.DequantRow(w, 0, hostRow);

        int groupStart = 5 * 16; // the run this scale governs: columns [80, 96)
        Assert.Contains(hostRow.AsSpan(groupStart, 16).ToArray(), v => v != 0f); // the fixture keeps some group 5 codes at q = 32, so a lone non-zero column is enough

        foreach (int c in Enumerable.Range(groupStart, 16).Append(0).Append(15).Append(32).Append(47).Append(96).Append(111))
        {
            float[] x = new float[cols];
            x[c] = 1f;
            float[] actual = new float[1];
            cuda.Gemv(x, w, actual);

            Assert.Equal(weights[c], hostRow[c]); // the fixture expansion and the CPU scalar chain agree on this column
            AssertSameWeight(hostRow[c], actual[0], $"cuda Q6_K column {c} against a lone group scale");
        }
    }
}
