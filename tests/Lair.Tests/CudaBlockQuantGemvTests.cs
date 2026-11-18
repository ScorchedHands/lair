using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// GPU gate for the packed block-quant GEMV kernels LAIR now carries on the CUDA seam: hand-built blocks per format whose
/// scales, mins, split header bits, signed quant codes, shared window bytes, fifth-bit planes, and high-bit mask fields walk
/// every unpacking class a real file produces — including negative and subnormal half-float headers, scale codes above 15 that
/// reassemble from two header words, one byte carrying four successive groups of the same quarter, a partial tail block whose
/// real columns stop inside a group, and zero-scaled groups next to non-zero ones. Every case runs the CUDA kernel against those
/// packed bytes exactly as a weight tensor would and compares against CpuBackend's scalar oracle over the same WeightRef: device
/// output must land inside that format's recorded band (measured on this box, asserted per element), batched launches must match
/// both the CPU oracle and the same kernel run solo for every activation vector in the batch, host-mapped weight postures must
/// answer from registered host bytes, and row dequantization — host-side scalar math with no accumulation chain — must agree bit
/// for bit with CpuBackend's expansion. Formats without a kernel here keep refusing rather than running through another device; the blocked-matmul
/// family now runs on this seam too — each packed quant format walks its own GEMV unpacking once per activation-row window instead of once per vector,
/// inside that format's recorded band — while the unquantized F32/F16/BF16 dense matmul keeps its own bit-for-bit posture against the scalar oracle in
/// <see cref="CudaDenseGemmTests"/>, and only the Q8_K i-quants refuse a dense matrix naming this backend.
/// </summary>
[Trait("Category", "Kernel")]
[Trait("Category", "Cuda")]
public sealed class CudaBlockQuantGemvTests
{
    // Recorded per-format bands: relative ceilings on the drift of the CUDA GEMV chain against the CPU scalar oracle, in the
    // same shape as the existing cross-device dot assertion — |expected - actual| <= band * max(|expected|, |actual|). Each is a
    // round-up of the widest drift measured on this box (RTX 4090, driver 610.88) over every case below: Q8_0 7.96e-7,
    // Q5_0 8.16e-7, Q2_0 7.41e-7, Q5_K 2.72e-6, Q2_K 4.22e-7, Q3_K 1.01e-5 (measured on the batched launch, where cancellation
    // in a longer chain shows up first). The ceilings leave headroom for a different device or driver JIT while staying far
    // tighter than any quality-relevant difference. These are CUDA-only bands: the CPU oracle's own scalar/AVX2/VNNI agreement
    // stays bit-identical and no dense-family band moves here.
    private const float BandQ8_0 = 1e-6f;
    private const float BandQ5_0 = 1e-6f;
    private const float BandQ2_0 = 1e-6f;
    private const float BandQ5K = 4e-6f;
    private const float BandQ2K = 1e-6f;
    private const float BandQ3K = 2e-5f;

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

    private static CpuBackend OpenCpuOracle() => new(threads: 1, allowAvx2: false);

    // The oracle every band is measured against: CpuBackend's scalar chain over the same packed bytes and activations.
    private static float[] CpuGemv(CpuBackend cpu, byte[] bytes, TensorShape shape, QuantFormat format, ReadOnlySpan<float> x)
    {
        int rows = shape.Dims[0];
        int cols = shape.Dims[^1];
        float[] output = new float[rows];
        cpu.Gemv(x, new WeightRef(bytes, shape, format), output);
        return output;
    }

    private static void AssertWithinBand(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float band, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        // Relative drift per element: the device chain and the oracle chain agree on every weight value, so what separates them
        // is summation rounding that grows with the size of the sum. The failure message names both this element's drift and the
        // widest drift in the case, so a run that misses its recorded band reports how far the device chain actually moved.
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]), $"{context}: output[{i}] is NaN against expected {expected[i]:G9}");

            float distance = Math.Abs(actual[i] - expected[i]);
            float scale = Math.Max(Math.Abs(expected[i]), Math.Abs(actual[i]));
            float relative = scale == 0f ? distance : distance / scale;
            if (relative > worst) worst = relative;
        }

        for (int i = 0; i < expected.Length; i++)
        {
            float distance = Math.Abs(actual[i] - expected[i]);
            float scale = Math.Max(Math.Abs(expected[i]), Math.Abs(actual[i]));
            bool accepted = scale == 0f ? distance == 0f : distance <= band * scale;

            Assert.True(
                accepted,
                $"{context}: output[{i}] drifts {(scale == 0f ? distance : distance / scale):G9} relative ({distance:G9} absolute) past the recorded band {band:G9}; this case measured a maximum of {worst:G9} (expected {expected[i]:G9}, got {actual[i]:G9})");
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

    // Half-float headers chosen per block so a run covers the normal range, a negative scale, and a subnormal magnitude whose
    // sign has to survive widening on both sides of the seam.
    private static readonly Half[] HeaderHalves = [(Half)0.25f, (Half)(-7.5e-06f), (Half)3.5f];

    private static void WriteHeaderLE(Span<byte> dst, int offset, Half value)
    {
        ushort bits = (ushort)BitConverter.HalfToInt16Bits(value);
        dst[offset] = (byte)bits;
        dst[offset + 1] = (byte)(bits >> 8);
    }

    // ---------------------------------------------------------------- Q8_0 fixtures: half d then 32 int8 quants per block.

    private static byte[] BuildQ8_0Fixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 31) / 32;
        byte[] bytes = new byte[rows * blocksPerRow * 34];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 34;
                WriteHeaderLE(bytes, block, HeaderHalves[(r + b) % HeaderHalves.Length]);

                int limit = Math.Min(32, cols - b * 32); // padded quants stay zero: the real column count bounds them.
                for (int l = 0; l < limit; l++)
                {
                    sbyte q = (sbyte)(((r * 17 + b * 5 + l * 41) % 256) - 128); // walks the whole int8 range including both extremes
                    bytes[block + 2 + l] = (byte)q;
                }
            }
        }

        return bytes;
    }

    [Fact]
    public void CudaQ8_0_Gemv_MatchesCpuOracleWithinBand()
    {
        const int rows = 3;
        const int cols = 64; // two whole blocks per row
        byte[] bytes = BuildQ8_0Fixture(rows, cols);
        TensorShape shape = new([rows, cols]);
        QuantFormat format = QuantFormat.Q8_0;

        float[] x = Activations(cols);
        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = CpuGemv(cpu, bytes, shape, format, x);

        using CudaBackend cuda = OpenCuda();
        Assert.True(cuda.SupportsGemv(format), "the CUDA seam stopped reporting its Q8_0 GEMV kernel");

        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, format), actual);
        AssertWithinBand(expected, actual, BandQ8_0, "cuda Q8_0 exact block multiple");
    }

    [Fact]
    public void CudaQ8_0_Gemv_PartialFinalBlock_StopsAtRealColumns()
    {
        const int rows = 2;
        const int cols = 36; // two blocks: the second carries only four real columns
        byte[] bytes = BuildQ8_0Fixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        float[] x = Activations(cols);
        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = CpuGemv(cpu, bytes, shape, QuantFormat.Q8_0, x);

        using CudaBackend cuda = OpenCuda();
        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q8_0), actual);
        AssertWithinBand(expected, actual, BandQ8_0, "cuda Q8_0 partial final block");
    }

    [Fact]
    public void CudaQ8_0_GemvBatch_MatchesCpuOracleAndSoloLaunch()
    {
        const int rows = 2;
        const int cols = 96; // three blocks per row: the batch loop re-walks every block for each activation vector
        const int count = 3;
        byte[] bytes = BuildQ8_0Fixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        AssertBatchMatchesSoloAndOracle(bytes, shape, QuantFormat.Q8_0, rows, cols, count, BandQ8_0);
    }

    [Fact]
    public void CudaQ8_0_DequantRow_MatchesCpuBackendScalarExpansion()
    {
        AssertDequantRowBitsEqual(2, 36, BuildQ8_0Fixture(2, 36), QuantFormat.Q8_0);
    }

    [Fact]
    public void CudaQ8_0_Gemv_HostMappedPosture_RunsFromRegisteredHostBytes() =>
        AssertHostMappedMatchesOracle(BuildQ8_0Fixture(3, 64), new([3, 64]), QuantFormat.Q8_0, BandQ8_0);

    // ---------------------------------------------------------------- Q5_0 fixtures: half d, a 32-bit fifth-bit word, then nibbles.

    private static byte[] BuildQ5_0Fixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 31) / 32;
        byte[] bytes = new byte[rows * blocksPerRow * 22];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 22;
                WriteHeaderLE(bytes, block, HeaderHalves[(r * 2 + b) % HeaderHalves.Length]);

                int limit = Math.Min(32, cols - b * 32); // padded quants stay zero: the real column count bounds them.
                for (int l = 0; l < limit; l++)
                {
                    byte nibble = (byte)((r * 3 + b * 7 + l * 11) & 0xF);
                    if (l < 16)
                    {
                        bytes[block + 6 + l] |= nibble; // low nibble of the pair byte for elements 0..15 ...
                    }
                    else
                    {
                        bytes[block + 6 + (l - 16)] |= (byte)(nibble << 4); // ... high nibble of the same byte for 16..31
                    }

                    int fifthBit = (r * 5 + b * 3 + l * 7) & 1; // both states across every element position
                    bytes[block + 2 + (l >> 3)] |= (byte)(fifthBit << (l & 7)); // little-endian word: element l keeps bit l
                }
            }
        }

        return bytes;
    }

    [Fact]
    public void CudaQ5_0_Gemv_MatchesCpuOracleWithinBand()
    {
        const int rows = 3;
        const int cols = 64; // two whole blocks per row: nibbles, fifth bits, and centering all walk both halves
        byte[] bytes = BuildQ5_0Fixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        float[] x = Activations(cols);
        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = CpuGemv(cpu, bytes, shape, QuantFormat.Q5_0, x);

        using CudaBackend cuda = OpenCuda();
        Assert.True(cuda.SupportsGemv(QuantFormat.Q5_0), "the CUDA seam stopped reporting its Q5_0 GEMV kernel");

        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q5_0), actual);
        AssertWithinBand(expected, actual, BandQ5_0, "cuda Q5_0 exact block multiple");
    }

    [Fact]
    public void CudaQ5_0_Gemv_PartialFinalBlock_StopsAtRealColumns()
    {
        const int rows = 2;
        const int cols = 40; // two blocks: the second fills its low-nibble arm only, and partially
        byte[] bytes = BuildQ5_0Fixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        float[] x = Activations(cols);
        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = CpuGemv(cpu, bytes, shape, QuantFormat.Q5_0, x);

        using CudaBackend cuda = OpenCuda();
        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q5_0), actual);
        AssertWithinBand(expected, actual, BandQ5_0, "cuda Q5_0 partial final block");
    }

    [Fact]
    public void CudaQ5_0_GemvBatch_MatchesCpuOracleAndSoloLaunch()
    {
        const int rows = 2;
        const int cols = 64;
        const int count = 3;
        byte[] bytes = BuildQ5_0Fixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        AssertBatchMatchesSoloAndOracle(bytes, shape, QuantFormat.Q5_0, rows, cols, count, BandQ5_0);
    }

    [Fact]
    public void CudaQ5_0_DequantRow_MatchesCpuBackendScalarExpansion() =>
        AssertDequantRowBitsEqual(2, 40, BuildQ5_0Fixture(2, 40), QuantFormat.Q5_0);

    [Fact]
    public void CudaQ5_0_Gemv_HostMappedPosture_RunsFromRegisteredHostBytes() =>
        AssertHostMappedMatchesOracle(BuildQ5_0Fixture(3, 64), new([3, 64]), QuantFormat.Q5_0, BandQ5_0);

    // ---------------------------------------------------------------- Q2_0 fixtures: half d then sixteen bytes of four codes each.

    private static byte[] BuildQ2_0Fixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 63) / 64;
        byte[] bytes = new byte[rows * blocksPerRow * 18];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 18;
                WriteHeaderLE(bytes, block, HeaderHalves[(r + b * 3) % HeaderHalves.Length]);

                int limit = Math.Min(64, cols - b * 64); // padded quants stay zero: the real column count bounds them.
                for (int l = 0; l < limit; l++)
                {
                    byte two = (byte)((r * 3 + b * 5 + l) & 3); // every shift class of the shared window byte appears per block
                    bytes[block + 2 + (l >> 2)] |= (byte)(two << (2 * (l & 3)));
                }
            }
        }

        return bytes;
    }

    [Fact]
    public void CudaQ2_0_Gemv_MatchesCpuOracleWithinBand()
    {
        const int rows = 3;
        const int cols = 128; // two whole blocks per row across every bit-shift class
        byte[] bytes = BuildQ2_0Fixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        float[] x = Activations(cols);
        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = CpuGemv(cpu, bytes, shape, QuantFormat.Q2_0, x);

        using CudaBackend cuda = OpenCuda();
        Assert.True(cuda.SupportsGemv(QuantFormat.Q2_0), "the CUDA seam stopped reporting its Q2_0 GEMV kernel");

        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q2_0), actual);
        AssertWithinBand(expected, actual, BandQ2_0, "cuda Q2_0 exact block multiple");
    }

    [Fact]
    public void CudaQ2_0_Gemv_PartialFinalBlock_StopsAtRealColumns()
    {
        const int rows = 2;
        const int cols = 84; // two blocks: the second carries twenty real columns, stopping inside its third window byte
        byte[] bytes = BuildQ2_0Fixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        float[] x = Activations(cols);
        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = CpuGemv(cpu, bytes, shape, QuantFormat.Q2_0, x);

        using CudaBackend cuda = OpenCuda();
        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q2_0), actual);
        AssertWithinBand(expected, actual, BandQ2_0, "cuda Q2_0 partial final block");
    }

    [Fact]
    public void CudaQ2_0_GemvBatch_MatchesCpuOracleAndSoloLaunch()
    {
        const int rows = 2;
        const int cols = 128;
        const int count = 3;
        byte[] bytes = BuildQ2_0Fixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        AssertBatchMatchesSoloAndOracle(bytes, shape, QuantFormat.Q2_0, rows, cols, count, BandQ2_0);
    }

    [Fact]
    public void CudaQ2_0_DequantRow_MatchesCpuBackendScalarExpansion() =>
        AssertDequantRowBitsEqual(2, 84, BuildQ2_0Fixture(2, 84), QuantFormat.Q2_0);

    [Fact]
    public void CudaQ2_0_Gemv_HostMappedPosture_RunsFromRegisteredHostBytes() =>
        AssertHostMappedMatchesOracle(BuildQ2_0Fixture(3, 128), new([3, 128]), QuantFormat.Q2_0, BandQ2_0);

    // ---------------------------------------------------------------- Q5_K fixtures: Q4_K scale layout plus a fixed fifth-bit plane.

    private static byte[] BuildQ5KFixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 255) / 256;
        byte[] bytes = new byte[rows * blocksPerRow * 176];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 176;
                WriteHeaderLE(bytes, block, HeaderHalves[(r + b) % HeaderHalves.Length]); // d at bytes 0..1 ...
                WriteHeaderLE(bytes, block + 2, (Half)(-0.5f)); // ... and min at bytes 2..3

                byte[] sc = new byte[8];
                byte[] mn = new byte[8];
                for (int j = 0; j < 8; j++)
                {
                    sc[j] = (byte)(6 + 7 * j); // groups 4-7 land at 34..55: their high bits split across two header words
                    mn[j] = (byte)((j == 2) ? 0 : (1 + 6 * j)); // one zero-scaled group sits beside non-zero ones
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
                    for (int l = 0; l < limit; l++)
                    {
                        byte nibble = (byte)((r * 3 + j * 7 + l * 5) & 0xF); // distinct codes across both halves of a pair window
                        int windowByte = block + 48 + 32 * (j / 2) + l;
                        if ((j & 1) == 0)
                        {
                            bytes[windowByte] |= nibble;
                        }
                        else
                        {
                            bytes[windowByte] |= (byte)(nibble << 4);
                        }

                        int fifthBit = (r * 5 + j * 3 + l * 11) & 1; // one plane byte carries eight different groups' bits
                        bytes[block + 16 + l] |= (byte)(fifthBit << j);
                    }
                }
            }
        }

        return bytes;
    }

    [Fact]
    public void CudaQ5K_Gemv_MatchesCpuOracleWithinBand()
    {
        const int rows = 3;
        const int cols = 256; // one whole super-block per row: eight groups, split scale bits, and the shared fifth-bit plane
        byte[] bytes = BuildQ5KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        float[] x = Activations(cols);
        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = CpuGemv(cpu, bytes, shape, QuantFormat.Q5_K, x);

        using CudaBackend cuda = OpenCuda();
        Assert.True(cuda.SupportsGemv(QuantFormat.Q5_K), "the CUDA seam stopped reporting its Q5_K GEMV kernel");

        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q5_K), actual);
        AssertWithinBand(expected, actual, BandQ5K, "cuda Q5_K exact block multiple");
    }

    [Fact]
    public void CudaQ5K_Gemv_PartialFinalBlock_StopsAtRealColumns()
    {
        const int rows = 2;
        const int cols = 300; // two super-blocks: the second fills groups 0 and 1 only, with group 1 partially filled
        byte[] bytes = BuildQ5KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        float[] x = Activations(cols);
        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = CpuGemv(cpu, bytes, shape, QuantFormat.Q5_K, x);

        using CudaBackend cuda = OpenCuda();
        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q5_K), actual);
        AssertWithinBand(expected, actual, BandQ5K, "cuda Q5_K partial final block");
    }

    [Fact]
    public void CudaQ5K_GemvBatch_MatchesCpuOracleAndSoloLaunch()
    {
        const int rows = 2;
        const int cols = 512; // a multi-block row: the batch loop must re-walk every super-block for each activation vector
        const int count = 3;
        byte[] bytes = BuildQ5KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        AssertBatchMatchesSoloAndOracle(bytes, shape, QuantFormat.Q5_K, rows, cols, count, BandQ5K);
    }

    [Fact]
    public void CudaQ5K_DequantRow_MatchesCpuBackendScalarExpansion() =>
        AssertDequantRowBitsEqual(2, 300, BuildQ5KFixture(2, 300), QuantFormat.Q5_K);

    [Fact]
    public void CudaQ5K_Gemv_HostMappedPosture_RunsFromRegisteredHostBytes() =>
        AssertHostMappedMatchesOracle(BuildQ5KFixture(3, 256), new([3, 256]), QuantFormat.Q5_K, BandQ5K);

    // ---------------------------------------------------------------- Q2_K fixtures: packed scale/min nibbles plus four-group window bytes.

    private static byte[] BuildQ2KFixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 255) / 256;
        byte[] bytes = new byte[rows * blocksPerRow * 84];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 84;

                byte[] scBytes = new byte[16];
                for (int g = 0; g < 16; g++)
                {
                    byte scaleNibble = (byte)((g == 3 || g == 12) ? 0 : ((r * 5 + b * 3 + g * 7) & 0xF)); // zero-scaled groups beside non-zero ones
                    byte minNibble = (byte)((b * 9 + g * 5 + r) & 0xF);
                    scBytes[g] = (byte)(scaleNibble | (minNibble << 4));
                    bytes[block + g] = scBytes[g]; // one byte carries both this group's four-bit scale and its four-bit min
                }

                WriteHeaderLE(bytes, block + 80, HeaderHalves[(r * 2 + b) % HeaderHalves.Length]); // d at bytes 80..81 ...
                WriteHeaderLE(bytes, block + 82, (Half)(-3.5f)); // ... and min at bytes 82..83

                for (int g = 0; g < 16; g++)
                {
                    int h = g >> 3;
                    int k = g & 7;

                    int subStart = g * 16;
                    if (subStart >= cols - b * 256)
                    {
                        break; // padded tail: no real columns in this group
                    }

                    int limit = Math.Min(16, cols - b * 256 - subStart);
                    for (int e = 0; e < limit; e++)
                    {
                        byte two = (byte)((r * 3 + g * 7 + e * 5) & 3); // every shift class of the shared window byte appears per quarter
                        bytes[block + 16 + 32 * h + 16 * (k & 1) + e] |= (byte)(two << (2 * (k >> 1)));
                    }
                }
            }
        }

        return bytes;
    }

    [Fact]
    public void CudaQ2K_Gemv_MatchesCpuOracleWithinBand()
    {
        const int rows = 3;
        const int cols = 256; // one whole super-block per row: sixteen groups over four shared window bytes each
        byte[] bytes = BuildQ2KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        float[] x = Activations(cols);
        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = CpuGemv(cpu, bytes, shape, QuantFormat.Q2_K, x);

        using CudaBackend cuda = OpenCuda();
        Assert.True(cuda.SupportsGemv(QuantFormat.Q2_K), "the CUDA seam stopped reporting its Q2_K GEMV kernel");

        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q2_K), actual);
        AssertWithinBand(expected, actual, BandQ2K, "cuda Q2_K exact block multiple");
    }

    [Fact]
    public void CudaQ2K_Gemv_PartialFinalBlock_StopsAtRealColumns()
    {
        const int rows = 2;
        const int cols = 300; // two super-blocks: the second fills groups 0..1 only, with group 1 partially filled
        byte[] bytes = BuildQ2KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        float[] x = Activations(cols);
        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = CpuGemv(cpu, bytes, shape, QuantFormat.Q2_K, x);

        using CudaBackend cuda = OpenCuda();
        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q2_K), actual);
        AssertWithinBand(expected, actual, BandQ2K, "cuda Q2_K partial final block");
    }

    [Fact]
    public void CudaQ2K_GemvBatch_MatchesCpuOracleAndSoloLaunch()
    {
        const int rows = 2;
        const int cols = 512; // a multi-block row: the batch loop must re-walk every super-block for each activation vector
        const int count = 3;
        byte[] bytes = BuildQ2KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        AssertBatchMatchesSoloAndOracle(bytes, shape, QuantFormat.Q2_K, rows, cols, count, BandQ2K);
    }

    [Fact]
    public void CudaQ2K_DequantRow_MatchesCpuBackendScalarExpansion() =>
        AssertDequantRowBitsEqual(2, 300, BuildQ2KFixture(2, 300), QuantFormat.Q2_K);

    [Fact]
    public void CudaQ2K_Gemv_HostMappedPosture_RunsFromRegisteredHostBytes() =>
        AssertHostMappedMatchesOracle(BuildQ2KFixture(3, 256), new([3, 256]), QuantFormat.Q2_K, BandQ2K);

    // ---------------------------------------------------------------- Q3_K fixtures: packed six-bit scales plus per-element high bits.

    private static byte[] BuildQ3KFixture(int rows, int cols)
    {
        int blocksPerRow = (cols + 255) / 256;
        byte[] bytes = new byte[rows * blocksPerRow * 110];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * 110;

                byte[] scales = new byte[12]; // twelve bytes pack sixteen six-bit group scales at bytes 96..107
                for (int g = 0; g < 16; g++)
                {
                    int code = (r * 5 + b * 3 + g * 4) & 0x3F; // walks below, at, and above the centering point of 32
                    byte low = (byte)(code & 0xF);
                    byte high = (byte)((code >> 4) & 0x3);

                    if (g < 8)
                    {
                        scales[g] |= low; // groups 0..7 keep their scale's low nibble in scales[g] ...
                    }
                    else
                    {
                        scales[g - 8] |= (byte)(low << 4); // ... groups 8..15 take the high nibble of scales[g-8]
                    }

                    scales[8 + (g & 3)] |= (byte)(high << (2 * (g >> 2))); // every scale's four high bits share one byte at a constant shift
                }

                for (int i = 0; i < 12; i++)
                {
                    bytes[block + 96 + i] = scales[i];
                }

                WriteHeaderLE(bytes, block + 108, HeaderHalves[(r + b * 3) % HeaderHalves.Length]); // d at bytes 108..109

                for (int g = 0; g < 16; g++)
                {
                    int h = g >> 3;
                    int k = g & 7;
                    int m = 4 * h + (k >> 1); // this group's bit position inside every element of its quarter's hmask byte

                    int subStart = g * 16;
                    if (subStart >= cols - b * 256)
                    {
                        break; // padded tail: no real columns in this group
                    }

                    int limit = Math.Min(16, cols - b * 256 - subStart);
                    for (int e = 0; e < limit; e++)
                    {
                        byte two = (byte)((r * 3 + g * 7 + e * 5) & 3); // low two bits of the three-bit code, at this group's shift ...
                        bytes[block + 32 + 32 * h + 16 * (k & 1) + e] |= (byte)(two << (2 * (k >> 1)));

                        int highBit = (r * 5 + g * 3 + e * 7) & 1; // ... and its high bit, both states across every group of a half
                        bytes[block + 16 * (k & 1) + e] |= (byte)(highBit << m);
                    }
                }
            }
        }

        return bytes;
    }

    [Fact]
    public void CudaQ3K_Gemv_MatchesCpuOracleWithinBand()
    {
        const int rows = 3;
        const int cols = 256; // one whole super-block per row: packed scales, shared window bytes, and both hmask halves
        byte[] bytes = BuildQ3KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        float[] x = Activations(cols);
        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = CpuGemv(cpu, bytes, shape, QuantFormat.Q3_K, x);

        using CudaBackend cuda = OpenCuda();
        Assert.True(cuda.SupportsGemv(QuantFormat.Q3_K), "the CUDA seam stopped reporting its Q3_K GEMV kernel");

        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q3_K), actual);
        AssertWithinBand(expected, actual, BandQ3K, "cuda Q3_K exact block multiple");
    }

    [Fact]
    public void CudaQ3K_Gemv_PartialFinalBlock_StopsAtRealColumns()
    {
        const int rows = 2;
        const int cols = 300; // two super-blocks: the second fills groups 0..1 only, with group 1 partially filled
        byte[] bytes = BuildQ3KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        float[] x = Activations(cols);
        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = CpuGemv(cpu, bytes, shape, QuantFormat.Q3_K, x);

        using CudaBackend cuda = OpenCuda();
        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.Q3_K), actual);
        AssertWithinBand(expected, actual, BandQ3K, "cuda Q3_K partial final block");
    }

    [Fact]
    public void CudaQ3K_GemvBatch_MatchesCpuOracleAndSoloLaunch()
    {
        const int rows = 2;
        const int cols = 512; // a multi-block row: the batch loop must re-walk every super-block for each activation vector
        const int count = 3;
        byte[] bytes = BuildQ3KFixture(rows, cols);
        TensorShape shape = new([rows, cols]);

        AssertBatchMatchesSoloAndOracle(bytes, shape, QuantFormat.Q3_K, rows, cols, count, BandQ3K);
    }

    [Fact]
    public void CudaQ3K_DequantRow_MatchesCpuBackendScalarExpansion() =>
        AssertDequantRowBitsEqual(2, 300, BuildQ3KFixture(2, 300), QuantFormat.Q3_K);

    [Fact]
    public void CudaQ3K_Gemv_HostMappedPosture_RunsFromRegisteredHostBytes() =>
        AssertHostMappedMatchesOracle(BuildQ3KFixture(3, 256), new([3, 256]), QuantFormat.Q3_K, BandQ3K);

    // ---------------------------------------------------------------- shared assertions.

    private static void AssertBatchMatchesSoloAndOracle(byte[] bytes, TensorShape shape, QuantFormat format, int rows, int cols, int count, float band)
    {
        float[] inputs = new float[count * cols];
        for (int bIdx = 0; bIdx < count; bIdx++)
        {
            for (int c = 0; c < cols; c++)
            {
                inputs[bIdx * cols + c] = Activation(c + bIdx * 7);
            }
        }

        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = new float[count * rows];
        for (int bIdx = 0; bIdx < count; bIdx++)
        {
            float[] one = CpuGemv(cpu, bytes, shape, format, inputs.AsSpan(bIdx * cols, cols));
            for (int r = 0; r < rows; r++)
            {
                expected[bIdx * rows + r] = one[r];
            }
        }

        using CudaBackend cuda = OpenCuda();

        float[] batched = new float[count * rows];
        cuda.GemvBatch(inputs, count, new WeightRef(bytes, shape, format), batched);
        AssertWithinBand(expected, batched, band, $"cuda {format} batched launch");

        // The same kernel run one activation vector at a time must answer the same values: batching re-walks blocks, it does not rescale them.
        float[] solo = new float[rows];
        for (int bIdx = 0; bIdx < count; bIdx++)
        {
            cuda.Gemv(inputs.AsSpan(bIdx * cols, cols), new WeightRef(bytes, shape, format), solo);
            for (int r = 0; r < rows; r++)
            {
                Assert.Equal(batched[bIdx * rows + r], solo[r]); // bit-for-bit: identical launch shape on both sides of the batch loop
            }
        }
    }

    private static void AssertDequantRowBitsEqual(int rows, int cols, byte[] bytes, QuantFormat format)
    {
        TensorShape shape = new([rows, cols]);
        WeightRef w = new(bytes, shape, format);

        using CudaBackend cuda = OpenCuda();
        using CpuBackend cpu = OpenCpuOracle();

        float[] cudaOut = new float[cols];
        float[] cpuOut = new float[cols];
        for (int r = 0; r < rows; r++)
        {
            cuda.DequantRow(w, r, cudaOut);
            cpu.DequantRow(w, r, cpuOut);

            // Row expansion carries no accumulation chain: the device-side host math is the CPU scalar math byte for byte.
            BatchKernelTests.AssertBitsEqual(cpuOut, cudaOut, $"cuda {format} row {r} vs CpuBackend scalar expansion");
        }
    }

    private static void AssertHostMappedMatchesOracle(byte[] bytes, TensorShape shape, QuantFormat format, float band)
    {
        int rows = shape.Dims[0];
        int cols = shape.Dims[^1];
        float[] x = Activations(cols);

        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = CpuGemv(cpu, bytes, shape, format, x);

        using CudaBackend cuda = OpenCuda(CudaWeightResidency.HostMapped);
        Assert.Equal(CudaWeightResidency.HostMapped, cuda.WeightResidency);

        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, format), actual);
        AssertWithinBand(expected, actual, band, $"cuda {format} host-mapped posture");
    }

    // ---------------------------------------------------------------- the blocked-matmul family: same packed bytes, one unpack per activation-row window.

    private static float BandFor(QuantFormat format) => format switch
    {
        QuantFormat.Q8_0 => BandQ8_0,
        QuantFormat.Q5_0 => BandQ5_0,
        QuantFormat.Q2_0 => BandQ2_0,
        QuantFormat.Q5_K => BandQ5K,
        QuantFormat.Q2_K => BandQ2K,
        QuantFormat.Q3_K => BandQ3K,
        _ => throw new ArgumentException($"this file carries no fixture band for {format}", nameof(format)),
    };

    // The exact postures each format's recorded band was measured against in the GEMV tests above — that format's batched
    // launch over an exact block multiple (run here one vector at a time and as a three-vector partial window) plus its
    // single-vector partial-tail row — so a blocked matmul is compared with the CPU oracle on cases those bands already cover.
    private static readonly (QuantFormat Format, int Rows, int Cols, int Count)[] BandPostures =
    [
        (QuantFormat.Q8_0, 2, 96, 1), (QuantFormat.Q8_0, 2, 96, 3), (QuantFormat.Q8_0, 2, 36, 1),
        (QuantFormat.Q5_0, 2, 64, 1), (QuantFormat.Q5_0, 2, 64, 3), (QuantFormat.Q5_0, 2, 40, 1),
        (QuantFormat.Q2_0, 2, 128, 1), (QuantFormat.Q2_0, 2, 128, 3), (QuantFormat.Q2_0, 2, 84, 1),
        (QuantFormat.Q5_K, 2, 512, 1), (QuantFormat.Q5_K, 2, 512, 3), (QuantFormat.Q5_K, 2, 300, 1),
        (QuantFormat.Q2_K, 2, 512, 1), (QuantFormat.Q2_K, 2, 512, 3), (QuantFormat.Q2_K, 2, 300, 1),
        (QuantFormat.Q3_K, 2, 512, 1), (QuantFormat.Q3_K, 2, 512, 3), (QuantFormat.Q3_K, 2, 300, 1),
    ];

    private static byte[] FixtureFor(QuantFormat format, int rows, int cols) => format switch
    {
        QuantFormat.Q8_0 => BuildQ8_0Fixture(rows, cols),
        QuantFormat.Q5_0 => BuildQ5_0Fixture(rows, cols),
        QuantFormat.Q2_0 => BuildQ2_0Fixture(rows, cols),
        QuantFormat.Q5_K => BuildQ5KFixture(rows, cols),
        QuantFormat.Q2_K => BuildQ2KFixture(rows, cols),
        _ => BuildQ3KFixture(rows, cols),
    };

    private static float[] GemmInputs(int count, int cols)
    {
        float[] inputs = new float[count * cols];
        for (int bIdx = 0; bIdx < count; bIdx++)
        {
            for (int c = 0; c < cols; c++)
            {
                inputs[bIdx * cols + c] = Activation(c + bIdx * 7); // distinct activation vectors per batch slot.
            }
        }

        return inputs;
    }

    private static void AssertGemmWindowMatchesSoloLaunch(byte[] bytes, TensorShape shape, QuantFormat format, int count)
    {
        float[] inputs = GemmInputs(count, shape.Dims[^1]);

        int rows = shape.Dims[0];
        int cols = shape.Dims[^1];

        using CudaBackend cuda = OpenCuda();
        Assert.True(cuda.SupportsGemm(format), $"the CUDA seam stopped claiming its {format} blocked matmul");

        float[] gemm = new float[count * rows];
        cuda.GemmBatch(inputs, count, new WeightRef(bytes, shape, format), gemm);

        // Every window lane answers bit-for-bit what that vector's own solo launch answers: the unpack runs once per window and each lane's chain advances over the same columns in the same order as its solo walk.
        float[] solo = new float[rows];
        for (int bIdx = 0; bIdx < count; bIdx++)
        {
            cuda.Gemv(inputs.AsSpan(bIdx * cols, cols), new WeightRef(bytes, shape, format), solo);

            for (int r = 0; r < rows; r++)
            {
                Assert.Equal(solo[r], gemm[bIdx * rows + r]); // bit-for-bit: the windowed dot and the solo dot are the same chain.
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q2_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    public void CudaBlockQuantGemm_WindowLanesAnswerEachVectorExactlyAsItsOwnLaunchDoes(QuantFormat format)
    {
        foreach ((_, int rows, int cols, _) in BandPostures.Where(p => p.Format == format).DistinctBy(p => (p.Rows, p.Cols)))
        {
            byte[] bytes = FixtureFor(format, rows, cols);
            TensorShape shape = new([rows, cols]);

            foreach (int count in new[] { 5, 7, 8, 9 }) // a partially filled window, an exact window, and a batch that spans more than one window
            {
                AssertGemmWindowMatchesSoloLaunch(bytes, shape, format, count);
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q2_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    public void CudaBlockQuantGemm_MatchesCpuOracleWithinBand(QuantFormat format)
    {
        float band = BandFor(format);

        foreach ((_, int rows, int cols, int count) in BandPostures.Where(p => p.Format == format)) // the posture this format's band was measured on: a partial activation-row window over an exact block multiple, and its single-vector partial-tail row.
        {
            byte[] bytes = FixtureFor(format, rows, cols);
            TensorShape shape = new([rows, cols]);

            float[] inputs = GemmInputs(count, cols);

            using CpuBackend cpu = OpenCpuOracle();
            float[] expected = new float[count * rows];
            for (int bIdx = 0; bIdx < count; bIdx++)
            {
                float[] one = CpuGemv(cpu, bytes, shape, format, inputs.AsSpan(bIdx * cols, cols));
                for (int r = 0; r < rows; r++)
                {
                    expected[bIdx * rows + r] = one[r];
                }
            }

            using CudaBackend cuda = OpenCuda();
            float[] gemm = new float[count * rows];
            cuda.GemmBatch(inputs, count, new WeightRef(bytes, shape, format), gemm);
            AssertWithinBand(expected, gemm, band, $"cuda {format} blocked matmul at {rows}x{cols} ({count} vectors)");

            // Same case, device to device: the windowed chain is bit-identical with each vector's own solo launch.
            float[] solo = new float[rows];
            for (int bIdx = 0; bIdx < count; bIdx++)
            {
                cuda.Gemv(inputs.AsSpan(bIdx * cols, cols), new WeightRef(bytes, shape, format), solo);

                for (int r = 0; r < rows; r++)
                {
                    Assert.Equal(solo[r], gemm[bIdx * rows + r]);
                }
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.F32)]
    [InlineData(QuantFormat.F16)]
    [InlineData(QuantFormat.BF16)]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.Q2_0)]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ2_S)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ3_S)]
    public void CudaCapability_ClaimsEveryGEMV_AndClaimsOrRefusesBlockedMatmulWithItsKernelTable(QuantFormat format)
    {
        Assert.True(CudaBackend.HasGemvKernel(format), $"the CUDA seam stopped reporting its {format} GEMV kernel");

        TensorShape shape = new([1, 32]);
        byte[] bytes = new byte[shape.RowStrideBytes(format)]; // a correctly sized packed row for every format the seam now executes.
        float[] x = Activations(32);
        float[] output = new float[1];

        using CudaBackend cuda = OpenCuda();
        Assert.True(cuda.SupportsGemv(format), $"the backend stopped claiming the {format} GEMV its kernel exists for"); // weight GEMV runs on the device for every format this seam accepts.

        bool carriesGemm = CudaBackend.HasGemmKernel(format);
        Assert.Equal(carriesGemm, cuda.SupportsGemm(format)); // the capability gate answers from the same table the function handles resolve from.

        if (carriesGemm)
        {
            cuda.GemmBatch(x, 1, new WeightRef(bytes, shape, format), output); // its own windowed kernel runs: no throw and no CPU fallback behind it.
        }
        else
        {
            NotSupportedException gemmError = Assert.Throws<NotSupportedException>(
                () => cuda.GemmBatch(x, 1, new WeightRef(bytes, shape, format), output));
            Assert.Contains($"no blocked matmul kernel for format {format}", gemmError.Message, StringComparison.Ordinal);
        }
    }
}
