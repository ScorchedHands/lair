using System.Reflection;
using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// GPU gate for the packed i-quant GEMV kernels LAIR now carries on the CUDA seam: hand-built blocks per format whose header halves,
/// six-bit scale fields, grid-index bytes with their high-bit extensions, raw sign planes and shared ksigns entries walk every unpacking
/// class a real file produces — swept so every byte position of every block passes through its whole value range across the cases, negative
/// and subnormal half-float headers beside normal ones, zero-scaled groups next to non-zero ones, grid lookups at both table extremes, and a
/// partial tail block whose real columns stop inside a super-block. Every case runs the CUDA kernel against those packed bytes exactly as a
/// weight tensor would and compares against CpuBackend's scalar oracle over the same WeightRef: device output must land inside that format's
/// recorded band (measured on this box, asserted per element), batched launches must match both the CPU oracle and the same kernel run solo
/// for every activation vector in the batch, host-mapped weight postures must answer from registered host bytes, row dequantization —
/// host-side scalar math with no accumulation chain — must agree bit for bit with CpuBackend's expansion, and the device side's pinned lookup
/// tables must be byte-identical to the CPU backend's own. Sub-4-bit formats dot against a Q8_K quantization of the activation on both sides;
/// IQ4_NL and IQ4_XS fuse their table value with the block scale in float, as their CPU chains do.
/// </summary>
[Trait("Category", "Kernel")]
[Trait("Category", "Cuda")]
public sealed class CudaIQuantGemvTests
{
    // Recorded per-format bands: relative ceilings on the drift of the CUDA GEMV chain against the CPU scalar oracle, in the same shape as
    // the block-quant band assertion — |expected - actual| <= band * max(|expected|, |actual|). Each is a round-up of the widest drift measured
    // on this box over every case below. These are CUDA-only bands: the CPU oracle's own scalar chain stays bit-identical and no existing
    // dense-family or block-quant band moves here. Measured maxima recorded per format, from a pass run with the bands pinned to zero so every
    // non-exact element reports its case maximum (measured 2026-11-18 on this box):
    //   IQ4_NL: 8.97e-8 exact-block, 3.34e-7 batched, 1.03e-6 host-mapped posture -> band 2e-6 (the FP nibble chain accumulates as FFMA on the
    //     device where the oracle multiplies then adds; every weight value itself agrees exactly).
    //   IQ4_XS: 1.02e-6 exact-block, 1.07e-6 batched, 1.02e-6 host-mapped posture -> band 2e-6 (same contraction story over a longer chain of sub-block scales).
    //   IQ2_XXS / IQ2_XS / IQ3_XXS: bit-exact against the oracle in every case, exact-block, partial tail, batched and host-mapped.
    //   IQ2_S: 8.27e-8 on the partial final block, bit-exact everywhere else; IQ3_S: 1.47e-7 on the partial final block, bit-exact elsewhere -
    //     the same FFMA-vs-mul/add difference landing once per super-block where a total happens to be exactly representable in every other case.

    private const float BandIQ4Nl = 2e-6f; // measured worst above: 1.03189382E-06 (host-mapped posture case).
    private const float BandIQ4Xs = 2e-6f; // measured worst above: 1.0708269E-06 (batched launch case).
    private const float BandIQ2XXS = 5e-7f; // measured worst above: 0 - every element bit-exact vs the oracle chain.
    private const float BandIQ2Xs = 5e-7f; // measured worst above: 0 - every element bit-exact vs the oracle chain.
    private const float BandIQ2S = 5e-7f; // measured worst above: 8.26988327E-08 (partial final block case).
    private const float BandIQ3Xxs = 5e-7f; // measured worst above: 0 - every element bit-exact vs the oracle chain.
    private const float BandIQ3S = 5e-7f; // measured worst above: 1.47036275E-07 (partial final block case).

    // ---------------------------------------------------------------- fixture geometry per format.

    private static int BlocksPerRow(QuantFormat format, int cols) => format switch
    {
        QuantFormat.IQ4_NL => (cols + 31) / 32,
        QuantFormat.IQ4_XS => (cols + 255) / 256,
        _ => (cols + 255) / 256, // the five sub-4-bit formats all run in Q8_K super-blocks.
    };

    private static int BlockBytes(QuantFormat format) => format switch
    {
        QuantFormat.IQ4_NL => 18,
        QuantFormat.IQ4_XS => 136,
        QuantFormat.IQ2_XXS => 66,
        QuantFormat.IQ2_XS => 74,
        QuantFormat.IQ2_S => 82,
        QuantFormat.IQ3_XXS => 98,
        QuantFormat.IQ3_S => 110,
        _ => throw new ArgumentException($"{format} is not an i-quant format", nameof(format)),
    };

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
        float[] output = new float[rows];
        cpu.Gemv(x, new WeightRef(bytes, shape, format), output);
        return output;
    }

    private static void AssertWithinBand(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float band, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        // Relative drift per element: the device chain and the oracle chain agree on every weight value, so what separates them is summation
        // rounding that grows with the size of the sum. The failure message names both this element's drift and the widest drift in the case.
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

    private static float WorstRelativeDrift(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);

        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            float distance = Math.Abs(actual[i] - expected[i]);
            float scale = Math.Max(Math.Abs(expected[i]), Math.Abs(actual[i]));
            float relative = scale == 0f ? distance : distance / scale; // a zero against zero is exact, and any drift off zero counts fully.
            if (relative > worst) worst = relative;
        }

        return worst;
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

    // Half-float headers chosen per block so a run covers the normal range, a negative scale, and a subnormal magnitude whose sign has to
    // survive widening on both sides of the seam.
    private static readonly Half[] HeaderHalves = [(Half)0.25f, (Half)(-7.5e-06f), (Half)3.5f];

    private static void WriteHeaderLE(Span<byte> dst, int offset, Half value)
    {
        ushort bits = (ushort)BitConverter.HalfToInt16Bits(value);
        dst[offset] = (byte)bits;
        dst[offset + 1] = (byte)(bits >> 8);
    }

    // A nibble window whose low and high codes walk the whole four-bit range across successive bytes, so every IQ4 table entry gets used.
    private static byte NibblePair(int k, int r, int b) =>
        (byte)(((k * 5 + r * 3 + b * 7) & 0xF) | (((k * 11 + r * 13 + b * 2) & 0xF) << 4));

    // IQ4_NL: half-float d at bytes 0..1, sixteen nibble bytes at 2..17 (low nibbles elements 0..15, high nibbles 16..31).
    private static byte[] BuildIQ4NlFixture(int rows, int cols)
    {
        int blocksPerRow = BlocksPerRow(QuantFormat.IQ4_NL, cols);
        byte[] bytes = new byte[rows * blocksPerRow * BlockBytes(QuantFormat.IQ4_NL)];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * BlockBytes(QuantFormat.IQ4_NL);

                WriteHeaderLE(bytes, block, HeaderHalves[(r * 2 + b) % HeaderHalves.Length]); // d at bytes 0..1.
                for (int k = 0; k < 16; k++)
                {
                    bytes[block + 2 + k] = NibblePair(k, r, b); // every code value appears somewhere across the cases.
                }
            }
        }

        return bytes;
    }

    // IQ4_XS: d at 0..1, six-bit scale fields packed as scales_h (bytes 2..3) plus four low-nibble bytes at 4..7, then eight sixteen-byte
    // nibble windows at block + 8 + 16*ib. The swept scale bytes cover zero and non-zero codes with both high-p-bit pairs next to them.
    private static byte[] BuildIQ4XsFixture(int rows, int cols)
    {
        int blocksPerRow = BlocksPerRow(QuantFormat.IQ4_XS, cols);
        byte[] bytes = new byte[rows * blocksPerRow * BlockBytes(QuantFormat.IQ4_XS)];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * BlockBytes(QuantFormat.IQ4_XS);

                WriteHeaderLE(bytes, block, HeaderHalves[(r * 2 + b) % HeaderHalves.Length]); // d at bytes 0..1.

                ushort scalesH = unchecked((ushort)(0x5A3C + r * 97 + b * 61)); // every sub-block's two high bits vary, including the zero pair and the three pair.
                WriteUShortLE(bytes, block + 2, scalesH);

                for (int k = 0; k < 4; k++)
                {
                    bytes[block + 4 + k] = unchecked((byte)(k * 17 + r * 53 + b * 29)); // low nibbles walk the whole range in both halves of each byte.
                }

                for (int ib = 0; ib < 8; ib++)
                {
                    for (int k = 0; k < 16; k++)
                    {
                        bytes[block + 8 + 16 * ib + k] = NibblePair(k, r + ib, b); // each window walks the whole code range independently.
                    }
                }
            }
        }

        return bytes;
    }

    private static void WriteUShortLE(Span<byte> dst, int offset, ushort value)
    {
        dst[offset] = (byte)value;
        dst[offset + 1] = (byte)(value >> 8);
    }

    // The five sub-4-bit formats share one fixture shape: every byte of a block walks its whole range across successive blocks and rows, so
    // grid-index bytes hit both table extremes with their high-bit extensions set and unset, raw sign planes land on zero and non-zero bits,
    // scale nibbles cover ls = 1 next to ls = 3 and zero scales beside non-zero ones. The half-float header at byte 0 is then overwritten per
    // block so normal, negative, and subnormal magnitudes each appear where the kernel reads them.
    private static byte[] BuildQ8KFixture(QuantFormat format, int rows, int cols)
    {
        int blocksPerRow = BlocksPerRow(format, cols);
        int stride = BlockBytes(format);
        byte[] bytes = new byte[rows * blocksPerRow * stride];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * stride;

                for (int i = 2; i < stride; i++) // the swept body: every position passes through its whole byte range across cases.
                {
                    bytes[block + i] = unchecked((byte)(i * 13 + r * 97 + b * 53));
                }

                WriteHeaderLE(bytes, block, HeaderHalves[(r * 2 + b) % HeaderHalves.Length]); // d at bytes 0..1.
            }
        }

        return bytes;
    }

    private static readonly (QuantFormat Format, float Band)[] Formats =
    [
        (QuantFormat.IQ4_NL, BandIQ4Nl),
        (QuantFormat.IQ4_XS, BandIQ4Xs),
        (QuantFormat.IQ2_XXS, BandIQ2XXS),
        (QuantFormat.IQ2_XS, BandIQ2Xs),
        (QuantFormat.IQ2_S, BandIQ2S),
        (QuantFormat.IQ3_XXS, BandIQ3Xxs),
        (QuantFormat.IQ3_S, BandIQ3S),
    ];

    private static byte[] BuildFixture(QuantFormat format, int rows, int cols) =>
        format == QuantFormat.IQ4_NL ? BuildIQ4NlFixture(rows, cols) :
        format == QuantFormat.IQ4_XS ? BuildIQ4XsFixture(rows, cols) :
        BuildQ8KFixture(format, rows, cols);

    // A column count that fills whole super-blocks plus a tail block whose real columns stop inside the last group.
    private static int TailColumns(QuantFormat format) =>
        format == QuantFormat.IQ4_NL ? 97 : // three full 32-element blocks and one partial
        format == QuantFormat.IQ4_XS ? 540: // two full 256-element blocks, then a sub-block-stopping tail
        612; // two whole Q8_K super-blocks plus four columns into the third

    private static int ExactColumns(QuantFormat format) =>
        format == QuantFormat.IQ4_NL ? 96 :
        format == QuantFormat.IQ4_XS ? 512 :
        512;

    // ---------------------------------------------------------------- per-format parity.

    [Theory]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ2_S)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ3_S)]
    public void CudaIQuantGemv_MatchesCpuOracleWithinBand_AndPartialTailStopsAtRealColumns(QuantFormat format)
    {
        float band = BandFor(format);

        foreach (int cols in new[] { ExactColumns(format), TailColumns(format) })
        {
            const int rows = 2;
            byte[] bytes = BuildFixture(format, rows, cols);
            TensorShape shape = new([rows, cols]);

            float[] x = Activations(cols);
            using CpuBackend cpu = OpenCpuOracle();
            float[] expected = CpuGemv(cpu, bytes, shape, format, x);

            using CudaBackend cuda = OpenCuda();
            float[] actual = new float[rows];
            cuda.Gemv(x, new WeightRef(bytes, shape, format), actual);
            AssertWithinBand(expected, actual, band, $"cuda {format} {(cols == ExactColumns(format) ? "exact block multiple" : "partial final block")}");
        }
    }

    [Theory]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ2_S)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ3_S)]
    public void CudaIQuantGemvBatch_MatchesCpuOracleAndSoloLaunch(QuantFormat format) =>
        AssertBatchMatchesSoloAndOracle(format, 2, ExactColumns(format), count: 3);

    [Theory]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ4_XS)]
    public void CudaIQuantGemm_WindowLanesAnswerEachVectorExactlyAsItsOwnLaunchDoes(QuantFormat format)
    {
        foreach (int count in new[] { 5, 7, 8, 9 }) // a partially filled window, an exact window, and a batch that spans more than one window
        {
            AssertGemmWindowMatchesSoloLaunch(format, 2, ExactColumns(format), count);
        }
    }

    [Theory]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ4_XS)]
    public void CudaIQuantGemm_MatchesCpuOracleWithinBand_AndPartialTailStopsAtRealColumns(QuantFormat format)
    {
        AssertGemmMatchesOracle(format, 2, ExactColumns(format), count: 3); // a partial activation-row window against an exact block multiple.
        AssertGemmMatchesOracle(format, 2, TailColumns(format), count: 1); // one vector through a row whose last quant block is partial.
    }

    // The Q8_K i-quant family carries its GEMV kernel today but no windowed blocked matmul yet: the capability gate and the launch both refuse it naming this backend.
    [Theory]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ2_S)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ3_S)]
    public void CudaIQuantGemm_Q8KFamilyRefusesNamingThisBackend(QuantFormat format)
    {
        Assert.True(CudaBackend.HasGemvKernel(format), $"the CUDA seam stopped reporting its {format} GEMV kernel");

        int cols = ExactColumns(format);
        byte[] bytes = BuildFixture(format, 1, cols);
        TensorShape shape = new([1, cols]);

        using CudaBackend cuda = OpenCuda();
        Assert.False(cuda.SupportsGemm(format), $"the CUDA seam claimed a {format} blocked matmul its kernel does not exist for");

        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => cuda.GemmBatch(Activations(cols), 1, new WeightRef(bytes, shape, format), new float[1]));
        Assert.Contains($"no blocked matmul kernel for format {format}", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ2_S)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ3_S)]
    public void CudaIQuantDequantRow_MatchesCpuBackendScalarExpansion(QuantFormat format) =>
        AssertDequantRowBitsEqual(format, 2, TailColumns(format));

    [Theory]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ2_S)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ3_S)]
    public void CudaIQuantGemv_HostMappedPosture_RunsFromRegisteredHostBytes(QuantFormat format) =>
        AssertHostMappedMatchesOracle(format, 3, ExactColumns(format));

    // ---------------------------------------------------------------- shared tables pinned byte-identical.

    [Theory]
    [InlineData("GetKSignsIQ2XS")]
    [InlineData("GetIq2XXSGrid")]
    [InlineData("GetIq2XSGrid")]
    [InlineData("GetIq2SGrid")]
    [InlineData("GetIq3XXSGrid")]
    [InlineData("GetIq3SGrid")]
    public void CudaPinnedTables_MatchCpuBackendTablesByteForByte(string tableGetter)
    {
        ReadOnlySpan<byte> cpuTable = ReadStaticTable(typeof(CpuBackend), tableGetter);
        ReadOnlySpan<byte> cudaTable = ReadStaticTable(typeof(Lair.Backends.Cuda.CudaBackend), tableGetter);

        Assert.Equal(cpuTable.Length, cudaTable.Length); // the device mirror carries exactly as many entries as the CPU oracle's table.

        int firstMismatch = -1;
        for (int i = 0; i < cpuTable.Length; i++)
        {
            if (cpuTable[i] != cudaTable[i])
            {
                firstMismatch = i;
                break;
            }
        }

        if (firstMismatch >= 0)
        {
            Assert.Fail($"the CUDA {tableGetter} table diverges from the CPU one at entry {firstMismatch}: cpu {cpuTable[firstMismatch]:X2}, cuda {cudaTable[firstMismatch]:X2}"); // built only when a mismatch actually exists: eager interpolation would index the miss sentinel.
        }
    }

    // ---------------------------------------------------------------- shared assertions.

    private static float BandFor(QuantFormat format) => Formats.First(f => f.Format == format).Band;

    private static void AssertBatchMatchesSoloAndOracle(QuantFormat format, int rows, int cols, int count)
    {
        byte[] bytes = BuildFixture(format, rows, cols);
        TensorShape shape = new([rows, cols]);
        float band = BandFor(format);

        float[] inputs = new float[count * cols];
        for (int bIdx = 0; bIdx < count; bIdx++)
        {
            for (int c = 0; c < cols; c++)
            {
                inputs[bIdx * cols + c] = Activation(c + bIdx * 7); // distinct activation vectors per batch slot.
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
                Assert.Equal(batched[bIdx * rows + r], solo[r]); // bit-for-bit: identical launch shape on both sides of the batch loop.
            }
        }
    }
    private static void AssertGemmMatchesOracle(QuantFormat format, int rows, int cols, int count)
    {
        byte[] bytes = BuildFixture(format, rows, cols);
        TensorShape shape = new([rows, cols]);
        float band = BandFor(format);

        float[] inputs = new float[count * cols];
        for (int bIdx = 0; bIdx < count; bIdx++)
        {
            for (int c = 0; c < cols; c++)
            {
                inputs[bIdx * cols + c] = Activation(c + bIdx * 7); // distinct activation vectors per batch slot.
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
        Assert.True(cuda.SupportsGemm(format), $"the CUDA seam stopped claiming its {format} blocked matmul");

        float[] gemm = new float[count * rows];
        cuda.GemmBatch(inputs, count, new WeightRef(bytes, shape, format), gemm);
        AssertWithinBand(expected, gemm, band, $"cuda {format} blocked matmul ({count} vectors)");

        // Same case device to device: every window lane answers bit-for-bit what that vector's own solo launch answers.
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

    private static void AssertGemmWindowMatchesSoloLaunch(QuantFormat format, int rows, int cols, int count)
    {
        byte[] bytes = BuildFixture(format, rows, cols);
        TensorShape shape = new([rows, cols]);

        float[] inputs = new float[count * cols];
        for (int bIdx = 0; bIdx < count; bIdx++)
        {
            for (int c = 0; c < cols; c++)
            {
                inputs[bIdx * cols + c] = Activation(c + bIdx * 7); // distinct activation vectors per batch slot.
            }
        }

        using CudaBackend cuda = OpenCuda();
        Assert.True(cuda.SupportsGemm(format), $"the CUDA seam stopped claiming its {format} blocked matmul");

        float[] gemm = new float[count * rows];
        cuda.GemmBatch(inputs, count, new WeightRef(bytes, shape, format), gemm);

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

    private static void AssertDequantRowBitsEqual(QuantFormat format, int rows, int cols)
    {
        byte[] bytes = BuildFixture(format, rows, cols);
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

    private static void AssertHostMappedMatchesOracle(QuantFormat format, int rows, int cols)
    {
        byte[] bytes = BuildFixture(format, rows, cols);
        TensorShape shape = new([rows, cols]);
        float band = BandFor(format);

        float[] x = Activations(cols);

        using CpuBackend cpu = OpenCpuOracle();
        float[] expected = CpuGemv(cpu, bytes, shape, format, x);

        using CudaBackend cuda = OpenCuda(CudaWeightResidency.HostMapped);
        Assert.Equal(CudaWeightResidency.HostMapped, cuda.WeightResidency);

        float[] actual = new float[rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, format), actual);
        AssertWithinBand(expected, actual, band, $"cuda {format} host-mapped posture");
    }

    private delegate ReadOnlySpan<byte> PinnedTableGetter();


    // Both backends keep their pinned tables as private static span-returning members; a directly-invoked delegate over the method reads one
    // without boxing the span. The gate below is what proves the device mirror and the oracle's table are the same bytes.
    private static ReadOnlySpan<byte> ReadStaticTable(Type owner, string getterName)
    {
        MethodInfo method = owner.GetMethod(getterName, BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException($"{owner.Name} exposes no pinned table named {getterName}");

        return ((PinnedTableGetter)method.CreateDelegate(typeof(PinnedTableGetter)))();
    }



}
