using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// GPU gate for the unquantized dense-matmul entries of the CUDA blocked family: hand-built F32/F16/BF16 weight rows walked by
/// <see cref="CudaBackend.GemmBatch"/> against exactly the packed bytes a weight tensor would carry. Two posture claims are made per
/// format. Device to device, the claim is bit-for-bit: every windowed lane advances its accumulator chain over the same columns in the
/// same order with the same operand association as that vector's own solo GEMV launch, against the very value that format's own embedded-PTX
/// widening produced (plain f32 loads, integer-bit half widening including subnormals and negatives, or the bf16 high-half shift), so a
/// windowed dot answers what repeated single-vector launches answer to the last bit — in either weight-residency posture. Against CpuBackend's
/// scalar oracle the claim is this seam's existing CUDA band posture rather than raw bytes: the driver's JIT folds each element's multiply and
/// accumulate into one fused operation, so a device chain lands within a per-format measured band of the host chain instead of on its bits —
/// the same drift every GEMV launch on these formats already answers (which is why batch==solo holds bit-for-bit while an oracle comparison does
/// not), inherited from the blocked family's band posture rather than widened. No format widens through a lossy conversion, no native math library
/// stands behind any of it, and the formats that still carry no windowed kernel keep refusing naming this backend.
/// </summary>
[Trait("Category", "Kernel")]
[Trait("Category", "Parity")]
[Trait("Category", "Cuda")]
public sealed class CudaDenseGemmTests
{
    // Recorded per-format bands for the unquantized dense-matmul family, in the same shape as the packed-quant CUDA bands: relative ceilings on the
    // drift of a device chain against CpuBackend's scalar oracle, |expected - actual| <= band * max(|expected|, |actual|). Each is a round-up with
    // headroom for another device or driver JIT over every case below — measured widest drift on this box: F32 2.02e-5, BF16 9.19e-6, and F16 1.35e-3,
    // where a half-weight fixture whose signs alternate leaves the longest chain most exposed to cancellation. These are CUDA-only numbers for these three
    // formats: no CPU band moves, no packed-quant band moves, and they measure what this seam's GEMV launches on the same bytes already answer — every
    // windowed lane is bit-for-bit its own vector's solo launch (pinned separately below), so nothing here separates a blocked matmul from a row pass.
    private const float BandF32 = 5e-5f;
    private const float BandBf16 = 2e-5f;
    private const float BandF16 = 4e-3f;

    private static float Activation(int c) => (float)Math.Sin(0.11 * c + 1.7) * 1.3f;

    // Half-float weights chosen so a run covers the normal range, a negative value, both subnormal magnitudes, and their signs — every case that
    // separates this seam's integer-bit widening from the cvt.f16 forms its driver rejects.
    private static readonly Half[] WeightHalves =
    [
        (Half)0.25f, (Half)(-7.5e-06f), (Half)3.5f, (Half)(-1.25f), (Half)5.9604645e-08f, (Half)(-6.0e-05f),
    ];

    // bf16 codes as stored: both signs of a whole number and a fraction, the smallest subnormal code in each sign, and an exponent-heavy value
    // whose low mantissa bits simply do not exist at 8 bits.
    private static readonly ushort[] WeightBf16Codes = [0x3F80, 0xBF80, 0x4120, 0xBE80, 0x0001, 0x8001];

    // The postures every case below runs: whole columns and a ragged tail width, one row and several rows (several weight rows land in the same
    // kernel block), against batch sizes that partially fill an activation-row window, fill it exactly, and span windows.
    private static readonly (int Rows, int Cols)[] Postures = [(2, 96), (5, 84), (3, 17), (1, 128)];

    private static readonly int[] BatchCounts = [1, 3, 5, 7, 8, 9];

    // ---------------------------------------------------------------- CPU-oracle band parity and batch==solo sanity

    [Theory]
    [InlineData(QuantFormat.F32)]
    [InlineData(QuantFormat.F16)]
    [InlineData(QuantFormat.BF16)]
    public void CudaDenseGemm_LandsInsideItsBandAgainstScalarOracle_WithinEveryWindowShape(QuantFormat format)
    {
        foreach ((int rows, int cols) in Postures)
        {
            byte[] bytes = Fixture(format, rows, cols);
            TensorShape shape = new([rows, cols]);

            foreach (int count in BatchCounts)
            {
                float[] inputs = GemmInputs(count, cols);

                using CpuBackend cpu = OpenCpuOracle();
                float[] expected = ScalarReference(cpu, bytes, shape, format, inputs, rows, cols, count);

                using CudaBackend cuda = OpenCuda();
                Assert.True(cuda.SupportsGemm(format), $"the CUDA seam stopped claiming its {format} blocked matmul");

                float[] gemm = new float[count * rows];
                cuda.GemmBatch(inputs, count, new WeightRef(bytes, shape, format), gemm);

                AssertWithinBand(expected, gemm, BandFor(format), $"cuda {format} blocked matmul at {rows}x{cols} ({count} vectors)");
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.F32)]
    [InlineData(QuantFormat.F16)]
    [InlineData(QuantFormat.BF16)]
    public void CudaDenseGemm_WindowLanesAnswerEachVectorExactlyAsItsOwnLaunchDoes(QuantFormat format)
    {
        foreach ((int rows, int cols) in Postures)
        {
            byte[] bytes = Fixture(format, rows, cols);
            TensorShape shape = new([rows, cols]);

            // A partially filled window (5, 7), an exact one (8, 16), a batch that spans windows (9), and single-vector calls.
            foreach (int count in new[] { 1, 5, 7, 8, 9, 16 })
            {
                float[] inputs = GemmInputs(count, cols);

                using CudaBackend cuda = OpenCuda();
                float[] gemm = new float[count * rows];
                cuda.GemmBatch(inputs, count, new WeightRef(bytes, shape, format), gemm);

                float[] solo = new float[rows];
                for (int bIdx = 0; bIdx < count; bIdx++)
                {
                    cuda.Gemv(inputs.AsSpan(bIdx * cols, cols), new WeightRef(bytes, shape, format), solo);

                    for (int r = 0; r < rows; r++)
                    {
                        Assert.Equal(solo[r], gemm[bIdx * rows + r]); // bit-for-bit: the windowed dot and that vector's own launch are the same chain.
                    }
                }
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.F32)]
    [InlineData(QuantFormat.F16)]
    [InlineData(QuantFormat.BF16)]
    public void CudaDenseGemm_HostMappedWeights_AnswerTheDeviceCopyPostureBitForBit(QuantFormat format)
    {
        foreach ((int rows, int cols) in Postures.Take(2))
        {
            byte[] bytes = Fixture(format, rows, cols);
            TensorShape shape = new([rows, cols]);

            float[] inputs = GemmInputs(9, cols); // a partially filled window against both postures.
            WeightRef w = new(bytes, shape, format);

            using CudaBackend deviceCopy = OpenCuda();
            float[] copied = new float[9 * rows];
            deviceCopy.GemmBatch(inputs, 9, w, copied);

            using CudaBackend mapped = OpenCuda(CudaWeightResidency.HostMapped);
            Assert.Equal(CudaWeightResidency.HostMapped, mapped.WeightResidency);

            float[] mappedOut = new float[9 * rows];
            mapped.GemmBatch(inputs, 9, w, mappedOut); // the same kernel against registered host bytes.

            BatchKernelTests.AssertBitsEqual(copied, mappedOut, $"cuda {format} blocked matmul over host-mapped weights at {rows}x{cols}");

            AssertWithinBand(
                ScalarReference(OpenCpuOracle(), bytes, shape, format, inputs, rows, cols, 9),
                mappedOut,
                BandFor(format),
                $"cuda {format} blocked matmul over host-mapped weights at {rows}x{cols}");
        }
    }

    // ---------------------------------------------------------------- capability posture honesty

    [Theory]
    [InlineData(QuantFormat.F32, true)]
    [InlineData(QuantFormat.F16, true)]
    [InlineData(QuantFormat.BF16, true)]
    [InlineData(QuantFormat.Q8_0, true)]
    [InlineData(QuantFormat.Q4_K, true)]
    [InlineData(QuantFormat.IQ4_XS, true)]
    [InlineData(QuantFormat.IQ2_XXS, false)] // the Q8_K i-quants still carry no windowed kernel: they keep refusing at load.
    [InlineData(QuantFormat.IQ2_XS, false)]
    [InlineData(QuantFormat.IQ2_S, false)]
    [InlineData(QuantFormat.IQ3_XXS, false)]
    [InlineData(QuantFormat.IQ3_S, false)]
    public void CudaDenseMatmulCapability_CoversExactlyTheFormatsWithWindowedKernels(QuantFormat format, bool supported)
    {
        Assert.Equal(supported, CudaBackend.HasGemmKernel(format));

        using CudaBackend cuda = OpenCuda();
        Assert.Equal(supported, cuda.SupportsGemm(format)); // the instance answers from the same table its function handles resolve from.

        TensorShape shape = new([2, 32]);

        // A correctly sized packed row pair for every format this lane names: the unquantized trio gets real weight patterns and every other
        // supported format its own block packing, so nothing here fails on bytes rather than on the kernel table.
        byte[] bytes = IsUnquantized(format) ? Fixture(format, 2, 32) : new byte[2 * shape.RowStrideBytes(format)];

        float[] inputs = GemmInputs(1, 32);
        float[] output = new float[2];

        if (supported)
        {
            cuda.GemmBatch(inputs, 1, new WeightRef(bytes, shape, format), output); // its own kernel runs: no throw and no CPU fallback behind it.
        }
        else
        {
            NotSupportedException error = Assert.Throws<NotSupportedException>(
                () => cuda.GemmBatch(inputs, 1, new WeightRef(bytes, shape, format), output));

            Assert.Contains("CudaBackend ('cuda')", error.Message);
            Assert.Contains($"no blocked matmul kernel for format {format}", error.Message, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void CudaDenseGemm_UnquantizedFormatsReportSupport_WithoutAnyNativeLibrarySeam()
    {
        // The capability facts answer from the embedded-PTX table alone: nothing here links a math library or resolves a handle on one.
        Assert.True(CudaBackend.HasGemmKernel(QuantFormat.F32), "the CUDA seam stopped reporting its hand-tiled F32 dense matmul");
        Assert.True(CudaBackend.HasGemmKernel(QuantFormat.F16), "the CUDA seam stopped reporting its windowed half-widening F16 dense matmul");
        Assert.True(CudaBackend.HasGemmKernel(QuantFormat.BF16), "the CUDA seam stopped reporting its high-half-shift BF16 dense matmul");

        using CudaBackend cuda = OpenCuda();
        foreach (QuantFormat format in new[] { QuantFormat.F32, QuantFormat.F16, QuantFormat.BF16 })
        {
            Assert.True(cuda.SupportsGemm(format), $"the opened backend stopped claiming its {format} blocked matmul");

            byte[] bytes = Fixture(format, 4, 96);
            float[] inputs = GemmInputs(8, 96); // an exact window across four weight rows.
            float[] gemm = new float[8 * 4];
            cuda.GemmBatch(inputs, 8, new WeightRef(bytes, new TensorShape([4, 96]), format), gemm);

            Assert.False(gemm.Any(float.IsNaN), $"cuda {format} dense matmul produced a NaN against well-formed weights");
        }
    }

    // ---------------------------------------------------------------- fixtures and helpers

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

    private static bool IsUnquantized(QuantFormat format) => format is QuantFormat.F32 or QuantFormat.F16 or QuantFormat.BF16;

    private static float BandFor(QuantFormat format) => format switch
    {
        QuantFormat.F32 => BandF32,
        QuantFormat.F16 => BandF16,
        _ => BandBf16,
    };

    // The oracle every band is measured against: CpuBackend's scalar single-vector walk over the same packed bytes and activations.
    private static float[] ScalarReference(CpuBackend cpu, byte[] bytes, TensorShape shape, QuantFormat format, ReadOnlySpan<float> inputs, int rows, int cols, int count)
    {
        float[] expected = new float[count * rows];
        float[] one = new float[rows];

        for (int bIdx = 0; bIdx < count; bIdx++)
        {
            cpu.Gemv(inputs.Slice(bIdx * cols, cols), new WeightRef(bytes, shape, format), one);

            for (int r = 0; r < rows; r++)
            {
                expected[bIdx * rows + r] = one[r];
            }
        }

        return expected;
    }

    // The same relative-drift gate the packed-quant CUDA bands use: a miss reports both this element's drift and the widest drift the case measured.
    private static void AssertWithinBand(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float band, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

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

    private static byte[] Fixture(QuantFormat format, int rows, int cols) => format switch
    {
        QuantFormat.F32 => F32Fixture(rows, cols),
        QuantFormat.F16 => F16Fixture(rows, cols),
        _ => Bf16Fixture(rows, cols),
    };

    private static byte[] F32Fixture(int rows, int cols)
    {
        float[] values = new float[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                values[r * cols + c] = Weight(r, c) * 1.5f; // plain f32 weights in the row-major packing a dense matrix stores.
            }
        }

        byte[] bytes = new byte[values.Length * sizeof(float)];
        MemoryMarshal.AsBytes(values.AsSpan()).CopyTo(bytes);
        return bytes;
    }

    private static byte[] F16Fixture(int rows, int cols)
    {
        byte[] bytes = new byte[rows * cols * 2];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                ushort bits = BitConverter.HalfToUInt16Bits(WeightHalf(r, c)); // little-endian halves: the same bytes CpuBackend's half reader walks.
                bytes[2 * (r * cols + c)] = (byte)(bits & 0xFF);
                bytes[2 * (r * cols + c) + 1] = (byte)(bits >> 8);
            }
        }

        return bytes;
    }

    private static byte[] Bf16Fixture(int rows, int cols)
    {
        byte[] bytes = new byte[rows * cols * 2];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                ushort code = WeightBf16Code(r, c); // stored bf16 codes: the high half of an f32 pattern, exactly as both sides read them.
                bytes[2 * (r * cols + c)] = (byte)(code & 0xFF);
                bytes[2 * (r * cols + c) + 1] = (byte)(code >> 8);
            }
        }

        return bytes;
    }

    private static float Weight(int row, int col) => MathF.Sin(0.37f * row + 0.19f * col) - 0.25f;

    private static Half WeightHalf(int row, int col) => WeightHalves[(row * 5 + col) % WeightHalves.Length];

    private static ushort WeightBf16Code(int row, int col) => WeightBf16Codes[(row * 3 + col) % WeightBf16Codes.Length];
}
