using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// Near-tie argmax guard for the CUDA GEMV formats this phase newly covers: every lane builds a two-candidate weight where both rows carry byte-identical
/// packed bodies and differ only in each block's half-float scale header, so their logits are scaled copies of one oracle chain separated by exactly one
/// mantissa step at fp16 precision (~9.7e-4 relative) — close candidates whose separation is still 5x to 2000x the recorded per-format CUDA band. CPU and
/// CUDA argmax over those two logits must pick the same candidate, which holds only while each row's device drift stays inside its recorded band: a flip
/// would require drifting past the gap, far outside what these bands document. The second lane pins both rows to byte-identical candidates (a true tie):
/// there the CPU oracle answers bit-equal logits, and the test records that candidate divergence is allowed but bounded by 4x each format's band — an
/// argmax flip under a below-band gap is documented posture, not agreement. Bands are copied verbatim from CudaIQuantGemvTests and CudaBlockQuantGemvTests;
/// this file never widens one, and the CPU oracle stays the scalar chain (threads: 1, allowAvx2: false) those bands were measured against. Fixtures are a
/// handful of kilobytes of hand-built blocks — no model files, no pinned loads.
/// </summary>
[Trait("Category", "Kernel")]
[Trait("Category", "Parity")]
public sealed class CudaNearTieArgMaxTests
{
    // The recorded per-format ceilings this guard runs against, copied unchanged from the band fixtures: 2e-6 for IQ4_NL/IQ4_XS; 1e-6 for Q8_0/Q5_0/
    // Q2_0/Q2_K; 4e-6 for Q5_K; 2e-5 for Q3_K; 5e-7 for the five sub-4-bit i-quants. These are identical to CudaIQuantGemvTests.Band* and
    // CudaBlockQuantGemvTests.Band*; any change there is a measured-bands decision, not something this guard may make on its own authority.
    private const float BandIQ4Nl = 2e-6f;
    private const float BandIQ4Xs = 2e-6f;
    private const float BandIQ2XXS = 5e-7f;
    private const float BandIQ2Xs = 5e-7f;
    private const float BandIQ2S = 5e-7f;
    private const float BandIQ3Xxs = 5e-7f;
    private const float BandIQ3S = 5e-7f;
    private const float BandQ8_0 = 1e-6f;
    private const float BandQ5_0 = 1e-6f;
    private const float BandQ2_0 = 1e-6f;
    private const float BandQ5K = 4e-6f;
    private const float BandQ2K = 1e-6f;
    private const float BandQ3K = 2e-5f;

    // The candidate pair's scale headers: adjacent finite half-float bit patterns at exponent -1 (0.5 and one mantissa step above it). Every LAIR GEMV
    // chain multiplies its whole row by the block header d read from a fixed offset, so these two candidates differ in logit space by exactly that one
    // mantissa step — about 9.74e-4 relative to each candidate's magnitude. Formats with an independent min half (Q5_K at bytes 2..3, Q2_K at 82..83) get a
    // +0f there so the shift term vanishes and the scaling stays strictly linear in d; no other header position is read as a half-float.
    private const ushort NearTieHeaderBitsLow = 0x3800;   // (Half)0.5f
    private const ushort NearTieHeaderBitsHigh = 0x3801;  // one mantissa step above it: the closest representable candidate separation at this exponent
    private const ushort ZeroMinHeaderBits = 0x0000;      // +0f: kills the min-shift term in Q5_K/Q2_K so logit scaling is exact

    // ---------------------------------------------------------------- per-format tables.

    private static float BandFor(QuantFormat format) => format switch
    {
        QuantFormat.IQ4_NL => BandIQ4Nl,
        QuantFormat.IQ4_XS => BandIQ4Xs,
        QuantFormat.IQ2_XXS => BandIQ2XXS,
        QuantFormat.IQ2_XS => BandIQ2Xs,
        QuantFormat.IQ2_S => BandIQ2S,
        QuantFormat.IQ3_XXS => BandIQ3Xxs,
        QuantFormat.IQ3_S => BandIQ3S,
        QuantFormat.Q8_0 => BandQ8_0,
        QuantFormat.Q5_0 => BandQ5_0,
        QuantFormat.Q2_0 => BandQ2_0,
        QuantFormat.Q5_K => BandQ5K,
        QuantFormat.Q2_K => BandQ2K,
        QuantFormat.Q3_K => BandQ3K,
        _ => throw new ArgumentException($"{format} is not a newly covered CUDA GEMV format", nameof(format)),
    };

    private static int BlockColumns(QuantFormat format) => format switch
    {
        QuantFormat.Q8_0 or QuantFormat.Q5_0 or QuantFormat.IQ4_NL => 32,
        QuantFormat.Q2_0 => 64,
        _ => 256, // IQ4_XS, the five sub-4-bit i-quants, and Q5_K/Q2_K/Q3_K run in whole 256-column super-blocks.
    };

    private static int CandidateColumns(QuantFormat format) => BlockColumns(format) switch
    {
        32 => 100, // three whole blocks plus a partial tail block for the 32-element families ...
        64 => 196, // ... and two whole Q2_0 blocks with four columns into the third.
        _ => 580,  // two whole super-blocks plus twelve columns into the third: every format also exercises its partial tail block here.
    };

    private static bool HasMinHalf(QuantFormat format) => format is QuantFormat.Q5_K or QuantFormat.Q2_K;

    // Where each newly covered format reads its per-block scale half, straight from CpuBackend's scalar chains: Q3_K keeps d at bytes 108..109 and
    // Q2_K carries its halves as a pair at the block tail (d at 80..81); every other format here takes it from byte 0 of its block.
    private static int ScaleHeaderOffset(QuantFormat format) => format switch
    {
        QuantFormat.Q3_K => 108,
        QuantFormat.Q2_K => 80,
        _ => 0,
    };

    private static int MinHalfOffset(QuantFormat format) => format switch
    {
        QuantFormat.Q5_K => 2,   // LAIR's Q5_K keeps d at bytes 0..1 and the min half right behind it.
        QuantFormat.Q2_K => 82,  // LAIR's Q2_K carries its halves as a pair at the block tail.
        _ => throw new ArgumentException($"{format} has no independent min half", nameof(format)),
    };

    private static readonly (QuantFormat Format, float Band)[] NewlyCoveredFormats =
    [
        (QuantFormat.Q8_0, BandQ8_0),
        (QuantFormat.Q5_0, BandQ5_0),
        (QuantFormat.Q2_0, BandQ2_0),
        (QuantFormat.Q5_K, BandQ5K),
        (QuantFormat.Q2_K, BandQ2K),
        (QuantFormat.Q3_K, BandQ3K),
        (QuantFormat.IQ4_NL, BandIQ4Nl),
        (QuantFormat.IQ4_XS, BandIQ4Xs),
        (QuantFormat.IQ2_XXS, BandIQ2XXS),
        (QuantFormat.IQ2_XS, BandIQ2Xs),
        (QuantFormat.IQ2_S, BandIQ2S),
        (QuantFormat.IQ3_XXS, BandIQ3Xxs),
        (QuantFormat.IQ3_S, BandIQ3S),
    ];

    // ---------------------------------------------------------------- fixtures.

    private static float Activation(int c) => (float)Math.Sin(0.11 * c + 1.7) * 1.3f;

    private static float[] Activations(int cols)
    {
        float[] x = new float[cols];
        for (int c = 0; c < cols; c++)
        {
            x[c] = Activation(c);
        }

        return x;
    }

    // Two candidate rows sharing one swept packed body — every byte of every block walks a value range across blocks exactly like the band fixtures do,
    // and both rows read identical codes, scales, grid indices and sign planes. Only the header halves differ per row, written at each format's fixed d
    // offset (and +0f min where one exists), so the candidate logits are scaled copies of a single oracle chain with no other asymmetry between them.
    private static byte[] BuildCandidateBytes(QuantFormat format, int cols, ushort headerBitsRowA, ushort headerBitsRowB)
    {
        long strideBytes = new TensorShape([cols]).RowStrideBytes(format); // RowStrideBytes reads only the column count.

        int blocksPerRow = (cols + BlockColumns(format) - 1) / BlockColumns(format);
        int blockBytes = checked((int)(strideBytes / blocksPerRow)); // every row stride here is exactly blocksPerRow whole blocks.
        byte[] bytes = new byte[checked(2 * blocksPerRow * blockBytes)];

        for (int r = 0; r < 2; r++)
        {
            ushort headerBits = r == 0 ? headerBitsRowA : headerBitsRowB;

            for (int b = 0; b < blocksPerRow; b++)
            {
                int block = (r * blocksPerRow + b) * blockBytes;

                for (int i = 2; i < blockBytes; i++)
                {
                    bytes[block + i] = unchecked((byte)(i * 13 + b * 53)); // row-invariant sweep: both candidates read the same body.
                }

                WriteHalfLE(bytes, block + ScaleHeaderOffset(format), headerBits); // d at each format's fixed offset; the swept values there are overwritten last.

                if (HasMinHalf(format))
                {
                    WriteHalfLE(bytes, block + MinHalfOffset(format), ZeroMinHeaderBits);
                }
            }
        }

        return bytes;
    }

    private static void WriteHalfLE(byte[] dst, int offset, ushort bits)
    {
        dst[offset] = (byte)bits;
        dst[offset + 1] = (byte)(bits >> 8);
    }

    // ---------------------------------------------------------------- lanes.

    [Fact]
    public void NearTieGuardTables_CoverEveryNewlyCoveredGEMVFormatOnce()
    {
        // The guard's own format table and the per-lane band lookup must agree, and every listed format is one the CUDA seam claims a GEMV kernel for:
        // this class exists exactly to pin argmax posture over that newly covered set.
        foreach ((QuantFormat format, float band) in NewlyCoveredFormats)
        {
            Assert.Equal(band, BandFor(format));
            Assert.True(CudaBackend.HasGemvKernel(format), $"the CUDA seam stopped reporting its {format} GEMV kernel this guard is written for");
        }

        // Q4_K and Q6_K are deliberately absent: they were already covered before this phase's GEMV expansion, so their lanes live in the earlier fixtures.
        Assert.DoesNotContain(NewlyCoveredFormats, entry => entry.Format == QuantFormat.Q4_K);
        Assert.DoesNotContain(NewlyCoveredFormats, entry => entry.Format == QuantFormat.Q6_K);
    }

    [Theory]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q2_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ2_S)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ3_S)]
    public void CudaNearTieArgMax_CpuAndCudaPickTheSameCandidate_GapStaysAboveRecordedBand(QuantFormat format)
    {
        float band = BandFor(format);
        int cols = CandidateColumns(format);

        byte[] bytes = BuildCandidateBytes(format, cols, NearTieHeaderBitsLow, NearTieHeaderBitsHigh);
        TensorShape shape = new([2, cols]);
        WeightRef w = new(bytes, shape, format);

        float[] x = Activations(cols);

        using CpuBackend cpu = new(threads: 1, allowAvx2: false); // the scalar oracle chain these bands were measured against.
        float[] expected = new float[2];
        cpu.Gemv(x, w, expected);

        int winner = ArgMax(expected);
        int runnerUp = 1 - winner;
        float magnitude = Math.Max(Math.Abs(expected[winner]), Math.Abs(expected[runnerUp]));
        Assert.True(magnitude > 0f, $"cuda near-tie guard {format}: the candidate logits cancelled to zero (expected {expected[0]:G9}, {expected[1]:G9}); pick a fixture whose oracle sum is non-zero");

        float gap = Math.Abs(expected[winner] - expected[runnerUp]);
        // The separation must clear this format's recorded band with room — an argmax flip then needs drift past the whole gap, not inside the documented
        // ceiling — yet stay a near tie: one fp16 mantissa step apart is under 0.2% of candidate magnitude, far below any quality-relevant logit distance.
        Assert.True(
            gap > 8f * band * magnitude,
            $"cuda near-tie guard {format}: candidates are only {gap / magnitude:G9} relative apart — not the documented one-mantissa-step separation (>= {8f * band:G9}, i.e. 8x the recorded band {band:G9})");
        Assert.True(
            gap <= 0.002f * magnitude,
            $"cuda near-tie guard {format}: candidates sit {gap / magnitude:G9} relative apart — wider than a near tie (<= 0.002); the header pair no longer builds close logits");

        using CudaBackend cuda = OpenCuda();
        Assert.True(cuda.SupportsGemv(format), $"the CUDA seam stopped reporting its {format} GEMV kernel this guard depends on");

        float[] actual = new float[2];
        cuda.Gemv(x, w, actual);

        // Both candidates land inside their recorded band against the oracle — the documented divergence posture this guard is about. The bands are not
        // widened here: they are the same constants CudaIQuantGemvTests and CudaBlockQuantGemvTests pin for these formats.
        AssertWithinBand(expected, actual, band, $"cuda {format} near-tie candidates");

        int cudaWinner = ArgMax(actual);
        Assert.Equal(winner, cudaWinner); // the whole point: with drift inside the bands and a gap above them, CPU and CUDA argmax cannot disagree.
    }

    [Theory]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q2_0)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    [InlineData(QuantFormat.IQ4_NL)]
    [InlineData(QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.IQ2_XXS)]
    [InlineData(QuantFormat.IQ2_XS)]
    [InlineData(QuantFormat.IQ2_S)]
    [InlineData(QuantFormat.IQ3_XXS)]
    [InlineData(QuantFormat.IQ3_S)]
    public void CudaExactTieCandidates_RecordedBandBoundsAnyAllowedDivergence(QuantFormat format)
    {
        float band = BandFor(format);
        int cols = CandidateColumns(format);

        byte[] bytes = BuildCandidateBytes(format, cols, NearTieHeaderBitsLow, NearTieHeaderBitsLow); // both headers identical: a true tie.
        TensorShape shape = new([2, cols]);
        WeightRef w = new(bytes, shape, format);

        float[] x = Activations(cols);

        using CpuBackend cpu = new(threads: 1, allowAvx2: false);
        float[] expected = new float[2];
        cpu.Gemv(x, w, expected);
        Assert.Equal(expected[0], expected[1]); // byte-identical candidates through the scalar oracle answer bit-equal logits.

        using CudaBackend cuda = OpenCuda();
        Assert.True(cuda.SupportsGemv(format), $"the CUDA seam stopped reporting its {format} GEMV kernel this guard depends on");

        float[] actual = new float[2];
        cuda.Gemv(x, w, actual);
        AssertWithinBand(expected, actual, band, $"cuda {format} exact-tie candidates");

        // Below-band posture: with a zero candidate gap an argmax flip is allowed behavior — but only inside the documented ceiling. Each row sits within
        // its band against the oracle, so the two device answers can no longer drift more than 4x that band relative to each other; this lane pins exactly
        // that envelope and records it as allowed divergence rather than demanding agreement (or faking it).
        float pairDistance = Math.Abs(actual[0] - actual[1]);
        float scale = Math.Max(Math.Abs(actual[0]), Math.Abs(actual[1]));
        bool bounded = scale == 0f ? pairDistance == 0f : pairDistance <= 4f * band * scale;
        Assert.True(
            bounded,
            $"cuda {format} exact-tie candidates drifted {pairDistance:G9} absolute ({(scale == 0f ? pairDistance : pairDistance / scale):G9} relative) past the allowed below-band envelope of 4x the recorded band {band:G9}; candidate divergence under a tie must stay inside documented posture");
    }

    // ---------------------------------------------------------------- shared helpers.

    private static CudaBackend OpenCuda()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        try
        {
            return new CudaBackend();
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

    private static void AssertWithinBand(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float band, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        // Same relative-drift shape as every recorded band in this repo: |expected - actual| <= band * max(|expected|, |actual|), per element. The failure
        // message names the widest drift in the case so a run that misses its recorded band reports how far the device chain actually moved.
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

    /// <summary>Picks the top id by descending logit with a strict "greater than" scan: equal logits keep the lower id, so tie posture is deterministic on both sides.</summary>
    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        float best = float.NegativeInfinity;
        int bestIdx = -1;

        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > best)
            {
                best = logits[i];
                bestIdx = i;
            }
        }

        Assert.True(bestIdx >= 0, "logit vector had no selectable entry");
        return bestIdx;
    }
}
