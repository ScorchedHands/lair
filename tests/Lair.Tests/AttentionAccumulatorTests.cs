using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Bit-identity tests for the attention accumulator helpers that replaced the kernels' scalar per-element loops.
/// <see cref="LayerOps"/> scales and accumulates head accumulators eight lanes at a time on an AVX2 host and one
/// element at a time otherwise, and the vector path must land on the same bits the scalar loops produced: a scaling
/// pass is one rounded multiply per element, an accumulation pass is a rounded multiply followed by a rounded add —
/// never a fused multiply-add, which rounds once and can land elsewhere. Every case compares the production kernels
/// and the helpers against a scalar reference written out in this file, byte for byte, and the head dimensions cover
/// whole 8-lane groups, a pure scalar tail, and mixed tails, so the tail handling is proven at the kernel boundary
/// too. The tiled kernel is compared against its own scalar reference and against the per-row kernel on the same
/// fixture, which is the same equality the engine's prefill identity suite relies on.
/// </summary>
[Trait("Category", "Attention")]
public sealed class AttentionAccumulatorTests
{
    // Head dimensions that straddle the 8-lane vector boundary: pure tail, exact groups, mixed tails, and the
    // production shapes. None of the odd ones is a multiple of 8, so each exercises the scalar tail path.
    private static readonly int[] HeadDims = [1, 2, 4, 7, 8, 9, 12, 13, 15, 16, 20, 31, 32, 33, 64, 128];

    // Factors and weights that stress rounding: inexact reciprocals, values just above a power of two, and signed
    // magnitudes so a sign or a rounding mode flip cannot slip through a bit comparison.
    private static readonly float[] Factors = [0.33333334f, 1.0000001f, 0.5f, 2f, -0.75f, 1.4142135f, 0.11111111f];

    [Theory]
    [MemberData(nameof(HeadDimsData))]
    public void ScaleAccumulator_MatchesScalarMultiplyLoop_BitForBit(int headDim)
    {
        foreach (float factor in Factors)
        {
            float[] accumulator = Staging(headDim);
            float[] expected = Staging(headDim);

            LayerOps.ScaleAccumulator(accumulator, factor);

            for (int j = 0; j < headDim; j++)
            {
                expected[j] *= factor;
            }

            AssertBitsEqual(expected, accumulator, $"ScaleAccumulator headDim={headDim} factor={factor}");
        }
    }

    [Fact]
    public void ScaleAccumulator_SpecialFactors_LandOnScalarBits()
    {
        // Zero, negative zero, NaN, and infinities must round through the vector path exactly as the scalar loop does.
        foreach (float factor in new[] { 0f, -0f, float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            float[] accumulator = [1f, -2f, 3.5f, 7f, 0.25f, -0.5f, 11f, 13f, 17f];
            float[] expected = (float[])accumulator.Clone();

            LayerOps.ScaleAccumulator(accumulator, factor);

            for (int j = 0; j < accumulator.Length; j++)
            {
                expected[j] *= factor;
            }

            AssertBitsEqual(expected, accumulator, $"ScaleAccumulator factor={factor}");
        }
    }

    [Theory]
    [MemberData(nameof(HeadDimsData))]
    public void AccumulateWeighted_MatchesTwoRoundingScalarLoop_BitForBit(int headDim)
    {
        foreach (float weight in Factors)
        {
            float[] accumulator = Staging(headDim);
            float[] values = Values(headDim);
            float[] expected = (float[])accumulator.Clone();

            LayerOps.AccumulateWeighted(accumulator, values, weight);

            for (int j = 0; j < headDim; j++)
            {
                expected[j] += weight * values[j]; // the scalar loop's two roundings: multiply, then add
            }

            AssertBitsEqual(expected, accumulator, $"AccumulateWeighted headDim={headDim} weight={weight}");
        }
    }

    [Fact]
    public void AccumulateWeighted_KeepsTwoRoundingSemantics_WhereFusionWouldDiffer()
    {
        // A weight/value pair whose product needs more precision than a float holds, added into an accumulator that
        // cancels against it: the scalar loop rounds the product first, and the vector path must round the same way.
        // A fused multiply-add keeps the extra precision through the add and lands on a different value here.
        float weight = 1.0000001f; // just above 1: the product against these values is not exactly representable
        float[] values = [1.0000001f, -1.0000001f, 1.0000001f, -1.0000001f, 1.0000001f, -1.0000001f, 1.0000001f, -1.0000001f, 1.0000001f];
        float[] accumulator = [-1f, 1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f];
        float[] expected = (float[])accumulator.Clone();

        LayerOps.AccumulateWeighted(accumulator, values, weight);

        for (int j = 0; j < values.Length; j++)
        {
            expected[j] += weight * values[j];
        }

        AssertBitsEqual(expected, accumulator, "AccumulateWeighted two-rounding probe");

        // The probe is only meaningful if it actually moved the accumulator away from the fused result: a fused
        // multiply-add over these operands keeps the product's extra bits, so its low-order lanes differ.
        Assert.True(
            accumulator[0] != -1f || expected[0] != -1f,
            "the probe left the accumulator untouched; it no longer distinguishes two roundings from a fused multiply-add");
    }

    [Theory]
    [InlineData(12)]
    [InlineData(13)]
    [InlineData(8)]
    public void CausalAttention_FloatPath_MatchesScalarReference_BitForBit(int headDim)
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        const int headCount = 4;
        const int kvHeadCount = 2;
        const int positions = 13;

        Synthetic fixture = Synthetic.Build(headCount, kvHeadCount, headDim, positions, rows: 1);

        using BlockPool pool = new(blockTokens: 4, layerCount: 1, kvHeads: kvHeadCount, headDim, capacityBlocks: 8, backend);
        using SequenceKv kv = new(pool);
        fixture.WriteTo(kv);

        float[] production = new float[headCount * headDim];
        LayerOps.CausalAttention(
            backend, fixture.QRows.AsSpan(0, headCount * headDim), headCount, kvHeadCount, headDim, kv, layer: 0, position: positions - 1,
            attnScale: 0.125f, softcap: 0f, window: 0, production);

        float[] reference = ReferenceCausalAttention(backend, fixture, kv, headCount, kvHeadCount, headDim, positions - 1);

        AssertBitsEqual(reference, production, $"CausalAttention float path against the scalar reference at headDim={headDim}");
    }

    [Theory]
    [InlineData(12)]
    [InlineData(13)]
    public void CausalAttention_Windowed_MatchesScalarReference_BitForBit(int headDim)
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        const int headCount = 4;
        const int kvHeadCount = 2;
        const int positions = 13;
        const int window = 5;

        Synthetic fixture = Synthetic.Build(headCount, kvHeadCount, headDim, positions, rows: 1);

        using BlockPool pool = new(blockTokens: 4, layerCount: 1, kvHeads: kvHeadCount, headDim, capacityBlocks: 8, backend);
        using SequenceKv kv = new(pool);
        fixture.WriteTo(kv);

        float[] production = new float[headCount * headDim];
        LayerOps.CausalAttention(
            backend, fixture.QRows.AsSpan(0, headCount * headDim), headCount, kvHeadCount, headDim, kv, layer: 0, position: positions - 1,
            attnScale: 0.125f, softcap: 4f, window, production);

        float[] reference = ReferenceCausalAttention(backend, fixture, kv, headCount, kvHeadCount, headDim, positions - 1, window, softcap: 4f);

        AssertBitsEqual(reference, production, $"windowed CausalAttention float path against the scalar reference at headDim={headDim}");
    }

    [Theory]
    [InlineData(12)]
    [InlineData(13)]
    public void CausalAttentionTiledRows_MatchesScalarReferenceAndPerRowKernel_BitForBit(int headDim)
    {
        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        const int headCount = 4;
        const int kvHeadCount = 2;
        const int positions = 13;
        const int rows = 5;
        const int fromPosition = 8;

        Synthetic fixture = Synthetic.Build(headCount, kvHeadCount, headDim, positions, rows);

        using BlockPool pool = new(blockTokens: 4, layerCount: 1, kvHeads: kvHeadCount, headDim, capacityBlocks: 8, backend);
        using SequenceKv kv = new(pool);
        fixture.WriteTo(kv);

        float[] tiledOut = new float[rows * headCount * headDim];
        LayerOps.CausalAttentionTiledRows(
            backend, fixture.QRows, fromPosition, rows, headCount, kvHeadCount, headDim, kv, layer: 0,
            attnScale: 0.125f, softcap: 0f, window: 0,
            runningMaxRowState: new float[rows * headCount], sumExpRowState: new float[rows * headCount], tiledOut);

        float[] reference = ReferenceTiledRows(backend, fixture, kv, fromPosition, rows, headCount, kvHeadCount, headDim);

        AssertBitsEqual(reference, tiledOut, $"CausalAttentionTiledRows against its scalar reference at headDim={headDim}");

        // The same equality the engine's prefill identity suite asserts: each tiled row lands bit-for-bit on that
        // row's own per-row causal pass, which is what the tiled walk claims to reproduce.
        for (int r = 0; r < rows; r++)
        {
            float[] perRowOut = new float[headCount * headDim];
            LayerOps.CausalAttention(
                backend, fixture.QRows.AsSpan(r * headCount * headDim, headCount * headDim), headCount, kvHeadCount, headDim,
                kv, layer: 0, position: fromPosition + r, attnScale: 0.125f, softcap: 0f, window: 0, perRowOut);

            AssertBitsEqual(
                perRowOut,
                tiledOut.AsSpan(r * headCount * headDim, headCount * headDim),
                $"tiled row {r} against its own causal pass at headDim={headDim}");
        }
    }

    // ---------------------------------------------------------------- scalar references

    // The exact operation sequence the kernels ran before the accumulator loops were vectorized: the same causal
    // walk, the same dot, the same online-softmax order, and plain scalar rescale/accumulate/normalize loops.
    private static float[] ReferenceCausalAttention(
        CpuBackend backend, Synthetic fixture, SequenceKv kv, int headCount, int kvHeadCount, int headDim, int position,
        int window = 0, float softcap = 0f)
    {
        int region = headCount * headDim;
        float[] outConcat = new float[region];

        BlockSelection read = kv.SelectCausalRead(position, window);
        int blockTokens = kv.BlockTokens;

        for (int h = 0; h < headCount; h++)
        {
            ReadOnlySpan<float> qh = fixture.QRows.AsSpan(0, headCount * headDim).Slice(h * headDim, headDim);
            int kvHead = h * kvHeadCount / headCount;
            Span<float> outHead = outConcat.AsSpan(h * headDim, headDim);

            float runningMax = float.NegativeInfinity;
            float sumExp = 0f;
            outHead.Clear();

            for (int blockIndex = read.FirstBlockIndex; blockIndex <= read.LastBlockIndex; blockIndex++)
            {
                int slotFirst = Math.Max(read.FirstPosition, blockIndex * blockTokens);
                int slotLast = Math.Min(read.LastPosition, (blockIndex + 1) * blockTokens - 1);

                for (int t = slotFirst; t <= slotLast; t++)
                {
                    ReadOnlySpan<float> kt = kv.ReadK(read, t, 0).Slice(kvHead * headDim, headDim);
                    float score = backend.Dot(qh, kt) * 0.125f;
                    if (softcap > 0f)
                    {
                        score = LayerOps.Softcap(score, softcap);
                    }

                    if (score > runningMax)
                    {
                        if (!float.IsNegativeInfinity(runningMax))
                        {
                            float rescale = MathF.Exp(runningMax - score);
                            sumExp *= rescale;

                            for (int j = 0; j < headDim; j++)
                            {
                                outHead[j] *= rescale;
                            }
                        }

                        runningMax = score;
                    }

                    float weight = MathF.Exp(score - runningMax);
                    sumExp += weight;

                    ReadOnlySpan<float> vt = kv.ReadV(read, t, 0).Slice(kvHead * headDim, headDim);
                    for (int j = 0; j < headDim; j++)
                    {
                        outHead[j] += weight * vt[j];
                    }
                }
            }

            float invSum = 1f / sumExp;
            for (int j = 0; j < headDim; j++)
            {
                outHead[j] *= invSum;
            }
        }

        return outConcat;
    }

    // The tiled kernel's scalar twin: the same grouped walk, per-row online state, and scalar accumulator loops.
    private static float[] ReferenceTiledRows(
        CpuBackend backend, Synthetic fixture, SequenceKv kv, int fromPosition, int rows, int headCount, int kvHeadCount, int headDim)
    {
        int region = headCount * headDim;
        float[] outConcat = new float[rows * region];
        float[] runningMaxRowState = new float[headCount * rows];
        float[] sumExpRowState = new float[headCount * rows];

        outConcat.AsSpan(0, rows * region).Clear();

        for (int h = 0; h < headCount; h++)
        {
            for (int r = 0; r < rows; r++)
            {
                runningMaxRowState[r * headCount + h] = float.NegativeInfinity;
                sumExpRowState[r * headCount + h] = 0f;
            }
        }

        BlockSelection read = kv.SelectRangeRead(0, fromPosition + rows - 1);
        int blockTokens = kv.BlockTokens;

        for (int hStart = 0; hStart < headCount;)
        {
            int kvHead = hStart * kvHeadCount / headCount;

            int groupEnd = hStart + 1;
            while (groupEnd < headCount && groupEnd * kvHeadCount / headCount == kvHead)
            {
                groupEnd++;
            }

            for (int blockIndex = read.FirstBlockIndex; blockIndex <= read.LastBlockIndex; blockIndex++)
            {
                int slotFirst = Math.Max(read.FirstPosition, blockIndex * blockTokens);
                int slotLast = Math.Min(read.LastPosition, (blockIndex + 1) * blockTokens - 1);

                for (int t = slotFirst; t <= slotLast; t++)
                {
                    ReadOnlySpan<float> kt = kv.ReadK(read, t, 0).Slice(kvHead * headDim, headDim);
                    ReadOnlySpan<float> vt = kv.ReadV(read, t, 0).Slice(kvHead * headDim, headDim);

                    int rLow = Math.Max(0, t - fromPosition);
                    int rHigh = rows - 1;

                    for (int r = rLow; r <= rHigh; r++)
                    {
                        ReadOnlySpan<float> qRow = fixture.QRows.AsSpan(r * region, region);
                        Span<float> outHeadRow = outConcat.AsSpan(r * region, region);

                        for (int h = hStart; h < groupEnd; h++)
                        {
                            ReadOnlySpan<float> qh = qRow.Slice(h * headDim, headDim);
                            int state = r * headCount + h;

                            float score = backend.Dot(qh, kt) * 0.125f;

                            if (score > runningMaxRowState[state])
                            {
                                if (!float.IsNegativeInfinity(runningMaxRowState[state]))
                                {
                                    float rescale = MathF.Exp(runningMaxRowState[state] - score);
                                    sumExpRowState[state] *= rescale;

                                    for (int j = 0; j < headDim; j++)
                                    {
                                        outHeadRow[h * headDim + j] *= rescale;
                                    }
                                }

                                runningMaxRowState[state] = score;
                            }

                            float weight = MathF.Exp(score - runningMaxRowState[state]);
                            sumExpRowState[state] += weight;

                            for (int j = 0; j < headDim; j++)
                            {
                                outHeadRow[h * headDim + j] += weight * vt[j];
                            }
                        }
                    }
                }
            }

            hStart = groupEnd;
        }

        for (int r = 0; r < rows; r++)
        {
            Span<float> outHeadRow = outConcat.AsSpan(r * region, region);

            for (int h = 0; h < headCount; h++)
            {
                float invSum = 1f / sumExpRowState[r * headCount + h];
                for (int j = 0; j < headDim; j++)
                {
                    outHeadRow[h * headDim + j] *= invSum;
                }
            }
        }

        return outConcat;
    }

    // ---------------------------------------------------------------- fixture

    /// <summary>
    /// A synthetic attention fixture: signed, varying Q/K/V values per position and kv-head so every dot is
    /// non-trivial and every rescale branch fires, with one query set per tiled row in row-major order.
    /// </summary>
    private sealed class Synthetic
    {
        public float[] QRows { get; private set; } = [];
        public float[][] KValues { get; private set; } = [];
        public float[][] VValues { get; private set; } = [];

        public static Synthetic Build(int headCount, int kvHeadCount, int headDim, int positions, int rows)
        {
            Synthetic fixture = new();

            fixture.QRows = new float[rows * headCount * headDim];
            for (int r = 0; r < rows; r++)
            {
                for (int h = 0; h < headCount; h++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        fixture.QRows[r * headCount * headDim + h * headDim + j] = (r - 2) * 0.5f + (h - 1.5f) * 0.5f + (j % 3 - 1) * 0.25f + 1f / (j + 3);
                    }
                }
            }

            fixture.KValues = new float[positions][];
            fixture.VValues = new float[positions][];
            for (int position = 0; position < positions; position++)
            {
                float[] kVec = new float[kvHeadCount * headDim];
                float[] vVec = new float[kvHeadCount * headDim];
                for (int h = 0; h < kvHeadCount; h++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        kVec[h * headDim + j] = (h + 1) * 1.5f * (position - positions / 2 + 0.5f) + (j % 5 - 2) * 0.125f;
                        vVec[h * headDim + j] = -(h + 1) - position * 0.25f + (j % 7) * 0.0625f;
                    }
                }

                fixture.KValues[position] = kVec;
                fixture.VValues[position] = vVec;
            }

            return fixture;
        }

        public void WriteTo(SequenceKv kv)
        {
            for (int position = 0; position < KValues.Length; position++)
            {
                kv.Append();
                kv.WriteK(position, 0, KValues[position]);
                kv.WriteV(position, 0, VValues[position]);
            }
        }
    }

    // ---------------------------------------------------------------- helpers

    public static TheoryData<int> HeadDimsData() => [.. HeadDims];

    // Mantissa-heavy staging values: products with the factors below need rounding, so a fused path would show up.
    private static float[] Staging(int headDim)
    {
        float[] values = new float[headDim];
        for (int j = 0; j < headDim; j++)
        {
            values[j] = (j % 2 == 0 ? 1f : -1f) * (1f + 0.125f * (j % 5) + 1f / (j + 7));
        }

        return values;
    }

    private static float[] Values(int headDim)
    {
        float[] values = new float[headDim];
        for (int j = 0; j < headDim; j++)
        {
            values[j] = 1.0000001f * (j % 3 - 1) + 0.33333334f / (j + 2);
        }

        return values;
    }

    /// <summary>Compares two float windows on the raw bits of the floats: NaN payloads and signed zeros must match too.</summary>
    private static void AssertBitsEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        ReadOnlySpan<byte> expectedBytes = MemoryMarshal.AsBytes(expected);
        ReadOnlySpan<byte> actualBytes = MemoryMarshal.AsBytes(actual);

        for (int i = 0; i < expectedBytes.Length; i++)
        {
            Assert.True(
                expectedBytes[i] == actualBytes[i],
                $"{context}: first differing byte at {i} (float {i / 4}): expected 0x{expectedBytes[i]:X2}, got 0x{actualBytes[i]:X2}");
        }
    }
}
