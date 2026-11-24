using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Layer-level coverage of the partial rotary composition the bounded qwen4exp forward runs for a target that declares
/// fewer rotary dimensions than its heads carry (the pinned Qwen3.8-Flash-Next shape: 256-wide head/key/value regions with
/// <c>rope.dimension_count = 64</c>). The gates read the same operations the engine composes — angle tables from
/// <see cref="LayerOps.BuildRopeSliceTables"/> and rotations through <see cref="QsaOps.RotateAtPosition"/>, which is what every
/// q/k head and indexer component of a sparse-full block goes through — so they pin what rotates, at which frequencies, and
/// what stays untouched: pairs inside the declared slice rotate, every dimension outside it passes through bit-for-bit, a
/// narrowed table keeps the full-head frequency spacing for the pairs that remain, and a declaration covering the whole
/// region builds exactly the table a full-head build builds. The shared-expert gate seam is here too because both qwen4exp
/// forward paths read it: the family arm composes plain sigmoid over its selector score while every other routed family
/// keeps silu(score) / score unchanged, including its 0-by-0 at a zero score.
/// </summary>
[Trait("Category", "Layers")]
public sealed class Qwen4ExpRopeSliceTests
{
    private const double RopeBase = 10_000.0;

    // A head region wider than the slice it rotates: eight dimensions carrying two pairs of a four-dimension slice.
    private const int HeadDim = 8;
    private const int SliceDimensions = 4;
    private const int HalfSlice = SliceDimensions / 2;
    private const int MaxPositions = 3;

    [Fact]
    public void BuildRopeTables_FullWidthDeclaration_IsBitIdenticalToTheFullHeadTable()
    {
        float[] fullCos = new float[MaxPositions * HeadDim / 2];
        float[] fullSin = new float[MaxPositions * HeadDim / 2];
        LayerOps.BuildRopeTables(RopeBase, HeadDim, MaxPositions, fullCos, fullSin);

        float[] declaredCos = new float[MaxPositions * HeadDim / 2];
        float[] declaredSin = new float[MaxPositions * HeadDim / 2];
        LayerOps.BuildRopeSliceTables(RopeBase, HeadDim, ropeDimensions: HeadDim, MaxPositions, declaredCos, declaredSin);

        Assert.True(BitsEqual(fullCos, declaredCos), "a slice declaration covering the whole head must build today's full-head table");
        Assert.True(BitsEqual(fullSin, declaredSin), "a slice declaration covering the whole head must build today's full-head table");
    }

    [Fact]
    public void BuildRopeTables_NarrowedSlice_KeepsTheFullHeadFrequencySpacingForItsPairs()
    {
        float[] fullCos = new float[MaxPositions * HeadDim / 2];
        float[] fullSin = new float[MaxPositions * HeadDim / 2];
        LayerOps.BuildRopeTables(RopeBase, HeadDim, MaxPositions, fullCos, fullSin);

        float[] sliceCos = new float[MaxPositions * HalfSlice];
        float[] sliceSin = new float[MaxPositions * HalfSlice];
        LayerOps.BuildRopeSliceTables(RopeBase, HeadDim, SliceDimensions, MaxPositions, sliceCos, sliceSin);

        // Narrowing truncates which pairs exist; it does not re-space the frequencies of the ones that remain. Pair i of a
        // narrowed table is pair i of the full-head table because both derive their angle from the whole-head denominator —
        // what the reference reads when n_rot falls below the head dimension, and what the YaRN tables already do under a variant.
        for (int pos = 0; pos < MaxPositions; pos++)
        {
            Assert.True(
                BitsEqual(fullCos.AsSpan(pos * HeadDim / 2, HalfSlice), sliceCos.AsSpan(pos * HalfSlice, HalfSlice)),
                $"cosine angles at position {pos} must match the full-head table entry for entry");
            Assert.True(
                BitsEqual(fullSin.AsSpan(pos * HeadDim / 2, HalfSlice), sliceSin.AsSpan(pos * HalfSlice, HalfSlice)),
                $"sine angles at position {pos} must match the full-head table entry for entry");
        }

        // The narrowed table carries only the slice's pairs: nothing beyond them exists to rotate.
        Assert.Equal(MaxPositions * HalfSlice, sliceCos.Length);
    }

    [Theory]
    [InlineData(3)] // an odd slice cannot form pairs
    [InlineData(9)] // a slice wider than its region has dimensions this table never had
    public void BuildRopeTables_InvalidSliceDeclaration_FailsNamingTheDeclaredSlice(int declared)
    {
        ArgumentException ex = Assert.Throws<ArgumentException>(() => LayerOps.BuildRopeSliceTables(
            RopeBase, HeadDim, declared, MaxPositions, new float[MaxPositions * 4], new float[MaxPositions * 4]));

        Assert.Contains(declared.ToString(), ex.Message);
    }

    [Fact]
    public void BuildRopeTables_ZeroWideSlice_FailsAsAnOutOfRangeDeclaration()
    {
        ArgumentOutOfRangeException ex = Assert.Throws<ArgumentOutOfRangeException>(() => LayerOps.BuildRopeSliceTables(
            RopeBase, HeadDim, ropeDimensions: 0, MaxPositions, new float[MaxPositions * 4], new float[MaxPositions * 4]));

        Assert.Equal("ropeDimensions", ex.ParamName);
    }

    [Fact]
    public void Qwen4ExpHeadRotation_DeclaredSlice_RotatesTheLeadingDimensionsAndPassesTheRestThrough()
    {
        float[] cos = new float[MaxPositions * HalfSlice];
        float[] sin = new float[MaxPositions * HalfSlice];
        LayerOps.BuildRopeSliceTables(RopeBase, HeadDim, SliceDimensions, MaxPositions, cos, sin);

        for (int position = 1; position < MaxPositions; position++) // position 0 is the identity rotation by construction
        {
            float[] input = HeadRegion(seed: position * 31);
            float[] head = [.. input];

            QsaOps.RotateAtPosition(head, ropeHalf: HalfSlice, cos, sin, position); // the engine's own call shape for a q/k head region

            ReadOnlySpan<float> cosAt = cos.AsSpan(position * HalfSlice, HalfSlice);
            ReadOnlySpan<float> sinAt = sin.AsSpan(position * HalfSlice, HalfSlice);

            bool anyPairMoved = false;
            for (int i = 0; i < HalfSlice; i++)
            {
                float c = cosAt[i];
                float s = sinAt[i];
                float lo = input[2 * i];
                float hi = input[2 * i + 1];

                Assert.Equal(lo * c - hi * s, head[2 * i]); // the adjacent-pair composition over this position's table angles
                Assert.Equal(hi * c + lo * s, head[2 * i + 1]);

                anyPairMoved |= BitConverter.SingleToInt32Bits(head[2 * i]) != BitConverter.SingleToInt32Bits(lo);
            }

            Assert.True(anyPairMoved, $"a nonzero angle at position {position} must actually rotate the slice");

            // The dimensions outside the slice read back exactly as they entered.
            for (int d = SliceDimensions; d < HeadDim; d++)
            {
                Assert.True(
                    BitConverter.SingleToInt32Bits(head[d]) == BitConverter.SingleToInt32Bits(input[d]),
                    $"head dimension {d} sits outside the declared slice and must pass through unchanged, got {head[d]:G9} against {input[d]:G9}");
            }
        }
    }

    [Fact]
    public void Qwen4ExpIndexerRotation_DeclaredSlice_NarrowsWiderComponentsAndLeavesTheirTailUntouched()
    {
        // A component region wider than the declared slice: six dimensions, four of them rotated. This is the engine's own
        // narrowing — a declared slice reaches an indexer component only as far as that region carries it.
        const int ComponentWidth = 6;

        float[] cos = new float[MaxPositions * HalfSlice];
        float[] sin = new float[MaxPositions * HalfSlice];
        LayerOps.BuildRopeSliceTables(RopeBase, ComponentWidth, SliceDimensions, MaxPositions, cos, sin);

        for (int position = 1; position < MaxPositions; position++)
        {
            float[] input = [.. HeadRegion(seed: position * 17).AsSpan(0, ComponentWidth)];
            float[] component = [.. input];

            QsaOps.RotateAtPosition(component, ropeHalf: HalfSlice, cos, sin, position);

            bool anyPairMoved = false;
            for (int i = 0; i < HalfSlice; i++)
            {
                float c = cos[position * HalfSlice + i];
                float s = sin[position * HalfSlice + i];

                Assert.Equal(input[2 * i] * c - input[2 * i + 1] * s, component[2 * i]);
                Assert.Equal(input[2 * i + 1] * c + input[2 * i] * s, component[2 * i + 1]);

                anyPairMoved |= BitConverter.SingleToInt32Bits(component[2 * i]) != BitConverter.SingleToInt32Bits(input[2 * i]);
            }

            Assert.True(anyPairMoved, $"a nonzero angle at position {position} must actually rotate the indexer slice");

            for (int d = SliceDimensions; d < ComponentWidth; d++)
            {
                Assert.True(
                    BitConverter.SingleToInt32Bits(component[d]) == BitConverter.SingleToInt32Bits(input[d]),
                    $"indexer dimension {d} sits outside the declared slice and must pass through unchanged");
            }
        }

        // A component region narrower than the declared slice is rotated whole: the slice narrows to what the region carries,
        // so no indexer dimension a file ships goes unrotated for want of width.
        const int NarrowComponent = 2;
        float[] narrowCos = new float[MaxPositions * NarrowComponent / 2];
        float[] narrowSin = new float[MaxPositions * NarrowComponent / 2];
        LayerOps.BuildRopeSliceTables(RopeBase, NarrowComponent, ropeDimensions: Math.Min(SliceDimensions, NarrowComponent), MaxPositions, narrowCos, narrowSin);

        for (int position = 1; position < MaxPositions; position++)
        {
            float[] input = [.. HeadRegion(seed: position * 23).AsSpan(0, NarrowComponent)];
            float[] component = [.. input];

            QsaOps.RotateAtPosition(component, ropeHalf: NarrowComponent / 2, narrowCos, narrowSin, position);

            for (int i = 0; i < NarrowComponent / 2; i++)
            {
                float c = narrowCos[position * NarrowComponent / 2 + i];
                float s = narrowSin[position * NarrowComponent / 2 + i];

                Assert.Equal(input[2 * i] * c - input[2 * i + 1] * s, component[2 * i]);
                Assert.NotEqual(BitConverter.SingleToInt32Bits(component[2 * i]), BitConverter.SingleToInt32Bits(input[2 * i]));
            }
        }
    }

    [Fact]
    public void SharedExpertGate_Qwen4ExpArm_IsPlainSigmoidFiniteAtZero_AndOtherRoutedFamiliesKeepSiluOverScore()
    {
        // The qwen4exp arm reads the reference gate: plain sigmoid over its hidden-sized selector score, finite everywhere and
        // exactly 0.5 at a zero score — where a qwen2moe-shaped block's composition divides 0 by 0.
        Assert.Equal(0.5f, LayerOps.SharedExpertGateWeight(plainSigmoidFamily: true, score: 0f));

        foreach (float score in new[] { -3.5f, -0.75f, 0.25f, 1.5f, 40f })
        {
            Assert.Equal(1f / (1f + MathF.Exp(-score)), LayerOps.SharedExpertGateWeight(plainSigmoidFamily: true, score));
            Assert.True(float.IsFinite(LayerOps.SharedExpertGateWeight(plainSigmoidFamily: true, score)));

            // Every other routed family keeps the pinned qwen2moe composition bit-for-bit.
            int pinned = BitConverter.SingleToInt32Bits(LayerOps.Silu(score) / score);
            Assert.Equal(pinned, BitConverter.SingleToInt32Bits(LayerOps.SharedExpertGateWeight(plainSigmoidFamily: false, score)));
        }

        Assert.True(float.IsNaN(LayerOps.SharedExpertGateWeight(plainSigmoidFamily: false, score: 0f)), "the qwen2moe arm divides 0 by 0 exactly as its reference graph does");
    }

    // ---------------------------------------------------------------- helpers

    private static float[] HeadRegion(int seed)
    {
        float[] values = new float[HeadDim];
        for (int i = 0; i < HeadDim; i++)
        {
            values[i] = MathF.Sin((seed * 7 + i) * 0.37f) + 1.5f; // finite magnitudes at unit scale, never zero
        }

        return values;
    }

    private static bool BitsEqual(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        if (left.Length != right.Length)
        {
            return false;
        }

        for (int i = 0; i < left.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(left[i]) != BitConverter.SingleToInt32Bits(right[i]))
            {
                return false;
            }
        }

        return true;
    }
}
