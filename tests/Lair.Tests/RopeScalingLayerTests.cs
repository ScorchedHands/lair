using System.Text;

using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Forward-level goldens for the resolved rope scaling and rotary slice: a YaRN angle table matches a
/// double-precision scalar oracle that blends each pair's interpolated angle toward the unscaled one across the
/// correction band and carries the variant's magnitude, a unit multiplier with no log multiplier reproduces the
/// standard tables bit-for-bit, a partial slice rotates only the leading dimensions in both pair layouts, and a
/// synthetic model carrying declared YaRN keys forwards to logits that match the oracle forward — including its
/// score-side attention factor — while demonstrably differing from the same weights with no scaling declared. A
/// narrowed slice leaves the untouched head dimensions bit-identical through the full forward, batched decode is
/// bit-identical to single-token decode under a variant and a slice, and an unscaled model decodes exactly as it
/// did before this resolution existed. Engine-to-engine comparisons are asserted bit-for-bit; oracle comparisons
/// use the double-precision tolerance shared by the strategy goldens.
/// </summary>
[Trait("Category", "Layers")]
public sealed class RopeScalingLayerTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

    // The synthetic model's rope geometry: base 10000, a doubled context, and the original length the variant
    // interpolates against. With the reference default betas (32 fast / 1 slow) on a head dimension of 4 the
    // correction band covers pair indices [0..2], so pair 0 keeps its unscaled angle, pair 1 blends halfway, and
    // the variant's magnitude scales every table entry.
    private const double RopeBase = 10_000.0;
    private const double YarnMultiplier = 2.0;
    private const int OriginalContextLength = 1024;

    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
            }
        }

        _tempFiles.Clear();
    }

    // ---------------------------------------------------------------- angle tables

    [Fact]
    public void BuildRopeTables_UnitMultiplier_MatchesStandardTablesBitForBit_AndUnitCorrectionKeepsInterpolatedAngles()
    {
        const int HeadDim = 4;
        const int Positions = 5;

        float[] standardCos = new float[Positions * HeadDim / 2];
        float[] standardSin = new float[Positions * HeadDim / 2];
        LayerOps.BuildRopeTables(RopeBase, HeadDim, Positions, standardCos, standardSin);

        // A unit multiplier with the correction blend active interpolates toward the unscaled angles, which is
        // the standard rotation itself, and a unit attention factor leaves the magnitudes untouched: the tables
        // must come out bit-for-bit identical.
        float[] yarnCos = new float[Positions * HeadDim / 2];
        float[] yarnSin = new float[Positions * HeadDim / 2];
        LayerOps.BuildRopeScalingTables(
            RopeBase, HeadDim, HeadDim, Positions,
            multiplier: 1.0, originalContext: OriginalContextLength, attentionFactor: 1.0,
            correctionScale: 1.0, betaFast: 32.0, betaSlow: 1.0,
            yarnCos, yarnSin);

        Assert.True(BitsEqual(yarnCos, standardCos), "a unit-multiplier YaRN table must match the standard cos table bit-for-bit");
        Assert.True(BitsEqual(yarnSin, standardSin), "a unit-multiplier YaRN table must match the standard sin table bit-for-bit");

        // With the correction scale disabled the variant degenerates to linear interpolation: every angle is the
        // unscaled angle divided by the multiplier, and the magnitude is the attention factor alone.
        float[] linearCos = new float[Positions * HeadDim / 2];
        float[] linearSin = new float[Positions * HeadDim / 2];
        LayerOps.BuildRopeScalingTables(
            RopeBase, HeadDim, HeadDim, Positions,
            multiplier: 2.0, originalContext: OriginalContextLength, attentionFactor: 1.5,
            correctionScale: 0.0, betaFast: 32.0, betaSlow: 1.0,
            linearCos, linearSin);

        for (int pos = 0; pos < Positions; pos++)
        {
            for (int i = 0; i < HeadDim / 2; i++)
            {
                double angle = pos * Math.Pow(RopeBase, -2.0 * i / HeadDim) / YarnMultiplier;
                int entry = pos * HeadDim / 2 + i;
                AssertEntry($"interpolated cos[pos {pos}][{i}]", linearCos[entry], Math.Cos(angle) * 1.5);
                AssertEntry($"interpolated sin[pos {pos}][{i}]", linearSin[entry], Math.Sin(angle) * 1.5);
            }
        }
    }

    [Fact]
    public void BuildRopeScalingTables_YarnGolden_MatchesTheReferenceBlendAndMagnitude_OverTheDeclaredSlice()
    {
        const int HeadDim = 4;
        const int Slice = 2; // only the leading pair rotates
        const int Positions = 6;

        double attentionFactor = ResolvedAttentionFactor(YarnMultiplier, declaredAttention: 1.0, logMultiplier: 0.0);
        double tableMagnitude = attentionFactor * (1.0 + 0.1 * Math.Log(YarnMultiplier)); // the blend's own magnitude

        float[] cos = new float[Positions * Slice / 2];
        float[] sin = new float[Positions * Slice / 2];
        LayerOps.BuildRopeScalingTables(
            RopeBase, HeadDim, Slice, Positions,
            YarnMultiplier, OriginalContextLength, attentionFactor,
            correctionScale: 1.0, betaFast: 32.0, betaSlow: 1.0,
            cos, sin);

        double corrLow = Math.Max(0.0, Math.Floor(CorrDim(32.0, HeadDim, OriginalContextLength, RopeBase)));
        double corrHigh = Math.Min(HeadDim - 1.0, Math.Ceiling(CorrDim(1.0, HeadDim, OriginalContextLength, RopeBase)));
        double bandWidth = Math.Max(0.001, corrHigh - corrLow);

        // The slice narrows the table to its pairs; the band still spans the full head's pair indices.
        Assert.True(corrLow == 0.0 && corrHigh == 2.0, $"band [0..2] expected for beta 32/1 on head dim {HeadDim}, got [{corrLow}..{corrHigh}]");

        for (int pos = 0; pos < Positions; pos++)
        {
            for (int i = 0; i < Slice / 2; i++)
            {
                double unscaled = pos * Math.Pow(RopeBase, -2.0 * i / HeadDim);
                double interp = unscaled / YarnMultiplier;
                double ramp = 1.0 - Math.Min(1.0, Math.Max(0.0, (i - corrLow) / bandWidth));
                double mix = ramp * 1.0; // the reference default correction scale
                double angle = interp * (1.0 - mix) + unscaled * mix;

                int entry = pos * Slice / 2 + i;
                AssertEntry($"yarn cos[pos {pos}][{i}]", cos[entry], Math.Cos(angle) * tableMagnitude);
                AssertEntry($"yarn sin[pos {pos}][{i}]", sin[entry], Math.Sin(angle) * tableMagnitude);
            }
        }

        // A declared log multiplier changes only the attention factor, never the angles: with it the resolved
        // factor divides out the all-dimensions magnitude scale instead.
        double withLogMul = ResolvedAttentionFactor(YarnMultiplier, declaredAttention: 1.0, logMultiplier: 0.7);
        Assert.True(
            Math.Abs(withLogMul - attentionFactor / (ResolvedMscale(YarnMultiplier, 0.7))) <= 1e-9,
            $"log-multiplier factor {withLogMul:G9} vs plain {attentionFactor:G9}");

        // A slice wider than the head or an odd slice is a table-build argument error, mirroring the load gate.
        Assert.Throws<ArgumentException>(() => LayerOps.BuildRopeScalingTables(
            RopeBase, HeadDim, HeadDim + 2, Positions, YarnMultiplier, OriginalContextLength, attentionFactor, 1.0, 32.0, 1.0, cos, sin));
        Assert.Throws<ArgumentException>(() => LayerOps.BuildRopeScalingTables(
            RopeBase, HeadDim, 3, Positions, YarnMultiplier, OriginalContextLength, attentionFactor, 1.0, 32.0, 1.0, cos, sin));
    }

    [Fact]
    public void ApplyRope_PartialSlice_RotatesOnlyTheLeadingDimensions_InBothPairings()
    {
        const int SliceHalf = 2; // two pairs: dimensions 0..3 rotate out of six, and 4 and 5 pass through

        float[] cos = [MathF.Cos(0.7f), MathF.Cos(1.3f)];
        float[] sin = [MathF.Sin(0.7f), MathF.Sin(1.3f)];

        // Adjacent pairing: pair i occupies dimensions 2i and 2i + 1 of the head region.
        float[] adjacent = [0.5f, -0.25f, 0.8f, -0.6f, 0.4f, -0.9f];
        LayerOps.ApplyRopeToHead(adjacent, SliceHalf, cos, sin);

        Assert.True(
            BitConverter.SingleToInt32Bits(adjacent[4]) == BitConverter.SingleToInt32Bits(0.4f) &&
            BitConverter.SingleToInt32Bits(adjacent[5]) == BitConverter.SingleToInt32Bits(-0.9f),
            "an adjacent-pair slice must leave the dimensions beyond it bit-identical");

        AssertEntry("adjacent pair 0 low", adjacent[0], 0.5 * cos[0] - (-0.25) * sin[0]);
        AssertEntry("adjacent pair 0 high", adjacent[1], -0.25 * cos[0] + 0.5 * sin[0]);
        AssertEntry("adjacent pair 1 low", adjacent[2], 0.8 * cos[1] - (-0.6) * sin[1]);
        AssertEntry("adjacent pair 1 high", adjacent[3], -0.6 * cos[1] + 0.8 * sin[1]);

        // Rotate-half pairing inside the slice: dimension i pairs with i + SliceHalf, so dimensions beyond the rotated region stay untouched.
        float[] split = [0.5f, -0.25f, 0.8f, -0.6f, 0.4f, -0.9f];
        LayerOps.ApplyRopeToSplitHalfHead(split, SliceHalf, cos, sin);

        Assert.True(
            BitConverter.SingleToInt32Bits(split[4]) == BitConverter.SingleToInt32Bits(0.4f) &&
            BitConverter.SingleToInt32Bits(split[5]) == BitConverter.SingleToInt32Bits(-0.9f),
            "a rotate-half slice must leave the dimensions beyond its rotated region bit-identical");

        AssertEntry("split pair 0 low", split[0], 0.5 * cos[0] - 0.8 * sin[0]);
        AssertEntry("split pair 1 low", split[1], -0.25 * cos[1] - (-0.6) * sin[1]);
        AssertEntry("split pair 0 high", split[2], 0.8 * cos[0] + 0.5 * sin[0]);
        AssertEntry("split pair 1 high", split[3], -0.6 * cos[1] + (-0.25) * sin[1]);
    }

    // ---------------------------------------------------------------- load announcement

    [Fact]
    public void Open_DeclaredVariantAndSlice_AnnouncesTheResolvedPlanOnce_AndPlainModelStaysSilent()
    {
        RopeModelSpec spec = new();
        string yarnPath = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.rope.scaling.type", "yarn"), ("llama.rope.scaling.factor", (float)YarnMultiplier), ("llama.rope.scaling.original_context_length", OriginalContextLength), ("llama.rope.dimension_count", 2)],
            spec.Tensors()));
        string plainPath = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors()));

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile yarnGguf = GgufFile.Open(yarnPath);
        using GgufFile plainGguf = GgufFile.Open(plainPath);

        List<string> yarnLines = [];
        using TransformerModel yarnModel = TransformerModel.Open(yarnGguf, backend, maxContextTokens: 8, logLine: yarnLines.Add);

        List<string> plainLines = [];
        using TransformerModel plainModel = TransformerModel.Open(plainGguf, backend, maxContextTokens: 8, logLine: plainLines.Add);

        string? line = yarnLines.Find(entry => entry.StartsWith("rope_scaling ", StringComparison.Ordinal));
        Assert.NotNull(line);
        Assert.Contains("variant=yarn", line, StringComparison.Ordinal);
        Assert.Contains("multiplier=2 ", line, StringComparison.Ordinal);
        Assert.Contains($"original_context_length={OriginalContextLength}", line, StringComparison.Ordinal);
        Assert.Contains("attention_factor=1 ", line, StringComparison.Ordinal);
        Assert.Contains("rotary_dimensions=2", line, StringComparison.Ordinal);
        Assert.Contains("architecture='llama'", line, StringComparison.Ordinal);

        Assert.Null(plainLines.Find(entry => entry.StartsWith("rope_scaling ", StringComparison.Ordinal)));
    }

    // ---------------------------------------------------------------- forward goldens

    [Fact]
    public void Forward_DeclaredYarnKeys_MatchesReferenceForward_AndDiffersFromUnscaledModel()
    {
        RopeModelSpec spec = new();
        string yarnPath = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.rope.scaling.type", "yarn"), ("llama.rope.scaling.factor", (float)YarnMultiplier), ("llama.rope.scaling.original_context_length", OriginalContextLength)],
            spec.Tensors()));

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile yarnGguf = GgufFile.Open(yarnPath);
        using TransformerModel yarnModel = TransformerModel.Open(yarnGguf, backend, maxContextTokens: 8);

        Assert.NotNull(yarnModel.Plan.RopeScaling);
        Assert.Equal("yarn", yarnModel.Plan.RopeScaling.Variant);
        Assert.Equal((float)YarnMultiplier, yarnModel.Plan.RopeScaling.Multiplier);
        Assert.Equal(OriginalContextLength, yarnModel.Plan.RopeScaling.OriginalContextLength);

        using BlockPool pool = new(blockTokens: 4, layerCount: yarnModel.Plan.LayerCount, kvHeads: yarnModel.Plan.KvHeadCount, headDim: yarnModel.Plan.HeadDim, capacityBlocks: 8, backend);
        using SequenceKv kv = new(pool);

        int[] history = [1, 4, 2, 5, 3];
        yarnModel.Prefill(history, kv);
        float[] logits = yarnModel.DecodeNext(0, kv).ToArray();

        RopeReference reference = new(spec);
        Assert.True(
            Closest(logits, reference.LogitsAt(history, queryToken: 0, ropeScaling: true)),
            "YaRN decode logits at position 5 must match the scaled reference forward");

        // The same weights with no scaling declared are a different model numerically.
        string plainPath = WriteTemp(BuildGguf(spec.MetadataPairs(), spec.Tensors()));
        using GgufFile plainGguf = GgufFile.Open(plainPath);
        using TransformerModel plainModel = TransformerModel.Open(plainGguf, backend, maxContextTokens: 8);

        Assert.Null(plainModel.Plan.RopeScaling);

        using SequenceKv plainKv = new(pool);
        plainModel.Prefill(history, plainKv);
        float[] plainLogits = plainModel.DecodeNext(0, plainKv).ToArray();

        Assert.True(
            !BitsEqual(logits, plainLogits),
            "the scaled forward must not decode bit-identically to the same model with no scaling declared");
        Assert.True(
            Differs(logits, reference.LogitsAt(history, queryToken: 0, ropeScaling: false)),
            "the scaled logits must move away from the unscaled reference forward by more than the tolerance");

        // The default path is untouched: an unscaled model still matches the plain reference exactly.
        Assert.True(
            Closest(plainLogits, reference.LogitsAt(history, queryToken: 0, ropeScaling: false)),
            "the unscaled decode must keep matching the plain reference forward");

        // Batched decode under the variant is bit-identical to single-token decode.
        using SequenceKv batched = new(pool);
        foreach (int token in history)
        {
            yarnModel.DecodeBatch([token], [batched]);
        }
        Span<float> lastBatchLogits = yarnModel.DecodeBatch([0], [batched]);

        // Compare against the same model stepped alone through DecodeNext on a fresh cache.
        using SequenceKv singleFresh = new(pool);
        foreach (int token in history)
        {
            yarnModel.DecodeNext(token, singleFresh);
        }
        float[] singleLast = yarnModel.DecodeNext(0, singleFresh).ToArray();

        Assert.True(
            lastBatchLogits.SequenceEqual(singleLast.AsSpan()),
            "a token batched through DecodeBatch under a scaling variant must be bit-identical to stepping the same sequence alone");
    }

    [Fact]
    public void Forward_DeclaredYarnKeysWithPartialSlice_RotatesOnlyTheLeadingDimensions()
    {
        RopeModelSpec spec = new();
        string path = WriteTemp(BuildGguf(
            [.. spec.MetadataPairs(), ("llama.rope.scaling.type", "yarn"), ("llama.rope.scaling.factor", (float)YarnMultiplier), ("llama.rope.scaling.original_context_length", OriginalContextLength), ("llama.rope.dimension_count", 2)],
            spec.Tensors()));

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using GgufFile gguf = GgufFile.Open(path);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 8);

        Assert.Equal(2, model.Plan.RopeRotaryDimensions); // only the leading pair rotates

        RopeReference reference = new(spec);

        using BlockPool pool = new(blockTokens: 4, layerCount: model.Plan.LayerCount, kvHeads: model.Plan.KvHeadCount, headDim: model.Plan.HeadDim, capacityBlocks: 8, backend);
        using SequenceKv kv = new(pool);

        int[] history = [1, 4, 2, 5, 3];
        model.Prefill(history, kv);
        float[] logits = model.DecodeNext(0, kv).ToArray();

        Assert.True(
            Closest(logits, reference.LogitsAt(history, queryToken: 0, ropeScaling: true, rotarySlice: 2)),
            "the sliced YaRN decode must match the reference forward that rotates only the leading dimensions");

        Assert.True(
            Differs(logits, reference.LogitsAt(history, queryToken: 0, ropeScaling: false)),
            "a narrowed slice under a variant must differ from the unscaled full-head rotation by more than the tolerance");
    }

    // ---------------------------------------------------------------- oracle geometry

    private static bool Closest(ReadOnlySpan<float> actual, double[] expected)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            if (Math.Abs(actual[i] - expected[i]) > 1e-5 * Math.Max(1.0, Math.Abs(expected[i])))
            {
                return false;
            }
        }

        return true;
    }

    private static bool Differs(ReadOnlySpan<float> actual, double[] expected) => !Closest(actual, expected);

    private static bool BitsEqual(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        for (int i = 0; i < a.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
            {
                return false;
            }
        }

        return true;
    }

    private static void AssertEntry(string label, float actual, double expected)
    {
        Assert.True(
            Math.Abs(actual - expected) <= 1e-5 * Math.Max(1.0, Math.Abs(expected)),
            $"{label}: expected {expected:G9}, got {actual:G9}");
    }

    // The reference's score-side attention factor for the variant (the plan resolves exactly this formula).
    private static double ResolvedMscale(double scale, double magnitude) =>
        scale <= 1.0 ? 1.0 : 0.1 * magnitude * Math.Log(scale) + 1.0;

    private static double ResolvedAttentionFactor(double multiplier, double declaredAttention, double logMultiplier)
    {
        double resolved = logMultiplier != 0.0
            ? ResolvedMscale(multiplier, 1.0) / ResolvedMscale(multiplier, logMultiplier)
            : ResolvedMscale(multiplier, 1.0);

        return resolved / (1.0 + 0.1 * Math.Log(multiplier)) * declaredAttention;
    }

    // The reference's solved correction dimension: the pair index whose unscaled frequency reaches beta at the
    // original context length.
    private static double CorrDim(double beta, int headDim, double originalContext, double thetaBase) =>
        headDim * Math.Log(originalContext / (beta * 2.0 * Math.PI)) / (2.0 * Math.Log(thetaBase));

    // A complete single-layer llama-shaped model with hidden 6, heads 2, kv-heads 1, a declared head dim of 4,
    // ffn 8, vocab 6 — all F32 tensors with hand-picked payloads the reference forward reads directly. No
    // output.weight: the logits come from the tied token-embedding rows. The payload magnitudes keep scores at
    // the unit scale so a scaled or sliced rotation moves the output by far more than the comparison tolerance.
    private sealed class RopeModelSpec
    {
        public const int Hidden = 6;
        public const int Heads = 2;
        public const int KvHeads = 1;
        public const int HeadDim = 4;
        public const int Intermediate = 8;
        public const int Vocab = 6;
        public const float Eps = 1e-5f;

        public List<(string Key, object Value)> MetadataPairs() =>
        [
            ("general.architecture", "llama"),
            ("llama.embedding_length", Hidden),
            ("llama.block_count", 1),
            ("llama.attention.head_count", Heads),
            ("llama.attention.head_count_kv", KvHeads),
            ("llama.feed_forward_length", Intermediate),
            ("llama.attention.head_length", HeadDim),
            ("llama.rope.freq_base", RopeBase),
            ("llama.attention.layer_norm_rms_epsilon", Eps),
            ("llama.vocab_size", Vocab),
        ];

        public List<TensorDef> Tensors() =>
        [
            new("token_embd.weight", [Vocab, Hidden], Embeddings),
            new("blk.0.attn_q.weight", [Heads * HeadDim, Hidden], AttnQ),
            new("blk.0.attn_k.weight", [KvHeads * HeadDim, Hidden], AttnK),
            new("blk.0.attn_v.weight", [KvHeads * HeadDim, Hidden], AttnV),
            new("blk.0.attn_output.weight", [Hidden, Heads * HeadDim], AttnO),
            new("blk.0.ffn_gate.weight", [Intermediate, Hidden], FfnGate),
            new("blk.0.ffn_up.weight", [Intermediate, Hidden], FfnUp),
            new("blk.0.ffn_down.weight", [Hidden, Intermediate], FfnDown),
            new("blk.0.attn_norm.weight", [Hidden], AttnNorm),
            new("blk.0.ffn_norm.weight", [Hidden], FfnNorm),
            new("output_norm.weight", [Hidden], OutputNorm),
        ];

        public readonly float[] Embeddings =
        [
            0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f,
            0.11f, -0.21f, 0.31f, -0.41f, 0.51f, -0.61f,
            0.32f, 0.22f, -0.32f, 0.42f, -0.52f, 0.62f,
            -0.13f, 0.23f, 0.33f, -0.43f, 0.53f, 0.63f,
            0.94f, -0.84f, 0.74f, -0.64f, 0.54f, -0.44f,
            0.15f, 0.25f, 0.35f, 0.45f, 0.55f, -0.65f,
        ];

        public readonly float[] AttnQ =
        [
            0.20f, 0.40f, 0.30f, 0.10f, 0.50f, 0.25f,
            0.15f, -0.35f, 0.45f, 0.20f, -0.10f, 0.30f,
            0.35f, 0.10f, -0.25f, 0.40f, 0.15f, -0.30f,
            -0.20f, 0.30f, 0.10f, -0.45f, 0.25f, 0.40f,
            0.45f, -0.15f, 0.20f, 0.35f, -0.40f, 0.10f,
            -0.30f, 0.25f, -0.40f, 0.15f, 0.35f, -0.20f,
            0.10f, 0.50f, -0.15f, -0.30f, 0.20f, 0.45f,
            -0.25f, 0.15f, 0.40f, -0.20f, -0.35f, 0.30f,
        ];

        public readonly float[] AttnK =
        [
            0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f,
            0.20f, -0.10f, 0.40f, 0.30f, -0.20f, 0.50f,
            -0.30f, 0.40f, 0.10f, -0.50f, 0.20f, 0.30f,
            0.40f, 0.15f, -0.35f, 0.25f, 0.45f, -0.25f,
        ];

        public readonly float[] AttnV =
        [
            0.09f, 0.08f, 0.07f, 0.06f, 0.05f, 0.04f,
            -0.45f, 0.38f, -0.31f, 0.24f, -0.17f, 0.10f,
            0.30f, -0.22f, 0.14f, -0.06f, 0.42f, -0.34f,
            0.06f, 0.05f, 0.04f, 0.03f, 0.02f, 0.01f,
        ];

        public readonly float[] AttnO =
        [
            0.10f, 0.00f, 0.20f, 0.00f, 0.30f, 0.00f, 0.15f, 0.00f,
            0.00f, 0.30f, 0.00f, 0.40f, 0.00f, 0.50f, 0.00f, 0.25f,
            0.15f, 0.00f, 0.25f, 0.00f, 0.35f, 0.00f, 0.20f, 0.00f,
            0.00f, 0.35f, 0.00f, 0.45f, 0.00f, 0.55f, 0.00f, 0.30f,
            0.10f, 0.00f, 0.15f, 0.00f, 0.25f, 0.00f, 0.35f, 0.00f,
            0.00f, 0.20f, 0.00f, 0.30f, 0.00f, 0.40f, 0.00f, 0.45f,
        ];

        public readonly float[] FfnGate =
        [
            0.10f, 0.11f, 0.12f, 0.13f, 0.14f, 0.15f,
            0.16f, 0.17f, 0.18f, 0.19f, 0.20f, 0.21f,
            0.22f, 0.23f, 0.24f, 0.25f, 0.26f, 0.27f,
            0.28f, 0.29f, 0.30f, 0.31f, 0.32f, 0.33f,
            0.34f, 0.35f, 0.36f, 0.37f, 0.38f, 0.39f,
            0.40f, 0.41f, 0.42f, 0.43f, 0.44f, 0.45f,
            0.46f, 0.47f, 0.48f, 0.49f, 0.50f, 0.51f,
            0.52f, 0.53f, 0.54f, 0.55f, 0.56f, 0.57f,
        ];

        public readonly float[] FfnUp =
        [
            0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f,
            0.07f, 0.08f, 0.09f, 0.10f, 0.11f, 0.12f,
            0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f,
            0.19f, 0.20f, 0.21f, 0.22f, 0.23f, 0.24f,
            0.25f, 0.26f, 0.27f, 0.28f, 0.29f, 0.30f,
            0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f,
            0.37f, 0.38f, 0.39f, 0.40f, 0.41f, 0.42f,
            0.43f, 0.44f, 0.45f, 0.46f, 0.47f, 0.48f,
        ];

        public readonly float[] FfnDown =
        [
            0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.10f, 0.11f, 0.12f,
            0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.20f,
            0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f, 0.27f, 0.28f,
            0.29f, 0.30f, 0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f,
            0.37f, 0.38f, 0.39f, 0.40f, 0.41f, 0.42f, 0.43f, 0.44f,
            0.45f, 0.46f, 0.47f, 0.48f, 0.49f, 0.50f, 0.51f, 0.52f,
        ];

        public readonly float[] AttnNorm = [1.10f, 0.90f, 1.05f, 0.95f, 1.00f, 1.15f];
        public readonly float[] FfnNorm = [0.90f, 1.10f, 0.95f, 1.05f, 1.00f, 0.85f];
        public readonly float[] OutputNorm = [1.00f, 1.02f, 0.98f, 1.01f, 0.99f, 1.03f];
    }

    // A scalar reference forward over the spec's hand-picked tensors: RMSNorm, the rope rotation chosen by the
    // scaling flag (unscaled full-head pairs, or the YaRN blend and magnitude over the declared slice), causal
    // attention with an explicit softmax whose scores carry the head-dimension gain times the variant's
    // attention factor, silu(gate) * up, and the tied token-embedding logits. Computed in double and compared
    // against the engine's float values with the shared tolerance.
    private sealed class RopeReference
    {
        private readonly RopeModelSpec _spec;

        public RopeReference(RopeModelSpec spec) => _spec = spec;

        public double[] LogitsAt(int[] historyTokens, int queryToken, bool ropeScaling, int rotarySlice = 0)
        {
            double attentionFactor = ropeScaling
                ? ResolvedAttentionFactor(YarnMultiplier, declaredAttention: 1.0, logMultiplier: 0.0)
                : 1.0;
            double tableMagnitude = ropeScaling
                ? attentionFactor * (1.0 + 0.1 * Math.Log(YarnMultiplier))
                : 1.0;
            double scoreScale = attentionFactor / Math.Sqrt(RopeModelSpec.HeadDim);

            double[][] kvK = new double[historyTokens.Length + 1][];
            double[][] kvV = new double[historyTokens.Length + 1][];

            for (int p = 0; p < historyTokens.Length; p++)
            {
                Advance(Row(_spec.Embeddings, historyTokens[p], RopeModelSpec.Hidden), p, kvK, kvV, ropeScaling, rotarySlice, tableMagnitude, scoreScale);
            }

            double[] cur = Row(_spec.Embeddings, queryToken, RopeModelSpec.Hidden);
            Advance(cur, historyTokens.Length, kvK, kvV, ropeScaling, rotarySlice, tableMagnitude, scoreScale);

            double[] normed = Norm(cur, _spec.OutputNorm);
            return Gemv(_spec.Embeddings, RopeModelSpec.Vocab, RopeModelSpec.Hidden, normed); // tied lm_head
        }

        private void Advance(double[] cur, int position, double[][] kvK, double[][] kvV, bool ropeScaling, int rotarySlice, double tableMagnitude, double scoreScale)
        {
            double[] normed = Norm(cur, _spec.AttnNorm);
            double[] q = Gemv(_spec.AttnQ, RopeModelSpec.Heads * RopeModelSpec.HeadDim, RopeModelSpec.Hidden, normed);
            double[] k = Gemv(_spec.AttnK, RopeModelSpec.KvHeads * RopeModelSpec.HeadDim, RopeModelSpec.Hidden, normed);
            double[] v = Gemv(_spec.AttnV, RopeModelSpec.KvHeads * RopeModelSpec.HeadDim, RopeModelSpec.Hidden, normed);

            Rope(q, position, ropeScaling, rotarySlice, tableMagnitude);
            Rope(k, position, ropeScaling, rotarySlice, tableMagnitude);

            kvK[position] = k;
            kvV[position] = v;

            double[] attnOut = new double[RopeModelSpec.Heads * RopeModelSpec.HeadDim];
            Attention(q, kvK, kvV, position, scoreScale, attnOut);

            double[] proj = Gemv(_spec.AttnO, RopeModelSpec.Hidden, RopeModelSpec.Heads * RopeModelSpec.HeadDim, attnOut);
            for (int i = 0; i < RopeModelSpec.Hidden; i++)
            {
                cur[i] += proj[i];
            }

            double[] ffnNormed = Norm(cur, _spec.FfnNorm);
            double[] gate = Gemv(_spec.FfnGate, RopeModelSpec.Intermediate, RopeModelSpec.Hidden, ffnNormed);
            double[] up = Gemv(_spec.FfnUp, RopeModelSpec.Intermediate, RopeModelSpec.Hidden, ffnNormed);
            for (int i = 0; i < RopeModelSpec.Intermediate; i++)
            {
                gate[i] = Silu(gate[i]) * up[i];
            }

            double[] down = Gemv(_spec.FfnDown, RopeModelSpec.Hidden, RopeModelSpec.Intermediate, gate);
            for (int i = 0; i < RopeModelSpec.Hidden; i++)
            {
                cur[i] += down[i];
            }
        }

        private static void Attention(double[] q, double[][] kvK, double[][] kvV, int position, double scoreScale, double[] attnOut)
        {
            const int headDim = RopeModelSpec.HeadDim;

            // heads 2, kv-heads 1: every head reads kv-head (h * 1 / 2) == 0.
            for (int h = 0; h < RopeModelSpec.Heads; h++)
            {
                double[] scores = new double[position + 1];
                double max = double.NegativeInfinity;
                for (int t = 0; t <= position; t++)
                {
                    double dot = 0.0;
                    for (int j = 0; j < headDim; j++)
                    {
                        dot += q[h * headDim + j] * kvK[t][j];
                    }

                    scores[t] = dot * scoreScale;
                    max = Math.Max(max, scores[t]);
                }

                double sum = 0.0;
                for (int t = 0; t <= position; t++)
                {
                    scores[t] = Math.Exp(scores[t] - max);
                    sum += scores[t];
                }

                for (int j = 0; j < headDim; j++)
                {
                    double acc = 0.0;
                    for (int t = 0; t <= position; t++)
                    {
                        acc += scores[t] / sum * kvV[t][j];
                    }

                    attnOut[h * headDim + j] = acc;
                }
            }
        }

        // The rotation the engine's tables produce: adjacent pairs over the declared slice (the full head when
        // the slice is zero), each pair carrying the variant's magnitude when scaling is declared.
        private static void Rope(double[] headRegion, int position, bool ropeScaling, int rotarySlice, double tableMagnitude)
        {
            int slice = rotarySlice > 0 ? rotarySlice : RopeModelSpec.HeadDim;

            double corrLow = Math.Max(0.0, Math.Floor(CorrDim(32.0, RopeModelSpec.HeadDim, OriginalContextLength, RopeBase)));
            double corrHigh = Math.Min(RopeModelSpec.HeadDim - 1.0, Math.Ceiling(CorrDim(1.0, RopeModelSpec.HeadDim, OriginalContextLength, RopeBase)));
            double bandWidth = Math.Max(0.001, corrHigh - corrLow);

            for (int h = 0; h < headRegion.Length / RopeModelSpec.HeadDim; h++)
            {
                for (int i = 0; i < slice / 2; i++)
                {
                    double unscaled = position * Math.Pow(RopeBase, -2.0 * i / RopeModelSpec.HeadDim);
                    double angle;
                    if (ropeScaling)
                    {
                        double interp = unscaled / YarnMultiplier;
                        double ramp = 1.0 - Math.Min(1.0, Math.Max(0.0, (i - corrLow) / bandWidth));
                        angle = interp * (1.0 - ramp) + unscaled * ramp; // the reference default correction scale
                    }
                    else
                    {
                        angle = unscaled;
                    }

                    double c = Math.Cos(angle) * tableMagnitude;
                    double s = Math.Sin(angle) * tableMagnitude;

                    double lo = headRegion[h * RopeModelSpec.HeadDim + 2 * i];
                    double hi = headRegion[h * RopeModelSpec.HeadDim + 2 * i + 1];
                    headRegion[h * RopeModelSpec.HeadDim + 2 * i] = lo * c - hi * s;
                    headRegion[h * RopeModelSpec.HeadDim + 2 * i + 1] = hi * c + lo * s;
                }
            }
        }

        private static double[] Norm(double[] x, float[] weight)
        {
            int n = x.Length;
            double sumSq = 0.0;
            for (int i = 0; i < n; i++)
            {
                sumSq += x[i] * x[i];
            }

            double invRms = 1.0 / Math.Sqrt(sumSq / n + RopeModelSpec.Eps);
            double[] result = new double[n];
            for (int i = 0; i < n; i++)
            {
                result[i] = x[i] * invRms * weight[i];
            }

            return result;
        }

        private static double Silu(double x) => x / (1.0 + Math.Exp(-x));

        private static double[] Gemv(float[] rows, int outCount, int inCount, double[] input)
        {
            double[] output = new double[outCount];
            for (int r = 0; r < outCount; r++)
            {
                double acc = 0.0;
                for (int c = 0; c < inCount; c++)
                {
                    acc += rows[r * inCount + c] * input[c];
                }

                output[r] = acc;
            }

            return output;
        }

        private static double[] Row(float[] data, int row, int length)
        {
            double[] result = new double[length];
            for (int i = 0; i < length; i++)
            {
                result[i] = data[row * length + i];
            }

            return result;
        }
    }

    private sealed record TensorDef(string Name, int[] Dims, float[] Payload); // dims in logical (row-major) order; vectors are one-dimensional

    /// <summary>Builds a GGUF v3 image from KV pairs and F32 tensors with the exact payloads the reference forward reads.</summary>
    private static byte[] BuildGguf(IReadOnlyList<(string Key, object Value)> kvPairs, IReadOnlyList<TensorDef> tensors)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(GgufMagic);
        writer.Write(3u); // version
        writer.Write((ulong)tensors.Count);
        writer.Write((ulong)kvPairs.Count);

        foreach ((string key, object value) in kvPairs)
        {
            WriteGgufString(writer, key);
            switch (value)
            {
                case string s:
                    writer.Write(8u); // STRING
                    WriteGgufString(writer, s);
                    break;
                case int i:
                    writer.Write(5u); // INT32
                    writer.Write(i);
                    break;
                case uint u:
                    writer.Write(4u); // UINT32
                    writer.Write(u);
                    break;
                case long l:
                    writer.Write(11u); // INT64
                    writer.Write(l);
                    break;
                case float f:
                    writer.Write(6u); // FLOAT32
                    writer.Write(f);
                    break;
                case double d:
                    writer.Write(12u); // FLOAT64
                    writer.Write(d);
                    break;
                case bool b:
                    writer.Write(7u); // BOOL
                    writer.Write((byte)(b ? 1 : 0));
                    break;
                default:
                    throw new ArgumentException($"Unsupported fixture value type {value.GetType().Name}.", nameof(kvPairs));
            }
        }

        long offset = 0;
        foreach (TensorDef tensor in tensors)
        {
            // GGUF stores dims fastest-varying first: the logical row-major order is written reversed.
            int[] stored = new int[tensor.Dims.Length];
            for (int d = 0; d < tensor.Dims.Length; d++)
            {
                stored[d] = tensor.Dims[tensor.Dims.Length - 1 - d];
            }

            WriteGgufString(writer, tensor.Name);
            writer.Write((uint)stored.Length);
            foreach (int dim in stored)
            {
                writer.Write((ulong)dim);
            }

            writer.Write(0u); // ggml type F32
            writer.Write((ulong)offset);

            long elements = 1;
            foreach (int dim in tensor.Dims)
            {
                elements *= dim;
            }

            offset += elements * 4;
        }

        long headerLength = stream.Position;
        long dataStart = (headerLength + 31) / 32 * 32; // GGUF default alignment is 32
        for (long i = headerLength; i < dataStart; i++)
        {
            writer.Write((byte)0);
        }

        foreach (TensorDef tensor in tensors)
        {
            long elements = 1;
            foreach (int dim in tensor.Dims)
            {
                elements *= dim;
            }

            if (tensor.Payload.Length != elements)
            {
                throw new ArgumentException($"tensor '{tensor.Name}' payload has {tensor.Payload.Length} entries, expected {elements}.", nameof(tensors));
            }

            foreach (float value in tensor.Payload)
            {
                writer.Write(value);
            }
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-rope-scaling-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }
}
