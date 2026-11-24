using Lair.Contracts;
using Lair.Backends.Cpu;
using Lair.Kv;

namespace Lair.Models.Engine;

/// <summary>
/// The pure numeric operations of the transformer forward pass, isolated so each can be verified against
/// hand-computed values independently of any model file. RoPE rotates q/k adjacent pairs (2i, 2i + 1) for the
/// llama-family pairing or rotate-half pairs (i, i + headDim/2) for the qwen3 pairing, each with
/// theta_i = base^(-2i/headDim); RMSNorm and LayerNorm accumulate in double; attention runs an online softmax
/// over the paged KV cache so no score row is ever materialized.
/// </summary>
internal static class LayerOps
{
    /// <summary>RMSNorm with weight: dst[i] = src[i] / sqrt(mean(src^2) + eps) * w[i]. The mean accumulates in double.</summary>
    internal static void RmsNormWeighted(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, float eps, Span<float> dst)
    {
        if (weight.Length != src.Length || dst.Length < src.Length)
        {
            throw new ArgumentException("RMSNorm operands must all cover the same element count.", nameof(weight));
        }

        double sumSq = 0.0;
        for (int i = 0; i < src.Length; i++)
        {
            sumSq += (double)src[i] * src[i];
        }

        float invRms = (float)(1.0 / Math.Sqrt(sumSq / src.Length + eps));
        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = src[i] * invRms * weight[i];
        }
    }

    /// <summary>
    /// RMSNorm applied to each head region of a head-major buffer independently: head h normalizes its own
    /// <c>headDim</c> slice with the head-dimension weight vector, <c>dst[h * headDim + i] = src[h * headDim + i] /
    /// sqrt(mean(src_h^2) + eps) * w[i]</c>. Each head's mean accumulates in double. This is the per-head form
    /// applied to q and k after their projection and before RoPE.
    /// </summary>
    internal static void RmsNormWeightedHeads(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, int headDim, float eps, Span<float> dst)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);

        if (weight.Length != headDim || src.Length % headDim != 0 || dst.Length < src.Length)
        {
            throw new ArgumentException("Per-head RMSNorm needs a weight vector covering exactly one head and operands covering whole heads.", nameof(weight));
        }

        for (int region = 0; region < src.Length; region += headDim)
        {
            double sumSq = 0.0;
            for (int i = 0; i < headDim; i++)
            {
                sumSq += (double)src[region + i] * src[region + i];
            }

            float invRms = (float)(1.0 / Math.Sqrt(sumSq / headDim + eps));
            for (int i = 0; i < headDim; i++)
            {
                dst[region + i] = src[region + i] * invRms * weight[i];
            }
        }
    }

    /// <summary>
    /// RMSNorm applied to each head region of a head-major buffer independently, with no weight vector: head h
    /// normalizes its own <c>headDim</c> slice with the epsilon alone, <c>dst[h * headDim + i] = src[h * headDim + i] /
    /// sqrt(mean(src_h^2) + eps)</c>. Each head's mean accumulates in double exactly as <see cref="RmsNormWeightedHeads"/>
    /// does, so the result is bit-for-bit what that operation produces for an all-ones weight vector. This is the
    /// per-head form applied to q and k after their projection when the architecture normalizes them without a norm
    /// tensor of its own.
    /// </summary>
    internal static void RmsNormUnweightedHeads(ReadOnlySpan<float> src, int headDim, float eps, Span<float> dst)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);

        if (src.Length % headDim != 0 || dst.Length < src.Length)
        {
            throw new ArgumentException("Per-head RMSNorm needs operands covering whole heads.", nameof(src));
        }

        for (int region = 0; region < src.Length; region += headDim)
        {
            double sumSq = 0.0;
            for (int i = 0; i < headDim; i++)
            {
                sumSq += (double)src[region + i] * src[region + i];
            }

            float invRms = (float)(1.0 / Math.Sqrt(sumSq / headDim + eps));
            for (int i = 0; i < headDim; i++)
            {
                dst[region + i] = src[region + i] * invRms;
            }
        }
    }

    /// <summary>
    /// LayerNorm with weight and optional bias: dst[i] = (src[i] - mean) / sqrt(var + eps) * w[i] (+ b[i] when a
    /// bias is supplied). Mean and population variance (divided by the element count) accumulate in double, the
    /// convention shared by the reference LayerNorm implementations.
    /// </summary>
    internal static void LayerNormWeighted(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias, float eps, Span<float> dst)
    {
        if (weight.Length != src.Length || dst.Length < src.Length)
        {
            throw new ArgumentException("LayerNorm operands must all cover the same element count.", nameof(weight));
        }

        if (!bias.IsEmpty && bias.Length != src.Length)
        {
            throw new ArgumentException("LayerNorm bias must cover the same element count as the input.", nameof(bias));
        }

        double sum = 0.0;
        for (int i = 0; i < src.Length; i++)
        {
            sum += src[i];
        }

        double mean = sum / src.Length;

        double sumSqDev = 0.0;
        for (int i = 0; i < src.Length; i++)
        {
            double deviation = src[i] - mean;
            sumSqDev += deviation * deviation;
        }

        double invStd = 1.0 / Math.Sqrt(sumSqDev / src.Length + eps);
        for (int i = 0; i < src.Length; i++)
        {
            double value = (src[i] - mean) * invStd * weight[i];
            if (!bias.IsEmpty)
            {
                value += bias[i];
            }

            dst[i] = (float)value;
        }
    }

    /// <summary>SiLU activation: x / (1 + exp(-x)).</summary>
    internal static float Silu(float x) => x / (1f + MathF.Exp(-x));

    /// <summary>Plain sigmoid: 1 / (1 + exp(-x)) — the gate a qwen4exp shared expert reads over its selector score.</summary>
    internal static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>
    /// The weight one routed family's declared shared expert contributes for its selector score. A family whose reference
    /// composes the gate as a plain sigmoid gets <c>sigmoid(score)</c>: finite at every score, 0.5 at exactly zero. Every
    /// other routed family keeps the pinned qwen2moe composition <c>silu(score) / score</c>, which divides 0 by 0 at a zero
    /// score exactly as that graph does. The arm is selected by the family the plan resolved, never probed from the score.
    /// </summary>
    internal static float SharedExpertGateWeight(bool plainSigmoidFamily, float score) =>
        plainSigmoidFamily ? Sigmoid(score) : Silu(score) / score;

    /// <summary>
    /// Logit softcap: cap * tanh(x / cap), evaluated in double and rounded once to float. The result stays
    /// inside [-cap, cap]: the identity for small magnitudes, saturating toward ±cap for large ones. Callers
    /// branch on a resolved cap and use this only when a cap is enabled.
    /// </summary>
    internal static float Softcap(float x, float cap) => (float)((double)cap * Math.Tanh((double)x / (double)cap));

    /// <summary>
    /// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))), evaluated in double and rounded once to float.
    /// </summary>
    internal static float Gelu(float x) => (float)(0.5 * (double)x * (1.0 + Erf((double)x / Math.Sqrt(2.0))));

    /// <summary>
    /// Tanh-approximated GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))), evaluated in double and
    /// rounded once to float — the approximation the dense families pair with <c>gelu_tanh</c> gated feed-forward blocks.
    /// </summary>
    internal static float GeluTanh(float x)
    {
        double d = x;
        return (float)(0.5 * d * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (d + 0.044715 * d * d * d))));
    }

    /// <summary>
    /// Error function in double precision: Cephes rational poles — a quartic-over-quintic pole for |x| <= 1, an
    /// eighth-degree pole times exp(-x^2) for 1 &lt; |x| &lt; 8, and a fifth-degree pole beyond 8 — with absolute
    /// error at the ~1e-15 level across the whole real line (|x| >= 26 saturates to ±1, where the true value is
    /// within 1e-300 of it).
    /// </summary>
    internal static double Erf(double x)
    {
        if (Math.Abs(x) > 1.0)
        {
            return 1.0 - ErfComplement(x);
        }

        double z = x * x;
        double numerator = PolynomialEvaluate(z, ErfSmallT);
        double denominator = MonicPolynomialEvaluate(z, ErfSmallU);
        return x * (numerator / denominator);
    }

    // Cephes erf pole for |x| <= 1: T(z) / U(z) * x with z = x^2, U monic.
    private static readonly double[] ErfSmallT = [9.60497373987051638749E0, 9.00260197203842689217E1, 2.23200534594684319226E3, 7.00332514112805075473E3, 5.55923013010394962768E4];
    private static readonly double[] ErfSmallU = [3.35617141647503099647E1, 5.21357949780152679795E2, 4.59432382970980127987E3, 2.26290000613890934246E4, 4.92673942608635921086E4];

    // Cephes erfc poles for |x| > 1: exp(-x^2) * P(x)/Q(x) below 8 and R(x)/S(x) from 8 up, Q and S monic.
    private static readonly double[] ErfComplementP = [2.46196981473530512524E-10, 5.64189564831068821977E-1, 7.46321056442269912687E0, 4.86371970985681366614E1, 1.96520832956077098242E2, 5.26445194995477358631E2, 9.34528527171957607540E2, 1.02755188689515710272E3, 5.57535335369399327526E2];
    private static readonly double[] ErfComplementQ = [1.32281951154744992508E1, 8.67072140885989742329E1, 3.54937778887819891062E2, 9.75708501743205489753E2, 1.82390916687909736289E3, 2.24633760818710981792E3, 1.65666309194161350182E3, 5.57535340817727675546E2];
    private static readonly double[] ErfComplementR = [5.64189583547755073984E-1, 1.27536670759978104416E0, 5.01905042251180477414E0, 6.16021097993053585195E0, 7.40974269950448939160E0, 2.97886665372100240670E0];
    private static readonly double[] ErfComplementS = [2.26052863220117276590E0, 9.39603524938001434673E0, 1.20489539808096656605E1, 1.70814450747565897222E1, 9.60896809063285878198E0, 3.36907645100081516050E0];

    // |x| at which erfc underflows the double range: erf is 1 to within 1e-300 well before this point.
    private const double ErfComplementSaturation = 26.0;

    private static double ErfComplement(double x)
    {
        double magnitude = Math.Abs(x);
        if (magnitude >= ErfComplementSaturation)
        {
            return x >= 0.0 ? 0.0 : 2.0;
        }

        double ratio = magnitude < 8.0
            ? PolynomialEvaluate(magnitude, ErfComplementP) / MonicPolynomialEvaluate(magnitude, ErfComplementQ)
            : PolynomialEvaluate(magnitude, ErfComplementR) / MonicPolynomialEvaluate(magnitude, ErfComplementS);

        double value = ratio * Math.Exp(-magnitude * magnitude);
        return x >= 0.0 ? value : 2.0 - value;
    }

    // Horner evaluation of c[0] * x^degree + c[1] * x^(degree-1) + ... + c[degree].
    private static double PolynomialEvaluate(double x, double[] coefficients)
    {
        double accumulator = coefficients[0];
        for (int i = 1; i < coefficients.Length; i++)
        {
            accumulator = accumulator * x + coefficients[i];
        }

        return accumulator;
    }

    // Horner evaluation of the monic polynomial x^degree + c[0] * x^(degree-1) + ... + c[degree-1].
    private static double MonicPolynomialEvaluate(double x, double[] coefficients)
    {
        double accumulator = x + coefficients[0];
        for (int i = 1; i < coefficients.Length; i++)
        {
            accumulator = accumulator * x + coefficients[i];
        }

        return accumulator;
    }

    /// <summary>
    /// Fills position-major cos/sin tables for standard full-head RoPE up to maxPositions. Each position stores
    /// headDim/2 entries: angle(pos, i) = pos * base^(-2i/headDim), the angle of pair i, computed in double and stored
    /// as float so the same inputs always produce bit-identical rotations. This is <see cref="BuildRopeSliceTables"/>
    /// over the whole head dimension, which is why a full-head table reads exactly as it always has.
    /// </summary>
    internal static void BuildRopeTables(double thetaBase, int headDim, int maxPositions, Span<float> cos, Span<float> sin) =>
        BuildRopeSliceTables(thetaBase, headDim, ropeDimensions: headDim, maxPositions, cos, sin);

    /// <summary>
    /// Fills position-major cos/sin tables for the leading <c>ropeDimensions</c> of a <c>headDim</c>-wide head up to
    /// maxPositions — the unscaled angle table a partial rotary declaration asks for. Each position stores
    /// <c>ropeDimensions / 2</c> entries, and pair i still carries the angle of the pair occupying head dimensions at
    /// its own place in the full head: <c>angle(pos, i) = pos * base^(-2i/headDim)</c>, computed over the whole-head
    /// denominator exactly as a full-head build computes it. Narrowing therefore truncates which pairs exist without
    /// re-spacing the frequencies of the ones that remain — what the reference reads for <c>n_rot &lt; n_embd_head</c>
    /// and what <see cref="BuildRopeScalingTables"/> already does under a scaling variant. A slice equal to the head
    /// dimension produces bit-identical tables to <see cref="BuildRopeTables"/>, and only pairs inside the slice exist,
    /// so every rotation driven off this table leaves the dimensions outside it untouched.
    /// </summary>
    internal static void BuildRopeSliceTables(double thetaBase, int headDim, int ropeDimensions, int maxPositions, Span<float> cos, Span<float> sin)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);

        if (headDim % 2 != 0)
        {
            throw new ArgumentException($"RoPE pairs require an even head dimension, got {headDim}.", nameof(headDim));
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ropeDimensions);
        if (ropeDimensions > headDim || ropeDimensions % 2 != 0)
        {
            throw new ArgumentException(
                $"the rotary slice must be an even count of head dimensions not exceeding {headDim}, got {ropeDimensions}.",
                nameof(ropeDimensions));
        }

        int half = ropeDimensions / 2;
        if (cos.Length < maxPositions * half || sin.Length < maxPositions * half)
        {
            throw new ArgumentException($"Cos/sin tables must cover maxPositions * {ropeDimensions}/2 entries each.", nameof(cos));
        }

        double[] invFreq = new double[half];
        for (int i = 0; i < half; i++)
        {
            invFreq[i] = Math.Pow(thetaBase, -2.0 * i / headDim);
        }

        for (int pos = 0; pos < maxPositions; pos++)
        {
            int tableBase = pos * half;
            for (int i = 0; i < half; i++)
            {
                double angle = pos * invFreq[i];
                cos[tableBase + i] = (float)Math.Cos(angle);
                sin[tableBase + i] = (float)Math.Sin(angle);
            }
        }
    }

    /// <summary>
    /// Fills position-major cos/sin tables for the YaRN scaling variant up to maxPositions, over the leading
    /// <c>ropeDimensions</c> head dimensions. Each position stores <c>ropeDimensions / 2</c> entries. Pair i of
    /// position p starts from the unscaled angle <c>p * base^(-2i/headDim)</c>: the table angle interpolates it
    /// toward <c>angle / multiplier</c> and blends the two back toward the unscaled angle across the correction
    /// band <c>[corrLow, corrHigh]</c> with weight <c>correctionScale * (1 - clamp((i - corrLow) / (corrHigh - corrLow)))</c>,
    /// where the band ends are the pair indices whose unscaled frequency reaches <c>betaFast</c> and
    /// <c>betaSlow</c> at the original context — <c>i(x) = headDim * ln(originalContext / (x * 2 * pi)) / (2 * ln(base))</c>,
    /// clamped to <c>[0, headDim - 1]</c>. Every entry is then multiplied by <c>attentionFactor</c>, which gains
    /// <c>1 + 0.1 * ln(multiplier)</c> when the correction scale is non-zero. Angles and scales are computed in
    /// double and stored as float, so the same inputs always produce bit-identical rotations.
    /// </summary>
    internal static void BuildRopeScalingTables(
        double thetaBase,
        int headDim,
        int ropeDimensions,
        int maxPositions,
        double multiplier,
        double originalContext,
        double attentionFactor,
        double correctionScale,
        double betaFast,
        double betaSlow,
        Span<float> cos,
        Span<float> sin)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);
        if (headDim % 2 != 0)
        {
            throw new ArgumentException($"RoPE pairs require an even head dimension, got {headDim}.", nameof(headDim));
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ropeDimensions);
        if (ropeDimensions > headDim || ropeDimensions % 2 != 0)
        {
            throw new ArgumentException(
                $"the rotary slice must be an even count of head dimensions not exceeding {headDim}, got {ropeDimensions}.",
                nameof(ropeDimensions));
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxPositions);

        int half = ropeDimensions / 2;
        if (cos.Length < maxPositions * half || sin.Length < maxPositions * half)
        {
            throw new ArgumentException("Cos/sin tables must cover maxPositions * ropeDimensions/2 entries each.", nameof(cos));
        }

        double freqScale = 1.0 / multiplier;

        // Band ends in pair indices, from the reference's solved correction dimensions.
        static double CorrDim(double beta, int headDim, double originalContext, double thetaBase) =>
            headDim * Math.Log(originalContext / (beta * 2.0 * Math.PI)) / (2.0 * Math.Log(thetaBase));

        double corrLow = Math.Max(0.0, Math.Floor(CorrDim(betaFast, headDim, originalContext, thetaBase)));
        double corrHigh = Math.Min(headDim - 1.0, Math.Ceiling(CorrDim(betaSlow, headDim, originalContext, thetaBase)));
        double bandWidth = Math.Max(0.001, corrHigh - corrLow);

        double magnitude = attentionFactor;
        if (correctionScale != 0.0)
        {
            magnitude *= 1.0 + 0.1 * Math.Log(multiplier); // 1 / freqScale == multiplier
        }

        for (int pos = 0; pos < maxPositions; pos++)
        {
            int tableBase = pos * half;
            for (int i = 0; i < half; i++)
            {
                double unscaled = pos * Math.Pow(thetaBase, -2.0 * i / headDim);
                double angle = unscaled * freqScale;

                if (correctionScale != 0.0)
                {
                    double ramp = 1.0 - Math.Min(1.0, Math.Max(0.0, (i - corrLow) / bandWidth));
                    double mix = ramp * correctionScale;
                    angle = angle * (1.0 - mix) + unscaled * mix;
                }

                cos[tableBase + i] = (float)(Math.Cos(angle) * magnitude);
                sin[tableBase + i] = (float)(Math.Sin(angle) * magnitude);
            }
        }
    }

    /// <summary>
    /// Rotates one head region in place with GGUF/llama-style adjacent pairs: out[2i] = x[2i]*cos_i - x[2i+1]*sin_i,
    /// out[2i + 1] = x[2i + 1]*cos_i + x[2i]*sin_i. Table entry i is the angle of the pair occupying adjacent head
    /// dimensions 2i and 2i + 1, which is how rotary q/k rows are laid out in converted llama-family GGUF checkpoints.
    /// </summary>
    internal static void ApplyRopeToHead(Span<float> headRegion, int ropeHalf, ReadOnlySpan<float> cosAtPosition, ReadOnlySpan<float> sinAtPosition)
    {
        if (headRegion.Length < 2 * ropeHalf || cosAtPosition.Length < ropeHalf || sinAtPosition.Length < ropeHalf)
        {
            throw new ArgumentException("RoPE operands must cover a full head region and half-dimension tables.", nameof(headRegion));
        }

        for (int i = 0; i < ropeHalf; i++)
        {
            float c = cosAtPosition[i];
            float s = sinAtPosition[i];

            int lowIndex = 2 * i;
            int highIndex = 2 * i + 1;

            float lo = headRegion[lowIndex];
            float hi = headRegion[highIndex];

            headRegion[lowIndex] = lo * c - hi * s;
            headRegion[highIndex] = hi * c + lo * s;
        }
    }

    /// <summary>
    /// Rotates one head region in place with rotate-half (non-adjacent) pairs: out[i] = x[i]*cos_i - x[i + ropeHalf]*sin_i,
    /// out[i + ropeHalf] = x[i + ropeHalf]*cos_i + x[i]*sin_i. Table entry i is the angle of the pair occupying head
    /// dimensions i and i + ropeHalf, which is how rotary q/k rows are laid out in converted qwen3 GGUF checkpoints.
    /// The angle table is the same one <see cref="BuildRopeTables"/> produces for the adjacent pairing.
    /// </summary>
    internal static void ApplyRopeToSplitHalfHead(Span<float> headRegion, int ropeHalf, ReadOnlySpan<float> cosAtPosition, ReadOnlySpan<float> sinAtPosition)
    {
        if (headRegion.Length < 2 * ropeHalf || cosAtPosition.Length < ropeHalf || sinAtPosition.Length < ropeHalf)
        {
            throw new ArgumentException("RoPE operands must cover a full head region and half-dimension tables.", nameof(headRegion));
        }

        for (int i = 0; i < ropeHalf; i++)
        {
            float c = cosAtPosition[i];
            float s = sinAtPosition[i];

            int lowIndex = i;
            int highIndex = i + ropeHalf;

            float lo = headRegion[lowIndex];
            float hi = headRegion[highIndex];

            headRegion[lowIndex] = lo * c - hi * s;
            headRegion[highIndex] = hi * c + lo * s;
        }
    }

    /// <summary>
    /// Causal attention for one layer at the current position: head h reads kv-head (h * kvHeadCount / headCount),
    /// scores are q·k multiplied by attnScale — then capped by <see cref="Softcap"/> when softcap is positive —
    /// over logical positions 0..position, and softmax runs online with a running max/sum. A positive window
    /// narrows that causal range to <c>[max(0, position - window + 1), position]</c>: keys further back than the
    /// window are never scored, so they contribute nothing to the softmax. The narrowing happens before the online
    /// max, so the running max and sum stay exact over the positions that do contribute. A non-positive window
    /// leaves the range unbounded. The narrowed range is resolved once into a <see cref="BlockSelection"/> by
    /// <see cref="SequenceKv.SelectCausalRead"/>, and every K/V read of this call addresses a position inside that
    /// selection through <see cref="SequenceKv.ReadK(BlockSelection, int, int)"/>: the pass walks exactly the selected
    /// blocks in ascending order — reading each block's slots that fall in range, partial at either edge — so an
    /// out-of-window physical block is never addressed rather than merely skipped by a loop bound, and a read outside
    /// the selection fails instead of reaching past the window. Positions excluded this way stay written and readable
    /// for every other layer's own selection. The weighted V accumulation is normalized in place into outConcat
    /// (head-major layout). A non-positive softcap disables the cap: the score path is then exactly the uncapped one.
    /// </summary>
    internal static void CausalAttention(
        IBackend backend,
        ReadOnlySpan<float> qHeads, // headCount * headDim
        int headCount,
        int kvHeadCount,
        int headDim,
        SequenceKv kv,
        int layer,
        int position,
        float attnScale,
        float softcap, // positive enables the score softcap; non-positive disables it
        int window, // positive narrows the causal range to the window; non-positive leaves it unbounded
        Span<float> outConcat) // headCount * headDim
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(kv);

        if (qHeads.Length < headCount * headDim || outConcat.Length < headCount * headDim)
        {
            throw new ArgumentException("Attention operands must cover every head's full region.", nameof(qHeads));
        }

        BlockSelection read = kv.SelectCausalRead(position, window); // the in-window positions and exactly which blocks they live in
        int blockTokens = kv.BlockTokens;

        for (int h = 0; h < headCount; h++)
        {
            ReadOnlySpan<float> qh = qHeads.Slice(h * headDim, headDim);
            int kvHead = h * kvHeadCount / headCount; // grouped-query mapping: consecutive heads share a kv-head
            Span<float> outHead = outConcat.Slice(h * headDim, headDim);

            float runningMax = float.NegativeInfinity;
            float sumExp = 0f;
            outHead.Clear();

            for (int blockIndex = read.FirstBlockIndex; blockIndex <= read.LastBlockIndex; blockIndex++)
            {
                // Only the selected block's slots inside the windowed range: a partial first or last block contributes its
                // in-range positions, and no position outside the selection is ever addressed.
                int slotFirst = Math.Max(read.FirstPosition, blockIndex * blockTokens);
                int slotLast = Math.Min(read.LastPosition, (blockIndex + 1) * blockTokens - 1);

                for (int t = slotFirst; t <= slotLast; t++)
                {
                    ReadOnlySpan<float> kt = kv.ReadK(read, t, layer).Slice(kvHead * headDim, headDim);
                    float score = backend.Dot(qh, kt) * attnScale;
                    if (softcap > 0f)
                    {
                        score = Softcap(score, softcap); // after scaling, before the online softmax sees it
                    }

                    if (score > runningMax)
                    {
                        // Rescale everything accumulated under the old max so all weights share the new exponent base.
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

                    float weight = MathF.Exp(score - runningMax); // the current max term contributes exactly 1
                    sumExp += weight;

                    ReadOnlySpan<float> vt = kv.ReadV(read, t, layer).Slice(kvHead * headDim, headDim);
                    for (int j = 0; j < headDim; j++)
                    {
                        outHead[j] += weight * vt[j];
                    }
                }
            }

            float invSum = 1f / sumExp; // >= 1 because the max term contributes exp(0)
            for (int j = 0; j < headDim; j++)
            {
                outHead[j] *= invSum;
            }
        }
    }

    /// <summary>
    /// Causal attention for one layer at the current position over quantized KV block storage: the generic twin of
    /// <see cref="CausalAttention"/> over a <see cref="SequenceKv{TK, TV}"/>, instantiated once per resolved (K,V) format
    /// pair so the hot body JIT-specializes on the concrete block structs and the loop carries no runtime format switch and
    /// no per-token virtual dispatch. The numeric path mirrors the float kernel exactly: head h reads kv-head
    /// (h * kvHeadCount / headCount); each score is the quantized dot result multiplied by attnScale — then capped by
    /// <see cref="Softcap"/> when softcap is positive — over the same causal range, and the online softmax keeps the same
    /// running max/sum update order, rescale-on-new-max rule, and final 1f / sumExp normalization. A positive window
    /// narrows the range exactly as the float pass does, resolved once into a <see cref="BlockSelection"/> by
    /// <see cref="SequenceKv{TK, TV}.SelectCausalRead"/>, and every K/V read addresses a position inside that selection —
    /// an out-of-window physical block is never addressed. The K side scores through the dot side's seam: the query heads
    /// are prepared once per call (one Q8_K-quantized activation per head, reused across calls through the caller-owned
    /// scratch and scale spans) and each scored position's kv-head region is dotted as exact integer sub-block totals under
    /// the resolved <paramref name="dot"/> encoding, scaled by the stored group scales and the activation super-block scales.
    /// The V side decodes each stored group under its own scale and accumulates <c>weight * value</c> in element order, the
    /// same order the float kernel's <c>outHead[j] += weight * vt[j]</c> uses. A non-positive softcap disables the cap: the
    /// score path is then exactly the uncapped one.
    /// </summary>
    internal static void CausalAttention<TK, TV>(
        ReadOnlySpan<float> qHeads, // headCount * headDim
        int headCount,
        int kvHeadCount,
        int headDim,
        SequenceKv<TK, TV> kv,
        int layer,
        int position,
        float attnScale,
        float softcap, // positive enables the score softcap; non-positive disables it
        int window, // positive narrows the causal range to the window; non-positive leaves it unbounded
        IntegerDotPath dot, // the integer-dot encoding resolved at backend open: this call carries it, it never re-resolves it
        Span<byte> queryScratch, // caller-owned prepared-query scratch, reused across calls: one contiguous run per head as the dot side lays it out
        Span<float> queryScales, // caller-owned activation scales, reused across calls, in the dot side's per-head order
        Span<float> outConcat) // headCount * headDim
        where TK : struct, IKvFormat<TK>, IKvDotSide<TK>
        where TV : struct, IKvFormat<TV>, IKvValueSide<TV>
    {
        ArgumentNullException.ThrowIfNull(kv);

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvHeadCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);

        int region = headCount * headDim;
        if (qHeads.Length < region || outConcat.Length < region)
        {
            throw new ArgumentException("Attention operands must cover every head's full region.", nameof(qHeads));
        }

        if (headDim % TK.ValuesPerBlock != 0 || headDim % TV.ValuesPerBlock != 0)
        {
            throw new ArgumentException(
                $"quantized attention needs whole groups per head region: headDim={headDim} is not a whole number of {TK.ValuesPerBlock}- and {TV.ValuesPerBlock}-value groups.",
                nameof(headDim));
        }

        BlockSelection read = kv.SelectCausalRead(position, window); // the in-window positions and exactly which blocks they live in
        int blockTokens = kv.BlockTokens;

        int kBlocksPerHead = headDim / TK.ValuesPerBlock; // each kv-head's groups sit contiguously inside the token region
        int vBlocksPerHead = headDim / TV.ValuesPerBlock;

        TK.PrepareQuery(qHeads, headCount, headDim, queryScratch, queryScales); // once per attention call: the quantized-activation step this pair stages

        for (int h = 0; h < headCount; h++)
        {
            ReadOnlySpan<float> qh = qHeads.Slice(h * headDim, headDim);
            int kvHead = h * kvHeadCount / headCount; // grouped-query mapping: consecutive heads share a kv-head
            int kBlockOffset = kvHead * kBlocksPerHead;
            int vBlockOffset = kvHead * vBlocksPerHead;
            Span<float> outHead = outConcat.Slice(h * headDim, headDim);

            float runningMax = float.NegativeInfinity;
            float sumExp = 0f;
            outHead.Clear();

            for (int blockIndex = read.FirstBlockIndex; blockIndex <= read.LastBlockIndex; blockIndex++)
            {
                // Only the selected block's slots inside the windowed range: a partial first or last block contributes its
                // in-range positions, and no position outside the selection is ever addressed.
                int slotFirst = Math.Max(read.FirstPosition, blockIndex * blockTokens);
                int slotLast = Math.Min(read.LastPosition, (blockIndex + 1) * blockTokens - 1);

                for (int t = slotFirst; t <= slotLast; t++)
                {
                    ReadOnlySpan<TK> kRegion = kv.ReadK(read, t, layer);
                    float score = TK.DotHead(kRegion, kBlockOffset, qh, queryScratch, queryScales, h, dot) * attnScale;
                    if (softcap > 0f)
                    {
                        score = Softcap(score, softcap); // after scaling, before the online softmax sees it
                    }

                    if (score > runningMax)
                    {
                        // Rescale everything accumulated under the old max so all weights share the new exponent base.
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

                    float weight = MathF.Exp(score - runningMax); // the current max term contributes exactly 1
                    sumExp += weight;

                    ReadOnlySpan<TV> vRegion = kv.ReadV(read, t, layer);
                    TV.AccumulateHead(outHead, vRegion, vBlockOffset, weight); // weighted values in element order, the same order the float kernel accumulates in
                }
            }

            float invSum = 1f / sumExp; // >= 1 because the max term contributes exp(0)
            for (int j = 0; j < headDim; j++)
            {
                outHead[j] *= invSum;
            }
        }
    }

    /// <summary>
    /// Row-tiled causal attention for one layer over a sub-batch of consecutive prompt rows: the walk inverts the
    /// single-row pass's nesting — each kv-head group, then each selected block, then each in-range position — so every
    /// attending row shares that position's K and V read instead of re-reading it from its own loop. Each head maps to
    /// kv-head (h * kvHeadCount / headCount) exactly as <see cref="CausalAttention"/> does, and the walk visits heads in
    /// ascending order within each group, so every query reads its assigned K/V slice under the same grouped-query
    /// mapping. Every row keeps its own online-softmax state — running max, weight sum, accumulator — across the whole
    /// block/position walk: a row's operation sequence is then ascending positions with the same <c>runningMax</c> update
    /// rule, the same rescale-on-new-max arithmetic (<c>sumExp *= rescale</c>, <c>outHead[j] *= rescale</c>, then
    /// <c>weight = exp(score - runningMax)</c>, <c>sumExp += weight</c>, <c>outHead[j] += weight * vt[j]</c>) and the same
    /// final <c>1f / sumExp</c> normalization as that row's own single-row pass over its selection, so a tiled window is
    /// bit-for-bit what forwarding each row alone would leave. The rows are consecutive positions of one sequence starting
    /// at <paramref name="fromPosition"/>: at position t only the rows whose own position reaches at least t attend — no
    /// zero-weight multiply and no skipped max update for a closed row, exactly as today where a row simply never scores
    /// past itself. A positive window narrows each row's causal range to <c>[max(0, q - window + 1), q]</c> relative to its
    /// own position before the online max sees any score; at t only rows whose backward span reaches t are in range — a
    /// row never reads outside its selection. The walk addresses exactly the union of those per-row causal selections as one
    /// contiguous run resolved through <see cref="SequenceKv.SelectRangeRead"/> and read via
    /// <see cref="SequenceKv.ReadK(BlockSelection, int, int)"/>, so no block outside any row's selection is addressed rather than
    /// merely skipped by a loop bound. The scores carry the same scaled-then-capped path as the single-row kernel: q·k times
    /// attnScale per row, then <see cref="Softcap"/> when softcap is positive, before that row's running max sees them. Row
    /// state lives in caller-owned arrays indexed by head and row (head-major within each row), reused across calls so the
    /// pass allocates nothing itself. A non-positive softcap disables the cap: the score path stays exactly the uncapped one.
    /// </summary>
    internal static void CausalAttentionTiledRows(
        IBackend backend,
        ReadOnlySpan<float> qRows, // rowCount * headCount * headDim, row-major: every row's own rotated queries
        int fromPosition, // absolute position of the first row; consecutive rows follow it in order
        int rowCount,
        int headCount,
        int kvHeadCount,
        int headDim,
        SequenceKv kv,
        int layer,
        float attnScale,
        float softcap, // positive enables the score softcap; non-positive disables it
        int window, // positive narrows each row's causal range to the window; non-positive leaves it unbounded
        Span<float> runningMaxRowState, // headCount * rowCount: every row's per-head online max across the walk
        Span<float> sumExpRowState, // same shape: every row's per-head weight sum across the walk
        Span<float> outConcat) // rowCount * headCount * headDim, row-major accumulator and result (head-major within each row)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(kv);

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rowCount);
        if (fromPosition < 0 || fromPosition + rowCount > kv.Length)
        {
            throw new ArgumentException(
                $"rows [{fromPosition}..{fromPosition + rowCount - 1}] must lie inside the positions this sequence holds ({kv.Length}): every row's K/V is written before any of its attention runs.",
                nameof(rowCount));
        }

        int region = headCount * headDim;
        if (qRows.Length < rowCount * region || outConcat.Length < rowCount * region)
        {
            throw new ArgumentException("Attention operands must cover every row's full set of heads.", nameof(qRows));
        }

        if (runningMaxRowState.Length < headCount * rowCount || sumExpRowState.Length < headCount * rowCount)
        {
            throw new ArgumentException("Row-state arrays must hold one entry per head for every row.", nameof(runningMaxRowState));
        }

        // The union of the rows' own causal reads: consecutive positions make each later row's backward span start at or after its predecessor's, so the selections join into exactly this contiguous run — from the earliest visible position to the last row's own. Under a positive window that is the first row's window edge; an unbounded read reaches back to position 0 of the sequence rather than merely where this sub-batch begins, covering every earlier position each row already attends. Reads walk only these blocks; a partial block at either edge contributes its in-range slots like every other causal read here does.
        BlockSelection read = kv.SelectRangeRead(
            window > 0 ? Math.Max(0, fromPosition - window + 1) : 0,
            fromPosition + rowCount - 1);

        int blockTokens = kv.BlockTokens;

        // Seed every row's online-softmax state exactly as the single-row pass seeds each head it runs: running max at negative infinity, weight sum at zero, accumulator cleared. The state arrays live across blocks and positions while a row walks them in ascending order.
        outConcat.Slice(0, rowCount * region).Clear();

        for (int h = 0; h < headCount; h++)
        {
            for (int r = 0; r < rowCount; r++)
            {
                runningMaxRowState[r * headCount + h] = float.NegativeInfinity;
                sumExpRowState[r * headCount + h] = 0f;
            }
        }

        // Grouped-query walk: heads sharing one kv-head read the same K/V slice at a position, so that pair is loaded once per group and block/position here — instead of once per row as in the single-row pass, where every row re-reads it from its own loop. The grouping comes from the single-row pass's own mapping: consecutive ascending runs of h * kvHeadCount / headCount, so each query pairs with exactly its assigned kv-head and heads walk in ascending order within a group.
        for (int hStart = 0; hStart < headCount;)
        {
            int kvHead = hStart * kvHeadCount / headCount; // grouped-query mapping: consecutive heads share a kv-head

            int groupEnd = hStart + 1;
            while (groupEnd < headCount && groupEnd * kvHeadCount / headCount == kvHead)
            {
                groupEnd++;
            }

            for (int blockIndex = read.FirstBlockIndex; blockIndex <= read.LastBlockIndex; blockIndex++)
            {
                // Only the selected block's slots inside the unioned range: a partial first or last block contributes its in-range positions, and no position outside any row's selection is ever addressed.
                int slotFirst = Math.Max(read.FirstPosition, blockIndex * blockTokens);
                int slotLast = Math.Min(read.LastPosition, (blockIndex + 1) * blockTokens - 1);

                for (int t = slotFirst; t <= slotLast; t++)
                {
                    ReadOnlySpan<float> kt = kv.ReadK(read, t, layer).Slice(kvHead * headDim, headDim); // this position's K read once for the whole group
                    ReadOnlySpan<float> vt = kv.ReadV(read, t, layer).Slice(kvHead * headDim, headDim); // and its V likewise

                    // The rows whose own causal selection covers this position form a contiguous run: causality needs each row's own position to reach at least t — earlier rows are closed against it; under a positive window only the rows whose backward span reaches t qualify. A closed or out-of-selection row never touches this slot at all — no zero-weight multiply, no skipped max update — exactly as today where a row simply scores neither past its own position nor outside its selection.
                    int rLow = Math.Max(0, t - fromPosition); // causal floor: only rows whose position reaches the scored one are open to it
                    int rHigh = window > 0 ? Math.Min(rowCount - 1, t + window - 1 - fromPosition) : rowCount - 1; // a positive window excludes every row whose own selection starts past the position

                    for (int r = rLow; r <= rHigh; r++)
                    {
                        ReadOnlySpan<float> qRow = qRows.Slice(r * region, region); // this row's queries: each head slices its assigned region under the same grouped-query mapping as the single-row pass
                        Span<float> outHeadRow = outConcat.Slice(r * region, region);

                        for (int h = hStart; h < groupEnd; h++)
                        {
                            ReadOnlySpan<float> qh = qRow.Slice(h * headDim, headDim);
                            int state = r * headCount + h; // this row's online-softmax entry for the walked head across blocks and positions

                            float score = backend.Dot(qh, kt) * attnScale;
                            if (softcap > 0f)
                            {
                                score = Softcap(score, softcap); // after scaling, before that row's online max sees it — per row as in the single-row pass
                            }

                            if (score > runningMaxRowState[state])
                            {
                                // Rescale everything accumulated under this row's old max so all its weights share the new exponent base.
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

                            float weight = MathF.Exp(score - runningMaxRowState[state]); // the current max term contributes exactly 1
                            sumExpRowState[state] += weight;

                            for (int j = 0; j < headDim; j++)
                            {
                                outHeadRow[h * headDim + j] += weight * vt[j];
                            }
                        }
                    }
                }
            }

            hStart = groupEnd; // the next group starts where this one's kv-head run ends
        }

        // Final normalization per row and per head, the same 1f / sumExp the single-row pass applies when its walk finishes. Every (row, head) pair saw exactly its selection's positions in ascending order with identical updates — so this kernel lands bit-for-bit on what forwarding each row alone would leave.
        for (int r = 0; r < rowCount; r++)
        {
            Span<float> outHeadRow = outConcat.Slice(r * region, region);

            for (int h = 0; h < headCount; h++)
            {
                float invSum = 1f / sumExpRowState[r * headCount + h]; // >= 1 because the max term contributes exp(0)
                for (int j = 0; j < headDim; j++)
                {
                    outHeadRow[h * headDim + j] *= invSum;
                }
            }
        }
    }

    /// <summary>
    /// Full-sequence attention over one layer's KV contents: every query position attends to all logical positions in
    /// the window instead of just a causal prefix. Queries are given position-major (queryCount * headCount * headDim)
    /// so a whole batch is processed in one pass; each head maps to kv-head (h * kvHeadCount / headCount) and runs the
    /// same online softmax as <see cref="CausalAttention"/>, including its scaled-then-capped score path when
    /// softcap is positive, with its score window widened from [0, position] to
    /// [fromPosition, toPosition], which must lie inside the sequence's existing KV (<see cref="SequenceKv.Length"/>).
    /// The queries' own absolute positions start at <paramref name="queryPositionOffset"/>, so a positive window narrows
    /// each query's range backward-only to <c>[max(fromPosition, q - window + 1), min(toPosition, q)]</c> relative to its
    /// own position <c>q</c>: positions further back than the window are never scored, and neither is any position past
    /// the query itself — a positive window never grants forward visibility. The narrowing happens before the online max,
    /// so the running max and sum stay exact over the positions that do contribute; a query whose windowed range does not
    /// intersect the given KV range is a failure. A non-positive window leaves the given range untouched, which is what
    /// keeps an unbounded layer bidirectional over it.
    /// Restricting that window yields a block-local variant used when long sequences are processed in chunks: every
    /// query then sees exactly one contiguous range of positions. With fromPosition == 0 and toPosition == p over a
    /// sequence of length p + 1 the numeric path is identical to <see cref="CausalAttention"/> at position p, which
    /// lets chunk boundaries be verified against the causal decode kernels bit-for-bit; with a positive window it is
    /// identical to that kernel's own narrowed range at the same query under the same window. Each query's resolved
    /// range becomes a <see cref="BlockSelection"/> from <see cref="SequenceKv.SelectRangeRead"/>, and its reads walk only
    /// that selection's blocks — positions outside it are never addressed, exactly as in the causal path.
    /// </summary>
    internal static void BidirectionalAttention(
        IBackend backend,
        ReadOnlySpan<float> qPositions, // queryCount * headCount * headDim, position-major
        int queryCount,
        int queryPositionOffset, // absolute position of the first query
        int fromPosition,
        int toPosition,
        int headCount,
        int kvHeadCount,
        int headDim,
        SequenceKv kv,
        int layer,
        float attnScale,
        float softcap, // positive enables the score softcap; non-positive disables it
        int window, // positive narrows each query's range to its backward span ending at the query itself; non-positive leaves it as given
        Span<float> outConcat) // queryCount * headCount * headDim, position-major
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(kv);

        ArgumentOutOfRangeException.ThrowIfNegative(queryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(queryPositionOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(fromPosition);
        if (toPosition < fromPosition || toPosition >= kv.Length)
        {
            throw new ArgumentException(
                $"query window [{fromPosition}..{toPosition}] must lie inside the 0..{kv.Length - 1} positions already held by the sequence.",
                nameof(toPosition));
        }

        int region = headCount * headDim;
        if (qPositions.Length < queryCount * region || outConcat.Length < queryCount * region)
        {
            throw new ArgumentException("Attention operands must cover every query's full set of heads.", nameof(qPositions));
        }

        for (int p = 0; p < queryCount; p++)
        {
            int queryPosition = queryPositionOffset + p;
            int firstVisible = fromPosition;
            int lastVisible = toPosition;
            if (window > 0)
            {
                // Backward-only: the window reaches behind the query and stops at the query itself, so this is exactly
                // the range SequenceKv.SelectCausalRead resolves for the same position and window. The literal
                // [fromPosition..toPosition] edges still bound it; a non-positive window keeps them as given.
                firstVisible = Math.Max(firstVisible, queryPosition - window + 1);
                lastVisible = Math.Min(lastVisible, queryPosition);

                if (firstVisible > lastVisible)
                {
                    throw new ArgumentException(
                        $"query at position {queryPosition} has no position inside [{fromPosition}..{toPosition}] within its window of {window}.",
                        nameof(window));
                }
            }

            ReadOnlySpan<float> qAtPosition = qPositions.Slice(p * region, region);
            Span<float> outAtPosition = outConcat.Slice(p * region, region);

            // This query's own windowed range resolved into the blocks it covers: reads walk that selection only.
            BlockSelection read = kv.SelectRangeRead(firstVisible, lastVisible);
            int blockTokens = kv.BlockTokens;

            for (int h = 0; h < headCount; h++)
            {
                ReadOnlySpan<float> qh = qAtPosition.Slice(h * headDim, headDim);
                int kvHead = h * kvHeadCount / headCount; // grouped-query mapping: consecutive heads share a kv-head
                Span<float> outHead = outAtPosition.Slice(h * headDim, headDim);

                float runningMax = float.NegativeInfinity;
                float sumExp = 0f;
                outHead.Clear();

                for (int blockIndex = read.FirstBlockIndex; blockIndex <= read.LastBlockIndex; blockIndex++)
                {
                    int slotFirst = Math.Max(read.FirstPosition, blockIndex * blockTokens);
                    int slotLast = Math.Min(read.LastPosition, (blockIndex + 1) * blockTokens - 1);

                    for (int t = slotFirst; t <= slotLast; t++)
                    {
                        ReadOnlySpan<float> kt = kv.ReadK(read, t, layer).Slice(kvHead * headDim, headDim);
                        float score = backend.Dot(qh, kt) * attnScale;
                        if (softcap > 0f)
                        {
                            score = Softcap(score, softcap); // after scaling, before the online softmax sees it
                        }

                        if (score > runningMax)
                        {
                            // Rescale everything accumulated under the old max so all weights share the new exponent base.
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

                        float weight = MathF.Exp(score - runningMax); // the current max term contributes exactly 1
                        sumExp += weight;

                        ReadOnlySpan<float> vt = kv.ReadV(read, t, layer).Slice(kvHead * headDim, headDim);
                        for (int j = 0; j < headDim; j++)
                        {
                            outHead[j] += weight * vt[j];
                        }
                    }
                }

                float invSum = 1f / sumExp; // >= 1 because the max term contributes exp(0)
                for (int j = 0; j < headDim; j++)
                {
                    outHead[j] *= invSum;
                }
            }
        }
    }
}
