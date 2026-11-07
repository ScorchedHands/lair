using Lair.Contracts;
using Lair.Kv;

namespace Lair.Models.Engine;

/// <summary>
/// The pure numeric operations of the transformer forward pass, isolated so each can be verified against
/// hand-computed values independently of any model file. RoPE uses GGUF/llama-style adjacent pairs (2i, 2i + 1)
/// with theta_i = base^(-2i/headDim); RMSNorm and LayerNorm accumulate in double; attention runs an online softmax
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
    /// as float so the same inputs always produce bit-identical rotations.
    /// </summary>
    internal static void BuildRopeTables(double thetaBase, int headDim, int maxPositions, Span<float> cos, Span<float> sin)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);

        if (headDim % 2 != 0)
        {
            throw new ArgumentException($"RoPE pairs require an even head dimension, got {headDim}.", nameof(headDim));
        }

        int half = headDim / 2;
        if (cos.Length < maxPositions * half || sin.Length < maxPositions * half)
        {
            throw new ArgumentException("Cos/sin tables must cover maxPositions * headDim/2 entries each.", nameof(cos));
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
    /// Rotates one head region in place with GGUF/llama-style adjacent pairs: out[2i] = x[2i]*cos_i - x[2i+1]*sin_i,
    /// out[2i + 1] = x[2i + 1]*cos_i + x[2i]*sin_i. Table entry i is the angle of the pair occupying adjacent head
    /// dimensions 2i and 2i + 1, which is how rotary q/k rows are laid out in converted GGUF checkpoints.
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
    /// Causal attention for one layer at the current position: head h reads kv-head (h * kvHeadCount / headCount),
    /// scores are q·k multiplied by attnScale — then capped by <see cref="Softcap"/> when softcap is positive —
    /// over logical positions 0..position, and softmax runs online with a running max/sum. The weighted V
    /// accumulation is normalized in place into outConcat (head-major layout). A non-positive softcap disables
    /// the cap: the score path is then exactly the uncapped one.
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
        Span<float> outConcat) // headCount * headDim
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(kv);

        if (qHeads.Length < headCount * headDim || outConcat.Length < headCount * headDim)
        {
            throw new ArgumentException("Attention operands must cover every head's full region.", nameof(qHeads));
        }

        for (int h = 0; h < headCount; h++)
        {
            ReadOnlySpan<float> qh = qHeads.Slice(h * headDim, headDim);
            int kvHead = h * kvHeadCount / headCount; // grouped-query mapping: consecutive heads share a kv-head
            Span<float> outHead = outConcat.Slice(h * headDim, headDim);

            float runningMax = float.NegativeInfinity;
            float sumExp = 0f;
            outHead.Clear();

            for (int t = 0; t <= position; t++)
            {
                ReadOnlySpan<float> kt = kv.ReadK(t, layer).Slice(kvHead * headDim, headDim);
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

                ReadOnlySpan<float> vt = kv.ReadV(t, layer).Slice(kvHead * headDim, headDim);
                for (int j = 0; j < headDim; j++)
                {
                    outHead[j] += weight * vt[j];
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
    /// Full-sequence attention over one layer's KV contents: every query position attends to all logical positions in
    /// the window instead of just a causal prefix. Queries are given position-major (queryCount * headCount * headDim)
    /// so a whole batch is processed in one pass; each head maps to kv-head (h * kvHeadCount / headCount) and runs the
    /// same online softmax as <see cref="CausalAttention"/>, including its scaled-then-capped score path when
    /// softcap is positive, with its score window widened from [0, position] to
    /// [fromPosition, toPosition], which must lie inside the sequence's existing KV (<see cref="SequenceKv.Length"/>).
    /// Restricting that window yields a block-local variant used when long sequences are processed in chunks: every
    /// query then sees exactly one contiguous range of positions. With fromPosition == 0 and toPosition == p over a
    /// sequence of length p + 1 the numeric path is identical to <see cref="CausalAttention"/> at position p, which
    /// lets chunk boundaries be verified against the causal decode kernels bit-for-bit.
    /// </summary>
    internal static void BidirectionalAttention(
        IBackend backend,
        ReadOnlySpan<float> qPositions, // queryCount * headCount * headDim, position-major
        int queryCount,
        int fromPosition,
        int toPosition,
        int headCount,
        int kvHeadCount,
        int headDim,
        SequenceKv kv,
        int layer,
        float attnScale,
        float softcap, // positive enables the score softcap; non-positive disables it
        Span<float> outConcat) // queryCount * headCount * headDim, position-major
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(kv);

        ArgumentOutOfRangeException.ThrowIfNegative(queryCount);
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
            ReadOnlySpan<float> qAtPosition = qPositions.Slice(p * region, region);
            Span<float> outAtPosition = outConcat.Slice(p * region, region);

            for (int h = 0; h < headCount; h++)
            {
                ReadOnlySpan<float> qh = qAtPosition.Slice(h * headDim, headDim);
                int kvHead = h * kvHeadCount / headCount; // grouped-query mapping: consecutive heads share a kv-head
                Span<float> outHead = outAtPosition.Slice(h * headDim, headDim);

                float runningMax = float.NegativeInfinity;
                float sumExp = 0f;
                outHead.Clear();

                for (int t = fromPosition; t <= toPosition; t++)
                {
                    ReadOnlySpan<float> kt = kv.ReadK(t, layer).Slice(kvHead * headDim, headDim);
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

                    ReadOnlySpan<float> vt = kv.ReadV(t, layer).Slice(kvHead * headDim, headDim);
                    for (int j = 0; j < headDim; j++)
                    {
                        outHead[j] += weight * vt[j];
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
