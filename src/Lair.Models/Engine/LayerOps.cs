using Lair.Contracts;
using Lair.Kv;

namespace Lair.Models.Engine;

/// <summary>
/// The pure numeric operations of the transformer forward pass, isolated so each can be verified against
/// hand-computed values independently of any model file. RoPE uses GGUF/llama-style adjacent pairs (2i, 2i + 1)
/// with theta_i = base^(-2i/headDim); RMSNorm accumulates in double; attention runs an online softmax over the paged
/// KV cache so no score row is ever materialized.
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

    /// <summary>SiLU activation: x / (1 + exp(-x)).</summary>
    internal static float Silu(float x) => x / (1f + MathF.Exp(-x));

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
    /// scores are q·k multiplied by attnScale over logical positions 0..position, and softmax runs online with a running
    /// max/sum. The weighted V accumulation is normalized in place into outConcat (head-major layout).
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
    /// same online softmax as <see cref="CausalAttention"/> with its score window widened from [0, position] to
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
