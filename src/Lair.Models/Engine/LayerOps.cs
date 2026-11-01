using Lair.Core.Contracts;
using Lair.Kv;

namespace Lair.Models.Engine;

/// <summary>
/// The pure numeric operations of the transformer forward pass, isolated so each can be verified against
/// hand-computed values independently of any model file. RoPE uses llama-style pairs (i, i + headDim/2) with
/// theta_i = base^(-2i/headDim); RMSNorm accumulates in double; attention runs an online softmax over the paged
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
    /// headDim/2 entries: angle(pos, i) = pos * base^(-2i/headDim), computed in double and stored as float so the
    /// same inputs always produce bit-identical rotations.
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

    /// <summary>Rotates one head region in place: out[i] = x[i]*cos - x[i+half]*sin, out[i+half] = x[i+half]*cos + x[i]*sin.</summary>
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

            float lo = headRegion[i];
            float hi = headRegion[i + ropeHalf];

            headRegion[i] = lo * c - hi * s;
            headRegion[i + ropeHalf] = hi * c + lo * s;
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
}
