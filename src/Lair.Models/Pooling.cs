namespace Lair.Models;

/// <summary>
/// Pure math over hidden-state vectors produced by an encoder forward: mean pooling (the default sentence-vector
/// recipe), CLS-style extraction of the first position's vector, and optional L2 normalization. Vectors are given as a
/// flat row-major buffer (<c>vectors[position * vectorLength + i]</c>) so no allocation happens on the way in; every
/// helper writes an exact <paramref name="vectorLength"/>-sized result into the caller's span and reads only inputs, so
/// repeated calls over unchanged values are bit-for-bit repeatable. The mean accumulates in double before the final
/// division, matching how the engine's norms accumulate.
/// </summary>
public static class Pooling
{
    /// <summary>pooled[i] = mean over positions of vectors[position * vectorLength + i].</summary>
    public static void MeanPool(ReadOnlySpan<float> vectors, int vectorLength, Span<float> pooled) =>
        AccumulateMean(vectors, vectorLength, pooled);

    /// <summary>
    /// Same row-wise mean as <see cref="MeanPool"/> over the first <paramref name="positionCount"/> rows only; the rest
    /// of the flat buffer is not read. A trailing partial chunk can therefore be folded in separately and combined with
    /// weighted addition to reproduce a whole-sequence mean bit-for-bit.
    /// </summary>
    public static void MeanPool(ReadOnlySpan<float> vectors, int vectorLength, int positionCount, Span<float> pooled)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(positionCount);
        AccumulateMean(vectors[..checked(positionCount * vectorLength)], vectorLength, pooled);
    }

    /// <summary>pooled[i] = vectors[i]: the first row of a flat buffer verbatim (CLS-style extraction).</summary>
    public static void FirstToken(ReadOnlySpan<float> vectors, int vectorLength, Span<float> pooled)
    {
        if (vectorLength <= 0 || vectors.Length < vectorLength || pooled.Length < vectorLength)
        {
            throw new ArgumentException($"need a flat buffer covering at least one row of {vectorLength} floats.", nameof(vectors));
        }

        vectors[..vectorLength].CopyTo(pooled[..vectorLength]); // verbatim bits: no arithmetic touches the values
    }

    /// <summary>Divides vector[0..length) by its L2 norm in place; a zero vector is left untouched.</summary>
    public static void NormalizeL2(Span<float> vector)
    {
        double sumSq = 0.0; // same accumulation discipline as the engine's rmsnorm
        for (int i = 0; i < vector.Length; i++)
        {
            sumSq += (double)vector[i] * vector[i];
        }

        if (sumSq == 0.0)
        {
            return; // no direction to preserve: the zero vector stays zero instead of dividing by zero
        }

        float invNorm = (float)(1.0 / Math.Sqrt(sumSq));
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] *= invNorm;
        }
    }

    private static void AccumulateMean(ReadOnlySpan<float> vectors, int vectorLength, Span<float> pooled)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vectorLength);
        if (vectors.Length % vectorLength != 0 || pooled.Length < vectorLength)
        {
            throw new ArgumentException($"flat buffer of {vectors.Length} floats must cover whole rows of {vectorLength}, and the output must hold one row.", nameof(vectors));
        }

        int positions = vectors.Length / vectorLength;
        if (positions == 0)
        {
            throw new ArgumentException("at least one position row is required.", nameof(vectors));
        }

        for (int i = 0; i < vectorLength; i++)
        {
            double sum = 0.0; // column runs down the rows in order: deterministic across calls and thread counts
            for (int p = 0; p < positions; p++)
            {
                sum += vectors[p * vectorLength + i];
            }

            pooled[i] = (float)(sum / positions);
        }
    }
}
