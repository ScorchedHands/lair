using Lair.Contracts;

namespace Lair.Backends.Cpu;

// Measurement-only visibility seams over the Q8_K activation-quant machinery this assembly already ships.
// Nothing here changes what any production path runs: each member is a pass-through to an existing kernel helper so a
// candidate KV-format seam measured by the bench/tests dots against the same integer-dot code the i-quant GEMVs use,
// rather than carrying its own copy of the quantizer or the dot. No host configuration path reaches this file.
public sealed unsafe partial class CpuBackend
{
    /// <summary>The Q8_K super-block count one activation vector of <paramref name="cols"/> values quantizes into.</summary>
    internal static int KvSpikeQ8KBlockCount(int cols) => Q8KBlockCount(cols);

    /// <summary>The Q8_K payload bytes one activation vector of <paramref name="cols"/> values occupies after quantization.</summary>
    internal static int KvSpikeQ8KActivationBytes(int cols) => Q8KActivationBytes(cols);

    /// <summary>Quantizes one FP activation vector exactly as the sub-4-bit GEMVs do: same scale rule, same rounding.</summary>
    internal static void KvSpikeQuantizeActivationToQ8K(float* x, int cols, byte* qa, float* da) =>
        QuantizeActivationToQ8K(x, cols, qa, da);

    /// <summary>The resolved-encoding integer dot over one 32-value sub-block: magnitudes times sign-folded Q8_K quants.</summary>
    internal static void KvSpikeSubBlockDotQ8K(byte* mag, uint signs, sbyte* act, out int loSum, out int hiSum, IntegerDotPath dot) =>
        SubBlockDotQ8KVector(mag, signs, act, out loSum, out hiSum, dot);

    /// <summary>The scalar reference sub-block dot the vector encodings are gated against bit-for-bit.</summary>
    internal static void KvSpikeSubBlockDotQ8KScalar(byte* mag, uint signs, sbyte* act, out int loSum, out int hiSum) =>
        SubBlockDotQ8KScalar(mag, signs, act, out loSum, out hiSum);

    /// <summary>The AVX2 float dot this backend's <see cref="IBackend.Dot"/> runs on its vector posture.</summary>
    internal static float KvSpikeFloatDot(float* a, float* b, int count) => DotCore(a, b, count);

    /// <summary>The scalar reference float dot the no-vector posture runs.</summary>
    internal static float KvSpikeFloatDotScalar(float* a, float* b, int count) => DotScalar(a, b, count);

    /// <summary>Whether this opened backend's dense float kernels run on their AVX2 posture (the same gate <see cref="IBackend.Dot"/> reads).</summary>
    internal bool KvSpikeUsesAvx2FloatKernels => _useAvx2;
}
