namespace Lair.Backends.Cpu;

// Production visibility seams over the Q8_K activation-quant machinery this assembly already ships, opened for the
// quantized KV attention kernels in the storage layer. Each member is a pass-through to an existing kernel helper — no
// second copy of any quantizer or dot arithmetic exists outside this assembly — so a quantized K-dot stages its query
// side through exactly the scale rule and rounding every sub-4-bit GEMV activation uses, and scores stored groups with
// exactly the integer dots those GEMVs run. The seams are static and internal: they reach no host configuration path on
// their own, carry no state, and a backend opened for another posture never consults them.
public sealed unsafe partial class CpuBackend
{
    /// <summary>The Q8_K payload bytes one activation vector of <paramref name="cols"/> values occupies after quantization.</summary>
    internal static int KvQ8KActivationBytes(int cols) => Q8KActivationBytes(cols);

    /// <summary>The Q8_K super-block count (and so scale count) one activation vector of <paramref name="cols"/> values quantizes into.</summary>
    internal static int KvQ8KBlockCount(int cols) => Q8KBlockCount(cols);

    /// <summary>Quantizes one FP activation vector exactly as the sub-4-bit GEMVs do: same scale rule, same rounding, zero block stays a zero dot contribution.</summary>
    internal static void KvQuantizeActivationToQ8K(float* x, int cols, byte* qa, float* da) =>
        QuantizeActivationToQ8K(x, cols, qa, da);

    /// <summary>The resolved-encoding integer dot over one 32-value sub-block: magnitudes times sign-folded Q8_K quants, exact int32 half totals.</summary>
    internal static void KvSubBlockDotQ8K(byte* mag, uint signs, sbyte* act, out int loSum, out int hiSum, IntegerDotPath dot) =>
        SubBlockDotQ8KVector(mag, signs, act, out loSum, out hiSum, dot);

    /// <summary>The scalar reference sub-block dot the vector encodings are gated against bit-for-bit.</summary>
    internal static void KvSubBlockDotQ8KScalar(byte* mag, uint signs, sbyte* act, out int loSum, out int hiSum) =>
        SubBlockDotQ8KScalar(mag, signs, act, out loSum, out hiSum);
}
