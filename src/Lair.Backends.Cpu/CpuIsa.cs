using System.Runtime.Intrinsics.X86;

namespace Lair.Backends.Cpu;

/// <summary>
/// The CPU instruction-set axis <c>engine.cpu.isa</c> declares for the sub-4-bit i-quant integer dots. A selection is one
/// fact of an opened backend: <see cref="CpuBackend"/> resolves it once at open and reads the resolved value per kernel
/// call, so every row, batch vector, worker, and lane of a model runs the same encoding for that engine's whole lifetime.
/// </summary>
public enum CpuIsa
{
    /// <summary>Pick the highest available integer-dot path: AVX-VNNI when this host reports it, else AVX2.</summary>
    Auto,

    /// <summary>The pmaddubsw+pmaddwd integer dot. Forced means forced: a host without the AVX2 baseline fails the load naming it.</summary>
    Avx2,

    /// <summary>The vpdpbusd integer dot over the same staged magnitudes and sign-folded activation. A host without AVX-VNNI fails the load naming that feature.</summary>
    AvxVnni,
}

/// <summary>The integer-dot encoding one Q8_K i-quant kernel runs once the ISA axis is resolved at engine open.</summary>
public enum IntegerDotPath
{
    /// <summary>The scalar reference dot — the oracle every vector path is gated against bit-for-bit.</summary>
    Scalar,

    /// <summary>Adjacent-pair products summed in int16 lanes (pmaddubsw), then quadruple sums in int32 lanes (pmaddwd).</summary>
    Avx2,

    /// <summary>The same four-product groups widened and accumulated straight into int32 lanes by vpdpbusd.</summary>
    Vnni,
}

/// <summary>The vocabulary one engine.cpu.isa key accepts and the names a resolved selection reports, each in the order a refusal reports it.</summary>
public static class CpuIsaNames
{
    public const string Auto = "auto";
    public const string Avx2 = "avx2";
    public const string AvxVnni = "avx-vnni";

    /// <summary>The names one engine.cpu.isa declaration may state, in the order a refusal reports them.</summary>
    public static IReadOnlyList<string> DeclaredNames { get; } = [Auto, Avx2, AvxVnni];

    public static string Of(CpuIsa isa) => isa switch
    {
        CpuIsa.Avx2 => Avx2,
        CpuIsa.AvxVnni => AvxVnni,
        _ => Auto,
    };

    /// <summary>Reads one config name as an ISA selection. An unrecognized name is not a selection: callers report it naming the accepted vocabulary.</summary>
    public static bool TryParse(string? name, out CpuIsa isa)
    {
        switch (name)
        {
            case Auto or null:
                isa = CpuIsa.Auto;
                return true;

            case Avx2:
                isa = CpuIsa.Avx2;
                return true;

            case AvxVnni:
                isa = CpuIsa.AvxVnni;
                return true;

            default:
                isa = default;
                return false;
        }
    }

    /// <summary>The accepted set as the text a refusal names it by.</summary>
    public static string AcceptedVocabulary => $"only {string.Join(", ", DeclaredNames.Select(name => $"'{name}'"))}";
}

/// <summary>What one host reports about the two vector integer-dot encodings LAIR ships.</summary>
public readonly record struct CpuFeaturePosture(bool Avx2Available, bool AvxVnniAvailable)
{
    /// <summary>The CPU feature one forced selection names: AVX-VNNI for the vpdpbusd dot, plain AVX2 for the pair-then-quadruple dot.</summary>
    public static string FeatureNameFor(CpuIsa isa) => isa == CpuIsa.AvxVnni ? "AVX-VNNI" : "AVX2";

    /// <summary>The feature text one refusal reports for a forced selection this posture cannot run, naming the baseline too when that is what is missing as well.</summary>
    public string MissingFeatureFor(CpuIsa isa) => isa == CpuIsa.AvxVnni && !Avx2Available
        ? "AVX-VNNI (and the AVX2 baseline its staging and sign fold use)"
        : FeatureNameFor(isa);
}

/// <summary>The load-time capability probe, and the one resolution rule engine.cpu.isa resolves through.</summary>
public static class CpuIsaProbe
{
    /// <summary>
    /// The vector baseline this backend has always required: AVX2 plus FMA plus SSSE3. Kept as one expression so a posture
    /// can never claim an integer-dot path whose staging and sign-fold intrinsics are missing on the host reporting it.
    /// </summary>
    public static bool Avx2BaselineAvailable => Avx2.IsSupported && Fma.IsSupported && Ssse3.IsSupported;

    /// <summary>The posture this process reports right now: AVX-VNNI counts as available only beside the vector baseline its kernels still use.</summary>
    public static CpuFeaturePosture Detect() => new(Avx2BaselineAvailable, AvxVnni.IsSupported && Avx2BaselineAvailable);

    /// <summary>
    /// Resolves one requested axis value against one posture. An auto selection takes the highest available path — AVX-VNNI
    /// over AVX2 — and lands on the scalar reference dots only when this host reports no runnable vector path at all, which is
    /// exactly what an unpinned backend did before this axis existed. A forced selection resolves to itself or not at all: a
    /// posture that cannot run it returns false and names the CPU feature it needs instead of substituting some other path.
    /// </summary>
    public static bool TryResolve(CpuIsa requested, CpuFeaturePosture posture, out IntegerDotPath path, out string? missingFeature)
    {
        switch (requested)
        {
            case CpuIsa.Auto:
                missingFeature = null;

                if (VnniRunnable(posture))
                {
                    path = IntegerDotPath.Vnni;
                    return true;
                }

                if (posture.Avx2Available)
                {
                    path = IntegerDotPath.Avx2;
                    return true;
                }

                path = IntegerDotPath.Scalar; // nothing runnable this host reports: the reference kernels run, as they always have.
                return true;

            case CpuIsa.Avx2:
                if (!posture.Avx2Available)
                {
                    path = default;
                    missingFeature = posture.MissingFeatureFor(requested);
                    return false;
                }

                path = IntegerDotPath.Avx2;
                missingFeature = null;
                return true;

            default: // CpuIsa.AvxVnni
                if (!VnniRunnable(posture))
                {
                    path = default;
                    missingFeature = posture.MissingFeatureFor(requested);
                    return false;
                }

                path = IntegerDotPath.Vnni;
                missingFeature = null;
                return true;
        }
    }

    // vpdpbusd runs on this backend only beside the vector baseline its staging and sign fold use, so no posture - detected or
    // stated by a caller - resolves to it while AVX2 is missing. Without this guard an injected posture could hand a kernel path
    // whose load/store intrinsics the same open has already gated off.
    private static bool VnniRunnable(CpuFeaturePosture posture) => posture.AvxVnniAvailable && posture.Avx2Available;
}

/// <summary>Thrown when engine.cpu.isa forces an integer-dot path the host's CPU does not report. No other path is substituted.</summary>
public sealed class CpuIsaUnavailableException : InvalidOperationException
{
    public CpuIsa Requested { get; }

    /// <summary>The CPU feature this forced selection needs and the host does not report.</summary>
    public string MissingFeature { get; }

    public CpuIsaUnavailableException(CpuIsa requested, string missingFeature)
        : base(
            $"engine.cpu.isa '{CpuIsaNames.Of(requested)}' is unavailable on this host: the CPU does not report " +
            $"{missingFeature}, and a forced ISA is never answered with some other integer-dot path that happens to be available.")
    {
        Requested = requested;
        MissingFeature = missingFeature;
    }
}
