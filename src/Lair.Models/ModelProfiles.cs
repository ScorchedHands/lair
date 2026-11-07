namespace Lair.Models;

/// <summary>
/// Registry of the supported model profiles, keyed by <c>general.architecture</c>. It is the single source
/// for profile selection and for the supported-profile list in the rejection message for an unsupported
/// architecture. A new profile is a new entry here plus its tensor map and strategy values. The generic
/// fallback profile lives outside the registry: it is never selectable by name and is applied only when a
/// file declares an unregistered architecture and its tensor table matches the llama-shaped map exactly.
/// </summary>
public static class ModelProfiles
{
    /// <summary>Identity of the generic llama-shaped fallback profile, recorded by the plan of a file accepted through it.</summary>
    public const string GenericArchitecture = "llama-generic";

    private static readonly ModelProfile[] Profiles =
    [
        new ModelProfile
        {
            Architecture = "llama",
            Prefix = "llama.",
            Norm = NormStrategy.RmsNorm,
            Activation = ActivationStrategy.SwiGLU,
            Bias = BiasPolicy.Declared,
            Rope = RopeStrategy.StandardFullHead,
            Tensors = CanonicalTensors(),
        },
        new ModelProfile
        {
            Architecture = "qwen2",
            Prefix = "qwen2.",
            Norm = NormStrategy.RmsNorm,
            Activation = ActivationStrategy.SwiGLU,
            Bias = BiasPolicy.Declared,
            Rope = RopeStrategy.StandardFullHead,
            Tensors = CanonicalTensors(),
        },
    ];

    private static readonly Dictionary<string, ModelProfile> ByArchitecture =
        Profiles.ToDictionary(profile => profile.Architecture, StringComparer.Ordinal);

    // The generic fallback profile: the llama strategy values and tensor map, re-prefixed to the declared
    // architecture when applied. It is deliberately absent from the registry array, so it is never selectable
    // by name and never appears in the supported-architecture list.
    private static readonly ModelProfile GenericProfile = new()
    {
        Architecture = GenericArchitecture,
        Prefix = GenericArchitecture + ".",
        Norm = NormStrategy.RmsNorm,
        Activation = ActivationStrategy.SwiGLU,
        Bias = BiasPolicy.ByTensorPresence,
        Rope = RopeStrategy.StandardFullHead,
        Tensors = CanonicalTensors(),
    };

    /// <summary>Supported architecture names, in registry order.</summary>
    public static IEnumerable<string> SupportedArchitectures => Profiles.Select(profile => profile.Architecture);

    /// <summary>
    /// Resolves the profile for one <c>general.architecture</c> value. An unsupported name fails at load with
    /// a <see cref="ModelException"/> naming the value and the registry's supported profiles.
    /// </summary>
    public static ModelProfile Resolve(string architecture)
    {
        if (ByArchitecture.TryGetValue(architecture, out ModelProfile? profile))
        {
            return profile;
        }

        throw new ModelException(
            $"model architecture '{architecture}' is not supported (supported architectures: " +
            $"{string.Join(", ", SupportedArchitectures)})");
    }

    /// <summary>
    /// Resolves the profile for one <c>general.architecture</c> value: the registered profile when the name is
    /// in the registry, otherwise the generic llama-shaped fallback profile re-prefixed to the declared
    /// architecture, whose metadata keys are read under <c>&lt;architecture&gt;.</c>. The fallback is accepted
    /// only after its exact tensor-table match is verified; the registry rejection stays available through
    /// <see cref="Resolve"/>.
    /// </summary>
    public static ModelProfile ResolveOrFallback(string architecture)
    {
        ArgumentNullException.ThrowIfNull(architecture);

        if (ByArchitecture.TryGetValue(architecture, out ModelProfile? profile))
        {
            return profile;
        }

        return GenericProfile with { Prefix = architecture + "." };
    }

    // The canonical llama-family tensor names: the llama.cpp GGUF layout both supported profiles ship, with
    // attention biases and the output head optional because llama files carry neither while qwen2 files may.
    // The per-head QK-norm vectors and the norm-bias vectors are optional too: they exist only for files that
    // declare the corresponding feature, and the declared flags decide whether they are required or ignored.
    private static TensorMap CanonicalTensors() =>
        new(
        [
            new TensorDescriptor(TensorRole.TokenEmbedding, TensorScope.Global, "token_embd.weight", Required: true, []),
            new TensorDescriptor(TensorRole.AttentionQuery, TensorScope.PerLayer, "blk.{layer}.attn_q.weight", Required: true, []),
            new TensorDescriptor(TensorRole.AttentionKey, TensorScope.PerLayer, "blk.{layer}.attn_k.weight", Required: true, []),
            new TensorDescriptor(TensorRole.AttentionValue, TensorScope.PerLayer, "blk.{layer}.attn_v.weight", Required: true, []),
            new TensorDescriptor(TensorRole.AttentionOutput, TensorScope.PerLayer, "blk.{layer}.attn_output.weight", Required: true, []),
            new TensorDescriptor(TensorRole.FeedForwardGate, TensorScope.PerLayer, "blk.{layer}.ffn_gate.weight", Required: true, []),
            new TensorDescriptor(TensorRole.FeedForwardUp, TensorScope.PerLayer, "blk.{layer}.ffn_up.weight", Required: true, []),
            new TensorDescriptor(TensorRole.FeedForwardDown, TensorScope.PerLayer, "blk.{layer}.ffn_down.weight", Required: true, []),
            new TensorDescriptor(TensorRole.AttentionNorm, TensorScope.PerLayer, "blk.{layer}.attn_norm.weight", Required: true, []),
            new TensorDescriptor(TensorRole.FeedForwardNorm, TensorScope.PerLayer, "blk.{layer}.ffn_norm.weight", Required: true, []),
            new TensorDescriptor(TensorRole.AttentionQueryBias, TensorScope.PerLayer, "blk.{layer}.attn_q.bias", Required: false, []),
            new TensorDescriptor(TensorRole.AttentionKeyBias, TensorScope.PerLayer, "blk.{layer}.attn_k.bias", Required: false, []),
            new TensorDescriptor(TensorRole.AttentionValueBias, TensorScope.PerLayer, "blk.{layer}.attn_v.bias", Required: false, []),
            new TensorDescriptor(TensorRole.AttentionOutputBias, TensorScope.PerLayer, "blk.{layer}.attn_output.bias", Required: false, []),
            new TensorDescriptor(TensorRole.OutputNorm, TensorScope.Global, "output_norm.weight", Required: true, []),
            new TensorDescriptor(TensorRole.OutputHead, TensorScope.Global, "output.weight", Required: false, []),
            new TensorDescriptor(TensorRole.OutputBias, TensorScope.Global, "output.bias", Required: false, []),
            new TensorDescriptor(TensorRole.AttentionQueryNorm, TensorScope.PerLayer, "blk.{layer}.attn_q_norm.weight", Required: false, []),
            new TensorDescriptor(TensorRole.AttentionKeyNorm, TensorScope.PerLayer, "blk.{layer}.attn_k_norm.weight", Required: false, []),
            new TensorDescriptor(TensorRole.AttentionNormBias, TensorScope.PerLayer, "blk.{layer}.attn_norm.bias", Required: false, []),
            new TensorDescriptor(TensorRole.FeedForwardNormBias, TensorScope.PerLayer, "blk.{layer}.ffn_norm.bias", Required: false, []),
            new TensorDescriptor(TensorRole.OutputNormBias, TensorScope.Global, "output_norm.bias", Required: false, []),
        ]);
}
