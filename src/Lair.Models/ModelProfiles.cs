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
            NormPlacement = NormPlacement.PreOnly,
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
            NormPlacement = NormPlacement.PreOnly,
            Tensors = CanonicalTensors(),
        },
        new ModelProfile
        {
            Architecture = "qwen3",
            Prefix = "qwen3.",
            Norm = NormStrategy.RmsNorm,
            Activation = ActivationStrategy.SwiGLU,
            Bias = BiasPolicy.Declared,
            // The qwen3 block carries its rotary q/k rows in rotate-half order, so the engine pairs each head's
            // halves (i with i + headDim/2) instead of adjacent dimensions; the llama-family profiles keep the
            // adjacent pairing their checkpoints lay their rows in.
            Rope = RopeStrategy.RotateHalfFullHead,
            // The qwen3 block always carries the per-head q/k norm vectors, so the feature defaults to the weighted
            // mode when the file declares no attention.use_qk_norm; a declared false overrides it either way. Attention
            // biases stay declared-only: qwen3 conversion ships them only in some variants and the file must say so.
            DefaultQkNorm = QkNormMode.PerHeadWeighted,
            NormPlacement = NormPlacement.PreOnly,
            Tensors = CanonicalTensors(),
        },
        new ModelProfile
        {
            Architecture = "gemma2",
            Prefix = "gemma2.",
            // The reference gemma2 graph runs every block norm — pre-sublayer, post-sublayer, and output — as an
            // RMS norm with the file's layer_norm_rms_epsilon, and normalizes each q and k head nowhere: the
            // conversion ships no per-head norm vectors and the graph applies no unweighted head normalization, so
            // QK-norm stays off unless the file declares it.
            Norm = NormStrategy.RmsNorm,
            // The reference FFN gates with the exact GELU, not the tanh approximation.
            Activation = ActivationStrategy.GeGluExact,
            Bias = BiasPolicy.Declared,
            // The gemma2 block carries its rotary q/k rows in rotate-half order, the layout the qwen3 checkpoints
            // use, so the engine pairs each head's halves rather than adjacent dimensions.
            Rope = RopeStrategy.RotateHalfFullHead,
            // The gemma2 conversion does ship q/k/v bias vectors, but the reference's forward leaves the measured
            // logits inside its own quantized band when they are not applied, so bias stays declared-only like the
            // other profiles: a declared use_bias=true still makes every layer's q/k/v vectors required at load.
            DefaultAttentionBias = false,
            DefaultQkNorm = QkNormMode.Off,
            // The gemma2 block normalizes each sublayer output with its own vector before the residual add, which
            // makes every layer's post-sublayer norm tensors required at load.
            NormPlacement = NormPlacement.PostSublayerOutputs,
            // The reference scales the token-embedding row by sqrt(hidden size) before the first layer; the
            // conversion leaves that gain out of the weights, so the engine applies it.
            ScaleInputEmbeddingsBySqrtHidden = true,
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

    // The canonical llama-family tensor names: the llama.cpp GGUF layout every supported profile ships, with
    // attention biases and the output head optional because llama files carry neither while qwen2 and qwen3
    // files may. The per-head QK-norm vectors are optional in the map: the resolved mode decides at load whether they
    // are required or ignored, and a file whose profile defaults them on but lacks the vectors fails naming them. The
    // post-sublayer norm vectors are declared here too because every family names them the same way; the gemma2
    // profile consumes them under its post-sublayer placement, while every profile that resolves NormPlacement.PreOnly
    // treats their presence as a layout failure rather than ignorable material. The norm-bias vectors are optional for
    // the same reason the QK-norm ones are: the declared flag decides.
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
            new TensorDescriptor(TensorRole.PostAttentionNorm, TensorScope.PerLayer, "blk.{layer}.post_attention_norm.weight", Required: false, []),
            new TensorDescriptor(TensorRole.PostFeedForwardNorm, TensorScope.PerLayer, "blk.{layer}.post_ffw_norm.weight", Required: false, []),
        ]);
}
