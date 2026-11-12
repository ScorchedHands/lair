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

    // The sparse-FFN shape each MoE profile declares: the router and expert roles its tensor map names, and
    // whether its reference block adds a shared expert. The counts the plan carries come from the file's
    // declared expert metadata, never from the profile, so these shapes carry zero counts. They name the factorized
    // layout's basis/coefficient/output-basis roles too — the map declares them as optional tensors a sidecar-produced
    // file can reach — while carrying the stacked layout and no rank, because which layout a block has is what its own
    // declaration says, not what the family prefers.
    private static readonly SparseFfnDescriptor Qwen2MoeSparseFfn = new(
        ExpertCount: 0,
        RoutedExpertsPerToken: 0,
        HasSharedExpert: true,
        RouterRole: TensorRole.MoeRouter,
        ExpertGateRole: TensorRole.MoeExpertGate,
        ExpertUpRole: TensorRole.MoeExpertUp,
        ExpertDownRole: TensorRole.MoeExpertDown,
        SharedGateRole: TensorRole.MoeSharedExpertGate,
        SharedUpRole: TensorRole.MoeSharedExpertUp,
        SharedDownRole: TensorRole.MoeSharedExpertDown,
        // The reference qwen2moe block ships one selector vector of the file's hidden size per layer and multiplies
        // this token's shared-expert output by its score, so this shape names that role. A sparse shape naming no
        // gate role adds its shared block ungated instead.
        SharedInputGateRole: TensorRole.MoeSharedExpertInputGate,
        Layout: SparseFfnLayout.StackedExperts,
        Rank: null,
        ExpertInputBasisRole: TensorRole.MoeExpertInputBasis,
        ExpertCoefficientRole: TensorRole.MoeExpertCoefficients,
        ExpertOutputBasisRole: TensorRole.MoeExpertOutputBasis,
        ExpertWeightsScale: 0f,
        NormalizeExpertWeights: false);

    private static readonly SparseFfnDescriptor Qwen3MoeSparseFfn = new(
        ExpertCount: 0,
        RoutedExpertsPerToken: 0,
        HasSharedExpert: false,
        RouterRole: TensorRole.MoeRouter,
        ExpertGateRole: TensorRole.MoeExpertGate,
        ExpertUpRole: TensorRole.MoeExpertUp,
        ExpertDownRole: TensorRole.MoeExpertDown,
        SharedGateRole: null,
        SharedUpRole: null,
        SharedDownRole: null,
        // The reference qwen3moe block ships no shared expert at all, so it names neither its matrices nor a gate.
        SharedInputGateRole: null,
        Layout: SparseFfnLayout.StackedExperts,
        Rank: null,
        ExpertInputBasisRole: TensorRole.MoeExpertInputBasis,
        ExpertCoefficientRole: TensorRole.MoeExpertCoefficients,
        ExpertOutputBasisRole: TensorRole.MoeExpertOutputBasis,
        ExpertWeightsScale: 0f,
        NormalizeExpertWeights: false);

    // The reference olmoe block routes every token through a per-layer router matrix over its experts and adds no
    // shared expert at all, so this shape names the router and the stacked gate/up/down triplets and none of the
    // shared-expert roles. Its counts come from the file's declared metadata like every other sparse shape, so they
    // stay zero here; the reference reads no routing-weight gain or renormalization for this family either — a real
    // olmoe conversion declares neither key, which resolves to scale 0f (disabled) and normalization off.
    private static readonly SparseFfnDescriptor OlmoeSparseFfn = new(
        ExpertCount: 0,
        RoutedExpertsPerToken: 0,
        HasSharedExpert: false,
        RouterRole: TensorRole.MoeRouter,
        ExpertGateRole: TensorRole.MoeExpertGate,
        ExpertUpRole: TensorRole.MoeExpertUp,
        ExpertDownRole: TensorRole.MoeExpertDown,
        SharedGateRole: null,
        SharedUpRole: null,
        SharedDownRole: null,
        // The reference olmoe block ships no shared expert at all, so it names neither its matrices nor a gate.
        SharedInputGateRole: null,
        Layout: SparseFfnLayout.StackedExperts,
        Rank: null,
        ExpertInputBasisRole: TensorRole.MoeExpertInputBasis,
        ExpertCoefficientRole: TensorRole.MoeExpertCoefficients,
        ExpertOutputBasisRole: TensorRole.MoeExpertOutputBasis,
        ExpertWeightsScale: 0f,
        NormalizeExpertWeights: false);

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
            // The reference applies its sliding window to alternating blocks — local attention on the even-indexed
            // layers, full attention on the odd-indexed ones — so the declared window masks only the even side here.
            WindowInterleave = WindowInterleave.AlternatingLocalFirst,
            Tensors = CanonicalTensors(),
        },
        new ModelProfile
        {
            Architecture = "gemma3",
            Prefix = "gemma3.",
            // The reference gemma3 graph runs every block norm — pre-sublayer, post-sublayer, and output — as an RMS
            // norm with the file's layer_norm_rms_epsilon, exactly like its predecessor. It also normalizes each q and
            // k head inside that sublayer: the conversion ships per-head norm vectors for every layer, so QK-norm
            // defaults to the weighted mode here rather than staying off until declared.
            Norm = NormStrategy.RmsNorm,
            // The reference FFN gates with the exact GELU, not the tanh approximation, as its predecessor's does.
            Activation = ActivationStrategy.GeGluExact,
            Bias = BiasPolicy.Declared,
            // The gemma3 block carries its rotary q/k rows in rotate-half order like the gemma2 and qwen3 blocks do,
            // so the engine pairs each head's halves rather than adjacent dimensions.
            Rope = RopeStrategy.RotateHalfFullHead,
            // This family ships no attention-bias vectors for these checkpoints, and nothing in its reference forward
            // adds one where they are absent: bias stays declared-only like every other profile, so a declared
            // use_bias=true still makes each layer's q/k/v vectors required at load.
            DefaultAttentionBias = false,
            DefaultQkNorm = QkNormMode.PerHeadWeighted,
            // The gemma3 block normalizes each sublayer output with its own vector before the residual add like
            // gemma2 does, which makes every layer's post-sublayer norm tensors required at load.
            NormPlacement = NormPlacement.PostSublayerOutputs,
            // The reference scales the token-embedding row by sqrt(hidden size) before the first layer; the conversion
            // leaves that gain out of the weights, so the engine applies it.
            ScaleInputEmbeddingsBySqrtHidden = true,
            // The reference declares this family's window stride itself and reads no per-layer declaration for it: local
            // attention on four layers in every group of six, full attention on the sixth — so the declared window masks
            // all but the last layer of each group here. Its logit softcaps stay metadata-only like every other profile:
            // a file that declares neither cap runs uncapped whatever this entry says about them.
            WindowInterleave = WindowInterleave.SixLayerGroupsLocalFirst,
            Tensors = CanonicalTensors(),
        },
        new ModelProfile
        {
            Architecture = "phi3",
            Prefix = "phi3.",
            // The reference phi3 block runs its pre-sublayer and output norms as RMS norms with the file's
            // layer_norm_rms_epsilon, gates its feed-forward with SiLU over a gate/up pair, and normalizes neither q nor
            // k per head: the conversion ships no per-head norm vectors for this family. Its attention carries no bias
            // vector either — the reference block projects each side from weights alone — so bias stays declared-only
            // like every other profile rather than defaulting on.
            Norm = NormStrategy.RmsNorm,
            Activation = ActivationStrategy.SwiGLU,
            Bias = BiasPolicy.Declared,
            DefaultQkNorm = QkNormMode.Off,
            // The reference rotary covers each head in full at the base a file declaring no rope frequency is read at:
            // this conversion ships no rope scaling keys and no rope-factor tensors for the 4k variant, so the rotation
            // stays unscaled over the whole head dimension whatever the declared slice says. Its rotary rows sit in
            // rotate-half order — each head's two halves of every pair, not adjacent dimensions — so the engine pairs the
            // head halves the way the qwen3 and gemma blocks' rows do.
            Rope = RopeStrategy.RotateHalfFullHead,
            // The reference block projects q, k, and v through one fused matrix per layer — its attention reads the query
            // region of that buffer, then the key region, then the value region — so this profile resolves the fused
            // layout and names that tensor in its map instead of the three split roles. Its head regions are contiguous:
            // every head of q first (head_count × head_dim rows), then every head of k (kv_head_count × head_dim rows),
            // then value again, which is how the conversion writes one linear projection's output columns in order.
            AttentionWeights = QkvLayout.FusedBufferSlices,
            // The reference block ships no gate matrix at all: its feed-forward reads one fused gate/up projection whose
            // first intermediate_size rows are the gate side and next intermediate_size rows are the up side, then gates
            // with SiLU. This profile resolves that shape so a file carrying only blk.N.ffn_up.weight loads, and a file
            // shipping a separate gate matrix is reported as material this block does not read.
            GateSource = FeedForwardGateSource.UpTensorHalves,
            NormPlacement = NormPlacement.PreOnly,
            Tensors = PhiTensors(),
        },
        new ModelProfile
        {
            Architecture = "qwen2moe",
            Prefix = "qwen2moe.",
            // The reference qwen2moe block routes every token through a per-layer router matrix over its experts,
            // sums the routed expert outputs with the routing weights, and adds one shared-expert block to that —
            // the profile names those tensors and declares the sparse-FFN shape. The shared expert is selected per
            // token by a separate hidden-sized vector of the same layer, named by its own role rather than folded
            // into the shared gate matrix's name.
            SparseFfn = Qwen2MoeSparseFfn,
            Norm = NormStrategy.RmsNorm,
            Activation = ActivationStrategy.SwiGLU,
            Bias = BiasPolicy.Declared,
            Rope = RopeStrategy.StandardFullHead,
            NormPlacement = NormPlacement.PreOnly,
            Tensors = MoeTensors(),
        },
        new ModelProfile
        {
            Architecture = "qwen3moe",
            Prefix = "qwen3moe.",
            // The reference qwen3moe block carries the same router-and-experts layout as qwen2moe without a
            // shared expert, and always ships the per-head q/k norm vectors like the dense qwen3 profile does:
            // the feature defaults to the weighted mode when the file declares no attention.use_qk_norm.
            SparseFfn = Qwen3MoeSparseFfn,
            Norm = NormStrategy.RmsNorm,
            Activation = ActivationStrategy.SwiGLU,
            Bias = BiasPolicy.Declared,
            Rope = RopeStrategy.RotateHalfFullHead,
            DefaultQkNorm = QkNormMode.PerHeadWeighted,
            NormPlacement = NormPlacement.PreOnly,
            Tensors = MoeTensors(),
        },
        new ModelProfile
        {
            Architecture = "olmoe",
            Prefix = "olmoe.",
            // The reference olmoe block routes every token through a per-layer router over its experts with no shared
            // expert, and normalizes each layer's projected q and k the way its graph does: one RMS norm over the whole
            // projection — before the heads are split — scaled element by element by that layer's hidden-sized vector.
            // The file ships those vectors whether or not it declares the feature, so the profile defaults the mode to
            // the stream-weighted form; a declared attention.use_qk_norm=false still turns it off.
            SparseFfn = OlmoeSparseFfn,
            Norm = NormStrategy.RmsNorm,
            Activation = ActivationStrategy.SwiGLU,
            Bias = BiasPolicy.Declared,
            // The reference reads no rope pairing declaration for this family and its graph asserts the rotation covers
            // each head in full, which is the adjacent-pair layout it applies to a file that declares none.
            Rope = RopeStrategy.StandardFullHead,
            DefaultQkNorm = QkNormMode.StreamWeighted,
            NormPlacement = NormPlacement.PreOnly,
            Tensors = MoeTensors(),
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
    private static TensorMap CanonicalTensors() => new(CanonicalDescriptors());

    private static TensorDescriptor[] CanonicalDescriptors() =>
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
        ];

    // The phi3 tensor names: the canonical map with this family's two fused roles swapped in. q/k/v arrive as one buffer
    // per layer, so its name is required and the three split roles stay declared-but-optional — a file shipping them for
    // this architecture is material the resolved layout does not read rather than an unnameable table, and the engine
    // reports it instead of loading it. The gate matrix role relaxes the same way: this family's conversion ships no
    // blk.N.ffn_gate.weight at all, its gate half living inside ffn_up.weight, so a file that does carry one is reported
    // as unread rather than rejected for naming a tensor this block never consumes.
    private static TensorMap PhiTensors() =>
        new(
        [
            .. CanonicalDescriptors().Select(descriptor =>
                descriptor.Role is TensorRole.AttentionQuery or TensorRole.AttentionKey or TensorRole.AttentionValue or TensorRole.FeedForwardGate
                    ? descriptor with { Required = false }
                    : descriptor),
            new TensorDescriptor(TensorRole.AttentionQkv, TensorScope.PerLayer, "blk.{layer}.attn_qkv.weight", Required: true, []),
        ]);

    // The MoE tensor map: the canonical map with the dense gate/up/down roles relaxed to optional — an
    // expert-routed block ships its gate/up/down triplets per expert instead of a dense triplet, so a MoE
    // file carries none of the dense names — plus the sparse-FFN roles under the names the reference uses:
    // the router matrix, the stacked per-expert gate/up/down triplets, the shared expert's gate/up/down
    // triplets, and the hidden-sized vector that selects the shared expert for a token. Every sparse role is
    // optional: the map names a MoE file's table so the load can report the file instead of rejecting its
    // material, and only the families whose reference ships each tensor name it — the shared selector vector is a
    // separate tensor from the shared gate matrix, with a shape of its own, so it carries no alias onto that name.
    // Dense profiles carry no sparse roles at all: their forward consumes none, and the generic llama-shaped fallback
    // treats a tensor under a sparse name as material its map cannot match.
    private static TensorMap MoeTensors() =>
        new(
        [
            .. CanonicalDescriptors().Select(descriptor =>
                descriptor.Role is TensorRole.FeedForwardGate or TensorRole.FeedForwardUp or TensorRole.FeedForwardDown
                    ? descriptor with { Required = false }
                    : descriptor),
            new TensorDescriptor(TensorRole.MoeRouter, TensorScope.PerLayer, "blk.{layer}.ffn_gate_inp.weight", Required: false, []),
            new TensorDescriptor(TensorRole.MoeExpertGate, TensorScope.PerLayer, "blk.{layer}.ffn_gate_exps.weight", Required: false, []),
            new TensorDescriptor(TensorRole.MoeExpertUp, TensorScope.PerLayer, "blk.{layer}.ffn_up_exps.weight", Required: false, []),
            new TensorDescriptor(TensorRole.MoeExpertDown, TensorScope.PerLayer, "blk.{layer}.ffn_down_exps.weight", Required: false, []),
            new TensorDescriptor(TensorRole.MoeSharedExpertGate, TensorScope.PerLayer, "blk.{layer}.ffn_gate_shexp.weight", Required: false, []),
            new TensorDescriptor(TensorRole.MoeSharedExpertUp, TensorScope.PerLayer, "blk.{layer}.ffn_up_shexp.weight", Required: false, []),
            new TensorDescriptor(TensorRole.MoeSharedExpertDown, TensorScope.PerLayer, "blk.{layer}.ffn_down_shexp.weight", Required: false, []),
            new TensorDescriptor(TensorRole.MoeSharedExpertInputGate, TensorScope.PerLayer, "blk.{layer}.ffn_gate_inp_shexp.weight", Required: false, []),

            // The factorized layout's tensors. No published conversion ships them — LAIR converts nothing itself, so a
            // file carrying these arrives through an alternate weight provider — and the names follow this family's own
            // GGUF conventions rather than inventing a vocabulary: block-scoped `blk.N.` prefix, `.weight` suffix, and
            // the same `ffn_…exps…` stem the dense-stacked expert roles use. The shared basis is one matrix per layer on
            // each side of the compressed space (`basis_in` reads hidden values, `basis_out` writes them), the per-expert
            // coefficients keep the plural `_coefficients` shape the stacked expert triplets use so their outermost
            // dimension indexes experts exactly as `ffn_gate_exps` does, and each role carries the shorter singular form
            // of its own name as an alias because a sidecar exporter is free to spell it either way. All stay optional:
            // only a file declaring the factorized layout declares these tensors, and nothing resident today reads them.
            new TensorDescriptor(
                TensorRole.MoeExpertInputBasis,
                TensorScope.PerLayer,
                "blk.{layer}.ffn_exps_basis_in.weight",
                Required: false,
                ["blk.{layer}.ffn_experts_basis_in.weight"]),
            new TensorDescriptor(
                TensorRole.MoeExpertCoefficients,
                TensorScope.PerLayer,
                "blk.{layer}.ffn_exps_coefficient.weight",
                Required: false,
                ["blk.{layer}.ffn_experts_coefficient.weight"]),
            new TensorDescriptor(
                TensorRole.MoeExpertOutputBasis,
                TensorScope.PerLayer,
                "blk.{layer}.ffn_exps_basis_out.weight",
                Required: false,
                ["blk.{layer}.ffn_experts_basis_out.weight"]),
        ]);
}
