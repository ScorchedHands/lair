using System.Globalization;

using Lair.Contracts;
using Lair.Loader;

namespace Lair.Models;

/// <summary>
/// Resolves a GGUF file into a <see cref="ModelPlan"/>: the architecture is looked up in
/// <see cref="ModelProfiles"/>, the geometry is read from the profile's metadata prefix, the vocabulary size
/// comes from the declared key or the tokenizer vocabulary length, the attention score window comes from the
/// declared <c>&lt;prefix&gt;attention.sliding_window</c> (unbounded when the key is absent), and the tensor map
/// comes from the profile together with the window interleave that decides which layers that window masks. The rotary
/// base those masked layers rotate at reads <c>&lt;prefix&gt;rope.freq_base_swa</c>, resolving to the same base an
/// undeclared global frequency takes when the key is absent. The rope scaling resolves the declared
/// <c>&lt;prefix&gt;rope.scaling.*</c> metadata into a scaling plan: an absent
/// variant declaration keeps the standard unscaled rotation, the declared <c>yarn</c> variant resolves the
/// multiplier, original context length, attention factor, correction scale, and beta pair, and any other variant
/// declaration fails at load naming the key and the value. The rotary slice resolves the declared
/// <c>&lt;prefix&gt;rope.dimension_count</c>: an absent key rotates the full head dimension, a declared even count
/// not exceeding it narrows the rotation to that leading slice, and any other declaration fails at load naming
/// the key and the value. The per-head QK-norm mode resolves the declared
/// <c>&lt;prefix&gt;attention.use_qk_norm</c> against the profile's default, the attention-bias flag resolves the
/// declared <c>&lt;prefix&gt;attention.use_bias</c> against the profile's default, and the norm placement comes from
/// the profile alone — no metadata key selects it. The block-norm epsilon read from
/// <c>&lt;prefix&gt;attention.layer_norm_epsilon</c> (zero when the key is absent) feeds the LayerNorm block norms;
    /// the RMS epsilon from <c>&lt;prefix&gt;attention.layer_norm_rms_epsilon</c> feeds the RMS norms and the per-head
    /// QK-norm. The input-embedding gain comes from the profile alone: a profile that scales embeddings resolves
    /// <c>sqrt(hidden size)</c> in the plan, every other profile resolves zero. The sparse-FFN slot resolves the
    /// declared <c>&lt;prefix&gt;expert_count</c> together with its routing, shared-expert, weight-scale, and
    /// weight-normalization keys against the resolved profile's declared shape; a file that declares no expert
    /// count keeps the dense gate/up/down block; its declared <c>&lt;prefix&gt;expert_layout</c> (<c>stacked</c> when the key is
    /// absent) and <c>&lt;prefix&gt;expert_rank</c> decide how factorized that storage is, and only a declaration of the
    /// shared-basis layout carries the basis/coefficient/output-basis tensor roles or names a rank. The sparse-FFN strategy
    /// resolves the declared <c>&lt;prefix&gt;sparse_ffn_strategy</c> against the resolved profile's default exactly as the
    /// QK-norm mode does, and lands in the plan only for a block that declares experts: an unrecognized strategy name fails
    /// at load naming the key and the value, and selecting <see cref="SparseFfnStrategy.LowRankSharedBasis"/> on a file whose
    /// declaration stores its experts as stacked matrices is refused there rather than downgraded. An unregistered architecture is accepted only through the generic llama-shaped fallback profile, which
/// requires the file's tensor table to match the llama map exactly and reports the substitution as one structured
/// log line when applied.
/// </summary>
public static class ModelPlans
{
    /// <summary>
    /// Parses the architecture metadata of a GGUF file into its resolved plan. The architecture must name a
    /// registered profile or pass the generic llama-shaped fallback check; an empty or whitespace-only name is
    /// rejected outright and is never a fallback candidate. Declared head dimensions must agree with each
    /// other, and the derived one must split the hidden size exactly.
    /// </summary>
    /// <param name="gguf">The GGUF to resolve.</param>
    /// <param name="logLine">Sink for the structured fallback-applied line; hosts may capture instead.</param>
    public static ModelPlan Resolve(GgufFile gguf, Action<string>? logLine = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);

        GgufMetadata metadata = gguf.Metadata;

        if (!metadata.Has("general.architecture"))
        {
            throw new ModelException(
                "GGUF metadata key 'general.architecture' is not present (supported architectures: " +
                $"{string.Join(", ", ModelProfiles.SupportedArchitectures)})");
        }

        string architecture = metadata.Get("general.architecture").AsString();

        // An empty or whitespace-only architecture name is a load failure in its own right. It must not reach
        // the generic fallback, where it would re-prefix the llama-shaped profile to a degenerate '.' and read
        // the file's geometry under that prefix.
        if (string.IsNullOrWhiteSpace(architecture))
        {
            throw new ModelException("GGUF metadata key 'general.architecture' is present but empty");
        }

        ModelProfile profile = ModelProfiles.ResolveOrFallback(architecture);
        bool fallback = profile.Architecture != architecture;

        // The fallback is only ever a disguise for a llama-shaped file, so its tensor-table match is verified
        // before any metadata read: a table the map cannot match is rejected with the registry wording, which
        // also carries the supported-profile list.
        if (fallback)
        {
            ValidateLlamaShapedTable(profile.Tensors, architecture, gguf.Tensors);
        }

        return ResolveWithProfile(gguf, profile, architecture, fallback, logLine);
    }

    /// <summary>
    /// Resolves a GGUF file against an explicitly supplied profile instead of the registry selection: the
    /// profile's prefix supplies every metadata key and its strategy values land in the plan. No fallback gate
    /// or architecture-name check runs; the declared <c>general.architecture</c> value is recorded as-is.
    /// </summary>
    internal static ModelPlan Resolve(GgufFile gguf, ModelProfile profile, Action<string>? logLine = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(profile);

        if (!gguf.Metadata.Has("general.architecture"))
        {
            throw new ModelException("GGUF metadata key 'general.architecture' is not present");
        }

        return ResolveWithProfile(gguf, profile, gguf.Metadata.Get("general.architecture").AsString(), fallback: false, logLine);
    }

    // Shared resolution body: geometry under the profile's prefix, the standard-RoPE-only gates, and the plan's
    // strategy values straight from the profile.
    private static ModelPlan ResolveWithProfile(GgufFile gguf, ModelProfile profile, string architecture, bool fallback, Action<string>? logLine)
    {
        GgufMetadata metadata = gguf.Metadata;
        string prefix = profile.Prefix;

        // Declared rope scaling: an absent <prefix>rope.scaling.type keeps the standard rotation, the declared
        // "yarn" variant resolves the scaling plan, and any other variant declaration — or any scaling operand
        // declared without a variant name, which is the reference's linear default — fails the load naming the
        // key and the declared value.
        RopeScalingPlan? ropeScaling = ResolveRopeScaling(metadata, prefix);

        int hidden = RequireInt(metadata, prefix + "embedding_length");
        int layers = RequireInt(metadata, prefix + "block_count");
        int heads = RequireInt(metadata, prefix + "attention.head_count");
        int kvHeads = RequireInt(metadata, prefix + "attention.head_count_kv");
        int ffn = ResolveFeedForwardLength(metadata, prefix, gguf, logLine);

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layers);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hidden);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(heads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ffn);

        // The per-head dimension is declared explicitly by head_length (the name the gemma family uses) or by
        // key_length / value_length; every declared source must agree with the others, and a file declaring one
        // of key_length / value_length must declare both. With no declaration the dimension is derived from
        // hidden / head_count, which must divide exactly. The declared dimension no longer has to split the
        // hidden size: head_count x head_dim may differ from embedding_length.
        int headDim;
        bool hasHeadLength = metadata.Has(prefix + "attention.head_length");
        bool hasKeyLength = metadata.Has(prefix + "attention.key_length");
        bool hasValueLength = metadata.Has(prefix + "attention.value_length");

        if (hasHeadLength || hasKeyLength || hasValueLength)
        {
            headDim = hasHeadLength
                ? RequireInt(metadata, prefix + "attention.head_length")
                : RequireInt(metadata, prefix + "attention.key_length");
            string headSource = hasHeadLength ? "head_length" : "key_length";

            if (hasKeyLength)
            {
                int keyLength = RequireInt(metadata, prefix + "attention.key_length");
                if (keyLength != headDim)
                {
                    throw new ModelException(
                        $"{prefix}attention.{headSource} ({headDim}) and {prefix}attention.key_length ({keyLength}) disagree; " +
                        "head dimensions must be equal for this architecture family");
                }
            }

            if (hasValueLength)
            {
                int valueLength = RequireInt(metadata, prefix + "attention.value_length");
                if (valueLength != headDim)
                {
                    throw new ModelException(
                        $"{prefix}attention.{headSource} ({headDim}) and {prefix}attention.value_length ({valueLength}) disagree; " +
                        "head dimensions must be equal for this architecture family");
                }
            }
        }
        else if (hidden % heads != 0)
        {
            throw new ModelException(
                $"{prefix}attention.head_count ({heads}) does not divide {prefix}embedding_length ({hidden}); " +
                $"the architecture needs an explicit {prefix}attention.key_length to define the head dimension");
        }
        else
        {
            headDim = hidden / heads;
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);

        // The declared rotary slice: an absent key means the full head dimension. A declared slice must be an
        // even count of head dimensions not exceeding the head dimension; the rotation then covers the leading
        // slice and leaves the remaining dimensions unchanged.
        int ropeRotaryDimensions = ResolveRotarySlice(metadata, prefix, headDim);

        float ropeTheta = metadata.Has(prefix + "rope.freq_base")
            ? (float)metadata.Get(prefix + "rope.freq_base").AsDouble()
            : ModelPlan.DefaultRopeTheta;

        // The declared rotary base for the layers this profile's interleave narrows to its window: an absent key means
        // the same base a file declaring no rotary frequency at all is read at. A declared value must land in the float
        // range as a positive finite number — a non-positive, NaN, infinite, or out-of-range declaration fails naming
        // the key and the declared value. Which layers it reaches is the plan's per-layer window resolution, not this call.
        float ropeWindowedTheta = ResolveRopeBase(metadata, prefix + "rope.freq_base_swa");

        if (!metadata.Has(prefix + "attention.layer_norm_rms_epsilon"))
        {
            throw new ModelException($"GGUF metadata key '{prefix}attention.layer_norm_rms_epsilon' is not present");
        }

        float eps = (float)metadata.Get(prefix + "attention.layer_norm_rms_epsilon").AsDouble();

        // Block-norm epsilon: the LayerNorm block norms run with this value, zero unless the file declares
        // <prefix>attention.layer_norm_epsilon. The RMS epsilon above stays the operand for the RMS norms and for
        // the per-head QK-norm whatever this resolves to.
        float epsLayerNorm = ResolveNormEpsilon(metadata, prefix + "attention.layer_norm_epsilon");

        // Declared feature flags: an absent key means the profile's feature default for it. Norm-bias defaults false
        // for every profile today; attention bias defaults false except where a profile defaults it on, and a
        // declaration overrides either direction. A declared use_qk_norm=false forces the per-head mode off whatever
        // the profile defaults to, while an absent declaration keeps that mode. The declaration or profile default,
        // not tensor presence, decides the behavior; the forward engine enforces the tensor side of each resolved
        // value at Open — required tensors for a weighted mode, one structured log line naming the ignored tensors for
        // a resolved off whose vectors are present, and no demand at all from an unweighted mode. Norm placement has
        // no declaration: it comes straight from the profile, which also decides whether each layer's post-sublayer
        // norm tensors are required or must be absent.
        bool useAttentionBias = metadata.Has(prefix + "attention.use_bias")
            ? metadata.GetBool(prefix + "attention.use_bias", fallback: false)
            : profile.DefaultAttentionBias;
        bool useNormBias = metadata.GetBool(prefix + "attention.norm_bias", fallback: false);
        QkNormMode qkNorm = ResolveQkNorm(metadata, prefix, profile.DefaultQkNorm);

        // Declared softcaps: an absent key means the cap is disabled (zero). A declared value must land in the
        // float range as a positive finite number; anything else is a load failure naming the key and value.
        float attentionSoftcap = ResolveSoftcap(metadata, prefix + "attn_logit_softcapping");
        float finalLogitSoftcap = ResolveSoftcap(metadata, prefix + "final_logit_softcapping");

        // Declared attention window: an absent key means the attention is unbounded (zero). A declared value
        // must be a positive integer that fits the int range; anything else is a load failure naming the key
        // and the declared value. The resolved value masks scores in every attention path and bounds no KV, and
        // which layers it masks has no declaration: it comes straight from the resolved profile.
        int attentionWindow = ResolveWindow(metadata, prefix + "attention.sliding_window");

        // Vocabulary size: the declared key wins when present. Without it the size is the length of the
        // tokenizer vocabulary; a file carrying neither fails with a message naming both keys. The fallback
        // and a declared/tokenizer disagreement are each reported as one structured line in the style of the
        // profile-default lines.
        int vocab = ResolveVocab(metadata, prefix, gguf, logLine);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocab);

        LogicalTensorTable tensors = gguf.Tensors;
        bool hasOutputWeights = profile.Tensors[TensorRole.OutputHead].Resolve(tensors) is not null;

        if (fallback)
        {
            // The fallback was accepted: one structured line naming the model, the declared architecture,
            // and the profile actually applied, in the same style as the tokenizer profile defaults.
            if (logLine is not null)
            {
                logLine(
                    $"model_profile model='{ModelName(metadata, gguf)}' " +
                    $"architecture='{architecture}' profile={profile.Architecture}");
            }
        }

        // The sparse-FFN slot and its resolved strategy resolve together: the descriptor says how the block stores its
        // experts, and the plan's strategy value is what a file declares on top of the profile's default — refused here,
        // not downgraded, when it selects a variant the declared storage cannot back.
        SparseFfnDescriptor? sparseFfn = ResolveSparseFfn(metadata, prefix, profile.SparseFfn);

        // The qwen4exp execution geometry resolves only under a profile that declares this family's semantics: its per-layer
        // attention interleave from the declared compress-ratio array (checked against <prefix>full_attention_interval when
        // that key is also present), the shared gated-delta-net geometry from the <prefix>ssm.* group, the indexer budget
        // from the <prefix>attention.indexer.* group, and the hyper-connection shape from <prefix>hyper_connection.*. The
        // file's tensor table is then validated against that resolved interleave — every linear layer must carry its fused
        // qkv buffer, gate matrix and recurrent-state tensors where a full layer instead carries split projections and an
        // indexer set — so a plan for this family never resolves past a table whose per-layer material contradicts what the
        // declared groups say those layers run. A profile that declares no such semantics resolves null and validates nothing:
        // every other family keeps today's posture, where tensor-table enforcement lands at engine open instead.
        Qwen4ExpGeometry? qwen4exp = profile.DeclaresQwen4ExpSemantics
            ? ResolveQwen4Exp(metadata, prefix, layers)
            : null;

        if (qwen4exp is not null)
        {
            ValidateQwen4ExpTable(profile.Tensors, architecture, gguf.Tensors, qwen4exp, hidden);
        }

        return new ModelPlan
        {
            Architecture = architecture,
            Profile = profile.Architecture,
            Prefix = prefix,
            LayerCount = layers,
            HiddenSize = hidden,
            HeadCount = heads,
            KvHeadCount = kvHeads,
            HeadDim = headDim,
            IntermediateSize = ffn,
            RopeTheta = ropeTheta,
            RopeWindowedTheta = ropeWindowedTheta,
            EpsRmsNorm = eps,
            EpsLayerNorm = epsLayerNorm,
            VocabSize = vocab,
            HasOutputWeights = hasOutputWeights,
            UseAttentionBias = useAttentionBias,
            UseNormBias = useNormBias,
            QkNorm = qkNorm,
            NormPlacement = profile.NormPlacement,
            AttentionSoftcap = attentionSoftcap,
            FinalLogitSoftcap = finalLogitSoftcap,
            AttentionWindow = attentionWindow,
            WindowInterleave = profile.WindowInterleave,
            InputEmbeddingScale = profile.ScaleInputEmbeddingsBySqrtHidden ? MathF.Sqrt(hidden) : 0f,
            Tensors = profile.Tensors,
            Norm = profile.Norm,
            Activation = profile.Activation,
            Bias = profile.Bias,
            Rope = profile.Rope,
            // The q/k/v projection layout and the feed-forward gate side come straight from the resolved profile like
            // the norm placement and the window interleave do: no metadata key selects either, and each decides which
            // tensors this block's forward reads. For a family that mixes both layouts across its layers, the per-layer
            // kind in <see cref="ModelPlan.Qwen4Exp"/> is what says which side each layer takes; this profile-level value
            // stays at the layout every other family runs on.
            AttentionWeights = profile.AttentionWeights,
            GateSource = profile.GateSource,
            RopeScaling = ropeScaling,
            RopeRotaryDimensions = ropeRotaryDimensions,
            SparseFfn = sparseFfn,
            SparseFfnStrategy = ResolveSparseFfnStrategy(metadata, prefix, profile.DefaultSparseFfnStrategy, sparseFfn),

            // The family geometry resolved above lands unchanged: null for every file whose profile declares no such semantics.
            Qwen4Exp = qwen4exp,
            // The execution path a file declares nothing about runs on: one CPU segment over DRAM-resident weights. A host
            // that knows its own placement resolves the segments it declares onto this value when it opens the engine; no
            // metadata key selects any part of a route here.
            Route = ModelRoute.CpuDefault(),
        };
    }

    // The generic fallback accepts a tensor table only when it matches the llama-shaped map exactly: every
    // required descriptor resolves, and every tensor in the table is named by some descriptor's primary or
    // alias name (the optional ones included). A tensor no descriptor can name is unexpected; the rejection
    // carries the registry wording with its supported-profile list, names both sides of the mismatch, and
    // states that the llama-shaped fallback was attempted. The block extent is derived from the table itself,
    // so this gate needs no metadata: it runs before any metadata read. Family-specific roles — sparse-FFN, gated-delta-net,
    // indexer, hyper-connection and PLE names — stay out of the nameable set even though some maps declare them: a dense-shaped
    // fallback cannot consume an expert-routed or linear-attention block, so a tensor under one of those names is unexpected
    // material and the file is rejected with the names it carries rather than accepted as a dense model with extra tensors.
    private static void ValidateLlamaShapedTable(TensorMap map, string architecture, LogicalTensorTable tensors)
    {
        int layers = 1;
        while (map[TensorRole.AttentionQuery].Resolve(tensors, layers) is not null)
        {
            layers++;
        }

        HashSet<string> nameable = new(StringComparer.Ordinal);
        List<string> missing = [];

        foreach (TensorDescriptor descriptor in map.Descriptors)
        {
            if (TensorMap.IsFamilySpecificRole(descriptor.Role))
            {
                continue;
            }

            bool perLayer = descriptor.Scope == TensorScope.PerLayer;
            int slots = perLayer ? layers : 1;

            for (int slot = 0; slot < slots; slot++)
            {
                int layer = perLayer ? slot : -1;
                nameable.Add(descriptor.NameFor(layer));
                foreach (string alias in descriptor.Aliases)
                {
                    nameable.Add(perLayer ? AliasName(alias, layer) : alias);
                }

                if (descriptor.Required && descriptor.Resolve(tensors, layer) is null)
                {
                    missing.Add(descriptor.NameFor(layer));
                }
            }
        }

        List<string> unexpected = [];
        foreach (LogicalTensor tensor in tensors.Tensors)
        {
            if (!nameable.Contains(tensor.Name) && !IsCompanionOfNameable(tensor.Name, nameable))
            {
                unexpected.Add(tensor.Name);
            }
        }

        if (missing.Count > 0 || unexpected.Count > 0)
        {
            throw new ModelException(
                $"model architecture '{architecture}' is not supported (supported architectures: " +
                $"{string.Join(", ", ModelProfiles.SupportedArchitectures)}): the llama-shaped fallback profile " +
                $"was attempted but the tensor table does not match it (missing tensors: " +
                $"{string.Join(", ", missing)}; unexpected tensors: {string.Join(", ", unexpected)})");
        }
    }

    private static string AliasName(string alias, int layer) =>
        alias.Replace("{layer}", layer.ToString(CultureInfo.InvariantCulture));

    // A quantization companion entry — a <tensor>.scale block or a <tensor>.input_scale scalar — of a tensor the
    // llama-shaped map can name is known-optional: the forward never reads it, so the fallback gate must not
    // mistake it for a layout mismatch.
    private static bool IsCompanionOfNameable(string name, HashSet<string> nameable)
    {
        foreach (string suffix in new string[] { ".input_scale", ".scale" })
        {
            if (name.EndsWith(suffix, StringComparison.Ordinal) && nameable.Contains(name[..^suffix.Length]))
            {
                return true;
            }
        }

        return false;
    }

    private static int RequireInt(GgufMetadata metadata, string key)
    {
        if (!metadata.Has(key))
        {
            throw new ModelException($"GGUF metadata key '{key}' is not present");
        }

        try
        {
            return checked((int)metadata.Get(key).AsLong());
        }
        catch (GgufException ex)
        {
            throw new ModelException($"GGUF metadata key '{key}' must be an integer", ex);
        }
    }

    // Shared vocabulary resolution: the declared <prefix>vocab_size wins when present; otherwise the size is
    // the length of tokenizer.ggml.tokens; a file carrying neither fails naming both keys. When the fallback
    // is applied, or when both sources are present and disagree (the declared key still wins), one structured
    // line names the model and the token count, in the style of the tokenizer profile-default lines.
    private static int ResolveVocab(GgufMetadata metadata, string prefix, GgufFile gguf, Action<string>? logLine)
    {
        bool declared = metadata.Has(prefix + "vocab_size");
        bool tokensDeclared = metadata.Has("tokenizer.ggml.tokens");

        if (!declared && !tokensDeclared)
        {
            throw new ModelException(
                $"GGUF metadata key '{prefix}vocab_size' is not present and 'tokenizer.ggml.tokens' is not " +
                "present: the vocabulary size must be declared or derivable from the tokenizer vocabulary");
        }

        int tokenCount = -1;
        if (tokensDeclared)
        {
            try
            {
                tokenCount = metadata.Get("tokenizer.ggml.tokens").AsArray().Count;
            }
            catch (GgufException ex)
            {
                throw new ModelException("GGUF metadata key 'tokenizer.ggml.tokens' must be an array", ex);
            }
        }

        if (!declared)
        {
            if (logLine is not null)
            {
                logLine(
                    $"vocab_fallback model='{ModelName(metadata, gguf)}' " +
                    $"source=tokenizer.ggml.tokens count={tokenCount.ToString(CultureInfo.InvariantCulture)}");
            }

            return tokenCount;
        }

        int vocab = RequireInt(metadata, prefix + "vocab_size");

        if (tokenCount >= 0 && tokenCount != vocab && logLine is not null)
        {
            logLine(
                $"vocab_disagree model='{ModelName(metadata, gguf)}' " +
                $"declared={vocab.ToString(CultureInfo.InvariantCulture)} source=tokenizer.ggml.tokens " +
                $"count={tokenCount.ToString(CultureInfo.InvariantCulture)}");
        }

        return vocab;
    }

    // Display name shared by the structured load-time log lines: general.name when declared and non-empty,
    // the file path otherwise.
    private static string ModelName(GgufMetadata metadata, GgufFile gguf)
    {
        string name = metadata.Has("general.name") ? metadata.Get("general.name").AsString() : "";
        return name.Length > 0 ? name : gguf.Path;
    }

    // Shared softcap resolution: an absent key means disabled (zero). A declared key is read as a double and
    // must land in the float range as a positive finite number — a non-positive, NaN, infinite, or out-of-range
    // declaration fails the load with a message naming the key and the declared value.
    private static float ResolveSoftcap(GgufMetadata metadata, string key)
    {
        if (!metadata.Has(key))
        {
            return 0f;
        }

        double declared;
        try
        {
            declared = metadata.Get(key).AsDouble();
        }
        catch (GgufException ex)
        {
            throw new ModelException($"GGUF metadata key '{key}' must be a number", ex);
        }

        float cap = (float)declared;
        if (!float.IsFinite(cap) || cap <= 0f)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares {declared.ToString(CultureInfo.InvariantCulture)}; " +
                "softcap values must be positive finite numbers");
        }

        return cap;
    }

    // Shared rotary-base resolution for the windowed base: an absent key means ModelPlan.DefaultRopeTheta, the value a
    // file declaring no rotary frequency is read at. A declared key is read as a double and must land in the float range
    // as a positive finite number — a non-positive, NaN, infinite, or out-of-range declaration fails the load with a
    // message naming the key and the declared value.
    private static float ResolveRopeBase(GgufMetadata metadata, string key)
    {
        if (!metadata.Has(key))
        {
            return ModelPlan.DefaultRopeTheta;
        }

        double declared;
        try
        {
            declared = metadata.Get(key).AsDouble();
        }
        catch (GgufException ex)
        {
            throw new ModelException($"GGUF metadata key '{key}' must be a number", ex);
        }

        float baseValue = (float)declared;
        if (!float.IsFinite(baseValue) || baseValue <= 0f)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares {declared.ToString(CultureInfo.InvariantCulture)}; " +
                "rotary base values must be positive finite numbers");
        }

        return baseValue;
    }

    // Shared block-norm epsilon resolution: an absent key means zero. A declared key is read as a double and must
    // land in the float range as a non-negative finite number — a negative, NaN, infinite, or out-of-range
    // declaration fails the load with a message naming the key and the declared value.
    private static float ResolveNormEpsilon(GgufMetadata metadata, string key)
    {
        if (!metadata.Has(key))
        {
            return 0f;
        }

        double declared;
        try
        {
            declared = metadata.Get(key).AsDouble();
        }
        catch (GgufException ex)
        {
            throw new ModelException($"GGUF metadata key '{key}' must be a number", ex);
        }

        float eps = (float)declared;
        if (!float.IsFinite(eps) || eps < 0f)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares {declared.ToString(CultureInfo.InvariantCulture)}; " +
                "norm epsilon values must be non-negative finite numbers");
        }

        return eps;
    }

    // Shared QK-norm resolution: an absent declaration keeps the resolved profile's mode. A declared false forces Off
    // whatever that default is, and a declared true requests the norm — which no metadata key can express as
    // unweighted or say which weighted form the architecture runs, so it keeps the profile's own weighted mode when it
    // has one and asks for the per-head weighted form on an architecture whose default leaves the feature off. The
    // tensor side of every mode is enforced by the forward engine at Open.
    private static QkNormMode ResolveQkNorm(GgufMetadata metadata, string prefix, QkNormMode profileDefault)
    {
        if (!metadata.Has(prefix + "attention.use_qk_norm"))
        {
            return profileDefault;
        }

        if (!metadata.GetBool(prefix + "attention.use_qk_norm", fallback: false))
        {
            return QkNormMode.Off;
        }

        return profileDefault is QkNormMode.PerHeadUnweighted or QkNormMode.StreamWeighted
            ? profileDefault
            : QkNormMode.PerHeadWeighted;
    }

    // Shared attention-window resolution: an absent key means unbounded (zero). A declared key must read as an
    // integer that fits the int range and is strictly positive — a fractional, text, zero, negative, or
    // out-of-int-range declaration fails the load with a message naming the key and the declared value.
    private static int ResolveWindow(GgufMetadata metadata, string key)
    {
        if (!metadata.Has(key))
        {
            return 0;
        }

        GgufValue declared = metadata.Get(key);

        long declaredValue;
        try
        {
            declaredValue = declared.AsLong();
        }
        catch (GgufException ex)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares {declared}; attention window values must be positive integers", ex);
        }

        if (declaredValue <= 0 || declaredValue > int.MaxValue)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares {declaredValue.ToString(CultureInfo.InvariantCulture)}; " +
                "attention window values must be positive integers");
        }

        return (int)declaredValue;
    }

    // Shared rope-scaling resolution: an absent <prefix>rope.scaling.type keeps the standard rotation (null).
    // A declared type resolves "none" to the same unscaled path and "yarn" to the YaRN plan; any other variant
    // name fails the load naming the key and the declared value, as does any scaling operand declared without
    // a variant name — the reference's linear default — which the rejection names by the key the file carries.
    // The YaRN plan reads the multiplier from <prefix>rope.scaling.factor (1f when absent), requires the
    // original context length under <prefix>rope.scaling.original_context_length, and reads the attention
    // factor, the correction scale, and the beta pair from <prefix>rope.scaling.attn_factor,
    // <prefix>rope.scaling.yarn_ext_factor, <prefix>rope.scaling.yarn_beta_fast, and
    // <prefix>rope.scaling.yarn_beta_slow against the reference's defaults (1f, 1f, 32f, 1f), with the
    // declared <prefix>rope.scaling.yarn_log_multiplier (0f when absent) selecting the reference's
    // all-dimensions magnitude divisor inside the attention factor.
    private static RopeScalingPlan? ResolveRopeScaling(GgufMetadata metadata, string prefix)
    {
        string typeKey = prefix + "rope.scaling.type";

        if (!metadata.Has(typeKey))
        {
            foreach (string key in metadata.Keys)
            {
                if (key.StartsWith(prefix + "rope.scaling.", StringComparison.Ordinal))
                {
                    throw new ModelException(
                        $"GGUF metadata key '{key}' declares rope scaling without a {typeKey} variant; " +
                        "only the 'yarn' variant is supported");
                }
            }

            return null;
        }

        string variant;
        try
        {
            variant = metadata.Get(typeKey).AsString();
        }
        catch (GgufException ex)
        {
            throw new ModelException($"GGUF metadata key '{typeKey}' must be a string", ex);
        }

        if (variant == "none")
        {
            return null;
        }

        if (variant != "yarn")
        {
            throw new ModelException(
                $"GGUF metadata key '{typeKey}' declares rope scaling variant '{variant}'; " +
                "only the 'yarn' variant is supported");
        }

        float multiplier = ResolveScalingFactor(metadata, prefix + "rope.scaling.factor");

        int originalContext = RequireInt(metadata, prefix + "rope.scaling.original_context_length");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(originalContext);

        float declaredAttention = ResolveScalingMagnitude(metadata, prefix + "rope.scaling.attn_factor", fallback: 1f);
        float logMultiplier = ResolveScalingMagnitude(metadata, prefix + "rope.scaling.yarn_log_multiplier", fallback: 0f);
        float correctionScale = ResolveScalingMagnitude(metadata, prefix + "rope.scaling.yarn_ext_factor", fallback: 1f);
        float betaFast = ResolveScalingMagnitude(metadata, prefix + "rope.scaling.yarn_beta_fast", fallback: 32f);
        float betaSlow = ResolveScalingMagnitude(metadata, prefix + "rope.scaling.yarn_beta_slow", fallback: 1f);

        return new RopeScalingPlan(
            Variant: "yarn",
            Multiplier: multiplier,
            OriginalContextLength: originalContext,
            AttentionFactor: ResolveAttentionFactor(multiplier, declaredAttention, logMultiplier, correctionScale),
            CorrectionScale: correctionScale,
            BetaFast: betaFast,
            BetaSlow: betaSlow);
    }

    // The variant's score-side attention factor. With a non-zero correction scale the reference rebuilds it from
    // the multiplier's magnitude scale — getMscale(m, 1) alone, or getMscale(m, 1) divided by getMscale(m, s)
    // when an all-dimensions log multiplier is declared — and divides out that same magnitude scale, because the
    // angle tables fold it into their own entries; with the correction scale disabled the declared factor stands
    // alone. The declared attention factor then multiplies whatever the variant resolved. getMscale(m, s) is 1
    // for a multiplier not above 1 and 0.1 * s * ln(m) + 1 above it.
    private static float ResolveAttentionFactor(float multiplier, float declaredAttention, float logMultiplier, float correctionScale)
    {
        if (correctionScale == 0f)
        {
            return declaredAttention;
        }

        double resolved = logMultiplier != 0f
            ? Mscale(multiplier, 1.0) / Mscale(multiplier, logMultiplier)
            : Mscale(multiplier, 1.0);

        resolved *= 1.0 / (1.0 + 0.1 * Math.Log(multiplier));

        return (float)(resolved * declaredAttention);
    }

    private static double Mscale(double scale, double magnitude) =>
        scale <= 1.0 ? 1.0 : 0.1 * magnitude * Math.Log(scale) + 1.0;

    // Shared scaling-factor resolution: an absent key means the unscaled multiplier (one). A declared key is
    // read as a double and must land in the float range as a positive finite number — a non-positive, NaN,
    // infinite, or out-of-range declaration fails the load with a message naming the key and the declared value.
    private static float ResolveScalingFactor(GgufMetadata metadata, string key)
    {
        if (!metadata.Has(key))
        {
            return 1f;
        }

        GgufValue declared = metadata.Get(key);

        double declaredValue;
        try
        {
            declaredValue = declared.AsDouble();
        }
        catch (GgufException ex)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares {declared}; rope scaling factor values must be positive finite numbers", ex);
        }

        float value = (float)declaredValue;
        if (!float.IsFinite(value) || value <= 0f)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares {declaredValue.ToString(CultureInfo.InvariantCulture)}; " +
                "rope scaling factor values must be positive finite numbers");
        }

        return value;
    }

    // Shared rope-magnitude resolution in the style of the block-norm epsilon: an absent key takes the fallback.
    // A declared key is read as a double and must land in the float range as a non-negative finite number — a
    // negative, NaN, infinite, or out-of-range declaration fails the load with a message naming the key and the
    // declared value.
    private static float ResolveScalingMagnitude(GgufMetadata metadata, string key, float fallback)
    {
        if (!metadata.Has(key))
        {
            return fallback;
        }

        GgufValue declared = metadata.Get(key);

        double declaredValue;
        try
        {
            declaredValue = declared.AsDouble();
        }
        catch (GgufException ex)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares {declared}; rope scaling magnitude values must be non-negative finite numbers", ex);
        }

        float value = (float)declaredValue;
        if (!float.IsFinite(value) || value < 0f)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares {declaredValue.ToString(CultureInfo.InvariantCulture)}; " +
                "rope scaling magnitude values must be non-negative finite numbers");
        }

        return value;
    }

    // Shared rotary-slice resolution: an absent <prefix>rope.dimension_count means the full head dimension (zero).
    // A declared slice must read as an integer that fits the int range and is a positive even count not exceeding
    // the head dimension — a fractional, text, zero, negative, odd, or out-of-range declaration fails the load
    // with a message naming the key and the declared value.
    private static int ResolveRotarySlice(GgufMetadata metadata, string prefix, int headDim)
    {
        string key = prefix + "rope.dimension_count";
        if (!metadata.Has(key))
        {
            return 0;
        }

        GgufValue declared = metadata.Get(key);

        long declaredValue;
        try
        {
            declaredValue = declared.AsLong();
        }
        catch (GgufException ex)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares {declared}; rope dimension count values must be positive even integers", ex);
        }

        if (declaredValue <= 0 || declaredValue > headDim || declaredValue % 2 != 0)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares {declaredValue.ToString(CultureInfo.InvariantCulture)}; " +
                $"the rope dimension count must be a positive even integer not exceeding the head dimension ({headDim})");
        }

        return (int)declaredValue;
    }

    // Shared sparse-FFN resolution: the declared <prefix>expert_count decides whether the block is expert-routed
    // — absent means the dense gate/up/down block. A declared count must read as a positive integer that fits
    // the int range; a fractional, text, zero, negative, or out-of-int-range declaration fails the load with a
    // message naming the key and the declared value, in the style of the attention window. The routing count
    // reads <prefix>expert_used_count, defaults to the expert count when the key is absent, must be a positive
    // integer, and must not exceed the expert count. The shared expert reads <prefix>expert_shared_count,
    // absent or zero meaning none. The routing-weight operands
    // read <prefix>expert_weights_scale — a non-negative finite number, zero (the reference's disabled default)
    // when the key is absent — and <prefix>expert_weights_norm, false when the key is absent. The roles come
    // from the resolved profile's declared sparse-FFN shape, or from the map's sparse names when the profile
    // declares no shape of its own; the shared-expert roles are carried only when the file declares a shared
    // expert and the resolved profile names them. The shared expert's per-token selector vector is named by the
    // profile alone: whether it gates anything is decided later, by whether the file ships that tensor. The storage
    // shape reads <prefix>expert_layout — absent means stacked, 'shared_basis' declares a factorized block — and a
    // factorized declaration must also declare its rank under <prefix>expert_rank while carrying the basis,
    // coefficient, and output-basis roles; a stacked block names none of them and declaring a rank for one is an
    // inconsistent declaration rather than something to ignore.
    private static SparseFfnDescriptor? ResolveSparseFfn(GgufMetadata metadata, string prefix, SparseFfnDescriptor? shape)
    {
        int expertCount = ResolveMoECount(metadata, prefix + "expert_count", fallback: 0, positive: true);
        if (expertCount == 0)
        {
            return null;
        }

        int routed = ResolveMoECount(metadata, prefix + "expert_used_count", fallback: expertCount, positive: true);
        if (routed > expertCount)
        {
            throw new ModelException(
                $"GGUF metadata key '{prefix}expert_used_count' declares {routed.ToString(CultureInfo.InvariantCulture)}; " +
                $"a sparse-FFN block cannot route more experts than its {prefix}expert_count ({expertCount})");
        }

        bool hasSharedExpert = ResolveMoECount(metadata, prefix + "expert_shared_count", fallback: 0, positive: false) > 0;

        SparseFfnLayout layout = ResolveMoELayout(metadata, prefix + "expert_layout");
        int? rank = ResolveMoERank(metadata, prefix, layout);

        return new SparseFfnDescriptor(
            ExpertCount: expertCount,
            RoutedExpertsPerToken: routed,
            HasSharedExpert: hasSharedExpert,
            RouterRole: shape?.RouterRole ?? TensorRole.MoeRouter,
            ExpertGateRole: shape?.ExpertGateRole ?? TensorRole.MoeExpertGate,
            ExpertUpRole: shape?.ExpertUpRole ?? TensorRole.MoeExpertUp,
            ExpertDownRole: shape?.ExpertDownRole ?? TensorRole.MoeExpertDown,
            SharedGateRole: hasSharedExpert ? shape?.SharedGateRole ?? TensorRole.MoeSharedExpertGate : null,
            SharedUpRole: hasSharedExpert ? shape?.SharedUpRole ?? TensorRole.MoeSharedExpertUp : null,
            SharedDownRole: hasSharedExpert ? shape?.SharedDownRole ?? TensorRole.MoeSharedExpertDown : null,
            // Whether a family selects its shared expert with a per-token vector is part of the reference's block
            // semantics, not something a canonical name can infer: only a profile whose sparse-FFN shape names the
            // role carries it, and the file decides at load whether that tensor is there.
            SharedInputGateRole: hasSharedExpert ? shape?.SharedInputGateRole : null,
            Layout: layout,
            Rank: rank,
            // The factorized roles name tensors a sidecar-produced file can reach; they belong to the declared storage
            // shape and nothing resident reads them yet. A stacked block names none of them, which is what keeps an
            // un-factorizable descriptor out of any shared-basis selection.
            ExpertInputBasisRole: layout == SparseFfnLayout.FactorizedSharedBasis ? shape?.ExpertInputBasisRole ?? TensorRole.MoeExpertInputBasis : null,
            ExpertCoefficientRole: layout == SparseFfnLayout.FactorizedSharedBasis ? shape?.ExpertCoefficientRole ?? TensorRole.MoeExpertCoefficients : null,
            ExpertOutputBasisRole: layout == SparseFfnLayout.FactorizedSharedBasis ? shape?.ExpertOutputBasisRole ?? TensorRole.MoeExpertOutputBasis : null,
            ExpertWeightsScale: ResolveMoEScale(metadata, prefix + "expert_weights_scale"),
            NormalizeExpertWeights: metadata.GetBool(prefix + "expert_weights_norm", fallback: false));
    }

    // Shared sparse-FFN layout resolution in the style of the rope-scaling variant name: an absent key means the
    // stacked expert storage every published conversion ships. A declared 'shared_basis' names a factorized block; any
    // other value — or a value that is not a string at all — fails the load naming the key and what it declares.
    private static SparseFfnLayout ResolveMoELayout(GgufMetadata metadata, string key)
    {
        if (!metadata.Has(key))
        {
            return SparseFfnLayout.StackedExperts;
        }

        string declared;
        try
        {
            declared = metadata.Get(key).AsString();
        }
        catch (GgufException ex)
        {
            throw new ModelException($"GGUF metadata key '{key}' must be a string", ex);
        }

        return declared switch
        {
            "stacked" => SparseFfnLayout.StackedExperts,
            "shared_basis" => SparseFfnLayout.FactorizedSharedBasis,
            _ => throw new ModelException(
                $"GGUF metadata key '{key}' declares sparse-FFN expert layout '{declared}'; only 'stacked' and 'shared_basis' are recognized"),
        };
    }

    // Shared factorization-rank resolution: a stacked block declares no rank, and declaring one for it fails naming the
    // key and its value rather than leaving a number nothing can use. A factorized declaration must declare its rank as
    // a positive integer that fits the int range — absent, fractional, text, zero, negative, or out-of-int-range all
    // fail naming the key — because a coefficient-space width is exactly what makes the declared layout readable. The
    // numeric bounds against the matrices it compresses are a kernel question: nothing resident computes with the rank
    // yet, so this validates that a declaration states one and leaves its fit where the tensors themselves get checked.
    private static int? ResolveMoERank(GgufMetadata metadata, string prefix, SparseFfnLayout layout)
    {
        string key = prefix + "expert_rank";

        if (layout != SparseFfnLayout.FactorizedSharedBasis)
        {
            if (metadata.Has(key))
            {
                GgufValue declared = metadata.Get(key);
                throw new ModelException(
                    $"GGUF metadata key '{key}' declares {declared}; only a block whose {prefix}expert_layout is 'shared_basis' can declare {prefix}expert_rank");
            }

            return null;
        }

        if (!metadata.Has(key))
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' is not present: a sparse-FFN block declaring {prefix}expert_layout 'shared_basis' must declare the rank its coefficient tensors carry");
        }

        return ResolveMoECount(metadata, key, fallback: 0, positive: true);
    }

    // Shared sparse-FFN strategy resolution in the style of the per-head QK-norm mode: an absent declaration keeps the
    // resolved profile's default, and a declared name overrides it. Only these two names are recognized — anything else
    // fails naming the key and the value, as every other variant-name declaration does. A selection is checked against
    // what the descriptor says the file stores rather than downgraded to something runnable: asking for shared-basis
    // low-rank work from a block whose experts are stacked dense matrices reports both sides of that mismatch with the
    // two declarations that would settle it, because silently running the resident path under a low-rank selection would
    // report an experiment that never ran. A block that declares no experts carries no strategy value at all.
    private static SparseFfnStrategy? ResolveSparseFfnStrategy(
        GgufMetadata metadata,
        string prefix,
        SparseFfnStrategy profileDefault,
        SparseFfnDescriptor? descriptor)
    {
        if (descriptor is null)
        {
            return null;
        }

        string key = prefix + "sparse_ffn_strategy";
        if (!metadata.Has(key))
        {
            return profileDefault;
        }

        string declared;
        try
        {
            declared = metadata.Get(key).AsString();
        }
        catch (GgufException ex)
        {
            throw new ModelException($"GGUF metadata key '{key}' must be a string", ex);
        }

        SparseFfnStrategy requested = declared switch
        {
            "dense_resident" => SparseFfnStrategy.DenseResident,
            "low_rank_shared_basis" => SparseFfnStrategy.LowRankSharedBasis,
            _ => throw new ModelException(
                $"GGUF metadata key '{key}' declares sparse-FFN strategy '{declared}'; only 'dense_resident' and 'low_rank_shared_basis' are recognized"),
        };

        if (requested == SparseFfnStrategy.LowRankSharedBasis && descriptor.Layout != SparseFfnLayout.FactorizedSharedBasis)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares sparse-FFN strategy 'low_rank_shared_basis' on a block whose {prefix}expert_layout stores its experts as stacked dense matrices with no rank: " +
                $"a shared-basis selection needs the factorized storage this file does not declare — declare {prefix}expert_layout 'shared_basis' with an {prefix}expert_rank, or select 'dense_resident'");
        }

        return requested;
    }

    // Shared MoE count resolution in the style of the attention window: an absent key takes the fallback. A
    // declared key must read as an integer that fits the int range and is positive when it names the expert
    // count or the routing count, non-negative when it names the shared-expert count — a fractional, text,
    // zero-for-a-routed-count, negative, or out-of-int-range declaration fails the load with a message naming
    // the key and the declared value.
    private static int ResolveMoECount(GgufMetadata metadata, string key, int fallback, bool positive)
    {
        if (!metadata.Has(key))
        {
            return fallback;
        }

        GgufValue declared = metadata.Get(key);

        long declaredValue;
        try
        {
            declaredValue = declared.AsLong();
        }
        catch (GgufException ex)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares {declared}; sparse-FFN count values must be {(positive ? "positive" : "non-negative")} integers", ex);
        }

        if (declaredValue < (positive ? 1 : 0) || declaredValue > int.MaxValue)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares {declaredValue.ToString(CultureInfo.InvariantCulture)}; " +
                $"sparse-FFN count values must be {(positive ? "positive" : "non-negative")} integers");
        }

        return (int)declaredValue;
    }

    // Shared MoE weight-scale resolution in the style of the block-norm epsilon: an absent key means the
    // reference's disabled default (zero). A declared key is read as a double and must land in the float range
    // as a non-negative finite number — a negative, NaN, infinite, or out-of-range declaration fails the load
    // with a message naming the key and the declared value.
    private static float ResolveMoEScale(GgufMetadata metadata, string key)
    {
        if (!metadata.Has(key))
        {
            return 0f;
        }

        double declared;
        try
        {
            declared = metadata.Get(key).AsDouble();
        }
        catch (GgufException ex)
        {
            throw new ModelException($"GGUF metadata key '{key}' must be a number", ex);
        }

        float scale = (float)declared;
        if (!float.IsFinite(scale) || scale < 0f)
        {
            throw new ModelException(
                $"GGUF metadata key '{key}' declares {declared.ToString(CultureInfo.InvariantCulture)}; " +
                "sparse-FFN weight scale values must be non-negative finite numbers");
        }

        return scale;
    }

    // Shared feed-forward width resolution: the declared <prefix>feed_forward_length wins when present, exactly as every
    // family today reads it. A file that names no dense intermediate size but declares an expert one under
    // <prefix>expert_feed_forward_length takes its sparse-FFN width instead — which is what a routed-only conversion whose
    // shared and routed blocks share one intermediate size spells (the qwen4exp target ships all of its FFN material as
    // *_exps/*_shexp stacks with no dense key at all), so reading the expert declaration records the block's actual width
    // rather than failing on a name this family never writes. The substitution reports itself once in the style of the
    // vocabulary-fallback line; it lands only when the dense key is absent and an expert one present, and every other file
    // resolves exactly what it declares — including today's rejection naming the missing dense key when neither exists.
    private static int ResolveFeedForwardLength(GgufMetadata metadata, string prefix, GgufFile gguf, Action<string>? logLine)
    {
        if (metadata.Has(prefix + "feed_forward_length"))
        {
            return RequireInt(metadata, prefix + "feed_forward_length");
        }

        if (!metadata.Has(prefix + "expert_feed_forward_length"))
        {
            throw new ModelException($"GGUF metadata key '{prefix}feed_forward_length' is not present");
        }

        int expertWidth = RequireInt(metadata, prefix + "expert_feed_forward_length");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(expertWidth);

        if (logLine is not null)
        {
            logLine(
                $"ffn_fallback model='{ModelName(metadata, gguf)}' source={prefix}expert_feed_forward_length " +
                $"value={expertWidth.ToString(CultureInfo.InvariantCulture)}");
        }

        return expertWidth;
    }

    // Shared qwen4exp geometry resolution — the exact key names pinned by the target file's own header (the loader gate over both shipped shards reads them unchanged): <prefix>ssm.group_count/state_size/inner_size/time_step_rank/conv_kernel for the gated-delta-net side, <prefix>attention.indexer.head_count/key_length/top_k for selection budgeting, and <prefix>hyper_connection.count/low_rank for stream mixing. The value-head count derives as inner size divided by head dimension — a declaration whose division leaves a remainder fails naming both keys with their values. The declared gate-row count (time_step_rank) must cover exactly those heads: decay weights, dt biases and gate rows each carry one entry per value head, so a mismatch names key + value against the derived count rather than silently padding or truncating gates. <prefix>attention.compress_ratios is an int32 array with exactly block_count non-negative entries — zero means linear attention, positive declares that full layer's pooling ratio for sparse selection (a ratio below 2 groups no cells and runs dense at the layer level: a legal declaration). An absent key fails naming it; wrong length or out-of-range entries name what was declared. When <prefix>full_attention_interval is also declared, its group ends must match the array exactly — every positive-ratio position lands on one and every group end carries one; either side failing names both sides with the layer index. Absent interval leaves the array sole authority (documented choice for synthetic fixtures). Indexer counts and hyper-connection count/rank are required positives.
    private static Qwen4ExpGeometry ResolveQwen4Exp(GgufMetadata metadata, string prefix, int layers)
    {
        int groupCount = RequireInt(metadata, prefix + "ssm.group_count");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(groupCount);

        int stateSize = RequireInt(metadata, prefix + "ssm.state_size");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stateSize);

        int innerSize = RequireInt(metadata, prefix + "ssm.inner_size");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(innerSize);

        if (innerSize % stateSize != 0)
        {
            throw new ModelException(
                $"{prefix}ssm.inner_size ({innerSize}) is not a whole number of value heads at {prefix}ssm.state_size ({stateSize}); " +
                "the concatenated value width must divide exactly by the shared head dimension");
        }

        int valueHeads = innerSize / stateSize;

        int timeStepRank = RequireInt(metadata, prefix + "ssm.time_step_rank");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(timeStepRank);

        if (timeStepRank != valueHeads)
        {
            throw new ModelException(
                $"{prefix}ssm.time_step_rank ({timeStepRank}) does not cover the derived value heads of " +
                $"{prefix}ssm.inner_size / {prefix}ssm.state_size = {valueHeads}; decay weights, dt biases and gate rows each carry one entry per value head");
        }

        int convKernel = RequireInt(metadata, prefix + "ssm.conv_kernel");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(convKernel);

        string ratiosKey = prefix + "attention.compress_ratios";
        if (!metadata.Has(ratiosKey))
        {
            throw new ModelException($"GGUF metadata key '{ratiosKey}' is not present: the per-layer attention interleave must be declared");
        }

        int[] ratios;
        try
        {
            ratios = metadata.GetInt32Array(ratiosKey);
        }
        catch (GgufException ex)
        {
            throw new ModelException($"GGUF metadata key '{ratiosKey}' must be an array of integers", ex);
        }

        if (ratios.Length != layers)
        {
            throw new ModelException(
                $"GGUF metadata key '{ratiosKey}' declares {ratios.Length} entries but {prefix}block_count is {layers}; " +
                "the compress-ratio array must cover every layer exactly once");
        }

        AttentionKind[] kinds = new AttentionKind[layers];
        for (int layer = 0; layer < layers; layer++)
        {
            if (ratios[layer] < 0)
            {
                throw new ModelException(
                    $"GGUF metadata key '{ratiosKey}'[{layer}] declares {ratios[layer].ToString(CultureInfo.InvariantCulture)}; " +
                    "compress ratios must be non-negative integers");
            }

            kinds[layer] = ratios[layer] > 0 ? AttentionKind.FullQuerySparseAttention : AttentionKind.LinearGatedDeltaNet;
        }

        if (metadata.Has(prefix + "full_attention_interval"))
        {
            int interval = RequireInt(metadata, prefix + "full_attention_interval");
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(interval);

            for (int layer = 0; layer < layers; layer++)
            {
                bool groupEnd = (layer + 1) % interval == 0;

                if (ratios[layer] > 0 && !groupEnd)
                {
                    throw new ModelException(
                        $"GGUF metadata key '{prefix}full_attention_interval' ({interval}) contradicts '{ratiosKey}'[{layer}] = " +
                        $"{ratios[layer].ToString(CultureInfo.InvariantCulture)}: a positive compress ratio may only land on the last position of an interval group");
                }

                if (groupEnd && ratios[layer] == 0)
                {
                    throw new ModelException(
                        $"GGUF metadata key '{prefix}full_attention_interval' ({interval}) contradicts '{ratiosKey}'[{layer}] = 0: " +
                        "every last position of an interval group must declare a positive compress ratio");
                }
            }
        }

        int indexerHeads = RequireInt(metadata, prefix + "attention.indexer.head_count");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(indexerHeads);

        int indexerKeyLength = RequireInt(metadata, prefix + "attention.indexer.key_length");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(indexerKeyLength);

        int topKCells = RequireInt(metadata, prefix + "attention.indexer.top_k");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topKCells);

        int streamCount = RequireInt(metadata, prefix + "hyper_connection.count");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(streamCount);

        int lowRank = RequireInt(metadata, prefix + "hyper_connection.low_rank");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(lowRank);

        return new Qwen4ExpGeometry(
            AttentionKinds: kinds,
            CompressRatios: ratios,
            LinearAttention: new GdnDescriptor(
                KeyHeadCount: groupCount,
                ValueHeadCount: valueHeads,
                StateDim: stateSize,
                InnerSize: innerSize,
                TimeStepRank: timeStepRank,
                ConvKernelSize: convKernel),
            SparseAttention: new QsaDescriptor(IndexerHeadCount: indexerHeads, IndexerKeyLength: indexerKeyLength, TopKCells: topKCells),
            HyperConnections: new HyperConnectionDescriptor(StreamCount: streamCount, LowRank: lowRank));
    }

    // The qwen4exp table gate runs against the resolved interleave rather than a flat required-role sweep. Every linear layer must carry its fused qkv buffer (whose row count is exactly the conv channels), its z-gate matrix and all seven recurrent-state tensors; in place of those, a full sparse layer carries split projections plus the indexer quartet; both sides share the block's hyper-connection trio on each sublayer side. A required name missing from the kind that needs it fails naming every such tensor with its concrete layer index — the same "tensor '<name>' is missing" identity the engine raises at open, raised here so a plan for this family never resolves past material its declared groups contradict. Where a family tensor lands on a layer whose ratio says the other attention runs, the file names which layers carry misplaced material: ssm_* or attn_gate under a positive ratio, split q/k/v or indexer.* under zero. Present hyper-connection and recurrent tensors are shape-checked against what the metadata declares (inject rows = stream count; down/up cross the flattened streams through the low rank in each direction; norms cover the whole flattened width; conv rows = 2·key-heads·state + inner over kernel columns; gate/out transpose across inner size), so a synthetic fixture proves its layouts match its own declarations without inventing names this family does not ship. Roles whose payload geometry only execution can judge (head-split widths inside fused buffers, rope pairing) stay checked where they run.
    private static void ValidateQwen4ExpTable(TensorMap map, string architecture, LogicalTensorTable tensors, Qwen4ExpGeometry geometry, int hiddenSize)
    {
        List<string> missing = [];
        List<string> misplaced = [];

        for (int layer = 0; layer < geometry.LayerCount; layer++)
        {
            AttentionKind kind = geometry.AttentionKinds[layer];
            TensorRole[] requiredHere = kind == AttentionKind.LinearGatedDeltaNet ? LinearAttentionRoles : FullSparseRoles;
            TensorRole[] forbiddenHere = kind == AttentionKind.LinearGatedDeltaNet ? FullSparseRoles : LinearAttentionRoles;

            foreach (TensorRole role in requiredHere)
            {
                if (map[role].Resolve(tensors, layer) is null)
                {
                    missing.Add(map[role].NameFor(layer));
                }
            }

            foreach (TensorRole role in forbiddenHere)
            {
                LogicalTensor? present = map[role].Resolve(tensors, layer);
                if (present is not null)
                {
                    misplaced.Add(present.Name);
                }
            }

            int flattened = geometry.HyperConnections.FlattenedStreamFloats(hiddenSize);
            foreach ((TensorRole role, int[] expectedShape) in HyperConnectionMixing(geometry.HyperConnections, hiddenSize))
            {
                TensorDescriptor descriptor = map[role];
                LogicalTensor? tensor = descriptor.Resolve(tensors, layer);

                if (tensor is null)
                {
                    missing.Add(descriptor.NameFor(layer));
                    continue;
                }

                bool ok = role == TensorRole.AttentionStreamNorm || role == TensorRole.FeedForwardStreamNorm
                    ? VectorMatches(tensor.Shape, flattened)
                    : ShapeMatches(tensor.Shape, expectedShape);

                if (!ok)
                {
                    throw new ModelException(
                        $"tensor '{descriptor.NameFor(layer)}' declares shape [{string.Join(", ", tensor.Shape.Dims)}] but the declared " +
                        $"{architecture} hyper-connection geometry (count x hidden = {flattened}, rank {geometry.HyperConnections.LowRank}) needs " +
                        $"[{string.Join(", ", expectedShape)}]: inject rows are stream count, down/up cross the flattened streams through the low rank in each direction, norms cover the whole flattened width");
                }
            }

            if (kind == AttentionKind.LinearGatedDeltaNet)
            {
                ValidateLinearBlockShapes(map, tensors, geometry.LinearAttention, architecture, layer, hiddenSize);
            }
        }

        foreach ((TensorRole role, int[] expectedShape) in OutputHyperConnectionMixing(geometry.HyperConnections, hiddenSize))
        {
            TensorDescriptor descriptor = map[role];
            LogicalTensor? tensor = descriptor.Resolve(tensors); // global scope: the layer argument is ignored entirely.

            if (tensor is null)
            {
                missing.Add(descriptor.NameFor(-1));
                continue;
            }

            bool ok = role == TensorRole.OutputStreamNorm ? VectorMatches(tensor.Shape, expectedShape[0]) : ShapeMatches(tensor.Shape, expectedShape);
            if (!ok)
            {
                throw new ModelException(
                    $"tensor '{descriptor.Name}' declares shape [{string.Join(", ", tensor.Shape.Dims)}] but the declared " +
                    $"{architecture} output hyper-connection geometry needs [{string.Join(", ", expectedShape)}]: one final down/up pair across the flattened streams through the low rank, with a norm over their whole width");
            }
        }

        if (map[TensorRole.TokenEmbedding].Resolve(tensors) is null)
        {
            missing.Add(map[TensorRole.TokenEmbedding].NameFor(-1));
        }

        if (missing.Count > 0 || misplaced.Count > 0)
        {
            throw new ModelException(
                $"model architecture '{architecture}' tensor table does not match its resolved per-layer attention interleave " +
                $"({geometry.LinearLayerCount} gated-delta-net layers, {geometry.FullLayerCount} sparse full-attention layers): " +
                $"missing tensors: {string.Join(", ", missing)}; misplaced tensors: {string.Join(", ", misplaced)}");
        }
    }

    private static void ValidateLinearBlockShapes(TensorMap map, LogicalTensorTable tensors, GdnDescriptor gdn, string architecture, int layer, int hiddenSize)
    {
        foreach ((TensorRole role, int[] expectedShape) in new (TensorRole, int[])[]
        {
            (TensorRole.AttentionQkv, [gdn.ConvChannels, hiddenSize]),
            (TensorRole.LinearAttentionGate, [gdn.InnerSize, hiddenSize]),
            (TensorRole.SsmOutputProjection, [hiddenSize, gdn.InnerSize]),
            (TensorRole.SsmConvolutionKernel, [gdn.ConvChannels, gdn.ConvKernelSize]),
            (TensorRole.SsmAlphaProjection, [gdn.TimeStepRank, hiddenSize]),
            (TensorRole.SsmBetaProjection, [gdn.TimeStepRank, hiddenSize]),
        })
        {
            TensorDescriptor descriptor = map[role];
            LogicalTensor? tensor = descriptor.Resolve(tensors, layer);

            // Presence belongs to the aggregate sweep — a required linear-block name that never arrives lands in its missing list naming the tensor; this pass judges only what did arrive.
            if (tensor is null)
            {
                continue;
            }

            if (!ShapeMatches(tensor.Shape, expectedShape))
            {
                throw new ModelException(
                    $"tensor '{descriptor.NameFor(layer)}' declares shape [{string.Join(", ", tensor.Shape.Dims)}] but the declared {architecture} gated-delta-net geometry needs " +
                    $"[{string.Join(", ", expectedShape)}]: conv rows are 2 key heads x state dim plus inner size over kernel columns, gate and out transpose across inner size, alpha/beta carry one row per value head");
            }
        }

        foreach ((TensorRole role, int length) in new (TensorRole, int)[] { (TensorRole.SsmDecayWeight, gdn.TimeStepRank), (TensorRole.SsmTimeStepBias, gdn.TimeStepRank), (TensorRole.SsmHeadNorm, gdn.StateDim) })
        {
            TensorDescriptor descriptor = map[role];
            LogicalTensor? tensor = descriptor.Resolve(tensors, layer);

            if (tensor is null)
            {
                continue; // same posture: the sweep above already names it missing.
            }

            if (!VectorMatches(tensor.Shape, length))
            {
                throw new ModelException(
                    $"tensor '{descriptor.NameFor(layer)}' declares shape [{string.Join(", ", tensor.Shape.Dims)}] but the declared {architecture} gated-delta-net geometry needs a single vector of length {length}: one entry per value head for decay and dt bias, one per state dimension for the head norm");
            }
        }
    }

    private static readonly TensorRole[] LinearAttentionRoles = [TensorRole.AttentionQkv, TensorRole.LinearAttentionGate, TensorRole.SsmDecayWeight, TensorRole.SsmAlphaProjection, TensorRole.SsmBetaProjection, TensorRole.SsmConvolutionKernel, TensorRole.SsmTimeStepBias, TensorRole.SsmHeadNorm, TensorRole.SsmOutputProjection];

    private static readonly TensorRole[] FullSparseRoles = [TensorRole.AttentionQuery, TensorRole.AttentionKey, TensorRole.AttentionValue, TensorRole.AttentionOutput, TensorRole.IndexerQueryProjection, TensorRole.IndexerKeyProjection, TensorRole.IndexerQueryNorm, TensorRole.IndexerKeyNorm];

    private static (TensorRole, int[])[] HyperConnectionMixing(HyperConnectionDescriptor hc, int hiddenSize)
    {
        int flattened = hc.FlattenedStreamFloats(hiddenSize);
        return
        [
            (TensorRole.AttentionStreamInject, [hc.StreamCount, flattened]),
            (TensorRole.AttentionStreamDown, [hc.LowRank, flattened]),
            (TensorRole.AttentionStreamUp, [flattened, hc.LowRank]),
            (TensorRole.AttentionStreamNorm, [flattened]),
            (TensorRole.FeedForwardStreamInject, [hc.StreamCount, flattened]),
            (TensorRole.FeedForwardStreamDown, [hc.LowRank, flattened]),
            (TensorRole.FeedForwardStreamUp, [flattened, hc.LowRank]),
            (TensorRole.FeedForwardStreamNorm, [flattened]),
        ];
    }

    private static (TensorRole, int[])[] OutputHyperConnectionMixing(HyperConnectionDescriptor hc, int hiddenSize)
    {
        int flattened = hc.FlattenedStreamFloats(hiddenSize);
        return [(TensorRole.OutputStreamDown, [hc.LowRank, flattened]), (TensorRole.OutputStreamUp, [flattened, hc.LowRank]), (TensorRole.OutputStreamNorm, [flattened])];
    }

    private static bool ShapeMatches(TensorShape? shape, int[] expected) =>
        shape is not null && shape.Dims.Length == expected.Length && shape.Dims.AsSpan().SequenceEqual(expected);

    private static bool VectorMatches(TensorShape? shape, int length) =>
        shape is not null && (shape.Dims.Length == 1 ? shape.Dims[0] : -1) == length;
}
