using System.Globalization;

using Lair.Contracts;
using Lair.Loader;

namespace Lair.Models;

/// <summary>
/// Resolves a GGUF file into a <see cref="ModelPlan"/>: the architecture is looked up in
/// <see cref="ModelProfiles"/>, the geometry is read from the profile's metadata prefix, the vocabulary size
/// comes from the declared key or the tokenizer vocabulary length, the attention score window comes from the
/// declared <c>&lt;prefix&gt;attention.sliding_window</c> (unbounded when the key is absent), and the tensor map
/// comes from the profile together with the window interleave that decides which layers that window masks. Only
/// standard RoPE over the full head dimension is accepted: any rope-scaling
/// variant or partial-rotary declaration fails at load. The per-head QK-norm mode resolves the declared
/// <c>&lt;prefix&gt;attention.use_qk_norm</c> against the profile's default, the attention-bias flag resolves the
/// declared <c>&lt;prefix&gt;attention.use_bias</c> against the profile's default, and the norm placement comes from
/// the profile alone — no metadata key selects it. The block-norm epsilon read from
/// <c>&lt;prefix&gt;attention.layer_norm_epsilon</c> (zero when the key is absent) feeds the LayerNorm block norms;
    /// the RMS epsilon from <c>&lt;prefix&gt;attention.layer_norm_rms_epsilon</c> feeds the RMS norms and the per-head
    /// QK-norm. The input-embedding gain comes from the profile alone: a profile that scales embeddings resolves
    /// <c>sqrt(hidden size)</c> in the plan, every other profile resolves zero. An unregistered architecture is accepted only through the generic llama-shaped fallback profile, which
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

        foreach (string key in metadata.Keys)
        {
            if (key.StartsWith(prefix + "rope.scaling.", StringComparison.Ordinal))
            {
                throw new ModelException(
                    $"model architecture '{architecture}' declares rope scaling via GGUF key '{key}'; " +
                    "only standard RoPE without scaling variants is supported");
            }
        }

        int hidden = RequireInt(metadata, prefix + "embedding_length");
        int layers = RequireInt(metadata, prefix + "block_count");
        int heads = RequireInt(metadata, prefix + "attention.head_count");
        int kvHeads = RequireInt(metadata, prefix + "attention.head_count_kv");
        int ffn = RequireInt(metadata, prefix + "feed_forward_length");

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

        // Standard full-head RoPE only: a declared rotary slice smaller than the head is not supported.
        if (metadata.Has(prefix + "rope.dimension_count"))
        {
            int rot = RequireInt(metadata, prefix + "rope.dimension_count");
            if (rot != headDim)
            {
                throw new ModelException(
                    $"{prefix}rope.dimension_count ({rot}) differs from the head dimension ({headDim}); " +
                    "partial rotary embeddings are not supported");
            }
        }

        float ropeTheta = metadata.Has(prefix + "rope.freq_base")
            ? (float)metadata.Get(prefix + "rope.freq_base").AsDouble()
            : 10_000f;

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
            SparseFfn = profile.SparseFfn,
        };
    }

    // The generic fallback accepts a tensor table only when it matches the llama-shaped map exactly: every
    // required descriptor resolves, and every tensor in the table is named by some descriptor's primary or
    // alias name (the optional ones included). A tensor no descriptor can name is unexpected; the rejection
    // carries the registry wording with its supported-profile list, names both sides of the mismatch, and
    // states that the llama-shaped fallback was attempted. The block extent is derived from the table itself,
    // so this gate needs no metadata: it runs before any metadata read.
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

    // Shared per-head QK-norm resolution: an absent declaration keeps the resolved profile's mode. A declared false
    // forces Off whatever that default is, and a declared true requests the norm — which no metadata key can express
    // as unweighted, so it lands on the weighted form unless the profile already resolves the unweighted one. The
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

        return profileDefault == QkNormMode.PerHeadUnweighted ? QkNormMode.PerHeadUnweighted : QkNormMode.PerHeadWeighted;
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
}
