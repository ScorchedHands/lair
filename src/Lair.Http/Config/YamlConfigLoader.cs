using System.Globalization;
using YamlDotNet.Core;
using YamlDotNet.RepresentationModel;

namespace Lair.Host.Config;

/// <summary>
/// Reads a LAIR configuration file once and returns a validated
/// <see cref="LairConfig"/>. Unknown keys, out-of-range values, and placement
/// choices this build cannot honour all fail with a <see cref="ConfigException"/>.
/// </summary>
public static class YamlConfigLoader
{
    private const string TopLevel = "the top level";

    private static readonly string[] TopLevelKeys = ["server", "engine", "models", "ui"];
    private static readonly string[] ServerKeys = ["bind", "port"];
    private static readonly string[] EngineKeys = ["threads", "block_size", "max_batch_tokens"];
    private static readonly string[] UiKeys = ["recent_requests"];
    private static readonly string[] ModelKeys = ["name", "path", "type", "max_context_tokens", "kv_blocks", "placement", "force_fenced_tools", "tool_parser", "reasoning"];
    private static readonly string[] PlacementKeys = ["device", "kv_residency", "prefill", "core_set", "thread_quota"];

    /// <summary>Loads and validates <paramref name="filePath"/>; no file watching.</summary>
    public static LairConfig Load(string filePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

        if (!File.Exists(filePath))
        {
            throw new ConfigException($"Configuration file '{filePath}' does not exist.");
        }

        YamlMappingNode root = ReadDocument(filePath);
        Dictionary<string, YamlNode> top = KeysOf(root, TopLevel);
        RejectUnknownKeys(top, TopLevel, TopLevelKeys);

        IReadOnlyList<ModelConfig> models = ParseModels(top);
        ValidateCoreSetsDisjoint(models);

        return new LairConfig
        {
            Server = ParseServer(top),
            Engine = ParseEngine(top),
            Models = models,
            Ui = ParseUi(top),
        };
    }

    private static YamlMappingNode ReadDocument(string filePath)
    {
        try
        {
            using FileStream file = File.OpenRead(filePath);
            using StreamReader reader = new(file);

            YamlStream yaml = new();
            yaml.Load(reader);

            if (yaml.Documents.Count == 0)
            {
                throw new ConfigException($"Configuration file '{filePath}' is empty.");
            }

            return AsMapping(yaml.Documents[0].RootNode, $"the root of '{filePath}'");
        }
        catch (ConfigException)
        {
            throw;
        }
        catch (Exception ex) when (ex is YamlException or IOException or UnauthorizedAccessException)
        {
            throw new ConfigException($"Configuration file '{filePath}' could not be read as YAML: {ex.Message}", ex);
        }
    }

    private static ServerConfig ParseServer(Dictionary<string, YamlNode> top)
    {
        const string where = "'server'";
        if (!top.TryGetValue("server", out YamlNode? node))
        {
            return new ServerConfig();
        }

        Dictionary<string, YamlNode> keys = KeysOf(AsMapping(node, where), where);
        RejectUnknownKeys(keys, where, ServerKeys);

        string bind = TryText(keys, "bind", $"server.bind") ?? "127.0.0.1";
        if (string.IsNullOrWhiteSpace(bind))
        {
            throw new ConfigException("server.bind must be a non-empty address to listen on.");
        }

        int port = TryInteger(keys, "port", "server.port") ?? 8080;
        if (port is < 1 or > 65535)
        {
            throw new ConfigException($"server.port {port} is out of range: 'port' must be between 1 and 65535.");
        }

        return new ServerConfig { Bind = bind, Port = port };
    }

    private static UiConfig ParseUi(Dictionary<string, YamlNode> top)
    {
        const string where = "'ui'";
        if (!top.TryGetValue("ui", out YamlNode? node))
        {
            return new UiConfig();
        }

        Dictionary<string, YamlNode> keys = KeysOf(AsMapping(node, where), where);
        RejectUnknownKeys(keys, where, UiKeys);

        int recentRequests = TryInteger(keys, "recent_requests", "ui.recent_requests") ?? 25;
        if (recentRequests is < 1 or > 1000)
        {
            throw new ConfigException($"ui.recent_requests {recentRequests} is out of range: 'recent_requests' must be between 1 and 1000.");
        }

        return new UiConfig { RecentRequests = recentRequests };
    }

    private static EngineConfig ParseEngine(Dictionary<string, YamlNode> top)
    {
        const string where = "'engine'";
        if (!top.TryGetValue("engine", out YamlNode? node))
        {
            return new EngineConfig();
        }

        Dictionary<string, YamlNode> keys = KeysOf(AsMapping(node, where), where);
        RejectUnknownKeys(keys, where, EngineKeys);

        int threads = TryInteger(keys, "threads", "engine.threads") ?? 0;
        if (threads < 0)
        {
            throw new ConfigException($"engine.threads {threads} is invalid: 'threads' must be 0 or greater (0 selects automatically).");
        }

        int blockSize = TryInteger(keys, "block_size", "engine.block_size") ?? 16;
        if (blockSize != 16)
        {
            throw new ConfigException($"engine.block_size {blockSize} is invalid: block size is fixed to 16 in this build.");
        }

        int maxBatchTokens = TryInteger(keys, "max_batch_tokens", "engine.max_batch_tokens") ?? 512;
        if (maxBatchTokens < 1)
        {
            throw new ConfigException($"engine.max_batch_tokens {maxBatchTokens} is invalid: 'max_batch_tokens' must be 1 or greater.");
        }

        return new EngineConfig { Threads = threads, BlockSize = blockSize, MaxBatchTokens = maxBatchTokens };
    }

    private static IReadOnlyList<ModelConfig> ParseModels(Dictionary<string, YamlNode> top)
    {
        const string where = "'models'";
        if (!top.TryGetValue("models", out YamlNode? node))
        {
            throw new ConfigException($"Missing required key 'models' at {TopLevel}; allowed keys are: {Allowed(TopLevelKeys)}.");
        }

        if (node.NodeType != YamlNodeType.Sequence)
        {
            throw new ConfigException($"{where} must be a list of model entries.");
        }

        List<YamlNode> entries = [.. ((YamlSequenceNode)node).Children];
        if (entries.Count != 1)
        {
            throw new ConfigException(
                $"{where} must contain exactly one model because this build serves a single model (found {entries.Count}).");
        }

        return [ParseModel(entries[0], 0)];
    }

    private static ModelConfig ParseModel(YamlNode node, int index)
    {
        string where = $"models[{index}]";
        Dictionary<string, YamlNode> keys = KeysOf(AsMapping(node, where), where);
        RejectUnknownKeys(keys, where, ModelKeys);

        string? name = TryText(keys, "name", $"models[{index}].name");
        if (name is null)
        {
            throw new ConfigException($"Missing required key 'name' under {where}; allowed keys are: {Allowed(ModelKeys)}.");
        }

        string? path = TryText(keys, "path", $"models[{index}].path");
        if (path is null)
        {
            throw new ConfigException($"Missing required key 'path' under {where}; allowed keys are: {Allowed(ModelKeys)}.");
        }
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ConfigException($"models[{index}].path must be a non-empty path to the model file on disk.");
        }
        if (!File.Exists(path))
        {
            throw new ConfigException($"models[{index}].path '{path}' does not exist on disk.");
        }

        string type = TryText(keys, "type", $"models[{index}].type") ?? "chat";
        if (type != "chat")
        {
            throw new ConfigException(
                $"models[{index}].type '{type}' is invalid: embed models are not supported in this build and 'chat' is the only accepted value.");
        }

        int maxContextTokens = TryInteger(keys, "max_context_tokens", $"models[{index}].max_context_tokens") ?? 4096;
        if (maxContextTokens < 128)
        {
            throw new ConfigException(
                $"models[{index}].max_context_tokens {maxContextTokens} is invalid: 'max_context_tokens' must be 128 or greater.");
        }

        int kvBlocks = TryInteger(keys, "kv_blocks", $"models[{index}].kv_blocks") ?? 256;
        if (kvBlocks < 8)
        {
            throw new ConfigException($"models[{index}].kv_blocks {kvBlocks} is invalid: 'kv_blocks' must be 8 or greater.");
        }

        // Minimal tool-precedence override: when true, LAIR's fenced tool block is used for declared tools even when
        // the model's chat template reports tool support. Absent means false — the template's capabilities decide.
        bool forceFencedTools = TryBoolean(keys, "force_fenced_tools", $"models[{index}].force_fenced_tools") ?? false;

        // Tool output parser: auto (default), or one of the three explicit choices. An unrecognized value fails
        // naming the model entry and the key with the accepted set.
        string toolParser = TryText(keys, "tool_parser", $"models[{index}].tool_parser") ?? "auto";
        if (toolParser is not ("auto" or "template" or "fenced" or "none"))
        {
            throw new ConfigException(
                $"models[{index}].tool_parser '{toolParser}' is invalid: 'tool_parser' accepts only 'auto', 'template', 'fenced', or 'none'.");
        }

        // Reasoning visibility policy: keep (default) or strip. An unrecognized value fails naming the model entry and
        // the key with the accepted set, exactly as the tool parser choice does.
        string reasoning = TryText(keys, "reasoning", $"models[{index}].reasoning") ?? "keep";
        if (reasoning is not ("keep" or "strip"))
        {
            throw new ConfigException(
                $"models[{index}].reasoning '{reasoning}' is invalid: 'reasoning' accepts only 'keep' or 'strip'.");
        }

        return new ModelConfig
        {
            Name = name,
            Path = path,
            Type = type,
            MaxContextTokens = maxContextTokens,
            KvBlocks = kvBlocks,
            ForceFencedTools = forceFencedTools,
            ToolParser = toolParser,
            Reasoning = reasoning,
            Placement = ParsePlacement(keys, index),
        };
    }

    private static PlacementConfig ParsePlacement(Dictionary<string, YamlNode> modelKeys, int index)
    {
        string where = $"models[{index}].placement";
        if (!modelKeys.TryGetValue("placement", out YamlNode? node))
        {
            return new PlacementConfig();
        }

        Dictionary<string, YamlNode> keys = KeysOf(AsMapping(node, where), where);
        RejectUnknownKeys(keys, where, PlacementKeys);

        string device = TryText(keys, "device", $"models[{index}].placement.device") ?? "cpu";
        if (device != "cpu")
        {
            throw new ConfigException(
                $"models[{index}].placement.device '{device}' is invalid: the only accepted value for 'device' is 'cpu'.");
        }

        string kvResidency = TryText(keys, "kv_residency", $"models[{index}].placement.kv_residency") ?? "dram";
        if (kvResidency != "dram")
        {
            throw new ConfigException(
                $"models[{index}].placement.kv_residency '{kvResidency}' is invalid: the only accepted value for 'kv_residency' is 'dram'.");
        }

        string prefill = TryText(keys, "prefill", $"models[{index}].placement.prefill") ?? "cpu";
        if (prefill != "cpu")
        {
            throw new ConfigException(
                $"models[{index}].placement.prefill '{prefill}' is invalid: the only accepted value for 'prefill' is 'cpu'.");
        }

        string? coreSetSpec = TryText(keys, "core_set", $"models[{index}].placement.core_set");
        IReadOnlySet<int> coreSet = string.IsNullOrWhiteSpace(coreSetSpec)
            ? new HashSet<int>()
            : ParseCoreSet(coreSetSpec!, $"models[{index}].placement.core_set");

        int? threadQuota = TryInteger(keys, "thread_quota", $"models[{index}].placement.thread_quota");
        if (threadQuota is < 1)
        {
            throw new ConfigException(
                $"models[{index}].placement.thread_quota {threadQuota} is invalid: 'thread_quota' must be 1 or greater.");
        }

        return new PlacementConfig
        {
            Device = device,
            KvResidency = kvResidency,
            Prefill = prefill,
            CoreSet = coreSet,
            ThreadQuota = threadQuota,
        };
    }

    /// <summary>
    /// Parses a comma-separated list of logical core ids and inclusive ranges,
    /// for example <c>"0-15"</c> or <c>"0,2,8-11"</c>.
    /// </summary>
    private static IReadOnlySet<int> ParseCoreSet(string spec, string key)
    {
        HashSet<int> cores = [];
        foreach (string rawPart in spec.Split(','))
        {
            string part = rawPart.Trim();
            if (part.Length == 0)
            {
                throw new ConfigException(
                    $"{key} '{spec}' is invalid: expected comma-separated core ids or ranges such as '0-15' or '0,2,8-11'.");
            }

            int dash = part.IndexOf('-');
            if (dash < 0)
            {
                if (!int.TryParse(part, NumberStyles.Integer, CultureInfo.InvariantCulture, out int single) || single < 0)
                {
                    throw new ConfigException(
                        $"{key} '{part}' is invalid: core ids must be non-negative integers (expected '0-15' or '0,2,8-11').");
                }

                cores.Add(single);
                continue;
            }

            string head = part[..dash].Trim();
            string tail = part[(dash + 1)..].Trim();
            if (!int.TryParse(head, NumberStyles.Integer, CultureInfo.InvariantCulture, out int from) ||
                !int.TryParse(tail, NumberStyles.Integer, CultureInfo.InvariantCulture, out int to) ||
                from < 0 ||
                to < from)
            {
                throw new ConfigException(
                    $"{key} '{part}' is invalid: a range must be 'low-high' with non-negative integers and low <= high.");
            }

            for (int core = from; core <= to; core++)
            {
                cores.Add(core);
            }
        }

        return cores;
    }

    public static void ValidateCoreSetsDisjoint(IReadOnlyList<ModelConfig> models)
    {
        ArgumentNullException.ThrowIfNull(models);

        // Two pinned models sharing a core id contend for the same execution resources and make the pinning contract
        // meaningless. The current build serves exactly one model, so this rule guards the multi-model path that the
        // placement vocabulary is written for; it validates parsed model objects directly so callers without a YAML
        // document apply the same rule.
        Dictionary<int, string> owners = [];
        for (int index = 0; index < models.Count; index++)
        {
            ModelConfig model = models[index];
            foreach (int core in model.Placement.CoreSet)
            {
                if (!owners.TryAdd(core, model.Name))
                {
                    throw new ConfigException(
                        $"models '{owners[core]}' and '{model.Name}' both pin core {core}: placement core sets must " +
                        "disjointly partition the machine so pinned models never share a core.");
                }
            }
        }
    }

    private static YamlMappingNode AsMapping(YamlNode node, string where)
    {
        if (node is YamlMappingNode mapping)
        {
            return mapping;
        }

        throw new ConfigException($"{where} must be a mapping of key/value pairs.");
    }

    private static Dictionary<string, YamlNode> KeysOf(YamlMappingNode mapping, string where)
    {
        Dictionary<string, YamlNode> keys = new(StringComparer.Ordinal);

        foreach (KeyValuePair<YamlNode, YamlNode> entry in mapping.Children)
        {
            if (entry.Key.NodeType != YamlNodeType.Scalar ||
                ((YamlScalarNode)entry.Key).Value is not string key ||
                key.Length == 0)
            {
                throw new ConfigException($"{where} contains a key that is not a plain name.");
            }

            if (!keys.TryAdd(key, entry.Value))
            {
                throw new ConfigException($"Duplicate key '{key}' under {where}.");
            }
        }

        return keys;
    }

    private static void RejectUnknownKeys(Dictionary<string, YamlNode> keys, string where, string[] allowed)
    {
        foreach (string key in keys.Keys)
        {
            if (Array.IndexOf(allowed, key) < 0)
            {
                throw new ConfigException($"Unknown key '{key}' under {where}; allowed keys are: {Allowed(allowed)}.");
            }
        }
    }

    private static string? TryText(Dictionary<string, YamlNode> keys, string key, string qualifiedKey)
    {
        if (!keys.TryGetValue(key, out YamlNode? node))
        {
            return null;
        }

        if (node.NodeType != YamlNodeType.Scalar || ((YamlScalarNode)node).Value is not string value)
        {
            throw new ConfigException($"{qualifiedKey} must be a text value.");
        }

        return value;
    }

    private static int? TryInteger(Dictionary<string, YamlNode> keys, string key, string qualifiedKey)
    {
        if (!keys.TryGetValue(key, out YamlNode? node))
        {
            return null;
        }

        if (node.NodeType != YamlNodeType.Scalar ||
            ((YamlScalarNode)node).Value is not string text ||
            !int.TryParse(text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out int value))
        {
            throw new ConfigException($"{qualifiedKey} must be an integer value.");
        }

        return value;
    }

    private static bool? TryBoolean(Dictionary<string, YamlNode> keys, string key, string qualifiedKey)
    {
        if (!keys.TryGetValue(key, out YamlNode? node))
        {
            return null;
        }

        if (node.NodeType != YamlNodeType.Scalar ||
            ((YamlScalarNode)node).Value is not string text ||
            !bool.TryParse(text, out bool value))
        {
            throw new ConfigException($"{qualifiedKey} must be a true or false value.");
        }

        return value;
    }

    private static string Allowed(string[] keys) => string.Join(", ", keys);
}
