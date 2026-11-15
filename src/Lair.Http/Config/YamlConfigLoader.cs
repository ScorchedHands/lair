using System.Globalization;
using Lair.Models;
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

    private static readonly string[] TopLevelKeys = ["server", "engine", "models", "ui", "tiers"];
    private static readonly string[] ServerKeys = ["bind", "port"];
    private static readonly string[] EngineKeys = ["threads", "block_size", "max_batch_tokens", "prefill_step_budget_ms", "prefill_kernel"];
    private static readonly string[] UiKeys = ["recent_requests"];
    private static readonly string[] ModelKeys = ["name", "path", "type", "max_context_tokens", "kv_blocks", "placement", "force_fenced_tools", "tool_parser", "reasoning", "max_reasoning_tokens", "max_response_tokens", "max_concurrent_requests", "budgets"];
    private static readonly string[] PlacementKeys = ["device", "kv_residency", "prefill", "core_set", "thread_quota", "prefill_lane", "core_overcommit", "numa_spread"];

    /// <summary>The resident capacity keys a top-level tiers block may declare, in the order a refusal reports them.</summary>
    private static readonly IReadOnlyList<string> TierKeys = ["dram_mb", "dram_cores", "nvme_path", "nodes"];

    /// <summary>The keys one per-node override entry may declare beside the node position it names.</summary>
    private static readonly IReadOnlyList<string> NodeOverrideKeys = ["index", "mb", "cores"];

    /// <summary>The resource keys a model's budgets block may declare, in the order a refusal reports them.</summary>
    private static readonly IReadOnlyList<string> BudgetResourceKeys = [BudgetResourceNames.Weights, BudgetResourceNames.Kv, BudgetResourceNames.PrefillBuffers];

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
            Engine = ParseEngine(top, models),
            Models = models,
            Ui = ParseUi(top),
            Tiers = ParseTiers(top),
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

    /// <summary>
    /// Parses the top-level tiers block: flat resident capacity keys for system memory, a disk path declared as presence and path only,
    /// and an optional list of per-node overrides keyed by node position. Shape is validated here — every known key that expects a number
    /// must state one, positive where it states anything, and a refusal names the qualified key with the offending value (naming the node
    /// index too for a per-node override). Whether declared totals agree with what a host reports is answered when an engine opens,
    /// warn-and-accept: this loader never consults topology. A disk byte ceiling is not accepted here because that position already holds
    /// the complete set; accelerator ceilings are not this block's vocabulary at all.
    /// </summary>
    private static TierCapacityConfig? ParseTiers(Dictionary<string, YamlNode> top)
    {
        if (!top.TryGetValue("tiers", out YamlNode? node))
        {
            return null; // no tiers block declares nothing: today's posture states no public totals.
        }

        const string where = "'tiers'";
        Dictionary<string, YamlNode> keys = KeysOf(AsMapping(node, where), where);
        RejectUnknownKeys(keys, where, TierKeys);

        long? dramMb = TryLong(keys, "dram_mb", "tiers.dram_mb");
        if (dramMb is < 1)
        {
            throw new ConfigException($"tiers.dram_mb {dramMb} is invalid: 'dram_mb' must be 1 or greater — omit the key to declare no byte total.");
        }

        int? dramCores = TryInteger(keys, "dram_cores", "tiers.dram_cores");
        if (dramCores is < 1)
        {
            throw new ConfigException($"tiers.dram_cores {dramCores} is invalid: 'dram_cores' must be 1 or greater — omit the key to declare no core total.");
        }

        // A disk path declares only that this host reads from one. No ceiling belongs at that position, so nothing here reads a number beside it.
        string? nvmePath = TryText(keys, "nvme_path", "tiers.nvme_path");
        if (nvmePath is not null && string.IsNullOrWhiteSpace(nvmePath))
        {
            throw new ConfigException("tiers.nvme_path must name a non-empty path this host reads from — omit the key to say nothing about disk storage.");
        }

        IReadOnlyList<NodeCapacityOverride> nodes = ParseNodeOverrides(keys);

        if (dramMb is null && dramCores is null && nvmePath is null && nodes.Count == 0)
        {
            throw new ConfigException(
                $"{where} declares nothing: a tiers block states at least one of {Vocabulary(TierKeys)} — omit 'tiers' entirely to declare no capacity.");
        }

        return new TierCapacityConfig { DramMb = dramMb, DramCores = dramCores, NvmePath = nvmePath, Nodes = nodes };
    }

    // Per-node overrides: each entry names the node position it disagrees with and states its own bytes and/or cores. A malformed value
    // fails naming key, value, and the index that entry names; a duplicate or out-of-range index cannot name one node position at all.
    private static IReadOnlyList<NodeCapacityOverride> ParseNodeOverrides(Dictionary<string, YamlNode> tierKeys)
    {
        if (!tierKeys.TryGetValue("nodes", out YamlNode? node))
        {
            return [];
        }

        const string where = "'tiers.nodes'";
        if (node.NodeType != YamlNodeType.Sequence)
        {
            throw new ConfigException($"{where} must be a list of per-node overrides, each naming an index and stating 'mb' and/or 'cores'.");
        }

        List<NodeCapacityOverride> overrides = [];
        HashSet<int> declaredIndexes = [];
        int position = 0;

        foreach (YamlNode entry in ((YamlSequenceNode)node).Children)
        {
            string entryWhere = $"{where}[{position}]";
            Dictionary<string, YamlNode> keys = KeysOf(AsMapping(entry, entryWhere), entryWhere);
            RejectUnknownKeys(keys, entryWhere, NodeOverrideKeys);

            int? index = TryInteger(keys, "index", $"{entryWhere}.index");
            if (index is null)
            {
                throw new ConfigException($"Missing required key 'index' under {entryWhere}; allowed keys are: {Allowed(NodeOverrideKeys)}.");
            }

            if (index.Value < 0)
            {
                throw new ConfigException(
                    $"{entryWhere}.index {index} is invalid for node index {index}: a per-node override names the position a host posture reports, counting from 0.");
            }

            long? mb = TryLong(keys, "mb", $"{entryWhere}.mb");
            if (mb is < 1)
            {
                throw new ConfigException($"{entryWhere}.mb {mb} is invalid for node index {index}: a per-node byte override must be 1 or greater.");
            }

            int? cores = TryInteger(keys, "cores", $"{entryWhere}.cores");
            if (cores is < 1)
            {
                throw new ConfigException($"{entryWhere}.cores {cores} is invalid for node index {index}: a per-node core override must be 1 or greater.");
            }

            if (mb is null && cores is null)
            {
                throw new ConfigException(
                    $"{entryWhere} states nothing about node index {index}: an override declares 'mb' and/or 'cores' — omit the entry to state no override there.");
            }

            if (!declaredIndexes.Add(index.Value))
            {
                throw new ConfigException($"{where} names node index {index} more than once (at position {position}): one entry overrides what one node position states.");
            }

            overrides.Add(new NodeCapacityOverride { Index = index.Value, Mb = mb, Cores = cores });
            position++;
        }

        return [.. overrides];
    }

    private static EngineConfig ParseEngine(Dictionary<string, YamlNode> top, IReadOnlyList<ModelConfig> models)
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

        // Step-time prefill budget: absent means the scheduler keeps forwarding chunks at the configured width, and a present
        // value gives that width from measured step cost instead. A rejected value fails naming the key and the offending
        // number exactly as the other sized keys do.
        int? prefillStepBudgetMs = TryInteger(keys, "prefill_step_budget_ms", "engine.prefill_step_budget_ms");
        if (prefillStepBudgetMs is < 1)
        {
            throw new ConfigException($"engine.prefill_step_budget_ms {prefillStepBudgetMs} is invalid: 'prefill_step_budget_ms' must be 1 or greater.");
        }

        // Prefill kernel selection: gemv (default — the batched row window this build runs), tiled, or gemm. All three
        // are accepted by the schema; whether one can run for the served file is the forward engine's answer at model
        // open. An unrecognized value fails naming the served model and the key with the accepted set, exactly as the
        // per-model choice keys do.
        string prefillKernel = TryText(keys, "prefill_kernel", "engine.prefill_kernel") ?? PrefillKernelNames.GemV;
        if (prefillKernel is not (PrefillKernelNames.GemV or PrefillKernelNames.Tiled or PrefillKernelNames.Gemm))
        {
            throw new ConfigException(
                $"engine.prefill_kernel '{prefillKernel}' is invalid for model '{models[0].Name}': 'prefill_kernel' accepts only '{PrefillKernelNames.GemV}', '{PrefillKernelNames.Tiled}', or '{PrefillKernelNames.Gemm}'.");
        }

        return new EngineConfig { Threads = threads, BlockSize = blockSize, MaxBatchTokens = maxBatchTokens, PrefillStepBudgetMs = prefillStepBudgetMs, PrefillKernel = prefillKernel };
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

        // Per-resource budgets parse before either spelling of the system-memory KV ceiling decides anything: a declared
        // 'budgets.kv.caps.dram_blocks' becomes canonical for what LAIR reserves and caps at that tier exactly as 'kv_blocks'
        // does today, so both spellings have to be in hand — and agree — before the reservation is sized. A legacy key alone
        // behaves exactly as it always did; budgets alone carry their own value; neither keeps today's default of 256.
        IReadOnlyList<BudgetChain>? budgets = ParseBudgets(keys, index, name);

        int? declaredKvBlocks = TryInteger(keys, "kv_blocks", $"models[{index}].kv_blocks");
        if (declaredKvBlocks is < 8)
        {
            throw new ConfigException($"models[{index}].kv_blocks {declaredKvBlocks} is invalid: 'kv_blocks' must be 8 or greater.");
        }

        long? budgetedKvCeiling = SystemMemoryKvCeiling(budgets);

        int kvBlocks;
        if (budgetedKvCeiling is not null && declaredKvBlocks is not null)
        {
            if (declaredKvBlocks.Value != budgetedKvCeiling.Value)
            {
                throw new ConfigException(
                    $"model '{name}' declares models[{index}].kv_blocks {declaredKvBlocks.Value} and models[{index}].budgets.kv.caps.dram_blocks {budgetedKvCeiling.Value}: " +
                    "both spell the same system-memory KV ceiling, so they must state the same number — keep one key or make them agree.");
            }

            kvBlocks = declaredKvBlocks.Value; // equal spellings: startup sizing stays exactly what it was today, and model open reports once that the legacy key is deprecated in favor of budgets.
        }
        else if (budgetedKvCeiling is not null)
        {
            kvBlocks = checked((int)budgetedKvCeiling.Value); // a budgeted ceiling alone sizes the reservation through the same path every other KV block count does.
        }
        else
        {
            kvBlocks = declaredKvBlocks ?? 256;
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

        // Per-model token budgets: absent means unbudgeted, and a present value must be 1 or greater. A rejected value
        // fails naming the model entry, key, and offending number exactly as the other sized keys do.
        int? maxReasoningTokens = TryInteger(keys, "max_reasoning_tokens", $"models[{index}].max_reasoning_tokens");
        if (maxReasoningTokens is < 1)
        {
            throw new ConfigException(
                $"models[{index}].max_reasoning_tokens {maxReasoningTokens} is invalid: 'max_reasoning_tokens' must be 1 or greater.");
        }

        int? maxResponseTokens = TryInteger(keys, "max_response_tokens", $"models[{index}].max_response_tokens");
        if (maxResponseTokens is < 1)
        {
            throw new ConfigException(
                $"models[{index}].max_response_tokens {maxResponseTokens} is invalid: 'max_response_tokens' must be 1 or greater.");
        }

        // Per-model admission concurrency: absent bounds nothing — every submission reaches admission as far as capacity allows,
        // exactly as before this key existed. A present value caps how many requests the model holds admitted at once and must
        // be 1 or greater; a rejected value fails naming the model entry, key, and offending number like the other sized keys do.
        int? maxConcurrentRequests = TryInteger(keys, "max_concurrent_requests", $"models[{index}].max_concurrent_requests");
        if (maxConcurrentRequests is < 1)
        {
            throw new ConfigException(
                $"models[{index}].max_concurrent_requests {maxConcurrentRequests} is invalid: 'max_concurrent_requests' must be 1 or greater.");
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
            MaxReasoningTokens = maxReasoningTokens,
            MaxResponseTokens = maxResponseTokens,
            MaxConcurrentRequests = maxConcurrentRequests,
            Budgets = budgets,
            LegacyKvBlocksDeclared = declaredKvBlocks is not null,
            Placement = ParsePlacement(keys, index, name),
        };
    }

    private static PlacementConfig ParsePlacement(Dictionary<string, YamlNode> modelKeys, int index, string modelName)
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
                $"models[{index}].placement.device '{device}' is invalid: the only accepted value for 'device' is 'cpu'. " +
                $"An accelerator device becomes placeable when a compute backend reports it in '{ModelRoute.CudaBackendSeam}'.");
        }

        string kvResidency = TryText(keys, "kv_residency", $"models[{index}].placement.kv_residency") ?? "dram";
        if (kvResidency != "dram")
        {
            throw new ConfigException(
                $"models[{index}].placement.kv_residency '{kvResidency}' is invalid: the only accepted value for 'kv_residency' is 'dram'. " +
                $"An accelerator residency becomes placeable when a compute backend reports that device in '{ModelRoute.CudaBackendSeam}'.");
        }

        string prefill = TryText(keys, "prefill", $"models[{index}].placement.prefill") ?? "cpu";
        if (prefill != "cpu")
        {
            throw new ConfigException(
                $"models[{index}].placement.prefill '{prefill}' is invalid: the only accepted value for 'prefill' is 'cpu'. " +
                $"An accelerator prefill stage becomes placeable when a compute backend reports that device in '{ModelRoute.CudaBackendSeam}'.");
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

        // Prefill lane posture: shared (default — prompt forwards ride the model's single execution backend exactly as
        // before this key existed) or dedicated (prompt forwards run on a second CPU execution backend capped to its own
        // share of the worker claim). An unrecognized value fails naming the served model, the key, and the accepted set.
        string prefillLane = TryText(keys, "prefill_lane", $"models[{index}].placement.prefill_lane") ?? PlacementPrefillLaneValues.Shared;
        if (prefillLane is not (PlacementPrefillLaneValues.Shared or PlacementPrefillLaneValues.Dedicated))
        {
            throw new ConfigException(
                $"models[{index}].placement.prefill_lane '{prefillLane}' is invalid for model '{modelName}': 'prefill_lane' accepts only '{PlacementPrefillLaneValues.Shared}' or '{PlacementPrefillLaneValues.Dedicated}'.");
        }

        // Core-set overcommit: absent means today's disjoint partition, and a present value opts this model's own claim into sharing cores
        // another model already claims. A non-boolean fails naming the key exactly as the other choice keys do; what the value costs a
        // running engine is decided when the engine opens, so the loader only reads the posture.
        bool coreOvercommit = TryBoolean(keys, "core_overcommit", $"models[{index}].placement.core_overcommit") ?? false;

        // NUMA spread posture: auto (default — placement chooses the node) or local (one node carries the model) or interleave
        // (a spread posture across the reported nodes). An unrecognized value fails naming the served model, the key, and the
        // accepted set; what the posture resolves to on this host's reported topology is decided when the engine opens.
        string numaSpread = TryText(keys, "numa_spread", $"models[{index}].placement.numa_spread") ?? NumaSpreadNames.Auto;
        if (!NumaSpreadNames.TryParse(numaSpread, out _))
        {
            throw new ConfigException(
                $"models[{index}].placement.numa_spread '{numaSpread}' is invalid for model '{modelName}': 'numa_spread' accepts {NumaSpreadNames.AcceptedVocabulary}.");
        }

        return new PlacementConfig
        {
            Device = device,
            KvResidency = kvResidency,
            Prefill = prefill,
            CoreSet = coreSet,
            ThreadQuota = threadQuota,
            PrefillLane = prefillLane,
            CoreOvercommit = coreOvercommit,
            NumaSpread = numaSpread,
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
        // document apply the same rule. A model that opts into core overcommit relaxes the rule for its own claim only:
        // it may share cores an earlier claim already holds, while every other pair keeps today's refusal naming both models.
        Dictionary<int, string> owners = [];
        for (int index = 0; index < models.Count; index++)
        {
            ModelConfig model = models[index];
            foreach (int core in model.Placement.CoreSet)
            {
                if (!owners.TryAdd(core, model.Name))
                {
                    if (model.Placement.CoreOvercommit)
                    {
                        continue; // this claim opted into sharing held cores: the owner keeps its seat and nothing is refused.
                    }

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

    private static void RejectUnknownKeys(Dictionary<string, YamlNode> keys, string where, IReadOnlyList<string> allowed)
    {
        foreach (string key in keys.Keys)
        {
            if (!IsKnown(key, allowed))
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

    // The same read as TryInteger for the byte totals a tiers block declares: capacity ceilings count in whole megabytes and can name a
    // number larger than an int carries, so these keys parse wide while still refusing anything that is not an integer.
    private static long? TryLong(Dictionary<string, YamlNode> keys, string key, string qualifiedKey)
    {
        if (!keys.TryGetValue(key, out YamlNode? node))
        {
            return null;
        }

        if (node.NodeType != YamlNodeType.Scalar ||
            ((YamlScalarNode)node).Value is not string text ||
            !long.TryParse(text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out long value))
        {
            throw new ConfigException($"{qualifiedKey} must be an integer value.");
        }

        return value;
    }

    private static bool? TryBoolean(Dictionary<string, YamlNode> keys, string key, string qualifiedKey)    {
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

    /// <summary>Parses one model entry's budgets block: every declared resource names its own ordered tier chain and the ceilings it declares along that chain, each refusal naming the qualified config path, the offending value, and the accepted vocabulary exactly as the other choice keys do. Chain shape — a known tier at each position, priority order, no repeats — is enforced here in config-path wording; ceiling ranges are checked against what each key sizes: a system-memory KV ceiling carries <c>kv_blocks</c>' floor of 8 because it reserves physical blocks exactly as the legacy key does, every other ceiling must be positive where it states one, and the read-side origin store takes no ceiling at all because that storage already holds the complete set. Nothing here consults capacity or opened pools — which claims a host can carry is resolution's answer when the engine opens.</summary>
    private static IReadOnlyList<BudgetChain>? ParseBudgets(Dictionary<string, YamlNode> modelKeys, int index, string modelName)
    {
        if (!modelKeys.TryGetValue("budgets", out YamlNode? node))
        {
            return null;
        }

        string where = $"models[{index}].budgets";
        Dictionary<string, YamlNode> resources = KeysOf(AsMapping(node, where), where);
        RejectUnknownKeys(resources, where, BudgetResourceKeys);

        List<BudgetChain> chains = [];
        foreach (string resourceKey in BudgetResourceKeys)
        {
            if (!resources.TryGetValue(resourceKey, out YamlNode? resourceNode))
            {
                continue; // a resource this entry says nothing about declares no chain at all.
            }

            chains.Add(ParseBudgetChain(index, modelName, resourceKey, resourceNode));
        }

        return [.. chains];
    }

    private static BudgetChain ParseBudgetChain(int index, string modelName, string resourceKey, YamlNode node)
    {
        if (!BudgetResourceNames.TryParse(resourceKey, out BudgetResource resource))
        {
            throw new ConfigException(
                $"models[{index}].budgets.{resourceKey} is invalid for model '{modelName}': budgets declare only {Vocabulary(BudgetResourceKeys)}.");
        }

        string where = $"models[{index}].budgets.{resourceKey}";
        Dictionary<string, YamlNode> keys = KeysOf(AsMapping(node, where), where);
        RejectUnknownKeys(keys, where, ["order", "caps"]);

        if (!keys.TryGetValue("order", out YamlNode? orderNode))
        {
            throw new ConfigException($"Missing required key 'order' under {where}; allowed keys are: {Allowed(["order", "caps"])}.");
        }

        if (orderNode.NodeType != YamlNodeType.Sequence)
        {
            throw new ConfigException($"{where}.order must be a list of tier names ({Vocabulary(BudgetTierNames.DeclaredNames)}).");
        }

        List<BudgetTier> order = [];
        foreach (YamlNode item in ((YamlSequenceNode)orderNode).Children)
        {
            if (item.NodeType != YamlNodeType.Scalar || ((YamlScalarNode)item).Value is not string tierName)
            {
                throw new ConfigException($"{where}.order must be a list of tier names ({Vocabulary(BudgetTierNames.DeclaredNames)}).");
            }

            if (!BudgetTierNames.TryParse(tierName, out BudgetTier tier))
            {
                throw new ConfigException(
                    $"{where}.order '{tierName}' is invalid for model '{modelName}': 'order' accepts only {Vocabulary(BudgetTierNames.DeclaredNames)}, highest-priority residency first.");
            }

            order.Add(tier);
        }

        ValidateChainShape(modelName, where, resourceKey, order);

        Dictionary<BudgetTier, long> caps = [];
        if (keys.TryGetValue("caps", out YamlNode? capsNode))
        {
            string capsWhere = $"{where}.caps";
            foreach (KeyValuePair<string, YamlNode> capEntry in KeysOf(AsMapping(capsNode, capsWhere), capsWhere))
            {
                string capKey = capEntry.Key;
                YamlNode value = capEntry.Value;
                if (!BudgetCapKeys.TryParse(capKey, resource, out BudgetTier tier))
                {
                    throw new ConfigException(
                        $"{capsWhere}.{capKey} is invalid for model '{modelName}': '{resourceKey}' ceilings are named {Vocabulary(BudgetCapKeys.For(resource))}, each binding a tier its own 'order' names.");
                }

                if (value.NodeType != YamlNodeType.Scalar || ((YamlScalarNode)value).Value is not string text)
                {
                    throw new ConfigException($"{capsWhere}.{capKey} must be an integer value.");
                }

                long declared;
                bool parsed = resource == BudgetResource.Kv && tier == BudgetTier.Dram
                    ? long.TryParse(text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out declared) && declared <= int.MaxValue
                    : long.TryParse(text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out declared);

                if (!parsed)
                {
                    throw new ConfigException(
                        $"{capsWhere}.{capKey} '{text}' is invalid for model '{modelName}': a system-memory KV ceiling sizes the physical reservation and must be an integer no larger than {int.MaxValue}.");
                }

                ValidateCap(modelName, capsWhere, capKey, resource, tier, declared, order);
                caps[tier] = declared;
            }
        }

        return new BudgetChain { Resource = resource, Order = [.. order], Caps = caps };
    }

    private static void ValidateChainShape(string modelName, string where, string resourceKey, List<BudgetTier> order)
    {
        if (order.Count == 0)
        {
            throw new ConfigException(
                $"{where}.order is empty for model '{modelName}': a chain must name at least the tier its copies start on — omit '{resourceKey}' entirely to declare nothing about it.");
        }

        HashSet<BudgetTier> seen = [];
        int previousPriority = -1;
        foreach (BudgetTier tier in order)
        {
            if (!seen.Add(tier))
            {
                throw new ConfigException(
                    $"{where}.order {DescribeTiers(order)} is invalid for model '{modelName}': a chain names each tier exactly once.");
            }

            int priority = RankOf(tier);
            if (priority <= previousPriority)
            {
                throw new ConfigException(
                    $"{where}.order {DescribeTiers(order)} is out of order for model '{modelName}': tiers must run from the highest-priority residency to " +
                    $"the lowest ({string.Join(" then ", BudgetTierNames.DeclaredNames.Select(name => $"'{name}'"))}), skipping any tier this resource does not use.");
            }

            previousPriority = priority;
        }
    }

    private static void ValidateCap(string modelName, string capsWhere, string capKey, BudgetResource resource, BudgetTier tier, long declared, List<BudgetTier> order)
    {
        if (!order.Contains(tier))
        {
            throw new ConfigException(
                $"{capsWhere}.{capKey} is invalid for model '{modelName}': a ceiling binds only tiers its own 'order' names — declare '{BudgetTierNames.Of(tier)}' there to cap it.");
        }

        if (resource == BudgetResource.Weights && tier == BudgetTier.Nvme)
        {
            throw new ConfigException(
                $"{capsWhere}.{capKey} is invalid for model '{modelName}': weights read their complete set from disk storage already, so no model-level ceiling describes that position.");
        }

        if (resource == BudgetResource.Kv && tier == BudgetTier.Dram)
        {
            if (declared < 8)
            {
                throw new ConfigException(
                    $"{capsWhere}.{capKey} {declared} is invalid for model '{modelName}': a system-memory KV ceiling reserves physical blocks exactly as 'kv_blocks' does and must be 8 or greater.");
            }

            return;
        }

        if (declared < 1)
        {
            throw new ConfigException(
                $"{capsWhere}.{capKey} {declared} is invalid for model '{modelName}': a ceiling must be 1 or greater — omit the key to declare no ceiling at that tier.");
        }
    }

    /// <summary>The declared system-memory KV ceiling of these chains, if any: what both spellings of this model's physical reservation and live cap have to agree on.</summary>
    private static long? SystemMemoryKvCeiling(IReadOnlyList<BudgetChain>? budgets)
    {
        if (budgets is null)
        {
            return null;
        }

        foreach (BudgetChain chain in budgets)
        {
            if (chain.Resource == BudgetResource.Kv && chain.Caps.TryGetValue(BudgetTier.Dram, out long declared))
            {
                return declared;
            }
        }

        return null;
    }

    private static int RankOf(BudgetTier tier) => tier switch
    {
        BudgetTier.Dram => 0,
        BudgetTier.Vram => 1,
        _ => 2,
    };

    private static string DescribeTiers(IReadOnlyList<BudgetTier> order) =>
        string.Join(", ", order.Select(tier => $"'{BudgetTierNames.Of(tier)}'"));

    private static bool IsKnown(string key, IReadOnlyList<string> allowed)
    {
        foreach (string candidate in allowed)
        {
            if (string.Equals(candidate, key, StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }

    private static string Vocabulary(IReadOnlyList<string> names) =>
        string.Join(", ", names.Select(name => $"'{name}'"));

    private static string Allowed(IReadOnlyList<string> keys) => string.Join(", ", keys);
}
