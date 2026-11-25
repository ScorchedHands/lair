using System.Globalization;
using Lair.Backends.Cpu;
using Lair.Kv;
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

    private static readonly string[] TopLevelKeys = ["server", "engine", "models", "ui", "host"];
    private static readonly string[] ServerKeys = ["bind", "port"];
    private static readonly string[] EngineKeys = ["threads", "block_size", "max_batch_tokens", "prefill_step_budget_ms", "prefill_kernel", "cpu"];
    private static readonly string[] CpuKeys = ["isa", "tuning"];

    // The CPU kernel tuning block: every key is an optional pin over the posture-resolved default. Each accepted span mirrors the
    // backend's own resolution bounds (CpuKernelTuning), so a value that passes here always pins cleanly there too.
    private static readonly string[] CpuTuningKeys = ["row_block", "parallel_row_threshold", "max_scratch_cols", "chain_group"];
    private static readonly string[] UiKeys = ["recent_requests"];
    private static readonly string[] ModelKeys = ["name", "path", "type", "max_context_tokens", "kv_blocks", "kv", "placement", "force_fenced_tools", "tool_parser", "reasoning", "max_reasoning_tokens", "max_response_tokens", "max_concurrent_requests", "max_queued_requests", "queue_timeout_ms", "compute_share", "pooling", "normalize", "budgets", "source"];
    private static readonly string[] PlacementKeys = ["device", "kv_residency", "prefill", "core_set", "thread_quota", "prefill_lane", "core_overcommit", "numa_spread", "gpu_layers"];

    /// <summary>The keys one model entry's KV block may declare.</summary>
    private static readonly string[] KvKeys = ["quant"];

    /// <summary>The per-side keys one KV quant nesting may declare.</summary>
    private static readonly string[] KvQuantKeys = ["k", "v"];

    /// <summary>The keys one model entry's source block may declare, beside exactly one of the two file spellings.</summary>
    private static readonly string[] SourceKeys = ["repo", "revision", "file", "files"];

    /// <summary>The keys one source file mapping may declare beside the path it names.</summary>
    private static readonly string[] SourceFileKeys = ["path", "sha256", "size"];

    /// <summary>The resident capacity keys a top-level host block may declare, in the order a refusal reports them.</summary>
    private static readonly IReadOnlyList<string> HostPostureKeys = ["dram_mb", "vram_mb", "cpu_cores", "disk_path", "numa"];

    /// <summary>The keys one posture's numa nesting may declare.</summary>
    private static readonly IReadOnlyList<string> NumaNestKeys = ["nodes"];

    /// <summary>The keys one per-node override entry may declare beside the node position it names.</summary>
    private static readonly IReadOnlyList<string> NodeOverrideKeys = ["index", "mb", "cores"];

    /// <summary>The resource keys a model's budgets block may declare, in the order a refusal reports them.</summary>
    private static readonly IReadOnlyList<string> BudgetResourceKeys = [BudgetResourceNames.Weights, BudgetResourceNames.Kv, BudgetResourceNames.PrefillBuffers, BudgetResourceNames.Ple];

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
            Host = ParseHost(top),
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
    /// Parses the top-level host block: flat resident capacity keys for system and accelerator memory, a disk path declared as presence
    /// and path only, and an optional numa nesting carrying per-node overrides keyed by node position. Shape is validated here — every
    /// known key that expects a number must state one, positive where it states anything, and a refusal names the qualified key with the
    /// offending value (naming the node index too for a per-node override). Whether a declared system-memory total agrees with what a host
    /// reports is answered when an engine opens, warn-and-accept: this loader never consults topology, and an accelerator total is declared
    /// posture alone because no driver-detected budget is consulted. A disk byte ceiling is not accepted here because that position already
    /// holds the complete set.
    /// </summary>
    private static HostCapacityConfig? ParseHost(Dictionary<string, YamlNode> top)
    {
        if (!top.TryGetValue("host", out YamlNode? node))
        {
            return null; // no host block declares nothing: today's posture states no public totals.
        }

        const string where = "'host'";
        Dictionary<string, YamlNode> keys = KeysOf(AsMapping(node, where), where);
        RejectUnknownKeys(keys, where, HostPostureKeys);

        long? dramMb = TryLong(keys, "dram_mb", "host.dram_mb");
        if (dramMb is < 1)
        {
            throw new ConfigException($"host.dram_mb {dramMb} is invalid: 'dram_mb' must be 1 or greater — omit the key to declare no byte total.");
        }

        // An accelerator total is the ceiling a device-resident weight claim seats against, stated by the operator rather than read
        // off any device: what a host backs without stating a size keeps the posture that states no public total.
        long? vramMb = TryLong(keys, "vram_mb", "host.vram_mb");
        if (vramMb is < 1)
        {
            throw new ConfigException($"host.vram_mb {vramMb} is invalid: 'vram_mb' must be 1 or greater — omit the key to declare no byte total.");
        }

        int? cpuCores = TryInteger(keys, "cpu_cores", "host.cpu_cores");
        if (cpuCores is < 1)
        {
            throw new ConfigException($"host.cpu_cores {cpuCores} is invalid: 'cpu_cores' must be 1 or greater — omit the key to declare no core total.");
        }

        // A disk path declares only that this host reads from one. No ceiling belongs at that position, so nothing here reads a number beside it.
        string? diskPath = TryText(keys, "disk_path", "host.disk_path");
        if (diskPath is not null && string.IsNullOrWhiteSpace(diskPath))
        {
            throw new ConfigException("host.disk_path must name a non-empty path this host reads from — omit the key to say nothing about disk storage.");
        }

        IReadOnlyList<NodeCapacityOverride> nodes = ParseNumaOverrides(keys);

        if (dramMb is null && vramMb is null && cpuCores is null && diskPath is null && nodes.Count == 0)
        {
            throw new ConfigException(
                $"{where} declares nothing: a host block states at least one of {Vocabulary(HostPostureKeys)} — omit 'host' entirely to declare no capacity.");
        }

        return new HostCapacityConfig { DramMb = dramMb, VramMb = vramMb, CpuCores = cpuCores, DiskPath = diskPath, Nodes = nodes };
    }

    // Per-node overrides sit in the host block's own numa nesting: each entry names the node position it disagrees with and states its own bytes and/or cores. A malformed value
    // fails naming key, value, and the index that entry names; a duplicate or out-of-range index cannot name one node position at all.
    private static IReadOnlyList<NodeCapacityOverride> ParseNumaOverrides(Dictionary<string, YamlNode> hostKeys)
    {
        if (!hostKeys.TryGetValue("numa", out YamlNode? numa))
        {
            return []; // no nesting: the posture states nothing about individual nodes.
        }

        const string nestWhere = "'host.numa'";
        Dictionary<string, YamlNode> nestKeys = KeysOf(AsMapping(numa, nestWhere), nestWhere);
        RejectUnknownKeys(nestKeys, nestWhere, NumaNestKeys);

        if (!nestKeys.TryGetValue("nodes", out YamlNode? node))
        {
            return []; // a nesting that names no list declares no per-node override.
        }

        const string where = "'host.numa.nodes'";
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

        EngineCpuConfig cpu = ParseEngineCpu(keys, models[0].Name);

        return new EngineConfig { Threads = threads, BlockSize = blockSize, MaxBatchTokens = maxBatchTokens, PrefillStepBudgetMs = prefillStepBudgetMs, PrefillKernel = prefillKernel, Cpu = cpu };
    }

    // CPU ISA selection: auto (default — the highest integer-dot path this host's CPU reports), avx2, or avx-vnni. All three
    // spellings are accepted by the schema; whether a forced one can run is what the backend answers when the engine opens. An
    // unrecognized value fails naming the served model and the key with the accepted set, exactly as the other choice keys do.
    private static EngineCpuConfig ParseEngineCpu(Dictionary<string, YamlNode> engineKeys, string modelName)
    {
        const string where = "'engine.cpu'";
        if (!engineKeys.TryGetValue("cpu", out YamlNode? node))
        {
            return new EngineCpuConfig();
        }

        Dictionary<string, YamlNode> keys = KeysOf(AsMapping(node, where), where);
        RejectUnknownKeys(keys, where, CpuKeys);

        string isa = TryText(keys, "isa", "engine.cpu.isa") ?? CpuIsaNames.Auto;
        if (!CpuIsaNames.TryParse(isa, out _))
        {
            throw new ConfigException(
                $"engine.cpu.isa '{isa}' is invalid for model '{modelName}': 'cpu.isa' accepts {CpuIsaNames.AcceptedVocabulary}.");
        }

        EngineCpuTuningConfig tuning = ParseEngineCpuTuning(keys);

        return new EngineCpuConfig { Isa = isa, Tuning = tuning };
    }

    // CPU kernel tuning pins: every key is optional and validated against the same accepted span the backend's own resolution applies.
    // An out-of-span value fails naming the qualified key and its bounds instead of clamping a stated pin quietly; an absent block or key
    // declares nothing — that knob stays posture-resolved (row block scaled by effective worker count, threshold tracking row block and
    // workers, chain grouping at the vector register budget, scratch width sized for cache residency).
    private static EngineCpuTuningConfig ParseEngineCpuTuning(Dictionary<string, YamlNode> cpuKeys)
    {
        const string where = "'engine.cpu.tuning'";
        if (!cpuKeys.TryGetValue("tuning", out YamlNode? node))
        {
            return new EngineCpuTuningConfig();
        }

        Dictionary<string, YamlNode> keys = KeysOf(AsMapping(node, where), where);
        RejectUnknownKeys(keys, where, CpuTuningKeys);

        int? rowBlock = TryInteger(keys, "row_block", "engine.cpu.tuning.row_block");
        if (rowBlock is not null and (< 1 or > 64))
        {
            throw new ConfigException($"engine.cpu.tuning.row_block {rowBlock} is invalid: 'cpu.tuning.row_block' must be between 1 and 64.");
        }

        int? parallelRowThreshold = TryInteger(keys, "parallel_row_threshold", "engine.cpu.tuning.parallel_row_threshold");
        if (parallelRowThreshold is < 1)
        {
            throw new ConfigException($"engine.cpu.tuning.parallel_row_threshold {parallelRowThreshold} is invalid: 'cpu.tuning.parallel_row_threshold' must be 1 or greater.");
        }

        int? chainGroup = TryInteger(keys, "chain_group", "engine.cpu.tuning.chain_group");
        if (chainGroup is not null and (< 1 or > 8))
        {
            throw new ConfigException($"engine.cpu.tuning.chain_group {chainGroup} is invalid: 'cpu.tuning.chain_group' must be between 1 and 8 — a chain group wider than eight spills accumulators past the vector posture's register budget.");
        }

        int? maxScratchCols = TryInteger(keys, "max_scratch_cols", "engine.cpu.tuning.max_scratch_cols");
        if (maxScratchCols is not null and (< 512 or > 65_536))
        {
            throw new ConfigException($"engine.cpu.tuning.max_scratch_cols {maxScratchCols} is invalid: 'cpu.tuning.max_scratch_cols' must be between 512 and 65536 — wider staging no longer stays cache-resident beside the packed row it replaces.");
        }

        return new EngineCpuTuningConfig { RowBlock = rowBlock, ParallelRowThreshold = parallelRowThreshold, ChainGroup = chainGroup, MaxScratchCols = maxScratchCols };
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
        if (entries.Count == 0)
        {
            throw new ConfigException($"{where} must declare at least one model entry.");
        }

        // Every declared entry parses through the same per-model validation; names are what requests and reload diffs route by, so two entries may not share one.
        List<ModelConfig> models = [];
        for (int index = 0; index < entries.Count; index++)
        {
            ModelConfig parsed = ParseModel(entries[index], index);

            int prior = models.FindIndex(held => string.Equals(held.Name, parsed.Name, StringComparison.Ordinal));
            if (prior >= 0)
            {
                throw new ConfigException(
                    $"models[{index}].name '{parsed.Name}' duplicates models[{prior}].name: every served model is named distinctly — give each entry its own 'name'.");
            }

            models.Add(parsed);
        }

        return models;
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
        if (!File.Exists(path) && !keys.ContainsKey("source"))
        {
            throw new ConfigException($"models[{index}].path '{path}' does not exist on disk.");
        }

        // Serving posture: chat (default) or embed. A chat entry behaves exactly as it always did; an embed entry opens the model's
        // encode-only serving posture, and any other spelling fails naming the path, the model, the value, and both accepted values.
        string type = TryText(keys, "type", $"models[{index}].type") ?? ModelTypeValues.Chat;
        if (type is not (ModelTypeValues.Chat or ModelTypeValues.Embed))
        {
            throw new ConfigException(
                $"models[{index}].type '{type}' is invalid for model '{name}': 'type' accepts only '{ModelTypeValues.Chat}' or '{ModelTypeValues.Embed}'.");
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

        // Per-model request queue: absent means today's fail-fast posture, and an explicit 0 states that same posture outright — the
        // documented default is zero-disabled. A positive value bounds how many submissions may wait for a concurrency slot before
        // further ones are refused with the overloaded shape. Only a negative number is rejected here, naming the model entry, key,
        // and offending value exactly as the other sized keys do.
        int? maxQueuedRequests = TryInteger(keys, "max_queued_requests", $"models[{index}].max_queued_requests");
        if (maxQueuedRequests is < 0)
        {
            throw new ConfigException(
                $"models[{index}].max_queued_requests {maxQueuedRequests} is invalid: 'max_queued_requests' must be 0 or greater, where 0 keeps the immediate refusal posture.");
        }

        // How long a waiting submission may block for its slot. A present value must be 1 millisecond or greater; absent resolves to
        // the documented default when queueing is enabled. Declared while queueing stands disabled it validates and reports like any
        // other key, but no submission waits, so no refusal posture changes.
        int? queueTimeoutMs = TryInteger(keys, "queue_timeout_ms", $"models[{index}].queue_timeout_ms");
        if (queueTimeoutMs is < 1)
        {
            throw new ConfigException(
                $"models[{index}].queue_timeout_ms {queueTimeoutMs} is invalid: 'queue_timeout_ms' must be 1 or greater.");
        }

        // Per-model device fair-share: absent declares no share — step composition bounds this model by its segment batch caps and quota
        // headroom alone, exactly as before this key existed. A present value is the percent of one host device co-scheduling window this
        // model's launched device work may hold and must be an integer from 1 to 100; a rejected value fails naming the model entry, the
        // model, the key, the offending number, and the accepted range like the other sized keys do. What a model without an
        // accelerator-backed segment declares here is validated the same way and bounds nothing until a device-backed segment exists —
        // the posture every declared posture this build cannot yet carry keeps.
        int? computeShare = TryInteger(keys, "compute_share", $"models[{index}].compute_share");
        if (computeShare is < 1 or > 100)
        {
            throw new ConfigException(
                $"models[{index}].compute_share {computeShare} is invalid for model '{name}': 'compute_share' must be an integer from 1 to 100 — " +
                "the percent of one device co-scheduling window this model may hold its launched device work in — or absent to declare no share.");
        }

        // Sentence-vector posture, read by embedding requests only and absent-by-default in both halves. An unrecognized pooling choice fails
        // naming the path, model, value, and accepted set exactly as the tool-parser and reasoning choices do; a present normalize must be a
        // boolean like every other flag key here. A chat entry may declare either key without effect: what an embed request answers with follows them.
        string pooling = TryText(keys, "pooling", $"models[{index}].pooling") ?? PoolingModeValues.Mean;
        if (pooling is not (PoolingModeValues.Mean or PoolingModeValues.Cls))
        {
            throw new ConfigException(
                $"models[{index}].pooling '{pooling}' is invalid for model '{name}': 'pooling' accepts only '{PoolingModeValues.Mean}' or '{PoolingModeValues.Cls}'.");
        }

        bool normalize = TryBoolean(keys, "normalize", $"models[{index}].normalize") ?? false;

        KvFormatPair kvQuant = ParseKvQuant(keys, index);

        return new ModelConfig
        {
            Name = name,
            Path = path,
            Type = type,
            MaxContextTokens = maxContextTokens,
            KvBlocks = kvBlocks,
            KvQuant = kvQuant,
            ForceFencedTools = forceFencedTools,
            ToolParser = toolParser,
            Reasoning = reasoning,
            MaxReasoningTokens = maxReasoningTokens,
            MaxResponseTokens = maxResponseTokens,
            MaxConcurrentRequests = maxConcurrentRequests,
            MaxQueuedRequests = maxQueuedRequests,
            QueueTimeoutMs = queueTimeoutMs,
            ComputeShare = computeShare,
            Pooling = pooling,
            Normalize = normalize,
            Budgets = budgets,
            LegacyKvBlocksDeclared = declaredKvBlocks is not null,
            Source = ParseSource(keys, index, name),
            Placement = ParsePlacement(keys, index, name),
        };
    }

    /// <summary>Parses one model entry's KV format axis: absent means today's FP32/FP32 pair, each declared side resolves to a canonical lowercase name, and the dropped q4_0/q4_0 combo refuses by naming both qualified paths rather than falling back.</summary>
    private static KvFormatPair ParseKvQuant(Dictionary<string, YamlNode> modelKeys, int index)
    {
        string where = $"models[{index}].kv";
        if (!modelKeys.TryGetValue("kv", out YamlNode? node))
        {
            return KvFormatPair.Default;
        }

        Dictionary<string, YamlNode> kvKeys = KeysOf(AsMapping(node, where), where);
        RejectUnknownKeys(kvKeys, where, KvKeys);

        if (!kvKeys.TryGetValue("quant", out YamlNode? quantNode))
        {
            return KvFormatPair.Default;
        }

        string quantWhere = $"{where}.quant";
        Dictionary<string, YamlNode> quantKeys = KeysOf(AsMapping(quantNode, quantWhere), quantWhere);
        RejectUnknownKeys(quantKeys, quantWhere, KvQuantKeys);

        string? declaredK = TryText(quantKeys, "k", $"{quantWhere}.k");
        string? declaredV = TryText(quantKeys, "v", $"{quantWhere}.v");

        KvSideFormat k = KvSideFormat.F32;
        KvSideFormat v = KvSideFormat.F32;

        if (declaredK is not null && !KvSideFormatNames.TryResolve(declaredK, out k))
        {
            throw new ConfigException(
                $"{quantWhere}.k '{declaredK}' is invalid: 'k' accepts {KvSideFormatNames.AcceptedVocabulary}.");
        }

        if (declaredV is not null && !KvSideFormatNames.TryResolve(declaredV, out v))
        {
            throw new ConfigException(
                $"{quantWhere}.v '{declaredV}' is invalid: 'v' accepts {KvSideFormatNames.AcceptedVocabulary}.");
        }

        KvFormatPair pair = new(k, v);
        if (pair.IsDroppedQ4Pair)
        {
            throw new ConfigException(
                $"{quantWhere}.k declares '{KvSideFormatNames.Q4_0}' and {quantWhere}.v declares '{KvSideFormatNames.Q4_0}': q4_0/q4_0 is the dropped KV pair, and LAIR has no quantized KV storage or attention execution path for it.");
        }

        return pair;
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

        string device = TryText(keys, "device", $"models[{index}].placement.device") ?? "auto";
        if (device is not ("auto" or "cpu"))
        {
            throw new ConfigException(
                $"models[{index}].placement.device '{device}' is invalid: 'device' accepts only 'auto' or 'cpu'. " +
                $"An accelerator device becomes placeable when a compute backend reports it in '{ModelRoute.CudaBackendSeam}'.");
        }

        // KV residency is its own placement axis beside the device keys: an absent key keeps today's host-resident posture,
        // and every declared mode is vocabulary this loader accepts without consulting what any opened backend can back —
        // whether a device-backed block pool can be seated is what route resolution answers when the engine opens. A value
        // outside the declared set fails naming the served model, the key, and the accepted set, exactly as the other
        // choice keys here do.
        string kvResidency = TryText(keys, "kv_residency", $"models[{index}].placement.kv_residency") ?? BlockResidencyModeNames.Dram;
        if (!BlockResidencyModeNames.TryResolvePlacementValue(kvResidency, out _))
        {
            throw new ConfigException(
                $"models[{index}].placement.kv_residency '{kvResidency}' is invalid for model '{modelName}': 'kv_residency' accepts {BlockResidencyModeNames.AcceptedVocabulary}.");
        }

        string prefill = TryText(keys, "prefill", $"models[{index}].placement.prefill") ?? "auto";
        if (prefill is not ("auto" or "cpu"))
        {
            throw new ConfigException(
                $"models[{index}].placement.prefill '{prefill}' is invalid: 'prefill' accepts only 'auto' or 'cpu'. " +
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

        // Layer split: absent means today's CPU-only posture, and a present value names how many transformer layers execution
        // opens on device 'cuda', counted from layer 0. A negative count says nothing about where a split lands, so it fails
        // naming the model, key, and value like every other out-of-range integer here; what the split actually opens is decided
        // when the engine opens — including refusing an open that names more layers than this file carries.
        int gpuLayers = TryInteger(keys, "gpu_layers", $"models[{index}].placement.gpu_layers") ?? 0;
        if (gpuLayers < 0)
        {
            throw new ConfigException(
                $"models[{index}].placement.gpu_layers {gpuLayers} is invalid for model '{modelName}': 'gpu_layers' must be a non-negative integer — " +
                "the count of this model's transformer layers execution opens on device 'cuda', counted from layer 0.");
        }

        return new PlacementConfig
        {
            Device = device is "auto" ? "cpu" : device,
            KvResidency = kvResidency,
            Prefill = prefill is "auto" ? "cpu" : prefill,
            CoreSet = coreSet,
            ThreadQuota = threadQuota,
            PrefillLane = prefillLane,
            CoreOvercommit = coreOvercommit,
            NumaSpread = numaSpread,
            GpuLayers = gpuLayers,
        };
    }

    /// <summary>
    /// Parses one model entry's source block: the published repo, the revision pinned as declared, and exactly one of
    /// <c>file</c> or <c>files</c> naming the model files the pack carries. Shape is validated here — every refusal names
    /// the qualified config path and the offending value exactly as the other blocks do — and an absent block preserves
    /// the local-path posture: the declared path is served with no provenance at all. A declared revision is stored
    /// verbatim; a branch name becomes a commit sha only when fetch resolves it into the manifest.
    /// </summary>
    private static ModelSourceConfig? ParseSource(Dictionary<string, YamlNode> modelKeys, int index, string modelName)
    {
        if (!modelKeys.TryGetValue("source", out YamlNode? node))
        {
            return null; // no source block: today's posture serves the declared local path exactly as before.
        }

        string where = $"models[{index}].source";
        Dictionary<string, YamlNode> keys = KeysOf(AsMapping(node, where), where);
        RejectUnknownKeys(keys, where, SourceKeys);

        string? repo = TryText(keys, "repo", $"{where}.repo");
        if (repo is null)
        {
            throw new ConfigException($"Missing required key 'repo' under {where}; allowed keys are: {Allowed(SourceKeys)}.");
        }

        if (string.IsNullOrWhiteSpace(repo))
        {
            throw new ConfigException($"{where}.repo must be a non-empty Hugging Face repo slug.");
        }

        // The revision is stored as declared: a commit sha is already the pin, and a branch name becomes one when fetch resolves it.
        string? revision = TryText(keys, "revision", $"{where}.revision");
        if (revision is null)
        {
            throw new ConfigException($"Missing required key 'revision' under {where}; allowed keys are: {Allowed(SourceKeys)}.");
        }

        if (string.IsNullOrWhiteSpace(revision))
        {
            throw new ConfigException($"{where}.revision must name a commit sha or branch to pin.");
        }

        if (keys.ContainsKey("file") && keys.ContainsKey("files"))
        {
            throw new ConfigException($"{where} declares both 'file' and 'files': the source block states exactly one of them.");
        }

        List<SourceFileConfig> files;
        if (keys.TryGetValue("file", out YamlNode? fileNode))
        {
            files = [ParseSourceFile(fileNode, $"{where}.file", index, modelName)];
        }
        else if (keys.TryGetValue("files", out YamlNode? filesNode))
        {
            files = ParseSourceFiles(filesNode, where, index, modelName);
        }
        else
        {
            throw new ConfigException($"{where} declares no model file: state exactly one of 'file' or 'files' — omit 'source' entirely to serve the declared local path.");
        }

        return new ModelSourceConfig { Repo = repo, Revision = revision, Files = [.. files] };
    }

    // One declared file entry: a plain path/name string, or a mapping stating the path beside its pinned hash and size.
    // A malformed value fails naming the qualified entry path and the offending value; an entry that is neither a string
    // nor a mapping cannot name a file at all.
    private static SourceFileConfig ParseSourceFile(YamlNode entry, string where, int index, string modelName)
    {
        if (entry.NodeType == YamlNodeType.Scalar)
        {
            if (((YamlScalarNode)entry).Value is not string barePath || string.IsNullOrWhiteSpace(barePath))
            {
                throw new ConfigException($"{where} must be a non-empty model file path/name.");
            }

            return new SourceFileConfig { Path = barePath };
        }

        Dictionary<string, YamlNode> keys = KeysOf(AsMapping(entry, where), where);
        RejectUnknownKeys(keys, where, SourceFileKeys);

        string? declaredPath = TryText(keys, "path", $"{where}.path");
        if (declaredPath is null)
        {
            throw new ConfigException($"Missing required key 'path' under {where}; allowed keys are: {Allowed(SourceFileKeys)}.");
        }

        if (string.IsNullOrWhiteSpace(declaredPath))
        {
            throw new ConfigException($"{where}.path must be a non-empty model file path/name.");
        }

        string? sha256 = TryText(keys, "sha256", $"{where}.sha256");
        if (sha256 is not null && !IsSha256Hex(sha256))
        {
            throw new ConfigException(
                $"{where}.sha256 '{sha256}' is invalid for model '{modelName}': 'sha256' must be 64 lowercase hex characters.");
        }

        long? size = TryLong(keys, "size", $"{where}.size");
        if (size is < 0)
        {
            throw new ConfigException(
                $"{where}.size {size} is invalid for model '{modelName}': 'size' must be a non-negative integer — omit the key to declare no size.");
        }

        return new SourceFileConfig { Path = declaredPath, Sha256 = sha256, Size = size };
    }

    private static List<SourceFileConfig> ParseSourceFiles(YamlNode node, string where, int index, string modelName)
    {
        string listWhere = $"{where}.files";
        if (node.NodeType != YamlNodeType.Sequence)
        {
            throw new ConfigException($"{listWhere} must be a list of model file entries, each a path/name string or a mapping stating 'path' with optional 'sha256' and 'size'.");
        }

        List<YamlNode> entries = [.. ((YamlSequenceNode)node).Children];
        if (entries.Count == 0)
        {
            throw new ConfigException(
                $"{listWhere} is empty for model '{modelName}': a source block names at least one model file — omit 'source' entirely to serve the declared local path.");
        }

        List<SourceFileConfig> files = [];
        int position = 0;
        foreach (YamlNode entry in entries)
        {
            files.Add(ParseSourceFile(entry, $"{listWhere}[{position}]", index, modelName));
            position++;
        }

        return files;
    }

    private static bool IsSha256Hex(string value)
    {
        if (value.Length != 64)
        {
            return false;
        }

        foreach (char ch in value)
        {
            if (!(ch is >= '0' and <= '9' or >= 'a' and <= 'f'))
            {
                return false;
            }
        }

        return true;
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
        // meaningless, so this rule guards every model entry one configuration declares — startup declarations are checked pair by pair here, while an addition to a live serve meets its own rule against what already-served models still pin. The check validates parsed model objects directly so callers without a YAML document apply the same rule. A model that opts into core overcommit relaxes the rule for its own claim only:
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

    // The same read as TryInteger for the byte totals a host posture declares: capacity ceilings count in whole megabytes and can name a
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

    /// <summary>Parses one model entry's budgets block: every declared resource names its own ordered tier chain and the ceilings it declares along that chain, each refusal naming the qualified config path, the offending value, and the accepted vocabulary exactly as the other choice keys do. Chain shape — a known tier at each position, priority order, no repeats — is enforced here in config-path wording; ceiling ranges are checked against what each key sizes: a system-memory KV ceiling carries <c>kv_blocks</c>' floor of 8 because it reserves physical blocks exactly as the legacy key does, every other ceiling must be positive where it states one (so an accelerator-tier KV ceiling is a finite block count, and stating none there declares no model-level bound rather than an unbounded reservation), and the read-side origin store takes no ceiling at all because that storage already holds the complete set. Nothing here consults capacity or opened pools — which claims a host can carry is resolution's answer when the engine opens.</summary>
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
                        resource == BudgetResource.Kv && tier == BudgetTier.Dram
                            ? $"{capsWhere}.{capKey} '{text}' is invalid for model '{modelName}': a system-memory KV ceiling sizes the physical reservation and must be an integer no larger than {int.MaxValue}."
                            : $"{capsWhere}.{capKey} '{text}' is invalid for model '{modelName}': '{capKey}' must state an integer ceiling.");
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

        if (resource is BudgetResource.Weights or BudgetResource.Ple && tier == BudgetTier.Disk)
        {
            throw new ConfigException(
                $"{capsWhere}.{capKey} is invalid for model '{modelName}': '{BudgetResourceNames.Of(resource)}' reads its complete set from disk storage already, so no model-level ceiling describes that position.");
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
