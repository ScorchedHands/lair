using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using Lair.Models;

namespace Lair.Host.Config;

/// <summary>
/// How a configuration edit may be applied to an already-running engine. Every editable field belongs to exactly
/// one class: <see cref="Live"/> changes take effect inside the running engine (at the next step boundary),
/// <see cref="NewRequestsOnly"/> changes alter admission validation for later requests but never disturb sequences
/// already in flight, and <see cref="RestartOnly"/> changes are refused with an actionable message because applying
/// them would mean rebuilding state a live engine cannot rebuild safely.
/// </summary>
public enum ConfigMutationClass
{
    Live,
    NewRequestsOnly,
    RestartOnly,
}

/// <summary>The wire vocabulary for mutation classes used in structured log lines and reload summaries.</summary>
public static class MutationClassNames
{
    public const string Live = "live";
    public const string NewRequests = "new-requests";
    public const string RestartOnly = "restart-only";

    public static string Of(ConfigMutationClass mutationClass) => mutationClass switch
    {
        ConfigMutationClass.Live => Live,
        ConfigMutationClass.NewRequestsOnly => NewRequests,
        _ => RestartOnly,
    };
}

/// <summary>One classified difference between the currently effective configuration and a reloaded candidate.</summary>
public sealed record FieldChange(string Field, string OldValue, string NewValue, ConfigMutationClass MutationClass);

/// <summary>What the change-application pipeline decided about one field: applied to the live engine, or refused.</summary>
public enum ConfigApplyOutcome
{
    Applied,
    Refused,
}

/// <summary>The per-field outcome produced by applying (or refusing) a classified change. The detail carries the effect for applied changes and the actionable reason for refusals.</summary>
public sealed record FieldResult(FieldChange Change, ConfigApplyOutcome Outcome, string Detail);

/// <summary>
/// One reload attempt as one structured document: whether validation passed, and every per-field outcome it
/// produced. Serialized with a consistent <c>config_reload_</c> prefix so reload summaries share the snapshot log's
/// vocabulary; hosts assert against this object instead of parsing their own emitted lines.
/// </summary>
public sealed record ReloadSummary(
    [property: JsonPropertyName("config_reload_ts")] string TimestampUtcIso8601,
    [property: JsonPropertyName("config_reload_validation_ok")] bool ValidationOk,

    /// <summary>The loader's rejection message when validation failed; null when the candidate validated.</summary>
    [property: JsonPropertyName("config_reload_error")] string? ValidationError,

    [property: JsonPropertyName("config_reload_results")] IReadOnlyList<ReloadFieldResult> Results)
{
    public static ReloadSummary FromResults(IReadOnlyList<FieldResult> results, bool validationOk, string? validationError) =>
        new(
            DateTimeOffset.UtcNow.ToString("O", CultureInfo.InvariantCulture),
            validationOk,
            validationError,
            [.. results.Select(result => new ReloadFieldResult(
                result.Change.Field,
                MutationClassNames.Of(result.Change.MutationClass),
                result.Outcome == ConfigApplyOutcome.Applied ? "applied" : "refused",
                result.Change.OldValue,
                result.Change.NewValue,
                result.Detail))]);

    /// <summary>The summary as one compact JSON document — the structured log line's body.</summary>
    public string ToJson() => JsonSerializer.Serialize(this);
}

/// <summary>One field outcome inside a <see cref="ReloadSummary"/>, in wire vocabulary.</summary>
public sealed record ReloadFieldResult(
    [property: JsonPropertyName("field")] string Field,
    [property: JsonPropertyName("class")] string MutationClass,
    [property: JsonPropertyName("outcome")] string Outcome,
    [property: JsonPropertyName("old")] string OldValue,
    [property: JsonPropertyName("new")] string NewValue,
    [property: JsonPropertyName("detail")] string Detail);

/// <summary>
/// The mutation table: which config field belongs to which mutation class, and the diff that turns a reloaded
/// candidate into classified per-field changes. Every key the loader accepts is covered; anything not explicitly
/// hot-reloadable (listener settings, backend sizing, pinning sets, model identity) classifies as restart-only so an
/// edit can never fall through unclassified. The classification itself applies nothing — application stays behind
/// full validation in the change-application pipeline.
/// </summary>
public static class ConfigMutationTable
{
    /// <summary>The effective configuration a candidate is diffed against.</summary>
    public static IReadOnlyList<FieldChange> Diff(LairConfig current, LairConfig candidate)
    {
        ArgumentNullException.ThrowIfNull(current);
        ArgumentNullException.ThrowIfNull(candidate);

        List<FieldChange> changes = [];

        EngineConfig oldEngine = current.Engine;
        EngineConfig newEngine = candidate.Engine;
        Add(changes, "engine.threads", Text(oldEngine.Threads), Text(newEngine.Threads));
        Add(changes, "engine.block_size", Text(oldEngine.BlockSize), Text(newEngine.BlockSize));
        if (newEngine.MaxBatchTokens != oldEngine.MaxBatchTokens)
        {
            changes.Add(new FieldChange("engine.max_batch_tokens", Text(oldEngine.MaxBatchTokens), Text(newEngine.MaxBatchTokens), ConfigMutationClass.Live));
        }

        if (newEngine.PrefillStepBudgetMs != oldEngine.PrefillStepBudgetMs)
        {
            changes.Add(new FieldChange("engine.prefill_step_budget_ms", TextOrNone(oldEngine.PrefillStepBudgetMs), TextOrNone(newEngine.PrefillStepBudgetMs), ConfigMutationClass.Live));
        }

        if (newEngine.PrefillKernel != oldEngine.PrefillKernel)
        {
            changes.Add(new FieldChange("engine.prefill_kernel", oldEngine.PrefillKernel, newEngine.PrefillKernel, ConfigMutationClass.NewRequestsOnly));
        }

        ServerConfig oldServer = current.Server;
        ServerConfig newServer = candidate.Server;
        Add(changes, "server.bind", oldServer.Bind, newServer.Bind);
        Add(changes, "server.port", Text(oldServer.Port), Text(newServer.Port));

        ModelConfig oldModel = current.Models[0];
        ModelConfig newModel = candidate.Models[0];
        Add(changes, "models[0].name", oldModel.Name, newModel.Name);
        Add(changes, "models[0].path", oldModel.Path, newModel.Path);
        Add(changes, "models[0].type", oldModel.Type, newModel.Type);
        if (newModel.MaxContextTokens != oldModel.MaxContextTokens)
        {
            changes.Add(new FieldChange("models[0].max_context_tokens", Text(oldModel.MaxContextTokens), Text(newModel.MaxContextTokens), ConfigMutationClass.NewRequestsOnly));
        }

        // The legacy key reports its own change only where both configurations still spell the ceiling that way. Where budgets hold it, the budget field is what a candidate actually edited and reports through — one edit, one classified entry — while the canonical block count below stays the value every enforcement seam consumes either way.
        if (newModel.KvBlocks != oldModel.KvBlocks && newModel.LegacyKvBlocksDeclared && oldModel.LegacyKvBlocksDeclared)
        {
            changes.Add(new FieldChange("models[0].kv_blocks", Text(oldModel.KvBlocks), Text(newModel.KvBlocks), ConfigMutationClass.Live));
        }

        if (newModel.ForceFencedTools != oldModel.ForceFencedTools)
        {
            changes.Add(new FieldChange("models[0].force_fenced_tools", Text(oldModel.ForceFencedTools), Text(newModel.ForceFencedTools), ConfigMutationClass.NewRequestsOnly));
        }

        if (newModel.ToolParser != oldModel.ToolParser)
        {
            changes.Add(new FieldChange("models[0].tool_parser", oldModel.ToolParser, newModel.ToolParser, ConfigMutationClass.NewRequestsOnly));
        }

        if (newModel.Reasoning != oldModel.Reasoning)
        {
            changes.Add(new FieldChange("models[0].reasoning", oldModel.Reasoning, newModel.Reasoning, ConfigMutationClass.NewRequestsOnly));
        }

        if (newModel.MaxReasoningTokens != oldModel.MaxReasoningTokens)
        {
            changes.Add(new FieldChange("models[0].max_reasoning_tokens", TextOrNone(oldModel.MaxReasoningTokens), TextOrNone(newModel.MaxReasoningTokens), ConfigMutationClass.NewRequestsOnly));
        }

        if (newModel.MaxResponseTokens != oldModel.MaxResponseTokens)
        {
            changes.Add(new FieldChange("models[0].max_response_tokens", TextOrNone(oldModel.MaxResponseTokens), TextOrNone(newModel.MaxResponseTokens), ConfigMutationClass.NewRequestsOnly));
        }

        if (newModel.MaxConcurrentRequests != oldModel.MaxConcurrentRequests)
        {
            changes.Add(new FieldChange("models[0].max_concurrent_requests", TextOrNone(oldModel.MaxConcurrentRequests), TextOrNone(newModel.MaxConcurrentRequests), ConfigMutationClass.NewRequestsOnly));
        }

        AddBudgetChanges(changes, oldModel, newModel);

        PlacementConfig oldPlacement = oldModel.Placement;
        PlacementConfig newPlacement = newModel.Placement;
        Add(changes, "models[0].placement.device", oldPlacement.Device, newPlacement.Device);
        Add(changes, "models[0].placement.kv_residency", oldPlacement.KvResidency, newPlacement.KvResidency);
        Add(changes, "models[0].placement.prefill", oldPlacement.Prefill, newPlacement.Prefill);
        if (!newPlacement.CoreSet.SetEquals(oldPlacement.CoreSet))
        {
            changes.Add(new FieldChange("models[0].placement.core_set", CanonicalCoreSet(oldPlacement.CoreSet), CanonicalCoreSet(newPlacement.CoreSet), ConfigMutationClass.RestartOnly));
        }

        if (newPlacement.ThreadQuota != oldPlacement.ThreadQuota)
        {
            changes.Add(new FieldChange("models[0].placement.thread_quota", TextOrNone(oldPlacement.ThreadQuota), TextOrNone(newPlacement.ThreadQuota), ConfigMutationClass.Live));
        }

        Add(changes, "models[0].placement.prefill_lane", oldPlacement.PrefillLane, newPlacement.PrefillLane);

        // Core-set overcommit changes what a running engine's workers contend for, so it is established when the engine opens: refused live,
        // exactly like the pinned set it relaxes. Declared capacity totals bind accounting at open for the same reason.
        if (newPlacement.CoreOvercommit != oldPlacement.CoreOvercommit)
        {
            changes.Add(new FieldChange("models[0].placement.core_overcommit", Text(oldPlacement.CoreOvercommit), Text(newPlacement.CoreOvercommit), ConfigMutationClass.RestartOnly));
        }

        AddTierChanges(changes, current.Tiers, candidate.Tiers);

        if (candidate.Ui.RecentRequests != current.Ui.RecentRequests)
        {
            changes.Add(new FieldChange("ui.recent_requests", Text(current.Ui.RecentRequests), Text(candidate.Ui.RecentRequests), ConfigMutationClass.Live));
        }

        return changes;
    }

    /// <summary>
    /// Best-effort classification of a reload candidate whose full validation failed: re-reads the file's raw scalar
    /// values for known keys and reports every field whose text differs from what the running engine currently holds,
    /// each carrying its mutation class. This lets a refused edit still name its field and class in the log even when
    /// the loader rejected it (for example an out-of-range restart-only value) — reported outcomes are refusals; the
    /// candidate validated nothing, so nothing here can be applied. <c>placement.core_set</c> is skipped because raw
    /// specs ("0-15" vs "0,1,2,...") have no single canonical text to compare against; a semantic core-set change on
    /// a file that does validate surfaces through the full <see cref="Diff"/> instead. Returns empty when the file
    /// cannot be parsed at all — an unparseable edit has no field values to name.
    /// </summary>
    public static IReadOnlyList<FieldChange> ExtractEdits(LairConfig current, string filePath)
    {
        ArgumentNullException.ThrowIfNull(current);

        YamlDotNet.RepresentationModel.YamlStream stream = new();
        try
        {
            using FileStream file = File.OpenRead(filePath);
            using StreamReader reader = new(file);
            stream.Load(reader);

            if (stream.Documents.Count == 0 ||
                stream.Documents[0].RootNode is not YamlDotNet.RepresentationModel.YamlMappingNode root)
            {
                return [];
            }

            List<FieldChange> changes = [];

            foreach ((string field, string? rawValue) in RawScalars(root))
            {
                if (rawValue is null || rawValue == CurrentText(current, field))
                {
                    continue; // unchanged or absent: nothing to report about this key
                }

                changes.Add(new FieldChange(field, CurrentText(current, field), rawValue, Classify(field)));
            }

            return changes;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            // A file this untrustworthy has no per-field values to name; the reload's validation error already says what happened.
            _ = ex;
            return [];
        }
    }

    /// <summary>The mutation class of one config field: every loader-accepted key maps here, and unknown keys classify as restart-only rather than falling through.</summary>
    public static ConfigMutationClass Classify(string field) => field switch
    {
        "engine.max_batch_tokens" => ConfigMutationClass.Live,
        "engine.prefill_step_budget_ms" => ConfigMutationClass.Live,
        "engine.prefill_kernel" => ConfigMutationClass.NewRequestsOnly,
        "models[0].kv_blocks" => ConfigMutationClass.Live,
        "models[0].placement.thread_quota" => ConfigMutationClass.Live,
        "ui.recent_requests" => ConfigMutationClass.Live,
        "models[0].max_context_tokens" => ConfigMutationClass.NewRequestsOnly,
        "models[0].force_fenced_tools" => ConfigMutationClass.NewRequestsOnly,
        "models[0].tool_parser" => ConfigMutationClass.NewRequestsOnly,
        "models[0].reasoning" => ConfigMutationClass.NewRequestsOnly,
        "models[0].max_reasoning_tokens" => ConfigMutationClass.NewRequestsOnly,
        "models[0].max_response_tokens" => ConfigMutationClass.NewRequestsOnly,
        "models[0].max_concurrent_requests" => ConfigMutationClass.NewRequestsOnly,
        "models[0].placement.core_overcommit" => ConfigMutationClass.RestartOnly,
        "tiers.dram_mb" or "tiers.dram_cores" or "tiers.nvme_path" or "tiers.nodes" => ConfigMutationClass.RestartOnly,
        _ => BudgetFieldClass(field) ?? ConfigMutationClass.RestartOnly,
    };

    // Every declared-capacity difference between two configurations, reported key by key so each carries its own restart-only class and a
    // refused edit cannot hide another. Per-node overrides are a structural list with no single canonical scalar to compare against raw
    // text — exactly like 'placement.core_set' and budget chain orders — so they report through one field carrying the whole declared posture.
    private static void AddTierChanges(List<FieldChange> changes, TierCapacityConfig? currentTiers, TierCapacityConfig? candidateTiers)
    {
        Add(changes, "tiers.dram_mb", NumberOrNone(currentTiers?.DramMb), NumberOrNone(candidateTiers?.DramMb));
        Add(changes, "tiers.dram_cores", NumberOrNone(currentTiers?.DramCores), NumberOrNone(candidateTiers?.DramCores));
        Add(changes, "tiers.nvme_path", currentTiers?.NvmePath ?? "(unset)", candidateTiers?.NvmePath ?? "(unset)");

        string currentNodeOverrides = CanonicalNodeOverrides(currentTiers);
        string candidateNodeOverrides = CanonicalNodeOverrides(candidateTiers);
        if (currentNodeOverrides != candidateNodeOverrides)
        {
            changes.Add(new FieldChange("tiers.nodes", currentNodeOverrides, candidateNodeOverrides, ConfigMutationClass.RestartOnly));
        }
    }

    private static string NumberOrNone(long? value) => value is null ? "(unset)" : value.Value.ToString(CultureInfo.InvariantCulture);

    // The declared per-node overrides in one canonical text so two postures compare like-for-like: each entry keyed by the node position it names.
    private static string CanonicalNodeOverrides(TierCapacityConfig? tiers) => tiers?.Nodes is not { Count: > 0 }
        ? "(none)"
        : string.Join(
            ";",
            tiers.Nodes.OrderBy(node => node.Index).Select(node => $"index={node.Index}:mb={(node.Mb is long mb ? mb.ToString(CultureInfo.InvariantCulture) : "unset")}:cores={(node.Cores is int cores ? cores.ToString(CultureInfo.InvariantCulture) : "unset")}"));

    // How one budget field classifies: a ceiling on the system-memory tier of the KV chain is the live soft cap 'kv_blocks' has always been;
    // weight ceilings bind what every forward reads and retarget only by re-reading slabs from the model file, so they are restart-only; prefill
    // buffer ceilings classify for later requests alone even where nothing clamps to them yet — a class is not an enforcement promise. A chain's
    // order names structure that was resolved when the engine opened, and any budget-shaped key this table does not recognize stays restart-only
    // rather than falling through unclassified.
    private static ConfigMutationClass? BudgetFieldClass(string field)
    {
        const string prefix = "models[0].budgets.";
        if (!field.StartsWith(prefix, StringComparison.Ordinal))
        {
            return null;
        }

        string[] parts = field[prefix.Length..].Split('.');

        if (parts is [var resourceKey, "order"])
        {
            return ConfigMutationClass.RestartOnly;
        }

        if (parts is not [var capResourceKey, "caps", var capKey] || !BudgetResourceNames.TryParse(capResourceKey, out BudgetResource resource))
        {
            return ConfigMutationClass.RestartOnly;
        }

        if (!BudgetCapKeys.TryParse(capKey, resource, out BudgetTier tier))
        {
            return ConfigMutationClass.RestartOnly;
        }

        return (resource, tier) switch
        {
            (BudgetResource.Kv, BudgetTier.Dram) => ConfigMutationClass.Live,
            (BudgetResource.PrefillBuffers, _) => ConfigMutationClass.NewRequestsOnly,
            _ => ConfigMutationClass.RestartOnly,
        };
    }

    private static void Add(List<FieldChange> changes, string field, string oldValue, string newValue)
    {
        if (oldValue != newValue)
        {
            changes.Add(new FieldChange(field, oldValue, newValue, Classify(field)));
        }
    }

    // Scalar text of every known config key straight from the YAML document: qualified name → raw scalar value,
    // null when the key is absent or its position in the document is not a plain scalar.
    private static IEnumerable<(string Field, string? Value)> RawScalars(YamlDotNet.RepresentationModel.YamlMappingNode root)
    {
        YamlDotNet.RepresentationModel.YamlMappingNode? engine = Child(root, "engine");
        yield return ("server.bind", ScalarText(Child(root, "server"), "bind"));
        yield return ("server.port", ScalarText(Child(root, "server"), "port"));
        yield return ("engine.threads", ScalarText(engine, "threads"));
        yield return ("engine.block_size", ScalarText(engine, "block_size"));
        yield return ("engine.max_batch_tokens", ScalarText(engine, "max_batch_tokens"));
        yield return ("engine.prefill_step_budget_ms", ScalarText(engine, "prefill_step_budget_ms"));
        yield return ("engine.prefill_kernel", ScalarText(engine, "prefill_kernel"));

        YamlDotNet.RepresentationModel.YamlMappingNode? model = FirstModel(root);
        yield return ("models[0].name", ScalarText(model, "name"));
        yield return ("models[0].path", ScalarText(model, "path"));
        yield return ("models[0].type", ScalarText(model, "type"));
        yield return ("models[0].max_context_tokens", ScalarText(model, "max_context_tokens"));
        yield return ("models[0].kv_blocks", ScalarText(model, "kv_blocks"));
        yield return ("models[0].force_fenced_tools", ScalarText(model, "force_fenced_tools"));
        yield return ("models[0].tool_parser", ScalarText(model, "tool_parser"));
        yield return ("models[0].reasoning", ScalarText(model, "reasoning"));
        yield return ("models[0].max_reasoning_tokens", ScalarText(model, "max_reasoning_tokens"));
        yield return ("models[0].max_response_tokens", ScalarText(model, "max_response_tokens"));
        yield return ("models[0].max_concurrent_requests", ScalarText(model, "max_concurrent_requests"));

        YamlDotNet.RepresentationModel.YamlMappingNode? placement = Child(model, "placement");
        yield return ("models[0].placement.device", ScalarText(placement, "device"));
        yield return ("models[0].placement.kv_residency", ScalarText(placement, "kv_residency"));
        yield return ("models[0].placement.prefill", ScalarText(placement, "prefill"));
        yield return ("models[0].placement.thread_quota", ScalarText(placement, "thread_quota"));
        yield return ("models[0].placement.prefill_lane", ScalarText(placement, "prefill_lane"));

        // Core-set overcommit is a plain scalar beside the pinned set it relaxes. Declared capacity totals are plain scalars too; per-node
        // override lists skip here exactly as 'placement.core_set' and budget chain orders do — a structural list has no single canonical text,
        // and a semantic override change on a file that does validate surfaces through the full Diff instead.
        yield return ("models[0].placement.core_overcommit", ScalarText(placement, "core_overcommit"));

        YamlDotNet.RepresentationModel.YamlMappingNode? tiers = Child(root, "tiers");
        yield return ("tiers.dram_mb", ScalarText(tiers, "dram_mb"));
        yield return ("tiers.dram_cores", ScalarText(tiers, "dram_cores"));
        yield return ("tiers.nvme_path", ScalarText(tiers, "nvme_path"));

        // Budget chain order lists are skipped here exactly as 'placement.core_set' is: a structural list has no single canonical text to compare against, and a semantic chain change on a file that does validate surfaces through the full Diff instead. Each ceiling is a plain scalar carrying its own qualified key below.
        YamlDotNet.RepresentationModel.YamlMappingNode? budgets = Child(model, "budgets");

        foreach (string resourceKey in BudgetResourceNames.DeclaredKeys)
        {
            if (!BudgetResourceNames.TryParse(resourceKey, out BudgetResource resource))
            {
                continue; // the vocabulary is fixed here; a name it does not recognize has no ceiling keys to read.
            }

            YamlDotNet.RepresentationModel.YamlMappingNode? ceilings = Child(Child(budgets, resourceKey), "caps");

            foreach (string capKey in BudgetCapKeys.For(resource))
            {
                yield return ($"models[0].budgets.{resourceKey}.caps.{capKey}", ScalarText(ceilings, capKey));
            }
        }

        yield return ("ui.recent_requests", ScalarText(Child(root, "ui"), "recent_requests"));
    }

    // The current effective value of a known key in the same text form RawScalars produces, so comparison is like-for-like.
    private static string CurrentText(LairConfig current, string field) => field switch
    {
        "server.bind" => current.Server.Bind,
        "server.port" => Text(current.Server.Port),
        "engine.threads" => Text(current.Engine.Threads),
        "engine.block_size" => Text(current.Engine.BlockSize),
        "engine.max_batch_tokens" => Text(current.Engine.MaxBatchTokens),
        "engine.prefill_step_budget_ms" => TextOrNone(current.Engine.PrefillStepBudgetMs),
        "engine.prefill_kernel" => current.Engine.PrefillKernel,
        "models[0].name" => current.Models[0].Name,
        "models[0].path" => current.Models[0].Path,
        "models[0].type" => current.Models[0].Type,
        "models[0].max_context_tokens" => Text(current.Models[0].MaxContextTokens),
        "models[0].kv_blocks" => Text(current.Models[0].KvBlocks),
        "models[0].force_fenced_tools" => Text(current.Models[0].ForceFencedTools),
        "models[0].tool_parser" => current.Models[0].ToolParser,
        "models[0].reasoning" => current.Models[0].Reasoning,
        "models[0].max_reasoning_tokens" => TextOrNone(current.Models[0].MaxReasoningTokens),
        "models[0].max_response_tokens" => TextOrNone(current.Models[0].MaxResponseTokens),
        "models[0].max_concurrent_requests" => TextOrNone(current.Models[0].MaxConcurrentRequests),
        "models[0].placement.device" => current.Models[0].Placement.Device,
        "models[0].placement.kv_residency" => current.Models[0].Placement.KvResidency,
        "models[0].placement.prefill" => current.Models[0].Placement.Prefill,
        "models[0].placement.thread_quota" => TextOrNone(current.Models[0].Placement.ThreadQuota),
        "models[0].placement.prefill_lane" => current.Models[0].Placement.PrefillLane,
        "models[0].placement.core_overcommit" => Text(current.Models[0].Placement.CoreOvercommit),
        "tiers.dram_mb" => NumberOrNone(current.Tiers?.DramMb),
        "tiers.dram_cores" => NumberOrNone(current.Tiers?.DramCores),
        "tiers.nvme_path" => current.Tiers?.NvmePath ?? "(unset)",
        "ui.recent_requests" => Text(current.Ui.RecentRequests),
        _ => BudgetCurrentText(current, field) ?? string.Empty, // a budget ceiling the running engine holds in its declared chains reads back in the same text form RawScalars produces.
    };

    private static YamlDotNet.RepresentationModel.YamlMappingNode? Child(YamlDotNet.RepresentationModel.YamlMappingNode? parent, string key) =>
        parent is not null && parent.Children.TryGetValue(new YamlDotNet.RepresentationModel.YamlScalarNode(key), out YamlDotNet.RepresentationModel.YamlNode? node)
            ? node as YamlDotNet.RepresentationModel.YamlMappingNode
            : null;

    private static string? ScalarText(YamlDotNet.RepresentationModel.YamlMappingNode? parent, string key) =>
        parent is not null && parent.Children.TryGetValue(new YamlDotNet.RepresentationModel.YamlScalarNode(key), out YamlDotNet.RepresentationModel.YamlNode? node)
            ? node as YamlDotNet.RepresentationModel.YamlScalarNode is YamlDotNet.RepresentationModel.YamlScalarNode scalar ? scalar.Value : null
            : null;

    private static YamlDotNet.RepresentationModel.YamlMappingNode? FirstModel(YamlDotNet.RepresentationModel.YamlMappingNode root) =>
        Child(root, "models") is null && root.Children.TryGetValue(new YamlDotNet.RepresentationModel.YamlScalarNode("models"), out YamlDotNet.RepresentationModel.YamlNode? models)
            ? (models as YamlDotNet.RepresentationModel.YamlSequenceNode)?.Children.FirstOrDefault() as YamlDotNet.RepresentationModel.YamlMappingNode
            : null;

    private static string CanonicalCoreSet(IReadOnlySet<int> cores) =>
        cores.Count == 0 ? "(none)" : string.Join(",", cores.OrderBy(core => core));

    private static string Text(int value) => value.ToString(CultureInfo.InvariantCulture);

    private static string Text(bool value) => value ? "true" : "false";

    private static string TextOrNone(int? value) => value is null ? "(unset)" : Text(value.Value);

    // Every budget-shaped difference between two configurations, reported field by field so each ceiling carries its own mutation class and a
    // refused sibling never hides the one edit that can apply. A resource neither side declares reports nothing: absent chains compare as absent.
    private static void AddBudgetChanges(List<FieldChange> changes, ModelConfig current, ModelConfig candidate)
    {
        foreach (string resourceKey in BudgetResourceNames.DeclaredKeys)
        {
            if (!BudgetResourceNames.TryParse(resourceKey, out BudgetResource resource))
            {
                continue; // the vocabulary is fixed here; a name it does not recognize declares no chain to compare.
            }

            BudgetChain? oldChain = ChainFor(current.Budgets, resource);
            BudgetChain? newChain = ChainFor(candidate.Budgets, resource);

            Add(changes, $"models[0].budgets.{resourceKey}.order", OrderText(oldChain), OrderText(newChain));

            foreach (string capKey in BudgetCapKeys.For(resource))
            {
                if (!BudgetCapKeys.TryParse(capKey, resource, out BudgetTier tier))
                {
                    continue;
                }

                Add(changes, $"models[0].budgets.{resourceKey}.caps.{capKey}", CeilingText(oldChain, tier), CeilingText(newChain, tier));
            }
        }
    }

    private static BudgetChain? ChainFor(IReadOnlyList<BudgetChain>? chains, BudgetResource resource)
    {
        if (chains is null)
        {
            return null;
        }

        foreach (BudgetChain chain in chains)
        {
            if (chain.Resource == resource)
            {
                return chain;
            }
        }

        return null;
    }

    private static string CeilingText(BudgetChain? chain, BudgetTier tier) =>
        chain is not null && chain.Caps.TryGetValue(tier, out long ceiling) ? ceiling.ToString(CultureInfo.InvariantCulture) : "(unset)";

    private static string OrderText(BudgetChain? chain) => chain is null
        ? "(unset)"
        : string.Join(",", chain.Order.Select(tier => BudgetTierNames.Of(tier)));

    // The current effective value of a budget field in the same text form RawScalars produces, or null when this field names no budget at all.
    private static string? BudgetCurrentText(LairConfig current, string field)
    {
        const string prefix = "models[0].budgets.";
        if (!field.StartsWith(prefix, StringComparison.Ordinal))
        {
            return null;
        }

        string[] parts = field[prefix.Length..].Split('.');

        if (parts is [var resourceKey, "order"])
        {
            return BudgetResourceNames.TryParse(resourceKey, out BudgetResource ordered) ? OrderText(ChainFor(current.Models[0].Budgets, ordered)) : null;
        }

        if (parts is not [var capResourceKey, "caps", var capKey] || !BudgetResourceNames.TryParse(capResourceKey, out BudgetResource resource))
        {
            return null;
        }

        return BudgetCapKeys.TryParse(capKey, resource, out BudgetTier tier) ? CeilingText(ChainFor(current.Models[0].Budgets, resource), tier) : null;
    }
}
