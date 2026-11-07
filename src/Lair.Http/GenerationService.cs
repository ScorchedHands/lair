using System.Collections.Concurrent;
using System.Globalization;
using Lair.Http.Monitoring;
using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;
using Lair.Templating.Minja;
using Lair.Tokenizer;
using Microsoft.Extensions.Logging;

namespace Lair.Http;

/// <summary>
/// Single-model generation facade: opens one configured GGUF model (weights, forward engine, tokenizer,
/// chat template, paged KV pool) and runs <see cref="GenerationRequest"/>s against it. Requests are admitted
/// to a continuous-batching scheduler, so admitted sequences decode together in shared steps and a long
/// prompt never blocks shorter ones. A request is refused with <see cref="EngineBusyException"/> only when
/// the KV pool cannot back its prompt plus token budget.
/// </summary>
public sealed class GenerationService : IDisposable
{
    private readonly GgufFile _file;
    private readonly CpuBackend _backend;
    private readonly TransformerModel _model;
    private readonly GgufTokenizer _tokenizer;
    private readonly MinjaChatTemplate? _chatTemplate;
    private readonly TemplateCapabilities? _chatTemplateCapabilities;
    private readonly ILogger? _logger;
    private readonly BlockPool _pool;
    private readonly GenerationScheduler _scheduler;
    private EngineMetricsBus? _metrics;

    // Generations currently publishing monitor lifecycle frames, keyed by their stream request id — the in-flight
    // half of a resync state dump. Trackers are added at admission and removed when their completion frame publishes.
    private readonly ConcurrentDictionary<string, RequestTracker> _monitoredRequests = new(StringComparer.Ordinal);

    // Change-application state: the configuration currently in force (restart-only refusals keep their old values),
    // the advertised model facts recomputed whenever a live or new-requests field changes, and the last reload attempt.
    // The applies serialize on one gate so concurrent apply calls never interleave half-applied states.
    private readonly object _reloadGate = new();
    private LairConfig _current;
    private ModelInfo _modelInfo; // reference-swapped under the reload gate so observers read a complete record
    private ReloadSummary? _lastReload;

    /// <summary>The metrics bus attached by the API host, if any — null for hosts that never attach one.</summary>
    internal EngineMetricsBus? Metrics => _metrics;

    private bool _disposed;

    private GenerationService(
        GgufFile file,
        CpuBackend backend,
        TransformerModel model,
        GgufTokenizer tokenizer,
        MinjaChatTemplate? chatTemplate,
        TemplateCapabilities? chatTemplateCapabilities,
        ILogger? logger,
        BlockPool pool,
        GenerationScheduler scheduler,
        ModelInfo info,
        LairConfig config)
    {
        _file = file;
        _backend = backend;
        _model = model;
        _tokenizer = tokenizer;
        _chatTemplate = chatTemplate;
        _chatTemplateCapabilities = chatTemplateCapabilities;
        _logger = logger;
        _pool = pool;
        _scheduler = scheduler;
        _current = config;
        _modelInfo = info;
    }

    /// <summary>The served model's advertised facts; requests must name <see cref="ModelInfo.Id"/>. Recomputed when a hot-reloaded field changes what the engine advertises.</summary>
    public ModelInfo Model => Volatile.Read(ref _modelInfo);

    /// <summary>
    /// Capability facts detected once for the loaded model's chat template (null when the model has no chat template).
    /// These are the facts a request-time tool-precedence decision consumes to choose between the template's own tool
    /// rendering and LAIR's fenced-tool injection; they are published here so that decision has a single source of
    /// truth on the loaded-model path.
    /// </summary>
    public TemplateCapabilities? ChatTemplateCapabilities => _chatTemplateCapabilities;

    /// <summary>The id that ends a generation: the model's declared EOT id (<c>tokenizer.ggml.eot_token_id</c>) when it has one, otherwise the EOS id.</summary>
    internal int StopId => _tokenizer.EotId >= 0 ? _tokenizer.EotId : _tokenizer.EosId;

    /// <summary>Vocabulary size of the served model; token ids at or above it have no vocabulary text.</summary>
    public int VocabSize => Model.VocabSize;

    /// <summary>Worker count the served model's CPU backend actually runs with, after engine threads and the model's thread quota are combined.</summary>
    public int EffectiveWorkerCount => _backend.EffectiveWorkerCount;

    /// <summary>The configured kind of the served model as advertised on <c>/v1/models</c>. The config loader accepts only "chat" in this build; surfaces that must exclude embed models filter on this fact instead of an assumption.</summary>
    public string ModelType => _current.Models[0].Type;

    /// <summary>
    /// Attaches the metrics bus that samples this engine's counters and feeds <c>/monitor/stream</c>. The API host
    /// calls it once while building the app; hosts that never attach one simply produce no snapshots. Idempotent:
    /// a second call returns the already-attached bus.
    /// </summary>
    internal EngineMetricsBus AttachMetrics(ILoggerFactory loggerFactory, TimeSpan interval)
    {
        ArgumentNullException.ThrowIfNull(loggerFactory);
        if (_metrics is not null)
        {
            return _metrics;
        }

        PrefixCache? cache = _cacheForMetrics;
        EngineMetricsBus bus = new(Model.Id, _scheduler, _pool, cache, loggerFactory, interval);

        // The monitor's request-side state starts from the configuration in force: the recent-request ring capacity,
        // the live in-flight views a resync can dump, and the model facts snapshots carry additively.
        bus.SetRecentRequestCapacity(_current.Ui.RecentRequests);
        bus.AttachInFlightProvider(InFlightRequestViews);
        bus.AttachModelFacts(BuildMonitorFacts(_current));

        _metrics = bus;
        return bus;
    }

    // Snapshot identity/placement facts straight from the configuration in force plus the backend's live worker count.
    private ModelFacts BuildMonitorFacts(LairConfig config) => new(
        config.Models[0].Type,
        _backend.EffectiveWorkerCount,
        FormatCoreSet(config.Models[0].Placement.CoreSet),
        config.Models[0].Placement.ThreadQuota,
        config.Models[0].Placement.Device,
        config.Models[0].Placement.Prefill,
        config.Models[0].Placement.KvResidency);

    // Pinned core ids as compressed ranges ("0-23", "1, 4-6"); null when nothing is pinned.
    private static string? FormatCoreSet(IReadOnlySet<int> cores)
    {
        if (cores.Count == 0)
        {
            return null;
        }

        int[] sorted = [.. cores];
        Array.Sort(sorted);

        List<string> ranges = [];
        int runStart = sorted[0];
        int previous = sorted[0];
        for (int i = 1; i <= sorted.Length; i++)
        {
            if (i == sorted.Length || sorted[i] != previous + 1)
            {
                ranges.Add(runStart == previous ? runStart.ToString() : $"{runStart}-{previous}");
                if (i < sorted.Length)
                {
                    runStart = sorted[i];
                    previous = sorted[i];
                }
            }
            else
            {
                previous = sorted[i];
            }
        }

        return string.Join(", ", ranges);
    }

    // Current in-flight request views for a resync: volatile reads off each tracker, ordered by id so repeated
    // dumps of the same moment read alike.
    private IReadOnlyList<RequestPayload> InFlightRequestViews()
    {
        List<RequestPayload> views = [];
        foreach (KeyValuePair<string, RequestTracker> entry in _monitoredRequests)
        {
            views.Add(entry.Value.View());
        }

        views.Sort(static (a, b) => string.CompareOrdinal(a.RequestId, b.RequestId));
        return views;
    }

    // The scheduler owns its prefix cache privately; the service keeps what it passed so metrics can report cached blocks.
    private PrefixCache? _cacheForMetrics;

    /// <summary>Decodes one token id through the served model's tokenizer, special tokens included so control pieces round-trip.</summary>
    public string DecodeToken(int tokenId) => _tokenizer.Decode([tokenId], includeSpecial: true);

    /// <summary>The KV block budget admission currently spends from — the soft cap within <see cref="ReservedKvBlocks"/>.</summary>
    public int KvBudgetBlocks => _pool.EffectiveBudgetBlocks;

    /// <summary>Physical KV blocks reserved for this engine at open; never changes while it runs.</summary>
    public int ReservedKvBlocks => _pool.CapacityBlocks;

    /// <summary>Physical blocks currently handed out by the pool, including shared cache references and active sequences.</summary>
    public int KvUsedBlocks => _pool.UsedBlocks;

    /// <summary>Prefix-cache entries holding a block reference right now — what a shrinking budget drains through LRU eviction.</summary>
    public int CachedPrefixBlocks => _cacheForMetrics?.CachedBlocks ?? 0;

    /// <summary>The scheduler's live engine counters (prefill chunks and tokens, decode steps, occupancy) as the step loop maintains them.</summary>
    public SchedulerStats Stats => _scheduler.Stats;

    /// <summary>The configuration currently in force: candidates applied so far merged over the opened one, with refused restart-only fields still at their old values. Read-only snapshot for hosts that diff reloads against it.</summary>
    public LairConfig EffectiveConfiguration => Volatile.Read(ref _current);

    /// <summary>The most recent completed reload attempt as one structured summary, or null if none has finished. Hosts assert against this instead of parsing log lines.</summary>
    public ReloadSummary? LastReload => Volatile.Read(ref _lastReload);

    /// <summary>Records the outcome of one reload attempt (applied or rejected) for programmatic observation by hosts and tests.</summary>
    public void ReportReload(ReloadSummary summary)
    {
        ArgumentNullException.ThrowIfNull(summary);
        Volatile.Write(ref _lastReload, summary);
    }

    /// <summary>
    /// Entry point of the change-application pipeline: diffs a fully validated candidate configuration against what is
    /// currently in force and dispatches every difference by its mutation class. Live changes apply to the running
    /// engine (prefill width at the next step boundary, worker participation from the next kernel call, KV budget as one
    /// gated cap-flip-plus-drain); new-requests-only changes update admission for later requests while sequences already in
    /// flight run on unchanged limits; restart-only fields are refused naming field and class, their old values retained.
    /// The effective configuration merges exactly the applied fields — a candidate carrying both an applicable live change
    /// and a refused restart-only change ends up with the first in force and the second still at its old value. Returns one
    /// per-field result per difference; nothing here re-validates (candidates arrive validated) and no failure leaves a
    /// half-applied state: each field's apply is independent, and refusals are inert by construction.
    /// </summary>
    public IReadOnlyList<FieldResult> ApplyConfiguration(LairConfig candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        ObjectDisposedException.ThrowIf(_disposed, typeof(GenerationService));

        lock (_reloadGate)
        {
            List<FieldResult> results = [];
            LairConfig effective = _current;

            foreach (FieldChange change in ConfigMutationTable.Diff(effective, candidate))
            {
                switch (change.Field)
                {
                    case "engine.max_batch_tokens":
                        _scheduler.ApplyMaxBatchTokens(candidate.Engine.MaxBatchTokens);
                        effective = effective with { Engine = effective.Engine with { MaxBatchTokens = candidate.Engine.MaxBatchTokens } };
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            "prefill chunk width takes effect at the next step boundary; sequences mid-chunk finish that chunk at the old width"));
                        break;

                    case "models[0].placement.thread_quota":
                        int workersBefore = _backend.EffectiveWorkerCount;
                        int workersAfter = _backend.ApplyThreadQuota(candidate.Models[0].Placement.ThreadQuota);
                        effective = effective with { Models = [effective.Models[0] with { Placement = effective.Models[0].Placement with { ThreadQuota = candidate.Models[0].Placement.ThreadQuota } }] };
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"effective workers {workersBefore} -> {workersAfter}; participation adjusts from the next kernel call (the scheduler's next step), lowered workers idle"));
                        break;

                    case "models[0].kv_blocks":
                        GenerationScheduler.KvBudgetApply budget = _scheduler.ApplyKvBudget(candidate.Models[0].KvBlocks);
                        effective = effective with { Models = [effective.Models[0] with { KvBlocks = candidate.Models[0].KvBlocks } ] };
                        RefreshAdvertisedModel(effective);
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"KV budget {change.OldValue} -> {budget.AppliedBlocks} blocks within the {_pool.CapacityBlocks}-block physical reservation; drained {budget.EvictedCacheEntries} cached prefix blocks above the new cap (active sequences untouched)"));
                        break;

                    case "models[0].max_context_tokens":
                        effective = effective with { Models = [effective.Models[0] with { MaxContextTokens = candidate.Models[0].MaxContextTokens } ] };
                        RefreshAdvertisedModel(effective);
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"admission now validates against a {_modelInfo.ContextLength}-token window; sequences already in flight are unaffected"));
                        break;

                    case "models[0].force_fenced_tools":
                        effective = effective with { Models = [effective.Models[0] with { ForceFencedTools = candidate.Models[0].ForceFencedTools }] };
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"chat requests admitted from now on {(candidate.Models[0].ForceFencedTools ? "inject LAIR's fenced tool block even for a tool-capable template" : "follow the loaded template's tool capabilities")}; sequences already in flight are unaffected"));
                        break;

                    case "ui.recent_requests":
                        _metrics?.SetRecentRequestCapacity(candidate.Ui.RecentRequests);
                        effective = effective with { Ui = candidate.Ui };
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Applied,
                            $"monitor recent-request ring now holds at most {candidate.Ui.RecentRequests} completed requests; entries beyond the new cap dropped oldest-first"));
                        break;

                    default:
                        results.Add(new FieldResult(change, ConfigApplyOutcome.Refused, RefusalDetail(change)));
                        break;
                }
            }

            _current = effective;

            // Snapshots must speak the configuration in force: re-push model facts after every apply so additive
            // snapshot fields (worker threads, quota) track live-applied changes without waiting for a new tick's config.
            EngineMetricsBus? metricsForFacts = _metrics;
            if (metricsForFacts is not null)
            {
                metricsForFacts.AttachModelFacts(BuildMonitorFacts(effective));
            }

            return results;
        }
    }

    // Advertised model facts recomputed from the effective configuration and live pool state: the window admission
    // validates against is the smallest of configured max context, what the forward engine can actually address
    // (its precomputed RoPE extent), and the token slots the current KV budget can back. A raise beyond the engine's
    // addressed extent clamps there — re-cutting those tables belongs to a model reload, not to a hot edit.
    private void RefreshAdvertisedModel(LairConfig effective)
    {
        long backed = Math.Min((long)_pool.EffectiveBudgetBlocks * _pool.BlockTokens, _model.MaxContextTokens);
        int context = (int)Math.Min(effective.Models[0].MaxContextTokens, backed);

        Volatile.Write(ref _modelInfo, _modelInfo with { ContextLength = context, KvBlocks = _pool.EffectiveBudgetBlocks });
    }

    private static string RefusalDetail(FieldChange change) => change.Field switch
    {
        "models[0].path" => $"{change.Field} requires restart (restart-only): '{change.OldValue}' -> '{change.NewValue}'. Server-side model reload is not implemented in this build — stop and restart lair to serve a different model file.",
        "engine.block_size" => $"{change.Field} requires restart (restart-only): {change.OldValue} -> {change.NewValue}. KV block geometry is fixed for the lifetime of a running engine; change it by restarting lair.",
        "server.bind" or "server.port" => $"{change.Field} requires restart (restart-only): {change.OldValue} -> {change.NewValue}. The HTTP listener binds once at startup.",
        _ => $"{change.Field} requires restart (restart-only): {change.OldValue} -> {change.NewValue}. This field is established when the engine opens; change it by restarting lair.",
    };

    /// <summary>
    /// Opens the first model declared in the configuration: GGUF container, mmap-backed weights through the
    /// forward engine, tokenizer, chat template, a KV pool sized by the model entry, a prefix cache over that pool,
    /// and the batch scheduler over them. The prefix cache lives exactly as long as the pool, so each opened
    /// service starts with an empty cache. The effective context is the smaller of the configured max context and
    /// the token slots the KV pool can hold.
    /// </summary>
    public static GenerationService Open(LairConfig config, ILogger? logger = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Models.Count == 0)
        {
            throw new InvalidOperationException(
                "the configuration declares no models: add at least one entry under 'models:' before opening the engine.");
        }

        ModelConfig modelConfig = config.Models[0];
        int blockTokens = config.Engine.BlockSize;

        // The KV pool holds KvBlocks * blockTokens token slots, reserved physically at open. Admission validates against the
        // smaller of the configured max context and those slots; the forward engine's addressable extent covers the whole
        // reservation so a later hot raise of max_context_tokens stays applicable up to what the pool can back — raising it
        // beyond the reservation would need new RoPE tables, i.e. a model reload.
        int reservedSlots = checked(modelConfig.KvBlocks * blockTokens);
        int maxContext = Math.Min(modelConfig.MaxContextTokens, reservedSlots);

        // The model's placement decides the backend's shape: the engine thread choice (0 = auto-sized from physical
        // cores) capped by the model's thread quota, and the model's core set pinned onto every pool worker.
        CpuBackend backend = new(config.Engine.Threads, modelConfig.Placement.CoreSet, modelConfig.Placement.ThreadQuota);
        GgufFile? file = null;
        TransformerModel? model = null;
        BlockPool? pool = null;
        GenerationScheduler? scheduler = null;
        try
        {
            file = GgufFile.Open(modelConfig.Path);
            // Profile-scoped tokenizer defaults applied at load (absent BOS/EOS keys) surface as structured
            // log lines on the host logger.
            GgufTokenizer tokenizer = GgufTokenizer.FromGguf(file, line => logger?.LogInformation("{TokenizerLog}", line));
            model = TransformerModel.Open(file, backend, reservedSlots);
            pool = new BlockPool(
                blockTokens,
                model.Plan.LayerCount,
                model.Plan.KvHeadCount,
                model.Plan.HeadDim,
                modelConfig.KvBlocks,
                backend);

            PrefixCache cache = new(pool);
            scheduler = new GenerationScheduler(model, pool, config.Engine.MaxBatchTokens, cache);

            // The chat template is parsed once here, at model load, and cached for the loaded instance's lifetime: no
            // per-request parse and no lazy first-request parse. An unparseable template fails model startup fast,
            // naming the model path plus the offending construct and its offset. The model's literal BOS/EOS pieces
            // are bound to the template's bos_token/eos_token context variables so the wrapper can report whether a
            // rendered prompt already carries the model BOS.
            MinjaChatTemplate? chatTemplate = null;
            TemplateCapabilities? chatTemplateCapabilities = null;
            if (file.Metadata.ChatTemplate is string source)
            {
                string bosToken = LiteralTokenPiece(tokenizer, tokenizer.BosId);
                string eosToken = LiteralTokenPiece(tokenizer, tokenizer.EosId);

                try
                {
                    chatTemplate = new MinjaChatTemplate(source, bosToken, eosToken);
                }
                catch (MinjaSyntaxException ex)
                {
                    logger?.LogError(
                        "model path={ModelPath} has an invalid chat template: construct={Construct} at offset={Position}",
                        modelConfig.Path,
                        ex.Construct,
                        ex.Position);

                    throw new InvalidOperationException(
                        $"model '{modelConfig.Name}' at '{modelConfig.Path}' has an invalid chat template: {ex.Message}",
                        ex);
                }

                // Capability facts are probed once at load so request-time tool-precedence decisions have them without
                // re-probing per request. Probing never throws: a probe that cannot render reports that capability absent.
                chatTemplateCapabilities = chatTemplate.DetectCapabilities();
            }

            ModelInfo info = new(
                modelConfig.Name,
                maxContext,
                modelConfig.KvBlocks,
                DominantQuant(file),
                model.Plan.Architecture,
                model.Plan.VocabSize);

            GenerationService service = new(file, backend, model, tokenizer, chatTemplate, chatTemplateCapabilities, logger, pool, scheduler, info, config);
            service._cacheForMetrics = cache; // the metrics bus reports cached blocks from the same cache the scheduler uses
            file = null;
            model = null;
            pool = null;
            scheduler = null;
            return service;
        }
        catch
        {
            scheduler?.Dispose();
            pool?.Dispose();
            model?.Dispose();
            file?.Dispose();
            backend.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Runs one generation to completion through the batch scheduler. The prompt is tokenized, prefilled in chunks,
    /// then decoded in shared steps until eos, the token budget, or a stop string. Prompt prefixes already computed by
    /// earlier requests are adopted from the prefix cache instead of recomputed, and the adopted token count is
    /// reported on the result as <see cref="GenerationResult.CachedPromptTokens"/>. Cancellation surfaces as
    /// <see cref="OperationCanceledException"/>.
    /// </summary>
    public async Task<GenerationResult> GenerateAsync(GenerationRequest request, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, typeof(GenerationService));

        if (!string.Equals(request.Model, Model.Id, StringComparison.Ordinal))
        {
            throw new ModelNotFoundException(request.Model);
        }

        cancellationToken.ThrowIfCancellationRequested();

        if (request.PresencePenalty is < -2f or > 2f)
        {
            throw new ArgumentException($"'presence_penalty' must be in [-2, 2], got {request.PresencePenalty}.", nameof(request));
        }

        if (request.FrequencyPenalty is < -2f or > 2f)
        {
            throw new ArgumentException($"'frequency_penalty' must be in [-2, 2], got {request.FrequencyPenalty}.", nameof(request));
        }

        if (request.TopLogprobs is < 0 or > 20)
        {
            throw new ArgumentException($"'top_logprobs' must be in [0, 20], got {request.TopLogprobs}.", nameof(request));
        }

        IReadOnlyDictionary<int, float>? logitBias = ResolveLogitBias(request.LogitBias);

        // Tool declarations and the choice are validated here too so programmatic callers meet the same rules as HTTP ones.
        // The pipeline is active only for chat requests that declare tools under a mode other than none: which tool surface
        // the request gets — the template's own rendering or LAIR's fenced block — is resolved once by capability
        // precedence, and detection reads fences out of the decoded text against the names this choice allows.
        ToolPipeline tools = ResolveToolPipeline(request);

        IReadOnlyCollection<string> callableNames = tools.Active
            ? tools.Choice.Mode == ToolChoiceMode.Function
                ? [tools.Choice.FunctionName!]
                : [.. tools.DeclaredTools!.Select(tool => tool.Name)]
            : [];

        IReadOnlyList<int> prompt = request.IsChat
            ? EncodeChat(request, tools, cancellationToken)
            : EncodeCompletion(request);

        if (prompt.Count >= Model.ContextLength)
        {
            throw new ContextLengthExceededException(prompt.Count, Model.ContextLength);
        }

        int maxTokens = Math.Min(request.MaxTokens, Model.ContextLength - prompt.Count);
        if (maxTokens <= 0)
        {
            // The clamp left no budget: the sequence generates nothing and reports the budget as spent.
            return new GenerationResult(string.Empty, prompt.Count, 0, "length");
        }

        // Monitor lifecycle keys on the caller's correlation id when one was given (the API host passes the same value it
        // answers with in x-lair-request-id); programmatic callers get a generated id so their stream rows still correlate.
        EngineMetricsBus? bus = _metrics;
        string monitorRequestId = !string.IsNullOrWhiteSpace(request.RequestId) ? request.RequestId! : Guid.NewGuid().ToString("N");
        RequestTracker? tracker = bus is null ? null : new(bus, monitorRequestId, Model.Id, request.IsChat ? "chat" : "completion", prompt.Count);

        List<string> pieces = [];
        string streamed = string.Empty;
        int decodedChars = 0;
        bool stopped = false;
        SequenceHandle? handle = null;

        // The stop signal is a cancellation the sequence observes at its next step boundary rather than a call through the
        // handle: the scheduler can invoke the streaming hook for the first sampled token before this call has its handle
        // back, and a stop literal seen in that token must still end the sequence.
        using CancellationTokenSource stopSignal = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

        // The stop signal is the model's declared EOT id when it has one (tokenizer.ggml.eot_token_id);
        // models without an EOT id keep today's EOS behavior.
        int stopId = StopId;

        ScheduledRequest scheduled = new()
        {
            PromptTokens = [.. prompt],
            MaxTokens = maxTokens,
            Sampling = new SamplingParams
            {
                Temperature = request.Temperature,
                TopK = request.TopK,
                TopP = request.TopP,
                Seed = request.Seed,
                PresencePenalty = request.PresencePenalty,
                FrequencyPenalty = request.FrequencyPenalty,
                LogitBias = logitBias,
            },
            EosId = stopId,
            LogProbTopN = request.LogProbs ? Math.Max(request.TopLogprobs, 0) : null,
            OnTokenId = tokenId =>
            {
                // Step-boundary tok/s sampling rides the hook that already runs for every generated token: one tick,
                // two array writes, and at most a throttled lifecycle publish.
                tracker?.ObserveToken();

                string piece = _tokenizer.Decode([tokenId]);
                if (piece.Length == 0)
                {
                    return;
                }

                pieces.Add(piece);
                if (request.Stream && request.OnToken is not null)
                {
                    request.OnToken(piece);
                }

                streamed += piece;
                int stopIndex = StopIndex(streamed, request.Stop);
                if (stopIndex >= 0)
                {
                    // The stop literal is already streamed; trim it out of the final text below.
                    decodedChars = stopIndex;
                    stopped = true;
                    stopSignal.Cancel();
                }
            },
        };

        try
        {
            handle = _scheduler.Submit(scheduled, stopSignal.Token);
        }
        catch (SchedulerFullException)
        {
            // The scheduler's capacity vocabulary is an engine-busy condition at this facade: surface it as the
            // API-level EngineBusyException rather than leaking a scheduling type to callers. No lifecycle frames were
            // published — a refused admission never entered the stream.
            throw new EngineBusyException();
        }

        if (tracker is RequestTracker admitted)
        {
            _monitoredRequests[admitted.RequestId] = admitted;
            admitted.PublishStart();
        }

        SequenceOutcome outcome;
        try
        {
            outcome = await handle.Completion.ConfigureAwait(false);
        }
        finally
        {
            // Safety net: a sequence that is still running when this call is interrupted must not hold its blocks.
            handle.Cancel();
        }

        string text = _tokenizer.Decode(outcome.GeneratedTokens, includeSpecial: false);
        string finishReason;
        if (stopped)
        {
            text = text[..Math.Min(decodedChars, text.Length)];
            finishReason = "stop";

            // The scheduler reports a stop-ended sequence as cancelled — only the facade sees it was really the stop
            // literal, so its tally feeds the metrics bus's finish-reason counts.
            _metrics?.RecordStopCompletion();
        }
        else if (outcome.FinishReason == FinishReason.Failed)
        {
            finishReason = "failed";
        }
        else if (outcome.FinishReason == FinishReason.Canceled)
        {
            finishReason = "cancelled";
        }
        else if (outcome.FinishReason == FinishReason.Eos)
        {
            finishReason = "eos";
        }
        else
        {
            finishReason = "length";
        }

        // The completion frame publishes before failure semantics surface: a failed or cancelled generation still gets
        // its final lifecycle row with the status it ended on, so stream state matches engine state.
        if (tracker is RequestTracker finished)
        {
            _monitoredRequests.TryRemove(finished.RequestId, out _);
            finished.Complete(finishReason, outcome.PrefillCachedTokens, finishReason);
        }

        if (finishReason == "failed")
        {
            throw new EngineBusyException();
        }

        if (finishReason == "cancelled")
        {
            // A sequence the scheduler cancelled without a stop literal is either the caller's token or the safety net.
            cancellationToken.ThrowIfCancellationRequested();
            throw new OperationCanceledException(cancellationToken);
        }

        // Recognized fences take over the result: the fences vanish from the visible text, the calls ride along as data,
        // and the finish reason becomes tool_calls no matter how decoding ended. Detection is unchanged by which tool
        // surface the prompt used: it reads fenced-JSON calls out of the decoded text whenever the pipeline is active,
        // whether LAIR injected the contract or the template rendered the tools itself.
        IReadOnlyList<ToolCall>? toolCalls = null;
        if (tools.Active)
        {
            ToolDetection detection = ToolCallDetector.Detect(text, callableNames);
            if (detection.Calls.Count > 0)
            {
                text = detection.VisibleText;
                finishReason = "tool_calls";
                toolCalls = detection.Calls;
            }
        }

        return new GenerationResult(
            text,
            outcome.PromptTokens,
            outcome.CompletionTokens,
            finishReason,
            outcome.PrefillCachedTokens,
            request.LogProbs ? outcome.LogProbs : null,
            toolCalls);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        // The metrics sampler notices disposal at its next tick and completes every open stream subscription on the
        // way out; it is not worth blocking engine teardown on one sampling delay, so this is deliberately unawaited.
        if (_metrics is not null)
        {
            ValueTask completion = _metrics.DisposeAsync();
            if (!completion.IsCompletedSuccessfully)
            {
                // The sampler swallows its own tick faults; this just keeps any late one from surfacing as an
                // unobserved task exception during teardown.
                completion.AsTask().ContinueWith(static faulted => _ = faulted.Exception, TaskContinuationOptions.OnlyOnFaulted);
            }

            _metrics = null;
        }

        // The scheduler releases every sequence's blocks before the pool they belong to goes away; the forward
        // engine's mmap view must be released before the file handle it maps.
        _scheduler.Dispose();
        _pool.Dispose();
        _model.Dispose();
        _file.Dispose();
        _backend.Dispose();
    }

    private IReadOnlyList<int> EncodeChat(GenerationRequest request, ToolPipeline tools, CancellationToken cancellationToken)
    {
        ChatPrompt prompt = BuildChatPrompt(request, tools, cancellationToken);

        // BOS policy: the template is rendered exactly, then the tokenizer's own BOS is suppressed only when the
        // rendered prompt already starts with the model BOS token. No blind stripping — when the template did not emit
        // the BOS, the model's add_bos_token policy still prefixes it.
        bool prefixBos = _tokenizer.AddBosToken && !prompt.RenderStartsWithBos;
        return prefixBos
            ? _tokenizer.EncodeWithBos(prompt.Text)
            : _tokenizer.Encode(prompt.Text);
    }

    // The full chat-prompt build the request path performs: validate the conversation, render the model's template,
    // and apply the resolved tool precedence — the template's own tool rendering or LAIR's fenced block, never both.
    private ChatPrompt BuildChatPrompt(GenerationRequest request, ToolPipeline tools, CancellationToken cancellationToken)
    {
        IReadOnlyList<Lair.Tokenizer.Templating.ChatMessage>? messages = request.Messages;
        if (messages is null || messages.Count == 0)
        {
            throw new ArgumentException(
                "a chat request requires a non-empty 'messages' list; send a completion request with a raw 'prompt' instead.",
                nameof(request));
        }

        if (_chatTemplate is null)
        {
            throw new InvalidOperationException(
                $"model '{Model.Id}' has no 'tokenizer.chat_template' in its GGUF metadata, so chat requests cannot be " +
                "rendered; serve a model with a chat template or send a completion request with a raw 'prompt'.");
        }

        ChatTemplateInput input = new()
        {
            Messages = [.. messages.Select(MapChatMessage)],
            AddGenerationPrompt = true,

            // DN-M32-7 precedence: when the template owns the tool surface, the declared tools enter the template
            // context and the template formats them; the fenced-injection path keeps them out of the context so
            // exactly one tool surface reaches the model.
            Tools = tools.TemplateTools is null ? null : [.. tools.TemplateTools.Select(TemplateToolValues.Of)],
        };

        ChatTemplateRenderResult render;
        try
        {
            render = _chatTemplate.Render(input, cancellationToken: cancellationToken);
        }
        catch (MinjaRenderException ex)
        {
            // A template that parsed but fails to render this request — a deliberate raise_exception, a limit, or a
            // cancelled render (the library reports cancellation as RenderLimitsException, never a raw
            // OperationCanceledException) — surfaces as a bounded, actionable template error naming the model and the
            // correlation id. The HTTP layer maps it to an invalid_request response.
            string requestId = string.IsNullOrWhiteSpace(request.RequestId) ? string.Empty : request.RequestId!;
            string detail = ex is RenderLimitsException limits ? $"{limits.Limit} limit ({ex.Construct})" : ex.Construct;
            _logger?.LogError(
                "model '{Model}' request_id={RequestId} chat template render failed ({Detail}): {Message}",
                Model.Id,
                requestId,
                detail,
                ex.Message);

            throw new TemplateRenderFailedException(Model.Id, requestId, ex.Message, ex);
        }

        // The tool block rides on the rendered text before encoding: the injection point is the prompt string, so every
        // template family receives the same instruction line, schema fences, and reply contract.
        string rendered = render.Prompt;
        if (tools.InjectedTools is not null)
        {
            rendered = ToolPromptComposer.Compose(rendered, tools.InjectedTools);
        }

        return new ChatPrompt(rendered, render.StartsWithBos);
    }

    // One built chat prompt: the exact string the tokenizer receives, and whether the template render already
    // carried the model BOS (the tokenizer's own BOS is suppressed only in that case).
    private sealed record ChatPrompt(string Text, bool RenderStartsWithBos);

    /// <summary>
    /// The exact chat-prompt build the request path performs for one chat request — tool validation, capability-driven
    /// tool precedence, template render, and fenced injection — returned as the prompt string exactly as it is
    /// tokenized. Hosts and tests observe the adapter's real prompt decision through this seam.
    /// </summary>
    internal string BuildChatPrompt(GenerationRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        return BuildChatPrompt(request, ResolveToolPipeline(request), cancellationToken).Text;
    }

    /// <summary>
    /// The request's resolved tool pipeline: the declared tools, the resolved choice, whether the pipeline is active
    /// for this request, and which tool surface the active pipeline uses — the tools the template renders itself or
    /// the tools LAIR injects as its fenced block. Exactly one of the two tool lists is non-null when active.
    /// </summary>
    private sealed record ToolPipeline(
        IReadOnlyList<ToolDefinition>? DeclaredTools,
        ToolChoiceSelection Choice,
        bool Active,
        IReadOnlyList<ToolDefinition>? TemplateTools,
        IReadOnlyList<ToolDefinition>? InjectedTools);

    private ToolPipeline ResolveToolPipeline(GenerationRequest request)
    {
        IReadOnlyList<ToolDefinition>? declaredTools = ValidateToolDeclarations(request.Tools);
        ToolChoiceSelection choice = ResolveToolChoice(request.ToolChoice, declaredTools);

        // ToolChoiceMode.None closes the pipeline before any precedence decision: no template tools, no injection.
        if (!request.IsChat || declaredTools is not { Count: > 0 } || choice.Mode == ToolChoiceMode.None)
        {
            return new ToolPipeline(declaredTools, choice, false, null, null);
        }

        return ShouldInjectFencedTools(_chatTemplateCapabilities, Volatile.Read(ref _current).Models[0].ForceFencedTools)
            ? new ToolPipeline(declaredTools, choice, true, null, declaredTools)
            : new ToolPipeline(declaredTools, choice, true, declaredTools, null);
    }

    // DN-M32-7 tool precedence: a loaded template that renders tools owns the tool surface for that request. LAIR's
    // fenced block is the fallback for a template without tool support (and for a model with no chat template at all),
    // and the model's force_fenced_tools override selects it even for a tool-capable template.
    private static bool ShouldInjectFencedTools(TemplateCapabilities? capabilities, bool forceOverride) =>
        forceOverride || capabilities is null || !capabilities.SupportsTools;

    private static ChatMessage MapChatMessage(Lair.Tokenizer.Templating.ChatMessage message) =>
        new() { Role = message.Role, Content = MinjaValue.Of(message.Content) };

    // The literal vocabulary text of a special token id, or empty when the id is unset. This is the string bound to the
    // template's bos_token/eos_token context variables and compared against a rendered prompt for BOS suppression.
    private static string LiteralTokenPiece(GgufTokenizer tokenizer, int tokenId) =>
        tokenId >= 0 ? tokenizer.Decode([tokenId], includeSpecial: true) : string.Empty;

    private IReadOnlyList<int> EncodeCompletion(GenerationRequest request)
    {
        if (request.Prompt is null)
        {
            throw new ArgumentException(
                "a completion request requires a 'prompt'; set 'is_chat' with 'messages' to use the model's chat template.",
                nameof(request));
        }

        return _tokenizer.AddBosToken ? _tokenizer.EncodeWithBos(request.Prompt) : _tokenizer.Encode(request.Prompt);
    }

    // logit_bias keys are token strings or numeric token ids. A numeric key names its id directly; a string
    // key resolves through the served tokenizer and must encode to exactly one vocabulary token. Anything
    // unresolvable, out of vocabulary, or out of the [-100, 100] value range is rejected naming the key.
    private IReadOnlyDictionary<int, float>? ResolveLogitBias(IReadOnlyDictionary<string, float>? bias)
    {
        if (bias is null || bias.Count == 0)
        {
            return null;
        }

        Dictionary<int, float> resolved = [];
        foreach ((string key, float value) in bias)
        {
            int tokenId;
            if (int.TryParse(key, NumberStyles.Integer, CultureInfo.InvariantCulture, out int numericId))
            {
                tokenId = numericId;
                if (tokenId < 0 || tokenId >= Model.VocabSize)
                {
                    throw new ArgumentException(
                        $"'logit_bias' key '{key}' is not a token id of this model's vocabulary.",
                        nameof(bias));
                }
            }
            else
            {
                IReadOnlyList<int> encoded = _tokenizer.Encode(key);
                if (encoded.Count != 1)
                {
                    throw new ArgumentException(
                        $"'logit_bias' key '{key}' does not resolve to a single token of this model's vocabulary.",
                        nameof(bias));
                }

                tokenId = encoded[0];
            }

            if (value is < -100f or > 100f)
            {
                throw new ArgumentException(
                    $"'logit_bias' value for '{key}' must be in [-100, 100], got {value}.",
                    nameof(bias));
            }

            resolved[tokenId] = value;
        }

        return resolved;
    }

    private static int StopIndex(string text, IReadOnlyList<string>? stopStrings)
    {
        if (stopStrings is null)
        {
            return -1;
        }

        int best = -1;
        foreach (string stop in stopStrings)
        {
            if (stop.Length == 0)
            {
                continue;
            }

            int index = text.IndexOf(stop, StringComparison.Ordinal);
            if (index >= 0 && (best < 0 || index < best))
            {
                best = index;
            }
        }

        return best;
    }

    // A tool declaration must carry a non-empty unique name; an absent or empty list declares nothing and the pipeline stays closed.
    private static IReadOnlyList<ToolDefinition>? ValidateToolDeclarations(IReadOnlyList<ToolDefinition>? tools)
    {
        if (tools is null || tools.Count == 0)
        {
            return null;
        }

        HashSet<string> names = new(StringComparer.Ordinal);
        foreach (ToolDefinition tool in tools)
        {
            if (tool.Name.Length == 0)
            {
                throw new ArgumentException("'tools' contains a declaration with an empty 'name'.", nameof(tools));
            }

            if (!names.Add(tool.Name))
            {
                throw new ArgumentException($"'tools' declares the tool '{tool.Name}' more than once.", nameof(tools));
            }
        }

        return tools;
    }

    // A null choice means auto. A forced function must name one of the declared tools: its fences are then the only ones detection accepts.
    private static ToolChoiceSelection ResolveToolChoice(ToolChoiceSelection? choice, IReadOnlyList<ToolDefinition>? declaredTools)
    {
        if (choice is null || choice.Mode == ToolChoiceMode.Auto)
        {
            return ToolChoiceSelection.Auto;
        }

        if (choice.Mode == ToolChoiceMode.None)
        {
            return ToolChoiceSelection.None;
        }

        string forced = choice.FunctionName ?? throw new ArgumentException(
            "'tool_choice' of mode function must name a tool.", nameof(choice));

        bool declared = false;
        foreach (ToolDefinition tool in declaredTools ?? [])
        {
            if (string.Equals(tool.Name, forced, StringComparison.Ordinal))
            {
                declared = true;
                break;
            }
        }

        if (!declared)
        {
            throw new ArgumentException(
                $"'tool_choice' names '{forced}', which is not one of the declared tools.", nameof(choice));
        }

        return ToolChoiceSelection.Function(forced);
    }

    // The file's quantization is the format most of its tensors are stored in; mixed files (quantized
    // weights with F16 embeddings) report the weights, not the embeddings.
    private static string DominantQuant(GgufFile file)
    {
        return file.Tensors.Tensors
            .GroupBy(tensor => tensor.Format)
            .MaxBy(group => group.Count())!
            .Key
            .ToString();
    }
}
