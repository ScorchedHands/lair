using System.Globalization;
using Lair.Api.Monitoring;
using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;
using Lair.Tokenizer;
using Lair.Tokenizer.Templating;
using Microsoft.Extensions.Logging;

namespace Lair.Api;

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
    private readonly ChatTemplateRenderer? _chatTemplate;
    private readonly BlockPool _pool;
    private readonly GenerationScheduler _scheduler;
    private EngineMetricsBus? _metrics;

    /// <summary>The metrics bus attached by the API host, if any — null for hosts that never attach one.</summary>
    internal EngineMetricsBus? Metrics => _metrics;

    private bool _disposed;

    private GenerationService(
        GgufFile file,
        CpuBackend backend,
        TransformerModel model,
        GgufTokenizer tokenizer,
        ChatTemplateRenderer? chatTemplate,
        BlockPool pool,
        GenerationScheduler scheduler,
        ModelInfo info)
    {
        _file = file;
        _backend = backend;
        _model = model;
        _tokenizer = tokenizer;
        _chatTemplate = chatTemplate;
        _pool = pool;
        _scheduler = scheduler;
        Model = info;
    }

    /// <summary>The served model's advertised facts; requests must name <see cref="ModelInfo.Id"/>.</summary>
    public ModelInfo Model { get; }

    /// <summary>Vocabulary size of the served model; token ids at or above it have no vocabulary text.</summary>
    public int VocabSize => Model.VocabSize;

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
        _metrics = bus;
        return bus;
    }

    // The scheduler owns its prefix cache privately; the service keeps what it passed so metrics can report cached blocks.
    private PrefixCache? _cacheForMetrics;

    /// <summary>Decodes one token id through the served model's tokenizer, special tokens included so control pieces round-trip.</summary>
    public string DecodeToken(int tokenId) => _tokenizer.Decode([tokenId], includeSpecial: true);

    /// <summary>
    /// Opens the first model declared in the configuration: GGUF container, mmap-backed weights through the
    /// forward engine, tokenizer, chat template, a KV pool sized by the model entry, a prefix cache over that pool,
    /// and the batch scheduler over them. The prefix cache lives exactly as long as the pool, so each opened
    /// service starts with an empty cache. The effective context is the smaller of the configured max context and
    /// the token slots the KV pool can hold.
    /// </summary>
    public static GenerationService Open(LairConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Models.Count == 0)
        {
            throw new InvalidOperationException(
                "the configuration declares no models: add at least one entry under 'models:' before opening the engine.");
        }

        ModelConfig modelConfig = config.Models[0];
        int blockTokens = config.Engine.BlockSize;

        // The KV pool holds KvBlocks * blockTokens token slots; the context cannot exceed what it can back.
        int maxContext = Math.Min(modelConfig.MaxContextTokens, modelConfig.KvBlocks * blockTokens);

        CpuBackend backend = new(config.Engine.Threads);
        GgufFile? file = null;
        TransformerModel? model = null;
        BlockPool? pool = null;
        GenerationScheduler? scheduler = null;
        try
        {
            file = GgufFile.Open(modelConfig.Path);
            GgufTokenizer tokenizer = GgufTokenizer.FromGguf(file);
            model = TransformerModel.Open(file, backend, maxContext);
            pool = new BlockPool(
                blockTokens,
                model.Desc.LayerCount,
                model.Desc.KvHeadCount,
                model.Desc.HeadDim,
                modelConfig.KvBlocks,
                backend);

            PrefixCache cache = new(pool);
            scheduler = new GenerationScheduler(model, pool, config.Engine.MaxBatchTokens, cache);

            ChatTemplateRenderer? chatTemplate = file.Metadata.ChatTemplate is string source
                ? ChatTemplateRenderer.Parse(source)
                : null;

            ModelInfo info = new(
                modelConfig.Name,
                maxContext,
                modelConfig.KvBlocks,
                DominantQuant(file),
                model.Desc.Architecture,
                model.Desc.VocabSize);

            GenerationService service = new(file, backend, model, tokenizer, chatTemplate, pool, scheduler, info);
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
        // The pipeline is active only for chat requests that declare tools under a mode other than none: injection happens on
        // the rendered prompt, and detection reads fences out of the decoded text against the names this choice allows.
        IReadOnlyList<ToolDefinition>? declaredTools = ValidateToolDeclarations(request.Tools);
        ToolChoiceSelection toolChoice = ResolveToolChoice(request.ToolChoice, declaredTools);
        bool toolsActive = request.IsChat && declaredTools is { Count: > 0 } && toolChoice.Mode != ToolChoiceMode.None;

        IReadOnlyCollection<string> callableNames = toolsActive
            ? toolChoice.Mode == ToolChoiceMode.Function
                ? [toolChoice.FunctionName!]
                : [.. declaredTools!.Select(tool => tool.Name)]
            : [];

        IReadOnlyList<int> prompt = request.IsChat
            ? EncodeChat(request, toolsActive ? declaredTools : null)
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

        List<string> pieces = [];
        string streamed = string.Empty;
        int decodedChars = 0;
        bool stopped = false;
        SequenceHandle? handle = null;

        // The stop signal is a cancellation the sequence observes at its next step boundary rather than a call through the
        // handle: the scheduler can invoke the streaming hook for the first sampled token before this call has its handle
        // back, and a stop literal seen in that token must still end the sequence.
        using CancellationTokenSource stopSignal = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

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
            EosId = _tokenizer.EosId,
            LogProbTopN = request.LogProbs ? Math.Max(request.TopLogprobs, 0) : null,
            OnTokenId = tokenId =>
            {
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
            // API-level EngineBusyException rather than leaking a scheduling type to callers.
            throw new EngineBusyException();
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
        else
        {
            finishReason = outcome.FinishReason switch
            {
                FinishReason.Failed => throw new EngineBusyException(),
                FinishReason.Canceled => throw new OperationCanceledException(cancellationToken),
                FinishReason.Eos => "eos",
                _ => "length",
            };
        }

        // Recognized fences take over the result: the fences vanish from the visible text, the calls ride along as data,
        // and the finish reason becomes tool_calls no matter how decoding ended.
        IReadOnlyList<ToolCall>? toolCalls = null;
        if (toolsActive)
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

    private IReadOnlyList<int> EncodeChat(GenerationRequest request, IReadOnlyList<ToolDefinition>? injectedTools)
    {
        IReadOnlyList<ChatMessage>? messages = request.Messages;
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

        // The tool block rides on the rendered text before encoding: the injection point is the prompt string, so every
        // template family receives the same instruction line, schema fences, and reply contract.
        string rendered = _chatTemplate.Render(messages, addGenerationPrompt: true);
        if (injectedTools is not null)
        {
            rendered = ToolPromptComposer.Compose(rendered, injectedTools);
        }

        return _tokenizer.AddBosToken
            ? _tokenizer.EncodeWithBos(rendered)
            : _tokenizer.Encode(rendered);
    }

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
