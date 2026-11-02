using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;
using Lair.Tokenizer;
using Lair.Tokenizer.Templating;

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

            scheduler = new GenerationScheduler(model, pool, config.Engine.MaxBatchTokens, new PrefixCache(pool));

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

        IReadOnlyList<int> prompt = request.IsChat
            ? EncodeChat(request)
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
            },
            EosId = _tokenizer.EosId,
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
                    handle?.Cancel();
                }
            },
        };

        try
        {
            handle = _scheduler.Submit(scheduled, cancellationToken);
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

        if (stopped)
        {
            string text = _tokenizer.Decode(outcome.GeneratedTokens, includeSpecial: false);
            return new GenerationResult(
                text[..Math.Min(decodedChars, text.Length)],
                outcome.PromptTokens,
                outcome.CompletionTokens,
                "stop",
                outcome.PrefillCachedTokens);
        }

        switch (outcome.FinishReason)
        {
            case FinishReason.Failed:
                throw new EngineBusyException();

            case FinishReason.Canceled:
                throw new OperationCanceledException(cancellationToken);

            case FinishReason.Eos:
                return new GenerationResult(
                    _tokenizer.Decode(outcome.GeneratedTokens, includeSpecial: false),
                    outcome.PromptTokens,
                    outcome.CompletionTokens,
                    "eos",
                    outcome.PrefillCachedTokens);

            default:
                return new GenerationResult(
                    _tokenizer.Decode(outcome.GeneratedTokens, includeSpecial: false),
                    outcome.PromptTokens,
                    outcome.CompletionTokens,
                    "length",
                    outcome.PrefillCachedTokens);
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        // The scheduler releases every sequence's blocks before the pool they belong to goes away; the forward
        // engine's mmap view must be released before the file handle it maps.
        _scheduler.Dispose();
        _pool.Dispose();
        _model.Dispose();
        _file.Dispose();
        _backend.Dispose();
    }

    private IReadOnlyList<int> EncodeChat(GenerationRequest request)
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

        return _tokenizer.AddBosToken
            ? _tokenizer.EncodeWithBos(_chatTemplate.Render(messages, addGenerationPrompt: true))
            : _tokenizer.Encode(_chatTemplate.Render(messages, addGenerationPrompt: true));
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
