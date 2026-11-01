using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Sampling;
using Lair.Tokenizer;
using Lair.Tokenizer.Templating;

namespace Lair.Api;

/// <summary>
/// Single-model generation facade: opens one configured GGUF model (weights, forward engine, tokenizer,
/// chat template, paged KV pool) and runs <see cref="GenerationRequest"/>s against it. The engine forwards
/// one sequence at a time; a second concurrent request fails immediately with <see cref="EngineBusyException"/>.
/// </summary>
public sealed class GenerationService : IDisposable
{
    private readonly GgufFile _file;
    private readonly CpuBackend _backend;
    private readonly TransformerModel _model;
    private readonly GgufTokenizer _tokenizer;
    private readonly ChatTemplateRenderer? _chatTemplate;
    private readonly BlockPool _pool;

    // 0 = idle, 1 = a generation is in flight.
    private int _inFlight;

    private bool _disposed;

    private GenerationService(
        GgufFile file,
        CpuBackend backend,
        TransformerModel model,
        GgufTokenizer tokenizer,
        ChatTemplateRenderer? chatTemplate,
        BlockPool pool,
        ModelInfo info)
    {
        _file = file;
        _backend = backend;
        _model = model;
        _tokenizer = tokenizer;
        _chatTemplate = chatTemplate;
        _pool = pool;
        Model = info;
    }

    /// <summary>The served model's advertised facts; requests must name <see cref="ModelInfo.Id"/>.</summary>
    public ModelInfo Model { get; }

    /// <summary>
    /// Opens the first model declared in the configuration: GGUF container, mmap-backed weights through the
    /// forward engine, tokenizer, chat template, and a KV pool sized by the model entry. The effective context
    /// is the smaller of the configured max context and the token slots the KV pool can hold.
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

            GenerationService service = new(file, backend, model, tokenizer, chatTemplate, pool, info);
            file = null;
            model = null;
            pool = null;
            return service;
        }
        catch
        {
            pool?.Dispose();
            model?.Dispose();
            file?.Dispose();
            backend.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Runs one generation to completion. The prompt is tokenized, prefilled, then decoded token by token
    /// until eos, the token budget, or a stop string. Cancellation surfaces as <see cref="OperationCanceledException"/>.
    /// </summary>
    public async Task<GenerationResult> GenerateAsync(GenerationRequest request, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, typeof(GenerationService));

        if (!string.Equals(request.Model, Model.Id, StringComparison.Ordinal))
        {
            throw new ModelNotFoundException(request.Model);
        }

        if (Interlocked.CompareExchange(ref _inFlight, 1, 0) != 0)
        {
            throw new EngineBusyException();
        }

        try
        {
            return await Task.Run(() => Generate(request, cancellationToken), CancellationToken.None)
                .ConfigureAwait(false);
        }
        finally
        {
            Volatile.Write(ref _inFlight, 0);
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        // SequenceKv instances hand out of the pool, so the pool must outlive them; the forward engine's
        // mmap view must be released before the file handle it maps.
        _pool.Dispose();
        _model.Dispose();
        _file.Dispose();
        _backend.Dispose();
    }

    private GenerationResult Generate(GenerationRequest request, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        IReadOnlyList<int> prompt = request.IsChat
            ? EncodeChat(request)
            : EncodeCompletion(request);

        if (prompt.Count >= Model.ContextLength)
        {
            throw new ContextLengthExceededException(prompt.Count, Model.ContextLength);
        }

        Sampler sampler = new(new SamplingParams
        {
            Temperature = request.Temperature,
            TopK = request.TopK,
            TopP = request.TopP,
            Seed = request.Seed,
        });

        string finishReason = "length";
        List<int> generated = [];
        string streamed = string.Empty;
        List<string> pieces = [];
        int decodedChars = 0;

        using SequenceKv kv = new(_pool);
        _model.Prefill(prompt, kv);

        int maxTokens = Math.Min(request.MaxTokens, Model.ContextLength - prompt.Count);

        // Prefill already forwarded the final prompt token: its logits are the first next-token distribution.
        ReadOnlySpan<float> logits = _model.LastLogits();
        for (int i = 0; i < maxTokens; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int token = sampler.Sample(logits);

            if (token == _tokenizer.EosId)
            {
                finishReason = "eos";
                break; // eos ends the turn and is never forwarded into the sequence.
            }

            generated.Add(token);
            logits = _model.DecodeNext(token, kv);

            string piece = _tokenizer.Decode([token]);
            if (piece.Length > 0)
            {
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
                    finishReason = "stop";
                    break;
                }

                decodedChars = streamed.Length;
            }
        }

        string text = _tokenizer.Decode(generated, includeSpecial: false);
        if (finishReason == "stop")
        {
            text = text[..Math.Min(decodedChars, text.Length)];
        }

        return new GenerationResult(text, prompt.Count, generated.Count, finishReason);
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
