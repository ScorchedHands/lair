namespace Lair.Api;

/// <summary>
/// What a served model advertises: the id requests must name, the effective context window, the KV pool size,
/// the dominant tensor quantization, the architecture, and the vocabulary size.
/// </summary>
public sealed record ModelInfo(
    string Id,
    int ContextLength,
    int KvBlocks,
    string Quant,
    string Architecture,
    int VocabSize);
