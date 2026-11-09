namespace Lair.Http;

/// <summary>
/// What a served model advertises: the id requests must name, the effective context window, the KV pool size,
/// the dominant tensor quantization, the architecture, the vocabulary size, and the tool output parser resolved for
/// this model at load. The parser field defaults to the configuration's <c>auto</c> key so records built without a
/// resolved plan stay well-formed; every engine-opened record carries its resolved choice instead.
/// </summary>
public sealed record ModelInfo(
    string Id,
    int ContextLength,
    int KvBlocks,
    string Quant,
    string Architecture,
    int VocabSize,
    string ToolParser = "auto");
