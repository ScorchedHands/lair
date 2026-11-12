namespace Lair.Contracts;

/// <summary>
/// Descriptor for a quantized tensor type: the ggml type id, its reference name,
/// the concrete <see cref="QuantFormat"/> the kernels dispatch on, and the block
/// geometry (elements and bytes per ggml quantization block) needed to size and
/// validate packed weight buffers.
/// </summary>
/// <param name="TypeId">ggml type id as stored in a GGUF tensor info.</param>
/// <param name="Name">reference name of the type, e.g. "Q4_K".</param>
/// <param name="Format">concrete quant format the kernels switch on.</param>
/// <param name="BlockSize">ggml blck_size: elements covered by one quantization block.</param>
/// <param name="TypeSize">ggml type_size: bytes occupied by one quantization block.</param>
public readonly record struct QuantDescriptor(
    uint TypeId,
    string Name,
    QuantFormat Format,
    int BlockSize,
    int TypeSize);
