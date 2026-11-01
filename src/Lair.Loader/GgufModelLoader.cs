using Lair.Core.Contracts;

namespace Lair.Loader;

/// <summary>
/// Reads the logical tensor table of a GGUF model file without materializing
/// tensor payload bytes; pair with <see cref="GgufMmapWeightSource"/> to access
/// the weights.
/// </summary>
public sealed class GgufModelLoader : IModelLoader
{
    public LogicalTensorTable Load(string modelPath) =>
        GgufFile.Open(modelPath).Tensors;
}
