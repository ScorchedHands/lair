using Lair.Contracts;

namespace Lair.Loader;

/// <summary>
/// Reads the logical tensor table of a GGUF model file without materializing
/// tensor payload bytes; pair with <see cref="GgufMmapWeightSource"/> to access
/// the weights. A path naming any member of a split GGUF set (file names of the
/// shape <c>-NNNNN-of-NNNNN</c>) loads the whole split as one logical tensor
/// table — sibling shards are discovered in the same directory — while an
/// ordinary single-file path is read exactly as before.
/// </summary>
public sealed class GgufModelLoader : IModelLoader
{
    public LogicalTensorTable Load(string modelPath) =>
        GgufFile.Open(modelPath).Tensors;
}
