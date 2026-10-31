using Lair.Core.Contracts;

namespace Lair.Loader;

public sealed class GGUFMetadataStub : IModelLoader
{
    public LogicalTensorTable Load(string modelPath) =>
        throw new NotImplementedException("GGUF metadata loading is not implemented.");
}
