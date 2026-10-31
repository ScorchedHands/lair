namespace Lair.Core.Contracts;

/// <summary>
/// Describes a model file. <see cref="Load"/> reads the file's metadata and returns a
/// logical tensor table; it does not materialize tensor bytes. Retrieval of tensor
/// payload is the job of an <see cref="IWeightSource"/>.
/// </summary>
public interface IModelLoader
{
    LogicalTensorTable Load(string modelPath);
}
