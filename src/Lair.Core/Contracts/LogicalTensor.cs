namespace Lair.Core.Contracts;

public sealed record LogicalTensor(
    string Name,
    TensorShape Shape,
    QuantFormat Format,
    long Offset,
    long SizeBytes);

public sealed class LogicalTensorTable
{
    private readonly Dictionary<string, LogicalTensor> _byName;

    public LogicalTensorTable(IReadOnlyList<LogicalTensor> tensors)
    {
        Tensors = tensors;
        _byName = new Dictionary<string, LogicalTensor>(StringComparer.Ordinal);
        foreach (LogicalTensor tensor in tensors)
        {
            _byName[tensor.Name] = tensor;
        }
    }

    public IReadOnlyList<LogicalTensor> Tensors { get; }

    public LogicalTensor? Find(string name) =>
        _byName.TryGetValue(name, out LogicalTensor? tensor) ? tensor : null;
}
