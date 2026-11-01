namespace Lair.Models;

/// <summary>
/// Raised when a GGUF file cannot be interpreted as a supported model architecture: unsupported or missing
/// metadata keys, absent required tensors, or tensor shapes that contradict the declared dimensions.
/// </summary>
public sealed class ModelException : Exception
{
    public ModelException(string message)
        : base(message)
    {
    }

    public ModelException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
