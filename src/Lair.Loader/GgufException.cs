namespace Lair.Loader;

public sealed class GgufException : Exception
{
    public GgufException(string message)
        : base(message)
    {
    }

    public GgufException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
