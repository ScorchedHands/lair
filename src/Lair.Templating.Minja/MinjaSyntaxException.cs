namespace Lair.Templating.Minja;

/// <summary>
/// Thrown when a template uses syntax outside minja's supported grammar or is malformed. The message names the
/// offending construct and its 0-based character offset in the template source.
/// </summary>
public sealed class MinjaSyntaxException : Exception
{
    /// <summary>Name of the offending construct.</summary>
    public string Construct { get; }

    /// <summary>0-based character offset of the construct in the template source.</summary>
    public int Position { get; }

    public MinjaSyntaxException(string construct, int position)
        : base($"unsupported minja construct '{construct}' at offset {position}")
    {
        Construct = construct;
        Position = position;
    }

    private MinjaSyntaxException(string message, string construct, int position)
        : base(message)
    {
        Construct = construct;
        Position = position;
    }

    internal static MinjaSyntaxException Malformed(int position, string detail) =>
        new($"invalid syntax: {detail} at offset {position}", "syntax", position);
}
