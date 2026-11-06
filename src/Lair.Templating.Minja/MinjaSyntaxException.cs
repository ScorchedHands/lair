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

    /// <summary>
    /// Parse-time resource guard: the template is well-formed enough to name but exceeds a parser limit
    /// (template size or parser depth). It stays a syntax error so malformed-source handling and
    /// limit handling share one taxonomy; <see cref="Construct"/> names the guard.
    /// </summary>
    internal static MinjaSyntaxException LimitExceeded(string limit, int position) =>
        new($"minja limit exceeded: {limit} at offset {position}", limit, position);

    /// <summary>Document-level parse guard (for example template size) where no construct offset applies.</summary>
    internal static MinjaSyntaxException LimitExceeded(string limit, string detail) =>
        new($"minja limit exceeded: {limit}: {detail}", limit, 0);
}
