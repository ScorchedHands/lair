namespace Lair.Templating.Minja;

/// <summary>
/// Thrown when a parsed template cannot be evaluated or rendered. The message names the failing construct
/// and carries its 0-based character offset in the template source when known. Built-in filters, tests and
/// function calls are reported here until the filters item implements them.
/// </summary>
public sealed class MinjaRenderException : Exception
{
    /// <summary>Name of the failing construct.</summary>
    public string Construct { get; }

    /// <summary>0-based character offset of the construct in the template source, or -1 when not applicable.</summary>
    public int Position { get; }

    public MinjaRenderException(string construct, int position, string detail)
        : base($"minja render error: {detail} at offset {position}")
    {
        Construct = construct;
        Position = position;
    }

    internal static MinjaRenderException UnsupportedExpression(string construct, int position) =>
        new(construct, position, $"'{construct}' is not supported by the expression evaluator yet");
}
