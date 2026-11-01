namespace Lair.Tokenizer.Templating;

/// <summary>
/// Thrown when a chat template uses syntax outside the supported Mini Jinja subset. The message names the offending
/// construct and its character position in the template source.
/// </summary>
public sealed class TemplateSyntaxException : Exception
{
    public TemplateSyntaxException(string message, int position)
        : base($"{message} at offset {position}")
    {
        Position = position;
    }

    /// <summary>Character offset of the offending construct in the template source.</summary>
    public int Position { get; }
}
