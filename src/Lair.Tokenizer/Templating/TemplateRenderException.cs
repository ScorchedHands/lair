namespace Lair.Tokenizer.Templating;

/// <summary>
/// Thrown when a template that parsed successfully fails to evaluate: unknown variables, type mismatches, out-of-range
/// indexes, or values that cannot be stringified or JSON-encoded. Parse-time failures use
/// <see cref="TemplateSyntaxException"/>; this type covers evaluation-time failures only.
/// </summary>
public sealed class TemplateRenderException : Exception
{
    public TemplateRenderException(string message)
        : base(message)
    {
    }
}
