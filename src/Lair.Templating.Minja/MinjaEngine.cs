namespace Lair.Templating.Minja;

/// <summary>
/// Managed template engine entry point. Parsing lives in <see cref="MinjaParser"/>, expression evaluation and
/// template rendering (expressions plus if/for/set/macro/call/filter statements) in <see cref="MinjaEvaluator"/>;
/// these methods are stable conveniences over both. The assembly carries no dependency on the rest of LAIR or
/// on any native code.
/// </summary>
public static class MinjaEngine
{
    /// <summary>Parses a chat template into its AST. Throws <see cref="MinjaSyntaxException"/> for unsupported or malformed syntax.</summary>
    public static TemplateNode Parse(string template) => MinjaParser.Parse(template);

    /// <summary>Evaluates a single expression string against an optional context.</summary>
    public static MinjaValue Evaluate(string expression, IReadOnlyDictionary<string, MinjaValue>? context = null) =>
        MinjaEvaluator.Evaluate(expression, context);

    /// <summary>Renders a template of text, comments, expression outputs and statements against an optional context.</summary>
    public static string Render(string template, IReadOnlyDictionary<string, MinjaValue>? context = null) =>
        MinjaEvaluator.Render(template, context);
}
