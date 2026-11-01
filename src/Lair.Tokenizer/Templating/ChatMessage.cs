namespace Lair.Tokenizer.Templating;

/// <summary>
/// One conversation turn handed to a chat template renderer.
/// </summary>
public sealed record ChatMessage(string Role, string Content);
