namespace Lair.Host.Config;

/// <summary>
/// The production credential seam: it reads the hub access token straight from the process environment and nowhere
/// else — never a config file, a pack manifest, a command-line argument, or a log. LAIR's own variable is consulted
/// first, then the conventional Hugging Face variable, so an operator already exporting the conventional one needs no
/// extra setup. A whitespace-only value counts as unset rather than a credential to send.
/// </summary>
public sealed class EnvironmentHubTokenProvider : IHubTokenProvider
{
    /// <summary>The LAIR-specific environment variable that carries the hub access token.</summary>
    public const string LairVariable = "LAIR_HF_TOKEN";

    /// <summary>The conventional Hugging Face variable accepted as a fallback for operators already exporting it.</summary>
    public const string ConventionalVariable = "HUGGING_FACE_HUB_TOKEN";

    /// <inheritdoc />
    public string? GetToken()
    {
        string? token = Environment.GetEnvironmentVariable(LairVariable);
        if (string.IsNullOrWhiteSpace(token))
        {
            token = Environment.GetEnvironmentVariable(ConventionalVariable);
        }

        return string.IsNullOrWhiteSpace(token) ? null : token;
    }
}
