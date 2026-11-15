namespace Lair.Host.Config;

/// <summary>
/// The credential seam a hub call consults before it reaches out: the access token an operator supplies for a gated
/// repo, or null when none was supplied. It exists so the hub seam can attach a credential to its own requests while
/// the fetch path, the pack manifest, the CLI, and the logs never carry the value — a test runs the same seam against
/// a stand-in provider and never reads the real environment.
/// </summary>
public interface IHubTokenProvider
{
    /// <summary>
    /// The access token to present to the hub, or null when the operator supplied none. A null, empty, or
    /// whitespace-only value is reported as null so an unset variable never sends a meaningless credential.
    /// </summary>
    string? GetToken();
}
