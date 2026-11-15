namespace Lair.Host.Config;

/// <summary>
/// Raised when a model fetch refuses: a file is missing at the pinned revision, a download is incomplete, a
/// fetched file's bytes do not match its pinned hash, or a lock cannot be held. Every message names the repo,
/// the resolved revision, and the file it was refusing, and a hash refusal names both the expected and the
/// actual sha256 so an operator can see exactly which pin the bytes disagreed with.
/// </summary>
public sealed class ModelFetchException : Exception
{
    public ModelFetchException(string message)
        : base(message)
    {
    }

    public ModelFetchException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
