namespace Lair.Tests;

/// <summary>
/// Locates the real-model fixture shipped in the repository. The test host's working directory varies per runner, so
/// walk up from the assembly location instead of relying on a relative path.
/// </summary>
internal static class ModelFixture
{
    public const string SmolLmRelativePath = "models/SmolLM-135M-Instruct-F16.gguf";

    public static string? Resolve() => ResolveRepoRelative(SmolLmRelativePath);

    /// <summary>
    /// Locates any repository-relative path (model files, recorded fixtures) the same way: walk up from the
    /// assembly location until the candidate exists.
    /// </summary>
    public static string? ResolveRepoRelative(string repoRelativePath)
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        for (int depth = 0; directory is not null && depth < 8; depth++)
        {
            string candidate = Path.Combine(directory.FullName, repoRelativePath);
            if (File.Exists(candidate))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        return null;
    }
}
