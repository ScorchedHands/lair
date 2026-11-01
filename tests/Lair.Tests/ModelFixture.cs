namespace Lair.Tests;

/// <summary>
/// Locates the real-model fixture shipped in the repository. The test host's working directory varies per runner, so
/// walk up from the assembly location instead of relying on a relative path.
/// </summary>
internal static class ModelFixture
{
    public const string SmolLmRelativePath = "models/SmolLM-135M-Instruct-F16.gguf";

    public static string? Resolve()
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        for (int depth = 0; directory is not null && depth < 8; depth++)
        {
            string candidate = Path.Combine(directory.FullName, SmolLmRelativePath);
            if (File.Exists(candidate))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        return null;
    }
}
