using System.Text.Json;
using System.Text.Json.Serialization;

namespace Lair.Host.Config;

/// <summary>
/// The pin record one model pack directory carries in its <c>manifest.json</c>: every fetched model with the repo and
/// resolved revision its bytes came from, and each file's LAIR-computed sha256, size, and fetch time. The manifest is
/// what loadability gates against — a config entry's declared source plus this record decide whether the bytes on disk
/// are the pinned bytes — and it survives config hot-reloads untouched: a live edit never rewrites a pin, and adopting
/// a new revision is an operator act behind a restart.
/// </summary>
public sealed record ModelManifest
{
    /// <summary>The fetched models this pack carries, keyed by the model name a config entry serves.</summary>
    [property: JsonPropertyName("models")] public required IReadOnlyList<ModelManifestEntry> Models { get; init; }
}

/// <summary>One fetched model in the pack: the repo it came from, the commit sha its declared revision resolved to at fetch, and the files carrying its bytes.</summary>
public sealed record ModelManifestEntry
{
    /// <summary>The model name this entry pins, matching the config entry it serves.</summary>
    [property: JsonPropertyName("model")] public required string Model { get; init; }

    /// <summary>The repo the files were fetched from, as the source block declared it.</summary>
    [property: JsonPropertyName("repo")] public required string Repo { get; init; }

    /// <summary>The commit sha the declared revision resolved to at fetch — the pin load gates on, whether the config named a branch or a sha.</summary>
    [property: JsonPropertyName("resolved_revision")] public required string ResolvedRevision { get; init; }

    /// <summary>The files this model's bytes are carried by, in fetch order.</summary>
    [property: JsonPropertyName("files")] public required IReadOnlyList<ModelManifestFile> Files { get; init; }
}

/// <summary>One fetched file in the pack: its path/name beside the repo, the sha256 LAIR computed over its bytes, its size, and when it landed.</summary>
public sealed record ModelManifestFile
{
    /// <summary>The file's path/name within the repo, kept as a readable name in the pack directory.</summary>
    [property: JsonPropertyName("path")] public required string Path { get; init; }

    /// <summary>The sha256 LAIR computed over the fetched bytes — the value loadability verifies against.</summary>
    [property: JsonPropertyName("sha256")] public required string Sha256 { get; init; }

    /// <summary>The byte size of the fetched file; null (absent) records no size.</summary>
    [property: JsonPropertyName("size")] public long? Size { get; init; }

    /// <summary>When this file landed in the pack, recorded as an instant.</summary>
    [property: JsonPropertyName("fetched_at")] public required DateTimeOffset FetchedAt { get; init; }
}

/// <summary>
/// Reads and writes one pack directory's <c>manifest.json</c>. The file is the pack's pin record, so a missing manifest
/// is no manifest (null) while a present-but-unreadable one fails naming the file and the reason — the store never
/// guesses a pin, and a save is a whole-file rewrite, never a partial edit.
/// </summary>
public static class ModelManifestStore
{
    /// <summary>The manifest file name every pack directory carries.</summary>
    public const string FileName = "manifest.json";

    private static readonly JsonSerializerOptions FileOptions = new() { WriteIndented = true };

    /// <summary>The manifest recorded in <paramref name="packDirectory"/>, or null when the pack carries no manifest file yet.</summary>
    public static ModelManifest? Load(string packDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(packDirectory);

        string path = Path.Combine(packDirectory, FileName);
        if (!File.Exists(path))
        {
            return null;
        }

        try
        {
            using FileStream file = File.OpenRead(path);
            return JsonSerializer.Deserialize<ModelManifest>(file, FileOptions);
        }
        catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
        {
            throw new ConfigException($"Model manifest '{path}' could not be read as JSON: {ex.Message}", ex);
        }
    }

    /// <summary>Writes <paramref name="manifest"/> to <paramref name="packDirectory"/>'s <c>manifest.json</c>, creating the directory when the pack is new.</summary>
    public static void Save(string packDirectory, ModelManifest manifest)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(packDirectory);
        ArgumentNullException.ThrowIfNull(manifest);

        Directory.CreateDirectory(packDirectory);

        string path = Path.Combine(packDirectory, FileName);
        try
        {
            using FileStream file = File.Create(path);
            JsonSerializer.Serialize(file, manifest, FileOptions);
        }
        catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
        {
            throw new ConfigException($"Model manifest '{path}' could not be written as JSON: {ex.Message}", ex);
        }
    }
}
