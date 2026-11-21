namespace Lair.Loader;

/// <summary>
/// The GGUF metadata keys llama.cpp uses to record a file's membership in a
/// split set. `split.no` is the zero-based position of the shard, `split.count`
/// the total number of shards, and `split.tensors.count` the total tensor count
/// declared across the whole set. A split model whose shards carry these keys
/// is validated against them at assembly; a set that carries none is assembled
/// from the file-name shape alone.
/// </summary>
public static class GgufSplitKeys
{
    public const string No = "split.no";
    public const string Count = "split.count";
    public const string TensorsCount = "split.tensors.count";
}
