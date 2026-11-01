namespace Lair.Kv;

/// <summary>Raised when a paged KV pool has no free blocks left.</summary>
public sealed class KvFullException : Exception
{
    public KvFullException()
        : base("KV block pool is full.")
    {
    }

    public KvFullException(string message)
        : base(message)
    {
    }

    public KvFullException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
