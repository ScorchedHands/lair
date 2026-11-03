namespace Lair.Contracts;

/// <summary>
/// Resolves a logical tensor to its raw bytes.
/// Resolution is plan-time: callers resolve each tensor once and cache the returned
/// spans for the duration of a hot loop, so no virtual dispatch happens per access.
/// Implementations must keep the backing memory alive and pinned-stable until
/// <see cref="IDisposable.Dispose"/> is called; returned spans remain valid for the
/// lifetime of the source.
/// </summary>
public interface IWeightSource : IDisposable
{
    ReadOnlySpan<byte> Resolve(LogicalTensor tensor);
}
