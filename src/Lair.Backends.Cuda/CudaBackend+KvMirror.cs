using Lair.Contracts;

namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // Device KV mirrors this instance has opened and not yet released. Each mirror pins a host arena it does not own,
    // so the backend tracks every outstanding one and releases its pin plus its device allocation before the context
    // goes away — a pool that never disposed its mirror still gets both resources returned at backend dispose.
    private readonly List<CudaKvMirror> _kvMirrors = [];

    /// <summary>
    /// Opens the device KV mirror for a declared block range: pins the requested host arena with the driver, allocates
    /// persistent device storage sized to exactly one whole block per device-backed id, and answers with the seam that
    /// crosses dirty blocks host-to-device on flush. The arena stays owned by whoever handed it in; this instance only
    /// records the pin and the allocation for their lifetimes. No attention kernel is opened or implied here — a mirror
    /// holds bytes resident on the device, nothing more.
    /// </summary>
    public IKvDeviceMirror OpenKvMirror(KvMirrorRequest request)
    {
        ThrowIfDisposed();
        return new CudaKvMirror(this, request);
    }

    // Serializes this instance's driver work against every other call on the same gate and re-binds the retained
    // primary context for the calling thread, exactly as a kernel launch does. A mirror uses it around its copies so a
    // flush never interleaves with weight residency or another mirror's crossing.
    internal void EnterDeviceWork()
    {
        Monitor.Enter(_gate);
        SetCurrentContext();
    }

    internal void ExitDeviceWork() => Monitor.Exit(_gate);

    internal void RegisterKvMirror(CudaKvMirror mirror) => _kvMirrors.Add(mirror);
    internal void UnregisterKvMirror(CudaKvMirror mirror) => _kvMirrors.Remove(mirror);

    // Releases every outstanding mirror, newest first: each dispose flushes its pending crossings while the arenas it
    // pins are still alive (the pools own them), frees its device storage, and unregisters its pin. A mirror removes
    // itself from this list as it goes, so draining by last entry terminates on its own.
    private void ReleaseKvMirrors()
    {
        while (_kvMirrors.Count > 0)
        {
            _kvMirrors[^1].Dispose();
        }
    }

    /// <summary>Device KV mirrors currently open on this instance.</summary>
    public int OpenKvMirrorCount => _kvMirrors.Count;
}
