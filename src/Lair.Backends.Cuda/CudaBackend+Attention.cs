namespace Lair.Backends.Cuda;

using System.Numerics;

using Lair.Contracts;

public sealed partial class CudaBackend
{
    // Device steps this instance has opened and not yet released. Each one owns persistent device allocations inside
    // this instance's context, so the backend tracks every outstanding step and frees them before the context goes
    // away — a caller that never disposed its step still gets its allocations returned at backend dispose.
    private readonly List<CudaDeviceStep> _deviceSteps = [];

    /// <summary>
    /// The capability fact every CUDA device-step posture answers from: this seam carries the on-device attention
    /// family — the rotation, the KV write, and the causal attention walk, in both the private-arena and the pool
    /// block-layout (paged) forms — as embedded PTX entry points resolved beside the GEMV and blocked-matmul family,
    /// so a dense transformer step can compose them against buffers the workspace owns instead of staging scores,
    /// softmax, or KV reads back over PCIe per layer. The capability is static so the posture is inspectable without
    /// opening a device, and it names kernels this seam actually has: attention composition is not claimed for a
    /// posture that only carries weight passes.
    /// </summary>
    public static bool HasDeviceStepKernel => true; // rope_pairs, rope_half, kv_store, kv_store_paged, attn_causal, and attn_causal_paged each resolve a function handle here

    /// <summary>
    /// Opens one on-device transformer step: persistent device buffers for the request's positions, layers, and staged
    /// window, plus the angle tables and KV arena the attention family reads. The arena starts unwritten — a step may
    /// only read positions its own KV write has covered — and the allocations live with this instance until the step
    /// is disposed or this instance is.
    /// </summary>
    public CudaDeviceStep OpenDeviceStep(CudaDeviceStepRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        ThrowIfDisposed();
        return new CudaDeviceStep(this, request);
    }

    /// <summary>
    /// Opens one on-device transformer step whose KV storage is a pool's device store rather than a private arena: the
    /// step's KV write and attention walk run against the pool's resident blocks through the pool's block layout, which
    /// is the device-side KV read/write posture a <c>vram</c> or <c>split</c> residency declares for a GPU step. The
    /// store must be one the mirror hands out in the device-step posture and match the request's positions, layers, and
    /// kv region; a store that is a host-read mirror's copy, or one whose geometry the request does not describe,
    /// refuses naming both sides rather than walking a layout the store does not hold.
    /// </summary>
    public CudaDeviceStep OpenDeviceStep(CudaDeviceStepRequest request, IKvDeviceMirror kvStore)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(kvStore);
        ThrowIfDisposed();
        return new CudaDeviceStep(this, request, kvStore);
    }

    /// <summary>
    /// Opens the composed attention window an engine's device prefill posture runs each prompt layer through: the step
    /// family above, answered in the model layout the request states rather than in whatever head staging the walk needs.
    /// A grouped-query group the committed shift mapping already divides answers with a plain device step; a group it
    /// cannot express — nine query heads over three kv-heads is that shape today — runs through padded head staging over
    /// the same kernels, so no model layout reaches a mapping the walk does not hold and no kernel here changes. The KV
    /// store arrives exactly as <see cref="OpenDeviceStep(CudaDeviceStepRequest, IKvDeviceMirror)"/> takes it: bound when
    /// one is supplied (the device-step posture of a resident pool), private-arena when none is, with the same geometry
    /// refusals naming both sides either way. The caller owns what this opens and disposes it after its last forward.
    /// </summary>
    public IDeviceAttentionStep OpenAttentionStep(CudaDeviceStepRequest request, IKvDeviceMirror? kvStore = null)
    {
        ArgumentNullException.ThrowIfNull(request);
        ThrowIfDisposed();

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(request.HeadCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(request.KvHeadCount);

        if (request.HeadCount % request.KvHeadCount != 0)
        {
            throw new ArgumentException(
                $"composed step request HeadCount {request.HeadCount} is not a whole multiple of KvHeadCount {request.KvHeadCount}: " +
                "grouped-query attention maps consecutive query heads onto one kv-head.",
                nameof(request));
        }

        int groupReal = request.HeadCount / request.KvHeadCount;
        if (BitOperations.IsPow2((uint)groupReal))
        {
            return new CudaDeviceStep(this, request, kvStore);
        }

        // The padded shape: each real head keeps its kv-head's group slot and the partial groups' tails pad out to whole power-of-two groups. Validate through that layout so every other geometry rule — head dimension, arena size, window width — answers exactly as a plain step request would.
        int stagedHeads = checked(request.KvHeadCount * (int)BitOperations.RoundUpToPowerOf2((uint)groupReal));

        return new CudaGroupedAttentionStep(new CudaDeviceStep(this, request with { HeadCount = stagedHeads }, kvStore), request.HeadCount, request.KvHeadCount);
    }

    /// <summary>Device steps currently open on this instance.</summary>
    public int OpenDeviceStepCount => _deviceSteps.Count;

    internal IntPtr RopePairsKernel => _ropePairs;

    // Whether this instance has released its context: a step that still holds buffers must refuse rather than reach
    // for a context its backend already gave back.
    internal bool IsDisposed => _disposed;

    internal IntPtr RopeHalfKernel => _ropeHalf;

    internal IntPtr KvStoreKernel => _kvStore;

    internal IntPtr KvStorePagedKernel => _kvStorePaged;

    internal IntPtr AttentionKernel => _attnCausal;

    internal IntPtr AttentionPagedKernel => _attnCausalPaged;

    // The two quantized paged-attention handles a device step resolves from a quantized store's combo at open: q8_0/q8_0
    // and q8_0/q4_0 each stand for exactly one entry point, so the step picks the handle by combo once and never branches
    // on format per position.
    internal IntPtr AttentionQ8Q8Kernel => _attnCausalQ8Q8;

    internal IntPtr AttentionK8V4Kernel => _attnCausalK8V4;

    internal void RegisterDeviceStep(CudaDeviceStep step) => _deviceSteps.Add(step);

    internal void UnregisterDeviceStep(CudaDeviceStep step) => _deviceSteps.Remove(step);

    // Folds one staged-window input crossing or one attention output crossing into the runtime transfer tallies, the
    // way a GEMV folds its own activation input and kernel output: a composed step counts the window it staged and
    // the attention result it read back, and nothing the kernels moved between devices enters the tally.
    internal void AccountStepTransfer(long bytes) => AccountRuntimeCopy(bytes);

    // The device-measured timing pair every step call brackets around its launches, so a step's kernel busy time lands
    // in the same tally a GEMV launch's does. The closing stamp lands after the call's last launch is enqueued, so the
    // elapsed answer covers that call's launches as one device-measured region.
    internal bool BeginStepKernelTiming() => BeginKernelTiming();

    internal void StampStepKernelEnd() => StampKernelEnd();

    internal void FoldStepKernelElapsed() => FoldKernelElapsed();

    // Releases every outstanding step, newest first: each frees its own allocations while this instance's context is
    // still current, and removes itself from this list as it goes, so draining by last entry terminates on its own.
    private void ReleaseDeviceSteps()
    {
        while (_deviceSteps.Count > 0)
        {
            _deviceSteps[^1].Dispose();
        }
    }
}
