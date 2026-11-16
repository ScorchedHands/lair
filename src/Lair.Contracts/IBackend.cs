using System.Buffers;

namespace Lair.Contracts;

/// <summary>
/// Compute backend: aligned arena management plus dense kernels. Kernels consume packed
/// weights through <see cref="WeightRef"/> and never materialize dequantized copies.
/// </summary>
public interface IBackend : IDisposable
{
    string DeviceId { get; }

    /// <summary>
    /// The weight tier this backend's kernels read weight bytes from while they compute: system memory where a weight
    /// source's host regions live, or persistent device memory this backend owns and copies each tensor into once. A
    /// route segment naming the device-resident tier is answered only by a backend reporting it; a backend reporting
    /// system memory answers host-resident segments on any device it runs.
    /// </summary>
    WeightTier WeightTier { get; }

    MemoryHandle AllocateAligned(long bytes, int alignment);

    void FreeAligned(MemoryHandle handle);

    /// <summary>
    /// Opens a persistent device-resident KV mirror over the caller's host arena for the device-backed block range the
    /// request names. The host arena stays the copy attention reads; the mirror keeps a persistent device copy of exactly
    /// that block range, with the host side pinned as the staging path dirty blocks cross on flush. A backend with no
    /// device-resident KV storage refuses the request naming its device id, the requested tier, and the resource rather
    /// than answering with a host-only stand-in.
    /// </summary>
    /// <exception cref="NotSupportedException">This backend holds no device-resident storage for the named resource at the requested tier.</exception>
    IKvDeviceMirror OpenKvMirror(KvMirrorRequest request);

    /// <summary>
    /// output[0..rows) = w.Bytes-as-matrix (rows x cols) times x. The weight matrix is the
    /// first rows columns of each row consumed as-is; x must be exactly cols long and output
    /// at least rows long. Results are deterministic regardless of internal parallelism.
    /// </summary>
    void Gemv(ReadOnlySpan<float> x, WeightRef w, Span<float> output);

    /// <summary>
    /// Batched GEMV: <paramref name="count"/> activation vectors (each exactly cols long) stored consecutively in
    /// <paramref name="inputs"/> are dotted against the same weight matrix in one pass over its packed bytes.
    /// outputs[batch * rows + row] receives the dot of inputs[batch] with weight row <paramref name="row"/>, exactly
    /// as a single-vector <see cref="Gemv"/> call would write output[row]. For every row and batch the result is
    /// bit-identical to the single-vector path and deterministic regardless of internal parallelism.
    /// </summary>
    void GemvBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs);

    /// <summary>
    /// Blocked matmul over dequantized weight tiles: <paramref name="count"/> activation vectors (each exactly cols
    /// long) stored consecutively in <paramref name="inputs"/> multiply the same weight matrix, with each weight tile
    /// dequantized once and consumed by every activation vector of the batch. Outputs use the same
    /// <c>outputs[batch * rows + row]</c> layout as <see cref="GemvBatch"/>. Every output element accumulates in
    /// exactly the order and with exactly the operand association the single-vector <see cref="Gemv"/> reference uses
    /// for that element, so the result is raw-byte identical to it and deterministic regardless of internal
    /// parallelism. Only formats this backend reports as supported reach this kernel; an unsupported format throws.
    /// </summary>
    void GemmBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs);

    /// <summary>
    /// Whether this backend's blocked matmul executes for weights in <paramref name="format"/>. Formats it reports as
    /// unsupported are refused rather than run through some other route.
    /// </summary>
    bool SupportsGemm(QuantFormat format);

    /// <summary>
    /// Dequantizes one weight row (w.Bytes-as-matrix row <paramref name="row"/>, cols = Dims[^1]) into an FP32
    /// buffer. The dequant math matches the fused GEMV kernels exactly: F32 rows copy, F16 rows expand halves,
    /// Q8_0/Q4_K rows unpack block scales and quant codes with the same arithmetic. Output must be at least cols long.
    /// </summary>
    void DequantRow(WeightRef w, int row, Span<float> output);

    float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b);

    void Axpy(float alpha, ReadOnlySpan<float> x, Span<float> y);

    void Fill(Span<float> y, float v);

    void Scale(Span<float> y, float s);
}
