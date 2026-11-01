using System.Buffers;

namespace Lair.Core.Contracts;

/// <summary>
/// Compute backend: aligned arena management plus dense kernels. Kernels consume packed
/// weights through <see cref="WeightRef"/> and never materialize dequantized copies.
/// </summary>
public interface IBackend : IDisposable
{
    string DeviceId { get; }

    MemoryHandle AllocateAligned(long bytes, int alignment);

    void FreeAligned(MemoryHandle handle);

    /// <summary>
    /// output[0..rows) = w.Bytes-as-matrix (rows x cols) times x. The weight matrix is the
    /// first rows columns of each row consumed as-is; x must be exactly cols long and output
    /// at least rows long. Results are deterministic regardless of internal parallelism.
    /// </summary>
    void Gemv(ReadOnlySpan<float> x, WeightRef w, Span<float> output);

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
