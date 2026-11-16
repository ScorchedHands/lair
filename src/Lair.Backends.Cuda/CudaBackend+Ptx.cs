using System.Text;

namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // Address arithmetic stays in computed-register form throughout: this driver seam rejects [base+reg] load forms, so
    // each element address is mad'd from its index, widened, shifted by the element size, and added to the base. Half
    // and bf16 weights widen through integer bit math into a per-thread local slot reloaded as f32; cvt.f16 forms are
    // rejected outright (CUDA_ERROR_INVALID_PTX) on this seam. Accumulation is plain mul then add in ascending column
    // order - the same chain the scalar CPU reference uses, so one device result matches one host dot bit for bit.
    private const string PtxHeader =
        ".version 7.0\n" +
        ".target sm_50\n" +
        ".address_size 64\n";

    private static readonly byte[] PtxBytes = Encoding.UTF8.GetBytes(
        PtxHeader + "\n" + GemvF32Ptx + "\n" + GemvF16Ptx + "\n" + GemvBf16Ptx + "\n" + GemvQ4KPtx + "\0");
}
