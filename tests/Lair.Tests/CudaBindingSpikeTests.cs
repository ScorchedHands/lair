using System.Runtime.InteropServices;

using Lair.Backends.Cuda;

namespace Lair.Tests;

/// <summary>
/// Driver-API smoke: the embedded PTX GEMM must reach the device through cuModuleLoadData, cuLaunchKernel, and the
/// pinned copies, and return values a managed CPU reference agrees with. A missing driver or device fails loudly
/// naming the nvcuda.dll seam instead of falling back to CPU.
/// </summary>
[Trait("Category", "Kernel")]
[Trait("Category", "Cuda")]
public sealed class CudaBindingSpikeTests
{
    private const float Tolerance = 1e-4f;

    [Fact]
    public void CudaGemmSmoke_MatchesCpuReference()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam smoke runs on Windows against nvcuda.dll.");
        }

        float[] cuda;
        try
        {
            cuda = CudaGemmSmoke.RunGemmSmoke();
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }

        Assert.NotNull(cuda);
        Assert.Equal(CudaGemmSmoke.MatrixM * CudaGemmSmoke.MatrixN, cuda.Length);

        float[] cpu = RunCpuGemm();
        Assert.Equal(cpu.Length, cuda.Length);
        for (int i = 0; i < cpu.Length; i++)
        {
            Assert.True(MathF.Abs(cpu[i] - cuda[i]) < Tolerance, $"C[{i}] CPU {cpu[i]} vs CUDA {cuda[i]}");
        }
    }

    private static float[] RunCpuGemm()
    {
        int m = CudaGemmSmoke.MatrixM;
        int n = CudaGemmSmoke.MatrixN;
        int k = CudaGemmSmoke.MatrixK;

        float[] a = new float[m * k];
        float[] b = new float[k * n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                a[i * k + j] = (i * k + j + 1) * 0.5f;
            }
        }
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < n; j++)
            {
                b[i * n + j] = (i * n + j + 1) * 0.25f;
            }
        }

        float[] c = new float[m * n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float sum = 0f;
                for (int p = 0; p < k; p++)
                {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        return c;
    }
}
