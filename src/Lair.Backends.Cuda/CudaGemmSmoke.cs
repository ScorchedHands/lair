using System.Runtime.InteropServices;
using System.Text;

using Lair.Contracts;

namespace Lair.Backends.Cuda;

/// <summary>
/// Row-major FP32 GEMM (C = A * B, A is MxK, B is KxN) executed through the CUDA Driver API against nvcuda.dll,
/// with the kernel supplied as embedded PTX JIT-compiled by the driver.
/// </summary>
public static class CudaGemmSmoke
{
    public const int MatrixM = 3;
    public const int MatrixN = 4;
    public const int MatrixK = 5;

    private const string KernelName = "gemm_fp32";

    private const string Ptx =
        ".version 7.0\n" +
        ".target sm_50\n" +
        ".address_size 64\n" +
        "\n" +
        ".visible .entry gemm_fp32(\n" +
        "    .param .u64 a_ptr,\n" +
        "    .param .u64 b_ptr,\n" +
        "    .param .u64 c_ptr,\n" +
        "    .param .u32 m,\n" +
        "    .param .u32 n,\n" +
        "    .param .u32 k\n" +
        ") {\n" +
        "    .reg .u64 a, b, c, addr;\n" +
        "    .reg .u32 row, col, kk, m_, n_, k_, tid, t0;\n" +
        "    .reg .pred p;\n" +
        "    .reg .f32 sum, av, bv;\n" +
        "\n" +
        "    ld.param.u64 a, [a_ptr];\n" +
        "    ld.param.u64 b, [b_ptr];\n" +
        "    ld.param.u64 c, [c_ptr];\n" +
        "    ld.param.u32 m_, [m];\n" +
        "    ld.param.u32 n_, [n];\n" +
        "    ld.param.u32 k_, [k];\n" +
        "\n" +
        "    mov.u32 tid, %ctaid.x;\n" +
        "    mov.u32 t0, %ntid.x;\n" +
        "    mul.lo.u32 tid, tid, t0;\n" +
        "    mov.u32 t0, %tid.x;\n" +
        "    add.u32 tid, tid, t0;\n" +
        "    rem.u32 col, tid, n_;\n" +
        "    div.u32 row, tid, n_;\n" +
        "    setp.ge.u32 p, row, m_;\n" +
        "    @p bra done;\n" +
        "\n" +
        "    mov.f32 sum, 0f00000000;\n" +
        "    mov.u32 kk, 0;\n" +
        "loop:\n" +
        "    setp.ge.u32 p, kk, k_;\n" +
        "    @p bra store;\n" +
        "    mad.lo.u32 t0, row, k_, kk;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, a, addr;\n" +
        "    ld.global.f32 av, [addr];\n" +
        "    mad.lo.u32 t0, kk, n_, col;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, b, addr;\n" +
        "    ld.global.f32 bv, [addr];\n" +
        "    fma.rn.f32 sum, av, bv, sum;\n" +
        "    add.u32 kk, kk, 1;\n" +
        "    bra loop;\n" +
        "store:\n" +
        "    mad.lo.u32 t0, row, n_, col;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, c, addr;\n" +
        "    st.global.f32 [addr], sum;\n" +
        "done:\n" +
        "    ret;\n" +
        "}\n";

    public static float[] RunGemmSmoke()
    {
        int m = MatrixM;
        int n = MatrixN;
        int k = MatrixK;

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
        RunGemm(a, b, c, m, n, k);
        return c;
    }

    public static void RunGemm(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int n, int k)
    {
        if (a.Length != m * k || b.Length != k * n || c.Length != m * n)
        {
            throw new ArgumentException($"GEMM shape mismatch: expected A {m}x{k}, B {k}x{n}, C {m}x{n}.");
        }

        TensorShape shape = new([m, k]);
        LogicalTensor tensor = new("gemm_a", shape, QuantFormat.F32, 0, shape.ElementCount * sizeof(float));
        if (tensor.SizeBytes != (long)a.Length * sizeof(float))
        {
            throw new InvalidOperationException($"Contract tensor size {tensor.SizeBytes} does not match GEMM input {a.Length} FP32 elements.");
        }

        CudaException.ThrowIf("cuInit", CudaNative.cuInit(0));

        CudaException.ThrowIf("cuDeviceGetCount", CudaNative.cuDeviceGetCount(out int deviceCount));
        if (deviceCount < 1)
        {
            throw new CudaException("cuDeviceGetCount", CudaResult.NoDevice);
        }

        CudaException.ThrowIf("cuDeviceGet", CudaNative.cuDeviceGet(out int device, 0));
        CudaException.ThrowIf("cuDevicePrimaryCtxRetain", CudaNative.cuDevicePrimaryCtxRetain(out IntPtr context, device));
        try
        {
            CudaException.ThrowIf("cuCtxSetCurrent", CudaNative.cuCtxSetCurrent(context));

            byte[] ptx = Encoding.UTF8.GetBytes(Ptx + "\0");
            IntPtr module = 0;
            ulong deviceA = 0;
            ulong deviceB = 0;
            ulong deviceC = 0;
            try
            {
                unsafe
                {
                    fixed (byte* ptxPtr = ptx)
                    {
                        CudaException.ThrowIf("cuModuleLoadData", CudaNative.cuModuleLoadData(out module, ptxPtr));
                    }
                }

                CudaException.ThrowIf("cuModuleGetFunction", CudaNative.cuModuleGetFunction(out IntPtr function, module, KernelName));

                nuint bytesA = (nuint)(m * k) * sizeof(float);
                nuint bytesB = (nuint)(k * n) * sizeof(float);
                nuint bytesC = (nuint)(m * n) * sizeof(float);
                CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out deviceA, bytesA));
                CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out deviceB, bytesB));
                CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out deviceC, bytesC));

                unsafe
                {
                    fixed (float* aPtr = a)
                    {
                        CudaException.ThrowIf("cuMemcpyHtoD_v2", CudaNative.cuMemcpyHtoD(deviceA, (byte*)aPtr, bytesA));
                    }
                    fixed (float* bPtr = b)
                    {
                        CudaException.ThrowIf("cuMemcpyHtoD_v2", CudaNative.cuMemcpyHtoD(deviceB, (byte*)bPtr, bytesB));
                    }

                    uint threads = (uint)(m * n);
                    void** kernelParams = stackalloc void*[6];
                    kernelParams[0] = &deviceA;
                    kernelParams[1] = &deviceB;
                    kernelParams[2] = &deviceC;
                    kernelParams[3] = &m;
                    kernelParams[4] = &n;
                    kernelParams[5] = &k;
                    CudaException.ThrowIf("cuLaunchKernel", CudaNative.cuLaunchKernel(function, 1, 1, 1, threads, 1, 1, 0, IntPtr.Zero, kernelParams, null));

                    CudaException.ThrowIf("cuCtxSynchronize", CudaNative.cuCtxSynchronize());

                    fixed (float* cPtr = c)
                    {
                        CudaException.ThrowIf("cuMemcpyDtoH_v2", CudaNative.cuMemcpyDtoH((byte*)cPtr, deviceC, bytesC));
                    }
                }

                CudaException.ThrowIf("cuMemFree_v2", CudaNative.cuMemFree(deviceA));
                CudaException.ThrowIf("cuMemFree_v2", CudaNative.cuMemFree(deviceB));
                CudaException.ThrowIf("cuMemFree_v2", CudaNative.cuMemFree(deviceC));
                deviceA = 0;
                deviceB = 0;
                deviceC = 0;
                CudaException.ThrowIf("cuModuleUnload", CudaNative.cuModuleUnload(module));
                module = IntPtr.Zero;
            }
            finally
            {
                if (deviceC != 0)
                {
                    CudaNative.cuMemFree(deviceC);
                }
                if (deviceB != 0)
                {
                    CudaNative.cuMemFree(deviceB);
                }
                if (deviceA != 0)
                {
                    CudaNative.cuMemFree(deviceA);
                }
                if (module != IntPtr.Zero)
                {
                    CudaNative.cuModuleUnload(module);
                }
            }
        }
        finally
        {
            CudaException.ThrowIf("cuDevicePrimaryCtxRelease", CudaNative.cuDevicePrimaryCtxRelease(device));
        }
    }
}
