using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Lair.Contracts;

namespace Lair.Backends.Cpu;

/// <summary>
/// Shared-operand FP32 dot batching: several input vectors dotted against one common vector in a single call, where the
/// backend's own chain-group budget decides how many accumulator chains share one pass over that operand. Every output is
/// produced by the same accumulation order an independent <see cref="CpuBackend.Dot"/> call for that input would use — the
/// eight-lane chunk walk with its FMA association, a scalar tail folded into its own accumulator, and the same lane reduction
/// before that tail — so batching only removes redundant reads of the shared operand and per-call dispatch. Grouping is over
/// independent chains, never inside one accumulator, so every output lands bit-for-bit on its individual dot at any posture.
/// </summary>
public sealed unsafe partial class CpuBackend : IDenseSharedInputDot
{
    public bool TryDotBatchSharedInput(ReadOnlySpan<float> inputs, int count, ReadOnlySpan<float> common, Span<float> outputs)
    {
        if (count < 2 || common.Length == 0)
        {
            return false; // one dot has no operand to share with anything: the caller's own Dot call already covers it.
        }

        CpuKernelTuning tuning = _tuning; // sampled per call like every other kernel here, so a quota applied elsewhere takes effect from this call on.
        int chainGroup = Math.Min(tuning.SharedInputDotChainGroup, CpuKernelTuning.SharedInputDotChainGroupMax);
        if (chainGroup <= 0)
        {
            return false; // the resolved posture keeps shared-operand dots individual: no batched chain kernel runs.
        }

        long requiredInputs = (long)count * common.Length;
        if (inputs.Length < requiredInputs)
        {
            throw new ArgumentException(
                $"A batch of {count} input vectors against a {common.Length}-wide shared operand needs {requiredInputs} floats.", nameof(inputs));
        }

        if (outputs.Length < count)
        {
            throw new ArgumentException($"One output slot per dotted input vector: outputs hold {outputs.Length}, the batch has {count}.", nameof(outputs));
        }

        int len = common.Length;

        fixed (float* pIn = inputs)
        fixed (float* pCommon = common)
        fixed (float* pOut = outputs)
        {
            // A batch wider than the posture's budget splits into consecutive groups of it. Chains are independent, so which group a chain
            // sits in never changes its accumulation order — the split is scheduling, not arithmetic.
            for (int firstChain = 0; firstChain < count; firstChain += chainGroup)
            {
                int chains = Math.Min(chainGroup, count - firstChain);

                if (_useAvx2)
                {
                    DotChainsCore(pIn + (long)firstChain * len, pCommon, len, chains, pOut + firstChain);
                }
                else
                {
                    for (int b = 0; b < chains; b++)
                    {
                        pOut[firstChain + b] = DotScalar(pIn + (long)(firstChain + b) * len, pCommon, len); // the scalar posture's own reference dot, one output at a time.
                    }
                }
            }
        }

        return true;
    }

    // One chain group held entirely in registers: each chain's accumulator is a local of this method, so the shared operand is loaded once per chunk and stays hot while
    // every chain takes its own input load and FMA. Keeping chains in stack scratch instead costs a store and reload per chain per chunk — more than the re-reads batching
    // removes. Arms 2..8 cover exactly the counts this call can produce, since a group is at most SharedInputDotChainGroupMax wide; each arm walks its own accumulators
    // through ascending chunks with DotCore's operand association, keeps one scalar tail per chain, and reduces before adding that tail — bit-for-bit an independent Dot
    // per chain. The default arm re-enters the single-vector kernel directly for any count outside those arms, which is also a plain individual dot per chain.
    private static void DotChainsCore(float* inputs, float* common, int len, int chains, float* outputs)
    {
        switch (chains)
        {
            case 2:
            {
                Vector256<float> c0 = Vector256<float>.Zero;
                Vector256<float> c1 = Vector256<float>.Zero;

                int i = 0;
                for (; i + 8 <= len; i += 8)
                {
                    Vector256<float> sharedChunk = Avx.LoadVector256(common + i); // one load of the shared operand feeds every chain in it.
                    c0 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + i), sharedChunk, c0);
                    c1 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + len + i), sharedChunk, c1);
                }

                float t0 = 0f; // each chain's tail is its own scalar accumulator, added after the reduction exactly as DotCore adds one.
                float t1 = 0f;
                for (; i < len; i++)
                {
                    float shared = common[i];
                    t0 += inputs[i] * shared;
                    t1 += inputs[len + i] * shared;
                }

                outputs[0] = HSum(c0) + t0; // the reduction happens before that chain's tail joins it.
                outputs[1] = HSum(c1) + t1;
                return;
            }
            case 3:
            {
                Vector256<float> c0 = Vector256<float>.Zero;
                Vector256<float> c1 = Vector256<float>.Zero;
                Vector256<float> c2 = Vector256<float>.Zero;

                int i = 0;
                for (; i + 8 <= len; i += 8)
                {
                    Vector256<float> sharedChunk = Avx.LoadVector256(common + i); // one load of the shared operand feeds every chain in it.
                    c0 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + i), sharedChunk, c0);
                    c1 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + len + i), sharedChunk, c1);
                    c2 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 2L * len + i), sharedChunk, c2);
                }

                float t0 = 0f; // each chain's tail is its own scalar accumulator, added after the reduction exactly as DotCore adds one.
                float t1 = 0f;
                float t2 = 0f;
                for (; i < len; i++)
                {
                    float shared = common[i];
                    t0 += inputs[i] * shared;
                    t1 += inputs[len + i] * shared;
                    t2 += inputs[2L * len + i] * shared;
                }

                outputs[0] = HSum(c0) + t0; // the reduction happens before that chain's tail joins it.
                outputs[1] = HSum(c1) + t1;
                outputs[2] = HSum(c2) + t2;
                return;
            }
            case 4:
            {
                Vector256<float> c0 = Vector256<float>.Zero;
                Vector256<float> c1 = Vector256<float>.Zero;
                Vector256<float> c2 = Vector256<float>.Zero;
                Vector256<float> c3 = Vector256<float>.Zero;

                int i = 0;
                for (; i + 8 <= len; i += 8)
                {
                    Vector256<float> sharedChunk = Avx.LoadVector256(common + i); // one load of the shared operand feeds every chain in it.
                    c0 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + i), sharedChunk, c0);
                    c1 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + len + i), sharedChunk, c1);
                    c2 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 2L * len + i), sharedChunk, c2);
                    c3 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 3L * len + i), sharedChunk, c3);
                }

                float t0 = 0f; // each chain's tail is its own scalar accumulator, added after the reduction exactly as DotCore adds one.
                float t1 = 0f;
                float t2 = 0f;
                float t3 = 0f;
                for (; i < len; i++)
                {
                    float shared = common[i];
                    t0 += inputs[i] * shared;
                    t1 += inputs[len + i] * shared;
                    t2 += inputs[2L * len + i] * shared;
                    t3 += inputs[3L * len + i] * shared;
                }

                outputs[0] = HSum(c0) + t0; // the reduction happens before that chain's tail joins it.
                outputs[1] = HSum(c1) + t1;
                outputs[2] = HSum(c2) + t2;
                outputs[3] = HSum(c3) + t3;
                return;
            }
            case 5:
            {
                Vector256<float> c0 = Vector256<float>.Zero;
                Vector256<float> c1 = Vector256<float>.Zero;
                Vector256<float> c2 = Vector256<float>.Zero;
                Vector256<float> c3 = Vector256<float>.Zero;
                Vector256<float> c4 = Vector256<float>.Zero;

                int i = 0;
                for (; i + 8 <= len; i += 8)
                {
                    Vector256<float> sharedChunk = Avx.LoadVector256(common + i); // one load of the shared operand feeds every chain in it.
                    c0 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + i), sharedChunk, c0);
                    c1 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + len + i), sharedChunk, c1);
                    c2 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 2L * len + i), sharedChunk, c2);
                    c3 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 3L * len + i), sharedChunk, c3);
                    c4 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 4L * len + i), sharedChunk, c4);
                }

                float t0 = 0f; // each chain's tail is its own scalar accumulator, added after the reduction exactly as DotCore adds one.
                float t1 = 0f;
                float t2 = 0f;
                float t3 = 0f;
                float t4 = 0f;
                for (; i < len; i++)
                {
                    float shared = common[i];
                    t0 += inputs[i] * shared;
                    t1 += inputs[len + i] * shared;
                    t2 += inputs[2L * len + i] * shared;
                    t3 += inputs[3L * len + i] * shared;
                    t4 += inputs[4L * len + i] * shared;
                }

                outputs[0] = HSum(c0) + t0; // the reduction happens before that chain's tail joins it.
                outputs[1] = HSum(c1) + t1;
                outputs[2] = HSum(c2) + t2;
                outputs[3] = HSum(c3) + t3;
                outputs[4] = HSum(c4) + t4;
                return;
            }
            case 6:
            {
                Vector256<float> c0 = Vector256<float>.Zero;
                Vector256<float> c1 = Vector256<float>.Zero;
                Vector256<float> c2 = Vector256<float>.Zero;
                Vector256<float> c3 = Vector256<float>.Zero;
                Vector256<float> c4 = Vector256<float>.Zero;
                Vector256<float> c5 = Vector256<float>.Zero;

                int i = 0;
                for (; i + 8 <= len; i += 8)
                {
                    Vector256<float> sharedChunk = Avx.LoadVector256(common + i); // one load of the shared operand feeds every chain in it.
                    c0 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + i), sharedChunk, c0);
                    c1 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + len + i), sharedChunk, c1);
                    c2 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 2L * len + i), sharedChunk, c2);
                    c3 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 3L * len + i), sharedChunk, c3);
                    c4 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 4L * len + i), sharedChunk, c4);
                    c5 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 5L * len + i), sharedChunk, c5);
                }

                float t0 = 0f; // each chain's tail is its own scalar accumulator, added after the reduction exactly as DotCore adds one.
                float t1 = 0f;
                float t2 = 0f;
                float t3 = 0f;
                float t4 = 0f;
                float t5 = 0f;
                for (; i < len; i++)
                {
                    float shared = common[i];
                    t0 += inputs[i] * shared;
                    t1 += inputs[len + i] * shared;
                    t2 += inputs[2L * len + i] * shared;
                    t3 += inputs[3L * len + i] * shared;
                    t4 += inputs[4L * len + i] * shared;
                    t5 += inputs[5L * len + i] * shared;
                }

                outputs[0] = HSum(c0) + t0; // the reduction happens before that chain's tail joins it.
                outputs[1] = HSum(c1) + t1;
                outputs[2] = HSum(c2) + t2;
                outputs[3] = HSum(c3) + t3;
                outputs[4] = HSum(c4) + t4;
                outputs[5] = HSum(c5) + t5;
                return;
            }
            case 7:
            {
                Vector256<float> c0 = Vector256<float>.Zero;
                Vector256<float> c1 = Vector256<float>.Zero;
                Vector256<float> c2 = Vector256<float>.Zero;
                Vector256<float> c3 = Vector256<float>.Zero;
                Vector256<float> c4 = Vector256<float>.Zero;
                Vector256<float> c5 = Vector256<float>.Zero;
                Vector256<float> c6 = Vector256<float>.Zero;

                int i = 0;
                for (; i + 8 <= len; i += 8)
                {
                    Vector256<float> sharedChunk = Avx.LoadVector256(common + i); // one load of the shared operand feeds every chain in it.
                    c0 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + i), sharedChunk, c0);
                    c1 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + len + i), sharedChunk, c1);
                    c2 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 2L * len + i), sharedChunk, c2);
                    c3 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 3L * len + i), sharedChunk, c3);
                    c4 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 4L * len + i), sharedChunk, c4);
                    c5 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 5L * len + i), sharedChunk, c5);
                    c6 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 6L * len + i), sharedChunk, c6);
                }

                float t0 = 0f; // each chain's tail is its own scalar accumulator, added after the reduction exactly as DotCore adds one.
                float t1 = 0f;
                float t2 = 0f;
                float t3 = 0f;
                float t4 = 0f;
                float t5 = 0f;
                float t6 = 0f;
                for (; i < len; i++)
                {
                    float shared = common[i];
                    t0 += inputs[i] * shared;
                    t1 += inputs[len + i] * shared;
                    t2 += inputs[2L * len + i] * shared;
                    t3 += inputs[3L * len + i] * shared;
                    t4 += inputs[4L * len + i] * shared;
                    t5 += inputs[5L * len + i] * shared;
                    t6 += inputs[6L * len + i] * shared;
                }

                outputs[0] = HSum(c0) + t0; // the reduction happens before that chain's tail joins it.
                outputs[1] = HSum(c1) + t1;
                outputs[2] = HSum(c2) + t2;
                outputs[3] = HSum(c3) + t3;
                outputs[4] = HSum(c4) + t4;
                outputs[5] = HSum(c5) + t5;
                outputs[6] = HSum(c6) + t6;
                return;
            }
            case 8:
            {
                Vector256<float> c0 = Vector256<float>.Zero;
                Vector256<float> c1 = Vector256<float>.Zero;
                Vector256<float> c2 = Vector256<float>.Zero;
                Vector256<float> c3 = Vector256<float>.Zero;
                Vector256<float> c4 = Vector256<float>.Zero;
                Vector256<float> c5 = Vector256<float>.Zero;
                Vector256<float> c6 = Vector256<float>.Zero;
                Vector256<float> c7 = Vector256<float>.Zero;

                int i = 0;
                for (; i + 8 <= len; i += 8)
                {
                    Vector256<float> sharedChunk = Avx.LoadVector256(common + i); // one load of the shared operand feeds every chain in it.
                    c0 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + i), sharedChunk, c0);
                    c1 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + len + i), sharedChunk, c1);
                    c2 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 2L * len + i), sharedChunk, c2);
                    c3 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 3L * len + i), sharedChunk, c3);
                    c4 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 4L * len + i), sharedChunk, c4);
                    c5 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 5L * len + i), sharedChunk, c5);
                    c6 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 6L * len + i), sharedChunk, c6);
                    c7 = Fma.MultiplyAdd(Avx.LoadVector256(inputs + 7L * len + i), sharedChunk, c7);
                }

                float t0 = 0f; // each chain's tail is its own scalar accumulator, added after the reduction exactly as DotCore adds one.
                float t1 = 0f;
                float t2 = 0f;
                float t3 = 0f;
                float t4 = 0f;
                float t5 = 0f;
                float t6 = 0f;
                float t7 = 0f;
                for (; i < len; i++)
                {
                    float shared = common[i];
                    t0 += inputs[i] * shared;
                    t1 += inputs[len + i] * shared;
                    t2 += inputs[2L * len + i] * shared;
                    t3 += inputs[3L * len + i] * shared;
                    t4 += inputs[4L * len + i] * shared;
                    t5 += inputs[5L * len + i] * shared;
                    t6 += inputs[6L * len + i] * shared;
                    t7 += inputs[7L * len + i] * shared;
                }

                outputs[0] = HSum(c0) + t0; // the reduction happens before that chain's tail joins it.
                outputs[1] = HSum(c1) + t1;
                outputs[2] = HSum(c2) + t2;
                outputs[3] = HSum(c3) + t3;
                outputs[4] = HSum(c4) + t4;
                outputs[5] = HSum(c5) + t5;
                outputs[6] = HSum(c6) + t6;
                outputs[7] = HSum(c7) + t7;
                return;
            }
            default:
            {
                for (int b = 0; b < chains; b++) // a count outside the arms is still one independent single-vector dot per chain.
                {
                    outputs[b] = DotCore(inputs + (long)b * len, common, len);
                }

                return;
            }
        }
    }
}
