using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;
using Lair.Kv;

namespace Lair.Tests;

/// <summary>
/// GPU gate for device-side quantized KV attention: a quantized generic pool opened in the device-step posture hands its
/// persistent device copy — the quantized groups the host writers produced and the mirror flushed — to a CUDA step as a
/// store, and the step's causal attention walk reads those resident quantized bytes through a combo-selected kernel. The
/// claims: the q8_0/q8_0 and q8_0/q4_0 device walks decode exactly the groups the committed writers stored and answer the
/// naive dequantized reference (the same stored values, the same FP32 queries) inside the device step's recorded band, so
/// no CPU quantized band or CPU parity band moves; the attention kernel a step runs is resolved once from the store's
/// combo, not branched per position; a quantized step reads its store but refuses to write it, because quantized KV is
/// written host-side and flushed; and a quantized combo with no device attention kernel here (a q4_0 K side, or the
/// dropped q4_0/q4_0 pair) refuses at pool open naming the format and posture rather than opening a store no kernel walks.
/// </summary>
[Trait("Category", "Kv")]
[Trait("Category", "Attention")]
[Trait("Category", "Cuda")]
public sealed class CudaQuantizedKvTests
{
    // The device step's own recorded attention band against the host reference, unchanged from the float store-bound seam:
    // the quantized walk runs the identical online-softmax chain through quantized addressing, so its drift allowance is the
    // existing seam's. The only new drift is the quantized decode the kernel and the naive reference both apply to the same
    // stored groups, which cancels — the two see identical values. No CPU quantized tolerance or CPU parity band is widened.
    private const float DeviceBand = 3e-4f;

    private const float DeviceAbsCeiling = 1e-5f;

    // One synthetic geometry shared by every case, matching the CPU quantized attention fixture's shape so the same stored
    // groups and the same FP32 queries are what both the device walk and the naive reference attend over. BlockTokens is a
    // power of two the store-bound kernels page by shift; headDim 64 over 2 kv-heads is kvDim 128 = four whole 32-value
    // groups per token region, so every group the kernel addresses is whole.
    private const int BlockTokens = 4;

    private const int Layers = 1;

    private const int HeadCount = 4;

    private const int KvHeadCount = 2;

    private const int HeadDim = 64;

    private const int KvDim = KvHeadCount * HeadDim;

    private const int Positions = 8; // two whole blocks; the walk attends positions [0..6] and never reads the unwritten slot 7

    private const int QueryPosition = 6;

    private static readonly float AttnScale = 1f / MathF.Sqrt(HeadDim);

    private const int Seed = 20_261_118;

    // ---------------------------------------------------------------- q8_0/q8_0 device attention decodes the stored groups

    [Fact]
    public void CudaQuantizedKv_Q8Q8DeviceAttention_MatchesNaiveDequantizedReference()
    {
        using CudaBackend cuda = OpenCuda();
        using BlockPool<Q8Block, Q8Block> pool = OpenQuantizedStore<Q8Block, Q8Block>(cuda, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0));

        Fixture fixture = Fixture.Build(Seed);

        WriteQuantizedHostSide(pool, fixture, KvSideFormat.Q8_0, KvSideFormat.Q8_0);
        Assert.True(pool.DeviceMirrorCrossedBytes > 0, "host quantized writes must have crossed whole blocks to the store the step reads");
        pool.FlushDeviceMirror();
        Assert.Equal(0, pool.DeviceMirrorPendingWrites); // the written blocks are current on the device copy

        using CudaDeviceStep step = cuda.OpenDeviceStep(StoreRequest(), pool.DeviceStepMirror!);
        Assert.True(step.KvStoreBacked);

        float[] deviceOut = new float[HeadCount * HeadDim];
        step.StageQueries(fixture.QHeads);
        step.Attend(layer: 0, QueryPosition, AttnScale, softcap: 0f, window: 0, deviceOut);

        float[] reference = NaiveDequantizedReference(fixture, KvSideFormat.Q8_0, KvSideFormat.Q8_0);
        Assert.False(deviceOut.Any(float.IsNaN), "device q8_0/q8_0 attention produced a NaN");
        AssertWithinBand(reference, deviceOut, DeviceBand, DeviceAbsCeiling, "device q8_0/q8_0 attention over the resident quantized groups");
    }

    // ---------------------------------------------------------------- q8_0/q4_0 device attention decodes nibbles under their group scale

    [Fact]
    public void CudaQuantizedKv_K8V4DeviceAttention_MatchesNaiveDequantizedReference()
    {
        using CudaBackend cuda = OpenCuda();
        using BlockPool<Q8Block, Q4Block> pool = OpenQuantizedStore<Q8Block, Q4Block>(cuda, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q4_0));

        Fixture fixture = Fixture.Build(Seed);

        WriteQuantizedHostSide(pool, fixture, KvSideFormat.Q8_0, KvSideFormat.Q4_0);
        pool.FlushDeviceMirror();
        Assert.Equal(0, pool.DeviceMirrorPendingWrites);

        using CudaDeviceStep step = cuda.OpenDeviceStep(StoreRequest(), pool.DeviceStepMirror!);

        float[] deviceOut = new float[HeadCount * HeadDim];
        step.StageQueries(fixture.QHeads);
        step.Attend(layer: 0, QueryPosition, AttnScale, softcap: 0f, window: 0, deviceOut);

        float[] reference = NaiveDequantizedReference(fixture, KvSideFormat.Q8_0, KvSideFormat.Q4_0);
        Assert.False(deviceOut.Any(float.IsNaN), "device q8_0/q4_0 attention produced a NaN");
        AssertWithinBand(reference, deviceOut, DeviceBand, DeviceAbsCeiling, "device q8_0/q4_0 attention over the resident quantized groups");
    }

    // ---------------------------------------------------------------- window narrowing bounds the quantized walk to the selection

    [Fact]
    public void CudaQuantizedKv_DeviceAttention_WindowBoundsWalkToSelection()
    {
        const int Window = 3;

        using CudaBackend cuda = OpenCuda();
        using BlockPool<Q8Block, Q8Block> pool = OpenQuantizedStore<Q8Block, Q8Block>(cuda, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0));

        Fixture fixture = Fixture.Build(Seed);

        // Poison every position the window excludes with huge magnitudes: a walk that reached them would move the output
        // far past the reference, so a match proves the selection bounded the walk rather than a loop bound skipping them.
        using (SequenceKv<Q8Block, Q8Block> kv = new(pool))
        {
            for (int pos = 0; pos < fixture.Positions; pos++)
            {
                kv.Append();
                kv.WriteK(pos, 0, fixture.KValues[pos]);
                kv.WriteV(pos, 0, fixture.VValues[pos]);
            }

            for (int pos = 0; pos < QueryPosition - Window + 1; pos++)
            {
                kv.WriteK(pos, 0, Filled(KvDim, 4096f));
                kv.WriteV(pos, 0, Filled(KvDim, -4096f));
            }
        }

        pool.FlushDeviceMirror();

        using CudaDeviceStep step = cuda.OpenDeviceStep(StoreRequest(), pool.DeviceStepMirror!);

        float[] deviceOut = new float[HeadCount * HeadDim];
        step.StageQueries(fixture.QHeads);
        step.Attend(layer: 0, QueryPosition, AttnScale, softcap: 0f, window: Window, deviceOut);

        // The reference runs over the same windowed range with the original (unpoisoned) values.
        float[] reference = NaiveDequantizedReference(fixture, KvSideFormat.Q8_0, KvSideFormat.Q8_0, window: Window);
        AssertWithinBand(reference, deviceOut, DeviceBand, DeviceAbsCeiling, "windowed device q8_0/q8_0 attention");
    }

    // ---------------------------------------------------------------- the kernel a step runs is resolved from the combo, not per position

    [Fact]
    public void CudaQuantizedKv_StoreComboResolvesTheAttentionKernel_OnceAtOpen()
    {
        using CudaBackend cuda = OpenCuda();

        using BlockPool<Q8Block, Q8Block> q8q8 = OpenQuantizedStore<Q8Block, Q8Block>(cuda, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0));
        using BlockPool<Q8Block, Q4Block> k8v4 = OpenQuantizedStore<Q8Block, Q4Block>(cuda, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q4_0));

        Assert.True(q8q8.TryGetDeviceStore(out KvDeviceStore q8q8Store));
        Assert.True(k8v4.TryGetDeviceStore(out KvDeviceStore k8v4Store));
        Assert.Equal(KvMirrorFormatCombo.Q8Q8, q8q8Store.Combo);
        Assert.Equal(KvMirrorFormatCombo.K8V4, k8v4Store.Combo);

        // The store carries the combo and the group geometry a step addresses the quantized bytes through.
        Assert.Equal(34, q8q8Store.KGroupBytes);
        Assert.Equal(34, q8q8Store.VGroupBytes);
        Assert.Equal(18, k8v4Store.VGroupBytes);
        Assert.Equal(KvDim / 32, q8q8Store.GroupsPerTokenK);
    }

    // ---------------------------------------------------------------- a quantized step reads its store but refuses to write it

    [Fact]
    public void CudaQuantizedKv_QuantizedStepRefusesWriteKv_NamingPosture()
    {
        using CudaBackend cuda = OpenCuda();
        using BlockPool<Q8Block, Q8Block> pool = OpenQuantizedStore<Q8Block, Q8Block>(cuda, new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0));

        using CudaDeviceStep step = cuda.OpenDeviceStep(StoreRequest(), pool.DeviceStepMirror!);

        step.StageQueries(Filled(HeadCount * HeadDim, 0.5f));
        InvalidOperationException refusal = Assert.Throws<InvalidOperationException>(() => step.WriteKv(layer: 0, positionBase: 0));
        Assert.Contains("quantized KV store", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("written host-side", refusal.Message, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- combos with no device attention kernel refuse at pool open

    [Fact]
    public void CudaQuantizedKv_UnsupportedQuantizedDeviceCombo_RefusesAtPoolOpen_NamingFormatAndPosture()
    {
        using CudaBackend cuda = OpenCuda();

        // A q4_0 K side has no device attention kernel here: the quantized route dots a q8_0 K side.
        ArgumentException q4KRefusal = Assert.Throws<ArgumentException>(() =>
            OpenQuantizedStore<Q4Block, Q4Block>(cuda, new KvFormatPair(KvSideFormat.Q4_0, KvSideFormat.Q4_0)));
        Assert.Contains("no device attention kernel", q4KRefusal.Message, StringComparison.Ordinal);
        Assert.Contains("q4_0", q4KRefusal.Message, StringComparison.Ordinal);

        // The dropped q4_0/q4_0 pair refuses the same way rather than walking quantized bytes with a float kernel.
        ArgumentException droppedRefusal = Assert.Throws<ArgumentException>(() =>
            OpenQuantizedStore<Q4Block, Q8Block>(cuda, new KvFormatPair(KvSideFormat.Q4_0, KvSideFormat.Q8_0)));
        Assert.Contains("q8_0/q8_0", droppedRefusal.Message, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- fixtures and helpers

    private static BlockPool<TK, TV> OpenQuantizedStore<TK, TV>(CudaBackend cuda, KvFormatPair formats)
        where TK : struct, IKvFormat<TK>
        where TV : struct, IKvFormat<TV> =>
        new(BlockTokens, Layers, KvHeadCount, HeadDim, capacityBlocks: Positions / BlockTokens, cuda, formats,
            BlockResidencyMode.Vram, KvMirrorExecution.DeviceStep);

    private static void WriteQuantizedHostSide<TK, TV>(BlockPool<TK, TV> pool, Fixture fixture, KvSideFormat k, KvSideFormat v)
        where TK : struct, IKvFormat<TK>
        where TV : struct, IKvFormat<TV>
    {
        using SequenceKv<TK, TV> kv = new(pool);
        for (int pos = 0; pos < fixture.Positions; pos++)
        {
            kv.Append();
            kv.WriteK(pos, 0, fixture.KValues[pos]);
            kv.WriteV(pos, 0, fixture.VValues[pos]);
        }
    }

    private static CudaDeviceStepRequest StoreRequest() => new()
    {
        MaxPositions = Positions,
        LayerCount = Layers,
        HeadCount = HeadCount,
        KvHeadCount = KvHeadCount,
        HeadDim = HeadDim,
    };

    private static CudaBackend OpenCuda(CudaWeightResidency residency = CudaWeightResidency.DeviceCopy)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        try
        {
            return new CudaBackend(residency);
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }
    }

    // The online-softmax semantics of the float kernel — grouped-query mapping, scaled scores, the rescale-on-new-max
    // rule, and the final 1f / sumExp normalization — run over the values the committed writer stored (re-quantized then
    // decoded) with the FP32 queries, one position at a time in ascending order. This is the same footing the CPU
    // quantized kernel is measured against, so a device walk that decodes the same groups answers the same values.
    private static float[] NaiveDequantizedReference(Fixture fixture, KvSideFormat k, KvSideFormat v, int window = 0)
    {
        float[][] kByPos = new float[fixture.Positions][];
        float[][] vByPos = new float[fixture.Positions][];
        for (int pos = 0; pos < fixture.Positions; pos++)
        {
            kByPos[pos] = DequantRegion(fixture.KValues[pos], k);
            vByPos[pos] = DequantRegion(fixture.VValues[pos], v);
        }

        int firstVisible = window > 0 ? Math.Max(0, QueryPosition - window + 1) : 0;
        float[] outConcat = new float[HeadCount * HeadDim];

        for (int h = 0; h < HeadCount; h++)
        {
            int kvHead = h * KvHeadCount / HeadCount;
            float runningMax = float.NegativeInfinity;
            float sumExp = 0f;
            Span<float> outHead = outConcat.AsSpan(h * HeadDim, HeadDim);
            outHead.Clear();

            for (int t = firstVisible; t <= QueryPosition; t++)
            {
                float score = 0f;
                for (int j = 0; j < HeadDim; j++)
                {
                    score += kByPos[t][kvHead * HeadDim + j] * fixture.QHeads[h * HeadDim + j];
                }

                score *= AttnScale;

                if (score > runningMax)
                {
                    if (!float.IsNegativeInfinity(runningMax))
                    {
                        float rescale = MathF.Exp(runningMax - score);
                        sumExp *= rescale;
                        for (int j = 0; j < HeadDim; j++)
                        {
                            outHead[j] *= rescale;
                        }
                    }

                    runningMax = score;
                }

                float weight = MathF.Exp(score - runningMax);
                sumExp += weight;

                for (int j = 0; j < HeadDim; j++)
                {
                    outHead[j] += weight * vByPos[t][kvHead * HeadDim + j];
                }
            }

            float invSum = 1f / sumExp;
            for (int j = 0; j < HeadDim; j++)
            {
                outHead[j] *= invSum;
            }
        }

        return outConcat;
    }

    private static float[] DequantRegion(float[] values, KvSideFormat side)
    {
        int groups = values.Length / 32;
        int groupBytes = KvFormatPair.GroupBytes(side);
        byte[] arena = new byte[groups * groupBytes];

        if (side == KvSideFormat.Q8_0)
        {
            Q8Block.WriteRegion(values, MemoryMarshal.Cast<byte, Q8Block>(arena.AsSpan()));
            return DequantQ8(MemoryMarshal.Cast<byte, Q8Block>(arena.AsSpan()));
        }

        Q4Block.WriteRegion(values, MemoryMarshal.Cast<byte, Q4Block>(arena.AsSpan()));
        return DequantQ4(MemoryMarshal.Cast<byte, Q4Block>(arena.AsSpan()));
    }

    private static unsafe float[] DequantQ8(ReadOnlySpan<Q8Block> blocks)
    {
        float[] values = new float[blocks.Length * Q8Block.Values];
        for (int b = 0; b < blocks.Length; b++)
        {
            ref readonly Q8Block block = ref System.Runtime.CompilerServices.Unsafe.Add(
                ref MemoryMarshal.GetReference(blocks), b);
            float d = (float)BitConverter.UInt16BitsToHalf(block.ScaleBits);
            for (int i = 0; i < Q8Block.Values; i++)
            {
                values[b * Q8Block.Values + i] = (float)block.Quants[i] * d;
            }
        }

        return values;
    }

    private static unsafe float[] DequantQ4(ReadOnlySpan<Q4Block> blocks)
    {
        float[] values = new float[blocks.Length * Q4Block.Values];
        for (int b = 0; b < blocks.Length; b++)
        {
            ref readonly Q4Block block = ref System.Runtime.CompilerServices.Unsafe.Add(
                ref MemoryMarshal.GetReference(blocks), b);
            float d = (float)BitConverter.UInt16BitsToHalf(block.ScaleBits);
            for (int j = 0; j < Q4Block.Values / 2; j++)
            {
                byte packed = block.Nibbles[j];
                values[b * Q4Block.Values + j] = ((packed & 0x0F) - 8) * d;
                values[b * Q4Block.Values + Q4Block.Values / 2 + j] = ((packed >> 4) - 8) * d;
            }
        }

        return values;
    }

    private static float[] Filled(int count, float value)
    {
        float[] values = new float[count];
        Array.Fill(values, value);
        return values;
    }

    private static void AssertWithinBand(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float band, float absoluteCeiling, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        float worstRelative = 0f;
        int relativeIndex = 0;
        double worstAbsolute = 0d;
        int absoluteIndex = 0;
        int offender = -1;

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]), $"{context}: NaN at element {i}");

            float delta = Math.Abs(expected[i] - actual[i]);
            float scale = Math.Max(Math.Abs(expected[i]), Math.Abs(actual[i]));
            float relative = scale == 0f ? delta : delta / scale;

            if (relative > worstRelative)
            {
                worstRelative = relative;
                relativeIndex = i;
            }

            if (delta > worstAbsolute)
            {
                worstAbsolute = delta;
                absoluteIndex = i;
            }

            double allowance = Math.Max((double)band * scale, absoluteCeiling);
            if (delta > allowance && offender < 0)
            {
                offender = i;
            }
        }

        if (offender >= 0)
        {
            Assert.Fail(
                $"{context}: widest relative drift {worstRelative:E6} at element {relativeIndex} and widest absolute " +
                $"drift {worstAbsolute:E6} at element {absoluteIndex} against the recorded band {band:E2}, ceiling {absoluteCeiling:E2}" +
                $" — offending element {offender}: expected {expected[offender]:E9}, actual {actual[offender]:E9}.");
        }
    }

    private sealed class Fixture
    {
        public int Positions { get; private set; }
        public float[] QHeads { get; private set; } = [];
        public float[][] KValues { get; private set; } = [];
        public float[][] VValues { get; private set; } = [];

        public static Fixture Build(int seed)
        {
            Random rng = new(seed);
            Fixture fixture = new()
            {
                Positions = CudaQuantizedKvTests.Positions,
                QHeads = new float[HeadCount * HeadDim],
            };

            for (int i = 0; i < fixture.QHeads.Length; i++)
            {
                fixture.QHeads[i] = (float)(rng.NextDouble() * 3.0 - 1.5);
            }

            fixture.KValues = new float[CudaQuantizedKvTests.Positions][];
            fixture.VValues = new float[CudaQuantizedKvTests.Positions][];
            for (int pos = 0; pos < CudaQuantizedKvTests.Positions; pos++)
            {
                fixture.KValues[pos] = new float[KvDim];
                fixture.VValues[pos] = new float[KvDim];
                for (int i = 0; i < KvDim; i++)
                {
                    fixture.KValues[pos][i] = (float)(rng.NextDouble() * 3.0 - 1.5);
                    fixture.VValues[pos][i] = (float)(rng.NextDouble() * 3.0 - 1.5);
                }

                // Deliberate exact zeros and negatives: zero magnitudes must land on zero scales, and negative values
                // must survive the signed-byte and nibble encodings the device kernel decodes.
                fixture.KValues[pos][0] = -fixture.KValues[pos][0];
                fixture.KValues[pos][1] = 0f;
                fixture.VValues[pos][2] = 0f;
                fixture.VValues[pos][31] = -fixture.VValues[pos][31];
            }

            return fixture;
        }
    }
}
