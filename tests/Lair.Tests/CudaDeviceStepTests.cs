using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;
using Lair.Kv;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// GPU gate for the on-device transformer-step family: the rotation, the KV write, and the causal
/// softmax(QK^T)V walk run by CudaDeviceStep against buffers the step owns, checked against the shipped CPU
/// attention and rotation for the same geometry. Against the host oracle the claim is this seam's CUDA band posture
/// rather than raw bytes: the device softmax answers through ex2.approx over a log2-e scaled argument where the host
/// answers through MathF.Exp, so a device walk lands inside a recorded band of the host online-softmax chain it
/// mirrors — element order, running-max rule, rescale-on-new-max arithmetic, and the final reciprocal all held in the
/// host kernel's order. Device to device the claim is bit-for-bit: the same staged window walked twice answers the
/// same bytes. The composition claim is a transfer claim too: one step crosses the seam for its staged rows and once
/// for its attention result — scores, softmax state, and every K/V read of the walk never cross, and a window the
/// arena already holds crosses only its queries and its result.
/// </summary>
[Trait("Category", "Kernel")]
[Trait("Category", "Parity")]
[Trait("Category", "Cuda")]
public sealed class CudaDeviceStepTests
{
    // Recorded bands for the device step against the host reference, in the same relative shape as the CUDA GEMV and
    // blocked-matmul bands: |expected - actual| <= band * max(|expected|, |actual|), with a companion absolute ceiling so
    // an element whose value comes out of cancellation still has to land near zero. The device walk answers through
    // ex2.approx over a log2-e scaled argument and an approximate-form tanh where the host answers through MathF.Exp and
    // a double-precision cap, and its dot is one scalar ascending chain while the host's seam runs its vectorized FMA
    // accumulation — this seam's cross-path numerics posture rather than raw bytes. Measured drift on these cases over
    // every element of all 96 band checks: an element fails only when its drift exceeds both allowances at once, and the
    // widest relative drift among elements whose absolute drift outruns the ceiling measures 5.71E-005 (at about
    // +2.29E-002, absolute 1.31E-006) — the band below carries that five times over; the largest absolute drift anywhere
    // in those families is 2.10E-006 at a value near -2.23E-001 (relative 9.4E-006, carried by its own magnitude) and no
    // composed-step element passes 1.15E-006. The ceiling carries the cancellation class: the widest relative drift
    // anywhere is 4.34E-004 at a value that cancels down to about -6.11E-005 with only 2.7E-008 of absolute drift, and a
    // gate without a ceiling would need a relative band near 4.4E-004 on these fixtures instead; the tightest margin
    // anywhere at the values below is still more than one and a half times — the closest element sits at -3.7612E-003
    // against this band's allowance of about 1.13E-006 for its measured drift of 6.6E-007. Rotation measures its
    // widest relative drift at 1.90E-005 (absolute 5.7E-008, on a value near +2.99E-003) and its widest absolute drift at
    // 3.0E-007 (on values near +-2.1E+000), so the band below covers that relative worst by more than half again and this
    // ceiling sits above its absolute worst by more than three times. These are CUDA-only numbers on this host and driver:
    // no CPU band moves, and no existing CUDA GEMV or blocked-matmul band moves — the bit-for-bit claim this seam keeps is
    // its own, in the walked-twice case below.
    private const float AttentionBand = 3e-4f;

    private const float AttentionAbsCeiling = 1e-6f;

    private const float RopeBand = 3e-5f;

    private const float RopeAbsCeiling = 1e-6f;

    // Geometries the step kernels must walk: grouped-query 2:1, plain MHA at a wide head, single-kv-head MQA with a
    // group of 4, and a group of 2 at a head dimension that is not a power of two.
    private static readonly (int Heads, int KvHeads, int HeadDim)[] Geometries =
    [
        (4, 2, 8),
        (2, 2, 64),
        (4, 1, 96),
        (6, 3, 16),
    ];

    // Window shapes: a single row, a ragged window, a narrowed window, and windows starting away from position 0 so
    // the arena's earlier positions are read rather than rewritten.
    private static readonly (int Rows, int PositionBase, int Window)[] WindowShapes =
    [
        (1, 0, 0),
        (3, 0, 0),
        (5, 0, 3),
        (4, 5, 0),
        (3, 6, 2),
        (2, 7, 1),
    ];

    private const float AttnScale = 0.125f;

    private static readonly float[] Softcaps = [0f, 5f, 50f];

    private static float Value(int seed) => (float)Math.Sin(0.37 * seed + 0.13) * 1.7f;

    private static float[] QueryRows(int rows, int heads, int headDim, int seedOffset) =>
        Filled(rows * heads * headDim, seedOffset);

    private static float[] KeyValueRows(int rows, int kvHeads, int headDim, int seedOffset) =>
        Filled(rows * kvHeads * headDim, seedOffset);

    private static float[] Filled(int length, int seedOffset)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = Value(seedOffset + i);
        }

        return values;
    }

    // ---------------------------------------------------------------- rotation parity

    [Theory]
    [InlineData(RopePairing.AdjacentPairs)]
    [InlineData(RopePairing.SplitHalfPairs)]
    public void DeviceStep_Rotation_LandsInsideBandAgainstHostRotation_InEveryPairingAndGeometry(RopePairing pairing)
    {
        foreach ((int heads, int kvHeads, int headDim) in Geometries)
        {
            const int MaxPositions = 24;
            const int Rows = 5;
            const int PositionBase = 3;
            int half = headDim / 2;

            float[] cos = new float[MaxPositions * half];
            float[] sin = new float[MaxPositions * half];
            LayerOps.BuildRopeTables(thetaBase: 10_000.0, headDim, MaxPositions, cos, sin);

            float[] qRows = QueryRows(Rows, heads, headDim, seedOffset: 11);
            float[] kRows = KeyValueRows(Rows, kvHeads, headDim, seedOffset: 101);
            float[] vRows = KeyValueRows(Rows, kvHeads, headDim, seedOffset: 202);

            float[] expectedQ = (float[])qRows.Clone();
            float[] expectedK = (float[])kRows.Clone();
            for (int r = 0; r < Rows; r++)
            {
                for (int h = 0; h < heads; h++)
                {
                    ApplyRopeToRegion(expectedQ, r * heads * headDim + h * headDim, headDim, pairing, cos, sin, PositionBase + r);
                }

                for (int h = 0; h < kvHeads; h++)
                {
                    ApplyRopeToRegion(expectedK, r * kvHeads * headDim + h * headDim, headDim, pairing, cos, sin, PositionBase + r);
                }
            }

            using CudaBackend cuda = OpenCuda();
            using CudaDeviceStep step = OpenStep(cuda, heads, kvHeads, headDim, layerCount: 1, MaxPositions, pairing: pairing);

            step.LoadRopeTables(cos, sin);
            Assert.True(step.RopeTablesLoaded);

            step.StageWindow(qRows, kRows, vRows);
            step.ApplyRope(PositionBase);

            float[] actualQ = new float[Rows * heads * headDim];
            float[] actualK = new float[Rows * kvHeads * headDim];
            float[] actualV = new float[Rows * kvHeads * headDim];
            step.ReadStagedQueries(actualQ);
            step.ReadStagedKeys(actualK);
            step.ReadStagedValues(actualV);

            AssertWithinBand(expectedQ, actualQ, RopeBand, RopeAbsCeiling, $"device {pairing} query rotation at heads {heads}x{headDim}");
            AssertWithinBand(expectedK, actualK, RopeBand, RopeAbsCeiling, $"device {pairing} key rotation at kv-heads {kvHeads}x{headDim}");

            // The value side is not rotated: the rotation touched the query and key rows only, and it left the values
            // exactly as the window staged them.
            BatchKernelTests.AssertBitsEqual(vRows, actualV, $"device {pairing} rotation left the value rows untouched");
        }
    }

    // ---------------------------------------------------------------- KV write and read parity

    [Fact]
    public void DeviceStep_KvWrite_LandsInArenaSlotsTheAttentionWalkReadsBack()
    {
        const int Heads = 4;
        const int KvHeads = 2;
        const int HeadDim = 8;
        const int Layers = 3;
        const int MaxPositions = 16;
        const int Rows = 4;

        // Two layers get one set of rows and the third gets another: an attention on each layer must read that
        // layer's own write, so a shared arena cannot answer a wrong layer's positions.
        float[] sharedK = KeyValueRows(Rows, KvHeads, HeadDim, seedOffset: 7);
        float[] sharedV = KeyValueRows(Rows, KvHeads, HeadDim, seedOffset: 40);
        float[] otherK = KeyValueRows(Rows, KvHeads, HeadDim, seedOffset: 77);
        float[] otherV = KeyValueRows(Rows, KvHeads, HeadDim, seedOffset: 91);

        using CudaBackend cuda = OpenCuda();
        using CudaDeviceStep step = OpenStep(cuda, Heads, KvHeads, HeadDim, Layers, MaxPositions);

        for (int layer = 0; layer < Layers; layer++)
        {
            bool shared = layer != 1;
            step.StageWindow(
                QueryRows(Rows, Heads, HeadDim, seedOffset: 300 + layer),
                shared ? sharedK : otherK,
                shared ? sharedV : otherV);
            step.WriteKv(layer, positionBase: 0);
        }

        for (int layer = 0; layer < Layers; layer++)
        {
            // The walk answers the queries that are staged when it runs, so each layer's attention is checked against
            // that layer's own rows: stage them first, then read back through the arena.
            float[] qRows = QueryRows(Rows, Heads, HeadDim, seedOffset: 300 + layer);
            step.StageQueries(qRows);

            float[] actual = new float[Rows * Heads * HeadDim];
            step.Attend(layer, positionBase: 0, attnScale: AttnScale, softcap: 0f, window: 0, actual);

            float[] expected = HostAttention(
                qRows, Rows, Heads, KvHeads, HeadDim, layer,
                layer != 1 ? sharedK : otherK,
                layer != 1 ? sharedV : otherV,
                writtenRows: Rows,
                attnScale: AttnScale,
                softcap: 0f,
                window: 0);

            AssertWithinBand(
                expected,
                actual,
                AttentionBand,
                AttentionAbsCeiling,
                $"device KV arena read at layer {layer}");
        }
    }

    // ---------------------------------------------------------------- causal attention parity

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void DeviceStep_CausalAttention_LandsInsideBandAgainstHostOnlineSoftmax(int shapeIndex)
    {
        foreach ((int heads, int kvHeads, int headDim) in Geometries)
        {
            (int rows, int positionBase, int window) = WindowShapes[shapeIndex];
            int written = positionBase + rows;
            int maxPositions = written + 4;

            float[] hostK = new float[written * kvHeads * headDim];
            float[] hostV = new float[written * kvHeads * headDim];

            using CudaBackend cuda = OpenCuda();
            using CudaDeviceStep step = OpenStep(cuda, heads, kvHeads, headDim, layerCount: 2, maxPositions);

            // Every position the walk reads is written by a window before it, in the order a forward pass runs them.
            for (int start = 0; start < written; start += rows)
            {
                int chunk = Math.Min(rows, written - start);
                float[] chunkK = KeyValueRows(chunk, kvHeads, headDim, seedOffset: 600 + start);
                float[] chunkV = KeyValueRows(chunk, kvHeads, headDim, seedOffset: 700 + start);

                step.StageWindow(QueryRows(chunk, heads, headDim, seedOffset: 500 + start), chunkK, chunkV);
                step.WriteKv(layer: 0, positionBase: start);

                chunkK.CopyTo(hostK.AsSpan(start * kvHeads * headDim));
                chunkV.CopyTo(hostV.AsSpan(start * kvHeads * headDim));
            }

            foreach (float softcap in Softcaps)
            {
                float[] qRows = QueryRows(rows, heads, headDim, seedOffset: 500 + positionBase);
                step.StageQueries(qRows);

                float[] actual = new float[rows * heads * headDim];
                step.Attend(layer: 0, positionBase, attnScale: AttnScale, softcap, window, actual);

                float[] expected = HostAttention(
                    qRows, rows, heads, kvHeads, headDim, layer: 0, hostK, hostV, written,
                    AttnScale, softcap, window, positionBase);

                Assert.False(actual.Any(float.IsNaN), $"device attention produced a NaN for heads {heads}x{headDim}");
                AssertWithinBand(
                    expected,
                    actual,
                    AttentionBand,
                    AttentionAbsCeiling,
                    $"device attention heads {heads}x{headDim} rows {rows} at {positionBase} window {window} softcap {softcap}");
            }
        }
    }

    // ---------------------------------------------------------------- composed step

    [Theory]
    [InlineData(RopePairing.AdjacentPairs)]
    [InlineData(RopePairing.SplitHalfPairs)]
    public void DeviceStep_RunStep_ComposesRotationWriteAndAttentionAgainstTheHostStep(RopePairing pairing)
    {
        const int Heads = 4;
        const int KvHeads = 2;
        const int HeadDim = 8;
        const int MaxPositions = 16;
        const int Rows = 4;
        const int PositionBase = 2;
        const float Softcap = 50f;

        float[] cos = new float[MaxPositions * HeadDim / 2];
        float[] sin = new float[MaxPositions * HeadDim / 2];
        LayerOps.BuildRopeTables(thetaBase: 10_000.0, HeadDim, MaxPositions, cos, sin);

        float[] qRows = QueryRows(Rows, Heads, HeadDim, seedOffset: 13);
        float[] kRows = KeyValueRows(Rows, KvHeads, HeadDim, seedOffset: 113);
        float[] vRows = KeyValueRows(Rows, KvHeads, HeadDim, seedOffset: 213);

        // The step's window reaches back before its own first row, so the positions it reads belong to an earlier
        // window that already wrote them — the forward order this seam documents. Both sides hold those rows once.
        const int EarlierRows = PositionBase;
        float[] earlierK = KeyValueRows(EarlierRows, KvHeads, HeadDim, seedOffset: 313);
        float[] earlierV = KeyValueRows(EarlierRows, KvHeads, HeadDim, seedOffset: 413);

        // The host step the composition is checked against: rotate the queries and the keys, write them, attend.
        float[] rotatedQ = (float[])qRows.Clone();
        float[] rotatedK = (float[])kRows.Clone();
        for (int r = 0; r < Rows; r++)
        {
            for (int h = 0; h < Heads; h++)
            {
                ApplyRopeToRegion(rotatedQ, r * Heads * HeadDim + h * HeadDim, HeadDim, pairing, cos, sin, PositionBase + r);
            }

            for (int h = 0; h < KvHeads; h++)
            {
                ApplyRopeToRegion(rotatedK, r * KvHeads * HeadDim + h * HeadDim, HeadDim, pairing, cos, sin, PositionBase + r);
            }
        }

        int kvRegion = KvHeads * HeadDim;
        float[] hostK = new float[(EarlierRows + Rows) * kvRegion];
        float[] hostV = new float[(EarlierRows + Rows) * kvRegion];
        earlierK.CopyTo(hostK, 0);
        earlierV.CopyTo(hostV, 0);
        rotatedK.CopyTo(hostK, EarlierRows * kvRegion);
        vRows.CopyTo(hostV, EarlierRows * kvRegion);

        using CudaBackend cuda = OpenCuda();

        using CudaDeviceStep step = OpenStep(cuda, Heads, KvHeads, HeadDim, layerCount: 1, MaxPositions, pairing: pairing);
        step.LoadRopeTables(cos, sin);

        // The earlier window writes its keys and values into the same arena the composed step then reads.
        step.StageWindow(
            QueryRows(EarlierRows, Heads, HeadDim, seedOffset: 613),
            earlierK,
            earlierV);
        step.WriteKv(layer: 0, positionBase: 0);

        long transfersBefore = cuda.RuntimeTransferCallsTotal;

        float[] actual = new float[Rows * Heads * HeadDim];
        step.RunStep(0, PositionBase, qRows, kRows, vRows, AttnScale, Softcap, window: 3, actual);

        float[] expected = HostAttention(
            rotatedQ, Rows, Heads, KvHeads, HeadDim, layer: 0, hostK, hostV, writtenRows: EarlierRows + Rows,
            AttnScale, Softcap, window: 3, positionBase: PositionBase);

        AssertWithinBand(
            expected,
            actual,
            AttentionBand,
            AttentionAbsCeiling,
            $"device composed step with {pairing} pairing");

        // The composition's crossing claim: three staged row groups in and one attention result out, and nothing else
        // — no per-layer score, softmax, or KV round trip between the kernels.
        Assert.Equal(4, cuda.RuntimeTransferCallsTotal - transfersBefore);
        Assert.True(cuda.KernelBusyMicrosecondsTotal > 0, "the device step recorded no kernel busy time");
        Assert.Equal(0, cuda.DeviceWeightCopies);
        Assert.Equal(0, cuda.OpenKvMirrorCount);
        Assert.True(step.DeviceBytes > 0);
    }

    [Fact]
    public void DeviceStep_ResidentArenaAnswersLaterWindowsWithoutRestagingOldKv()
    {
        const int Heads = 4;
        const int KvHeads = 2;
        const int HeadDim = 8;
        const int MaxPositions = 16;

        using CudaBackend cuda = OpenCuda();
        using CudaDeviceStep step = OpenStep(cuda, Heads, KvHeads, HeadDim, layerCount: 1, MaxPositions);

        // First window: positions 0..3 written and attended.
        step.StageWindow(
            QueryRows(4, Heads, HeadDim, seedOffset: 17),
            KeyValueRows(4, KvHeads, HeadDim, seedOffset: 117),
            KeyValueRows(4, KvHeads, HeadDim, seedOffset: 217));
        step.WriteKv(0, positionBase: 0);

        float[] firstOut = new float[4 * Heads * HeadDim];
        step.Attend(0, positionBase: 0, attnScale: AttnScale, softcap: 0f, window: 0, firstOut);

        // Second window: its own rows cross, and the arena's earlier positions cross no second time.
        long transfersBefore = cuda.RuntimeTransferCallsTotal;
        step.StageWindow(
            QueryRows(2, Heads, HeadDim, seedOffset: 317),
            KeyValueRows(2, KvHeads, HeadDim, seedOffset: 117 + 4 * KvHeads * HeadDim),
            KeyValueRows(2, KvHeads, HeadDim, seedOffset: 217 + 4 * KvHeads * HeadDim));
        step.WriteKv(0, positionBase: 4);
        Assert.Equal(3, cuda.RuntimeTransferCallsTotal - transfersBefore);

        float[] secondOut = new float[2 * Heads * HeadDim];
        step.Attend(0, positionBase: 4, attnScale: AttnScale, softcap: 0f, window: 0, secondOut);
        Assert.Equal(4, cuda.RuntimeTransferCallsTotal - transfersBefore);

        float[] allK = KeyValueRows(6, KvHeads, HeadDim, seedOffset: 117);
        float[] allV = KeyValueRows(6, KvHeads, HeadDim, seedOffset: 217);

        float[] expectedFirst = HostAttention(
            QueryRows(4, Heads, HeadDim, seedOffset: 17), 4, Heads, KvHeads, HeadDim, 0,
            allK, allV, writtenRows: 4, AttnScale, 0f, 0);
        AssertWithinBand(
            expectedFirst,
            firstOut,
            AttentionBand,
            AttentionAbsCeiling,
            "device first window over its own writes");

        // The second window's reference attends all six positions: the first four come from the first window's write
        // and the last two from the second's, and only the new queries and the result crossed.
        float[] expectedSecond = HostAttention(
            QueryRows(2, Heads, HeadDim, seedOffset: 317), 2, Heads, KvHeads, HeadDim, 0,
            allK, allV, writtenRows: 6, AttnScale, 0f, 0, positionBase: 4);
        AssertWithinBand(
            expectedSecond,
            secondOut,
            AttentionBand,
            AttentionAbsCeiling,
            "device second window over the resident arena");
    }

    // ---------------------------------------------------------------- determinism posture

    [Fact]
    public void DeviceStep_SameWindowWalkedTwiceAnswersIdenticalBytes()
    {
        const int Heads = 4;
        const int KvHeads = 2;
        const int HeadDim = 8;
        const int MaxPositions = 16;
        const int Rows = 4;

        float[] qRows = QueryRows(Rows, Heads, HeadDim, seedOffset: 23);
        float[] kRows = KeyValueRows(Rows, KvHeads, HeadDim, seedOffset: 123);
        float[] vRows = KeyValueRows(Rows, KvHeads, HeadDim, seedOffset: 223);

        using CudaBackend cuda = OpenCuda();
        using CudaDeviceStep first = OpenStep(cuda, Heads, KvHeads, HeadDim, layerCount: 1, MaxPositions);
        using CudaDeviceStep second = OpenStep(cuda, Heads, KvHeads, HeadDim, layerCount: 1, MaxPositions);

        float[] runA = new float[Rows * Heads * HeadDim];
        first.StageWindow(qRows, kRows, vRows);
        first.WriteKv(0, 0);
        first.Attend(0, 0, AttnScale, softcap: 5f, window: 0, runA);
        float[] runAagain = (float[])runA.Clone();

        // The same window walked again on the same step answers its own bytes: the walk reads only the arena, so no
        // leftover state from an earlier walk can shift a result.
        float[] runC = new float[Rows * Heads * HeadDim];
        first.Attend(0, 0, AttnScale, softcap: 5f, window: 0, runC);
        BatchKernelTests.AssertBitsEqual(runAagain, runC, "one device step walked twice");

        float[] runB = new float[Rows * Heads * HeadDim];
        second.StageWindow(qRows, kRows, vRows);
        second.WriteKv(0, 0);
        second.Attend(0, 0, AttnScale, softcap: 5f, window: 0, runB);
        BatchKernelTests.AssertBitsEqual(runAagain, runB, "two device steps over the same window");
    }

    [Fact]
    public void DeviceStep_AnswersIdenticalBytesUnderBothWeightResidencyPostures()
    {
        const int Heads = 2;
        const int KvHeads = 2;
        const int HeadDim = 8;
        const int MaxPositions = 16;
        const int Rows = 3;

        float[] qRows = QueryRows(Rows, Heads, HeadDim, seedOffset: 31);
        float[] kRows = KeyValueRows(Rows, KvHeads, HeadDim, seedOffset: 131);
        float[] vRows = KeyValueRows(Rows, KvHeads, HeadDim, seedOffset: 231);

        float[] cos = new float[MaxPositions * HeadDim / 2];
        float[] sin = new float[MaxPositions * HeadDim / 2];
        LayerOps.BuildRopeTables(thetaBase: 10_000.0, HeadDim, MaxPositions, cos, sin);

        using CudaBackend copied = OpenCuda();
        using CudaBackend mapped = OpenCuda(CudaWeightResidency.HostMapped);

        using CudaDeviceStep copiedStep = OpenStep(copied, Heads, KvHeads, HeadDim, layerCount: 1, MaxPositions);
        using CudaDeviceStep mappedStep = OpenStep(mapped, Heads, KvHeads, HeadDim, layerCount: 1, MaxPositions);

        copiedStep.LoadRopeTables(cos, sin);
        mappedStep.LoadRopeTables(cos, sin);

        float[] copiedOut = new float[Rows * Heads * HeadDim];
        copiedStep.RunStep(0, 0, qRows, kRows, vRows, AttnScale, softcap: 0f, window: 0, copiedOut);

        float[] mappedOut = new float[Rows * Heads * HeadDim];
        mappedStep.RunStep(0, 0, qRows, kRows, vRows, AttnScale, softcap: 0f, window: 0, mappedOut);

        BatchKernelTests.AssertBitsEqual(copiedOut, mappedOut, "device step over both weight-residency postures");
    }

    // ---------------------------------------------------------------- refusal posture

    [Fact]
    public void DeviceStepRequest_RefusesHeadGeometriesItsMappingCannotWalk_NamingEach()
    {
        ArgumentException nonMultiple = Assert.Throws<ArgumentException>(
            () => new CudaDeviceStepRequest
            {
                MaxPositions = 8,
                LayerCount = 1,
                HeadCount = 6,
                KvHeadCount = 4,
                HeadDim = 8,
            }.Validate());
        Assert.Contains("not a whole multiple of KvHeadCount 4", nonMultiple.Message, StringComparison.Ordinal);

        // The attention kernel maps each query head to its kv-head by a bit shift, so a group size that is not a
        // power of two refuses at open time rather than walking the wrong kv-head mid-step.
        ArgumentException nonPowerOfTwo = Assert.Throws<ArgumentException>(
            () => new CudaDeviceStepRequest
            {
                MaxPositions = 8,
                LayerCount = 1,
                HeadCount = 9,
                KvHeadCount = 3,
                HeadDim = 8,
            }.Validate());
        Assert.Contains("a group of 3 query heads per kv-head", nonPowerOfTwo.Message, StringComparison.Ordinal);
        Assert.Contains("power of two", nonPowerOfTwo.Message, StringComparison.Ordinal);

        foreach ((int heads, int kvHeads) in new[] { (2, 2), (4, 2), (8, 1), (16, 2), (12, 3) })
        {
            new CudaDeviceStepRequest
            {
                MaxPositions = 8,
                LayerCount = 1,
                HeadCount = heads,
                KvHeadCount = kvHeads,
                HeadDim = 8,
            }.Validate();
        }
    }

    [Fact]
    public void DeviceStep_RefusesUnsupportedPositionLayerAndWindowOrdering_NamingEach()
    {
        using CudaBackend cuda = OpenCuda();
        using CudaDeviceStep step = OpenStep(cuda, heads: 4, kvHeads: 2, headDim: 8, layerCount: 2, maxPositions: 16);

        step.StageWindow(
            QueryRows(2, 4, 8, seedOffset: 5),
            KeyValueRows(2, 2, 8, seedOffset: 5),
            KeyValueRows(2, 2, 8, seedOffset: 5));

        ArgumentOutOfRangeException layerError = Assert.Throws<ArgumentOutOfRangeException>(
            () => step.WriteKv(layer: 2, positionBase: 0));
        Assert.Contains("layer 2", layerError.Message, StringComparison.Ordinal);
        Assert.Contains("2 arena layers", layerError.Message, StringComparison.Ordinal);

        ArgumentOutOfRangeException positionError = Assert.Throws<ArgumentOutOfRangeException>(
            () => step.WriteKv(layer: 0, positionBase: 15));
        Assert.Contains("outside this step's MaxPositions 16", positionError.Message, StringComparison.Ordinal);

        Assert.Throws<InvalidOperationException>(() => step.ApplyRope(positionBase: 0)); // no resident tables yet

        step.ClearWindow();
        InvalidOperationException stagedError = Assert.Throws<InvalidOperationException>(
            () => step.Attend(0, 0, AttnScale, 0f, 0, new float[2 * 4 * 8]));
        Assert.Contains("no staged window rows", stagedError.Message, StringComparison.Ordinal);

        step.StageWindow(
            QueryRows(2, 4, 8, seedOffset: 5),
            KeyValueRows(2, 2, 8, seedOffset: 5),
            KeyValueRows(2, 2, 8, seedOffset: 5));

        ArgumentException outputError = Assert.Throws<ArgumentException>(
            () => step.Attend(0, 0, AttnScale, 0f, 0, new float[2 * 4 * 8 - 1]));
        Assert.Contains("attention output must cover 2 rows of HeadCount x HeadDim (32 elements each)", outputError.Message, StringComparison.Ordinal);

        ArgumentException raggedError = Assert.Throws<ArgumentException>(
            () => step.StageWindow(
                QueryRows(2, 4, 8, seedOffset: 5),
                KeyValueRows(2, 2, 8, seedOffset: 5),
                KeyValueRows(1, 2, 8, seedOffset: 5)));
        Assert.Contains("got 32 and 16", raggedError.Message, StringComparison.Ordinal);

        ArgumentException widthError = Assert.Throws<ArgumentException>(
            () => step.StageWindow(new float[70 * 4 * 8], new float[70 * 2 * 8], new float[70 * 2 * 8]));
        Assert.Contains("exceeds this step's MaxWindowRows 64", widthError.Message, StringComparison.Ordinal);

        ArgumentException tableError = Assert.Throws<ArgumentException>(
            () => step.LoadRopeTables(new float[15 * 4], new float[16 * 4]));
        Assert.Contains("need 64 entries each (16 positions x 4 pairs)", tableError.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DeviceStep_TracksOpenSteps_AndBackendDisposeReleasesAnUndisposedStep()
    {
        CudaBackend cuda = OpenCuda();
        Assert.Equal(0, cuda.OpenDeviceStepCount);

        CudaDeviceStep first = cuda.OpenDeviceStep(new CudaDeviceStepRequest
        {
            MaxPositions = 8,
            LayerCount = 1,
            HeadCount = 2,
            KvHeadCount = 2,
            HeadDim = 4,
        });
        CudaDeviceStep second = cuda.OpenDeviceStep(new CudaDeviceStepRequest
        {
            MaxPositions = 8,
            LayerCount = 1,
            HeadCount = 2,
            KvHeadCount = 2,
            HeadDim = 4,
        });

        Assert.Equal(2, cuda.OpenDeviceStepCount);
        Assert.True(first.DeviceBytes > 0);

        first.Dispose();
        Assert.Equal(1, cuda.OpenDeviceStepCount);
        Assert.Throws<ObjectDisposedException>(() => first.StageQueries(new float[2 * 2 * 4]));

        // The backend releases whatever step is still open, so a caller that never disposed one still gets its
        // allocations returned rather than a leak into the context it shares.
        cuda.Dispose();
        Assert.Throws<ObjectDisposedException>(() => second.StageQueries(new float[2 * 2 * 4]));
    }

    [Fact]
    public void DeviceStep_OpeningAStepLeavesWeightResidencyAndExistingKernelsUntouched()
    {
        const int Rows = 4;
        const int Cols = 32;

        TensorShape shape = new([Rows, Cols]);
        float[] weightValues = Filled(Rows * Cols, seedOffset: 0);
        byte[] bytes = new byte[Rows * Cols * sizeof(float)];
        MemoryMarshal.AsBytes(weightValues.AsSpan()).CopyTo(bytes);

        float[] x = Filled(Cols, seedOffset: 3);

        using CpuBackend cpu = new();
        float[] expected = new float[Rows];
        for (int r = 0; r < Rows; r++)
        {
            expected[r] = cpu.Dot(x, weightValues.AsSpan(r * Cols, Cols));
        }

        using CudaBackend cuda = OpenCuda();
        Assert.True(cuda.SupportsGemv(QuantFormat.F32));
        Assert.True(cuda.SupportsGemm(QuantFormat.F32));

        float[] beforeWeights = new float[Rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.F32), beforeWeights);
        AssertWithinBand(expected, beforeWeights, 5e-5f, "F32 GEMV before any device step opened");
        int copiesBefore = cuda.DeviceWeightCopies;
        long weightBytesBefore = cuda.PersistentDeviceWeightBytes;
        Assert.Equal(0, cuda.OpenKvMirrorCount);

        using CudaDeviceStep step = cuda.OpenDeviceStep(new CudaDeviceStepRequest
        {
            MaxPositions = 16,
            LayerCount = 1,
            HeadCount = 2,
            KvHeadCount = 2,
            HeadDim = 8,
        });

        Assert.Equal(copiesBefore, cuda.DeviceWeightCopies); // a step opens no weight residency
        Assert.Equal(weightBytesBefore, cuda.PersistentDeviceWeightBytes);
        Assert.Equal(0, cuda.OpenKvMirrorCount); // a step's arena is its own storage, not a mirror

        float[] afterWeights = new float[Rows];
        cuda.Gemv(x, new WeightRef(bytes, shape, QuantFormat.F32), afterWeights);
        BatchKernelTests.AssertBitsEqual(beforeWeights, afterWeights, "F32 GEMV after a device step opened");

        float[] afterGemm = new float[Rows];
        cuda.GemmBatch(x, 1, new WeightRef(bytes, shape, QuantFormat.F32), afterGemm);
        BatchKernelTests.AssertBitsEqual(beforeWeights, afterGemm, "F32 blocked matmul after a device step opened");
    }

    // ---------------------------------------------------------------- fixtures and helpers

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

    private static CudaDeviceStep OpenStep(
        CudaBackend backend,
        int heads,
        int kvHeads,
        int headDim,
        int layerCount,
        int maxPositions,
        RopePairing pairing = RopePairing.AdjacentPairs) =>
        backend.OpenDeviceStep(new CudaDeviceStepRequest
        {
            MaxPositions = maxPositions,
            LayerCount = layerCount,
            HeadCount = heads,
            KvHeadCount = kvHeads,
            HeadDim = headDim,
            Pairing = pairing,
        });

    private static void ApplyRopeToRegion(
        float[] data,
        int region,
        int headDim,
        RopePairing pairing,
        float[] cos,
        float[] sin,
        int position)
    {
        int half = headDim / 2;
        if (pairing == RopePairing.AdjacentPairs)
        {
            LayerOps.ApplyRopeToHead(data.AsSpan(region, headDim), half, cos.AsSpan(position * half, half), sin.AsSpan(position * half, half));
        }
        else
        {
            LayerOps.ApplyRopeToSplitHalfHead(data.AsSpan(region, headDim), half, cos.AsSpan(position * half, half), sin.AsSpan(position * half, half));
        }
    }

    // The shipped CPU attention over the same geometry: a host KV sequence holds every written position's K and V and
    // each row attends its own causal range through LayerOps.CausalAttention — the per-row walk the tiled host kernel
    // is documented to answer bit-for-bit.
    private static float[] HostAttention(
        float[] qRows,
        int rows,
        int heads,
        int kvHeads,
        int headDim,
        int layer,
        float[] kRows,
        float[] vRows,
        int writtenRows,
        float attnScale,
        float softcap,
        int window,
        int positionBase = 0)
    {
        using CpuBackend cpu = new();
        using BlockPool pool = new(
            blockTokens: 16,
            layerCount: layer + 1,
            kvHeads: kvHeads,
            headDim: headDim,
            capacityBlocks: writtenRows / 16 + 2,
            backend: cpu);

        SequenceKv kv = new(pool);
        try
        {
            for (int pos = 0; pos < writtenRows; pos++)
            {
                kv.Append();
                kv.WriteK(pos, layer, new ReadOnlySpan<float>(kRows, pos * kvHeads * headDim, kvHeads * headDim));
                kv.WriteV(pos, layer, new ReadOnlySpan<float>(vRows, pos * kvHeads * headDim, kvHeads * headDim));
            }

            float[] outRows = new float[rows * heads * headDim];
            for (int r = 0; r < rows; r++)
            {
                LayerOps.CausalAttention(
                    cpu,
                    new ReadOnlySpan<float>(qRows, r * heads * headDim, heads * headDim),
                    heads,
                    kvHeads,
                    headDim,
                    kv,
                    layer,
                    positionBase + r,
                    attnScale,
                    softcap,
                    window,
                    outRows.AsSpan(r * heads * headDim, heads * headDim));
            }

            return outRows;
        }
        finally
        {
            kv.Dispose();
        }
    }

    // The relative-only shape, kept for the pre-existing seam check that compares a device GEMV against the host one.
    private static void AssertWithinBand(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float band, string context) =>
        AssertWithinBand(expected, actual, band, absoluteCeiling: null, context);

    // The device-step shape: an element passes when its drift stays inside the relative band or under the absolute
    // ceiling, whichever is wider for that element. The ceiling carries the elements whose value comes out of
    // cancellation — their magnitude shrinks while their error is inherited from the rows they were summed from, so a
    // pure relative claim on them measures how close two numerics paths came to zero rather than how accurate either
    // one was. Every element still has to land within one band or the other; nothing here skips an element.
    private static void AssertWithinBand(
        ReadOnlySpan<float> expected,
        ReadOnlySpan<float> actual,
        float band,
        float absoluteCeiling,
        string context) =>
        AssertWithinBand(expected, actual, band, (float?)absoluteCeiling, context);

    private static void AssertWithinBand(
        ReadOnlySpan<float> expected,
        ReadOnlySpan<float> actual,
        float band,
        float? absoluteCeiling,
        string context)
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

            double allowance = absoluteCeiling is null ? band : Math.Max((double)band * scale, absoluteCeiling.Value);
            bool outside = absoluteCeiling is null ? relative > allowance : delta > allowance;
            if (outside && offender < 0)
            {
                offender = i;
            }
        }

        if (offender >= 0)
        {
            Assert.Fail(
                $"{context}: widest relative drift {worstRelative:E6} at element {relativeIndex} and widest absolute " +
                $"drift {worstAbsolute:E6} at element {absoluteIndex} against the recorded band {band:E2}" +
                (absoluteCeiling is null ? string.Empty : $", ceiling {absoluteCeiling.Value:E2}") +
                $" — offending element {offender}: expected {expected[offender]:E9}, actual {actual[offender]:E9}.");
        }
    }
}
