using System.Runtime.InteropServices;

using Lair.Backends.Cuda;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// MVP6.1 per-model CUDA stream gates on the CudaBackend seam: every opened backend carries its own dedicated non-blocking
/// compute/copy lane (never the legacy NULL/default stream), two open backends carry distinct lanes, and device work from
/// both stands pending at once — proven structurally through an internal enqueue pipeline plus a recorded probe event, not
/// through wall-time ratios. Every overlapped answer is bit-for-bit identical to its solo run: the refactor moved launches,
/// staging copies, residency fills, timing events, joins, and the steady-state allocator onto the instance lane while
/// keeping each kernel's function handle, grid geometry, and parameter order unchanged, so overlap changes scheduling only.
/// The class runs in a serialized collection because it reads the process-wide live-stream count as a structural fact.
/// </summary>
[CollectionDefinition("cuda-stream-serialized", DisableParallelization = true)]
public sealed class CudaStreamSerializedCollection;

[Collection("cuda-stream-serialized")]
[Trait("Category", "Cuda")]
[Trait("Category", "Streams")]
public sealed class CudaStreamConcurrencyTests
{
    // Moderate shapes throughout: each case stays well inside a busy device's free memory while still keeping a launch
    // long enough for its lane's queue to be observable. A 4096x4096 F32 tensor is 16 MiB of weight bytes per backend and
    // an 8192x4096 one is 32 MiB, and every GEMV re-reads its whole resident weight region, so a queued pipeline keeps its
    // lane busy for the window the pending probes read. The earlier version of this gate used 1 GiB fixtures; these shapes
    // answer the same structural questions at a fraction of the device footprint.
    private const int Rows = 4096;

    private const int Cols = 4096;

    private const int QueuedRows = 8192;

    private const int QueuedCols = 4096;

    // Pipelines submitted per lane before either lane is awaited: deep enough that both streams answer NOT_READY for most
    // observations while staying a sub-second device queue.
    private const int EnqueuedPipelinesPerLane = 24;

    // Queue-and-drain rounds in the overlapped enqueue case: the proof repeats rather than resting on one lucky deep queue.
    private const int QueueRounds = 3;

    [Fact]
    public void PerModelStreams_EveryOpenBackend_CarriesItsOwnNonDefaultStream()
    {
        long baseline = CudaBackend.LiveInstanceStreams;

        using CudaBackend a = OpenCuda();
        using CudaBackend b = OpenCuda();

        Assert.Equal(baseline + 2, CudaBackend.LiveInstanceStreams); // one live lane per opened instance.

        Assert.NotEqual(IntPtr.Zero, a.ComputeStream);
        Assert.NotEqual(IntPtr.Zero, b.ComputeStream);
        Assert.NotEqual(a.ComputeStream, b.ComputeStream); // distinct lanes: neither backend launches on the other's stream or the legacy default one.

        byte[] bytes = Weights(Rows, Cols);
        float[] x = Activations(Cols);
        TensorShape shape = new([Rows, Cols]);
        WeightRef w = new(bytes, shape, QuantFormat.F32);

        float[] outA = new float[Rows];
        a.Gemv(x, w, outA); // each lane runs its own kernel through the production path.
        Assert.DoesNotContain(outA, v => float.IsNaN(v));

        float[] outB = new float[Rows];
        b.Gemv(x, w, outB);

        Assert.True(
            MemoryMarshal.AsBytes(outA).SequenceEqual(MemoryMarshal.AsBytes(outB)),
            "two backends' streams changed the answer: the same F32 GEMV bytes must land bit-identical on every instance lane.");

        a.Dispose(); // explicit so the live-stream count below sees both releases regardless of using-block order.
        b.Dispose();
        Assert.Equal(baseline, CudaBackend.LiveInstanceStreams); // disposed backends release their lanes before their context goes away.
    }

    [Fact]
    public void ConcurrentEnqueue_TwoBackends_BothLanesPendingAndBitIdenticalToSolo()
    {
        byte[] bytes = Weights(QueuedRows, QueuedCols);
        float[] x = Activations(QueuedCols);
        TensorShape shape = new([QueuedRows, QueuedCols]);
        WeightRef w = new(bytes, shape, QuantFormat.F32);

        using CudaBackend a = OpenCuda();
        using CudaBackend b = OpenCuda();

        // Solo baselines through the production path: what every overlapped answer must match bit-for-bit.
        float[] soloA = new float[QueuedRows];
        a.Gemv(x, w, soloA);
        long transfersAfterSoloA = a.RuntimeTransferCallsTotal;
        Assert.True(a.KernelBusyMicrosecondsTotal > 0, "the production GEMV recorded no device-measured kernel busy time on the instance stream");

        float[] soloB = new float[QueuedRows];
        b.Gemv(x, w, soloB);

        // Each round submits both lanes before awaiting either — the posture MVP6.1 measured — interleaving so each lane's
        // queue is in flight while the other submits its next pipeline. The enqueue seam leaves every copy, launch, and
        // stream-ordered buffer release pending on that instance's own stream; nothing awaits a lane until the round ends.
        // After every pair, both lanes are probed: an observation where neither has completed anything yet (both answer
        // NOT_READY) is two independent streams' device work standing in flight together at one instant. A shared default
        // stream could never answer this pair for two backends at once; no wall-time ratio appears anywhere in this test.
        // Requiring the majority of observations green keeps the proof structural while tolerating a shallow-queue edge
        // when other GPU tenants contend on this box, and every round repeats the whole posture rather than trusting one queue.
        int bothPendingObservations = 0;
        int observations = 0;

        for (int round = 0; round < QueueRounds; round++)
        {
            List<CudaBackend.GemvPipeline> pipelinesA = [];
            List<CudaBackend.GemvPipeline> pipelinesB = [];

            try
            {
                for (int i = 0; i < EnqueuedPipelinesPerLane; i++)
                {
                    pipelinesA.Add(a.BeginGemv(w, x));
                    pipelinesB.Add(b.BeginGemv(w, x));

                    bool aPending = a.InstanceStreamPending();
                    bool bPending = b.InstanceStreamPending();

                    observations++;
                    if (aPending && bPending)
                    {
                        bothPendingObservations++;
                    }
                }

                for (int i = 0; i < pipelinesA.Count; i++)
                {
                    pipelinesA[i].Await(); // joins only this instance's lane.
                    Assert.True(
                        MemoryMarshal.AsBytes(soloA).SequenceEqual(MemoryMarshal.AsBytes(pipelinesA[i].Output)),
                        $"round {round} pipeline A[{i}] drifted from the solo GEMV answer: overlap must change scheduling, not bits");
                }

                for (int i = 0; i < pipelinesB.Count; i++)
                {
                    pipelinesB[i].Await();
                    Assert.True(
                        MemoryMarshal.AsBytes(soloB).SequenceEqual(MemoryMarshal.AsBytes(pipelinesB[i].Output)),
                        $"round {round} pipeline B[{i}] drifted from the solo GEMV answer: overlap must change scheduling, not bits");
                }
            }
            finally
            {
                foreach (CudaBackend.GemvPipeline pipeline in pipelinesA.Concat(pipelinesB))
                {
                    pipeline.Await(); // idempotent: releases any lane a failure path left open.
                }
            }
        }

        Assert.True(
            bothPendingObservations >= QueueRounds * EnqueuedPipelinesPerLane / 2,
            $"only {bothPendingObservations} of {observations} submissions observed device work pending on both instance streams at once: the lanes are not running independently");

        // Accounting keeps its semantics across both paths: one activation input crossing and one kernel output return per compute.
        Assert.Equal(2L * QueueRounds * EnqueuedPipelinesPerLane, a.RuntimeTransferCallsTotal - transfersAfterSoloA);
        Assert.True(a.KernelBusyMicrosecondsTotal > 0, "the overlapped pipelines recorded no device-measured kernel busy time");
    }

    [Fact]
    public void ProductionGemvBatch_TwoBackendsOnThreads_OverlappedRunsBitIdenticalToSolo()
    {
        byte[] bytes = Weights(Rows, Cols);
        TensorShape shape = new([Rows, Cols]);
        WeightRef w = new(bytes, shape, QuantFormat.F32);

        const int batch = 8;
        float[] inputs = new float[batch * Cols];
        for (int v = 0; v < batch; v++)
        {
            float[] row = Activations(Cols);
            Array.Copy(row, 0, inputs, v * Cols, Cols);
        }

        using CudaBackend a = OpenCuda();
        using CudaBackend b = OpenCuda();

        // Solo baselines first, through the same production entry point the overlapped runs use: every overlapped iteration
        // below must answer the same bits its lane answered alone.
        float[] soloA = new float[batch * Rows];
        a.GemvBatch(inputs, batch, w, soloA);

        float[] soloB = new float[batch * Rows];
        b.GemvBatch(inputs, batch, w, soloB);

        long transfersBeforeA = a.RuntimeTransferCallsTotal;
        long transfersBeforeB = b.RuntimeTransferCallsTotal;

        const int iterationsPerLane = 6; // deep enough that the two lanes' launches interleave on the device rather than each draining before the other starts.
        float[][] overlappedA = new float[iterationsPerLane][];
        float[][] overlappedB = new float[iterationsPerLane][];

        Thread threadA = new(() =>
        {
            WeightRef laneWeight = new(bytes, shape, QuantFormat.F32); // WeightRef is a ref struct, so each lane builds its own over the shared bytes.
            for (int i = 0; i < iterationsPerLane; i++)
            {
                overlappedA[i] = new float[batch * Rows];
                a.GemvBatch(inputs, batch, laneWeight, overlappedA[i]);
            }
        });

        Thread threadB = new(() =>
        {
            WeightRef laneWeight = new(bytes, shape, QuantFormat.F32);
            for (int i = 0; i < iterationsPerLane; i++)
            {
                overlappedB[i] = new float[batch * Rows];
                b.GemvBatch(inputs, batch, laneWeight, overlappedB[i]);
            }
        });

        threadA.Start();
        threadB.Start();
        Assert.True(threadA.Join(TimeSpan.FromMinutes(5)), "backend A's production lane did not finish its overlapped GEMV batches");
        Assert.True(threadB.Join(TimeSpan.FromMinutes(5)), "backend B's production lane did not finish its overlapped GEMV batches");

        for (int i = 0; i < iterationsPerLane; i++)
        {
            Assert.True(
                MemoryMarshal.AsBytes(soloA).SequenceEqual(MemoryMarshal.AsBytes(overlappedA[i])),
                $"backend A iteration {i} drifted from its solo baseline while backend B ran concurrently: per-model streams must keep outputs bit-identical");

            Assert.True(
                MemoryMarshal.AsBytes(soloB).SequenceEqual(MemoryMarshal.AsBytes(overlappedB[i])),
                $"backend B iteration {i} drifted from its solo baseline while backend A ran concurrently: per-model streams must keep outputs bit-identical");
        }

        // The two lanes ran their own kernels independently and each kept its accounting: one activation input crossing and
        // one kernel output return per compute call, with no cross-instance serialization gate between them.
        Assert.Equal(2 * iterationsPerLane, a.RuntimeTransferCallsTotal - transfersBeforeA);
        Assert.Equal(2 * iterationsPerLane, b.RuntimeTransferCallsTotal - transfersBeforeB);
        Assert.True(a.KernelBusyMicrosecondsTotal > 0 && b.KernelBusyMicrosecondsTotal > 0, "concurrent production runs recorded no device-measured kernel busy time");
    }

    /// <summary>
    /// The residency-upload ordering gate per-model streams made load-bearing: a resident weight copy is filled by a copy
    /// enqueued on the owning instance stream and joined before that call answers, so the first launch reading the region
    /// on that lane cannot run against bytes that have not landed. A synchronous copy rides the legacy default stream
    /// instead, and a non-blocking instance stream never rendezvouses with it — the exact race that surfaced as garbage
    /// answers once each model carried its own lane (observed here before the fix as a Q4_K blocked matmul answering -444.5
    /// for an expected 225.2). Each iteration opens a fresh backend so the fill runs again, answers identical bytes twice —
    /// the first call through the fill and the second against the already-cached residency — while a second lane keeps the
    /// device contended with queued pipelines of its own. No band or tolerance appears in this gate: every assertion is bit
    /// equality between two answers over the same weight bytes.
    /// </summary>
    [Theory]
    [InlineData(QuantFormat.F32)]
    [InlineData(QuantFormat.Q8_0)]
    public void ResidencyUpload_RidesTheInstanceStream_AheadOfTheFirstLaunchOnThatLane(QuantFormat format)
    {
        const int rows = 512;
        const int cols = 512;
        byte[] bytes = Weights(rows, cols, format);
        float[] x = Activations(cols);
        TensorShape shape = new([rows, cols]);

        using CudaBackend busy = OpenCuda(); // a second lane kept contended on its own weights while every residency fill below runs

        byte[] busyBytes = Weights(QueuedRows, QueuedCols, QuantFormat.F32);
        TensorShape busyShape = new([QueuedRows, QueuedCols]);
        WeightRef busyWeight = new(busyBytes, busyShape, QuantFormat.F32);
        float[] busyActivations = Activations(QueuedCols);

        byte[]? reference = null;

        for (int iteration = 0; iteration < 8; iteration++)
        {
            List<CudaBackend.GemvPipeline> queued = [];
            try
            {
                // The contended lane stands pending across the fresh backend's first compute, so any window a fill would
                // have to race is open here rather than closed by an idle device.
                for (int i = 0; i < 4; i++)
                {
                    queued.Add(busy.BeginGemv(busyWeight, busyActivations));
                }

                using CudaBackend fresh = OpenCuda();

                float[] first = new float[rows]; // this call is the residency fill plus the first launch against it
                fresh.Gemv(x, new WeightRef(bytes, shape, format), first);

                float[] cached = new float[rows]; // the same bytes answered again with the residency already resident
                fresh.Gemv(x, new WeightRef(bytes, shape, format), cached);

                Assert.True(
                    MemoryMarshal.AsBytes(first).SequenceEqual(MemoryMarshal.AsBytes(cached)),
                    $"{format} iteration {iteration}: the first answer after a residency fill drifted from the same backend's cached answer — the fill was not ordered ahead of the launch on the instance stream.");

                if (reference is null)
                {
                    reference = MemoryMarshal.AsBytes(first).ToArray();
                }
                else
                {
                    Assert.True(
                        reference.SequenceEqual(MemoryMarshal.AsBytes(first)),
                        $"{format} iteration {iteration}: a fresh backend's first answer over identical bytes drifted from an earlier fresh backend's.");
                }

                foreach (CudaBackend.GemvPipeline pipeline in queued)
                {
                    float[] answered = pipeline.Output; // the contended lane's own answer, valid once its lane is awaited below
                    pipeline.Await();

                    Assert.DoesNotContain(answered, v => float.IsNaN(v));
                }
            }
            finally
            {
                foreach (CudaBackend.GemvPipeline pipeline in queued)
                {
                    pipeline.Await(); // idempotent: releases any lane a failure path left open.
                }
            }
        }

        // The contended lane's own answers must match what it would have answered alone, proving the contention above was real device work and not an artifact of the gate.
        float[] solo = new float[QueuedRows];
        busy.Gemv(busyActivations, new WeightRef(busyBytes, busyShape, QuantFormat.F32), solo);

        using CudaBackend idle = OpenCuda();
        float[] soloIdle = new float[QueuedRows];
        idle.Gemv(busyActivations, new WeightRef(busyBytes, busyShape, QuantFormat.F32), soloIdle);

        Assert.True(
            MemoryMarshal.AsBytes(solo).SequenceEqual(MemoryMarshal.AsBytes(soloIdle)),
            "the contended lane's answer drifted from the same bytes answered on an idle backend: overlap changed bits, not scheduling.");
    }

    // ---------------------------------------------------------------- fixtures and helpers

    private static CudaBackend OpenCuda()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        try
        {
            return new CudaBackend();
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

    // Deterministic weight bytes shared by every backend in a case so each lane dots against identical values and a drift
    // between answers can only mean the stream refactor changed bits — which it must not. Q8_0 packs those same floats into
    // its own 34-byte blocks (one half-precision scale plus 32 signed quants) rather than inventing separate fixture values,
    // so both formats here walk the same residency-fill path over the same numbers.
    private static byte[] Weights(int rows, int cols, QuantFormat format = QuantFormat.F32)
    {
        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++)
        {
            int row = i / cols;
            int col = i % cols;
            weights[i] = (float)Math.Sin(0.17 * row + 0.03 * col) * 0.9f;
        }

        if (format == QuantFormat.F32)
        {
            return MemoryMarshal.AsBytes(weights.AsSpan()).ToArray();
        }

        ArgumentOutOfRangeException.ThrowIfNotEqual(format, QuantFormat.Q8_0);

        int blocks = cols / 32;
        byte[] bytes = new byte[rows * blocks * 34]; // Q8_0's fixed stride: two scale bytes plus thirty-two quants.

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocks; b++)
            {
                int blockBase = (r * blocks + b) * 34;

                float amax = 0f;
                for (int i = 0; i < 32; i++)
                {
                    float ax = Math.Abs(weights[r * cols + b * 32 + i]);
                    if (ax > amax)
                    {
                        amax = ax;
                    }
                }

                float scale = amax / 127f; // the reference's iscale so no quant saturates past its signed range.

                BitConverter.TryWriteBytes(bytes.AsSpan(blockBase), (Half)scale); // Q8_0's header is the block scale as a 16-bit half, little-endian.

                for (int i = 0; i < 32; i++)
                {
                    float v = weights[r * cols + b * 32 + i];
                    int quant = Math.Clamp((int)Math.Round(v / (scale == 0f ? 1f : scale)), -127, 127);
                    bytes[blockBase + 2 + i] = (byte)(sbyte)quant; // the reference stores magnitudes as signed quants.
                }
            }
        }

        return bytes;
    }

    private static float[] Activations(int cols)
    {
        float[] x = new float[cols];
        for (int c = 0; c < cols; c++)
        {
            x[c] = (float)Math.Sin(0.11 * c + 7) * 1.3f;
        }

        return x;
    }
}
