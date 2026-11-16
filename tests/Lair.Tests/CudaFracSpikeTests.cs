using System.Buffers;
using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Mapped-host residency over the synthetic F16 llama fixture: the model's weight bytes are read once into page-aligned
/// unmanaged host allocations owned by an anonymous weight source, and a CUDA span opened in host-mapped residency
/// registers those live regions with the driver and launches its kernels against the device pointers the registrations
/// answer with — no weight tensor is copied into persistent device memory on this posture, and the copy counter stays
/// at zero while the mapped-region counter names every transformer matrix the span registered once. The route is a
/// hybrid: the transformer span computes on the mapped CUDA backend from layer 0, and the tail span plus the output
/// head run on the opened CPU backend against the same host-resident source. The token stream must be identical to a
/// CPU-only baseline of the same file, the crossing facts must name the cpu-side tail at the layer-count boundary, and
/// the opened CPU backend must answer the head's weight passes while running fewer counted passes than the baseline.
/// The default posture is proven beside it: the same file opened whole on a default CUDA backend crosses every dense
/// matrix and the tied head to device memory, while the same whole-model route on the mapped posture registers the same
/// set of host regions and copies none.
/// </summary>
[Trait("Category", "Parity")]
[Trait("Category", "Scheduling")]
[Trait("Category", "Cuda")]
public sealed class CudaFracSpikeTests : IDisposable
{
    private const int DecodeSteps = 4;

    private static readonly int[] PromptTokens = [1, 2, 3];

    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
                // a temp file this run could not remove is no one's problem but its own
            }
        }
    }

    [Fact]
    public void HostMappedCudaTransformerSpan_AndCpuTail_OverHostResidentWeights_DecodesSameTokensAsCpuOnlyBaseline()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        string path = SyntheticLlamaGguf.Write("frac-host-mapped-f16", withReasoningMarkers: true, QuantFormat.F16);
        _tempFiles.Add(path);

        List<int> baselineTokens;
        int baselineCpuPasses;
        int layerCount;

        using GgufFile gguf = GgufFile.Open(path);

        // CPU-only baseline: the default single-CPU-segment route, every weight pass visible on the opened backend.
        try
        {
            using CpuBackend cpuOnlyInner = new();
            CountingDecodeBackend countingBaseline = new(cpuOnlyInner);
            using TransformerModel baseline = TransformerModel.Open(gguf, countingBaseline, maxContextTokens: 32, logLine: null, PrefillKernel.GemV);

            Assert.Single(baseline.Plan.Route.Segments);
            Assert.Equal(ModelRoute.CpuDeviceId, baseline.Plan.Route.Segments[0].DeviceId);

            layerCount = baseline.Desc.LayerCount;
            baselineTokens = GreedyDecode(baseline, countingBaseline);
            baselineCpuPasses = countingBaseline.GemvCalls + countingBaseline.GemvBatchCalls;
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }

        List<int> fracTokens;
        int deviceWeightCopies;
        int hostMappedRegions;
        int cpuTailPasses;
        TransformerModel.ExecutionCrossing[] stagingFacts;

        using CpuBackend tailInner = new();
        CountingDecodeBackend countingTail = new(tailInner); // the service-side CPU backend, visible as it answers its own span and the output head.
        IBackend tail = countingTail;

        try
        {
            using HostResidentWeightSource weights = new(gguf); // every tensor payload lives in its own page-aligned anonymous allocation for the source's lifetime.
            using CudaBackend hostMapped = new(CudaWeightResidency.HostMapped); // the mapped posture: kernels launch against registered host regions, never device copies.

            ModelRoute frac = CudaFirstCpuTailRoute("frac-host-mapped", layerCount);

            using TransformerModel fracModel = TransformerModel.Open(
                gguf, weights, tail, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, frac, [hostMapped, tail]); // both resolved positions answered explicitly.

            Assert.Equal(2, frac.Segments.Count);
            Assert.Equal(CudaBackend.DeviceIdValue, frac.Segments[0].DeviceId); // the transformer span names device 'cuda' from layer 0.
            Assert.Equal(ModelRoute.CpuDeviceId, frac.Segments[1].DeviceId);    // the tail range and the output head run on cpu.
            Assert.Equal(CudaWeightResidency.HostMapped, hostMapped.WeightResidency);

            stagingFacts = [.. fracModel.ExecutionStaging];

            fracTokens = GreedyDecode(fracModel, tail);

            deviceWeightCopies = hostMapped.DeviceWeightCopies;
            hostMappedRegions = hostMapped.HostMappedWeightRegions;
            cpuTailPasses = countingTail.GemvCalls + countingTail.GemvBatchCalls;
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }

        Assert.Equal(baselineTokens, fracTokens); // prompt window and continued steps token-identical across the mapped crossing

        TransformerModel.ExecutionCrossing tailCrossing = Assert.Single(stagingFacts); // one crossing into the cpu-side tail span
        Assert.Equal(1, tailCrossing.SegmentIndex);
        Assert.Equal(layerCount, tailCrossing.LayerStart);
        Assert.Equal(layerCount, tailCrossing.LayerEnd); // at the layer-count boundary the range is empty: every transformer layer ran on the mapped span and only the output head remains on cpu.
        Assert.Equal(ModelRoute.CpuDeviceId, tailCrossing.StagingOwnerDeviceId);

        Assert.Equal(0, deviceWeightCopies); // the mapped route crossed no weight tensor into persistent device memory
        Assert.Equal(layerCount * 7, hostMappedRegions); // every transformer matrix of the cuda span registered from the host-resident source exactly once
        Assert.True(hostMappedRegions > 0, "the mapped backend registered no host weight region — its span never computed over the anonymous allocations");

        Assert.True(cpuTailPasses > 0, "the opened CPU backend answered no weight pass — its tail span and the output head never executed there");
        Assert.True(cpuTailPasses < baselineCpuPasses, "the mapped span took the transformer layers but the opened CPU backend still ran every counted pass this model took");
    }

    [Fact]
    public void DefaultCudaBackend_CrossesWeightsToDevice_AndHostMappedMode_RunsWholeModelFromRegisteredHostRegions()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        string path = SyntheticLlamaGguf.Write("frac-mode-accounting", withReasoningMarkers: true, QuantFormat.F16);
        _tempFiles.Add(path);

        List<int> deviceCopyTokens;
        int deviceCopies;
        List<int> mappedTokens;
        int mappedCopies;
        int mappedRegions;

        using GgufFile gguf = GgufFile.Open(path);

        try
        {
            // Default posture, unchanged: the whole model on a default-opened CUDA backend copies every dense matrix and the tied head to device memory once per instance.
            using (CudaBackend deviceCopy = new())
            {
                Assert.Equal(CudaWeightResidency.DeviceCopy, deviceCopy.WeightResidency);

                using TransformerModel model = TransformerModel.Open(gguf, deviceCopy, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, CudaWholeModelRoute("frac-device-copy"));
                deviceCopyTokens = GreedyDecode(model, deviceCopy);
                deviceCopies = deviceCopy.DeviceWeightCopies;
            }

            // Mapped posture over the same whole-model route: the same tensors are answered by registered host regions instead, and none is copied.
            using (HostResidentWeightSource weights = new(gguf))
            using (CudaBackend hostMapped = new(CudaWeightResidency.HostMapped))
            {
                using TransformerModel model = TransformerModel.Open(gguf, weights, hostMapped, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, CudaWholeModelRoute("frac-host-mapped-whole"), []);
                mappedTokens = GreedyDecode(model, hostMapped);
                mappedCopies = hostMapped.DeviceWeightCopies;
                mappedRegions = hostMapped.HostMappedWeightRegions;
            }
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }

        int wholeModelTensors = DenseMatrixCount(gguf) + 1; // every per-layer dense matrix plus the tied output head over token-embedding rows

        Assert.True(deviceCopies > 0, "the default posture crossed no weight tensor to device memory");
        Assert.Equal(wholeModelTensors, deviceCopies);
        Assert.Equal(0, mappedCopies); // the mapped posture keeps its copy counter at zero for the same tensor set
        Assert.Equal(wholeModelTensors, mappedRegions);
        Assert.Equal(deviceCopyTokens, mappedTokens); // the same file decodes identically whether its weights ride device copies or registered host regions
    }

    // ---------------------------------------------------------------- helpers

    private static ModelRoute CudaFirstCpuTailRoute(string modelName, int boundary) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = modelName,
            Stages =
            [
                new RouteStage(ModelRoutes.PrefillStage, CudaBackend.DeviceIdValue) { SplitPosition = 0 },
                new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId) { SplitPosition = boundary },
            ],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaBackend.DeviceIdValue],
        });

    private static ModelRoute CudaWholeModelRoute(string modelName) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = modelName,
            Stages = [new(ModelRoutes.PrefillStage, CudaBackend.DeviceIdValue), new(ModelRoutes.DecodeStage, CudaBackend.DeviceIdValue)],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaBackend.DeviceIdValue],
        });

    private static int DenseMatrixCount(GgufFile gguf) => ModelPlans.Resolve(gguf, logLine: null).LayerCount * 7; // q,k,v,o,gate,up,down per layer

    private static List<int> GreedyDecode(TransformerModel model, IBackend poolOwner)
    {
        using BlockPool pool = new(
            blockTokens: 16,
            layerCount: model.Desc.LayerCount,
            kvHeads: model.Desc.KvHeadCount,
            headDim: model.Desc.HeadDim,
            capacityBlocks: 16,
            poolOwner);

        using SequenceKv kv = new(pool);

        int token = ArgMax(model.Prefill(PromptTokens, kv));
        List<int> sampled = [token];

        for (int step = 0; step < DecodeSteps; step++)
        {
            token = ArgMax(model.DecodeNext(token, kv));
            sampled.Add(token);
        }

        return sampled;
    }

    /// <summary>Picks the top id by descending logit with a strict "greater than" scan: equal logits keep the lower id.</summary>
    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        float best = float.NegativeInfinity;
        int bestIdx = -1;

        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > best)
            {
                best = logits[i];
                bestIdx = i;
            }
        }

        Assert.True(bestIdx >= 0, "logit vector had no selectable entry");
        return bestIdx;
    }

    /// <summary>
    /// An anonymous host-resident weight source: every tensor payload the file declares is read once into its own
    /// page-aligned unmanaged allocation and held there for the source's lifetime. Resolutions are stable spans over
    /// those allocations — not file map pages, and not device memory — which is the residency a mapped CUDA span
    /// registers and computes over.
    /// </summary>
    private sealed unsafe class HostResidentWeightSource : IWeightSource
    {
        // Page granularity of the driver's host-registration seam: each allocation starts on this boundary and covers
        // whole pages, so a registration never reaches past its own allocation into a neighbour's pages.
        private const long PageGranularity = 65536;

        private readonly Dictionary<string, (IntPtr Pointer, int Length)> _byName = new(StringComparer.Ordinal);
        private readonly List<(IntPtr Pointer, nuint AllocationBytes)> _allocations = [];
        private bool _disposed;

        public HostResidentWeightSource(GgufFile gguf)
        {
            ArgumentNullException.ThrowIfNull(gguf);

            byte[] fileBytes = File.ReadAllBytes(gguf.Path);

            foreach (LogicalTensor tensor in gguf.Tensors.Tensors)
            {
                int length = checked((int)tensor.SizeBytes);
                long allocationBytes = (tensor.SizeBytes + PageGranularity - 1) / PageGranularity * PageGranularity;

                void* block = NativeMemory.AlignedAlloc((nuint)allocationBytes, (nuint)PageGranularity);
                if (block is null)
                {
                    throw new OutOfMemoryException($"Failed to allocate {allocationBytes} page-aligned bytes for tensor '{tensor.Name}'.");
                }

                _allocations.Add(((IntPtr)block, (nuint)allocationBytes));

                int fileOffset = checked((int)(gguf.DataStart + tensor.Offset));
                fileBytes.AsSpan(fileOffset, length).CopyTo(new Span<byte>(block, length));

                _byName[tensor.Name] = ((IntPtr)block, length);
            }
        }

        public ReadOnlySpan<byte> Resolve(LogicalTensor tensor)
        {
            ArgumentNullException.ThrowIfNull(tensor);
            ObjectDisposedException.ThrowIf(_disposed, this);

            if (!_byName.TryGetValue(tensor.Name, out (IntPtr Pointer, int Length) region))
            {
                throw new GgufException($"tensor '{tensor.Name}' is not held by this host-resident weight source");
            }

            return new ReadOnlySpan<byte>((void*)region.Pointer, region.Length);
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            foreach ((IntPtr pointer, nuint _) in _allocations)
            {
                NativeMemory.AlignedFree((void*)pointer);
            }

            _allocations.Clear();
            _byName.Clear();
        }
    }

    /// <summary>A CPU compute backend that counts the weight passes it was asked to run: what a span resolved onto this engine's own device executes is visible as calls against its kernels.</summary>
    private sealed class CountingDecodeBackend : IBackend
    {
        private readonly CpuBackend _inner;

        public CountingDecodeBackend(CpuBackend inner)
        {
            ArgumentNullException.ThrowIfNull(inner);
            _inner = inner;
        }

        /// <summary>Single-token weight passes this backend ran.</summary>
        public int GemvCalls { get; private set; }

        /// <summary>Batched-row weight passes this backend ran.</summary>
        public int GemvBatchCalls { get; private set; }

        public string DeviceId => _inner.DeviceId;

        public Lair.Contracts.WeightTier WeightTier => Lair.Contracts.WeightTier.Dram;

        public MemoryHandle AllocateAligned(long bytes, int alignment) => _inner.AllocateAligned(bytes, alignment);

        public void FreeAligned(MemoryHandle handle) => _inner.FreeAligned(handle);

        public IKvDeviceMirror OpenKvMirror(KvMirrorRequest request) => _inner.OpenKvMirror(request);

        public void Gemv(ReadOnlySpan<float> x, WeightRef w, Span<float> output)
        {
            GemvCalls++;
            _inner.Gemv(x, w, output);
        }

        public void GemvBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs)
        {
            GemvBatchCalls++;
            _inner.GemvBatch(inputs, count, w, outputs); // a batched-row weight pass through this engine's own kernels; the CPU backend answers it bit-identically to its single-vector path.
        }

        public void GemmBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs) => _inner.GemmBatch(inputs, count, w, outputs);

        public bool SupportsGemm(QuantFormat format) => _inner.SupportsGemm(format);

        public void DequantRow(WeightRef w, int row, Span<float> output) => _inner.DequantRow(w, row, output);

        public float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b) => _inner.Dot(a, b);

        public void Axpy(float alpha, ReadOnlySpan<float> x, Span<float> y) => _inner.Axpy(alpha, x, y);

        public void Fill(Span<float> y, float v) => _inner.Fill(y, v);

        public void Scale(Span<float> y, float s) => _inner.Scale(y, s);

        public void Dispose() => _inner.Dispose();
    }
}
