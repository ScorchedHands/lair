using System.Buffers;
using System.Runtime.InteropServices;
using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// End-to-end proof that the engine runs over an alternate IWeightSource: every tensor is copied out of the mmap
/// view into backend-allocated aligned blocks before the model opens, and the forward then reads only from those
/// blocks. Greedy decode through the arena-backed model must produce exactly the token sequence and bit-equal
/// logits of the same model opened over its default mmap source — weight provenance is a swappable seam, not an
/// engine assumption.
/// </summary>
[Trait("Category", "Parity")]
public sealed class ArenaWeightSourceTests
{
    private const int DecodeSteps = 8;

    [Fact]
    public void AlternateIWeightSource_DecodesIdenticallyToMmap()
    {
        string modelPath = ModelFixture.Resolve()!;
        using GgufFile file = GgufFile.Open(modelPath);
        using CpuBackend backend = new(4);

        using TransformerModel mmapModel = TransformerModel.Open(file, backend, 512);
        using ArenaWeightSource arena = new(file, backend);
        using TransformerModel arenaModel = TransformerModel.Open(file, arena, backend, 512);

        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(file);
        IReadOnlyList<int> prompt = tokenizer.AddBosToken
            ? tokenizer.EncodeWithBos("The capital of France is")
            : tokenizer.Encode("The capital of France is");

        int[] mmapTokens = GreedySequence(mmapModel, backend, prompt);
        int[] arenaTokens = GreedySequence(arenaModel, backend, prompt);

        Assert.Equal(mmapTokens, arenaTokens);
    }

    private static int[] GreedySequence(TransformerModel model, CpuBackend backend, IReadOnlyList<int> prompt)
    {
        using BlockPool pool = new(16, model.Desc.LayerCount, model.Desc.KvHeadCount, model.Desc.HeadDim, 32, backend);
        using SequenceKv kv = new(pool);
        model.Prefill(prompt, kv);

        int[] tokens = new int[DecodeSteps];
        ReadOnlySpan<float> logits = model.LastLogits();
        for (int step = 0; step < DecodeSteps; step++)
        {
            int best = 0;
            for (int i = 1; i < logits.Length; i++)
            {
                if (logits[i] > logits[best])
                {
                    best = i;
                }
            }

            tokens[step] = best;
            logits = model.DecodeNext(best, kv);
        }

        return tokens;
    }

    private sealed class ArenaWeightSource : IWeightSource
    {
        private readonly GgufMmapWeightSource _mmap;
        private readonly IBackend _backend;
        private readonly Dictionary<string, (nint Ptr, long Size)> _byName = new(StringComparer.Ordinal);
        private bool _disposed;

        public unsafe ArenaWeightSource(GgufFile file, IBackend backend)
        {
            ArgumentNullException.ThrowIfNull(file);
            _backend = backend;
            _mmap = new GgufMmapWeightSource(file);

            foreach (LogicalTensor tensor in file.Tensors.Tensors)
            {
                ReadOnlySpan<byte> source = _mmap.Resolve(tensor);
                MemoryHandle handle = backend.AllocateAligned(tensor.SizeBytes, 64);
                source.CopyTo(new Span<byte>(handle.Pointer, checked((int)tensor.SizeBytes)));

                // The model's Resolve hands out spans for its whole lifetime: keep every block until Dispose.
                _handles.Add(handle);
                _byName[tensor.Name] = ((nint)handle.Pointer, tensor.SizeBytes);
            }
        }

        private readonly List<MemoryHandle> _handles = [];

        public unsafe ReadOnlySpan<byte> Resolve(LogicalTensor tensor)
        {
            ObjectDisposedException.ThrowIf(_disposed, typeof(ArenaWeightSource));

            if (!_byName.TryGetValue(tensor.Name, out (nint Ptr, long Size) block))
            {
                throw new InvalidOperationException($"arena weight source has no copy of tensor '{tensor.Name}'.");
            }

            return new ReadOnlySpan<byte>((void*)block.Ptr, checked((int)block.Size));
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            foreach (MemoryHandle handle in _handles)
            {
                _backend.FreeAligned(handle);
            }

            _handles.Clear();
            _mmap.Dispose();
        }
    }
}
