using Lair.Http;
using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Prefix-cache tests over a synthetic small pool plus one end-to-end run on the real SmolLM F16 fixture: shared
/// prompt prefixes are adopted instead of prefilled and stay bit-identical to a cold copy, a write into a shared
/// block copies it private first without disturbing the original, eviction is least-recently-used and never disturbs
/// an active sequence, refcounts free a shared block only after the last holder releases it, a differing token cuts
/// adoption at its block, and a forged 64-bit key collision is refused by the stored-token guard.
/// </summary>
[Trait("Category", "Cache")]
public sealed class PrefixCacheTests
{
    // ---------------------------------------------------------------- integration

    [Fact]
    public async Task SharedPromptPrefix_SecondRequestAdoptsBlocks_AndOutputMatchesColdRun()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        string prefix = BuildSharedPrefix(modelPath); // ~120 tokens of identical leading text
        string promptA = prefix + " Question one: what happened next?";
        string promptB = prefix + " Question two: who was watching?";

        using GenerationService service = ConcurrencyTests.OpenService(kvBlocks: 64, maxContextTokens: 512);

        GenerationResult first = await service.GenerateAsync(
            ConcurrencyTests.CompletionRequest(promptA, maxTokens: 24),
            CancellationToken.None);
        GenerationResult second = await service.GenerateAsync(
            ConcurrencyTests.CompletionRequest(promptB, maxTokens: 24),
            CancellationToken.None);

        Assert.Equal(0, first.CachedPromptTokens); // cold: nothing was cached before it
        Assert.True(second.CachedPromptTokens >= 96, $"second request adopted only {second.CachedPromptTokens} cached prompt tokens");
        Assert.True(second.CachedPromptTokens < second.PromptTokens, "the whole prompt should not be cached — the differing tail must prefill fresh");

        // Determinism invariant: the cache-hit run must produce exactly what a cold run on a fresh service produces.
        using GenerationService baseline = ConcurrencyTests.OpenService(kvBlocks: 64, maxContextTokens: 512);
        GenerationResult cold = await baseline.GenerateAsync(
            ConcurrencyTests.CompletionRequest(promptB, maxTokens: 24),
            CancellationToken.None);

        Assert.False(string.IsNullOrEmpty(cold.Text), "the cold baseline produced no text");
        Assert.False(string.IsNullOrEmpty(second.Text), "the cache-hit request produced no text");
        Assert.Equal(cold.Text, second.Text);
        Assert.Equal(cold.FinishReason, second.FinishReason);
        Assert.Equal(cold.CompletionTokens, second.CompletionTokens);
        Assert.Equal(cold.PromptTokens, second.PromptTokens);
        Assert.Equal(0, cold.CachedPromptTokens);
    }

    // ---------------------------------------------------------------- storage layer

    [Fact]
    public void Adopt_SharedPrefixBlocks_AreReadonlyShared_AndBitIdenticalToPublishersCopy()
    {
        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 8);
        var cache = new PrefixCache(pool);

        int[] prompt = MakeTokens(0, 48); // three full blocks
        SequenceKv publisher = new(pool);
        FillRange(publisher, pool, 0, 48);
        cache.Publish(prompt, publisher);
        Assert.Equal(3, cache.CachedBlocks);

        int[] longer = [.. prompt, .. MakeTokens(1000, 1016)]; // same 48-token prefix, fresh tail block
        SequenceKv adopter = new(pool);
        int adopted = cache.TryAdopt(longer, adopter);
        Assert.Equal(3, adopted);
        Assert.Equal(48, adopter.Length); // prefill resumes at position 48

        int[] publisherBlocks = [.. publisher.Blocks];
        int[] adopterHead = [.. adopter.Blocks.Take(3)];
        Assert.Equal(publisherBlocks, adopterHead); // the same physical blocks back both tables
        Assert.True(adopter.IsAdopted(0) && adopter.IsAdopted(16) && adopter.IsAdopted(32), "adopted blocks must be marked shared");
        foreach (int block in publisherBlocks)
        {
            Assert.Equal(3, pool.RefCount(block)); // publisher + cache entry + adopter
        }

        float[] coldCopy = pool.BlockView(publisherBlocks[1]).ToArray(); // what a cold recomputation of block 1 must equal

        FillRange(adopter, pool, 48, 64); // the tail block prefills private
        Assert.False(adopter.IsAdopted(48));
        Assert.NotEqual(publisherBlocks[1], adopter.BlockFor(48));

        SpansEqual(coldCopy, pool.BlockView(adopter.BlockFor(16)), "shared bytes must equal the publisher's block bytes");

        // The shared prefix region is bit-identical across both sequences: same physical blocks, same reads.
        for (int pos = 0; pos < 48; pos++)
        {
            for (int layer = 0; layer < pool.LayerCount; layer++)
            {
                SpansEqual(publisher.ReadK(pos, layer), adopter.ReadK(pos, layer), $"shared K diverged at position {pos}, layer {layer}");
                SpansEqual(publisher.ReadV(pos, layer), adopter.ReadV(pos, layer), $"shared V diverged at position {pos}, layer {layer}");
            }
        }

        // The adopted tail block holds the fresh pattern in private storage.
        for (int pos = 48; pos < 64; pos++)
        {
            for (int layer = 0; layer < pool.LayerCount; layer++)
            {
                SpansEqual(MakeKPattern(pool, pos, layer), adopter.ReadK(pos, layer), $"tail K wrong at position {pos}, layer {layer}");
                SpansEqual(Pattern(-pos - 1, layer, pool.KvDim), adopter.ReadV(pos, layer), $"tail V wrong at position {pos}, layer {layer}");
            }
        }

        publisher.Dispose();
        adopter.Dispose();
        Assert.Equal(1, pool.RefCount(publisherBlocks[0])); // only the cache reference survives the sequences
        Assert.Equal(3, cache.EvictBlocks(3)); // and it releases every cached block when evicted
        Assert.Equal(pool.CapacityBlocks, pool.FreeBlocks);
    }

    [Fact]
    public void CopyOnWrite_MidBlockWriteIntoAdoptedBlock_CopiesPrivate_AndLeavesOriginalIntact()
    {
        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 6);
        var cache = new PrefixCache(pool);

        int[] prompt = MakeTokens(0, 16);
        SequenceKv original = new(pool);
        FillRange(original, pool, 0, 16);
        cache.Publish(prompt, original);

        SequenceKv sharer = new(pool);
        Assert.Equal(1, cache.TryAdopt(prompt, sharer));
        int sharedBlock = sharer.BlockFor(0);
        Assert.True(sharer.IsAdopted(5));
        Assert.Equal(3, pool.RefCount(sharedBlock)); // original + cache + sharer

        float[] before = pool.BlockView(sharedBlock).ToArray();

        float[] rewrite = Pattern(999, 0, pool.KvDim); // a deliberate rewrite of position 5, layer 0
        sharer.WriteK(5, 0, rewrite);

        Assert.False(sharer.IsAdopted(5)); // the block became private to the writer
        Assert.NotEqual(sharedBlock, sharer.BlockFor(0));
        Assert.Equal(2, pool.RefCount(sharedBlock)); // the writer released its reference on the shared block
        SpansEqual(before, pool.BlockView(sharedBlock), "the original holders' copy must be bit-identical to before");

        SpansEqual(SlotOf(before, pool, 5, 0, isV: false), original.ReadK(5, 0), "the original sequence's K slot was disturbed");
        SpansEqual(rewrite, sharer.ReadK(5, 0), "the writer must see its own value after copy-on-write");
        SpansEqual(SlotOf(before, pool, 4, 0, isV: false), sharer.ReadK(4, 0), "untouched slots must come along in the private copy");
        SpansEqual(SlotOf(before, pool, 6, 0, isV: false), sharer.ReadK(6, 0), "untouched slots must come along in the private copy");

        original.Dispose();
        sharer.Dispose();
        Assert.Equal(1, cache.EvictBlocks(1));
        Assert.Equal(pool.CapacityBlocks, pool.FreeBlocks); // every reference accounted for: nothing leaked
    }

    [Fact]
    public void Eviction_IsLeastRecentlyUsed_SkipsActiveSequences_AndKeepsTheirDataIntact()
    {
        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 4);
        var cache = new PrefixCache(pool);

        int[] promptP = MakeTokens(0, 16);
        int[] promptQ = MakeTokens(16, 32);
        int[] promptS = MakeTokens(32, 48);

        SequenceKv seqP = new(pool);
        FillRange(seqP, pool, 0, 16);
        cache.Publish(promptP, seqP); // published first: the oldest entry

        SequenceKv seqQ = new(pool);
        FillRange(seqQ, pool, 0, 16);
        cache.Publish(promptQ, seqQ);

        SequenceKv seqS = new(pool);
        FillRange(seqS, pool, 0, 16);
        cache.Publish(promptS, seqS);

        Assert.Equal(3, cache.CachedBlocks);
        Assert.Equal(1, pool.FreeBlocks); // three published blocks, one free

        // An active sequence adopts the oldest prefix, pinning its block and making it the most recently used.
        SequenceKv active = new(pool);
        Assert.Equal(1, cache.TryAdopt(promptP, active));
        int pinnedBlock = active.BlockFor(0);
        Assert.Equal(1, pool.FreeBlocks); // adoption shares blocks; it does not allocate

        seqP.Dispose();
        seqQ.Dispose();
        seqS.Dispose(); // their sequences let go; the cache entries keep the blocks alive

        Assert.Equal(2, pool.RefCount(pinnedBlock)); // active sequence + cache entry: eviction must skip this one

        // Pressure: the pool needs a block. The least-recently-used cache-only entry goes first.
        Assert.Equal(1, cache.EvictBlocks(1));
        Assert.Equal(2, cache.CachedBlocks);
        Assert.Equal(2, pool.FreeBlocks);

        SequenceKv goneProbe = new(pool); // the evicted prefix no longer adopts
        Assert.Equal(0, cache.TryAdopt(promptQ, goneProbe));
        goneProbe.Dispose();

        // Further pressure: S (next oldest) goes; P stays because an active sequence still references it.
        Assert.Equal(1, cache.EvictBlocks(2));
        Assert.Equal(1, cache.CachedBlocks);
        Assert.Equal(3, pool.FreeBlocks);
        Assert.Equal(2, pool.RefCount(pinnedBlock)); // active sequence + cache entry

        SequenceKv pinnedProbe = new(pool); // the pinned entry survives eviction attempts and still adopts
        Assert.Equal(1, cache.TryAdopt(promptP, pinnedProbe));
        pinnedProbe.Dispose();
        Assert.Equal(2, pool.RefCount(pinnedBlock)); // probe released its own reference again

        for (int pos = 0; pos < 16; pos++)
        {
            for (int layer = 0; layer < pool.LayerCount; layer++)
            {
                SpansEqual(MakeKPattern(pool, pos, layer), active.ReadK(pos, layer), $"active sequence's KV changed at position {pos}, layer {layer}");
            }
        }

        // The active sequence crosses into a fresh block using the reclaimed capacity — eviction made admission possible.
        active.Append();
        Assert.Equal(2, pool.FreeBlocks);

        active.Dispose();
        Assert.Equal(1, pool.RefCount(pinnedBlock)); // back to cache-only
        Assert.Equal(1, cache.EvictBlocks(1));
        Assert.Equal(0, cache.CachedBlocks);
        Assert.Equal(pool.CapacityBlocks, pool.FreeBlocks);
        Assert.Equal(0, pool.UsedBlocks);
    }

    [Fact]
    public void Refcount_SharedBlockFreesOnlyAfterEveryHolderReleases()
    {
        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 6);
        var cache = new PrefixCache(pool);

        int[] prompt = MakeTokens(0, 16);
        SequenceKv first = new(pool);
        FillRange(first, pool, 0, 16);
        cache.Publish(prompt, first);
        int block = first.BlockFor(0);
        Assert.Equal(2, pool.RefCount(block)); // sequence + cache entry

        SequenceKv second = new(pool);
        Assert.Equal(1, cache.TryAdopt(prompt, second));
        Assert.Equal(3, pool.RefCount(block)); // two sequences + cache entry

        int freeBefore = pool.FreeBlocks;
        first.Dispose();
        Assert.Equal(2, pool.RefCount(block)); // the finished sequence lets go; the block stays shared
        Assert.Equal(freeBefore, pool.FreeBlocks);
        SpansEqual(MakeKPattern(pool, 3, 0), second.ReadK(3, 0), "the surviving sequence's shared data must be intact");

        second.Dispose();
        Assert.Equal(1, pool.RefCount(block)); // only the cache reference remains — still not free
        Assert.Equal(freeBefore, pool.FreeBlocks);

        Assert.Equal(1, cache.EvictBlocks(1));
        Assert.Equal(0, pool.RefCount(block)); // freed exactly when the last reference went
        Assert.Equal(freeBefore + 1, pool.FreeBlocks);
    }

    [Fact]
    public void Adoption_StopsAtFirstDifferingBlock_AndForgedKeyCollisionIsRefused()
    {
        using CpuBackend backend = new();
        using BlockPool pool = NewPool(backend, capacityBlocks: 8);
        var cache = new PrefixCache(pool);

        int[] published = MakeTokens(0, 48); // three full blocks cached
        SequenceKv publisher = new(pool);
        FillRange(publisher, pool, 0, 48);
        cache.Publish(published, publisher);

        // A prompt identical for blocks 0 and 1 but with a differing token inside block 2 adopts exactly two blocks.
        int[] tailDiffers = [.. MakeTokens(0, 32), 777, .. MakeTokens(33, 48)];
        SequenceKv a = new(pool);
        Assert.Equal(2, cache.TryAdopt(tailDiffers, a)); // blocks 0 and 1 verify against the real entries; block 2's chain diverges
        a.Dispose();

        // A prompt differing inside block 0 adopts nothing.
        int[] headDiffers = [0, 1, 2, 999, .. MakeTokens(4, 48)];
        SequenceKv b = new(pool);
        Assert.Equal(0, cache.TryAdopt(headDiffers, b));
        Assert.True(b.Length == 0, "a rejected adoption must leave the sequence empty");
        Assert.Empty(b.Blocks);
        b.Dispose();

        // Collision guard: register an entry under the exact key the real chain would produce for blocks 0+1, but
        // storing the wrong block-1 tokens. Adoption of the real prompt must stop after block 0 despite the key hit.
        int scratch = pool.Allocate();
        ulong k0 = PrefixCache.ChainKey(0UL, published[..16]);
        ulong k1 = PrefixCache.ChainKey(k0, published[16..32]);
        cache.PublishEntry(k1, scratch, MakeTokens(500, 516)); // forged: right key, wrong stored tokens

        SequenceKv c = new(pool);
        Assert.Equal(1, cache.TryAdopt(published, c)); // block 0 verifies and adopts; block 1 hits the forged key but fails token verification
        Assert.Equal(16, c.Length);
        c.Dispose();

        // A forged entry with the right key and right tokens is adopted — the guard checks content, not provenance.
        cache.PublishEntry(k1, scratch, published[16..32]); // re-forged under the same block: the displaced entry's reference returns to normal bookkeeping
        SequenceKv d = new(pool);
        Assert.Equal(3, cache.TryAdopt(published, d)); // with corrected tokens the whole verified chain adopts again: blocks 0, 1 (the scratch block), and 2
        SpansEqual(MakeKPattern(pool, 7, 1), d.ReadK(7, 1), "adoption through a verified forged entry must read the block's real KV");
        d.Dispose();

        publisher.Dispose();
        Assert.Equal(2, cache.EvictBlocks(4)); // k0's and k2's blocks release; the forged entry stays pinned by the test's own scratch reference
        Assert.Equal(1, cache.CachedBlocks);
        Assert.Equal(pool.CapacityBlocks - 1, pool.FreeBlocks);

        pool.Release(scratch); // the test lets go of its own reference...
        Assert.Equal(1, pool.RefCount(scratch)); // ...cache reference alone: still not free
        Assert.Equal(1, cache.EvictBlocks(1));
        Assert.Equal(0, pool.RefCount(scratch));
        Assert.Equal(0, cache.CachedBlocks);
        Assert.Equal(pool.CapacityBlocks, pool.FreeBlocks);
    }

    // ---------------------------------------------------------------- helpers

    private static BlockPool NewPool(CpuBackend backend, int capacityBlocks) =>
        new(blockTokens: 16, layerCount: 2, kvHeads: 2, headDim: 8, capacityBlocks: capacityBlocks, backend);

    private static int[] MakeTokens(int fromId, int toId) => [.. Enumerable.Range(fromId, toId - fromId)];

    private static float[] Pattern(int seedValue, int layer, int length)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = seedValue * 100 + layer * 10 + i + 1;
        }

        return values;
    }

    private static float[] MakeKPattern(BlockPool pool, int pos, int layer) => Pattern(pos, layer, pool.KvDim);

    private static void FillRange(SequenceKv kv, BlockPool pool, int fromPos, int toPos)
    {
        for (int pos = fromPos; pos < toPos; pos++)
        {
            kv.Append();
            for (int layer = 0; layer < pool.LayerCount; layer++)
            {
                kv.WriteK(pos, layer, MakeKPattern(pool, pos, layer));
                kv.WriteV(pos, layer, Pattern(-pos - 1, layer, pool.KvDim));
            }
        }
    }

    // Extracts one position's K or V slot out of a captured whole-block snapshot, using the pool's documented layout.
    private static float[] SlotOf(float[] blockBytes, BlockPool pool, int pos, int layer, bool isV)
    {
        int floatsPerLayer = 2 * pool.BlockTokens * pool.KvDim;
        int start = layer * floatsPerLayer + (isV ? pool.BlockTokens * pool.KvDim : 0) + pos * pool.KvDim;
        return blockBytes[start..(start + pool.KvDim)];
    }

    private static void SpansEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string message)
    {
        Assert.True(expected.SequenceEqual(actual), $"{message} (expected {Fmt(expected)}, got {Fmt(actual)})");
    }

    private static string Fmt(ReadOnlySpan<float> values)
    {
        int shown = Math.Min(values.Length, 4);
        List<string> parts = [];
        for (int i = 0; i < shown; i++)
        {
            parts.Add(values[i].ToString("R"));
        }

        return string.Join(", ", parts) + (values.Length > shown ? ", ..." : string.Empty);
    }

    // Sizes the shared-prefix fixture against the real tokenizer so the integration prompt really crosses six blocks.
    private static string BuildSharedPrefix(string modelPath)
    {
        const string line = "The quick brown fox jumps over the lazy dog while the patient cat watches the rain. ";
        string prefix = string.Concat(Enumerable.Repeat(line, 7));

        using GgufFile gguf = GgufFile.Open(modelPath);
        int tokens = GgufTokenizer.FromGguf(gguf).Encode(prefix).Count;
        Assert.True(tokens >= 100 && tokens <= 140, $"shared prefix built {tokens} tokens; expected around 120");

        return prefix;
    }
}
