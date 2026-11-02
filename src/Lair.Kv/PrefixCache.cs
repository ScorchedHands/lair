namespace Lair.Kv;

/// <summary>
/// Reference-counted prefix block cache over a paged KV pool. Completed full prompt blocks are published under
/// rolling token-prefix keys and shared into later sequences whose prompts extend them, so those prefixes skip
/// prefill entirely. Each entry holds one reference to its physical block on top of every sequence that adopted
/// or published it: an entry survives while any owner keeps the block alive, and releasing a cache entry returns
/// the block to the free list only when no other reference remains. Entries are evicted least-recently-published
/// first whenever the pool needs room for new allocations; blocks still shared by active sequences are never
/// evicted out from under them. The cache is keyed at the pool's block granularity and lives exactly as long as
/// the pool: a new pool starts with an empty cache, so any model, quantization, or block-size change invalidates
/// every cached prefix automatically.
/// </summary>
public sealed class PrefixCache
{
    /// <summary>Seed of the key chain; a fixed odd constant so token streams combine into a non-trivial first key.</summary>
    private const ulong ChainSeed = 0x9E37_79B9_7F4A_7C15UL;

    private readonly BlockPool _pool;

    // Key → entry. One entry per distinct prefix chain; equal keys with different token content cannot coexist,
    // so a colliding publish replaces the stale entry and hands its block back to normal refcount bookkeeping.
    private readonly Dictionary<ulong, PrefixEntry> _entries = [];

    // Publication order: oldest at the head (eviction scans forward), most recent at the tail. Every dictionary
    // entry has exactly one node here; a touch moves its node to the tail.
    private readonly LinkedList<PrefixEntry> _lru = new();

    public PrefixCache(BlockPool pool)
    {
        ArgumentNullException.ThrowIfNull(pool);
        _pool = pool;
    }

    /// <summary>The prefix granularity: cached keys cover whole blocks of this many token ids.</summary>
    public int BlockTokens => _pool.BlockTokens;

    /// <summary>Number of physical blocks currently held by cache entries.</summary>
    public int CachedBlocks => _entries.Count;

    /// <summary>
    /// One hop of the rolling key chain over token ids: <c>key(block_i) = ChainKey(key(block_{i-1}), tokens)</c>,
    /// seeded with a fixed constant for block 0. The mix is non-cryptographic by design — two xorshift-multiply
    /// rounds around an FNV-style per-token avalanche — so equal token streams always produce equal keys and any
    /// differing prefix almost certainly produces a different one; the residual collision risk is covered because
    /// adoption verifies each entry's stored block tokens against the prompt before sharing. Callers seed with 0:
    /// <see cref="TryAdopt"/> and <see cref="Publish"/> fold in the constant themselves.
    /// </summary>
    public static ulong ChainKey(ulong previousKey, ReadOnlySpan<int> blockTokens)
    {
        // Fold a fixed odd constant once per hop so chains of zero token ids still avalanche through the mix.
        ulong key = previousKey ^ ChainSeed;

        foreach (int tokenId in blockTokens)
        {
            // FNV-style combine: multiply by an odd prime, then xor the id into both halves of the 64-bit state.
            key *= 0x0000_0100_0000_01F3UL;
            ulong wide = (uint)tokenId;
            key ^= wide;
            key ^= wide << 32;

            // xorshift-multiply round: spreads each token's bits across the whole chain state.
            key ^= key >> 33;
            key *= 0xFF51_AFD7_ED55_8CCDUL;
            key ^= key >> 29;
        }

        return key;
    }

    /// <summary>
    /// Adopts the leading cached prefix blocks into a fresh sequence. Walks full prompt blocks in order while
    /// their rolling keys hit cache entries and each entry's stored token verification passes, adds a reference
    /// to every matched physical block, marks it adopted (read-only) in the sequence's table, and sets the
    /// sequence length so prefill resumes at the first unmatched block. Returns the number of blocks adopted; 0
    /// leaves the sequence exactly as if no cache existed. A partial tail prompt token block is never adopted:
    /// its positions must be forwarded fresh to produce that request's own logits path.
    /// </summary>
    public int TryAdopt(ReadOnlySpan<int> promptTokens, SequenceKv kv)
    {
        ArgumentNullException.ThrowIfNull(kv);

        int blockTokens = _pool.BlockTokens;
        ulong key = 0UL;
        int adopted = 0;

        while ((adopted + 1) * blockTokens <= promptTokens.Length)
        {
            ReadOnlySpan<int> tokens = promptTokens.Slice(adopted * blockTokens, blockTokens);
            key = ChainKey(key, tokens);

            if (!_entries.TryGetValue(key, out PrefixEntry? entry))
            {
                break; // chain ends: everything from this block onward prefills fresh
            }

            // Collision guard: an equal 64-bit key with different stored token ids is not a prefix match.
            if (!entry.LastTokens.SequenceEqual(tokens) || _pool.RefCount(entry.PhysicalBlockId) == 0)
            {
                break;
            }

            _pool.AddRef(entry.PhysicalBlockId); // the sequence's ownership of this block joins the entry's cache reference
            kv.MarkAdopted(entry.PhysicalBlockId, adopted * blockTokens);
            Touch(entry);
            adopted++;
        }

        if (adopted > 0)
        {
            kv.SetLength(adopted * blockTokens); // prefill continues from the first position after the shared prefix
        }

        return adopted;
    }

    /// <summary>
    /// Publishes every full prompt-prefix block of a prefilled sequence in hash-chain order. Each newly cached
    /// block takes one cache reference on top of the publishing sequence's own ownership; blocks already under
    /// their key (adopted earlier by this same sequence, or published by an identical prefix) are only touched to
    /// mark them most-recently-used — they keep the physical block already registered. Partial final blocks stay
    /// out of the cache because their KV is not fully written at publish time. Returns how many blocks this call
    /// newly registered (renewals excluded).
    /// </summary>
    public int Publish(ReadOnlySpan<int> promptTokens, SequenceKv kv)
    {
        ArgumentNullException.ThrowIfNull(kv);

        int blockTokens = _pool.BlockTokens;
        ulong key = 0UL;
        int depth = 0;
        int published = 0;

        while ((depth + 1) * blockTokens <= promptTokens.Length)
        {
            ReadOnlySpan<int> tokens = promptTokens.Slice(depth * blockTokens, blockTokens);
            key = ChainKey(key, tokens);

            if (_entries.TryGetValue(key, out PrefixEntry? existing))
            {
                Touch(existing); // already cached — this publish just renews its recency
            }
            else
            {
                int physicalBlockId = kv.BlockFor(depth * blockTokens);
                _pool.AddRef(physicalBlockId);
                var entry = new PrefixEntry(key, physicalBlockId, tokens.ToArray());
                _entries[key] = entry;
                _lru.AddLast(entry.Node = entry.CreateNode());
                published++;
            }

            depth++;
        }

        return published;
    }

    /// <summary>
    /// Frees up to <paramref name="blocksNeeded"/> blocks back to the pool's free list by evicting cache entries,
    /// least-recently-published first. An entry whose physical block still carries references beyond its own
    /// cache reference is shared by an active sequence and is skipped — eviction only ever hands over blocks no
    /// live sequence depends on. Returns the number of blocks actually released; a shortfall means every
    /// remaining cached block is pinned in use.
    /// </summary>
    public int EvictBlocks(int blocksNeeded)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(blocksNeeded);

        int freed = 0;
        while (freed < blocksNeeded && _entries.Count > 0)
        {
            // Scan from the oldest entry forward, skipping pinned ones. A pass that frees nothing means every
            // remaining entry is shared by an active sequence: no further eviction can help this request.
            int scanned = 0;
            bool freedAnyThisPass = false;

            while (scanned < _entries.Count && freed < blocksNeeded)
            {
                PrefixEntry? oldest = _lru.First?.Value;
                if (oldest is null)
                {
                    break; // dictionary and list are kept in lockstep: an empty list means no entries left
                }

                scanned++;

                if (_pool.RefCount(oldest.PhysicalBlockId) > 1)
                {
                    Touch(oldest); // pinned by a live sequence — it is the least-recent entry, but not evictable now
                    continue;
                }

                RemoveEntry(oldest);
                _pool.Release(oldest.PhysicalBlockId); // cache reference was the last one: the block joins the free list
                freed++;
                freedAnyThisPass = true;
            }

            if (!freedAnyThisPass)
            {
                break;
            }
        }

        return freed;
    }

    /// <summary>
    /// Drains cache entries down to a block budget: while the pool's in-use count exceeds <paramref name="budgetBlocks"/>,
    /// evicts least-recently-published entries (blocks pinned by active sequences are skipped, exactly as under capacity
    /// pressure) until enough blocks have left the used side of the pool. This is the shrink path of a hot budget edit:
    /// it frees cached copies down to the new cap and never touches memory an active sequence depends on. Returns the
    /// number of cache entries evicted; a shortfall means every remaining cached block is pinned in use.
    /// </summary>
    public int TrimToBudget(int budgetBlocks)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(budgetBlocks);

        // Cached blocks sit on the used side of the pool (their cache reference keeps them allocated), so exceeding
        // the cap can only be relieved by releasing entries — from the LRU head forward, pinned ones skipped. A block
        // shared with a live sequence leaves refcount &gt; 1 after its entry is evicted and stays in use: that alone
        // does not end the drain, because later unpinned entries still sit behind it. Each step removes one more
        // entry (or reports every remaining entry pinned), so the loop always terminates.
        int evicted = 0;
        while (_pool.UsedBlocks > budgetBlocks && _entries.Count > 0)
        {
            if (EvictBlocks(1) == 0)
            {
                break; // every remaining entry is pinned by a live sequence: nothing left to drain
            }

            evicted++;
        }

        return evicted;
    }

    private void Touch(PrefixEntry entry)
    {
        LinkedListNode<PrefixEntry> node = entry.Node ?? entry.CreateNode();
        _lru.Remove(node);
        _lru.AddLast(node);
    }

    private void RemoveEntry(PrefixEntry entry)
    {
        _entries.Remove(entry.Key);
        if (entry.Node is not null)
        {
            _lru.Remove(entry.Node);
        }
    }

    /// <summary>Test-only hook: registers an entry under a caller-chosen key and physical block so forged 64-bit collisions can be exercised. Production paths never call this.</summary>
    internal void PublishEntry(ulong key, int physicalBlockId, ReadOnlySpan<int> lastBlockTokens)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(physicalBlockId);

        if (_entries.TryGetValue(key, out PrefixEntry? stale))
        {
            RemoveEntry(stale); // equal keys cannot coexist: the displaced block falls back to its remaining owners' refs
            _pool.Release(stale.PhysicalBlockId);
        }

        _pool.AddRef(physicalBlockId);
        var entry = new PrefixEntry(key, physicalBlockId, lastBlockTokens.ToArray());
        _entries[key] = entry;
        _lru.AddLast(entry.Node = entry.CreateNode());
    }

    /// <summary>One cached prefix block: its chain key, the physical block holding its KV, and the token ids that must match before adoption.</summary>
    internal sealed class PrefixEntry(ulong key, int physicalBlockId, int[] lastTokens)
    {
        public ulong Key { get; } = key;

        public int PhysicalBlockId { get; } = physicalBlockId;

        /// <summary>The block's own token ids: the collision guard compares these against the prompt before adoption.</summary>
        public int[] LastTokens { get; } = lastTokens;

        // The entry is its own list node holder: one node per dictionary entry, moved to the tail on every touch.
        public LinkedListNode<PrefixEntry>? Node { get; set; }

        public LinkedListNode<PrefixEntry> CreateNode() => new(this);
    }
}
