namespace Lair.Models;

/// <summary>
/// One model-side resource whose working copies carry their own ceilings: the weight bytes a forward reads, the KV blocks
/// sequences write, and the activation buffers a prompt window works in. Each resource answers for itself — an exceeded
/// ceiling moves that one resource down its own chain rather than relocating the whole model — because movement costs are
/// not comparable across them: KV blocks move cheaply, weight slabs are re-read by every token, and prefill working sets
/// are transient work no tier wants on disk.
/// </summary>
public enum BudgetResource
{
    /// <summary>The weights a forward reads: read-side material whose complete set already exists in the model file.</summary>
    Weights,

    /// <summary>Paged KV blocks sequences append to: write-side state, resident where its pool was opened.</summary>
    Kv,

    /// <summary>The activation buffers a batched prompt window works inside: transient working-set bytes, never disk material today.</summary>
    PrefillBuffers,
}

/// <summary>The config key name of each budget resource, and the reverse read that decides which resources are public vocabulary.</summary>
public static class BudgetResourceNames
{
    public const string Weights = "weights";
    public const string Kv = "kv";
    public const string PrefillBuffers = "prefill_buffers";

    /// <summary>The resource keys a configuration may declare, in the order a refusal reports them.</summary>
    public static IReadOnlyList<string> DeclaredKeys { get; } = [Weights, Kv, PrefillBuffers];

    public static string Of(BudgetResource resource) => resource switch
    {
        BudgetResource.Kv => Kv,
        BudgetResource.PrefillBuffers => PrefillBuffers,
        _ => Weights,
    };

    /// <summary>Reads one config key as a resource. An unrecognized name is not a resource: callers report it naming the value.</summary>
    public static bool TryParse(string key, out BudgetResource resource)
    {
        switch (key)
        {
            case Weights:
                resource = BudgetResource.Weights;
                return true;

            case Kv:
                resource = BudgetResource.Kv;
                return true;

            case PrefillBuffers:
                resource = BudgetResource.PrefillBuffers;
                return true;

            default:
                resource = default;
                return false;
        }
    }
}

/// <summary>
/// Where a resource's working copies can sit. DRAM is what every pool this build opens holds; an accelerator tier and the
/// disk tier are vocabulary a chain may name before anything resident sits there, exactly as a route segment naming an
/// accelerator device names a span no backend runs yet.
/// </summary>
public enum BudgetTier
{
    /// <summary>System memory: where weights, KV blocks, and activation buffers live for every model served today.</summary>
    Dram,

    /// <summary>An accelerator's own memory. Named by the vocabulary only until a backend reports such a device.</summary>
    Vram,

    /// <summary>Disk storage. A read-side resource's origin store; write-side bytes land there only where something writes them.</summary>
    Nvme,
}

/// <summary>The config key name of each budget tier, and the reverse read that decides which tiers are public vocabulary.</summary>
public static class BudgetTierNames
{
    public const string Dram = "dram";
    public const string Vram = "vram";
    public const string Nvme = "nvme";

    /// <summary>The tier names a chain may declare, in the priority order every legal chain follows.</summary>
    public static IReadOnlyList<string> DeclaredNames { get; } = [Dram, Vram, Nvme];

    public static string Of(BudgetTier tier) => tier switch
    {
        BudgetTier.Vram => Vram,
        BudgetTier.Nvme => Nvme,
        _ => Dram,
    };

    /// <summary>Reads one config name as a tier. An unrecognized name is not a tier: callers report it naming the value.</summary>
    public static bool TryParse(string name, out BudgetTier tier)
    {
        switch (name)
        {
            case Dram:
                tier = BudgetTier.Dram;
                return true;

            case Vram:
                tier = BudgetTier.Vram;
                return true;

            case Nvme:
                tier = BudgetTier.Nvme;
                return true;

            default:
                tier = default;
                return false;
        }
    }
}

/// <summary>
/// The ceiling-key vocabulary of one resource: every cap key is a tier name joined to the unit that resource counts its
/// ceilings in — <c>dram_mb</c>, <c>vram_blocks</c>, and so on. Callers share this mapping instead of each spelling the
/// suffixes out, so a config layer and a reload table agree on which keys exist for which resource. A key built from one
/// resource's unit never parses as another's: a megabyte ceiling is not how KV counts anything, and blocks are not how
/// weights or buffers count theirs.
/// </summary>
public static class BudgetCapKeys
{
    private const string MegabyteSuffix = "_mb";
    private const string BlocksSuffix = "_blocks";

    /// <summary>The qualified name of one tier's ceiling under one resource, in that resource's unit.</summary>
    public static string Of(BudgetTier tier, BudgetResource resource) => $"{BudgetTierNames.Of(tier)}{UnitSuffix(resource)}";

    /// <summary>All cap keys one resource accepts, highest-priority residency first — the vocabulary a refusal names when a key does not belong to this resource.</summary>
    public static IReadOnlyList<string> For(BudgetResource resource) => [.. BudgetTierNames.DeclaredNames.Select(name => $"{name}{UnitSuffix(resource)}")];

    /// <summary>Reads one cap key as a tier of one resource. A wrong unit, an unknown tier, or anything malformed is not a ceiling this resource declares: callers report it naming the accepted vocabulary.</summary>
    public static bool TryParse(string key, BudgetResource resource, out BudgetTier tier)
    {
        tier = default;

        string suffix = UnitSuffix(resource);
        if (!key.EndsWith(suffix, StringComparison.Ordinal))
        {
            return false;
        }

        return BudgetTierNames.TryParse(key[..(key.Length - suffix.Length)], out tier);
    }

    private static string UnitSuffix(BudgetResource resource) => resource == BudgetResource.Kv ? BlocksSuffix : MegabyteSuffix;
}

/// <summary>
/// What one resource's configuration declares: its own ordered chain of tiers and the ceiling declared at each tier that
/// appears in it. Caps are expressed in the unit that resource counts in — megabytes for weights and prefill buffers, KV
/// blocks for KV — so a cap is always a number against bytes or blocks LAIR holds there rather than a share of some pool.
/// A tier named in <see cref="Order"/> with no entry here states "no model-level ceiling at that position", which is a
/// different answer from any declared value: the resource may still be bounded by what the machine has, just not by this
/// model's declaration. Non-positive ceilings never reach this shape — callers reject them before building a chain.
/// </summary>
public sealed record BudgetChain
{
    public required BudgetResource Resource { get; init; }

    /// <summary>The tiers this resource demotes through, highest-priority residency first. Never empty, and no tier named twice.</summary>
    public required IReadOnlyList<BudgetTier> Order { get; init; }

    /// <summary>Declared ceilings keyed by the tier they bind, each in this resource's unit. Every key must name a tier of <see cref="Order"/>.</summary>
    public IReadOnlyDictionary<BudgetTier, long> Caps { get; init; } = new Dictionary<BudgetTier, long>();
}

/// <summary>
/// What resolution decided about one position of a resource chain: the tier it landed on, whether that position is the
/// read-side origin store rather than a working-copy claim, and the ceiling this model declares there — null when the
/// declaration states no ceiling at all. A resolved position names what load decided; claiming bytes against capacity or
/// moving copies between positions is later work that consumes these values unchanged.
/// </summary>
public sealed record ResolvedBudgetTier
{
    public required BudgetTier Tier { get; init; }

    /// <summary>True when this position holds the resource's complete set outside LAIR's working copies — a read-side origin store, which no ceiling of ours describes and no pool claim needs.</summary>
    public bool IsOriginStore { get; init; }

    /// <summary>The declared ceiling at this position in the resource's own unit (megabytes for weights/prefill buffers, KV blocks for KV); null means the declaration states none there.</summary>
    public long? Cap { get; init; }

    /// <summary>The identity an error or diagnostic names this position by: its tier plus whether it holds copies at all.</summary>
    public string Describe() => IsOriginStore ? $"'{BudgetTierNames.Of(Tier)}' (origin store)" : $"'{BudgetTierNames.Of(Tier)}'";

    /// <summary>The ceiling as the text a message reports, in the unit that resource counts ceilings in, with absent stated as absent rather than guessed at.</summary>
    public string DescribeCap(BudgetResource resource) => Cap is null ? "no ceiling" : $"{Cap} {ModelBudgets.CapUnitName(resource)}s";
}

/// <summary>One resource's fully resolved chain: its identity and every position resolution landed, in declaration order.</summary>
public sealed record ResolvedBudgetChain
{
    public required BudgetResource Resource { get; init; }

    /// <summary>The chain positions as declared, each with what it holds there. Never empty — a resource that declares nothing resolves to no chain at all rather than an empty one.</summary>
    public required IReadOnlyList<ResolvedBudgetTier> Tiers { get; init; }

    /// <summary>Looks up the resolved position of one tier of this chain; null when the chain does not name it.</summary>
    public ResolvedBudgetTier? At(BudgetTier tier)
    {
        foreach (ResolvedBudgetTier position in Tiers)
        {
            if (position.Tier == tier)
            {
                return position;
            }
        }

        return null;
    }

    /// <summary>The first position that holds this resource's working copies — where a claim starts before any demotion down the chain.</summary>
    public ResolvedBudgetTier? FirstCopyHoldingTier()
    {
        foreach (ResolvedBudgetTier position in Tiers)
        {
            if (!position.IsOriginStore)
            {
                return position;
            }
        }

        return null;
    }
}

/// <summary>
/// The budget value set one model resolves to: every resource its configuration declared, each with its own chain landed
/// against the pools actually opened. A model that declares nothing carries an empty set — today's behavior exactly, where
/// only KV block counts bound anything. Nothing here claims bytes or moves copies; this is the immutable shape later
/// capacity work accounts against.
/// </summary>
public sealed record ResolvedBudgetSet
{
    /// <summary>The set a model carries when its configuration declares no budgets at all.</summary>
    public static readonly ResolvedBudgetSet Empty = new() { Resources = [] };

    public required IReadOnlyList<ResolvedBudgetChain> Resources { get; init; }

    /// <summary>Looks up one resolved resource chain; null when that resource declares nothing.</summary>
    public ResolvedBudgetChain? For(BudgetResource resource)
    {
        foreach (ResolvedBudgetChain chain in Resources)
        {
            if (chain.Resource == resource)
            {
                return chain;
            }
        }

        return null;
    }

    /// <summary>True when no resource declares anything.</summary>
    public bool IsEmpty => Resources.Count == 0;
}

/// <summary>
/// What budget resolution consumes: the model's name for every message this owns, the chains its configuration declared,
/// and what the host actually has open to hold working copies — the tiers its opened pools can back plus whether anything
/// writes that resource's bytes out rather than only reading them. Callers map their own state onto these values; nothing
/// here reads a config key or opens a pool.
/// </summary>
public sealed record BudgetRequest
{
    public required string ModelName { get; init; }

    /// <summary>The chains this model declares, one per resource it states anything about.</summary>
    public required IReadOnlyList<BudgetChain> Declared { get; init; }

    /// <summary>The tiers opened pools can hold working copies at. A chain naming a tier outside this list resolves structurally and refuses here rather than dropping that position silently — the posture every unimplemented capability takes.</summary>
    public required IReadOnlyList<BudgetTier> OpenedPoolTiers { get; init; }

    /// <summary>Whether opened state can write bytes out to disk storage for a write-side resource. False everywhere this build runs: nothing writes KV or activation material anywhere but system memory yet, so a chain asking for it reports what is missing.</summary>
    public bool DiskWriteSupport { get; init; }
}

/// <summary>
/// Resolves declared budget chains into the value set a model's plan carries, and owns every rule that resolution enforces.
/// A chain must name known tiers in priority order (<c>dram</c> ahead of <c>vram</c> ahead of <c>nvme</c>, any tier skippable)
/// with no tier named twice; a ceiling binds only a tier its own chain names; and the read/write split decides which
/// positions hold copies at all — a read-side resource's disk position is its origin store, legal only as the final position
/// of its chain and never under a ceiling of ours because that storage already holds the complete set. Every other position
/// needs an opened pool to hold working copies there: a tier no pool backs refuses naming the model, the resource, the tier,
/// and what is missing rather than falling back to residency that happens to work, exactly as a route segment on a device no
/// backend reports does. Nothing in this resolution claims capacity or relocates a copy — it lands the shape later capacity
/// work consumes unchanged.
/// </summary>
public static class ModelBudgets
{
    /// <summary>The priority order every chain must follow, highest-priority residency first.</summary>
    private static readonly BudgetTier[] PriorityOrder = [BudgetTier.Dram, BudgetTier.Vram, BudgetTier.Nvme];

    /// <summary>
    /// Resolves one model's declared chains against the pools its host opened. Each resource resolves independently: mixed
    /// shapes are legal material (weights demoting to their origin store beside a KV chain that names only system memory), and
    /// every unresolvable declaration — an unknown tier, a repeated tier, a chain out of priority order, a ceiling at a tier its
    /// chain does not name, a ceiling on a read-side origin position, or a working-copy claim against a tier no opened pool backs —
    /// throws naming the model, the resource, and what this host lacks. A CPU-only posture resolves DRAM positions only: an
    /// accelerator-resident claim reports the missing compute backend by name instead of being dropped from the chain it declared.
    /// </summary>
    public static ResolvedBudgetSet Resolve(BudgetRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.Declared);
        ArgumentNullException.ThrowIfNull(request.OpenedPoolTiers);

        string model = RequireModelName(request.ModelName);

        // Every declaration passes the same structural rules a configuration layer applies before it accepts one — this is
        // what makes accepting a future-shaped chain at load safe: posture below only ever decides against declarations that already hold.
        ValidateDeclaredChains(model, request.Declared);

        List<ResolvedBudgetChain> chains = [];
        foreach (BudgetChain chain in request.Declared)
        {
            chains.Add(ResolveChain(model, chain, request.OpenedPoolTiers, request.DiskWriteSupport));
        }

        return new ResolvedBudgetSet { Resources = [.. chains] };
    }

    /// <summary>
    /// The structural rules one model's declared chains must hold before anything asks whether a host can carry them: known
    /// resources named once, each with its own non-empty order of known tiers in priority order and no tier repeated, ceilings
    /// bound only to tiers that order names and positive wherever they state one, an origin-store position last rather than
    /// mid-chain, and never under a ceiling. A configuration layer that knows nothing about opened pools runs this pass on what
    /// it parses; <see cref="Resolve"/> runs the same rules first, so every refusal naming these reasons reads identically at
    /// both layers. Nothing here consults capacity: which claims an opened host can carry is resolution's answer, not this one.
    /// </summary>
    public static void ValidateDeclaredChains(string model, IReadOnlyList<BudgetChain> chains)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        ArgumentNullException.ThrowIfNull(chains);

        HashSet<BudgetResource> seenResources = [];
        foreach (BudgetChain chain in chains)
        {
            ArgumentNullException.ThrowIfNull(chain);

            if (!Enum.IsDefined(chain.Resource))
            {
                throw new ModelException(
                    $"model '{model}' declares budgets for resource '{chain.Resource}', which is not a budgetable resource " +
                    $"({string.Join(", ", BudgetResourceNames.DeclaredKeys)})");
            }

            if (!seenResources.Add(chain.Resource))
            {
                throw new ModelException(
                    $"model '{model}' declares budgets for resource '{BudgetResourceNames.Of(chain.Resource)}' more than once: one resource names its own order and caps");
            }

            ValidateStructure(model, chain);
        }
    }

    private static void ValidateStructure(string model, BudgetChain chain)
    {
        string resource = DescribeResource(chain.Resource);

        if (chain.Order.Count == 0)
        {
            throw new ModelException(
                $"model '{model}', budgets for resource '{resource}': an order must name at least the tier its copies start on — omit this resource to declare nothing about it");
        }

        HashSet<BudgetTier> seenTiers = [];
        int previousPriority = -1;

        foreach (BudgetTier tier in chain.Order)
        {
            if (!IsKnownTier(tier))
            {
                throw new ModelException(
                    $"model '{model}', budgets for resource '{resource}': order names tier '{tier}', which is not a residency budget chains can declare " +
                    $"({string.Join(", ", BudgetTierNames.DeclaredNames)})");
            }

            if (!seenTiers.Add(tier))
            {
                throw new ModelException(
                    $"model '{model}', budgets for resource '{resource}': order names tier '{BudgetTierNames.Of(tier)}' more than once — one chain holds this resource at a tier exactly once");
            }

            int priority = PriorityOf(tier);
            if (priority <= previousPriority)
            {
                throw new ModelException(
                    $"model '{model}', budgets for resource '{resource}': order [{DescribeOrder(chain.Order)}] is out of order — tiers must run from the highest-priority residency to " +
                    $"the lowest ({string.Join(" then ", BudgetTierNames.DeclaredNames)}), skipping any tier this resource does not use");
            }

            previousPriority = priority;
        }

        foreach (KeyValuePair<BudgetTier, long> cap in chain.Caps)
        {
            if (!IsKnownTier(cap.Key))
            {
                throw new ModelException(
                    $"model '{model}', budgets for resource '{resource}': a ceiling names tier '{cap.Key}', which is not a residency budget chains can declare " +
                    $"({string.Join(", ", BudgetTierNames.DeclaredNames)})");
            }

            if (!seenTiers.Contains(cap.Key))
            {
                throw new ModelException(
                    $"model '{model}', budgets for resource '{resource}': a ceiling binds tier '{BudgetTierNames.Of(cap.Key)}', which this chain's order " +
                    $"[{DescribeOrder(chain.Order)}] does not name — declare the tier in 'order' to cap it");
            }

            if (cap.Value <= 0)
            {
                throw new ModelException(
                    $"model '{model}', budgets for resource '{resource}': a ceiling at tier '{BudgetTierNames.Of(cap.Key)}' is {cap.Value}; " +
                    $"{CapUnitName(chain.Resource)} ceilings must be 1 or greater — omit the cap to declare no ceiling there");
            }

            // A read-side origin store already holds this resource's complete set outside LAIR's working copies, so a ceiling
            // of ours there can only misrepresent it. Skipping that check here would let a config carry a number nothing could honor.
            if (chain.Resource == BudgetResource.Weights && cap.Key == BudgetTier.Nvme)
            {
                throw new ModelException(
                    $"model '{model}', budgets for resource '{resource}': a ceiling at tier '{BudgetTierNames.Of(BudgetTier.Nvme)}' is not meaningful — " +
                    $"'{resource}' reads its complete set from that storage already, so it caps only the tiers holding working copies ({CapUnitName(chain.Resource)})");
            }
        }

        // A read-side resource's disk position is where its complete set already lives: legal only as the last position of its own chain.
        for (int index = 0; index < chain.Order.Count - 1; index++)
        {
            if (chain.Order[index] == BudgetTier.Nvme && IsReadSide(chain.Resource))
            {
                throw new ModelException(
                    $"model '{model}', budgets for resource '{resource}': order [{DescribeOrder(chain.Order)}] names its origin store before the tier it demotes to — " +
                    $"'{BudgetTierNames.Of(BudgetTier.Nvme)}' must be this chain's last position");
            }
        }
    }

    // The posture pass: every structural rule already holds, so each position either lands as the read-side origin store or needs an opened pool.
    private static ResolvedBudgetChain ResolveChain(string model, BudgetChain chain, IReadOnlyList<BudgetTier> openedPools, bool diskWrites)
    {
        string resource = DescribeResource(chain.Resource);

        List<ResolvedBudgetTier> positions = [];

        for (int index = 0; index < chain.Order.Count; index++)
        {
            BudgetTier tier = chain.Order[index];

            // A read-side origin store holds the complete set outside LAIR's working copies: no ceiling of ours describes it and no claim needs it.
            if (tier == BudgetTier.Nvme && IsReadSide(chain.Resource))
            {
                positions.Add(new ResolvedBudgetTier { Tier = tier, IsOriginStore = true });
                continue;
            }

            // Every other position holds a working copy LAIR owns, so it needs an opened pool of that residency. A CPU-only host
            // has the system-memory tier open only: an accelerator-resident claim names what is missing rather than dropping the
            // position its chain declared, and disk-backed write-side bytes name the writing machinery this build does not carry.
            if (!ContainsTier(openedPools, tier))
            {
                throw new ModelException(
                    $"model '{model}', budgets for resource '{resource}': no opened pool backs working copies at tier '{BudgetTierNames.Of(tier)}' — " +
                    DescribeMissingCapability(resource, tier, diskWrites));
            }

            positions.Add(new ResolvedBudgetTier
            {
                Tier = tier,
                IsOriginStore = false,
                Cap = chain.Caps.TryGetValue(tier, out long declaredCap) ? declaredCap : null,
            });
        }

        return new ResolvedBudgetChain { Resource = chain.Resource, Tiers = [.. positions] };
    }

    // What a claim against an unopened residency is missing: for the accelerator tier, whether the resource needs a compute backend opened
    // with device-resident weight copies or a device-backed pool that residency does not hold yet (the same seam route resolution names), and
    // for write-side bytes at the disk tier, whether this host writes them out at all — declared vocabulary in every case, never a silent drop.
    private static string DescribeMissingCapability(string resource, BudgetTier tier, bool diskWrites) => tier switch
    {
        BudgetTier.Vram when resource == BudgetResourceNames.Weights =>
            $"'{resource}' resident in accelerator memory needs a compute backend opened with device-resident weight copies in " +
            $"'{ModelRoute.CudaBackendSeam}', which this open has none",

        BudgetTier.Vram =>
            $"'{resource}' has no device-backed pool opened for it — accelerator memory holds weight copies today, and device-backed pools " +
            $"for '{resource}' run in '{ModelRoute.CudaBackendSeam}' once they are opened",

        _ when !diskWrites =>
            $"'{resource}' has no disk-backed pool opened for it — write-side bytes reach storage only where something writes them, " +
            $"and nothing resident reads a chain through {BudgetTierNames.Of(tier)} yet",

        _ =>
            $"'{resource}' would hold working copies in {BudgetTierNames.Of(tier)}, and the disk-backed pool those claims need is not opened for it",
    };

    private static string RequireModelName(string modelName) => !string.IsNullOrWhiteSpace(modelName)
        ? modelName
        : throw new ModelException("budget resolution declares no model name: every refusal this reports names the model its budgets belong to");

    /// <summary>Whether this resource's complete set already exists on its origin store rather than only in LAIR's working copies.</summary>
    private static bool IsReadSide(BudgetResource resource) => resource == BudgetResource.Weights;

    private static int PriorityOf(BudgetTier tier) => Array.IndexOf(PriorityOrder, tier);

    private static bool IsKnownTier(BudgetTier tier) => PriorityOf(tier) >= 0;

    private static bool ContainsTier(IReadOnlyList<BudgetTier> tiers, BudgetTier tier)
    {
        foreach (BudgetTier opened in tiers)
        {
            if (opened == tier)
            {
                return true;
            }
        }

        return false;
    }

    private static string DescribeOrder(IReadOnlyList<BudgetTier> order) =>
        string.Join(", ", order.Select(tier => $"'{BudgetTierNames.Of(tier)}'"));

    private static string DescribeResource(BudgetResource resource) => BudgetResourceNames.TryParse(BudgetResourceNames.Of(resource), out _)
        ? BudgetResourceNames.Of(resource)
        : resource.ToString();

    /// <summary>The unit one resource's ceilings count in, named in every message that states a ceiling rule.</summary>
    internal static string CapUnitName(BudgetResource resource) => resource == BudgetResource.Kv ? "KV block" : "megabyte";
}
