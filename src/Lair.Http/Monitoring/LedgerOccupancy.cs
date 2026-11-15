using System.Text.Json.Serialization;

namespace Lair.Http.Monitoring;

/// <summary>
/// One tier position the host device ledger has claims seated at, reported additively on monitor snapshots: the tier's
/// capacity posture (null capacity states the explicit unbounded posture — a public total no configuration declared),
/// what the seated claims hold against it in the resource's native unit, and the contested holders when more than one
/// model shares the tier. An engine whose ledger has seated no claims reports no entries at all, so snapshots of an
/// engine that claims nothing keep the shape they had before the ledger seam existed.
/// </summary>
public sealed record LedgerOccupancyEntry(
    [property: JsonPropertyName("resource")] string Resource,
    [property: JsonPropertyName("tier")] string Tier,
    [property: JsonPropertyName("unit")] string Unit,

    /// <summary>The capacity the opened seam states for this tier and resource, in the entry's unit; null states the unbounded posture.</summary>
    [property: JsonPropertyName("capacity")] long? Capacity,

    /// <summary>The sum of claims holding working copies at this tier for this resource, in the entry's unit.</summary>
    [property: JsonPropertyName("occupied")] long Occupied,

    /// <summary>True when more than one claimant shares this tier for this resource — the contested posture an operator resolves by reading <see cref="Holders"/>.</summary>
    [property: JsonPropertyName("contested")] bool Contested,

    /// <summary>The claims sharing this tier in registration order, when contested; null when a single claimant holds it.</summary>
    [property: JsonPropertyName("holders")] IReadOnlyList<LedgerHolderEntry>? Holders = null);

/// <summary>One claimant sharing a ledger tier: the model holding it and the amount, in the tier's unit.</summary>
public sealed record LedgerHolderEntry(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("amount")] long Amount);
