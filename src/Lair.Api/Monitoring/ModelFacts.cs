namespace Lair.Api.Monitoring;

/// <summary>
/// Configuration- and backend-backed facts about the model a bus observes, attached once by the owning service
/// (and re-pushed when hot reload changes them). They feed only snapshot payload fields — sampling cost is one
/// volatile record read per tick — so an unattached bus publishes exactly its v1 shape with nulls in these slots.
/// </summary>
public sealed record ModelFacts(
    string Type,
    int WorkerThreads,
    string? CoreSet,
    int? ThreadQuota,
    string DecodeDevice,
    string PrefillDevice,
    string KvResidency);
