namespace Lair.Http.Monitoring;

/// <summary>
/// One read of an opened engine's accelerator accounting, taken from the compute backends it holds rather than from any
/// OS- or hardware-level telemetry view: how long its kernels have been busy on the device in total, what bytes and how
/// many operations their runtime copies have crossed, and how much default-copy weight residency those instances hold
/// open right now. Cumulative fields read total-since-open; persistent weight bytes are a current reading that drops as an
/// instance releases allocations. A serve with no accelerator backend answers nothing here at all — the provider reports
/// null so its snapshots keep the shape they had before this seam existed.
/// </summary>
public sealed record AcceleratorAccounting(
    long KernelBusyMicrosecondsTotal,
    long RuntimeTransferBytesTotal,
    long RuntimeTransferCallsTotal,
    long PersistentDeviceWeightBytes);
