namespace Lair.Backends.Cuda;

/// <summary>How a <see cref="CudaBackend"/> instance puts weight bytes in reach of its kernels. DeviceCopy is the posture every default-opened instance runs: each tensor is copied into device memory once per instance and stays there. HostMapped launches against the device pointer of a host region the weight source already owns — the weights never enter persistent device memory on this posture.</summary>
public enum CudaWeightResidency
{
    /// <summary>Each weight tensor is copied to device memory once per instance and the kernels read those copies.</summary>
    DeviceCopy,

    /// <summary>Each weight tensor's live host region is registered with the driver once per instance and the kernels read through the device pointer the registration answers with; no weight tensor is copied into persistent device memory.</summary>
    HostMapped,
}
