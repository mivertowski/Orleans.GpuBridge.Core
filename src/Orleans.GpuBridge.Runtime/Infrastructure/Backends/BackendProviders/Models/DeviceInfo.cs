using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Runtime.BackendProviders.Models;

/// <summary>
/// Device information
/// </summary>
public sealed record DeviceInfo(
    int Index,
    string Name,
    GpuBackend Backend,
    long TotalMemory,
    int ComputeUnits,
    string[] Extensions);