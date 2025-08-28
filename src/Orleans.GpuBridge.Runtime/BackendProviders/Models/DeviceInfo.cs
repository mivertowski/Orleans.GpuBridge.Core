using Orleans.GpuBridge.Runtime.BackendProviders.Enums;

namespace Orleans.GpuBridge.Runtime.BackendProviders.Models;

/// <summary>
/// Device information
/// </summary>
public sealed record DeviceInfo(
    int Index,
    string Name,
    BackendType Backend,
    long TotalMemory,
    int ComputeUnits,
    string[] Extensions);