using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Abstractions;

/// <summary>
/// Represents a GPU device
/// </summary>
public sealed record GpuDevice(
    int Index,
    string Name,
    DeviceType Type,
    long TotalMemoryBytes,
    long AvailableMemoryBytes,
    int ComputeUnits,
    IReadOnlyList<string> Capabilities)
{
    public double MemoryUtilization => TotalMemoryBytes > 0 
        ? (TotalMemoryBytes - AvailableMemoryBytes) / (double)TotalMemoryBytes 
        : 0;
}