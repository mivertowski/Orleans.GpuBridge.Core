using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Abstractions;

/// <summary>
/// Information about the GPU bridge
/// </summary>
public sealed record GpuBridgeInfo(
    string Version,
    int DeviceCount,
    long TotalMemoryBytes,
    GpuBackend Backend,
    bool IsGpuAvailable,
    IReadOnlyDictionary<string, object>? Metadata = null);