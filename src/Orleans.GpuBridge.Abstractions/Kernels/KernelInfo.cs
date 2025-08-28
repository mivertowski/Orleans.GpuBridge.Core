using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Abstractions.Kernels;

/// <summary>
/// Information about a kernel
/// </summary>
public sealed record KernelInfo(
    KernelId Id,
    string Description,
    Type InputType,
    Type OutputType,
    bool SupportsGpu,
    int PreferredBatchSize,
    IReadOnlyDictionary<string, object>? Metadata = null);