using System;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// Memory pool statistics
/// </summary>
public record MemoryPoolStats
{
    public long TotalAllocatedBytes { get; init; }
    public long TotalRentedBytes { get; init; }
    public int TotalBuffers { get; init; }
    public double EfficiencyPercent { get; init; }
    public BucketStats[] BucketStats { get; init; } = Array.Empty<BucketStats>();
}

public record BucketStats
{
    public int Size { get; init; }
    public int Available { get; init; }
    public int Total { get; init; }
}
