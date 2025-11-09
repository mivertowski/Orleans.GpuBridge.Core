namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// GPU memory statistics
/// </summary>
public sealed class GpuMemoryStats
{
    public long TotalAllocated { get; init; }
    public long TotalReturned { get; init; }
    public long CurrentInUse { get; init; }
    public long PeakUsage { get; init; }
    public int SegmentCount { get; init; }
    public int AllocationCount { get; init; }

    public double UtilizationPercent => TotalAllocated > 0
        ? (CurrentInUse / (double)TotalAllocated) * 100
        : 0;

    public long LeakedBytes => TotalAllocated - TotalReturned;
}
