namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// GPU memory statistics
/// </summary>
public sealed class GpuMemoryStats
{
    /// <summary>Gets the total bytes allocated</summary>
    public long TotalAllocated { get; init; }

    /// <summary>Gets the total bytes returned to the pool</summary>
    public long TotalReturned { get; init; }

    /// <summary>Gets the current bytes in use</summary>
    public long CurrentInUse { get; init; }

    /// <summary>Gets the peak memory usage in bytes</summary>
    public long PeakUsage { get; init; }

    /// <summary>Gets the number of memory segments</summary>
    public int SegmentCount { get; init; }

    /// <summary>Gets the total number of allocations</summary>
    public int AllocationCount { get; init; }

    /// <summary>Gets the memory utilization percentage</summary>
    public double UtilizationPercent => TotalAllocated > 0
        ? (CurrentInUse / (double)TotalAllocated) * 100
        : 0;

    /// <summary>Gets the number of leaked bytes (allocated but not returned)</summary>
    public long LeakedBytes => TotalAllocated - TotalReturned;
}
