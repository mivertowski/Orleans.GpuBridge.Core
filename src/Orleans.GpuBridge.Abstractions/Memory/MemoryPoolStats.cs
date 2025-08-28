namespace Orleans.GpuBridge.Abstractions.Memory;

/// <summary>
/// Statistics for memory pool
/// </summary>
public sealed record MemoryPoolStats(
    long TotalAllocated,
    long InUse,
    long Available,
    int BufferCount,
    int RentCount,
    int ReturnCount)
{
    public double UtilizationPercent => TotalAllocated > 0 
        ? (InUse / (double)TotalAllocated) * 100 
        : 0;
}