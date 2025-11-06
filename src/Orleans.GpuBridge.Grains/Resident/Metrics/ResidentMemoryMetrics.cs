using System;
using Orleans;

namespace Orleans.GpuBridge.Grains.Resident.Metrics;

/// <summary>
/// Comprehensive metrics for GPU-resident memory grain with Ring Kernel statistics.
/// Tracks memory pool efficiency, Ring Kernel throughput, kernel cache performance,
/// and overall GPU utilization.
/// </summary>
[GenerateSerializer]
public sealed record ResidentMemoryMetrics(
    // Memory pool metrics (6 properties)
    [property: Id(0)] long TotalPoolSizeBytes,
    [property: Id(1)] long UsedPoolSizeBytes,
    [property: Id(2)] double PoolUtilization,
    [property: Id(3)] long PoolHitCount,
    [property: Id(4)] long PoolMissCount,
    [property: Id(5)] double PoolHitRate,

    // Ring Kernel metrics (4 properties)
    [property: Id(6)] long TotalMessagesProcessed,
    [property: Id(7)] double MessagesPerSecond,
    [property: Id(8)] double AverageMessageLatencyNs,
    [property: Id(9)] long PendingMessageCount,

    // Allocation metrics (4 properties)
    [property: Id(10)] int ActiveAllocationCount,
    [property: Id(11)] long TotalAllocatedBytes,
    [property: Id(12)] int KernelCacheSize,
    [property: Id(13)] double KernelCacheHitRate,

    // Device info (3 properties)
    [property: Id(14)] string DeviceType,
    [property: Id(15)] string DeviceName,
    [property: Id(16)] DateTime StartTime)
{
    /// <summary>
    /// Grain uptime since activation.
    /// </summary>
    public TimeSpan Uptime => DateTime.UtcNow - StartTime;

    /// <summary>
    /// Average pool size per allocation in bytes.
    /// </summary>
    public double AverageAllocationSize =>
        ActiveAllocationCount > 0
            ? (double)TotalAllocatedBytes / ActiveAllocationCount
            : 0;

    /// <summary>
    /// Memory efficiency: pool hits divided by total allocations.
    /// High values (>90%) indicate excellent pool reuse.
    /// </summary>
    public double MemoryEfficiency => PoolHitRate;

    /// <summary>
    /// Ring Kernel throughput in millions of messages per second.
    /// </summary>
    public double ThroughputMMessagesPerSec => MessagesPerSecond / 1_000_000.0;

    /// <summary>
    /// Average message latency in microseconds.
    /// </summary>
    public double AverageMessageLatencyMicroseconds => AverageMessageLatencyNs / 1000.0;

    /// <summary>
    /// Kernel cache efficiency: cache hits divided by total compiles.
    /// High values (>95%) indicate excellent kernel reuse.
    /// </summary>
    public double KernelEfficiency => KernelCacheHitRate;

    /// <summary>
    /// Total GPU memory in megabytes.
    /// </summary>
    public double TotalMemoryMB => TotalAllocatedBytes / (1024.0 * 1024.0);

    /// <summary>
    /// Used GPU memory in megabytes.
    /// </summary>
    public double UsedMemoryMB => UsedPoolSizeBytes / (1024.0 * 1024.0);
}

/// <summary>
/// Internal metrics tracker for the GPU resident grain.
/// Collects grain-level statistics to complement Ring Kernel metrics.
/// </summary>
internal sealed class ResidentMemoryMetricsTracker
{
    private long _totalAllocations;
    private long _totalWrites;
    private long _totalReads;
    private long _totalComputes;
    private long _totalReleases;
    private long _poolHits;
    private long _kernelCacheHits;
    private DateTime _startTime;

    public void Start()
    {
        _startTime = DateTime.UtcNow;
    }

    public void RecordAllocation(long sizeBytes, bool isPoolHit)
    {
        Interlocked.Increment(ref _totalAllocations);
        if (isPoolHit)
        {
            Interlocked.Increment(ref _poolHits);
        }
    }

    public void RecordWrite(long bytesWritten, double transferTimeMicroseconds)
    {
        Interlocked.Increment(ref _totalWrites);
    }

    public void RecordRead(long bytesRead, double transferTimeMicroseconds)
    {
        Interlocked.Increment(ref _totalReads);
    }

    public void RecordCompute(double totalTimeMicroseconds, bool isCacheHit)
    {
        Interlocked.Increment(ref _totalComputes);
        if (isCacheHit)
        {
            Interlocked.Increment(ref _kernelCacheHits);
        }
    }

    public void RecordRelease(long sizeBytes, bool returnedToPool)
    {
        Interlocked.Increment(ref _totalReleases);
    }

    public GrainLevelMetrics GetMetrics()
    {
        return new GrainLevelMetrics
        {
            TotalAllocations = _totalAllocations,
            TotalWrites = _totalWrites,
            TotalReads = _totalReads,
            TotalComputes = _totalComputes,
            TotalReleases = _totalReleases,
            PoolHits = _poolHits,
            KernelCacheHits = _kernelCacheHits,
            StartTime = _startTime
        };
    }
}

/// <summary>
/// Grain-level metrics (not Ring Kernel metrics).
/// </summary>
internal sealed record GrainLevelMetrics
{
    public long TotalAllocations { get; init; }
    public long TotalWrites { get; init; }
    public long TotalReads { get; init; }
    public long TotalComputes { get; init; }
    public long TotalReleases { get; init; }
    public long PoolHits { get; init; }
    public long KernelCacheHits { get; init; }
    public DateTime StartTime { get; init; }
}
