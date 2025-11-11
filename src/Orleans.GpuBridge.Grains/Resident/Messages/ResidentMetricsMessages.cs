using System;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Grains.Resident.Messages;

/// <summary>
/// Message to get Ring Kernel metrics.
/// </summary>
public sealed record GetMetricsMessage : ResidentMessage
{
    /// <summary>
    /// Whether to include detailed per-allocation metrics.
    /// </summary>
    public bool IncludeDetails { get; init; }

    public GetMetricsMessage(bool includeDetails = false)
    {
        IncludeDetails = includeDetails;
    }
}

/// <summary>
/// Response message containing Ring Kernel metrics.
/// </summary>
public sealed record MetricsResponse : ResidentMessage
{
    /// <summary>
    /// Request ID this response corresponds to.
    /// </summary>
    public Guid OriginalRequestId { get; init; }

    /// <summary>
    /// Total messages processed by Ring Kernel.
    /// </summary>
    public long TotalMessagesProcessed { get; init; }

    /// <summary>
    /// Current messages per second throughput.
    /// </summary>
    public double MessagesPerSecond { get; init; }

    /// <summary>
    /// Average message processing latency in nanoseconds.
    /// </summary>
    public double AverageLatencyNanoseconds { get; init; }

    /// <summary>
    /// Memory pool hit count.
    /// </summary>
    public long PoolHitCount { get; init; }

    /// <summary>
    /// Memory pool miss count.
    /// </summary>
    public long PoolMissCount { get; init; }

    /// <summary>
    /// Kernel cache hit count.
    /// </summary>
    public long KernelCacheHitCount { get; init; }

    /// <summary>
    /// Kernel cache miss count.
    /// </summary>
    public long KernelCacheMissCount { get; init; }

    /// <summary>
    /// Total GPU memory allocated in bytes.
    /// </summary>
    public long TotalAllocatedBytes { get; init; }

    /// <summary>
    /// Active allocation count.
    /// </summary>
    public int ActiveAllocationCount { get; init; }

    public MetricsResponse(
        Guid originalRequestId,
        long totalMessagesProcessed,
        double messagesPerSecond,
        double averageLatencyNanoseconds,
        long poolHitCount,
        long poolMissCount,
        long kernelCacheHitCount,
        long kernelCacheMissCount,
        long totalAllocatedBytes,
        int activeAllocationCount)
    {
        OriginalRequestId = originalRequestId;
        TotalMessagesProcessed = totalMessagesProcessed;
        MessagesPerSecond = messagesPerSecond;
        AverageLatencyNanoseconds = averageLatencyNanoseconds;
        PoolHitCount = poolHitCount;
        PoolMissCount = poolMissCount;
        KernelCacheHitCount = kernelCacheHitCount;
        KernelCacheMissCount = kernelCacheMissCount;
        TotalAllocatedBytes = totalAllocatedBytes;
        ActiveAllocationCount = activeAllocationCount;
    }
}
