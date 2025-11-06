using System;
using Orleans;

namespace Orleans.GpuBridge.Grains.Stream.Metrics;

/// <summary>
/// Enhanced stream processing metrics with GPU stats, throughput, and backpressure information
/// </summary>
[GenerateSerializer]
public sealed record StreamProcessingMetrics(
    // Basic metrics
    [property: Id(0)] long TotalItemsProcessed,
    [property: Id(1)] long TotalItemsFailed,
    [property: Id(2)] TimeSpan TotalProcessingTime,

    // Batch metrics
    [property: Id(3)] long TotalBatchesProcessed,
    [property: Id(4)] double AverageBatchSize,
    [property: Id(5)] double BatchEfficiency,  // avg_batch_size / max_batch_size

    // Latency metrics
    [property: Id(6)] double AverageLatencyMs,  // Average item latency
    [property: Id(7)] double P50LatencyMs,      // Median latency
    [property: Id(8)] double P99LatencyMs,      // 99th percentile latency

    // GPU metrics
    [property: Id(9)] TimeSpan TotalKernelExecutionTime,
    [property: Id(10)] TimeSpan TotalMemoryTransferTime,
    [property: Id(11)] double KernelEfficiency,  // kernel_time / (kernel_time + transfer_time)
    [property: Id(12)] double MemoryBandwidthMBps,
    [property: Id(13)] long TotalGpuMemoryAllocated,

    // Throughput metrics
    [property: Id(14)] double CurrentThroughput,  // items/second (last 10 seconds)
    [property: Id(15)] double PeakThroughput,

    // Backpressure metrics
    [property: Id(16)] long BufferCurrentSize,
    [property: Id(17)] long BufferCapacity,
    [property: Id(18)] double BufferUtilization,  // current / capacity
    [property: Id(19)] long TotalPauseCount,
    [property: Id(20)] TimeSpan TotalPauseDuration,

    // Device info
    [property: Id(21)] string DeviceType,
    [property: Id(22)] string DeviceName,
    [property: Id(23)] DateTime StartTime,
    [property: Id(24)] DateTime? LastProcessedTime)
{
    /// <summary>
    /// Processing uptime
    /// </summary>
    public TimeSpan Uptime => (LastProcessedTime ?? DateTime.UtcNow) - StartTime;

    /// <summary>
    /// Average throughput over entire session (items/second)
    /// </summary>
    public double AverageThroughput =>
        Uptime.TotalSeconds > 0 ? TotalItemsProcessed / Uptime.TotalSeconds : 0;

    /// <summary>
    /// Success rate (percentage of items successfully processed)
    /// </summary>
    public double SuccessRate
    {
        get
        {
            var total = TotalItemsProcessed + TotalItemsFailed;
            return total > 0 ? (double)TotalItemsProcessed / total * 100 : 0;
        }
    }

    /// <summary>
    /// Average pause duration (when backpressure occurs)
    /// </summary>
    public TimeSpan AveragePauseDuration =>
        TotalPauseCount > 0
            ? TimeSpan.FromMilliseconds(TotalPauseDuration.TotalMilliseconds / TotalPauseCount)
            : TimeSpan.Zero;
}
