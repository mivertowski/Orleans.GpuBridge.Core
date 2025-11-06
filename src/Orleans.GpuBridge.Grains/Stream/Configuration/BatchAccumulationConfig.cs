using System;

namespace Orleans.GpuBridge.Grains.Stream.Configuration;

/// <summary>
/// Configuration for intelligent batch accumulation
/// </summary>
public sealed class BatchAccumulationConfig
{
    /// <summary>
    /// Minimum items to accumulate before processing (latency vs throughput tradeoff)
    /// </summary>
    public int MinBatchSize { get; init; } = 32;

    /// <summary>
    /// Maximum items per batch (GPU memory constraint)
    /// </summary>
    public int MaxBatchSize { get; init; } = 10_000;

    /// <summary>
    /// Maximum time to wait before flushing partial batch (latency SLA)
    /// </summary>
    public TimeSpan MaxBatchWaitTime { get; init; } = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Target GPU memory utilization (0.0 - 1.0)
    /// </summary>
    public double GpuMemoryUtilizationTarget { get; init; } = 0.7; // 70%

    /// <summary>
    /// Enable adaptive batch sizing based on throughput
    /// </summary>
    public bool EnableAdaptiveBatching { get; init; } = true;

    /// <summary>
    /// Default configuration
    /// </summary>
    public static BatchAccumulationConfig Default => new();
}
