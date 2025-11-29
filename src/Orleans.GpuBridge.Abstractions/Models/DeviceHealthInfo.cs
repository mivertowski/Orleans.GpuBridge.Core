using System;
using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Abstractions.Models;

/// <summary>
/// Comprehensive device health information for production monitoring
/// </summary>
public sealed record DeviceHealthInfo
{
    /// <summary>
    /// Device identifier
    /// </summary>
    public string DeviceId { get; init; } = string.Empty;

    /// <summary>
    /// Last health check timestamp
    /// </summary>
    public DateTime LastCheckTime { get; init; } = DateTime.UtcNow;

    /// <summary>
    /// Current device temperature in Celsius
    /// </summary>
    public int TemperatureCelsius { get; init; }

    /// <summary>
    /// Maximum safe operating temperature
    /// </summary>
    public int MaxTemperatureCelsius { get; init; } = 85;

    /// <summary>
    /// Whether the device is currently thermal throttling
    /// </summary>
    public bool IsThermalThrottling { get; init; }

    /// <summary>
    /// Current power usage in watts
    /// </summary>
    public double PowerUsageWatts { get; init; }

    /// <summary>
    /// Maximum power limit in watts
    /// </summary>
    public double PowerLimitWatts { get; init; }

    /// <summary>
    /// Memory utilization percentage (0-100)
    /// </summary>
    public double MemoryUtilizationPercent { get; init; }

    /// <summary>
    /// GPU utilization percentage (0-100)
    /// </summary>
    public double GpuUtilizationPercent { get; init; }

    /// <summary>
    /// Error count since last reset
    /// </summary>
    public int ErrorCount { get; init; }

    /// <summary>
    /// Consecutive failed health checks
    /// </summary>
    public int ConsecutiveFailures { get; init; }

    /// <summary>
    /// Overall health score (0.0 to 1.0, higher is better)
    /// </summary>
    public double HealthScore => CalculateHealthScore();

    /// <summary>
    /// Current device status
    /// </summary>
    public DeviceStatus Status { get; init; } = DeviceStatus.Available;

    /// <summary>
    /// Whether the device is considered healthy
    /// </summary>
    public bool IsHealthy => HealthScore >= 0.7 &&
                            Status == DeviceStatus.Available &&
                            ConsecutiveFailures < 3;

    /// <summary>
    /// Predicted failure probability in next hour (0.0 to 1.0)
    /// </summary>
    public double PredictedFailureProbability { get; init; }

    /// <summary>
    /// Time since last successful operation
    /// </summary>
    public TimeSpan TimeSinceLastSuccess { get; init; }

    private double CalculateHealthScore()
    {
        double score = 1.0;

        // Temperature penalty
        if (MaxTemperatureCelsius > 0)
        {
            var tempRatio = TemperatureCelsius / (double)MaxTemperatureCelsius;
            if (tempRatio > 0.8) score -= (tempRatio - 0.8) * 2.5; // Heavy penalty for high temps
        }

        // Throttling penalty
        if (IsThermalThrottling) score -= 0.3;

        // Power penalty
        if (PowerLimitWatts > 0 && PowerUsageWatts > PowerLimitWatts * 0.95)
        {
            score -= 0.2; // Penalty for near power limit
        }

        // Memory utilization penalty
        if (MemoryUtilizationPercent > 90) score -= 0.2;

        // Error penalty
        score -= Math.Min(0.4, ErrorCount * 0.05);

        // Consecutive failure penalty
        score -= Math.Min(0.5, ConsecutiveFailures * 0.1);

        // Predicted failure penalty
        score -= PredictedFailureProbability * 0.3;

        return Math.Max(0.0, Math.Min(1.0, score));
    }
}

/// <summary>
/// Benchmark scores for performance comparison
/// </summary>
public sealed record DeviceBenchmarkScores
{
    /// <summary>
    /// Overall performance score (0.0 to 10.0, higher is better)
    /// </summary>
    public double OverallScore { get; init; } = 1.0;

    /// <summary>
    /// Compute performance score
    /// </summary>
    public double ComputeScore { get; init; } = 1.0;

    /// <summary>
    /// Memory bandwidth score
    /// </summary>
    public double MemoryBandwidthScore { get; init; } = 1.0;

    /// <summary>
    /// Latency performance score
    /// </summary>
    public double LatencyScore { get; init; } = 1.0;

    /// <summary>
    /// Throughput performance score
    /// </summary>
    public double ThroughputScore { get; init; } = 1.0;

    /// <summary>
    /// Benchmark timestamp
    /// </summary>
    public DateTime BenchmarkTime { get; init; } = DateTime.UtcNow;

    /// <summary>
    /// Benchmark duration
    /// </summary>
    public TimeSpan BenchmarkDuration { get; init; }

    /// <summary>
    /// Whether benchmark results are reliable
    /// </summary>
    public bool IsReliable { get; init; } = true;
}

/// <summary>
/// Device load balancing information
/// </summary>
public sealed record DeviceLoadInfo
{
    /// <summary>
    /// Device identifier
    /// </summary>
    public string DeviceId { get; init; } = string.Empty;

    /// <summary>
    /// Current queue depth
    /// </summary>
    public int CurrentQueueDepth { get; init; }

    /// <summary>
    /// Average queue depth over last minute
    /// </summary>
    public double AverageQueueDepth { get; init; }

    /// <summary>
    /// Current processing rate (operations per second)
    /// </summary>
    public double ProcessingRate { get; init; }

    /// <summary>
    /// Average processing latency in milliseconds
    /// </summary>
    public double AverageLatencyMs { get; init; }

    /// <summary>
    /// Success rate percentage (0-100)
    /// </summary>
    public double SuccessRatePercent { get; init; } = 100.0;

    /// <summary>
    /// Load factor for selection weighting (higher = prefer less)
    /// </summary>
    public double LoadFactor => CalculateLoadFactor();

    /// <summary>
    /// Selection weight for load balancing (higher = more likely to be selected)
    /// </summary>
    public double SelectionWeight { get; set; } = 1.0;

    /// <summary>
    /// Last update timestamp
    /// </summary>
    public DateTime LastUpdateTime { get; init; } = DateTime.UtcNow;

    /// <summary>
    /// Predicted completion time for average workload
    /// </summary>
    public TimeSpan PredictedCompletionTime { get; init; }

    /// <summary>
    /// Current device utilization percentage (0-100)
    /// </summary>
    public double CurrentUtilization { get; set; }

    /// <summary>
    /// Queue depth for pending work items
    /// </summary>
    public int QueueDepth { get; set; }

    /// <summary>
    /// Throttle deadline (if device is being throttled)
    /// </summary>
    public DateTime? ThrottleUntil { get; set; }

    /// <summary>
    /// Performance history for trend analysis
    /// </summary>
    public List<double> PerformanceHistory { get; set; } = new();

    private double CalculateLoadFactor()
    {
        // Combine multiple load indicators
        var queueFactor = Math.Min(2.0, CurrentQueueDepth / 10.0); // Queue depth impact
        var latencyFactor = Math.Min(1.5, AverageLatencyMs / 100.0); // Latency impact
        var successFactor = Math.Max(0.1, SuccessRatePercent / 100.0); // Success rate impact

        // Lower load factor is better (inverse of success rate for penalty)
        return (queueFactor + latencyFactor) / successFactor;
    }
}

/// <summary>
/// Device capability cache information
/// </summary>
public sealed record DeviceCapabilityCache
{
    /// <summary>
    /// Device identifier
    /// </summary>
    public string DeviceId { get; init; } = string.Empty;

    /// <summary>
    /// Cached capabilities dictionary
    /// </summary>
    public IReadOnlyDictionary<string, object> Capabilities { get; init; } =
        new Dictionary<string, object>();

    /// <summary>
    /// Cache creation timestamp
    /// </summary>
    public DateTime CacheTime { get; init; } = DateTime.UtcNow;

    /// <summary>
    /// Cache expiry time
    /// </summary>
    public DateTime ExpiryTime { get; init; } = DateTime.UtcNow.AddHours(1);

    /// <summary>
    /// Whether the cache is still valid
    /// </summary>
    public bool IsValid => DateTime.UtcNow < ExpiryTime;

    /// <summary>
    /// Cache hit count for performance tracking
    /// </summary>
    public int HitCount { get; init; }

    /// <summary>
    /// Last access timestamp
    /// </summary>
    public DateTime LastAccessTime { get; init; } = DateTime.UtcNow;

    /// <summary>
    /// Last update timestamp
    /// </summary>
    public DateTime LastUpdated { get; init; } = DateTime.UtcNow;

    /// <summary>
    /// Benchmark scores for performance comparison
    /// </summary>
    public DeviceBenchmarkScores BenchmarkScores { get; init; } = new();
}