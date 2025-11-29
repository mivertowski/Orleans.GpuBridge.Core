// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

namespace Orleans.GpuBridge.Abstractions.Placement;

/// <summary>
/// Monitors queue depth across GPU ring kernels for intelligent placement decisions.
/// </summary>
/// <remarks>
/// <para>
/// This interface provides real-time visibility into ring kernel message queue utilization,
/// enabling the placement director to make informed decisions about grain placement.
/// </para>
/// <para>
/// <strong>Key Metrics:</strong>
/// <list type="bullet">
/// <item><description>Input queue depth: Messages waiting to be processed</description></item>
/// <item><description>Output queue depth: Responses waiting to be sent</description></item>
/// <item><description>Queue utilization: Percentage of queue capacity in use</description></item>
/// <item><description>Throughput: Messages processed per second</description></item>
/// </list>
/// </para>
/// </remarks>
public interface IQueueDepthMonitor
{
    /// <summary>
    /// Gets the current queue depth snapshot for a specific GPU/silo.
    /// </summary>
    /// <param name="siloId">Silo identifier (null for local silo).</param>
    /// <param name="deviceIndex">GPU device index (default: 0).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Queue depth snapshot with all metrics.</returns>
    Task<QueueDepthSnapshot> GetQueueDepthAsync(
        string? siloId = null,
        int deviceIndex = 0,
        CancellationToken ct = default);

    /// <summary>
    /// Gets aggregated queue depth metrics across all ring kernels on a device.
    /// </summary>
    /// <param name="siloId">Silo identifier (null for local silo).</param>
    /// <param name="deviceIndex">GPU device index.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Aggregated metrics across all kernels.</returns>
    Task<AggregatedQueueMetrics> GetAggregatedMetricsAsync(
        string? siloId = null,
        int deviceIndex = 0,
        CancellationToken ct = default);

    /// <summary>
    /// Gets queue depth history for trend analysis.
    /// </summary>
    /// <param name="siloId">Silo identifier.</param>
    /// <param name="deviceIndex">GPU device index.</param>
    /// <param name="duration">History duration to retrieve.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Historical queue depth samples.</returns>
    Task<QueueDepthHistory> GetHistoryAsync(
        string? siloId = null,
        int deviceIndex = 0,
        TimeSpan? duration = null,
        CancellationToken ct = default);

    /// <summary>
    /// Checks if a device has capacity for additional grains.
    /// </summary>
    /// <param name="siloId">Silo identifier.</param>
    /// <param name="deviceIndex">GPU device index.</param>
    /// <param name="maxQueueUtilization">Maximum acceptable queue utilization (0.0-1.0).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>True if device has capacity, false otherwise.</returns>
    Task<bool> HasCapacityAsync(
        string? siloId = null,
        int deviceIndex = 0,
        double maxQueueUtilization = 0.8,
        CancellationToken ct = default);

    /// <summary>
    /// Subscribes to queue depth threshold alerts.
    /// </summary>
    /// <param name="threshold">Utilization threshold that triggers alert (0.0-1.0).</param>
    /// <param name="callback">Callback when threshold is exceeded.</param>
    /// <returns>Subscription handle for unsubscription.</returns>
    IDisposable SubscribeToAlerts(double threshold, Action<QueueDepthAlert> callback);
}

/// <summary>
/// Point-in-time snapshot of queue depth metrics.
/// </summary>
public readonly record struct QueueDepthSnapshot
{
    /// <summary>
    /// Timestamp when snapshot was taken (nanoseconds since epoch).
    /// </summary>
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Silo identifier.
    /// </summary>
    public required string SiloId { get; init; }

    /// <summary>
    /// GPU device index.
    /// </summary>
    public required int DeviceIndex { get; init; }

    /// <summary>
    /// Number of active ring kernels on this device.
    /// </summary>
    public required int ActiveKernelCount { get; init; }

    /// <summary>
    /// Total input queue depth across all kernels.
    /// </summary>
    public required int TotalInputQueueDepth { get; init; }

    /// <summary>
    /// Total output queue depth across all kernels.
    /// </summary>
    public required int TotalOutputQueueDepth { get; init; }

    /// <summary>
    /// Maximum input queue capacity across all kernels.
    /// </summary>
    public required int TotalInputQueueCapacity { get; init; }

    /// <summary>
    /// Maximum output queue capacity across all kernels.
    /// </summary>
    public required int TotalOutputQueueCapacity { get; init; }

    /// <summary>
    /// Input queue utilization (0.0 - 1.0).
    /// </summary>
    public double InputQueueUtilization =>
        TotalInputQueueCapacity > 0 ? (double)TotalInputQueueDepth / TotalInputQueueCapacity : 0.0;

    /// <summary>
    /// Output queue utilization (0.0 - 1.0).
    /// </summary>
    public double OutputQueueUtilization =>
        TotalOutputQueueCapacity > 0 ? (double)TotalOutputQueueDepth / TotalOutputQueueCapacity : 0.0;

    /// <summary>
    /// Average queue utilization across input and output.
    /// </summary>
    public double AverageQueueUtilization => (InputQueueUtilization + OutputQueueUtilization) / 2.0;

    /// <summary>
    /// Messages processed per second (aggregate throughput).
    /// </summary>
    public required double ThroughputMsgsPerSec { get; init; }

    /// <summary>
    /// GPU compute utilization (0.0 - 1.0).
    /// </summary>
    public required double GpuUtilization { get; init; }

    /// <summary>
    /// Available GPU memory in bytes.
    /// </summary>
    public required long AvailableMemoryBytes { get; init; }

    /// <summary>
    /// Total GPU memory in bytes.
    /// </summary>
    public required long TotalMemoryBytes { get; init; }

    /// <summary>
    /// Memory utilization (0.0 - 1.0).
    /// </summary>
    public double MemoryUtilization =>
        TotalMemoryBytes > 0 ? 1.0 - ((double)AvailableMemoryBytes / TotalMemoryBytes) : 0.0;

    /// <summary>
    /// Available memory ratio (0.0 - 1.0).
    /// </summary>
    public double AvailableMemoryRatio =>
        TotalMemoryBytes > 0 ? (double)AvailableMemoryBytes / TotalMemoryBytes : 0.0;
}

/// <summary>
/// Aggregated queue metrics across multiple ring kernels.
/// </summary>
public readonly record struct AggregatedQueueMetrics
{
    /// <summary>
    /// Timestamp when metrics were aggregated.
    /// </summary>
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Silo identifier.
    /// </summary>
    public required string SiloId { get; init; }

    /// <summary>
    /// GPU device index.
    /// </summary>
    public required int DeviceIndex { get; init; }

    /// <summary>
    /// Number of ring kernels included in aggregation.
    /// </summary>
    public required int KernelCount { get; init; }

    /// <summary>
    /// Minimum queue utilization across all kernels.
    /// </summary>
    public required double MinQueueUtilization { get; init; }

    /// <summary>
    /// Maximum queue utilization across all kernels.
    /// </summary>
    public required double MaxQueueUtilization { get; init; }

    /// <summary>
    /// Average queue utilization across all kernels.
    /// </summary>
    public required double AvgQueueUtilization { get; init; }

    /// <summary>
    /// Standard deviation of queue utilization (for load balance assessment).
    /// </summary>
    public required double StdDevQueueUtilization { get; init; }

    /// <summary>
    /// Total throughput across all kernels (msgs/sec).
    /// </summary>
    public required double TotalThroughput { get; init; }

    /// <summary>
    /// Average processing latency (nanoseconds).
    /// </summary>
    public required double AvgProcessingLatencyNanos { get; init; }

    /// <summary>
    /// P99 processing latency (nanoseconds).
    /// </summary>
    public required double P99ProcessingLatencyNanos { get; init; }

    /// <summary>
    /// Indicates whether load is balanced (stddev &lt; 0.1).
    /// </summary>
    public bool IsLoadBalanced => StdDevQueueUtilization < 0.1;

    /// <summary>
    /// Indicates whether any kernel is overloaded (max &gt; 0.9).
    /// </summary>
    public bool HasOverloadedKernel => MaxQueueUtilization > 0.9;
}

/// <summary>
/// Historical queue depth data for trend analysis.
/// </summary>
public sealed class QueueDepthHistory
{
    /// <summary>
    /// Silo identifier.
    /// </summary>
    public required string SiloId { get; init; }

    /// <summary>
    /// GPU device index.
    /// </summary>
    public required int DeviceIndex { get; init; }

    /// <summary>
    /// Start of history window (nanoseconds since epoch).
    /// </summary>
    public required long StartTimestampNanos { get; init; }

    /// <summary>
    /// End of history window (nanoseconds since epoch).
    /// </summary>
    public required long EndTimestampNanos { get; init; }

    /// <summary>
    /// Historical samples ordered by timestamp.
    /// </summary>
    public required IReadOnlyList<QueueDepthSample> Samples { get; init; }

    /// <summary>
    /// Trend direction (-1: decreasing, 0: stable, +1: increasing).
    /// </summary>
    public required int TrendDirection { get; init; }

    /// <summary>
    /// Predicted utilization in 1 minute based on trend.
    /// </summary>
    public required double PredictedUtilization1Min { get; init; }

    /// <summary>
    /// Duration of history window.
    /// </summary>
    public TimeSpan Duration =>
        TimeSpan.FromTicks((EndTimestampNanos - StartTimestampNanos) / 100);
}

/// <summary>
/// Single sample in queue depth history.
/// </summary>
public readonly record struct QueueDepthSample
{
    /// <summary>
    /// Timestamp of sample (nanoseconds since epoch).
    /// </summary>
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Queue utilization at this sample (0.0 - 1.0).
    /// </summary>
    public required double QueueUtilization { get; init; }

    /// <summary>
    /// Throughput at this sample (msgs/sec).
    /// </summary>
    public required double Throughput { get; init; }
}

/// <summary>
/// Alert triggered when queue depth exceeds threshold.
/// </summary>
public readonly record struct QueueDepthAlert
{
    /// <summary>
    /// Timestamp when alert was triggered.
    /// </summary>
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Silo that triggered the alert.
    /// </summary>
    public required string SiloId { get; init; }

    /// <summary>
    /// GPU device index.
    /// </summary>
    public required int DeviceIndex { get; init; }

    /// <summary>
    /// Current queue utilization.
    /// </summary>
    public required double CurrentUtilization { get; init; }

    /// <summary>
    /// Threshold that was exceeded.
    /// </summary>
    public required double Threshold { get; init; }

    /// <summary>
    /// Alert severity level.
    /// </summary>
    public required QueueAlertSeverity Severity { get; init; }

    /// <summary>
    /// Recommended action.
    /// </summary>
    public required string RecommendedAction { get; init; }
}

/// <summary>
/// Queue alert severity levels.
/// </summary>
public enum QueueAlertSeverity
{
    /// <summary>
    /// Informational: Queue utilization is elevated but acceptable.
    /// </summary>
    Info = 0,

    /// <summary>
    /// Warning: Queue utilization is high, consider load balancing.
    /// </summary>
    Warning = 1,

    /// <summary>
    /// Critical: Queue is near capacity, immediate action required.
    /// </summary>
    Critical = 2,

    /// <summary>
    /// Emergency: Queue overflow imminent, rejecting new grains.
    /// </summary>
    Emergency = 3
}
