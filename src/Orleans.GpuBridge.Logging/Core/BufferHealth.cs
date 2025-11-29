namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// Buffer health status.
/// </summary>
public sealed record BufferHealth
{
    /// <summary>
    /// Gets the current health status of the buffer.
    /// </summary>
    public BufferHealthStatus Status { get; init; }

    /// <summary>
    /// Gets the buffer utilization as a percentage (0-100).
    /// </summary>
    public double UtilizationPercent { get; init; }

    /// <summary>
    /// Gets the number of entries currently buffered.
    /// </summary>
    public int BufferedEntries { get; init; }

    /// <summary>
    /// Gets the maximum capacity of the buffer.
    /// </summary>
    public int Capacity { get; init; }

    /// <summary>
    /// Gets the buffer statistics including throughput and latency metrics.
    /// </summary>
    public BufferStatistics Statistics { get; init; } = new();
}

/// <summary>
/// Buffer health status enumeration.
/// </summary>
public enum BufferHealthStatus
{
    /// <summary>
    /// Buffer is operating normally with adequate capacity.
    /// </summary>
    Healthy,

    /// <summary>
    /// Buffer is approaching capacity limits.
    /// </summary>
    Warning,

    /// <summary>
    /// Buffer is at or near capacity; entries may be dropped.
    /// </summary>
    Critical
}
