namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Memory pool health status information.
/// </summary>
public sealed class MemoryPoolHealth
{
    /// <summary>
    /// Gets the current health status of the memory pool.
    /// </summary>
    public HealthStatus Status { get; init; }

    /// <summary>
    /// Gets the total memory usage in bytes across all pools.
    /// </summary>
    public long TotalUsageBytes { get; init; }

    /// <summary>
    /// Gets the total memory limit in bytes.
    /// </summary>
    public long TotalLimitBytes { get; init; }

    /// <summary>
    /// Gets the memory utilization as a percentage (0-100).
    /// </summary>
    public double UtilizationPercent { get; init; }

    /// <summary>
    /// Gets the number of active memory pools.
    /// </summary>
    public int PoolCount { get; init; }

    /// <summary>
    /// Gets a human-readable message describing the health status.
    /// </summary>
    public string Message { get; init; } = string.Empty;
}

/// <summary>
/// Health status levels for memory pool monitoring.
/// </summary>
public enum HealthStatus
{
    /// <summary>
    /// Memory pool is operating normally with adequate capacity.
    /// </summary>
    Healthy,

    /// <summary>
    /// Memory pool usage is elevated and approaching limits.
    /// </summary>
    Warning,

    /// <summary>
    /// Memory pool is critically low on available capacity.
    /// </summary>
    Critical
}
