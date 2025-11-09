namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Memory pool health status
/// </summary>
public sealed class MemoryPoolHealth
{
    public HealthStatus Status { get; init; }
    public long TotalUsageBytes { get; init; }
    public long TotalLimitBytes { get; init; }
    public double UtilizationPercent { get; init; }
    public int PoolCount { get; init; }
    public string Message { get; init; } = string.Empty;
}

public enum HealthStatus
{
    Healthy,
    Warning,
    Critical
}
