namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// Buffer health status.
/// </summary>
public sealed record BufferHealth
{
    public BufferHealthStatus Status { get; init; }
    public double UtilizationPercent { get; init; }
    public int BufferedEntries { get; init; }
    public int Capacity { get; init; }
    public BufferStatistics Statistics { get; init; } = new();
}

/// <summary>
/// Buffer health status enumeration.
/// </summary>
public enum BufferHealthStatus
{
    Healthy,
    Warning,
    Critical
}
