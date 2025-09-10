namespace Orleans.GpuBridge.Abstractions.Domain.ValueObjects;

/// <summary>
/// Performance metrics for device monitoring
/// </summary>
public sealed record PerformanceMetrics(
    double UtilizationPercent,
    long MemoryUsedBytes,
    double PowerUsageWatts,
    int ClockSpeedMHz,
    int MemoryClockMHz)
{
    /// <summary>
    /// Memory utilization percentage
    /// </summary>
    public double MemoryUtilizationPercent { get; init; }

    /// <summary>
    /// Power efficiency metric (operations per watt estimate)
    /// </summary>
    public double PowerEfficiency => PowerUsageWatts > 0 
        ? UtilizationPercent / PowerUsageWatts 
        : 0.0;

    /// <summary>
    /// Overall device health score (0.0 to 1.0)
    /// </summary>
    public double HealthScore => Math.Max(0.0, Math.Min(1.0, 
        (1.0 - Math.Max(0, UtilizationPercent - 80) / 20.0) * // Penalize high utilization
        (PowerUsageWatts > 0 ? Math.Min(1.0, 100.0 / PowerUsageWatts) : 1.0))); // Reward efficiency
}