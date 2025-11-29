namespace Orleans.GpuBridge.Abstractions.Domain.ValueObjects;

/// <summary>
/// Thermal information for device monitoring
/// </summary>
public sealed record ThermalInfo(
    int TemperatureCelsius,
    int MaxTemperatureCelsius,
    int ThrottleTemperatureCelsius,
    bool IsThrottling)
{
    /// <summary>
    /// Temperature utilization as percentage of max safe temperature
    /// </summary>
    public double TemperatureUtilization => MaxTemperatureCelsius > 0
        ? Math.Min(1.0, TemperatureCelsius / (double)MaxTemperatureCelsius)
        : 0.0;

    /// <summary>
    /// Whether the device is approaching thermal limits
    /// </summary>
    public bool IsNearThermalLimit => TemperatureCelsius >= (ThrottleTemperatureCelsius * 0.9);
}