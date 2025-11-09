namespace Orleans.GpuBridge.Resilience;

/// <summary>
/// Health status enumeration
/// </summary>
public enum HealthStatus
{
    Unknown = 0,
    Healthy = 1,
    Degraded = 2,
    Unhealthy = 3
}
