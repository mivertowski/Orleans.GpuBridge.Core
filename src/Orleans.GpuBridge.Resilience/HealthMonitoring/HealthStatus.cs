namespace Orleans.GpuBridge.Resilience;

/// <summary>
/// Health status enumeration for GPU components.
/// </summary>
public enum HealthStatus
{
    /// <summary>Health status is not yet determined.</summary>
    Unknown = 0,
    /// <summary>Component is functioning normally.</summary>
    Healthy = 1,
    /// <summary>Component is functioning with reduced capacity or performance.</summary>
    Degraded = 2,
    /// <summary>Component is not functioning properly.</summary>
    Unhealthy = 3
}
