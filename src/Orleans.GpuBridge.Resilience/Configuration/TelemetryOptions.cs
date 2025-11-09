using System;

namespace Orleans.GpuBridge.Resilience;

/// <summary>
/// Telemetry configuration options
/// </summary>
public sealed class TelemetryOptions
{
    /// <summary>
    /// Metrics retention period
    /// </summary>
    public TimeSpan RetentionPeriod { get; set; } = TimeSpan.FromHours(24);

    /// <summary>
    /// Health check interval
    /// </summary>
    public TimeSpan HealthCheckInterval { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Whether to enable detailed tracing
    /// </summary>
    public bool EnableDetailedTracing { get; set; } = false;
}
