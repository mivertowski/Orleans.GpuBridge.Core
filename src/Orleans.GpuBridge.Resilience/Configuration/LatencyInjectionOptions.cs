using System;

namespace Orleans.GpuBridge.Resilience.Policies;

/// <summary>
/// Latency injection configuration
/// </summary>
public sealed class LatencyInjectionOptions
{
    /// <summary>
    /// Whether to enable latency injection
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Minimum latency to inject
    /// </summary>
    public TimeSpan MinLatency { get; set; } = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Maximum latency to inject
    /// </summary>
    public TimeSpan MaxLatency { get; set; } = TimeSpan.FromSeconds(5);

    /// <summary>
    /// Probability of injecting latency (0.0 to 1.0)
    /// </summary>
    public double InjectionProbability { get; set; } = 0.05;
}
