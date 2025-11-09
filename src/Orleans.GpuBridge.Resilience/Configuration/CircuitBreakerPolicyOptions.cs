using System;

namespace Orleans.GpuBridge.Resilience.Policies;

/// <summary>
/// Circuit breaker policy configuration options
/// </summary>
public sealed class CircuitBreakerPolicyOptions
{
    /// <summary>
    /// Failure ratio threshold to open circuit (0.0 to 1.0)
    /// </summary>
    public double FailureRatio { get; set; } = 0.5;

    /// <summary>
    /// Sampling duration for failure rate calculation
    /// </summary>
    public TimeSpan SamplingDuration { get; set; } = TimeSpan.FromMinutes(2);

    /// <summary>
    /// Minimum throughput required before opening circuit
    /// </summary>
    public int MinimumThroughput { get; set; } = 10;

    /// <summary>
    /// Duration to keep circuit open
    /// </summary>
    public TimeSpan BreakDuration { get; set; } = TimeSpan.FromMinutes(1);

    /// <summary>
    /// Whether to enable circuit breaker
    /// </summary>
    public bool Enabled { get; set; } = true;
}
