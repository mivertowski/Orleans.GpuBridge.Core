using System;

namespace Orleans.GpuBridge.Resilience.Policies;

/// <summary>
/// Chaos engineering configuration options
/// </summary>
public sealed class ChaosEngineeringOptions
{
    /// <summary>
    /// Whether to enable chaos engineering features
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Probability of injecting faults (0.0 to 1.0)
    /// </summary>
    public double FaultInjectionProbability { get; set; } = 0.01;

    /// <summary>
    /// Latency injection configuration
    /// </summary>
    public LatencyInjectionOptions LatencyInjection { get; set; } = new();

    /// <summary>
    /// Exception injection configuration
    /// </summary>
    public ExceptionInjectionOptions ExceptionInjection { get; set; } = new();

    /// <summary>
    /// Resource exhaustion simulation configuration
    /// </summary>
    public ResourceExhaustionOptions ResourceExhaustion { get; set; } = new();
}
