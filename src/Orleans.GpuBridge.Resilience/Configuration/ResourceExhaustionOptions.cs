using System;

namespace Orleans.GpuBridge.Resilience.Policies;

/// <summary>
/// Resource exhaustion simulation configuration
/// </summary>
public sealed class ResourceExhaustionOptions
{
    /// <summary>
    /// Whether to enable resource exhaustion simulation
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Probability of simulating memory exhaustion (0.0 to 1.0)
    /// </summary>
    public double MemoryExhaustionProbability { get; set; } = 0.01;

    /// <summary>
    /// Probability of simulating compute exhaustion (0.0 to 1.0)
    /// </summary>
    public double ComputeExhaustionProbability { get; set; } = 0.01;

    /// <summary>
    /// Duration to simulate resource exhaustion
    /// </summary>
    public TimeSpan ExhaustionDuration { get; set; } = TimeSpan.FromSeconds(30);
}
