using System;

namespace Orleans.GpuBridge.Resilience.Policies;

/// <summary>
/// Exception injection configuration
/// </summary>
public sealed class ExceptionInjectionOptions
{
    /// <summary>
    /// Whether to enable exception injection
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Probability of injecting exceptions (0.0 to 1.0)
    /// </summary>
    public double InjectionProbability { get; set; } = 0.02;

    /// <summary>
    /// Types of exceptions to inject
    /// </summary>
    public string[] ExceptionTypes { get; set; } =
    {
        "Orleans.GpuBridge.Abstractions.Exceptions.GpuOperationException",
        "Orleans.GpuBridge.Abstractions.Exceptions.GpuMemoryException",
        "System.TimeoutException"
    };
}
