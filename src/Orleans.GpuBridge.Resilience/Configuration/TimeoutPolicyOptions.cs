using System;

namespace Orleans.GpuBridge.Resilience.Policies;

/// <summary>
/// Timeout policy configuration options
/// </summary>
public sealed class TimeoutPolicyOptions
{
    /// <summary>
    /// Timeout for kernel execution operations
    /// </summary>
    public TimeSpan KernelExecution { get; set; } = TimeSpan.FromMinutes(5);

    /// <summary>
    /// Timeout for device operations
    /// </summary>
    public TimeSpan DeviceOperation { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Timeout for memory allocation operations
    /// </summary>
    public TimeSpan MemoryAllocation { get; set; } = TimeSpan.FromSeconds(10);

    /// <summary>
    /// Timeout for kernel compilation operations
    /// </summary>
    public TimeSpan KernelCompilation { get; set; } = TimeSpan.FromMinutes(10);

    /// <summary>
    /// Timeout for data transfer operations
    /// </summary>
    public TimeSpan DataTransfer { get; set; } = TimeSpan.FromMinutes(2);

    /// <summary>
    /// Default timeout for unspecified operations
    /// </summary>
    public TimeSpan DefaultOperation { get; set; } = TimeSpan.FromMinutes(1);
}
