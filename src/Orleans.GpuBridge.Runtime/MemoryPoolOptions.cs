using System;
using System.Collections.Generic;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Configuration options for memory pools.
/// </summary>
public sealed class MemoryPoolOptions
{
    /// <summary>
    /// Gets or sets the maximum total memory in bytes across all pools.
    /// Default is 4GB.
    /// </summary>
    public long MaxTotalMemoryBytes { get; set; } = 4L * 1024 * 1024 * 1024; // 4GB default

    /// <summary>
    /// Gets or sets the default memory limit per type in bytes.
    /// Default is 512MB.
    /// </summary>
    public long DefaultPerTypeLimit { get; set; } = 512L * 1024 * 1024; // 512MB per type

    /// <summary>
    /// Gets the per-type memory limits for specific element types.
    /// </summary>
    public Dictionary<Type, long> PerTypeMemoryLimits { get; } = new();

    /// <summary>
    /// Gets or sets the maximum number of pooled buffers per type.
    /// Default is 100.
    /// </summary>
    public int MaxPooledBuffersPerType { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to enable memory pressure monitoring.
    /// Default is true.
    /// </summary>
    public bool EnableMemoryPressureMonitoring { get; set; } = true;

    /// <summary>
    /// Gets or sets the memory pressure threshold percentage (0-100).
    /// Default is 80%.
    /// </summary>
    public double MemoryPressureThresholdPercent { get; set; } = 80.0;
}
