using System;
using System.Collections.Generic;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Configuration options for memory pools
/// </summary>
public sealed class MemoryPoolOptions
{
    public long MaxTotalMemoryBytes { get; set; } = 4L * 1024 * 1024 * 1024; // 4GB default
    public long DefaultPerTypeLimit { get; set; } = 512L * 1024 * 1024; // 512MB per type
    public Dictionary<Type, long> PerTypeMemoryLimits { get; } = new();
    public int MaxPooledBuffersPerType { get; set; } = 100;
    public bool EnableMemoryPressureMonitoring { get; set; } = true;
    public double MemoryPressureThresholdPercent { get; set; } = 80.0;
}
