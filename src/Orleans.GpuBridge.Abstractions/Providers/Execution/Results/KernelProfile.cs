using System;
using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Results;

/// <summary>
/// Kernel execution profile
/// </summary>
public sealed record KernelProfile(
    TimeSpan AverageExecutionTime,
    TimeSpan MinExecutionTime,
    TimeSpan MaxExecutionTime,
    double StandardDeviation,
    long MemoryBandwidthBytesPerSecond,
    double ComputeThroughputGFlops,
    int OptimalBlockSize,
    IReadOnlyDictionary<string, object>? ExtendedMetrics = null);