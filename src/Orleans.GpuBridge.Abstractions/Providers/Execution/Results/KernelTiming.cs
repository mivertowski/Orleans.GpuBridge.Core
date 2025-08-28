using System;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Results;

/// <summary>
/// Kernel timing information
/// </summary>
public sealed record KernelTiming(
    TimeSpan QueueTime,
    TimeSpan KernelTime,
    TimeSpan TotalTime,
    long BytesTransferred = 0,
    double GFlops = 0);