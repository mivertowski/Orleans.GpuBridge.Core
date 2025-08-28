using System;
using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Results.Statistics;

/// <summary>
/// Execution statistics
/// </summary>
public sealed record ExecutionStatistics(
    long TotalKernelsExecuted,
    long TotalBatchesExecuted,
    long TotalGraphsExecuted,
    TimeSpan TotalExecutionTime,
    TimeSpan AverageKernelTime,
    long TotalBytesTransferred,
    long TotalErrors,
    IReadOnlyDictionary<string, long> KernelExecutionCounts);