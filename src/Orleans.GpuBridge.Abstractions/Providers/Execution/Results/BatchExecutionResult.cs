using System;
using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Results;

/// <summary>
/// Result of batch execution
/// </summary>
public sealed record BatchExecutionResult(
    int SuccessCount,
    int FailureCount,
    IReadOnlyList<KernelExecutionResult> Results,
    TimeSpan TotalExecutionTime);