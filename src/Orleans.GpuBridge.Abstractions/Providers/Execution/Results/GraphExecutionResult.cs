using System;
using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Results;

/// <summary>
/// Result of graph execution
/// </summary>
public sealed record GraphExecutionResult(
    bool Success,
    int NodesExecuted,
    TimeSpan ExecutionTime,
    IReadOnlyDictionary<string, KernelTiming>? NodeTimings = null);