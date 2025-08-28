using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Results;

/// <summary>
/// Result of kernel execution
/// </summary>
public sealed record KernelExecutionResult(
    bool Success,
    string? ErrorMessage = null,
    KernelTiming? Timing = null,
    IReadOnlyDictionary<string, object>? Metadata = null);