using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Results;

/// <summary>
/// Graph validation result
/// </summary>
public sealed record GraphValidationResult(
    bool IsValid,
    IReadOnlyList<string>? Errors = null,
    IReadOnlyList<string>? Warnings = null,
    bool HasCycles = false);