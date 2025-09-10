using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Results;

/// <summary>
/// Graph validation result
/// </summary>
public sealed record GraphValidationResult(
    bool IsValid,
    IReadOnlyList<string>? Errors = null,
    IReadOnlyList<string>? Warnings = null,
    bool HasCycles = false)
{
    /// <summary>
    /// Creates a successful validation result
    /// </summary>
    public static GraphValidationResult Success(IReadOnlyList<string>? warnings = null, bool hasCycles = false) =>
        new(IsValid: true, Warnings: warnings, HasCycles: hasCycles);
        
    /// <summary>
    /// Creates an error validation result
    /// </summary>
    public static GraphValidationResult Error(string error, IReadOnlyList<string>? warnings = null, bool hasCycles = false) =>
        new(IsValid: false, Errors: new[] { error }, Warnings: warnings, HasCycles: hasCycles);
        
    /// <summary>
    /// Creates an error validation result with multiple errors
    /// </summary>
    public static GraphValidationResult Error(IReadOnlyList<string> errors, IReadOnlyList<string>? warnings = null, bool hasCycles = false) =>
        new(IsValid: false, Errors: errors, Warnings: warnings, HasCycles: hasCycles);
}