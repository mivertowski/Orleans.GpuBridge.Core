namespace Orleans.GpuBridge.Logging.Configuration;

/// <summary>
/// Configuration validation result.
/// </summary>
public sealed record ValidationResult(bool IsValid, IReadOnlyList<string> Errors);
