using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Logging.Configuration;

/// <summary>
/// Console logging configuration.
/// </summary>
public sealed class ConsoleConfiguration
{
    /// <summary>
    /// Whether console logging is enabled.
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Minimum log level for console output.
    /// </summary>
    public LogLevel MinimumLevel { get; set; } = LogLevel.Information;

    /// <summary>
    /// Whether to use console colors.
    /// </summary>
    public bool UseColors { get; set; } = true;

    /// <summary>
    /// Timestamp format for console output.
    /// </summary>
    public string TimestampFormat { get; set; } = "Local";

    /// <summary>
    /// Whether to include the logger category.
    /// </summary>
    public bool IncludeCategory { get; set; } = true;

    /// <summary>
    /// Whether to include correlation IDs.
    /// </summary>
    public bool IncludeCorrelationId { get; set; } = true;

    /// <summary>
    /// Whether to include structured properties.
    /// </summary>
    public bool IncludeProperties { get; set; } = true;

    /// <summary>
    /// Whether to include logging scopes.
    /// </summary>
    public bool IncludeScopes { get; set; } = false;

    /// <summary>
    /// Whether to include thread ID.
    /// </summary>
    public bool IncludeThreadId { get; set; } = false;

    /// <summary>
    /// Whether to include context information.
    /// </summary>
    public bool IncludeContext { get; set; } = true;

    /// <summary>
    /// Whether to include performance metrics.
    /// </summary>
    public bool IncludeMetrics { get; set; } = true;
}
