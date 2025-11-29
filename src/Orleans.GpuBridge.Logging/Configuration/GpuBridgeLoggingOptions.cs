using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Logging.Configuration;

/// <summary>
/// Configuration options for Orleans GPU Bridge logging system.
/// </summary>
public sealed class GpuBridgeLoggingOptions
{
    /// <summary>
    /// The configuration section name for GPU Bridge logging options.
    /// </summary>
    public const string SectionName = "GpuBridgeLogging";

    /// <summary>
    /// Global minimum log level.
    /// </summary>
    public LogLevel MinimumLevel { get; set; } = LogLevel.Information;

    /// <summary>
    /// Whether to enable structured logging.
    /// </summary>
    public bool EnableStructuredLogging { get; set; } = true;

    /// <summary>
    /// Whether to enable performance metrics logging.
    /// </summary>
    public bool EnablePerformanceMetrics { get; set; } = true;

    /// <summary>
    /// Whether to enable correlation ID tracking.
    /// </summary>
    public bool EnableCorrelationTracking { get; set; } = true;

    /// <summary>
    /// Buffer configuration.
    /// </summary>
    public BufferConfiguration Buffer { get; set; } = new();

    /// <summary>
    /// Delegate manager configuration.
    /// </summary>
    public DelegateManagerConfiguration DelegateManager { get; set; } = new();

    /// <summary>
    /// Console logging configuration.
    /// </summary>
    public ConsoleConfiguration Console { get; set; } = new();

    /// <summary>
    /// File logging configuration.
    /// </summary>
    public FileConfiguration File { get; set; } = new();

    /// <summary>
    /// Telemetry logging configuration.
    /// </summary>
    public TelemetryConfiguration Telemetry { get; set; } = new();

    /// <summary>
    /// Category-specific log levels.
    /// </summary>
    public Dictionary<string, LogLevel> CategoryLevels { get; set; } = [];
}
