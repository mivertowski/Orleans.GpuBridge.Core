using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Logging.Configuration;

/// <summary>
/// Configuration options for Orleans GPU Bridge logging system.
/// </summary>
public sealed class GpuBridgeLoggingOptions
{
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
    public Dictionary<string, LogLevel> CategoryLevels { get; set; } = new();
}

/// <summary>
/// Buffer configuration options.
/// </summary>
public sealed class BufferConfiguration
{
    /// <summary>
    /// Buffer capacity for log entries.
    /// </summary>
    public int Capacity { get; set; } = 10000;

    /// <summary>
    /// Maximum batch size for processing.
    /// </summary>
    public int MaxBatchSize { get; set; } = 100;

    /// <summary>
    /// Flush interval for batched processing.
    /// </summary>
    public TimeSpan FlushInterval { get; set; } = TimeSpan.FromSeconds(1);

    /// <summary>
    /// Whether to drop entries when buffer is full.
    /// </summary>
    public bool DropOnFull { get; set; } = true;

    /// <summary>
    /// Whether to prioritize high-severity entries.
    /// </summary>
    public bool PrioritizeHighSeverity { get; set; } = true;
}

/// <summary>
/// Delegate manager configuration options.
/// </summary>
public sealed class DelegateManagerConfiguration
{
    /// <summary>
    /// Maximum queue size for delegate manager.
    /// </summary>
    public int MaxQueueSize { get; set; } = 10000;

    /// <summary>
    /// Processing interval for delegate manager.
    /// </summary>
    public TimeSpan ProcessingInterval { get; set; } = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Whether to drop entries when queue is full.
    /// </summary>
    public bool DropOnQueueFull { get; set; } = true;
}

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

/// <summary>
/// File logging configuration.
/// </summary>
public sealed class FileConfiguration
{
    /// <summary>
    /// Whether file logging is enabled.
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Directory where log files will be stored.
    /// </summary>
    public string LogDirectory { get; set; } = "logs";

    /// <summary>
    /// Base name for log files.
    /// </summary>
    public string BaseFileName { get; set; } = "gpu-bridge";

    /// <summary>
    /// Minimum log level for file output.
    /// </summary>
    public LogLevel MinimumLevel { get; set; } = LogLevel.Debug;

    /// <summary>
    /// Maximum file size in bytes before rotation.
    /// </summary>
    public long? MaxFileSizeBytes { get; set; } = 100 * 1024 * 1024; // 100MB

    /// <summary>
    /// Time interval for file rotation.
    /// </summary>
    public TimeSpan? RotationInterval { get; set; } = TimeSpan.FromDays(1);

    /// <summary>
    /// Number of days to retain log files.
    /// </summary>
    public int? RetentionDays { get; set; } = 30;

    /// <summary>
    /// Maximum number of log files to retain.
    /// </summary>
    public int? MaxRetainedFiles { get; set; } = 100;

    /// <summary>
    /// Whether to flush after each write.
    /// </summary>
    public bool AutoFlush { get; set; } = false;

    /// <summary>
    /// Buffer size for file operations.
    /// </summary>
    public int BufferSize { get; set; } = 64 * 1024; // 64KB

    /// <summary>
    /// Maximum batch size for batch writes.
    /// </summary>
    public int MaxBatchSize { get; set; } = 1000;

    /// <summary>
    /// Timestamp format for file names.
    /// </summary>
    public string FileNameTimestampFormat { get; set; } = "yyyyMMdd_HHmmss";

    /// <summary>
    /// Whether to include context information.
    /// </summary>
    public bool IncludeContext { get; set; } = true;
}

/// <summary>
/// Telemetry logging configuration.
/// </summary>
public sealed class TelemetryConfiguration
{
    /// <summary>
    /// Whether telemetry logging is enabled.
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Minimum log level for telemetry.
    /// </summary>
    public LogLevel MinimumLevel { get; set; } = LogLevel.Information;

    /// <summary>
    /// Service name for telemetry.
    /// </summary>
    public string ServiceName { get; set; } = "Orleans.GpuBridge";

    /// <summary>
    /// Service version for telemetry.
    /// </summary>
    public string ServiceVersion { get; set; } = "1.0.0";

    /// <summary>
    /// Service instance identifier.
    /// </summary>
    public string ServiceInstance { get; set; } = Environment.MachineName;

    /// <summary>
    /// Maximum batch size for telemetry processing.
    /// </summary>
    public int MaxBatchSize { get; set; } = 100;

    /// <summary>
    /// Interval for flushing telemetry data.
    /// </summary>
    public TimeSpan FlushInterval { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Whether to flush immediately on error.
    /// </summary>
    public bool FlushOnError { get; set; } = true;

    /// <summary>
    /// Whether to create activities for all log levels.
    /// </summary>
    public bool TraceAllLevels { get; set; } = false;

    /// <summary>
    /// OTLP endpoint for telemetry export.
    /// </summary>
    public string? OtlpEndpoint { get; set; }

    /// <summary>
    /// API key for telemetry service.
    /// </summary>
    public string? ApiKey { get; set; }

    /// <summary>
    /// Custom headers for telemetry requests.
    /// </summary>
    public Dictionary<string, string> CustomHeaders { get; set; } = new();

    /// <summary>
    /// Whether to include detailed resource information.
    /// </summary>
    public bool IncludeResourceInfo { get; set; } = true;

    /// <summary>
    /// Whether to sample telemetry data.
    /// </summary>
    public bool EnableSampling { get; set; } = false;

    /// <summary>
    /// Sampling rate (0.0 to 1.0).
    /// </summary>
    public double SamplingRate { get; set; } = 0.1;
}

/// <summary>
/// Configuration extensions for Orleans GPU Bridge logging.
/// </summary>
public static class LoggingConfigurationExtensions
{
    /// <summary>
    /// Binds configuration to GpuBridgeLoggingOptions.
    /// </summary>
    public static GpuBridgeLoggingOptions BindGpuBridgeLogging(this IConfiguration configuration)
    {
        var options = new GpuBridgeLoggingOptions();
        configuration.GetSection(GpuBridgeLoggingOptions.SectionName).Bind(options);
        return options;
    }

    /// <summary>
    /// Gets a specific category log level from configuration.
    /// </summary>
    public static LogLevel GetCategoryLevel(this GpuBridgeLoggingOptions options, string categoryName)
    {
        // Check for exact match first
        if (options.CategoryLevels.TryGetValue(categoryName, out var level))
        {
            return level;
        }

        // Check for parent namespace matches
        var parts = categoryName.Split('.');
        for (int i = parts.Length - 1; i > 0; i--)
        {
            var parentCategory = string.Join('.', parts.Take(i));
            if (options.CategoryLevels.TryGetValue(parentCategory, out level))
            {
                return level;
            }
        }

        return options.MinimumLevel;
    }

    /// <summary>
    /// Validates configuration options.
    /// </summary>
    public static ValidationResult ValidateConfiguration(this GpuBridgeLoggingOptions options)
    {
        var errors = new List<string>();

        if (options.Buffer.Capacity <= 0)
            errors.Add("Buffer capacity must be greater than 0");

        if (options.Buffer.MaxBatchSize <= 0)
            errors.Add("Buffer max batch size must be greater than 0");

        if (options.DelegateManager.MaxQueueSize <= 0)
            errors.Add("Delegate manager max queue size must be greater than 0");

        if (options.File.Enabled)
        {
            if (string.IsNullOrWhiteSpace(options.File.LogDirectory))
                errors.Add("File log directory must be specified when file logging is enabled");

            if (string.IsNullOrWhiteSpace(options.File.BaseFileName))
                errors.Add("File base name must be specified when file logging is enabled");
        }

        if (options.Telemetry.Enabled)
        {
            if (string.IsNullOrWhiteSpace(options.Telemetry.ServiceName))
                errors.Add("Service name must be specified when telemetry is enabled");

            if (options.Telemetry.SamplingRate is < 0 or > 1)
                errors.Add("Telemetry sampling rate must be between 0 and 1");
        }

        return new ValidationResult(errors.Count == 0, errors);
    }
}

/// <summary>
/// Configuration validation result.
/// </summary>
public sealed record ValidationResult(bool IsValid, IReadOnlyList<string> Errors);