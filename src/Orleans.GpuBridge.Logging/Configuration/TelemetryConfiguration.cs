using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Logging.Configuration;

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
