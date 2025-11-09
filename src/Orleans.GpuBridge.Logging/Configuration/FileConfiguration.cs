using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Logging.Configuration;

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
