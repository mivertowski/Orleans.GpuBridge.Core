using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// Configuration options for the logger factory.
/// </summary>
public sealed class LoggerFactoryOptions
{
    /// <summary>
    /// Default minimum log level for new loggers.
    /// </summary>
    public LogLevel DefaultMinimumLevel { get; set; } = LogLevel.Information;

    /// <summary>
    /// Buffer capacity for log entries.
    /// </summary>
    public int BufferCapacity { get; set; } = 10000;

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
    public bool DropOnBufferFull { get; set; } = true;

    /// <summary>
    /// Maximum queue size for delegate manager.
    /// </summary>
    public int MaxQueueSize { get; set; } = 10000;

    /// <summary>
    /// Processing interval for delegate manager.
    /// </summary>
    public TimeSpan ProcessingInterval { get; set; } = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Whether to drop entries when delegate queue is full.
    /// </summary>
    public bool DropOnQueueFull { get; set; } = true;
}
