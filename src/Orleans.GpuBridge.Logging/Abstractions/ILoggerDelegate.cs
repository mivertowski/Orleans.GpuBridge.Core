using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Logging.Abstractions;

/// <summary>
/// Core delegate interface for centralized logging operations.
/// Provides a flexible abstraction for different logging targets.
/// </summary>
public interface ILoggerDelegate
{
    /// <summary>
    /// Gets the name/identifier of this logger delegate.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the minimum log level this delegate will handle.
    /// </summary>
    LogLevel MinimumLevel { get; }

    /// <summary>
    /// Determines if this delegate is enabled for the specified log level.
    /// </summary>
    /// <param name="logLevel">The log level to check</param>
    /// <returns>true if enabled, false otherwise</returns>
    bool IsEnabled(LogLevel logLevel);

    /// <summary>
    /// Writes a log entry asynchronously.
    /// </summary>
    /// <param name="entry">The log entry to write</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task WriteAsync(LogEntry entry, CancellationToken cancellationToken = default);

    /// <summary>
    /// Flushes any pending log entries.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    Task FlushAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs cleanup operations.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    ValueTask DisposeAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Extended logger delegate interface with batch processing capabilities.
/// </summary>
public interface IBatchLoggerDelegate : ILoggerDelegate
{
    /// <summary>
    /// Writes multiple log entries as a batch.
    /// </summary>
    /// <param name="entries">Collection of log entries</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task WriteBatchAsync(IReadOnlyCollection<LogEntry> entries, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the maximum batch size supported.
    /// </summary>
    int MaxBatchSize { get; }
}

/// <summary>
/// Structured logger delegate interface with context enrichment.
/// </summary>
public interface IStructuredLoggerDelegate : ILoggerDelegate
{
    /// <summary>
    /// Enriches log entries with additional context.
    /// </summary>
    /// <param name="entry">Log entry to enrich</param>
    /// <param name="context">Additional context</param>
    /// <returns>Enriched log entry</returns>
    LogEntry EnrichEntry(LogEntry entry, LogContext? context = null);
}