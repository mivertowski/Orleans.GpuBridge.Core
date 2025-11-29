using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Logging.Abstractions;
using System.Text.Json;

namespace Orleans.GpuBridge.Logging.Delegates;

/// <summary>
/// Logger delegate that writes to the console with color-coded output.
/// </summary>
public sealed class ConsoleLoggerDelegate : IStructuredLoggerDelegate
{
    private readonly ConsoleLoggerOptions _options;
    private readonly object _lock = new();

    /// <summary>
    /// Gets the delegate name.
    /// </summary>
    public string Name => "Console";

    /// <summary>
    /// Gets the minimum log level that will be processed.
    /// </summary>
    public LogLevel MinimumLevel { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ConsoleLoggerDelegate"/> class.
    /// </summary>
    /// <param name="options">Configuration options for console logging.</param>
    public ConsoleLoggerDelegate(ConsoleLoggerOptions? options = null)
    {
        _options = options ?? new ConsoleLoggerOptions();
        MinimumLevel = _options.MinimumLevel;
    }

    /// <summary>
    /// Determines whether logging is enabled for the specified log level.
    /// </summary>
    /// <param name="logLevel">The log level to check.</param>
    /// <returns>true if logging is enabled for the level; otherwise, false.</returns>
    public bool IsEnabled(LogLevel logLevel) => logLevel >= MinimumLevel;

    /// <summary>
    /// Writes a log entry to the console.
    /// </summary>
    /// <param name="entry">The log entry to write.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    public Task WriteAsync(LogEntry entry, CancellationToken cancellationToken = default)
    {
        if (!IsEnabled(entry.Level) || cancellationToken.IsCancellationRequested)
            return Task.CompletedTask;

        var enrichedEntry = EnrichEntry(entry);
        var formattedMessage = FormatLogEntry(enrichedEntry);

        lock (_lock)
        {
            WriteToConsoleWithColor(enrichedEntry.Level, formattedMessage);
        }

        return Task.CompletedTask;
    }

    /// <summary>
    /// Enriches a log entry with additional context information.
    /// </summary>
    /// <param name="entry">The log entry to enrich.</param>
    /// <param name="context">Optional log context to use for enrichment.</param>
    /// <returns>The enriched log entry.</returns>
    public LogEntry EnrichEntry(LogEntry entry, LogContext? context = null)
    {
        var enrichedProperties = new Dictionary<string, object?>(entry.Properties);

        // Add context properties if available
        context ??= LogContext.Current;
        if (context != null && _options.IncludeContext)
        {
            enrichedProperties["Machine"] = context.MachineName;
            enrichedProperties["Process"] = context.ProcessId;
            if (context.UserId != null) enrichedProperties["User"] = context.UserId;
            if (context.Component != null) enrichedProperties["Component"] = context.Component;
        }

        // Add thread information if requested
        if (_options.IncludeThreadId)
        {
            enrichedProperties["ThreadId"] = entry.ThreadId;
        }

        // Add performance metrics if available
        if (entry.Metrics != null && _options.IncludeMetrics)
        {
            if (entry.Metrics.Duration.HasValue)
                enrichedProperties["Duration"] = $"{entry.Metrics.Duration.Value.TotalMilliseconds:F2}ms";

            if (entry.Metrics.MemoryUsage.HasValue)
                enrichedProperties["Memory"] = $"{entry.Metrics.MemoryUsage.Value / 1024.0:F1}KB";

            foreach (var counter in entry.Metrics.Counters)
            {
                enrichedProperties[$"Metric.{counter.Key}"] = counter.Value;
            }
        }

        return entry.WithProperties(enrichedProperties);
    }

    /// <summary>
    /// Flushes any buffered log entries.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    public Task FlushAsync(CancellationToken cancellationToken = default)
    {
        // Console output is immediately flushed
        return Task.CompletedTask;
    }

    /// <summary>
    /// Disposes the console logger delegate asynchronously.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A value task representing the asynchronous operation.</returns>
    public ValueTask DisposeAsync(CancellationToken cancellationToken = default)
    {
        // Nothing to dispose for console output
        return ValueTask.CompletedTask;
    }

    private string FormatLogEntry(LogEntry entry)
    {
        var timestamp = _options.TimestampFormat switch
        {
            ConsoleTimestampFormat.None => "",
            ConsoleTimestampFormat.Local => $"{entry.Timestamp.ToLocalTime():HH:mm:ss.fff} ",
            ConsoleTimestampFormat.Utc => $"{entry.Timestamp:HH:mm:ss.fff} ",
            ConsoleTimestampFormat.Full => $"{entry.Timestamp:yyyy-MM-dd HH:mm:ss.fff} ",
            _ => ""
        };

        var levelText = GetLevelText(entry.Level);
        var category = _options.IncludeCategory ? $"[{entry.Category}] " : "";
        var correlationId = !string.IsNullOrEmpty(entry.CorrelationId) && _options.IncludeCorrelationId
            ? $"({entry.CorrelationId[..8]}) "
            : "";

        var message = $"{timestamp}{levelText} {category}{correlationId}{entry.Message}";

        // Add exception details
        if (entry.Exception != null)
        {
            message += Environment.NewLine + FormatException(entry.Exception);
        }

        // Add structured properties
        if (entry.Properties.Count > 0 && _options.IncludeProperties)
        {
            var properties = string.Join(", ",
                entry.Properties.Select(kvp => $"{kvp.Key}={FormatValue(kvp.Value)}"));
            message += Environment.NewLine + $"Properties: {properties}";
        }

        // Add scopes
        if (entry.Scopes.Count > 0 && _options.IncludeScopes)
        {
            foreach (var scope in entry.Scopes)
            {
                message += Environment.NewLine + $"Scope[{scope.Name}]: " +
                    string.Join(", ", scope.Properties.Select(kvp => $"{kvp.Key}={FormatValue(kvp.Value)}"));
            }
        }

        return message;
    }

    private void WriteToConsoleWithColor(LogLevel level, string message)
    {
        if (!_options.UseColors)
        {
            Console.WriteLine(message);
            return;
        }

        var originalColor = Console.ForegroundColor;

        try
        {
            Console.ForegroundColor = GetLevelColor(level);
            Console.WriteLine(message);
        }
        finally
        {
            Console.ForegroundColor = originalColor;
        }
    }

    private static string GetLevelText(LogLevel level) => level switch
    {
        LogLevel.Trace => "TRCE",
        LogLevel.Debug => "DBUG",
        LogLevel.Information => "INFO",
        LogLevel.Warning => "WARN",
        LogLevel.Error => "ERRO",
        LogLevel.Critical => "CRIT",
        _ => "UNKN"
    };

    private static ConsoleColor GetLevelColor(LogLevel level) => level switch
    {
        LogLevel.Trace => ConsoleColor.DarkGray,
        LogLevel.Debug => ConsoleColor.Gray,
        LogLevel.Information => ConsoleColor.White,
        LogLevel.Warning => ConsoleColor.Yellow,
        LogLevel.Error => ConsoleColor.Red,
        LogLevel.Critical => ConsoleColor.Magenta,
        _ => ConsoleColor.White
    };

    private static string FormatException(Exception exception)
    {
        return $"{exception.GetType().Name}: {exception.Message}" +
               Environment.NewLine + exception.StackTrace;
    }

    private static string FormatValue(object? value)
    {
        return value switch
        {
            null => "null",
            string s => $"\"{s}\"",
            _ => value.ToString() ?? "null"
        };
    }
}

/// <summary>
/// Configuration options for console logging.
/// </summary>
public sealed class ConsoleLoggerOptions
{
    /// <summary>
    /// Minimum log level to output.
    /// </summary>
    public LogLevel MinimumLevel { get; set; } = LogLevel.Information;

    /// <summary>
    /// Whether to use console colors.
    /// </summary>
    public bool UseColors { get; set; } = true;

    /// <summary>
    /// Timestamp format for console output.
    /// </summary>
    public ConsoleTimestampFormat TimestampFormat { get; set; } = ConsoleTimestampFormat.Local;

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
/// Timestamp format options for console output.
/// </summary>
public enum ConsoleTimestampFormat
{
    /// <summary>
    /// No timestamp is displayed.
    /// </summary>
    None,

    /// <summary>
    /// Local time with HH:mm:ss.fff format.
    /// </summary>
    Local,

    /// <summary>
    /// UTC time with HH:mm:ss.fff format.
    /// </summary>
    Utc,

    /// <summary>
    /// Full date and time with yyyy-MM-dd HH:mm:ss.fff format.
    /// </summary>
    Full
}