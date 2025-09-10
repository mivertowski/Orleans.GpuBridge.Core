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

    public string Name => "Console";
    public LogLevel MinimumLevel { get; }

    public ConsoleLoggerDelegate(ConsoleLoggerOptions? options = null)
    {
        _options = options ?? new ConsoleLoggerOptions();
        MinimumLevel = _options.MinimumLevel;
    }

    public bool IsEnabled(LogLevel logLevel) => logLevel >= MinimumLevel;

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

    public Task FlushAsync(CancellationToken cancellationToken = default)
    {
        // Console output is immediately flushed
        return Task.CompletedTask;
    }

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
    None,
    Local,
    Utc,
    Full
}