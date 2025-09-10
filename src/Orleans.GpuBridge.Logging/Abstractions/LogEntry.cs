using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace Orleans.GpuBridge.Logging.Abstractions;

/// <summary>
/// Represents a structured log entry with rich metadata.
/// </summary>
public sealed record LogEntry
{
    /// <summary>
    /// Unique identifier for this log entry.
    /// </summary>
    public string Id { get; init; } = Guid.NewGuid().ToString("N");

    /// <summary>
    /// Timestamp when the log entry was created.
    /// </summary>
    public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>
    /// Log level of this entry.
    /// </summary>
    public LogLevel Level { get; init; }

    /// <summary>
    /// Category/source of the log entry (typically the logger name).
    /// </summary>
    public string Category { get; init; } = string.Empty;

    /// <summary>
    /// The log message.
    /// </summary>
    public string Message { get; init; } = string.Empty;

    /// <summary>
    /// Exception associated with this log entry, if any.
    /// </summary>
    public Exception? Exception { get; init; }

    /// <summary>
    /// Event ID associated with this log entry.
    /// </summary>
    public EventId EventId { get; init; }

    /// <summary>
    /// Correlation ID for distributed tracing.
    /// </summary>
    public string? CorrelationId { get; init; }

    /// <summary>
    /// Operation ID for tracing nested operations.
    /// </summary>
    public string? OperationId { get; init; }

    /// <summary>
    /// Activity ID for distributed tracing compatibility.
    /// </summary>
    public string? ActivityId { get; init; }

    /// <summary>
    /// Structured properties associated with this log entry.
    /// </summary>
    public IReadOnlyDictionary<string, object?> Properties { get; init; } = 
        new Dictionary<string, object?>();

    /// <summary>
    /// Scope information captured at log time.
    /// </summary>
    public IReadOnlyList<LogScope> Scopes { get; init; } = Array.Empty<LogScope>();

    /// <summary>
    /// Performance metrics associated with this log entry.
    /// </summary>
    public PerformanceMetrics? Metrics { get; init; }

    /// <summary>
    /// Thread ID where this log entry was created.
    /// </summary>
    public int ThreadId { get; init; } = Environment.CurrentManagedThreadId;

    /// <summary>
    /// Creates a new log entry with the specified level, category, and message.
    /// </summary>
    public static LogEntry Create(LogLevel level, string category, string message)
    {
        return new LogEntry
        {
            Level = level,
            Category = category,
            Message = message,
            CorrelationId = Activity.Current?.Id,
            OperationId = Activity.Current?.SpanId.ToString(),
            ActivityId = Activity.Current?.RootId
        };
    }

    /// <summary>
    /// Creates a new log entry with exception.
    /// </summary>
    public static LogEntry CreateWithException(LogLevel level, string category, string message, Exception exception)
    {
        return Create(level, category, message) with { Exception = exception };
    }

    /// <summary>
    /// Creates a new log entry with structured properties.
    /// </summary>
    public static LogEntry CreateWithProperties(LogLevel level, string category, string message, 
        IReadOnlyDictionary<string, object?> properties)
    {
        return Create(level, category, message) with { Properties = properties };
    }

    /// <summary>
    /// Creates a copy of this log entry with additional properties.
    /// </summary>
    public LogEntry WithProperties(IReadOnlyDictionary<string, object?> additionalProperties)
    {
        var mergedProperties = new Dictionary<string, object?>(Properties);
        foreach (var kvp in additionalProperties)
        {
            mergedProperties[kvp.Key] = kvp.Value;
        }
        
        return this with { Properties = mergedProperties };
    }

    /// <summary>
    /// Creates a copy of this log entry with correlation information.
    /// </summary>
    public LogEntry WithCorrelation(string correlationId, string? operationId = null)
    {
        return this with 
        { 
            CorrelationId = correlationId,
            OperationId = operationId ?? OperationId
        };
    }

    /// <summary>
    /// Creates a copy of this log entry with performance metrics.
    /// </summary>
    public LogEntry WithMetrics(PerformanceMetrics metrics)
    {
        return this with { Metrics = metrics };
    }
}

/// <summary>
/// Represents a logging scope with key-value pairs.
/// </summary>
public sealed record LogScope(string Name, IReadOnlyDictionary<string, object?> Properties);

/// <summary>
/// Performance metrics for log entries.
/// </summary>
public sealed record PerformanceMetrics
{
    /// <summary>
    /// Duration of the operation being logged.
    /// </summary>
    public TimeSpan? Duration { get; init; }

    /// <summary>
    /// Memory usage at the time of logging (in bytes).
    /// </summary>
    public long? MemoryUsage { get; init; }

    /// <summary>
    /// CPU usage percentage.
    /// </summary>
    public double? CpuUsage { get; init; }

    /// <summary>
    /// Custom performance counters.
    /// </summary>
    public IReadOnlyDictionary<string, double> Counters { get; init; } = 
        new Dictionary<string, double>();

    /// <summary>
    /// Creates performance metrics with duration.
    /// </summary>
    public static PerformanceMetrics WithDuration(TimeSpan duration) =>
        new() { Duration = duration };

    /// <summary>
    /// Creates performance metrics with memory usage.
    /// </summary>
    public static PerformanceMetrics WithMemory(long bytes) =>
        new() { MemoryUsage = bytes };

    /// <summary>
    /// Creates performance metrics with custom counters.
    /// </summary>
    public static PerformanceMetrics WithCounters(IReadOnlyDictionary<string, double> counters) =>
        new() { Counters = counters };
}