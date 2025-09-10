using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Logging.Abstractions;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// High-performance logger implementation for Orleans GPU Bridge.
/// Integrates with the delegate-based logging system for centralized control.
/// </summary>
public sealed class GpuBridgeLogger : ILogger
{
    private readonly string _categoryName;
    private readonly LogBuffer _buffer;
    private readonly LogLevel _minimumLevel;
    private readonly ConcurrentStack<LogScope> _scopes = new();

    public GpuBridgeLogger(string categoryName, LogBuffer buffer, LogLevel minimumLevel)
    {
        _categoryName = categoryName ?? throw new ArgumentNullException(nameof(categoryName));
        _buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
        _minimumLevel = minimumLevel;
    }

    public IDisposable BeginScope<TState>(TState state) where TState : notnull
    {
        var scope = new LogScope(
            state?.ToString() ?? "Unknown",
            ExtractProperties(state));
        
        _scopes.Push(scope);
        return new ScopeDisposable(_scopes);
    }

    public bool IsEnabled(LogLevel logLevel) => logLevel >= _minimumLevel;

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, 
        Func<TState, Exception?, string> formatter)
    {
        if (!IsEnabled(logLevel))
            return;

        var message = formatter(state, exception);
        var properties = ExtractProperties(state);
        var scopes = _scopes.ToArray().AsEnumerable().Reverse().ToList();

        var entry = new LogEntry
        {
            Level = logLevel,
            Category = _categoryName,
            Message = message,
            Exception = exception,
            EventId = eventId,
            Properties = properties,
            Scopes = scopes,
            CorrelationId = LogContext.Current?.CorrelationId ?? Activity.Current?.Id,
            OperationId = LogContext.Current?.OperationId ?? Activity.Current?.SpanId.ToString(),
            ActivityId = Activity.Current?.RootId
        };

        // Try to add to buffer, fall back to console if full
        if (!_buffer.TryAdd(entry) && logLevel >= LogLevel.Error)
        {
            // For critical entries, ensure they're logged even if buffer is full
            FallbackLog(entry);
        }
    }

    /// <summary>
    /// Logs with performance metrics tracking.
    /// </summary>
    public void LogWithMetrics<TState>(LogLevel logLevel, EventId eventId, TState state, 
        Exception? exception, Func<TState, Exception?, string> formatter, PerformanceMetrics metrics)
    {
        if (!IsEnabled(logLevel))
            return;

        var message = formatter(state, exception);
        var properties = ExtractProperties(state);
        var scopes = _scopes.ToArray().AsEnumerable().Reverse().ToList();

        var entry = new LogEntry
        {
            Level = logLevel,
            Category = _categoryName,
            Message = message,
            Exception = exception,
            EventId = eventId,
            Properties = properties,
            Scopes = scopes,
            Metrics = metrics,
            CorrelationId = LogContext.Current?.CorrelationId ?? Activity.Current?.Id,
            OperationId = LogContext.Current?.OperationId ?? Activity.Current?.SpanId.ToString(),
            ActivityId = Activity.Current?.RootId
        };

        if (!_buffer.TryAdd(entry) && logLevel >= LogLevel.Error)
        {
            FallbackLog(entry);
        }
    }

    /// <summary>
    /// Logs GPU operation with specific context.
    /// </summary>
    public void LogGpuOperation(LogLevel logLevel, string operation, string? kernelId = null, 
        TimeSpan? duration = null, long? memoryUsage = null, Exception? exception = null)
    {
        if (!IsEnabled(logLevel))
            return;

        var properties = new Dictionary<string, object?>
        {
            ["Operation"] = operation,
            ["OperationType"] = "GPU",
            ["KernelId"] = kernelId
        };

        var metrics = new PerformanceMetrics
        {
            Duration = duration,
            MemoryUsage = memoryUsage
        };

        var entry = new LogEntry
        {
            Level = logLevel,
            Category = _categoryName,
            Message = $"GPU Operation: {operation}" + (kernelId != null ? $" (Kernel: {kernelId})" : ""),
            Exception = exception,
            EventId = new EventId(1001, "GpuOperation"),
            Properties = properties,
            Scopes = _scopes.ToArray().AsEnumerable().Reverse().ToList(),
            Metrics = metrics,
            CorrelationId = LogContext.Current?.CorrelationId ?? Activity.Current?.Id,
            OperationId = LogContext.Current?.OperationId ?? Activity.Current?.SpanId.ToString(),
            ActivityId = Activity.Current?.RootId
        };

        if (!_buffer.TryAdd(entry) && logLevel >= LogLevel.Error)
        {
            FallbackLog(entry);
        }
    }

    /// <summary>
    /// Logs Orleans grain operation.
    /// </summary>
    public void LogGrainOperation(LogLevel logLevel, string grainType, string method, 
        string? grainId = null, TimeSpan? duration = null, Exception? exception = null)
    {
        if (!IsEnabled(logLevel))
            return;

        var properties = new Dictionary<string, object?>
        {
            ["GrainType"] = grainType,
            ["Method"] = method,
            ["GrainId"] = grainId,
            ["OperationType"] = "Grain"
        };

        var metrics = duration.HasValue ? PerformanceMetrics.WithDuration(duration.Value) : null;

        var entry = new LogEntry
        {
            Level = logLevel,
            Category = _categoryName,
            Message = $"Grain {grainType}.{method}" + (grainId != null ? $" ({grainId})" : ""),
            Exception = exception,
            EventId = new EventId(1002, "GrainOperation"),
            Properties = properties,
            Scopes = _scopes.ToArray().AsEnumerable().Reverse().ToList(),
            Metrics = metrics,
            CorrelationId = LogContext.Current?.CorrelationId ?? Activity.Current?.Id,
            OperationId = LogContext.Current?.OperationId ?? Activity.Current?.SpanId.ToString(),
            ActivityId = Activity.Current?.RootId
        };

        if (!_buffer.TryAdd(entry) && logLevel >= LogLevel.Error)
        {
            FallbackLog(entry);
        }
    }

    private static Dictionary<string, object?> ExtractProperties<TState>(TState state)
    {
        var properties = new Dictionary<string, object?>();

        if (state is null)
            return properties;

        // Handle structured logging state
        if (state is IEnumerable<KeyValuePair<string, object?>> kvps)
        {
            foreach (var kvp in kvps)
            {
                // Skip original format and template
                if (kvp.Key != "{OriginalFormat}")
                {
                    properties[kvp.Key] = kvp.Value;
                }
            }
        }
        else if (state is IReadOnlyList<KeyValuePair<string, object?>> roList)
        {
            foreach (var kvp in roList)
            {
                if (kvp.Key != "{OriginalFormat}")
                {
                    properties[kvp.Key] = kvp.Value;
                }
            }
        }
        else
        {
            // For simple states, add as a generic property
            properties["State"] = state;
        }

        return properties;
    }

    private void FallbackLog(LogEntry entry)
    {
        try
        {
            var timestamp = entry.Timestamp.ToString("HH:mm:ss.fff");
            var level = entry.Level.ToString().ToUpperInvariant();
            var message = $"[{timestamp}] {level} [{entry.Category}] {entry.Message}";
            
            if (entry.Exception != null)
            {
                message += Environment.NewLine + entry.Exception.ToString();
            }

            Console.WriteLine(message);
        }
        catch
        {
            // Ultimate fallback - don't throw from logging
        }
    }

    private sealed class ScopeDisposable : IDisposable
    {
        private readonly ConcurrentStack<LogScope> _scopes;
        private bool _disposed;

        public ScopeDisposable(ConcurrentStack<LogScope> scopes)
        {
            _scopes = scopes;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _scopes.TryPop(out _);
                _disposed = true;
            }
        }
    }
}

/// <summary>
/// Extension methods for GpuBridgeLogger.
/// </summary>
public static class GpuBridgeLoggerExtensions
{
    /// <summary>
    /// Logs GPU kernel execution.
    /// </summary>
    public static void LogKernelExecution(this ILogger logger, string kernelId, 
        TimeSpan duration, long inputSize, long outputSize, bool success = true)
    {
        if (logger is GpuBridgeLogger gpuLogger)
        {
            var level = success ? LogLevel.Information : LogLevel.Error;
            var metrics = new PerformanceMetrics
            {
                Duration = duration,
                Counters = new Dictionary<string, double>
                {
                    ["InputSize"] = inputSize,
                    ["OutputSize"] = outputSize,
                    ["Throughput"] = inputSize / duration.TotalSeconds
                }
            };

            gpuLogger.LogWithMetrics(level, new EventId(1003, "KernelExecution"),
                "Kernel execution completed", null,
                (state, ex) => $"Kernel {kernelId} executed in {duration.TotalMilliseconds:F2}ms",
                metrics);
        }
        else
        {
            logger.LogInformation("Kernel {KernelId} executed in {Duration}ms", 
                kernelId, duration.TotalMilliseconds);
        }
    }

    /// <summary>
    /// Logs GPU memory operations.
    /// </summary>
    public static void LogMemoryOperation(this ILogger logger, string operation, 
        long bytes, TimeSpan duration)
    {
        if (logger is GpuBridgeLogger gpuLogger)
        {
            gpuLogger.LogGpuOperation(LogLevel.Debug, $"Memory {operation}", 
                null, duration, bytes);
        }
        else
        {
            logger.LogDebug("GPU Memory {Operation}: {Bytes} bytes in {Duration}ms", 
                operation, bytes, duration.TotalMilliseconds);
        }
    }

    /// <summary>
    /// Logs grain activation/deactivation.
    /// </summary>
    public static void LogGrainLifecycle(this ILogger logger, string grainType, 
        string grainId, string operation)
    {
        if (logger is GpuBridgeLogger gpuLogger)
        {
            gpuLogger.LogGrainOperation(LogLevel.Debug, grainType, operation, grainId);
        }
        else
        {
            logger.LogDebug("Grain {GrainType} ({GrainId}) {Operation}", 
                grainType, grainId, operation);
        }
    }

    /// <summary>
    /// Logs with correlation context.
    /// </summary>
    public static void LogWithCorrelation(this ILogger logger, LogLevel level, 
        string message, string correlationId, params object?[] args)
    {
        using var context = new LogContext(correlationId).Push();
        logger.Log(level, message, args);
    }

    /// <summary>
    /// Logs performance metrics.
    /// </summary>
    public static void LogPerformance(this ILogger logger, string operation, 
        TimeSpan duration, Dictionary<string, double>? counters = null)
    {
        var metrics = new PerformanceMetrics
        {
            Duration = duration,
            Counters = counters ?? new Dictionary<string, double>()
        };

        if (logger is GpuBridgeLogger gpuLogger)
        {
            gpuLogger.LogWithMetrics(LogLevel.Information, new EventId(1004, "Performance"),
                operation, null,
                (state, ex) => $"Performance: {operation} completed in {duration.TotalMilliseconds:F2}ms",
                metrics);
        }
        else
        {
            logger.LogInformation("Performance: {Operation} completed in {Duration}ms", 
                operation, duration.TotalMilliseconds);
        }
    }
}