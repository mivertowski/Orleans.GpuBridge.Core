using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Logging.Abstractions;
using OpenTelemetry;
using OpenTelemetry.Logs;
using System.Diagnostics;
using System.Collections.Concurrent;

namespace Orleans.GpuBridge.Logging.Delegates;

/// <summary>
/// Logger delegate that integrates with OpenTelemetry for distributed tracing and telemetry.
/// </summary>
public sealed class TelemetryLoggerDelegate : IBatchLoggerDelegate, IStructuredLoggerDelegate, IAsyncDisposable
{
    private readonly TelemetryLoggerOptions _options;
    private readonly ActivitySource _activitySource;
    private readonly ConcurrentQueue<LogEntry> _telemetryQueue = new();
    private readonly Timer _flushTimer;
    private readonly SemaphoreSlim _processingLock = new(1, 1);
    private bool _disposed;

    // OpenTelemetry integration
    private readonly ILogger<TelemetryLoggerDelegate>? _otlpLogger;

    public string Name => "Telemetry";
    public LogLevel MinimumLevel { get; }
    public int MaxBatchSize => _options.MaxBatchSize;

    public TelemetryLoggerDelegate(TelemetryLoggerOptions options, 
        ILogger<TelemetryLoggerDelegate>? otlpLogger = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _otlpLogger = otlpLogger;
        MinimumLevel = _options.MinimumLevel;
        
        _activitySource = new ActivitySource(_options.ServiceName, _options.ServiceVersion);
        
        // Setup periodic flush timer
        _flushTimer = new Timer(FlushTelemetryQueue, null, 
            _options.FlushInterval, _options.FlushInterval);
    }

    public bool IsEnabled(LogLevel logLevel) => logLevel >= MinimumLevel;

    public async Task WriteAsync(LogEntry entry, CancellationToken cancellationToken = default)
    {
        if (!IsEnabled(entry.Level) || _disposed)
            return;

        var enrichedEntry = EnrichEntry(entry);
        
        // Queue for batched telemetry processing
        _telemetryQueue.Enqueue(enrichedEntry);
        
        // Create OpenTelemetry activity for critical events
        if (entry.Level >= LogLevel.Warning || _options.TraceAllLevels)
        {
            await CreateTelemetryActivity(enrichedEntry, cancellationToken);
        }

        // Forward to OTLP logger if available
        if (_otlpLogger != null)
        {
            ForwardToOtlpLogger(enrichedEntry);
        }

        // Immediate flush for critical errors
        if (entry.Level >= LogLevel.Error && _options.FlushOnError)
        {
            await ProcessTelemetryQueue(cancellationToken);
        }
    }

    public async Task WriteBatchAsync(IReadOnlyCollection<LogEntry> entries, CancellationToken cancellationToken = default)
    {
        if (entries.Count == 0 || _disposed)
            return;

        var filteredEntries = entries.Where(e => IsEnabled(e.Level)).ToList();
        if (filteredEntries.Count == 0)
            return;

        var hasErrors = false;
        
        foreach (var entry in filteredEntries)
        {
            var enrichedEntry = EnrichEntry(entry);
            _telemetryQueue.Enqueue(enrichedEntry);
            
            if (entry.Level >= LogLevel.Error)
                hasErrors = true;

            // Create activities for warnings and above
            if (entry.Level >= LogLevel.Warning || _options.TraceAllLevels)
            {
                await CreateTelemetryActivity(enrichedEntry, cancellationToken);
            }

            // Forward to OTLP logger
            if (_otlpLogger != null)
            {
                ForwardToOtlpLogger(enrichedEntry);
            }
        }

        // Flush immediately if errors are present
        if (hasErrors && _options.FlushOnError)
        {
            await ProcessTelemetryQueue(cancellationToken);
        }
    }

    public LogEntry EnrichEntry(LogEntry entry, LogContext? context = null)
    {
        var enrichedProperties = new Dictionary<string, object?>(entry.Properties);

        // Add service information
        enrichedProperties["Service.Name"] = _options.ServiceName;
        enrichedProperties["Service.Version"] = _options.ServiceVersion;
        enrichedProperties["Service.Instance"] = _options.ServiceInstance;

        // Add OpenTelemetry trace information
        var activity = Activity.Current;
        if (activity != null)
        {
            enrichedProperties["TraceId"] = activity.TraceId.ToString();
            enrichedProperties["SpanId"] = activity.SpanId.ToString();
            enrichedProperties["TraceFlags"] = activity.ActivityTraceFlags.ToString();
            
            // Copy activity tags
            foreach (var tag in activity.Tags)
            {
                enrichedProperties[$"Activity.{tag.Key}"] = tag.Value;
            }
        }

        // Add context properties
        context ??= LogContext.Current;
        if (context != null)
        {
            enrichedProperties["Telemetry.CorrelationId"] = context.CorrelationId;
            enrichedProperties["Telemetry.OperationId"] = context.OperationId;
            enrichedProperties["Telemetry.UserId"] = context.UserId;
            enrichedProperties["Telemetry.SessionId"] = context.SessionId;
            enrichedProperties["Telemetry.TenantId"] = context.TenantId;
            enrichedProperties["Telemetry.Component"] = context.Component;
            enrichedProperties["Telemetry.Environment"] = context.Environment;
            
            // Add custom context properties
            foreach (var kvp in context.Properties)
            {
                enrichedProperties[$"Context.{kvp.Key}"] = kvp.Value;
            }
        }

        // Add resource information
        enrichedProperties["Resource.Machine"] = Environment.MachineName;
        enrichedProperties["Resource.Process"] = Environment.ProcessId;
        enrichedProperties["Resource.Thread"] = entry.ThreadId;
        enrichedProperties["Resource.OS"] = Environment.OSVersion.ToString();

        // Add performance telemetry
        if (entry.Metrics != null)
        {
            enrichedProperties["Performance.Duration"] = entry.Metrics.Duration?.TotalMilliseconds;
            enrichedProperties["Performance.Memory"] = entry.Metrics.MemoryUsage;
            enrichedProperties["Performance.CPU"] = entry.Metrics.CpuUsage;
            
            foreach (var counter in entry.Metrics.Counters)
            {
                enrichedProperties[$"Performance.{counter.Key}"] = counter.Value;
            }
        }

        return entry.WithProperties(enrichedProperties);
    }

    public async Task FlushAsync(CancellationToken cancellationToken = default)
    {
        if (_disposed)
            return;

        await ProcessTelemetryQueue(cancellationToken);
    }

    ValueTask IAsyncDisposable.DisposeAsync() => DisposeAsync(CancellationToken.None);

    public async ValueTask DisposeAsync(CancellationToken cancellationToken = default)
    {
        if (_disposed)
            return;

        _disposed = true;

        await _flushTimer.DisposeAsync();
        
        // Process remaining telemetry
        await ProcessTelemetryQueue(cancellationToken);
        
        _activitySource.Dispose();
        _processingLock.Dispose();
    }

    private async Task CreateTelemetryActivity(LogEntry entry, CancellationToken cancellationToken)
    {
        using var activity = _activitySource.StartActivity($"Log.{entry.Level}");
        
        if (activity == null)
            return;

        // Set activity tags
        activity.SetTag("log.level", entry.Level.ToString());
        activity.SetTag("log.category", entry.Category);
        activity.SetTag("log.message", entry.Message);
        activity.SetTag("log.timestamp", entry.Timestamp.ToString("O"));
        
        if (!string.IsNullOrEmpty(entry.CorrelationId))
            activity.SetTag("log.correlation_id", entry.CorrelationId);
        
        if (!string.IsNullOrEmpty(entry.OperationId))
            activity.SetTag("log.operation_id", entry.OperationId);

        // Set status based on log level
        if (entry.Level >= LogLevel.Error)
        {
            activity.SetStatus(ActivityStatusCode.Error, entry.Message);
        }
        else if (entry.Level == LogLevel.Warning)
        {
            activity.SetStatus(ActivityStatusCode.Ok, "Warning logged");
        }

        // Add structured properties as tags
        foreach (var prop in entry.Properties.Take(20)) // Limit tags to avoid performance issues
        {
            var value = prop.Value?.ToString();
            if (!string.IsNullOrEmpty(value))
            {
                activity.SetTag($"log.property.{prop.Key}", value);
            }
        }

        // Add exception details
        if (entry.Exception != null)
        {
            activity.SetTag("exception.type", entry.Exception.GetType().FullName);
            activity.SetTag("exception.message", entry.Exception.Message);
            activity.SetTag("exception.stack_trace", entry.Exception.StackTrace);
            
            // Note: RecordException is not available in basic Activity, would need OpenTelemetry.Api
            // activity.RecordException(entry.Exception);
        }

        // Add performance metrics as events
        if (entry.Metrics != null)
        {
            var tags = new ActivityTagsCollection();
            
            if (entry.Metrics.Duration.HasValue)
                tags.Add("duration_ms", entry.Metrics.Duration.Value.TotalMilliseconds);
            
            if (entry.Metrics.MemoryUsage.HasValue)
                tags.Add("memory_bytes", entry.Metrics.MemoryUsage.Value);
            
            if (entry.Metrics.CpuUsage.HasValue)
                tags.Add("cpu_usage", entry.Metrics.CpuUsage.Value);

            foreach (var counter in entry.Metrics.Counters)
            {
                tags.Add(counter.Key, counter.Value);
            }

            activity.AddEvent(new ActivityEvent("performance_metrics", DateTimeOffset.UtcNow, tags));
        }

        await Task.Delay(1, cancellationToken); // Yield for async processing
    }

    private void ForwardToOtlpLogger(LogEntry entry)
    {
        if (_otlpLogger == null)
            return;

        try
        {
            var logLevel = entry.Level;
            var message = entry.Message;
            var exception = entry.Exception;
            var eventId = entry.EventId;

            // Create structured state from properties
            var state = entry.Properties.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

            _otlpLogger.Log(logLevel, eventId, state, exception, (s, ex) => message);
        }
        catch (Exception ex)
        {
            // Fallback logging to prevent cascade failures
            Console.WriteLine($"Failed to forward log to OTLP: {ex.Message}");
        }
    }

    private void FlushTelemetryQueue(object? state)
    {
        _ = Task.Run(async () =>
        {
            try
            {
                await ProcessTelemetryQueue(CancellationToken.None);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing telemetry queue: {ex.Message}");
            }
        });
    }

    private async Task ProcessTelemetryQueue(CancellationToken cancellationToken)
    {
        if (_disposed || _telemetryQueue.IsEmpty)
            return;

        await _processingLock.WaitAsync(cancellationToken);
        try
        {
            var batch = new List<LogEntry>();
            
            // Collect batch
            while (_telemetryQueue.TryDequeue(out var entry) && batch.Count < _options.MaxBatchSize)
            {
                batch.Add(entry);
            }

            if (batch.Count == 0)
                return;

            // Process telemetry batch
            await ProcessTelemetryBatch(batch, cancellationToken);
        }
        finally
        {
            _processingLock.Release();
        }
    }

    private async Task ProcessTelemetryBatch(List<LogEntry> entries, CancellationToken cancellationToken)
    {
        try
        {
            // Group by severity for different processing
            var errorEntries = entries.Where(e => e.Level >= LogLevel.Error).ToList();
            var warningEntries = entries.Where(e => e.Level == LogLevel.Warning).ToList();
            var infoEntries = entries.Where(e => e.Level < LogLevel.Warning).ToList();

            // Process high-priority entries first
            if (errorEntries.Count > 0)
            {
                await ProcessHighPriorityTelemetry(errorEntries, cancellationToken);
            }

            if (warningEntries.Count > 0)
            {
                await ProcessMediumPriorityTelemetry(warningEntries, cancellationToken);
            }

            if (infoEntries.Count > 0)
            {
                await ProcessLowPriorityTelemetry(infoEntries, cancellationToken);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error processing telemetry batch: {ex.Message}");
        }
    }

    private async Task ProcessHighPriorityTelemetry(List<LogEntry> entries, CancellationToken cancellationToken)
    {
        // For high-priority telemetry, we might want to send to external systems
        // This is a placeholder for actual telemetry processing
        await Task.Delay(1, cancellationToken);
        
        // Log statistics
        Console.WriteLine($"Processed {entries.Count} high-priority telemetry entries");
    }

    private async Task ProcessMediumPriorityTelemetry(List<LogEntry> entries, CancellationToken cancellationToken)
    {
        await Task.Delay(1, cancellationToken);
        Console.WriteLine($"Processed {entries.Count} medium-priority telemetry entries");
    }

    private async Task ProcessLowPriorityTelemetry(List<LogEntry> entries, CancellationToken cancellationToken)
    {
        await Task.Delay(1, cancellationToken);
        Console.WriteLine($"Processed {entries.Count} low-priority telemetry entries");
    }
}

/// <summary>
/// Configuration options for telemetry logging.
/// </summary>
public sealed class TelemetryLoggerOptions
{
    /// <summary>
    /// Minimum log level to process for telemetry.
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