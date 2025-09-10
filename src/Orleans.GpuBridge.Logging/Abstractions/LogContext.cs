using System.Collections.Concurrent;
using System.Diagnostics;

namespace Orleans.GpuBridge.Logging.Abstractions;

/// <summary>
/// Provides contextual information for log enrichment and correlation tracking.
/// </summary>
public sealed class LogContext
{
    private static readonly AsyncLocal<LogContext?> _current = new();
    private readonly ConcurrentDictionary<string, object?> _properties = new();

    /// <summary>
    /// Gets the current log context for the async flow.
    /// </summary>
    public static LogContext? Current => _current.Value;

    /// <summary>
    /// Correlation ID for this context.
    /// </summary>
    public string CorrelationId { get; }

    /// <summary>
    /// Operation ID for nested operation tracking.
    /// </summary>
    public string? OperationId { get; set; }

    /// <summary>
    /// User ID associated with this context.
    /// </summary>
    public string? UserId { get; set; }

    /// <summary>
    /// Session ID for user session tracking.
    /// </summary>
    public string? SessionId { get; set; }

    /// <summary>
    /// Request ID for HTTP request correlation.
    /// </summary>
    public string? RequestId { get; set; }

    /// <summary>
    /// Tenant ID for multi-tenant applications.
    /// </summary>
    public string? TenantId { get; set; }

    /// <summary>
    /// Component name (e.g., service, grain type).
    /// </summary>
    public string? Component { get; set; }

    /// <summary>
    /// Environment name (Development, Staging, Production).
    /// </summary>
    public string? Environment { get; set; }

    /// <summary>
    /// Machine/host name where the log originated.
    /// </summary>
    public string MachineName { get; } = System.Environment.MachineName;

    /// <summary>
    /// Process ID.
    /// </summary>
    public int ProcessId { get; } = System.Environment.ProcessId;

    /// <summary>
    /// Custom properties for this context.
    /// </summary>
    public IReadOnlyDictionary<string, object?> Properties => _properties;

    /// <summary>
    /// Timestamp when this context was created.
    /// </summary>
    public DateTimeOffset CreatedAt { get; } = DateTimeOffset.UtcNow;

    /// <summary>
    /// Creates a new log context with auto-generated correlation ID.
    /// </summary>
    public LogContext() : this(GenerateCorrelationId())
    {
    }

    /// <summary>
    /// Creates a new log context with the specified correlation ID.
    /// </summary>
    /// <param name="correlationId">Correlation ID for this context</param>
    public LogContext(string correlationId)
    {
        CorrelationId = correlationId ?? throw new ArgumentNullException(nameof(correlationId));
        
        // Initialize with current Activity information if available
        var activity = Activity.Current;
        if (activity != null)
        {
            OperationId = activity.SpanId.ToString();
            SetProperty("TraceId", activity.TraceId.ToString());
            SetProperty("SpanId", activity.SpanId.ToString());
        }
    }

    /// <summary>
    /// Sets a property in this context.
    /// </summary>
    /// <param name="key">Property key</param>
    /// <param name="value">Property value</param>
    public void SetProperty(string key, object? value)
    {
        _properties.TryAdd(key, value);
    }

    /// <summary>
    /// Gets a property from this context.
    /// </summary>
    /// <param name="key">Property key</param>
    /// <returns>Property value or null if not found</returns>
    public T? GetProperty<T>(string key)
    {
        return _properties.TryGetValue(key, out var value) && value is T typedValue 
            ? typedValue 
            : default;
    }

    /// <summary>
    /// Creates a new context that inherits from this one.
    /// </summary>
    /// <param name="operationId">Operation ID for the child context</param>
    /// <returns>New child context</returns>
    public LogContext CreateChild(string? operationId = null)
    {
        var child = new LogContext(CorrelationId)
        {
            OperationId = operationId ?? Guid.NewGuid().ToString("N")[..8],
            UserId = UserId,
            SessionId = SessionId,
            RequestId = RequestId,
            TenantId = TenantId,
            Component = Component,
            Environment = Environment
        };

        // Copy custom properties
        foreach (var kvp in _properties)
        {
            child._properties.TryAdd(kvp.Key, kvp.Value);
        }

        return child;
    }

    /// <summary>
    /// Pushes this context as the current context for the async flow.
    /// </summary>
    /// <returns>Disposable that restores the previous context</returns>
    public IDisposable Push()
    {
        var previous = _current.Value;
        _current.Value = this;
        return new ContextScope(previous);
    }

    /// <summary>
    /// Creates a log context from the current Activity.
    /// </summary>
    /// <returns>Log context initialized with Activity information</returns>
    public static LogContext FromActivity()
    {
        var activity = Activity.Current;
        var correlationId = activity?.Id ?? GenerateCorrelationId();
        
        var context = new LogContext(correlationId);
        
        if (activity != null)
        {
            context.OperationId = activity.SpanId.ToString();
            context.SetProperty("TraceId", activity.TraceId.ToString());
            context.SetProperty("SpanId", activity.SpanId.ToString());
            context.SetProperty("ParentId", activity.ParentId);
            
            // Copy Activity tags
            foreach (var tag in activity.Tags)
            {
                context.SetProperty($"Activity.{tag.Key}", tag.Value);
            }
        }

        return context;
    }

    /// <summary>
    /// Converts this context to a dictionary for serialization.
    /// </summary>
    /// <returns>Dictionary representation</returns>
    public Dictionary<string, object?> ToDictionary()
    {
        var dict = new Dictionary<string, object?>
        {
            ["CorrelationId"] = CorrelationId,
            ["OperationId"] = OperationId,
            ["UserId"] = UserId,
            ["SessionId"] = SessionId,
            ["RequestId"] = RequestId,
            ["TenantId"] = TenantId,
            ["Component"] = Component,
            ["Environment"] = Environment,
            ["MachineName"] = MachineName,
            ["ProcessId"] = ProcessId,
            ["CreatedAt"] = CreatedAt
        };

        foreach (var kvp in _properties)
        {
            dict[kvp.Key] = kvp.Value;
        }

        return dict;
    }

    private static string GenerateCorrelationId() => 
        Guid.NewGuid().ToString("N")[..12];

    private sealed class ContextScope : IDisposable
    {
        private readonly LogContext? _previousContext;

        public ContextScope(LogContext? previousContext)
        {
            _previousContext = previousContext;
        }

        public void Dispose()
        {
            _current.Value = _previousContext;
        }
    }
}