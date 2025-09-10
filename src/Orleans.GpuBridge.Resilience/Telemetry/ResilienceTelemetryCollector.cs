using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics.Metrics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Resilience.Fallback;
using Orleans.GpuBridge.Resilience.Policies;
using Orleans.GpuBridge.Resilience.RateLimit;

namespace Orleans.GpuBridge.Resilience.Telemetry;

/// <summary>
/// Comprehensive telemetry collector for resilience patterns
/// </summary>
public sealed class ResilienceTelemetryCollector : IDisposable
{
    private readonly ILogger<ResilienceTelemetryCollector> _logger;
    private readonly Meter _meter;
    private readonly Timer _metricsTimer;
    
    // Counters
    private readonly Counter<long> _retryCounter;
    private readonly Counter<long> _circuitBreakerCounter;
    private readonly Counter<long> _timeoutCounter;
    private readonly Counter<long> _fallbackCounter;
    private readonly Counter<long> _rateLimitCounter;
    private readonly Counter<long> _chaosCounter;
    
    // Histograms
    private readonly Histogram<double> _operationDuration;
    private readonly Histogram<double> _retryDelay;
    private readonly Histogram<double> _circuitBreakerDuration;
    
    // Gauges (using ObservableGauge)
    private readonly ObservableGauge<double> _bulkheadUtilization;
    private readonly ObservableGauge<double> _circuitBreakerState;
    private readonly ObservableGauge<double> _fallbackLevel;
    private readonly ObservableGauge<double> _rateLimiterTokens;
    
    // Internal state tracking
    private readonly ConcurrentDictionary<string, CircuitBreakerState> _circuitStates;
    private readonly ConcurrentDictionary<string, double> _bulkheadUtilizations;
    private readonly ConcurrentDictionary<string, FallbackLevel> _fallbackLevels;
    private readonly ConcurrentDictionary<string, double> _rateLimiterStates;
    
    // Health monitoring
    private readonly ConcurrentQueue<HealthEvent> _healthEvents;
    private readonly ConcurrentDictionary<string, ComponentHealth> _componentHealth;
    
    private bool _disposed;

    public ResilienceTelemetryCollector(ILogger<ResilienceTelemetryCollector> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        
        _meter = new Meter("Orleans.GpuBridge.Resilience", "1.0.0");
        
        // Initialize counters
        _retryCounter = _meter.CreateCounter<long>("resilience_retries_total", 
            description: "Total number of retries executed");
        _circuitBreakerCounter = _meter.CreateCounter<long>("resilience_circuit_breaker_events_total", 
            description: "Total circuit breaker state changes");
        _timeoutCounter = _meter.CreateCounter<long>("resilience_timeouts_total", 
            description: "Total number of timeouts");
        _fallbackCounter = _meter.CreateCounter<long>("resilience_fallbacks_total", 
            description: "Total number of fallback executions");
        _rateLimitCounter = _meter.CreateCounter<long>("resilience_rate_limit_events_total", 
            description: "Total rate limiting events");
        _chaosCounter = _meter.CreateCounter<long>("resilience_chaos_faults_total", 
            description: "Total chaos engineering faults injected");
        
        // Initialize histograms
        _operationDuration = _meter.CreateHistogram<double>("resilience_operation_duration_ms", 
            description: "Duration of resilient operations in milliseconds");
        _retryDelay = _meter.CreateHistogram<double>("resilience_retry_delay_ms", 
            description: "Delay between retries in milliseconds");
        _circuitBreakerDuration = _meter.CreateHistogram<double>("resilience_circuit_breaker_duration_ms", 
            description: "Duration circuit breaker remained in open state");
        
        // Initialize state tracking
        _circuitStates = new ConcurrentDictionary<string, CircuitBreakerState>();
        _bulkheadUtilizations = new ConcurrentDictionary<string, double>();
        _fallbackLevels = new ConcurrentDictionary<string, FallbackLevel>();
        _rateLimiterStates = new ConcurrentDictionary<string, double>();
        _healthEvents = new ConcurrentQueue<HealthEvent>();
        _componentHealth = new ConcurrentDictionary<string, ComponentHealth>();
        
        // Initialize observable gauges
        _bulkheadUtilization = _meter.CreateObservableGauge<double>("resilience_bulkhead_utilization", 
            description: "Current bulkhead utilization percentage",
            observeValues: ObserveBulkheadUtilization);
        
        _circuitBreakerState = _meter.CreateObservableGauge<double>("resilience_circuit_breaker_state", 
            description: "Current circuit breaker state (0=Closed, 1=Open, 2=HalfOpen)",
            observeValues: ObserveCircuitBreakerState);
        
        _fallbackLevel = _meter.CreateObservableGauge<double>("resilience_fallback_level", 
            description: "Current fallback level",
            observeValues: ObserveFallbackLevel);
        
        _rateLimiterTokens = _meter.CreateObservableGauge<double>("resilience_rate_limiter_tokens", 
            description: "Available tokens in rate limiter",
            observeValues: ObserveRateLimiterTokens);
        
        // Start periodic health monitoring
        _metricsTimer = new Timer(CollectMetrics, null, TimeSpan.FromSeconds(10), TimeSpan.FromSeconds(10));
        
        _logger.LogInformation("Resilience telemetry collector initialized");
    }

    /// <summary>
    /// Records a retry event
    /// </summary>
    public void RecordRetry(string operation, int attemptNumber, TimeSpan delay, Exception? exception = null)
    {
        _retryCounter.Add(1, new KeyValuePair<string, object?>[]
        {
            new("operation", operation),
            new("attempt", attemptNumber),
            new("exception_type", exception?.GetType().Name ?? "unknown")
        });
        
        _retryDelay.Record(delay.TotalMilliseconds, new KeyValuePair<string, object?>[]
        {
            new("operation", operation)
        });
        
        _logger.LogDebug("Recorded retry event: {Operation} attempt {Attempt} with {Delay}ms delay", 
            operation, attemptNumber, delay.TotalMilliseconds);
    }

    /// <summary>
    /// Records a circuit breaker event
    /// </summary>
    public void RecordCircuitBreakerEvent(string circuitName, CircuitBreakerState newState, TimeSpan? duration = null)
    {
        _circuitStates[circuitName] = newState;
        
        _circuitBreakerCounter.Add(1, new KeyValuePair<string, object?>[]
        {
            new("circuit", circuitName),
            new("state", newState.ToString())
        });
        
        if (duration.HasValue)
        {
            _circuitBreakerDuration.Record(duration.Value.TotalMilliseconds, new KeyValuePair<string, object?>[]
            {
                new("circuit", circuitName)
            });
        }
        
        UpdateComponentHealth(circuitName, newState == CircuitBreakerState.Open ? 
            ComponentHealth.Degraded : ComponentHealth.Healthy);
        
        _logger.LogInformation("Recorded circuit breaker event: {Circuit} changed to {State}", 
            circuitName, newState);
    }

    /// <summary>
    /// Records a timeout event
    /// </summary>
    public void RecordTimeout(string operation, TimeSpan timeout, Exception? exception = null)
    {
        _timeoutCounter.Add(1, new KeyValuePair<string, object?>[]
        {
            new("operation", operation),
            new("timeout_ms", timeout.TotalMilliseconds),
            new("exception_type", exception?.GetType().Name ?? "timeout")
        });
        
        _logger.LogWarning("Recorded timeout event: {Operation} timed out after {Timeout}ms", 
            operation, timeout.TotalMilliseconds);
    }

    /// <summary>
    /// Records a fallback execution
    /// </summary>
    public void RecordFallback(string operation, FallbackLevel fromLevel, FallbackLevel toLevel, bool successful)
    {
        _fallbackLevels[operation] = toLevel;
        
        _fallbackCounter.Add(1, new KeyValuePair<string, object?>[]
        {
            new("operation", operation),
            new("from_level", fromLevel.ToString()),
            new("to_level", toLevel.ToString()),
            new("successful", successful)
        });
        
        UpdateComponentHealth(operation, successful ? ComponentHealth.Degraded : ComponentHealth.Unhealthy);
        
        _logger.LogInformation("Recorded fallback event: {Operation} from {FromLevel} to {ToLevel}, successful: {Successful}", 
            operation, fromLevel, toLevel, successful);
    }

    /// <summary>
    /// Records a rate limiting event
    /// </summary>
    public void RecordRateLimit(string operation, bool accepted, double availableTokens)
    {
        _rateLimiterStates[operation] = availableTokens;
        
        _rateLimitCounter.Add(1, new KeyValuePair<string, object?>[]
        {
            new("operation", operation),
            new("accepted", accepted),
            new("available_tokens", availableTokens)
        });
        
        if (!accepted)
        {
            _logger.LogWarning("Recorded rate limit rejection: {Operation} with {AvailableTokens} tokens", 
                operation, availableTokens);
        }
    }

    /// <summary>
    /// Records a chaos engineering fault injection
    /// </summary>
    public void RecordChaosFault(string operation, string faultType, TimeSpan? duration = null)
    {
        _chaosCounter.Add(1, new KeyValuePair<string, object?>[]
        {
            new("operation", operation),
            new("fault_type", faultType),
            new("duration_ms", duration?.TotalMilliseconds ?? 0)
        });
        
        _logger.LogInformation("Recorded chaos fault: {FaultType} injected into {Operation}", 
            faultType, operation);
    }

    /// <summary>
    /// Records operation execution metrics
    /// </summary>
    public void RecordOperation(string operation, TimeSpan duration, bool successful, string? errorType = null)
    {
        _operationDuration.Record(duration.TotalMilliseconds, new KeyValuePair<string, object?>[]
        {
            new("operation", operation),
            new("successful", successful),
            new("error_type", errorType ?? "none")
        });
        
        UpdateComponentHealth(operation, successful ? ComponentHealth.Healthy : ComponentHealth.Degraded);
    }

    /// <summary>
    /// Updates bulkhead utilization
    /// </summary>
    public void UpdateBulkheadUtilization(string bulkheadName, double utilizationPercentage)
    {
        _bulkheadUtilizations[bulkheadName] = utilizationPercentage;
    }

    /// <summary>
    /// Gets comprehensive resilience metrics
    /// </summary>
    public ResilienceMetrics GetMetrics()
    {
        var healthEvents = new List<HealthEvent>();
        while (_healthEvents.TryDequeue(out var evt))
        {
            healthEvents.Add(evt);
        }

        return new ResilienceMetrics(
            ComponentHealth: new Dictionary<string, ComponentHealth>(_componentHealth),
            CircuitBreakerStates: new Dictionary<string, CircuitBreakerState>(_circuitStates),
            BulkheadUtilizations: new Dictionary<string, double>(_bulkheadUtilizations),
            FallbackLevels: new Dictionary<string, FallbackLevel>(_fallbackLevels),
            RateLimiterStates: new Dictionary<string, double>(_rateLimiterStates),
            RecentHealthEvents: healthEvents,
            LastUpdated: DateTimeOffset.UtcNow);
    }

    /// <summary>
    /// Updates component health status
    /// </summary>
    private void UpdateComponentHealth(string component, ComponentHealth health)
    {
        var previousHealth = _componentHealth.GetValueOrDefault(component, ComponentHealth.Unknown);
        _componentHealth[component] = health;
        
        if (previousHealth != health)
        {
            var healthEvent = new HealthEvent(
                DateTimeOffset.UtcNow,
                component,
                previousHealth,
                health);
            
            _healthEvents.Enqueue(healthEvent);
            
            // Limit queue size
            while (_healthEvents.Count > 1000)
            {
                _healthEvents.TryDequeue(out _);
            }
        }
    }

    /// <summary>
    /// Observes bulkhead utilization for metrics
    /// </summary>
    private IEnumerable<Measurement<double>> ObserveBulkheadUtilization()
    {
        foreach (var kvp in _bulkheadUtilizations)
        {
            yield return new Measurement<double>(kvp.Value, new KeyValuePair<string, object?>[]
            {
                new("bulkhead", kvp.Key)
            });
        }
    }

    /// <summary>
    /// Observes circuit breaker states for metrics
    /// </summary>
    private IEnumerable<Measurement<double>> ObserveCircuitBreakerState()
    {
        foreach (var kvp in _circuitStates)
        {
            yield return new Measurement<double>((int)kvp.Value, new KeyValuePair<string, object?>[]
            {
                new("circuit", kvp.Key)
            });
        }
    }

    /// <summary>
    /// Observes fallback levels for metrics
    /// </summary>
    private IEnumerable<Measurement<double>> ObserveFallbackLevel()
    {
        foreach (var kvp in _fallbackLevels)
        {
            yield return new Measurement<double>((int)kvp.Value, new KeyValuePair<string, object?>[]
            {
                new("operation", kvp.Key)
            });
        }
    }

    /// <summary>
    /// Observes rate limiter token availability
    /// </summary>
    private IEnumerable<Measurement<double>> ObserveRateLimiterTokens()
    {
        foreach (var kvp in _rateLimiterStates)
        {
            yield return new Measurement<double>(kvp.Value, new KeyValuePair<string, object?>[]
            {
                new("rate_limiter", kvp.Key)
            });
        }
    }

    /// <summary>
    /// Periodic metrics collection callback
    /// </summary>
    private void CollectMetrics(object? state)
    {
        if (_disposed) return;
        
        try
        {
            // Clean up old health events
            var cutoff = DateTimeOffset.UtcNow.AddHours(-1);
            var eventsToKeep = new List<HealthEvent>();
            
            while (_healthEvents.TryDequeue(out var evt))
            {
                if (evt.Timestamp >= cutoff)
                {
                    eventsToKeep.Add(evt);
                }
            }
            
            foreach (var evt in eventsToKeep)
            {
                _healthEvents.Enqueue(evt);
            }
            
            _logger.LogDebug("Metrics collection completed: {Components} components, {Events} health events", 
                _componentHealth.Count, eventsToKeep.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during metrics collection");
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        _disposed = true;
        _metricsTimer?.Dispose();
        _meter?.Dispose();
        
        _logger.LogInformation("Resilience telemetry collector disposed");
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Circuit breaker states for telemetry
/// </summary>
public enum CircuitBreakerState
{
    Closed = 0,
    Open = 1,
    HalfOpen = 2
}

/// <summary>
/// Component health states
/// </summary>
public enum ComponentHealth
{
    Unknown = 0,
    Healthy = 1,
    Degraded = 2,
    Unhealthy = 3
}

/// <summary>
/// Health event record
/// </summary>
public readonly record struct HealthEvent(
    DateTimeOffset Timestamp,
    string Component,
    ComponentHealth PreviousHealth,
    ComponentHealth NewHealth);

/// <summary>
/// Comprehensive resilience metrics
/// </summary>
public readonly record struct ResilienceMetrics(
    IReadOnlyDictionary<string, ComponentHealth> ComponentHealth,
    IReadOnlyDictionary<string, CircuitBreakerState> CircuitBreakerStates,
    IReadOnlyDictionary<string, double> BulkheadUtilizations,
    IReadOnlyDictionary<string, FallbackLevel> FallbackLevels,
    IReadOnlyDictionary<string, double> RateLimiterStates,
    IReadOnlyList<HealthEvent> RecentHealthEvents,
    DateTimeOffset LastUpdated);