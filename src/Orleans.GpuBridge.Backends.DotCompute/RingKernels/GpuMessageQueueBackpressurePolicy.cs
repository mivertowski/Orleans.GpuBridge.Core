using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Backends.DotCompute.RingKernels;

/// <summary>
/// Implements backpressure policy for GPU message queues to prevent overflow and message loss.
/// Slows down message senders when queue utilization is high.
/// </summary>
/// <remarks>
/// Backpressure Strategy:
/// - Monitor queue utilization in real-time
/// - Apply progressive delays when queue >threshold
/// - Circuit breaker trips when queue reaches capacity
/// - Automatic recovery when queue drains
///
/// Thresholds (default):
/// - <70% utilization: No backpressure
/// - 70-85% utilization: Warning (light delay)
/// - 85-95% utilization: Critical (moderate delay)
/// - >95% utilization: Circuit breaker (reject new messages)
///
/// Benefits:
/// - Prevents message loss from queue overflow
/// - Maintains system stability under load
/// - Provides graceful degradation
/// - Metrics for monitoring queue health
/// </remarks>
public sealed class GpuMessageQueueBackpressurePolicy
{
    private readonly ILogger<GpuMessageQueueBackpressurePolicy> _logger;
    private readonly BackpressureOptions _options;
    private readonly object _stateLock = new();

    private BackpressureState _currentState;
    private long _totalBackpressureEvents;
    private long _totalMessagesRejected;
    private DateTimeOffset _lastStateChange;

    public GpuMessageQueueBackpressurePolicy(
        ILogger<GpuMessageQueueBackpressurePolicy> logger,
        BackpressureOptions? options = null)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options ?? new BackpressureOptions();
        _currentState = BackpressureState.Normal;
        _lastStateChange = DateTimeOffset.UtcNow;

        _logger.LogInformation(
            "Message queue backpressure policy initialized - " +
            "Thresholds: Warning={Warning}%, Critical={Critical}%, CircuitBreaker={CircuitBreaker}%",
            _options.WarningThresholdPercent,
            _options.CriticalThresholdPercent,
            _options.CircuitBreakerThresholdPercent);
    }

    /// <summary>
    /// Checks if a message can be enqueued and applies backpressure if needed.
    /// Returns delay in milliseconds (0 = no delay, -1 = reject message).
    /// </summary>
    public async Task<BackpressureDecision> ShouldEnqueueAsync(
        int currentQueueDepth,
        int queueCapacity,
        CancellationToken ct = default)
    {
        var utilizationPercent = (currentQueueDepth / (double)queueCapacity) * 100.0;

        // Determine backpressure state based on utilization
        var newState = DetermineState(utilizationPercent);

        // Update state if changed
        if (newState != _currentState)
        {
            UpdateState(newState, utilizationPercent);
        }

        // Apply backpressure based on current state
        return newState switch
        {
            BackpressureState.Normal => new BackpressureDecision
            {
                ShouldEnqueue = true,
                DelayMilliseconds = 0,
                State = BackpressureState.Normal,
                Reason = null
            },

            BackpressureState.Warning => await ApplyWarningBackpressureAsync(
                utilizationPercent, ct),

            BackpressureState.Critical => await ApplyCriticalBackpressureAsync(
                utilizationPercent, ct),

            BackpressureState.CircuitBreakerOpen => HandleCircuitBreakerOpen(
                utilizationPercent),

            _ => throw new InvalidOperationException($"Unknown backpressure state: {newState}")
        };
    }

    /// <summary>
    /// Gets current backpressure statistics.
    /// </summary>
    public BackpressureStatistics GetStatistics()
    {
        lock (_stateLock)
        {
            return new BackpressureStatistics
            {
                CurrentState = _currentState,
                TotalBackpressureEvents = _totalBackpressureEvents,
                TotalMessagesRejected = _totalMessagesRejected,
                LastStateChange = _lastStateChange,
                TimeInCurrentState = DateTimeOffset.UtcNow - _lastStateChange
            };
        }
    }

    private BackpressureState DetermineState(double utilizationPercent)
    {
        if (utilizationPercent >= _options.CircuitBreakerThresholdPercent)
            return BackpressureState.CircuitBreakerOpen;

        if (utilizationPercent >= _options.CriticalThresholdPercent)
            return BackpressureState.Critical;

        if (utilizationPercent >= _options.WarningThresholdPercent)
            return BackpressureState.Warning;

        return BackpressureState.Normal;
    }

    private void UpdateState(BackpressureState newState, double utilizationPercent)
    {
        lock (_stateLock)
        {
            var oldState = _currentState;
            _currentState = newState;
            _lastStateChange = DateTimeOffset.UtcNow;

            _logger.LogWarning(
                "Backpressure state changed: {OldState} → {NewState} - " +
                "Queue utilization: {Utilization:F1}%",
                oldState,
                newState,
                utilizationPercent);
        }
    }

    private async Task<BackpressureDecision> ApplyWarningBackpressureAsync(
        double utilizationPercent,
        CancellationToken ct)
    {
        // Light delay - give queue time to drain
        var delayMs = _options.WarningDelayMilliseconds;

        Interlocked.Increment(ref _totalBackpressureEvents);

        _logger.LogDebug(
            "Applying warning backpressure - Delay: {Delay}ms, Utilization: {Utilization:F1}%",
            delayMs,
            utilizationPercent);

        if (delayMs > 0)
        {
            await Task.Delay(delayMs, ct);
        }

        return new BackpressureDecision
        {
            ShouldEnqueue = true,
            DelayMilliseconds = delayMs,
            State = BackpressureState.Warning,
            Reason = $"Queue utilization at {utilizationPercent:F1}% (warning threshold)"
        };
    }

    private async Task<BackpressureDecision> ApplyCriticalBackpressureAsync(
        double utilizationPercent,
        CancellationToken ct)
    {
        // Moderate delay - queue is getting full
        var delayMs = _options.CriticalDelayMilliseconds;

        Interlocked.Increment(ref _totalBackpressureEvents);

        _logger.LogWarning(
            "Applying critical backpressure - Delay: {Delay}ms, Utilization: {Utilization:F1}%",
            delayMs,
            utilizationPercent);

        if (delayMs > 0)
        {
            await Task.Delay(delayMs, ct);
        }

        return new BackpressureDecision
        {
            ShouldEnqueue = true,
            DelayMilliseconds = delayMs,
            State = BackpressureState.Critical,
            Reason = $"Queue utilization at {utilizationPercent:F1}% (critical threshold)"
        };
    }

    private BackpressureDecision HandleCircuitBreakerOpen(double utilizationPercent)
    {
        // Reject message - queue is full
        Interlocked.Increment(ref _totalMessagesRejected);

        _logger.LogError(
            "Circuit breaker OPEN - Rejecting message. " +
            "Queue utilization: {Utilization:F1}%",
            utilizationPercent);

        return new BackpressureDecision
        {
            ShouldEnqueue = false,
            DelayMilliseconds = -1,
            State = BackpressureState.CircuitBreakerOpen,
            Reason = $"Queue utilization at {utilizationPercent:F1}% (circuit breaker threshold)"
        };
    }
}

/// <summary>
/// Backpressure policy configuration options.
/// </summary>
public sealed class BackpressureOptions
{
    /// <summary>
    /// Queue utilization percentage that triggers warning backpressure.
    /// Default: 70%.
    /// </summary>
    public double WarningThresholdPercent { get; set; } = 70.0;

    /// <summary>
    /// Queue utilization percentage that triggers critical backpressure.
    /// Default: 85%.
    /// </summary>
    public double CriticalThresholdPercent { get; set; } = 85.0;

    /// <summary>
    /// Queue utilization percentage that trips circuit breaker (rejects messages).
    /// Default: 95%.
    /// </summary>
    public double CircuitBreakerThresholdPercent { get; set; } = 95.0;

    /// <summary>
    /// Delay applied during warning backpressure in milliseconds.
    /// Default: 10ms.
    /// </summary>
    public int WarningDelayMilliseconds { get; set; } = 10;

    /// <summary>
    /// Delay applied during critical backpressure in milliseconds.
    /// Default: 50ms.
    /// </summary>
    public int CriticalDelayMilliseconds { get; set; } = 50;
}

/// <summary>
/// Backpressure state indicating queue health.
/// </summary>
public enum BackpressureState
{
    /// <summary>Queue utilization is normal (<70%).</summary>
    Normal,

    /// <summary>Queue utilization is elevated (70-85%) - light backpressure.</summary>
    Warning,

    /// <summary>Queue utilization is high (85-95%) - moderate backpressure.</summary>
    Critical,

    /// <summary>Queue is near capacity (>95%) - rejecting messages.</summary>
    CircuitBreakerOpen
}

/// <summary>
/// Decision about whether to enqueue a message and any required delay.
/// </summary>
public sealed class BackpressureDecision
{
    /// <summary>Whether the message should be enqueued.</summary>
    public required bool ShouldEnqueue { get; init; }

    /// <summary>Delay in milliseconds before enqueuing (-1 = rejected).</summary>
    public required int DelayMilliseconds { get; init; }

    /// <summary>Current backpressure state.</summary>
    public required BackpressureState State { get; init; }

    /// <summary>Reason for the decision (if applicable).</summary>
    public required string? Reason { get; init; }
}

/// <summary>
/// Statistics about backpressure policy behavior.
/// </summary>
public sealed class BackpressureStatistics
{
    public required BackpressureState CurrentState { get; init; }
    public required long TotalBackpressureEvents { get; init; }
    public required long TotalMessagesRejected { get; init; }
    public required DateTimeOffset LastStateChange { get; init; }
    public required TimeSpan TimeInCurrentState { get; init; }
}
