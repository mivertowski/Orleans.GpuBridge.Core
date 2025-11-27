// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Runtime.Temporal.FaultTolerance;

/// <summary>
/// Handles network failures with configurable exponential backoff and jitter.
/// </summary>
/// <remarks>
/// <para>
/// <b>Features:</b>
/// </para>
/// <list type="bullet">
/// <item>Exponential backoff with configurable base and max delay</item>
/// <item>Random jitter to prevent thundering herd</item>
/// <item>Circuit breaker pattern for persistent failures</item>
/// <item>Retry statistics and telemetry</item>
/// <item>Cancellation support</item>
/// </list>
/// <para>
/// <b>Retry Formula:</b>
/// <code>
/// delay = min(maxDelay, baseDelay * 2^attempt) * (1 + random(-jitter, +jitter))
/// </code>
/// </para>
/// </remarks>
public sealed class NetworkRetryHandler : IDisposable
{
    private readonly ILogger<NetworkRetryHandler> _logger;
    private readonly NetworkRetryOptions _options;
    private readonly Random _jitterRandom = new();
    private readonly object _lock = new();

    private long _totalAttempts;
    private long _totalSuccesses;
    private long _totalFailures;
    private long _circuitBreakerTrips;
    private CircuitState _circuitState = CircuitState.Closed;
    private DateTimeOffset _lastFailureTime = DateTimeOffset.MinValue;
    private DateTimeOffset _circuitOpenedAt = DateTimeOffset.MinValue;
    private int _consecutiveFailures;
    private bool _disposed;

    /// <summary>
    /// Gets the current circuit breaker state.
    /// </summary>
    public CircuitState CurrentCircuitState
    {
        get
        {
            lock (_lock)
            {
                return _circuitState;
            }
        }
    }

    /// <summary>
    /// Gets whether the circuit is allowing requests.
    /// </summary>
    public bool IsAllowingRequests
    {
        get
        {
            lock (_lock)
            {
                return _circuitState != CircuitState.Open ||
                       ShouldAttemptReset();
            }
        }
    }

    /// <summary>
    /// Gets total retry attempts made.
    /// </summary>
    public long TotalAttempts => Interlocked.Read(ref _totalAttempts);

    /// <summary>
    /// Gets total successful operations.
    /// </summary>
    public long TotalSuccesses => Interlocked.Read(ref _totalSuccesses);

    /// <summary>
    /// Gets total failed operations (after all retries exhausted).
    /// </summary>
    public long TotalFailures => Interlocked.Read(ref _totalFailures);

    /// <summary>
    /// Gets number of times circuit breaker was tripped.
    /// </summary>
    public long CircuitBreakerTrips => Interlocked.Read(ref _circuitBreakerTrips);

    /// <summary>
    /// Occurs when a retry is attempted.
    /// </summary>
    public event EventHandler<RetryAttemptEventArgs>? RetryAttempted;

    /// <summary>
    /// Occurs when all retries are exhausted.
    /// </summary>
    public event EventHandler<RetryExhaustedEventArgs>? RetryExhausted;

    /// <summary>
    /// Occurs when the circuit breaker state changes.
    /// </summary>
    public event EventHandler<CircuitStateChangedEventArgs>? CircuitStateChanged;

    /// <summary>
    /// Initializes a new network retry handler.
    /// </summary>
    /// <param name="logger">Logger for diagnostic messages.</param>
    /// <param name="options">Retry configuration options.</param>
    public NetworkRetryHandler(
        ILogger<NetworkRetryHandler> logger,
        NetworkRetryOptions? options = null)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options ?? NetworkRetryOptions.Default;

        _logger.LogInformation(
            "NetworkRetryHandler initialized: MaxRetries={MaxRetries}, BaseDelay={BaseDelay}ms, MaxDelay={MaxDelay}ms",
            _options.MaxRetryAttempts,
            _options.BaseDelay.TotalMilliseconds,
            _options.MaxDelay.TotalMilliseconds);
    }

    /// <summary>
    /// Executes an async operation with retry logic.
    /// </summary>
    /// <typeparam name="T">Return type of the operation.</typeparam>
    /// <param name="operation">The operation to execute.</param>
    /// <param name="operationName">Name for logging purposes.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The operation result.</returns>
    /// <exception cref="RetryExhaustedException">Thrown when all retries are exhausted.</exception>
    public async Task<T> ExecuteAsync<T>(
        Func<CancellationToken, Task<T>> operation,
        string operationName,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(operation);
        ArgumentException.ThrowIfNullOrEmpty(operationName);

        // Check circuit breaker
        if (!IsAllowingRequests)
        {
            throw new CircuitBreakerOpenException(
                $"Circuit breaker is open for operation: {operationName}");
        }

        Exception? lastException = null;
        var startTime = Stopwatch.GetTimestamp();

        for (int attempt = 0; attempt <= _options.MaxRetryAttempts; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            Interlocked.Increment(ref _totalAttempts);

            try
            {
                // Half-open: Test with single request
                if (CurrentCircuitState == CircuitState.HalfOpen)
                {
                    _logger.LogDebug(
                        "Circuit half-open, attempting probe request for {Operation}",
                        operationName);
                }

                var result = await operation(cancellationToken);

                // Success - reset failure tracking
                RecordSuccess();
                Interlocked.Increment(ref _totalSuccesses);

                var elapsed = Stopwatch.GetElapsedTime(startTime);
                _logger.LogDebug(
                    "Operation {Operation} succeeded after {Attempts} attempt(s), elapsed: {Elapsed}ms",
                    operationName,
                    attempt + 1,
                    elapsed.TotalMilliseconds);

                return result;
            }
            catch (OperationCanceledException)
            {
                throw; // Don't retry cancellation
            }
            catch (Exception ex) when (ShouldRetry(ex, attempt))
            {
                lastException = ex;
                RecordFailure();

                var delay = CalculateDelay(attempt);
                RaiseRetryAttemptEvent(attempt, delay, ex, operationName);

                _logger.LogWarning(
                    "Operation {Operation} failed (attempt {Attempt}/{MaxAttempts}), retrying in {Delay}ms. Error: {Error}",
                    operationName,
                    attempt + 1,
                    _options.MaxRetryAttempts + 1,
                    delay.TotalMilliseconds,
                    ex.Message);

                if (attempt < _options.MaxRetryAttempts)
                {
                    await Task.Delay(delay, cancellationToken);
                }
            }
            catch (Exception ex)
            {
                // Non-retryable exception
                lastException = ex;
                RecordFailure();
                Interlocked.Increment(ref _totalFailures);

                _logger.LogError(ex,
                    "Operation {Operation} failed with non-retryable exception",
                    operationName);

                throw;
            }
        }

        // All retries exhausted
        Interlocked.Increment(ref _totalFailures);
        var totalElapsed = Stopwatch.GetElapsedTime(startTime);

        RaiseRetryExhaustedEvent(_options.MaxRetryAttempts + 1, totalElapsed, lastException!, operationName);

        _logger.LogError(
            "Operation {Operation} failed after {Attempts} attempts, total elapsed: {Elapsed}ms",
            operationName,
            _options.MaxRetryAttempts + 1,
            totalElapsed.TotalMilliseconds);

        throw new RetryExhaustedException(
            operationName,
            _options.MaxRetryAttempts + 1,
            totalElapsed,
            lastException!);
    }

    /// <summary>
    /// Executes an async operation with retry logic (no return value).
    /// </summary>
    /// <param name="operation">The operation to execute.</param>
    /// <param name="operationName">Name for logging purposes.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task ExecuteAsync(
        Func<CancellationToken, Task> operation,
        string operationName,
        CancellationToken cancellationToken = default)
    {
        await ExecuteAsync(
            async ct =>
            {
                await operation(ct);
                return true; // Dummy return value
            },
            operationName,
            cancellationToken);
    }

    /// <summary>
    /// Calculates the delay for a retry attempt with exponential backoff and jitter.
    /// </summary>
    /// <param name="attempt">Zero-based attempt number.</param>
    /// <returns>Delay duration.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TimeSpan CalculateDelay(int attempt)
    {
        // Exponential backoff: baseDelay * 2^attempt
        var exponentialDelay = _options.BaseDelay.TotalMilliseconds * Math.Pow(2, attempt);

        // Cap at max delay
        var cappedDelay = Math.Min(exponentialDelay, _options.MaxDelay.TotalMilliseconds);

        // Add jitter (-jitter% to +jitter%)
        double jitter;
        lock (_jitterRandom)
        {
            jitter = (_jitterRandom.NextDouble() * 2 - 1) * _options.JitterFactor;
        }

        var finalDelay = cappedDelay * (1 + jitter);
        return TimeSpan.FromMilliseconds(Math.Max(finalDelay, 1));
    }

    /// <summary>
    /// Manually resets the circuit breaker to closed state.
    /// </summary>
    public void ResetCircuitBreaker()
    {
        lock (_lock)
        {
            var previousState = _circuitState;
            _circuitState = CircuitState.Closed;
            _consecutiveFailures = 0;

            if (previousState != CircuitState.Closed)
            {
                RaiseCircuitStateChangedEvent(previousState, CircuitState.Closed);
            }
        }

        _logger.LogInformation("Circuit breaker manually reset to closed state");
    }

    /// <summary>
    /// Gets current retry statistics.
    /// </summary>
    public NetworkRetryStatistics GetStatistics()
    {
        return new NetworkRetryStatistics
        {
            TotalAttempts = TotalAttempts,
            TotalSuccesses = TotalSuccesses,
            TotalFailures = TotalFailures,
            CircuitBreakerTrips = CircuitBreakerTrips,
            CurrentCircuitState = CurrentCircuitState,
            SuccessRate = TotalAttempts > 0
                ? (double)TotalSuccesses / TotalAttempts
                : 1.0,
            LastFailureTime = _lastFailureTime,
            ConsecutiveFailures = _consecutiveFailures
        };
    }

    private bool ShouldRetry(Exception exception, int attempt)
    {
        // Only check if exception type is retryable
        // Attempt count is handled in the retry loop
        var isRetryable = _options.RetryableExceptionPredicate?.Invoke(exception) ??
                          IsDefaultRetryableException(exception);

        // If not retryable, don't retry regardless of attempt count
        if (!isRetryable)
        {
            return false;
        }

        // If retryable but at max attempts, still return true to enter retry handler
        // (which will not delay and let loop exit to exhaustion handler)
        return true;
    }

    private static bool IsDefaultRetryableException(Exception exception)
    {
        // Default retryable exceptions
        return exception is TimeoutException ||
               exception is System.Net.Sockets.SocketException ||
               exception is System.Net.Http.HttpRequestException ||
               exception is System.IO.IOException;
    }

    private void RecordSuccess()
    {
        lock (_lock)
        {
            _consecutiveFailures = 0;

            if (_circuitState == CircuitState.HalfOpen)
            {
                var previousState = _circuitState;
                _circuitState = CircuitState.Closed;
                RaiseCircuitStateChangedEvent(previousState, CircuitState.Closed);

                _logger.LogInformation("Circuit breaker closed after successful probe");
            }
        }
    }

    private void RecordFailure()
    {
        lock (_lock)
        {
            _consecutiveFailures++;
            _lastFailureTime = DateTimeOffset.UtcNow;

            // Check if we should trip the circuit breaker
            if (_circuitState == CircuitState.Closed &&
                _consecutiveFailures >= _options.CircuitBreakerThreshold)
            {
                TripCircuitBreaker();
            }
            else if (_circuitState == CircuitState.HalfOpen)
            {
                // Failed during probe - reopen circuit
                TripCircuitBreaker();
            }
        }
    }

    private void TripCircuitBreaker()
    {
        var previousState = _circuitState;
        _circuitState = CircuitState.Open;
        _circuitOpenedAt = DateTimeOffset.UtcNow;
        Interlocked.Increment(ref _circuitBreakerTrips);

        RaiseCircuitStateChangedEvent(previousState, CircuitState.Open);

        _logger.LogWarning(
            "Circuit breaker tripped after {Failures} consecutive failures",
            _consecutiveFailures);
    }

    private bool ShouldAttemptReset()
    {
        if (_circuitState != CircuitState.Open)
        {
            return false;
        }

        // Check if enough time has passed to try resetting
        var elapsed = DateTimeOffset.UtcNow - _circuitOpenedAt;
        if (elapsed >= _options.CircuitBreakerResetTimeout)
        {
            lock (_lock)
            {
                if (_circuitState == CircuitState.Open)
                {
                    var previousState = _circuitState;
                    _circuitState = CircuitState.HalfOpen;
                    RaiseCircuitStateChangedEvent(previousState, CircuitState.HalfOpen);

                    _logger.LogInformation(
                        "Circuit breaker transitioning to half-open after {Elapsed}ms",
                        elapsed.TotalMilliseconds);
                }
            }

            return true;
        }

        return false;
    }

    private void RaiseRetryAttemptEvent(int attempt, TimeSpan delay, Exception exception, string operationName)
    {
        try
        {
            RetryAttempted?.Invoke(this, new RetryAttemptEventArgs
            {
                Attempt = attempt,
                Delay = delay,
                Exception = exception,
                OperationName = operationName,
                Timestamp = DateTimeOffset.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in RetryAttempted event handler");
        }
    }

    private void RaiseRetryExhaustedEvent(int totalAttempts, TimeSpan totalElapsed, Exception lastException, string operationName)
    {
        try
        {
            RetryExhausted?.Invoke(this, new RetryExhaustedEventArgs
            {
                TotalAttempts = totalAttempts,
                TotalElapsed = totalElapsed,
                LastException = lastException,
                OperationName = operationName,
                Timestamp = DateTimeOffset.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in RetryExhausted event handler");
        }
    }

    private void RaiseCircuitStateChangedEvent(CircuitState previousState, CircuitState newState)
    {
        try
        {
            CircuitStateChanged?.Invoke(this, new CircuitStateChangedEventArgs
            {
                PreviousState = previousState,
                NewState = newState,
                ChangedAt = DateTimeOffset.UtcNow,
                ConsecutiveFailures = _consecutiveFailures
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in CircuitStateChanged event handler");
        }
    }

    /// <summary>
    /// Disposes resources used by the retry handler.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;

        _disposed = true;

        _logger.LogDebug(
            "NetworkRetryHandler disposed. Stats: {Attempts} attempts, {Successes} successes, {Failures} failures, {Trips} circuit trips",
            TotalAttempts,
            TotalSuccesses,
            TotalFailures,
            CircuitBreakerTrips);
    }
}

/// <summary>
/// Configuration options for network retry handling.
/// </summary>
public sealed class NetworkRetryOptions
{
    /// <summary>
    /// Maximum number of retry attempts (not including the initial attempt).
    /// Default: 3.
    /// </summary>
    public int MaxRetryAttempts { get; init; } = 3;

    /// <summary>
    /// Base delay for exponential backoff.
    /// Default: 100ms.
    /// </summary>
    public TimeSpan BaseDelay { get; init; } = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Maximum delay cap for exponential backoff.
    /// Default: 30 seconds.
    /// </summary>
    public TimeSpan MaxDelay { get; init; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Jitter factor for randomizing delays (0-1).
    /// Default: 0.2 (±20%).
    /// </summary>
    public double JitterFactor { get; init; } = 0.2;

    /// <summary>
    /// Number of consecutive failures before tripping circuit breaker.
    /// Default: 5.
    /// </summary>
    public int CircuitBreakerThreshold { get; init; } = 5;

    /// <summary>
    /// How long to wait before attempting to close the circuit.
    /// Default: 30 seconds.
    /// </summary>
    public TimeSpan CircuitBreakerResetTimeout { get; init; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Custom predicate to determine if an exception is retryable.
    /// If null, uses default retryable exception types.
    /// </summary>
    public Func<Exception, bool>? RetryableExceptionPredicate { get; init; }

    /// <summary>
    /// Default retry options suitable for production use.
    /// </summary>
    public static NetworkRetryOptions Default { get; } = new();

    /// <summary>
    /// Aggressive retry options for high-availability scenarios.
    /// </summary>
    public static NetworkRetryOptions HighAvailability { get; } = new()
    {
        MaxRetryAttempts = 5,
        BaseDelay = TimeSpan.FromMilliseconds(50),
        MaxDelay = TimeSpan.FromSeconds(10),
        JitterFactor = 0.3,
        CircuitBreakerThreshold = 10,
        CircuitBreakerResetTimeout = TimeSpan.FromSeconds(15)
    };

    /// <summary>
    /// Conservative retry options for latency-sensitive scenarios.
    /// </summary>
    public static NetworkRetryOptions LowLatency { get; } = new()
    {
        MaxRetryAttempts = 2,
        BaseDelay = TimeSpan.FromMilliseconds(10),
        MaxDelay = TimeSpan.FromMilliseconds(100),
        JitterFactor = 0.1,
        CircuitBreakerThreshold = 3,
        CircuitBreakerResetTimeout = TimeSpan.FromSeconds(5)
    };
}

/// <summary>
/// Circuit breaker states.
/// </summary>
public enum CircuitState
{
    /// <summary>
    /// Circuit is closed - requests are allowed through.
    /// </summary>
    Closed,

    /// <summary>
    /// Circuit is open - requests are blocked.
    /// </summary>
    Open,

    /// <summary>
    /// Circuit is half-open - testing with single request.
    /// </summary>
    HalfOpen
}

/// <summary>
/// Statistics about network retry operations.
/// </summary>
public sealed class NetworkRetryStatistics
{
    /// <summary>
    /// Total retry attempts made.
    /// </summary>
    public required long TotalAttempts { get; init; }

    /// <summary>
    /// Total successful operations.
    /// </summary>
    public required long TotalSuccesses { get; init; }

    /// <summary>
    /// Total failed operations (after all retries).
    /// </summary>
    public required long TotalFailures { get; init; }

    /// <summary>
    /// Times circuit breaker was tripped.
    /// </summary>
    public required long CircuitBreakerTrips { get; init; }

    /// <summary>
    /// Current circuit breaker state.
    /// </summary>
    public required CircuitState CurrentCircuitState { get; init; }

    /// <summary>
    /// Success rate (0-1).
    /// </summary>
    public required double SuccessRate { get; init; }

    /// <summary>
    /// Time of last failure.
    /// </summary>
    public required DateTimeOffset LastFailureTime { get; init; }

    /// <summary>
    /// Current consecutive failure count.
    /// </summary>
    public required int ConsecutiveFailures { get; init; }
}

/// <summary>
/// Event arguments for retry attempts.
/// </summary>
public sealed class RetryAttemptEventArgs : EventArgs
{
    /// <summary>
    /// Zero-based attempt number.
    /// </summary>
    public required int Attempt { get; init; }

    /// <summary>
    /// Delay before next retry.
    /// </summary>
    public required TimeSpan Delay { get; init; }

    /// <summary>
    /// Exception that triggered the retry.
    /// </summary>
    public required Exception Exception { get; init; }

    /// <summary>
    /// Name of the operation being retried.
    /// </summary>
    public required string OperationName { get; init; }

    /// <summary>
    /// When the retry was attempted.
    /// </summary>
    public required DateTimeOffset Timestamp { get; init; }
}

/// <summary>
/// Event arguments when all retries are exhausted.
/// </summary>
public sealed class RetryExhaustedEventArgs : EventArgs
{
    /// <summary>
    /// Total number of attempts made.
    /// </summary>
    public required int TotalAttempts { get; init; }

    /// <summary>
    /// Total time elapsed across all attempts.
    /// </summary>
    public required TimeSpan TotalElapsed { get; init; }

    /// <summary>
    /// Last exception thrown.
    /// </summary>
    public required Exception LastException { get; init; }

    /// <summary>
    /// Name of the operation that failed.
    /// </summary>
    public required string OperationName { get; init; }

    /// <summary>
    /// When retries were exhausted.
    /// </summary>
    public required DateTimeOffset Timestamp { get; init; }
}

/// <summary>
/// Event arguments for circuit state changes.
/// </summary>
public sealed class CircuitStateChangedEventArgs : EventArgs
{
    /// <summary>
    /// Previous circuit state.
    /// </summary>
    public required CircuitState PreviousState { get; init; }

    /// <summary>
    /// New circuit state.
    /// </summary>
    public required CircuitState NewState { get; init; }

    /// <summary>
    /// When the state changed.
    /// </summary>
    public required DateTimeOffset ChangedAt { get; init; }

    /// <summary>
    /// Consecutive failures at time of change.
    /// </summary>
    public required int ConsecutiveFailures { get; init; }
}

/// <summary>
/// Exception thrown when all retry attempts are exhausted.
/// </summary>
public sealed class RetryExhaustedException : Exception
{
    /// <summary>
    /// Name of the operation that failed.
    /// </summary>
    public string OperationName { get; }

    /// <summary>
    /// Total number of attempts made.
    /// </summary>
    public int TotalAttempts { get; }

    /// <summary>
    /// Total elapsed time across all attempts.
    /// </summary>
    public TimeSpan TotalElapsed { get; }

    /// <summary>
    /// Creates a new retry exhausted exception.
    /// </summary>
    public RetryExhaustedException()
        : base("Retry attempts exhausted")
    {
        OperationName = "Unknown";
    }

    /// <summary>
    /// Creates a new retry exhausted exception with a message.
    /// </summary>
    /// <param name="message">Exception message.</param>
    public RetryExhaustedException(string message)
        : base(message)
    {
        OperationName = "Unknown";
    }

    /// <summary>
    /// Creates a new retry exhausted exception with a message and inner exception.
    /// </summary>
    /// <param name="message">Exception message.</param>
    /// <param name="innerException">Inner exception.</param>
    public RetryExhaustedException(string message, Exception innerException)
        : base(message, innerException)
    {
        OperationName = "Unknown";
    }

    /// <summary>
    /// Creates a new retry exhausted exception with details.
    /// </summary>
    /// <param name="operationName">Name of the operation.</param>
    /// <param name="totalAttempts">Total attempts made.</param>
    /// <param name="totalElapsed">Total time elapsed.</param>
    /// <param name="innerException">The last exception thrown.</param>
    public RetryExhaustedException(
        string operationName,
        int totalAttempts,
        TimeSpan totalElapsed,
        Exception innerException)
        : base($"Operation '{operationName}' failed after {totalAttempts} attempts over {totalElapsed.TotalMilliseconds:F0}ms", innerException)
    {
        OperationName = operationName;
        TotalAttempts = totalAttempts;
        TotalElapsed = totalElapsed;
    }
}

/// <summary>
/// Exception thrown when the circuit breaker is open.
/// </summary>
public sealed class CircuitBreakerOpenException : Exception
{
    /// <summary>
    /// Creates a new circuit breaker open exception.
    /// </summary>
    public CircuitBreakerOpenException()
        : base("Circuit breaker is open")
    {
    }

    /// <summary>
    /// Creates a new circuit breaker open exception with a message.
    /// </summary>
    /// <param name="message">Exception message.</param>
    public CircuitBreakerOpenException(string message)
        : base(message)
    {
    }

    /// <summary>
    /// Creates a new circuit breaker open exception with a message and inner exception.
    /// </summary>
    /// <param name="message">Exception message.</param>
    /// <param name="innerException">Inner exception.</param>
    public CircuitBreakerOpenException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
