namespace Orleans.GpuBridge.HealthChecks.CircuitBreaker;

/// <summary>
/// Configuration options for circuit breaker resilience pattern.
/// </summary>
public class CircuitBreakerOptions
{
    /// <summary>
    /// Gets or sets the number of consecutive failures before the circuit opens.
    /// </summary>
    public int FailureThreshold { get; set; } = 3;

    /// <summary>
    /// Gets or sets the duration the circuit stays open before attempting recovery.
    /// </summary>
    public TimeSpan BreakDuration { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Gets or sets the number of retry attempts for failed operations.
    /// </summary>
    public int RetryCount { get; set; } = 3;

    /// <summary>
    /// Gets or sets the base delay between retry attempts in milliseconds.
    /// </summary>
    public int RetryDelayMs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the timeout duration for individual operations.
    /// </summary>
    public TimeSpan OperationTimeout { get; set; } = TimeSpan.FromSeconds(30);
}