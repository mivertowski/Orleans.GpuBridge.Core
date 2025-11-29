namespace Orleans.GpuBridge.HealthChecks.CircuitBreaker;

/// <summary>
/// Represents the state of a circuit breaker.
/// </summary>
public enum CircuitState
{
    /// <summary>
    /// Circuit is closed, operations are allowed to execute normally.
    /// </summary>
    Closed,

    /// <summary>
    /// Circuit is open, operations are blocked due to failures.
    /// </summary>
    Open,

    /// <summary>
    /// Circuit is half-open, testing if the system has recovered.
    /// </summary>
    HalfOpen,

    /// <summary>
    /// Circuit is manually isolated, operations are blocked.
    /// </summary>
    Isolated
}