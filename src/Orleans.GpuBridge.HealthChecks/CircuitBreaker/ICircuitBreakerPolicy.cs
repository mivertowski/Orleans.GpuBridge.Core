namespace Orleans.GpuBridge.HealthChecks.CircuitBreaker;

/// <summary>
/// Interface for circuit breaker policy implementations.
/// </summary>
public interface ICircuitBreakerPolicy
{
    /// <summary>
    /// Executes an operation with circuit breaker protection.
    /// </summary>
    /// <typeparam name="TResult">The type of result returned by the operation.</typeparam>
    /// <param name="operation">The operation to execute.</param>
    /// <param name="operationName">The name of the operation for tracking purposes.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The result of the operation.</returns>
    Task<TResult> ExecuteAsync<TResult>(
        Func<Task<TResult>> operation,
        string operationName,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Executes an action with circuit breaker protection.
    /// </summary>
    /// <param name="operation">The action to execute.</param>
    /// <param name="operationName">The name of the operation for tracking purposes.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    Task ExecuteAsync(
        Func<Task> operation,
        string operationName,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the current circuit state for an operation.
    /// </summary>
    /// <param name="operationName">The name of the operation.</param>
    /// <returns>The current circuit state.</returns>
    CircuitState GetCircuitState(string operationName);

    /// <summary>
    /// Resets the circuit to closed state for an operation.
    /// </summary>
    /// <param name="operationName">The name of the operation.</param>
    void Reset(string operationName);

    /// <summary>
    /// Forces the circuit to open state (isolated) for an operation.
    /// </summary>
    /// <param name="operationName">The name of the operation.</param>
    void Isolate(string operationName);
}
