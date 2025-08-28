namespace Orleans.GpuBridge.HealthChecks.CircuitBreaker;

public interface ICircuitBreakerPolicy
{
    Task<TResult> ExecuteAsync<TResult>(
        Func<Task<TResult>> operation,
        string operationName,
        CancellationToken cancellationToken = default);
    
    Task ExecuteAsync(
        Func<Task> operation,
        string operationName,
        CancellationToken cancellationToken = default);
    
    CircuitState GetCircuitState(string operationName);
    void Reset(string operationName);
    void Isolate(string operationName);
}