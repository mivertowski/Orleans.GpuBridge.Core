namespace Orleans.GpuBridge.HealthChecks.CircuitBreaker;

public class CircuitBreakerOptions
{
    public int FailureThreshold { get; set; } = 3;
    public TimeSpan BreakDuration { get; set; } = TimeSpan.FromSeconds(30);
    public int RetryCount { get; set; } = 3;
    public int RetryDelayMs { get; set; } = 100;
    public TimeSpan OperationTimeout { get; set; } = TimeSpan.FromSeconds(30);
}