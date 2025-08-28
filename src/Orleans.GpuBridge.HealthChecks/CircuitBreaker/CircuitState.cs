namespace Orleans.GpuBridge.HealthChecks.CircuitBreaker;

public enum CircuitState
{
    Closed,
    Open,
    HalfOpen,
    Isolated
}