namespace Orleans.GpuBridge.Resilience.Policies;

/// <summary>
/// Configuration options for GPU resilience policies
/// </summary>
public sealed class GpuResiliencePolicyOptions
{
    /// <summary>
    /// The configuration section name for GPU resilience policy options.
    /// </summary>
    public const string SectionName = "GpuResilience";

    /// <summary>
    /// Retry policy configuration
    /// </summary>
    public RetryPolicyOptions RetryOptions { get; set; } = new();

    /// <summary>
    /// Circuit breaker configuration
    /// </summary>
    public CircuitBreakerPolicyOptions CircuitBreakerOptions { get; set; } = new();

    /// <summary>
    /// Timeout policy configuration
    /// </summary>
    public TimeoutPolicyOptions TimeoutOptions { get; set; } = new();

    /// <summary>
    /// Bulkhead isolation configuration
    /// </summary>
    public BulkheadPolicyOptions BulkheadOptions { get; set; } = new();

    /// <summary>
    /// Rate limiting configuration
    /// </summary>
    public RateLimitingOptions RateLimitingOptions { get; set; } = new();

    /// <summary>
    /// Chaos engineering configuration
    /// </summary>
    public ChaosEngineeringOptions ChaosOptions { get; set; } = new();
}