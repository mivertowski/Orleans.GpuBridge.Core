using System;

namespace Orleans.GpuBridge.Resilience.Policies;

/// <summary>
/// Configuration options for GPU resilience policies
/// </summary>
public sealed class GpuResiliencePolicyOptions
{
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

/// <summary>
/// Retry policy configuration options
/// </summary>
public sealed class RetryPolicyOptions
{
    /// <summary>
    /// Maximum number of retry attempts
    /// </summary>
    public int MaxAttempts { get; set; } = 3;

    /// <summary>
    /// Base delay between retries
    /// </summary>
    public TimeSpan BaseDelay { get; set; } = TimeSpan.FromMilliseconds(500);

    /// <summary>
    /// Maximum delay between retries
    /// </summary>
    public TimeSpan MaxDelay { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Whether to use exponential backoff
    /// </summary>
    public bool UseExponentialBackoff { get; set; } = true;

    /// <summary>
    /// Whether to add jitter to retry delays
    /// </summary>
    public bool UseJitter { get; set; } = true;

    /// <summary>
    /// Jitter factor (0.0 to 1.0)
    /// </summary>
    public double JitterFactor { get; set; } = 0.1;
}

/// <summary>
/// Circuit breaker policy configuration options
/// </summary>
public sealed class CircuitBreakerPolicyOptions
{
    /// <summary>
    /// Failure ratio threshold to open circuit (0.0 to 1.0)
    /// </summary>
    public double FailureRatio { get; set; } = 0.5;

    /// <summary>
    /// Sampling duration for failure rate calculation
    /// </summary>
    public TimeSpan SamplingDuration { get; set; } = TimeSpan.FromMinutes(2);

    /// <summary>
    /// Minimum throughput required before opening circuit
    /// </summary>
    public int MinimumThroughput { get; set; } = 10;

    /// <summary>
    /// Duration to keep circuit open
    /// </summary>
    public TimeSpan BreakDuration { get; set; } = TimeSpan.FromMinutes(1);

    /// <summary>
    /// Whether to enable circuit breaker
    /// </summary>
    public bool Enabled { get; set; } = true;
}

/// <summary>
/// Timeout policy configuration options
/// </summary>
public sealed class TimeoutPolicyOptions
{
    /// <summary>
    /// Timeout for kernel execution operations
    /// </summary>
    public TimeSpan KernelExecution { get; set; } = TimeSpan.FromMinutes(5);

    /// <summary>
    /// Timeout for device operations
    /// </summary>
    public TimeSpan DeviceOperation { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Timeout for memory allocation operations
    /// </summary>
    public TimeSpan MemoryAllocation { get; set; } = TimeSpan.FromSeconds(10);

    /// <summary>
    /// Timeout for kernel compilation operations
    /// </summary>
    public TimeSpan KernelCompilation { get; set; } = TimeSpan.FromMinutes(10);

    /// <summary>
    /// Timeout for data transfer operations
    /// </summary>
    public TimeSpan DataTransfer { get; set; } = TimeSpan.FromMinutes(2);

    /// <summary>
    /// Default timeout for unspecified operations
    /// </summary>
    public TimeSpan DefaultOperation { get; set; } = TimeSpan.FromMinutes(1);
}

/// <summary>
/// Bulkhead isolation policy configuration options
/// </summary>
public sealed class BulkheadPolicyOptions
{
    /// <summary>
    /// Maximum concurrent operations allowed
    /// </summary>
    public int MaxConcurrentOperations { get; set; } = 10;

    /// <summary>
    /// Maximum queued operations when at capacity
    /// </summary>
    public int MaxQueuedOperations { get; set; } = 50;

    /// <summary>
    /// Whether to enable bulkhead isolation
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Timeout for waiting to acquire bulkhead slot
    /// </summary>
    public TimeSpan AcquisitionTimeout { get; set; } = TimeSpan.FromSeconds(30);
}

/// <summary>
/// Rate limiting configuration options
/// </summary>
public sealed class RateLimitingOptions
{
    /// <summary>
    /// Maximum requests per time window
    /// </summary>
    public int MaxRequests { get; set; } = 100;

    /// <summary>
    /// Time window for rate limiting
    /// </summary>
    public TimeSpan TimeWindow { get; set; } = TimeSpan.FromMinutes(1);

    /// <summary>
    /// Whether to enable rate limiting
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Rate limiting algorithm type
    /// </summary>
    public RateLimitingAlgorithm Algorithm { get; set; } = RateLimitingAlgorithm.TokenBucket;

    /// <summary>
    /// Token bucket refill rate (tokens per second)
    /// </summary>
    public double TokenRefillRate { get; set; } = 10.0;

    /// <summary>
    /// Maximum burst size for token bucket
    /// </summary>
    public int MaxBurstSize { get; set; } = 20;
}

/// <summary>
/// Rate limiting algorithm types
/// </summary>
public enum RateLimitingAlgorithm
{
    /// <summary>
    /// Token bucket algorithm
    /// </summary>
    TokenBucket,

    /// <summary>
    /// Fixed window algorithm
    /// </summary>
    FixedWindow,

    /// <summary>
    /// Sliding window algorithm
    /// </summary>
    SlidingWindow
}

/// <summary>
/// Chaos engineering configuration options
/// </summary>
public sealed class ChaosEngineeringOptions
{
    /// <summary>
    /// Whether to enable chaos engineering features
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Probability of injecting faults (0.0 to 1.0)
    /// </summary>
    public double FaultInjectionProbability { get; set; } = 0.01;

    /// <summary>
    /// Latency injection configuration
    /// </summary>
    public LatencyInjectionOptions LatencyInjection { get; set; } = new();

    /// <summary>
    /// Exception injection configuration
    /// </summary>
    public ExceptionInjectionOptions ExceptionInjection { get; set; } = new();

    /// <summary>
    /// Resource exhaustion simulation configuration
    /// </summary>
    public ResourceExhaustionOptions ResourceExhaustion { get; set; } = new();
}

/// <summary>
/// Latency injection configuration
/// </summary>
public sealed class LatencyInjectionOptions
{
    /// <summary>
    /// Whether to enable latency injection
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Minimum latency to inject
    /// </summary>
    public TimeSpan MinLatency { get; set; } = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Maximum latency to inject
    /// </summary>
    public TimeSpan MaxLatency { get; set; } = TimeSpan.FromSeconds(5);

    /// <summary>
    /// Probability of injecting latency (0.0 to 1.0)
    /// </summary>
    public double InjectionProbability { get; set; } = 0.05;
}

/// <summary>
/// Exception injection configuration
/// </summary>
public sealed class ExceptionInjectionOptions
{
    /// <summary>
    /// Whether to enable exception injection
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Probability of injecting exceptions (0.0 to 1.0)
    /// </summary>
    public double InjectionProbability { get; set; } = 0.02;

    /// <summary>
    /// Types of exceptions to inject
    /// </summary>
    public string[] ExceptionTypes { get; set; } = 
    {
        "Orleans.GpuBridge.Abstractions.Exceptions.GpuOperationException",
        "Orleans.GpuBridge.Abstractions.Exceptions.GpuMemoryException",
        "System.TimeoutException"
    };
}

/// <summary>
/// Resource exhaustion simulation configuration
/// </summary>
public sealed class ResourceExhaustionOptions
{
    /// <summary>
    /// Whether to enable resource exhaustion simulation
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Probability of simulating memory exhaustion (0.0 to 1.0)
    /// </summary>
    public double MemoryExhaustionProbability { get; set; } = 0.01;

    /// <summary>
    /// Probability of simulating compute exhaustion (0.0 to 1.0)
    /// </summary>
    public double ComputeExhaustionProbability { get; set; } = 0.01;

    /// <summary>
    /// Duration to simulate resource exhaustion
    /// </summary>
    public TimeSpan ExhaustionDuration { get; set; } = TimeSpan.FromSeconds(30);
}