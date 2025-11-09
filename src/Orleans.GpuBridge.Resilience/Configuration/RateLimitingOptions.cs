using System;

namespace Orleans.GpuBridge.Resilience.Policies;

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
