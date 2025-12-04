using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions.Exceptions;
using Orleans.GpuBridge.Resilience.Policies;

namespace Orleans.GpuBridge.Resilience.RateLimit;

/// <summary>
/// Thread-safe token bucket rate limiter for GPU operations
/// </summary>
public sealed class TokenBucketRateLimiter : IRateLimiter, IDisposable
{
    private readonly ILogger<TokenBucketRateLimiter> _logger;
    private readonly RateLimitingOptions _options;
    private readonly Timer _refillTimer;
    private readonly object _lock = new();
    private double _tokens;
    private DateTimeOffset _lastRefill;
    private long _totalRequests;
    private long _rejectedRequests;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="TokenBucketRateLimiter"/> class.
    /// </summary>
    /// <param name="logger">The logger instance for rate limiter logging.</param>
    /// <param name="options">The rate limiting configuration options.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="logger"/> or <paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when token refill rate or max burst size is not positive.</exception>
    public TokenBucketRateLimiter(
        ILogger<TokenBucketRateLimiter> logger,
        IOptions<RateLimitingOptions> options)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
        
        if (_options.TokenRefillRate <= 0)
            throw new ArgumentException("Token refill rate must be positive", nameof(options));
        
        if (_options.MaxBurstSize <= 0)
            throw new ArgumentException("Max burst size must be positive", nameof(options));

        _tokens = _options.MaxBurstSize;
        _lastRefill = DateTimeOffset.UtcNow;
        
        // Start refill timer
        var refillInterval = TimeSpan.FromMilliseconds(100); // Refill every 100ms for smooth rate limiting
        _refillTimer = new Timer(RefillTokens, null, refillInterval, refillInterval);
        
        _logger.LogInformation(
            "Token bucket rate limiter initialized: {RefillRate} tokens/sec, {BurstSize} max burst",
            _options.TokenRefillRate, _options.MaxBurstSize);
    }

    /// <summary>
    /// Attempts to acquire a token for rate limiting
    /// </summary>
    public async Task<bool> TryAcquireAsync(int tokens = 1, CancellationToken cancellationToken = default)
    {
        if (!_options.Enabled || _disposed)
            return true;

        Interlocked.Increment(ref _totalRequests);

        lock (_lock)
        {
            RefillTokensInternal();
            
            if (_tokens >= tokens)
            {
                _tokens -= tokens;
                return true;
            }
            
            Interlocked.Increment(ref _rejectedRequests);
            return false;
        }
    }

    /// <summary>
    /// Waits for tokens to become available or throws if rate limit exceeded
    /// </summary>
    public async Task AcquireAsync(int tokens = 1, CancellationToken cancellationToken = default)
    {
        if (!_options.Enabled || _disposed)
            return;

        var startTime = DateTimeOffset.UtcNow;
        var maxWaitTime = TimeSpan.FromSeconds(30); // Maximum wait time
        
        while (!cancellationToken.IsCancellationRequested)
        {
            if (await TryAcquireAsync(tokens, cancellationToken))
            {
                return;
            }
            
            // Check if we've been waiting too long
            if (DateTimeOffset.UtcNow - startTime > maxWaitTime)
            {
                var metrics = GetMetrics();
                throw new RateLimitExceededException(
                    (int)metrics.TotalRequests,
                    _options.MaxBurstSize,
                    TimeSpan.FromSeconds(1.0 / _options.TokenRefillRate * tokens));
            }
            
            // Wait before retrying
            var waitTime = TimeSpan.FromMilliseconds(Math.Max(10, 1000.0 / _options.TokenRefillRate));
            await Task.Delay(waitTime, cancellationToken);
        }
        
        cancellationToken.ThrowIfCancellationRequested();
    }

    /// <summary>
    /// Executes an operation with rate limiting
    /// </summary>
    public async Task<T> ExecuteAsync<T>(
        Func<CancellationToken, Task<T>> operation,
        int tokens = 1,
        CancellationToken cancellationToken = default)
    {
        await AcquireAsync(tokens, cancellationToken);
        return await operation(cancellationToken);
    }

    /// <summary>
    /// Gets current rate limiter metrics
    /// </summary>
    public RateLimiterMetrics GetMetrics()
    {
        lock (_lock)
        {
            RefillTokensInternal();
            
            var rejectionRate = _totalRequests == 0 ? 0.0 : (double)_rejectedRequests / _totalRequests;
            
            return new RateLimiterMetrics(
                TotalRequests: _totalRequests,
                RejectedRequests: _rejectedRequests,
                RejectionRate: rejectionRate,
                AvailableTokens: _tokens,
                MaxTokens: _options.MaxBurstSize,
                RefillRate: _options.TokenRefillRate,
                Utilization: 1.0 - (_tokens / _options.MaxBurstSize));
        }
    }

    /// <summary>
    /// Resets the rate limiter statistics (tokens remain unchanged)
    /// </summary>
    public void ResetStats()
    {
        Interlocked.Exchange(ref _totalRequests, 0);
        Interlocked.Exchange(ref _rejectedRequests, 0);
        
        _logger.LogInformation("Rate limiter statistics reset");
    }

    /// <summary>
    /// Adds tokens to the bucket (for testing or manual adjustment)
    /// </summary>
    public void AddTokens(double tokens)
    {
        if (tokens <= 0 || _disposed) return;
        
        lock (_lock)
        {
            _tokens = Math.Min(_tokens + tokens, _options.MaxBurstSize);
        }
        
        _logger.LogDebug("Added {TokenCount} tokens to bucket", tokens);
    }

    /// <summary>
    /// Timer callback to refill tokens
    /// </summary>
    private void RefillTokens(object? state)
    {
        if (_disposed) return;
        
        lock (_lock)
        {
            RefillTokensInternal();
        }
    }

    /// <summary>
    /// Internal method to refill tokens (must be called within lock)
    /// </summary>
    private void RefillTokensInternal()
    {
        var now = DateTimeOffset.UtcNow;
        var timeSinceLastRefill = now - _lastRefill;
        
        if (timeSinceLastRefill.TotalSeconds > 0)
        {
            var tokensToAdd = _options.TokenRefillRate * timeSinceLastRefill.TotalSeconds;
            _tokens = Math.Min(_tokens + tokensToAdd, _options.MaxBurstSize);
            _lastRefill = now;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;

        _disposed = true;
        _refillTimer?.Dispose();

        _logger.LogInformation("Token bucket rate limiter disposed");
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Interface for rate limiting implementations
/// </summary>
public interface IRateLimiter
{
    /// <summary>
    /// Attempts to acquire tokens without waiting
    /// </summary>
    Task<bool> TryAcquireAsync(int tokens = 1, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Waits for tokens to become available
    /// </summary>
    Task AcquireAsync(int tokens = 1, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Executes an operation with rate limiting
    /// </summary>
    Task<T> ExecuteAsync<T>(Func<CancellationToken, Task<T>> operation, int tokens = 1, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets current metrics
    /// </summary>
    RateLimiterMetrics GetMetrics();
}

/// <summary>
/// Rate limiter metrics
/// </summary>
public readonly record struct RateLimiterMetrics(
    long TotalRequests,
    long RejectedRequests,
    double RejectionRate,
    double AvailableTokens,
    double MaxTokens,
    double RefillRate,
    double Utilization);