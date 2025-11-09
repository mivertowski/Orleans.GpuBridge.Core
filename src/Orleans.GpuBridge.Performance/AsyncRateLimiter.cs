using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// High-performance async rate limiter
/// </summary>
public sealed class AsyncRateLimiter : IDisposable
{
    private readonly SemaphoreSlim _semaphore;
    private readonly Timer _refillTimer;
    private readonly int _maxTokens;
    private readonly int _refillRate;
    private volatile int _currentTokens;

    public AsyncRateLimiter(int maxTokens, int refillRate, TimeSpan refillInterval)
    {
        _maxTokens = maxTokens;
        _refillRate = refillRate;
        _currentTokens = maxTokens;
        _semaphore = new SemaphoreSlim(maxTokens, maxTokens);

        _refillTimer = new Timer(RefillTokens, null, refillInterval, refillInterval);
    }

    public async ValueTask<bool> TryAcquireAsync(int tokenCount = 1, CancellationToken cancellationToken = default)
    {
        if (tokenCount <= 0 || tokenCount > _maxTokens)
            return false;

        // Fast path: try to acquire immediately
        var current = _currentTokens;
        if (current >= tokenCount &&
            Interlocked.CompareExchange(ref _currentTokens, current - tokenCount, current) == current)
        {
            return true;
        }

        // Slow path: wait for tokens
        try
        {
            var timeout = TimeSpan.FromMilliseconds(100); // Short timeout for rate limiting
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            cts.CancelAfter(timeout);

            for (int i = 0; i < tokenCount; i++)
            {
                await _semaphore.WaitAsync(cts.Token);
            }

            Interlocked.Add(ref _currentTokens, -tokenCount);
            return true;
        }
        catch (OperationCanceledException)
        {
            return false;
        }
    }

    private void RefillTokens(object? state)
    {
        var current = _currentTokens;
        var toAdd = Math.Min(_refillRate, _maxTokens - current);

        if (toAdd > 0)
        {
            Interlocked.Add(ref _currentTokens, toAdd);
            _semaphore.Release(toAdd);
        }
    }

    public void Dispose()
    {
        _refillTimer?.Dispose();
        _semaphore?.Dispose();
    }
}
