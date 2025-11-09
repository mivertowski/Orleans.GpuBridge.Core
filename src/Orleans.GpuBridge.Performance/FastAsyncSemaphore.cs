using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// Optimized async semaphore with minimal allocations
/// </summary>
public sealed class FastAsyncSemaphore : IDisposable
{
    private readonly SemaphoreSlim _semaphore;
    private readonly ConcurrentQueue<TaskCompletionSource<bool>> _waiters;
    private volatile int _currentCount;

    public FastAsyncSemaphore(int initialCount, int maxCount)
    {
        _semaphore = new SemaphoreSlim(initialCount, maxCount);
        _waiters = new ConcurrentQueue<TaskCompletionSource<bool>>();
        _currentCount = initialCount;
    }

    public int CurrentCount => _currentCount;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ValueTask WaitAsync(CancellationToken cancellationToken = default)
    {
        if (TryWait())
            return ValueTask.CompletedTask;

        return new ValueTask(WaitSlowAsync(cancellationToken));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private bool TryWait()
    {
        var count = _currentCount;
        return count > 0 && Interlocked.CompareExchange(ref _currentCount, count - 1, count) == count;
    }

    private async Task WaitSlowAsync(CancellationToken cancellationToken)
    {
        await _semaphore.WaitAsync(cancellationToken);
        Interlocked.Decrement(ref _currentCount);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Release(int releaseCount = 1)
    {
        _semaphore.Release(releaseCount);
        Interlocked.Add(ref _currentCount, releaseCount);
    }

    public void Dispose()
    {
        _semaphore?.Dispose();
    }
}
