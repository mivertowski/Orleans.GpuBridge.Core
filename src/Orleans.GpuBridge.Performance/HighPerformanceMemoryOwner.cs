using System;
using System.Buffers;
using System.Threading;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// High-performance memory owner with efficient disposal
/// </summary>
public sealed class HighPerformanceMemoryOwner<T> : IMemoryOwner<T> where T : unmanaged
{
    private Memory<T> _memory;
    private readonly HighPerformanceMemoryPool<T> _pool;
    private readonly int _bucketIndex;
    private readonly ILogger _logger;
    private int _isDisposed;

    public HighPerformanceMemoryOwner(
        Memory<T> memory,
        HighPerformanceMemoryPool<T> pool,
        int bucketIndex,
        ILogger logger)
    {
        _memory = memory;
        _pool = pool;
        _bucketIndex = bucketIndex;
        _logger = logger;
    }

    public Memory<T> Memory => _isDisposed == 0 ? _memory : throw new ObjectDisposedException(nameof(HighPerformanceMemoryOwner<T>));

    public bool IsDisposed => _isDisposed != 0;

    public void Reset()
    {
        if (_isDisposed == 0)
        {
            _memory.Span.Clear();
        }
    }

    public void Dispose()
    {
        if (Interlocked.CompareExchange(ref _isDisposed, 1, 0) == 0)
        {
            _pool.Return(this, _bucketIndex);
        }
    }
}
