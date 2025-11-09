using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// Custom ValueTask source that minimizes allocations for hot paths
/// </summary>
public sealed class PooledValueTaskSource<T> : IValueTaskSource<T>, IThreadPoolWorkItem
{
    private static readonly ConcurrentQueue<PooledValueTaskSource<T>> Pool = new();

    private ManualResetValueTaskSourceCore<T> _core;
    private readonly Action<object?> _continuation;
    private T _result = default!;
    private Exception? _exception;
    private bool _completed;

    private PooledValueTaskSource()
    {
        _continuation = state => ((PooledValueTaskSource<T>)state!).Execute();
    }

    public static PooledValueTaskSource<T> Rent()
    {
        return Pool.TryDequeue(out var source) ? source : new PooledValueTaskSource<T>();
    }

    public void Return()
    {
        Reset();
        Pool.Enqueue(this);
    }

    public ValueTask<T> Task => new(this, _core.Version);

    public void SetResult(T result)
    {
        _result = result;
        _completed = true;
        _core.SetResult(result);
    }

    public void SetException(Exception exception)
    {
        _exception = exception;
        _completed = true;
        _core.SetException(exception);
    }

    public T GetResult(short token)
    {
        try
        {
            return _core.GetResult(token);
        }
        finally
        {
            Return();
        }
    }

    public ValueTaskSourceStatus GetStatus(short token) => _core.GetStatus(token);

    public void OnCompleted(Action<object?> continuation, object? state, short token, ValueTaskSourceOnCompletedFlags flags)
    {
        _core.OnCompleted(continuation, state, token, flags);
    }

    public void Execute()
    {
        if (_completed)
        {
            if (_exception != null)
                SetException(_exception);
            else
                SetResult(_result);
        }
    }

    private void Reset()
    {
        _core.Reset();
        _result = default!;
        _exception = null;
        _completed = false;
    }
}
