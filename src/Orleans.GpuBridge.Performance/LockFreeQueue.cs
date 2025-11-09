using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// Lock-free producer-consumer queue with back-pressure support
/// </summary>
public sealed class LockFreeQueue<T> : IDisposable
{
    private readonly Channel<T> _channel;
    private readonly ChannelWriter<T> _writer;
    private readonly ChannelReader<T> _reader;

    public LockFreeQueue(int capacity = 1024)
    {
        var options = new BoundedChannelOptions(capacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = false,
            SingleWriter = false
        };

        _channel = Channel.CreateBounded<T>(options);
        _writer = _channel.Writer;
        _reader = _channel.Reader;
    }

    public ValueTask<bool> TryEnqueueAsync(T item, CancellationToken cancellationToken = default)
    {
        return _writer.TryWrite(item) ? new(true) : new(EnqueueSlowAsync(item, cancellationToken));
    }

    private async Task<bool> EnqueueSlowAsync(T item, CancellationToken cancellationToken)
    {
        try
        {
            await _writer.WriteAsync(item, cancellationToken);
            return true;
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return false;
        }
    }

    public ValueTask<T?> TryDequeueAsync(CancellationToken cancellationToken = default)
    {
        return _reader.TryRead(out var item) ? new(item) : new(DequeueSlowAsync(cancellationToken));
    }

    private async Task<T?> DequeueSlowAsync(CancellationToken cancellationToken)
    {
        try
        {
            return await _reader.ReadAsync(cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return default;
        }
    }

    public IAsyncEnumerable<T> ConsumeAllAsync(CancellationToken cancellationToken = default)
    {
        return _reader.ReadAllAsync(cancellationToken);
    }

    public void Complete() => _writer.Complete();

    public void Dispose() => Complete();
}
