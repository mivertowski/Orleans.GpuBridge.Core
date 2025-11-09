using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// High-performance batch processor with optimal async patterns
/// </summary>
public sealed class OptimizedBatchProcessor<T> : IDisposable
{
    private readonly Channel<BatchItem<T>> _channel;
    private readonly Task[] _workers;
    private readonly CancellationTokenSource _cancellation;
    private readonly ILogger<OptimizedBatchProcessor<T>> _logger;
    private readonly int _batchSize;
    private readonly TimeSpan _batchTimeout;
    private readonly Func<T[], ValueTask> _processor;

    public OptimizedBatchProcessor(
        ILogger<OptimizedBatchProcessor<T>> logger,
        Func<T[], ValueTask> processor,
        int workerCount = 0,
        int batchSize = 100,
        TimeSpan batchTimeout = default)
    {
        _logger = logger;
        _processor = processor;
        _batchSize = batchSize;
        _batchTimeout = batchTimeout == default ? TimeSpan.FromMilliseconds(50) : batchTimeout;

        workerCount = workerCount <= 0 ? Environment.ProcessorCount : workerCount;

        var options = new BoundedChannelOptions(batchSize * workerCount * 2)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = false,
            SingleWriter = false
        };

        _channel = Channel.CreateBounded<BatchItem<T>>(options);
        _cancellation = new CancellationTokenSource();

        _workers = new Task[workerCount];
        for (int i = 0; i < workerCount; i++)
        {
            var workerId = i;
            _workers[i] = Task.Run(() => ProcessBatchesAsync(workerId, _cancellation.Token));
        }

        _logger.LogInformation("Optimized batch processor started with {WorkerCount} workers", workerCount);
    }

    public async ValueTask<bool> EnqueueAsync(T item, CancellationToken cancellationToken = default)
    {
        var source = PooledValueTaskSource<bool>.Rent();
        var batchItem = new BatchItem<T>(item, source);

        try
        {
            await _channel.Writer.WriteAsync(batchItem, cancellationToken);
            return await source.Task;
        }
        catch (Exception ex)
        {
            source.SetException(ex);
            throw;
        }
    }

    private async Task ProcessBatchesAsync(int workerId, CancellationToken cancellationToken)
    {
        var batch = new List<BatchItem<T>>(_batchSize);
        var timer = new Timer(FlushBatch, batch, Timeout.Infinite, Timeout.Infinite);

        try
        {
            _logger.LogDebug("Batch worker {WorkerId} started", workerId);

            await foreach (var item in _channel.Reader.ReadAllAsync(cancellationToken))
            {
                batch.Add(item);

                if (batch.Count >= _batchSize)
                {
                    await ProcessBatch(batch);
                    batch.Clear();
                    timer.Change(Timeout.Infinite, Timeout.Infinite);
                }
                else if (batch.Count == 1)
                {
                    // Start timeout timer for first item in batch
                    timer.Change(_batchTimeout, Timeout.InfiniteTimeSpan);
                }
            }

            // Process remaining items
            if (batch.Count > 0)
            {
                await ProcessBatch(batch);
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            // Expected cancellation
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Batch worker {WorkerId} failed", workerId);
        }
        finally
        {
            timer.Dispose();
            _logger.LogDebug("Batch worker {WorkerId} stopped", workerId);
        }

        void FlushBatch(object? state)
        {
            var currentBatch = (List<BatchItem<T>>)state!;
            if (currentBatch.Count > 0)
            {
                _ = Task.Run(async () =>
                {
                    await ProcessBatch(currentBatch);
                    currentBatch.Clear();
                }, cancellationToken);
            }
        }
    }

    private async ValueTask ProcessBatch(List<BatchItem<T>> batch)
    {
        if (batch.Count == 0) return;

        try
        {
            var items = batch.ConvertAll(b => b.Item).ToArray();
            await _processor(items);

            // Signal completion to all items in batch
            foreach (var batchItem in batch)
            {
                batchItem.Source.SetResult(true);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Batch processing failed for {ItemCount} items", batch.Count);

            // Signal failure to all items in batch
            foreach (var batchItem in batch)
            {
                batchItem.Source.SetException(ex);
            }
        }
    }

    public async ValueTask CompleteAsync()
    {
        _channel.Writer.Complete();
        await Task.WhenAll(_workers).ConfigureAwait(false);
    }

    public void Dispose()
    {
        _channel.Writer.Complete();
        _cancellation.Cancel();

        try
        {
            Task.WaitAll(_workers, TimeSpan.FromSeconds(5));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error waiting for batch workers to complete");
        }

        _cancellation.Dispose();
    }

    private record BatchItem<TItem>(TItem Item, PooledValueTaskSource<bool> Source);
}
