using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// High-performance async patterns and optimizations
/// </summary>
public static class AsyncPatternOptimizations
{
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
}

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
        var source = AsyncPatternOptimizations.PooledValueTaskSource<bool>.Rent();
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

    private record BatchItem<TItem>(TItem Item, AsyncPatternOptimizations.PooledValueTaskSource<bool> Source);
}

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

/// <summary>
/// Optimized task scheduler for CPU-bound work with work stealing
/// </summary>
public sealed class OptimizedTaskScheduler : TaskScheduler, IDisposable
{
    private readonly WorkStealingQueue[] _queues;
    private readonly Thread[] _workers;
    private readonly ManualResetEventSlim _shutdown;
    private volatile bool _disposed;

    public OptimizedTaskScheduler(int workerCount = 0)
    {
        workerCount = workerCount <= 0 ? Environment.ProcessorCount : workerCount;
        
        _queues = new WorkStealingQueue[workerCount];
        _workers = new Thread[workerCount];
        _shutdown = new ManualResetEventSlim(false);
        
        // Create work queues and worker threads
        for (int i = 0; i < workerCount; i++)
        {
            _queues[i] = new WorkStealingQueue();
            var workerId = i;
            _workers[i] = new Thread(() => WorkerLoop(workerId))
            {
                IsBackground = true,
                Name = $"OptimizedWorker-{workerId}"
            };
            _workers[i].Start();
        }
    }

    protected override void QueueTask(Task task)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(OptimizedTaskScheduler));
        
        // Get current thread's queue or use random queue
        var threadId = Thread.CurrentThread.ManagedThreadId;
        var queueIndex = threadId % _queues.Length;
        _queues[queueIndex].Enqueue(task);
    }

    protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued)
    {
        // Allow inline execution for better performance
        return TryExecuteTask(task);
    }

    protected override IEnumerable<Task>? GetScheduledTasks()
    {
        var tasks = new List<Task>();
        foreach (var queue in _queues)
        {
            tasks.AddRange(queue.GetTasks());
        }
        return tasks;
    }

    private void WorkerLoop(int workerId)
    {
        var localQueue = _queues[workerId];
        
        while (!_shutdown.IsSet)
        {
            Task? task = null;
            
            // Try local queue first
            if (localQueue.TryDequeue(out task) ||
                TryStealWork(workerId, out task))
            {
                TryExecuteTask(task);
            }
            else
            {
                // No work available, wait briefly
                Thread.SpinWait(1000);
            }
        }
    }

    private bool TryStealWork(int excludeIndex, out Task? task)
    {
        task = null;
        
        // Try to steal from other queues
        for (int i = 0; i < _queues.Length; i++)
        {
            if (i == excludeIndex) continue;
            
            if (_queues[i].TrySteal(out task))
                return true;
        }
        
        return false;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        
        _shutdown.Set();
        
        foreach (var thread in _workers)
        {
            thread.Join(TimeSpan.FromSeconds(1));
        }
        
        _shutdown.Dispose();
    }

    /// <summary>
    /// Work-stealing queue implementation
    /// </summary>
    private sealed class WorkStealingQueue
    {
        private readonly ConcurrentQueue<Task> _queue = new();
        private volatile int _count;

        public void Enqueue(Task task)
        {
            _queue.Enqueue(task);
            Interlocked.Increment(ref _count);
        }

        public bool TryDequeue(out Task? task)
        {
            if (_queue.TryDequeue(out task))
            {
                Interlocked.Decrement(ref _count);
                return true;
            }
            return false;
        }

        public bool TrySteal(out Task? task)
        {
            // Same as dequeue for this simple implementation
            return TryDequeue(out task);
        }

        public IEnumerable<Task> GetTasks()
        {
            return _queue.ToArray();
        }
    }
}

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