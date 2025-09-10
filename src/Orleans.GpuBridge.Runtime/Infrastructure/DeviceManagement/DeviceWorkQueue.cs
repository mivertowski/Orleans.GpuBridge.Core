using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Runtime.Infrastructure.DeviceManagement;

/// <summary>
/// Work queue for a specific device
/// </summary>
internal sealed class DeviceWorkQueue
{
    private readonly GpuDevice _device;
    private readonly ConcurrentQueue<WorkItem> _queue;
    private readonly SemaphoreSlim _semaphore;
    private readonly CancellationTokenSource _cts;
    private readonly Task _processingTask;
    private long _processedItems;
    private long _failedItems;
    private bool _shutdown;
    
    public GpuDevice Device => _device;
    public int QueuedItems => _queue.Count;
    
    public DeviceWorkQueue(GpuDevice device)
    {
        _device = device;
        _queue = new ConcurrentQueue<WorkItem>();
        _semaphore = new SemaphoreSlim(0);
        _cts = new CancellationTokenSource();
        _processingTask = ProcessQueueAsync(_cts.Token);
    }
    
    public Task<WorkHandle> EnqueueAsync(
        Func<CancellationToken, Task> work,
        CancellationToken ct = default)
    {
        if (_shutdown)
        {
            throw new InvalidOperationException("Queue is shutting down");
        }
        
        var item = new WorkItem
        {
            Id = Guid.NewGuid().ToString(),
            Work = work,
            CompletionSource = new TaskCompletionSource(),
            EnqueuedAt = DateTime.UtcNow
        };
        
        _queue.Enqueue(item);
        _semaphore.Release();
        
        return Task.FromResult(new WorkHandle(item.Id, item.CompletionSource.Task));
    }
    
    private async Task ProcessQueueAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                await _semaphore.WaitAsync(ct);
                
                if (_queue.TryDequeue(out var item))
                {
                    try
                    {
                        await item.Work(ct);
                        item.CompletionSource.SetResult();
                        Interlocked.Increment(ref _processedItems);
                    }
                    catch (Exception ex)
                    {
                        item.CompletionSource.SetException(ex);
                        Interlocked.Increment(ref _failedItems);
                    }
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }
    }
    
    public DeviceWorkQueueMetrics GetMetrics()
    {
        var processed = Interlocked.Read(ref _processedItems);
        var failed = Interlocked.Read(ref _failedItems);
        var total = processed + failed;
        
        return new DeviceWorkQueueMetrics
        {
            QueuedItems = _queue.Count,
            ProcessedItems = processed,
            FailedItems = failed,
            ErrorRate = total > 0 ? failed / (double)total : 0
        };
    }
    
    public async Task ShutdownAsync(CancellationToken ct)
    {
        _shutdown = true;
        _cts.Cancel();
        
        try
        {
            await _processingTask.WaitAsync(ct);
        }
        catch (OperationCanceledException)
        {
            // Expected
        }
        
        // Complete remaining items with cancellation
        while (_queue.TryDequeue(out var item))
        {
            item.CompletionSource.SetCanceled();
        }
    }
    
    private sealed class WorkItem
    {
        public string Id { get; init; } = default!;
        public Func<CancellationToken, Task> Work { get; init; } = default!;
        public TaskCompletionSource CompletionSource { get; init; } = default!;
        public DateTime EnqueuedAt { get; init; }
    }
}