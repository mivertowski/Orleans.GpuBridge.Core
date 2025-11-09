using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Performance;

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
