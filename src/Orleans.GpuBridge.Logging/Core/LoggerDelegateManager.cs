using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Logging.Abstractions;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// Central manager for all logger delegates. Provides coordinated logging
/// across multiple targets with performance optimization and error handling.
/// </summary>
public sealed class LoggerDelegateManager : IAsyncDisposable
{
    private readonly ConcurrentDictionary<string, ILoggerDelegate> _delegates = new();
    private readonly ConcurrentQueue<LogEntry> _logQueue = new();
    private readonly SemaphoreSlim _flushSemaphore = new(1, 1);
    private readonly CancellationTokenSource _shutdownTokenSource = new();
    private readonly Task _processingTask;
    private volatile bool _disposed;

    /// <summary>
    /// Event fired when a logging error occurs.
    /// </summary>
    public event EventHandler<LoggingErrorEventArgs>? LoggingError;

    /// <summary>
    /// Gets all registered delegate names.
    /// </summary>
    public IEnumerable<string> DelegateNames => _delegates.Keys;

    /// <summary>
    /// Gets the number of pending log entries in the queue.
    /// </summary>
    public int PendingLogCount => _logQueue.Count;

    /// <summary>
    /// Gets or sets the maximum queue size before entries are dropped.
    /// </summary>
    public int MaxQueueSize { get; set; } = 10000;

    /// <summary>
    /// Gets or sets the batch processing interval.
    /// </summary>
    public TimeSpan ProcessingInterval { get; set; } = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Gets or sets whether to drop log entries when queue is full.
    /// </summary>
    public bool DropOnQueueFull { get; set; } = true;

    public LoggerDelegateManager()
    {
        _processingTask = ProcessLogEntriesAsync(_shutdownTokenSource.Token);
    }

    /// <summary>
    /// Registers a logger delegate.
    /// </summary>
    /// <param name="delegate">The delegate to register</param>
    /// <returns>true if registered successfully, false if name already exists</returns>
    public bool RegisterDelegate(ILoggerDelegate @delegate)
    {
        ArgumentNullException.ThrowIfNull(@delegate);
        return _delegates.TryAdd(@delegate.Name, @delegate);
    }

    /// <summary>
    /// Unregisters a logger delegate.
    /// </summary>
    /// <param name="name">Name of the delegate to remove</param>
    /// <returns>true if removed, false if not found</returns>
    public bool UnregisterDelegate(string name)
    {
        return _delegates.TryRemove(name, out var removed) && removed != null;
    }

    /// <summary>
    /// Gets a registered delegate by name.
    /// </summary>
    /// <param name="name">Delegate name</param>
    /// <returns>The delegate if found, null otherwise</returns>
    public ILoggerDelegate? GetDelegate(string name)
    {
        return _delegates.TryGetValue(name, out var @delegate) ? @delegate : null;
    }

    /// <summary>
    /// Queues a log entry for processing by all appropriate delegates.
    /// </summary>
    /// <param name="entry">The log entry to process</param>
    /// <returns>true if queued successfully, false if queue is full</returns>
    public bool EnqueueLogEntry(LogEntry entry)
    {
        if (_disposed)
            return false;

        // Check queue size
        if (_logQueue.Count >= MaxQueueSize)
        {
            if (DropOnQueueFull)
            {
                // Try to dequeue an old entry to make space
                _logQueue.TryDequeue(out _);
            }
            else
            {
                return false;
            }
        }

        _logQueue.Enqueue(entry);
        return true;
    }

    /// <summary>
    /// Writes a log entry immediately to all appropriate delegates.
    /// </summary>
    /// <param name="entry">The log entry to write</param>
    /// <param name="cancellationToken">Cancellation token</param>
    public async Task WriteImmediateAsync(LogEntry entry, CancellationToken cancellationToken = default)
    {
        if (_disposed)
            return;

        var tasks = new List<Task>();

        foreach (var @delegate in _delegates.Values)
        {
            if (@delegate.IsEnabled(entry.Level))
            {
                tasks.Add(WriteToDelegate(@delegate, entry, cancellationToken));
            }
        }

        if (tasks.Count > 0)
        {
            try
            {
                await Task.WhenAll(tasks);
            }
            catch (Exception ex)
            {
                OnLoggingError(new LoggingErrorEventArgs("Failed to write to delegates", ex));
            }
        }
    }

    /// <summary>
    /// Flushes all delegates.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    public async Task FlushAsync(CancellationToken cancellationToken = default)
    {
        await _flushSemaphore.WaitAsync(cancellationToken);
        try
        {
            var tasks = _delegates.Values.Select(d => d.FlushAsync(cancellationToken));
            await Task.WhenAll(tasks);
        }
        finally
        {
            _flushSemaphore.Release();
        }
    }

    /// <summary>
    /// Gets statistics about the logging system.
    /// </summary>
    public LoggingStatistics GetStatistics()
    {
        return new LoggingStatistics
        {
            RegisteredDelegates = _delegates.Count,
            PendingEntries = _logQueue.Count,
            MaxQueueSize = MaxQueueSize,
            ProcessingInterval = ProcessingInterval,
            IsProcessing = !_processingTask.IsCompleted
        };
    }

    private async Task ProcessLogEntriesAsync(CancellationToken cancellationToken)
    {
        var batch = new List<LogEntry>(1000);
        
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                // Collect batch
                batch.Clear();
                var batchStartTime = Stopwatch.GetTimestamp();
                
                while (_logQueue.TryDequeue(out var entry) && batch.Count < 1000)
                {
                    batch.Add(entry);
                }

                if (batch.Count > 0)
                {
                    await ProcessBatch(batch, cancellationToken);
                    
                    var elapsed = Stopwatch.GetElapsedTime(batchStartTime);
                    if (elapsed > ProcessingInterval)
                    {
                        // Log processing time warning if it takes too long
                        OnLoggingError(new LoggingErrorEventArgs(
                            $"Batch processing took {elapsed.TotalMilliseconds:F2}ms for {batch.Count} entries"));
                    }
                }

                await Task.Delay(ProcessingInterval, cancellationToken);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                OnLoggingError(new LoggingErrorEventArgs("Error in log processing loop", ex));
                await Task.Delay(TimeSpan.FromSeconds(1), cancellationToken);
            }
        }
    }

    private async Task ProcessBatch(List<LogEntry> entries, CancellationToken cancellationToken)
    {
        var delegateGroups = new Dictionary<ILoggerDelegate, List<LogEntry>>();

        // Group entries by delegate based on log level filtering
        foreach (var entry in entries)
        {
            foreach (var @delegate in _delegates.Values)
            {
                if (@delegate.IsEnabled(entry.Level))
                {
                    if (!delegateGroups.TryGetValue(@delegate, out var list))
                    {
                        list = new List<LogEntry>();
                        delegateGroups[@delegate] = list;
                    }
                    list.Add(entry);
                }
            }
        }

        // Process each delegate's batch
        var tasks = new List<Task>();
        foreach (var (delegateInstance, batch) in delegateGroups)
        {
            if (delegateInstance is IBatchLoggerDelegate batchDelegate)
            {
                tasks.Add(WriteBatchToDelegate(batchDelegate, batch, cancellationToken));
            }
            else
            {
                foreach (var entry in batch)
                {
                    tasks.Add(WriteToDelegate(delegateInstance, entry, cancellationToken));
                }
            }
        }

        if (tasks.Count > 0)
        {
            try
            {
                await Task.WhenAll(tasks);
            }
            catch (Exception ex)
            {
                OnLoggingError(new LoggingErrorEventArgs("Failed to process log batch", ex));
            }
        }
    }

    private async Task WriteToDelegate(ILoggerDelegate @delegate, LogEntry entry, CancellationToken cancellationToken)
    {
        try
        {
            await @delegate.WriteAsync(entry, cancellationToken);
        }
        catch (Exception ex)
        {
            OnLoggingError(new LoggingErrorEventArgs($"Failed to write to delegate '{@delegate.Name}'", ex));
        }
    }

    private async Task WriteBatchToDelegate(IBatchLoggerDelegate @delegate, IReadOnlyCollection<LogEntry> entries, 
        CancellationToken cancellationToken)
    {
        try
        {
            await @delegate.WriteBatchAsync(entries, cancellationToken);
        }
        catch (Exception ex)
        {
            OnLoggingError(new LoggingErrorEventArgs($"Failed to write batch to delegate '{@delegate.Name}'", ex));
        }
    }

    private void OnLoggingError(LoggingErrorEventArgs args)
    {
        LoggingError?.Invoke(this, args);
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;

        // Stop processing
        await _shutdownTokenSource.CancelAsync();

        try
        {
            await _processingTask;
        }
        catch (OperationCanceledException)
        {
            // Expected during shutdown
        }

        // Process remaining entries
        var remainingEntries = new List<LogEntry>();
        while (_logQueue.TryDequeue(out var entry))
        {
            remainingEntries.Add(entry);
        }

        if (remainingEntries.Count > 0)
        {
            await ProcessBatch(remainingEntries, CancellationToken.None);
        }

        // Dispose all delegates
        var disposeTasks = _delegates.Values.Select(d => d.DisposeAsync().AsTask());
        await Task.WhenAll(disposeTasks);

        _delegates.Clear();
        _flushSemaphore.Dispose();
        _shutdownTokenSource.Dispose();
    }
}

/// <summary>
/// Event arguments for logging errors.
/// </summary>
public sealed class LoggingErrorEventArgs : EventArgs
{
    public string Message { get; }
    public Exception? Exception { get; }
    public DateTimeOffset Timestamp { get; } = DateTimeOffset.UtcNow;

    public LoggingErrorEventArgs(string message, Exception? exception = null)
    {
        Message = message ?? throw new ArgumentNullException(nameof(message));
        Exception = exception;
    }
}

/// <summary>
/// Statistics about the logging system performance.
/// </summary>
public sealed record LoggingStatistics
{
    public int RegisteredDelegates { get; init; }
    public int PendingEntries { get; init; }
    public int MaxQueueSize { get; init; }
    public TimeSpan ProcessingInterval { get; init; }
    public bool IsProcessing { get; init; }
}