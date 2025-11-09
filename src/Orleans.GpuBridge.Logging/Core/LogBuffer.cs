using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Logging.Abstractions;
using System.Collections.Concurrent;
using System.Threading.Channels;

namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// High-performance log buffer for batching log entries with backpressure control.
/// Optimized for high-throughput scenarios with minimal memory allocation.
/// </summary>
public sealed class LogBuffer : IAsyncDisposable
{
    private readonly LogBufferOptions _options;
    private readonly Channel<LogEntry> _channel;
    private readonly ChannelWriter<LogEntry> _writer;
    private readonly ChannelReader<LogEntry> _reader;
    private readonly Task _processingTask;
    private readonly CancellationTokenSource _shutdownToken = new();
    private volatile bool _disposed;

    /// <summary>
    /// Event fired when a batch is ready for processing.
    /// </summary>
    public event Func<IReadOnlyList<LogEntry>, CancellationToken, Task>? BatchReady;

    /// <summary>
    /// Event fired when buffer overflow occurs.
    /// </summary>
    public event EventHandler<BufferOverflowEventArgs>? BufferOverflow;

    /// <summary>
    /// Gets the current number of buffered entries.
    /// </summary>
    public int BufferedCount => _channel.Reader.CanCount ? _channel.Reader.Count : -1;

    /// <summary>
    /// Gets buffer statistics.
    /// </summary>
    public BufferStatistics Statistics { get; private set; } = new();

    public LogBuffer(LogBufferOptions? options = null)
    {
        _options = options ?? new LogBufferOptions();

        // Create bounded channel for backpressure control
        var channelOptions = new BoundedChannelOptions(_options.Capacity)
        {
            FullMode = _options.DropOnOverflow
                ? BoundedChannelFullMode.DropOldest
                : BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = false,
            AllowSynchronousContinuations = false
        };

        _channel = Channel.CreateBounded<LogEntry>(channelOptions);
        _writer = _channel.Writer;
        _reader = _channel.Reader;

        _processingTask = ProcessBufferedEntriesAsync(_shutdownToken.Token);
    }

    /// <summary>
    /// Adds a log entry to the buffer.
    /// </summary>
    /// <param name="entry">The log entry to buffer</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>true if added successfully, false if buffer is full and dropping is disabled</returns>
    public async ValueTask<bool> AddAsync(LogEntry entry, CancellationToken cancellationToken = default)
    {
        if (_disposed)
            return false;

        try
        {
            await _writer.WriteAsync(entry, cancellationToken).ConfigureAwait(false);

            // Update statistics
            var stats = Statistics;
            Statistics = stats with
            {
                TotalEnqueued = stats.TotalEnqueued + 1,
                LastEnqueueTime = DateTimeOffset.UtcNow
            };

            return true;
        }
        catch (ChannelClosedException)
        {
            return false;
        }
        catch (InvalidOperationException) when (_options.DropOnOverflow)
        {
            // Channel is full and dropping is enabled
            OnBufferOverflow(entry);
            return false;
        }
    }

    /// <summary>
    /// Tries to add a log entry to the buffer without blocking.
    /// </summary>
    /// <param name="entry">The log entry to buffer</param>
    /// <returns>true if added successfully, false if buffer is full</returns>
    public bool TryAdd(LogEntry entry)
    {
        if (_disposed)
            return false;

        try
        {
            var added = _writer.TryWrite(entry);

            if (added)
            {
                var stats = Statistics;
                Statistics = stats with
                {
                    TotalEnqueued = stats.TotalEnqueued + 1,
                    LastEnqueueTime = DateTimeOffset.UtcNow
                };
            }
            else if (_options.DropOnOverflow)
            {
                OnBufferOverflow(entry);
            }

            return added;
        }
        catch (ChannelClosedException)
        {
            return false;
        }
    }

    /// <summary>
    /// Forces immediate flush of all buffered entries.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    public async Task FlushAsync(CancellationToken cancellationToken = default)
    {
        if (_disposed)
            return;

        // Signal flush and wait for processing
        var batch = new List<LogEntry>();
        while (_reader.TryRead(out var entry))
        {
            batch.Add(entry);

            if (batch.Count >= _options.MaxBatchSize)
            {
                await ProcessBatch(batch, cancellationToken).ConfigureAwait(false);
                batch.Clear();
            }
        }

        if (batch.Count > 0)
        {
            await ProcessBatch(batch, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Gets current buffer health status.
    /// </summary>
    public BufferHealth GetHealth()
    {
        var currentCount = BufferedCount;
        var utilizationPercent = currentCount >= 0
            ? (double)currentCount / _options.Capacity * 100
            : 0;

        return new BufferHealth
        {
            Status = utilizationPercent switch
            {
                < 50 => BufferHealthStatus.Healthy,
                < 80 => BufferHealthStatus.Warning,
                _ => BufferHealthStatus.Critical
            },
            UtilizationPercent = utilizationPercent,
            BufferedEntries = currentCount,
            Capacity = _options.Capacity,
            Statistics = Statistics
        };
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;

        // Stop accepting new entries
        _writer.TryComplete();

        // Cancel processing
        await _shutdownToken.CancelAsync().ConfigureAwait(false);

        try
        {
            await _processingTask.ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            // Expected during shutdown
        }

        // Process remaining entries
        await FlushAsync(CancellationToken.None).ConfigureAwait(false);

        _shutdownToken.Dispose();
    }

    private async Task ProcessBufferedEntriesAsync(CancellationToken cancellationToken)
    {
        var batch = new List<LogEntry>(_options.MaxBatchSize);
        var lastFlushTime = DateTimeOffset.UtcNow;

        try
        {
            await foreach (var entry in _reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                batch.Add(entry);

                var shouldFlush = batch.Count >= _options.MaxBatchSize ||
                                  DateTimeOffset.UtcNow - lastFlushTime >= _options.FlushInterval ||
                                  entry.Level >= LogLevel.Error; // Immediate flush for errors

                if (shouldFlush)
                {
                    await ProcessBatch(batch, cancellationToken).ConfigureAwait(false);
                    batch.Clear();
                    lastFlushTime = DateTimeOffset.UtcNow;
                }
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            // Expected during shutdown
        }

        // Process remaining entries
        if (batch.Count > 0)
        {
            await ProcessBatch(batch, CancellationToken.None).ConfigureAwait(false);
        }
    }

    private async Task ProcessBatch(List<LogEntry> batch, CancellationToken cancellationToken)
    {
        if (batch.Count == 0 || BatchReady == null)
            return;

        try
        {
            var startTime = DateTimeOffset.UtcNow;

            await BatchReady(batch, cancellationToken).ConfigureAwait(false);

            var processingTime = DateTimeOffset.UtcNow - startTime;

            // Update statistics
            var stats = Statistics;
            Statistics = stats with
            {
                TotalProcessed = stats.TotalProcessed + batch.Count,
                TotalBatches = stats.TotalBatches + 1,
                LastProcessTime = DateTimeOffset.UtcNow,
                AverageProcessingTime = TimeSpan.FromTicks(
                    (stats.AverageProcessingTime.Ticks * stats.TotalBatches + processingTime.Ticks) /
                    (stats.TotalBatches + 1)),
                MaxBatchSize = Math.Max(stats.MaxBatchSize, batch.Count)
            };
        }
        catch (Exception ex)
        {
            // Update error statistics
            var stats = Statistics;
            Statistics = stats with
            {
                TotalErrors = stats.TotalErrors + 1,
                LastError = ex.Message,
                LastErrorTime = DateTimeOffset.UtcNow
            };

            // Don't rethrow to avoid stopping the processing loop
            Console.WriteLine($"Error processing log batch: {ex.Message}");
        }
    }

    private void OnBufferOverflow(LogEntry droppedEntry)
    {
        var stats = Statistics;
        Statistics = stats with
        {
            TotalDropped = stats.TotalDropped + 1,
            LastDropTime = DateTimeOffset.UtcNow
        };

        BufferOverflow?.Invoke(this, new BufferOverflowEventArgs(droppedEntry));
    }
}
