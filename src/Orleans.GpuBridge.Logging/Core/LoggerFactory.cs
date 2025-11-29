using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Logging.Abstractions;
using Orleans.GpuBridge.Logging.Delegates;
using System.Collections.Concurrent;

namespace Orleans.GpuBridge.Logging.Core;

/// <summary>
/// Factory for creating and configuring loggers with delegate-based architecture.
/// Provides fluent configuration API for centralized logging management.
/// </summary>
public sealed class LoggerFactory : IAsyncDisposable
{
    private readonly LoggerDelegateManager _delegateManager;
    private readonly LogBuffer _buffer;
    private readonly ConcurrentDictionary<string, GpuBridgeLogger> _loggers = new();
    private readonly LoggerFactoryOptions _options;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="LoggerFactory"/> class.
    /// </summary>
    /// <param name="options">Configuration options for the logger factory.</param>
    public LoggerFactory(LoggerFactoryOptions? options = null)
    {
        _options = options ?? new LoggerFactoryOptions();
        _delegateManager = new LoggerDelegateManager
        {
            MaxQueueSize = _options.MaxQueueSize,
            ProcessingInterval = _options.ProcessingInterval,
            DropOnQueueFull = _options.DropOnQueueFull
        };

        _buffer = new LogBuffer(new LogBufferOptions
        {
            Capacity = _options.BufferCapacity,
            MaxBatchSize = _options.MaxBatchSize,
            FlushInterval = _options.FlushInterval,
            DropOnOverflow = _options.DropOnBufferFull
        });

        // Connect buffer to delegate manager
        _buffer.BatchReady += OnBatchReady;
        _delegateManager.LoggingError += OnLoggingError;
    }

    /// <summary>
    /// Creates a logger for the specified category.
    /// </summary>
    /// <param name="categoryName">Logger category name</param>
    /// <returns>Configured logger instance</returns>
    public ILogger CreateLogger(string categoryName)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(LoggerFactory));

        return _loggers.GetOrAdd(categoryName,
            name => new GpuBridgeLogger(name, _buffer, _options.DefaultMinimumLevel));
    }

    /// <summary>
    /// Creates a typed logger.
    /// </summary>
    /// <typeparam name="T">Type to create logger for</typeparam>
    /// <returns>Configured logger instance</returns>
    public ILogger<T> CreateLogger<T>() => new Logger<T>(CreateLogger(typeof(T).FullName ?? typeof(T).Name));

    /// <summary>
    /// Gets the delegate manager for advanced configuration.
    /// </summary>
    public LoggerDelegateManager DelegateManager => _delegateManager;

    /// <summary>
    /// Gets the log buffer for monitoring.
    /// </summary>
    public LogBuffer Buffer => _buffer;

    /// <summary>
    /// Gets factory statistics.
    /// </summary>
    public LoggerFactoryStatistics GetStatistics()
    {
        return new LoggerFactoryStatistics
        {
            RegisteredLoggers = _loggers.Count,
            RegisteredDelegates = _delegateManager.DelegateNames.Count(),
            BufferHealth = _buffer.GetHealth(),
            DelegateManagerStats = _delegateManager.GetStatistics()
        };
    }

    /// <summary>
    /// Flushes all loggers and delegates.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    public async Task FlushAsync(CancellationToken cancellationToken = default)
    {
        if (_disposed)
            return;

        await _buffer.FlushAsync(cancellationToken).ConfigureAwait(false);
        await _delegateManager.FlushAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Disposes the logger factory asynchronously, flushing all pending logs and disposing all resources.
    /// </summary>
    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;

        // Flush remaining logs
        await FlushAsync(CancellationToken.None).ConfigureAwait(false);

        // Dispose components
        await _buffer.DisposeAsync().ConfigureAwait(false);
        await _delegateManager.DisposeAsync().ConfigureAwait(false);

        _loggers.Clear();
    }

    private async Task OnBatchReady(IReadOnlyList<LogEntry> entries, CancellationToken cancellationToken)
    {
        foreach (var entry in entries)
        {
            _delegateManager.EnqueueLogEntry(entry);
        }

        // For high-priority entries, process immediately
        var hasErrors = entries.Any(e => e.Level >= LogLevel.Error);
        if (hasErrors)
        {
            await _delegateManager.FlushAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    private void OnLoggingError(object? sender, LoggingErrorEventArgs args)
    {
        // Fallback error logging to console
        Console.WriteLine($"[{DateTime.UtcNow:O}] Logging Error: {args.Message}");
        if (args.Exception != null)
        {
            Console.WriteLine($"Exception: {args.Exception}");
        }
    }
}
