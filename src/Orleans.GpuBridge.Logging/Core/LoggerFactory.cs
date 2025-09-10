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

        await _buffer.FlushAsync(cancellationToken);
        await _delegateManager.FlushAsync(cancellationToken);
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;

        // Flush remaining logs
        await FlushAsync(CancellationToken.None);

        // Dispose components
        await _buffer.DisposeAsync();
        await _delegateManager.DisposeAsync();

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
            await _delegateManager.FlushAsync(cancellationToken);
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

/// <summary>
/// Fluent builder for configuring the logger factory.
/// </summary>
public sealed class LoggerFactoryBuilder
{
    private readonly LoggerFactoryOptions _options = new();
    private readonly List<Func<LoggerFactory, ILoggerDelegate>> _delegateConfigurators = new();

    /// <summary>
    /// Sets the default minimum log level.
    /// </summary>
    public LoggerFactoryBuilder WithMinimumLevel(LogLevel level)
    {
        _options.DefaultMinimumLevel = level;
        return this;
    }

    /// <summary>
    /// Configures buffer settings.
    /// </summary>
    public LoggerFactoryBuilder WithBuffer(Action<BufferConfigurationBuilder> configure)
    {
        var builder = new BufferConfigurationBuilder(_options);
        configure(builder);
        return this;
    }

    /// <summary>
    /// Configures delegate manager settings.
    /// </summary>
    public LoggerFactoryBuilder WithDelegateManager(Action<DelegateManagerConfigurationBuilder> configure)
    {
        var builder = new DelegateManagerConfigurationBuilder(_options);
        configure(builder);
        return this;
    }

    /// <summary>
    /// Adds console logging delegate.
    /// </summary>
    public LoggerFactoryBuilder AddConsole(Action<ConsoleLoggerOptions>? configure = null)
    {
        _delegateConfigurators.Add(factory =>
        {
            var options = new ConsoleLoggerOptions();
            configure?.Invoke(options);
            return new ConsoleLoggerDelegate(options);
        });
        return this;
    }

    /// <summary>
    /// Adds file logging delegate.
    /// </summary>
    public LoggerFactoryBuilder AddFile(Action<FileLoggerOptions>? configure = null)
    {
        _delegateConfigurators.Add(factory =>
        {
            var options = new FileLoggerOptions();
            configure?.Invoke(options);
            return new FileLoggerDelegate(options);
        });
        return this;
    }

    /// <summary>
    /// Adds telemetry logging delegate.
    /// </summary>
    public LoggerFactoryBuilder AddTelemetry(Action<TelemetryLoggerOptions>? configure = null)
    {
        _delegateConfigurators.Add(factory =>
        {
            var options = new TelemetryLoggerOptions();
            configure?.Invoke(options);
            return new TelemetryLoggerDelegate(options);
        });
        return this;
    }

    /// <summary>
    /// Adds a custom logger delegate.
    /// </summary>
    public LoggerFactoryBuilder AddDelegate<T>(Func<LoggerFactory, T> factory) where T : ILoggerDelegate
    {
        _delegateConfigurators.Add(f => factory(f));
        return this;
    }

    /// <summary>
    /// Builds the configured logger factory.
    /// </summary>
    public LoggerFactory Build()
    {
        var factory = new LoggerFactory(_options);

        // Register all configured delegates
        foreach (var configurator in _delegateConfigurators)
        {
            var @delegate = configurator(factory);
            factory.DelegateManager.RegisterDelegate(@delegate);
        }

        return factory;
    }
}

/// <summary>
/// Builder for configuring buffer settings.
/// </summary>
public sealed class BufferConfigurationBuilder
{
    private readonly LoggerFactoryOptions _options;

    internal BufferConfigurationBuilder(LoggerFactoryOptions options)
    {
        _options = options;
    }

    public BufferConfigurationBuilder WithCapacity(int capacity)
    {
        _options.BufferCapacity = capacity;
        return this;
    }

    public BufferConfigurationBuilder WithMaxBatchSize(int batchSize)
    {
        _options.MaxBatchSize = batchSize;
        return this;
    }

    public BufferConfigurationBuilder WithFlushInterval(TimeSpan interval)
    {
        _options.FlushInterval = interval;
        return this;
    }

    public BufferConfigurationBuilder DropOnFull(bool drop = true)
    {
        _options.DropOnBufferFull = drop;
        return this;
    }
}

/// <summary>
/// Builder for configuring delegate manager settings.
/// </summary>
public sealed class DelegateManagerConfigurationBuilder
{
    private readonly LoggerFactoryOptions _options;

    internal DelegateManagerConfigurationBuilder(LoggerFactoryOptions options)
    {
        _options = options;
    }

    public DelegateManagerConfigurationBuilder WithMaxQueueSize(int size)
    {
        _options.MaxQueueSize = size;
        return this;
    }

    public DelegateManagerConfigurationBuilder WithProcessingInterval(TimeSpan interval)
    {
        _options.ProcessingInterval = interval;
        return this;
    }

    public DelegateManagerConfigurationBuilder DropOnQueueFull(bool drop = true)
    {
        _options.DropOnQueueFull = drop;
        return this;
    }
}

/// <summary>
/// Configuration options for the logger factory.
/// </summary>
public sealed class LoggerFactoryOptions
{
    /// <summary>
    /// Default minimum log level for new loggers.
    /// </summary>
    public LogLevel DefaultMinimumLevel { get; set; } = LogLevel.Information;

    /// <summary>
    /// Buffer capacity for log entries.
    /// </summary>
    public int BufferCapacity { get; set; } = 10000;

    /// <summary>
    /// Maximum batch size for processing.
    /// </summary>
    public int MaxBatchSize { get; set; } = 100;

    /// <summary>
    /// Flush interval for batched processing.
    /// </summary>
    public TimeSpan FlushInterval { get; set; } = TimeSpan.FromSeconds(1);

    /// <summary>
    /// Whether to drop entries when buffer is full.
    /// </summary>
    public bool DropOnBufferFull { get; set; } = true;

    /// <summary>
    /// Maximum queue size for delegate manager.
    /// </summary>
    public int MaxQueueSize { get; set; } = 10000;

    /// <summary>
    /// Processing interval for delegate manager.
    /// </summary>
    public TimeSpan ProcessingInterval { get; set; } = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Whether to drop entries when delegate queue is full.
    /// </summary>
    public bool DropOnQueueFull { get; set; } = true;
}

/// <summary>
/// Statistics about the logger factory.
/// </summary>
public sealed record LoggerFactoryStatistics
{
    public int RegisteredLoggers { get; init; }
    public int RegisteredDelegates { get; init; }
    public BufferHealth BufferHealth { get; init; } = new();
    public LoggingStatistics DelegateManagerStats { get; init; } = new();
}

/// <summary>
/// Generic logger wrapper for typed logging.
/// </summary>
internal sealed class Logger<T> : ILogger<T>
{
    private readonly ILogger _logger;

    public Logger(ILogger logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => _logger.BeginScope(state);
    public bool IsEnabled(LogLevel logLevel) => _logger.IsEnabled(logLevel);
    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter) =>
        _logger.Log(logLevel, eventId, state, exception, formatter);
}