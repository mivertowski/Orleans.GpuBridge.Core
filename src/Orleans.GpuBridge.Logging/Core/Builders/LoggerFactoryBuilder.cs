using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Logging.Abstractions;
using Orleans.GpuBridge.Logging.Delegates;

namespace Orleans.GpuBridge.Logging.Core;

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
