using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Logging.Abstractions;
using Orleans.GpuBridge.Logging.Configuration;
using Orleans.GpuBridge.Logging.Core;
using Orleans.GpuBridge.Logging.Delegates;

namespace Orleans.GpuBridge.Logging.Extensions;

/// <summary>
/// Dependency injection extensions for Orleans GPU Bridge logging system.
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Adds Orleans GPU Bridge logging system to the service collection.
    /// </summary>
    /// <param name="services">Service collection</param>
    /// <param name="configuration">Configuration section</param>
    /// <returns>Service collection for chaining</returns>
    public static IServiceCollection AddGpuBridgeLogging(this IServiceCollection services,
        IConfiguration configuration)
    {
        return services.AddGpuBridgeLogging(options =>
        {
            configuration.GetSection(GpuBridgeLoggingOptions.SectionName).Bind(options);
        });
    }

    /// <summary>
    /// Adds Orleans GPU Bridge logging system to the service collection.
    /// </summary>
    /// <param name="services">Service collection</param>
    /// <param name="configure">Configuration action</param>
    /// <returns>Service collection for chaining</returns>
    public static IServiceCollection AddGpuBridgeLogging(this IServiceCollection services,
        Action<GpuBridgeLoggingOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);

        // Configure options
        services.Configure(configure);
        services.AddSingleton<IValidateOptions<GpuBridgeLoggingOptions>, GpuBridgeLoggingOptionsValidator>();

        // Add core services
        services.TryAddSingleton<LoggerDelegateManager>();
        services.TryAddSingleton<LogBuffer>(provider =>
        {
            var options = provider.GetRequiredService<IOptions<GpuBridgeLoggingOptions>>().Value;
            return new LogBuffer(new LogBufferOptions
            {
                Capacity = options.Buffer.Capacity,
                MaxBatchSize = options.Buffer.MaxBatchSize,
                FlushInterval = options.Buffer.FlushInterval,
                DropOnOverflow = options.Buffer.DropOnFull,
                PrioritizeHighSeverity = options.Buffer.PrioritizeHighSeverity
            });
        });

        // Add logger factory
        services.TryAddSingleton<Core.LoggerFactory>(provider =>
        {
            var options = provider.GetRequiredService<IOptions<GpuBridgeLoggingOptions>>().Value;
            var factoryOptions = new Core.LoggerFactoryOptions
            {
                DefaultMinimumLevel = options.MinimumLevel,
                BufferCapacity = options.Buffer.Capacity,
                MaxBatchSize = options.Buffer.MaxBatchSize,
                FlushInterval = options.Buffer.FlushInterval,
                DropOnBufferFull = options.Buffer.DropOnFull,
                MaxQueueSize = options.DelegateManager.MaxQueueSize,
                ProcessingInterval = options.DelegateManager.ProcessingInterval,
                DropOnQueueFull = options.DelegateManager.DropOnQueueFull
            };

            var factory = new Core.LoggerFactory(factoryOptions);
            ConfigureDelegates(factory, options, provider);
            return factory;
        });

        // Replace default logging with GPU Bridge logging
        services.RemoveAll<ILoggerProvider>();
        services.TryAddSingleton<ILoggerProvider, GpuBridgeLoggerProvider>();

        return services;
    }

    /// <summary>
    /// Adds Orleans GPU Bridge logging with fluent configuration.
    /// </summary>
    /// <param name="services">Service collection</param>
    /// <param name="configure">Fluent configuration builder</param>
    /// <returns>Service collection for chaining</returns>
    public static IServiceCollection AddGpuBridgeLogging(this IServiceCollection services,
        Func<LoggerFactoryBuilder, LoggerFactoryBuilder> configure)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);

        var builder = new LoggerFactoryBuilder();
        builder = configure(builder);
        var loggerFactory = builder.Build();

        services.TryAddSingleton(loggerFactory);
        services.TryAddSingleton<ILoggerProvider, GpuBridgeLoggerProvider>();

        return services;
    }

    /// <summary>
    /// Adds console logging delegate.
    /// </summary>
    public static IServiceCollection AddGpuBridgeConsoleLogging(this IServiceCollection services,
        Action<ConsoleLoggerOptions>? configure = null)
    {
        services.TryAddSingleton<ConsoleLoggerDelegate>(provider =>
        {
            var options = new ConsoleLoggerOptions();
            configure?.Invoke(options);
            return new ConsoleLoggerDelegate(options);
        });

        return services;
    }

    /// <summary>
    /// Adds file logging delegate.
    /// </summary>
    public static IServiceCollection AddGpuBridgeFileLogging(this IServiceCollection services,
        Action<FileLoggerOptions>? configure = null)
    {
        services.TryAddSingleton<FileLoggerDelegate>(provider =>
        {
            var options = new FileLoggerOptions();
            configure?.Invoke(options);
            return new FileLoggerDelegate(options);
        });

        return services;
    }

    /// <summary>
    /// Adds telemetry logging delegate.
    /// </summary>
    public static IServiceCollection AddGpuBridgeTelemetryLogging(this IServiceCollection services,
        Action<TelemetryLoggerOptions>? configure = null)
    {
        services.TryAddSingleton<TelemetryLoggerDelegate>(provider =>
        {
            var options = new TelemetryLoggerOptions();
            configure?.Invoke(options);
            return new TelemetryLoggerDelegate(options);
        });

        return services;
    }

    private static void ConfigureDelegates(Core.LoggerFactory factory,
        GpuBridgeLoggingOptions options, IServiceProvider provider)
    {
        // Configure console delegate
        if (options.Console.Enabled)
        {
            var consoleOptions = new ConsoleLoggerOptions
            {
                MinimumLevel = options.Console.MinimumLevel,
                UseColors = options.Console.UseColors,
                IncludeCategory = options.Console.IncludeCategory,
                IncludeCorrelationId = options.Console.IncludeCorrelationId,
                IncludeProperties = options.Console.IncludeProperties,
                IncludeScopes = options.Console.IncludeScopes,
                IncludeThreadId = options.Console.IncludeThreadId,
                IncludeContext = options.Console.IncludeContext,
                IncludeMetrics = options.Console.IncludeMetrics,
                TimestampFormat = Enum.TryParse<ConsoleTimestampFormat>(options.Console.TimestampFormat, out var tsFormat)
                    ? tsFormat : ConsoleTimestampFormat.Local
            };

            factory.DelegateManager.RegisterDelegate(new ConsoleLoggerDelegate(consoleOptions));
        }

        // Configure file delegate
        if (options.File.Enabled)
        {
            var fileOptions = new FileLoggerOptions
            {
                LogDirectory = options.File.LogDirectory,
                BaseFileName = options.File.BaseFileName,
                MinimumLevel = options.File.MinimumLevel,
                MaxFileSizeBytes = options.File.MaxFileSizeBytes,
                RotationInterval = options.File.RotationInterval,
                RetentionDays = options.File.RetentionDays,
                MaxRetainedFiles = options.File.MaxRetainedFiles,
                AutoFlush = options.File.AutoFlush,
                BufferSize = options.File.BufferSize,
                MaxBatchSize = options.File.MaxBatchSize,
                FileNameTimestampFormat = options.File.FileNameTimestampFormat,
                IncludeContext = options.File.IncludeContext
            };

            factory.DelegateManager.RegisterDelegate(new FileLoggerDelegate(fileOptions));
        }

        // Configure telemetry delegate
        if (options.Telemetry.Enabled)
        {
            var telemetryOptions = new TelemetryLoggerOptions
            {
                MinimumLevel = options.Telemetry.MinimumLevel,
                ServiceName = options.Telemetry.ServiceName,
                ServiceVersion = options.Telemetry.ServiceVersion,
                ServiceInstance = options.Telemetry.ServiceInstance,
                MaxBatchSize = options.Telemetry.MaxBatchSize,
                FlushInterval = options.Telemetry.FlushInterval,
                FlushOnError = options.Telemetry.FlushOnError,
                TraceAllLevels = options.Telemetry.TraceAllLevels,
                OtlpEndpoint = options.Telemetry.OtlpEndpoint,
                ApiKey = options.Telemetry.ApiKey,
                CustomHeaders = options.Telemetry.CustomHeaders,
                IncludeResourceInfo = options.Telemetry.IncludeResourceInfo,
                EnableSampling = options.Telemetry.EnableSampling,
                SamplingRate = options.Telemetry.SamplingRate
            };

            var otlpLogger = provider.GetService<ILogger<TelemetryLoggerDelegate>>();
            factory.DelegateManager.RegisterDelegate(new TelemetryLoggerDelegate(telemetryOptions, otlpLogger));
        }
    }
}

/// <summary>
/// Logger provider for GPU Bridge logging system.
/// </summary>
internal sealed class GpuBridgeLoggerProvider : ILoggerProvider
{
    private readonly Core.LoggerFactory _loggerFactory;
    private readonly GpuBridgeLoggingOptions _options;
    private bool _disposed;

    public GpuBridgeLoggerProvider(Core.LoggerFactory loggerFactory,
        IOptions<GpuBridgeLoggingOptions> options)
    {
        _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
        _options = options.Value ?? throw new ArgumentNullException(nameof(options));
    }

    public ILogger CreateLogger(string categoryName)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GpuBridgeLoggerProvider));

        var minimumLevel = _options.GetCategoryLevel(categoryName);
        return new GpuBridgeLogger(categoryName, _loggerFactory.Buffer, minimumLevel);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _loggerFactory?.DisposeAsync().GetAwaiter().GetResult();
            _disposed = true;
        }
    }
}

/// <summary>
/// Options validator for GPU Bridge logging configuration.
/// </summary>
internal sealed class GpuBridgeLoggingOptionsValidator : IValidateOptions<GpuBridgeLoggingOptions>
{
    public ValidateOptionsResult Validate(string? name, GpuBridgeLoggingOptions options)
    {
        var validation = options.ValidateConfiguration();

        return validation.IsValid
            ? ValidateOptionsResult.Success
            : ValidateOptionsResult.Fail(validation.Errors);
    }
}