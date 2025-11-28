using DotCompute.Abstractions.Timing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using System;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// Service collection extensions for GPU timing features.
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Adds GPU timing services for temporal correctness.
    /// </summary>
    /// <param name="services">Service collection.</param>
    /// <param name="configure">Optional configuration action.</param>
    /// <returns>Service collection for chaining.</returns>
    public static IServiceCollection AddGpuTiming(
        this IServiceCollection services,
        Action<GpuTimingOptions>? configure = null)
    {
        var options = new GpuTimingOptions();
        configure?.Invoke(options);

        // Register options
        services.TryAddSingleton(options);

        // Register timing provider - try DotCompute first, fall back to CPU
        services.TryAddSingleton<IGpuTimingProvider>(sp =>
        {
            var loggerFactory = sp.GetRequiredService<ILoggerFactory>();

            // Try to get DotCompute timing provider if registered
            var dotComputeProvider = sp.GetService<ITimingProvider>();
            if (dotComputeProvider != null)
            {
                var dcLogger = loggerFactory.CreateLogger<DotComputeTimingProvider>();
                var provider = new DotComputeTimingProvider(dotComputeProvider, dcLogger);

                if (options.EnableTimestampInjection)
                {
                    provider.EnableTimestampInjection(true);
                }

                return provider;
            }

            // Fall back to CPU timing provider
            var cpuLogger = loggerFactory.CreateLogger<CpuTimingProvider>();
            return new CpuTimingProvider(cpuLogger);
        });

        // Register clock calibrator with timing provider
        services.TryAddSingleton<GpuClockCalibrator>(sp =>
        {
            var timingProvider = sp.GetRequiredService<IGpuTimingProvider>();
            var logger = sp.GetRequiredService<ILogger<GpuClockCalibrator>>();
            return new GpuClockCalibrator(timingProvider, logger);
        });

        return services;
    }

    /// <summary>
    /// Adds temporal actor services with GPU timing and ring kernels.
    /// </summary>
    /// <param name="services">Service collection.</param>
    /// <param name="configure">Optional configuration action.</param>
    /// <returns>Service collection for chaining.</returns>
    public static IServiceCollection AddTemporalActors(
        this IServiceCollection services,
        Action<TemporalActorOptions>? configure = null)
    {
        var options = new TemporalActorOptions();
        configure?.Invoke(options);

        // Add timing services
        services.AddGpuTiming(timingOpts =>
        {
            timingOpts.DeviceIndex = options.DeviceIndex;
            timingOpts.EnableTimestampInjection = options.EnableTimestampInjection;
            timingOpts.AutoCalibrate = options.AutoCalibrate;
            timingOpts.CalibrationSampleCount = options.CalibrationSampleCount;
        });

        // Register ring kernel manager for persistent GPU threads
        if (options.EnableRingKernels)
        {
            services.TryAddSingleton<RingKernelManager>();
        }

        return services;
    }
}

/// <summary>
/// Configuration options for GPU timing features.
/// </summary>
public sealed class GpuTimingOptions
{
    /// <summary>
    /// GPU device index to use for timing (default: 0).
    /// </summary>
    public int DeviceIndex { get; set; } = 0;

    /// <summary>
    /// Enable automatic timestamp injection at kernel entry (default: true).
    /// </summary>
    public bool EnableTimestampInjection { get; set; } = true;

    /// <summary>
    /// Automatically calibrate clock on startup (default: true).
    /// </summary>
    public bool AutoCalibrate { get; set; } = true;

    /// <summary>
    /// Number of samples for clock calibration (default: 1000).
    /// </summary>
    public int CalibrationSampleCount { get; set; } = 1000;

    /// <summary>
    /// Calibration refresh interval (default: 5 minutes).
    /// </summary>
    public TimeSpan CalibrationInterval { get; set; } = TimeSpan.FromMinutes(5);
}

/// <summary>
/// Configuration options for temporal actors.
/// </summary>
public sealed class TemporalActorOptions
{
    /// <summary>
    /// GPU device index (default: 0).
    /// </summary>
    public int DeviceIndex { get; set; } = 0;

    /// <summary>
    /// Enable timestamp injection (default: true).
    /// </summary>
    public bool EnableTimestampInjection { get; set; } = true;

    /// <summary>
    /// Auto-calibrate clocks (default: true).
    /// </summary>
    public bool AutoCalibrate { get; set; } = true;

    /// <summary>
    /// Calibration sample count (default: 1000).
    /// </summary>
    public int CalibrationSampleCount { get; set; } = 1000;

    /// <summary>
    /// Enable ring kernels for persistent GPU threads (default: true).
    /// </summary>
    public bool EnableRingKernels { get; set; } = true;

    /// <summary>
    /// Ring buffer size for message queue (default: 4096).
    /// </summary>
    public int MessageQueueSize { get; set; } = 4096;

    /// <summary>
    /// Maximum number of actors per ring kernel (default: 1024).
    /// </summary>
    public int MaxActorsPerRing { get; set; } = 1024;

    /// <summary>
    /// Enable device-wide barriers for coordination (default: false).
    /// </summary>
    public bool EnableBarriers { get; set; } = false;

    /// <summary>
    /// Memory ordering mode for causal correctness (default: ReleaseAcquire).
    /// </summary>
    public string MemoryOrdering { get; set; } = "ReleaseAcquire";
}
