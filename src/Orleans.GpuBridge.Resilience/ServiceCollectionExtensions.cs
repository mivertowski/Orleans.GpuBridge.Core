using System;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Orleans.GpuBridge.Resilience.Policies;
using Orleans.GpuBridge.Resilience.Fallback;
using Orleans.GpuBridge.Resilience.RateLimit;
using Orleans.GpuBridge.Resilience.Chaos;
using Orleans.GpuBridge.Resilience.Telemetry;

namespace Orleans.GpuBridge.Resilience;

/// <summary>
/// Extension methods for adding resilience patterns to the service collection
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Adds comprehensive GPU bridge resilience patterns
    /// </summary>
    public static IServiceCollection AddGpuBridgeResilience(
        this IServiceCollection services,
        IConfiguration? configuration = null,
        Action<GpuResiliencePolicyOptions>? configureOptions = null)
    {
        if (services == null) throw new ArgumentNullException(nameof(services));

        // Configure options
        var optionsBuilder = services.Configure<GpuResiliencePolicyOptions>(options =>
        {
            // Set default values
            options.RetryOptions.MaxAttempts = 3;
            options.RetryOptions.BaseDelay = TimeSpan.FromMilliseconds(500);
            options.RetryOptions.MaxDelay = TimeSpan.FromSeconds(30);
            options.RetryOptions.UseExponentialBackoff = true;
            options.RetryOptions.UseJitter = true;

            options.CircuitBreakerOptions.FailureRatio = 0.5;
            options.CircuitBreakerOptions.SamplingDuration = TimeSpan.FromMinutes(2);
            options.CircuitBreakerOptions.MinimumThroughput = 10;
            options.CircuitBreakerOptions.BreakDuration = TimeSpan.FromMinutes(1);
            options.CircuitBreakerOptions.Enabled = true;

            options.TimeoutOptions.KernelExecution = TimeSpan.FromMinutes(5);
            options.TimeoutOptions.DeviceOperation = TimeSpan.FromSeconds(30);
            options.TimeoutOptions.MemoryAllocation = TimeSpan.FromSeconds(10);
            options.TimeoutOptions.KernelCompilation = TimeSpan.FromMinutes(10);

            options.BulkheadOptions.MaxConcurrentOperations = 10;
            options.BulkheadOptions.MaxQueuedOperations = 50;
            options.BulkheadOptions.Enabled = true;
            options.BulkheadOptions.AcquisitionTimeout = TimeSpan.FromSeconds(30);

            options.RateLimitingOptions.MaxRequests = 100;
            options.RateLimitingOptions.TimeWindow = TimeSpan.FromMinutes(1);
            options.RateLimitingOptions.Enabled = true;
            options.RateLimitingOptions.Algorithm = RateLimitingAlgorithm.TokenBucket;
            options.RateLimitingOptions.TokenRefillRate = 10.0;
            options.RateLimitingOptions.MaxBurstSize = 20;

            options.ChaosOptions.Enabled = false;
            options.ChaosOptions.FaultInjectionProbability = 0.01;
            options.ChaosOptions.LatencyInjection.Enabled = false;
            options.ChaosOptions.LatencyInjection.InjectionProbability = 0.05;
            options.ChaosOptions.ExceptionInjection.Enabled = false;
            options.ChaosOptions.ExceptionInjection.InjectionProbability = 0.02;
            options.ChaosOptions.ResourceExhaustion.Enabled = false;
        });

        // Bind configuration if provided
        if (configuration != null)
        {
            optionsBuilder.Bind(configuration.GetSection(GpuResiliencePolicyOptions.SectionName));
        }

        // Apply custom configuration
        if (configureOptions != null)
        {
            services.Configure(configureOptions);
        }

        // Configure fallback chain options
        services.Configure<FallbackChainOptions>(options =>
        {
            options.AutoDegradationEnabled = true;
            options.AutoRecoveryEnabled = true;
            options.DegradationErrorThreshold = 0.5;
            options.RecoveryErrorThreshold = 0.1;
            options.MinimumRequestsForDegradation = 10;
            options.MinimumRequestsForRecovery = 5;
            options.MinimumRecoveryInterval = TimeSpan.FromMinutes(5);
            options.ErrorRateWindow = TimeSpan.FromMinutes(10);
        });

        // Register core resilience services
        services.TryAddSingleton<GpuResiliencePolicy>();
        services.TryAddSingleton<ResilienceTelemetryCollector>();

        // Register rate limiter
        services.TryAddSingleton<IRateLimiter, TokenBucketRateLimiter>();

        // Register chaos engineer (disabled by default)
        services.TryAddSingleton<IChaosEngineer, ChaosEngineer>();

        // Register fallback executors - these will be implemented by specific GPU backends
        services.TryAddSingleton<IFallbackExecutor<float[], float>, CpuFallbackExecutor>();
        services.TryAddSingleton<IFallbackExecutor<object, object>, GenericCpuFallbackExecutor>();

        // Register metrics and monitoring
        services.TryAddSingleton<IResilienceHealthMonitor, ResilienceHealthMonitor>();

        return services;
    }

    /// <summary>
    /// Adds rate limiting with custom configuration
    /// </summary>
    public static IServiceCollection AddGpuRateLimit(
        this IServiceCollection services,
        Action<RateLimitingOptions> configureOptions)
    {
        if (services == null) throw new ArgumentNullException(nameof(services));
        if (configureOptions == null) throw new ArgumentNullException(nameof(configureOptions));

        services.Configure<GpuResiliencePolicyOptions>(options =>
        {
            configureOptions(options.RateLimitingOptions);
        });

        services.TryAddSingleton<IRateLimiter, TokenBucketRateLimiter>();

        return services;
    }

    /// <summary>
    /// Adds chaos engineering with custom configuration
    /// </summary>
    public static IServiceCollection AddGpuChaosEngineering(
        this IServiceCollection services,
        Action<ChaosEngineeringOptions> configureOptions)
    {
        if (services == null) throw new ArgumentNullException(nameof(services));
        if (configureOptions == null) throw new ArgumentNullException(nameof(configureOptions));

        services.Configure<GpuResiliencePolicyOptions>(options =>
        {
            configureOptions(options.ChaosOptions);
            options.ChaosOptions.Enabled = true; // Enable when explicitly configured
        });

        services.TryAddSingleton<IChaosEngineer, ChaosEngineer>();

        return services;
    }

    /// <summary>
    /// Adds fallback chain with custom configuration
    /// </summary>
    public static IServiceCollection AddGpuFallbackChain(
        this IServiceCollection services,
        Action<FallbackChainOptions> configureOptions)
    {
        if (services == null) throw new ArgumentNullException(nameof(services));
        if (configureOptions == null) throw new ArgumentNullException(nameof(configureOptions));

        services.Configure(configureOptions);

        return services;
    }

    /// <summary>
    /// Configures telemetry and monitoring
    /// </summary>
    public static IServiceCollection AddGpuTelemetry(
        this IServiceCollection services,
        Action<TelemetryOptions>? configureOptions = null)
    {
        if (services == null) throw new ArgumentNullException(nameof(services));

        if (configureOptions != null)
        {
            services.Configure(configureOptions);
        }

        services.TryAddSingleton<ResilienceTelemetryCollector>();
        services.TryAddSingleton<IResilienceHealthMonitor, ResilienceHealthMonitor>();

        return services;
    }
}
