using System;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
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

/// <summary>
/// Fallback executor for CPU operations with float arrays
/// </summary>
internal sealed class CpuFallbackExecutor : IFallbackExecutor<float[], float>, IFallbackAware
{
    private readonly ILogger<CpuFallbackExecutor> _logger;

    public FallbackLevel Level => FallbackLevel.Degraded;

    public CpuFallbackExecutor(ILogger<CpuFallbackExecutor> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public async Task<float> ExecuteAsync(float[] input, string operationName, CancellationToken cancellationToken)
    {
        _logger.LogDebug("Executing CPU fallback for {OperationName} with {InputLength} elements", 
            operationName, input.Length);

        await Task.Yield(); // Make it async
        
        // Simple CPU implementation - sum all elements
        var sum = 0f;
        for (int i = 0; i < input.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            sum += input[i];
        }

        return sum;
    }

    public bool ShouldFallback(Exception exception)
    {
        // Fallback on GPU-specific exceptions but not on general compute errors
        return exception is Orleans.GpuBridge.Abstractions.Exceptions.GpuDeviceException or
               Orleans.GpuBridge.Abstractions.Exceptions.GpuKernelException or
               Orleans.GpuBridge.Abstractions.Exceptions.GpuMemoryException;
    }
}

/// <summary>
/// Generic CPU fallback executor
/// </summary>
internal sealed class GenericCpuFallbackExecutor : IFallbackExecutor<object, object>, IFallbackAware
{
    private readonly ILogger<GenericCpuFallbackExecutor> _logger;

    public FallbackLevel Level => FallbackLevel.Degraded;

    public GenericCpuFallbackExecutor(ILogger<GenericCpuFallbackExecutor> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public async Task<object> ExecuteAsync(object input, string operationName, CancellationToken cancellationToken)
    {
        _logger.LogDebug("Executing generic CPU fallback for {OperationName}", operationName);

        await Task.Yield(); // Make it async
        
        // Generic fallback - just return the input (identity operation)
        return input;
    }

    public bool ShouldFallback(Exception exception)
    {
        return exception is Orleans.GpuBridge.Abstractions.Exceptions.GpuBridgeException;
    }
}

/// <summary>
/// Health monitor for resilience patterns
/// </summary>
public interface IResilienceHealthMonitor
{
    /// <summary>
    /// Gets overall health status
    /// </summary>
    Task<HealthStatus> GetHealthStatusAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets detailed health report
    /// </summary>
    Task<HealthReport> GetHealthReportAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Implementation of resilience health monitor
/// </summary>
internal sealed class ResilienceHealthMonitor : IResilienceHealthMonitor
{
    private readonly ResilienceTelemetryCollector _telemetryCollector;
    private readonly GpuResiliencePolicy _resiliencePolicy;
    private readonly ILogger<ResilienceHealthMonitor> _logger;

    public ResilienceHealthMonitor(
        ResilienceTelemetryCollector telemetryCollector,
        GpuResiliencePolicy resiliencePolicy,
        ILogger<ResilienceHealthMonitor> logger)
    {
        _telemetryCollector = telemetryCollector ?? throw new ArgumentNullException(nameof(telemetryCollector));
        _resiliencePolicy = resiliencePolicy ?? throw new ArgumentNullException(nameof(resiliencePolicy));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public async Task<HealthStatus> GetHealthStatusAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var metrics = _telemetryCollector.GetMetrics();
            var bulkheadMetrics = _resiliencePolicy.GetBulkheadMetrics();

            // Determine overall health based on various factors
            if (bulkheadMetrics.UtilizationPercentage > 90)
                return HealthStatus.Unhealthy;

            var unhealthyComponents = 0;
            var degradedComponents = 0;

            foreach (var health in metrics.ComponentHealth.Values)
            {
                switch (health)
                {
                    case ComponentHealth.Unhealthy:
                        unhealthyComponents++;
                        break;
                    case ComponentHealth.Degraded:
                        degradedComponents++;
                        break;
                }
            }

            if (unhealthyComponents > 0)
                return HealthStatus.Unhealthy;

            if (degradedComponents > 0 || bulkheadMetrics.UtilizationPercentage > 70)
                return HealthStatus.Degraded;

            return HealthStatus.Healthy;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error determining health status");
            return HealthStatus.Unhealthy;
        }
    }

    public async Task<HealthReport> GetHealthReportAsync(CancellationToken cancellationToken = default)
    {
        var metrics = _telemetryCollector.GetMetrics();
        var bulkheadMetrics = _resiliencePolicy.GetBulkheadMetrics();
        var overallStatus = await GetHealthStatusAsync(cancellationToken);

        return new HealthReport(
            OverallStatus: overallStatus,
            ComponentHealth: metrics.ComponentHealth,
            BulkheadUtilization: bulkheadMetrics.UtilizationPercentage,
            CircuitBreakerStates: metrics.CircuitBreakerStates,
            FallbackLevels: metrics.FallbackLevels,
            RecentEvents: metrics.RecentHealthEvents,
            Timestamp: DateTimeOffset.UtcNow);
    }
}

/// <summary>
/// Health status enumeration
/// </summary>
public enum HealthStatus
{
    Unknown = 0,
    Healthy = 1,
    Degraded = 2,
    Unhealthy = 3
}

/// <summary>
/// Comprehensive health report
/// </summary>
public readonly record struct HealthReport(
    HealthStatus OverallStatus,
    IReadOnlyDictionary<string, ComponentHealth> ComponentHealth,
    double BulkheadUtilization,
    IReadOnlyDictionary<string, CircuitBreakerState> CircuitBreakerStates,
    IReadOnlyDictionary<string, FallbackLevel> FallbackLevels,
    IReadOnlyList<HealthEvent> RecentEvents,
    DateTimeOffset Timestamp);

/// <summary>
/// Telemetry configuration options
/// </summary>
public sealed class TelemetryOptions
{
    /// <summary>
    /// Metrics retention period
    /// </summary>
    public TimeSpan RetentionPeriod { get; set; } = TimeSpan.FromHours(24);

    /// <summary>
    /// Health check interval
    /// </summary>
    public TimeSpan HealthCheckInterval { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Whether to enable detailed tracing
    /// </summary>
    public bool EnableDetailedTracing { get; set; } = false;
}