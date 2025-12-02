// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Polly;
using Polly.CircuitBreaker;
using Polly.Retry;
using Polly.Timeout;
using Orleans.GpuBridge.Abstractions.Exceptions;

namespace Orleans.GpuBridge.Resilience.Policies;

/// <summary>
/// Comprehensive resilience policy for GPU operations with Polly integration.
/// </summary>
public sealed class GpuResiliencePolicy : IDisposable
{
    private readonly ILogger<GpuResiliencePolicy> _logger;
    private readonly GpuResiliencePolicyOptions _options;
    private readonly ResiliencePipeline _kernelExecutionPipeline;
    private readonly ResiliencePipeline _deviceOperationPipeline;
    private readonly ResiliencePipeline _memoryAllocationPipeline;
    private readonly ResiliencePipeline _compilationPipeline;
    private readonly SemaphoreSlim _bulkheadSemaphore;
    private readonly CancellationTokenSource _shutdownTokenSource;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuResiliencePolicy"/> class.
    /// </summary>
    /// <param name="logger">The logger.</param>
    /// <param name="options">The policy options.</param>
    public GpuResiliencePolicy(
        ILogger<GpuResiliencePolicy> logger,
        IOptions<GpuResiliencePolicyOptions> options)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
        _shutdownTokenSource = new CancellationTokenSource();

        // Initialize bulkhead semaphore for resource isolation
        _bulkheadSemaphore = new SemaphoreSlim(
            _options.BulkheadOptions.MaxConcurrentOperations,
            _options.BulkheadOptions.MaxConcurrentOperations);

        // Build resilience pipelines
        _kernelExecutionPipeline = BuildKernelExecutionPipeline();
        _deviceOperationPipeline = BuildDeviceOperationPipeline();
        _memoryAllocationPipeline = BuildMemoryAllocationPipeline();
        _compilationPipeline = BuildCompilationPipeline();

        _logger.LogInformation("GpuResiliencePolicy initialized with comprehensive resilience patterns");
    }

    /// <summary>
    /// Executes a kernel operation with full resilience patterns.
    /// </summary>
    /// <typeparam name="TResult">The result type.</typeparam>
    /// <param name="operation">The operation to execute.</param>
    /// <param name="operationName">The operation name for logging.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The operation result.</returns>
    public async Task<TResult> ExecuteKernelOperationAsync<TResult>(
        Func<CancellationToken, Task<TResult>> operation,
        string operationName,
        CancellationToken cancellationToken = default)
    {
        using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken, _shutdownTokenSource.Token);

        return await _kernelExecutionPipeline.ExecuteAsync(
            async ct =>
            {
                _logger.LogDebug("Executing kernel operation: {OperationName}", operationName);

                var result = await operation(ct);

                _logger.LogDebug("Kernel operation completed successfully: {OperationName}", operationName);
                return result;
            },
            linkedCts.Token);
    }

    /// <summary>
    /// Executes a device operation with circuit breaker and retry logic.
    /// </summary>
    /// <typeparam name="TResult">The result type.</typeparam>
    /// <param name="operation">The operation to execute.</param>
    /// <param name="deviceName">The device name.</param>
    /// <param name="operationName">The operation name for logging.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The operation result.</returns>
    public async Task<TResult> ExecuteDeviceOperationAsync<TResult>(
        Func<CancellationToken, Task<TResult>> operation,
        string deviceName,
        string operationName,
        CancellationToken cancellationToken = default)
    {
        using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken, _shutdownTokenSource.Token);

        return await _deviceOperationPipeline.ExecuteAsync(
            async ct =>
            {
                _logger.LogDebug("Executing device operation: {DeviceName}.{OperationName}",
                    deviceName, operationName);

                var result = await operation(ct);

                _logger.LogDebug("Device operation completed successfully: {DeviceName}.{OperationName}",
                    deviceName, operationName);
                return result;
            },
            linkedCts.Token);
    }

    /// <summary>
    /// Executes memory allocation with bulkhead isolation.
    /// </summary>
    /// <typeparam name="TResult">The result type.</typeparam>
    /// <param name="operation">The operation to execute.</param>
    /// <param name="requestedBytes">The requested bytes.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The operation result.</returns>
    public async Task<TResult> ExecuteMemoryAllocationAsync<TResult>(
        Func<CancellationToken, Task<TResult>> operation,
        long requestedBytes,
        CancellationToken cancellationToken = default)
    {
        using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken, _shutdownTokenSource.Token);

        return await _memoryAllocationPipeline.ExecuteAsync(
            async ct =>
            {
                await _bulkheadSemaphore.WaitAsync(ct);

                try
                {
                    _logger.LogDebug("Executing memory allocation: {RequestedBytes:N0} bytes", requestedBytes);

                    var result = await operation(ct);

                    _logger.LogDebug("Memory allocation completed successfully: {RequestedBytes:N0} bytes",
                        requestedBytes);
                    return result;
                }
                finally
                {
                    _bulkheadSemaphore.Release();
                }
            },
            linkedCts.Token);
    }

    /// <summary>
    /// Executes kernel compilation with extended timeouts and fallback strategies.
    /// </summary>
    /// <typeparam name="TResult">The result type.</typeparam>
    /// <param name="operation">The operation to execute.</param>
    /// <param name="kernelName">The kernel name.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The operation result.</returns>
    public async Task<TResult> ExecuteCompilationAsync<TResult>(
        Func<CancellationToken, Task<TResult>> operation,
        string kernelName,
        CancellationToken cancellationToken = default)
    {
        using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken, _shutdownTokenSource.Token);

        return await _compilationPipeline.ExecuteAsync(
            async ct =>
            {
                _logger.LogDebug("Executing kernel compilation: {KernelName}", kernelName);

                var result = await operation(ct);

                _logger.LogDebug("Kernel compilation completed successfully: {KernelName}", kernelName);
                return result;
            },
            linkedCts.Token);
    }

    /// <summary>
    /// Builds the kernel execution resilience pipeline.
    /// </summary>
    private ResiliencePipeline BuildKernelExecutionPipeline()
    {
        return new ResiliencePipelineBuilder()
            .AddRetry(new RetryStrategyOptions
            {
                MaxRetryAttempts = _options.RetryOptions.MaxAttempts,
                BackoffType = DelayBackoffType.Exponential,
                Delay = _options.RetryOptions.BaseDelay,
                MaxDelay = _options.RetryOptions.MaxDelay,
                UseJitter = true,
                ShouldHandle = new PredicateBuilder()
                    .Handle<GpuKernelException>()
                    .Handle<GpuOperationException>()
                    .Handle<TaskCanceledException>(ex => !ex.CancellationToken.IsCancellationRequested),
                OnRetry = args =>
                {
                    _logger.LogWarning(
                        "Kernel operation retry {Attempt}/{MaxAttempts}: {Exception}",
                        args.AttemptNumber + 1,
                        _options.RetryOptions.MaxAttempts,
                        args.Outcome.Exception?.Message);
                    return ValueTask.CompletedTask;
                }
            })
            .AddCircuitBreaker(new CircuitBreakerStrategyOptions
            {
                FailureRatio = _options.CircuitBreakerOptions.FailureRatio,
                SamplingDuration = _options.CircuitBreakerOptions.SamplingDuration,
                MinimumThroughput = _options.CircuitBreakerOptions.MinimumThroughput,
                BreakDuration = _options.CircuitBreakerOptions.BreakDuration,
                ShouldHandle = new PredicateBuilder()
                    .Handle<GpuKernelException>()
                    .Handle<GpuOperationException>(),
                OnOpened = args =>
                {
                    _logger.LogError("Kernel execution circuit breaker opened: {Exception}",
                        args.Outcome.Exception?.Message);
                    return ValueTask.CompletedTask;
                },
                OnClosed = args =>
                {
                    _logger.LogInformation("Kernel execution circuit breaker closed");
                    return ValueTask.CompletedTask;
                },
                OnHalfOpened = args =>
                {
                    _logger.LogInformation("Kernel execution circuit breaker half-opened");
                    return ValueTask.CompletedTask;
                }
            })
            .AddTimeout(_options.TimeoutOptions.KernelExecution)
            .Build();
    }

    /// <summary>
    /// Builds the device operation resilience pipeline.
    /// </summary>
    private ResiliencePipeline BuildDeviceOperationPipeline()
    {
        return new ResiliencePipelineBuilder()
            .AddRetry(new RetryStrategyOptions
            {
                MaxRetryAttempts = _options.RetryOptions.MaxAttempts / 2,
                BackoffType = DelayBackoffType.Linear,
                Delay = _options.RetryOptions.BaseDelay,
                MaxDelay = TimeSpan.FromSeconds(5),
                UseJitter = true,
                ShouldHandle = new PredicateBuilder()
                    .Handle<GpuDeviceException>()
                    .Handle<GpuOperationException>(),
                OnRetry = args =>
                {
                    _logger.LogWarning(
                        "Device operation retry {Attempt}/{MaxAttempts}: {Exception}",
                        args.AttemptNumber + 1,
                        _options.RetryOptions.MaxAttempts / 2,
                        args.Outcome.Exception?.Message);
                    return ValueTask.CompletedTask;
                }
            })
            .AddCircuitBreaker(new CircuitBreakerStrategyOptions
            {
                FailureRatio = 0.3,
                SamplingDuration = TimeSpan.FromMinutes(1),
                MinimumThroughput = 5,
                BreakDuration = TimeSpan.FromMinutes(2),
                ShouldHandle = new PredicateBuilder()
                    .Handle<GpuDeviceException>()
                    .Handle<GpuOperationException>(),
                OnOpened = args =>
                {
                    _logger.LogError("Device operation circuit breaker opened: {Exception}",
                        args.Outcome.Exception?.Message);
                    return ValueTask.CompletedTask;
                }
            })
            .AddTimeout(_options.TimeoutOptions.DeviceOperation)
            .Build();
    }

    /// <summary>
    /// Builds the memory allocation resilience pipeline.
    /// </summary>
    private ResiliencePipeline BuildMemoryAllocationPipeline()
    {
        return new ResiliencePipelineBuilder()
            .AddRetry(new RetryStrategyOptions
            {
                MaxRetryAttempts = 2,
                BackoffType = DelayBackoffType.Linear,
                Delay = TimeSpan.FromMilliseconds(100),
                MaxDelay = TimeSpan.FromSeconds(1),
                ShouldHandle = new PredicateBuilder()
                    .Handle<GpuMemoryException>()
                    .Handle<OutOfMemoryException>(),
                OnRetry = args =>
                {
                    _logger.LogWarning(
                        "Memory allocation retry {Attempt}/2: {Exception}",
                        args.AttemptNumber + 1,
                        args.Outcome.Exception?.Message);
                    return ValueTask.CompletedTask;
                }
            })
            .AddTimeout(_options.TimeoutOptions.MemoryAllocation)
            .Build();
    }

    /// <summary>
    /// Builds the compilation resilience pipeline.
    /// </summary>
    private ResiliencePipeline BuildCompilationPipeline()
    {
        return new ResiliencePipelineBuilder()
            .AddRetry(new RetryStrategyOptions
            {
                MaxRetryAttempts = 1,
                BackoffType = DelayBackoffType.Constant,
                Delay = TimeSpan.FromSeconds(1),
                ShouldHandle = new PredicateBuilder()
                    .Handle<GpuKernelException>()
                    .Handle<InvalidOperationException>(),
                OnRetry = args =>
                {
                    _logger.LogWarning(
                        "Kernel compilation retry: {Exception}",
                        args.Outcome.Exception?.Message);
                    return ValueTask.CompletedTask;
                }
            })
            .AddTimeout(_options.TimeoutOptions.KernelCompilation)
            .Build();
    }

    /// <summary>
    /// Gets current bulkhead utilization metrics.
    /// </summary>
    /// <returns>The bulkhead metrics.</returns>
    public BulkheadMetrics GetBulkheadMetrics()
    {
        var available = _bulkheadSemaphore.CurrentCount;
        var total = _options.BulkheadOptions.MaxConcurrentOperations;
        var inUse = total - available;

        return new BulkheadMetrics(
            TotalSlots: total,
            AvailableSlots: available,
            InUseSlots: inUse,
            UtilizationPercentage: (double)inUse / total * 100);
    }

    /// <summary>
    /// Initiates graceful shutdown of resilience policies.
    /// </summary>
    /// <param name="timeout">The shutdown timeout.</param>
    /// <returns>A task representing the shutdown operation.</returns>
    public async Task ShutdownAsync(TimeSpan timeout = default)
    {
        if (timeout == default)
            timeout = TimeSpan.FromSeconds(30);

        _logger.LogInformation("Initiating graceful shutdown of resilience policies");

        _shutdownTokenSource.Cancel();

        // Wait for bulkhead to drain
        var deadline = DateTime.UtcNow.Add(timeout);
        while (DateTime.UtcNow < deadline && _bulkheadSemaphore.CurrentCount < _options.BulkheadOptions.MaxConcurrentOperations)
        {
            await Task.Delay(100);
        }

        _logger.LogInformation("Resilience policy shutdown completed");
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;

        _disposed = true;
        _shutdownTokenSource?.Cancel();
        _shutdownTokenSource?.Dispose();
        _bulkheadSemaphore?.Dispose();

        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Metrics for bulkhead isolation monitoring.
/// </summary>
public readonly record struct BulkheadMetrics(
    int TotalSlots,
    int AvailableSlots,
    int InUseSlots,
    double UtilizationPercentage);
