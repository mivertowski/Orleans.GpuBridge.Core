// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Resilience.Policies;
using Orleans.GpuBridge.Resilience.Telemetry;
using Orleans.GpuBridge.Resilience.Chaos;

namespace Orleans.GpuBridge.Resilience.Wrappers;

/// <summary>
/// Resilient wrapper for GPU kernels with comprehensive error handling, retries, and fallback.
/// </summary>
/// <typeparam name="TIn">Input type for the kernel.</typeparam>
/// <typeparam name="TOut">Output type for the kernel.</typeparam>
public sealed class ResilientKernel<TIn, TOut> : IGpuKernel<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    private readonly IGpuKernel<TIn, TOut> _innerKernel;
    private readonly GpuResiliencePolicy _resiliencePolicy;
    private readonly ResilienceTelemetryCollector _telemetryCollector;
    private readonly IChaosEngineer? _chaosEngineer;
    private readonly ILogger _logger;
    private readonly SemaphoreSlim _executionSemaphore;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="ResilientKernel{TIn, TOut}"/> class.
    /// </summary>
    /// <param name="innerKernel">The inner kernel to wrap.</param>
    /// <param name="resiliencePolicy">The resilience policy to apply.</param>
    /// <param name="telemetryCollector">The telemetry collector.</param>
    /// <param name="logger">The logger.</param>
    /// <param name="chaosEngineer">Optional chaos engineer for testing.</param>
    public ResilientKernel(
        IGpuKernel<TIn, TOut> innerKernel,
        GpuResiliencePolicy resiliencePolicy,
        ResilienceTelemetryCollector telemetryCollector,
        ILogger logger,
        IChaosEngineer? chaosEngineer = null)
    {
        _innerKernel = innerKernel ?? throw new ArgumentNullException(nameof(innerKernel));
        _resiliencePolicy = resiliencePolicy ?? throw new ArgumentNullException(nameof(resiliencePolicy));
        _telemetryCollector = telemetryCollector ?? throw new ArgumentNullException(nameof(telemetryCollector));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _chaosEngineer = chaosEngineer;
        _executionSemaphore = new SemaphoreSlim(1, 1);

        _logger.LogDebug("ResilientKernel wrapper created for {KernelType}", _innerKernel.GetType().Name);
    }

    /// <inheritdoc />
    public string KernelId => _innerKernel.KernelId;

    /// <inheritdoc />
    public string DisplayName => _innerKernel.DisplayName;

    /// <inheritdoc />
    public string BackendProvider => _innerKernel.BackendProvider;

    /// <inheritdoc />
    public bool IsInitialized => _innerKernel.IsInitialized;

    /// <inheritdoc />
    public bool IsGpuAccelerated => _innerKernel.IsGpuAccelerated;

    /// <inheritdoc />
    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        const string operationName = "Initialize";
        var startTime = DateTimeOffset.UtcNow;

        try
        {
            await ExecuteWithResilienceAsync(
                async ct =>
                {
                    await _innerKernel.InitializeAsync(ct);
                    return true;
                },
                operationName,
                cancellationToken);

            var duration = DateTimeOffset.UtcNow - startTime;
            _telemetryCollector.RecordOperation(operationName, duration, true);
            _logger.LogDebug("Kernel initialized successfully in {Duration}ms", duration.TotalMilliseconds);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            var duration = DateTimeOffset.UtcNow - startTime;
            _telemetryCollector.RecordOperation(operationName, duration, false, ex.GetType().Name);
            _logger.LogError(ex, "Kernel initialization failed after {Duration}ms", duration.TotalMilliseconds);
            throw;
        }
    }

    /// <inheritdoc />
    public async Task<TOut> ExecuteAsync(TIn input, CancellationToken cancellationToken = default)
    {
        var operationName = $"Execute_{typeof(TIn).Name}";
        var startTime = DateTimeOffset.UtcNow;

        try
        {
            var result = await ExecuteWithResilienceAsync(
                async ct =>
                {
                    await _executionSemaphore.WaitAsync(ct);
                    try
                    {
                        return await _innerKernel.ExecuteAsync(input, ct);
                    }
                    finally
                    {
                        _executionSemaphore.Release();
                    }
                },
                operationName,
                cancellationToken);

            var duration = DateTimeOffset.UtcNow - startTime;
            _telemetryCollector.RecordOperation(operationName, duration, true);
            _logger.LogDebug("Kernel executed successfully in {Duration}ms", duration.TotalMilliseconds);

            return result;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            var duration = DateTimeOffset.UtcNow - startTime;
            _telemetryCollector.RecordOperation(operationName, duration, false, ex.GetType().Name);
            _logger.LogError(ex, "Kernel execution failed after {Duration}ms", duration.TotalMilliseconds);
            throw;
        }
    }

    /// <inheritdoc />
    public async Task<TOut[]> ExecuteBatchAsync(TIn[] inputs, CancellationToken cancellationToken = default)
    {
        var operationName = $"ExecuteBatch_{typeof(TIn).Name}_{inputs.Length}items";
        var startTime = DateTimeOffset.UtcNow;

        try
        {
            var result = await ExecuteWithResilienceAsync(
                async ct =>
                {
                    await _executionSemaphore.WaitAsync(ct);
                    try
                    {
                        _logger.LogDebug("Executing batch of {ItemCount} items", inputs.Length);
                        return await _innerKernel.ExecuteBatchAsync(inputs, ct);
                    }
                    finally
                    {
                        _executionSemaphore.Release();
                    }
                },
                operationName,
                cancellationToken);

            var duration = DateTimeOffset.UtcNow - startTime;
            _telemetryCollector.RecordOperation(operationName, duration, true);
            _logger.LogDebug("Batch execution completed: {InputCount} inputs -> {OutputCount} outputs in {Duration}ms",
                inputs.Length, result.Length, duration.TotalMilliseconds);

            return result;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            var duration = DateTimeOffset.UtcNow - startTime;
            _telemetryCollector.RecordOperation(operationName, duration, false, ex.GetType().Name);
            _logger.LogError(ex, "Batch execution failed for {ItemCount} items after {Duration}ms",
                inputs.Length, duration.TotalMilliseconds);
            throw;
        }
    }

    /// <inheritdoc />
    public long GetEstimatedExecutionTimeMicroseconds(int inputSize)
    {
        return _innerKernel.GetEstimatedExecutionTimeMicroseconds(inputSize);
    }

    /// <inheritdoc />
    public KernelMemoryRequirements GetMemoryRequirements()
    {
        return _innerKernel.GetMemoryRequirements();
    }

    /// <inheritdoc />
    public KernelValidationResult ValidateInput(TIn input)
    {
        return _innerKernel.ValidateInput(input);
    }

    /// <inheritdoc />
    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        const string operationName = "Warmup";
        var startTime = DateTimeOffset.UtcNow;

        try
        {
            await ExecuteWithResilienceAsync(
                async ct =>
                {
                    await _innerKernel.WarmupAsync(ct);
                    return true;
                },
                operationName,
                cancellationToken);

            var duration = DateTimeOffset.UtcNow - startTime;
            _telemetryCollector.RecordOperation(operationName, duration, true);
            _logger.LogDebug("Kernel warmup completed in {Duration}ms", duration.TotalMilliseconds);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            var duration = DateTimeOffset.UtcNow - startTime;
            _telemetryCollector.RecordOperation(operationName, duration, false, ex.GetType().Name);
            _logger.LogError(ex, "Kernel warmup failed after {Duration}ms", duration.TotalMilliseconds);
            throw;
        }
    }

    /// <summary>
    /// Performs health check on the kernel.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>True if the kernel is healthy.</returns>
    public async Task<bool> HealthCheckAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            return _innerKernel.IsInitialized && await Task.FromResult(_innerKernel.IsGpuAccelerated || true);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Kernel health check failed: {ErrorMessage}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Gets bulkhead metrics for this kernel.
    /// </summary>
    /// <returns>The bulkhead metrics.</returns>
    public BulkheadMetrics GetBulkheadMetrics()
    {
        return _resiliencePolicy.GetBulkheadMetrics();
    }

    private async Task<TResult> ExecuteWithResilienceAsync<TResult>(
        Func<CancellationToken, Task<TResult>> operation,
        string operationName,
        CancellationToken cancellationToken)
    {
        if (_chaosEngineer != null)
        {
            return await _chaosEngineer.ExecuteWithChaosAsync(
                async ct => await _resiliencePolicy.ExecuteKernelOperationAsync(operation, operationName, ct),
                operationName,
                cancellationToken);
        }

        return await _resiliencePolicy.ExecuteKernelOperationAsync(operation, operationName, cancellationToken);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;

        _disposed = true;

        try
        {
            _executionSemaphore?.Dispose();

            if (_innerKernel is IDisposable disposableKernel)
            {
                disposableKernel.Dispose();
            }

            _logger.LogDebug("ResilientKernel disposed for {KernelType}", _innerKernel.GetType().Name);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing ResilientKernel: {ErrorMessage}", ex.Message);
        }

        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Finalizer for cleanup.
    /// </summary>
    ~ResilientKernel()
    {
        Dispose();
    }
}
