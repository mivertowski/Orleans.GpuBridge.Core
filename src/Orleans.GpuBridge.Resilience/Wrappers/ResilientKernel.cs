using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Abstractions.Exceptions;
using Orleans.GpuBridge.Resilience.Policies;
using Orleans.GpuBridge.Resilience.Telemetry;
using Orleans.GpuBridge.Resilience.Chaos;

namespace Orleans.GpuBridge.Resilience.Wrappers;

/// <summary>
/// Resilient wrapper for GPU kernels with comprehensive error handling, retries, and fallback
/// </summary>
public sealed class ResilientKernel<TIn, TOut> : IGpuKernel<TIn, TOut>, IDisposable
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

    /// <summary>
    /// Submits a batch for execution with comprehensive resilience patterns
    /// </summary>
    public async ValueTask<KernelHandle> SubmitBatchAsync(
        IReadOnlyList<TIn> items,
        GpuExecutionHints? hints = null,
        CancellationToken ct = default)
    {
        var operationName = $"SubmitBatch_{typeof(TIn).Name}_{items.Count}items";
        var startTime = DateTimeOffset.UtcNow;

        try
        {
            // Apply chaos engineering if enabled
            if (_chaosEngineer != null)
            {
                return await _chaosEngineer.ExecuteWithChaosAsync(
                    async (cancellationToken) => await ExecuteSubmitBatchInternalAsync(items, hints, operationName, cancellationToken),
                    operationName,
                    ct);
            }
            else
            {
                return await ExecuteSubmitBatchInternalAsync(items, hints, operationName, ct);
            }
        }
        catch (Exception ex) when (!(ex is OperationCanceledException))
        {
            var duration = DateTimeOffset.UtcNow - startTime;
            _telemetryCollector.RecordOperation(operationName, duration, false, ex.GetType().Name);
            
            _logger.LogError(ex, 
                "Batch submission failed for {ItemCount} items after {Duration}ms: {ErrorType}",
                items.Count, duration.TotalMilliseconds, ex.GetType().Name);
            
            throw;
        }
    }

    /// <summary>
    /// Reads results with resilience patterns
    /// </summary>
    public async IAsyncEnumerable<TOut> ReadResultsAsync(
        KernelHandle handle,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var operationName = $"ReadResults_{handle.Id}";
        var startTime = DateTimeOffset.UtcNow;
        var resultCount = 0;

        try
        {
            await foreach (var result in ExecuteReadResultsInternalAsync(handle, operationName, ct))
            {
                resultCount++;
                yield return result;
            }

            var duration = DateTimeOffset.UtcNow - startTime;
            _telemetryCollector.RecordOperation(operationName, duration, true);
            
            _logger.LogDebug(
                "Successfully read {ResultCount} results from handle {HandleId} in {Duration}ms",
                resultCount, handle.Id, duration.TotalMilliseconds);
        }
        catch (Exception ex) when (!(ex is OperationCanceledException))
        {
            var duration = DateTimeOffset.UtcNow - startTime;
            _telemetryCollector.RecordOperation(operationName, duration, false, ex.GetType().Name);
            
            _logger.LogError(ex,
                "Failed to read results from handle {HandleId} after {Duration}ms (read {ResultCount} results): {ErrorType}",
                handle.Id, duration.TotalMilliseconds, resultCount, ex.GetType().Name);
            
            throw;
        }
    }

    /// <summary>
    /// Gets kernel information with resilience
    /// </summary>
    public async ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default)
    {
        var operationName = "GetKernelInfo";
        var startTime = DateTimeOffset.UtcNow;

        try
        {
            return await _resiliencePolicy.ExecuteKernelOperationAsync(
                async (cancellationToken) =>
                {
                    var info = await _innerKernel.GetInfoAsync(cancellationToken);
                    
                    _logger.LogDebug("Retrieved kernel info: {KernelId} - {Description}", 
                        info.Id.Value, info.Description);
                    
                    return info;
                },
                operationName,
                ct);
        }
        catch (Exception ex) when (!(ex is OperationCanceledException))
        {
            var duration = DateTimeOffset.UtcNow - startTime;
            _telemetryCollector.RecordOperation(operationName, duration, false, ex.GetType().Name);
            
            _logger.LogError(ex, "Failed to get kernel info after {Duration}ms: {ErrorType}",
                duration.TotalMilliseconds, ex.GetType().Name);
            
            throw;
        }
        finally
        {
            var duration = DateTimeOffset.UtcNow - startTime;
            _telemetryCollector.RecordOperation(operationName, duration, true);
        }
    }

    /// <summary>
    /// Internal batch submission with resilience
    /// </summary>
    private async Task<KernelHandle> ExecuteSubmitBatchInternalAsync(
        IReadOnlyList<TIn> items,
        GpuExecutionHints? hints,
        string operationName,
        CancellationToken ct)
    {
        return await _resiliencePolicy.ExecuteKernelOperationAsync(
            async (cancellationToken) =>
            {
                await _executionSemaphore.WaitAsync(cancellationToken);
                try
                {
                    _logger.LogDebug("Submitting batch of {ItemCount} items to kernel", items.Count);
                    
                    var handle = await _innerKernel.SubmitBatchAsync(items, hints, cancellationToken);
                    
                    _logger.LogDebug("Batch submitted successfully with handle {HandleId}", handle.Id);
                    return handle;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, 
                        "Batch submission attempt failed for {ItemCount} items: {ErrorMessage}",
                        items.Count, ex.Message);
                    throw;
                }
                finally
                {
                    _executionSemaphore.Release();
                }
            },
            operationName,
            ct);
    }

    /// <summary>
    /// Internal result reading with resilience
    /// </summary>
    private async IAsyncEnumerable<TOut> ExecuteReadResultsInternalAsync(
        KernelHandle handle,
        string operationName,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var resultEnumerator = _innerKernel.ReadResultsAsync(handle, ct).GetAsyncEnumerator(ct);
        var resultCount = 0;
        
        try
        {
            while (await _resiliencePolicy.ExecuteKernelOperationAsync(
                async (cancellationToken) => await resultEnumerator.MoveNextAsync(cancellationToken).ConfigureAwait(false),
                $"{operationName}_MoveNext",
                ct))
            {
                var result = resultEnumerator.Current;
                resultCount++;
                
                _logger.LogTrace("Read result {ResultIndex} from handle {HandleId}", 
                    resultCount, handle.Id);
                
                yield return result;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, 
                "Error reading results from handle {HandleId} after {ResultCount} results: {ErrorMessage}",
                handle.Id, resultCount, ex.Message);
            throw;
        }
        finally
        {
            if (resultEnumerator != null)
            {
                try
                {
                    await resultEnumerator.DisposeAsync();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error disposing result enumerator for handle {HandleId}", handle.Id);
                }
            }
        }
    }

    /// <summary>
    /// Gets resilience metrics for this kernel
    /// </summary>
    public BulkheadMetrics GetBulkheadMetrics()
    {
        return _resiliencePolicy.GetBulkheadMetrics();
    }

    /// <summary>
    /// Performs health check on the kernel
    /// </summary>
    public async Task<bool> HealthCheckAsync(CancellationToken ct = default)
    {
        try
        {
            var info = await GetInfoAsync(ct);
            return !string.IsNullOrEmpty(info.Id.Value);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Kernel health check failed: {ErrorMessage}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Disposes the resilient kernel wrapper
    /// </summary>
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
    /// Finalizer for cleanup
    /// </summary>
    ~ResilientKernel()
    {
        Dispose();
    }
}