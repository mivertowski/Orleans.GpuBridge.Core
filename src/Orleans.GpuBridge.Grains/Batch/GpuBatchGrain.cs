using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Concurrency;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Runtime;

namespace Orleans.GpuBridge.Grains.Batch;

/// <summary>
/// GPU batch processing grain with placement strategy
/// </summary>
[StatelessWorker(1)] // One per silo for better GPU utilization
[Reentrant] // Allow concurrent calls
public sealed class GpuBatchGrain<TIn, TOut> : Grain, IGpuBatchGrain<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    private readonly ILogger<GpuBatchGrain<TIn, TOut>> _logger;
    private readonly SemaphoreSlim _concurrencyLimit;
    private IGpuBridge _bridge = default!;
    private IGpuKernel<TIn, TOut> _kernel = default!;
    private KernelId _kernelId = default!;
    
    public GpuBatchGrain(ILogger<GpuBatchGrain<TIn, TOut>> logger)
    {
        _logger = logger;
        _concurrencyLimit = new SemaphoreSlim(
            Environment.ProcessorCount * 2,
            Environment.ProcessorCount * 2);
    }
    
    public override async Task OnActivateAsync(CancellationToken ct)
    {
        // Extract kernelId from compound primary key (format: "guid+kernelId" or just "kernelId")
        var primaryKey = this.GetPrimaryKeyString();
        var kernelIdString = primaryKey.Contains('+')
            ? primaryKey.Split('+', 2)[1]
            : primaryKey;

        _kernelId = KernelId.Parse(kernelIdString);
        _bridge = ServiceProvider.GetRequiredService<IGpuBridge>();
        _kernel = await _bridge.GetKernelAsync<TIn, TOut>(_kernelId, ct);

        _logger.LogInformation(
            "Activated GPU batch grain for kernel {KernelId}",
            _kernelId);

        await base.OnActivateAsync(ct);
    }
    
    public async Task<GpuBatchResult<TOut>> ExecuteAsync(
        IReadOnlyList<TIn> batch,
        GpuExecutionHints? hints = null)
    {
        await _concurrencyLimit.WaitAsync();
        try
        {
            var stopwatch = Stopwatch.StartNew();
            
            _logger.LogDebug(
                "Executing batch of {Count} items on kernel {KernelId}",
                batch.Count, _kernelId);
            
            // Submit batch to kernel
            var handle = await _kernel.SubmitBatchAsync(batch, hints);
            
            // Collect results
            var results = new List<TOut>();
            await foreach (var result in _kernel.ReadResultsAsync(handle))
            {
                results.Add(result);
            }
            
            stopwatch.Stop();

            _logger.LogInformation(
                "Executed batch of {Count} items in {ElapsedMs}ms",
                batch.Count, stopwatch.ElapsedMilliseconds);

            // Create basic metrics for monitoring
            var metrics = new GpuBatchMetrics(
                TotalItems: batch.Count,
                SubBatchCount: 1,
                SuccessfulSubBatches: 1,
                TotalExecutionTime: stopwatch.Elapsed,
                KernelExecutionTime: stopwatch.Elapsed,
                MemoryTransferTime: TimeSpan.Zero,
                Throughput: batch.Count / stopwatch.Elapsed.TotalSeconds,
                MemoryAllocated: 0,
                DeviceType: "CPU",
                DeviceName: "CPU Fallback");

            return new GpuBatchResult<TOut>(
                results,
                stopwatch.Elapsed,
                handle.Id,
                _kernelId,
                Error: null,
                Metrics: metrics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Failed to execute batch on kernel {KernelId}",
                _kernelId);
            
            return new GpuBatchResult<TOut>(
                Array.Empty<TOut>(),
                TimeSpan.Zero,
                string.Empty,
                _kernelId,
                Error: ex.Message,
                Metrics: null);
        }
        finally
        {
            _concurrencyLimit.Release();
        }
    }
    
    public async Task<GpuBatchResult<TOut>> ExecuteWithCallbackAsync(
        IReadOnlyList<TIn> batch,
        IGpuResultObserver<TOut> observer,
        GpuExecutionHints? hints = null)
    {
        try
        {
            var result = await ExecuteAsync(batch, hints);
            
            if (result.Success)
            {
                // Stream results to observer
                foreach (var item in result.Results)
                {
                    await observer.OnNextAsync(item);
                }
                await observer.OnCompletedAsync();
            }
            else
            {
                await observer.OnErrorAsync(
                    new Exception(result.Error));
            }
            
            return result;
        }
        catch (Exception ex)
        {
            await observer.OnErrorAsync(ex);
            throw;
        }
    }
    
    public Task<GpuBatchResult<TOut>> ProcessBatchAsync(
        IReadOnlyList<TIn> batch,
        GpuExecutionHints? hints = null)
    {
        // Alias for ExecuteAsync method for backward compatibility
        return ExecuteAsync(batch, hints);
    }
}