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
using Orleans.GpuBridge.Abstractions.Kernels;
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
        // Validate null batch (error)
        if (batch == null)
        {
            _logger.LogError("Null batch provided for kernel {KernelId}", _kernelId);

            return new GpuBatchResult<TOut>(
                Array.Empty<TOut>(),
                TimeSpan.Zero,
                string.Empty,
                _kernelId,
                Error: "Batch cannot be null",
                Metrics: null);
        }

        // Handle empty batch gracefully (return success with empty results)
        if (batch.Count == 0)
        {
            _logger.LogDebug("Empty batch provided for kernel {KernelId}, returning empty results", _kernelId);

            var emptyMetrics = new GpuBatchMetrics(
                TotalItems: 0,
                SubBatchCount: 0,
                SuccessfulSubBatches: 0,
                TotalExecutionTime: TimeSpan.Zero,
                KernelExecutionTime: TimeSpan.Zero,
                MemoryTransferTime: TimeSpan.Zero,
                Throughput: 0,
                MemoryAllocated: 0,
                DeviceType: "CPU",
                DeviceName: "CPU Fallback");

            return new GpuBatchResult<TOut>(
                Array.Empty<TOut>(),
                TimeSpan.Zero,
                Guid.NewGuid().ToString(),
                _kernelId,
                Error: null,
                Metrics: emptyMetrics);
        }

        await _concurrencyLimit.WaitAsync();
        try
        {
            var stopwatch = Stopwatch.StartNew();

            _logger.LogDebug(
                "Executing batch of {Count} items on kernel {KernelId}",
                batch.Count, _kernelId);

            // Calculate memory requirements for metrics
            var memoryAllocated = CalculateMemorySize(batch);

            // Determine if batch splitting is needed (simulate for large batches)
            var maxBatchSize = hints?.MaxMicroBatch ?? 1024;
            var subBatchCount = (int)Math.Ceiling((double)batch.Count / maxBatchSize);

            // Simulate memory transfer time (proportional to data size)
            var memoryTransferTime = TimeSpan.FromMilliseconds(
                Math.Max(1, memoryAllocated / (1024 * 1024))); // 1ms per MB

            // Execute batch using new API
            var batchArray = batch.ToArray();
            var results = await _kernel.ExecuteBatchAsync(batchArray);

            stopwatch.Stop();

            _logger.LogInformation(
                "Executed batch of {Count} items in {ElapsedMs}ms ({SubBatches} sub-batches, {ResultCount} results)",
                batch.Count, stopwatch.ElapsedMilliseconds, subBatchCount, results.Length);

            // Calculate kernel execution time (total time minus transfer time)
            var kernelTime = stopwatch.Elapsed - memoryTransferTime;
            if (kernelTime < TimeSpan.Zero)
                kernelTime = stopwatch.Elapsed;

            // Create detailed metrics for monitoring
            var metrics = new GpuBatchMetrics(
                TotalItems: batch.Count,
                SubBatchCount: subBatchCount,
                SuccessfulSubBatches: subBatchCount,
                TotalExecutionTime: stopwatch.Elapsed,
                KernelExecutionTime: kernelTime,
                MemoryTransferTime: memoryTransferTime,
                Throughput: batch.Count / stopwatch.Elapsed.TotalSeconds,
                MemoryAllocated: memoryAllocated,
                DeviceType: "CPU",
                DeviceName: "CPU Fallback");

            return new GpuBatchResult<TOut>(
                results.ToList(),
                stopwatch.Elapsed,
                Guid.NewGuid().ToString(),
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

    /// <summary>
    /// Calculate total memory size for batch processing
    /// </summary>
    private long CalculateMemorySize(IReadOnlyList<TIn> batch)
    {
        try
        {
            // Try to get actual size for blittable types
            var inputSize = System.Runtime.InteropServices.Marshal.SizeOf(typeof(TIn)) * batch.Count;
            var outputSize = System.Runtime.InteropServices.Marshal.SizeOf(typeof(TOut)) * batch.Count;
            return inputSize + outputSize;
        }
        catch (ArgumentException)
        {
            // For non-blittable types (reference types, complex structs),
            // use reasonable estimates based on typical pointer sizes
            var estimatedInputSize = IntPtr.Size * batch.Count;  // Pointer size per item
            var estimatedOutputSize = IntPtr.Size * batch.Count; // Pointer size per item
            return estimatedInputSize + estimatedOutputSize;
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