using System;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;

namespace Orleans.GpuBridge.Grains.Batch;

public sealed partial class GpuBatchGrainEnhanced<TIn, TOut>
{
    #region Public API

    public async Task<GpuBatchResult<TOut>> ExecuteAsync(
        IReadOnlyList<TIn> batch,
        GpuExecutionHints? hints = null)
    {
        if (batch == null || batch.Count == 0)
        {
            return new GpuBatchResult<TOut>(
                Array.Empty<TOut>(),
                TimeSpan.Zero,
                string.Empty,
                _kernelId,
                Error: "Empty batch provided");
        }

        await _concurrencyLimit.WaitAsync().ConfigureAwait(false);
        try
        {
            var stopwatch = Stopwatch.StartNew();

            _logger.LogDebug(
                "Executing batch of {Count} items on kernel {KernelId}",
                batch.Count, _kernelId);

            // GPU execution path
            if (_kernelExecutor != null && _compiledKernel != null && _memoryAllocator != null)
            {
                return await ExecuteOnGpuAsync(batch, hints, stopwatch).ConfigureAwait(false);
            }
            // CPU fallback path
            else
            {
                _logger.LogInformation(
                    "Executing batch on CPU (GPU unavailable) for kernel {KernelId}",
                    _kernelId);
                return await ExecuteOnCpuAsync(batch, stopwatch).ConfigureAwait(false);
            }
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
                Error: $"Batch execution failed: {ex.Message}");
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
            var result = await ExecuteAsync(batch, hints).ConfigureAwait(false);

            if (result.Success)
            {
                // Stream results to observer
                foreach (var item in result.Results)
                {
                    await observer.OnNextAsync(item).ConfigureAwait(false);
                }
                await observer.OnCompletedAsync().ConfigureAwait(false);
            }
            else
            {
                await observer.OnErrorAsync(
                    new Exception(result.Error)).ConfigureAwait(false);
            }

            return result;
        }
        catch (Exception ex)
        {
            await observer.OnErrorAsync(ex).ConfigureAwait(false);
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

    #endregion

    #region GPU Execution

    private async Task<GpuBatchResult<TOut>> ExecuteOnGpuAsync(
        IReadOnlyList<TIn> batch,
        GpuExecutionHints? hints,
        Stopwatch stopwatch)
    {
        // Calculate optimal batch size based on GPU memory
        var optimalBatchSize = CalculateOptimalBatchSize(batch, hints);

        // Split into sub-batches if necessary
        var subBatches = SplitIntoBatches(batch, optimalBatchSize);

        var allResults = new List<TOut>();
        var totalKernelTime = TimeSpan.Zero;
        var totalMemoryTransferTime = TimeSpan.Zero;
        var successfulBatches = 0;

        foreach (var subBatch in subBatches)
        {
            try
            {
                // Allocate GPU memory
                var (inputMemory, outputMemory, allocTime) = await AllocateGpuMemoryAsync(subBatch).ConfigureAwait(false);

                // Prepare execution parameters
                var execParams = PrepareExecutionParameters(inputMemory, outputMemory, subBatch.Count);

                // Execute kernel on GPU
                var kernelStopwatch = Stopwatch.StartNew();
                var kernelResult = await _kernelExecutor!.ExecuteAsync(
                    _compiledKernel!,
                    execParams,
                    CancellationToken.None).ConfigureAwait(false);
                kernelStopwatch.Stop();

                if (kernelResult.Success)
                {
                    // Read results from GPU memory
                    var (results, readTime) = await ReadResultsFromGpuAsync(outputMemory, subBatch.Count).ConfigureAwait(false);
                    allResults.AddRange(results);

                    totalKernelTime += kernelResult.Timing.KernelTime;
                    totalMemoryTransferTime += allocTime + readTime;
                    successfulBatches++;

                    _logger.LogDebug(
                        "Executed sub-batch: {Items} items in {KernelTime}ms (transfer: {TransferTime}ms)",
                        subBatch.Count,
                        kernelResult.Timing.KernelTime.TotalMilliseconds,
                        (allocTime + readTime).TotalMilliseconds);
                }
                else
                {
                    _logger.LogError(
                        "Sub-batch execution failed: {ErrorMessage}",
                        kernelResult.ErrorMessage);

                    // Free memory even on failure
                    await FreeGpuMemoryAsync(inputMemory, outputMemory).ConfigureAwait(false);

                    throw new InvalidOperationException(
                        $"Kernel execution failed: {kernelResult.ErrorMessage}");
                }

                // Free GPU memory
                await FreeGpuMemoryAsync(inputMemory, outputMemory).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Sub-batch execution failed, aborting batch");
                throw;
            }
        }

        stopwatch.Stop();

        // Update lifetime statistics
        _totalItemsProcessed += batch.Count;
        _totalBatchesProcessed++;
        _totalGpuExecutionTime += totalKernelTime;

        // Create comprehensive metrics
        var metrics = new GpuBatchMetrics(
            TotalItems: batch.Count,
            SubBatchCount: subBatches.Count,
            SuccessfulSubBatches: successfulBatches,
            TotalExecutionTime: stopwatch.Elapsed,
            KernelExecutionTime: totalKernelTime,
            MemoryTransferTime: totalMemoryTransferTime,
            Throughput: batch.Count / stopwatch.Elapsed.TotalSeconds,
            MemoryAllocated: CalculateTotalMemorySize(batch),
            DeviceType: _primaryDevice!.Type.ToString(),
            DeviceName: _primaryDevice.Name);

        _logger.LogInformation(
            "Executed batch: {Items} items in {Time}ms ({Throughput:F2} items/sec, {SubBatches} sub-batches)",
            batch.Count,
            stopwatch.ElapsedMilliseconds,
            metrics.Throughput,
            subBatches.Count);

        return new GpuBatchResult<TOut>(
            allResults,
            stopwatch.Elapsed,
            Guid.NewGuid().ToString(),
            _kernelId,
            Error: null,
            Metrics: metrics);
    }

    #endregion

    #region CPU Fallback

    private async Task<GpuBatchResult<TOut>> ExecuteOnCpuAsync(
        IReadOnlyList<TIn> batch,
        Stopwatch stopwatch)
    {
        // CPU passthrough - for testing purposes only
        // In production, you would implement actual CPU kernels here
        await Task.Yield();

        var results = new List<TOut>(batch.Count);

        // Simple passthrough if types are compatible
        if (typeof(TIn) == typeof(TOut))
        {
            foreach (var item in batch)
            {
                if (item is TOut result)
                {
                    results.Add(result);
                }
            }
        }
        else
        {
            // Default value for incompatible types
            for (int i = 0; i < batch.Count; i++)
            {
                results.Add(default(TOut));
            }
        }

        stopwatch.Stop();

        _logger.LogWarning(
            "Executed batch on CPU: {Items} items in {Time}ms (GPU unavailable)",
            batch.Count,
            stopwatch.ElapsedMilliseconds);

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
            Guid.NewGuid().ToString(),
            _kernelId,
            Error: null,
            Metrics: metrics);
    }

    #endregion
}
