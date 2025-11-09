using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;

namespace Orleans.GpuBridge.Backends.DotCompute.Execution;

/// <summary>
/// Batch execution functionality for DotComputeKernelExecutor
/// </summary>
internal sealed partial class DotComputeKernelExecutor
{
    /// <summary>
    /// Executes a batch of kernels with configurable parallelism
    /// </summary>
    public async Task<BatchExecutionResult> ExecuteBatchAsync(
        IReadOnlyList<KernelBatchItem> batch,
        BatchExecutionOptions options,
        CancellationToken cancellationToken = default)
    {
        if (batch == null)
            throw new ArgumentNullException(nameof(batch));

        if (options == null)
            options = new BatchExecutionOptions();

        var stopwatch = Stopwatch.StartNew();
        var results = new List<KernelExecutionResult>();
        var successCount = 0;
        var failureCount = 0;

        try
        {
            _logger.LogInformation("Executing DotCompute kernel batch: {BatchSize} kernels", batch.Count);

            if (options.ExecuteInParallel && options.MaxParallelism > 1)
            {
                // Parallel execution
                var semaphore = new SemaphoreSlim(options.MaxParallelism, options.MaxParallelism);
                var tasks = batch.Select(async item =>
                {
                    await semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
                    try
                    {
                        return await ExecuteAsync(item.Kernel, item.Parameters, cancellationToken).ConfigureAwait(false);
                    }
                    finally
                    {
                        semaphore.Release();
                    }
                });

                var batchResults = await Task.WhenAll(tasks).ConfigureAwait(false);
                results.AddRange(batchResults);
            }
            else
            {
                // Sequential execution
                foreach (var item in batch)
                {
                    try
                    {
                        var result = await ExecuteAsync(item.Kernel, item.Parameters, cancellationToken).ConfigureAwait(false);
                        results.Add(result);

                        if (!result.Success && options.StopOnFirstError)
                        {
                            _logger.LogWarning("Stopping batch execution due to error in kernel: {KernelName}", item.Kernel.Name);
                            break;
                        }
                    }
                    catch (Exception ex)
                    {
                        var errorResult = new KernelExecutionResult(false, ex.Message);
                        results.Add(errorResult);

                        if (options.StopOnFirstError)
                        {
                            _logger.LogWarning(ex, "Stopping batch execution due to exception in kernel: {KernelName}", item.Kernel.Name);
                            break;
                        }
                    }
                }
            }

            // Count successes and failures
            foreach (var result in results)
            {
                if (result.Success)
                    successCount++;
                else
                    failureCount++;
            }

            _logger.LogInformation(
                "DotCompute kernel batch execution completed: {SuccessCount} succeeded, {FailureCount} failed in {TotalTime}ms",
                successCount, failureCount, stopwatch.ElapsedMilliseconds);

            return new BatchExecutionResult(
                SuccessCount: successCount,
                FailureCount: failureCount,
                Results: results,
                TotalExecutionTime: stopwatch.Elapsed);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "DotCompute kernel batch execution failed");
            throw;
        }
    }
}
