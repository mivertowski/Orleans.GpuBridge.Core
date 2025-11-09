using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using OrleansCompiledKernel = Orleans.GpuBridge.Abstractions.Models.CompiledKernel;

namespace Orleans.GpuBridge.Backends.DotCompute.Execution;

/// <summary>
/// Profiling functionality for DotComputeKernelExecutor
/// </summary>
internal sealed partial class DotComputeKernelExecutor
{
    /// <summary>
    /// Profiles a kernel by executing it multiple times and collecting performance metrics
    /// </summary>
    public async Task<KernelProfile> ProfileAsync(
        OrleansCompiledKernel kernel,
        KernelExecutionParameters parameters,
        int iterations = 100,
        CancellationToken cancellationToken = default)
    {
        if (kernel == null)
            throw new ArgumentNullException(nameof(kernel));

        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        if (iterations <= 0)
            throw new ArgumentOutOfRangeException(nameof(iterations), "Iterations must be greater than zero");

        try
        {
            _logger.LogInformation("Profiling DotCompute kernel: {KernelName} with {Iterations} iterations", kernel.Name, iterations);

            var executionTimes = new List<TimeSpan>();
            var totalStopwatch = Stopwatch.StartNew();

            // Warm-up execution
            await ExecuteAsync(kernel, parameters, cancellationToken).ConfigureAwait(false);

            // Profile iterations
            for (int i = 0; i < iterations; i++)
            {
                var result = await ExecuteAsync(kernel, parameters, cancellationToken).ConfigureAwait(false);
                if (result.Success && result.Timing != null)
                {
                    executionTimes.Add(result.Timing.KernelTime);
                }

                cancellationToken.ThrowIfCancellationRequested();
            }

            totalStopwatch.Stop();

            // Calculate statistics
            var avgTime = TimeSpan.FromMilliseconds(executionTimes.Average(t => t.TotalMilliseconds));
            var minTime = executionTimes.Min();
            var maxTime = executionTimes.Max();
            var variance = executionTimes.Average(t => Math.Pow(t.TotalMilliseconds - avgTime.TotalMilliseconds, 2));
            var stdDev = Math.Sqrt(variance);

            _logger.LogInformation(
                "DotCompute kernel profiling completed: {KernelName} - Avg: {AvgTime}ms, Min: {MinTime}ms, Max: {MaxTime}ms",
                kernel.Name, avgTime.TotalMilliseconds, minTime.TotalMilliseconds, maxTime.TotalMilliseconds);

            return new KernelProfile(
                AverageExecutionTime: avgTime,
                MinExecutionTime: minTime,
                MaxExecutionTime: maxTime,
                StandardDeviation: stdDev,
                MemoryBandwidthBytesPerSecond: 0, // Would need to calculate based on data transfers
                ComputeThroughputGFlops: 0, // Would need kernel-specific calculation
                OptimalBlockSize: parameters.LocalWorkSize?.FirstOrDefault() ?? 256,
                ExtendedMetrics: new Dictionary<string, object>
                {
                    ["total_iterations"] = iterations,
                    ["total_profiling_time"] = totalStopwatch.Elapsed,
                    ["successful_iterations"] = executionTimes.Count
                });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to profile DotCompute kernel: {KernelName}", kernel.Name);
            throw;
        }
    }
}
