using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results.Statistics;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Backends.ILGPU.DeviceManagement;
using Orleans.GpuBridge.Backends.ILGPU.Kernels;
using Orleans.GpuBridge.Backends.ILGPU.Memory;

namespace Orleans.GpuBridge.Backends.ILGPU.Execution;

/// <summary>
/// ILGPU kernel executor implementation
/// </summary>
internal sealed class ILGPUKernelExecutor : IKernelExecutor
{
    private readonly ILogger<ILGPUKernelExecutor> _logger;
    private readonly ILGPUDeviceManager _deviceManager;
    private readonly ILGPUMemoryAllocator _memoryAllocator;
    private readonly ILGPUKernelCompiler? _kernelCompiler;
    private readonly ConcurrentDictionary<string, ILGPUKernelExecution> _activeExecutions;
    private readonly SemaphoreSlim _executionSemaphore;
    private long _totalKernelsExecuted;
    private long _totalBatchesExecuted;
    private long _totalGraphsExecuted;
    private long _totalExecutionTimeMs;
    private long _totalBytesTransferred;
    private long _totalErrors;
    private readonly ConcurrentDictionary<string, long> _kernelExecutionCounts;
    private bool _disposed;

    public ILGPUKernelExecutor(
        ILogger<ILGPUKernelExecutor> logger,
        ILGPUDeviceManager deviceManager,
        ILGPUMemoryAllocator memoryAllocator)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _deviceManager = deviceManager ?? throw new ArgumentNullException(nameof(deviceManager));
        _memoryAllocator = memoryAllocator ?? throw new ArgumentNullException(nameof(memoryAllocator));
        _activeExecutions = new ConcurrentDictionary<string, ILGPUKernelExecution>();
        _executionSemaphore = new SemaphoreSlim(50, 50); // Limit concurrent executions
        _kernelExecutionCounts = new ConcurrentDictionary<string, long>();
    }

    public async Task<KernelExecutionResult> ExecuteAsync(
        CompiledKernel kernel,
        KernelExecutionParameters parameters,
        CancellationToken cancellationToken = default)
    {
        if (kernel == null)
            throw new ArgumentNullException(nameof(kernel));

        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        var stopwatch = Stopwatch.StartNew();

        try
        {
            _logger.LogDebug("Executing ILGPU kernel: {KernelName}", kernel.Name);

            await _executionSemaphore.WaitAsync(cancellationToken);
            
            try
            {
                // Select device for execution
                var device = SelectExecutionDevice(parameters);
                if (device is not ILGPUComputeDevice ilgpuDevice)
                {
                    throw new InvalidOperationException("Selected device is not an ILGPU device");
                }

                var accelerator = ilgpuDevice.Accelerator;

                // Get the compiled ILGPU kernel
                var ilgpuKernel = _kernelCompiler?.GetCachedILGPUKernel(kernel.KernelId);
                if (ilgpuKernel == null)
                {
                    throw new InvalidOperationException($"ILGPU kernel not found: {kernel.KernelId}");
                }

                // Prepare kernel arguments
                var kernelArgs = await PrepareKernelArgumentsAsync(parameters, ilgpuDevice, cancellationToken);

                // Calculate launch configuration
                var config = CalculateKernelConfig(parameters, ilgpuDevice);

                // Execute the kernel
                var stream = parameters.PreferredQueue is ILGPUCommandQueue ilgpuQueue 
                    ? GetStreamFromQueue(ilgpuQueue) 
                    : accelerator.DefaultStream;

                var executionStart = Stopwatch.StartNew();
                
                try
                {
                    // Convert config to Index1D and args to ArrayView
                    var index = (Index1D)(config.GridDim.X * config.GroupDim.X);
                    var arrayView = kernelArgs.Length > 0 && kernelArgs[0] is ArrayView<int> av ? av : new ArrayView<int>();
                    ilgpuKernel(index, arrayView);
                    stream.Synchronize();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Kernel execution failed: {KernelName}", kernel.Name);
                    throw new InvalidOperationException($"ILGPU kernel execution failed: {ex.Message}", ex);
                }

                executionStart.Stop();

                // Update statistics
                UpdateExecutionStatistics(kernel.Name, executionStart.Elapsed, true);

                var timing = new KernelTiming(
                    QueueTime: TimeSpan.FromMilliseconds(stopwatch.ElapsedMilliseconds - executionStart.ElapsedMilliseconds),
                    KernelTime: executionStart.Elapsed,
                    TotalTime: stopwatch.Elapsed);

                _logger.LogDebug(
                    "ILGPU kernel execution completed: {KernelName} in {ExecutionTime}ms",
                    kernel.Name, executionStart.ElapsedMilliseconds);

                return new KernelExecutionResult(
                    Success: true,
                    Timing: timing,
                    Metadata: new Dictionary<string, object>
                    {
                        ["device_name"] = ilgpuDevice.Name,
                        ["accelerator_type"] = ilgpuDevice.Accelerator.AcceleratorType.ToString(),
                        ["global_work_size"] = parameters.GlobalWorkSize,
                        ["local_work_size"] = parameters.LocalWorkSize ?? Array.Empty<int>()
                    });
            }
            finally
            {
                _executionSemaphore.Release();
            }
        }
        catch (Exception ex)
        {
            UpdateExecutionStatistics(kernel.Name, stopwatch.Elapsed, false);
            
            _logger.LogError(ex, "Failed to execute ILGPU kernel: {KernelName}", kernel.Name);
            
            return new KernelExecutionResult(
                Success: false,
                ErrorMessage: ex.Message,
                Timing: new KernelTiming(
                    QueueTime: TimeSpan.Zero,
                    KernelTime: TimeSpan.Zero,
                    TotalTime: stopwatch.Elapsed));
        }
    }

    public Task<IKernelExecution> ExecuteAsyncNonBlocking(
        CompiledKernel kernel,
        KernelExecutionParameters parameters,
        CancellationToken cancellationToken = default)
    {
        if (kernel == null)
            throw new ArgumentNullException(nameof(kernel));

        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        var executionId = Guid.NewGuid().ToString("N");
        var execution = new ILGPUKernelExecution(
            executionId,
            kernel,
            this,
            new KernelExecutionLogger(_logger));

        _activeExecutions[executionId] = execution;

        // Start execution in background
        _ = Task.Run(async () =>
        {
            try
            {
                var result = await ExecuteAsync(kernel, parameters, cancellationToken);
                execution.SetResult(result);
            }
            catch (Exception ex)
            {
                execution.SetError(ex);
            }
            finally
            {
                _activeExecutions.TryRemove(executionId, out _);
            }
        }, cancellationToken);

        return Task.FromResult<IKernelExecution>(execution);
    }

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
            _logger.LogInformation("Executing ILGPU kernel batch: {BatchSize} kernels", batch.Count);

            if (options.ExecuteInParallel && options.MaxParallelism > 1)
            {
                // Parallel execution
                var semaphore = new SemaphoreSlim(options.MaxParallelism, options.MaxParallelism);
                var tasks = batch.Select(async item =>
                {
                    await semaphore.WaitAsync(cancellationToken);
                    try
                    {
                        return await ExecuteAsync(item.Kernel, item.Parameters, cancellationToken);
                    }
                    finally
                    {
                        semaphore.Release();
                    }
                });

                var batchResults = await Task.WhenAll(tasks);
                results.AddRange(batchResults);
            }
            else
            {
                // Sequential execution
                foreach (var item in batch)
                {
                    try
                    {
                        var result = await ExecuteAsync(item.Kernel, item.Parameters, cancellationToken);
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

            // Update batch statistics
            Interlocked.Increment(ref _totalBatchesExecuted);
            Interlocked.Add(ref _totalExecutionTimeMs, stopwatch.ElapsedMilliseconds);
            if (failureCount > 0)
            {
                Interlocked.Add(ref _totalErrors, failureCount);
            }
            
            _logger.LogInformation(
                "ILGPU kernel batch execution completed: {SuccessCount} succeeded, {FailureCount} failed in {TotalTime}ms",
                successCount, failureCount, stopwatch.ElapsedMilliseconds);

            return new BatchExecutionResult(
                SuccessCount: successCount,
                FailureCount: failureCount,
                Results: results,
                TotalExecutionTime: stopwatch.Elapsed);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "ILGPU kernel batch execution failed");
            throw;
        }
    }

    public IKernelGraph CreateGraph(string graphName)
    {
        if (string.IsNullOrEmpty(graphName))
            throw new ArgumentException("Graph name cannot be null or empty", nameof(graphName));

        return new ILGPUKernelGraph(graphName, this, new KernelGraphLogger(_logger));
    }

    public async Task<KernelProfile> ProfileAsync(
        CompiledKernel kernel,
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
            _logger.LogInformation("Profiling ILGPU kernel: {KernelName} with {Iterations} iterations", kernel.Name, iterations);

            var executionTimes = new List<TimeSpan>();
            var totalStopwatch = Stopwatch.StartNew();

            // Warm-up execution
            await ExecuteAsync(kernel, parameters, cancellationToken);

            // Profile iterations
            for (int i = 0; i < iterations; i++)
            {
                var result = await ExecuteAsync(kernel, parameters, cancellationToken);
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
                "ILGPU kernel profiling completed: {KernelName} - Avg: {AvgTime}ms, Min: {MinTime}ms, Max: {MaxTime}ms",
                kernel.Name, avgTime.TotalMilliseconds, minTime.TotalMilliseconds, maxTime.TotalMilliseconds);

            return new KernelProfile(
                AverageExecutionTime: avgTime,
                MinExecutionTime: minTime,
                MaxExecutionTime: maxTime,
                StandardDeviation: stdDev,
                MemoryBandwidthBytesPerSecond: 0, // Would need to calculate based on actual data transfers
                ComputeThroughputGFlops: 0, // Would need kernel-specific calculation based on operations
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
            _logger.LogError(ex, "Failed to profile ILGPU kernel: {KernelName}", kernel.Name);
            throw;
        }
    }

    public ExecutionStatistics GetStatistics()
    {
        var totalExecutionTime = TimeSpan.FromMilliseconds(_totalExecutionTimeMs);
        var averageKernelTime = _totalKernelsExecuted > 0 
            ? TimeSpan.FromMilliseconds(_totalExecutionTimeMs / (double)_totalKernelsExecuted)
            : TimeSpan.Zero;
            
        return new ExecutionStatistics(
            TotalKernelsExecuted: _totalKernelsExecuted,
            TotalBatchesExecuted: _totalBatchesExecuted,
            TotalGraphsExecuted: _totalGraphsExecuted,
            TotalExecutionTime: totalExecutionTime,
            AverageKernelTime: averageKernelTime,
            TotalBytesTransferred: _totalBytesTransferred,
            TotalErrors: _totalErrors,
            KernelExecutionCounts: new Dictionary<string, long>(_kernelExecutionCounts));
    }

    public void ResetStatistics()
    {
        _logger.LogInformation("Resetting ILGPU kernel execution statistics");
        
        // Reset all statistics counters
        Interlocked.Exchange(ref _totalKernelsExecuted, 0);
        Interlocked.Exchange(ref _totalBatchesExecuted, 0);
        Interlocked.Exchange(ref _totalGraphsExecuted, 0);
        Interlocked.Exchange(ref _totalExecutionTimeMs, 0);
        Interlocked.Exchange(ref _totalBytesTransferred, 0);
        Interlocked.Exchange(ref _totalErrors, 0);
        _kernelExecutionCounts.Clear();
        
        _logger.LogInformation("ILGPU kernel execution statistics reset completed");
    }

    private IComputeDevice SelectExecutionDevice(KernelExecutionParameters parameters)
    {
        // Use preferred device from parameters if available
        if (parameters.PreferredQueue?.Context?.Device != null)
        {
            return parameters.PreferredQueue.Context.Device;
        }

        // Select based on device requirements (simplified)
        var devices = _deviceManager.GetDevices();
        var gpuDevices = devices.Where(d => d.Type != DeviceType.CPU).ToList();

        return gpuDevices.FirstOrDefault() ?? devices.FirstOrDefault() ?? 
               throw new InvalidOperationException("No suitable device available for kernel execution");
    }

    private Task<object[]> PrepareKernelArgumentsAsync(
        KernelExecutionParameters parameters,
        ILGPUComputeDevice device,
        CancellationToken cancellationToken)
    {
        var args = new List<object>();

        // Add memory arguments
        foreach (var memArg in parameters.MemoryArguments)
        {
            if (memArg.Value is ILGPUDeviceMemoryWrapper deviceMemory)
            {
                // ILGPU kernels typically take ArrayView parameters
                args.Add(deviceMemory);
            }
            else
            {
                _logger.LogWarning("Unsupported memory argument type: {Type}", memArg.Value.GetType());
                args.Add(memArg.Value);
            }
        }

        // Add scalar arguments
        foreach (var scalarArg in parameters.ScalarArguments)
        {
            args.Add(scalarArg.Value);
        }

        return Task.FromResult(args.ToArray());
    }

    private KernelConfig CalculateKernelConfig(
        KernelExecutionParameters parameters,
        ILGPUComputeDevice device)
    {
        var globalSize = parameters.GlobalWorkSize.Length > 0 ? parameters.GlobalWorkSize[0] : 1;
        var groupSize = parameters.LocalWorkSize?.Length > 0 ? parameters.LocalWorkSize[0] : 
                       Math.Min(256, device.MaxThreadsPerBlock);

        return new KernelConfig(
            gridDim: (globalSize + groupSize - 1) / groupSize,
            groupDim: groupSize);
    }

    private AcceleratorStream GetStreamFromQueue(ILGPUCommandQueue queue)
    {
        // In a complete implementation, we would extract the stream from the command queue
        // Extract the stream from the ILGPU command queue
        if (queue.Context.Device is ILGPUComputeDevice device)
        {
            // In ILGPU, we can access the accelerator's streams
            // For now, use the default stream but in production this could be optimized
            // by maintaining a pool of streams or extracting from queue context
            return device.Accelerator.DefaultStream;
        }
        
        throw new InvalidOperationException($"Cannot get stream from queue - device type is {queue.Context.Device?.GetType().Name}");
    }

    private void UpdateExecutionStatistics(string kernelName, TimeSpan executionTime, bool success)
    {
        // Update execution statistics atomically
        Interlocked.Increment(ref _totalKernelsExecuted);
        Interlocked.Add(ref _totalExecutionTimeMs, (long)executionTime.TotalMilliseconds);
        
        if (!success)
        {
            Interlocked.Increment(ref _totalErrors);
        }
        
        // Update per-kernel execution counts
        _kernelExecutionCounts.AddOrUpdate(kernelName, 1, (key, oldValue) => oldValue + 1);
        
        _logger.LogTrace(
            "Updated execution statistics for kernel {KernelName}: {ExecutionTime}ms, Success: {Success}, " +
            "Total Kernels: {TotalKernels}, Total Errors: {TotalErrors}",
            kernelName, executionTime.TotalMilliseconds, success, _totalKernelsExecuted, _totalErrors);
    }

    internal void RemoveActiveExecution(string executionId)
    {
        _activeExecutions.TryRemove(executionId, out _);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing ILGPU kernel executor");

        try
        {
            // Cancel all active executions
            foreach (var execution in _activeExecutions.Values)
            {
                try
                {
                    execution.Cancel();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error cancelling active execution: {ExecutionId}", execution.ExecutionId);
                }
            }

            _activeExecutions.Clear();
            _executionSemaphore?.Dispose();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing ILGPU kernel executor");
        }

        _disposed = true;
    }
}

/// <summary>
/// Simple logger wrapper for kernel execution
/// </summary>
internal class KernelExecutionLogger : ILogger<ILGPUKernelExecution>
{
    private readonly ILogger _baseLogger;

    public KernelExecutionLogger(ILogger baseLogger)
    {
        _baseLogger = baseLogger;
    }

    public IDisposable? BeginScope<TState>(TState state) where TState : notnull
    {
        return _baseLogger.BeginScope(state);
    }

    public bool IsEnabled(LogLevel logLevel)
    {
        return _baseLogger.IsEnabled(logLevel);
    }

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        _baseLogger.Log(logLevel, eventId, state, exception, formatter);
    }
}

/// <summary>
/// Simple logger wrapper for kernel graph
/// </summary>
internal class KernelGraphLogger : ILogger<ILGPUKernelGraph>
{
    private readonly ILogger _baseLogger;

    public KernelGraphLogger(ILogger baseLogger)
    {
        _baseLogger = baseLogger;
    }

    public IDisposable? BeginScope<TState>(TState state) where TState : notnull
    {
        return _baseLogger.BeginScope(state);
    }

    public bool IsEnabled(LogLevel logLevel)
    {
        return _baseLogger.IsEnabled(logLevel);
    }

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        _baseLogger.Log(logLevel, eventId, state, exception, formatter);
    }
}