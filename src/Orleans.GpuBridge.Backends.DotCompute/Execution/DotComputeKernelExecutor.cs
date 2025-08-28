using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results.Statistics;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;

namespace Orleans.GpuBridge.Backends.DotCompute.Execution;

/// <summary>
/// DotCompute kernel executor implementation
/// </summary>
internal sealed class DotComputeKernelExecutor : IKernelExecutor
{
    private readonly ILogger<DotComputeKernelExecutor> _logger;
    private readonly IDeviceManager _deviceManager;
    private readonly IMemoryAllocator _memoryAllocator;
    private readonly IKernelCompiler _kernelCompiler;
    private readonly ConcurrentDictionary<string, DotComputeKernelExecution> _activeExecutions;
    private readonly ExecutionStatistics _statistics;
    private readonly SemaphoreSlim _executionSemaphore;
    private bool _disposed;

    public DotComputeKernelExecutor(
        ILogger<DotComputeKernelExecutor> logger,
        IDeviceManager deviceManager,
        IMemoryAllocator memoryAllocator,
        IKernelCompiler kernelCompiler)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _deviceManager = deviceManager ?? throw new ArgumentNullException(nameof(deviceManager));
        _memoryAllocator = memoryAllocator ?? throw new ArgumentNullException(nameof(memoryAllocator));
        _kernelCompiler = kernelCompiler ?? throw new ArgumentNullException(nameof(kernelCompiler));
        _activeExecutions = new ConcurrentDictionary<string, DotComputeKernelExecution>();
        _statistics = new ExecutionStatistics(0, 0, 0, TimeSpan.Zero, TimeSpan.Zero, 0, 0, new Dictionary<string, long>());
        _executionSemaphore = new SemaphoreSlim(50, 50); // Limit concurrent executions
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
            _logger.LogDebug("Executing DotCompute kernel: {KernelName}", kernel.Name);

            await _executionSemaphore.WaitAsync(cancellationToken);
            
            try
            {
                // Select device for execution
                var device = SelectExecutionDevice(parameters);

                // Get the compiled DotCompute kernel
                var dotComputeKernel = _kernelCompiler?.GetCachedDotComputeKernel(kernel.KernelId);
                if (dotComputeKernel == null)
                {
                    throw new InvalidOperationException($"DotCompute kernel not found: {kernel.KernelId}");
                }

                // Prepare kernel arguments
                var kernelArgs = await PrepareKernelArgumentsAsync(parameters, device, cancellationToken);

                // Calculate work dimensions
                var workDimensions = CalculateWorkDimensions(parameters);

                // Execute the kernel
                var executionStart = Stopwatch.StartNew();
                
                try
                {
                    await ExecuteDotComputeKernelAsync(
                        dotComputeKernel, 
                        kernelArgs, 
                        workDimensions, 
                        device,
                        cancellationToken);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "DotCompute kernel execution failed: {KernelName}", kernel.Name);
                    throw new InvalidOperationException($"DotCompute kernel execution failed: {ex.Message}", ex);
                }

                executionStart.Stop();

                // Update statistics
                UpdateExecutionStatistics(kernel.Name, executionStart.Elapsed, true);

                var timing = new KernelTiming(
                    QueueTime: TimeSpan.FromMilliseconds(stopwatch.ElapsedMilliseconds - executionStart.ElapsedMilliseconds),
                    KernelTime: executionStart.Elapsed,
                    TotalTime: stopwatch.Elapsed);

                _logger.LogDebug(
                    "DotCompute kernel execution completed: {KernelName} in {ExecutionTime}ms",
                    kernel.Name, executionStart.ElapsedMilliseconds);

                return new KernelExecutionResult(
                    Success: true,
                    Timing: timing,
                    Metadata: new Dictionary<string, object>
                    {
                        ["device_name"] = device.Name,
                        ["device_type"] = device.Type.ToString(),
                        ["global_work_size"] = parameters.GlobalWorkSize,
                        ["local_work_size"] = parameters.LocalWorkSize
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
            
            _logger.LogError(ex, "Failed to execute DotCompute kernel: {KernelName}", kernel.Name);
            
            return new KernelExecutionResult(
                Success: false,
                ErrorMessage: ex.Message,
                Timing: new KernelTiming(
                    QueueTime: TimeSpan.Zero,
                    KernelTime: TimeSpan.Zero,
                    TotalTime: stopwatch.Elapsed));
        }
    }

    public async Task<IKernelExecution> ExecuteAsyncNonBlocking(
        CompiledKernel kernel,
        KernelExecutionParameters parameters,
        CancellationToken cancellationToken = default)
    {
        if (kernel == null)
            throw new ArgumentNullException(nameof(kernel));

        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        var executionId = Guid.NewGuid().ToString("N");
        var execution = new DotComputeKernelExecution(
            executionId,
            kernel,
            this,
            _logger.CreateLogger<DotComputeKernelExecution>());

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

        return execution;
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
            _logger.LogInformation("Executing DotCompute kernel batch: {BatchSize} kernels", batch.Count);

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

    public IKernelGraph CreateGraph(string graphName)
    {
        if (string.IsNullOrEmpty(graphName))
            throw new ArgumentException("Graph name cannot be null or empty", nameof(graphName));

        return new DotComputeKernelGraph(graphName, this, _logger.CreateLogger<DotComputeKernelGraph>());
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
            _logger.LogInformation("Profiling DotCompute kernel: {KernelName} with {Iterations} iterations", kernel.Name, iterations);

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

    public ExecutionStatistics GetStatistics()
    {
        return _statistics;
    }

    public void ResetStatistics()
    {
        _logger.LogInformation("Resetting DotCompute kernel execution statistics");
        
        // In a complete implementation, we would reset the statistics fields
        // For now, this is a placeholder
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
        var gpuDevices = devices.Where(d => d.Type != DeviceType.Cpu).ToList();

        return gpuDevices.FirstOrDefault() ?? devices.FirstOrDefault() ?? 
               throw new InvalidOperationException("No suitable device available for kernel execution");
    }

    private async Task<object[]> PrepareKernelArgumentsAsync(
        KernelExecutionParameters parameters,
        IComputeDevice device,
        CancellationToken cancellationToken)
    {
        var args = new List<object>();

        // Add memory arguments
        foreach (var memArg in parameters.MemoryArguments)
        {
            args.Add(memArg.Value);
        }

        // Add scalar arguments
        foreach (var scalarArg in parameters.ScalarArguments)
        {
            args.Add(scalarArg.Value);
        }

        return args.ToArray();
    }

    private WorkDimensions CalculateWorkDimensions(KernelExecutionParameters parameters)
    {
        var globalSize = parameters.GlobalWorkSize.Length > 0 ? parameters.GlobalWorkSize : new[] { 1 };
        var localSize = parameters.LocalWorkSize?.Length > 0 ? parameters.LocalWorkSize : null;

        return new WorkDimensions(globalSize, localSize);
    }

    private async Task ExecuteDotComputeKernelAsync(
        object kernel,
        object[] arguments,
        WorkDimensions workDimensions,
        IComputeDevice device,
        CancellationToken cancellationToken)
    {
        // In a real DotCompute implementation, this would execute the kernel
        // using DotCompute APIs with the specified work dimensions
        
        // Simulate kernel execution time
        var executionTime = Math.Max(1, workDimensions.GlobalSize.Aggregate(1, (a, b) => a * b) / 1000000);
        await Task.Delay(executionTime, cancellationToken);
    }

    private void UpdateExecutionStatistics(string kernelName, TimeSpan executionTime, bool success)
    {
        // In a complete implementation, we would update the statistics fields
        // This is a placeholder for the statistics update logic
        _logger.LogTrace(
            "Updated execution statistics for kernel {KernelName}: {ExecutionTime}ms, Success: {Success}",
            kernelName, executionTime.TotalMilliseconds, success);
    }

    internal void RemoveActiveExecution(string executionId)
    {
        _activeExecutions.TryRemove(executionId, out _);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing DotCompute kernel executor");

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
            _logger.LogError(ex, "Error disposing DotCompute kernel executor");
        }

        _disposed = true;
    }
}

/// <summary>
/// Work dimensions for DotCompute kernel execution
/// </summary>
internal record WorkDimensions(int[] GlobalSize, int[]? LocalSize);

/// <summary>
/// DotCompute kernel execution wrapper
/// </summary>
internal sealed class DotComputeKernelExecution : IKernelExecution
{
    private readonly CompiledKernel _kernel;
    private readonly DotComputeKernelExecutor _executor;
    private readonly ILogger _logger;
    private readonly TaskCompletionSource<KernelExecutionResult> _completionSource;
    private readonly CancellationTokenSource _cancellationSource;
    private volatile bool _isCompleted;
    private volatile bool _isCanceled;

    public string ExecutionId { get; }
    public bool IsCompleted => _isCompleted;
    public bool IsCanceled => _isCanceled;

    public DotComputeKernelExecution(
        string executionId,
        CompiledKernel kernel,
        DotComputeKernelExecutor executor,
        ILogger logger)
    {
        ExecutionId = executionId ?? throw new ArgumentNullException(nameof(executionId));
        _kernel = kernel ?? throw new ArgumentNullException(nameof(kernel));
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _completionSource = new TaskCompletionSource<KernelExecutionResult>();
        _cancellationSource = new CancellationTokenSource();
    }

    public async Task<KernelExecutionResult> WaitForCompletionAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            using var combinedCts = CancellationTokenSource.CreateLinkedTokenSource(
                cancellationToken, _cancellationSource.Token);
            
            return await _completionSource.Task.WaitAsync(combinedCts.Token);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (OperationCanceledException)
        {
            // Internal cancellation
            return new KernelExecutionResult(false, "Kernel execution was canceled");
        }
    }

    public void Cancel()
    {
        if (_isCompleted)
            return;

        try
        {
            _logger.LogDebug("Cancelling DotCompute kernel execution: {ExecutionId}", ExecutionId);
            _isCanceled = true;
            _cancellationSource.Cancel();
            _completionSource.TrySetCanceled();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error cancelling DotCompute kernel execution: {ExecutionId}", ExecutionId);
        }
    }

    internal void SetResult(KernelExecutionResult result)
    {
        if (_isCompleted)
            return;

        _isCompleted = true;
        _completionSource.TrySetResult(result);
    }

    internal void SetError(Exception exception)
    {
        if (_isCompleted)
            return;

        _isCompleted = true;
        _completionSource.TrySetException(exception);
    }

    public void Dispose()
    {
        Cancel();
        _cancellationSource?.Dispose();
        _executor.RemoveActiveExecution(ExecutionId);
    }
}

/// <summary>
/// DotCompute kernel graph implementation
/// </summary>
internal sealed class DotComputeKernelGraph : IKernelGraph
{
    private readonly string _graphName;
    private readonly DotComputeKernelExecutor _executor;
    private readonly ILogger _logger;
    private readonly List<KernelGraphNode> _nodes;
    private bool _disposed;

    public string GraphName => _graphName;
    public IReadOnlyList<KernelGraphNode> Nodes => _nodes.AsReadOnly();

    public DotComputeKernelGraph(string graphName, DotComputeKernelExecutor executor, ILogger logger)
    {
        _graphName = graphName ?? throw new ArgumentNullException(nameof(graphName));
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _nodes = new List<KernelGraphNode>();
    }

    public IKernelGraph AddNode(string nodeId, CompiledKernel kernel, KernelExecutionParameters parameters)
    {
        if (string.IsNullOrEmpty(nodeId))
            throw new ArgumentException("Node ID cannot be null or empty", nameof(nodeId));

        if (kernel == null)
            throw new ArgumentNullException(nameof(kernel));

        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        if (_nodes.Any(n => n.NodeId == nodeId))
            throw new ArgumentException($"Node with ID '{nodeId}' already exists", nameof(nodeId));

        var node = new KernelGraphNode(nodeId, kernel, parameters);
        _nodes.Add(node);

        _logger.LogDebug("Added node {NodeId} to DotCompute kernel graph {GraphName}", nodeId, _graphName);
        return this;
    }

    public IKernelGraph AddDependency(string fromNodeId, string toNodeId)
    {
        var fromNode = _nodes.FirstOrDefault(n => n.NodeId == fromNodeId) ??
            throw new ArgumentException($"Node '{fromNodeId}' not found", nameof(fromNodeId));

        var toNode = _nodes.FirstOrDefault(n => n.NodeId == toNodeId) ??
            throw new ArgumentException($"Node '{toNodeId}' not found", nameof(toNodeId));

        fromNode.Dependencies.Add(toNodeId);
        _logger.LogDebug("Added dependency {FromNode} -> {ToNode} in graph {GraphName}", fromNodeId, toNodeId, _graphName);
        
        return this;
    }

    public async Task<GraphExecutionResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("Executing DotCompute kernel graph: {GraphName} with {NodeCount} nodes", _graphName, _nodes.Count);

            var stopwatch = Stopwatch.StartNew();
            var results = new Dictionary<string, KernelExecutionResult>();
            var executed = new HashSet<string>();

            // Simple topological execution (in a real implementation, this would be more sophisticated)
            while (executed.Count < _nodes.Count)
            {
                var readyNodes = _nodes
                    .Where(n => !executed.Contains(n.NodeId))
                    .Where(n => n.Dependencies.All(dep => executed.Contains(dep)))
                    .ToList();

                if (!readyNodes.Any())
                {
                    var remaining = _nodes.Where(n => !executed.Contains(n.NodeId)).Select(n => n.NodeId);
                    throw new InvalidOperationException($"Circular dependency detected in graph. Remaining nodes: {string.Join(", ", remaining)}");
                }

                // Execute ready nodes in parallel
                var nodeTasks = readyNodes.Select(async node =>
                {
                    var result = await _executor.ExecuteAsync(node.Kernel, node.Parameters, cancellationToken);
                    return (node.NodeId, result);
                });

                var nodeResults = await Task.WhenAll(nodeTasks);
                
                foreach (var (nodeId, result) in nodeResults)
                {
                    results[nodeId] = result;
                    executed.Add(nodeId);
                    
                    if (!result.Success)
                    {
                        _logger.LogError("Node {NodeId} failed in graph {GraphName}: {ErrorMessage}", 
                            nodeId, _graphName, result.ErrorMessage);
                    }
                }
            }

            stopwatch.Stop();
            
            var successCount = results.Values.Count(r => r.Success);
            var failureCount = results.Values.Count(r => !r.Success);

            _logger.LogInformation(
                "DotCompute kernel graph execution completed: {GraphName} - {SuccessCount} succeeded, {FailureCount} failed in {TotalTime}ms",
                _graphName, successCount, failureCount, stopwatch.ElapsedMilliseconds);

            return new GraphExecutionResult(
                GraphName: _graphName,
                Success: failureCount == 0,
                NodeResults: results,
                TotalExecutionTime: stopwatch.Elapsed);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to execute DotCompute kernel graph: {GraphName}", _graphName);
            throw;
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing DotCompute kernel graph: {GraphName}", _graphName);
        _disposed = true;
    }
}