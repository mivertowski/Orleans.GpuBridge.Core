using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;

namespace Orleans.GpuBridge.Backends.ILGPU.Execution;

/// <summary>
/// ILGPU compiled graph implementation
/// </summary>
internal sealed class ILGPUCompiledGraph : ICompiledGraph
{
    private readonly string _name;
    private readonly List<ILGPUGraphNode> _executionOrder;
    private readonly ILGPUKernelExecutor _executor;
    private readonly ILogger<ILGPUCompiledGraph> _logger;
    private readonly Dictionary<string, KernelExecutionParameters> _updatedParameters;
    private bool _disposed;

    public ILGPUCompiledGraph(
        string name,
        List<ILGPUGraphNode> executionOrder,
        ILGPUKernelExecutor executor,
        ILogger<ILGPUCompiledGraph> logger)
    {
        _name = name ?? throw new ArgumentNullException(nameof(name));
        _executionOrder = executionOrder ?? throw new ArgumentNullException(nameof(executionOrder));
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _updatedParameters = new Dictionary<string, KernelExecutionParameters>();
    }

    public async Task<GraphExecutionResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var nodeTimings = new Dictionary<string, KernelTiming>();
        var nodesExecuted = 0;

        try
        {
            _logger.LogInformation("Executing ILGPU compiled graph: {GraphName} with {NodeCount} nodes",
                _name, _executionOrder.Count);

            foreach (var node in _executionOrder)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var nodeStopwatch = System.Diagnostics.Stopwatch.StartNew();

                try
                {
                    await ExecuteNodeAsync(node, cancellationToken).ConfigureAwait(false);
                    nodesExecuted++;

                    nodeStopwatch.Stop();
                    nodeTimings[node.NodeId] = new KernelTiming(
                        QueueTime: TimeSpan.Zero,
                        KernelTime: nodeStopwatch.Elapsed,
                        TotalTime: nodeStopwatch.Elapsed);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to execute graph node: {NodeId} in graph: {GraphName}",
                        node.NodeId, _name);

                    return new GraphExecutionResult(
                        Success: false,
                        NodesExecuted: nodesExecuted,
                        ExecutionTime: stopwatch.Elapsed,
                        NodeTimings: nodeTimings);
                }
            }

            stopwatch.Stop();

            _logger.LogInformation(
                "ILGPU compiled graph execution completed: {GraphName}, {NodesExecuted} nodes in {ExecutionTime}ms",
                _name, nodesExecuted, stopwatch.ElapsedMilliseconds);

            return new GraphExecutionResult(
                Success: true,
                NodesExecuted: nodesExecuted,
                ExecutionTime: stopwatch.Elapsed,
                NodeTimings: nodeTimings);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "ILGPU compiled graph execution failed: {GraphName}", _name);
            throw;
        }
    }

    public void UpdateParameters(string nodeId, KernelExecutionParameters parameters)
    {
        if (string.IsNullOrEmpty(nodeId))
            throw new ArgumentException("Node ID cannot be null or empty", nameof(nodeId));

        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        _updatedParameters[nodeId] = parameters;

        _logger.LogDebug("Updated parameters for graph node: {NodeId} in graph: {GraphName}",
            nodeId, _name);
    }

    private async Task ExecuteNodeAsync(ILGPUGraphNode node, CancellationToken cancellationToken)
    {
        switch (node.Type)
        {
            case GraphNodeType.Kernel:
                await ExecuteKernelNodeAsync(node, cancellationToken).ConfigureAwait(false);
                break;

            case GraphNodeType.MemCopy:
                await ExecuteMemCopyNodeAsync(node, cancellationToken).ConfigureAwait(false);
                break;

            case GraphNodeType.Barrier:
                await ExecuteBarrierNodeAsync(node, cancellationToken).ConfigureAwait(false);
                break;

            default:
                throw new NotSupportedException($"Graph node type not supported: {node.Type}");
        }
    }

    private async Task ExecuteKernelNodeAsync(ILGPUGraphNode node, CancellationToken cancellationToken)
    {
        if (node.Kernel == null || node.Parameters == null)
        {
            throw new InvalidOperationException($"Kernel node {node.NodeId} missing kernel or parameters");
        }

        // Use updated parameters if available
        var parameters = _updatedParameters.TryGetValue(node.NodeId, out var updatedParams)
            ? updatedParams
            : node.Parameters;

        var result = await _executor.ExecuteAsync(node.Kernel, parameters, cancellationToken).ConfigureAwait(false);

        if (!result.Success)
        {
            throw new InvalidOperationException($"Kernel execution failed: {result.ErrorMessage}");
        }
    }

    private async Task ExecuteMemCopyNodeAsync(ILGPUGraphNode node, CancellationToken cancellationToken)
    {
        if (node.SourceMemory == null || node.DestinationMemory == null)
        {
            throw new InvalidOperationException($"Memory copy node {node.NodeId} missing source or destination");
        }

        await node.DestinationMemory.CopyFromAsync(
            node.SourceMemory,
            0,
            0,
            node.SizeBytes,
            cancellationToken).ConfigureAwait(false);
    }

    private async Task ExecuteBarrierNodeAsync(ILGPUGraphNode node, CancellationToken cancellationToken)
    {
        // Barrier nodes just ensure all dependencies have completed
        // In our execution model, this is implicit since we execute sequentially
        await Task.CompletedTask.ConfigureAwait(false);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing ILGPU compiled graph: {GraphName}", _name);

        try
        {
            _executionOrder.Clear();
            _updatedParameters.Clear();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing ILGPU compiled graph: {GraphName}", _name);
        }

        _disposed = true;
    }
}
