using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using OrleansCompiledKernel = Orleans.GpuBridge.Abstractions.Models.CompiledKernel;

namespace Orleans.GpuBridge.Backends.DotCompute.Execution;

/// <summary>
/// DotCompute kernel graph implementation for managing kernel execution dependencies
/// </summary>
internal sealed class DotComputeKernelGraph : IKernelGraph
{
    private readonly string _graphName;
    private readonly DotComputeKernelExecutor _executor;
    private readonly ILogger _logger;
    private readonly List<KernelGraphNode> _nodes;
    private bool _disposed;

    public string Name => _graphName;
    public string GraphName => _graphName;
    public IReadOnlyList<KernelGraphNode> Nodes => _nodes.AsReadOnly();

    public DotComputeKernelGraph(string graphName, DotComputeKernelExecutor executor, ILogger logger)
    {
        _graphName = graphName ?? throw new ArgumentNullException(nameof(graphName));
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _nodes = new List<KernelGraphNode>();
    }

    /// <summary>
    /// Adds a kernel node to the graph
    /// </summary>
    public IKernelGraph AddNode(string nodeId, OrleansCompiledKernel kernel, KernelExecutionParameters parameters)
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

    /// <summary>
    /// Adds a dependency between two nodes
    /// </summary>
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

    /// <summary>
    /// Executes the kernel graph with topological ordering
    /// </summary>
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
                    var result = await _executor.ExecuteAsync(node.Kernel, node.Parameters, cancellationToken).ConfigureAwait(false);
                    return (node.NodeId, result);
                });

                var nodeResults = await Task.WhenAll(nodeTasks).ConfigureAwait(false);

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
                Success: failureCount == 0,
                NodesExecuted: results.Count,
                ExecutionTime: stopwatch.Elapsed,
                NodeTimings: results.ToDictionary(
                    kvp => kvp.Key,
                    kvp => new KernelTiming(
                        QueueTime: kvp.Value.Timing?.QueueTime ?? TimeSpan.Zero,
                        KernelTime: kvp.Value.Timing?.KernelTime ?? TimeSpan.Zero,
                        TotalTime: kvp.Value.Timing?.TotalTime ?? TimeSpan.Zero)));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to execute DotCompute kernel graph: {GraphName}", _graphName);
            throw;
        }
    }

    /// <summary>
    /// Adds a kernel to the graph with dependencies
    /// </summary>
    public IGraphNode AddKernel(OrleansCompiledKernel kernel, KernelExecutionParameters parameters, IReadOnlyList<IGraphNode>? dependencies = null)
    {
        var nodeId = Guid.NewGuid().ToString();
        AddNode(nodeId, kernel, parameters);
        var kernelNode = new KernelGraphNode(nodeId, kernel, parameters);
        return new GraphNodeAdapter(kernelNode, dependencies ?? Array.Empty<IGraphNode>());
    }

    /// <summary>
    /// Adds a memory copy operation to the graph
    /// </summary>
    public IGraphNode AddMemCopy(IDeviceMemory source, IDeviceMemory destination, long size, IReadOnlyList<IGraphNode>? dependencies = null)
    {
        // Memory copy operations are currently not supported in DotCompute backend
        throw new NotSupportedException("Memory copy operations are not supported in the DotCompute backend");
    }

    /// <summary>
    /// Adds a synchronization barrier to the graph
    /// </summary>
    public IGraphNode AddBarrier(IReadOnlyList<IGraphNode> dependencies)
    {
        // Barriers are currently not supported in DotCompute backend
        throw new NotSupportedException("Barriers are not supported in the DotCompute backend");
    }

    /// <summary>
    /// Compiles the graph for optimized execution
    /// </summary>
    public async Task<ICompiledGraph> CompileAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        return new DotComputeCompiledGraph(_graphName, _nodes.AsReadOnly());
    }

    /// <summary>
    /// Validates the graph for circular dependencies and correctness
    /// </summary>
    public GraphValidationResult Validate()
    {
        try
        {
            // Basic validation - check for circular dependencies
            var visited = new HashSet<string>();
            var recursionStack = new HashSet<string>();

            foreach (var node in _nodes)
            {
                if (!visited.Contains(node.NodeId))
                {
                    ValidateNode(node.NodeId, visited, recursionStack);
                }
            }

            return GraphValidationResult.Success();
        }
        catch (Exception ex)
        {
            return GraphValidationResult.Error(ex.Message);
        }
    }

    private void ValidateNode(string nodeId, HashSet<string> visited, HashSet<string> recursionStack)
    {
        visited.Add(nodeId);
        recursionStack.Add(nodeId);

        var node = _nodes.FirstOrDefault(n => n.NodeId == nodeId);
        if (node != null)
        {
            foreach (var dependency in node.Dependencies)
            {
                if (!visited.Contains(dependency))
                {
                    ValidateNode(dependency, visited, recursionStack);
                }
                else if (recursionStack.Contains(dependency))
                {
                    throw new InvalidOperationException($"Circular dependency detected: {nodeId} -> {dependency}");
                }
            }
        }

        recursionStack.Remove(nodeId);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing DotCompute kernel graph: {GraphName}", _graphName);
        _disposed = true;
    }
}
