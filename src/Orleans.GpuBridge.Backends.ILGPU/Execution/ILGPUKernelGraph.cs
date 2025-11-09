using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Models;

namespace Orleans.GpuBridge.Backends.ILGPU.Execution;

/// <summary>
/// ILGPU kernel graph implementation
/// </summary>
internal sealed class ILGPUKernelGraph : IKernelGraph
{
    private readonly ILGPUKernelExecutor _executor;
    private readonly ILogger<ILGPUKernelGraph> _logger;
    private readonly List<ILGPUGraphNode> _nodes;
    private readonly Dictionary<string, ILGPUGraphNode> _nodeMap;
    private bool _disposed;

    public string Name { get; }

    public ILGPUKernelGraph(
        string name,
        ILGPUKernelExecutor executor,
        ILogger<ILGPUKernelGraph> logger)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _nodes = new List<ILGPUGraphNode>();
        _nodeMap = new Dictionary<string, ILGPUGraphNode>();

        _logger.LogDebug("Created ILGPU kernel graph: {GraphName}", Name);
    }

    public IGraphNode AddKernel(
        CompiledKernel kernel,
        KernelExecutionParameters parameters,
        IReadOnlyList<IGraphNode>? dependencies = null)
    {
        if (kernel == null)
            throw new ArgumentNullException(nameof(kernel));

        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        var nodeId = $"kernel-{Guid.NewGuid():N}";
        var node = new ILGPUGraphNode(
            nodeId,
            GraphNodeType.Kernel,
            dependencies ?? Array.Empty<IGraphNode>(),
            kernel,
            parameters);

        _nodes.Add(node);
        _nodeMap[nodeId] = node;

        _logger.LogDebug("Added kernel node to graph {GraphName}: {NodeId} ({KernelName})",
            Name, nodeId, kernel.Name);

        return node;
    }

    public IGraphNode AddMemCopy(
        IDeviceMemory source,
        IDeviceMemory destination,
        long sizeBytes,
        IReadOnlyList<IGraphNode>? dependencies = null)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));

        if (destination == null)
            throw new ArgumentNullException(nameof(destination));

        var nodeId = $"memcpy-{Guid.NewGuid():N}";
        var node = new ILGPUGraphNode(
            nodeId,
            GraphNodeType.MemCopy,
            dependencies ?? Array.Empty<IGraphNode>(),
            source,
            destination,
            sizeBytes);

        _nodes.Add(node);
        _nodeMap[nodeId] = node;

        _logger.LogDebug("Added memory copy node to graph {GraphName}: {NodeId} ({SizeBytes} bytes)",
            Name, nodeId, sizeBytes);

        return node;
    }

    public IGraphNode AddBarrier(IReadOnlyList<IGraphNode> dependencies)
    {
        if (dependencies == null)
            throw new ArgumentNullException(nameof(dependencies));

        var nodeId = $"barrier-{Guid.NewGuid():N}";
        var node = new ILGPUGraphNode(
            nodeId,
            GraphNodeType.Barrier,
            dependencies);

        _nodes.Add(node);
        _nodeMap[nodeId] = node;

        _logger.LogDebug("Added barrier node to graph {GraphName}: {NodeId} (depends on {DependencyCount} nodes)",
            Name, nodeId, dependencies.Count);

        return node;
    }

    public async Task<ICompiledGraph> CompileAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("Compiling ILGPU kernel graph: {GraphName} with {NodeCount} nodes",
                Name, _nodes.Count);

            // Validate graph
            var validation = Validate();
            if (!validation.IsValid)
            {
                throw new InvalidOperationException($"Graph validation failed: {validation.Errors?.FirstOrDefault()}");
            }

            // Topological sort to determine execution order
            var executionOrder = TopologicalSort();

            var compiledGraph = new ILGPUCompiledGraph(
                Name,
                executionOrder,
                _executor,
                new CompiledGraphLogger(_logger)); // Use a logger wrapper

            _logger.LogInformation("ILGPU kernel graph compilation completed: {GraphName}", Name);

            return await Task.FromResult<ICompiledGraph>(compiledGraph).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to compile ILGPU kernel graph: {GraphName}", Name);
            throw;
        }
    }

    public GraphValidationResult Validate()
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        try
        {
            // Check for cycles
            var hasCycles = HasCycles();
            if (hasCycles)
            {
                errors.Add("Graph contains cycles");
            }

            // Check node dependencies
            foreach (var node in _nodes)
            {
                foreach (var dependency in node.Dependencies)
                {
                    if (!_nodes.Contains(dependency))
                    {
                        errors.Add($"Node {node.NodeId} depends on node not in graph: {dependency.NodeId}");
                    }
                }
            }

            // Check for unreachable nodes
            var reachableNodes = GetReachableNodes();
            if (reachableNodes.Count < _nodes.Count)
            {
                warnings.Add($"Graph has {_nodes.Count - reachableNodes.Count} unreachable nodes");
            }

            var isValid = errors.Count == 0;

            _logger.LogDebug(
                "Graph validation completed: {GraphName}, Valid: {IsValid}, Errors: {ErrorCount}, Warnings: {WarningCount}",
                Name, isValid, errors.Count, warnings.Count);

            return new GraphValidationResult(
                IsValid: isValid,
                Errors: errors.Count > 0 ? errors : null,
                Warnings: warnings.Count > 0 ? warnings : null,
                HasCycles: hasCycles);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during graph validation: {GraphName}", Name);
            return new GraphValidationResult(false, new[] { $"Validation error: {ex.Message}" });
        }
    }

    private bool HasCycles()
    {
        var visited = new HashSet<ILGPUGraphNode>();
        var recursionStack = new HashSet<ILGPUGraphNode>();

        foreach (var node in _nodes)
        {
            if (HasCyclesUtil(node, visited, recursionStack))
            {
                return true;
            }
        }

        return false;
    }

    private bool HasCyclesUtil(ILGPUGraphNode node, HashSet<ILGPUGraphNode> visited, HashSet<ILGPUGraphNode> recursionStack)
    {
        visited.Add(node);
        recursionStack.Add(node);

        foreach (var dependency in node.Dependencies.Cast<ILGPUGraphNode>())
        {
            if (!visited.Contains(dependency))
            {
                if (HasCyclesUtil(dependency, visited, recursionStack))
                    return true;
            }
            else if (recursionStack.Contains(dependency))
            {
                return true;
            }
        }

        recursionStack.Remove(node);
        return false;
    }

    private HashSet<ILGPUGraphNode> GetReachableNodes()
    {
        var reachable = new HashSet<ILGPUGraphNode>();
        var queue = new Queue<ILGPUGraphNode>();

        // Start with nodes that have no dependencies (entry points)
        foreach (var node in _nodes.Where(n => n.Dependencies.Count == 0))
        {
            queue.Enqueue(node);
            reachable.Add(node);
        }

        while (queue.Count > 0)
        {
            var current = queue.Dequeue();

            // Find nodes that depend on the current node
            foreach (var node in _nodes.Where(n => n.Dependencies.Contains(current)))
            {
                if (reachable.Add(node))
                {
                    queue.Enqueue(node);
                }
            }
        }

        return reachable;
    }

    private List<ILGPUGraphNode> TopologicalSort()
    {
        var result = new List<ILGPUGraphNode>();
        var visited = new HashSet<ILGPUGraphNode>();
        var stack = new Stack<ILGPUGraphNode>();

        foreach (var node in _nodes)
        {
            if (!visited.Contains(node))
            {
                TopologicalSortUtil(node, visited, stack);
            }
        }

        while (stack.Count > 0)
        {
            result.Add(stack.Pop());
        }

        return result;
    }

    private void TopologicalSortUtil(ILGPUGraphNode node, HashSet<ILGPUGraphNode> visited, Stack<ILGPUGraphNode> stack)
    {
        visited.Add(node);

        foreach (var dependency in node.Dependencies.Cast<ILGPUGraphNode>())
        {
            if (!visited.Contains(dependency))
            {
                TopologicalSortUtil(dependency, visited, stack);
            }
        }

        stack.Push(node);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing ILGPU kernel graph: {GraphName}", Name);

        try
        {
            _nodes.Clear();
            _nodeMap.Clear();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing ILGPU kernel graph: {GraphName}", Name);
        }

        _disposed = true;
    }
}
