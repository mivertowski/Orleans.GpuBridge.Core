using System;
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
/// ILGPU kernel execution tracking
/// </summary>
internal sealed class ILGPUKernelExecution : IKernelExecution
{
    private readonly ILGPUKernelExecutor _executor;
    private readonly ILogger<ILGPUKernelExecution> _logger;
    private readonly TaskCompletionSource<KernelExecutionResult> _completionSource;
    private readonly CancellationTokenSource _cancellationTokenSource;
    private KernelTiming? _timing;
    private volatile KernelExecutionStatus _status;
    private volatile int _progressInt; // Progress as int (0-100)

    public string ExecutionId { get; }
    public CompiledKernel Kernel { get; }
    public KernelExecutionStatus Status => _status;
    public bool IsComplete => _status == KernelExecutionStatus.Completed || 
                             _status == KernelExecutionStatus.Failed || 
                             _status == KernelExecutionStatus.Cancelled ||
                             _status == KernelExecutionStatus.Timeout;
    public double Progress => _progressInt / 100.0;

    public ILGPUKernelExecution(
        string executionId,
        CompiledKernel kernel,
        ILGPUKernelExecutor executor,
        ILogger<ILGPUKernelExecution> logger)
    {
        ExecutionId = executionId ?? throw new ArgumentNullException(nameof(executionId));
        Kernel = kernel ?? throw new ArgumentNullException(nameof(kernel));
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        
        _completionSource = new TaskCompletionSource<KernelExecutionResult>();
        _cancellationTokenSource = new CancellationTokenSource();
        _status = KernelExecutionStatus.Queued;
        _progressInt = 0;

        _logger.LogDebug("Created kernel execution: {ExecutionId} for kernel: {KernelName}", 
            ExecutionId, Kernel.Name);
    }

    public async Task<KernelExecutionResult> WaitForCompletionAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogDebug("Waiting for kernel execution completion: {ExecutionId}", ExecutionId);

            using var combinedTokenSource = CancellationTokenSource.CreateLinkedTokenSource(
                cancellationToken, _cancellationTokenSource.Token);

            var result = await _completionSource.Task.ConfigureAwait(false);

            _logger.LogDebug(
                "Kernel execution completed: {ExecutionId}, Success: {Success}",
                ExecutionId, result.Success);

            return result;
        }
        catch (OperationCanceledException)
        {
            _logger.LogDebug("Kernel execution wait cancelled: {ExecutionId}", ExecutionId);
            
            if (cancellationToken.IsCancellationRequested)
                throw;

            // Internal cancellation
            return new KernelExecutionResult(false, "Execution was cancelled");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error waiting for kernel execution: {ExecutionId}", ExecutionId);
            throw;
        }
    }

    public async Task CancelAsync()
    {
        if (IsComplete)
        {
            _logger.LogDebug("Cannot cancel already completed execution: {ExecutionId}", ExecutionId);
            return;
        }

        try
        {
            _logger.LogInformation("Cancelling kernel execution: {ExecutionId}", ExecutionId);

            _status = KernelExecutionStatus.Cancelled;
            _cancellationTokenSource.Cancel();

            var result = new KernelExecutionResult(false, "Execution was cancelled");
            _completionSource.TrySetResult(result);

            // Remove from active executions
            _executor.RemoveActiveExecution(ExecutionId);

            _logger.LogInformation("Kernel execution cancelled: {ExecutionId}", ExecutionId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error cancelling kernel execution: {ExecutionId}", ExecutionId);
            throw;
        }
    }

    public KernelTiming? GetTiming()
    {
        return _timing;
    }

    internal void SetRunning()
    {
        _status = KernelExecutionStatus.Running;
        _progressInt = 10; // Small progress to indicate start (10%)
        
        _logger.LogDebug("Kernel execution started: {ExecutionId}", ExecutionId);
    }

    internal void UpdateProgress(double progress)
    {
        if (progress < 0.0 || progress > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(progress), "Progress must be between 0.0 and 1.0");
        }

        _progressInt = (int)(progress * 100);
        
        _logger.LogTrace("Kernel execution progress: {ExecutionId} - {Progress:P1}", ExecutionId, progress);
    }

    internal void SetResult(KernelExecutionResult result)
    {
        if (result == null)
            throw new ArgumentNullException(nameof(result));

        try
        {
            _status = result.Success ? KernelExecutionStatus.Completed : KernelExecutionStatus.Failed;
            _progress = 1.0;
            _timing = result.Timing;

            _completionSource.TrySetResult(result);

            _logger.LogDebug(
                "Kernel execution result set: {ExecutionId}, Success: {Success}",
                ExecutionId, result.Success);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting kernel execution result: {ExecutionId}", ExecutionId);
            SetError(ex);
        }
    }

    internal void SetError(Exception exception)
    {
        if (exception == null)
            throw new ArgumentNullException(nameof(exception));

        try
        {
            _status = KernelExecutionStatus.Failed;
            _progress = 1.0;

            var result = new KernelExecutionResult(false, exception.Message);
            _completionSource.TrySetResult(result);

            _logger.LogError(exception, "Kernel execution failed: {ExecutionId}", ExecutionId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting kernel execution error: {ExecutionId}", ExecutionId);
            
            // Last resort - set exception directly
            _completionSource.TrySetException(ex);
        }
    }

    internal void SetTimeout()
    {
        try
        {
            _status = KernelExecutionStatus.Timeout;
            _progress = 1.0;

            var result = new KernelExecutionResult(false, "Kernel execution timed out");
            _completionSource.TrySetResult(result);

            _logger.LogWarning("Kernel execution timed out: {ExecutionId}", ExecutionId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting kernel execution timeout: {ExecutionId}", ExecutionId);
        }
    }

    internal void Cancel()
    {
        if (!IsComplete)
        {
            _ = CancelAsync();
        }
    }
}

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
                _logger); // Use the existing logger

            _logger.LogInformation("ILGPU kernel graph compilation completed: {GraphName}", Name);

            return compiledGraph;
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

/// <summary>
/// ILGPU graph node implementation
/// </summary>
internal sealed class ILGPUGraphNode : IGraphNode
{
    public string NodeId { get; }
    public GraphNodeType Type { get; }
    public IReadOnlyList<IGraphNode> Dependencies { get; }
    
    // Optional node data
    public CompiledKernel? Kernel { get; }
    public KernelExecutionParameters? Parameters { get; }
    public IDeviceMemory? SourceMemory { get; }
    public IDeviceMemory? DestinationMemory { get; }
    public long SizeBytes { get; }

    public ILGPUGraphNode(
        string nodeId,
        GraphNodeType type,
        IReadOnlyList<IGraphNode> dependencies,
        CompiledKernel? kernel = null,
        KernelExecutionParameters? parameters = null)
    {
        NodeId = nodeId ?? throw new ArgumentNullException(nameof(nodeId));
        Type = type;
        Dependencies = dependencies ?? throw new ArgumentNullException(nameof(dependencies));
        Kernel = kernel;
        Parameters = parameters;
    }

    public ILGPUGraphNode(
        string nodeId,
        GraphNodeType type,
        IReadOnlyList<IGraphNode> dependencies,
        IDeviceMemory sourceMemory,
        IDeviceMemory destinationMemory,
        long sizeBytes)
    {
        NodeId = nodeId ?? throw new ArgumentNullException(nameof(nodeId));
        Type = type;
        Dependencies = dependencies ?? throw new ArgumentNullException(nameof(dependencies));
        SourceMemory = sourceMemory ?? throw new ArgumentNullException(nameof(sourceMemory));
        DestinationMemory = destinationMemory ?? throw new ArgumentNullException(nameof(destinationMemory));
        SizeBytes = sizeBytes;
    }
}

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
                    await ExecuteNodeAsync(node, cancellationToken);
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
                await ExecuteKernelNodeAsync(node, cancellationToken);
                break;

            case GraphNodeType.MemCopy:
                await ExecuteMemCopyNodeAsync(node, cancellationToken);
                break;

            case GraphNodeType.Barrier:
                await ExecuteBarrierNodeAsync(node, cancellationToken);
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

        var result = await _executor.ExecuteAsync(node.Kernel, parameters, cancellationToken);
        
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
            cancellationToken);
    }

    private async Task ExecuteBarrierNodeAsync(ILGPUGraphNode node, CancellationToken cancellationToken)
    {
        // Barrier nodes just ensure all dependencies have completed
        // In our execution model, this is implicit since we execute sequentially
        await Task.CompletedTask;
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