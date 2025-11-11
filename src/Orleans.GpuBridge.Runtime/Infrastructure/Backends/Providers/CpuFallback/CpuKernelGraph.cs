using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Models.Compilation;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;

namespace Orleans.GpuBridge.Runtime.Providers;

/// <summary>
/// CPU kernel graph builder for fallback provider
/// </summary>
internal sealed class CpuKernelGraph : IKernelGraph
{
    public string Name { get; }

    public CpuKernelGraph(string name)
    {
        Name = name;
    }

    public IGraphNode AddKernel(CompiledKernel kernel, KernelExecutionParameters parameters, IReadOnlyList<IGraphNode>? dependencies = null)
    {
        return new CpuGraphNode("kernel-" + Guid.NewGuid().ToString(), GraphNodeType.Kernel, dependencies ?? Array.Empty<IGraphNode>());
    }

    public IGraphNode AddMemCopy(IDeviceMemory source, IDeviceMemory destination, long sizeBytes, IReadOnlyList<IGraphNode>? dependencies = null)
    {
        return new CpuGraphNode("memcpy-" + Guid.NewGuid().ToString(), GraphNodeType.MemCopy, dependencies ?? Array.Empty<IGraphNode>());
    }

    public IGraphNode AddBarrier(IReadOnlyList<IGraphNode> dependencies)
    {
        return new CpuGraphNode("barrier-" + Guid.NewGuid().ToString(), GraphNodeType.Barrier, dependencies);
    }

    public Task<ICompiledGraph> CompileAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult<ICompiledGraph>(new CpuCompiledGraph());
    }

    public GraphValidationResult Validate()
    {
        return new GraphValidationResult(IsValid: true);
    }

    public void Dispose() { }
}

/// <summary>
/// CPU graph node for fallback provider
/// </summary>
internal sealed class CpuGraphNode : IGraphNode
{
    public string NodeId { get; }
    public GraphNodeType Type { get; }
    public IReadOnlyList<IGraphNode> Dependencies { get; }

    public CpuGraphNode(string nodeId, GraphNodeType type, IReadOnlyList<IGraphNode> dependencies)
    {
        NodeId = nodeId;
        Type = type;
        Dependencies = dependencies;
    }
}

/// <summary>
/// CPU compiled graph for fallback provider
/// </summary>
internal sealed class CpuCompiledGraph : ICompiledGraph
{
    public Task<GraphExecutionResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new GraphExecutionResult(
            Success: true,
            NodesExecuted: 0,
            ExecutionTime: TimeSpan.Zero));
    }

    public void UpdateParameters(string nodeId, KernelExecutionParameters parameters) { }

    public void Dispose() { }
}
