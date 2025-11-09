using System;
using System.Collections.Generic;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;

namespace Orleans.GpuBridge.Backends.DotCompute.Execution;

/// <summary>
/// Adapter to make KernelGraphNode compatible with IGraphNode interface
/// </summary>
/// <remarks>
/// This adapter allows KernelGraphNode instances to be used where IGraphNode
/// is required, following the Adapter design pattern.
/// </remarks>
internal sealed class GraphNodeAdapter : IGraphNode
{
    private readonly KernelGraphNode _kernelNode;

    public string NodeId => _kernelNode.NodeId;
    public GraphNodeType Type => GraphNodeType.Kernel;
    public IReadOnlyList<IGraphNode> Dependencies { get; }

    public GraphNodeAdapter(KernelGraphNode kernelNode, IReadOnlyList<IGraphNode> dependencies)
    {
        _kernelNode = kernelNode ?? throw new ArgumentNullException(nameof(kernelNode));
        Dependencies = dependencies ?? Array.Empty<IGraphNode>();
    }
}
