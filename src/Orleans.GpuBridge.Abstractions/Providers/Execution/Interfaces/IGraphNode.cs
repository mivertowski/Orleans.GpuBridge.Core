using System.Collections.Generic;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;

/// <summary>
/// Node in a kernel execution graph
/// </summary>
public interface IGraphNode
{
    /// <summary>
    /// Node ID
    /// </summary>
    string NodeId { get; }

    /// <summary>
    /// Node type
    /// </summary>
    GraphNodeType Type { get; }

    /// <summary>
    /// Dependencies
    /// </summary>
    IReadOnlyList<IGraphNode> Dependencies { get; }
}