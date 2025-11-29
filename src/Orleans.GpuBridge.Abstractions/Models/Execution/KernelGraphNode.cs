using System.Collections.Generic;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;

namespace Orleans.GpuBridge.Abstractions.Models.Execution;

/// <summary>
/// Represents a node in a kernel execution graph.
/// </summary>
/// <remarks>
/// A kernel graph node encapsulates a compiled kernel along with its execution parameters
/// and dependency information for graph-based execution workflows.
/// </remarks>
public sealed class KernelGraphNode
{
    /// <summary>
    /// Gets the unique identifier for this node within the graph.
    /// </summary>
    public string NodeId { get; }

    /// <summary>
    /// Gets the compiled kernel associated with this node.
    /// </summary>
    public CompiledKernel Kernel { get; }

    /// <summary>
    /// Gets the execution parameters for this kernel.
    /// </summary>
    public KernelExecutionParameters Parameters { get; }

    /// <summary>
    /// Gets the collection of node IDs that this node depends on.
    /// This node will not execute until all dependencies have completed.
    /// </summary>
    public ICollection<string> Dependencies { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="KernelGraphNode"/> class.
    /// </summary>
    /// <param name="nodeId">The unique identifier for the node.</param>
    /// <param name="kernel">The compiled kernel to execute.</param>
    /// <param name="parameters">The execution parameters for the kernel.</param>
    /// <param name="dependencies">Optional collection of node IDs this node depends on.</param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="nodeId"/>, <paramref name="kernel"/>, or <paramref name="parameters"/> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="nodeId"/> is empty or whitespace.
    /// </exception>
    public KernelGraphNode(
        string nodeId,
        CompiledKernel kernel,
        KernelExecutionParameters parameters,
        ICollection<string>? dependencies = null)
    {
        if (string.IsNullOrWhiteSpace(nodeId))
            throw new ArgumentException("Node ID cannot be null or empty", nameof(nodeId));

        NodeId = nodeId;
        Kernel = kernel ?? throw new ArgumentNullException(nameof(kernel));
        Parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
        Dependencies = dependencies ?? new List<string>();
    }

    /// <summary>
    /// Determines whether this node has any dependencies.
    /// </summary>
    /// <returns>
    /// <c>true</c> if this node has dependencies; otherwise, <c>false</c>.
    /// </returns>
    public bool HasDependencies => Dependencies.Count > 0;

    /// <summary>
    /// Determines whether all dependencies are satisfied by the given collection of completed nodes.
    /// </summary>
    /// <param name="completedNodes">The collection of node IDs that have completed execution.</param>
    /// <returns>
    /// <c>true</c> if all dependencies are satisfied; otherwise, <c>false</c>.
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="completedNodes"/> is null.
    /// </exception>
    public bool AreDependenciesSatisfied(ICollection<string> completedNodes)
    {
        if (completedNodes == null)
            throw new ArgumentNullException(nameof(completedNodes));

        if (!HasDependencies)
            return true;

        return Dependencies.All(dep => completedNodes.Contains(dep));
    }

    /// <summary>
    /// Gets a collection of unsatisfied dependencies given the completed nodes.
    /// </summary>
    /// <param name="completedNodes">The collection of node IDs that have completed execution.</param>
    /// <returns>
    /// A collection of node IDs representing unsatisfied dependencies.
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="completedNodes"/> is null.
    /// </exception>
    public IEnumerable<string> GetUnsatisfiedDependencies(ICollection<string> completedNodes)
    {
        if (completedNodes == null)
            throw new ArgumentNullException(nameof(completedNodes));

        return Dependencies.Where(dep => !completedNodes.Contains(dep));
    }

    /// <summary>
    /// Returns a string representation of this kernel graph node.
    /// </summary>
    /// <returns>
    /// A string containing the node ID, kernel name, and dependency count.
    /// </returns>
    public override string ToString()
    {
        var depCount = Dependencies.Count;
        var depText = depCount == 1 ? "1 dependency" : $"{depCount} dependencies";
        return $"Node '{NodeId}' [Kernel: {Kernel.Name}, {depText}]";
    }
}