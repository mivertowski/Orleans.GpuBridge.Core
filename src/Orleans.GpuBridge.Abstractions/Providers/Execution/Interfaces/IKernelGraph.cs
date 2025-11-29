using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;

/// <summary>
/// Kernel execution graph for optimized multi-kernel execution
/// </summary>
public interface IKernelGraph : IDisposable
{
    /// <summary>
    /// Graph name
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Adds a kernel node to the graph
    /// </summary>
    IGraphNode AddKernel(
        CompiledKernel kernel,
        KernelExecutionParameters parameters,
        IReadOnlyList<IGraphNode>? dependencies = null);

    /// <summary>
    /// Adds a memory copy node to the graph
    /// </summary>
    IGraphNode AddMemCopy(
        IDeviceMemory source,
        IDeviceMemory destination,
        long sizeBytes,
        IReadOnlyList<IGraphNode>? dependencies = null);

    /// <summary>
    /// Adds a synchronization barrier
    /// </summary>
    IGraphNode AddBarrier(IReadOnlyList<IGraphNode> dependencies);

    /// <summary>
    /// Compiles the graph for execution
    /// </summary>
    Task<ICompiledGraph> CompileAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Validates the graph structure
    /// </summary>
    GraphValidationResult Validate();
}