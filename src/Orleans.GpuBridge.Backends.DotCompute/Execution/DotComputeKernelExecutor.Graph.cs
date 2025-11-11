using System;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;

namespace Orleans.GpuBridge.Backends.DotCompute.Execution;

/// <summary>
/// Graph creation functionality for DotComputeKernelExecutor
/// </summary>
internal sealed partial class DotComputeKernelExecutor
{
    /// <summary>
    /// Creates a new kernel execution graph
    /// </summary>
    public IKernelGraph CreateGraph(string graphName)
    {
        if (string.IsNullOrEmpty(graphName))
            throw new ArgumentException("Graph name cannot be null or empty", nameof(graphName));

        return new DotComputeKernelGraph(graphName, this, _loggerFactory.CreateLogger<DotComputeKernelGraph>());
    }
}
