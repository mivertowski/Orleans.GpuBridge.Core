using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;

/// <summary>
/// Compiled execution graph
/// </summary>
public interface ICompiledGraph : IDisposable
{
    /// <summary>
    /// Executes the compiled graph
    /// </summary>
    Task<GraphExecutionResult> ExecuteAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Updates kernel parameters without recompiling
    /// </summary>
    void UpdateParameters(string nodeId, KernelExecutionParameters parameters);
}