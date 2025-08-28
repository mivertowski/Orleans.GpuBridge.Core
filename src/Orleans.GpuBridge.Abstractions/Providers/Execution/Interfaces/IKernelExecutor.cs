using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results.Statistics;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;

/// <summary>
/// Interface for kernel execution in GPU backends
/// </summary>
public interface IKernelExecutor
{
    /// <summary>
    /// Executes a kernel synchronously
    /// </summary>
    Task<KernelExecutionResult> ExecuteAsync(
        CompiledKernel kernel,
        KernelExecutionParameters parameters,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Executes a kernel asynchronously (non-blocking)
    /// </summary>
    Task<IKernelExecution> ExecuteAsyncNonBlocking(
        CompiledKernel kernel,
        KernelExecutionParameters parameters,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Executes multiple kernels in a batch
    /// </summary>
    Task<BatchExecutionResult> ExecuteBatchAsync(
        IReadOnlyList<KernelBatchItem> batch,
        BatchExecutionOptions options,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Creates a kernel execution graph for optimized execution
    /// </summary>
    IKernelGraph CreateGraph(string graphName);
    
    /// <summary>
    /// Profiles kernel execution
    /// </summary>
    Task<KernelProfile> ProfileAsync(
        CompiledKernel kernel,
        KernelExecutionParameters parameters,
        int iterations = 100,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets execution statistics
    /// </summary>
    ExecutionStatistics GetStatistics();
    
    /// <summary>
    /// Resets execution statistics
    /// </summary>
    void ResetStatistics();
}