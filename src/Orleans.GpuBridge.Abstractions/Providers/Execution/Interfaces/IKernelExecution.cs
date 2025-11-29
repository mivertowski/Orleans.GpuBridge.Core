using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;

/// <summary>
/// Represents an ongoing kernel execution
/// </summary>
public interface IKernelExecution
{
    /// <summary>
    /// Unique execution ID
    /// </summary>
    string ExecutionId { get; }

    /// <summary>
    /// Associated kernel
    /// </summary>
    CompiledKernel Kernel { get; }

    /// <summary>
    /// Execution status
    /// </summary>
    KernelExecutionStatus Status { get; }

    /// <summary>
    /// Waits for the execution to complete
    /// </summary>
    Task<KernelExecutionResult> WaitForCompletionAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Checks if the execution is complete without waiting
    /// </summary>
    bool IsComplete { get; }

    /// <summary>
    /// Gets the execution progress (0.0 to 1.0)
    /// </summary>
    double Progress { get; }

    /// <summary>
    /// Cancels the execution if possible
    /// </summary>
    Task CancelAsync();

    /// <summary>
    /// Gets timing information if available
    /// </summary>
    KernelTiming? GetTiming();
}