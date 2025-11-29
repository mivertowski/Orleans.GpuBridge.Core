using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Models;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Represents a command queue (also known as a stream) for submitting work to a compute device.
/// Command queues provide asynchronous execution of kernels and memory operations,
/// allowing for efficient overlapping of computation and data transfer operations.
/// </summary>
public interface ICommandQueue : IDisposable
{
    /// <summary>
    /// Gets the unique identifier for this command queue.
    /// This ID can be used for debugging, profiling, and queue management operations.
    /// </summary>
    string QueueId { get; }

    /// <summary>
    /// Gets the associated compute context that owns this command queue.
    /// All operations submitted to this queue will execute within the scope of this context.
    /// </summary>
    IComputeContext Context { get; }

    /// <summary>
    /// Enqueues a compiled kernel for asynchronous execution on the device.
    /// The kernel will be executed with the specified launch parameters,
    /// and execution may begin immediately or be scheduled based on device availability.
    /// </summary>
    /// <param name="kernel">The compiled kernel to execute.</param>
    /// <param name="parameters">Launch parameters including work sizes, memory configuration, and kernel arguments.</param>
    /// <param name="cancellationToken">Token to cancel the kernel enqueue operation.</param>
    /// <returns>A task that completes when the kernel has been successfully enqueued (not necessarily executed).</returns>
    Task EnqueueKernelAsync(
        CompiledKernel kernel,
        KernelLaunchParameters parameters,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Enqueues a memory copy operation for asynchronous execution.
    /// This allows for efficient data transfer between host and device memory,
    /// or between different memory locations on the device.
    /// </summary>
    /// <param name="source">Pointer to the source memory location.</param>
    /// <param name="destination">Pointer to the destination memory location.</param>
    /// <param name="sizeBytes">Number of bytes to copy.</param>
    /// <param name="cancellationToken">Token to cancel the memory copy operation.</param>
    /// <returns>A task that completes when the memory copy has been successfully enqueued.</returns>
    Task EnqueueCopyAsync(
        IntPtr source,
        IntPtr destination,
        long sizeBytes,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Waits for all previously enqueued operations in this queue to complete.
    /// This method blocks until all kernels, memory operations, and other commands
    /// that were submitted to this queue have finished execution.
    /// </summary>
    /// <param name="cancellationToken">Token to cancel the flush operation.</param>
    /// <returns>A task that completes when all enqueued operations have finished.</returns>
    Task FlushAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Inserts a synchronization barrier in the command queue.
    /// This ensures that all operations enqueued before this barrier complete
    /// before any operations enqueued after the barrier begin execution.
    /// This is useful for enforcing ordering constraints between operations.
    /// </summary>
    void EnqueueBarrier();
}