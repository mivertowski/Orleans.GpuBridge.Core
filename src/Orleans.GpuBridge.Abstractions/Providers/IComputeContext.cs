using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Models;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Represents a compute context for kernel execution on a specific device.
/// A compute context maintains the execution environment, manages resources,
/// and provides isolation between different computation sessions on the same device.
/// </summary>
public interface IComputeContext : IDisposable
{
    /// <summary>
    /// Gets the associated compute device for this context.
    /// This is the device on which all operations within this context will be executed.
    /// </summary>
    IComputeDevice Device { get; }
    
    /// <summary>
    /// Gets the unique identifier for this context.
    /// This ID can be used for debugging, logging, and context management operations.
    /// </summary>
    string ContextId { get; }
    
    /// <summary>
    /// Makes this context current for the calling thread.
    /// This is required for certain backend APIs that maintain thread-local context state.
    /// After calling this method, subsequent operations on the current thread will use this context.
    /// </summary>
    void MakeCurrent();
    
    /// <summary>
    /// Synchronizes all pending operations in this context.
    /// This method blocks until all previously submitted work to this context has completed,
    /// ensuring that all kernels, memory operations, and other commands have finished execution.
    /// </summary>
    /// <param name="cancellationToken">Token to cancel the synchronization operation.</param>
    /// <returns>A task that completes when all operations in the context have finished.</returns>
    Task SynchronizeAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Creates a command queue (also known as a stream) for this context.
    /// Command queues allow asynchronous submission of work to the device and can be used
    /// to overlap computation with memory transfers or to execute multiple kernels concurrently.
    /// </summary>
    /// <param name="options">Configuration options for the command queue creation.</param>
    /// <returns>A new command queue associated with this context.</returns>
    ICommandQueue CreateCommandQueue(CommandQueueOptions options);
}