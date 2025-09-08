using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.DotCompute.Interfaces;

/// <summary>
/// Represents an execution stream on an accelerator for sequential kernel execution
/// </summary>
/// <remarks>
/// Streams provide a mechanism for ordering kernel executions and memory operations.
/// Operations within a stream are executed sequentially, while operations in different
/// streams may execute concurrently.
/// </remarks>
public interface IAcceleratorStream : IDisposable
{
    /// <summary>
    /// Gets the unique identifier for this stream
    /// </summary>
    string StreamId { get; }
    
    /// <summary>
    /// Gets the accelerator this stream belongs to
    /// </summary>
    IAccelerator Accelerator { get; }
    
    /// <summary>
    /// Gets whether this stream is the default stream
    /// </summary>
    bool IsDefault { get; }
    
    /// <summary>
    /// Gets whether this stream supports concurrent kernel execution
    /// </summary>
    bool SupportsConcurrentKernels { get; }
    
    /// <summary>
    /// Synchronizes the stream, waiting for all operations to complete
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task representing the synchronization operation</returns>
    Task SynchronizeAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Waits for an event to occur on another stream
    /// </summary>
    /// <param name="eventToWaitFor">The event to wait for</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task representing the wait operation</returns>
    Task WaitForEventAsync(IAcceleratorEvent eventToWaitFor, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Records an event in this stream
    /// </summary>
    /// <returns>The recorded event</returns>
    IAcceleratorEvent RecordEvent();
    
    /// <summary>
    /// Sets the stream priority
    /// </summary>
    /// <param name="priority">Priority level (higher values = higher priority)</param>
    void SetPriority(int priority);
    
    /// <summary>
    /// Gets the current stream priority
    /// </summary>
    int GetPriority();
}